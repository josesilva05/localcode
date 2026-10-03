/* q38_ref_dump - intermediate tensors of Qwen3.8-Flash-Next from llama.cpp, for validation.
 *
 *   q38_ref_dump <model.gguf> <prompt.txt> <out_dir> [names] [--raw] [--cpu]
 *
 * Tokenizes the prompt with llama.cpp (wrapped in the chat template unless --raw), writes
 * the ids to out_dir/tokens.txt, then feeds the tokens ONE AT A TIME, so every layer
 * computes every token (a batch would skip the last layer for all but the last token).
 * For each token it appends to out_dir/<tensor name>.f32 every tensor whose name matches
 * one of the comma-separated prefixes in names, and the logits to out_dir/logits.f32.
 * out_dir/index.txt lists, per file, the element count of one token's row.
 *
 * Default names: l_last (each layer's output, the 4 hyper-connection streams),
 * hc_mixed, attn_output, linear_attn_out, ffn_out, result_norm.
 *
 * Reference/diagnostic only. Built when CMake gets -DQ38_LLAMA_CPP_DIR.
 */
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILES 512

typedef struct {
    const char *dir;
    char       *names[64];
    int         n_names;
    char        file_name[MAX_FILES][96];
    long long   file_row[MAX_FILES];
    int         n_files;
    float      *buf;
    size_t      buf_cap;
} dump;

static int wanted(const dump *d, const char *name)
{
    for (int i = 0; i < d->n_names; i++) {
        size_t n = strlen(d->names[i]);
        /* "<prefix>" or "<prefix>-<layer>", nothing else ("(reshaped)" views are skipped) */
        if (strncmp(name, d->names[i], n)) continue;
        const char *r = name + n;
        if (!*r) return 1;
        if (*r == '-' && r[1] && !r[1 + strspn(r + 1, "0123456789")]) return 1;
    }
    return 0;
}

static void append(dump *d, const char *name, const float *x, long long n)
{
    int f = 0;
    for (; f < d->n_files; f++)
        if (!strcmp(d->file_name[f], name)) break;
    if (f == d->n_files) {
        if (d->n_files == MAX_FILES) return;
        snprintf(d->file_name[f], sizeof d->file_name[f], "%s", name);
        d->file_row[f] = n;
        d->n_files++;
    }
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.f32", d->dir, name);
    FILE *fp = fopen(path, "ab");
    if (!fp) return;
    fwrite(x, sizeof(float), (size_t)n, fp);
    fclose(fp);
}

static bool on_tensor(struct ggml_tensor *t, bool ask, void *user)
{
    dump *d = (dump *)user;
    if (!wanted(d, t->name)) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "skipping %s (type %d, contiguous %d)\n", t->name, (int)t->type, (int)ggml_is_contiguous(t));
        return true;
    }
    size_t n = (size_t)ggml_nelements(t);
    if (n > d->buf_cap) {
        d->buf = (float *)realloc(d->buf, n * sizeof(float));
        d->buf_cap = n;
    }
    ggml_backend_tensor_get(t, d->buf, 0, n * sizeof(float));
    append(d, t->name, d->buf, (long long)n);
    return true;
}

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = (char *)malloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    s[n] = 0;
    fclose(f);
    return s;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt.txt> <out_dir> [names] [--raw] [--cpu]\n", argv[0]);
        return 2;
    }
    static dump d;
    d.dir = argv[3];
    const char *names = "l_last,hc_mixed,attn_output,linear_attn_out,ffn_out,result_norm";
    int raw = 0, cpu = 0, from = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "--cpu")) cpu = 1;
        else if (!strcmp(argv[i], "--from") && i + 1 < argc) from = atoi(argv[++i]);
        else names = argv[i];
    }
    char *nm = (char *)malloc(strlen(names) + 1);
    strcpy(nm, names);
    for (char *p = strtok(nm, ","); p && d.n_names < 64; p = strtok(NULL, ",")) d.names[d.n_names++] = p;

    char *user = slurp(argv[2]);
    if (!user) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }

    llama_backend_init();
    ggml_backend_load_all();

    struct llama_model_tensor_buft_override ov[2] = {
        { "ffn_(up|gate|down)_exps", ggml_backend_cpu_buffer_type() },
        { NULL, NULL },
    };
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = cpu ? 0 : 999;
    if (!cpu) mp.tensor_buft_overrides = ov;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &d;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    char *text = user;
    int len = (int)strlen(user);
    if (!raw) {
        struct llama_chat_message msg = { "user", user };
        const char *tmpl = llama_model_chat_template(model, NULL);
        len = llama_chat_apply_template(tmpl, &msg, 1, true, NULL, 0);
        text = (char *)malloc((size_t)len + 1);
        llama_chat_apply_template(tmpl, &msg, 1, true, text, len + 1);
        text[len] = 0;
    }
    int n = -llama_tokenize(vocab, text, len, NULL, 0, true, true);
    llama_token *toks = (llama_token *)malloc((size_t)n * sizeof(llama_token));
    llama_tokenize(vocab, text, len, toks, n, true, true);

    char path[1024];
    snprintf(path, sizeof path, "%s/tokens.txt", d.dir);
    FILE *ft = fopen(path, "w");
    if (!ft) { fprintf(stderr, "cannot write %s (does the directory exist?)\n", path); return 1; }
    for (int i = 0; i < n; i++) fprintf(ft, "%d\n", toks[i]);
    fclose(ft);
    snprintf(path, sizeof path, "%s/text.txt", d.dir);
    ft = fopen(path, "wb");
    fwrite(text, 1, (size_t)len, ft);
    fclose(ft);
    fprintf(stderr, "%d tokens\n", n);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < n; i++) {
        if (llama_decode(ctx, llama_batch_get_one(&toks[i], 1))) { fprintf(stderr, "decode failed at %d\n", i); return 1; }
        if (i >= from) append(&d, "logits", llama_get_logits_ith(ctx, -1), n_vocab);
        fprintf(stderr, "\rtoken %d/%d", i + 1, n);
    }
    fprintf(stderr, "\n");

    snprintf(path, sizeof path, "%s/index.txt", d.dir);
    ft = fopen(path, "w");
    for (int f = 0; f < d.n_files; f++) fprintf(ft, "%s %lld\n", d.file_name[f], d.file_row[f]);
    fclose(ft);
    fprintf(stderr, "%d files in %s\n", d.n_files, d.dir);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
