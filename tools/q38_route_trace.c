/* q38_route_trace - record Qwen3.8-Flash-Next's expert routing with llama.cpp.
 *
 *   q38_route_trace <model.gguf> <prompt.txt> <trace.txt> [n_gen=256] [n_ctx=8192]
 *
 * Runs the model through libllama (trunk on the GPUs, routed experts on the CPU, the
 * same placement as the llama-bench baseline), wraps the prompt in the model's chat
 * template, decodes greedily, and taps every layer's router output ("ffn_moe_probs-<il>",
 * the softmax over the 512 experts) with the scheduler's eval callback. For each token and
 * MoE layer it writes one line:
 *
 *     <P|D> <position> <layer> id:prob id:prob ...   (16 best experts, best first)
 *
 * P = prompt (prefill) token, D = generated token. The first 10 are the experts the model
 * used. tools/q38_route_sim.py replays the trace through an expert cache.
 *
 * Built only when CMake gets -DQ38_LLAMA_CPP_DIR=<llama.cpp checkout with a build/>.
 */
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TOPN 16
#define MAX_LAYERS 128

typedef struct {
    int n_exp;                  /* router width (512), learned from the first tensor */
    int n_tok;                  /* tokens in the current batch (the widest layer) */
    float *probs[MAX_LAYERS];   /* [have][n_exp] per layer, for the current batch */
    int have[MAX_LAYERS];       /* tokens the layer computed: llama.cpp runs the last layer
                                 * of a prefill batch only for the tokens whose logits are
                                 * wanted (the batch's last), so this can be < n_tok */
    int cap_tok;
} tap;

static bool on_tensor(struct ggml_tensor *t, bool ask, void *user)
{
    tap *tp = (tap *)user;
    static const char pre[] = "ffn_moe_probs-";
    if (strncmp(t->name, pre, sizeof pre - 1)) return ask ? false : true;
    if (getenv("Q38_TRACE_DEBUG") && !ask)
        fprintf(stderr, "[tap] %s type %d ne %lld %lld %lld %lld\n", t->name, (int)t->type,
                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    if (ask) return true;
    const char *num = t->name + sizeof pre - 1;
    if (!*num || num[strspn(num, "0123456789")]) return ask ? false : true;   /* skip "-12 (reshaped)" views */
    int il = atoi(num);
    if (il < 0 || il >= MAX_LAYERS || t->type != GGML_TYPE_F32) return true;
    int n_exp = (int)t->ne[0], n_tok = (int)(ggml_nelements(t) / t->ne[0]);
    if (!tp->n_exp) tp->n_exp = n_exp;
    if (n_exp != tp->n_exp || n_tok > tp->cap_tok) {
        fprintf(stderr, "unexpected router tensor %s [%d x %d]\n", t->name, n_exp, n_tok);
        return true;
    }
    if (!tp->probs[il]) tp->probs[il] = (float *)malloc((size_t)tp->cap_tok * n_exp * sizeof(float));
    ggml_backend_tensor_get(t, tp->probs[il], 0, (size_t)n_tok * n_exp * sizeof(float));
    tp->have[il] = n_tok;
    if (n_tok > tp->n_tok) tp->n_tok = n_tok;
    return true;
}

/* Writes and clears the batch held in tp. pos0 = position of its first token. */
static void flush(tap *tp, FILE *out, char phase, int pos0)
{
    for (int k = 0; k < tp->n_tok; k++) {
        for (int il = 0; il < MAX_LAYERS; il++) {
            const int kk = k - (tp->n_tok - tp->have[il]);   /* rows are the batch's last tokens */
            if (!tp->have[il] || kk < 0) continue;
            const float *p = tp->probs[il] + (size_t)kk * tp->n_exp;
            int best[TOPN];
            int nb = 0;
            for (int e = 0; e < tp->n_exp; e++) {          /* insertion into a top-16 list */
                int j = nb < TOPN ? nb++ : TOPN;
                if (j == TOPN && p[e] <= p[best[TOPN - 1]]) continue;
                if (j == TOPN) j = TOPN - 1;
                while (j > 0 && p[best[j - 1]] < p[e]) { best[j] = best[j - 1]; j--; }
                best[j] = e;
            }
            fprintf(out, "%c %d %d", phase, pos0 + k, il);
            for (int j = 0; j < nb; j++) fprintf(out, " %d:%.6g", best[j], p[best[j]]);
            fputc('\n', out);
        }
    }
    memset(tp->have, 0, sizeof tp->have);
    tp->n_tok = 0;
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

static double now(void) { return (double)clock() / CLOCKS_PER_SEC; }

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt.txt> <trace.txt> [n_gen=256] [n_ctx=8192]\n", argv[0]);
        return 2;
    }
    const int n_gen = argc > 4 ? atoi(argv[4]) : 256;
    const int n_ctx = argc > 5 ? atoi(argv[5]) : 8192;
    char *user = slurp(argv[2]);
    if (!user) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    FILE *out = fopen(argv[3], "w");
    if (!out) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }

    llama_backend_init();
    ggml_backend_load_all();

    struct llama_model_tensor_buft_override ov[2] = {
        { "ffn_(up|gate|down)_exps", ggml_backend_cpu_buffer_type() },
        { NULL, NULL },
    };
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.tensor_buft_overrides = ov;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    tap tp;
    memset(&tp, 0, sizeof tp);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &tp;
    tp.cap_tok = (int)cp.n_ubatch;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    /* chat template, generation prompt appended */
    struct llama_chat_message msg = { "user", user };
    const char *tmpl = llama_model_chat_template(model, NULL);
    int len = llama_chat_apply_template(tmpl, &msg, 1, true, NULL, 0);
    char *text = (char *)malloc((size_t)len + 1);
    llama_chat_apply_template(tmpl, &msg, 1, true, text, len + 1);
    text[len] = 0;

    int n_prompt = -llama_tokenize(vocab, text, len, NULL, 0, true, true);
    llama_token *toks = (llama_token *)malloc((size_t)(n_prompt + n_gen + 1) * sizeof(llama_token));
    llama_tokenize(vocab, text, len, toks, n_prompt, true, true);
    fprintf(stderr, "prompt: %d tokens, generating up to %d\n", n_prompt, n_gen);

    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    /* prefill in ubatch-sized pieces so each callback holds one piece */
    double t0 = now();
    for (int i = 0; i < n_prompt; i += (int)cp.n_ubatch) {
        int n = n_prompt - i < (int)cp.n_ubatch ? n_prompt - i : (int)cp.n_ubatch;
        if (llama_decode(ctx, llama_batch_get_one(toks + i, n))) { fprintf(stderr, "prefill failed\n"); return 1; }
        flush(&tp, out, 'P', i);
    }
    double t1 = now();
    int pos = n_prompt, n_out = 0;
    char piece[256];
    for (; n_out < n_gen; n_out++) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        int pn = llama_token_to_piece(vocab, tok, piece, sizeof piece - 1, 0, true);
        if (pn > 0) { piece[pn] = 0; fputs(piece, stderr); }
        toks[pos] = tok;
        if (llama_decode(ctx, llama_batch_get_one(&toks[pos], 1))) { fprintf(stderr, "\ndecode failed\n"); return 1; }
        flush(&tp, out, 'D', pos);
        pos++;
    }
    double t2 = now();
    fprintf(stderr, "\n\nprefill %d tokens %.1f s, decode %d tokens %.1f s (%.2f tok/s with the tap)\n",
            n_prompt, t1 - t0, n_out, t2 - t1, n_out / (t2 - t1 > 0 ? t2 - t1 : 1));

    fclose(out);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    for (int il = 0; il < MAX_LAYERS; il++) free(tp.probs[il]);
    free(toks);
    free(text);
    free(user);
    return 0;
}
