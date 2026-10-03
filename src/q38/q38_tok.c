/* q38_tok.c - see q38_tok.h. */
#include "tok.h"
#include "q38_tok.h"

struct q38_tok { Tok T; int *type; };

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER_DEFINED = 4, TT_UNUSED = 5, TT_BYTE = 6 };

q38_tok *q38_tok_load(const q38_gguf *g)
{
    const q38_kv *toks = q38_gguf_kv(g, "tokenizer.ggml.tokens");
    const q38_kv *types = q38_gguf_kv(g, "tokenizer.ggml.token_type");
    const q38_kv *merges = q38_gguf_kv(g, "tokenizer.ggml.merges");
    const char *model = q38_gguf_str(g, "tokenizer.ggml.model");
    const char *pre = q38_gguf_str(g, "tokenizer.ggml.pre");
    if (!toks || !toks->strs || !merges || !merges->strs || !types || !types->iarr || types->n != toks->n) {
        fprintf(stderr, "q38_tok: the GGUF has no complete BPE vocabulary\n");
        return NULL;
    }
    if (!model || strcmp(model, "gpt2") || !pre || strcmp(pre, "qwen35")) {
        fprintf(stderr, "q38_tok: tokenizer %s/%s, expected gpt2/qwen35\n", model ? model : "?", pre ? pre : "?");
        return NULL;
    }
    q38_tok *t = (q38_tok *)calloc(1, sizeof *t);
    Tok *T = &t->T;
    tk_build_bytemap(T);
    T->o200k = 2;
    T->use_merges_only = 1;
    const int n = (int)toks->n;
    T->n_ids = n;
    T->id2str = (char **)calloc((size_t)n, sizeof(char *));
    T->id_added = (int *)calloc((size_t)n, sizeof(int));
    T->id_special = (int *)calloc((size_t)n, sizeof(int));
    t->type = (int *)calloc((size_t)n, sizeof(int));
    int vc = 1;
    while (vc < n * 2) vc <<= 1;
    hm_init(&T->vocab, vc);
    int nsp = 0;
    for (int i = 0; i < n; i++) {
        t->type[i] = (int)types->iarr[i];
        T->id2str[i] = toks->strs[i];
        if (t->type[i] == TT_CONTROL || t->type[i] == TT_USER_DEFINED) nsp++;
        else hm_put(&T->vocab, toks->strs[i], (int)strlen(toks->strs[i]), i);
    }
    int mc = 1;
    while (mc < (int)merges->n * 2) mc <<= 1;
    hm_init(&T->merges, mc);
    for (uint64_t i = 0; i < merges->n; i++) {
        const char *s = merges->strs[i];
        const char *sp = strchr(s, ' ');
        if (!sp) { fprintf(stderr, "q38_tok: malformed merge %llu\n", (unsigned long long)i); q38_tok_free(t); return NULL; }
        int ll = (int)(sp - s), rl = (int)strlen(sp + 1);
        char *key = (char *)malloc((size_t)ll + 1 + rl);
        memcpy(key, s, (size_t)ll);
        key[ll] = 0;
        memcpy(key + ll + 1, sp + 1, (size_t)rl);
        hm_put(&T->merges, key, ll + 1 + rl, (int)i);
    }
    T->nsp = nsp;
    T->sp = (Special *)calloc((size_t)(nsp ? nsp : 1), sizeof(Special));
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (t->type[i] != TT_CONTROL && t->type[i] != TT_USER_DEFINED) continue;
        T->sp[k].str = toks->strs[i];
        T->sp[k].len = (int)strlen(toks->strs[i]);
        T->sp[k].id = i;
        T->id_added[i] = 1;
        if (t->type[i] == TT_CONTROL) T->id_special[i] = 1;
        k++;
    }
    qsort(T->sp, (size_t)T->nsp, sizeof(Special), cmp_sp_len);
    return t;
}

void q38_tok_free(q38_tok *t)
{
    if (!t) return;
    /* the token strings belong to the GGUF metadata; the merge keys are ours */
    for (int i = 0; i < t->T.merges.cap; i++)
        if (t->T.merges.e[i].used) free((void *)t->T.merges.e[i].k);
    free(t->T.merges.e);
    free(t->T.vocab.e);
    free(t->T.id2str);
    free(t->T.id_added);
    free(t->T.id_special);
    free(t->T.sp);
    free(t->type);
    free(t);
}

int q38_tok_n_vocab(const q38_tok *t) { return t->T.n_ids; }

int q38_tok_encode(q38_tok *t, const char *text, int len, int *out, int max)
{
    return tok_encode(&t->T, text, len, out, max);
}

int q38_tok_piece(q38_tok *t, int id, char *out, int max)
{
    int n = tok_decode(&t->T, &id, 1, out, max - 1);
    out[n] = 0;
    return n;
}

int q38_tok_id(q38_tok *t, const char *text) { return tok_id_of(&t->T, text); }

int q38_tok_is_control(const q38_tok *t, int id)
{
    return id >= 0 && id < t->T.n_ids && t->type[id] == TT_CONTROL;
}
