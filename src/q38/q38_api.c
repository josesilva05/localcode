/* q38_api.c - see q38_api.h. */
#include "q38_api.h"
#include "q38_model.h"
#include "q38_tok.h"
#ifdef Q38_CUDA
#include "q38_gpu.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

struct q38api {
    q38_model m;
    q38_tok  *tok;
    int       gpu;            /* device of the trunk, -1 = CPU */
    int       snap_pos;       /* position of the saved state, -1 = none */
    char      prof[1024];     /* usage profile path, "" = none */
};

static void profile_path(char *out, size_t cap)
{
    out[0] = 0;
    const char *base = getenv("LOCALAPPDATA");
    if (!base) return;
    char dir[900];
    snprintf(dir, sizeof dir, "%s\\localcode", base);
#ifdef _WIN32
    CreateDirectoryA(dir, NULL);
#endif
    snprintf(out, cap, "%s\\qwen38_expert_profile.bin", dir);
}

q38api *q38api_open(const char *model, int ctx, int gpu, double preload_gb)
{
    q38api *h = (q38api *)calloc(1, sizeof *h);
    if (!h) return NULL;
    h->gpu = -1;
    h->snap_pos = -1;
    if (q38_model_open(&h->m, model, ctx)) { free(h); return NULL; }
#ifdef Q38_CUDA
    if (gpu >= 0) {
        if (q38_gpu_open(&h->m, gpu) == 0) h->gpu = gpu;
    } else if (gpu == -1) {
        if (q38_gpu_open(&h->m, 1) == 0) h->gpu = 1;
        else if (q38_gpu_open(&h->m, 0) == 0) h->gpu = 0;
    }
#else
    (void)gpu;
#endif
    h->tok = q38_tok_load(&h->m.g);
    if (!h->tok) { q38_model_close(&h->m); free(h); return NULL; }
    profile_path(h->prof, sizeof h->prof);
    if (h->prof[0] && q38_model_profile_load(&h->m, h->prof) == 0 && preload_gb > 0) {
        const int nr = q38_model_preload(&h->m, preload_gb);
        int nv = 0;
#ifdef Q38_CUDA
        if (h->gpu >= 0) nv = q38_gpu_prefill(&h->m);
#endif
        fprintf(stderr, "q38: warmed %d experts into RAM (%.1f GB) and %d into VRAM\n", nr, h->m.cache_used / 1e9, nv);
    }
    return h;
}

void q38api_close(q38api *h)
{
    if (!h) return;
    q38api_save_profile(h);
    q38_tok_free(h->tok);
    q38_model_close(&h->m);
    free(h);
}

int q38api_n_ctx(q38api *h) { return h->m.max_ctx; }
int q38api_n_vocab(q38api *h) { return h->m.c.n_vocab; }
int q38api_n_past(q38api *h) { return h->m.n_past; }

int q38api_meta_str(q38api *h, const char *key, char *out, int cap)
{
    const char *v = q38_gguf_str(&h->m.g, key);
    if (!v) { if (cap > 0) out[0] = 0; return -1; }
    const int n = (int)strlen(v);
    if (cap > 0) {
        const int k = n < cap - 1 ? n : cap - 1;
        memcpy(out, v, (size_t)k);
        out[k] = 0;
    }
    return n;
}

long long q38api_meta_int(q38api *h, const char *key, long long def) { return (long long)q38_gguf_int(&h->m.g, key, def); }

int q38api_tokenize(q38api *h, const char *text, int len, int *out, int max) { return q38_tok_encode(h->tok, text, len, out, max); }
int q38api_piece(q38api *h, int id, char *out, int cap) { return q38_tok_piece(h->tok, id, out, cap); }
int q38api_is_control(q38api *h, int id) { return q38_tok_is_control(h->tok, id); }
int q38api_token_id(q38api *h, const char *text) { return q38_tok_id(h->tok, text); }

int q38api_eval(q38api *h, const int *toks, int n, float *logits)
{
    if (n <= 0) return -1;
    const float *lg = q38_model_forward_batch(&h->m, toks, n);
    if (!lg) return -1;
    if (logits) memcpy(logits, lg, (size_t)h->m.c.n_vocab * sizeof(float));
    return 0;
}

void q38api_reset(q38api *h)
{
    q38_model_reset(&h->m);
    h->snap_pos = -1;
}

int q38api_snapshot(q38api *h)
{
#ifdef Q38_CUDA
    if (h->gpu >= 0 && q38_gpu_state(&h->m, 1) == 0) {
        h->snap_pos = h->m.n_past;
        return h->snap_pos;
    }
#endif
    h->snap_pos = -1;
    return -1;
}

int q38api_restore(q38api *h)
{
#ifdef Q38_CUDA
    if (h->gpu >= 0 && h->snap_pos >= 0 && q38_gpu_state(&h->m, 0) == 0) {
        h->m.n_past = h->snap_pos;
        return h->snap_pos;
    }
#endif
    return -1;
}

int q38api_save_profile(q38api *h)
{
    return h->prof[0] ? q38_model_profile_save(&h->m, h->prof) : -1;
}
