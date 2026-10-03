/* q38_model.c - see q38_model.h. The comments name the llama.cpp graph node each step
 * reproduces, so a difference against tools/q38_reference.py can be traced to its source. */
#include "q38_model.h"
#include "q38_quant.h"
#include "q38_kernels.h"
#ifdef Q38_CUDA
#include "q38_gpu.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

static double now_s(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------- config -- */
static int64_t need_int(const q38_gguf *g, const char *key, int *bad)
{
    int64_t v = q38_gguf_int(g, key, -1);
    if (v < 0) { fprintf(stderr, "q38: missing %s\n", key); *bad = 1; }
    return v;
}

static int read_cfg(q38_cfg *c, const q38_gguf *g)
{
    const char *arch = q38_gguf_str(g, "general.architecture");
    if (!arch || strcmp(arch, "qwen4exp")) {
        fprintf(stderr, "q38: architecture %s, expected qwen4exp\n", arch ? arch : "(none)");
        return -1;
    }
    int bad = 0;
    memset(c, 0, sizeof *c);
    c->n_layer    = (int)need_int(g, "qwen4exp.block_count", &bad);
    c->n_embd     = (int)need_int(g, "qwen4exp.embedding_length", &bad);
    c->n_head     = (int)need_int(g, "qwen4exp.attention.head_count", &bad);
    c->n_head_kv  = (int)need_int(g, "qwen4exp.attention.head_count_kv", &bad);
    c->head_dim   = (int)need_int(g, "qwen4exp.attention.key_length", &bad);
    c->n_rot      = (int)need_int(g, "qwen4exp.rope.dimension_count", &bad);
    c->rope_base  = q38_gguf_float(g, "qwen4exp.rope.freq_base", 10000.0);
    c->eps        = (float)q38_gguf_float(g, "qwen4exp.attention.layer_norm_rms_epsilon", 1e-6);
    c->n_exp      = (int)need_int(g, "qwen4exp.expert_count", &bad);
    c->n_exp_used = (int)need_int(g, "qwen4exp.expert_used_count", &bad);
    c->ff_exp     = (int)need_int(g, "qwen4exp.expert_feed_forward_length", &bad);
    c->ff_shexp   = (int)need_int(g, "qwen4exp.expert_shared_feed_forward_length", &bad);
    c->interval   = (int)q38_gguf_int(g, "qwen4exp.full_attention_interval", 4);
    c->hc         = (int)need_int(g, "qwen4exp.hyper_connection.count", &bad);
    c->hc_rank    = (int)need_int(g, "qwen4exp.hyper_connection.low_rank", &bad);
    c->conv_k     = (int)need_int(g, "qwen4exp.ssm.conv_kernel", &bad);
    c->d_state    = (int)need_int(g, "qwen4exp.ssm.state_size", &bad);
    c->n_kh       = (int)need_int(g, "qwen4exp.ssm.group_count", &bad);
    c->n_vh       = (int)need_int(g, "qwen4exp.ssm.time_step_rank", &bad);
    c->idx_top_k  = (int)q38_gguf_int(g, "qwen4exp.attention.indexer.top_k", 0);
    if (bad) return -1;

    /* dense attention == the indexer's choice while n_kv <= top_k + ratio - 1 */
    int ratio = 0;
    const q38_kv *cr = q38_gguf_kv(g, "qwen4exp.attention.compress_ratios");
    if (cr && cr->iarr)
        for (uint64_t i = 0; i < cr->n; i++) if (cr->iarr[i] > ratio) ratio = (int)cr->iarr[i];
    c->max_dense_ctx = c->idx_top_k > 0 ? c->idx_top_k + (ratio > 0 ? ratio : 1) - 1 : 1 << 30;
    c->idx_ratio = ratio;
    c->idx_heads = (int)q38_gguf_int(g, "qwen4exp.attention.indexer.head_count", 0);
    c->idx_dim = (int)q38_gguf_int(g, "qwen4exp.attention.indexer.key_length", 0);

    const q38_kv *pl = q38_gguf_kv(g, "qwen4exp.ple.layers");
    c->ple_layer = -1;
    if (pl && pl->iarr && pl->n == 1) {
        c->ple_layer   = (int)pl->iarr[0];
        c->ple_ngram   = (int)need_int(g, "qwen4exp.ple.ngram_size", &bad);
        c->ple_per_gram = (int)need_int(g, "qwen4exp.ple.heads_per_ngram", &bad);
        c->ple_kern    = (int)need_int(g, "qwen4exp.ple.conv_kernel", &bad);
        c->ple_eos     = (int)need_int(g, "qwen4exp.ple.eos_token_id", &bad);
        c->ple_dim     = (int)need_int(g, "qwen4exp.embedding_length_per_layer_input", &bad);
        c->ple_heads   = (c->ple_ngram - 1) * c->ple_per_gram;
        const q38_kv *mu = q38_gguf_kv(g, "qwen4exp.ple.layer_multipliers");
        const q38_kv *of = q38_gguf_kv(g, "qwen4exp.ple.head_offsets");
        const q38_kv *vo = q38_gguf_kv(g, "qwen4exp.ple.head_vocab_sizes");
        if (bad || !mu || !of || !vo || !mu->iarr || !of->iarr || !vo->iarr ||
            (int)mu->n < c->ple_ngram || c->ple_ngram > 8 || c->ple_heads > Q38_MAX_PLE_HEADS ||
            (int)of->n < c->ple_heads || (int)vo->n < c->ple_heads) {
            fprintf(stderr, "q38: PLE metadata incomplete\n");
            return -1;
        }
        for (int i = 0; i < c->ple_ngram; i++) c->ple_mult[i] = (uint64_t)mu->iarr[i];
        for (int h = 0; h < c->ple_heads; h++) { c->ple_off[h] = of->iarr[h]; c->ple_voc[h] = vo->iarr[h]; }
    } else if (pl && pl->n > 1) {
        fprintf(stderr, "q38: %llu PLE layers, only one is supported\n", (unsigned long long)pl->n);
        return -1;
    }
    if (c->n_layer > Q38_MAX_LAYERS || c->hc < 2 || c->n_vh % c->n_kh || c->n_head % c->n_head_kv ||
        c->n_rot > c->head_dim || c->n_rot % 2) {
        fprintf(stderr, "q38: unsupported shape\n");
        return -1;
    }
    const q38_kv *voc = q38_gguf_kv(g, "tokenizer.ggml.tokens");
    c->n_vocab = voc ? (int)voc->n : 0;
    return 0;
}

/* ------------------------------------------------------------------ weights -- */
static int bind(q38_model *m, q38_mat *w, const char *name, int64_t ne0, int64_t rows, int load)
{
    const q38_tensor *t = q38_gguf_tensor(&m->g, name);
    memset(w, 0, sizeof *w);
    if (!t) { fprintf(stderr, "q38: missing tensor %s\n", name); return -1; }
    int64_t all_rows = t->ne[1] * t->ne[2] * t->ne[3];
    if (t->ne[0] != ne0 || (rows >= 0 && all_rows != rows)) {
        fprintf(stderr, "q38: %s is [%lld x %lld], expected [%lld x %lld]\n", name, (long long)t->ne[0],
                (long long)all_rows, (long long)ne0, (long long)rows);
        return -1;
    }
    if (!q38_quant_supported(t->type)) {
        fprintf(stderr, "q38: %s has type %s, which has no kernel yet\n", name, q38_type_name(t->type));
        return -1;
    }
    w->type = t->type;
    w->ne0 = t->ne[0];
    w->rows = all_rows;
    w->row_bytes = q38_row_bytes(t->type, t->ne[0]);
    w->t = t;
    if (load) {
        w->data = (uint8_t *)malloc((size_t)t->nbytes);
        if (!w->data || q38_gguf_read(&m->g, t, 0, w->data, t->nbytes)) {
            fprintf(stderr, "q38: cannot load %s (%.1f MB)\n", name, t->nbytes / 1e6);
            return -1;
        }
        m->trunk_bytes += t->nbytes;
    }
    return 0;
}

#define B(field, nm, ne0, rows) do { snprintf(name, sizeof name, nm, il); \
    if (bind(m, &l->field, name, ne0, rows, 1)) return -1; } while (0)
#define BX(field, nm, ne0, rows) do { snprintf(name, sizeof name, nm, il); \
    if (bind(m, &l->field, name, ne0, rows, 0)) return -1; } while (0)

static int bind_all(q38_model *m)
{
    const q38_cfg *c = &m->c;
    const int64_t E = c->n_embd, hcE = (int64_t)c->hc * E, R = c->hc_rank;
    const int64_t key_dim = (int64_t)c->d_state * c->n_kh, val_dim = (int64_t)c->d_state * c->n_vh;
    char name[128];
    if (bind(m, &m->tok_embd, "token_embd.weight", E, c->n_vocab, 1)) return -1;
    if (bind(m, &m->output, "output.weight", E, c->n_vocab, 1)) return -1;
    if (bind(m, &m->out_hc_norm, "output_hc_norm.weight", hcE, 1, 1)) return -1;
    if (bind(m, &m->out_hc_down, "output_hc_down.weight", hcE, R, 1)) return -1;
    if (bind(m, &m->out_hc_up, "output_hc_up.weight", R, hcE, 1)) return -1;
    if (c->ple_layer >= 0 &&
        bind(m, &m->ple_table, "per_layer_token_embd.weight", c->ple_dim, -1, 0)) return -1;
    for (int il = 0; il < c->n_layer; il++) {
        q38_layer *l = &m->L[il];
        l->recurrent = (il + 1) % c->interval != 0;
        B(hc_attn_norm, "blk.%d.hc_attn_norm.weight", hcE, 1);
        B(hc_attn_down, "blk.%d.hc_attn_down.weight", hcE, R);
        B(hc_attn_up, "blk.%d.hc_attn_up.weight", R, hcE);
        B(hc_attn_inject, "blk.%d.hc_attn_inject.weight", hcE, c->hc);
        B(hc_ffn_norm, "blk.%d.hc_ffn_norm.weight", hcE, 1);
        B(hc_ffn_down, "blk.%d.hc_ffn_down.weight", hcE, R);
        B(hc_ffn_up, "blk.%d.hc_ffn_up.weight", R, hcE);
        B(hc_ffn_inject, "blk.%d.hc_ffn_inject.weight", hcE, c->hc);
        if (l->recurrent) {
            B(qkv, "blk.%d.attn_qkv.weight", E, 2 * key_dim + val_dim);
            B(gate, "blk.%d.attn_gate.weight", E, val_dim);
            B(beta, "blk.%d.ssm_beta.weight", E, c->n_vh);
            B(alpha, "blk.%d.ssm_alpha.weight", E, c->n_vh);
            B(dt_bias, "blk.%d.ssm_dt.bias", c->n_vh, 1);
            B(a, "blk.%d.ssm_a", c->n_vh, 1);
            B(conv, "blk.%d.ssm_conv1d.weight", c->conv_k, 2 * key_dim + val_dim);
            B(ssm_norm, "blk.%d.ssm_norm.weight", c->d_state, 1);
            B(ssm_out, "blk.%d.ssm_out.weight", val_dim, E);
        } else {
            B(wq, "blk.%d.attn_q.weight", E, (int64_t)c->n_head * c->head_dim * 2);
            B(wk, "blk.%d.attn_k.weight", E, (int64_t)c->n_head_kv * c->head_dim);
            B(wv, "blk.%d.attn_v.weight", E, (int64_t)c->n_head_kv * c->head_dim);
            B(wo, "blk.%d.attn_output.weight", (int64_t)c->n_head * c->head_dim, E);
            B(q_norm, "blk.%d.attn_q_norm.weight", c->head_dim, 1);
            B(k_norm, "blk.%d.attn_k_norm.weight", c->head_dim, 1);
            if (c->idx_top_k > 0) {
                B(idx_q, "blk.%d.indexer.q_proj.weight", E, (int64_t)c->idx_heads * c->idx_dim);
                B(idx_k, "blk.%d.indexer.k_proj.weight", E, c->idx_dim);
                B(idx_qn, "blk.%d.indexer.q_norm.weight", c->idx_dim, 1);
                B(idx_kn, "blk.%d.indexer.k_norm.weight", c->idx_dim, 1);
            }
        }
        if (il == c->ple_layer) {
            const int64_t pin = (int64_t)c->ple_heads * c->ple_dim;
            B(ple_key, "blk.%d.ple_key.weight", pin, hcE);
            B(ple_value, "blk.%d.ple_value.weight", pin, E);
            B(ple_norm_key, "blk.%d.ple_norm_key.weight", hcE, 1);
            B(ple_norm_query, "blk.%d.ple_norm_query.weight", hcE, 1);
            B(ple_norm_conv, "blk.%d.ple_norm_conv.weight", hcE, 1);
            B(ple_conv, "blk.%d.ple_conv1d.weight", c->ple_kern, hcE);
        }
        B(router, "blk.%d.ffn_gate_inp.weight", E, c->n_exp);
        B(shexp_gate_inp, "blk.%d.ffn_gate_inp_shexp.weight", E, 1);
        B(sh_gate, "blk.%d.ffn_gate_shexp.weight", E, c->ff_shexp);
        B(sh_up, "blk.%d.ffn_up_shexp.weight", E, c->ff_shexp);
        B(sh_down, "blk.%d.ffn_down_shexp.weight", c->ff_shexp, E);
        BX(ex_gate, "blk.%d.ffn_gate_exps.weight", E, (int64_t)c->ff_exp * c->n_exp);
        BX(ex_up, "blk.%d.ffn_up_exps.weight", E, (int64_t)c->ff_exp * c->n_exp);
        BX(ex_down, "blk.%d.ffn_down_exps.weight", c->ff_exp, E * c->n_exp);
    }
    return 0;
}

/* -------------------------------------------------------------------- kernels -- */
#define MAX_NE0 16384
#define MAX_THREADS 64
static float g_rowbuf[MAX_THREADS][MAX_NE0];

static int thread_id(void)
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/* Q38_NAIVE=1: every product dequantises the row and sums in double (the first, slowest
 * implementation, kept as a control for validation). Otherwise q38_kernels.c. */
static int g_naive = -1;

/* y[r] = sum_i W[r][i] * x[i] for r < n_rows; W rows at data. */
static void matvec_rows(int type, const uint8_t *data, int64_t row_bytes, int64_t ne0, int64_t n_rows,
                        const float *x, float *y)
{
    if (g_naive < 0) g_naive = getenv("Q38_NAIVE") && atoi(getenv("Q38_NAIVE")) != 0;
    if (!g_naive) {
        float bsum[MAX_NE0 / 32];
        q38_xvec v;
        q38_xprep(&v, x, ne0, bsum);
        int n = (int)n_rows, r;
#pragma omp parallel for schedule(static) if (n_rows * ne0 > 65536)
        for (r = 0; r < n; r++) y[r] = q38_dot(type, data + (int64_t)r * row_bytes, &v);
        return;
    }
    if (type == Q38_F32) {
        const float *w = (const float *)data;
        int n = (int)n_rows, r;
#pragma omp parallel for schedule(static) if (n_rows * ne0 > 65536)
        for (r = 0; r < n; r++) {
            const float *wr = w + (int64_t)r * ne0;
            double s = 0;
            for (int64_t i = 0; i < ne0; i++) s += (double)wr[i] * x[i];
            y[r] = (float)s;
        }
        return;
    }
    int n = (int)n_rows, r;
#pragma omp parallel for schedule(static) if (n_rows * ne0 > 65536)
    for (r = 0; r < n; r++) {
        float *buf = g_rowbuf[thread_id()];
        q38_dequant_row(type, data + (int64_t)r * row_bytes, buf, ne0);
        double s = 0;
        for (int64_t i = 0; i < ne0; i++) s += (double)buf[i] * x[i];
        y[r] = (float)s;
    }
}

static void matvec(const q38_mat *w, const float *x, float *y)
{
    matvec_rows(w->type, w->data, w->row_bytes, w->ne0, w->rows, x, y);
}

static const float *vecf(const q38_mat *w) { return (const float *)w->data; }

static float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
static float siluf_(float x) { return x / (1.0f + expf(-x)); }
static float softplusf_(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

/* x / sqrt(mean(x^2) + eps) * w (w may be NULL), n elements */
static void rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float sc = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) y[i] = x[i] * sc * (w ? w[i] : 1.0f);
}

/* ----------------------------------------------------------- hyper-connections -- */
/* build_hc_mix: grouped RMSNorm per stream, low-rank sigmoid gate, mean over streams.
 * xn (hc*E) is left in m->xn for the inject product. */
static void hc_mix(q38_model *m, const float *res, const q38_mat *norm, const q38_mat *down,
                   const q38_mat *up, const q38_mat *inj, float *mixed, float *inject)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, hc = c->hc, hcE = hc * E;
    for (int s = 0; s < hc; s++) rmsnorm(m->xn + s * E, res + s * E, vecf(norm) + s * E, E, c->eps);
    float *lo = m->tmp, *gate = m->big;
    matvec(down, m->xn, lo);
    for (int i = 0; i < c->hc_rank; i++) lo[i] = siluf_(lo[i] / hc);
    matvec(up, lo, gate);
    for (int d = 0; d < E; d++) {
        double s = 0;
        for (int st = 0; st < hc; st++) s += (double)m->xn[st * E + d] * sigmoidf_(gate[st * E + d]);
        mixed[d] = (float)(s / hc);
    }
    if (inj) matvec(inj, m->xn, inject);
    (void)hcE;
}

/* build_hc_combine: res[s] += 2*sigmoid(inject[s]/hc) * out */
static void hc_combine(q38_model *m, float *res, const float *out, const float *inject)
{
    const int E = m->c.n_embd, hc = m->c.hc;
    for (int s = 0; s < hc; s++) {
        float w = 2.0f * sigmoidf_(inject[s] / hc);
        for (int d = 0; d < E; d++) res[s * E + d] += w * out[d];
    }
}

/* ------------------------------------------------------------------------ PLE -- */
/* The PLE input of token tok at position m->n_past: the n-gram hash rows of the table,
 * dequantised into emb (ple_heads * ple_dim floats). The rows are read in parallel. */
int q38_model_ple_emb(q38_model *m, int tok, float *emb)
{
    const q38_cfg *c = &m->c;
    int64_t ctx[8], rows[Q38_MAX_PLE_HEADS];
    ctx[0] = tok;
    int cut = 0;
    for (int s = 1; s < c->ple_ngram; s++) {
        int64_t idx = (int64_t)m->n_past - s;
        int64_t t = idx >= 0 ? m->toks[idx] : -1;
        cut = cut || t < 0 || t == c->ple_eos;
        ctx[s] = cut ? c->ple_eos : t;
    }
    for (int n = 2; n <= c->ple_ngram; n++) {
        uint64_t mixed = (uint64_t)ctx[0] * c->ple_mult[0];
        for (int j = 1; j < n; j++) mixed ^= (uint64_t)ctx[j] * c->ple_mult[j];
        const int base = (n - 2) * c->ple_per_gram;
        for (int g = 0; g < c->ple_per_gram; g++) {
            const int h = base + g;
            rows[h] = (int64_t)(mixed % (uint64_t)c->ple_voc[h]) + c->ple_off[h];
        }
    }
    if (m->ple_table.row_bytes > 1024) return -1;
    int bad = 0, h;
#pragma omp parallel for schedule(static) reduction(+ : bad)
    for (h = 0; h < c->ple_heads; h++) {
        uint8_t raw[1024];
        if (q38_gguf_pread(&m->g, m->ple_table.t, rows[h] * m->ple_table.row_bytes, raw, m->ple_table.row_bytes)) bad++;
        else q38_dequant_row(m->ple_table.type, raw, emb + (int64_t)h * c->ple_dim, c->ple_dim);
    }
    return bad ? -1 : 0;
}

/* q38_model_ple_emb for n tokens at positions p0.. (their ids already in m->toks): all the
 * rows of all the tokens read in one parallel loop */
int q38_model_ple_emb_batch(q38_model *m, int p0, int n, float *emb)
{
    const q38_cfg *c = &m->c;
    const int H = c->ple_heads, pin = H * c->ple_dim;
    int64_t *rows = (int64_t *)malloc((size_t)n * H * sizeof(int64_t));
    if (!rows || m->ple_table.row_bytes > 1024) { free(rows); return -1; }
    for (int t = 0; t < n; t++) {
        int64_t ctx[8];
        ctx[0] = m->toks[p0 + t];
        int cut = 0;
        for (int s2 = 1; s2 < c->ple_ngram; s2++) {
            const int64_t idx = (int64_t)p0 + t - s2;
            const int64_t tk = idx >= 0 ? m->toks[idx] : -1;
            cut = cut || tk < 0 || tk == c->ple_eos;
            ctx[s2] = cut ? c->ple_eos : tk;
        }
        for (int ng = 2; ng <= c->ple_ngram; ng++) {
            uint64_t mixed = (uint64_t)ctx[0] * c->ple_mult[0];
            for (int j = 1; j < ng; j++) mixed ^= (uint64_t)ctx[j] * c->ple_mult[j];
            const int base = (ng - 2) * c->ple_per_gram;
            for (int g = 0; g < c->ple_per_gram; g++) {
                const int h = base + g;
                rows[(int64_t)t * H + h] = (int64_t)(mixed % (uint64_t)c->ple_voc[h]) + c->ple_off[h];
            }
        }
    }
    int bad = 0, i;
    const int total = n * H;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : bad) num_threads(32)
    for (i = 0; i < total; i++) {
        uint8_t raw[1024];
        if (q38_gguf_pread(&m->g, m->ple_table.t, rows[i] * m->ple_table.row_bytes, raw, m->ple_table.row_bytes)) bad++;
        else q38_dequant_row(m->ple_table.type, raw, emb + (int64_t)(i / H) * pin + (int64_t)(i % H) * c->ple_dim, c->ple_dim);
    }
    free(rows);
    return bad ? -1 : 0;
}

static int ple(q38_model *m, int il, float *res, int tok)
{
    const q38_cfg *c = &m->c;
    const q38_layer *l = &m->L[il];
    const int E = c->n_embd, hc = c->hc, hcE = hc * E;
    int64_t ctx[8];
    ctx[0] = tok;
    int cut = 0;
    for (int s = 1; s < c->ple_ngram; s++) {
        int64_t idx = (int64_t)m->n_past - s;
        int64_t t = idx >= 0 ? m->toks[idx] : -1;
        cut = cut || t < 0 || t == c->ple_eos;
        ctx[s] = cut ? c->ple_eos : t;
    }
    const int pin = c->ple_heads * c->ple_dim;
    float *emb = m->tmp2;
    uint8_t raw[1024];
    if (m->ple_table.row_bytes > (int64_t)sizeof raw) return -1;
    for (int n = 2; n <= c->ple_ngram; n++) {
        uint64_t mixed = (uint64_t)ctx[0] * c->ple_mult[0];
        for (int j = 1; j < n; j++) mixed ^= (uint64_t)ctx[j] * c->ple_mult[j];
        int base = (n - 2) * c->ple_per_gram;
        for (int g = 0; g < c->ple_per_gram; g++) {
            int h = base + g;
            int64_t row = (int64_t)(mixed % (uint64_t)c->ple_voc[h]) + c->ple_off[h];
            if (q38_gguf_read(&m->g, m->ple_table.t, row * m->ple_table.row_bytes, raw, m->ple_table.row_bytes))
                return -1;
            q38_dequant_row(m->ple_table.type, raw, emb + h * c->ple_dim, c->ple_dim);
        }
    }
    float *key = m->big, *value = m->out, *query = m->xn;
    matvec(&l->ple_key, emb, key);
    matvec(&l->ple_value, emb, value);
    float gate[16];
    for (int s = 0; s < hc; s++) {
        rmsnorm(key + s * E, key + s * E, vecf(&l->ple_norm_key) + s * E, E, c->eps);
        rmsnorm(query + s * E, res + s * E, vecf(&l->ple_norm_query) + s * E, E, c->eps);
        double d = 0;
        for (int i = 0; i < E; i++) d += (double)key[s * E + i] * query[s * E + i];
        float sc = (float)(d / sqrt((double)E));
        float mag = sqrtf(fabsf(sc) > 1e-6f ? fabsf(sc) : 1e-6f);
        gate[s] = sigmoidf_(sc > 0 ? mag : sc < 0 ? -mag : 0.0f);
    }
    if (m->tap) m->tap(m->tap_ctx, "ple_gate", gate, hc);
    /* gated value per stream, then grouped norm -> conv input */
    float *gated = m->big;            /* key no longer needed */
    for (int s = 0; s < hc; s++)
        for (int i = 0; i < E; i++) gated[s * E + i] = value[i] * gate[s];
    float *normed = m->xn;
    for (int s = 0; s < hc; s++) rmsnorm(normed + s * E, gated + s * E, vecf(&l->ple_norm_conv) + s * E, E, c->eps);
    /* depthwise causal conv, dilation ple_ngram: tap k reads (kern-1-k)*ngram positions back */
    const int hist = (c->ple_kern - 1) * c->ple_ngram;
    const float *wk = vecf(&l->ple_conv);
    for (int ch = 0; ch < hcE; ch++) {
        float acc = 0;
        for (int k = 0; k < c->ple_kern; k++) {
            int back = (c->ple_kern - 1 - k) * c->ple_ngram;
            float x = back == 0 ? normed[ch] : m->ple_hist[(int64_t)(hist - back) * hcE + ch];
            acc += wk[(int64_t)ch * c->ple_kern + k] * x;
        }
        res[ch] += gated[ch] + siluf_(acc);
    }
    memmove(m->ple_hist, m->ple_hist + hcE, (size_t)(hist - 1) * hcE * sizeof(float));
    memcpy(m->ple_hist + (int64_t)(hist - 1) * hcE, normed, (size_t)hcE * sizeof(float));
    return 0;
}

/* --------------------------------------------------------- Gated DeltaNet -- */
static void gdn(q38_model *m, int il, const float *x, float *y)
{
    const q38_cfg *c = &m->c;
    const q38_layer *l = &m->L[il];
    const int S = c->d_state, Hk = c->n_kh, Hv = c->n_vh;
    const int ch = 2 * Hk * S + Hv * S;
    float *qkv = m->big;                       /* [ch] */
    float *z = m->big + ch;                    /* [Hv*S] */
    float *o = m->big + ch + Hv * S;           /* [Hv*S] */
    float beta[256], g[256];
    matvec(&l->qkv, x, qkv);
    matvec(&l->gate, x, z);
    matvec(&l->beta, x, beta);
    matvec(&l->alpha, x, g);
    for (int h = 0; h < Hv; h++) {
        beta[h] = sigmoidf_(beta[h]);
        g[h] = softplusf_(g[h] + vecf(&l->dt_bias)[h]) * vecf(&l->a)[h];
    }
    /* ggml_ssm_conv over [state (conv_k-1 columns, oldest first) | this token], then silu */
    float *cs = m->conv_state[il];
    const float *wc = vecf(&l->conv);
    const int K = c->conv_k;
    for (int i = 0; i < ch; i++) {
        float acc = 0;
        for (int k = 0; k < K - 1; k++) acc += cs[(int64_t)k * ch + i] * wc[(int64_t)i * K + k];
        acc += qkv[i] * wc[(int64_t)i * K + K - 1];
        for (int k = 0; k < K - 2; k++) cs[(int64_t)k * ch + i] = cs[(int64_t)(k + 1) * ch + i];
        cs[(int64_t)(K - 2) * ch + i] = qkv[i];
        qkv[i] = siluf_(acc);
    }
    float *q = qkv, *k = qkv + Hk * S, *v = qkv + 2 * Hk * S;
    /* build_gdn_l2_norm: x / sqrt(sum(x^2) + eps) */
    for (int h = 0; h < 2 * Hk; h++) {
        float *p = qkv + h * S;
        double ss = 0;
        for (int i = 0; i < S; i++) ss += (double)p[i] * p[i];
        float sc = (float)(1.0 / sqrt(ss + c->eps));
        for (int i = 0; i < S; i++) p[i] *= sc;
    }
    const float scale = 1.0f / sqrtf((float)S);
    int h;
#pragma omp parallel for schedule(static)
    for (h = 0; h < Hv; h++) {
        const int kh = h % Hk;                 /* ggml tiles the key heads */
        float *St = m->ssm_state[il] + (int64_t)h * S * S;   /* [i][j] */
        const float *kk = k + kh * S, *qq = q + kh * S, *vv = v + h * S;
        const float eg = expf(g[h]);
        float sk[512], d[512];
        for (int j = 0; j < S; j++) sk[j] = 0;
        for (int i = 0; i < S; i++) {
            float *row = St + (int64_t)i * S;
            for (int j = 0; j < S; j++) { row[j] *= eg; sk[j] += row[j] * kk[i]; }
        }
        for (int j = 0; j < S; j++) d[j] = (vv[j] - sk[j]) * beta[h];
        float *oo = o + h * S;
        for (int j = 0; j < S; j++) oo[j] = 0;
        for (int i = 0; i < S; i++) {
            float *row = St + (int64_t)i * S;
            for (int j = 0; j < S; j++) { row[j] += kk[i] * d[j]; oo[j] += row[j] * qq[i]; }
        }
        for (int j = 0; j < S; j++) oo[j] *= scale;
    }
    if (m->tap) {
        char nm[64];
        snprintf(nm, sizeof nm, "attn_output-%d", il);
        m->tap(m->tap_ctx, nm, o, (int64_t)Hv * S);
    }
    /* build_norm_gated: rmsnorm(o) * ssm_norm * sigmoid(z) */
    for (int h = 0; h < Hv; h++) {
        rmsnorm(o + h * S, o + h * S, vecf(&l->ssm_norm), S, c->eps);
        for (int j = 0; j < S; j++) o[h * S + j] *= sigmoidf_(z[h * S + j]);
    }
    matvec(&l->ssm_out, o, y);
}

/* ------------------------------------------------------------------ attention -- */
static void rope(float *x, int n_rot, int pos, double base)
{
    const int half = n_rot / 2;
    for (int i = 0; i < half; i++) {
        double ang = pos * pow(base, -2.0 * i / n_rot);
        float cs = (float)cos(ang), sn = (float)sin(ang);
        float x0 = x[i], x1 = x[i + half];
        x[i] = x0 * cs - x1 * sn;
        x[i + half] = x0 * sn + x1 * cs;
    }
}

/* The lightning indexer (llama.cpp build_qsa_top_k, set_input_qsa) for one token:
 * stores this position's raw key, pools a block when its last member arrives, and when the
 * context exceeds max_dense_ctx returns the cells to attend (*T of them): the incomplete tail
 * block (always), then whole blocks in order of sum_h relu(q_h . k_block), the last one cut
 * to fit (its lowest positions first). NULL: attend every cell (dense). */
static int g_sel[1 << 16];
static const float *g_score;

static int by_score(const void *a, const void *b)
{
    const int i = *(const int *)a, j = *(const int *)b;
    if (g_score[i] != g_score[j]) return g_score[i] > g_score[j] ? -1 : 1;
    return i - j;
}

static const int *indexer_select(q38_model *m, int il, const float *x, int *T)
{
    const q38_cfg *c = &m->c;
    const q38_layer *l = &m->L[il];
    if (c->idx_top_k <= 0 || !m->idx_raw[il]) return NULL;
    const int pos = m->n_past, R = c->idx_ratio, DI = c->idx_dim, W = c->max_dense_ctx;
    float *raw = m->idx_raw[il] + (int64_t)pos * DI;
    matvec(&l->idx_k, x, raw);
    if ((pos + 1) % R == 0) {
        const int b = pos / R;
        float *pk = m->idx_pool[il] + (int64_t)b * DI;
        for (int i = 0; i < DI; i++) {
            float sum = 0;
            for (int k = 0; k < R; k++) sum += m->idx_raw[il][(int64_t)(b * R + k) * DI + i];
            pk[i] = sum / R;
        }
        rmsnorm(pk, pk, vecf(&l->idx_kn), DI, c->eps);
        rope(pk, c->n_rot, b * R, c->rope_base);
    }
    if (pos + 1 <= W) return NULL;

    float qi[16 * 512];
    if (c->idx_heads * DI > (int)(sizeof qi / sizeof qi[0])) return NULL;
    matvec(&l->idx_q, x, qi);
    for (int hh = 0; hh < c->idx_heads; hh++) {
        rmsnorm(qi + hh * DI, qi + hh * DI, vecf(&l->idx_qn), DI, c->eps);
        rope(qi + hh * DI, c->n_rot, pos, c->rope_base);
    }
    const int tail_start = (pos + 1) / R * R, nfull = tail_start / R;
    float *score = (float *)malloc((size_t)nfull * sizeof(float));
    int *order = (int *)malloc((size_t)nfull * sizeof(int));
    for (int b = 0; b < nfull; b++) {
        const float *pk = m->idx_pool[il] + (int64_t)b * DI;
        float s = 0;
        for (int hh = 0; hh < c->idx_heads; hh++) {
            float d = 0;
            for (int i = 0; i < DI; i++) d += qi[hh * DI + i] * pk[i];
            s += d > 0 ? d : 0;
        }
        score[b] = s;
        order[b] = b;
    }
    /* blocks by score, highest first; equal scores: lower block first */
    g_score = score;
    qsort(order, (size_t)nfull, sizeof(int), by_score);
    int n = 0;
    for (int p = tail_start; p <= pos && n < W; p++) g_sel[n++] = p;
    for (int i = 0; i < nfull && n < W; i++)
        for (int k = 0; k < R && n < W; k++) g_sel[n++] = order[i] * R + k;
    free(score);
    free(order);
    *T = n;
    if (m->tap) {
        float *f = (float *)malloc((size_t)n * sizeof(float));
        char nm[64];
        for (int i = 0; i < n; i++) f[i] = (float)g_sel[i];
        snprintf(nm, sizeof nm, "indexer_top_k-%d", il);
        m->tap(m->tap_ctx, nm, f, n);
        free(f);
    }
    return g_sel;
}

static void attn(q38_model *m, int il, const float *x, float *y)
{
    const q38_cfg *c = &m->c;
    const q38_layer *l = &m->L[il];
    const int H = c->n_head, Hkv = c->n_head_kv, D = c->head_dim, pos = m->n_past;
    float *qf = m->big;                       /* [H][2][D]: q | gate per head */
    float *out = m->big + (int64_t)H * 2 * D; /* [H][D] */
    float *kc = m->kcache[il] + (int64_t)pos * Hkv * D, *vc = m->vcache[il] + (int64_t)pos * Hkv * D;
    matvec(&l->wq, x, qf);
    matvec(&l->wk, x, kc);
    matvec(&l->wv, x, vc);
    for (int h = 0; h < H; h++) {
        float *q = qf + (int64_t)h * 2 * D;
        rmsnorm(q, q, vecf(&l->q_norm), D, c->eps);
        rope(q, c->n_rot, pos, c->rope_base);
    }
    for (int h = 0; h < Hkv; h++) {
        rmsnorm(kc + h * D, kc + h * D, vecf(&l->k_norm), D, c->eps);
        rope(kc + h * D, c->n_rot, pos, c->rope_base);
    }
    const int r = H / Hkv;
    /* the cells this query attends: all of them up to max_dense_ctx, the indexer's choice after */
    int T = pos + 1;
    const int *sel = indexer_select(m, il, x, &T);
    const float scale = 1.0f / sqrtf((float)D);
    int h;
#pragma omp parallel for schedule(static)
    for (h = 0; h < H; h++) {
        const int kvh = h / r;
        const float *q = qf + (int64_t)h * 2 * D, *gate = q + D;
        double *sc = (double *)malloc((size_t)T * sizeof(double));
        double mx = -1e300, sum = 0;
        for (int t = 0; t < T; t++) {
            const float *kk = m->kcache[il] + ((int64_t)(sel ? sel[t] : t) * Hkv + kvh) * D;
            double s = 0;
            for (int i = 0; i < D; i++) s += (double)q[i] * kk[i];
            sc[t] = s * scale;
            if (sc[t] > mx) mx = sc[t];
        }
        for (int t = 0; t < T; t++) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
        float *o = out + (int64_t)h * D;
        for (int i = 0; i < D; i++) {
            double a = 0;
            for (int t = 0; t < T; t++) a += sc[t] * m->vcache[il][((int64_t)(sel ? sel[t] : t) * Hkv + kvh) * D + i];
            o[i] = (float)(a / sum) * sigmoidf_(gate[i]);
        }
        free(sc);
    }
    matvec(&l->wo, out, y);
    if (m->tap) {
        char nm[64];
        snprintf(nm, sizeof nm, "attn_output-%d", il);
        m->tap(m->tap_ctx, nm, y, c->n_embd);
    }
}

/* ------------------------------------------------------------- expert cache -- */
static int64_t expert_bytes(const q38_model *m, int il)
{
    const q38_layer *l = &m->L[il];
    const int64_t F = m->c.ff_exp, E = m->c.n_embd;
    return F * l->ex_gate.row_bytes + F * l->ex_up.row_bytes + E * l->ex_down.row_bytes;
}

static void lru_unlink(q38_model *m, q38_slot *s)
{
    if (s->prev) s->prev->next = s->next; else m->lru_head = s->next;
    if (s->next) s->next->prev = s->prev; else m->lru_tail = s->prev;
    s->prev = s->next = NULL;
}

static void lru_front(q38_model *m, q38_slot *s)
{
    s->prev = NULL;
    s->next = m->lru_head;
    if (m->lru_head) m->lru_head->prev = s;
    m->lru_head = s;
    if (!m->lru_tail) m->lru_tail = s;
}

/* Frees least-recently-used experts not used by the current token until need more bytes fit. */
static void lru_make_room(q38_model *m, int64_t need)
{
    q38_slot *s = m->lru_tail;
    while (s && m->cache_used + need > m->cache_budget) {
        q38_slot *prev = s->prev;
        if (s->stamp != m->token_stamp && !s->busy) {
            lru_unlink(m, s);
            m->table[s->key] = NULL;
            m->cache_used -= s->bytes;
            m->cache_count--;
            free(s->data);
            free(s);
        }
        s = prev;
    }
}

void q38_model_set_cache(q38_model *m, double gb)
{
    int64_t all = 0;
    for (int il = 0; il < m->c.n_layer; il++) all += expert_bytes(m, il) * m->c.n_exp;
    int64_t budget = (int64_t)(gb * 1e9);
    if (gb <= 0) {
        int64_t avail = 0;
#ifdef _WIN32
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof ms;
        if (GlobalMemoryStatusEx(&ms)) avail = (int64_t)ms.ullAvailPhys;
#endif
        budget = avail - (int64_t)8e9;       /* leave room for the OS, the session and the page cache */
        if (budget < (int64_t)2e9) budget = (int64_t)2e9;
    }
    if (budget > all) budget = all;
    m->cache_budget = budget;
    lru_make_room(m, 0);
}

/* The expert's three matrices in RAM, reading them from the GGUF on a miss. Misses of one
 * layer are read in parallel by the caller; this only allocates. */
static q38_slot *slot_alloc(q38_model *m, int il, int e)
{
    const int64_t b = expert_bytes(m, il);
    lru_make_room(m, b);
    q38_slot *s = (q38_slot *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->data = (uint8_t *)malloc((size_t)b);
    if (!s->data) { free(s); return NULL; }
    s->bytes = b;
    s->key = il * m->c.n_exp + e;
    s->stamp = m->token_stamp;
    m->table[s->key] = s;
    m->cache_used += b;
    m->cache_count++;
    lru_front(m, s);
    return s;
}

static int slot_read(q38_model *m, int il, int e, q38_slot *s)
{
    const q38_layer *l = &m->L[il];
    const int64_t F = m->c.ff_exp, E = m->c.n_embd;
    const int64_t bg = F * l->ex_gate.row_bytes, bu = F * l->ex_up.row_bytes, bd = E * l->ex_down.row_bytes;
    if (q38_gguf_pread(&m->g, l->ex_gate.t, (int64_t)e * bg, s->data, bg)) return -1;
    if (q38_gguf_pread(&m->g, l->ex_up.t, (int64_t)e * bu, s->data + bg, bu)) return -1;
    if (q38_gguf_pread(&m->g, l->ex_down.t, (int64_t)e * bd, s->data + bg + bu, bd)) return -1;
    return 0;
}

/* ------------------------------------------------------------------------ MoE -- */
/* Router: softmax over the experts, the K largest (ties: lower id first, as a stable
 * argsort), weights renormalised over the K. */
static void route(q38_model *m, int il, const float *logits, int *top, float *w)
{
    const q38_cfg *c = &m->c;
    const int NE = c->n_exp, K = c->n_exp_used;
    double mx = -1e300, sum = 0;
    for (int e = 0; e < NE; e++) if (logits[e] > mx) mx = logits[e];
    double pr[1024];
    for (int e = 0; e < NE; e++) { pr[e] = exp(logits[e] - mx); sum += pr[e]; }
    int nb = 0;
    for (int e = 0; e < NE; e++) {
        pr[e] /= sum;
        int j = nb < K ? nb++ : K;
        if (j == K) {
            if (!(pr[e] > pr[top[K - 1]])) continue;
            j = K - 1;
        }
        while (j > 0 && pr[top[j - 1]] < pr[e]) { top[j] = top[j - 1]; j--; }
        top[j] = e;
    }
    double wsum = 0;
    for (int k = 0; k < K; k++) wsum += pr[top[k]];
    for (int k = 0; k < K; k++) w[k] = (float)(pr[top[k]] / wsum);
    if (m->use)
        for (int k = 0; k < K; k++) m->use[il * NE + top[k]] += 1.0f;
    if (m->tap) {
        float ft[64];
        char nm[64];
        for (int k = 0; k < K; k++) ft[k] = (float)top[k];
        snprintf(nm, sizeof nm, "experts-%d", il);
        m->tap(m->tap_ctx, nm, ft, K);
    }
}

/* y = sum_k w[k] * expert_ids[k](x) over n routed experts of layer il, from the RAM cache
 * (misses read in parallel); with_shared adds the gated shared expert. n may be 0. */
static int experts_compute(q38_model *m, int il, const float *x, const int *ids, const float *wts, int n,
                           int with_shared, float *y)
{
    const q38_cfg *c = &m->c;
    const q38_layer *l = &m->L[il];
    const int E = c->n_embd, F = c->ff_exp, NE = c->n_exp;

    /* resident experts: hits move to the front, misses get a slot and are read together */
    q38_slot *sl[64];
    int miss[64], nmiss = 0;
    for (int k = 0; k < n; k++) {
        q38_slot *s = m->table[il * NE + ids[k]];
        if (s) {
            s->stamp = m->token_stamp;
            lru_unlink(m, s);
            lru_front(m, s);
            m->cache_hits++;
        } else {
            s = slot_alloc(m, il, ids[k]);
            if (!s) return -1;
            miss[nmiss++] = k;
            m->cache_misses++;
        }
        sl[k] = s;
    }
    if (nmiss) {
        double t0 = now_s();
        int bad = 0, i;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : bad)
        for (i = 0; i < nmiss; i++) bad += slot_read(m, il, ids[miss[i]], sl[miss[i]]) != 0;
        m->t_read += now_s() - t0;
        for (i = 0; i < nmiss; i++) m->expert_bytes_read += sl[miss[i]]->bytes;
        if (bad) { fprintf(stderr, "q38: reading experts of layer %d failed\n", il); return -1; }
    }

    /* the routed experts and the shared one (index n) as one batch of matrices */
    const int NX = with_shared ? n + 1 : n;
    const uint8_t *wg[65], *wu[65], *wd[65];
    int tg[65], tu[65], td[65];
    int64_t rg[65], ru[65], rd[65];
    for (int k = 0; k < n; k++) {
        const int64_t bg = (int64_t)F * l->ex_gate.row_bytes, bu = (int64_t)F * l->ex_up.row_bytes;
        wg[k] = sl[k]->data; wu[k] = sl[k]->data + bg; wd[k] = sl[k]->data + bg + bu;
        tg[k] = l->ex_gate.type; tu[k] = l->ex_up.type; td[k] = l->ex_down.type;
        rg[k] = l->ex_gate.row_bytes; ru[k] = l->ex_up.row_bytes; rd[k] = l->ex_down.row_bytes;
    }
    if (with_shared) {
        wg[n] = l->sh_gate.data; wu[n] = l->sh_up.data; wd[n] = l->sh_down.data;
        tg[n] = l->sh_gate.type; tu[n] = l->sh_up.type; td[n] = l->sh_down.type;
        rg[n] = l->sh_gate.row_bytes; ru[n] = l->sh_up.row_bytes; rd[n] = l->sh_down.row_bytes;
        if (c->ff_shexp != F) { fprintf(stderr, "q38: shared expert width %d != %d\n", c->ff_shexp, F); return -1; }
    }

    double t0 = now_s();
    float *gu = m->moe_gu, *h = m->moe_h, *dn = m->moe_dn;
    if (g_naive) {
        for (int k = 0; k < NX; k++) {
            matvec_rows(tg[k], wg[k], rg[k], E, F, x, gu + (int64_t)k * 2 * F);
            matvec_rows(tu[k], wu[k], ru[k], E, F, x, gu + (int64_t)k * 2 * F + F);
            for (int i = 0; i < F; i++) h[k * F + i] = siluf_(gu[k * 2 * F + i]) * gu[k * 2 * F + F + i];
            matvec_rows(td[k], wd[k], rd[k], F, E, h + (int64_t)k * F, dn + (int64_t)k * E);
        }
    } else if (NX > 0) {
        float bsx[MAX_NE0 / 32];
        q38_xvec vx;
        q38_xprep(&vx, x, E, bsx);
        int r, nr = NX * 2 * F;
#pragma omp parallel for schedule(static)
        for (r = 0; r < nr; r++) {
            const int k = r / (2 * F), j = r % (2 * F);
            gu[r] = j < F ? q38_dot(tg[k], wg[k] + (int64_t)j * rg[k], &vx)
                          : q38_dot(tu[k], wu[k] + (int64_t)(j - F) * ru[k], &vx);
        }
        q38_xvec vh[65];
        for (int k = 0; k < NX; k++) {
            for (int i = 0; i < F; i++) h[k * F + i] = siluf_(gu[k * 2 * F + i]) * gu[k * 2 * F + F + i];
            q38_xprep(&vh[k], h + (int64_t)k * F, F, m->moe_bsum + (int64_t)k * (F / 32));
        }
        nr = NX * E;
#pragma omp parallel for schedule(static)
        for (r = 0; r < nr; r++) {
            const int k = r / E, i = r % E;
            dn[r] = q38_dot(td[k], wd[k] + (int64_t)i * rd[k], &vh[k]);
        }
    }
    float sg = 0;
    if (with_shared) {
        matvec(&l->shexp_gate_inp, x, &sg);
        sg = sigmoidf_(sg);
    }
    for (int i = 0; i < E; i++) {
        float a = 0;
        for (int k = 0; k < n; k++) a += wts[k] * dn[(int64_t)k * E + i];
        y[i] = with_shared ? a + sg * dn[(int64_t)n * E + i] : a;
    }
    m->t_experts += now_s() - t0;
    return 0;
}

/* The MoE block on the host (the CPU path). */
static int moe(q38_model *m, int il, const float *x, float *y)
{
    float *logits = m->tmp;
    int top[64];
    float w[64];
    matvec(&m->L[il].router, x, logits);
    route(m, il, logits, top, w);
    return experts_compute(m, il, x, top, w, m->c.n_exp_used, 1, y);
}

/* Routed experts for many tokens of one layer at once (the batched prompt pass): the pairs
 * (token, expert, weight) are grouped by expert, each expert is read from the RAM cache once
 * and applied to all of its tokens (the weight rows stay in the CPU caches across them), and
 * every token's sum is then formed over its pairs in the order given. y[t] is overwritten
 * for every t < n_tok (zero for a token without pairs). */
int q38_model_experts_batch(q38_model *m, int il, const float *X, int n_tok, const int *ptok, const int *pexp,
                            const float *pw, int npair, float *Y)
{
    const q38_cfg *c = &m->c;
    const q38_layer *l = &m->L[il];
    const int E = c->n_embd, F = c->ff_exp, NE = c->n_exp;
    for (int64_t i = 0; i < (int64_t)n_tok * E; i++) Y[i] = 0.0f;
    if (npair <= 0) return 0;
    /* protect only this layer's experts from eviction: a whole batch's would not fit */
    m->token_stamp++;

    /* distinct experts and their pairs */
    int *first = (int *)malloc((size_t)NE * sizeof(int)), *next = (int *)malloc((size_t)npair * sizeof(int));
    int *ue = (int *)malloc((size_t)NE * sizeof(int));
    q38_slot **us = (q38_slot **)malloc((size_t)NE * sizeof(q38_slot *));
    float *gu = (float *)malloc((size_t)npair * 2 * F * sizeof(float)), *h = (float *)malloc((size_t)npair * F * sizeof(float));
    float *dn = (float *)malloc((size_t)npair * E * sizeof(float));
    float *xb = (float *)malloc((size_t)n_tok * (E / 32) * sizeof(float)), *hb = (float *)malloc((size_t)npair * (F / 32) * sizeof(float));
    q38_xvec *vx = (q38_xvec *)malloc((size_t)n_tok * sizeof(q38_xvec)), *vh = (q38_xvec *)malloc((size_t)npair * sizeof(q38_xvec));
    int rc = -1;
    if (!first || !next || !ue || !us || !gu || !h || !dn || !xb || !hb || !vx || !vh) goto out;
    for (int e = 0; e < NE; e++) first[e] = -1;
    int nu = 0;
    for (int p = npair - 1; p >= 0; p--) {             /* lists in pair order */
        const int e = pexp[p];
        if (first[e] < 0) ue[nu++] = e;
        next[p] = first[e];
        first[e] = p;
    }

    /* resident in RAM: hits to the front, misses read together */
    {
        int miss[1024], nmiss = 0;
        for (int i = 0; i < nu; i++) {
            q38_slot *s = m->table[il * NE + ue[i]];
            if (s) {
                s->stamp = m->token_stamp;
                lru_unlink(m, s);
                lru_front(m, s);
                m->cache_hits++;
            } else {
                s = slot_alloc(m, il, ue[i]);
                if (!s) goto out;
                if (nmiss < 1024) miss[nmiss++] = i;
                m->cache_misses++;
            }
            us[i] = s;
        }
        if (nmiss) {
            const double t0 = now_s();
            int bad = 0, i;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : bad)
            for (i = 0; i < nmiss; i++) bad += slot_read(m, il, ue[miss[i]], us[miss[i]]) != 0;
            m->t_read += now_s() - t0;
            for (i = 0; i < nmiss; i++) m->expert_bytes_read += us[miss[i]]->bytes;
            if (bad) { fprintf(stderr, "q38: reading experts of layer %d failed\n", il); goto out; }
        }
    }

    const double t0 = now_s();
    for (int t = 0; t < n_tok; t++) q38_xprep(&vx[t], X + (int64_t)t * E, E, xb + (int64_t)t * (E / 32));
    const int64_t bg = (int64_t)F * l->ex_gate.row_bytes, bu = (int64_t)F * l->ex_up.row_bytes;
    /* gate and up: work item = (expert, row); inner loop over the expert's tokens */
    {
        const int items = nu * 2 * F;
        int it;
#pragma omp parallel for schedule(static)
        for (it = 0; it < items; it++) {
            const int i = it / (2 * F), j = it % (2 * F);
            const uint8_t *row = j < F ? us[i]->data + (int64_t)j * l->ex_gate.row_bytes
                                       : us[i]->data + bg + (int64_t)(j - F) * l->ex_up.row_bytes;
            const int ty = j < F ? l->ex_gate.type : l->ex_up.type;
            for (int p = first[ue[i]]; p >= 0; p = next[p]) gu[(int64_t)p * 2 * F + j] = q38_dot(ty, row, &vx[ptok[p]]);
        }
    }
    for (int p = 0; p < npair; p++) {
        for (int j = 0; j < F; j++) h[(int64_t)p * F + j] = siluf_(gu[(int64_t)p * 2 * F + j]) * gu[(int64_t)p * 2 * F + F + j];
        q38_xprep(&vh[p], h + (int64_t)p * F, F, hb + (int64_t)p * (F / 32));
    }
    {
        const int items = nu * E;
        int it;
#pragma omp parallel for schedule(static)
        for (it = 0; it < items; it++) {
            const int i = it / E, r = it % E;
            const uint8_t *row = us[i]->data + bg + bu + (int64_t)r * l->ex_down.row_bytes;
            for (int p = first[ue[i]]; p >= 0; p = next[p]) dn[(int64_t)p * E + r] = q38_dot(l->ex_down.type, row, &vh[p]);
        }
    }
    for (int p = 0; p < npair; p++) {
        float *y = Y + (int64_t)ptok[p] * E;
        const float *d = dn + (int64_t)p * E;
        for (int r = 0; r < E; r++) y[r] += pw[p] * d[r];
    }
    m->t_experts += now_s() - t0;
    rc = 0;
out:
    free(first); free(next); free(ue); free(us); free(gu); free(h); free(dn); free(xb); free(hb); free(vx); free(vh);
    return rc;
}

int q38_model_experts_resident(q38_model *m, int il, const int *ids, int n, const uint8_t **data)
{
    const int NE = m->c.n_exp;
    m->token_stamp++;
    q38_slot **sl = (q38_slot **)malloc((size_t)(n ? n : 1) * sizeof(q38_slot *));
    int *miss = (int *)malloc((size_t)(n ? n : 1) * sizeof(int)), nmiss = 0, rc = 0;
    if (!sl || !miss) { free(sl); free(miss); return -1; }
    for (int i = 0; i < n; i++) {
        q38_slot *s = m->table[il * NE + ids[i]];
        if (s) {
            s->stamp = m->token_stamp;
            lru_unlink(m, s);
            lru_front(m, s);
            m->cache_hits++;
        } else {
            s = slot_alloc(m, il, ids[i]);
            if (!s) { rc = -1; break; }
            miss[nmiss++] = i;
            m->cache_misses++;
        }
        sl[i] = s;
    }
    if (!rc && nmiss) {
        const double t0 = now_s();
        int bad = 0, i;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : bad)
        for (i = 0; i < nmiss; i++) bad += slot_read(m, il, ids[miss[i]], sl[miss[i]]) != 0;
        m->t_read += now_s() - t0;
        for (i = 0; i < nmiss; i++) m->expert_bytes_read += sl[miss[i]]->bytes;
        if (bad) rc = -1;
    }
    if (!rc) for (int i = 0; i < n; i++) data[i] = sl[i]->data;
    free(sl);
    free(miss);
    return rc;
}

void q38_model_route(q38_model *m, int il, const float *logits, int *top, float *w) { route(m, il, logits, top, w); }

int q38_model_experts(q38_model *m, int il, const float *x, const int *ids, const float *w, int n, float *y)
{
    return experts_compute(m, il, x, ids, w, n, 0, y);
}

int q38_model_expert_slot(q38_model *m, int il, int e, const uint8_t **data, int64_t *bytes)
{
    q38_slot *s = m->table[il * m->c.n_exp + e];
    if (!s) return -1;
    *data = s->data;
    *bytes = s->bytes;
    return 0;
}

void *q38_model_expert_hold(q38_model *m, int il, int e, const uint8_t **data, int64_t *bytes)
{
    q38_slot *s = m->table[il * m->c.n_exp + e];
    if (!s) return NULL;
    s->busy++;
    *data = s->data;
    *bytes = s->bytes;
    return s;
}

void q38_model_expert_release(void *handle)
{
    if (handle) ((q38_slot *)handle)->busy--;
}

int q38_model_ple(q38_model *m, int il, float *res, int tok) { return ple(m, il, res, tok); }

/* ---------------------------------------------------------- usage profile -- */
#define Q38_PROFILE_MAGIC 0x31503851u   /* "Q8P1" */

int q38_model_profile_load(q38_model *m, const char *path)
{
    const int n = m->c.n_layer * m->c.n_exp;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t hdr[3];
    int ok = fread(hdr, 4, 3, f) == 3 && hdr[0] == Q38_PROFILE_MAGIC && (int)hdr[1] == m->c.n_layer &&
             (int)hdr[2] == m->c.n_exp;
    float *v = ok ? (float *)malloc((size_t)n * sizeof(float)) : NULL;
    ok = ok && v && fread(v, sizeof(float), (size_t)n, f) == (size_t)n;
    fclose(f);
    if (ok) for (int i = 0; i < n; i++) m->use[i] += v[i];
    free(v);
    return ok ? 0 : -1;
}

int q38_model_profile_save(q38_model *m, const char *path)
{
    const int n = m->c.n_layer * m->c.n_exp;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const uint32_t hdr[3] = { Q38_PROFILE_MAGIC, (uint32_t)m->c.n_layer, (uint32_t)m->c.n_exp };
    int ok = fwrite(hdr, 4, 3, f) == 3 && fwrite(m->use, sizeof(float), (size_t)n, f) == (size_t)n;
    ok = fclose(f) == 0 && ok;
    return ok ? 0 : -1;
}

static const float *g_use;
static int by_use(const void *a, const void *b)
{
    const float x = g_use[*(const int *)a], y = g_use[*(const int *)b];
    return x > y ? -1 : x < y ? 1 : 0;
}

int q38_model_preload(q38_model *m, double gb)
{
    const q38_cfg *c = &m->c;
    const int n = c->n_layer * c->n_exp;
    int *ord = (int *)malloc((size_t)n * sizeof(int));
    if (!ord) return 0;
    int nz = 0;
    for (int i = 0; i < n; i++) if (m->use[i] > 0) ord[nz++] = i;
    g_use = m->use;
    qsort(ord, (size_t)nz, sizeof(int), by_use);
    int64_t limit = (int64_t)(gb * 1e9);
    if (limit > m->cache_budget) limit = m->cache_budget;
    int done = 0;
    int64_t bytes = 0;
    for (int i0 = 0; i0 < nz && bytes < limit; i0 += 64) {
        q38_slot *sl[64];
        int key[64], cnt = 0;
        for (int i = i0; i < nz && i < i0 + 64; i++) {
            const int il = ord[i] / c->n_exp, e = ord[i] % c->n_exp;
            if (m->table[ord[i]]) continue;
            if (bytes + expert_bytes(m, il) > limit || m->cache_used + expert_bytes(m, il) > m->cache_budget) break;
            q38_slot *s = slot_alloc(m, il, e);
            if (!s) break;
            s->stamp = m->token_stamp - 1;       /* not "in use": evictable later */
            sl[cnt] = s;
            key[cnt] = ord[i];
            cnt++;
            bytes += s->bytes;
        }
        if (!cnt) break;
        int bad = 0, i;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : bad)
        for (i = 0; i < cnt; i++) bad += slot_read(m, key[i] / c->n_exp, key[i] % c->n_exp, sl[i]) != 0;
        if (bad) break;
        done += cnt;
    }
    /* reads went most-used first, so the most used sit at the LRU tail: put them back in
     * front, least used first, most used last (at the head) */
    for (int i = nz - 1; i >= 0; i--) {
        q38_slot *s = m->table[ord[i]];
        if (s) { lru_unlink(m, s); lru_front(m, s); }
    }
    free(ord);
    return done;
}

/* -------------------------------------------------------------------- forward -- */
const float *q38_model_forward(q38_model *m, int token)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, hc = c->hc, hcE = hc * E;
    char nm[64];
    if (token < 0 || token >= c->n_vocab) { fprintf(stderr, "q38: token %d out of range\n", token); return NULL; }
    if (m->n_past >= m->max_ctx) {
        fprintf(stderr, "q38: context full (%d tokens)\n", m->max_ctx);
        return NULL;
    }
    double t0 = now_s();
    m->token_stamp++;
#ifdef Q38_CUDA
    if (m->gpu) {
        if (q38_gpu_forward(m, token)) return NULL;
        m->toks[m->n_past++] = token;
        m->t_trunk += now_s() - t0;
        return m->logits;
    }
#endif
    float *x = m->mixed;
    q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)token * m->tok_embd.row_bytes, x, E);
    for (int s = 0; s < hc; s++) memcpy(m->res + s * E, x, (size_t)E * sizeof(float));
    float *y = m->tmp2 + 4 * 4096;   /* block output, kept apart from the MoE scratch */
    for (int il = 0; il < c->n_layer; il++) {
        q38_layer *l = &m->L[il];
        if (il == c->ple_layer && ple(m, il, m->res, token)) return NULL;
        hc_mix(m, m->res, &l->hc_attn_norm, &l->hc_attn_down, &l->hc_attn_up, &l->hc_attn_inject, m->mixed, m->inject);
        if (m->tap) { snprintf(nm, sizeof nm, "hc_mixed_attn-%d", il); m->tap(m->tap_ctx, nm, m->mixed, E); }
        if (l->recurrent) gdn(m, il, m->mixed, y);
        else attn(m, il, m->mixed, y);
        hc_combine(m, m->res, y, m->inject);
        hc_mix(m, m->res, &l->hc_ffn_norm, &l->hc_ffn_down, &l->hc_ffn_up, &l->hc_ffn_inject, m->mixed, m->inject);
        if (m->tap) { snprintf(nm, sizeof nm, "hc_mixed_ffn-%d", il); m->tap(m->tap_ctx, nm, m->mixed, E); }
        if (moe(m, il, m->mixed, y)) return NULL;
        if (m->tap) { snprintf(nm, sizeof nm, "ffn_out-%d", il); m->tap(m->tap_ctx, nm, y, E); }
        hc_combine(m, m->res, y, m->inject);
        if (m->tap) { snprintf(nm, sizeof nm, "l_last-%d", il); m->tap(m->tap_ctx, nm, m->res, hcE); }
    }
    hc_mix(m, m->res, &m->out_hc_norm, &m->out_hc_down, &m->out_hc_up, NULL, m->mixed, NULL);
    if (m->tap) m->tap(m->tap_ctx, "result_norm", m->mixed, E);
    matvec(&m->output, m->mixed, m->logits);
    if (m->tap) m->tap(m->tap_ctx, "logits", m->logits, c->n_vocab);
    m->toks[m->n_past++] = token;
    m->t_trunk += now_s() - t0;
    return m->logits;
}

const float *q38_model_forward_batch(q38_model *m, const int *toks, int n)
{
    if (n <= 0) return NULL;
    if (m->n_past + n > m->max_ctx) {
        fprintf(stderr, "q38: context full (%d + %d tokens > %d)\n", m->n_past, n, m->max_ctx);
        return NULL;
    }
#ifdef Q38_CUDA
    if (m->gpu && n > 1 && !m->tap && !(getenv("Q38_NO_BATCH") && atoi(getenv("Q38_NO_BATCH")))) {
        const double t0 = now_s();
        m->token_stamp++;
        if (q38_gpu_forward_batch(m, toks, n)) return NULL;
        m->t_trunk += now_s() - t0;
        return m->logits;
    }
#endif
    const float *lg = NULL;
    for (int i = 0; i < n; i++) if (!(lg = q38_model_forward(m, toks[i]))) return NULL;
    return lg;
}

/* ------------------------------------------------------------------- lifetime -- */
void q38_model_reset(q38_model *m)
{
#ifdef Q38_CUDA
    if (m->gpu) q38_gpu_reset(m);
#endif
    const q38_cfg *c = &m->c;
    const int ch = 2 * c->n_kh * c->d_state + c->n_vh * c->d_state;
    m->n_past = 0;
    for (int il = 0; il < c->n_layer; il++) {
        if (m->conv_state[il]) memset(m->conv_state[il], 0, (size_t)(c->conv_k - 1) * ch * sizeof(float));
        if (m->ssm_state[il]) memset(m->ssm_state[il], 0, (size_t)c->n_vh * c->d_state * c->d_state * sizeof(float));
    }
    if (m->ple_hist)
        memset(m->ple_hist, 0, (size_t)(c->ple_kern - 1) * c->ple_ngram * c->hc * c->n_embd * sizeof(float));
}

int q38_model_open(q38_model *m, const char *path, int max_ctx)
{
    memset(m, 0, sizeof *m);
#ifdef _OPENMP
    /* one thread per physical core: with SMT siblings the CPU experts ran slower (measured
     * 2026-10-03: 8 threads 21.97 tok/s, 16 threads 20.32 on an 8-core / 16-thread Ryzen) */
    if (!getenv("OMP_NUM_THREADS") && omp_get_num_procs() >= 4) omp_set_num_threads(omp_get_num_procs() / 2);
#endif
    if (q38_gguf_open(&m->g, path)) return -1;
    if (read_cfg(&m->c, &m->g)) { q38_model_close(m); return -1; }
    q38_cfg *c = &m->c;
    if (getenv("Q38_DENSE") && atoi(getenv("Q38_DENSE"))) {
        /* control for validation: attend every cell at every length (not the model) */
        c->idx_top_k = 0;
        c->max_dense_ctx = 1 << 30;
    }
    if (max_ctx <= 0) max_ctx = 8192;
    if (c->idx_top_k <= 0 && max_ctx > c->max_dense_ctx) max_ctx = c->max_dense_ctx;
    if (max_ctx > (1 << 16)) max_ctx = 1 << 16;
    m->max_ctx = max_ctx;
    double t0 = now_s();
    if (bind_all(m)) { q38_model_close(m); return -1; }
    double t1 = now_s();

    const int E = c->n_embd, hcE = c->hc * E;
    const int ch = 2 * c->n_kh * c->d_state + c->n_vh * c->d_state;
    for (int il = 0; il < c->n_layer; il++) {
        if (m->L[il].recurrent) {
            m->conv_state[il] = (float *)calloc((size_t)(c->conv_k - 1) * ch, sizeof(float));
            m->ssm_state[il] = (float *)calloc((size_t)c->n_vh * c->d_state * c->d_state, sizeof(float));
        } else {
            m->kcache[il] = (float *)calloc((size_t)max_ctx * c->n_head_kv * c->head_dim, sizeof(float));
            m->vcache[il] = (float *)calloc((size_t)max_ctx * c->n_head_kv * c->head_dim, sizeof(float));
            if (c->idx_top_k > 0) {
                m->idx_raw[il] = (float *)calloc((size_t)max_ctx * c->idx_dim, sizeof(float));
                m->idx_pool[il] = (float *)calloc((size_t)(max_ctx / c->idx_ratio + 1) * c->idx_dim, sizeof(float));
            }
        }
    }
    if (c->ple_layer >= 0)
        m->ple_hist = (float *)calloc((size_t)(c->ple_kern - 1) * c->ple_ngram * hcE, sizeof(float));
    m->toks = (int *)calloc((size_t)max_ctx, sizeof(int));
    m->res = (float *)calloc((size_t)hcE, sizeof(float));
    m->xn = (float *)calloc((size_t)hcE, sizeof(float));
    m->mixed = (float *)calloc((size_t)E, sizeof(float));
    m->out = (float *)calloc((size_t)hcE, sizeof(float));
    m->inject = (float *)calloc(64, sizeof(float));
    m->tmp = (float *)calloc((size_t)(c->n_exp > c->hc_rank ? c->n_exp : c->hc_rank) + 64, sizeof(float));
    m->tmp2 = (float *)calloc(65536, sizeof(float));
    int64_t big = (int64_t)hcE;
    int64_t g = 2 * (int64_t)ch + c->n_vh * c->d_state;
    if (g > big) big = g;
    int64_t a = (int64_t)c->n_head * c->head_dim * 3;
    if (a > big) big = a;
    m->big = (float *)calloc((size_t)big, sizeof(float));
    m->logits = (float *)calloc((size_t)c->n_vocab, sizeof(float));
    const int NX = c->n_exp_used + 1;
    m->table = (q38_slot **)calloc((size_t)c->n_layer * c->n_exp, sizeof(q38_slot *));
    m->moe_gu = (float *)calloc((size_t)NX * 2 * c->ff_exp, sizeof(float));
    m->moe_h = (float *)calloc((size_t)NX * c->ff_exp, sizeof(float));
    m->moe_dn = (float *)calloc((size_t)NX * E, sizeof(float));
    m->moe_bsum = (float *)calloc((size_t)NX * (c->ff_exp / 32 + 1), sizeof(float));
    m->use = (float *)calloc((size_t)c->n_layer * c->n_exp, sizeof(float));
    q38_model_set_cache(m, -1);
    if (!m->toks || !m->res || !m->big || !m->logits || !m->table || !m->moe_gu || !m->moe_h || !m->moe_dn ||
        !m->moe_bsum || 4 * 4096 + E > 65536 || 3 * c->ff_exp > 4 * 4096 || c->n_embd > MAX_NE0 ||
        c->n_exp > 1024 || c->n_exp_used > 64 || c->ff_exp % 32) {
        fprintf(stderr, "q38: out of memory or unsupported sizes\n");
        q38_model_close(m);
        return -1;
    }
    fprintf(stderr, "q38: %d layers, hidden %d, %d experts (top %d), context %d | trunk %.2f GB loaded in %.1f s\n",
            c->n_layer, E, c->n_exp, c->n_exp_used, max_ctx, m->trunk_bytes / 1e9, t1 - t0);
    return 0;
}

void q38_model_close(q38_model *m)
{
#ifdef Q38_CUDA
    if (m->gpu) q38_gpu_close(m);
#endif
    q38_mat *all[] = { &m->tok_embd, &m->output, &m->out_hc_norm, &m->out_hc_down, &m->out_hc_up };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) free(all[i]->data);
    for (int il = 0; il < Q38_MAX_LAYERS; il++) {
        q38_mat *w = (q38_mat *)&m->L[il];
        for (size_t i = 0; i < sizeof m->L[il] / sizeof(q38_mat); i++) free(w[i].data);
        free(m->conv_state[il]);
        free(m->ssm_state[il]);
        free(m->kcache[il]);
        free(m->vcache[il]);
        free(m->idx_raw[il]);
        free(m->idx_pool[il]);
    }
    free(m->ple_hist); free(m->toks); free(m->res); free(m->xn); free(m->mixed); free(m->out);
    free(m->inject); free(m->tmp); free(m->tmp2); free(m->big); free(m->logits); free(m->exbuf);
    for (q38_slot *sl = m->lru_head; sl;) { q38_slot *nx = sl->next; free(sl->data); free(sl); sl = nx; }
    free(m->use);
    free(m->table); free(m->moe_gu); free(m->moe_h); free(m->moe_dn); free(m->moe_bsum);
    q38_gguf_close(&m->g);
    memset(m, 0, sizeof *m);
}
