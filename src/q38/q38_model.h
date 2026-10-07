/* q38_model.h - Qwen3.8-Flash-Next (GGUF arch qwen4exp) forward pass, one token at a time;
 * also the plain members of the family (qwen35moe: Qwen3.5/3.6 MoE), see q38_cfg.plain.
 *
 * The graph follows llama.cpp's src/models/qwen4exp.cpp; tools/q38_reference.py is the same
 * graph in float64 numpy and is what this file is checked against (tools/q38_compare.py).
 *
 * Per token: the embedding is copied into hc = 4 residual streams; every layer runs
 *   [PLE n-gram embedding, layer 1 only] -> hyper-connection mix -> token mixer
 *   (Gated DeltaNet on 3 of every 4 layers, gated GQA attention on the 4th) -> combine
 *   -> mix -> MoE (top-10 of 512 routed experts + a gated shared expert) -> combine,
 * and a final mix (no inject) feeds the output projection.
 *
 * Weights stay in their GGUF quantisation. The trunk (everything but the routed experts and
 * the PLE table) is read into RAM at open; routed experts and PLE rows are read from the
 * GGUF when a token needs them. Attention is dense, which matches the model while the
 * context is at most indexer_top_k + compress_ratio - 1 (2051) tokens: until then the
 * indexer selects every position. Longer contexts are refused for now.
 */
#ifndef Q38_MODEL_H
#define Q38_MODEL_H

#include <stdint.h>
#include "q38_gguf.h"

#define Q38_MAX_LAYERS 64
#define Q38_MAX_PLE_HEADS 32

typedef struct {
    int     n_layer, n_embd, n_vocab;
    int     n_head, n_head_kv, head_dim;     /* full attention */
    int     n_rot;
    double  rope_base;
    float   eps;
    int     n_exp, n_exp_used, ff_exp, ff_shexp;
    int     interval;                        /* every interval-th layer is full attention */
    int     hc, hc_rank;
    int     conv_k, d_state, n_kh, n_vh;     /* Gated DeltaNet */
    int     idx_top_k, max_dense_ctx;        /* indexer budget; dense attention limit */
    int     idx_heads, idx_dim, idx_ratio;   /* lightning indexer: query heads, key length, block size */
    /* PLE */
    int     ple_layer;                       /* -1 when absent */
    int     ple_ngram, ple_per_gram, ple_heads, ple_dim, ple_kern;
    int     ple_eos;
    uint64_t ple_mult[8];
    int64_t ple_off[Q38_MAX_PLE_HEADS], ple_voc[Q38_MAX_PLE_HEADS];
    /* the family member: qwen4exp (Qwen3.8-Flash-Next) has hyper-connections, PLE and the
     * indexer; qwen35moe (Qwen3.5/3.6 MoE) is "plain": one residual stream, a pre-norm block
     * x + mixer(norm(x)), then + moe(norm(x)), and a silu (not sigmoid) DeltaNet output gate */
    char    arch[32], name[96];
    int     plain;                           /* hc is 1 then */
    int     gdn_silu;
    int     n_nextn;                         /* MTP blocks after the n_layer ones (not run yet) */
} q38_cfg;

/* A weight matrix as it sits in the GGUF: rows of ne0 elements of one ggml type. */
typedef struct {
    int      type;
    int64_t  ne0, rows, row_bytes;
    uint8_t *data;                   /* the trunk: loaded; experts: NULL */
    const q38_tensor *t;             /* source tensor */
} q38_mat;

typedef struct {
    q38_mat hc_attn_norm, hc_attn_down, hc_attn_up, hc_attn_inject;
    q38_mat hc_ffn_norm, hc_ffn_down, hc_ffn_up, hc_ffn_inject;
    /* Gated DeltaNet */
    q38_mat qkv, gate, beta, alpha, dt_bias, a, conv, ssm_norm, ssm_out;
    /* attention */
    q38_mat wq, wk, wv, wo, q_norm, k_norm;
    q38_mat idx_q, idx_k, idx_qn, idx_kn;   /* indexer (attention layers) */
    /* PLE */
    q38_mat ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv;
    /* MoE */
    q38_mat router, shexp_gate_inp, sh_gate, sh_up, sh_down;
    q38_mat ex_gate, ex_up, ex_down;     /* data NULL: read per expert */
    q38_mat attn_norm, post_norm;        /* plain members: the block's two RMSNorms */
    int     recurrent;
} q38_layer;

/* One routed expert held in RAM: gate rows, up rows, down rows, back to back. */
typedef struct q38_slot {
    uint8_t *data;
    int64_t  bytes;
    int      key;                    /* layer * n_exp + expert */
    int      stamp;                  /* last token that used it */
    volatile int busy;               /* > 0: being copied to VRAM, not to be evicted */
    volatile long loading;           /* 1: a background read fills data (not to be used or evicted
                                      * yet); 2: that read failed; 0: ready */
    int      pf;                     /* read ahead and not used yet */
    struct q38_slot *prev, *next;    /* LRU list, most recent first */
} q38_slot;

typedef struct {
    q38_cfg   c;
    q38_gguf  g;
    /* expert cache (LRU, byte budget) */
    q38_slot **table;                /* [n_layer * n_exp] */
    q38_slot  *lru_head, *lru_tail;
    int64_t    cache_budget, cache_used;
    int64_t    cache_hits, cache_misses, cache_count;
    int        token_stamp;
    /* evicted experts' buffers, kept for the next expert of the same size */
    int64_t    fb_size[8];
    uint8_t  **fb[8];
    int        fb_n[8], n_fb;
    /* the prompt pass's experts once the cache is full: read into these reused buffers and
     * not kept (q38_model_experts_stream); their bytes come out of the cache budget */
    uint8_t  **tr_buf;
    int        tr_n;
    int64_t    tr_size;
    /* the budget set_cache chose, and when the free RAM was last compared with the floor */
    int64_t    cache_budget0;
    double     t_memcheck;
    float     *moe_gu, *moe_h, *moe_dn, *moe_bsum;
    q38_layer L[Q38_MAX_LAYERS];
    q38_mat   tok_embd, output, out_hc_norm, out_hc_down, out_hc_up, ple_table;
    q38_mat   output_norm;               /* plain members */
    int64_t   trunk_bytes;

    /* state for one sequence */
    int       n_past;
    int       max_ctx;
    int      *toks;                  /* tokens seen, for the PLE n-grams */
    float    *conv_state[Q38_MAX_LAYERS];   /* [conv_k-1][channels] */
    float    *ssm_state[Q38_MAX_LAYERS];    /* [n_vh][d_state (key i)][d_state (value j)] */
    float    *kcache[Q38_MAX_LAYERS], *vcache[Q38_MAX_LAYERS];  /* [ctx][n_head_kv*head_dim] */
    float    *idx_raw[Q38_MAX_LAYERS];  /* [ctx][idx_dim]: indexer keys as projected */
    float    *idx_pool[Q38_MAX_LAYERS]; /* [ctx/ratio][idx_dim]: mean of a full block, normed, roped */
    float    *ple_hist;              /* [(ple_kern-1)*ple_ngram][hc*n_embd] */

    /* scratch */
    float    *res, *xn, *mixed, *out, *inject, *tmp, *tmp2, *big;
    float    *logits;
    uint8_t  *exbuf;
    int64_t   exbuf_cap;

    /* optional per-token taps for validation (q38_run --dump) */
    void    (*tap)(void *ctx, const char *name, const float *x, int64_t n);
    void     *tap_ctx;

    void     *gpu;                   /* q38_gpu.cu state when the trunk is on a device */
    void     *pf;                    /* background expert reads (q38_model_prefetch) */
    int64_t   pf_issued, pf_used, pf_waited;
    double    t_pf_wait;             /* time a used expert's background read was still running */
    float    *use;                   /* [n_layer * n_exp] router choices: this session + loaded profile */

    /* counters */
    double    t_read, t_experts, t_trunk;
    double    t_gpu_wait, t_ple, t_tail;   /* GPU path: waiting for the device before each MoE, PLE on the host, final sync */
    int64_t   expert_bytes_read;
} q38_model;

/* Opens the model (any shard of a split GGUF) and loads the trunk. max_ctx bounds the KV
 * cache. Returns 0, or -1 with the reason printed. */
int  q38_model_open(q38_model *m, const char *path, int max_ctx);
void q38_model_close(q38_model *m);
/* For the GPU path (q38_gpu.cu): the routed experts' weighted sum for layer il given the
 * mixed activation and the router logits, and the PLE block on the host residual. */
/* the router's choice: top[n_exp_used] ids and their renormalised weights */
void q38_model_route(q38_model *m, int il, const float *logits, int *top, float *w);
/* y = sum_k w[k] * expert ids[k](x) for n experts of layer il, from the RAM cache */
int  q38_model_experts(q38_model *m, int il, const float *x, const int *ids, const float *w, int n, float *y);
/* Make n experts of layer il resident in the RAM cache (misses read in parallel) and return
 * their bytes; they stay protected from eviction until the next call. 0 or -1. */
int  q38_model_experts_resident(q38_model *m, int il, const int *ids, int n, const uint8_t **data);
/* The prompt pass's form of it: hits come from the cache, misses go into the cache while it
 * has room and into reused buffers outside it once it is full, so a pass over more experts
 * than the cache holds keeps what the cache has instead of evicting what the next batch
 * needs. The bytes stay valid until the next call. 0 or -1. */
int  q38_model_experts_stream(q38_model *m, int il, const int *ids, int n, const uint8_t **data);
/* the same for many tokens: pairs (token ptok[p], expert pexp[p], weight pw[p]), each expert
 * read once for all of its tokens; Y[n_tok][n_embd] gets each token's sum in pair order */
int  q38_model_experts_batch(q38_model *m, int il, const float *X, int n_tok, const int *ptok, const int *pexp,
                             const float *pw, int npair, float *Y);
/* an expert's bytes in the RAM cache (gate | up | down), or -1 when it is not resident */
/* Starts reading expert e of layer il from the GGUF into the RAM cache on a background
 * thread; nothing when it is resident or already on its way. Its first use waits for it. */
void q38_model_prefetch(q38_model *m, int il, int e);
/* 1 when expert e of layer il is in the RAM cache or being read into it */
int  q38_model_expert_known(const q38_model *m, int il, int e);
int  q38_model_expert_slot(q38_model *m, int il, int e, const uint8_t **data, int64_t *bytes);
/* The same, holding the expert in RAM until q38_model_expert_release (it may be released
 * from another thread). Returns a handle, or NULL when it is not resident. */
void *q38_model_expert_hold(q38_model *m, int il, int e, const uint8_t **data, int64_t *bytes);
void  q38_model_expert_release(void *handle);
int  q38_model_ple(q38_model *m, int il, float *res, int tok);
/* the PLE layer's input for token tok at position n_past (ple_heads * ple_dim floats) */
int  q38_model_ple_emb(q38_model *m, int tok, float *emb);
/* the same for n tokens at positions p0.. whose ids are already in m->toks */
int  q38_model_ple_emb_batch(q38_model *m, int p0, int n, float *emb);
/* Expert usage profile: counts of router choices, kept across sessions (a profile file of
 * n_layer * n_exp floats). load adds the file's counts; save writes the current ones. */
int  q38_model_profile_load(q38_model *m, const char *path);
int  q38_model_profile_save(q38_model *m, const char *path);
/* Read the most used experts (by the profile) into the RAM cache, up to gb gigabytes or the
 * cache budget. Returns the number of experts read. */
int  q38_model_preload(q38_model *m, double gb);
/* Expert cache budget in GB; <= 0 sizes it from the free RAM (leaving a margin). */
void q38_model_set_cache(q38_model *m, double gb);
void q38_model_reset(q38_model *m);

/* Feeds one token at position m->n_past and returns the logits (n_vocab floats, owned by
 * the model), or NULL on failure. */
const float *q38_model_forward(q38_model *m, int token);
/* n tokens at once (batched on the GPU, one by one on the CPU); logits of the last one. */
const float *q38_model_forward_batch(q38_model *m, const int *toks, int n);

#endif
