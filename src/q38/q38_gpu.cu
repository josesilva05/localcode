/* q38_gpu.cu - see q38_gpu.h. Each kernel repeats the CPU step of the same name in
 * q38_model.c; the comments name the llama.cpp graph node when it helps. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <atomic>

extern "C" {
#include "q38_model.h"
#include "q38_gguf.h"
#include "q38_quant.h"
}
#include "q38_gpu.h"

#include <time.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif
static double now_s(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

#include <immintrin.h>
/* Wait for an event by polling: the host has nothing else to do, and a blocking wait costs a
 * thread wake-up (tens of microseconds on Windows) at every one of the ~100 waits per token.
 * Q38_SPIN=0 goes back to blocking waits. */
static int g_spin = -1;
static cudaError_t ev_wait(cudaEvent_t e)
{
    if (g_spin < 0) g_spin = !(getenv("Q38_SPIN") && atoi(getenv("Q38_SPIN")) == 0);
    if (!g_spin) return cudaEventSynchronize(e);
    cudaError_t r;
    while ((r = cudaEventQuery(e)) == cudaErrorNotReady) _mm_pause();
    return r;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "q38_gpu: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

/* Q8_0 matrices are repacked on upload (Q38_Q8R): int8 quants row by row, 16-byte aligned,
 * and the block scales as float32 beside them; values are unchanged (fp16 -> fp32 is exact). */
#define Q38_Q8R 100
typedef struct { int type; int64_t ne0, rows, row_bytes; uint8_t *d; float *scale; } DMat;
typedef struct { const uint8_t *g, *u, *d; float w; int pad; } ExDesc;
/* one layer's routed experts resident in VRAM, as the host found them for this token */
typedef struct { ExDesc e[16]; int n; int pad[3]; } LayerDesc;
/* one background upload of an expert to VRAM (state 0 free, 1 queued, 2 DMA in flight) */
struct UpJob { volatile int state; int slot, key; void *hold; const uint8_t *data; int64_t bytes; uint8_t *stage; cudaEvent_t ev; };

#define NMAT ((int)(sizeof(q38_layer) / sizeof(q38_mat)))
#define LM(il, field) (G->L[il][offsetof(q38_layer, field) / sizeof(q38_mat)])

typedef struct {
    int       dev;
    cudaStream_t st;
    cudaEvent_t  ev;
    DMat      L[Q38_MAX_LAYERS][64];
    DMat      output, out_hc_norm, out_hc_down, out_hc_up, output_norm;
    /* the MTP block's nextn tensors and buffers (the dense models' last part; its block in L[n_layer]) */
    DMat      mtp_eh, mtp_enorm, mtp_hnorm, mtp_shnorm;
    float    *mtp_hprev, *mtp_cat, *mtp_emb, *mtp_logits, *h_mtp_logits;
    float    *mtp_hchain;               /* the last draft's normed output: the h of a chained draft */
    /* activations */
    float    *res, *xn, *mixed, *lo, *gate, *inject, *y, *routed;
    float    *fa_part;                     /* flash decode partials: segments x H x (D + 2) */
    int8_t   *q8x;                         /* x in 8 bits for k_mvq (dense trunks), its scales and sums */
    float    *q8d, *q8s;
    float    *qkv, *z, *o, *beta, *g, *qf, *aout;
    float    *router, *sh_gu, *sh_h, *sh_dn, *sg, *logits;
    /* state */
    float    *conv[Q38_MAX_LAYERS], *ssm[Q38_MAX_LAYERS];
    void     *kc[Q38_MAX_LAYERS], *vc[Q38_MAX_LAYERS];   /* float, __half when kv_half, int8 when kv_q8 */
    int       kv_half, kv_q8, kvh_n;      /* kvh_n: K/V heads (the int8 store's rows) */
    float    *ksc[Q38_MAX_LAYERS], *vsc[Q38_MAX_LAYERS];  /* kv_q8: a scale per 32 values (llama.cpp's q8_0) */
    /* pinned host buffers */
    float    *h_x, *h_router, *h_y, *h_res, *h_tap;
    int64_t   bytes;
    /* routed experts in VRAM (pool_open / pool_promote) */
    int       n_cls, cls_of_layer[Q38_MAX_LAYERS], cls_layers[8];
    int64_t   cls_bytes[8];
    int       npd, pdev[2];               /* pool devices: [0] the trunk's, [1] the other one */
    int       cls_n[2][8], cls_first[2][8], free_n[2][8];
    uint8_t  *cls_mem[2][8];
    int       nvs, *vs_key, *free_list, *vslot, *cand, n_cand, swaps;
    unsigned char *vs_pd;
    uint8_t **vs_ptr;
    int64_t   vram_bytes_pd[2];
    /* the second device's share of a MoE layer */
    cudaStream_t st2;
    cudaEvent_t  ev2;
    LayerDesc *h_ldesc2, *d_ldesc2;
    float    *x2, *ex_gu2, *ex_h2, *ex_dn2, *routed2, *h_y2;
    cudaGraphExec_t gexec2[Q38_MAX_LAYERS];
    int64_t   aux_hits;
    /* next-layer prediction: layer il+1's router on layer il's MoE input, its top experts
     * read ahead from the SSD while layer il computes (Q38_PREDICT = how many, 0 = off) */
    int       predict, pred_ahead;    /* experts per layer; layers ahead (Q38_PREDICT_AHEAD, 1-2) */
    float    *router_next, *h_router_next;   /* [pred_ahead][n_exp] */
    DMat      pred_router[Q38_MAX_LAYERS];    /* int8 copies of the routers, for the guesses only */
    cudaEvent_t ev_pred[Q38_MAX_LAYERS];   /* one per layer: the launcher submits layer il+1 (which
                                             * records its own) before the host reads layer il's */
    int       pred_ids[2][64], pred_n[2], pred_layer[2];
    int64_t   pred_total, pred_hit;
    float    *freq;
    ExDesc   *h_desc, *d_desc;
    float    *routed_dev, *ex_gu, *ex_h, *ex_dn;
    int64_t   vram_bytes, uploads, tokens, vram_hits, vram_misses;
    /* background uploads to VRAM: jobs filled by pool_promote, copied by up_main */
    cudaStream_t st_up[2];
    UpJob     up[32];
    cudaEvent_t up_ev[2][32];
    uint8_t  *dstage[2][32];          /* device staging: raw bytes before k_repack */
    int       ex_types[Q38_MAX_LAYERS][3];
    int       ex_rows[3];
    int       n_exp_, ff_exp_, n_embd_;
    int64_t   rb[Q38_MAX_LAYERS][3];
    int       n_up;
    char     *pend;                   /* key has an upload in flight */
    volatile int up_stop;
#ifdef _WIN32
    HANDLE    up_thread;
#else
    pthread_t up_thread;
#endif
    double    t_launch, t_route, t_post, t_promote;   /* host time per phase (Q38_PROFILE) */
    double    t_rt, t_auxl, t_gxl, t_pred, t_auxw, t_xcall;  /* inside route; aux wait; experts call */
    /* the launcher thread (Q38_LAUNCHER, default on with graphs): while the host computes a
     * layer's experts it submits the second device's experts, this device's experts and the
     * next layer's graph, whose first kernel waits for h_flag[il] (mapped host memory) to move */
    int       launcher;
    uint8_t  *snap;                 /* q38_gpu_state: the host copy of the sequence state */
    size_t    snap_bytes;
    volatile long job_seq, done_aux_seq, done_f_seq, launch_stop, launch_err;
    int       job_il, job_aux, job_last;
    volatile unsigned *h_flag;
    unsigned *d_flag, *d_seen;
#ifdef _WIN32
    HANDLE    launch_thread;
#endif
    /* prompt pass: experts streamed from RAM into a staging area and computed here */
    uint8_t  *pstage, *praw[16], *phost[16];  /* streamed experts: device raw bytes, pinned bounce */
    cudaEvent_t pev[16];                  /* slot i's raw bytes consumed */
    cudaStream_t st_cp;                   /* their copies and rearrangement, beside the compute */
    cudaEvent_t ev_stage, ev_pairs, ev_pup;   /* a round staged; its pairs done; pair lists uploaded */
    int       pst_n, pring;
    int64_t   pst_slot, stream_bytes;
    double    t_stream, t_sp_copy, t_sp_pairs, t_attn;   /* t_attn: Q38_ATT_PROF=1 */
    /* per-token inputs read by the device (graph-stable) */
    int      *d_pos, *h_pos;
    float    *h_emb, *h_logits, *knew, *vnew;
    LayerDesc *h_ldesc, *d_ldesc;
    int       hcE, use_graphs;
    /* PLE on the device: input rows from the host, history in a ring indexed by position */
    float    *h_ple_emb, *ple_emb, *ple_key, *ple_val, *ple_gated, *ple_normed, *ple_gate, *ple_ring;
    /* indexer */
    float    *idx_raw[Q38_MAX_LAYERS], *idx_pool[Q38_MAX_LAYERS], *ikraw, *iq, *iscore;
    int      *isel, ins;
    cudaGraphExec_t gexec[2 * Q38_MAX_LAYERS + 4];
    int64_t   tokens_fwd;
    /* keep-warm (see warm_main) */
    cudaStream_t warm_st;
    cudaEvent_t  warm_ev;
    float       *warm_buf, *warm_out;
    volatile int warm_want, warm_stop, warm_on;
    int          warm_blocks, warm_n4, warm_dev;
#ifdef _WIN32
    HANDLE       warm_thread;
#else
    pthread_t    warm_thread;
#endif
} Gpu;

/* ------------------------------------------------------------------- kernels -- */
__device__ __forceinline__ float warp_sum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ double warp_sumd(double v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.0f + expf(-x)); }

#define ROWS_PER_BLOCK 8
__device__ __forceinline__ float block_sum(float v, float *part);

/* Q8_0 rows: lane l takes element l of every block (34 bytes: fp16 d, int8 qs[32]) */
__global__ void k_mv_q8_0(const uint8_t *__restrict__ W, int64_t rb, int ne0, int rows,
                          const float *__restrict__ x, float *__restrict__ y)
{
    extern __shared__ float xs[];
    for (int i = threadIdx.y * 32 + threadIdx.x; i < ne0; i += 32 * ROWS_PER_BLOCK) xs[i] = x[i];
    __syncthreads();
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (r >= rows) return;
    const uint8_t *w = W + (int64_t)r * rb;
    const int lane = threadIdx.x, nb = ne0 / 32;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = w + (int64_t)b * 34;
        const float d = __half2float(*(const __half *)(const void *)blk);   /* 2-byte aligned */
        acc += d * ((float)(int8_t)blk[2 + lane] * xs[b * 32 + lane]);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[r] = acc;
}

/* repacked Q8_0: a warp computes RW rows; per step each lane loads 16 x values once and
 * 16 quants of every row (int4 loads), so x is read once per RW rows */
#define RW 4
template <int RW_>
__global__ void k_mv_q8r(const int8_t *__restrict__ Q, const float *__restrict__ D, int ne0, int rows,
                         const float *__restrict__ x, float *__restrict__ y)
{
#undef RW
#define RW RW_
    const int r0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.y) * RW;
    if (r0 >= rows) return;
    const int nr = rows - r0 < RW ? rows - r0 : RW, nb = ne0 / 32;
    float acc[RW];
#pragma unroll
    for (int k = 0; k < RW; k++) acc[k] = 0.0f;
    for (int i = threadIdx.x * 16; i < ne0; i += 32 * 16) {
        const float4 x0 = __ldg((const float4 *)(x + i)), x1 = __ldg((const float4 *)(x + i + 4));
        const float4 x2 = __ldg((const float4 *)(x + i + 8)), x3 = __ldg((const float4 *)(x + i + 12));
#pragma unroll
        for (int k = 0; k < RW; k++) {
            if (k >= nr) break;
            const int4 v = __ldg((const int4 *)(Q + (int64_t)(r0 + k) * ne0 + i));
            const int8_t *b = (const int8_t *)&v;
            float t = b[0] * x0.x + b[1] * x0.y + b[2] * x0.z + b[3] * x0.w;
            t += b[4] * x1.x + b[5] * x1.y + b[6] * x1.z + b[7] * x1.w;
            t += b[8] * x2.x + b[9] * x2.y + b[10] * x2.z + b[11] * x2.w;
            t += b[12] * x3.x + b[13] * x3.y + b[14] * x3.z + b[15] * x3.w;
            acc[k] += __ldg(D + (int64_t)(r0 + k) * nb + i / 32) * t;
        }
    }
#pragma unroll
    for (int k = 0; k < RW; k++) {
        const float v = warp_sum(acc[k]);
        if (threadIdx.x == 0 && k < nr) y[r0 + k] = v;
    }
}
#undef RW
#define RW 4

/* few long rows (hc down: 320 x 10240): one block of 256 threads per row */
__global__ void k_mv_q8r_blk(const int8_t *__restrict__ Q, const float *__restrict__ D, int ne0,
                             const float *__restrict__ x, float *__restrict__ y)
{
    __shared__ float part[8];
    const int r = blockIdx.x, nb = ne0 / 32;
    const int8_t *q = Q + (int64_t)r * ne0;
    float acc = 0.0f;
    for (int i = threadIdx.x * 16; i < ne0; i += blockDim.x * 16) {
        const int4 v = __ldg((const int4 *)(q + i));
        const float4 x0 = __ldg((const float4 *)(x + i)), x1 = __ldg((const float4 *)(x + i + 4));
        const float4 x2 = __ldg((const float4 *)(x + i + 8)), x3 = __ldg((const float4 *)(x + i + 12));
        const int8_t *b = (const int8_t *)&v;
        float t = b[0] * x0.x + b[1] * x0.y + b[2] * x0.z + b[3] * x0.w;
        t += b[4] * x1.x + b[5] * x1.y + b[6] * x1.z + b[7] * x1.w;
        t += b[8] * x2.x + b[9] * x2.y + b[10] * x2.z + b[11] * x2.w;
        t += b[12] * x3.x + b[13] * x3.y + b[14] * x3.z + b[15] * x3.w;
        acc += __ldg(D + (int64_t)r * nb + i / 32) * t;
    }
    acc = block_sum(acc, part);
    if (threadIdx.x == 0) y[r] = acc;
}

/* many short rows (hc up: 10240 x 320): 4 lanes per row, 8 rows per warp */
__global__ void k_mv_q8r_short(const int8_t *__restrict__ Q, const float *__restrict__ D, int ne0, int rows,
                               const float *__restrict__ x, float *__restrict__ y)
{
    const int lane = threadIdx.x, sub = lane & 3;
    const int r = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.y) * 8 + (lane >> 2);
    float acc = 0.0f;
    if (r < rows) {
        const int8_t *q = Q + (int64_t)r * ne0;
        const int nb = ne0 / 32;
        for (int i = sub * 16; i < ne0; i += 64) {
            const int4 v = __ldg((const int4 *)(q + i));
            const float4 x0 = __ldg((const float4 *)(x + i)), x1 = __ldg((const float4 *)(x + i + 4));
            const float4 x2 = __ldg((const float4 *)(x + i + 8)), x3 = __ldg((const float4 *)(x + i + 12));
            const int8_t *b = (const int8_t *)&v;
            float t = b[0] * x0.x + b[1] * x0.y + b[2] * x0.z + b[3] * x0.w;
            t += b[4] * x1.x + b[5] * x1.y + b[6] * x1.z + b[7] * x1.w;
            t += b[8] * x2.x + b[9] * x2.y + b[10] * x2.z + b[11] * x2.w;
            t += b[12] * x3.x + b[13] * x3.y + b[14] * x3.z + b[15] * x3.w;
            acc += __ldg(D + (int64_t)r * nb + i / 32) * t;
        }
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    if (sub == 0 && r < rows) y[r] = acc;
}

/* few long rows (hc inject, gates): one block of 256 threads per row */
__global__ void k_mv_f32_blk(const float *__restrict__ W, int ne0, const float *__restrict__ x, float *__restrict__ y)
{
    __shared__ float part[8];
    const float *w = W + (int64_t)blockIdx.x * ne0;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) acc += w[i] * x[i];
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += part[i];
        y[blockIdx.x] = t;
    }
}

__global__ void k_mv_f32(const float *__restrict__ W, int ne0, int rows, const float *__restrict__ x,
                         float *__restrict__ y)
{
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (r >= rows) return;
    const float *w = W + (int64_t)r * ne0;
    float acc = 0.0f;
    if ((ne0 & 127) == 0) {
        for (int i = threadIdx.x * 4; i < ne0; i += 128) {
            const float4 a = __ldg((const float4 *)(w + i)), b = __ldg((const float4 *)(x + i));
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = threadIdx.x; i < ne0; i += 32) acc += w[i] * x[i];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[r] = acc;
}

/* rows of n elements: y = x / sqrt(mean(x^2)+eps) * w; one block (256 threads) per row */
__device__ __forceinline__ float block_sum(float v, float *part)
{
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += part[i];
        part[0] = t;
    }
    __syncthreads();
    return part[0];
}

__global__ void k_rmsnorm_rows(const float *x, const float *w, float *y, int n, float eps)
{
    __shared__ float part[8];
    const float *xr = x + (int64_t)blockIdx.x * n;
    const float *wr = w ? w + (int64_t)blockIdx.x * n : NULL;
    float *yr = y + (int64_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float sc = 1.0f / sqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * sc * (wr ? wr[i] : 1.0f);
}

/* rows of n: y = x / sqrt(sum(x^2) + eps) (build_gdn_l2_norm); one warp per row */
__global__ void k_l2norm_rows(float *x, int n, float eps)
{
    float *xr = x + (int64_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += 32) ss += xr[i] * xr[i];
    ss = warp_sum(ss);
    const float sc = 1.0f / sqrtf(ss + eps);
    for (int i = threadIdx.x; i < n; i += 32) xr[i] *= sc;
}

__global__ void k_silu_scale(float *x, int n, float inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = siluf_(x[i] * inv);
}

/* mixed[d] = mean over streams of xn * sigmoid(gate) */
__global__ void k_hc_mean(const float *xn, const float *gate, float *mixed, int E, int hc)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= E) return;
    float s = 0.0f;
    for (int st = 0; st < hc; st++) s += xn[st * E + d] * sigmoidf_(gate[st * E + d]);
    mixed[d] = s / hc;
}

/* res[s] += 2*sigmoid(inject[s]/hc) * out */
__global__ void k_hc_combine(float *res, const float *out, const float *inject, int E, int hc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E * hc) return;
    const int s = i / E;
    const float w = 2.0f * sigmoidf_(inject[s] / hc);
    res[i] += w * out[i % E];
}

__global__ void k_tile(float *res, const float *x, int E, int hc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E * hc) res[i] = x[i % E];
}

/* beta = sigmoid(beta); g = softplus(alpha + dt_bias) * a */
__global__ void k_gdn_gates(float *beta, float *g, const float *dt_bias, const float *a, int H)
{
    const int h = threadIdx.x;
    if (h >= H) return;
    beta[h] = sigmoidf_(beta[h]);
    const float t = g[h] + dt_bias[h];
    g[h] = (t > 20.0f ? t : logf(1.0f + expf(t))) * a[h];
}

/* ggml_ssm_conv over [state | x] then silu; the state keeps the last K-1 inputs */
__global__ void k_conv_silu(float *x, float *state, const float *w, int ch, int K)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ch) return;
    float acc = 0.0f;
    for (int k = 0; k < K - 1; k++) acc += state[(int64_t)k * ch + i] * w[(int64_t)i * K + k];
    const float xi = x[i];
    acc += xi * w[(int64_t)i * K + K - 1];
    for (int k = 0; k < K - 2; k++) state[(int64_t)k * ch + i] = state[(int64_t)(k + 1) * ch + i];
    state[(int64_t)(K - 2) * ch + i] = xi;
    x[i] = siluf_(acc);
}

/* One value head per block of (S, 4) threads: column j of the S x S state [i][j] is split
 * over 4 thread rows by i, partial sums meet in shared memory. */
__global__ void k_gdn_step(float *state, const float *q, const float *k, const float *v, const float *g,
                           const float *beta, float *o, int S, int Hk)
{
    extern __shared__ float sh[];
    float *ks = sh, *qs = sh + S, *red = sh + 2 * S;     /* red: [4][S] */
    const int h = blockIdx.x, j = threadIdx.x, p = threadIdx.y, P = blockDim.y, kh = h % Hk;
    if (p == 0) { ks[j] = k[kh * S + j]; qs[j] = q[kh * S + j]; }
    __syncthreads();
    float *St = state + (int64_t)h * S * S;
    const float eg = expf(g[h]);
    const int i0 = p * (S / P), i1 = i0 + S / P;
    float sk = 0.0f;
    for (int i = i0; i < i1; i++) sk += St[(int64_t)i * S + j] * eg * ks[i];
    red[p * S + j] = sk;
    __syncthreads();
    float skt = 0.0f;
    for (int t = 0; t < P; t++) skt += red[t * S + j];
    const float d = (v[h * S + j] - skt) * beta[h];
    __syncthreads();
    float acc = 0.0f;
    for (int i = i0; i < i1; i++) {
        const float s2 = St[(int64_t)i * S + j] * eg + ks[i] * d;
        St[(int64_t)i * S + j] = s2;
        acc += s2 * qs[i];
    }
    red[p * S + j] = acc;
    __syncthreads();
    if (p == 0) {
        float ot = 0.0f;
        for (int t = 0; t < P; t++) ot += red[t * S + j];
        o[h * S + j] = ot * (1.0f / sqrtf((float)S));
    }
}

/* build_norm_gated: rmsnorm(o) * w * sigmoid(z) (qwen4exp) or * silu(z) (qwen35moe), one block per head */
__global__ void k_norm_gate(float *o, const float *z, const float *w, int S, float eps, int silu)
{
    __shared__ float part[8];
    float *oh = o + (int64_t)blockIdx.x * S;
    const float *zh = z + (int64_t)blockIdx.x * S;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) ss += oh[i] * oh[i];
    ss = block_sum(ss, part);
    const float sc = 1.0f / sqrtf(ss / S + eps);
    for (int i = threadIdx.x; i < S; i += blockDim.x)
        oh[i] = oh[i] * sc * w[i] * (silu ? siluf_(zh[i]) : sigmoidf_(zh[i]));
}

/* rmsnorm with weight then NeoX rope on the first n_rot dims, one block per head row */
__global__ void k_norm_rope(float *x, int64_t stride, const float *w, int D, int n_rot, const int *ppos, double base, float eps)
{
    const int pos = *ppos;
    __shared__ float part[8];
    __shared__ float buf[1024];
    float *xr = x + (int64_t)blockIdx.x * stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float sc = 1.0f / sqrtf(ss / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) buf[i] = xr[i] * sc * w[i];
    __syncthreads();
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = buf[i];
        if (i < n_rot) {
            const int f = i < half ? i : i - half;
            const double ang = pos * pow(base, -2.0 * f / n_rot);
            const float cs = (float)cos(ang), sn = (float)sin(ang);
            v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
        }
        xr[i] = v;
    }
}

/* dense causal attention for one query head per block (256 threads):
 * out = softmax(q K^T / sqrt(D)) V * sigmoid(gate) */
/* The K/V cache holds float or, for long contexts, half (Q38_KV; half above 32k positions
 * unless told: what llama.cpp keeps by default). */
__device__ __forceinline__ float kv_ld(const float *p) { return *p; }
__device__ __forceinline__ float kv_ld(const __half *p) { return __half2float(*p); }
__device__ __forceinline__ void kv_st(float *p, float v) { *p = v; }
__device__ __forceinline__ void kv_st(__half *p, float v) { *p = __float2half(v); }

/* the K/V of N tokens (rows of n) at positions *ppos.. */
template <typename KT>
__global__ void k_kv_store(const float *k, const float *v, KT *kc, KT *vc, const int *ppos, int n, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * N) return;
    kv_st(kc + (int64_t)*ppos * n + i, k[i]);
    kv_st(vc + (int64_t)*ppos * n + i, v[i]);
}

/* Gated attention of token blockIdx.y (position *ppos + blockIdx.y) for head blockIdx.x over
 * its cells (all of them, or the indexer's W), in tiles of ATT_TILE cells with an online
 * softmax: any context length in a fixed 8 KB of shared memory (the scores of every cell at
 * once capped the plain models near 12k positions). D up to 4 * blockDim. */
#define ATT_TILE 2048
template <typename KT>
__global__ void k_attn_t(const float *qf, int64_t qstride, const KT *kc, const KT *vc, float *out, int64_t ostride,
                         const int *ppos, int H, int Hkv, int D, const int *sel, int64_t sstride, int W)
{
    const int t = blockIdx.y;
    const int T0 = *ppos + t + 1, use_sel = sel && T0 > W, T = use_sel ? W : T0;
    const int *sl = use_sel ? sel + (int64_t)t * sstride : NULL;
    __shared__ float sc[ATT_TILE];
    __shared__ float part[8];
    const int h = blockIdx.x, kvh = h / (H / Hkv), lane = threadIdx.x & 31, wp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const float *q = qf + (int64_t)t * qstride + (int64_t)h * 2 * D, *gate = q + D;
    const float scale = 1.0f / sqrtf((float)D);
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f }, m = -1e30f, l = 0.0f;
    for (int t0 = 0; t0 < T; t0 += ATT_TILE) {
        const int n = T - t0 < ATT_TILE ? T - t0 : ATT_TILE;
        for (int c = wp; c < n; c += nw) {
            const KT *kk = kc + ((int64_t)(use_sel ? sl[t0 + c] : t0 + c) * Hkv + kvh) * D;
            float s = 0.0f;
            for (int i = lane; i < D; i += 32) s += q[i] * kv_ld(kk + i);
            s = warp_sum(s);
            if (lane == 0) sc[c] = s * scale;
        }
        __syncthreads();
        float mt = -1e30f;
        for (int c = threadIdx.x; c < n; c += blockDim.x) mt = fmaxf(mt, sc[c]);
        for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
        if (lane == 0) part[wp] = mt;
        __syncthreads();
        if (threadIdx.x == 0) { float m2 = -1e30f; for (int i = 0; i < nw; i++) m2 = fmaxf(m2, part[i]); part[0] = m2; }
        __syncthreads();
        const float mn = fmaxf(m, part[0]), f = expf(m - mn);
        __syncthreads();
        float sum = 0.0f;
        for (int c = threadIdx.x; c < n; c += blockDim.x) { const float e = expf(sc[c] - mn); sc[c] = e; sum += e; }
        l = l * f + block_sum(sum, part);              /* block_sum syncs: sc is complete after it */
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = threadIdx.x + j * blockDim.x;
            if (i >= D) break;
            float a = acc[j] * f;
            for (int c = 0; c < n; c++) a += sc[c] * kv_ld(vc + ((int64_t)(use_sel ? sl[t0 + c] : t0 + c) * Hkv + kvh) * D + i);
            acc[j] = a;
        }
        m = mn;
        __syncthreads();                               /* before the next tile's scores */
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int i = threadIdx.x + j * blockDim.x;
        if (i < D) out[(int64_t)t * ostride + (int64_t)h * D + i] = acc[j] / l * sigmoidf_(gate[i]);
    }
}

/* Flash attention (no indexer selection). Block (kv head, QT tokens, segment) holds the
 * R = 8 * RW query rows that share one K/V head: QT tokens x G = H / Hkv heads, RW rows per
 * warp. K/V go through shared memory 32 cells at a time, read once for all R rows (the
 * per-head kernel read them again for every head and every token: O(N^2) traffic on long
 * prompts); lane c scores cell c, online softmax per row; each lane owns D / 32 output
 * dims. The causal limit is per row. With gridDim.z > 1 the cells are split in segments
 * (decode: enough blocks for one token) and each writes its partial (acc, m, l) to part,
 * merged by k_fattn_merge. D a multiple of 32 up to 256. */
#define FA_DMAX 256
template <typename KT, int FRW>
__global__ void __launch_bounds__(256) k_fattn(const float *qf, const KT *kc, const KT *vc, float *out, float *part,
                                               const int *ppos, int H, int Hkv, int D, int N)
{
    extern __shared__ float sm[];
    const int G = H / Hkv, R = 8 * FRW, QT = R / G, DP = D + 1, DJ = D / 32;
    const int kvh = blockIdx.x, tq0 = blockIdx.y * QT, seg = blockIdx.z, S = gridDim.z;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    float *Qs = sm, *Ks = Qs + R * D, *Vs = Ks + 32 * DP;
    const int p0 = *ppos;
    for (int e = threadIdx.x; e < R * D; e += blockDim.x) {
        const int r = e / D, i = e % D, t = tq0 + r / G;
        Qs[e] = t < N ? qf[(int64_t)t * H * 2 * D + (int64_t)(kvh * G + r % G) * 2 * D + i] : 0.0f;
    }
    const int Tb = p0 + (tq0 + QT < N ? tq0 + QT : N);      /* cells the block's last token sees */
    const int L = ((Tb + S - 1) / S + 31) & ~31;
    const int k0 = seg * L, k1 = Tb < k0 + L ? Tb : k0 + L;
    const float scale = rsqrtf((float)D);
    float m[FRW], l[FRW], acc[FRW][FA_DMAX / 32];
#pragma unroll
    for (int w = 0; w < FRW; w++) {
        m[w] = -1e30f; l[w] = 0.0f;
#pragma unroll
        for (int j = 0; j < FA_DMAX / 32; j++) acc[w][j] = 0.0f;
    }
    for (int kb = k0; kb < k1; kb += 32) {
        const int nk = k1 - kb < 32 ? k1 - kb : 32;
        __syncthreads();                                   /* Q stored / the last tile used */
        for (int e = threadIdx.x; e < 32 * D; e += blockDim.x) {
            const int c = e / D, i = e % D;
            float kv = 0.0f, vv = 0.0f;
            if (c < nk) {
                const int64_t o = ((int64_t)(kb + c) * Hkv + kvh) * D + i;
                kv = kv_ld(kc + o);
                vv = kv_ld(vc + o);
            }
            Ks[c * DP + i] = kv;
            Vs[c * D + i] = vv;
        }
        __syncthreads();
        float s[FRW];
#pragma unroll
        for (int w = 0; w < FRW; w++) s[w] = 0.0f;
        const float *kr = Ks + lane * DP, *qr = Qs + wp * FRW * D;
        for (int i = 0; i < D; i++) {
            const float k = kr[i];
#pragma unroll
            for (int w = 0; w < FRW; w++) s[w] += qr[w * D + i] * k;
        }
#pragma unroll
        for (int w = 0; w < FRW; w++) {
            const int t = tq0 + (wp * FRW + w) / G;
            const bool ok = lane < nk && t < N && kb + lane <= p0 + t;
            const float sv = ok ? s[w] * scale : -1e30f;
            float mx = sv;
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mn = fmaxf(m[w], mx), f = expf(m[w] - mn), p = ok ? expf(sv - mn) : 0.0f;
            l[w] = l[w] * f + warp_sum(p);
#pragma unroll
            for (int j = 0; j < FA_DMAX / 32; j++) acc[w][j] *= f;
            for (int c = 0; c < nk; c++) {
                const float pc = __shfl_sync(0xffffffffu, p, c);
                const float *vr = Vs + c * D + lane;
#pragma unroll
                for (int j = 0; j < FA_DMAX / 32; j++) if (j < DJ) acc[w][j] += pc * vr[32 * j];
            }
            m[w] = mn;
        }
    }
#pragma unroll
    for (int w = 0; w < FRW; w++) {
        const int r = wp * FRW + w, t = tq0 + r / G, h = kvh * G + r % G;
        if (t >= N) continue;
        if (S == 1) {
            const float *gate = qf + (int64_t)t * H * 2 * D + (int64_t)h * 2 * D + D;
            const float il = l[w] > 0.0f ? 1.0f / l[w] : 0.0f;
#pragma unroll
            for (int j = 0; j < FA_DMAX / 32; j++)
                if (j < DJ) out[(int64_t)t * H * D + (int64_t)h * D + lane + 32 * j] = acc[w][j] * il * sigmoidf_(gate[lane + 32 * j]);
        } else {
            float *pp = part + ((int64_t)(seg * N + t) * H + h) * (D + 2);
#pragma unroll
            for (int j = 0; j < FA_DMAX / 32; j++) if (j < DJ) pp[lane + 32 * j] = acc[w][j];
            if (lane == 0) { pp[D] = m[w]; pp[D + 1] = l[w]; }
        }
    }
}

/* Flash attention of a prompt batch on the tensor cores (what llama.cpp's FA does): block (kv
 * head, QT tokens) holds 32 query rows (QT tokens x the G heads of that K/V head) in fp16,
 * prescaled; per tile of 32 cells, S = Q K^T and O += P V as 16x16x16 fp16 products with fp32
 * accumulation, the online softmax in between (O rescaled in shared memory). The CUDA-core
 * kernel above took ~0.2 s more per 1024 tokens of a prompt for every 1k cells before it. */
/* 32 cells of K and V (rows of D) into shared fp16, 8 elements a load; cells from nk on zero */
__device__ __forceinline__ void kv_tile(const __half *kc, const __half *vc, int kb, int nk, int Hkv, int kvh, int D,
                                        __half *Kh, __half *Vh)
{
    const int D8 = D / 8;
    for (int e = threadIdx.x; e < 32 * D8; e += blockDim.x) {
        const int c = e / D8, i = (e % D8) * 8;
        uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
        if (c < nk) {
            const int64_t o = ((int64_t)(kb + c) * Hkv + kvh) * D + i;
            kv = *(const uint4 *)(const void *)(kc + o);
            vv = *(const uint4 *)(const void *)(vc + o);
        }
        *(uint4 *)(void *)(Kh + c * D + i) = kv;
        *(uint4 *)(void *)(Vh + c * D + i) = vv;
    }
}

__device__ __forceinline__ uint4 h8(float4 a, float4 b)
{
    __half2 h[4] = { __floats2half2_rn(a.x, a.y), __floats2half2_rn(a.z, a.w), __floats2half2_rn(b.x, b.y),
                     __floats2half2_rn(b.z, b.w) };
    return *(const uint4 *)(const void *)h;
}

__device__ __forceinline__ void kv_tile(const float *kc, const float *vc, int kb, int nk, int Hkv, int kvh, int D,
                                        __half *Kh, __half *Vh)
{
    const int D8 = D / 8;
    for (int e = threadIdx.x; e < 32 * D8; e += blockDim.x) {
        const int c = e / D8, i = (e % D8) * 8;
        const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 k0 = z, k1 = z, v0 = z, v1 = z;
        if (c < nk) {
            const int64_t o = ((int64_t)(kb + c) * Hkv + kvh) * D + i;
            k0 = *(const float4 *)(kc + o); k1 = *(const float4 *)(kc + o + 4);
            v0 = *(const float4 *)(vc + o); v1 = *(const float4 *)(vc + o + 4);
        }
        *(uint4 *)(void *)(Kh + c * D + i) = h8(k0, k1);
        *(uint4 *)(void *)(Vh + c * D + i) = h8(v0, v1);
    }
}

template <typename KT>
__global__ void __launch_bounds__(256) k_fattn_tc(const float *qf, const KT *kc, const KT *vc, float *out, const int *ppos,
                                                  int H, int Hkv, int D, int N)
{
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smb[];
    const int G = H / Hkv, QT = 32 / G;
    const int kvh = blockIdx.x, tq0 = blockIdx.y * QT;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    float *O = (float *)smb, *S = O + 32 * D, *fs = S + 2 * 32 * 32, *ls = fs + 32, *ms = ls + 32;   /* S: two halves of D */
    __half *Qh = (__half *)(ms + 32), *Kh = Qh + 32 * D, *Vh = Kh + 32 * D, *Ph = Vh + 32 * D;
    const int p0 = *ppos;
    const float scale = rsqrtf((float)D);
    for (int e = threadIdx.x; e < 32 * D; e += blockDim.x) {
        const int r = e / D, i = e % D, t = tq0 + r / G;
        Qh[e] = __float2half(t < N ? qf[(int64_t)t * H * 2 * D + (int64_t)(kvh * G + r % G) * 2 * D + i] * scale : 0.0f);
        O[e] = 0.0f;
    }
    if (threadIdx.x < 32) { ms[threadIdx.x] = -1e30f; ls[threadIdx.x] = 0.0f; }
    const int Tb = p0 + (tq0 + QT < N ? tq0 + QT : N);       /* cells the block's last token sees */
    for (int kb = 0; kb < Tb; kb += 32) {
        const int nk = Tb - kb < 32 ? Tb - kb : 32;
        __syncthreads();                                    /* the last tile's K/V and P used */
        kv_tile(kc, vc, kb, nk, Hkv, kvh, D, Kh, Vh);
        __syncthreads();
        {                                                   /* S: four 16x16 blocks, each over two halves of D */
            const int wi = (wp >> 1) & 1, wj = wp & 1, kh = wp >> 2;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int k = kh * (D / 2); k < (kh + 1) * (D / 2); k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, Qh + wi * 16 * D + k, D);
                wmma::load_matrix_sync(b, Kh + wj * 16 * D + k, D);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(S + kh * 32 * 32 + wi * 16 * 32 + wj * 16, acc, 32, wmma::mem_row_major);
        }
        __syncthreads();
        for (int rr = 0; rr < 4; rr++) {                    /* softmax: 4 rows a warp, lane = cell */
            const int r = wp * 4 + rr, t = tq0 + r / G;
            const bool ok = lane < nk && t < N && kb + lane <= p0 + t;
            const float sv = ok ? S[r * 32 + lane] + S[32 * 32 + r * 32 + lane] : -1e30f;
            float mx = sv;
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mo = ms[r], mn = fmaxf(mo, mx), f = expf(mo - mn), p = ok ? expf(sv - mn) : 0.0f;
            const float sum = warp_sum(p);
            Ph[r * 32 + lane] = __float2half(p);
            if (lane == 0) { ms[r] = mn; ls[r] = ls[r] * f + sum; fs[r] = f; }
        }
        __syncthreads();
        for (int fi = wp; fi < 2 * (D / 16); fi += 8) {     /* O = O f + P V: 2 x D/16 blocks, each warp its own */
            const int oi = fi / (D / 16), oj = fi % (D / 16);
            for (int q = lane; q < 256; q += 32) O[(oi * 16 + q / 16) * D + oj * 16 + q % 16] *= fs[oi * 16 + q / 16];
            __syncwarp();
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, O + oi * 16 * D + oj * 16, D, wmma::mem_row_major);
            for (int k = 0; k < 32; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(a, Ph + oi * 16 * 32 + k, 32);
                wmma::load_matrix_sync(b, Vh + k * D + oj * 16, D);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(O + oi * 16 * D + oj * 16, acc, D, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 32 * D; e += blockDim.x) {
        const int r = e / D, i = e % D, t = tq0 + r / G, h = kvh * G + r % G;
        if (t >= N) continue;
        const float *gate = qf + (int64_t)t * H * 2 * D + (int64_t)h * 2 * D + D;
        out[(int64_t)t * H * D + (int64_t)h * D + i] = (ls[r] > 0.0f ? O[e] / ls[r] : 0.0f) * sigmoidf_(gate[i]);
    }
}

/* D += A B for one m16n8k16 tile: fp16 inputs, fp32 accumulators (PTX ISA layouts, g = lane / 4,
 * t = lane % 4): a0 (row g, cols 2t, 2t+1), a1 (row g+8), a2 (row g, cols 2t+8, 2t+9), a3 (row g+8,
 * cols 2t+8..); b0 (rows 2t, 2t+1, col g), b1 (rows 2t+8, 2t+9); c0 c1 (row g, cols 2t, 2t+1), c2 c3
 * (row g+8) */
__device__ __forceinline__ void mma16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_h2(float x, float y)
{
    const __half2 h = __floats2half2_rn(x, y);
    return *(const uint32_t *)(const void *)&h;
}

__device__ __forceinline__ uint32_t ld_u32(const __half *p) { return *(const uint32_t *)(const void *)p; }

/* 16 cells of K (rows of DT) and V (transposed: Vt[d][cell]) into shared fp16 */
#define FM_KP 8                     /* row padding (halves): no bank conflicts on the fragment loads */
#define FM_VS 24                    /* Vt row stride: 16 cells + 8 */
__device__ __forceinline__ void kv_tile16(const __half *kc, const __half *vc, const float *, const float *, int kb, int nk,
                                          int Hkv, int kvh, int DT, __half *Ks, __half *Vt)
{
    for (int e = threadIdx.x; e < 16 * (DT / 8); e += blockDim.x) {
        const int c = e / (DT / 8), i = (e % (DT / 8)) * 8;
        uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
        if (c < nk) {
            const int64_t o = ((int64_t)(kb + c) * Hkv + kvh) * DT + i;
            kv = *(const uint4 *)(const void *)(kc + o);
            vv = *(const uint4 *)(const void *)(vc + o);
        }
        *(uint4 *)(void *)(Ks + c * (DT + FM_KP) + i) = kv;
        const __half *v = (const __half *)&vv;
#pragma unroll
        for (int j = 0; j < 8; j++) Vt[(i + j) * FM_VS + c] = v[j];
    }
}

__device__ __forceinline__ void kv_tile16(const float *kc, const float *vc, const float *, const float *, int kb, int nk,
                                          int Hkv, int kvh, int DT, __half *Ks, __half *Vt)
{
    for (int e = threadIdx.x; e < 16 * (DT / 8); e += blockDim.x) {
        const int c = e / (DT / 8), i = (e % (DT / 8)) * 8;
        const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 k0 = z, k1 = z, v0 = z, v1 = z;
        if (c < nk) {
            const int64_t o = ((int64_t)(kb + c) * Hkv + kvh) * DT + i;
            k0 = *(const float4 *)(kc + o); k1 = *(const float4 *)(kc + o + 4);
            v0 = *(const float4 *)(vc + o); v1 = *(const float4 *)(vc + o + 4);
        }
        *(uint4 *)(void *)(Ks + c * (DT + FM_KP) + i) = h8(k0, k1);
        const float v[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
#pragma unroll
        for (int j = 0; j < 8; j++) Vt[(i + j) * FM_VS + c] = __float2half(v[j]);
    }
}

/* int8 K/V (kv_q8): 8 values a load, times their row's scale */
__device__ __forceinline__ void kv_tile16(const int8_t *kc, const int8_t *vc, const float *ks, const float *vs, int kb,
                                          int nk, int Hkv, int kvh, int DT, __half *Ks, __half *Vt)
{
    for (int e = threadIdx.x; e < 16 * (DT / 8); e += blockDim.x) {
        const int c = e / (DT / 8), i = (e % (DT / 8)) * 8;
        uint2 kq = make_uint2(0, 0), vq = kq;
        float sk = 0.0f, sv = 0.0f;
        if (c < nk) {
            const int64_t row = (int64_t)(kb + c) * Hkv + kvh, o = row * DT + i, b = row * (DT / 32) + i / 32;
            kq = *(const uint2 *)(const void *)(kc + o);
            vq = *(const uint2 *)(const void *)(vc + o);
            sk = ks[b];
            sv = vs[b];
        }
        const int8_t *k8 = (const int8_t *)&kq, *v8 = (const int8_t *)&vq;
        __half kh[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            kh[j] = __float2half((float)k8[j] * sk);
            Vt[(i + j) * FM_VS + c] = __float2half((float)v8[j] * sv);
        }
        *(uint4 *)(void *)(Ks + c * (DT + FM_KP) + i) = *(const uint4 *)kh;
    }
}

/* the K/V of N tokens as int8 rows of D (blockDim = D), a scale per 32 values (amax / 127; each
 * warp one block): one per row was 9.5e-3 from float64 on the 27B (outliers in the keys) */
__global__ void k_kv_store_q8(const float *k, const float *v, int8_t *kc, int8_t *vc, float *ks, float *vs, const int *ppos,
                              int Hkv, int D)
{
    const int row = blockIdx.x, t = row / Hkv, h = row % Hkv, i = threadIdx.x;
    const int64_t dst = (int64_t)(*ppos + t) * Hkv + h;
    for (int w = 0; w < 2; w++) {
        const float x = (w ? v : k)[(int64_t)row * D + i];
        float m = fabsf(x);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        (w ? vc : kc)[dst * D + i] = (int8_t)__float2int_rn(m > 0.0f ? x * (127.0f / m) : 0.0f);
        if ((i & 31) == 0) (w ? vs : ks)[dst * (D / 32) + (i >> 5)] = m / 127.0f;
    }
}

/* Flash attention of a prompt batch, FlashAttention-2 style on mma.sync: block (kv head, QT tokens)
 * = 128 query rows (QT tokens x the G heads of that K/V head), 8 warps of 16 rows; each warp keeps
 * its rows' output (16 x DT, fp32) in registers, the scores of a 16-cell tile too: their
 * accumulator layout is the A operand's of P V, so the softmax works in place. K/V read once per
 * 16 tokens (the WMMA kernel: per 4, with O through shared memory). */
template <typename KT, int DT>
__global__ void __launch_bounds__(256) k_fattn_mma(const float *qf, const KT *kc, const KT *vc, float *out, const int *ppos,
                                                   int H, int Hkv, int N, const float *ks = NULL, const float *vs = NULL)
{
    extern __shared__ __align__(16) unsigned char smb[];
    __half *Qs = (__half *)smb, *Ks = Qs + 128 * (DT + FM_KP), *Vt = Ks + 16 * (DT + FM_KP);
    const int G = H / Hkv, QT = 128 / G, RV = QT * G;       /* rows from RV on: padding (G = 6) */
    const int kvh = blockIdx.x, tq0 = blockIdx.y * QT;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int p0 = *ppos;
    const float scale = rsqrtf((float)DT);
    for (int e = threadIdx.x; e < 128 * DT; e += blockDim.x) {
        const int r = e / DT, i = e % DT, t = r < RV ? tq0 + r / G : N;
        Qs[r * (DT + FM_KP) + i] =
            __float2half(t < N ? qf[(int64_t)t * H * 2 * DT + (int64_t)(kvh * G + r % G) * 2 * DT + i] * scale : 0.0f);
    }
    const int r0 = wp * 16 + g, r1 = r0 + 8;                 /* this lane's two rows */
    const int t0 = r0 < RV ? tq0 + r0 / G : N, t1 = r1 < RV ? tq0 + r1 / G : N;
    float o[DT / 8][4];
#pragma unroll
    for (int j = 0; j < DT / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;
    const __half *q0 = Qs + r0 * (DT + FM_KP), *q1 = Qs + r1 * (DT + FM_KP);
    const int Tb = p0 + (tq0 + QT < N ? tq0 + QT : N);
    for (int kb = 0; kb < Tb; kb += 16) {
        const int nk = Tb - kb < 16 ? Tb - kb : 16;
        __syncthreads();                                     /* Q stored / the last tile used */
        kv_tile16(kc, vc, ks, vs, kb, nk, Hkv, kvh, DT, Ks, Vt);
        __syncthreads();
        float s[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
#pragma unroll 4
        for (int k = 0; k < DT; k += 16) {                   /* S = Q K^T: two n8 tiles of cells */
            const uint32_t a[4] = { ld_u32(q0 + k + 2 * tq), ld_u32(q1 + k + 2 * tq), ld_u32(q0 + k + 8 + 2 * tq),
                                    ld_u32(q1 + k + 8 + 2 * tq) };
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const __half *kr = Ks + (8 * j + g) * (DT + FM_KP) + k;
                mma16816(s[j], a, ld_u32(kr + 2 * tq), ld_u32(kr + 8 + 2 * tq));
            }
        }
        /* mask (cells past the tile, after the row's token, rows past the batch), online softmax */
        float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int c = 8 * j + 2 * tq + e;
                const bool ok0 = c < nk && t0 < N && kb + c <= p0 + t0, ok1 = c < nk && t1 < N && kb + c <= p0 + t1;
                s[j][e] = ok0 ? s[j][e] : -1e30f;
                s[j][2 + e] = ok1 ? s[j][2 + e] : -1e30f;
                mx0 = fmaxf(mx0, s[j][e]);
                mx1 = fmaxf(mx1, s[j][2 + e]);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1), f0 = expf(m0 - mn0), f1 = expf(m1 - mn1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[j][e] = s[j][e] > -1e29f ? expf(s[j][e] - mn0) : 0.0f;
                s[j][2 + e] = s[j][2 + e] > -1e29f ? expf(s[j][2 + e] - mn1) : 0.0f;
                sum0 += s[j][e];
                sum1 += s[j][2 + e];
            }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1); sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1); sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l0 = l0 * f0 + sum0;
        l1 = l1 * f1 + sum1;
        m0 = mn0;
        m1 = mn1;
        /* O = O f + P V: P's two n8 tiles are the A operand over the 16 cells */
        const uint32_t a[4] = { pack_h2(s[0][0], s[0][1]), pack_h2(s[0][2], s[0][3]), pack_h2(s[1][0], s[1][1]),
                                pack_h2(s[1][2], s[1][3]) };
#pragma unroll
        for (int j = 0; j < DT / 8; j++) {
            o[j][0] *= f0; o[j][1] *= f0; o[j][2] *= f1; o[j][3] *= f1;
            const __half *vr = Vt + (8 * j + g) * FM_VS;
            mma16816(o[j], a, ld_u32(vr + 2 * tq), ld_u32(vr + 8 + 2 * tq));
        }
    }
    /* out = O / l, gated; rows past the batch are not written */
    const int h0 = kvh * G + r0 % G, h1 = kvh * G + r1 % G;
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < DT / 8; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int d = 8 * j + 2 * tq + e;
            if (t0 < N) out[(int64_t)t0 * H * DT + (int64_t)h0 * DT + d] =
                o[j][e] * i0 * sigmoidf_(qf[(int64_t)t0 * H * 2 * DT + (int64_t)h0 * 2 * DT + DT + d]);
            if (t1 < N) out[(int64_t)t1 * H * DT + (int64_t)h1 * DT + d] =
                o[j][2 + e] * i1 * sigmoidf_(qf[(int64_t)t1 * H * 2 * DT + (int64_t)h1 * 2 * DT + DT + d]);
        }
}

/* One token's attention on mma.sync: block (kv head, segment) is one warp; the G heads of the
 * K/V head are the rows of the 16-row tiles (the rest zero), the segment's cells go by 16; the
 * partial (O, m, l) goes to part for k_fattn_merge. FA_SEG_MMA segments: enough blocks for the
 * GPU at any length (the CUDA-core decode kernel took 90 ms a token at 128k cells). */
#define FA_SEG_MMA 128
template <typename KT, int DT>
__global__ void __launch_bounds__(32) k_fattn_dec_mma(const float *qf, const KT *kc, const KT *vc, float *part,
                                                      const int *ppos, int H, int Hkv, const float *ks = NULL,
                                                      const float *vs = NULL, int toff = 0)
{
    extern __shared__ __align__(16) unsigned char smb[];
    __half *Qs = (__half *)smb, *Ks = Qs + 16 * (DT + FM_KP), *Vt = Ks + 16 * (DT + FM_KP);
    const int G = H / Hkv, kvh = blockIdx.x, seg = blockIdx.y, S = gridDim.y;
    const int lane = threadIdx.x, g = lane >> 2, tq = lane & 3;
    const int tt = toff + blockIdx.z, T = *ppos + tt + 1;     /* token tt of a few (a verification): blockIdx.z */
    const int L = ((T + S - 1) / S + 15) & ~15, k0 = seg * L, k1 = T < k0 + L ? T : k0 + L;
    const float scale = rsqrtf((float)DT);
    for (int e = lane; e < 16 * DT; e += 32) {
        const int r = e / DT, i = e % DT;
        Qs[r * (DT + FM_KP) + i] = __float2half(r < G ? qf[(int64_t)tt * H * 2 * DT + (int64_t)(kvh * G + r) * 2 * DT + i] * scale : 0.0f);
    }
    float o[DT / 8][4];
#pragma unroll
    for (int j = 0; j < DT / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;
    const __half *q0 = Qs + g * (DT + FM_KP), *q1 = Qs + (g + 8) * (DT + FM_KP);
    for (int kb = k0; kb < k1; kb += 16) {
        const int nk = k1 - kb < 16 ? k1 - kb : 16;
        __syncwarp();
        kv_tile16(kc, vc, ks, vs, kb, nk, Hkv, kvh, DT, Ks, Vt);
        __syncwarp();
        float s[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
#pragma unroll 4
        for (int k = 0; k < DT; k += 16) {
            const uint32_t a[4] = { ld_u32(q0 + k + 2 * tq), ld_u32(q1 + k + 2 * tq), ld_u32(q0 + k + 8 + 2 * tq),
                                    ld_u32(q1 + k + 8 + 2 * tq) };
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const __half *kr = Ks + (8 * j + g) * (DT + FM_KP) + k;
                mma16816(s[j], a, ld_u32(kr + 2 * tq), ld_u32(kr + 8 + 2 * tq));
            }
        }
        float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const bool ok = 8 * j + 2 * tq + e < nk;
                s[j][e] = ok ? s[j][e] : -1e30f;
                s[j][2 + e] = ok ? s[j][2 + e] : -1e30f;
                mx0 = fmaxf(mx0, s[j][e]);
                mx1 = fmaxf(mx1, s[j][2 + e]);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1), f0 = expf(m0 - mn0), f1 = expf(m1 - mn1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[j][e] = s[j][e] > -1e29f ? expf(s[j][e] - mn0) : 0.0f;
                s[j][2 + e] = s[j][2 + e] > -1e29f ? expf(s[j][2 + e] - mn1) : 0.0f;
                sum0 += s[j][e];
                sum1 += s[j][2 + e];
            }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1); sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1); sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l0 = l0 * f0 + sum0;
        l1 = l1 * f1 + sum1;
        m0 = mn0;
        m1 = mn1;
        const uint32_t a[4] = { pack_h2(s[0][0], s[0][1]), pack_h2(s[0][2], s[0][3]), pack_h2(s[1][0], s[1][1]),
                                pack_h2(s[1][2], s[1][3]) };
#pragma unroll
        for (int j = 0; j < DT / 8; j++) {
            o[j][0] *= f0; o[j][1] *= f0; o[j][2] *= f1; o[j][3] *= f1;
            const __half *vr = Vt + (8 * j + g) * FM_VS;
            mma16816(o[j], a, ld_u32(vr + 2 * tq), ld_u32(vr + 8 + 2 * tq));
        }
    }
    /* the partial of each real row (g, and g + 8 when G > 8) */
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const int r = g + 8 * half;
        if (r >= G) continue;
        float *pp = part + (((int64_t)seg * gridDim.z + blockIdx.z) * H + kvh * G + r) * (DT + 2);
#pragma unroll
        for (int j = 0; j < DT / 8; j++) {
            pp[8 * j + 2 * tq] = o[j][2 * half];
            pp[8 * j + 2 * tq + 1] = o[j][2 * half + 1];
        }
        if (tq == 0) { pp[DT] = half ? m1 : m0; pp[DT + 1] = half ? l1 : l0; }
    }
}

/* the S segments' partials of (head blockIdx.x, token blockIdx.y), gated */
__global__ void k_fattn_merge(const float *part, const float *qf, float *out, int H, int D, int N, int S)
{
    const int h = blockIdx.x, t = blockIdx.y;
    const int64_t st = (int64_t)N * H * (D + 2);
    const float *p = part + ((int64_t)t * H + h) * (D + 2);
    float M = -1e30f, L = 0.0f;
    for (int s = 0; s < S; s++) M = fmaxf(M, p[s * st + D]);
    for (int s = 0; s < S; s++) L += expf(p[s * st + D] - M) * p[s * st + D + 1];
    const float *gate = qf + (int64_t)t * H * 2 * D + (int64_t)h * 2 * D + D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < S; s++) a += expf(p[s * st + D] - M) * p[s * st + i];
        out[(int64_t)t * H * D + (int64_t)h * D + i] = (L > 0.0f ? a / L : 0.0f) * sigmoidf_(gate[i]);
    }
}

__global__ void k_swiglu(const float *gu, float *h, int F)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < F) h[i] = siluf_(gu[i]) * gu[F + i];
}

/* y = routed + sigmoid(sg) * shared */
__global__ void k_moe_final(float *y, const float *routed, const float *sh, const float *sg, int E)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E) y[i] = routed[i] + sigmoidf_(sg[0]) * sh[i];
}

/* ------------------------------------------------- routed experts in VRAM -- */
/* Expert rows by type, lane-level partial sums (the caller reduces over the warp). The
 * weight arithmetic is q38_dequant_row's: d*sc*q - dmin*m (Q4_K/Q5_K), d*q + m (Q5_1). */
__device__ __forceinline__ float h2f_d(const uint8_t *p) { return __half2float(*(const __half *)(const void *)p); }

__device__ __forceinline__ void scale_min_k4_d(int j, const uint8_t *q, int *d, int *m)
{
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

__device__ __forceinline__ int sc_byte(const uint32_t *w3, int i) { return (w3[i >> 2] >> (8 * (i & 3))) & 0xFF; }

/* -- the IQ types (UD-IQ4_XS's experts): IQ4_NL, IQ4_XS, IQ3_S, kept as stored (a slot holds
 * them unrearranged). A row is cut into chunks of 8 weights, each decoded to floats with
 * q38_dequant_row's arithmetic and applied to two runs of 4 activations, x[o0..o0+4) and
 * x[o1..o1+4); lane l takes chunks l, l+32, ... */
__device__ const int8_t d_kv_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
__device__ const uint32_t d_iq3s_grid[512] = {
#include "q38_iq3s_grid.inc"
};

/* (Q6_K, Qwen3.6-35B-A3B's gate/up experts, goes the same way) */
__device__ __forceinline__ bool is_iq(int type)
{
    return type == Q38_IQ4_NL || type == Q38_IQ4_XS || type == Q38_IQ3_S || type == Q38_Q6_K;
}

__device__ __forceinline__ int64_t iq_row_bytes(int type, int ne0)
{
    if (type == Q38_IQ4_NL) return (int64_t)(ne0 / 32) * 18;
    return (int64_t)(ne0 / 256) * (type == Q38_IQ4_XS ? 136 : type == Q38_Q6_K ? 210 : 110);
}

__device__ __forceinline__ int iq_chunks(int type, int ne0) { return type == Q38_IQ4_NL ? (ne0 / 32) * 4 : (ne0 / 256) * 32; }

__device__ __forceinline__ void iq_chunk(int type, const uint8_t *w, int c, float *wt, int *o0, int *o1)
{
    if (type == Q38_Q6_K) {
        /* super-block sb, half n (128), quarter q (32): runs of 4 at l0 and l0 + 16; element l of
         * a quarter is ql[l (+32 for q odd)] low (q < 2) or high nibble, 2 bits of qh[l] at 2q,
         * minus 32, times d * scale[l / 16 + 2q] (q38_dequant_row's dequant_q6_K) */
        const int sb = c >> 5, n = (c >> 4) & 1, q = (c >> 2) & 3, l0 = (c & 3) * 4;
        const uint8_t *blk = w + (int64_t)sb * 210;
        const float d = __half2float(__ushort_as_half((unsigned short)(blk[208] | (blk[209] << 8))));
        const uint8_t *ql = blk + 64 * n + ((q & 1) ? 32 : 0), *qh = blk + 128 + 32 * n;
        const int8_t *sc = (const int8_t *)(blk + 192 + 8 * n);
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int lb = l0 + 16 * r;
            const float ds = d * sc[lb / 16 + 2 * q];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int l = lb + k;
                const int nib = q < 2 ? (ql[l] & 0xF) : (ql[l] >> 4);
                wt[4 * r + k] = ds * (float)((nib | (((qh[l] >> (2 * q)) & 3) << 4)) - 32);
            }
        }
        *o0 = 256 * sb + 128 * n + 32 * q + l0;
        *o1 = *o0 + 16;
    } else if (type == Q38_IQ4_NL) {                /* block of 32: bytes p..p+3 -> elements p.., 16+p.. */
        const int b = c >> 2, p = (c & 3) * 4;
        const uint8_t *blk = w + (int64_t)b * 18;
        const float d = __half2float(__ushort_as_half((unsigned short)(blk[0] | (blk[1] << 8))));
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int q = blk[2 + p + k];
            wt[k] = d * (float)__ldg(&d_kv_iq4nl[q & 15]);
            wt[4 + k] = d * (float)__ldg(&d_kv_iq4nl[q >> 4]);
        }
        *o0 = 32 * b + p;
        *o1 = *o0 + 16;
    } else if (type == Q38_IQ4_XS) {                /* super-block of 256: sub-block j, bytes p..p+3 */
        const int sb = c >> 5, j = (c & 31) >> 2, p = (c & 3) * 4;
        const uint8_t *blk = w + (int64_t)sb * 136;
        const float d = __half2float(__ushort_as_half((unsigned short)(blk[0] | (blk[1] << 8))));
        const int sh = blk[2] | (blk[3] << 8);
        const int ls = ((blk[4 + j / 2] >> 4 * (j % 2)) & 0xf) | (((sh >> 2 * j) & 3) << 4);
        const float dl = d * (ls - 32);
        const uint8_t *qs = blk + 8 + 16 * j + p;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            wt[k] = dl * (float)__ldg(&d_kv_iq4nl[qs[k] & 15]);
            wt[4 + k] = dl * (float)__ldg(&d_kv_iq4nl[qs[k] >> 4]);
        }
        *o0 = 256 * sb + 32 * j + p;
        *o1 = *o0 + 16;
    } else {                                        /* IQ3_S: sub-block j, group pair l: 8 elements */
        const int sb = c >> 5, j = (c & 31) >> 2, l = c & 3;
        const uint8_t *blk = w + (int64_t)sb * 110;
        const float d = __half2float(__ushort_as_half((unsigned short)(blk[0] | (blk[1] << 8))));
        const uint8_t *qs = blk + 2 + 8 * j;
        const int qh = blk[66 + j], signs = blk[74 + 4 * j + l];
        const float db = d * (1 + 2 * ((blk[106 + j / 2] >> 4 * (j & 1)) & 0xf));
        const uint32_t g1 = __ldg(&d_iq3s_grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)]);
        const uint32_t g2 = __ldg(&d_iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)]);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            wt[k] = db * (float)((g1 >> (8 * k)) & 0xFF) * (signs & (1 << k) ? -1.f : 1.f);
            wt[4 + k] = db * (float)((g2 >> (8 * k)) & 0xFF) * (signs & (1 << (k + 4)) ? -1.f : 1.f);
        }
        *o0 = 256 * sb + 32 * j + 8 * l;
        *o1 = *o0 + 4;
    }
}

__device__ __forceinline__ float iq_apply(const float *wt, const float *x, int o0, int o1)
{
    return wt[0] * x[o0] + wt[1] * x[o0 + 1] + wt[2] * x[o0 + 2] + wt[3] * x[o0 + 3] +
           wt[4] * x[o1] + wt[5] * x[o1 + 1] + wt[6] * x[o1 + 2] + wt[7] * x[o1 + 3];
}

__device__ float iq_lane_dot(int type, const uint8_t *w, const float *xs, int ne0, int lane)
{
    float acc = 0.0f, wt[8];
    const int n = iq_chunks(type, ne0);
    for (int c = lane; c < n; c += 32) {
        int o0, o1;
        iq_chunk(type, w, c, wt, &o0, &o1);
        acc += iq_apply(wt, xs, o0, o1);
    }
    return acc;
}

__device__ float lane_dot(int type, const uint8_t *w, const float *xs, int ne0, int lane)
{
    if (is_iq(type)) return iq_lane_dot(type, w, xs, ne0, lane);
    float acc = 0.0f;
    const int g = lane >> 3, t = lane & 7;          /* 4 groups of 8 lanes, one 32-block each */
    if (type == Q38_Q8_0) {
        for (int b = g; b < ne0 / 32; b += 4) {
            const uint8_t *blk = w + (int64_t)b * 34;
            const float d = h2f_d(blk);
            const uint16_t qa = *(const uint16_t *)(const void *)(blk + 2 + 4 * t);
            const uint16_t qb = *(const uint16_t *)(const void *)(blk + 4 + 4 * t);
            const float4 x = *(const float4 *)(xs + b * 32 + 4 * t);
            acc += d * ((float)(int8_t)(qa & 0xFF) * x.x + (float)(int8_t)(qa >> 8) * x.y +
                        (float)(int8_t)(qb & 0xFF) * x.z + (float)(int8_t)(qb >> 8) * x.w);
        }
    } else if (type == Q38_Q5_1) {
        /* lane t of a group: bytes 2t, 2t+1 of qs -> elements 2t, 2t+1 (low nibbles) and
         * 16+2t, 17+2t (high nibbles); the 5th bit of element j is bit j of qh */
        for (int b = g; b < ne0 / 32; b += 4) {
            const uint8_t *blk = w + (int64_t)b * 24;
            const float2 dm = __half22float2(*(const __half2 *)(const void *)blk);
            const uint32_t qh = *(const uint32_t *)(const void *)(blk + 4);
            const uint16_t q2 = *(const uint16_t *)(const void *)(blk + 8 + 2 * t);
            const int q0 = q2 & 0xFF, q1 = q2 >> 8;
            const float v0 = (float)((q0 & 15) | (((qh >> (2 * t)) & 1) << 4));
            const float v1 = (float)((q1 & 15) | (((qh >> (2 * t + 1)) & 1) << 4));
            const float v2 = (float)((q0 >> 4) | (((qh >> (16 + 2 * t)) & 1) << 4));
            const float v3 = (float)((q1 >> 4) | (((qh >> (17 + 2 * t)) & 1) << 4));
            const float2 xa = *(const float2 *)(xs + b * 32 + 2 * t), xb = *(const float2 *)(xs + b * 32 + 16 + 2 * t);
            acc += (v0 * dm.x + dm.y) * xa.x + (v1 * dm.x + dm.y) * xa.y + (v2 * dm.x + dm.y) * xb.x + (v3 * dm.x + dm.y) * xb.y;
        }
    } else if (type == Q38_Q4_K || type == Q38_Q5_K) {
        /* lane l: chunk j = l / 8 of the super-block, bytes p..p+3 of its 32 (p = 4 * (l % 8)):
         * low nibbles are elements 64j+p.., high nibbles 64j+32+p.. */
        const int five = type == Q38_Q5_K, bs = five ? 176 : 144, j = lane >> 3, p = (lane & 7) * 4;
        for (int b = 0; b < ne0 / 256; b++) {
            const uint8_t *blk = w + (int64_t)b * bs;
            const float2 dd = __half22float2(*(const __half2 *)(const void *)blk);
            const uint32_t *scw = (const uint32_t *)(const void *)(blk + 4);
            const uint32_t w3[3] = { scw[0], scw[1], scw[2] };
            int s1, m1, s2, m2;
            /* get_scale_min_k4 for sub-blocks 2j and 2j+1, from the three words */
            {
                const int i1 = 2 * j, i2 = 2 * j + 1;
                if (i1 < 4) { s1 = sc_byte(w3, i1) & 63; m1 = sc_byte(w3, i1 + 4) & 63; }
                else { s1 = (sc_byte(w3, i1 + 4) & 0xF) | ((sc_byte(w3, i1 - 4) >> 6) << 4);
                       m1 = (sc_byte(w3, i1 + 4) >> 4) | ((sc_byte(w3, i1) >> 6) << 4); }
                if (i2 < 4) { s2 = sc_byte(w3, i2) & 63; m2 = sc_byte(w3, i2 + 4) & 63; }
                else { s2 = (sc_byte(w3, i2 + 4) & 0xF) | ((sc_byte(w3, i2 - 4) >> 6) << 4);
                       m2 = (sc_byte(w3, i2 + 4) >> 4) | ((sc_byte(w3, i2) >> 6) << 4); }
            }
            const uint32_t q = *(const uint32_t *)(const void *)(blk + (five ? 48 : 16) + 32 * j + p);
            const uint32_t qh = five ? *(const uint32_t *)(const void *)(blk + 16 + p) : 0u;
            const float4 xl = *(const float4 *)(xs + b * 256 + 64 * j + p);
            const float4 xh = *(const float4 *)(xs + b * 256 + 64 * j + 32 + p);
            float lo[4], hi[4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int qb = (q >> (8 * u)) & 0xFF, hb = (qh >> (8 * u)) & 0xFF;
                lo[u] = (float)((qb & 15) + (five ? (((hb >> (2 * j)) & 1) << 4) : 0));
                hi[u] = (float)((qb >> 4) + (five ? (((hb >> (2 * j + 1)) & 1) << 4) : 0));
            }
            const float sl = lo[0] * xl.x + lo[1] * xl.y + lo[2] * xl.z + lo[3] * xl.w;
            const float sh2 = hi[0] * xh.x + hi[1] * xh.y + hi[2] * xh.z + hi[3] * xh.w;
            const float xsl = xl.x + xl.y + xl.z + xl.w, xsh = xh.x + xh.y + xh.z + xh.w;
            acc += dd.x * s1 * sl - dd.y * m1 * xsl + dd.x * s2 * sh2 - dd.y * m2 * xsh;
        }
    }
    return acc;
}

/* ------------------------------------------- experts repacked for the device -- */
/* An expert slot holds gate | up | down with each matrix rearranged (same bytes, same
 * values) so a lane reads 16 contiguous bytes of quants:
 *   Q4_K  qs[rows][ne0/2] | hdr[rows][nsb][16]          (hdr = d, dmin, scales[12])
 *   Q5_K  qs[rows][ne0/2] | qh[rows][nsb][32] | hdr[rows][nsb][16]
 *   Q5_1  qs[rows][ne0/2] | qh[rows][nb] u32 | dm[rows][nb] half2
 *   Q8_0  qs[rows][ne0]   | d[rows][nb] half                                          */
/* One 32-bit word (Q8_0: 16-bit) per thread, so neighbouring threads read neighbouring words;
 * the three matrices of an expert in one launch (blockIdx.y). */
__device__ __forceinline__ void repack_unit(const uint8_t *src, uint8_t *dst, int type, int rows, int ne0, int64_t w)
{
    if (type == Q38_Q4_K || type == Q38_Q5_K) {
        const int nsb = ne0 / 256, five = type == Q38_Q5_K, wpb = five ? 44 : 36;
        const int64_t i = w / wpb, r = i / nsb, b = i % nsb;
        const int k = (int)(w % wpb);
        const uint32_t v = ((const uint32_t *)(const void *)src)[w];
        const int64_t qsz = (int64_t)rows * (ne0 / 2);
        uint8_t *o;
        if (k < 4) o = dst + qsz + (five ? (int64_t)rows * nsb * 32 : 0) + i * 16 + 4 * k;
        else if (five && k < 12) o = dst + qsz + i * 32 + 4 * (k - 4);
        else o = dst + r * (ne0 / 2) + b * 128 + 4 * (k - (five ? 12 : 4));
        *(uint32_t *)(void *)o = v;
    } else if (type == Q38_Q5_1) {
        const int nb = ne0 / 32;
        const int64_t i = w / 6, r = i / nb, b = i % nb;
        const int k = (int)(w % 6);
        const uint32_t v = ((const uint32_t *)(const void *)src)[w];
        const int64_t qsz = (int64_t)rows * (ne0 / 2);
        uint8_t *o;
        if (k == 0) o = dst + qsz + (int64_t)rows * nb * 4 + i * 4;
        else if (k == 1) o = dst + qsz + i * 4;
        else o = dst + r * (ne0 / 2) + b * 16 + 4 * (k - 2);
        *(uint32_t *)(void *)o = v;
    } else if (type == Q38_Q8_0) {
        const int nb = ne0 / 32;
        const int64_t i = w / 17, r = i / nb, b = i % nb;
        const int k = (int)(w % 17);
        const uint16_t v = ((const uint16_t *)(const void *)src)[w];
        uint8_t *o = k == 0 ? dst + (int64_t)rows * ne0 + i * 2 : dst + r * ne0 + b * 32 + 2 * (k - 1);
        *(uint16_t *)(void *)o = v;
    } else if (is_iq(type)) {                       /* kept as stored: copied a 16-bit word at a time */
        ((uint16_t *)(void *)dst)[w] = ((const uint16_t *)(const void *)src)[w];
    }
}

__device__ __forceinline__ int64_t repack_units(int type, int rows, int ne0)
{
    if (type == Q38_Q4_K) return (int64_t)rows * (ne0 / 256) * 36;
    if (type == Q38_Q5_K) return (int64_t)rows * (ne0 / 256) * 44;
    if (type == Q38_Q5_1) return (int64_t)rows * (ne0 / 32) * 6;
    if (type == Q38_Q8_0) return (int64_t)rows * (ne0 / 32) * 17;
    if (is_iq(type)) return (int64_t)rows * iq_row_bytes(type, ne0) / 2;
    return 0;
}

__global__ void k_repack3(const uint8_t *src, uint8_t *dst, int t0, int t1, int t2, int F, int E, int64_t bg, int64_t bu)
{
    const int y = blockIdx.y;
    const int type = y == 0 ? t0 : (y == 1 ? t1 : t2), rows = y < 2 ? F : E, ne0 = y < 2 ? E : F;
    const int64_t off = y == 0 ? 0 : (y == 1 ? bg : bg + bu);
    const int64_t n = repack_units(type, rows, ne0);
    for (int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; w < n; w += (int64_t)gridDim.x * blockDim.x)
        repack_unit(src + off, dst + off, type, rows, ne0, w);
}

static void repack_expert(const uint8_t *raw, uint8_t *dst, const int *ty, int F, int E, int64_t bg, int64_t bu,
                          cudaStream_t st)
{
    k_repack3<<<dim3(256, 3), 256, 0, st>>>(raw, dst, ty[0], ty[1], ty[2], F, E, bg, bu);
}

/* byte i (0..11) of the scales, held in three registers: selects, not an indexed array
 * (a runtime-indexed array would live in local memory) */
__device__ __forceinline__ int scb(uint32_t w0, uint32_t w1, uint32_t w2, int i)
{
    const uint32_t w = i < 4 ? w0 : (i < 8 ? w1 : w2);
    return (int)((w >> (8 * (i & 3))) & 0xFF);
}

__device__ __forceinline__ void kscales(const uint8_t *hd, int j, float *s1, float *m1, float *s2, float *m2)
{
    const uint32_t w0 = *(const uint32_t *)(const void *)(hd + 4), w1 = *(const uint32_t *)(const void *)(hd + 8),
                   w2 = *(const uint32_t *)(const void *)(hd + 12);
    const int i1 = 2 * j, i2 = 2 * j + 1;
    int a, b;
    if (i1 < 4) { a = scb(w0, w1, w2, i1) & 63; b = scb(w0, w1, w2, i1 + 4) & 63; }
    else { a = (scb(w0, w1, w2, i1 + 4) & 0xF) | ((scb(w0, w1, w2, i1 - 4) >> 6) << 4);
           b = (scb(w0, w1, w2, i1 + 4) >> 4) | ((scb(w0, w1, w2, i1) >> 6) << 4); }
    *s1 = (float)a; *m1 = (float)b;
    if (i2 < 4) { a = scb(w0, w1, w2, i2) & 63; b = scb(w0, w1, w2, i2 + 4) & 63; }
    else { a = (scb(w0, w1, w2, i2 + 4) & 0xF) | ((scb(w0, w1, w2, i2 - 4) >> 6) << 4);
           b = (scb(w0, w1, w2, i2 + 4) >> 4) | ((scb(w0, w1, w2, i2) >> 6) << 4); }
    *s2 = (float)a; *m2 = (float)b;
}

/* row r of a repacked matrix (base = start of the matrix region) times xs; warp-reduced by the caller */
__device__ float lane_dot_rp(int type, const uint8_t *base, int rows, int r, const float *xs, int ne0, int lane)
{
    if (is_iq(type)) return iq_lane_dot(type, base + (int64_t)r * iq_row_bytes(type, ne0), xs, ne0, lane);
    float acc = 0.0f;
    if (type == Q38_Q4_K || type == Q38_Q5_K) {
        const int nsb = ne0 / 256, five = type == Q38_Q5_K;
        const uint8_t *qs = base + (int64_t)r * (ne0 / 2);
        const uint8_t *qh = base + (int64_t)rows * (ne0 / 2) + (int64_t)r * nsb * 32;
        const uint8_t *hd = base + (int64_t)rows * (ne0 / 2) + (five ? (int64_t)rows * nsb * 32 : 0) + (int64_t)r * nsb * 16;
        for (int c = lane; c < nsb * 8; c += 32) {               /* 16-byte chunks of qs */
            const int sb = c >> 3, bo = (c & 7) * 16, j = bo >> 5, p = bo & 31;
            const uint8_t *h = hd + sb * 16;
            const float2 dd = __half22float2(*(const __half2 *)(const void *)h);
            float s1, m1, s2, m2;
            kscales(h, j, &s1, &m1, &s2, &m2);
            const int4 q4 = *(const int4 *)(const void *)(qs + sb * 128 + bo);
            int4 h4 = make_int4(0, 0, 0, 0);
            if (five) h4 = *(const int4 *)(const void *)(qh + sb * 32 + p);
            const uint32_t qw[4] = { (uint32_t)q4.x, (uint32_t)q4.y, (uint32_t)q4.z, (uint32_t)q4.w };
            const uint32_t hw[4] = { (uint32_t)h4.x, (uint32_t)h4.y, (uint32_t)h4.z, (uint32_t)h4.w };
            const float4 *xl4 = (const float4 *)(xs + sb * 256 + 64 * j + p), *xh4 = xl4 + 8;
            float xl[16], xh[16];
#pragma unroll
            for (int v = 0; v < 4; v++) {
                const float4 a = __ldg(xl4 + v), b2 = __ldg(xh4 + v);
                xl[4 * v] = a.x; xl[4 * v + 1] = a.y; xl[4 * v + 2] = a.z; xl[4 * v + 3] = a.w;
                xh[4 * v] = b2.x; xh[4 * v + 1] = b2.y; xh[4 * v + 2] = b2.z; xh[4 * v + 3] = b2.w;
            }
            float sl = 0.0f, sh = 0.0f, xsl = 0.0f, xsh = 0.0f;
#pragma unroll
            for (int u = 0; u < 16; u++) {
                const int qb = (qw[u >> 2] >> (8 * (u & 3))) & 0xFF;
                const int hb = (hw[u >> 2] >> (8 * (u & 3))) & 0xFF;
                const float lo = (float)((qb & 15) + (five ? (((hb >> (2 * j)) & 1) << 4) : 0));
                const float hi = (float)((qb >> 4) + (five ? (((hb >> (2 * j + 1)) & 1) << 4) : 0));
                sl += lo * xl[u]; sh += hi * xh[u]; xsl += xl[u]; xsh += xh[u];
            }
            acc += dd.x * s1 * sl - dd.y * m1 * xsl + dd.x * s2 * sh - dd.y * m2 * xsh;
        }
    } else if (type == Q38_Q5_1) {
        const int nb = ne0 / 32;
        const uint8_t *qs = base + (int64_t)r * (ne0 / 2);
        const uint32_t *qh = (const uint32_t *)(const void *)(base + (int64_t)rows * (ne0 / 2)) + (int64_t)r * nb;
        const __half2 *dm = (const __half2 *)(const void *)(base + (int64_t)rows * (ne0 / 2) + (int64_t)rows * nb * 4) + (int64_t)r * nb;
        for (int b = lane; b < nb; b += 32) {                     /* one 32-block per lane */
            const int4 q4 = *(const int4 *)(const void *)(qs + b * 16);
            const uint32_t h = qh[b];
            const float2 f = __half22float2(dm[b]);
            const uint32_t qw[4] = { (uint32_t)q4.x, (uint32_t)q4.y, (uint32_t)q4.z, (uint32_t)q4.w };
            const float4 *x4 = (const float4 *)(xs + b * 32);
            float x[32];
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const float4 a = __ldg(x4 + v);
                x[4 * v] = a.x; x[4 * v + 1] = a.y; x[4 * v + 2] = a.z; x[4 * v + 3] = a.w;
            }
            float s = 0.0f, xsum = 0.0f;
#pragma unroll
            for (int u = 0; u < 16; u++) {
                const int qb = (qw[u >> 2] >> (8 * (u & 3))) & 0xFF;
                const float v0 = (float)((qb & 15) | (((h >> u) & 1) << 4));
                const float v1 = (float)((qb >> 4) | (((h >> (u + 16)) & 1) << 4));
                s += v0 * x[u] + v1 * x[u + 16];
                xsum += x[u] + x[u + 16];
            }
            acc += f.x * s + f.y * xsum;
        }
    } else if (type == Q38_Q8_0) {
        const int nb = ne0 / 32;
        const int8_t *qs = (const int8_t *)(base + (int64_t)r * ne0);
        const __half *d = (const __half *)(const void *)(base + (int64_t)rows * ne0) + (int64_t)r * nb;
        for (int c = lane; c < nb * 2; c += 32) {                 /* 16-byte halves of the blocks */
            const int4 q4 = *(const int4 *)(const void *)(qs + c * 16);
            const int8_t *q = (const int8_t *)&q4;
            const float4 *x4 = (const float4 *)(xs + c * 16);
            float s = 0.0f;
#pragma unroll
            for (int v = 0; v < 4; v++) {
                const float4 a = __ldg(x4 + v);
                s += (float)q[4 * v] * a.x + (float)q[4 * v + 1] * a.y + (float)q[4 * v + 2] * a.z + (float)q[4 * v + 3] * a.w;
            }
            acc += __half2float(d[c >> 1]) * s;
        }
    }
    return acc;
}

/* gate and up rows of the resident experts (repacked): blockIdx.y = expert, a warp per row;
 * x is read through the L1 cache (staging it in shared memory per block cost more traffic
 * than the weights) */
__global__ void k_ex_gu_rp(const LayerDesc *ld, int tg, int tu, int E, int F, const float *__restrict__ x, float *gu)
{
    const int k = blockIdx.y;
    if (k >= ld->n) return;
    const int j = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (j >= 2 * F) return;
    const ExDesc *ex = ld->e + k;
    const float v = j < F ? lane_dot_rp(tg, ex->g, F, j, x, E, threadIdx.x)
                          : lane_dot_rp(tu, ex->u, F, j - F, x, E, threadIdx.x);
    const float s = warp_sum(v);
    if (threadIdx.x == 0) gu[(int64_t)k * 2 * F + j] = s;
}

__global__ void k_ex_down_rp(const LayerDesc *ld, int td, int F, int E, const float *__restrict__ h, float *dn)
{
    const int k = blockIdx.y;
    if (k >= ld->n) return;
    const int i = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (i >= E) return;
    const float s = warp_sum(lane_dot_rp(td, ld->e[k].d, E, i, h + (int64_t)k * F, F, threadIdx.x));
    if (threadIdx.x == 0) dn[(int64_t)k * E + i] = s;
}

/* gate and up rows of n experts: row r -> expert r / (2F), gate for j < F, up after */
__global__ void k_ex_gu(const LayerDesc *ld, int tg, int tu, int64_t rbg, int64_t rbu, int E, int F,
                        const float *x, float *gu)
{
    const ExDesc *ex = ld->e;
    const int n = ld->n;
    extern __shared__ float xs[];
    for (int i = threadIdx.y * 32 + threadIdx.x; i < E; i += 32 * ROWS_PER_BLOCK) xs[i] = x[i];
    __syncthreads();
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (r >= n * 2 * F) return;
    const int k = r / (2 * F), j = r % (2 * F);
    const float v = j < F ? lane_dot(tg, ex[k].g + (int64_t)j * rbg, xs, E, threadIdx.x)
                          : lane_dot(tu, ex[k].u + (int64_t)(j - F) * rbu, xs, E, threadIdx.x);
    const float s = warp_sum(v);
    if (threadIdx.x == 0) gu[r] = s;
}

__global__ void k_ex_swiglu(const LayerDesc *ld, const float *gu, float *h, int F)
{
    const int n = ld->n;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * F) return;
    const int k = i / F, j = i % F;
    h[i] = siluf_(gu[(int64_t)k * 2 * F + j]) * gu[(int64_t)k * 2 * F + F + j];
}

/* down rows of n experts: row r -> expert r / E */
__global__ void k_ex_down(const LayerDesc *ld, int td, int64_t rbd, int F, int E, const float *h, float *dn)
{
    const ExDesc *ex = ld->e;
    const int n = ld->n;
    extern __shared__ float hs[];
    for (int i = threadIdx.y * 32 + threadIdx.x; i < n * F; i += 32 * ROWS_PER_BLOCK) hs[i] = h[i];
    __syncthreads();
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (r >= n * E) return;
    const int k = r / E, i = r % E;
    const float s = warp_sum(lane_dot(td, ex[k].d + (int64_t)i * rbd, hs + (int64_t)k * F, F, threadIdx.x));
    if (threadIdx.x == 0) dn[r] = s;
}

__global__ void k_ex_sum(const LayerDesc *ld, const float *dn, int E, float *out)
{
    const ExDesc *ex = ld->e;
    const int n = ld->n;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    float a = 0.0f;
    for (int k = 0; k < n; k++) a += ex[k].w * dn[(int64_t)k * E + i];
    out[i] = a;
}

/* y = routed (host) + routed (device) + sigmoid(sg) * shared */
__global__ void k_moe_final2(float *y, const float *routed, const float *routed_dev, const float *sh, const float *sg, int E)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E) y[i] = routed[i] + routed_dev[i] + sigmoidf_(sg[0]) * sh[i];
}

/* ------------------------------------------------------------------- PLE -- */
/* gate[s] = sigmoid(sgn(d) sqrt(max(|d|, 1e-6))), d = rms(key_s)*nk_s . rms(res_s)*nq_s / sqrt(E) */
__global__ void k_ple_gate(const float *key, const float *res, const float *nk, const float *nq, float *gate, int E, float eps)
{
    __shared__ float part[8];
    const int s = blockIdx.x;
    const float *k = key + (int64_t)s * E, *r = res + (int64_t)s * E, *wk = nk + (int64_t)s * E, *wq = nq + (int64_t)s * E;
    float sk = 0.0f, sq = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { sk += k[i] * k[i]; sq += r[i] * r[i]; }
    sk = block_sum(sk, part);
    __syncthreads();
    sq = block_sum(sq, part);
    const float ck = 1.0f / sqrtf(sk / E + eps), cq = 1.0f / sqrtf(sq / E + eps);
    float d = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) d += (k[i] * ck * wk[i]) * (r[i] * cq * wq[i]);
    __syncthreads();
    d = block_sum(d, part);
    if (threadIdx.x == 0) {
        const float sc = d / sqrtf((float)E);
        const float mag = sqrtf(fabsf(sc) > 1e-6f ? fabsf(sc) : 1e-6f);
        gate[s] = sigmoidf_(sc > 0 ? mag : sc < 0 ? -mag : 0.0f);
    }
}

/* gated_s = value * gate[s]; normed_s = rms(gated_s) * nconv_s */
__global__ void k_ple_gated(const float *value, const float *gate, const float *nconv, float *gated, float *normed, int E, float eps)
{
    __shared__ float part[8];
    const int s = blockIdx.x;
    float *gs = gated + (int64_t)s * E, *ns = normed + (int64_t)s * E;
    const float gv = gate[s];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { const float v = value[i] * gv; gs[i] = v; ss += v * v; }
    ss = block_sum(ss, part);
    const float c = 1.0f / sqrtf(ss / E + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) ns[i] = gs[i] * c * nconv[(int64_t)s * E + i];
}

/* depthwise causal conv dilated by the n-gram size; res += gated + silu(conv) */
__global__ void k_ple_conv(float *res, const float *gated, const float *normed, float *ring, const float *w,
                           const int *ppos, int n, int K, int dil, int hist)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n) return;
    const int pos = *ppos;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        const int back = (K - 1 - k) * dil;
        const float x = back == 0 ? normed[ch] : (pos - back >= 0 ? ring[(int64_t)((pos - back) % hist) * n + ch] : 0.0f);
        acc += w[(int64_t)ch * K + k] * x;
    }
    res[ch] += gated[ch] + siluf_(acc);
    ring[(int64_t)(pos % hist) * n + ch] = normed[ch];
}

/* ------------------------------------------------------- the lightning indexer -- */
/* The same steps as indexer_select in q38_model.c (llama.cpp build_qsa_top_k): */

/* store this position's raw key; when it completes a block, pool, norm and rope the block */
__global__ void k_idx_store(const float *kraw, float *raw, float *pool, const float *kn, const int *ppos,
                            int DI, int R, int n_rot, double base, float eps)
{
    __shared__ float part[8];
    __shared__ float buf[512];
    const int pos = *ppos, i = threadIdx.x;
    if (i < DI) raw[(int64_t)pos * DI + i] = kraw[i];
    if ((pos + 1) % R) return;
    __syncthreads();
    const int b = pos / R;
    float v = 0.0f;
    if (i < DI) {
        for (int k = 0; k < R; k++) v += raw[(int64_t)(b * R + k) * DI + i];
        v /= R;
    }
    const float ss = block_sum(i < DI ? v * v : 0.0f, part);
    const float sc = 1.0f / sqrtf(ss / DI + eps);
    if (i < DI) buf[i] = v * sc * kn[i];
    __syncthreads();
    if (i < DI) {
        const int half = n_rot / 2;
        float o = buf[i];
        if (i < n_rot) {
            const int f = i < half ? i : i - half;
            const double ang = (double)(b * R) * pow(base, -2.0 * f / n_rot);
            const float cs = (float)cos(ang), sn = (float)sin(ang);
            o = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
        }
        pool[(int64_t)b * DI + i] = o;
    }
}

/* score of every complete block before the tail: sum over heads of relu(q_h . k_block) */
__global__ void k_idx_score(const float *qi, const float *pool, float *score, const int *ppos, int W, int R, int NH, int DI)
{
    const int pos = *ppos;
    if (pos + 1 <= W) return;
    const int nfull = (pos + 1) / R;
    const int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b >= nfull) return;
    const float *pk = pool + (int64_t)b * DI;
    float s = 0.0f;
    for (int h = 0; h < NH; h++) {
        float d = 0.0f;
        for (int i = lane; i < DI; i += 32) d += qi[h * DI + i] * pk[i];
        d = warp_sum(d);
        s += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) score[b] = s;
}

/* order the blocks (score high to low, equal scores: lower block first; bitonic sort in
 * shared memory, NS a power of two >= the block count) and write the attended cells:
 * the tail, then whole blocks, the last one cut to fit */
__global__ void k_idx_select(const float *score, int *sel, const int *ppos, int W, int R, int NS)
{
    extern __shared__ float sk[];          /* NS keys, then NS indices */
    int *si = (int *)(sk + NS);
    const int pos = *ppos;
    if (pos + 1 <= W) return;
    const int tail_start = (pos + 1) / R * R, nfull = tail_start / R;
    for (int i = threadIdx.x; i < NS; i += blockDim.x) {
        sk[i] = i < nfull ? score[i] : -1e30f;
        si[i] = i;
    }
    __syncthreads();
    for (int k = 2; k <= NS; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < NS; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    /* "a before b" = higher score, or equal score and lower index */
                    const float ka = sk[i], kb = sk[ixj];
                    const int ia = si[i], ib = si[ixj];
                    const bool a_first = ka > kb || (ka == kb && ia < ib);
                    const bool up = (i & k) == 0;
                    if (up ? !a_first : a_first) { sk[i] = kb; sk[ixj] = ka; si[i] = ib; si[ixj] = ia; }
                }
            }
            __syncthreads();
        }
    }
    const int nt = pos + 1 - tail_start;    /* tail cells, always attended */
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        int cell;
        if (c < nt) cell = tail_start + c;
        else { const int k = c - nt; cell = si[k / R] * R + k % R; }
        sel[c] = cell;
    }
}

/* keep-warm: a read of 32 MB, so the memory clock has a reason to stay up */
__global__ void k_warm(const float4 *buf, int n, float *out)
{
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float4 v = buf[i];
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if (a.x + a.y + a.z + a.w == 12345.678f) out[0] = a.x;   /* never true: keeps the reads */
}

/* f32 rows -> the repacked Q8 layout (int8 quants row by row, a float scale per 32): one
 * thread per block of 32 */
__global__ void k_f32_to_q8r(const float *w, int64_t nblk, int8_t *q, float *d)
{
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const float *x = w + b * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
    const float sc = amax / 127.0f, inv = sc > 0 ? 1.0f / sc : 0.0f;
    for (int i = 0; i < 32; i++) q[b * 32 + i] = (int8_t)__float2int_rn(x[i] * inv);
    d[b] = sc;
}

/* the first node of a pre-submitted layer graph: waits until the host has moved its flag */
__global__ void k_wait_flag(const volatile unsigned *flag, unsigned *seen)
{
    const unsigned s0 = *seen;
    unsigned f;
    while ((f = *flag) == s0) __nanosleep(200);
    *seen = f;
    __threadfence_system();
}

/* Q6_K rows as stored (UD-IQ4_XS's output head): a warp per row; in each half (128) of a
 * super-block lane l takes elements l, l+32, l+64, l+96, with q38_dequant_row's arithmetic
 * d * sc * (q - 32). Block: ql[128], qh[64], int8 scales[16], fp16 d at 208. */
__global__ void k_mv_q6_K(const uint8_t *w, int64_t row_bytes, int ne0, int rows, const float *x, float *y)
{
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (r >= rows) return;
    const int l = threadIdx.x, is = l / 16;
    const uint8_t *row = w + (int64_t)r * row_bytes;
    float acc = 0.0f;
    for (int b = 0; b < ne0 / 256; b++) {
        const uint8_t *blk = row + (int64_t)b * 210;
        const float d = __half2float(__ushort_as_half((unsigned short)(blk[208] | (blk[209] << 8))));
        const int8_t *sc = (const int8_t *)(blk + 192);
        const float *xb = x + b * 256;
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql = blk + 64 * n, *qh = blk + 128 + 32 * n;
            const int8_t *s = sc + 8 * n;
            const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            const float *xh = xb + 128 * n;
            acc += d * s[is + 0] * q1 * xh[l] + d * s[is + 2] * q2 * xh[l + 32] + d * s[is + 4] * q3 * xh[l + 64] +
                   d * s[is + 6] * q4 * xh[l + 96];
        }
    }
    acc = warp_sum(acc);
    if (l == 0) y[r] = acc;
}

/* --------------------------------------------------------------------- host -- */
static Gpu *G;
static Gpu *GD[2];                  /* dense models: the parts in layer order (q38_gpu_dense.cuh) */
static int  GD_n, GD_split;         /* parts; the first layer of GD[1] */

static int upload(DMat *d, const q38_mat *w, Gpu *g)
{
    memset(d, 0, sizeof *d);
    if (!w->data) return 0;
    d->type = w->type; d->ne0 = w->ne0; d->rows = w->rows; d->row_bytes = w->row_bytes;
    if (w->type == Q38_BF16) {                      /* exact widening: the F32 kernels serve it */
        const int64_t n = w->rows * w->ne0;
        float *f = (float *)malloc((size_t)n * sizeof(float));
        if (!f) return -1;
        for (int64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, w->data + 2 * i, 2); f[i] = q38_bf16_to_fp32(h); }
        CK(cudaMalloc(&d->d, (size_t)n * sizeof(float)));
        CK(cudaMemcpy(d->d, f, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
        free(f);
        d->type = Q38_F32;
        d->row_bytes = w->ne0 * 4;
        g->bytes += n * 4;
        return 0;
    }
    if (w->type == Q38_Q8_0 && !getenv("Q38_NO_REPACK")) {
        const int64_t nb = w->ne0 / 32, nq = w->rows * w->ne0;
        int8_t *q = (int8_t *)malloc((size_t)nq);
        float *sc = (float *)malloc((size_t)(w->rows * nb) * sizeof(float));
        if (!q || !sc) { free(q); free(sc); return -1; }
        for (int64_t r = 0; r < w->rows; r++) {
            const uint8_t *row = w->data + r * w->row_bytes;
            for (int64_t b = 0; b < nb; b++) {
                uint16_t h;
                memcpy(&h, row + b * 34, 2);
                sc[r * nb + b] = q38_fp16_to_fp32(h);
                memcpy(q + r * w->ne0 + b * 32, row + b * 34 + 2, 32);
            }
        }
        CK(cudaMalloc(&d->d, (size_t)nq));
        CK(cudaMalloc(&d->scale, (size_t)(w->rows * nb) * sizeof(float)));
        CK(cudaMemcpy(d->d, q, (size_t)nq, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d->scale, sc, (size_t)(w->rows * nb) * sizeof(float), cudaMemcpyHostToDevice));
        free(q);
        free(sc);
        d->type = Q38_Q8R;
        g->bytes += nq + w->rows * nb * 4;
        return 0;
    }
    const size_t n = (size_t)(w->row_bytes * w->rows);
    CK(cudaMalloc(&d->d, n));
    CK(cudaMemcpy(d->d, w->data, n, cudaMemcpyHostToDevice));
    g->bytes += (int64_t)n;
    return 0;
}

/* y = W x for a matrix kept in its GGUF type (Q4_K, Q5_K, IQ4_XS ...: the dense models' trunk);
 * a warp per row, lane_dot's native layouts, x read through the cache */
__global__ void k_mv_native(const uint8_t *__restrict__ W, int64_t rb, int ne0, int rows, int type,
                            const float *__restrict__ x, float *__restrict__ y)
{
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (r >= rows) return;
    const float s = warp_sum(lane_dot(type, W + (int64_t)r * rb, x, ne0, threadIdx.x));
    if (threadIdx.x == 0) y[r] = s;
}

#include "q38_gpu_mvq.cuh"

/* Chunk c of a native Q4_K/Q5_K row as 8 weights: super-block c / 32, its 64-element quarter j,
 * bytes p..p+3 of that quarter's 32: low nibbles are elements 64j+p.. (*o0), high ones 64j+32+p..
 * (*o1); lane_dot's arithmetic, one chunk at a time */
__device__ __forceinline__ void kq_chunk(int five, const uint8_t *w, int c, float *wt, int *o0, int *o1)
{
    const int sb = c >> 5, j = (c >> 3) & 3, p = (c & 7) * 4;
    const uint8_t *blk = w + (int64_t)sb * (five ? 176 : 144);
    const float2 dd = __half22float2(*(const __half2 *)(const void *)blk);
    float s1, m1, s2, m2;
    kscales(blk, j, &s1, &m1, &s2, &m2);
    const uint32_t q = *(const uint32_t *)(const void *)(blk + (five ? 48 : 16) + 32 * j + p);
    const uint32_t qh = five ? *(const uint32_t *)(const void *)(blk + 16 + p) : 0u;
    const float a1 = dd.x * s1, b1 = dd.y * m1, a2 = dd.x * s2, b2 = dd.y * m2;
#pragma unroll
    for (int u = 0; u < 4; u++) {
        const int qb = (q >> (8 * u)) & 0xFF, hb = (qh >> (8 * u)) & 0xFF;
        wt[u] = a1 * (float)((qb & 15) + (five ? (((hb >> (2 * j)) & 1) << 4) : 0)) - b1;
        wt[4 + u] = a2 * (float)((qb >> 4) + (five ? (((hb >> (2 * j + 1)) & 1) << 4) : 0)) - b2;
    }
    *o0 = 256 * sb + 64 * j + p;
    *o1 = *o0 + 32;
}

/* y = W x, a block of 128 threads per MVR rows: thread t takes the 8-weight chunks t, t+128, ...
 * of each row (coalesced 4-byte loads, many in flight), x read once for the rows. One warp per
 * row (k_mv_native) ran the 27B's trunk at ~140 GB/s. */
#define MVR 2
__global__ void __launch_bounds__(128) k_mv_rows(const uint8_t *__restrict__ W, int64_t rb, int ne0, int rows, int type,
                                                 const float *__restrict__ x, float *__restrict__ y)
{
    __shared__ float part[MVR][4];
    const int r0 = blockIdx.x * MVR, t = threadIdx.x;
    const int kq = type == Q38_Q4_K || type == Q38_Q5_K, five = type == Q38_Q5_K;
    const int nch = kq ? (ne0 / 256) * 32 : iq_chunks(type, ne0);
    float acc[MVR];
#pragma unroll
    for (int r = 0; r < MVR; r++) acc[r] = 0.0f;
    for (int c = t; c < nch; c += 128) {
        float wt[8];
        int o0, o1;
        float xv[8];
#pragma unroll
        for (int r = 0; r < MVR; r++) {
            if (r0 + r >= rows) break;
            const uint8_t *row = W + (int64_t)(r0 + r) * rb;
            if (kq) kq_chunk(five, row, c, wt, &o0, &o1);
            else iq_chunk(type, row, c, wt, &o0, &o1);
            if (r == 0) {
                const float4 a = __ldg((const float4 *)(x + o0)), b = __ldg((const float4 *)(x + o1));
                xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w; xv[4] = b.x; xv[5] = b.y; xv[6] = b.z; xv[7] = b.w;
            }
            float s2 = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; k++) s2 += wt[k] * xv[k];
            acc[r] += s2;
        }
    }
#pragma unroll
    for (int r = 0; r < MVR; r++) {
        const float v = warp_sum(acc[r]);
        if ((t & 31) == 0) part[r][t >> 5] = v;
    }
    __syncthreads();
    if (t < MVR && r0 + t < rows) y[r0 + t] = part[t][0] + part[t][1] + part[t][2] + part[t][3];
}

static int mv(const DMat *w, const float *x, float *y, cudaStream_t st)
{
    const dim3 blk(32, ROWS_PER_BLOCK);
    const int grid = (int)((w->rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    if (mvq_ok(w->type) && G && G->q8x && w->ne0 % 256 == 0 && w->ne0 <= 32768) {
        const int nb = (int)(w->ne0 / 32);
        k_q8x<<<(nb + 3) / 4, 128, 0, st>>>(x, (int)w->ne0, G->q8x, G->q8d, G->q8s);
        const dim3 b4(32, 4);
        const int g4 = (int)((w->rows + 3) / 4);
        const uint8_t *W = (const uint8_t *)w->d;
        switch (w->type) {
        case Q38_Q4_K:   k_mvq<Q38_Q4_K><<<g4, b4, 0, st>>>(W, w->row_bytes, (int)w->ne0, (int)w->rows, G->q8x, G->q8d, G->q8s, y); break;
        case Q38_Q5_K:   k_mvq<Q38_Q5_K><<<g4, b4, 0, st>>>(W, w->row_bytes, (int)w->ne0, (int)w->rows, G->q8x, G->q8d, G->q8s, y); break;
        case Q38_Q6_K:   k_mvq<Q38_Q6_K><<<g4, b4, 0, st>>>(W, w->row_bytes, (int)w->ne0, (int)w->rows, G->q8x, G->q8d, G->q8s, y); break;
        case Q38_IQ4_XS: k_mvq<Q38_IQ4_XS><<<g4, b4, 0, st>>>(W, w->row_bytes, (int)w->ne0, (int)w->rows, G->q8x, G->q8d, G->q8s, y); break;
        default:         k_mvq<Q38_IQ4_NL><<<g4, b4, 0, st>>>(W, w->row_bytes, (int)w->ne0, (int)w->rows, G->q8x, G->q8d, G->q8s, y); break;
        }
        return 0;
    }
    if (w->type == Q38_Q8R) {
        const int8_t *q = (const int8_t *)w->d;
        if (w->rows <= 512 && w->ne0 >= 2048) {
            k_mv_q8r_blk<<<(int)w->rows, 256, 0, st>>>(q, w->scale, (int)w->ne0, x, y);
        } else if (w->ne0 <= 512 && (w->ne0 % 64) == 0) {
            k_mv_q8r_short<<<(int)((w->rows + ROWS_PER_BLOCK * 8 - 1) / (ROWS_PER_BLOCK * 8)), blk, 0, st>>>(
                q, w->scale, (int)w->ne0, (int)w->rows, x, y);
        } else if (w->rows < 2048) {
            k_mv_q8r<1><<<(int)((w->rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), blk, 0, st>>>(
                q, w->scale, (int)w->ne0, (int)w->rows, x, y);
        } else {
            const int g8 = (int)((w->rows + ROWS_PER_BLOCK * RW - 1) / (ROWS_PER_BLOCK * RW));
            k_mv_q8r<RW><<<g8, blk, 0, st>>>(q, w->scale, (int)w->ne0, (int)w->rows, x, y);
        }
    } else if (w->type == Q38_Q8_0) {
        k_mv_q8_0<<<grid, blk, (size_t)w->ne0 * sizeof(float), st>>>(w->d, w->row_bytes, (int)w->ne0, (int)w->rows, x, y);
    } else if (w->type == Q38_F32 && w->rows <= 64) {
        k_mv_f32_blk<<<(int)w->rows, 256, 0, st>>>((const float *)w->d, (int)w->ne0, x, y);
    } else if (w->type == Q38_F32) {
        k_mv_f32<<<grid, blk, 0, st>>>((const float *)w->d, (int)w->ne0, (int)w->rows, x, y);
    } else if ((w->type == Q38_Q4_K || w->type == Q38_Q5_K || w->type == Q38_IQ4_XS || w->type == Q38_IQ4_NL ||
                w->type == Q38_IQ3_S || (w->type == Q38_Q6_K && getenv("Q38_MV_ROWS_Q6"))) && !getenv("Q38_MV_OLD")) {
        k_mv_rows<<<(int)((w->rows + MVR - 1) / MVR), 128, 0, st>>>((const uint8_t *)w->d, w->row_bytes, (int)w->ne0,
                                                                     (int)w->rows, w->type, x, y);
    } else if (w->type == Q38_Q6_K) {
        k_mv_q6_K<<<grid, blk, 0, st>>>((const uint8_t *)w->d, w->row_bytes, (int)w->ne0, (int)w->rows, x, y);
    } else if (w->type == Q38_Q4_K || w->type == Q38_Q5_K || w->type == Q38_Q5_1 || w->type == Q38_IQ4_XS ||
               w->type == Q38_IQ4_NL || w->type == Q38_IQ3_S) {
        k_mv_native<<<grid, blk, 0, st>>>((const uint8_t *)w->d, w->row_bytes, (int)w->ne0, (int)w->rows, w->type, x, y);
    } else {
        fprintf(stderr, "q38_gpu: no device kernel for type %s\n", q38_type_name(w->type));
        return -1;
    }
    return 0;
}

static float *dalloc(size_t n, Gpu *g);

/* store N tokens' K/V of layer il and run its attention (qf rows of H*2*D: q then gate) */
static void kv_store(Gpu *g, int il, const float *k, const float *v, int n, int N, cudaStream_t st)
{
    if (g->kv_q8) {
        const int D = n / g->kvh_n;
        k_kv_store_q8<<<N * g->kvh_n, D, 0, st>>>(k, v, (int8_t *)g->kc[il], (int8_t *)g->vc[il], g->ksc[il], g->vsc[il],
                                                  g->d_pos, g->kvh_n, D);
        return;
    }
    const int nb = (int)(((int64_t)n * N + 255) / 256);
    if (g->kv_half) k_kv_store<<<nb, 256, 0, st>>>(k, v, (__half *)g->kc[il], (__half *)g->vc[il], g->d_pos, n, N);
    else k_kv_store<<<nb, 256, 0, st>>>(k, v, (float *)g->kc[il], (float *)g->vc[il], g->d_pos, n, N);
}

#define FA_SEG 32                       /* decode: segments of the cells, merged after */

/* the tensor-core kernel's shared memory: O, S, fs/ls/ms in fp32; Q, K, V, P in fp16 */
static size_t fa_tc_smem(int D) { return (size_t)(32 * D + 2 * 32 * 32 + 96) * 4 + (size_t)(3 * 32 * D + 32 * 32) * 2; }

/* the mma kernel's: Q (128 rows), K (16 cells) padded, V transposed, fp16 */
static size_t fa_mma_smem(int D) { return (size_t)(144 * (D + FM_KP) + D * FM_VS) * 2; }

static size_t fa_smem(int rw, int D) { return (size_t)(8 * rw * D + 32 * (D + 1) + 32 * D) * sizeof(float); }

/* the flash kernel fits: a head group that divides the block's rows, D up to 256 */
static int fa_ok(int H, int Hkv, int D, const int *sel)
{
    const int G = H / Hkv;
    return !sel && Hkv > 0 && H % Hkv == 0 && 8 % G == 0 && D % 32 == 0 && D <= FA_DMAX && !getenv("Q38_NO_FA");
}

/* once per process and device: the flash kernels' shared memory is above the default 48 KB */
/* the shared-x verification products: up to 99 KB (ffn_down's 4 x 17408 bytes of x) */
static void mvq_setup(void)
{
#define MVQ_ATTR(T) do { cudaFuncSetAttribute(k_mvq_ncs<T, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024); \
                         cudaFuncSetAttribute(k_mvq_ncs<T, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024); \
                         cudaFuncSetAttribute(k_mvq_ncs<T, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024); } while (0)
    MVQ_ATTR(Q38_Q4_K); MVQ_ATTR(Q38_Q5_K); MVQ_ATTR(Q38_Q6_K); MVQ_ATTR(Q38_IQ4_XS); MVQ_ATTR(Q38_IQ4_NL);
#undef MVQ_ATTR
}

static void fa_setup(int D)
{
    mvq_setup();
    cudaFuncSetAttribute(k_fattn<float, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_smem(1, D));
    cudaFuncSetAttribute(k_fattn<float, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_smem(4, D));
    cudaFuncSetAttribute(k_fattn<__half, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_smem(1, D));
    cudaFuncSetAttribute(k_fattn<__half, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_smem(4, D));
    cudaFuncSetAttribute(k_fattn_tc<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_tc_smem(D));
    cudaFuncSetAttribute(k_fattn_tc<__half>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_tc_smem(D));
    cudaFuncSetAttribute(k_fattn_mma<float, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_mma_smem(256));
    cudaFuncSetAttribute(k_fattn_mma<__half, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_mma_smem(256));
    cudaFuncSetAttribute(k_fattn_mma<float, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_mma_smem(128));
    cudaFuncSetAttribute(k_fattn_mma<__half, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_mma_smem(128));
    cudaFuncSetAttribute(k_fattn_mma<int8_t, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_mma_smem(256));
    cudaFuncSetAttribute(k_fattn_mma<int8_t, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fa_mma_smem(128));
}

template <typename KT>
static void attn_t(Gpu *g, int il, const float *qf, float *out, int H, int Hkv, int D, const int *sel, int W, int N,
                   cudaStream_t st)
{
    const KT *kc = (const KT *)g->kc[il], *vc = (const KT *)g->vc[il];
    /* the fp16 tensor-core kernels go with the fp16 cache (long contexts: llama.cpp's precision);
     * a float cache keeps float attention, exact against the reference */
    const int tc = sizeof(KT) == 2 && !getenv("Q38_NO_FA_TC");
    /* the mma kernels take any head group up to 16 (the 27B's is 6); the CUDA-core flash ones
     * need it to divide 8, the WMMA one too */
    const int mma = (D == 256 || D == 128) && H % Hkv == 0 && H / Hkv <= 16 && tc && !sel;
    const int wmma_asked = getenv("Q38_FA_KERNEL") && !strcmp(getenv("Q38_FA_KERNEL"), "wmma");
    if (!mma && !fa_ok(H, Hkv, D, sel)) {
        k_attn_t<<<dim3(H, N), 256, 0, st>>>(qf, (int64_t)H * 2 * D, kc, vc, out, (int64_t)H * D, g->d_pos, H, Hkv, D, sel, W, W);
    } else if (N <= 4 && mma) {
        /* one token (or a few: a verification) on mma.sync: FA_SEG_MMA one-warp blocks per K/V head,
         * then the merge; a few tokens one launch each, at their positions */
        const size_t sm = (size_t)(32 * (D + FM_KP) + D * FM_VS) * 2;
        if (D == 256) k_fattn_dec_mma<KT, 256><<<dim3(Hkv, FA_SEG_MMA, N), 32, sm, st>>>(qf, kc, vc, g->fa_part, g->d_pos, H, Hkv);
        else k_fattn_dec_mma<KT, 128><<<dim3(Hkv, FA_SEG_MMA, N), 32, sm, st>>>(qf, kc, vc, g->fa_part, g->d_pos, H, Hkv);
        k_fattn_merge<<<dim3(H, N), 256, 0, st>>>(g->fa_part, qf, out, H, D, N, FA_SEG_MMA);
    } else if (N == 1) {
        /* one token: QT = 1, every segment its own block, then the merge */
        k_fattn<KT, 1><<<dim3(Hkv, 1, FA_SEG), 256, fa_smem(1, D), st>>>(qf, kc, vc, out, g->fa_part, g->d_pos, H, Hkv, D, 1);
        k_fattn_merge<<<dim3(H, 1), 256, 0, st>>>(g->fa_part, qf, out, H, D, 1, FA_SEG);
    } else if (mma && (!wmma_asked || !fa_ok(H, Hkv, D, sel))) {
        /* a prompt batch: mma.sync, 128 / G tokens a block */
        const dim3 grid(Hkv, (N + 128 / (H / Hkv) - 1) / (128 / (H / Hkv)));
        if (D == 256) k_fattn_mma<KT, 256><<<grid, 256, fa_mma_smem(256), st>>>(qf, kc, vc, out, g->d_pos, H, Hkv, N);
        else k_fattn_mma<KT, 128><<<grid, 256, fa_mma_smem(128), st>>>(qf, kc, vc, out, g->d_pos, H, Hkv, N);
    } else if (D % 16 == 0 && tc) {
        /* a prompt batch: the tensor cores */
        const int QT = 32 / (H / Hkv);
        k_fattn_tc<KT><<<dim3(Hkv, (N + QT - 1) / QT), 256, fa_tc_smem(D), st>>>(qf, kc, vc, out, g->d_pos, H, Hkv, D, N);
    } else {
        const int QT = 32 / (H / Hkv);
        k_fattn<KT, 4><<<dim3(Hkv, (N + QT - 1) / QT, 1), 256, fa_smem(4, D), st>>>(qf, kc, vc, out, NULL, g->d_pos, H, Hkv, D, N);
    }
}

/* int8 K/V: the mma kernels only (head dims 128/256, head groups up to 16: checked at open) */
static void attn_q8(Gpu *g, int il, const float *qf, float *out, int H, int Hkv, int D, int N, cudaStream_t st)
{
    const int8_t *kc = (const int8_t *)g->kc[il], *vc = (const int8_t *)g->vc[il];
    const float *ks = g->ksc[il], *vs = g->vsc[il];
    if (N <= 4) {                                /* one token, or a few one launch each */
        const size_t sm = (size_t)(32 * (D + FM_KP) + D * FM_VS) * 2;
        if (D == 256) k_fattn_dec_mma<int8_t, 256><<<dim3(Hkv, FA_SEG_MMA, N), 32, sm, st>>>(qf, kc, vc, g->fa_part, g->d_pos, H, Hkv, ks, vs);
        else k_fattn_dec_mma<int8_t, 128><<<dim3(Hkv, FA_SEG_MMA, N), 32, sm, st>>>(qf, kc, vc, g->fa_part, g->d_pos, H, Hkv, ks, vs);
        k_fattn_merge<<<dim3(H, N), 256, 0, st>>>(g->fa_part, qf, out, H, D, N, FA_SEG_MMA);
    } else {
        const dim3 grid(Hkv, (N + 128 / (H / Hkv) - 1) / (128 / (H / Hkv)));
        if (D == 256) k_fattn_mma<int8_t, 256><<<grid, 256, fa_mma_smem(256), st>>>(qf, kc, vc, out, g->d_pos, H, Hkv, N, ks, vs);
        else k_fattn_mma<int8_t, 128><<<grid, 256, fa_mma_smem(128), st>>>(qf, kc, vc, out, g->d_pos, H, Hkv, N, ks, vs);
    }
}

static void attn(Gpu *g, int il, const float *qf, float *out, int H, int Hkv, int D, const int *sel, int W, int N,
                 cudaStream_t st)
{
    if (g->kv_q8) { attn_q8(g, il, qf, out, H, Hkv, D, N, st); return; }
    if (g->kv_half) attn_t<__half>(g, il, qf, out, H, Hkv, D, sel, W, N, st);
    else attn_t<float>(g, il, qf, out, H, Hkv, D, sel, W, N, st);
}

static float *dalloc(size_t n, Gpu *g)
{
    float *p = NULL;
    if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) return NULL;
    cudaMemset(p, 0, n * sizeof(float));
    g->bytes += (int64_t)(n * sizeof(float));
    return p;
}

/* ------------------------------------------------- the VRAM expert pool -- */
/* Slots in size classes (experts of a few layers are larger: Q8_0 down, Q5_K gate/up).
 * Admission by use frequency: an expert the CPU computed is uploaded when a slot of its
 * class is free, or when it is used more often than the coldest resident (decayed counts),
 * at most `swaps` replacements per token so the PCIe link does not eat the gain. */
static int64_t ex_bytes_layer(const q38_model *m, int il)
{
    const q38_layer *l = &m->L[il];
    return (int64_t)m->c.ff_exp * (l->ex_gate.row_bytes + l->ex_up.row_bytes) + (int64_t)m->c.n_embd * l->ex_down.row_bytes;
}

static int pool_open_dev(q38_model *m, Gpu *g, int pd, int64_t budget)
{
    const q38_cfg *c = &m->c;
    int64_t weight = 0;
    for (int il = 0; il < c->n_layer; il++) weight += ex_bytes_layer(m, il);
    CK(cudaSetDevice(g->pdev[pd]));
    for (int k = 0; k < g->n_cls; k++) {
        const double share = (double)budget * (double)(g->cls_layers[k] * g->cls_bytes[k]) / (double)weight;
        int n = (int)(share / (double)g->cls_bytes[k]);
        if (n > g->cls_layers[k] * c->n_exp) n = g->cls_layers[k] * c->n_exp;
        g->cls_n[pd][k] = n;
        g->cls_first[pd][k] = g->nvs;
        g->nvs += n;
    }
    return 0;
}

static int pool_alloc_dev(Gpu *g, int pd)
{
    CK(cudaSetDevice(g->pdev[pd]));
    CK(cudaStreamCreateWithFlags(&g->st_up[pd], cudaStreamNonBlocking));
    int64_t maxb = 0;
    for (int k = 0; k < g->n_cls; k++) if (g->cls_bytes[k] > maxb) maxb = g->cls_bytes[k];
    for (int i = 0; i < g->n_up; i++) CK(cudaMalloc(&g->dstage[pd][i], (size_t)maxb));
    for (int k = 0; k < g->n_cls; k++) {
        const int n = g->cls_n[pd][k];
        if (!n) continue;
        CK(cudaMalloc(&g->cls_mem[pd][k], (size_t)n * (size_t)g->cls_bytes[k]));
        g->vram_bytes += (int64_t)n * g->cls_bytes[k];
        g->vram_bytes_pd[pd] += (int64_t)n * g->cls_bytes[k];
        for (int i = 0; i < n; i++) {
            const int s = g->cls_first[pd][k] + i;
            g->vs_key[s] = -1;
            g->vs_pd[s] = (unsigned char)pd;
            g->vs_ptr[s] = g->cls_mem[pd][k] + (size_t)i * (size_t)g->cls_bytes[k];
            g->free_list[g->cls_first[pd][k] + g->free_n[pd][k]++] = s;
        }
    }
    return 0;
}

/* The second device (the one without the trunk): its own copy of the expert kernels' inputs
 * and outputs, a stream, and one graph per layer. */
static int aux_open(q38_model *m, Gpu *g)
{
    const q38_cfg *c = &m->c;
    const int K = c->n_exp_used, E = c->n_embd, F = c->ff_exp;
    CK(cudaSetDevice(g->pdev[1]));
    CK(cudaStreamCreateWithFlags(&g->st2, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&g->ev2, cudaEventDisableTiming | cudaEventBlockingSync));
    CK(cudaMalloc(&g->d_ldesc2, (size_t)c->n_layer * sizeof(LayerDesc)));
    CK(cudaMemset(g->d_ldesc2, 0, (size_t)c->n_layer * sizeof(LayerDesc)));
    CK(cudaMalloc(&g->x2, (size_t)E * sizeof(float)));
    CK(cudaMalloc(&g->ex_gu2, (size_t)K * 2 * F * sizeof(float)));
    CK(cudaMalloc(&g->ex_h2, (size_t)K * F * sizeof(float)));
    CK(cudaMalloc(&g->ex_dn2, (size_t)K * E * sizeof(float)));
    CK(cudaMalloc(&g->routed2, (size_t)E * sizeof(float)));
    CK(cudaMallocHost(&g->h_ldesc2, (size_t)c->n_layer * sizeof(LayerDesc)));
    memset(g->h_ldesc2, 0, (size_t)c->n_layer * sizeof(LayerDesc));
    CK(cudaMallocHost(&g->h_y2, (size_t)E * sizeof(float)));
    return 0;
}

static int pool_open(q38_model *m, Gpu *g)
{
    const q38_cfg *c = &m->c;
    const int nkeys = c->n_layer * c->n_exp;
    g->vslot = (int *)malloc((size_t)nkeys * sizeof(int));
    g->freq = (float *)calloc((size_t)nkeys, sizeof(float));
    g->cand = (int *)malloc((size_t)c->n_layer * 64 * sizeof(int));
    g->pend = (char *)calloc((size_t)nkeys, 1);
    for (int i = 0; i < nkeys; i++) g->vslot[i] = -1;
    const char *sw = getenv("Q38_VRAM_SWAPS");
    g->swaps = sw ? atoi(sw) : 32;   /* 2: 17.1 tok/s, 8: 20.8, 32: 20.8 warm and a faster warm-up (2026-10-03) */
    g->n_up = 32;
    g->n_exp_ = c->n_exp; g->ff_exp_ = c->ff_exp; g->n_embd_ = c->n_embd;
    for (int il = 0; il < c->n_layer; il++) {
        const int64_t b = ex_bytes_layer(m, il);
        int k = 0;
        while (k < g->n_cls && g->cls_bytes[k] != b) k++;
        if (k == g->n_cls) { if (k == 8) return -1; g->cls_bytes[k] = b; g->n_cls++; }
        g->cls_of_layer[il] = k;
        g->cls_layers[k]++;
        g->rb[il][0] = m->L[il].ex_gate.row_bytes;
        g->rb[il][1] = m->L[il].ex_up.row_bytes;
        g->rb[il][2] = m->L[il].ex_down.row_bytes;
        g->ex_types[il][0] = m->L[il].ex_gate.type;
        g->ex_types[il][1] = m->L[il].ex_up.type;
        g->ex_types[il][2] = m->L[il].ex_down.type;
    }

    /* budgets: the trunk device keeps a margin for its graphs and buffers; the second device
     * (Q38_AUX_GPU, default: the other one of two; -1 = none) gives almost all of its memory */
    size_t fr = 0, tot = 0;
    cudaSetDevice(g->dev);
    cudaMemGetInfo(&fr, &tot);
    const char *ev = getenv("Q38_VRAM_GB");
    int64_t b0 = ev ? (int64_t)(atof(ev) * 1e9) : (int64_t)fr - (int64_t)900e6;
    if (b0 > (int64_t)fr - (int64_t)500e6) b0 = (int64_t)fr - (int64_t)500e6;
    g->npd = 1;
    g->pdev[0] = g->dev;
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    const char *ea = getenv("Q38_AUX_GPU");
    int aux = ea ? atoi(ea) : (n_dev == 2 ? 1 - g->dev : -1);
    int64_t b1 = 0;
    if (aux >= 0 && aux < n_dev && aux != g->dev) {
        cudaSetDevice(aux);
        size_t f1 = 0, t1 = 0;
        cudaMemGetInfo(&f1, &t1);
        const char *ev1 = getenv("Q38_VRAM2_GB");
        b1 = ev1 ? (int64_t)(atof(ev1) * 1e9) : (int64_t)f1 - (int64_t)700e6;
        if (b1 > (int64_t)f1 - (int64_t)400e6) b1 = (int64_t)f1 - (int64_t)400e6;
        if (b1 > 0) { g->npd = 2; g->pdev[1] = aux; }
    }
    g->nvs = 0;
    if (b0 > 0) pool_open_dev(m, g, 0, b0);
    if (g->npd == 2) pool_open_dev(m, g, 1, b1);
    if (!g->nvs) { cudaSetDevice(g->dev); return 0; }
    const int total = g->nvs;
    g->vs_key = (int *)malloc((size_t)(total + 1) * sizeof(int));
    g->vs_ptr = (uint8_t **)malloc((size_t)(total + 1) * sizeof(uint8_t *));
    g->vs_pd = (unsigned char *)calloc((size_t)(total + 1), 1);
    g->free_list = (int *)malloc((size_t)(total + 1) * sizeof(int));
    int64_t maxb = 0;
    for (int k = 0; k < g->n_cls; k++) if (g->cls_bytes[k] > maxb) maxb = g->cls_bytes[k];
    cudaSetDevice(g->dev);
    for (int i = 0; i < g->n_up; i++) {
        CK(cudaMallocHost(&g->up[i].stage, (size_t)maxb));
        CK(cudaEventCreateWithFlags(&g->up[i].ev, cudaEventDisableTiming));   /* re-created per device in up_main */
        g->up[i].state = 0;
    }
    for (int pd = 0; pd < g->npd; pd++) if (pool_alloc_dev(g, pd)) return -1;
    for (int i = 0; i < g->n_up; i++)
        for (int pd = 0; pd < g->npd; pd++) {
            CK(cudaSetDevice(g->pdev[pd]));
            CK(cudaEventCreateWithFlags(&g->up_ev[pd][i], cudaEventDisableTiming));
        }
    if (g->npd == 2 && aux_open(m, g)) return -1;
    cudaSetDevice(g->dev);
    return 0;
}

/* Upload the raw expert in stage (pinned) into slot s, rearranged; on the slot's device. */
static void upload_slot(Gpu *g, int ji, int s, int key, const uint8_t *stage, int64_t bytes)
{
    const int pd = g->vs_pd[s];
    cudaStream_t st = g->st_up[pd];
    uint8_t *ds = g->dstage[pd][ji];
    cudaMemcpyAsync(ds, stage, (size_t)bytes, cudaMemcpyHostToDevice, st);
    const int il = key / g->n_exp_, F = g->ff_exp_, E = g->n_embd_;
    const int *ty = g->ex_types[il];
    const int64_t bg = (int64_t)F * g->rb[il][0], bu = (int64_t)F * g->rb[il][1];
    const int nb_g = ty[0] == Q38_Q4_K || ty[0] == Q38_Q5_K ? E / 256 : E / 32;
    const int nb_d = ty[2] == Q38_Q4_K || ty[2] == Q38_Q5_K ? F / 256 : F / 32;
    repack_expert(ds, g->vs_ptr[s], ty, F, E, bg, bu, st);
}

/* The upload worker: copies a held expert from the RAM cache into a pinned staging buffer,
 * releases it, and queues the DMA and the rearrangement on the slot's device. pool_promote
 * commits the slot once the job's event has fired. */
#ifdef _WIN32
static DWORD WINAPI up_main(LPVOID arg)
#else
static void *up_main(void *arg)
#endif
{
    Gpu *g = (Gpu *)arg;
    while (!g->up_stop) {
        int did = 0;
        for (int i = 0; i < g->n_up; i++) {
            struct UpJob *j = &g->up[i];
            if (j->state != 1) continue;
            std::atomic_thread_fence(std::memory_order_acquire);
            memcpy(j->stage, j->data, (size_t)j->bytes);
            q38_model_expert_release(j->hold);
            const int pd = g->vs_pd[j->slot];
            cudaSetDevice(g->pdev[pd]);
            upload_slot(g, i, j->slot, j->key, j->stage, j->bytes);
            j->ev = g->up_ev[pd][i];
            cudaEventRecord(j->ev, g->st_up[pd]);
            std::atomic_thread_fence(std::memory_order_release);
            j->state = 2;
            did = 1;
        }
        if (!did) {
#ifdef _WIN32
            Sleep(1);
#else
            usleep(1000);
#endif
        }
    }
    return 0;
}

static void pool_commit(Gpu *g)
{
    for (int i = 0; i < g->n_up; i++) {
        struct UpJob *j = &g->up[i];
        if (j->state != 2) continue;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (cudaEventQuery(j->ev) != cudaSuccess) continue;
        g->vs_key[j->slot] = j->key;
        g->vslot[j->key] = j->slot;
        g->pend[j->key] = 0;
        g->uploads++;
        j->state = 0;
    }
}

/* a free slot of class k on any device (the trunk's first), or -1 */
static int take_free(Gpu *g, int k)
{
    for (int pd = 0; pd < g->npd; pd++)
        if (g->free_n[pd][k] > 0) return g->free_list[g->cls_first[pd][k] + --g->free_n[pd][k]];
    return -1;
}

static void give_back(Gpu *g, int s, int k)
{
    const int pd = g->vs_pd[s];
    g->vs_key[s] = -1;
    g->free_list[g->cls_first[pd][k] + g->free_n[pd][k]++] = s;
}

/* After a token: queue uploads for the CPU-computed experts that earned a slot. */
static void pool_promote(q38_model *m, Gpu *g)
{
    const q38_cfg *c = &m->c;
    pool_commit(g);
    int filled = 0, swapped = 0;
    for (int i = 0; i < g->n_cand; i++) {
        const int key = g->cand[i];
        if (g->vslot[key] >= 0 || g->pend[key]) continue;
        int jb = -1;
        for (int u = 0; u < g->n_up; u++) if (g->up[u].state == 0) { jb = u; break; }
        if (jb < 0) break;
        const int il = key / c->n_exp, e = key % c->n_exp, k = g->cls_of_layer[il];
        int s = -1;
        if (filled < 64 && (s = take_free(g, k)) >= 0) {
            filled++;
        } else {
            if (swapped >= g->swaps) continue;
            int victim = -1;
            float vf = 1e30f;
            for (int pd = 0; pd < g->npd; pd++)
                for (int j = g->cls_first[pd][k]; j < g->cls_first[pd][k] + g->cls_n[pd][k]; j++) {
                    if (g->vs_key[j] < 0) continue;              /* free or pending */
                    const float f = g->freq[g->vs_key[j]];
                    if (f < vf) { vf = f; victim = j; }
                }
            if (victim < 0 || !(g->freq[key] > vf + 1.0f)) continue;
            g->vslot[g->vs_key[victim]] = -1;
            s = victim;
            swapped++;
        }
        const uint8_t *data = NULL;
        int64_t bytes = 0;
        void *h = q38_model_expert_hold(m, il, e, &data, &bytes);
        if (!h || bytes != g->cls_bytes[k]) {
            q38_model_expert_release(h);
            give_back(g, s, k);
            continue;
        }
        g->vs_key[s] = -2;                                   /* pending: not a victim, not usable */
        g->pend[key] = 1;
        struct UpJob *j = &g->up[jb];
        j->slot = s; j->key = key; j->hold = h; j->data = data; j->bytes = bytes;
        std::atomic_thread_fence(std::memory_order_release);
        j->state = 1;
    }
    g->n_cand = 0;
    if (++g->tokens % 128 == 0)
        for (int i = 0; i < c->n_layer * c->n_exp; i++) g->freq[i] *= 0.5f;
}

static void tap(q38_model *m, const char *name, const float *d, int64_t n)
{
    if (!m->tap) return;
    cudaStreamSynchronize(G->st);
    float *h = (float *)malloc((size_t)n * sizeof(float));
    cudaMemcpy(h, d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    m->tap(m->tap_ctx, name, h, n);
    free(h);
}

/* the norms in front of the mixer, the MoE and the head: the hyper-connection mix (qwen4exp),
 * or the plain members' RMSNorm (attn_norm, post_attention_norm, output_norm) */
#define NORM_ATTN(il) (m->c.plain ? &LM(il, attn_norm) : &LM(il, hc_attn_norm))
#define NORM_FFN(il)  (m->c.plain ? &LM(il, post_norm) : &LM(il, hc_ffn_norm))
#define NORM_OUT      (m->c.plain ? &G->output_norm : &G->out_hc_norm)

static int hc_mix(q38_model *m, const DMat *norm, const DMat *down, const DMat *up, const DMat *inj, float *mixed,
                  float *inject)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, hc = c->hc;
    if (c->plain) {
        /* mixed = rmsnorm(res) * norm; inject 0 makes k_hc_combine (hc = 1) res += out */
        k_rmsnorm_rows<<<1, 256, 0, G->st>>>(G->res, (const float *)norm->d, mixed, E, c->eps);
        if (inject) CK(cudaMemsetAsync(inject, 0, sizeof(float), G->st));
        return 0;
    }
    k_rmsnorm_rows<<<hc, 256, 0, G->st>>>(G->res, (const float *)norm->d, G->xn, E, c->eps);
    if (mv(down, G->xn, G->lo, G->st)) return -1;
    k_silu_scale<<<(c->hc_rank + 255) / 256, 256, 0, G->st>>>(G->lo, c->hc_rank, 1.0f / hc);
    if (mv(up, G->lo, G->gate, G->st)) return -1;
    k_hc_mean<<<(E + 255) / 256, 256, 0, G->st>>>(G->xn, G->gate, mixed, E, hc);
    if (inj && mv(inj, G->xn, inject, G->st)) return -1;
    return 0;
}

/* A decode step alternates bursts of device work with host work (routing, routed experts on
 * the CPU, launches). Over that duty cycle the driver lowers the device's performance state
 * (measured: P3/P5, 300-800 MHz core and 5 GHz memory instead of 1.8 GHz / 7.3 GHz), and
 * every kernel slows down. While a token is being computed this thread keeps one 32 MB read
 * in flight on its own stream at a time (blocking event: no CPU spin). Q38_WARM=1 turns it
 * on. Measured 2026-10-03: the clocks rise, but the reads compete with the model's kernels
 * and the driver lock with its launches; 16.45 tok/s without it, 15.2-15.9 with it. So it
 * is off by default; the remedy that works is fewer gaps (CUDA graphs). */
#define WARM_FLOATS (8 << 20)
#ifdef _WIN32
static DWORD WINAPI warm_main(LPVOID arg)
#else
static void *warm_main(void *arg)
#endif
{
    Gpu *g = (Gpu *)arg;
    cudaSetDevice(g->warm_dev);
    while (!g->warm_stop) {
        if (!g->warm_want) {
#ifdef _WIN32
            Sleep(1);
#else
            usleep(1000);
#endif
            continue;
        }
        k_warm<<<g->warm_blocks, 256, 0, g->warm_st>>>((const float4 *)g->warm_buf, g->warm_n4, g->warm_out);
        cudaEventRecord(g->warm_ev, g->warm_st);
        cudaEventSynchronize(g->warm_ev);
    }
    return 0;
}

static void warm_start(Gpu *g)
{
    /* the trunk's device: off by default (measured slower, contention). The second expert
     * device has no trunk to compete with and idles between short bursts (P3/P5 measured):
     * kept warm by default (Q38_WARM_AUX=0 turns it off). */
    const char *e = getenv("Q38_WARM"), *ea = getenv("Q38_WARM_AUX");
    if (g->npd == 2 && !(ea && atoi(ea) == 0)) g->warm_dev = g->pdev[1];
    else if (e && atoi(e)) g->warm_dev = g->dev;
    else return;
    cudaSetDevice(g->warm_dev);
    if (cudaStreamCreateWithFlags(&g->warm_st, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&g->warm_ev, cudaEventDisableTiming | cudaEventBlockingSync) != cudaSuccess ||
        cudaMalloc(&g->warm_buf, (size_t)WARM_FLOATS * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g->warm_out, sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "q38_gpu: keep-warm unavailable\n");
        return;
    }
    cudaMemset(g->warm_buf, 0, (size_t)WARM_FLOATS * sizeof(float));
    const char *wb = getenv("Q38_WARM_BLOCKS"), *wm = getenv("Q38_WARM_MB");
    g->warm_blocks = wb ? atoi(wb) : 2;
    g->warm_n4 = (int)((wm ? atof(wm) : 4.0) * (1 << 20) / 16);
    if (g->warm_n4 > WARM_FLOATS / 4) g->warm_n4 = WARM_FLOATS / 4;
#ifdef _WIN32
    g->warm_thread = CreateThread(NULL, 0, warm_main, g, 0, NULL);
    g->warm_on = g->warm_thread != NULL;
#else
    g->warm_on = pthread_create(&g->warm_thread, NULL, warm_main, g) == 0;
#endif    cudaSetDevice(g->dev);
}

static void warm_end(Gpu *g)
{
    if (!g->warm_on) return;
    g->warm_stop = 1;
#ifdef _WIN32
    WaitForSingleObject(g->warm_thread, INFINITE);
    CloseHandle(g->warm_thread);
#else
    pthread_join(g->warm_thread, NULL);
#endif
    g->warm_on = 0;
}

/* Fill the VRAM pool with the most used experts that are in the RAM cache (after
 * q38_model_preload), synchronously; the pool's frequencies start from the profile. */
static const float *g_puse;
static int by_puse(const void *a, const void *b)
{
    const float x = g_puse[*(const int *)a], y = g_puse[*(const int *)b];
    return x > y ? -1 : x < y ? 1 : 0;
}

extern "C" int q38_gpu_prefill(void *model)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    if (!G || !G->nvs || !m->use || c->dense) return 0;
    const int n = c->n_layer * c->n_exp;
    int *ord = (int *)malloc((size_t)n * sizeof(int));
    if (!ord) return 0;
    for (int i = 0; i < n; i++) { ord[i] = i; G->freq[i] = m->use[i]; }
    g_puse = m->use;
    qsort(ord, (size_t)n, sizeof(int), by_puse);
    int done = 0;
    for (int i = 0; i < n; i++) {
        const int key = ord[i];
        if (m->use[key] <= 0) break;
        const int il = key / c->n_exp, e = key % c->n_exp, k = G->cls_of_layer[il];
        if (G->vslot[key] >= 0) continue;
        const uint8_t *data = NULL;
        int64_t bytes = 0;
        if (q38_model_expert_slot(m, il, e, &data, &bytes) || bytes != G->cls_bytes[k]) continue;
        const int sl = take_free(G, k);
        if (sl < 0) continue;
        memcpy(G->up[0].stage, data, (size_t)bytes);
        const int pd = G->vs_pd[sl];
        cudaSetDevice(G->pdev[pd]);
        upload_slot(G, 0, sl, key, G->up[0].stage, bytes);
        CK(cudaStreamSynchronize(G->st_up[pd]));
        G->vs_key[sl] = key;
        G->vslot[key] = sl;
        done++;
    }
    cudaSetDevice(G->dev);
    free(ord);
    return done;
}

extern "C" void q38_gpu_warm(void *model, int on)
{
    (void)model;
    if (G) G->warm_want = on;
}

static int bat_open(q38_model *m, int nb);    /* q38_gpu_batch.cuh */
static int dense_open(q38_model *m, int device);   /* q38_gpu_dense.cuh */
static void dense_reset(q38_model *m);
static void dense_close(q38_model *m);
static int launcher_start(q38_model *m);
static void launcher_stop(void);

static int bat_open_aux(q38_model *m);

extern "C" int q38_gpu_open(void *model, int device)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || device < 0 || device >= n_dev) {
        fprintf(stderr, "q38_gpu: no CUDA device %d (%d found)\n", device, n_dev);
        return -1;
    }
    if (c->dense) return dense_open(m, device);
    /* the expert kernels (lane_dot, lane_dot_rp, the tiles) know these types; another one
     * would add nothing there, silently: such a model runs on the CPU, which has them all */
    for (int il = 0; il < c->n_layer; il++) {
        const int ty[3] = { m->L[il].ex_gate.type, m->L[il].ex_up.type, m->L[il].ex_down.type };
        for (int k = 0; k < 3; k++) {
            const int t = ty[k];
            if (t != Q38_Q4_K && t != Q38_Q5_K && t != Q38_Q5_1 && t != Q38_Q8_0 && t != Q38_IQ4_NL && t != Q38_IQ4_XS && t != Q38_Q6_K &&
                t != Q38_IQ3_S) {
                fprintf(stderr, "q38_gpu: the experts of layer %d are %s, which the device kernels do not compute; "
                                "running on the CPU\n", il, q38_type_name(t));
                return -1;
            }
        }
    }
    CK(cudaSetDevice(device));
    Gpu *g = (Gpu *)calloc(1, sizeof(Gpu));
    G = g;
    g->dev = device;
    CK(cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&g->ev, cudaEventDisableTiming | cudaEventBlockingSync));
    if (NMAT > 64) { fprintf(stderr, "q38_gpu: layer has %d matrices\n", NMAT); return -1; }
    for (int il = 0; il < c->n_layer; il++) {
        const q38_mat *w = (const q38_mat *)&m->L[il];
        for (int i = 0; i < NMAT; i++)
            if (upload(&g->L[il][i], &w[i], g)) return -1;
    }
    if (upload(&g->output, &m->output, g) || upload(&g->output_norm, &m->output_norm, g) || upload(&g->out_hc_norm, &m->out_hc_norm, g) ||
        upload(&g->out_hc_down, &m->out_hc_down, g) || upload(&g->out_hc_up, &m->out_hc_up, g))
        return -1;

    const int E = c->n_embd, hcE = c->hc * E, S = c->d_state;
    const int ch = 2 * c->n_kh * S + c->n_vh * S;
    g->res = dalloc(hcE, g); g->xn = dalloc(hcE, g); g->mixed = dalloc(E, g); g->lo = dalloc(c->hc_rank > 0 ? c->hc_rank : 1, g);   /* 0 without hyper-connections: a NULL that reads as a failure */
    g->gate = dalloc(hcE, g); g->inject = dalloc(64, g); g->y = dalloc(E, g); g->routed = dalloc(E, g);
    g->qkv = dalloc(ch, g); g->z = dalloc((size_t)c->n_vh * S, g); g->o = dalloc((size_t)c->n_vh * S, g);
    g->beta = dalloc(256, g); g->g = dalloc(256, g);
    g->qf = dalloc((size_t)c->n_head * c->head_dim * 2, g); g->aout = dalloc((size_t)c->n_head * c->head_dim, g);
    g->fa_part = dalloc((size_t)4 * (FA_SEG_MMA > FA_SEG ? FA_SEG_MMA : FA_SEG) * c->n_head * (c->head_dim + 2), g);
    fa_setup(c->head_dim);
    g->router = dalloc(c->n_exp, g); g->sh_gu = dalloc(2 * c->ff_shexp, g); g->sh_h = dalloc(c->ff_shexp, g);
    g->sh_dn = dalloc(E, g); g->sg = dalloc(1, g); g->logits = dalloc(c->n_vocab, g);
    {
        /* Q38_KV=16|32: the cache's element; half by default above 32k positions (halves the VRAM) */
        const char *e = getenv("Q38_KV");
        g->kv_half = e ? atoi(e) == 16 : m->max_ctx > 32768;
    }
    for (int il = 0; il < c->n_layer; il++) {
        if (m->L[il].recurrent) {
            g->conv[il] = dalloc((size_t)(c->conv_k - 1) * ch, g);
            g->ssm[il] = dalloc((size_t)c->n_vh * S * S, g);
        } else {
            const size_t kv_f = ((size_t)m->max_ctx * c->n_head_kv * c->head_dim * (g->kv_half ? 2 : 4) + 3) / 4;
            g->kc[il] = dalloc(kv_f, g);
            g->vc[il] = dalloc(kv_f, g);
        }
    }
    if (!g->logits) { fprintf(stderr, "q38_gpu: out of device memory\n"); return -1; }
    CK(cudaMallocHost(&g->h_x, (size_t)E * sizeof(float)));
    CK(cudaMallocHost(&g->h_router, (size_t)c->n_exp * sizeof(float)));
    {
        const char *e = getenv("Q38_PREDICT");
        g->predict = e ? atoi(e) : 12;
        if (g->predict > 64) g->predict = 64;
        const char *a = getenv("Q38_PREDICT_AHEAD");
        g->pred_ahead = a ? atoi(a) : 2;
        if (g->pred_ahead < 1) g->pred_ahead = 1;
        if (g->pred_ahead > 2) g->pred_ahead = 2;
        if (g->predict > 0) {
            g->router_next = dalloc((size_t)2 * c->n_exp, g);
            CK(cudaMallocHost(&g->h_router_next, (size_t)2 * c->n_exp * sizeof(float)));
            for (int il = 0; il < c->n_layer; il++) {
                const DMat *r = &g->L[il][offsetof(q38_layer, router) / sizeof(q38_mat)];
                DMat *p = &g->pred_router[il];
                memset(p, 0, sizeof *p);
                if (r->type != Q38_F32 || !r->d || r->ne0 % 32) { g->predict = 0; break; }
                const int64_t n = r->rows * r->ne0;
                p->type = Q38_Q8R; p->ne0 = r->ne0; p->rows = r->rows; p->row_bytes = r->ne0;
                CK(cudaMalloc(&p->d, (size_t)n));
                CK(cudaMalloc(&p->scale, (size_t)(n / 32) * sizeof(float)));
                k_f32_to_q8r<<<(int)((n / 32 + 255) / 256), 256>>>((const float *)r->d, n / 32, (int8_t *)p->d, p->scale);
                g->bytes += n + n / 8;
            }
            CK(cudaDeviceSynchronize());
        }
        g->pred_layer[0] = g->pred_layer[1] = -1;
    }
    CK(cudaMallocHost(&g->h_y, (size_t)E * sizeof(float)));
    CK(cudaMallocHost(&g->h_res, (size_t)hcE * sizeof(float)));
    {
        const int K = c->n_exp_used;
        if (K > 16) { fprintf(stderr, "q38_gpu: %d experts per token (max 16)\n", K); return -1; }
        CK(cudaMallocHost(&g->h_ldesc, (size_t)c->n_layer * sizeof(LayerDesc)));
        memset(g->h_ldesc, 0, (size_t)c->n_layer * sizeof(LayerDesc));
        CK(cudaMalloc(&g->d_ldesc, (size_t)c->n_layer * sizeof(LayerDesc)));
        CK(cudaMemset(g->d_ldesc, 0, (size_t)c->n_layer * sizeof(LayerDesc)));
        CK(cudaMallocHost(&g->h_pos, sizeof(int)));
        CK(cudaMalloc(&g->d_pos, sizeof(int)));
        CK(cudaMallocHost(&g->h_emb, (size_t)E * sizeof(float)));
        CK(cudaMallocHost(&g->h_logits, (size_t)c->n_vocab * sizeof(float)));
        g->knew = dalloc((size_t)c->n_head_kv * c->head_dim, g);
        g->vnew = dalloc((size_t)c->n_head_kv * c->head_dim, g);
        g->routed_dev = dalloc(E, g);
        g->ex_gu = dalloc((size_t)K * 2 * c->ff_exp, g);
        g->ex_h = dalloc((size_t)K * c->ff_exp, g);
        g->ex_dn = dalloc((size_t)K * E, g);
        g->hcE = hcE;
        if (c->idx_top_k > 0) {
            const int nblk = m->max_ctx / c->idx_ratio + 1;
            /* the sort only ever holds complete blocks: at most max_ctx / ratio of them */
            int ns = 1;
            while (ns < m->max_ctx / c->idx_ratio) ns <<= 1;
            if (ns > 8192) { fprintf(stderr, "q38_gpu: context %d is above the device indexer's limit (32768)\n", m->max_ctx); return -1; }
            g->ins = ns;
            for (int il = 0; il < c->n_layer; il++) {
                if (m->L[il].recurrent) continue;
                g->idx_raw[il] = dalloc((size_t)m->max_ctx * c->idx_dim, g);
                g->idx_pool[il] = dalloc((size_t)nblk * c->idx_dim, g);
            }
            g->ikraw = dalloc(c->idx_dim, g);
            g->iq = dalloc((size_t)c->idx_heads * c->idx_dim, g);
            g->iscore = dalloc(nblk, g);
            CK(cudaMalloc(&g->isel, (size_t)c->max_dense_ctx * sizeof(int)));
            CK(cudaFuncSetAttribute(k_idx_select, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * ns * 4));
        }
        if (c->ple_layer >= 0) {
            const int pin = c->ple_heads * c->ple_dim, hist = (c->ple_kern - 1) * c->ple_ngram;
            CK(cudaMallocHost(&g->h_ple_emb, (size_t)pin * sizeof(float)));
            g->ple_emb = dalloc(pin, g);
            g->ple_key = dalloc(hcE, g);
            g->ple_val = dalloc(E, g);
            g->ple_gated = dalloc(hcE, g);
            g->ple_normed = dalloc(hcE, g);
            g->ple_gate = dalloc(64, g);
            g->ple_ring = dalloc((size_t)hist * hcE, g);
        }
        const char *eg = getenv("Q38_GRAPH");
        g->use_graphs = !eg || atoi(eg) != 0;
    }
    if (bat_open(m, 0)) return -1;
    if (pool_open(m, g)) return -1;
    if (bat_open_aux(m)) return -1;
    if (g->nvs) {
#ifdef _WIN32
        g->up_thread = CreateThread(NULL, 0, up_main, g, 0, NULL);
#else
        pthread_create(&g->up_thread, NULL, up_main, g);
#endif
    }
    warm_start(g);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, device);
    m->gpu = g;
    launcher_start(m);
    /* the host no longer needs what the device computes: give that RAM to the expert cache.
     * Kept: the token embedding (looked up on the host) and the PLE layer (runs on the host). */
    int64_t freed = 0;
    for (int il = 0; il < c->n_layer; il++) {
        q38_layer *L = &m->L[il];
        q38_mat *w = (q38_mat *)L;
        for (int i = 0; i < NMAT; i++) {
            if (!w[i].data) continue;
            if (il == c->ple_layer && (&w[i] == &L->ple_key || &w[i] == &L->ple_value || &w[i] == &L->ple_norm_key ||
                                       &w[i] == &L->ple_norm_query || &w[i] == &L->ple_norm_conv || &w[i] == &L->ple_conv))
                continue;
            freed += w[i].row_bytes * w[i].rows;
            free(w[i].data);
            w[i].data = NULL;
        }
        /* the host's K/V and indexer caches: the device keeps its own */
        free(m->kcache[il]); free(m->vcache[il]); free(m->idx_raw[il]); free(m->idx_pool[il]);
        m->kcache[il] = m->vcache[il] = m->idx_raw[il] = m->idx_pool[il] = NULL;
    }
    q38_mat *top[] = { &m->output, &m->out_hc_norm, &m->out_hc_down, &m->out_hc_up };
    for (int i = 0; i < 4; i++) { freed += top[i]->row_bytes * top[i]->rows; free(top[i]->data); top[i]->data = NULL; }
    q38_model_set_cache(m, -1);
    fprintf(stderr, "q38: trunk on GPU %d (%s): %.2f GB on the device; %.2f GB of RAM handed to the expert cache"
            " (now up to %.1f GB); %d expert slots in VRAM: %.2f GB on GPU %d%s\n", device, p.name, (g->bytes) / 1e9,
            freed / 1e9, m->cache_budget / 1e9, g->nvs, g->vram_bytes_pd[0] / 1e9, device, "");
    if (g->npd == 2)
        fprintf(stderr, "q38: second expert pool on GPU %d: %.2f GB\n", g->pdev[1], g->vram_bytes_pd[1] / 1e9);
    return 0;
}

extern "C" void q38_gpu_reset(void *model)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    if (!G) return;
    if (c->dense) { dense_reset(m); return; }
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    cudaSetDevice(G->dev);
    for (int il = 0; il < c->n_layer; il++) {
        if (G->conv[il]) cudaMemset(G->conv[il], 0, (size_t)(c->conv_k - 1) * ch * sizeof(float));
        if (G->ssm[il]) cudaMemset(G->ssm[il], 0, (size_t)c->n_vh * S * S * sizeof(float));
    }
    cudaDeviceSynchronize();
}

extern "C" int q38_gpu_state(void *model, int save)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    if (!G || c->dense) return -1;                    /* dense: not yet (the parts' layers) */
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    const size_t conv_b = (size_t)(c->conv_k - 1) * ch * sizeof(float), ssm_b = (size_t)c->n_vh * S * S * sizeof(float);
    const size_t ring_b = G->ple_ring ? (size_t)(c->ple_kern - 1) * c->ple_ngram * c->hc * c->n_embd * sizeof(float) : 0;
    cudaSetDevice(G->dev);
    CK(cudaDeviceSynchronize());
    if (!G->snap) {
        size_t tot = ring_b;
        for (int il = 0; il < c->n_layer; il++) tot += (G->conv[il] ? conv_b : 0) + (G->ssm[il] ? ssm_b : 0);
        CK(cudaMallocHost(&G->snap, tot));
        G->snap_bytes = tot;
        if (!save) return -1;                       /* nothing saved yet */
    }
    const cudaMemcpyKind k = save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    size_t off = 0;
    for (int il = 0; il < c->n_layer; il++) {
        if (G->conv[il]) {
            CK(save ? cudaMemcpyAsync(G->snap + off, G->conv[il], conv_b, k, G->st) : cudaMemcpyAsync(G->conv[il], G->snap + off, conv_b, k, G->st));
            off += conv_b;
        }
        if (G->ssm[il]) {
            CK(save ? cudaMemcpyAsync(G->snap + off, G->ssm[il], ssm_b, k, G->st) : cudaMemcpyAsync(G->ssm[il], G->snap + off, ssm_b, k, G->st));
            off += ssm_b;
        }
    }
    if (ring_b) CK(save ? cudaMemcpyAsync(G->snap + off, G->ple_ring, ring_b, k, G->st) : cudaMemcpyAsync(G->ple_ring, G->snap + off, ring_b, k, G->st));
    CK(cudaStreamSynchronize(G->st));
    return 0;
}

/* The whole sequence state at n_past to a file, or back from one: the token ids, the attention
 * and indexer caches up to n_past, the recurrent states and the PLE history. A prompt's fixed
 * beginning (a harness's system prompt and tools) is then read once, not once per
 * conversation. Returns the position (n_past), or -1 (the state is then undefined: reset). */
#define Q38_STATE_MAGIC 0x53383351          /* "Q38S" */
#define Q38_STATE_VERSION 1
extern "C" int q38_gpu_state_file(void *model, const char *path, int save)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    if (!G) return -1;
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    const size_t conv_b = (size_t)(c->conv_k - 1) * ch * sizeof(float), ssm_b = (size_t)c->n_vh * S * S * sizeof(float);
    const size_t ring_b = G->ple_ring ? (size_t)(c->ple_kern - 1) * c->ple_ngram * c->hc * c->n_embd * sizeof(float) : 0;
    const size_t kv_row = (size_t)c->n_head_kv * c->head_dim * (G->kv_q8 ? 1 : G->kv_half ? 2 : 4), ix_row = (size_t)c->idx_dim * sizeof(float);
    const size_t sc_row = G->kv_q8 ? (size_t)c->n_head_kv * (c->head_dim / 32) * sizeof(float) : 0;   /* int8 K/V: the scales */
    /* what the file's layout depends on: a state from another shape is refused */
    const int64_t shape[10] = { Q38_STATE_MAGIC, Q38_STATE_VERSION, c->n_layer, (int64_t)kv_row, (int64_t)ix_row,
                                c->idx_ratio, (int64_t)conv_b, (int64_t)ssm_b, (int64_t)ring_b, c->n_vocab };
    FILE *f = fopen(path, save ? "wb" : "rb");
    if (!f) return -1;
    int64_t n = save ? m->n_past : -1;
    int ok = 1;
    if (save) {
        ok = fwrite(shape, sizeof shape, 1, f) == 1 && fwrite(&n, sizeof n, 1, f) == 1;
    } else {
        int64_t got[10];
        ok = fread(got, sizeof got, 1, f) == 1 && memcmp(got, shape, sizeof got) == 0 && fread(&n, sizeof n, 1, f) == 1
             && n > 0 && n < m->max_ctx;
    }
    size_t big = ring_b > conv_b ? ring_b : conv_b;
    if (ssm_b > big) big = ssm_b;
    if (ok && (size_t)n * kv_row > big) big = (size_t)n * kv_row;
    if (ok && (size_t)n * sc_row > big) big = (size_t)n * sc_row;
    if (ok && (size_t)n * ix_row > big) big = (size_t)n * ix_row;
    uint8_t *buf = ok ? (uint8_t *)malloc(big) : NULL;
    if (!buf) ok = 0;
    /* the dense models' layers live on two devices (their parts): each layer's own */
    Gpu *const parts[2] = { G, c->dense && GD_n == 2 ? GD[1] : NULL };
    for (int p = 0; p < 2 && ok; p++)
        if (parts[p]) { cudaSetDevice(parts[p]->dev); if (cudaDeviceSynchronize() != cudaSuccess) ok = 0; }
    cudaSetDevice(G->dev);
    /* the file side in 8 MB pieces, each after the disk governor's quiet window */
    auto fio = [&](uint8_t *p, size_t bytes) {
        for (size_t done = 0; done < bytes;) {
            const size_t k = bytes - done > ((size_t)8 << 20) ? (size_t)8 << 20 : bytes - done;
            q38_gguf_disk_wait();
            if ((save ? fwrite(p + done, 1, k, f) : fread(p + done, 1, k, f)) != k) return false;
            done += k;
        }
        return true;
    };
    /* one device array: through buf to the file, or from the file to the device */
    auto io = [&](void *dev, size_t bytes) {
        if (!ok || !bytes) return;
        if (save) {
            ok = cudaMemcpy(buf, dev, bytes, cudaMemcpyDeviceToHost) == cudaSuccess && fio(buf, bytes);
        } else {
            ok = fio(buf, bytes) && cudaMemcpy(dev, buf, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        }
    };
    if (ok) ok = save ? fwrite(m->toks, sizeof(int), (size_t)n, f) == (size_t)n : fread(m->toks, sizeof(int), (size_t)n, f) == (size_t)n;
    for (int il = 0; il < c->n_layer && ok; il++) {
        Gpu *gl = parts[1] && il >= GD_split ? parts[1] : G;
        cudaSetDevice(gl->dev);
        if (gl->conv[il]) io(gl->conv[il], conv_b);
        if (gl->ssm[il]) io(gl->ssm[il], ssm_b);
        if (gl->kc[il]) { io(gl->kc[il], (size_t)n * kv_row); io(gl->vc[il], (size_t)n * kv_row); }
        if (gl->ksc[il]) { io(gl->ksc[il], (size_t)n * sc_row); io(gl->vsc[il], (size_t)n * sc_row); }
        if (gl->idx_raw[il]) {
            io(gl->idx_raw[il], (size_t)n * ix_row);
            io(gl->idx_pool[il], (size_t)(n / c->idx_ratio + 1) * ix_row);   /* with the partial block */
        }
    }
    cudaSetDevice(G->dev);
    if (ring_b) io(G->ple_ring, ring_b);
    free(buf);
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        if (save) remove(path);
        return -1;
    }
    if (!save) m->n_past = (int)n;
    return (int)n;
}

/* Times each trunk matrix-vector product of one recurrent and one attention layer on the
 * device (kept busy, so at full clocks) and the per-token total they add up to. */
extern "C" int q38_gpu_mvbench(void *model)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    static const char *nm[] = { "hc_attn_norm", "hc_attn_down", "hc_attn_up", "hc_attn_inject", "hc_ffn_norm",
                                "hc_ffn_down", "hc_ffn_up", "hc_ffn_inject", "qkv", "gate", "beta", "alpha", "dt_bias",
                                "a", "conv", "ssm_norm", "ssm_out", "wq", "wk", "wv", "wo", "q_norm", "k_norm", "idx_q",
                                "idx_k", "idx_qn", "idx_kn", "ple_key", "ple_value", "ple_norm_key", "ple_norm_query",
                                "ple_norm_conv", "ple_conv", "router", "shexp_gate_inp", "sh_gate", "sh_up", "sh_down" };
    cudaSetDevice(G->dev);
    float *x = NULL, *y = NULL;
    CK(cudaMalloc(&x, 65536 * sizeof(float)));
    CK(cudaMalloc(&y, (size_t)c->n_vocab * sizeof(float)));
    CK(cudaMemset(x, 0, 65536 * sizeof(float)));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    int la = -1;
    for (int il = 0; il < c->n_layer; il++) if (!m->L[il].recurrent) { la = il; break; }
    const int layers[2] = { 0, la };
    double per_token = 0;
    for (int li = 0; li < 2; li++) {
        const int il = layers[li];
        if (il < 0) continue;
        const int count = m->L[il].recurrent ? 36 : 12;      /* layers of this kind */
        for (int i = 0; i < NMAT && i < (int)(sizeof nm / sizeof nm[0]); i++) {
            const DMat *w = &G->L[il][i];
            if (!w->d || w->rows < 2 || w->ne0 < 2) continue;
            for (int r = 0; r < 20; r++) if (mv(w, x, y, G->st)) return -1;
            CK(cudaEventRecord(e0, G->st));
            const int reps = 300;
            for (int r = 0; r < reps; r++) if (mv(w, x, y, G->st)) return -1;
            CK(cudaEventRecord(e1, G->st));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = 1e3 * ms / reps;
            const double bytes = w->type == Q38_Q8R ? (double)w->rows * w->ne0 * (1.0 + 4.0 / 32) :
                                 w->type == Q38_F32 ? (double)w->rows * w->ne0 * 4 : (double)w->rows * w->row_bytes;
            const int shared = i < 8 || i >= 33;                 /* in every layer */
            per_token += us * (shared ? (li == 0 ? 48 : 0) : count);
            printf("mv layer %2d %-15s %6lld x %6lld %-5s %8.1f us %6.1f GB/s\n", il, nm[i], (long long)w->ne0,
                   (long long)w->rows, w->type == Q38_Q8R ? "q8r" : q38_type_name(w->type), us, bytes / us / 1e3);
        }
    }
    {
        for (int r = 0; r < 5; r++) if (mv(&G->output, x, y, G->st)) return -1;
        CK(cudaEventRecord(e0, G->st));
        for (int r = 0; r < 50; r++) if (mv(&G->output, x, y, G->st)) return -1;
        CK(cudaEventRecord(e1, G->st));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = 1e3 * ms / 50;
        per_token += us;
        printf("mv output head %lld x %lld: %.1f us %.1f GB/s\n", (long long)G->output.ne0, (long long)G->output.rows, us,
               (double)G->output.rows * G->output.ne0 * 1.125 / us / 1e3);
    }
    printf("matrix-vector products of one token: %.2f ms\n", per_token / 1e3);
    cudaFree(x);
    cudaFree(y);
    return 0;
}

extern "C" void q38_gpu_close(void *model)
{
    q38_model *m = (q38_model *)model;
    if (!G) return;
    if (m->c.dense) { dense_close(m); return; }
    launcher_stop();
    if (G->vram_hits + G->vram_misses)
        fprintf(stderr, "q38: routed experts on the GPUs %.1f%% (%lld of %lld; %lld on the second), %lld uploaded to VRAM\n",
                100.0 * G->vram_hits / (double)(G->vram_hits + G->vram_misses), (long long)G->vram_hits,
                (long long)(G->vram_hits + G->vram_misses), (long long)G->aux_hits, (long long)G->uploads);
    if (G->pred_total)
        fprintf(stderr, "q38: next-layer prediction (top %d, %d ahead): %.1f%% of the routed experts; %lld experts read ahead, "
                        "%lld of them used, %lld still being read when needed (%.2f s)\n",
                G->predict, G->pred_ahead, 100.0 * G->pred_hit / (double)G->pred_total, (long long)m->pf_issued, (long long)m->pf_used,
                (long long)m->pf_waited, m->t_pf_wait);
    if (G->t_stream > 0)
        fprintf(stderr, "q38: prompt passes spent %.1f s copying %.1f GB of experts to the device\n", G->t_stream,
                G->stream_bytes / 1e9);
    if (G->t_sp_copy > 0)
        fprintf(stderr, "q38: streamed experts: copy+repack %.1f s (%.1f GB/s), their pairs %.1f s\n", G->t_sp_copy,
                G->stream_bytes / 1e9 / G->t_sp_copy, G->t_sp_pairs);
    if (G->tokens_fwd)
        fprintf(stderr, "q38: host per token: launch %.4f s, route %.4f s, promote %.4f s (%s)\n",
                G->t_launch / G->tokens_fwd, G->t_route / G->tokens_fwd, G->t_promote / G->tokens_fwd,
                G->use_graphs ? "CUDA graphs" : "plain launches");
    if (G->tokens_fwd)
        fprintf(stderr, "q38: host per token: route %.4f, launch aux %.4f, launch experts %.4f, predict %.4f, "
                        "experts call overhead %.4f, waiting for GPU %d %.4f s\n",
                G->t_rt / G->tokens_fwd, G->t_auxl / G->tokens_fwd, G->t_gxl / G->tokens_fwd, G->t_pred / G->tokens_fwd,
                G->t_xcall / G->tokens_fwd, G->npd == 2 ? G->pdev[1] : -1, G->t_auxw / G->tokens_fwd);
    warm_end(G);
    if (G->nvs) {
        G->up_stop = 1;
#ifdef _WIN32
        WaitForSingleObject(G->up_thread, INFINITE);
        CloseHandle(G->up_thread);
#else
        pthread_join(G->up_thread, NULL);
#endif
    }
    cudaSetDevice(G->dev);
    cudaDeviceSynchronize();
    if (G->snap) cudaFreeHost(G->snap);
    cudaDeviceReset();
    /* the other device's expert pool too, or a model opened next (/model) finds its memory taken */
    for (int pd = 1; pd < G->npd; pd++) {
        if (G->pdev[pd] == G->dev) continue;
        cudaSetDevice(G->pdev[pd]);
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
    cudaSetDevice(G->dev);
    free(G);
    G = NULL;
    m->gpu = NULL;
}

/* ------------------------------------------------------------ the token -- */
/* A token is a fixed sequence of segments between the host's steps:
 *   start                              position and embedding in, residual tiled
 *   F(il) = post(il-1) + front(il)     previous MoE combined; mixer; router; x and router
 *                                      logits out; shared expert (runs while the host routes)
 *   X(il) = experts(il)                the routed experts resident in VRAM
 *   T     = post(47) + head            final mix, output head, logits out
 * plus, around the PLE layer, P = post(ple-1) + residual out and residual in. Every
 * per-token value (position, expert lists) is read from device memory, so each segment is
 * captured once as a CUDA graph and relaunched (Q38_GRAPH=0: plain launches; taps always
 * use plain launches). */
typedef struct {
    int capturing;
} Seg;

static int seg_post(q38_model *m, int il, int capturing)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, hc = c->hc, hcE = hc * E;
    cudaStream_t st = G->st;
    char nm[64];
    CK(cudaMemcpyAsync(G->routed, G->h_y, (size_t)E * sizeof(float), cudaMemcpyHostToDevice, st));
    k_moe_final2<<<(E + 255) / 256, 256, 0, st>>>(G->y, G->routed, G->routed_dev, G->sh_dn, G->sg, E);
    if (m->tap && !capturing) { snprintf(nm, sizeof nm, "ffn_out-%d", il); tap(m, nm, G->y, E); }
    k_hc_combine<<<(hcE + 255) / 256, 256, 0, st>>>(G->res, G->y, G->inject, E, hc);
    if (m->tap && !capturing) { snprintf(nm, sizeof nm, "l_last-%d", il); tap(m, nm, G->res, hcE); }
    return 0;
}

static int seg_front(q38_model *m, int il, int capturing)
{
    const q38_cfg *c = &m->c;
    const q38_layer *l = &m->L[il];
    const int E = c->n_embd, hc = c->hc, hcE = hc * E, S = c->d_state, Hk = c->n_kh, Hv = c->n_vh;
    const int ch = 2 * Hk * S + Hv * S;
    cudaStream_t st = G->st;
    char nm[64];
    if (hc_mix(m, NORM_ATTN(il), &LM(il, hc_attn_down), &LM(il, hc_attn_up), &LM(il, hc_attn_inject),
               G->mixed, G->inject)) return -1;
    if (m->tap && !capturing) { snprintf(nm, sizeof nm, "hc_mixed_attn-%d", il); tap(m, nm, G->mixed, E); }
    if (l->recurrent) {
        if (mv(&LM(il, qkv), G->mixed, G->qkv, st) || mv(&LM(il, gate), G->mixed, G->z, st) ||
            mv(&LM(il, beta), G->mixed, G->beta, st) || mv(&LM(il, alpha), G->mixed, G->g, st)) return -1;
        k_gdn_gates<<<1, 256, 0, st>>>(G->beta, G->g, (const float *)LM(il, dt_bias).d, (const float *)LM(il, a).d, Hv);
        k_conv_silu<<<(ch + 255) / 256, 256, 0, st>>>(G->qkv, G->conv[il], (const float *)LM(il, conv).d, ch, c->conv_k);
        k_l2norm_rows<<<2 * Hk, 32, 0, st>>>(G->qkv, S, c->eps);
        k_gdn_step<<<Hv, dim3(S, 4), 6 * S * sizeof(float), st>>>(G->ssm[il], G->qkv, G->qkv + Hk * S, G->qkv + 2 * Hk * S,
                                                                 G->g, G->beta, G->o, S, Hk);
        if (m->tap && !capturing) { snprintf(nm, sizeof nm, "attn_output-%d", il); tap(m, nm, G->o, (int64_t)Hv * S); }
        k_norm_gate<<<Hv, 128, 0, st>>>(G->o, G->z, (const float *)LM(il, ssm_norm).d, S, c->eps, c->gdn_silu);
        if (mv(&LM(il, ssm_out), G->o, G->y, st)) return -1;
    } else {
        const int H = c->n_head, Hkv = c->n_head_kv, D = c->head_dim;
        if (mv(&LM(il, wq), G->mixed, G->qf, st) || mv(&LM(il, wk), G->mixed, G->knew, st) ||
            mv(&LM(il, wv), G->mixed, G->vnew, st)) return -1;
        k_norm_rope<<<H, 256, 0, st>>>(G->qf, 2 * D, (const float *)LM(il, q_norm).d, D, c->n_rot, G->d_pos, c->rope_base, c->eps);
        k_norm_rope<<<Hkv, 256, 0, st>>>(G->knew, D, (const float *)LM(il, k_norm).d, D, c->n_rot, G->d_pos, c->rope_base, c->eps);
        kv_store(G, il, G->knew, G->vnew, Hkv * D, 1, st);
        const int W = c->max_dense_ctx;
        if (G->idx_raw[il]) {
            const int DI = c->idx_dim, R = c->idx_ratio, NH = c->idx_heads, nblk = m->max_ctx / R + 1;
            if (mv(&LM(il, idx_k), G->mixed, G->ikraw, st) || mv(&LM(il, idx_q), G->mixed, G->iq, st)) return -1;
            k_idx_store<<<1, DI, 0, st>>>(G->ikraw, G->idx_raw[il], G->idx_pool[il], (const float *)LM(il, idx_kn).d,
                                          G->d_pos, DI, R, c->n_rot, c->rope_base, c->eps);
            k_norm_rope<<<NH, 256, 0, st>>>(G->iq, DI, (const float *)LM(il, idx_qn).d, DI, c->n_rot, G->d_pos, c->rope_base, c->eps);
            k_idx_score<<<(nblk + 7) / 8, 256, 0, st>>>(G->iq, G->idx_pool[il], G->iscore, G->d_pos, W, R, NH, DI);
            k_idx_select<<<1, 1024, (size_t)2 * G->ins * 4, st>>>(G->iscore, G->isel, G->d_pos, W, R, G->ins);
            if (m->tap && !capturing && m->n_past + 1 > W) {
                int *hs = (int *)malloc((size_t)W * sizeof(int));
                float *fs = (float *)malloc((size_t)W * sizeof(float));
                CK(cudaStreamSynchronize(st));
                CK(cudaMemcpy(hs, G->isel, (size_t)W * sizeof(int), cudaMemcpyDeviceToHost));
                for (int i = 0; i < W; i++) fs[i] = (float)hs[i];
                snprintf(nm, sizeof nm, "indexer_top_k-%d", il);
                m->tap(m->tap_ctx, nm, fs, W);
                free(hs);
                free(fs);
            }
        }
        attn(G, il, G->qf, G->aout, H, Hkv, D, G->idx_raw[il] ? G->isel : NULL, W, 1, st);
        if (mv(&LM(il, wo), G->aout, G->y, st)) return -1;
        if (m->tap && !capturing) { snprintf(nm, sizeof nm, "attn_output-%d", il); tap(m, nm, G->y, E); }
    }
    k_hc_combine<<<(hcE + 255) / 256, 256, 0, st>>>(G->res, G->y, G->inject, E, hc);
    if (hc_mix(m, NORM_FFN(il), &LM(il, hc_ffn_down), &LM(il, hc_ffn_up), &LM(il, hc_ffn_inject),
               G->mixed, G->inject)) return -1;
    if (m->tap && !capturing) { snprintf(nm, sizeof nm, "hc_mixed_ffn-%d", il); tap(m, nm, G->mixed, E); }
    if (c->dense) {                                 /* the FFN on the device: res += down(silu(gate) * up) */
        const int F = c->ff_shexp;
        if (mv(&LM(il, sh_gate), G->mixed, G->sh_gu, st) || mv(&LM(il, sh_up), G->mixed, G->sh_gu + F, st)) return -1;
        k_swiglu<<<(F + 255) / 256, 256, 0, st>>>(G->sh_gu, G->sh_h, F);
        if (mv(&LM(il, sh_down), G->sh_h, G->y, st)) return -1;
        if (m->tap && !capturing) { snprintf(nm, sizeof nm, "ffn_out-%d", il); tap(m, nm, G->y, E); }
        k_hc_combine<<<(hcE + 255) / 256, 256, 0, st>>>(G->res, G->y, G->inject, E, hc);
        if (m->tap && !capturing) { snprintf(nm, sizeof nm, "l_out-%d", il); tap(m, nm, G->res, E); }
        return 0;
    }
    /* router on the device; x and the logits to the host; the shared expert meanwhile */
    if (mv(&LM(il, router), G->mixed, G->router, st)) return -1;
    /* the next layers' routers (int8 copies: guesses only) on this MoE input, for reading ahead */
    if (G->predict > 0 && il + 1 < c->n_layer) {
        for (int a = 1; a <= G->pred_ahead && il + a < c->n_layer; a++)
            if (mv(&G->pred_router[il + a], G->mixed, G->router_next + (size_t)(a - 1) * c->n_exp, st)) return -1;
        CK(cudaMemcpyAsync(G->h_router_next, G->router_next, (size_t)G->pred_ahead * c->n_exp * sizeof(float),
                           cudaMemcpyDeviceToHost, st));
    }
    CK(cudaMemcpyAsync(G->h_x, G->mixed, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(G->h_router, G->router, (size_t)c->n_exp * sizeof(float), cudaMemcpyDeviceToHost, st));
    CK(cudaEventRecordWithFlags(G->ev, st, capturing ? cudaEventRecordExternal : cudaEventRecordDefault));
    if (mv(&LM(il, sh_gate), G->mixed, G->sh_gu, st) || mv(&LM(il, sh_up), G->mixed, G->sh_gu + c->ff_shexp, st))
        return -1;
    k_swiglu<<<(c->ff_shexp + 255) / 256, 256, 0, st>>>(G->sh_gu, G->sh_h, c->ff_shexp);
    if (mv(&LM(il, sh_down), G->sh_h, G->sh_dn, st) || mv(&LM(il, shexp_gate_inp), G->mixed, G->sg, st)) return -1;
    return 0;
}

static int seg_experts(q38_model *m, int il)
{
    const q38_cfg *c = &m->c;
    const q38_layer *L = &m->L[il];
    const int E = c->n_embd, F = c->ff_exp, K = c->n_exp_used;
    cudaStream_t st = G->st;
    LayerDesc *dd = G->d_ldesc + il;
    CK(cudaMemcpyAsync(dd, G->h_ldesc + il, sizeof(LayerDesc), cudaMemcpyHostToDevice, st));
    const dim3 blk(32, ROWS_PER_BLOCK);
    k_ex_gu_rp<<<dim3((2 * F + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, K), blk, 0, st>>>(
        dd, L->ex_gate.type, L->ex_up.type, E, F, G->mixed, G->ex_gu);
    k_ex_swiglu<<<(K * F + 255) / 256, 256, 0, st>>>(dd, G->ex_gu, G->ex_h, F);
    k_ex_down_rp<<<dim3((E + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, K), blk, 0, st>>>(
        dd, L->ex_down.type, F, E, G->ex_h, G->ex_dn);
    k_ex_sum<<<(E + 255) / 256, 256, 0, st>>>(dd, G->ex_dn, E, G->routed_dev);
    return 0;
}

/* the second device's experts of layer il: x in, the weighted sum out to h_y2 */
static int seg_aux(q38_model *m, int il, int capturing)
{
    const q38_cfg *c = &m->c;
    const q38_layer *L = &m->L[il];
    const int E = c->n_embd, F = c->ff_exp, K = c->n_exp_used;
    cudaStream_t st = G->st2;
    LayerDesc *dd = G->d_ldesc2 + il;
    CK(cudaMemcpyAsync(G->x2, G->h_x, (size_t)E * sizeof(float), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(dd, G->h_ldesc2 + il, sizeof(LayerDesc), cudaMemcpyHostToDevice, st));
    const dim3 blk(32, ROWS_PER_BLOCK);
    k_ex_gu_rp<<<dim3((2 * F + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, K), blk, 0, st>>>(
        dd, L->ex_gate.type, L->ex_up.type, E, F, G->x2, G->ex_gu2);
    k_ex_swiglu<<<(K * F + 255) / 256, 256, 0, st>>>(dd, G->ex_gu2, G->ex_h2, F);
    k_ex_down_rp<<<dim3((E + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, K), blk, 0, st>>>(
        dd, L->ex_down.type, F, E, G->ex_h2, G->ex_dn2);
    k_ex_sum<<<(E + 255) / 256, 256, 0, st>>>(dd, G->ex_dn2, E, G->routed2);
    CK(cudaMemcpyAsync(G->h_y2, G->routed2, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost, st));
    CK(cudaEventRecordWithFlags(G->ev2, st, capturing ? cudaEventRecordExternal : cudaEventRecordDefault));
    return 0;
}

static int run_aux(q38_model *m, int il)
{
    CK(cudaSetDevice(G->pdev[1]));
    int rc = 0;
    if (!G->use_graphs || m->tap) {
        rc = seg_aux(m, il, 0);
    } else {
        if (!G->gexec2[il]) {
            cudaGraph_t gr;
            CK(cudaStreamBeginCapture(G->st2, cudaStreamCaptureModeThreadLocal));
            rc = seg_aux(m, il, 1);
            CK(cudaStreamEndCapture(G->st2, &gr));
            if (!rc) CK(cudaGraphInstantiate(&G->gexec2[il], gr, 0));
            cudaGraphDestroy(gr);
        }
        if (!rc) CK(cudaGraphLaunch(G->gexec2[il], G->st2));
    }
    cudaStreamQuery(G->st2);                    /* hand it to the device now (WDDM batches) */
    CK(cudaSetDevice(G->dev));
    return rc;
}

static int seg_head(q38_model *m, int capturing)
{
    const q38_cfg *c = &m->c;
    if (hc_mix(m, NORM_OUT, &G->out_hc_down, &G->out_hc_up, NULL, G->mixed, NULL)) return -1;
    if (m->tap && !capturing) tap(m, "result_norm", G->mixed, c->n_embd);
    if (mv(&G->output, G->mixed, G->logits, G->st)) return -1;
    CK(cudaMemcpyAsync(G->h_logits, G->logits, (size_t)c->n_vocab * sizeof(float), cudaMemcpyDeviceToHost, G->st));
    return 0;
}

static int seg_ple(q38_model *m, int il)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, hc = c->hc, hcE = hc * E, hist = (c->ple_kern - 1) * c->ple_ngram;
    cudaStream_t st = G->st;
    if (mv(&LM(il, ple_key), G->ple_emb, G->ple_key, st) || mv(&LM(il, ple_value), G->ple_emb, G->ple_val, st)) return -1;
    k_ple_gate<<<hc, 256, 0, st>>>(G->ple_key, G->res, (const float *)LM(il, ple_norm_key).d,
                                   (const float *)LM(il, ple_norm_query).d, G->ple_gate, E, c->eps);
    k_ple_gated<<<hc, 256, 0, st>>>(G->ple_val, G->ple_gate, (const float *)LM(il, ple_norm_conv).d, G->ple_gated,
                                    G->ple_normed, E, c->eps);
    k_ple_conv<<<(hcE + 255) / 256, 256, 0, st>>>(G->res, G->ple_gated, G->ple_normed, G->ple_ring,
                                                  (const float *)LM(il, ple_conv).d, G->d_pos, hcE, c->ple_kern,
                                                  c->ple_ngram, hist);
    return 0;
}

static int seg_start(q38_model *m)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, hcE = c->hc * E;
    if (c->ple_layer >= 0)
        CK(cudaMemcpyAsync(G->ple_emb, G->h_ple_emb, (size_t)c->ple_heads * c->ple_dim * sizeof(float),
                           cudaMemcpyHostToDevice, G->st));
    CK(cudaMemcpyAsync(G->d_pos, G->h_pos, sizeof(int), cudaMemcpyHostToDevice, G->st));
    CK(cudaMemcpyAsync(G->mixed, G->h_emb, (size_t)E * sizeof(float), cudaMemcpyHostToDevice, G->st));
    k_tile<<<(hcE + 255) / 256, 256, 0, G->st>>>(G->res, G->mixed, E, c->hc);
    return 0;
}

static int seg_res_out(void)
{
    const int hcE = G->hcE;
    CK(cudaMemcpyAsync(G->h_res, G->res, (size_t)hcE * sizeof(float), cudaMemcpyDeviceToHost, G->st));
    return 0;
}

static int seg_res_in(void)
{
    CK(cudaMemcpyAsync(G->res, G->h_res, (size_t)G->hcE * sizeof(float), cudaMemcpyHostToDevice, G->st));
    return 0;
}

/* graph ids: 0 start, 1..L F(il), L+1..2L X(il), 2L+1 T, 2L+2 P */
#define GF(il) (1 + (il))
#define GX(il) (1 + c->n_layer + (il))
#define GT (1 + 2 * c->n_layer)
#define GP (2 + 2 * c->n_layer)

static int build_segment(q38_model *m, int id, int capturing)
{
    const q38_cfg *c = &m->c;
    if (id == 0) return seg_start(m);
    if (id == GT) return (!c->dense && seg_post(m, c->n_layer - 1, capturing)) || seg_head(m, capturing) ? -1 : 0;
    if (id == GP) return (c->ple_layer > 0 && seg_post(m, c->ple_layer - 1, capturing)) || seg_res_out() ? -1 : 0;
    if (id >= GX(0)) return seg_experts(m, id - GX(0));
    const int il = id - GF(0);
    if (il > 0 && G->launcher) k_wait_flag<<<1, 1, 0, G->st>>>(G->d_flag + (il - 1), G->d_seen + (il - 1));
    if (il > 0 && !c->dense && seg_post(m, il - 1, capturing)) return -1;
    if (il == c->ple_layer && seg_ple(m, il)) return -1;
    return seg_front(m, il, capturing);
}

static int run_segment(q38_model *m, int id)
{
    if (!G->use_graphs || m->tap) return build_segment(m, id, 0);
    if (!G->gexec[id]) {
        cudaGraph_t gr;
        CK(cudaStreamBeginCapture(G->st, cudaStreamCaptureModeThreadLocal));
        const int rc = build_segment(m, id, 1);
        CK(cudaStreamEndCapture(G->st, &gr));
        if (rc) return -1;
        CK(cudaGraphInstantiate(&G->gexec[id], gr, 0));
        cudaGraphDestroy(gr);
    }
    CK(cudaGraphLaunch(G->gexec[id], G->st));
    return 0;
}

/* Host cost of launching the instantiated per-layer graphs, back to back and alternating with
 * waits, and of switching devices (needs one decoded token first, for the graphs). */
extern "C" int q38_gpu_launchbench(void *model)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    cudaSetDevice(G->dev);
    const int il = 3, idf = GF(il), idx = GX(il);
    if (!G->gexec[idf] || !G->gexec[idx]) { printf("launch bench: graphs not built\n"); return 0; }
    CK(cudaStreamSynchronize(G->st));
    double t0 = now_s();
    for (int r = 0; r < 50; r++) CK(cudaGraphLaunch(G->gexec[idf], G->st));
    const double a = (now_s() - t0) / 50;
    CK(cudaStreamSynchronize(G->st));
    double tl = 0, tw = 0;
    for (int r = 0; r < 50; r++) {
        double t1 = now_s();
        CK(cudaGraphLaunch(G->gexec[idf], G->st));
        double t2 = now_s();
        CK(ev_wait(G->ev));
        tl += t2 - t1;
        tw += now_s() - t2;
    }
    CK(cudaStreamSynchronize(G->st));
    t0 = now_s();
    for (int r = 0; r < 50; r++) CK(cudaGraphLaunch(G->gexec[idx], G->st));
    const double b = (now_s() - t0) / 50;
    CK(cudaStreamSynchronize(G->st));
    t0 = now_s();
    for (int r = 0; r < 200; r++) { cudaSetDevice(G->npd == 2 ? G->pdev[1] : G->dev); cudaSetDevice(G->dev); }
    const double sd = (now_s() - t0) / 200;
    printf("launch bench: front graph %.1f us back to back, %.1f us after a wait (then %.1f us to its event); "
           "experts graph %.1f us; cudaSetDevice there and back %.1f us\n",
           a * 1e6, tl / 50 * 1e6, tw / 50 * 1e6, b * 1e6, sd * 1e6);
    return 0;
}

#ifdef _WIN32
static DWORD WINAPI launch_main(LPVOID arg)
{
    q38_model *m = (q38_model *)arg;
    const q38_cfg *c = &m->c;
    cudaSetDevice(G->dev);
    long last = G->job_seq;
    int idle = 0;
    while (!G->launch_stop) {
        const long js = G->job_seq;
        if (js == last) {
            if (++idle < 20000) _mm_pause();
            else { SwitchToThread(); idle = 20000; }
            continue;
        }
        idle = 0;
        last = js;
        MemoryBarrier();
        const int il = G->job_il;
        int rc = 0;
        if (G->job_aux) rc = run_aux(m, il);
        G->done_aux_seq = js;
        if (!rc) rc = run_segment(m, GX(il));
        if (!rc && !G->job_last) rc = run_segment(m, GF(il + 1));
        cudaStreamQuery(G->st);                 /* hand the work to the device now */
        if (rc) G->launch_err = 1;
        MemoryBarrier();
        G->done_f_seq = js;
    }
    (void)c;
    return 0;
}
#endif

static int launcher_start(q38_model *m)
{
#ifdef _WIN32
    const char *e = getenv("Q38_LAUNCHER");
    if (e && !atoi(e)) return 0;
    if (!G->use_graphs) return 0;
    unsigned *hf = NULL;
    cudaSetDevice(G->dev);
    if (cudaHostAlloc((void **)&hf, (size_t)m->c.n_layer * sizeof(unsigned), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
        return 0;
    memset(hf, 0, (size_t)m->c.n_layer * sizeof(unsigned));
    if (cudaHostGetDevicePointer((void **)&G->d_flag, hf, 0) != cudaSuccess) return 0;
    if (cudaMalloc(&G->d_seen, (size_t)m->c.n_layer * sizeof(unsigned)) != cudaSuccess) return 0;
    cudaMemset(G->d_seen, 0, (size_t)m->c.n_layer * sizeof(unsigned));
    G->h_flag = hf;
    G->launch_thread = CreateThread(NULL, 0, launch_main, m, 0, NULL);
    if (!G->launch_thread) return 0;
    G->launcher = 1;
#else
    (void)m;
#endif
    return 0;
}

static void launcher_stop(void)
{
#ifdef _WIN32
    if (!G->launcher) return;
    G->launch_stop = 1;
    WaitForSingleObject(G->launch_thread, INFINITE);
    CloseHandle(G->launch_thread);
    G->launcher = 0;
#endif
}

static inline void spin_until(volatile long *v, long want)
{
    while (*v != want) _mm_pause();
    MemoryBarrier();
}

/* the experts layer il will most likely route to, from its router on the previous layer's
 * MoE input: those in neither VRAM nor RAM start reading now */
static void predict_next(q38_model *m, int il, const float *logits, int slot)
{
    const int NE = m->c.n_exp, P = G->predict;
    int n = 0, *ids = G->pred_ids[slot];
    float tv[64];
    for (int e = 0; e < NE; e++) {
        const float v = logits[e];
        if (n == P && !(v > tv[n - 1])) continue;
        int j = n < P ? n++ : n - 1;
        while (j > 0 && tv[j - 1] < v) { tv[j] = tv[j - 1]; ids[j] = ids[j - 1]; j--; }
        tv[j] = v;
        ids[j] = e;
    }
    G->pred_n[slot] = n;
    G->pred_layer[slot] = il;
    for (int i = 0; i < n; i++) {
        const int e = ids[i], key = il * NE + e;
        if (G->vslot && G->vslot[key] >= 0) continue;
        if (q38_model_expert_known(m, il, e)) continue;
        q38_model_prefetch(m, il, e);
    }
}

#include "q38_gpu_dense.cuh"

extern "C" int q38_gpu_forward(void *model, int token)
{
    q38_model *m = (q38_model *)model;
    if (m->c.dense) return dense_forward(m, token);
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, K = c->n_exp_used, F = c->ff_exp;
    cudaSetDevice(G->dev);
    G->warm_want = 1;

    *G->h_pos = m->n_past;
    q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)token * m->tok_embd.row_bytes, G->h_emb, E);
    if (c->ple_layer >= 0) {
        double tp = now_s();
        if (q38_model_ple_emb(m, token, G->h_ple_emb)) return -1;
        m->t_ple += now_s() - tp;
    }
    double tl0 = now_s();
    if (run_segment(m, 0)) return -1;
    const int lau = G->launcher && G->use_graphs && !m->tap;
    for (int il = 0; il < c->n_layer; il++) {
        if (!lau || il == 0) {
            if (run_segment(m, GF(il))) return -1;
        } else {
            spin_until(&G->done_f_seq, G->job_seq);          /* F(il) submitted by the launcher */
            if (G->launch_err) return -1;
        }
        double tw = now_s();
        G->t_launch += tw - tl0;
        CK(ev_wait(G->ev));
        m->t_gpu_wait += now_s() - tw;
        double tr0 = now_s();
        int top[64], cpu_ids[64], n_gpu = 0, n_cpu = 0, n_aux = 0;
        float w[64], cpu_w[64];
        q38_model_route(m, il, G->h_router, top, w);
        double th = now_s();
        G->t_rt += th - tr0;
        if (G->pred_layer[0] == il) {      /* the one-layer-ahead guess for this layer */
            for (int k = 0; k < K; k++)
                for (int i = 0; i < G->pred_n[0]; i++) if (G->pred_ids[0][i] == top[k]) { G->pred_hit++; break; }
            G->pred_total += K;
        }
        LayerDesc *hd = G->h_ldesc + il;
        LayerDesc *hd2 = G->h_ldesc2 ? G->h_ldesc2 + il : NULL;
        const q38_layer *L = &m->L[il];
        const int64_t bg = (int64_t)F * L->ex_gate.row_bytes, bu = (int64_t)F * L->ex_up.row_bytes;
        for (int k = 0; k < K; k++) {
            const int key = il * c->n_exp + top[k];
            if (G->freq) G->freq[key] += 1.0f;
            const int vs = G->vslot ? G->vslot[key] : -1;
            if (vs >= 0 && G->vs_pd[vs] == 1 && hd2) {
                hd2->e[n_aux].g = G->vs_ptr[vs];
                hd2->e[n_aux].u = G->vs_ptr[vs] + bg;
                hd2->e[n_aux].d = G->vs_ptr[vs] + bg + bu;
                hd2->e[n_aux].w = w[k];
                n_aux++;
            } else if (vs >= 0) {
                hd->e[n_gpu].g = G->vs_ptr[vs];
                hd->e[n_gpu].u = G->vs_ptr[vs] + bg;
                hd->e[n_gpu].d = G->vs_ptr[vs] + bg + bu;
                hd->e[n_gpu].w = w[k];
                n_gpu++;
            } else {
                cpu_ids[n_cpu] = top[k];
                cpu_w[n_cpu] = w[k];
                n_cpu++;
                if (G->cand) G->cand[G->n_cand++] = key;
            }
        }
        hd->n = n_gpu;
        if (hd2) hd2->n = n_aux;
        G->vram_hits += n_gpu + n_aux;
        G->aux_hits += n_aux;
        double t1 = now_s();
        G->vram_misses += n_cpu;
        if (lau) {
            G->job_il = il;
            G->job_aux = n_aux > 0;
            G->job_last = il == c->n_layer - 1;
            MemoryBarrier();
            G->job_seq++;
        } else {
            if (n_aux && run_aux(m, il)) return -1;
        }
        double t2 = now_s();
        G->t_auxl += t2 - t1;
        if (!lau && run_segment(m, GX(il))) return -1;
        double t3 = now_s();
        G->t_gxl += t3 - t2;
        if (G->predict > 0 && il + 1 < c->n_layer) {
            predict_next(m, il + 1, G->h_router_next, 0);
            if (G->pred_ahead > 1 && il + 2 < c->n_layer) predict_next(m, il + 2, G->h_router_next + c->n_exp, 1);
        }
        double t4 = now_s();
        G->t_pred += t4 - t3;
        G->t_route += t4 - tr0;
        const double xr = m->t_read, xe = m->t_experts;
        const int xrc = q38_model_experts(m, il, G->h_x, cpu_ids, cpu_w, n_cpu, G->h_y);
        double t5 = now_s();
        G->t_xcall += (t5 - t4) - (m->t_read - xr) - (m->t_experts - xe);
        if (n_aux) {
            if (lau) spin_until(&G->done_aux_seq, G->job_seq);
            CK(ev_wait(G->ev2));
            for (int i = 0; i < E; i++) G->h_y[i] += G->h_y2[i];
        }
        if (lau && il + 1 < c->n_layer) {
            _mm_sfence();
            G->h_flag[il] = G->h_flag[il] + 1;               /* F(il+1) may combine this layer now */
        }
        if (xrc) return -1;
        G->t_auxw += now_s() - t5;
        tl0 = now_s();
    }
    if (lau) {
        spin_until(&G->done_f_seq, G->job_seq);              /* GX(last) before the tail */
        if (G->launch_err) return -1;
    }
    if (run_segment(m, GT)) return -1;
    double tt = now_s();
    CK(cudaStreamSynchronize(G->st));
    memcpy(m->logits, G->h_logits, (size_t)c->n_vocab * sizeof(float));
    m->t_tail += now_s() - tt;
    if (m->tap) m->tap(m->tap_ctx, "logits", m->logits, c->n_vocab);
    CK(cudaGetLastError());
    double tp0 = now_s();
    if (G->nvs) pool_promote(m, G);
    G->t_promote += now_s() - tp0;
    G->tokens_fwd++;
    return 0;
}

#include "q38_gpu_batch.cuh"
#include "q38_gpu_dense_batch.cuh"
