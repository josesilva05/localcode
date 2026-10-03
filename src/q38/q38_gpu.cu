/* q38_gpu.cu - see q38_gpu.h. Each kernel repeats the CPU step of the same name in
 * q38_model.c; the comments name the llama.cpp graph node when it helps. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    DMat      output, out_hc_norm, out_hc_down, out_hc_up;
    /* activations */
    float    *res, *xn, *mixed, *lo, *gate, *inject, *y, *routed;
    float    *qkv, *z, *o, *beta, *g, *qf, *aout;
    float    *router, *sh_gu, *sh_h, *sh_dn, *sg, *logits;
    /* state */
    float    *conv[Q38_MAX_LAYERS], *ssm[Q38_MAX_LAYERS], *kc[Q38_MAX_LAYERS], *vc[Q38_MAX_LAYERS];
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
    /* prompt pass: experts streamed from RAM into a staging area and computed here */
    uint8_t  *pstage, *praw[16], *phost[16];  /* streamed experts: device raw bytes, pinned bounce */
    cudaEvent_t pev[16];                  /* slot i's raw bytes consumed */
    cudaStream_t st_cp;                   /* their copies and rearrangement, beside the compute */
    cudaEvent_t ev_stage, ev_pairs, ev_pup;   /* a round staged; its pairs done; pair lists uploaded */
    int       pst_n, pring;
    int64_t   pst_slot, stream_bytes;
    double    t_stream, t_sp_copy, t_sp_pairs;
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

/* build_norm_gated: rmsnorm(o) * w * sigmoid(z), one block per head */
__global__ void k_norm_gate(float *o, const float *z, const float *w, int S, float eps)
{
    __shared__ float part[8];
    float *oh = o + (int64_t)blockIdx.x * S;
    const float *zh = z + (int64_t)blockIdx.x * S;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) ss += oh[i] * oh[i];
    ss = block_sum(ss, part);
    const float sc = 1.0f / sqrtf(ss / S + eps);
    for (int i = threadIdx.x; i < S; i += blockDim.x) oh[i] = oh[i] * sc * w[i] * sigmoidf_(zh[i]);
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
__global__ void k_kv_store(const float *k, const float *v, float *kc, float *vc, const int *ppos, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    kc[(int64_t)*ppos * n + i] = k[i];
    vc[(int64_t)*ppos * n + i] = v[i];
}

__global__ void k_attn(const float *qf, const float *kc, const float *vc, float *out, const int *ppos, int H, int Hkv, int D,
                       const int *sel, int W)
{
    const int T0 = *ppos + 1, use_sel = sel && T0 > W, T = use_sel ? W : T0;
    extern __shared__ float sc[];   /* T scores */
    __shared__ float part[8];
    const int h = blockIdx.x, kvh = h / (H / Hkv), lane = threadIdx.x & 31, wp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const float *q = qf + (int64_t)h * 2 * D, *gate = q + D;
    const float scale = 1.0f / sqrtf((float)D);
    for (int t = wp; t < T; t += nw) {
        const float *kk = kc + ((int64_t)(use_sel ? sel[t] : t) * Hkv + kvh) * D;
        float s = 0.0f;
        for (int i = lane; i < D; i += 32) s += q[i] * kk[i];
        s = warp_sum(s);
        if (lane == 0) sc[t] = s * scale;
    }
    __syncthreads();
    float mx = -1e30f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) mx = fmaxf(mx, sc[t]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) part[wp] = mx;
    __syncthreads();
    if (threadIdx.x == 0) { float m2 = -1e30f; for (int i = 0; i < nw; i++) m2 = fmaxf(m2, part[i]); part[0] = m2; }
    __syncthreads();
    mx = part[0];
    __syncthreads();
    float sum = 0.0f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) { const float e = expf(sc[t] - mx); sc[t] = e; sum += e; }
    const float tot = block_sum(sum, part);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float a2 = 0.0f;
        for (int t = 0; t < T; t++) a2 += sc[t] * vc[((int64_t)(use_sel ? sel[t] : t) * Hkv + kvh) * D + i];
        out[(int64_t)h * D + i] = a2 / tot * sigmoidf_(gate[i]);
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

__device__ float lane_dot(int type, const uint8_t *w, const float *xs, int ne0, int lane)
{
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
    }
}

__device__ __forceinline__ int64_t repack_units(int type, int rows, int ne0)
{
    if (type == Q38_Q4_K) return (int64_t)rows * (ne0 / 256) * 36;
    if (type == Q38_Q5_K) return (int64_t)rows * (ne0 / 256) * 44;
    if (type == Q38_Q5_1) return (int64_t)rows * (ne0 / 32) * 6;
    if (type == Q38_Q8_0) return (int64_t)rows * (ne0 / 32) * 17;
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

/* --------------------------------------------------------------------- host -- */
static Gpu *G;

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

static int mv(const DMat *w, const float *x, float *y, cudaStream_t st)
{
    const dim3 blk(32, ROWS_PER_BLOCK);
    const int grid = (int)((w->rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
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
    } else {
        fprintf(stderr, "q38_gpu: no device kernel for type %s\n", q38_type_name(w->type));
        return -1;
    }
    return 0;
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

static int hc_mix(q38_model *m, const DMat *norm, const DMat *down, const DMat *up, const DMat *inj, float *mixed,
                  float *inject)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, hc = c->hc;
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
    if (!G || !G->nvs || !m->use) return 0;
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
    if (upload(&g->output, &m->output, g) || upload(&g->out_hc_norm, &m->out_hc_norm, g) ||
        upload(&g->out_hc_down, &m->out_hc_down, g) || upload(&g->out_hc_up, &m->out_hc_up, g))
        return -1;

    const int E = c->n_embd, hcE = c->hc * E, S = c->d_state;
    const int ch = 2 * c->n_kh * S + c->n_vh * S;
    g->res = dalloc(hcE, g); g->xn = dalloc(hcE, g); g->mixed = dalloc(E, g); g->lo = dalloc(c->hc_rank, g);
    g->gate = dalloc(hcE, g); g->inject = dalloc(64, g); g->y = dalloc(E, g); g->routed = dalloc(E, g);
    g->qkv = dalloc(ch, g); g->z = dalloc((size_t)c->n_vh * S, g); g->o = dalloc((size_t)c->n_vh * S, g);
    g->beta = dalloc(256, g); g->g = dalloc(256, g);
    g->qf = dalloc((size_t)c->n_head * c->head_dim * 2, g); g->aout = dalloc((size_t)c->n_head * c->head_dim, g);
    g->router = dalloc(c->n_exp, g); g->sh_gu = dalloc(2 * c->ff_shexp, g); g->sh_h = dalloc(c->ff_shexp, g);
    g->sh_dn = dalloc(E, g); g->sg = dalloc(1, g); g->logits = dalloc(c->n_vocab, g);
    for (int il = 0; il < c->n_layer; il++) {
        if (m->L[il].recurrent) {
            g->conv[il] = dalloc((size_t)(c->conv_k - 1) * ch, g);
            g->ssm[il] = dalloc((size_t)c->n_vh * S * S, g);
        } else {
            g->kc[il] = dalloc((size_t)m->max_ctx * c->n_head_kv * c->head_dim, g);
            g->vc[il] = dalloc((size_t)m->max_ctx * c->n_head_kv * c->head_dim, g);
        }
    }
    if (!g->logits) { fprintf(stderr, "q38_gpu: out of device memory\n"); return -1; }
    CK(cudaMallocHost(&g->h_x, (size_t)E * sizeof(float)));
    CK(cudaMallocHost(&g->h_router, (size_t)c->n_exp * sizeof(float)));
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
            int ns = 1;
            while (ns < nblk) ns <<= 1;
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
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    cudaSetDevice(G->dev);
    for (int il = 0; il < c->n_layer; il++) {
        if (G->conv[il]) cudaMemset(G->conv[il], 0, (size_t)(c->conv_k - 1) * ch * sizeof(float));
        if (G->ssm[il]) cudaMemset(G->ssm[il], 0, (size_t)c->n_vh * S * S * sizeof(float));
    }
    cudaDeviceSynchronize();
}

extern "C" void q38_gpu_close(void *model)
{
    q38_model *m = (q38_model *)model;
    if (!G) return;
    if (G->vram_hits + G->vram_misses)
        fprintf(stderr, "q38: routed experts on the GPUs %.1f%% (%lld of %lld; %lld on the second), %lld uploaded to VRAM\n",
                100.0 * G->vram_hits / (double)(G->vram_hits + G->vram_misses), (long long)G->vram_hits,
                (long long)(G->vram_hits + G->vram_misses), (long long)G->aux_hits, (long long)G->uploads);
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
    cudaDeviceReset();
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
    if (hc_mix(m, &LM(il, hc_attn_norm), &LM(il, hc_attn_down), &LM(il, hc_attn_up), &LM(il, hc_attn_inject),
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
        k_norm_gate<<<Hv, 128, 0, st>>>(G->o, G->z, (const float *)LM(il, ssm_norm).d, S, c->eps);
        if (mv(&LM(il, ssm_out), G->o, G->y, st)) return -1;
    } else {
        const int H = c->n_head, Hkv = c->n_head_kv, D = c->head_dim;
        if (mv(&LM(il, wq), G->mixed, G->qf, st) || mv(&LM(il, wk), G->mixed, G->knew, st) ||
            mv(&LM(il, wv), G->mixed, G->vnew, st)) return -1;
        k_norm_rope<<<H, 256, 0, st>>>(G->qf, 2 * D, (const float *)LM(il, q_norm).d, D, c->n_rot, G->d_pos, c->rope_base, c->eps);
        k_norm_rope<<<Hkv, 256, 0, st>>>(G->knew, D, (const float *)LM(il, k_norm).d, D, c->n_rot, G->d_pos, c->rope_base, c->eps);
        k_kv_store<<<(Hkv * D + 255) / 256, 256, 0, st>>>(G->knew, G->vnew, G->kc[il], G->vc[il], G->d_pos, Hkv * D);
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
        k_attn<<<H, 256, (size_t)(m->max_ctx < W ? m->max_ctx : W) * sizeof(float), st>>>(G->qf, G->kc[il], G->vc[il], G->aout, G->d_pos, H, Hkv, D,
                                                                   G->idx_raw[il] ? G->isel : NULL, W);
        if (mv(&LM(il, wo), G->aout, G->y, st)) return -1;
        if (m->tap && !capturing) { snprintf(nm, sizeof nm, "attn_output-%d", il); tap(m, nm, G->y, E); }
    }
    k_hc_combine<<<(hcE + 255) / 256, 256, 0, st>>>(G->res, G->y, G->inject, E, hc);
    if (hc_mix(m, &LM(il, hc_ffn_norm), &LM(il, hc_ffn_down), &LM(il, hc_ffn_up), &LM(il, hc_ffn_inject),
               G->mixed, G->inject)) return -1;
    if (m->tap && !capturing) { snprintf(nm, sizeof nm, "hc_mixed_ffn-%d", il); tap(m, nm, G->mixed, E); }
    /* router on the device; x and the logits to the host; the shared expert meanwhile */
    if (mv(&LM(il, router), G->mixed, G->router, st)) return -1;
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
    CK(cudaSetDevice(G->dev));
    return rc;
}

static int seg_head(q38_model *m, int capturing)
{
    const q38_cfg *c = &m->c;
    if (hc_mix(m, &G->out_hc_norm, &G->out_hc_down, &G->out_hc_up, NULL, G->mixed, NULL)) return -1;
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
    if (id == GT) return seg_post(m, c->n_layer - 1, capturing) || seg_head(m, capturing) ? -1 : 0;
    if (id == GP) return (c->ple_layer > 0 && seg_post(m, c->ple_layer - 1, capturing)) || seg_res_out() ? -1 : 0;
    if (id >= GX(0)) return seg_experts(m, id - GX(0));
    const int il = id - GF(0);
    if (il > 0 && seg_post(m, il - 1, capturing)) return -1;
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

extern "C" int q38_gpu_forward(void *model, int token)
{
    q38_model *m = (q38_model *)model;
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
    for (int il = 0; il < c->n_layer; il++) {
        if (run_segment(m, GF(il))) return -1;
        double tw = now_s();
        G->t_launch += tw - tl0;
        CK(ev_wait(G->ev));
        m->t_gpu_wait += now_s() - tw;
        double tr0 = now_s();
        int top[64], cpu_ids[64], n_gpu = 0, n_cpu = 0, n_aux = 0;
        float w[64], cpu_w[64];
        q38_model_route(m, il, G->h_router, top, w);
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
        if (n_aux && run_aux(m, il)) return -1;
        G->vram_misses += n_cpu;
        if (run_segment(m, GX(il))) return -1;
        G->t_route += now_s() - tr0;
        if (q38_model_experts(m, il, G->h_x, cpu_ids, cpu_w, n_cpu, G->h_y)) return -1;
        if (n_aux) {
            CK(ev_wait(G->ev2));
            for (int i = 0; i < E; i++) G->h_y[i] += G->h_y2[i];
        }
        tl0 = now_s();
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
