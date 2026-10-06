#include <cublas_v2.h>
#include <algorithm>
/* q38_gpu_batch.cuh - the forward pass for N consecutive tokens at once (included by
 * q38_gpu.cu, after its kernels and helpers).
 *
 * Layer by layer for all N tokens: every trunk matrix is read once per group of 8 tokens
 * (k_mm_*), the steps that carry state from one token to the next (Gated DeltaNet, its
 * convolution, the PLE convolution, the indexer's block pooling) walk the N tokens in order
 * inside one kernel, attention gives each token its own position and indexer selection, and
 * the MoE costs one round trip to the host per layer instead of one per token: the host
 * routes all N tokens, the (token, expert) pairs whose expert is in VRAM run on that device,
 * the rest on the CPU, and each token's sum is formed in a fixed order.
 *
 * Arithmetic per token is the decode path's; the order of some float additions differs
 * (several tokens per weight load), so the results agree with q38_gpu_forward to rounding.
 * Used to read prompts (q38_gpu_forward_batch with want_all = 0) and, later, to verify
 * drafted tokens. */

#define TB 8                        /* tokens per weight load in the matrix products */

/* experts in tiles of up to XT tokens (k_ext_*) */
#define XT 8
typedef struct { const uint8_t *g, *u, *d; int p0, n; } ExTile;

typedef struct {
    int    nb;                      /* capacity in tokens */
    int    maxp;                    /* capacity in (token, expert) pairs */
    /* activations, [nb][...] */
    float *res, *xn, *mixed, *lo, *gate, *inject, *y, *routed, *ygpu;
    float *qkv, *z, *o, *beta, *g, *qf, *knew, *vnew, *aout;
    float *router, *shgu, *shh, *sh, *sg;
    float *ple_emb, *ple_key, *ple_val, *ple_gated, *ple_normed, *ple_gate;
    float *ikraw, *iq, *iscore, *logits;
    int   *isel;
    int    nblk;                    /* indexer score row length */
    /* experts on the trunk's device */
    ExDesc *h_pairs, *d_pairs;
    int   *h_np, *d_np;
    int   *h_tp, *d_tp;             /* [nb][K]: each token's pair indices, -1 after the last */
    float *h_tw, *d_tw;             /* [nb][K]: their weights */
    float *egu, *eh, *edn;
    ExTile *h_tiles, *d_tiles;
    /* experts on the second device */
    ExDesc *h_pairs2, *d_pairs2;
    ExTile *h_tiles2, *d_tiles2;
    int   *h_np2, *d_np2, *h_tp2, *d_tp2;
    float *h_tw2, *d_tw2, *x2, *egu2, *eh2, *edn2, *y2, *h_y2;
    /* pinned host */
    float *h_emb, *h_ple, *h_x, *h_router, *h_y, *h_logits;
    int   *h_pos;
} Bat;

static Bat *GB;

/* ------------------------------------------------------------------- kernels -- */
/* Y[n][r] = row r . X[n] for N tokens, repacked Q8_0; a warp per row, TB tokens per pass */
__global__ void k_mm_q8r(const int8_t *__restrict__ Q, const float *__restrict__ D, int ne0, int rows,
                         const float *__restrict__ X, int ldx, float *__restrict__ Y, int ldy, int N)
{
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y, n0 = blockIdx.y * TB;
    if (r >= rows || n0 >= N) return;
    const int nt = N - n0 < TB ? N - n0 : TB, nb = ne0 / 32;
    const int8_t *q = Q + (int64_t)r * ne0;
    float acc[TB];
#pragma unroll
    for (int t = 0; t < TB; t++) acc[t] = 0.0f;
    for (int i = threadIdx.x * 16; i < ne0; i += 32 * 16) {
        const int4 v = __ldg((const int4 *)(q + i));
        const int8_t *b = (const int8_t *)&v;
        const float d = __ldg(D + (int64_t)r * nb + i / 32);
#pragma unroll
        for (int t = 0; t < TB; t++) {
            if (t >= nt) break;
            const float *x = X + (int64_t)(n0 + t) * ldx + i;
            const float4 x0 = __ldg((const float4 *)x), x1 = __ldg((const float4 *)(x + 4));
            const float4 x2 = __ldg((const float4 *)(x + 8)), x3 = __ldg((const float4 *)(x + 12));
            float s = b[0] * x0.x + b[1] * x0.y + b[2] * x0.z + b[3] * x0.w;
            s += b[4] * x1.x + b[5] * x1.y + b[6] * x1.z + b[7] * x1.w;
            s += b[8] * x2.x + b[9] * x2.y + b[10] * x2.z + b[11] * x2.w;
            s += b[12] * x3.x + b[13] * x3.y + b[14] * x3.z + b[15] * x3.w;
            acc[t] += d * s;
        }
    }
#pragma unroll
    for (int t = 0; t < TB; t++) {
        const float v = warp_sum(acc[t]);
        if (threadIdx.x == 0 && t < nt) Y[(int64_t)(n0 + t) * ldy + r] = v;
    }
}

__global__ void k_mm_f32(const float *__restrict__ W, int ne0, int rows, const float *__restrict__ X, int ldx,
                         float *__restrict__ Y, int ldy, int N)
{
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y, n0 = blockIdx.y * TB;
    if (r >= rows || n0 >= N) return;
    const int nt = N - n0 < TB ? N - n0 : TB;
    const float *w = W + (int64_t)r * ne0;
    float acc[TB];
#pragma unroll
    for (int t = 0; t < TB; t++) acc[t] = 0.0f;
    for (int i = threadIdx.x * 4; i < ne0; i += 128) {
        const float4 a = __ldg((const float4 *)(w + i));
#pragma unroll
        for (int t = 0; t < TB; t++) {
            if (t >= nt) break;
            const float4 b = __ldg((const float4 *)(X + (int64_t)(n0 + t) * ldx + i));
            acc[t] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
#pragma unroll
    for (int t = 0; t < TB; t++) {
        const float v = warp_sum(acc[t]);
        if (threadIdx.x == 0 && t < nt) Y[(int64_t)(n0 + t) * ldy + r] = v;
    }
}

/* rows of n: y = x / sqrt(mean(x^2) + eps) * w[row % wrows] */
__global__ void k_rmsnorm_rows_w(const float *x, const float *w, float *y, int n, float eps, int wrows)
{
    __shared__ float part[8];
    const float *xr = x + (int64_t)blockIdx.x * n;
    const float *wr = w + (int64_t)(blockIdx.x % wrows) * n;
    float *yr = y + (int64_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float sc = 1.0f / sqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * sc * wr[i];
}

__global__ void k_hc_mean_b(const float *xn, const float *gate, float *mixed, int E, int hc, int N)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * E) return;
    const int64_t t = i / E, d = i % E;
    float s = 0.0f;
    for (int st = 0; st < hc; st++) s += xn[(t * hc + st) * E + d] * sigmoidf_(gate[(t * hc + st) * E + d]);
    mixed[i] = s / hc;
}

__global__ void k_hc_combine_b(float *res, const float *out, const float *inject, int E, int hc, int N)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * hc * E) return;
    const int64_t t = i / ((int64_t)hc * E), s = (i / E) % hc, d = i % E;
    res[i] += 2.0f * sigmoidf_(inject[t * hc + s] / hc) * out[t * E + d];
}

__global__ void k_tile_b(float *res, const float *x, int E, int hc, int N)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * hc * E) return;
    res[i] = x[(i / ((int64_t)hc * E)) * E + i % E];
}

__global__ void k_silu_scale_b(float *x, int64_t n, float inv)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = siluf_(x[i] * inv);
}

__global__ void k_gdn_gates_b(float *beta, float *g, const float *dt_bias, const float *a, int H, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * N) return;
    const int h = i % H;
    beta[i] = sigmoidf_(beta[i]);
    const float t = g[i] + dt_bias[h];
    g[i] = (t > 20.0f ? t : logf(1.0f + expf(t))) * a[h];
}

/* the short convolution over N tokens in order; the state keeps the last K-1 inputs */
__global__ void k_conv_silu_b(float *x, int ldx, float *state, const float *w, int ch, int K, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ch || K > 8) return;
    float prev[7];
#pragma unroll
    for (int k = 0; k < 7; k++) prev[k] = k < K - 1 ? state[(int64_t)k * ch + i] : 0.0f;
    for (int t = 0; t < N; t++) {
        const float xi = x[(int64_t)t * ldx + i];
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k < K - 1) acc += prev[k] * w[(int64_t)i * K + k];
        acc += xi * w[(int64_t)i * K + K - 1];
#pragma unroll
        for (int k = 0; k < 6; k++) if (k < K - 2) prev[k] = prev[k + 1];
#pragma unroll
        for (int k = 0; k < 7; k++) if (k == K - 2) prev[k] = xi;
        x[(int64_t)t * ldx + i] = siluf_(acc);
    }
#pragma unroll
    for (int k = 0; k < 7; k++) if (k < K - 1) state[(int64_t)k * ch + i] = prev[k];
}

/* rows of S inside each token's vector: row r -> token r / rpt, piece r % rpt */
__global__ void k_l2norm_b(float *x, int S, int rpt, int ld, float eps)
{
    float *xr = x + (int64_t)(blockIdx.x / rpt) * ld + (int64_t)(blockIdx.x % rpt) * S;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < S; i += 32) ss += xr[i] * xr[i];
    ss = warp_sum(ss);
    const float sc = 1.0f / sqrtf(ss + eps);
    for (int i = threadIdx.x; i < S; i += 32) xr[i] *= sc;
}

/* the Gated DeltaNet recurrence of one value head over N tokens in order */
__global__ void k_gdn_step_b(float *state, const float *qkv, int ld, const float *g, const float *beta, float *o,
                             int S, int Hk, int Hv, int N)
{
    extern __shared__ float sh[];
    float *ks = sh, *qs = sh + S, *red = sh + 2 * S;
    const int h = blockIdx.x, j = threadIdx.x, p = threadIdx.y, P = blockDim.y, kh = h % Hk;
    float *St = state + (int64_t)h * S * S;
    const int i0 = p * (S / P), i1 = i0 + S / P;
    for (int t = 0; t < N; t++) {
        const float *row = qkv + (int64_t)t * ld;
        if (p == 0) { qs[j] = row[kh * S + j]; ks[j] = row[Hk * S + kh * S + j]; }
        __syncthreads();
        const float eg = expf(g[t * Hv + h]);
        float sk = 0.0f;
        for (int i = i0; i < i1; i++) sk += St[(int64_t)i * S + j] * eg * ks[i];
        red[p * S + j] = sk;
        __syncthreads();
        float skt = 0.0f;
        for (int u = 0; u < P; u++) skt += red[u * S + j];
        const float d = (row[2 * Hk * S + h * S + j] - skt) * beta[t * Hv + h];
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
            for (int u = 0; u < P; u++) ot += red[u * S + j];
            o[(int64_t)t * Hv * S + h * S + j] = ot * (1.0f / sqrtf((float)S));
        }
        __syncthreads();
    }
}

/* rmsnorm + rope of head rows: block = token * rows + head; the token's position is p0 + token */
__global__ void k_norm_rope_b(float *x, int64_t stride, int rows, int64_t tok_stride, const float *w, int D, int n_rot,
                              const int *ppos, double base, float eps)
{
    const int t = blockIdx.x / rows, hr = blockIdx.x % rows, pos = *ppos + t;
    __shared__ float part[8];
    __shared__ float buf[1024];
    float *xr = x + (int64_t)t * tok_stride + (int64_t)hr * stride;
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

__global__ void k_kv_store_b(const float *k, const float *v, float *kc, float *vc, const int *ppos, int n, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * N) return;
    const int t = i / n, e = i % n;
    kc[(int64_t)(*ppos + t) * n + e] = k[i];
    vc[(int64_t)(*ppos + t) * n + e] = v[i];
}

/* causal attention of token blockIdx.y (position p0 + t) for head blockIdx.x */
__global__ void k_attn_b(const float *qf, const float *kc, const float *vc, float *out, const int *ppos, int H, int Hkv, int D,
                         const int *sel, int W)
{
    const int t = blockIdx.y;
    const int T0 = *ppos + t + 1, use_sel = sel && T0 > W, T = use_sel ? W : T0;
    const int *sl = sel ? sel + (int64_t)t * W : NULL;
    extern __shared__ float sc[];
    __shared__ float part[8];
    const int h = blockIdx.x, kvh = h / (H / Hkv), lane = threadIdx.x & 31, wp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const float *q = qf + (int64_t)t * H * 2 * D + (int64_t)h * 2 * D, *gate = q + D;
    const float scale = 1.0f / sqrtf((float)D);
    for (int c = wp; c < T; c += nw) {
        const float *kk = kc + ((int64_t)(use_sel ? sl[c] : c) * Hkv + kvh) * D;
        float s = 0.0f;
        for (int i = lane; i < D; i += 32) s += q[i] * kk[i];
        s = warp_sum(s);
        if (lane == 0) sc[c] = s * scale;
    }
    __syncthreads();
    float mx = -1e30f;
    for (int c = threadIdx.x; c < T; c += blockDim.x) mx = fmaxf(mx, sc[c]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) part[wp] = mx;
    __syncthreads();
    if (threadIdx.x == 0) { float m2 = -1e30f; for (int i = 0; i < nw; i++) m2 = fmaxf(m2, part[i]); part[0] = m2; }
    __syncthreads();
    mx = part[0];
    __syncthreads();
    float sum = 0.0f;
    for (int c = threadIdx.x; c < T; c += blockDim.x) { const float e = expf(sc[c] - mx); sc[c] = e; sum += e; }
    const float tot = block_sum(sum, part);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float a2 = 0.0f;
        for (int c = 0; c < T; c++) a2 += sc[c] * vc[((int64_t)(use_sel ? sl[c] : c) * Hkv + kvh) * D + i];
        out[(int64_t)t * H * D + (int64_t)h * D + i] = a2 / tot * sigmoidf_(gate[i]);
    }
}

/* indexer keys of N tokens in order: store raw, pool each block its last member completes */
__global__ void k_idx_store_b(const float *kraw, float *raw, float *pool, const float *kn, const int *ppos,
                              int DI, int R, int n_rot, double base, float eps, int N)
{
    __shared__ float part[8];
    __shared__ float buf[512];
    const int i = threadIdx.x;
    for (int t = 0; t < N; t++) {
        const int pos = *ppos + t;
        if (i < DI) raw[(int64_t)pos * DI + i] = kraw[(int64_t)t * DI + i];
        __syncthreads();
        if ((pos + 1) % R) continue;
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
        __syncthreads();
    }
}

__global__ void k_idx_score_b(const float *qi, const float *pool, float *score, int ldq, int lds, const int *ppos, int W,
                              int R, int NH, int DI)
{
    const int t = blockIdx.y, pos = *ppos + t;
    if (pos + 1 <= W) return;
    const int nfull = (pos + 1) / R;
    const int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b >= nfull) return;
    const float *pk = pool + (int64_t)b * DI, *q = qi + (int64_t)t * ldq;
    float s = 0.0f;
    for (int h = 0; h < NH; h++) {
        float d = 0.0f;
        for (int i = lane; i < DI; i += 32) d += q[h * DI + i] * pk[i];
        d = warp_sum(d);
        s += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) score[(int64_t)t * lds + b] = s;
}

__global__ void k_idx_select_b(const float *score, int lds, int *sel, const int *ppos, int W, int R, int NS)
{
    extern __shared__ float sk[];
    int *si = (int *)(sk + NS);
    const int t = blockIdx.x, pos = *ppos + t;
    if (pos + 1 <= W) return;
    const float *sc = score + (int64_t)t * lds;
    const int tail_start = (pos + 1) / R * R, nfull = tail_start / R;
    for (int i = threadIdx.x; i < NS; i += blockDim.x) { sk[i] = i < nfull ? sc[i] : -1e30f; si[i] = i; }
    __syncthreads();
    for (int k = 2; k <= NS; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < NS; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
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
    const int nt = pos + 1 - tail_start;
    int *out = sel + (int64_t)t * W;
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        int cell;
        if (c < nt) cell = tail_start + c;
        else { const int k = c - nt; cell = si[k / R] * R + k % R; }
        out[c] = cell;
    }
}

/* PLE for N tokens: gates per (token, stream), then the dilated convolution in order */
__global__ void k_ple_gate_b(const float *key, const float *res, const float *nk, const float *nq, float *gate, int E,
                             int hc, float eps)
{
    __shared__ float part[8];
    const int s = blockIdx.x, t = blockIdx.y, hcE = hc * E;
    const float *k = key + (int64_t)t * hcE + (int64_t)s * E, *r = res + (int64_t)t * hcE + (int64_t)s * E;
    const float *wk = nk + (int64_t)s * E, *wq = nq + (int64_t)s * E;
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
        const float scv = d / sqrtf((float)E);
        const float mag = sqrtf(fabsf(scv) > 1e-6f ? fabsf(scv) : 1e-6f);
        gate[t * hc + s] = sigmoidf_(scv > 0 ? mag : scv < 0 ? -mag : 0.0f);
    }
}

__global__ void k_ple_gated_b(const float *value, const float *gate, const float *nconv, float *gated, float *normed, int E,
                              int hc, float eps)
{
    __shared__ float part[8];
    const int s = blockIdx.x, t = blockIdx.y, hcE = hc * E;
    float *gs = gated + (int64_t)t * hcE + (int64_t)s * E, *ns = normed + (int64_t)t * hcE + (int64_t)s * E;
    const float gv = gate[t * hc + s];
    const float *val = value + (int64_t)t * E;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { const float v = val[i] * gv; gs[i] = v; ss += v * v; }
    ss = block_sum(ss, part);
    const float c = 1.0f / sqrtf(ss / E + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) ns[i] = gs[i] * c * nconv[(int64_t)s * E + i];
}

__global__ void k_ple_conv_b(float *res, const float *gated, const float *normed, float *ring, const float *w,
                             const int *ppos, int n, int K, int dil, int hist, int N)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n) return;
    const int p0 = *ppos;
    for (int t = 0; t < N; t++) {
        const int pos = p0 + t;
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            const int q = pos - (K - 1 - k) * dil;
            float x = 0.0f;
            if (q >= p0) x = normed[(int64_t)(q - p0) * n + ch];
            else if (q >= 0) x = ring[(int64_t)(q % hist) * n + ch];
            acc += w[(int64_t)ch * K + k] * x;
        }
        res[(int64_t)t * n + ch] += gated[(int64_t)t * n + ch] + siluf_(acc);
    }
    for (int t = (N > hist ? N - hist : 0); t < N; t++) ring[(int64_t)((p0 + t) % hist) * n + ch] = normed[(int64_t)t * n + ch];
}

/* routed experts of N tokens: pair k = (expert in VRAM, token ex[k].pad, weight) */
__global__ void k_exb_gu(const ExDesc *ex, const int *pn, int tg, int tu, int E, int F, const float *__restrict__ X, float *gu)
{
    const int k = blockIdx.y;
    if (k >= *pn) return;
    const int j = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (j >= 2 * F) return;
    const float *x = X + (int64_t)ex[k].pad * E;
    const float v = j < F ? lane_dot_rp(tg, ex[k].g, F, j, x, E, threadIdx.x)
                          : lane_dot_rp(tu, ex[k].u, F, j - F, x, E, threadIdx.x);
    const float s = warp_sum(v);
    if (threadIdx.x == 0) gu[(int64_t)k * 2 * F + j] = s;
}

__global__ void k_exb_swiglu(const int *pn, const float *gu, float *h, int F, int maxp)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)maxp * F) return;
    const int k = (int)(i / F), j = (int)(i % F);
    if (k >= *pn) return;
    h[i] = siluf_(gu[(int64_t)k * 2 * F + j]) * gu[(int64_t)k * 2 * F + F + j];
}

__global__ void k_exb_down(const ExDesc *ex, const int *pn, int td, int F, int E, const float *__restrict__ h, float *dn)
{
    const int k = blockIdx.y;
    if (k >= *pn) return;
    const int i = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (i >= E) return;
    const float s = warp_sum(lane_dot_rp(td, ex[k].d, E, i, h + (int64_t)k * F, F, threadIdx.x));
    if (threadIdx.x == 0) dn[(int64_t)k * E + i] = s;
}

/* ------------------------------------------------ experts in tiles of tokens -- */
/* The pairs of one expert in tiles of up to XT tokens: a warp decodes a weight row's chunk
 * once and applies it to every token of the tile. Each token's sum is formed exactly as in
 * lane_dot_rp (same chunks per lane, same order), so the results do not change. */

__device__ __forceinline__ void lane_dot_rp_t(int type, const uint8_t *base, int rows, int r, const float *X,
                                              const int *xoff, int n, int ne0, int lane, float *acc)
{
#pragma unroll
    for (int t = 0; t < XT; t++) acc[t] = 0.0f;
    if (is_iq(type)) {                              /* a chunk decoded once, applied to every token */
        const uint8_t *w = base + (int64_t)r * iq_row_bytes(type, ne0);
        const int nch = iq_chunks(type, ne0);
        for (int c = lane; c < nch; c += 32) {
            float wt[8];
            int o0, o1;
            iq_chunk(type, w, c, wt, &o0, &o1);
#pragma unroll
            for (int t = 0; t < XT; t++) {
                if (t >= n) break;
                acc[t] += iq_apply(wt, X + xoff[t], o0, o1);
            }
        }
    } else if (type == Q38_Q4_K || type == Q38_Q5_K) {
        const int nsb = ne0 / 256, five = type == Q38_Q5_K;
        const uint8_t *qs = base + (int64_t)r * (ne0 / 2);
        const uint8_t *qh = base + (int64_t)rows * (ne0 / 2) + (int64_t)r * nsb * 32;
        const uint8_t *hd = base + (int64_t)rows * (ne0 / 2) + (five ? (int64_t)rows * nsb * 32 : 0) + (int64_t)r * nsb * 16;
        for (int c = lane; c < nsb * 8; c += 32) {
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
            float lo[16], hi[16];
#pragma unroll
            for (int u = 0; u < 16; u++) {
                const int qb = (qw[u >> 2] >> (8 * (u & 3))) & 0xFF;
                const int hb = (hw[u >> 2] >> (8 * (u & 3))) & 0xFF;
                lo[u] = (float)((qb & 15) + (five ? (((hb >> (2 * j)) & 1) << 4) : 0));
                hi[u] = (float)((qb >> 4) + (five ? (((hb >> (2 * j + 1)) & 1) << 4) : 0));
            }
#pragma unroll
            for (int t = 0; t < XT; t++) {
                if (t >= n) break;
                const float4 *xl4 = (const float4 *)(X + xoff[t] + sb * 256 + 64 * j + p), *xh4 = xl4 + 8;
                float sl = 0.0f, sh = 0.0f, xsl = 0.0f, xsh = 0.0f;
#pragma unroll
                for (int v = 0; v < 4; v++) {
                    const float4 a = __ldg(xl4 + v), b2 = __ldg(xh4 + v);
                    const float xl[4] = { a.x, a.y, a.z, a.w }, xh[4] = { b2.x, b2.y, b2.z, b2.w };
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        sl += lo[4 * v + k] * xl[k]; sh += hi[4 * v + k] * xh[k]; xsl += xl[k]; xsh += xh[k];
                    }
                }
                acc[t] += dd.x * s1 * sl - dd.y * m1 * xsl + dd.x * s2 * sh - dd.y * m2 * xsh;
            }
        }
    } else if (type == Q38_Q5_1) {
        const int nb = ne0 / 32;
        const uint8_t *qs = base + (int64_t)r * (ne0 / 2);
        const uint32_t *qh = (const uint32_t *)(const void *)(base + (int64_t)rows * (ne0 / 2)) + (int64_t)r * nb;
        const __half2 *dm = (const __half2 *)(const void *)(base + (int64_t)rows * (ne0 / 2) + (int64_t)rows * nb * 4) + (int64_t)r * nb;
        for (int b = lane; b < nb; b += 32) {
            const int4 q4 = *(const int4 *)(const void *)(qs + b * 16);
            const uint32_t h = qh[b];
            const float2 f = __half22float2(dm[b]);
            const uint32_t qw[4] = { (uint32_t)q4.x, (uint32_t)q4.y, (uint32_t)q4.z, (uint32_t)q4.w };
            float v0[16], v1[16];
#pragma unroll
            for (int u = 0; u < 16; u++) {
                const int qb = (qw[u >> 2] >> (8 * (u & 3))) & 0xFF;
                v0[u] = (float)((qb & 15) | (((h >> u) & 1) << 4));
                v1[u] = (float)((qb >> 4) | (((h >> (u + 16)) & 1) << 4));
            }
#pragma unroll
            for (int t = 0; t < XT; t++) {
                if (t >= n) break;
                const float4 *x4 = (const float4 *)(X + xoff[t] + b * 32);
                float x[32];
#pragma unroll
                for (int v = 0; v < 8; v++) {
                    const float4 a = __ldg(x4 + v);
                    x[4 * v] = a.x; x[4 * v + 1] = a.y; x[4 * v + 2] = a.z; x[4 * v + 3] = a.w;
                }
                float s = 0.0f, xsum = 0.0f;
#pragma unroll
                for (int u = 0; u < 16; u++) {
                    s += v0[u] * x[u] + v1[u] * x[u + 16];
                    xsum += x[u] + x[u + 16];
                }
                acc[t] += f.x * s + f.y * xsum;
            }
        }
    } else if (type == Q38_Q8_0) {
        const int nb = ne0 / 32;
        const int8_t *qs = (const int8_t *)(base + (int64_t)r * ne0);
        const __half *d = (const __half *)(const void *)(base + (int64_t)rows * ne0) + (int64_t)r * nb;
        for (int c = lane; c < nb * 2; c += 32) {
            const int4 q4 = *(const int4 *)(const void *)(qs + c * 16);
            const int8_t *q = (const int8_t *)&q4;
            const float dc = __half2float(d[c >> 1]);
#pragma unroll
            for (int t = 0; t < XT; t++) {
                if (t >= n) break;
                const float4 *x4 = (const float4 *)(X + xoff[t] + c * 16);
                float s = 0.0f;
#pragma unroll
                for (int v = 0; v < 4; v++) {
                    const float4 a = __ldg(x4 + v);
                    s += (float)q[4 * v] * a.x + (float)q[4 * v + 1] * a.y + (float)q[4 * v + 2] * a.z + (float)q[4 * v + 3] * a.w;
                }
                acc[t] += dc * s;
            }
        }
    }
}

/* gate and up: blockIdx.y = tile, a warp per row; token of pair p is ex[p].pad */
__global__ void k_ext_gu(const ExTile *tl, const ExDesc *ex, int tg, int tu, int E, int F, const float *__restrict__ X, float *gu)
{
    const ExTile T = tl[blockIdx.y];
    const int j = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (j >= 2 * F) return;
    int xoff[XT];
#pragma unroll
    for (int t = 0; t < XT; t++) xoff[t] = t < T.n ? ex[T.p0 + t].pad * E : 0;
    float acc[XT];
    if (j < F) lane_dot_rp_t(tg, T.g, F, j, X, xoff, T.n, E, threadIdx.x, acc);
    else lane_dot_rp_t(tu, T.u, F, j - F, X, xoff, T.n, E, threadIdx.x, acc);
#pragma unroll
    for (int t = 0; t < XT; t++) {
        const float s = warp_sum(acc[t]);
        if (threadIdx.x == 0 && t < T.n) gu[(int64_t)(T.p0 + t) * 2 * F + j] = s;
    }
}

/* down: the tile's pairs are consecutive, so their h rows are too */
__global__ void k_ext_down(const ExTile *tl, int td, int F, int E, const float *__restrict__ h, float *dn)
{
    const ExTile T = tl[blockIdx.y];
    const int i = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (i >= E) return;
    int xoff[XT];
#pragma unroll
    for (int t = 0; t < XT; t++) xoff[t] = (T.p0 + (t < T.n ? t : 0)) * F;
    float acc[XT];
    lane_dot_rp_t(td, T.d, E, i, h, xoff, T.n, F, threadIdx.x, acc);
#pragma unroll
    for (int t = 0; t < XT; t++) {
        const float s = warp_sum(acc[t]);
        if (threadIdx.x == 0 && t < T.n) dn[(int64_t)(T.p0 + t) * E + i] = s;
    }
}

/* each token's routed sum over its pairs, in the order the host listed them */
__global__ void k_exb_reduce(const int *tp, const float *tw, int KT, const float *dn, int E, float *y, int N)
{   /* accumulates: y is cleared once per layer, then each group of pairs adds its share */
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * E) return;
    const int t = (int)(i / E), d = (int)(i % E);
    float a = 0.0f;
    for (int j = 0; j < KT; j++) {
        const int p = tp[t * KT + j];
        if (p < 0) break;
        a += tw[t * KT + j] * dn[(int64_t)p * E + d];
    }
    y[i] += a;
}

__global__ void k_swiglu_b(const float *gu, float *h, int F, int N)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * F) return;
    const int64_t t = i / F, j = i % F;
    h[i] = siluf_(gu[t * 2 * F + j]) * gu[t * 2 * F + F + j];
}

/* y = routed (host + second device) + routed (this device) + sigmoid(sg) * shared, per token */
__global__ void k_moe_final_b(float *y, const float *routed, const float *ygpu, const float *sh, const float *sg, int E, int N)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * E) return;
    y[i] = routed[i] + ygpu[i] + sigmoidf_(sg[i / E]) * sh[i];
}

/* --------------------------------------------------------------------- host -- */
static inline int nblocks(int64_t n, int b) { return (int)((n + b - 1) / b); }

/* Many tokens at once: the matrix is expanded to fp32 in a scratch buffer and multiplied by
 * cuBLAS (fp32 SGEMM, no reduced precision unless Q38_TF32=1); few tokens use k_mm_*. */
static cublasHandle_t g_blas;
static float  *g_wscr;
static int64_t g_wscr_n;
static int     g_blas_min = 1 << 30;

__global__ void k_deq_q8r(const int8_t *__restrict__ Q, const float *__restrict__ D, int64_t n, float *__restrict__ W)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        W[i] = (float)Q[i] * D[i >> 5];
}

static int blas_open(q38_model *m)
{
    const char *e = getenv("Q38_BLAS_MIN");
    const int mn = e ? atoi(e) : 32;
    if (mn <= 0) return 0;                                /* Q38_BLAS_MIN=0: never */
    int64_t mx = 0;
    for (int il = 0; il < m->c.n_layer; il++)
        for (int i = 0; i < NMAT; i++) {
            const DMat *w = &G->L[il][i];
            if (w->type == Q38_Q8R && w->rows * w->ne0 > mx) mx = w->rows * w->ne0;
        }
    if (cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "q38_gpu: cuBLAS unavailable\n"); return 0; }
    const char *t = getenv("Q38_TF32");
    if (t && atoi(t)) cublasSetMathMode(g_blas, CUBLAS_TF32_TENSOR_OP_MATH);
    if (mx > 0 && cudaMalloc(&g_wscr, (size_t)mx * sizeof(float)) != cudaSuccess) { cudaGetLastError(); mx = 0; g_wscr = NULL; }
    g_wscr_n = mx;
    G->bytes += mx * (int64_t)sizeof(float);
    g_blas_min = mn;
    return 0;
}

/* Y[N][rows] = W . X[N] */
static int mm(const DMat *w, const float *X, int ldx, float *Y, int ldy, int N, cudaStream_t st)
{
    if (N >= g_blas_min && (w->type == Q38_F32 || (w->type == Q38_Q8R && w->rows * w->ne0 <= g_wscr_n))) {
        /* row-major Y[N][rows] = X[N][ne0] W^T is column-major Y^T = W^T' X^T' with W read as
         * the ne0 x rows column-major matrix it is in memory */
        const float *W = (const float *)w->d;
        if (w->type == Q38_Q8R) {
            k_deq_q8r<<<1024, 256, 0, st>>>((const int8_t *)w->d, w->scale, w->rows * w->ne0, g_wscr);
            W = g_wscr;
        }
        const float one = 1.0f, zero = 0.0f;
        cublasSetStream(g_blas, st);
        if (cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)w->rows, N, (int)w->ne0, &one, W, (int)w->ne0, X, ldx, &zero,
                        Y, ldy) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "q38_gpu: cublasSgemm failed\n");
            return -1;
        }
        return 0;
    }
    const dim3 blk(32, ROWS_PER_BLOCK), grid(nblocks(w->rows, ROWS_PER_BLOCK), nblocks(N, TB));
    if (w->type == Q38_Q8R) k_mm_q8r<<<grid, blk, 0, st>>>((const int8_t *)w->d, w->scale, (int)w->ne0, (int)w->rows, X, ldx, Y, ldy, N);
    else if (w->type == Q38_F32) k_mm_f32<<<grid, blk, 0, st>>>((const float *)w->d, (int)w->ne0, (int)w->rows, X, ldx, Y, ldy, N);
    else {                                                  /* no batched kernel (Q6_K): token by token */
        for (int t = 0; t < N; t++) if (mv(w, X + (int64_t)t * ldx, Y + (int64_t)t * ldy, st)) return -1;
    }
    return 0;
}

static int bat_open(q38_model *m, int nb)
{
    const q38_cfg *c = &m->c;
    if (nb <= 0) {
        const char *e = getenv("Q38_BATCH");
        nb = e ? atoi(e) : 1024;
        if (nb < 1) nb = 1;
        if (nb > 1024) nb = 1024;
    }
    if (blas_open(m)) return -1;
    Bat *b = (Bat *)calloc(1, sizeof(Bat));
    if (!b) return -1;
    {   /* the staging area for experts streamed from RAM during a prompt pass */
        int64_t mx = 0;
        for (int il = 0; il < c->n_layer; il++) {
            const q38_layer *l = &m->L[il];
            const int64_t eb = (int64_t)c->ff_exp * (l->ex_gate.row_bytes + l->ex_up.row_bytes) + (int64_t)c->n_embd * l->ex_down.row_bytes;
            if (eb > mx) mx = eb;
        }
        const char *eg = getenv("Q38_PREFILL_STAGE_GB");
        const double gb = eg ? atof(eg) : 1.0;
        G->pst_slot = (mx + 255) / 256 * 256;
        G->pst_n = (int)(gb * 1e9 / (double)G->pst_slot);
        if (G->pst_n > 0) {
            CK(cudaSetDevice(G->dev));
            CK(cudaMalloc(&G->pstage, (size_t)G->pst_n * (size_t)G->pst_slot));
            /* a ring of pinned bounce buffers: CPU threads copy experts into one half while the
             * other half's bytes go over the bus (Q38_PREFILL_PINNED=0: pageable copies) */
            const char *ep = getenv("Q38_PREFILL_PINNED");
            G->pring = ep && !atoi(ep) ? 0 : 16;
            for (int i = 0; i < 16; i++) CK(cudaMalloc(&G->praw[i], (size_t)G->pst_slot));
            for (int i = 0; i < G->pring; i++) {
                if (cudaHostAlloc((void **)&G->phost[i], (size_t)G->pst_slot, cudaHostAllocDefault) != cudaSuccess) {
                    G->pring = 0;
                    break;
                }
                CK(cudaEventCreateWithFlags(&G->pev[i], cudaEventDisableTiming));
            }
            CK(cudaStreamCreateWithFlags(&G->st_cp, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&G->ev_stage, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&G->ev_pairs, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&G->ev_pup, cudaEventDisableTiming));
            G->bytes += (int64_t)G->pst_n * G->pst_slot;
        }
    }
    const int E = c->n_embd, hc = c->hc, hcE = hc * E, S = c->d_state, K = c->n_exp_used, F = c->ff_exp;
    const int ch = 2 * c->n_kh * S + c->n_vh * S;
    b->nb = nb;
    b->maxp = nb * K;
    cudaSetDevice(G->dev);
#define DA(p, n) do { if (!((p) = dalloc((size_t)(n), G))) return -1; } while (0)
    DA(b->res, (int64_t)nb * hcE); DA(b->xn, (int64_t)nb * hcE); DA(b->gate, (int64_t)nb * hcE);
    DA(b->mixed, (int64_t)nb * E); DA(b->lo, (int64_t)nb * c->hc_rank); DA(b->inject, (int64_t)nb * hc);
    DA(b->y, (int64_t)nb * E); DA(b->routed, (int64_t)nb * E); DA(b->ygpu, (int64_t)nb * E);
    DA(b->qkv, (int64_t)nb * ch); DA(b->z, (int64_t)nb * c->n_vh * S); DA(b->o, (int64_t)nb * c->n_vh * S);
    DA(b->beta, (int64_t)nb * c->n_vh); DA(b->g, (int64_t)nb * c->n_vh);
    DA(b->qf, (int64_t)nb * c->n_head * c->head_dim * 2);
    DA(b->knew, (int64_t)nb * c->n_head_kv * c->head_dim); DA(b->vnew, (int64_t)nb * c->n_head_kv * c->head_dim);
    DA(b->aout, (int64_t)nb * c->n_head * c->head_dim);
    DA(b->router, (int64_t)nb * c->n_exp); DA(b->shgu, (int64_t)nb * 2 * c->ff_shexp); DA(b->shh, (int64_t)nb * c->ff_shexp);
    DA(b->sh, (int64_t)nb * E); DA(b->sg, nb);
    DA(b->logits, c->n_vocab);
    DA(b->egu, (int64_t)b->maxp * 2 * F); DA(b->eh, (int64_t)b->maxp * F); DA(b->edn, (int64_t)b->maxp * E);
    if (c->ple_layer >= 0) {
        const int pin = c->ple_heads * c->ple_dim;
        DA(b->ple_emb, (int64_t)nb * pin); DA(b->ple_key, (int64_t)nb * hcE); DA(b->ple_val, (int64_t)nb * E);
        DA(b->ple_gated, (int64_t)nb * hcE); DA(b->ple_normed, (int64_t)nb * hcE); DA(b->ple_gate, (int64_t)nb * hc);
        CK(cudaMallocHost(&b->h_ple, (size_t)nb * pin * sizeof(float)));
    }
    if (c->idx_top_k > 0 && G->ins) {
        b->nblk = m->max_ctx / c->idx_ratio + 1;
        DA(b->ikraw, (int64_t)nb * c->idx_dim); DA(b->iq, (int64_t)nb * c->idx_heads * c->idx_dim);
        DA(b->iscore, (int64_t)nb * b->nblk);
        CK(cudaMalloc(&b->isel, (size_t)nb * c->max_dense_ctx * sizeof(int)));
        CK(cudaFuncSetAttribute(k_idx_select_b, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * G->ins * 4));
    }
    CK(cudaMallocHost(&b->h_pairs, (size_t)b->maxp * sizeof(ExDesc)));
    CK(cudaMalloc(&b->d_pairs, (size_t)b->maxp * sizeof(ExDesc)));
    CK(cudaMallocHost(&b->h_tiles, (size_t)b->maxp * sizeof(ExTile)));
    CK(cudaMalloc(&b->d_tiles, (size_t)b->maxp * sizeof(ExTile)));
    CK(cudaMallocHost(&b->h_np, sizeof(int)));
    CK(cudaMalloc(&b->d_np, sizeof(int)));
    CK(cudaMallocHost(&b->h_tp, (size_t)nb * K * sizeof(int)));
    CK(cudaMalloc(&b->d_tp, (size_t)nb * K * sizeof(int)));
    CK(cudaMallocHost(&b->h_tw, (size_t)nb * K * sizeof(float)));
    CK(cudaMalloc(&b->d_tw, (size_t)nb * K * sizeof(float)));
    CK(cudaMallocHost(&b->h_emb, (size_t)nb * E * sizeof(float)));
    CK(cudaMallocHost(&b->h_x, (size_t)nb * E * sizeof(float)));
    CK(cudaMallocHost(&b->h_router, (size_t)nb * c->n_exp * sizeof(float)));
    CK(cudaMallocHost(&b->h_y, (size_t)nb * E * sizeof(float)));
    CK(cudaMallocHost(&b->h_logits, (size_t)c->n_vocab * sizeof(float)));
    CK(cudaMallocHost(&b->h_pos, sizeof(int)));
#undef DA
    GB = b;
    return 0;
}


/* the second device's share of the batch buffers (after pool_open has found that device) */
static int bat_open_aux(q38_model *m)
{
    const q38_cfg *c = &m->c;
    Bat *b = GB;
    const int nb = b->nb, E = c->n_embd, K = c->n_exp_used, F = c->ff_exp;
    if (G->npd == 2 && b) {
        CK(cudaSetDevice(G->pdev[1]));
        CK(cudaMalloc(&b->d_pairs2, (size_t)b->maxp * sizeof(ExDesc)));
        CK(cudaMalloc(&b->d_tiles2, (size_t)b->maxp * sizeof(ExTile)));
        CK(cudaMallocHost(&b->h_tiles2, (size_t)b->maxp * sizeof(ExTile)));
        CK(cudaMalloc(&b->d_np2, sizeof(int)));
        CK(cudaMalloc(&b->d_tp2, (size_t)nb * K * sizeof(int)));
        CK(cudaMalloc(&b->d_tw2, (size_t)nb * K * sizeof(float)));
        CK(cudaMalloc(&b->x2, (size_t)nb * E * sizeof(float)));
        CK(cudaMalloc(&b->egu2, (size_t)b->maxp * 2 * F * sizeof(float)));
        CK(cudaMalloc(&b->eh2, (size_t)b->maxp * F * sizeof(float)));
        CK(cudaMalloc(&b->edn2, (size_t)b->maxp * E * sizeof(float)));
        CK(cudaMalloc(&b->y2, (size_t)nb * E * sizeof(float)));
        CK(cudaMallocHost(&b->h_pairs2, (size_t)b->maxp * sizeof(ExDesc)));
        CK(cudaMallocHost(&b->h_np2, sizeof(int)));
        CK(cudaMallocHost(&b->h_tp2, (size_t)nb * K * sizeof(int)));
        CK(cudaMallocHost(&b->h_tw2, (size_t)nb * K * sizeof(float)));
        CK(cudaMallocHost(&b->h_y2, (size_t)nb * E * sizeof(float)));
        CK(cudaSetDevice(G->dev));
    }
    return 0;
}

/* hyper-connection mix of N tokens: res [N][hc*E] -> mixed [N][E], inject [N][hc] */
static int hc_mix_b(q38_model *m, const float *res, int N, const DMat *norm, const DMat *down, const DMat *up,
                    const DMat *inj, float *mixed, float *inject)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, hc = c->hc, hcE = hc * E, R = c->hc_rank;
    cudaStream_t st = G->st;
    k_rmsnorm_rows_w<<<N * hc, 256, 0, st>>>(res, (const float *)norm->d, GB->xn, E, c->eps, hc);
    if (mm(down, GB->xn, hcE, GB->lo, R, N, st)) return -1;
    k_silu_scale_b<<<nblocks((int64_t)N * R, 256), 256, 0, st>>>(GB->lo, (int64_t)N * R, 1.0f / hc);
    if (mm(up, GB->lo, R, GB->gate, hcE, N, st)) return -1;
    k_hc_mean_b<<<nblocks((int64_t)N * E, 256), 256, 0, st>>>(GB->xn, GB->gate, mixed, E, hc, N);
    if (inj && mm(inj, GB->xn, hcE, inject, hc, N, st)) return -1;
    return 0;
}

/* routed experts of one layer for N tokens on one device: pairs in h_pairs, per-token lists */
static int run_pairs(q38_model *m, int il, int dev2, int np, const float *X, float *Y, int N)
{
    const q38_cfg *c = &m->c;
    const q38_layer *L = &m->L[il];
    const int E = c->n_embd, F = c->ff_exp, K = c->n_exp_used;
    Bat *b = GB;
    cudaStream_t st = dev2 ? G->st2 : G->st;
    ExDesc *hp = dev2 ? b->h_pairs2 : b->h_pairs, *dp = dev2 ? b->d_pairs2 : b->d_pairs;
    int *hn = dev2 ? b->h_np2 : b->h_np, *dn = dev2 ? b->d_np2 : b->d_np;
    int *htp = dev2 ? b->h_tp2 : b->h_tp, *dtp = dev2 ? b->d_tp2 : b->d_tp;
    float *htw = dev2 ? b->h_tw2 : b->h_tw, *dtw = dev2 ? b->d_tw2 : b->d_tw;
    float *egu = dev2 ? b->egu2 : b->egu, *eh = dev2 ? b->eh2 : b->eh, *edn = dev2 ? b->edn2 : b->edn;
    ExTile *ht = dev2 ? b->h_tiles2 : b->h_tiles, *dt = dev2 ? b->d_tiles2 : b->d_tiles;
    static int tiled = -1;
    if (tiled < 0) tiled = !(getenv("Q38_EXPERT_TILES") && !atoi(getenv("Q38_EXPERT_TILES")));
    int nt = 0;
    if (tiled) {
        /* pairs grouped by expert (stable), the per-token lists renamed to match, then tiles of
         * up to XT consecutive pairs of one expert */
        static int ord[16384], inv[16384];
        static ExDesc tmp[16384];
        if (np > 16384) return -1;
        for (int i = 0; i < np; i++) ord[i] = i;
        std::stable_sort(ord, ord + np, [hp](int a, int c) { return hp[a].g < hp[c].g; });
        for (int i = 0; i < np; i++) { tmp[i] = hp[ord[i]]; inv[ord[i]] = i; }
        memcpy(hp, tmp, (size_t)np * sizeof(ExDesc));
        for (int t = 0; t < N; t++)
            for (int j = 0; j < K; j++) {
                const int pp = htp[t * K + j];
                if (pp < 0) break;
                htp[t * K + j] = inv[pp];
            }
        for (int i = 0; i < np;) {
            int k = i + 1;
            while (k < np && k - i < XT && hp[k].g == hp[i].g) k++;
            ExTile *T = &ht[nt++];
            T->g = hp[i].g; T->u = hp[i].u; T->d = hp[i].d; T->p0 = i; T->n = k - i;
            i = k;
        }
        CK(cudaMemcpyAsync(dt, ht, (size_t)nt * sizeof(ExTile), cudaMemcpyHostToDevice, st));
    }
    *hn = np;
    CK(cudaMemcpyAsync(dn, hn, sizeof(int), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(dp, hp, (size_t)np * sizeof(ExDesc), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(dtp, htp, (size_t)N * K * sizeof(int), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(dtw, htw, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice, st));
    if (!dev2 && G->ev_pup) CK(cudaEventRecord(G->ev_pup, st));   /* the host may rewrite its lists after this */
    const dim3 blk(32, ROWS_PER_BLOCK);
    if (tiled) {
        k_ext_gu<<<dim3(nblocks(2 * F, ROWS_PER_BLOCK), nt), blk, 0, st>>>(dt, dp, L->ex_gate.type, L->ex_up.type, E, F, X, egu);
        k_exb_swiglu<<<nblocks((int64_t)np * F, 256), 256, 0, st>>>(dn, egu, eh, F, np);
        k_ext_down<<<dim3(nblocks(E, ROWS_PER_BLOCK), nt), blk, 0, st>>>(dt, L->ex_down.type, F, E, eh, edn);
        static int chk = -1;
        if (chk < 0) chk = getenv("Q38_TILE_CHECK") ? atoi(getenv("Q38_TILE_CHECK")) : 0;
        if (chk && !dev2) {           /* the per-pair kernels on the same inputs: must match bit for bit */
            static float *cg, *cd, *hg, *hc;
            static size_t cap;
            const size_t ng = (size_t)np * 2 * F, nd = (size_t)np * E;
            if (ng + nd > cap) {
                cudaFree(cg); cudaFree(cd); free(hg); free(hc);
                cap = (ng + nd) * 2;
                CK(cudaMalloc(&cg, cap * sizeof(float))); CK(cudaMalloc(&cd, cap * sizeof(float)));
                hg = (float *)malloc(cap * sizeof(float)); hc = (float *)malloc(cap * sizeof(float));
            }
            k_exb_gu<<<dim3(nblocks(2 * F, ROWS_PER_BLOCK), np), blk, 0, st>>>(dp, dn, L->ex_gate.type, L->ex_up.type, E, F, X, cg);
            k_exb_down<<<dim3(nblocks(E, ROWS_PER_BLOCK), np), blk, 0, st>>>(dp, dn, L->ex_down.type, F, E, eh, cd);
            CK(cudaStreamSynchronize(st));
            int64_t bad = 0;
            CK(cudaMemcpy(hg, cg, ng * sizeof(float), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc, egu, ng * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < ng; i++) bad += memcmp(&hg[i], &hc[i], 4) != 0;
            const int64_t badg = bad;
            CK(cudaMemcpy(hg, cd, nd * sizeof(float), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hc, edn, nd * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < nd; i++) bad += memcmp(&hg[i], &hc[i], 4) != 0;
            fprintf(stderr, "q38: tile check layer %d: %d pairs in %d tiles, %lld gate/up and %lld down values differ (types %d %d %d)\n",
                    il, np, nt, (long long)badg, (long long)(bad - badg), L->ex_gate.type, L->ex_up.type, L->ex_down.type);
        }
    } else {
        k_exb_gu<<<dim3(nblocks(2 * F, ROWS_PER_BLOCK), np), blk, 0, st>>>(dp, dn, L->ex_gate.type, L->ex_up.type, E, F, X, egu);
        k_exb_swiglu<<<nblocks((int64_t)np * F, 256), 256, 0, st>>>(dn, egu, eh, F, np);
        k_exb_down<<<dim3(nblocks(E, ROWS_PER_BLOCK), np), blk, 0, st>>>(dp, dn, L->ex_down.type, F, E, eh, edn);
    }
    k_exb_reduce<<<nblocks((int64_t)N * E, 256), 256, 0, st>>>(dtp, dtw, K, edn, E, Y, N);
    return 0;
}

/* N tokens at positions n_past .. n_past+N-1. want: -1 = logits of the last token into
 * m->logits; otherwise nothing (more batches follow). */
/* The pairs whose expert is in neither VRAM pool: make the experts resident in RAM, then in
 * rounds of the staging area's size copy them to the device, rearrange them and run their
 * pairs there (adding into ygpu). */
/* last: the prompt's last batch. Its experts go into the cache as decoding uses it (least
 * recent out), since the answer that follows needs mostly the same ones; earlier batches
 * keep a full cache as it is (q38_model_experts_stream), or a long prompt would evict what
 * its next batch needs. */
static int stream_experts(q38_model *m, int il, int N, const int *ptok, const int *pexp, const float *pw, int npc, int last)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, F = c->ff_exp, K = c->n_exp_used, NE = c->n_exp;
    Bat *b = GB;
    cudaStream_t st = G->st, sc = G->st_cp;
    if (!npc) return 0;
    static int ue[1024], slot_of[1024];
    static const uint8_t *data[1024];
    int nu = 0;
    for (int e = 0; e < NE; e++) slot_of[e] = -1;
    for (int p = 0; p < npc; p++) if (slot_of[pexp[p]] < 0) { slot_of[pexp[p]] = nu; ue[nu++] = pexp[p]; }
    if (last ? q38_model_experts_resident(m, il, ue, nu, data) : q38_model_experts_stream(m, il, ue, nu, data)) return -1;
    const q38_layer *L = &m->L[il];
    const int64_t bg = (int64_t)F * L->ex_gate.row_bytes, bu = (int64_t)F * L->ex_up.row_bytes;
    const int64_t eb = bg + bu + (int64_t)E * L->ex_down.row_bytes;
    const int *ty = G->ex_types[il];
    static int prof = -1;
    if (prof < 0) prof = getenv("Q38_STREAM_PROF") ? atoi(getenv("Q38_STREAM_PROF")) : 0;
    for (int r0 = 0; r0 < nu; r0 += G->pst_n) {
        const int r1 = nu - r0 < G->pst_n ? nu : r0 + G->pst_n;
        /* the copy stream: the stage is free once the pairs that last read it are done */
        CK(cudaStreamWaitEvent(sc, G->ev_pairs, 0));
        const double ts = now_s();
        if (G->pring) {
            /* groups of half the ring: fill a group's pinned buffers in parallel (after their
             * previous bytes were consumed), then queue its copies and rearrangements */
            const int half = G->pring / 2;
            for (int g0 = r0, grp = 0; g0 < r1; g0 += half, grp ^= 1) {
                const int g1 = r1 - g0 < half ? r1 : g0 + half;
                for (int i = g0; i < g1; i++) CK(cudaEventSynchronize(G->pev[grp * half + (i - g0)]));
                int i;
#pragma omp parallel for schedule(static, 1) num_threads(8)
                for (i = g0; i < g1; i++) memcpy(G->phost[grp * half + (i - g0)], data[i], (size_t)eb);
                for (i = g0; i < g1; i++) {
                    const int k = grp * half + (i - g0);
                    uint8_t *raw = G->praw[k], *dst = G->pstage + (int64_t)(i - r0) * G->pst_slot;
                    CK(cudaMemcpyAsync(raw, G->phost[k], (size_t)eb, cudaMemcpyHostToDevice, sc));
                    repack_expert(raw, dst, ty, F, E, bg, bu, sc);
                    CK(cudaEventRecord(G->pev[k], sc));
                }
            }
        } else {
            for (int i = r0; i < r1; i++) {
                uint8_t *raw = G->praw[i & 15], *dst = G->pstage + (int64_t)(i - r0) * G->pst_slot;
                CK(cudaMemcpyAsync(raw, data[i], (size_t)eb, cudaMemcpyHostToDevice, sc));
                repack_expert(raw, dst, ty, F, E, bg, bu, sc);
            }
        }
        CK(cudaEventRecord(G->ev_stage, sc));
        G->stream_bytes += (int64_t)(r1 - r0) * eb;
        if (prof) { CK(cudaStreamSynchronize(sc)); G->t_sp_copy += now_s() - ts; }
        G->t_stream += now_s() - ts;
        /* the pair lists are rewritten in place: the last upload of them must have happened */
        CK(cudaEventSynchronize(G->ev_pup));
        int np = 0;
        for (int t = 0; t < N; t++) b->h_tp[t * K] = -1;
        static int cnt[1024];
        for (int t = 0; t < N; t++) cnt[t] = 0;
        for (int p = 0; p < npc; p++) {
            const int i = slot_of[pexp[p]];
            if (i < r0 || i >= r1) continue;
            const uint8_t *base = G->pstage + (int64_t)(i - r0) * G->pst_slot;
            ExDesc *e = &b->h_pairs[np];
            e->g = base; e->u = base + bg; e->d = base + bg + bu; e->w = pw[p]; e->pad = ptok[p];
            const int t = ptok[p];
            b->h_tp[t * K + cnt[t]] = np;
            b->h_tw[t * K + cnt[t]] = pw[p];
            cnt[t]++;
            if (cnt[t] < K) b->h_tp[t * K + cnt[t]] = -1;
            np++;
        }
        CK(cudaStreamWaitEvent(st, G->ev_stage, 0));
        const double tq = now_s();
        if (np && run_pairs(m, il, 0, np, b->mixed, b->ygpu, N)) return -1;
        CK(cudaEventRecord(G->ev_pairs, st));
        if (prof) { CK(cudaStreamSynchronize(st)); G->t_sp_pairs += now_s() - tq; }
    }
    return 0;
}

static int forward_batch_chunk(q38_model *m, const int *toks, int N, int want_last)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, hc = c->hc, hcE = hc * E, S = c->d_state, Hk = c->n_kh, Hv = c->n_vh;
    const int ch = 2 * Hk * S + Hv * S, K = c->n_exp_used, F = c->ff_exp, NE = c->n_exp;
    const int p0 = m->n_past;
    Bat *b = GB;
    cudaStream_t st = G->st;
    cudaSetDevice(G->dev);

    /* inputs: positions, embeddings, PLE rows (the n-grams need the tokens in m->toks) */
    for (int t = 0; t < N; t++) m->toks[p0 + t] = toks[t];
    *b->h_pos = p0;
    CK(cudaMemcpyAsync(G->d_pos, b->h_pos, sizeof(int), cudaMemcpyHostToDevice, st));
    for (int t = 0; t < N; t++)
        q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)toks[t] * m->tok_embd.row_bytes, b->h_emb + (int64_t)t * E, E);
    CK(cudaMemcpyAsync(b->mixed, b->h_emb, (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, st));
    k_tile_b<<<nblocks((int64_t)N * hcE, 256), 256, 0, st>>>(b->res, b->mixed, E, hc, N);
    if (c->ple_layer >= 0) {
        const int pin = c->ple_heads * c->ple_dim;
        double tp = now_s();
        if (q38_model_ple_emb_batch(m, p0, N, b->h_ple)) return -1;
        m->t_ple += now_s() - tp;
        CK(cudaMemcpyAsync(b->ple_emb, b->h_ple, (size_t)N * pin * sizeof(float), cudaMemcpyHostToDevice, st));
    }

    for (int il = 0; il < c->n_layer; il++) {
        const q38_layer *l = &m->L[il];
        if (il == c->ple_layer) {
            const int pin = c->ple_heads * c->ple_dim, hist = (c->ple_kern - 1) * c->ple_ngram;
            if (mm(&LM(il, ple_key), b->ple_emb, pin, b->ple_key, hcE, N, st) ||
                mm(&LM(il, ple_value), b->ple_emb, pin, b->ple_val, E, N, st)) return -1;
            k_ple_gate_b<<<dim3(hc, N), 256, 0, st>>>(b->ple_key, b->res, (const float *)LM(il, ple_norm_key).d,
                                                     (const float *)LM(il, ple_norm_query).d, b->ple_gate, E, hc, c->eps);
            k_ple_gated_b<<<dim3(hc, N), 256, 0, st>>>(b->ple_val, b->ple_gate, (const float *)LM(il, ple_norm_conv).d,
                                                      b->ple_gated, b->ple_normed, E, hc, c->eps);
            k_ple_conv_b<<<nblocks(hcE, 256), 256, 0, st>>>(b->res, b->ple_gated, b->ple_normed, G->ple_ring,
                                                           (const float *)LM(il, ple_conv).d, G->d_pos, hcE, c->ple_kern,
                                                           c->ple_ngram, hist, N);
        }
        if (hc_mix_b(m, b->res, N, &LM(il, hc_attn_norm), &LM(il, hc_attn_down), &LM(il, hc_attn_up), &LM(il, hc_attn_inject),
                     b->mixed, b->inject)) return -1;
        if (l->recurrent) {
            if (mm(&LM(il, qkv), b->mixed, E, b->qkv, ch, N, st) || mm(&LM(il, gate), b->mixed, E, b->z, Hv * S, N, st) ||
                mm(&LM(il, beta), b->mixed, E, b->beta, Hv, N, st) || mm(&LM(il, alpha), b->mixed, E, b->g, Hv, N, st)) return -1;
            k_gdn_gates_b<<<nblocks(Hv * N, 256), 256, 0, st>>>(b->beta, b->g, (const float *)LM(il, dt_bias).d,
                                                               (const float *)LM(il, a).d, Hv, N);
            k_conv_silu_b<<<nblocks(ch, 256), 256, 0, st>>>(b->qkv, ch, G->conv[il], (const float *)LM(il, conv).d, ch, c->conv_k, N);
            k_l2norm_b<<<N * 2 * Hk, 32, 0, st>>>(b->qkv, S, 2 * Hk, ch, c->eps);
            k_gdn_step_b<<<Hv, dim3(S, 4), 6 * S * sizeof(float), st>>>(G->ssm[il], b->qkv, ch, b->g, b->beta, b->o, S, Hk, Hv, N);
            k_norm_gate<<<N * Hv, 128, 0, st>>>(b->o, b->z, (const float *)LM(il, ssm_norm).d, S, c->eps);
            if (mm(&LM(il, ssm_out), b->o, Hv * S, b->y, E, N, st)) return -1;
        } else {
            const int H = c->n_head, Hkv = c->n_head_kv, D = c->head_dim, W = c->max_dense_ctx;
            if (mm(&LM(il, wq), b->mixed, E, b->qf, H * 2 * D, N, st) || mm(&LM(il, wk), b->mixed, E, b->knew, Hkv * D, N, st) ||
                mm(&LM(il, wv), b->mixed, E, b->vnew, Hkv * D, N, st)) return -1;
            k_norm_rope_b<<<N * H, 256, 0, st>>>(b->qf, 2 * D, H, (int64_t)H * 2 * D, (const float *)LM(il, q_norm).d, D,
                                                c->n_rot, G->d_pos, c->rope_base, c->eps);
            k_norm_rope_b<<<N * Hkv, 256, 0, st>>>(b->knew, D, Hkv, (int64_t)Hkv * D, (const float *)LM(il, k_norm).d, D,
                                                  c->n_rot, G->d_pos, c->rope_base, c->eps);
            k_kv_store_b<<<nblocks(Hkv * D * N, 256), 256, 0, st>>>(b->knew, b->vnew, G->kc[il], G->vc[il], G->d_pos, Hkv * D, N);
            const int *sel = NULL;
            if (G->idx_raw[il] && b->isel) {
                const int DI = c->idx_dim, R = c->idx_ratio, NH = c->idx_heads;
                if (mm(&LM(il, idx_k), b->mixed, E, b->ikraw, DI, N, st) || mm(&LM(il, idx_q), b->mixed, E, b->iq, NH * DI, N, st))
                    return -1;
                k_idx_store_b<<<1, DI, 0, st>>>(b->ikraw, G->idx_raw[il], G->idx_pool[il], (const float *)LM(il, idx_kn).d,
                                                G->d_pos, DI, R, c->n_rot, c->rope_base, c->eps, N);
                if (p0 + N > W) {
                    k_norm_rope_b<<<N * NH, 256, 0, st>>>(b->iq, DI, NH, (int64_t)NH * DI, (const float *)LM(il, idx_qn).d, DI,
                                                         c->n_rot, G->d_pos, c->rope_base, c->eps);
                    k_idx_score_b<<<dim3(nblocks(b->nblk, 8), N), 256, 0, st>>>(b->iq, G->idx_pool[il], b->iscore, NH * DI,
                                                                               b->nblk, G->d_pos, W, R, NH, DI);
                    k_idx_select_b<<<N, 1024, (size_t)2 * G->ins * 4, st>>>(b->iscore, b->nblk, b->isel, G->d_pos, W, R, G->ins);
                    sel = b->isel;
                }
            }
            k_attn_b<<<dim3(H, N), 256, (size_t)(m->max_ctx < W ? m->max_ctx : W) * sizeof(float), st>>>(
                b->qf, G->kc[il], G->vc[il], b->aout, G->d_pos, H, Hkv, D, sel, W);
            if (mm(&LM(il, wo), b->aout, H * D, b->y, E, N, st)) return -1;
        }
        k_hc_combine_b<<<nblocks((int64_t)N * hcE, 256), 256, 0, st>>>(b->res, b->y, b->inject, E, hc, N);
        if (hc_mix_b(m, b->res, N, &LM(il, hc_ffn_norm), &LM(il, hc_ffn_down), &LM(il, hc_ffn_up), &LM(il, hc_ffn_inject),
                     b->mixed, b->inject)) return -1;

        /* MoE: router and x to the host; the shared expert meanwhile */
        if (mm(&LM(il, router), b->mixed, E, b->router, NE, N, st)) return -1;
        CK(cudaMemcpyAsync(b->h_x, b->mixed, (size_t)N * E * sizeof(float), cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(b->h_router, b->router, (size_t)N * NE * sizeof(float), cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(G->ev, st));
        const int FS = c->ff_shexp;
        if (mm(&LM(il, sh_gate), b->mixed, E, b->shgu, 2 * FS, N, st) || mm(&LM(il, sh_up), b->mixed, E, b->shgu + FS, 2 * FS, N, st))
            return -1;
        k_swiglu_b<<<nblocks((int64_t)N * FS, 256), 256, 0, st>>>(b->shgu, b->shh, FS, N);
        if (mm(&LM(il, sh_down), b->shh, FS, b->sh, E, N, st) || mm(&LM(il, shexp_gate_inp), b->mixed, E, b->sg, 1, N, st))
            return -1;
        double tw = now_s();
        CK(ev_wait(G->ev));
        m->t_gpu_wait += now_s() - tw;

        /* route every token; split its experts between this device, the second one and the CPU */
        const q38_layer *Lr = &m->L[il];
        const int64_t bg = (int64_t)F * Lr->ex_gate.row_bytes, bu = (int64_t)F * Lr->ex_up.row_bytes;
        int np1 = 0, np2 = 0;
        static int cpu_ids[1024][16];
        static float cpu_w[1024][16];
        static int ncpu[1024];
        for (int t = 0; t < N; t++) {
            int top[64];
            float w[64];
            q38_model_route(m, il, b->h_router + (int64_t)t * NE, top, w);
            int j1 = 0, j2 = 0;
            ncpu[t] = 0;
            for (int k = 0; k < K; k++) {
                const int key = il * NE + top[k];
                if (G->freq) G->freq[key] += 1.0f;
                const int vs = G->vslot ? G->vslot[key] : -1;
                if (vs >= 0 && G->vs_pd[vs] == 1 && b->d_pairs2) {
                    ExDesc *e = &b->h_pairs2[np2];
                    e->g = G->vs_ptr[vs]; e->u = G->vs_ptr[vs] + bg; e->d = G->vs_ptr[vs] + bg + bu; e->w = w[k]; e->pad = t;
                    b->h_tp2[t * K + j2] = np2; b->h_tw2[t * K + j2] = w[k]; j2++; np2++;
                } else if (vs >= 0) {
                    ExDesc *e = &b->h_pairs[np1];
                    e->g = G->vs_ptr[vs]; e->u = G->vs_ptr[vs] + bg; e->d = G->vs_ptr[vs] + bg + bu; e->w = w[k]; e->pad = t;
                    b->h_tp[t * K + j1] = np1; b->h_tw[t * K + j1] = w[k]; j1++; np1++;
                } else {
                    cpu_ids[t][ncpu[t]] = top[k];
                    cpu_w[t][ncpu[t]] = w[k];
                    ncpu[t]++;
                    if (G->cand && G->n_cand < c->n_layer * 64) G->cand[G->n_cand++] = key;
                }
            }
            if (j1 < K) b->h_tp[t * K + j1] = -1;
            if (b->h_tp2 && j2 < K) b->h_tp2[t * K + j2] = -1;
            G->vram_hits += j1 + j2;
            G->aux_hits += j2;
            G->vram_misses += ncpu[t];
        }
        CK(cudaMemsetAsync(b->ygpu, 0, (size_t)N * E * sizeof(float), st));
        if (np1 && run_pairs(m, il, 0, np1, b->mixed, b->ygpu, N)) return -1;
        if (np2) {
            CK(cudaSetDevice(G->pdev[1]));
            CK(cudaMemcpyAsync(b->x2, b->h_x, (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, G->st2));
            CK(cudaMemsetAsync(b->y2, 0, (size_t)N * E * sizeof(float), G->st2));
            if (run_pairs(m, il, 1, np2, b->x2, b->y2, N)) return -1;
            CK(cudaMemcpyAsync(b->h_y2, b->y2, (size_t)N * E * sizeof(float), cudaMemcpyDeviceToHost, G->st2));
            CK(cudaEventRecord(G->ev2, G->st2));
            CK(cudaSetDevice(G->dev));
        }
        {
            static int ptok[1024 * 16], pexp[1024 * 16];
            static float pw[1024 * 16];
            int npc = 0;
            for (int t = 0; t < N; t++)
                for (int k = 0; k < ncpu[t]; k++) { ptok[npc] = t; pexp[npc] = cpu_ids[t][k]; pw[npc] = cpu_w[t][k]; npc++; }
            if (G->pst_n > 0 && N > 1 && !(getenv("Q38_PREFILL_CPU") && atoi(getenv("Q38_PREFILL_CPU")))) {
                for (int64_t i = 0; i < (int64_t)N * E; i++) b->h_y[i] = 0.0f;
                if (stream_experts(m, il, N, ptok, pexp, pw, npc, want_last)) return -1;
            } else if (q38_model_experts_batch(m, il, b->h_x, N, ptok, pexp, pw, npc, b->h_y)) {
                return -1;
            }
        }
        if (np2) {
            CK(ev_wait(G->ev2));
            for (int64_t i = 0; i < (int64_t)N * E; i++) b->h_y[i] += b->h_y2[i];
        }
        CK(cudaMemcpyAsync(b->routed, b->h_y, (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, st));
        k_moe_final_b<<<nblocks((int64_t)N * E, 256), 256, 0, st>>>(b->y, b->routed, b->ygpu, b->sh, b->sg, E, N);
        k_hc_combine_b<<<nblocks((int64_t)N * hcE, 256), 256, 0, st>>>(b->res, b->y, b->inject, E, hc, N);
    }
    /* the head, for the last token */
    if (want_last) {
        if (hc_mix_b(m, b->res + (int64_t)(N - 1) * hcE, 1, &G->out_hc_norm, &G->out_hc_down, &G->out_hc_up, NULL, b->mixed, NULL))
            return -1;
        if (mm(&G->output, b->mixed, E, b->logits, c->n_vocab, 1, st)) return -1;
        CK(cudaMemcpyAsync(b->h_logits, b->logits, (size_t)c->n_vocab * sizeof(float), cudaMemcpyDeviceToHost, st));
    }
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    if (want_last) memcpy(m->logits, b->h_logits, (size_t)c->n_vocab * sizeof(float));
    m->n_past = p0 + N;
    if (G->nvs) pool_promote(m, G);
    return 0;
}

/* n tokens through the model; logits of the last one in m->logits. Chunks of the batch size. */
extern "C" int q38_gpu_forward_batch(void *model, const int *toks, int n)
{
    q38_model *m = (q38_model *)model;
    if (!GB && bat_open(m, 0)) return -1;
    G->warm_want = 1;
    for (int i = 0; i < n; i += GB->nb) {
        const int k = n - i < GB->nb ? n - i : GB->nb;
        const double t0 = now_s(), r0 = m->t_read;
        const int64_t b0 = m->expert_bytes_read;
        if (forward_batch_chunk(m, toks + i, k, i + k == n)) return -1;
        if (getenv("Q38_BATCH_VERBOSE"))
            fprintf(stderr, "q38: batch of %d tokens at %d: %.1f s, %.2f GB read in %.1f s\n", k, m->n_past - k,
                    now_s() - t0, (m->expert_bytes_read - b0) / 1e9, m->t_read - r0);
    }
    return 0;
}
