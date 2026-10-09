/* q38_gpu_mvq.cuh - y = W x with x in 8 bits and integer dot products (dp4a), for matrices kept in
 * their GGUF types (included by q38_gpu.cu before mv).
 *
 * llama.cpp's mmvq: x is quantised once into blocks of 32 (int8, a scale, the scaled sum of the
 * block), each weight unit is read 16-32 bytes at a time and multiplied 4 int8 pairs an
 * instruction. The float kernels (k_mv_native, k_mv_rows: every weight converted and multiplied
 * in fp32) ran the dense 27B's trunk at ~150 GB/s; this one is bandwidth-bound. Its error is
 * llama.cpp's (8-bit activations); Q38_ACT8=0 keeps the float kernels, exact against the
 * reference. Units: Q4_K/Q5_K 64 elements (one 32-byte run of nibbles), Q6_K/IQ4_XS/IQ4_NL 32. */

/* x in blocks of 32: q (int8), d (scale), s = d * sum(q); a warp per block */
__global__ void k_q8x(const float *__restrict__ x, int n, int8_t *__restrict__ q, float *__restrict__ d, float *__restrict__ s)
{
    const int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= n) return;
    const float v = x[b * 32 + lane];
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float sc = a / 127.0f, inv = a > 0.0f ? 127.0f / a : 0.0f;
    const int qi = __float2int_rn(v * inv);
    q[b * 32 + lane] = (int8_t)qi;
    int sum = qi;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) { d[b] = sc; s[b] = sc * (float)sum; }
}

__device__ __forceinline__ int ld_i32(const uint8_t *p) { return *(const int *)(const void *)p; }
/* 4 bytes at a 2-byte boundary (Q6_K's 210-byte blocks) */
__device__ __forceinline__ int ld_i32_a2(const uint8_t *p)
{
    const uint16_t *h = (const uint16_t *)(const void *)p;
    return (int)((uint32_t)h[0] | ((uint32_t)h[1] << 16));
}

/* the 6-bit scale and min of sub-block i (0..7) of a K-quant super-block (get_scale_min_k4) */
__device__ __forceinline__ void k_scale_min(const uint8_t *sc, int i, int *s, int *m)
{
    if (i < 4) { *s = sc[i] & 63; *m = sc[i + 4] & 63; }
    else { *s = (sc[i + 4] & 0xF) | ((sc[i - 4] >> 6) << 4); *m = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4); }
}

/* Q4_K / Q5_K unit u: super-block u / 4, quarter j = u % 4 (elements 64j..64j+63): the 32 bytes of
 * nibbles hold 64j+l (low) and 64j+32+l (high); Q5_K's qh adds bit 2j / 2j+1 of qh[l] */
template <int FIVE>
__device__ __forceinline__ float dot_q45k(const uint8_t *row, int u, const int8_t *xq, const float *xd, const float *xs)
{
    const int sb = u >> 2, j = u & 3;
    const uint8_t *blk = row + (int64_t)sb * (FIVE ? 176 : 144);
    const float2 dm = __half22float2(*(const __half2 *)(const void *)blk);
    int s1, m1, s2, m2;
    k_scale_min(blk + 4, 2 * j, &s1, &m1);
    k_scale_min(blk + 4, 2 * j + 1, &s2, &m2);
    const uint8_t *qs = blk + (FIVE ? 48 : 16) + 32 * j;
    const int b0 = sb * 8 + 2 * j, b1 = b0 + 1;
    /* 16-byte loads: the nibbles, Q5_K's high bits, the two blocks of x (all 16-byte aligned) */
    int q[8], h[8], xa[8], xb[8];
    *(int4 *)&q[0] = *(const int4 *)(const void *)qs; *(int4 *)&q[4] = *(const int4 *)(const void *)(qs + 16);
    if (FIVE) { *(int4 *)&h[0] = *(const int4 *)(const void *)(blk + 16); *(int4 *)&h[4] = *(const int4 *)(const void *)(blk + 32); }
    *(int4 *)&xa[0] = *(const int4 *)(xq + b0 * 32); *(int4 *)&xa[4] = *(const int4 *)(xq + b0 * 32 + 16);
    *(int4 *)&xb[0] = *(const int4 *)(xq + b1 * 32); *(int4 *)&xb[4] = *(const int4 *)(xq + b1 * 32 + 16);
    int lo = 0, hi = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int ql = q[i] & 0x0F0F0F0F, qh2 = (q[i] >> 4) & 0x0F0F0F0F;
        if (FIVE) {
            ql |= ((h[i] >> (2 * j)) & 0x01010101) << 4;
            qh2 |= ((h[i] >> (2 * j + 1)) & 0x01010101) << 4;
        }
        lo = __dp4a(ql, xa[i], lo);
        hi = __dp4a(qh2, xb[i], hi);
    }
    return dm.x * ((float)s1 * xd[b0] * (float)lo + (float)s2 * xd[b1] * (float)hi) -
           dm.y * ((float)m1 * xs[b0] + (float)m2 * xs[b1]);
}

/* Q6_K unit u: super-block u / 8, k = u % 8: half n = k / 4, quarter qq = k % 4: elements 128n+32qq+l,
 * l < 32, from ql[64n + 32(qq&1) + l] (low or high nibble) and 2 bits of qh[32n + l] at 2qq, minus 32;
 * scales sc[8n + 2qq] for l < 16, sc[8n + 2qq + 1] after */
__device__ __forceinline__ float dot_q6k(const uint8_t *row, int u, const int8_t *xq, const float *xd)
{
    const int sb = u >> 3, k = u & 7, n = k >> 2, qq = k & 3;
    const uint8_t *blk = row + (int64_t)sb * 210;
    const uint8_t *ql = blk + 64 * n + 32 * (qq & 1), *qh = blk + 128 + 32 * n;
    const int8_t *sc = (const int8_t *)(blk + 192) + 8 * n + 2 * qq;
    const float d = __half2float(*(const __half *)(const void *)(blk + 208));
    const int b = sb * 8 + 4 * n + qq;
    /* 32 bytes at a 2-byte boundary: 9 aligned words, shifted into place */
    const int sh = (int)((uintptr_t)ql & 3) * 8;          /* ql and qh share the block's alignment */
    const int *qa = (const int *)(const void *)((uintptr_t)ql & ~(uintptr_t)3);
    const int *ha = (const int *)(const void *)((uintptr_t)qh & ~(uintptr_t)3);
    int wl[9], wh[9], xv8[8];
#pragma unroll
    for (int i = 0; i < 9; i++) { wl[i] = i < 8 || sh ? qa[i] : 0; wh[i] = i < 8 || sh ? ha[i] : 0; }
    *(int4 *)&xv8[0] = *(const int4 *)(xq + b * 32); *(int4 *)&xv8[4] = *(const int4 *)(xq + b * 32 + 16);
    int dot[2] = { 0, 0 }, sx[2] = { 0, 0 };
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int lw = (int)__funnelshift_r((unsigned)wl[i], (unsigned)wl[i + 1], sh);
        const int hw = (int)__funnelshift_r((unsigned)wh[i], (unsigned)wh[i + 1], sh);
        const int v = ((lw >> (4 * (qq >> 1))) & 0x0F0F0F0F) | (((hw >> (2 * qq)) & 0x03030303) << 4);
        const int xv = xv8[i];
        dot[i >> 2] = __dp4a(v, xv, dot[i >> 2]);
        sx[i >> 2] = __dp4a(0x01010101, xv, sx[i >> 2]);
    }
    return d * xd[b] * ((float)sc[0] * (float)(dot[0] - 32 * sx[0]) + (float)sc[1] * (float)(dot[1] - 32 * sx[1]));
}

/* 4 nibbles (one byte each, in bits 0-3 of each byte of q) through the IQ4 table, packed */
__device__ __forceinline__ int iq4_lut(int q)
{
    const int8_t *t = d_kv_iq4nl;
    return (int)(uint8_t)t[q & 15] | ((int)(uint8_t)t[(q >> 8) & 15] << 8) | ((int)(uint8_t)t[(q >> 16) & 15] << 16) |
           ((int)(uint8_t)t[(q >> 24) & 15] << 24);
}

/* IQ4_XS unit u: super-block u / 8, sub-block j = u % 8: qs[16j..16j+15], low nibbles elements 32j+k,
 * high 32j+16+k, scale d * (ls - 32) */
__device__ __forceinline__ float dot_iq4xs(const uint8_t *row, int u, const int8_t *xq, const float *xd)
{
    const int sb = u >> 3, j = u & 7;
    const uint8_t *blk = row + (int64_t)sb * 136;
    const float d = __half2float(*(const __half *)(const void *)blk);
    const int sh = blk[2] | (blk[3] << 8);
    const int ls = ((blk[4 + j / 2] >> 4 * (j % 2)) & 0xf) | (((sh >> 2 * j) & 3) << 4);
    const uint8_t *qs = blk + 8 + 16 * j;
    const int b = sb * 8 + j;
    const int8_t *x = xq + b * 32;
    int dot = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int q = ld_i32(qs + 4 * i);
        dot = __dp4a(iq4_lut(q), ld_i32((const uint8_t *)(x + 4 * i)), dot);
        dot = __dp4a(iq4_lut(q >> 4), ld_i32((const uint8_t *)(x + 16 + 4 * i)), dot);
    }
    return d * (float)(ls - 32) * xd[b] * (float)dot;
}

/* IQ4_NL unit u: block u (18 bytes: d, 16 bytes of nibbles as IQ4_XS's sub-block) */
__device__ __forceinline__ float dot_iq4nl(const uint8_t *row, int u, const int8_t *xq, const float *xd)
{
    const uint8_t *blk = row + (int64_t)u * 18;
    const float d = __half2float(__ushort_as_half((unsigned short)(blk[0] | (blk[1] << 8))));
    const int8_t *x = xq + u * 32;
    int dot = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int q = blk[2 + 4 * i] | (blk[3 + 4 * i] << 8) | (blk[4 + 4 * i] << 16) | (blk[5 + 4 * i] << 24);   /* 2-byte aligned */
        dot = __dp4a(iq4_lut(q), ld_i32((const uint8_t *)(x + 4 * i)), dot);
        dot = __dp4a(iq4_lut(q >> 4), ld_i32((const uint8_t *)(x + 16 + 4 * i)), dot);
    }
    return d * xd[u] * (float)dot;
}

/* a warp per row, lane l takes units l, l+32, ... */
template <int TYPE>
__global__ void __launch_bounds__(128) k_mvq(const uint8_t *__restrict__ W, int64_t rb, int ne0, int rows,
                                             const int8_t *__restrict__ xq, const float *__restrict__ xd,
                                             const float *__restrict__ xs, float *__restrict__ y)
{
    const int r = blockIdx.x * 4 + threadIdx.y;
    if (r >= rows) return;
    const uint8_t *row = W + (int64_t)r * rb;
    const int units = TYPE == Q38_Q4_K || TYPE == Q38_Q5_K ? ne0 / 64 : ne0 / 32;
    float acc = 0.0f;
    for (int u = threadIdx.x; u < units; u += 32) {
        if (TYPE == Q38_Q4_K) acc += dot_q45k<0>(row, u, xq, xd, xs);
        else if (TYPE == Q38_Q5_K) acc += dot_q45k<1>(row, u, xq, xd, xs);
        else if (TYPE == Q38_Q6_K) acc += dot_q6k(row, u, xq, xd);
        else if (TYPE == Q38_IQ4_XS) acc += dot_iq4xs(row, u, xq, xd);
        else acc += dot_iq4nl(row, u, xq, xd);
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[r] = acc;
}

/* NC columns of one unit, the weights decoded once (into 8-bit lanes and their scales) and then
 * multiplied with each column's x: decoding again for every column made 4 columns cost 1.6x (Q5_K)
 * to 2.5x (IQ4_XS) one. xq/xd/xs: NC rows of ne0 (nb = ne0 / 32 scales a row). */
template <int FIVE, int NC>
__device__ __forceinline__ void unit_q45k(const uint8_t *row, int u, const int8_t *xq, const float *xd, const float *xs,
                                          int ne0, float *acc)
{
    const int sb = u >> 2, j = u & 3, nb = ne0 / 32;
    const uint8_t *blk = row + (int64_t)sb * (FIVE ? 176 : 144);
    const float2 dm = __half22float2(*(const __half2 *)(const void *)blk);
    int s1, m1, s2, m2;
    k_scale_min(blk + 4, 2 * j, &s1, &m1);
    k_scale_min(blk + 4, 2 * j + 1, &s2, &m2);
    const uint8_t *qs = blk + (FIVE ? 48 : 16) + 32 * j;
    int q[8], h[8], lo4[8], hi4[8];
    *(int4 *)&q[0] = *(const int4 *)(const void *)qs; *(int4 *)&q[4] = *(const int4 *)(const void *)(qs + 16);
    if (FIVE) { *(int4 *)&h[0] = *(const int4 *)(const void *)(blk + 16); *(int4 *)&h[4] = *(const int4 *)(const void *)(blk + 32); }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        lo4[i] = q[i] & 0x0F0F0F0F;
        hi4[i] = (q[i] >> 4) & 0x0F0F0F0F;
        if (FIVE) {
            lo4[i] |= ((h[i] >> (2 * j)) & 0x01010101) << 4;
            hi4[i] |= ((h[i] >> (2 * j + 1)) & 0x01010101) << 4;
        }
    }
    const int b0 = sb * 8 + 2 * j, b1 = b0 + 1;
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const int8_t *x = xq + (int64_t)c * ne0;
        int xa[8], xb[8];
        *(int4 *)&xa[0] = *(const int4 *)(x + b0 * 32); *(int4 *)&xa[4] = *(const int4 *)(x + b0 * 32 + 16);
        *(int4 *)&xb[0] = *(const int4 *)(x + b1 * 32); *(int4 *)&xb[4] = *(const int4 *)(x + b1 * 32 + 16);
        int lo = 0, hi = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) { lo = __dp4a(lo4[i], xa[i], lo); hi = __dp4a(hi4[i], xb[i], hi); }
        const float *d = xd + c * nb, *s = xs + c * nb;
        acc[c] += dm.x * ((float)s1 * d[b0] * (float)lo + (float)s2 * d[b1] * (float)hi) - dm.y * ((float)m1 * s[b0] + (float)m2 * s[b1]);
    }
}

template <int NC>
__device__ __forceinline__ void unit_q6k(const uint8_t *row, int u, const int8_t *xq, const float *xd, int ne0, float *acc)
{
    const int sb = u >> 3, k = u & 7, n = k >> 2, qq = k & 3, nb = ne0 / 32;
    const uint8_t *blk = row + (int64_t)sb * 210;
    const uint8_t *ql = blk + 64 * n + 32 * (qq & 1), *qh = blk + 128 + 32 * n;
    const int8_t *sc = (const int8_t *)(blk + 192) + 8 * n + 2 * qq;
    const float d = __half2float(*(const __half *)(const void *)(blk + 208));
    const int b = sb * 8 + 4 * n + qq;
    const int sh = (int)((uintptr_t)ql & 3) * 8;
    const int *qa = (const int *)(const void *)((uintptr_t)ql & ~(uintptr_t)3);
    const int *ha = (const int *)(const void *)((uintptr_t)qh & ~(uintptr_t)3);
    int wl[9], wh[9], v[8];
#pragma unroll
    for (int i = 0; i < 9; i++) { wl[i] = i < 8 || sh ? qa[i] : 0; wh[i] = i < 8 || sh ? ha[i] : 0; }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int lw = (int)__funnelshift_r((unsigned)wl[i], (unsigned)wl[i + 1], sh);
        const int hw = (int)__funnelshift_r((unsigned)wh[i], (unsigned)wh[i + 1], sh);
        v[i] = ((lw >> (4 * (qq >> 1))) & 0x0F0F0F0F) | (((hw >> (2 * qq)) & 0x03030303) << 4);
    }
    const float s0 = (float)sc[0], s1 = (float)sc[1];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const int8_t *x = xq + (int64_t)c * ne0 + b * 32;
        int xv[8];
        *(int4 *)&xv[0] = *(const int4 *)x; *(int4 *)&xv[4] = *(const int4 *)(x + 16);
        int dot0 = 0, dot1 = 0, sx0 = 0, sx1 = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            dot0 = __dp4a(v[i], xv[i], dot0); sx0 = __dp4a(0x01010101, xv[i], sx0);
            dot1 = __dp4a(v[4 + i], xv[4 + i], dot1); sx1 = __dp4a(0x01010101, xv[4 + i], sx1);
        }
        acc[c] += d * xd[c * nb + b] * (s0 * (float)(dot0 - 32 * sx0) + s1 * (float)(dot1 - 32 * sx1));
    }
}

/* IQ4_XS (sub-block u of 32) and IQ4_NL (block u): the table looked up once */
template <int XS, int NC>
__device__ __forceinline__ void unit_iq4(const uint8_t *row, int u, const int8_t *xq, const float *xd, int ne0, float *acc)
{
    const int nb = ne0 / 32;
    float scale;
    int lo[4], hi[4];
    if (XS) {
        const int sb = u >> 3, j = u & 7;
        const uint8_t *blk = row + (int64_t)sb * 136;
        const float d = __half2float(*(const __half *)(const void *)blk);
        const int sh = blk[2] | (blk[3] << 8);
        const int ls = ((blk[4 + j / 2] >> 4 * (j % 2)) & 0xf) | (((sh >> 2 * j) & 3) << 4);
        scale = d * (float)(ls - 32);
        const uint8_t *qs = blk + 8 + 16 * j;
#pragma unroll
        for (int i = 0; i < 4; i++) { const int q = ld_i32(qs + 4 * i); lo[i] = iq4_lut(q); hi[i] = iq4_lut(q >> 4); }
    } else {
        const uint8_t *blk = row + (int64_t)u * 18;
        scale = __half2float(__ushort_as_half((unsigned short)(blk[0] | (blk[1] << 8))));
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int q = blk[2 + 4 * i] | (blk[3 + 4 * i] << 8) | (blk[4 + 4 * i] << 16) | (blk[5 + 4 * i] << 24);
            lo[i] = iq4_lut(q);
            hi[i] = iq4_lut(q >> 4);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const int8_t *x = xq + (int64_t)c * ne0 + u * 32;
        int xv[8];
        *(int4 *)&xv[0] = *(const int4 *)x; *(int4 *)&xv[4] = *(const int4 *)(x + 16);
        int dot = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) { dot = __dp4a(lo[i], xv[i], dot); dot = __dp4a(hi[i], xv[4 + i], dot); }
        acc[c] += scale * xd[c * nb + u] * (float)dot;
    }
}

/* NC vectors at once (a few tokens: speculative verification): each weight unit read and decoded
 * once for all of them; x: NC rows of ne0 quantised */
template <int TYPE, int NC>
__global__ void __launch_bounds__(128) k_mvq_nc(const uint8_t *__restrict__ W, int64_t rb, int ne0, int rows,
                                                const int8_t *__restrict__ xq, const float *__restrict__ xd,
                                                const float *__restrict__ xs, float *__restrict__ y, int ldy)
{
    const int r = blockIdx.x * 4 + threadIdx.y;
    if (r >= rows) return;
    const uint8_t *row = W + (int64_t)r * rb;
    const int units = TYPE == Q38_Q4_K || TYPE == Q38_Q5_K ? ne0 / 64 : ne0 / 32;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) acc[c] = 0.0f;
    for (int u = threadIdx.x; u < units; u += 32) {
        if (TYPE == Q38_Q4_K) unit_q45k<0, NC>(row, u, xq, xd, xs, ne0, acc);
        else if (TYPE == Q38_Q5_K) unit_q45k<1, NC>(row, u, xq, xd, xs, ne0, acc);
        else if (TYPE == Q38_Q6_K) unit_q6k<NC>(row, u, xq, xd, ne0, acc);
        else if (TYPE == Q38_IQ4_XS) unit_iq4<1, NC>(row, u, xq, xd, ne0, acc);
        else unit_iq4<0, NC>(row, u, xq, xd, ne0, acc);
    }
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const float v = warp_sum(acc[c]);
        if (threadIdx.x == 0) y[(int64_t)c * ldy + r] = v;
    }
}

/* A Q4_K/Q5_K unit's raw bytes, loaded one unit ahead of its arithmetic (the loop below): with 4
 * columns of work between two loads, a warp kept too few reads in flight and 4 columns cost 1.6x one */
struct Q45Raw { int4 q0, q1, h0, h1; uint32_t dm, w0, w1, w2; };

template <int FIVE>
__device__ __forceinline__ void q45_load(const uint8_t *row, int u, Q45Raw &w)
{
    const uint8_t *blk = row + (int64_t)(u >> 2) * (FIVE ? 176 : 144);
    const uint32_t *hw = (const uint32_t *)(const void *)blk;
    w.dm = hw[0]; w.w0 = hw[1]; w.w1 = hw[2]; w.w2 = hw[3];
    const uint8_t *qs = blk + (FIVE ? 48 : 16) + 32 * (u & 3);
    w.q0 = *(const int4 *)(const void *)qs;
    w.q1 = *(const int4 *)(const void *)(qs + 16);
    if (FIVE) { w.h0 = *(const int4 *)(const void *)(blk + 16); w.h1 = *(const int4 *)(const void *)(blk + 32); }
}

template <int FIVE, int NC>
__device__ __forceinline__ void q45_math(const Q45Raw &w, int u, const int8_t *xq, const float *xd, const float *xs, int ne0,
                                         float *acc)
{
    const int sb = u >> 2, j = u & 3, nb = ne0 / 32;
    const float2 dm = __half22float2(*(const __half2 *)&w.dm);
    int s1, m1, s2, m2;
    {
        const int i1 = 2 * j, i2 = 2 * j + 1;
        if (i1 < 4) { s1 = scb(w.w0, w.w1, w.w2, i1) & 63; m1 = scb(w.w0, w.w1, w.w2, i1 + 4) & 63; }
        else { s1 = (scb(w.w0, w.w1, w.w2, i1 + 4) & 0xF) | ((scb(w.w0, w.w1, w.w2, i1 - 4) >> 6) << 4);
               m1 = (scb(w.w0, w.w1, w.w2, i1 + 4) >> 4) | ((scb(w.w0, w.w1, w.w2, i1) >> 6) << 4); }
        if (i2 < 4) { s2 = scb(w.w0, w.w1, w.w2, i2) & 63; m2 = scb(w.w0, w.w1, w.w2, i2 + 4) & 63; }
        else { s2 = (scb(w.w0, w.w1, w.w2, i2 + 4) & 0xF) | ((scb(w.w0, w.w1, w.w2, i2 - 4) >> 6) << 4);
               m2 = (scb(w.w0, w.w1, w.w2, i2 + 4) >> 4) | ((scb(w.w0, w.w1, w.w2, i2) >> 6) << 4); }
    }
    const int q[8] = { w.q0.x, w.q0.y, w.q0.z, w.q0.w, w.q1.x, w.q1.y, w.q1.z, w.q1.w };
    const int h[8] = { w.h0.x, w.h0.y, w.h0.z, w.h0.w, w.h1.x, w.h1.y, w.h1.z, w.h1.w };
    int lo4[8], hi4[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        lo4[i] = q[i] & 0x0F0F0F0F;
        hi4[i] = (q[i] >> 4) & 0x0F0F0F0F;
        if (FIVE) {
            lo4[i] |= ((h[i] >> (2 * j)) & 0x01010101) << 4;
            hi4[i] |= ((h[i] >> (2 * j + 1)) & 0x01010101) << 4;
        }
    }
    const int b0 = sb * 8 + 2 * j, b1 = b0 + 1;
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const int8_t *x = xq + (int64_t)c * ne0;
        int xa[8], xb[8];
        *(int4 *)&xa[0] = *(const int4 *)(x + b0 * 32); *(int4 *)&xa[4] = *(const int4 *)(x + b0 * 32 + 16);
        *(int4 *)&xb[0] = *(const int4 *)(x + b1 * 32); *(int4 *)&xb[4] = *(const int4 *)(x + b1 * 32 + 16);
        int lo = 0, hi = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) { lo = __dp4a(lo4[i], xa[i], lo); hi = __dp4a(hi4[i], xb[i], hi); }
        const float *d = xd + c * nb, *s = xs + c * nb;
        acc[c] += dm.x * ((float)s1 * d[b0] * (float)lo + (float)s2 * d[b1] * (float)hi) - dm.y * ((float)m1 * s[b0] + (float)m2 * s[b1]);
    }
}

/* The same with the NC quantised vectors in shared memory, loaded once per block of 32 rows (8 warps,
 * 4 rows each): from L1/L2 for every unit, x (4 x 64 bytes against Q5_K's 80 of weights) made 4
 * columns cost 1.5x one. Shared: NC * ne0 bytes, then the scales and sums (NC * ne0 / 32 floats each). */
template <int TYPE, int NC>
__global__ void __launch_bounds__(256) k_mvq_ncs(const uint8_t *__restrict__ W, int64_t rb, int ne0, int rows,
                                                 const int8_t *__restrict__ xq, const float *__restrict__ xd,
                                                 const float *__restrict__ xs, float *__restrict__ y, int ldy)
{
    extern __shared__ __align__(16) unsigned char sx[];
    const int nb = ne0 / 32, tid = threadIdx.y * 32 + threadIdx.x;
    int8_t *q = (int8_t *)sx;
    float *d = (float *)(sx + NC * ne0), *s = d + NC * nb;
    for (int i = tid * 16; i < NC * ne0; i += 256 * 16) *(int4 *)(q + i) = *(const int4 *)(xq + i);
    for (int i = tid; i < NC * nb; i += 256) { d[i] = xd[i]; s[i] = xs[i]; }
    __syncthreads();
    const int units = TYPE == Q38_Q4_K || TYPE == Q38_Q5_K ? ne0 / 64 : ne0 / 32;
#pragma unroll 1
    for (int k = 0; k < 4; k++) {
        const int r = blockIdx.x * 32 + threadIdx.y * 4 + k;
        if (r >= rows) return;
        const uint8_t *row = W + (int64_t)r * rb;
        float acc[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) acc[c] = 0.0f;
        if (TYPE == Q38_Q4_K || TYPE == Q38_Q5_K) {      /* one unit's bytes ahead of its arithmetic */
            Q45Raw cur, nxt;
            int u = threadIdx.x;
            if (u < units) q45_load<TYPE == Q38_Q5_K>(row, u, cur);
            for (; u < units; u += 32) {
                if (u + 32 < units) q45_load<TYPE == Q38_Q5_K>(row, u + 32, nxt);
                q45_math<TYPE == Q38_Q5_K, NC>(cur, u, q, d, s, ne0, acc);
                cur = nxt;
            }
        } else {
            for (int u = threadIdx.x; u < units; u += 32) {
                if (TYPE == Q38_Q6_K) unit_q6k<NC>(row, u, q, d, ne0, acc);
                else if (TYPE == Q38_IQ4_XS) unit_iq4<1, NC>(row, u, q, d, ne0, acc);
                else unit_iq4<0, NC>(row, u, q, d, ne0, acc);
            }
        }
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const float v = warp_sum(acc[c]);
            if (threadIdx.x == 0) y[(int64_t)c * ldy + r] = v;
        }
    }
}

/* NC rows of x (stride ldx) into blocks of 32: as k_q8x, the row c's blocks after row c-1's */
__global__ void k_q8x_nc(const float *__restrict__ x, int ldx, int n, int NC, int8_t *__restrict__ q, float *__restrict__ d,
                         float *__restrict__ s)
{
    const int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, nb = n / 32;
    if (b >= nb * NC) return;
    const int c = b / nb, j = b % nb;
    const float v = x[(int64_t)c * ldx + j * 32 + lane];
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float inv = a > 0.0f ? 127.0f / a : 0.0f;
    const int qi = __float2int_rn(v * inv);
    q[(int64_t)c * n + j * 32 + lane] = (int8_t)qi;
    int sum = qi;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) { d[c * nb + j] = a / 127.0f; s[c * nb + j] = a / 127.0f * (float)sum; }
}

static int g_mvq = 0;               /* set by the dense models' open: their trunk goes through k_mvq */

static int mvq_ok(int type)
{
    return g_mvq && (type == Q38_Q4_K || type == Q38_Q5_K || type == Q38_Q6_K || type == Q38_IQ4_XS || type == Q38_IQ4_NL);
}
