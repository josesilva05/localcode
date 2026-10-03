/* q38_kernels.c - see q38_kernels.h. AVX2 + FMA + F16C, with a portable fallback that
 * dequantises the row (q38_dequant_row) and sums in float. */
#include "q38_kernels.h"
#include "q38_gguf.h"
#include "q38_quant.h"

#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define Q38_AVX2 1
#endif

void q38_xprep(q38_xvec *v, const float *x, int64_t n, float *bsum)
{
    v->x = x;
    v->n = n;
    v->bsum = bsum;
    for (int64_t b = 0; b < n / 32; b++) {
        float s = 0;
        for (int i = 0; i < 32; i++) s += x[b * 32 + i];
        bsum[b] = s;
    }
}

int q38_dot_supported(int type)
{
    return q38_quant_supported(type);
}

/* ------------------------------------------------------------------ fallback -- */
static float dot_generic(int type, const uint8_t *row, const q38_xvec *v)
{
    float buf[16384];
    if (v->n > 16384) return 0.0f;
    q38_dequant_row(type, row, buf, v->n);
    float s = 0;
    for (int64_t i = 0; i < v->n; i++) s += buf[i] * v->x[i];
    return s;
}

#ifdef Q38_AVX2
static inline float hsum8(__m256 a)
{
    __m128 lo = _mm256_castps256_ps128(a), hi = _mm256_extractf128_ps(a, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    s = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

static inline float h2f(const uint8_t *p)
{
    uint16_t h;
    memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

/* 8 bytes (unsigned) -> 8 floats */
static inline __m256 u8x8(const uint8_t *p)
{
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p)));
}

/* sum of q[0..32) * x[0..32), q as 32 unsigned bytes in a register */
static inline __m256 dot32_u8(__m256i q, const float *x)
{
    __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
    __m256 a = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), _mm256_loadu_ps(x));
    a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))), _mm256_loadu_ps(x + 8), a);
    a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), _mm256_loadu_ps(x + 16), a);
    a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))), _mm256_loadu_ps(x + 24), a);
    return a;
}

static float dot_f32(const uint8_t *row, const q38_xvec *v)
{
    const float *w = (const float *)row;
    __m256 a = _mm256_setzero_ps(), b = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= v->n; i += 16) {
        a = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(v->x + i), a);
        b = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(v->x + i + 8), b);
    }
    float s = hsum8(_mm256_add_ps(a, b));
    for (; i < v->n; i++) s += w[i] * v->x[i];
    return s;
}

static float dot_bf16(const uint8_t *row, const q38_xvec *v)
{
    const uint16_t *w = (const uint16_t *)row;
    __m256 a = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= v->n; i += 8) {
        __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + i))), 16);
        a = _mm256_fmadd_ps(_mm256_castsi256_ps(u), _mm256_loadu_ps(v->x + i), a);
    }
    float s = hsum8(a);
    for (; i < v->n; i++) s += q38_bf16_to_fp32(w[i]) * v->x[i];
    return s;
}

/* Q8_0: fp16 d, int8 qs[32] */
static float dot_q8_0(const uint8_t *row, const q38_xvec *v)
{
    const int64_t nb = v->n / 32;
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++, row += 34) {
        const float *x = v->x + b * 32;
        const __m128i q0 = _mm_loadu_si128((const __m128i *)(row + 2)), q1 = _mm_loadu_si128((const __m128i *)(row + 18));
        __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)), _mm256_loadu_ps(x));
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8))), _mm256_loadu_ps(x + 8), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)), _mm256_loadu_ps(x + 16), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8))), _mm256_loadu_ps(x + 24), s);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(row)), s, acc);
    }
    return hsum8(acc);
}

/* 32 bits -> 32 bytes, 0xFF where the bit is set (byte i <-> bit i) */
static inline __m256i bits_to_bytes(uint32_t x)
{
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0);
    __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32((int)x), shuf);
    b = _mm256_or_si256(b, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL));
    return _mm256_cmpeq_epi8(b, _mm256_set1_epi64x(-1));
}

/* Q5_1: fp16 d, fp16 m, u8 qh[4], u8 qs[16]; y = d*q + m */
static float dot_q5_1(const uint8_t *row, const q38_xvec *v)
{
    const int64_t nb = v->n / 32;
    __m256 acc = _mm256_setzero_ps();
    float macc = 0;
    const __m256i m4 = _mm256_set1_epi8(0x0F), m16 = _mm256_set1_epi8(0x10);
    for (int64_t b = 0; b < nb; b++, row += 24) {
        uint32_t qh;
        memcpy(&qh, row + 4, 4);
        const __m128i qs = _mm_loadu_si128((const __m128i *)(row + 8));
        /* bytes 0..15 = low nibbles (elements 0..15), 16..31 = high nibbles (16..31) */
        __m256i q = _mm256_set_m128i(_mm_srli_epi16(qs, 4), qs);
        q = _mm256_and_si256(q, m4);
        q = _mm256_or_si256(q, _mm256_and_si256(bits_to_bytes(qh), m16));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(row)), dot32_u8(q, v->x + b * 32), acc);
        macc += h2f(row + 2) * v->bsum[b];
    }
    return hsum8(acc) + macc;
}

static inline void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

/* Q4_K: fp16 d, fp16 dmin, scales[12], qs[128]; y = d*sc*q - dmin*m per 32 */
static float dot_q4_K(const uint8_t *row, const q38_xvec *v)
{
    const int64_t nb = v->n / 256;
    __m256 acc = _mm256_setzero_ps();
    float macc = 0;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (int64_t b = 0; b < nb; b++, row += 144) {
        const float d = h2f(row), dmin = h2f(row + 2);
        const uint8_t *sc = row + 4, *qs = row + 16;
        const float *x = v->x + b * 256;
        const float *bs = v->bsum + b * 8;
        for (int j = 0; j < 4; j++) {
            uint8_t s1, m1, s2, m2;
            scale_min_k4(2 * j, sc, &s1, &m1);
            scale_min_k4(2 * j + 1, sc, &s2, &m2);
            const __m256i q = _mm256_loadu_si256((const __m256i *)(qs + 32 * j));
            const __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * s1), dot32_u8(lo, x + 64 * j), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * s2), dot32_u8(hi, x + 64 * j + 32), acc);
            macc += dmin * m1 * bs[2 * j] + dmin * m2 * bs[2 * j + 1];
        }
    }
    return hsum8(acc) - macc;
}

/* Q5_K: fp16 d, fp16 dmin, scales[12], qh[32], qs[128] */
static float dot_q5_K(const uint8_t *row, const q38_xvec *v)
{
    const int64_t nb = v->n / 256;
    __m256 acc = _mm256_setzero_ps();
    float macc = 0;
    const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
    for (int64_t b = 0; b < nb; b++, row += 176) {
        const float d = h2f(row), dmin = h2f(row + 2);
        const uint8_t *sc = row + 4, *qhp = row + 16, *qs = row + 48;
        const __m256i qh = _mm256_loadu_si256((const __m256i *)qhp);
        const float *x = v->x + b * 256;
        const float *bs = v->bsum + b * 8;
        for (int j = 0; j < 4; j++) {
            uint8_t s1, m1, s2, m2;
            scale_min_k4(2 * j, sc, &s1, &m1);
            scale_min_k4(2 * j + 1, sc, &s2, &m2);
            const __m256i q = _mm256_loadu_si256((const __m256i *)(qs + 32 * j));
            __m256i h1 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * j), one);
            __m256i h2 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * j + 1), one);
            const __m256i lo = _mm256_or_si256(_mm256_and_si256(q, m4), _mm256_slli_epi16(h1, 4));
            const __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4), _mm256_slli_epi16(h2, 4));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * s1), dot32_u8(lo, x + 64 * j), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * s2), dot32_u8(hi, x + 64 * j + 32), acc);
            macc += dmin * m1 * bs[2 * j] + dmin * m2 * bs[2 * j + 1];
        }
    }
    return hsum8(acc) - macc;
}
#endif /* Q38_AVX2 */

float q38_dot(int type, const uint8_t *row, const q38_xvec *v)
{
#ifdef Q38_AVX2
    switch (type) {
    case Q38_F32:  return dot_f32(row, v);
    case Q38_BF16: return dot_bf16(row, v);
    case Q38_Q8_0: return dot_q8_0(row, v);
    case Q38_Q5_1: return dot_q5_1(row, v);
    case Q38_Q4_K: return dot_q4_K(row, v);
    case Q38_Q5_K: return dot_q5_K(row, v);
    default: break;
    }
#endif
    return dot_generic(type, row, v);
}

void q38_matvec_rows(int type, const uint8_t *w, int64_t row_bytes, int64_t n_rows, const q38_xvec *v, float *y)
{
    for (int64_t r = 0; r < n_rows; r++) y[r] = q38_dot(type, w + r * row_bytes, v);
}
