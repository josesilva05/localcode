/* q38_quant.c - see q38_quant.h. Each dequant_* mirrors the ggml function of the same
 * name; keep the arithmetic order identical (it decides the last bit). */
#include "q38_quant.h"
#include "q38_gguf.h"

#include <string.h>

float q38_fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                       /* subnormal: normalise */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

float q38_bf16_to_fp32(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
#define H(p) q38_fp16_to_fp32(rd16(p))

/* Q8_0: fp16 d, int8 qs[32]. */
static void dequant_q8_0(const uint8_t *x, float *y, int64_t nb)
{
    for (int64_t i = 0; i < nb; i++, x += 34, y += 32) {
        const float d = H(x);
        const int8_t *qs = (const int8_t *)(x + 2);
        for (int j = 0; j < 32; j++) y[j] = qs[j] * d;
    }
}

/* Q5_1: fp16 d, fp16 m, u8 qh[4], u8 qs[16]. */
static void dequant_q5_1(const uint8_t *x, float *y, int64_t nb)
{
    for (int64_t i = 0; i < nb; i++, x += 24, y += 32) {
        const float d = H(x), m = H(x + 2);
        uint32_t qh;
        memcpy(&qh, x + 4, 4);
        const uint8_t *qs = x + 8;
        for (int j = 0; j < 16; j++) {
            const uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            const uint8_t xh_1 = (uint8_t)(((qh >> (j + 12))) & 0x10);
            const int x0 = (qs[j] & 0x0F) | xh_0;
            const int x1 = (qs[j] >> 4) | xh_1;
            y[j + 0] = x0 * d + m;
            y[j + 16] = x1 * d + m;
        }
    }
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

/* Q4_K: fp16 d, fp16 dmin, u8 scales[12], u8 qs[128]. */
static void dequant_q4_K(const uint8_t *x, float *y, int64_t nb)
{
    for (int64_t i = 0; i < nb; i++, x += 144) {
        const float d = H(x), min = H(x + 2);
        const uint8_t *scales = x + 4, *q = x + 16;
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

/* Q5_K: fp16 d, fp16 dmin, u8 scales[12], u8 qh[32], u8 qs[128]. */
static void dequant_q5_K(const uint8_t *x, float *y, int64_t nb)
{
    for (int64_t i = 0; i < nb; i++, x += 176) {
        const float d = H(x), min = H(x + 2);
        const uint8_t *scales = x + 4, *qh = x + 16, *ql = x + 48;
        int is = 0;
        uint8_t sc, m, u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
}

/* Q6_K: u8 ql[128], u8 qh[64], i8 scales[16], fp16 d (last). */
static void dequant_q6_K(const uint8_t *x, float *y, int64_t nb)
{
    for (int64_t i = 0; i < nb; i++, x += 210) {
        const float d = H(x + 208);
        const uint8_t *ql = x, *qh = x + 128;
        const int8_t *sc = (const int8_t *)(x + 192);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

/* IQ4_NL: fp16 d, u8 qs[16] (indices into a non-linear 16-value table). */
static void dequant_iq4_nl(const uint8_t *x, float *y, int64_t nb)
{
    for (int64_t i = 0; i < nb; i++, x += 18, y += 32) {
        const float d = H(x);
        const uint8_t *qs = x + 2;
        for (int j = 0; j < 16; ++j) {
            y[j + 0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = d * kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

int q38_quant_supported(int type)
{
    switch (type) {
    case Q38_F32: case Q38_F16: case Q38_BF16: case Q38_Q8_0: case Q38_Q5_1:
    case Q38_Q4_K: case Q38_Q5_K: case Q38_Q6_K: case Q38_IQ4_NL:
        return 1;
    default:
        return 0;
    }
}

int q38_dequant_row(int type, const void *src, float *dst, int64_t n)
{
    const uint8_t *x = (const uint8_t *)src;
    const int64_t bs = q38_type_block(type);
    if (!bs || n % bs) return -1;
    const int64_t nb = n / bs;
    switch (type) {
    case Q38_F32:  memcpy(dst, src, (size_t)n * 4); return 0;
    case Q38_F16:  for (int64_t i = 0; i < n; i++) dst[i] = H(x + 2 * i); return 0;
    case Q38_BF16: for (int64_t i = 0; i < n; i++) dst[i] = q38_bf16_to_fp32(rd16(x + 2 * i)); return 0;
    case Q38_Q8_0: dequant_q8_0(x, dst, nb); return 0;
    case Q38_Q5_1: dequant_q5_1(x, dst, nb); return 0;
    case Q38_Q4_K: dequant_q4_K(x, dst, nb); return 0;
    case Q38_Q5_K: dequant_q5_K(x, dst, nb); return 0;
    case Q38_Q6_K: dequant_q6_K(x, dst, nb); return 0;
    case Q38_IQ4_NL: dequant_iq4_nl(x, dst, nb); return 0;
    default: return -1;
    }
}
