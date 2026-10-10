/* q38_kernels.h - dot products of a quantised weight row with a float32 activation.
 *
 * The activation stays float32 (no 8-bit activation quantisation, unlike llama.cpp's CPU
 * path): each weight is expanded to float in registers exactly as q38_dequant_row would
 * produce it, so the only difference from "dequantise, then dot" is the order of the
 * float32 additions. Block formats with an offset (Q4_K, Q5_K, Q5_1) use the identity
 *   sum_i (a*q_i - b) x_i = a * sum_i q_i x_i - b * sum_i x_i
 * with the per-32 sums of x computed once per activation (q38_xprep).
 */
#ifndef Q38_KERNELS_H
#define Q38_KERNELS_H

#include <stdint.h>

/* An activation prepared for the dot products: the values and their per-32 block sums. */
typedef struct {
    const float *x;
    float       *bsum;     /* n/32 sums */
    int64_t      n;
} q38_xvec;

void  q38_xprep(q38_xvec *v, const float *x, int64_t n, float *bsum_storage);
/* 1 when q38_dot handles the type */
int   q38_dot_supported(int type);
float q38_dot(int type, const uint8_t *row, const q38_xvec *v);

/* The activation quantised to 8 bits as llama.cpp's CPU dot products take it: q8_K blocks
 * of 256 (Q4_K, Q5_K rows) and q8_1 blocks of 32 (Q5_1, Q8_0 rows); integer products. */
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } q38_q8K;
typedef struct { float d, s; int8_t qs[32]; } q38_q8b;     /* s = d * sum(qs) */
typedef struct {
    q38_q8K *k;            /* n/256 blocks, NULL when n % 256 != 0 or not wanted */
    q38_q8b *b;            /* n/32 blocks, NULL when not wanted */
    int64_t  n;
} q38_xq;

void  q38_xquant(q38_xq *v, const float *x, int64_t n, q38_q8K *k_storage, q38_q8b *b_storage);
/* 1 when q38_dot8 handles the type with what v holds */
int   q38_dot8_supported(int type, const q38_xq *v);
float q38_dot8(int type, const uint8_t *row, const q38_xq *v);

/* y[r] = row r . x for n_rows rows of row_bytes each; single thread. */
void  q38_matvec_rows(int type, const uint8_t *w, int64_t row_bytes, int64_t n_rows, const q38_xvec *v, float *y);

#endif
