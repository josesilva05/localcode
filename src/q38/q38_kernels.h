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

/* y[r] = row r . x for n_rows rows of row_bytes each; single thread. */
void  q38_matvec_rows(int type, const uint8_t *w, int64_t row_bytes, int64_t n_rows, const q38_xvec *v, float *y);

#endif
