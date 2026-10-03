/* q38_quant.h - ggml block formats used by the Qwen3.8-Flash-Next GGUFs.
 *
 * Dequantisation reproduces llama.cpp's reference dequantize_row_* (ggml-quants.c)
 * operation for operation, so a dequantised weight is bit-identical to what ggml's
 * reference path produces. Supported: F32, F16, BF16, Q8_0, Q5_1, Q4_K, Q5_K, Q6_K,
 * IQ4_NL (every type in UD-Q4_K_XL plus Q6_K).
 */
#ifndef Q38_QUANT_H
#define Q38_QUANT_H

#include <stdint.h>

float q38_fp16_to_fp32(uint16_t h);
float q38_bf16_to_fp32(uint16_t h);

/* 1 when q38_dequant_row handles the type. */
int q38_quant_supported(int type);

/* Dequantises n elements (a multiple of the type's block) starting at src. Returns 0, or
 * -1 for an unsupported type. */
int q38_dequant_row(int type, const void *src, float *dst, int64_t n);

#endif
