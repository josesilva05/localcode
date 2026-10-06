/* q38_iq_tables.h - the grid of IQ3_S, from llama.cpp (ggml/src/ggml-common.h, iq3s_grid, MIT
 * licence): 512 entries, each the four 8-bit magnitudes of a group of four weights. The values
 * live in q38_iq3s_grid.inc, which q38_gpu.cu includes too for its device copy. */
#ifndef Q38_IQ_TABLES_H
#define Q38_IQ_TABLES_H
#include <stdint.h>
static const uint32_t q38_iq3s_grid[512] = {
#include "q38_iq3s_grid.inc"
};
#endif
