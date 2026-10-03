/* q38_gguf_info - inspect a (split) GGUF with the engine's reader.
 *
 *   q38_gguf_info <any shard.gguf>
 *       metadata summary and one line per tensor kind (layer numbers folded).
 *   q38_gguf_info <any shard.gguf> dump <tensor> <first_row> <n_rows> <out.f32>
 *       dequantises n_rows rows (ne0 elements each; rows are counted across ne1*ne2*ne3)
 *       to raw little-endian float32, for tools/q38_check_dequant.py.
 */
#include "q38_gguf.h"
#include "q38_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fold_name(const char *in, char *out, size_t cap)
{
    /* blk.17.ffn_up_exps.weight -> blk.N.ffn_up_exps.weight */
    size_t o = 0;
    for (size_t i = 0; in[i] && o + 2 < cap; i++) {
        if (!strncmp(in + i, "blk.", 4) && in[i + 4] >= '0' && in[i + 4] <= '9') {
            memcpy(out + o, "blk.N", 5);
            o += 5;
            i += 4;
            while (in[i] >= '0' && in[i] <= '9') i++;
            i--;
            continue;
        }
        out[o++] = in[i];
    }
    out[o] = 0;
}

static int info(const q38_gguf *g)
{
    printf("shards %d, tensors %d, keys %d, alignment %d\n", g->n_shards, g->n_tensors, g->n_kv, g->alignment);
    const char *arch = q38_gguf_str(g, "general.architecture");
    printf("architecture %s, name %s\n", arch ? arch : "?", q38_gguf_str(g, "general.name") ? q38_gguf_str(g, "general.name") : "?");
    for (int k = 0; k < g->n_kv; k++) {
        const q38_kv *kv = &g->kv[k];
        if (!strncmp(kv->key, "tokenizer.", 10) && kv->type == Q38_V_ARR) {
            printf("  %s [%llu]\n", kv->key, (unsigned long long)kv->n);
            continue;
        }
        if (!strcmp(kv->key, "tokenizer.chat_template")) { printf("  %s (%zu chars)\n", kv->key, strlen(kv->s)); continue; }
        printf("  %s = ", kv->key);
        if (kv->type == Q38_V_STR) printf("%s", kv->s);
        else if (kv->type == Q38_V_ARR) {
            printf("[");
            for (uint64_t j = 0; j < kv->n && j < 8; j++) {
                if (kv->strs) printf("%s\"%s\"", j ? ", " : "", kv->strs[j]);
                else if (kv->farr) printf("%s%g", j ? ", " : "", kv->farr[j]);
                else printf("%s%lld", j ? ", " : "", (long long)kv->iarr[j]);
            }
            printf("%s] (%llu)", kv->n > 8 ? ", ..." : "", (unsigned long long)kv->n);
        } else if (kv->type == Q38_V_F32 || kv->type == Q38_V_F64) printf("%g", kv->f);
        else printf("%lld", (long long)kv->i);
        printf("\n");
    }

    /* fold per-layer tensors: count, type, dims, total bytes */
    typedef struct { char name[128]; int type, count; int64_t ne[4], bytes; } row;
    row *rows = (row *)calloc((size_t)g->n_tensors, sizeof(row));
    int nr = 0;
    int64_t total = 0;
    int type_count[Q38_TYPE_COUNT] = {0};
    for (int k = 0; k < g->n_tensors; k++) {
        const q38_tensor *t = &g->tensors[k];
        char f[128];
        fold_name(t->name, f, sizeof f);
        int r = 0;
        for (; r < nr; r++)
            if (!strcmp(rows[r].name, f) && rows[r].type == t->type && !memcmp(rows[r].ne, t->ne, sizeof t->ne)) break;
        if (r == nr) {
            strcpy(rows[nr].name, f);
            rows[nr].type = t->type;
            memcpy(rows[nr].ne, t->ne, sizeof t->ne);
            nr++;
        }
        rows[r].count++;
        rows[r].bytes += t->nbytes;
        total += t->nbytes;
        type_count[t->type]++;
    }
    printf("\n%-4s %-8s %-40s %-24s %10s\n", "n", "type", "tensor", "ne", "MiB");
    for (int r = 0; r < nr; r++) {
        char dims[64];
        snprintf(dims, sizeof dims, "[%lld,%lld,%lld]", (long long)rows[r].ne[0], (long long)rows[r].ne[1], (long long)rows[r].ne[2]);
        printf("%-4d %-8s %-40s %-24s %10.1f%s\n", rows[r].count, q38_type_name(rows[r].type), rows[r].name, dims,
               rows[r].bytes / 1048576.0, q38_quant_supported(rows[r].type) ? "" : "  (no dequant)");
    }
    printf("\ntotal %.2f GiB\n", total / 1073741824.0);
    free(rows);
    return 0;
}

static int dump(const q38_gguf *g, const char *name, int64_t first, int64_t n, const char *out)
{
    const q38_tensor *t = q38_gguf_tensor(g, name);
    if (!t) { fprintf(stderr, "no tensor %s\n", name); return 1; }
    int64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
    if (first < 0 || n <= 0 || first + n > rows) { fprintf(stderr, "rows %lld..%lld outside 0..%lld\n", (long long)first, (long long)(first + n), (long long)rows); return 1; }
    if (!q38_quant_supported(t->type)) { fprintf(stderr, "type %s not supported\n", q38_type_name(t->type)); return 1; }
    int64_t rb = q38_row_bytes(t->type, t->ne[0]);
    unsigned char *raw = (unsigned char *)malloc((size_t)rb);
    float *y = (float *)malloc((size_t)t->ne[0] * sizeof(float));
    FILE *f = fopen(out, "wb");
    if (!raw || !y || !f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    for (int64_t r = first; r < first + n; r++) {
        if (q38_gguf_read(g, t, r * rb, raw, rb)) { fprintf(stderr, "read failed at row %lld\n", (long long)r); return 1; }
        q38_dequant_row(t->type, raw, y, t->ne[0]);
        fwrite(y, sizeof(float), (size_t)t->ne[0], f);
    }
    fclose(f);
    printf("%s %s shard %d: rows %lld..%lld of %lld (ne0 %lld) -> %s\n", name, q38_type_name(t->type), t->shard + 1,
           (long long)first, (long long)(first + n - 1), (long long)rows, (long long)t->ne[0], out);
    free(raw);
    free(y);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [dump <tensor> <first_row> <n_rows> <out.f32>]\n", argv[0]);
        return 2;
    }
    q38_gguf g;
    if (q38_gguf_open(&g, argv[1])) return 1;
    int rc;
    if (argc == 7 && !strcmp(argv[2], "dump"))
        rc = dump(&g, argv[3], atoll(argv[4]), atoll(argv[5]), argv[6]);
    else
        rc = info(&g);
    q38_gguf_close(&g);
    return rc;
}
