/* q38_gguf.c - see q38_gguf.h. */
#include "q38_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#define q38_fseek _fseeki64
#define q38_ftell _ftelli64
#else
#define q38_fseek fseeko
#define q38_ftell ftello
#endif

/* --------------------------------------------------------------------- type table -- */
static const struct { int block, bytes; const char *name; } TT[Q38_TYPE_COUNT] = {
    [Q38_F32] = {1, 4, "F32"},        [Q38_F16] = {1, 2, "F16"},
    [Q38_Q4_0] = {32, 18, "Q4_0"},    [Q38_Q4_1] = {32, 20, "Q4_1"},
    [Q38_Q5_0] = {32, 22, "Q5_0"},    [Q38_Q5_1] = {32, 24, "Q5_1"},
    [Q38_Q8_0] = {32, 34, "Q8_0"},    [Q38_Q8_1] = {32, 36, "Q8_1"},
    [Q38_Q2_K] = {256, 84, "Q2_K"},   [Q38_Q3_K] = {256, 110, "Q3_K"},
    [Q38_Q4_K] = {256, 144, "Q4_K"},  [Q38_Q5_K] = {256, 176, "Q5_K"},
    [Q38_Q6_K] = {256, 210, "Q6_K"},  [Q38_Q8_K] = {256, 292, "Q8_K"},
    [Q38_IQ2_XXS] = {256, 66, "IQ2_XXS"}, [Q38_IQ2_XS] = {256, 74, "IQ2_XS"},
    [Q38_IQ3_XXS] = {256, 98, "IQ3_XXS"}, [Q38_IQ1_S] = {256, 50, "IQ1_S"},
    [Q38_IQ4_NL] = {32, 18, "IQ4_NL"},    [Q38_IQ3_S] = {256, 110, "IQ3_S"},
    [Q38_IQ2_S] = {256, 82, "IQ2_S"},     [Q38_IQ4_XS] = {256, 136, "IQ4_XS"},
    [Q38_I8] = {1, 1, "I8"},   [Q38_I16] = {1, 2, "I16"}, [Q38_I32] = {1, 4, "I32"},
    [Q38_I64] = {1, 8, "I64"}, [Q38_F64] = {1, 8, "F64"}, [Q38_IQ1_M] = {256, 56, "IQ1_M"},
    [Q38_BF16] = {1, 2, "BF16"}, [Q38_TQ1_0] = {256, 54, "TQ1_0"},
    [Q38_TQ2_0] = {256, 66, "TQ2_0"}, [Q38_MXFP4] = {32, 17, "MXFP4"},
};

int q38_type_block(int t) { return (t >= 0 && t < Q38_TYPE_COUNT) ? TT[t].block : 0; }
int q38_type_bytes(int t) { return (t >= 0 && t < Q38_TYPE_COUNT) ? TT[t].bytes : 0; }
const char *q38_type_name(int t)
{
    return (t >= 0 && t < Q38_TYPE_COUNT && TT[t].name) ? TT[t].name : "?";
}
int64_t q38_row_bytes(int t, int64_t ne0)
{
    int b = q38_type_block(t);
    return b ? ne0 / b * q38_type_bytes(t) : 0;
}

/* ------------------------------------------------------------------ file reading -- */
typedef struct { FILE *f; const char *path; int err; } rd;

static void rd_raw(rd *r, void *p, size_t n)
{
    if (r->err) { memset(p, 0, n); return; }
    if (fread(p, 1, n, r->f) != n) {
        fprintf(stderr, "q38_gguf: %s: truncated header\n", r->path);
        r->err = 1;
        memset(p, 0, n);
    }
}
static uint32_t rd_u32(rd *r) { uint32_t v; rd_raw(r, &v, 4); return v; }
static uint64_t rd_u64(rd *r) { uint64_t v; rd_raw(r, &v, 8); return v; }

static char *rd_str(rd *r)
{
    uint64_t n = rd_u64(r);
    if (r->err) return NULL;
    if (n > ((uint64_t)1 << 30)) {
        fprintf(stderr, "q38_gguf: %s: string of %llu bytes\n", r->path, (unsigned long long)n);
        r->err = 1;
        return NULL;
    }
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) { r->err = 1; return NULL; }
    rd_raw(r, s, (size_t)n);
    s[n] = 0;
    return s;
}

static int vsize(int t)
{
    switch (t) {
    case Q38_V_U8: case Q38_V_I8: case Q38_V_BOOL: return 1;
    case Q38_V_U16: case Q38_V_I16: return 2;
    case Q38_V_U32: case Q38_V_I32: case Q38_V_F32: return 4;
    case Q38_V_U64: case Q38_V_I64: case Q38_V_F64: return 8;
    default: return 0;
    }
}

/* One scalar of type t; integers widen to *i, floats to *f. */
static void rd_scalar(rd *r, int t, int64_t *i, double *f)
{
    union { uint8_t u8; int8_t i8; uint16_t u16; int16_t i16; uint32_t u32; int32_t i32;
            float f32; uint64_t u64; int64_t i64; double f64; } u;
    int n = vsize(t);
    if (!n) { fprintf(stderr, "q38_gguf: %s: bad value type %d\n", r->path, t); r->err = 1; return; }
    rd_raw(r, &u, (size_t)n);
    *i = 0; *f = 0;
    switch (t) {
    case Q38_V_U8: case Q38_V_BOOL: *i = u.u8; break;
    case Q38_V_I8: *i = u.i8; break;
    case Q38_V_U16: *i = u.u16; break;
    case Q38_V_I16: *i = u.i16; break;
    case Q38_V_U32: *i = u.u32; break;
    case Q38_V_I32: *i = u.i32; break;
    case Q38_V_U64: *i = (int64_t)u.u64; break;
    case Q38_V_I64: *i = u.i64; break;
    case Q38_V_F32: *f = u.f32; break;
    case Q38_V_F64: *f = u.f64; break;
    }
}

static int is_float(int t) { return t == Q38_V_F32 || t == Q38_V_F64; }

static void rd_kv(rd *r, q38_kv *kv)
{
    memset(kv, 0, sizeof *kv);
    kv->key = rd_str(r);
    kv->type = (int)rd_u32(r);
    kv->n = 1;
    if (r->err) return;
    if (kv->type == Q38_V_STR) {
        kv->s = rd_str(r);
    } else if (kv->type == Q38_V_ARR) {
        kv->atype = (int)rd_u32(r);
        kv->n = rd_u64(r);
        if (r->err) return;
        if (kv->n > ((uint64_t)1 << 32)) { r->err = 1; return; }
        if (kv->atype == Q38_V_STR) {
            kv->strs = (char **)calloc((size_t)kv->n ? (size_t)kv->n : 1, sizeof(char *));
            for (uint64_t j = 0; j < kv->n && !r->err; j++) kv->strs[j] = rd_str(r);
        } else if (is_float(kv->atype)) {
            kv->farr = (double *)calloc((size_t)kv->n ? (size_t)kv->n : 1, sizeof(double));
            for (uint64_t j = 0; j < kv->n && !r->err; j++) {
                int64_t dummy;
                rd_scalar(r, kv->atype, &dummy, &kv->farr[j]);
            }
        } else {
            kv->iarr = (int64_t *)calloc((size_t)kv->n ? (size_t)kv->n : 1, sizeof(int64_t));
            for (uint64_t j = 0; j < kv->n && !r->err; j++) {
                double dummy;
                rd_scalar(r, kv->atype, &kv->iarr[j], &dummy);
            }
        }
    } else {
        rd_scalar(r, kv->type, &kv->i, &kv->f);
    }
}

static void kv_free(q38_kv *kv)
{
    if (kv->strs)
        for (uint64_t j = 0; j < kv->n; j++) free(kv->strs[j]);
    free(kv->strs);
    free(kv->iarr);
    free(kv->farr);
    free(kv->s);
    free(kv->key);
}

/* Reads one shard's header. Metadata is kept from shard 0 only (later shards repeat a few
 * split.* keys); every shard contributes its tensors. */
static int read_shard(q38_gguf *g, int shard, int keep_kv)
{
    const char *path = g->paths[shard];
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "q38_gguf: cannot open %s\n", path); return -1; }
    g->fp[shard] = f;
    setvbuf(f, NULL, _IOFBF, 1 << 20);
    rd r = { f, path, 0 };

    char magic[4];
    rd_raw(&r, magic, 4);
    if (r.err || memcmp(magic, "GGUF", 4)) {
        fprintf(stderr, "q38_gguf: %s is not a GGUF file\n", path);
        return -1;
    }
    uint32_t ver = rd_u32(&r);
    if (ver < 2 || ver > 3) {
        fprintf(stderr, "q38_gguf: %s: GGUF version %u not supported\n", path, ver);
        return -1;
    }
    uint64_t nt = rd_u64(&r), nk = rd_u64(&r);
    if (r.err || nt > 1000000 || nk > 100000) {
        fprintf(stderr, "q38_gguf: %s: implausible header (%llu tensors, %llu keys)\n", path,
                (unsigned long long)nt, (unsigned long long)nk);
        return -1;
    }

    int alignment = 32;
    for (uint64_t k = 0; k < nk && !r.err; k++) {
        q38_kv kv;
        rd_kv(&r, &kv);
        if (r.err) { kv_free(&kv); break; }
        if (kv.key && !strcmp(kv.key, "general.alignment") && kv.type != Q38_V_ARR) alignment = (int)kv.i;
        if (keep_kv) {
            q38_kv *nkv = (q38_kv *)realloc(g->kv, (size_t)(g->n_kv + 1) * sizeof *nkv);
            if (!nkv) { kv_free(&kv); r.err = 1; break; }
            g->kv = nkv;
            g->kv[g->n_kv++] = kv;
        } else {
            kv_free(&kv);
        }
    }
    if (r.err) return -1;
    if (alignment <= 0 || (alignment & (alignment - 1))) {
        fprintf(stderr, "q38_gguf: %s: alignment %d\n", path, alignment);
        return -1;
    }
    if (shard == 0) g->alignment = alignment;

    int first = g->n_tensors;
    q38_tensor *nt_arr = (q38_tensor *)realloc(g->tensors, (size_t)(first + nt) * sizeof *nt_arr);
    if (!nt_arr && nt) return -1;
    g->tensors = nt_arr;
    for (uint64_t k = 0; k < nt && !r.err; k++) {
        q38_tensor *t = &g->tensors[first + k];
        memset(t, 0, sizeof *t);
        t->name = rd_str(&r);
        t->n_dims = (int)rd_u32(&r);
        if (t->n_dims < 1 || t->n_dims > 4) {
            fprintf(stderr, "q38_gguf: %s: tensor %s has %d dims\n", path, t->name ? t->name : "?", t->n_dims);
            r.err = 1;
            g->n_tensors = first + (int)k + 1;
            break;
        }
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (int d = 0; d < t->n_dims; d++) t->ne[d] = (int64_t)rd_u64(&r);
        t->type = (int)rd_u32(&r);
        t->offset = (int64_t)rd_u64(&r);
        t->shard = shard;
        g->n_tensors = first + (int)k + 1;
        if (!q38_type_block(t->type)) {
            fprintf(stderr, "q38_gguf: %s: tensor %s has unknown type %d\n", path, t->name, t->type);
            r.err = 1;
            break;
        }
        if (t->ne[0] % q38_type_block(t->type)) {
            fprintf(stderr, "q38_gguf: %s: tensor %s: ne0 %lld not a multiple of the %s block\n",
                    path, t->name, (long long)t->ne[0], q38_type_name(t->type));
            r.err = 1;
            break;
        }
        t->nbytes = q38_row_bytes(t->type, t->ne[0]) * t->ne[1] * t->ne[2] * t->ne[3];
    }
    if (r.err) return -1;

    int64_t pos = q38_ftell(f);
    int64_t data = (pos + alignment - 1) / alignment * alignment;
    q38_fseek(f, 0, SEEK_END);
    int64_t size = q38_ftell(f);
    for (int k = first; k < g->n_tensors; k++) {
        q38_tensor *t = &g->tensors[k];
        if (t->offset % alignment) {
            fprintf(stderr, "q38_gguf: %s: tensor %s offset %lld not aligned\n", path, t->name,
                    (long long)t->offset);
            return -1;
        }
        t->offset += data;
        if (t->offset + t->nbytes > size) {
            fprintf(stderr, "q38_gguf: %s: tensor %s runs past the end of the file\n", path, t->name);
            return -1;
        }
    }
    return 0;
}

/* "dir/name-00002-of-00004.gguf" -> prefix "dir/name-", count 4. Returns 0 when the path
 * is not a split name. */
static int split_count(const char *path, size_t *prefix_len)
{
    size_t n = strlen(path);
    /* -NNNNN-of-NNNNN.gguf is 20 characters */
    if (n < 20 || strcmp(path + n - 5, ".gguf")) return 0;
    const char *p = path + n - 20;
    if (p[0] != '-' || strncmp(p + 6, "-of-", 4)) return 0;
    for (int i = 1; i <= 5; i++) if (p[i] < '0' || p[i] > '9') return 0;
    for (int i = 10; i <= 14; i++) if (p[i] < '0' || p[i] > '9') return 0;
    *prefix_len = (size_t)(p - path) + 1;
    return atoi(p + 10);
}

int q38_gguf_open(q38_gguf *g, const char *path)
{
    memset(g, 0, sizeof *g);
    size_t plen = 0;
    int count = split_count(path, &plen);
    if (count > Q38_MAX_SHARDS) {
        fprintf(stderr, "q38_gguf: %s: %d shards (max %d)\n", path, count, Q38_MAX_SHARDS);
        return -1;
    }
    if (count <= 0) {
        g->n_shards = 1;
        g->paths[0] = (char *)malloc(strlen(path) + 1);
        strcpy(g->paths[0], path);
    } else {
        g->n_shards = count;
        for (int s = 0; s < count; s++) {
            g->paths[s] = (char *)malloc(plen + 32);
            memcpy(g->paths[s], path, plen);
            snprintf(g->paths[s] + plen, 32, "%05d-of-%05d.gguf", s + 1, count);
        }
    }
    for (int s = 0; s < g->n_shards; s++) {
        if (read_shard(g, s, s == 0)) {
            q38_gguf_close(g);
            return -1;
        }
    }
    for (int s = 0; s < g->n_shards; s++) {
#ifdef _WIN32
        HANDLE h = CreateFileA(g->paths[s], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
        if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "q38_gguf: cannot open %s\n", g->paths[s]); q38_gguf_close(g); return -1; }
        g->fh[s] = (intptr_t)h;
        HANDLE hd = CreateFileA(g->paths[s], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
        g->fhd[s] = hd == INVALID_HANDLE_VALUE ? 0 : (intptr_t)hd;
#else
        int fd = open(g->paths[s], O_RDONLY);
        if (fd < 0) { fprintf(stderr, "q38_gguf: cannot open %s\n", g->paths[s]); q38_gguf_close(g); return -1; }
        g->fh[s] = (intptr_t)fd + 1;
#endif
    }
    int64_t declared = q38_gguf_int(g, "split.tensors.count", -1);
    if (declared >= 0 && declared != g->n_tensors) {
        fprintf(stderr, "q38_gguf: %s: split.tensors.count %lld but the shards hold %d tensors\n",
                path, (long long)declared, g->n_tensors);
        q38_gguf_close(g);
        return -1;
    }
    return 0;
}

void q38_gguf_close(q38_gguf *g)
{
    for (int k = 0; k < g->n_kv; k++) kv_free(&g->kv[k]);
    free(g->kv);
    for (int k = 0; k < g->n_tensors; k++) free(g->tensors[k].name);
    free(g->tensors);
    for (int s = 0; s < Q38_MAX_SHARDS; s++) {
        if (g->fp[s]) fclose((FILE *)g->fp[s]);
#ifdef _WIN32
        if (g->fh[s]) CloseHandle((HANDLE)g->fh[s]);
        if (g->fhd[s]) CloseHandle((HANDLE)g->fhd[s]);
#else
        if (g->fh[s]) close((int)g->fh[s] - 1);
#endif
        free(g->paths[s]);
    }
    memset(g, 0, sizeof *g);
}

const q38_kv *q38_gguf_kv(const q38_gguf *g, const char *key)
{
    for (int k = 0; k < g->n_kv; k++)
        if (g->kv[k].key && !strcmp(g->kv[k].key, key)) return &g->kv[k];
    return NULL;
}

const q38_tensor *q38_gguf_tensor(const q38_gguf *g, const char *name)
{
    for (int k = 0; k < g->n_tensors; k++)
        if (!strcmp(g->tensors[k].name, name)) return &g->tensors[k];
    return NULL;
}

int64_t q38_gguf_int(const q38_gguf *g, const char *key, int64_t def)
{
    const q38_kv *kv = q38_gguf_kv(g, key);
    if (!kv || kv->type == Q38_V_ARR || kv->type == Q38_V_STR || is_float(kv->type)) return def;
    return kv->i;
}

double q38_gguf_float(const q38_gguf *g, const char *key, double def)
{
    const q38_kv *kv = q38_gguf_kv(g, key);
    if (!kv || kv->type == Q38_V_ARR || kv->type == Q38_V_STR) return def;
    return is_float(kv->type) ? kv->f : (double)kv->i;
}

const char *q38_gguf_str(const q38_gguf *g, const char *key)
{
    const q38_kv *kv = q38_gguf_kv(g, key);
    return (kv && kv->type == Q38_V_STR) ? kv->s : NULL;
}

int q38_gguf_read(const q38_gguf *g, const q38_tensor *t, int64_t off, void *dst, int64_t nbytes)
{
    if (off < 0 || nbytes < 0 || off + nbytes > t->nbytes) return -1;
    FILE *f = (FILE *)g->fp[t->shard];
    if (q38_fseek(f, t->offset + off, SEEK_SET)) return -1;
    return fread(dst, 1, (size_t)nbytes, f) == (size_t)nbytes ? 0 : -1;
}

#ifdef _WIN32
#define Q38_DIO_MAX 8
static __declspec(thread) uint8_t *tl_dbuf[Q38_DIO_MAX];
static __declspec(thread) int64_t tl_dcap[Q38_DIO_MAX];
static __declspec(thread) HANDLE tl_ev[Q38_DIO_MAX];
#endif

int q38_gguf_pread_multi(q38_gguf *g, int n, const q38_tensor *const *t, const int64_t *off, void *const *dst,
                         const int64_t *nbytes)
{
#ifdef _WIN32
    for (int c0 = 0; c0 < n; c0 += Q38_DIO_MAX) {
        const int cn = n - c0 < Q38_DIO_MAX ? n - c0 : Q38_DIO_MAX;
        OVERLAPPED ov[Q38_DIO_MAX];
        int64_t pos[Q38_DIO_MAX], a0[Q38_DIO_MAX];
        int issued[Q38_DIO_MAX];
        int bad = 0;
        for (int k = 0; k < cn; k++) {
            const int i = c0 + k;
            issued[k] = 0;
            if (off[i] < 0 || nbytes[i] < 0 || off[i] + nbytes[i] > t[i]->nbytes) { bad = 1; continue; }
            const HANDLE h = (HANDLE)g->fhd[t[i]->shard];
            const int64_t SEC = 4096;
            pos[k] = t[i]->offset + off[i];
            a0[k] = pos[k] & ~(SEC - 1);
            const int64_t len = ((pos[k] + nbytes[i] + SEC - 1) & ~(SEC - 1)) - a0[k];
            if (!h || len > ((int64_t)1 << 26)) continue;              /* read below, the plain way */
            if (len > tl_dcap[k]) {
                if (tl_dbuf[k]) VirtualFree(tl_dbuf[k], 0, MEM_RELEASE);
                tl_dcap[k] = len + ((int64_t)1 << 20);
                tl_dbuf[k] = (uint8_t *)VirtualAlloc(NULL, (SIZE_T)tl_dcap[k], MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                if (!tl_dbuf[k]) { tl_dcap[k] = 0; continue; }
            }
            if (!tl_ev[k]) tl_ev[k] = CreateEventA(NULL, TRUE, FALSE, NULL);
            if (!tl_ev[k]) continue;
            ResetEvent(tl_ev[k]);
            memset(&ov[k], 0, sizeof ov[k]);
            ov[k].Offset = (DWORD)(uint64_t)a0[k];
            ov[k].OffsetHigh = (DWORD)((uint64_t)a0[k] >> 32);
            ov[k].hEvent = tl_ev[k];
            if (!ReadFile(h, tl_dbuf[k], (DWORD)len, NULL, &ov[k]) && GetLastError() != ERROR_IO_PENDING) continue;
            issued[k] = 1;
        }
        for (int k = 0; k < cn; k++) {
            const int i = c0 + k;
            int ok = 0;
            if (issued[k]) {
                DWORD got = 0;
                if (GetOverlappedResult((HANDLE)g->fhd[t[i]->shard], &ov[k], &got, TRUE) && (int64_t)got >= pos[k] + nbytes[i] - a0[k]) {
                    memcpy(dst[i], tl_dbuf[k] + (pos[k] - a0[k]), (size_t)nbytes[i]);
                    ok = 1;
                }
            }
            if (!ok && !(off[i] < 0 || nbytes[i] < 0 || off[i] + nbytes[i] > t[i]->nbytes))
                ok = q38_gguf_pread(g, t[i], off[i], dst[i], nbytes[i]) == 0;
            if (!ok) bad = 1;
        }
        if (bad) return -1;
    }
    return 0;
#else
    for (int i = 0; i < n; i++) if (q38_gguf_pread(g, t[i], off[i], dst[i], nbytes[i])) return -1;
    return 0;
#endif
}

int q38_gguf_pread_direct(q38_gguf *g, const q38_tensor *t, int64_t off, void *dst, int64_t nbytes)
{
    return q38_gguf_pread_multi(g, 1, &t, &off, &dst, &nbytes);
}

int q38_gguf_pread(q38_gguf *g, const q38_tensor *t, int64_t off, void *dst, int64_t nbytes)
{
    if (off < 0 || nbytes < 0 || off + nbytes > t->nbytes) return -1;
    int64_t pos = t->offset + off, done = 0;
    char *p = (char *)dst;
    while (done < nbytes) {
        int64_t want = nbytes - done;
        if (want > ((int64_t)1 << 30)) want = (int64_t)1 << 30;
#ifdef _WIN32
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        ov.Offset = (DWORD)(uint64_t)(pos + done);
        ov.OffsetHigh = (DWORD)((uint64_t)(pos + done) >> 32);
        DWORD got = 0;
        if (!ReadFile((HANDLE)g->fh[t->shard], p + done, (DWORD)want, &got, &ov) || got == 0) return -1;
#else
        ssize_t got = pread((int)g->fh[t->shard] - 1, p + done, (size_t)want, (off_t)(pos + done));
        if (got <= 0) return -1;
#endif
        done += (int64_t)got;
    }
    return 0;
}
