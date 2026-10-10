/* q38_gguf.h - GGUF reader for Qwen3.8-Flash-Next (llama.cpp's file format).
 *
 * Reads the header of every shard of a split GGUF (name-00001-of-0000N.gguf), keeps the
 * metadata (key/value pairs) and the tensor table, and leaves the weights on disk: a
 * tensor is a (shard, absolute offset, ggml type, dims) record that the loaders pread.
 *
 * Layout per shard (GGUF v3, little endian):
 *   "GGUF" u32 version, u64 n_tensors, u64 n_kv,
 *   n_kv x (string key, u32 type, value),
 *   n_tensors x (string name, u32 n_dims, u64 ne[n_dims], u32 ggml_type, u64 offset),
 *   padding to general.alignment (32 when absent), then the data section; a tensor's
 *   offset is relative to the start of its own shard's data section.
 * Strings are u64 length + bytes (no terminator). ne[0] is the contiguous dimension.
 */
#ifndef Q38_GGUF_H
#define Q38_GGUF_H

#include <stdint.h>
#include <stddef.h>

/* ggml tensor types (ggml.h numbering, which the file stores). */
enum q38_type {
    Q38_F32 = 0, Q38_F16 = 1, Q38_Q4_0 = 2, Q38_Q4_1 = 3, Q38_Q5_0 = 6, Q38_Q5_1 = 7,
    Q38_Q8_0 = 8, Q38_Q8_1 = 9, Q38_Q2_K = 10, Q38_Q3_K = 11, Q38_Q4_K = 12, Q38_Q5_K = 13,
    Q38_Q6_K = 14, Q38_Q8_K = 15, Q38_IQ2_XXS = 16, Q38_IQ2_XS = 17, Q38_IQ3_XXS = 18,
    Q38_IQ1_S = 19, Q38_IQ4_NL = 20, Q38_IQ3_S = 21, Q38_IQ2_S = 22, Q38_IQ4_XS = 23,
    Q38_I8 = 24, Q38_I16 = 25, Q38_I32 = 26, Q38_I64 = 27, Q38_F64 = 28, Q38_IQ1_M = 29,
    Q38_BF16 = 30, Q38_TQ1_0 = 34, Q38_TQ2_0 = 35, Q38_MXFP4 = 39,
    Q38_TYPE_COUNT = 40
};

/* Elements per block and bytes per block; 0/0 for an id ggml does not define. */
int         q38_type_block(int type);
int         q38_type_bytes(int type);
const char *q38_type_name(int type);
/* Bytes of one row of ne0 elements (ne0 must be a multiple of the block size). */
int64_t     q38_row_bytes(int type, int64_t ne0);

/* GGUF metadata value types. */
enum q38_vtype {
    Q38_V_U8 = 0, Q38_V_I8 = 1, Q38_V_U16 = 2, Q38_V_I16 = 3, Q38_V_U32 = 4, Q38_V_I32 = 5,
    Q38_V_F32 = 6, Q38_V_BOOL = 7, Q38_V_STR = 8, Q38_V_ARR = 9, Q38_V_U64 = 10,
    Q38_V_I64 = 11, Q38_V_F64 = 12
};

typedef struct {
    char    *key;
    int      type;      /* enum q38_vtype */
    int      atype;     /* element type when type == Q38_V_ARR */
    uint64_t n;         /* array length (1 for a scalar) */
    /* Scalars: i (all integer types and bool), f (both float types), s (string).
     * Arrays of numbers: arr holds n elements widened to int64 (ints) or double (floats);
     * arrays of strings: strs holds n NUL-terminated strings. */
    int64_t  i;
    double   f;
    char    *s;
    int64_t *iarr;
    double  *farr;
    char   **strs;
} q38_kv;

typedef struct {
    char    *name;
    int      type;      /* enum q38_type */
    int      n_dims;
    int64_t  ne[4];     /* unused dims are 1 */
    int      shard;     /* index into q38_gguf.paths / fp */
    int64_t  offset;    /* absolute byte offset in the shard file */
    int64_t  nbytes;
} q38_tensor;

#define Q38_MAX_SHARDS 16

typedef struct {
    int         n_shards;
    char       *paths[Q38_MAX_SHARDS];
    void       *fp[Q38_MAX_SHARDS];   /* FILE *, kept open for q38_gguf_read */
    intptr_t    fh[Q38_MAX_SHARDS];   /* positioned-read handles (HANDLE / fd), 0 = not open */
    intptr_t    fhd[Q38_MAX_SHARDS];  /* the same without the OS file cache (Windows), 0 = none */
    int         n_kv;
    q38_kv     *kv;
    int         n_tensors;
    q38_tensor *tensors;
    int         alignment;
} q38_gguf;

/* Opens path (any shard of a split, or a single-file GGUF), reads every shard's header.
 * Returns 0 on success; on failure prints the reason and returns -1. */
int  q38_gguf_open(q38_gguf *g, const char *path);
void q38_gguf_close(q38_gguf *g);

const q38_kv     *q38_gguf_kv(const q38_gguf *g, const char *key);
const q38_tensor *q38_gguf_tensor(const q38_gguf *g, const char *name);

/* Typed metadata getters: return def when the key is missing or of another kind. */
int64_t     q38_gguf_int(const q38_gguf *g, const char *key, int64_t def);
double      q38_gguf_float(const q38_gguf *g, const char *key, double def);
const char *q38_gguf_str(const q38_gguf *g, const char *key);

/* Reads nbytes at byte offset off of the tensor into dst. Returns 0 on success.
 * Not thread-safe (shares a FILE position); used at open time. */
int  q38_gguf_read(const q38_gguf *g, const q38_tensor *t, int64_t off, void *dst, int64_t nbytes);

/* The same as a positioned read (pread / ReadFile with an offset): safe to call from any
 * number of threads at once. Opens its own handles on first use. */
/* q38_gguf_pread past the OS file cache when the shard could be opened that way (sector-aligned
 * reads into a per-thread buffer, then copied); otherwise the same as q38_gguf_pread. */
int  q38_gguf_pread_direct(q38_gguf *g, const q38_tensor *t, int64_t off, void *dst, int64_t nbytes);
/* n such reads issued together (asynchronous, all in flight at once), then waited for */
int  q38_gguf_pread_multi(q38_gguf *g, int n, const q38_tensor *const *t, const int64_t *off, void *const *dst,
                          const int64_t *nbytes);
int  q38_gguf_pread(q38_gguf *g, const q38_tensor *t, int64_t off, void *dst, int64_t nbytes);
/* A whole tensor into dst, past the OS file cache in 8 MB pieces paced by the governor. */
int  q38_gguf_read_tensor(q38_gguf *g, const q38_tensor *t, void *dst);
/* The disk governor (q38_gguf.c): the last measured active time of the model's disk in
 * percent (-1 unknown), and the pause it currently puts before each expert read. */
int  q38_gguf_disk_busy(void);
int  q38_gguf_disk_pause_ms(void);
/* Waits while the governor holds the disk quiet; call before each piece of a big transfer. */
void q38_gguf_disk_wait(void);

#endif
