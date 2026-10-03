/* q38_run - feed token ids through the engine (validation / timing).
 *
 *   q38_run <model.gguf> <tokens.txt> [--dump dir] [--gen N]
 *
 * tokens.txt holds one id per line (q38_ref_dump writes one). Each token is fed in turn;
 * --dump appends every tap (layer outputs, mixer outputs, logits ...) to dir/<name>.f32,
 * the layout tools/q38_compare.py reads. --gen N then continues greedily for N tokens.
 */
#include "q38_model.h"
#ifdef Q38_CUDA
#include "q38_gpu.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *g_dump;

static const char *g_names;   /* --names: comma-separated prefixes to keep (all when NULL) */
static int g_cur, g_from;

static void tap(void *ctx, const char *name, const float *x, int64_t n)
{
    (void)ctx;
    if (g_cur < g_from) return;
    if (g_names) {
        int keep = 0;
        const char *p = g_names;
        while (*p && !keep) {
            const char *e = strchr(p, ',');
            const size_t l = e ? (size_t)(e - p) : strlen(p);
            if (!strncmp(name, p, l)) keep = 1;
            p = e ? e + 1 : p + l;
        }
        if (!keep) return;
    }
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.f32", g_dump, name);
    FILE *f = fopen(path, "ab");
    if (!f) return;
    fwrite(x, sizeof(float), (size_t)n, f);
    fclose(f);
}

static double now_s(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void top5(const float *l, int n, int *out)
{
    for (int k = 0; k < 5; k++) {
        int b = -1;
        for (int i = 0; i < n; i++) {
            int used = 0;
            for (int j = 0; j < k; j++) if (out[j] == i) used = 1;
            if (!used && (b < 0 || l[i] > l[b])) b = i;
        }
        out[k] = b;
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <tokens.txt> [--dump dir] [--gen N] [--gpu N] [--cache-gb G]\n", argv[0]);
        return 2;
    }
    int gen = 0;
    double cache_gb = 0;
    int gpu = -1;
    const char *logits_dir = NULL;
    int from = 0, quiet = 0, batch = 0;
    double preload = 0;               /* GB of experts read at start from localcode's usage profile */  /* --batch: the prompt in one batched pass */     /* logits only, without taps (taps force plain GPU launches) */
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--dump") && i + 1 < argc) g_dump = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits") && i + 1 < argc) logits_dir = argv[++i];
        else if (!strcmp(argv[i], "--from") && i + 1 < argc) from = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--batch")) batch = 1;
        else if (!strcmp(argv[i], "--preload") && i + 1 < argc) preload = atof(argv[++i]);
        else if (!strcmp(argv[i], "--names") && i + 1 < argc) g_names = argv[++i];
    }
    FILE *f = fopen(argv[2], "r");
    if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    int toks[8192], n = 0;
    while (n < 8192 && fscanf(f, "%d", &toks[n]) == 1) n++;
    fclose(f);

    static q38_model m;
    if (q38_model_open(&m, argv[1], 4096)) return 1;
#ifdef Q38_CUDA
    if (gpu >= 0 && q38_gpu_open(&m, gpu)) return 1;
#endif
    if (cache_gb > 0) q38_model_set_cache(&m, cache_gb);
    if (g_dump) m.tap = tap;
    if (preload > 0) {
        char prof[1024];
        const char *base = getenv("LOCALAPPDATA");
        snprintf(prof, sizeof prof, "%s/localcode/qwen38_expert_profile.bin", base ? base : ".");
        const double tp = now_s();
        if (q38_model_profile_load(&m, prof) == 0) {
            const int nr = q38_model_preload(&m, preload);
            int nv = 0;
#ifdef Q38_CUDA
            if (gpu >= 0) nv = q38_gpu_prefill(&m);
#endif
            fprintf(stderr, "preloaded %d experts into RAM, %d into VRAM in %.1f s\n", nr, nv, now_s() - tp);
        }
        m.t_read = 0;
    }

    int next = -1;
    int i0 = 0;
    if (batch && n > 1) {
        const double tb = now_s();
        const float *lg = q38_model_forward_batch(&m, toks, n);
        if (!lg) return 1;
        printf("batch time: read %.1f s, cpu experts %.1f s, gpu wait %.1f s, ple %.1f s, total %.1f s\n", m.t_read,
               m.t_experts, m.t_gpu_wait, m.t_ple, now_s() - tb);
        int t5[5];
        top5(lg, m.c.n_vocab, t5);
        next = t5[0];
        printf("prompt of %d tokens in one batch: %.3fs (%.1f tok/s), top5 %d %d %d %d %d\n", n, now_s() - tb,
               n / (now_s() - tb), t5[0], t5[1], t5[2], t5[3], t5[4]);
        if (logits_dir) {
            char path[1024];
            snprintf(path, sizeof path, "%s/logits_last.f32", logits_dir);
            FILE *lf = fopen(path, "wb");
            if (lf) { fwrite(lg, sizeof(float), (size_t)m.c.n_vocab, lf); fclose(lf); }
        }
        i0 = n;
    }
    for (int i = i0; i < n + gen; i++) {
        int tok = i < n ? toks[i] : next;
        double t0 = now_s(), r0 = m.t_read, e0 = m.t_experts, w0 = m.t_gpu_wait, p0 = m.t_ple, l0 = m.t_tail;
        g_cur = i;
        g_from = from;
        const float *lg = q38_model_forward(&m, tok);
        if (!lg) return 1;
        if (logits_dir && i >= from) {
            char path[1024];
            snprintf(path, sizeof path, "%s/logits.f32", logits_dir);
            FILE *lf = fopen(path, "ab");
            if (lf) { fwrite(lg, sizeof(float), (size_t)m.c.n_vocab, lf); fclose(lf); }
        }
        int t5[5];
        top5(lg, m.c.n_vocab, t5);
        next = t5[0];
        if (!quiet || i >= n) printf("%s %3d id %6d %6.3fs (read %.3f, experts %.3f, gpuwait %.3f, ple %.3f, tail %.3f) top5 %d %d %d %d %d\n",
               i < n ? "prompt" : "gen   ", i, tok, now_s() - t0, m.t_read - r0, m.t_experts - e0, m.t_gpu_wait - w0,
               m.t_ple - p0, m.t_tail - l0, t5[0], t5[1], t5[2], t5[3], t5[4]);
        fflush(stdout);
    }
    printf("experts read %.2f GB\n", m.expert_bytes_read / 1e9);
    q38_model_close(&m);
    return 0;
}
