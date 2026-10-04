/* q38_run - feed token ids through the engine (validation / timing).
 *
 *   q38_run <model.gguf> <tokens.txt> [--dump dir] [--gen N]
 *
 * tokens.txt holds one id per line (q38_ref_dump writes one). Each token is fed in turn;
 * --dump appends every tap (layer outputs, mixer outputs, logits ...) to dir/<name>.f32,
 * the layout tools/q38_compare.py reads. --gen N then continues greedily for N tokens.
 */
#ifdef _OPENMP
#include <omp.h>
#endif
#include "q38_model.h"
#include "q38_kernels.h"
#include "q38_quant.h"
#include "q38_tok.h"
#ifdef Q38_CUDA
#include "q38_gpu.h"
#endif

#include <stdio.h>
#include <math.h>
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
    int from = 0, quiet = 0, batch = 0, text = 0, vbench = 0, xbench = 0, d8check = 0, mvbench = 0, lbench = 0;
    static int gtoks[8192];
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
        else if (!strcmp(argv[i], "--text")) text = 1;
        else if (!strcmp(argv[i], "--verify-bench")) vbench = 1;
        else if (!strcmp(argv[i], "--expert-bench")) xbench = 1;
        else if (!strcmp(argv[i], "--dot8-check")) d8check = 1;
        else if (!strcmp(argv[i], "--mv-bench")) mvbench = 1;
        else if (!strcmp(argv[i], "--launch-bench")) lbench = 1;
        else if (!strcmp(argv[i], "--preload") && i + 1 < argc) preload = atof(argv[++i]);
        else if (!strcmp(argv[i], "--names") && i + 1 < argc) g_names = argv[++i];
    }
    static int toks[8192];
    int n = 0;
    if (!text) {
        FILE *f = fopen(argv[2], "r");
        if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        while (n < 8192 && fscanf(f, "%d", &toks[n]) == 1) n++;
        fclose(f);
    }

    static q38_model m;
    if (q38_model_open(&m, argv[1], text || n + gen > 4000 ? 8192 : 4096)) return 1;
    if (text) {                       /* tokens.txt is plain text: encode it, up to 8000 tokens */
        FILE *f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        fseek(f, 0, SEEK_END);
        const long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = (char *)malloc((size_t)len + 1);
        if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        fclose(f);
        buf[len] = 0;
        q38_tok *tk = q38_tok_load(&m.g);
        if (!tk) return 1;
        n = q38_tok_encode(tk, buf, (int)len, toks, 8000 - gen);
        q38_tok_free(tk);
        free(buf);
        fprintf(stderr, "%d tokens from the text\n", n);
    }
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
        if (i >= n - 1 && i - (n - 1) < 8192) gtoks[i - (n - 1)] = next;
        if (!quiet || i >= n) printf("%s %3d id %6d %6.3fs (read %.3f, experts %.3f, gpuwait %.3f, ple %.3f, tail %.3f) top5 %d %d %d %d %d\n",
               i < n ? "prompt" : "gen   ", i, tok, now_s() - t0, m.t_read - r0, m.t_experts - e0, m.t_gpu_wait - w0,
               m.t_ple - p0, m.t_tail - l0, t5[0], t5[1], t5[2], t5[3], t5[4]);
        fflush(stdout);
    }
#ifdef Q38_CUDA
    if (mvbench && gpu >= 0) q38_gpu_mvbench(&m);
    if (lbench && gpu >= 0) q38_gpu_launchbench(&m);
#endif
    if (d8check) {
        /* the 8-bit dot products against (a) the same quantised activation dequantised and
         * summed in double (the kernel itself: should agree to float rounding) and (b) the
         * float32 activation (the cost of quantising the activation) */
        const int E = m.c.n_embd, F = m.c.ff_exp;
        float *x = (float *)malloc((size_t)E * sizeof(float)), *w = (float *)malloc((size_t)E * sizeof(float));
        float *xd = (float *)malloc((size_t)E * sizeof(float)), *bs = (float *)malloc((size_t)E / 32 * sizeof(float));
        static q38_q8K kq[64];
        static q38_q8b bq[512];
        double worst_k = 0, worst_a = 0, sum_a = 0;
        int nchk = 0;
        for (int il = 0; il < m.c.n_layer; il++) {
            const int ids[1] = { 7 };
            const float wt[1] = { 1.0f };
            float *y = (float *)malloc((size_t)E * sizeof(float));
            q38_model_experts(&m, il, x, ids, wt, 0, y);     /* nothing; keeps the API warm */
            free(y);
            const uint8_t *data = NULL;
            int64_t bytes = 0;
            {   /* make expert 7 resident and get its bytes */
                float *yy = (float *)malloc((size_t)E * sizeof(float));
                for (int i = 0; i < E; i++) x[i] = 0.02f * (float)(((i + il) * 7919) % 401 - 200) * (1.0f + (i % 97 == 0) * 8);
                q38_model_experts(&m, il, x, ids, wt, 1, yy);
                free(yy);
            }
            if (q38_model_expert_slot(&m, il, 7, &data, &bytes)) continue;
            const q38_layer *l = &m.L[il];
            for (int mat = 0; mat < 2; mat++) {
                const int ty = mat == 0 ? l->ex_gate.type : l->ex_down.type;
                const int ne0 = mat == 0 ? E : F, rows = mat == 0 ? F : E;
                const int64_t rb = mat == 0 ? l->ex_gate.row_bytes : l->ex_down.row_bytes;
                const uint8_t *base = mat == 0 ? data : data + (int64_t)F * (l->ex_gate.row_bytes + l->ex_up.row_bytes);
                for (int i = 0; i < ne0; i++) x[i] = 0.02f * (float)(((i + il + mat) * 7919) % 401 - 200) * (1.0f + (i % 97 == 0) * 8);
                q38_xq vq;
                q38_xquant(&vq, x, ne0, kq, bq);
                if (!q38_dot8_supported(ty, &vq)) continue;
                const int kpath = ty == Q38_Q4_K || ty == Q38_Q5_K;
                for (int i = 0; i < ne0; i++)
                    xd[i] = kpath ? vq.k[i / 256].d * vq.k[i / 256].qs[i % 256] : vq.b[i / 32].d * vq.b[i / 32].qs[i % 32];
                q38_xvec vx;
                q38_xprep(&vx, x, ne0, bs);
                for (int r = 0; r < rows; r += 37) {
                    q38_dequant_row(ty, base + r * rb, w, ne0);
                    double ref = 0, refx = 0, mag = 0;
                    for (int i = 0; i < ne0; i++) { ref += (double)w[i] * xd[i]; refx += (double)w[i] * x[i]; mag += fabs((double)w[i] * x[i]); }
                    const float d8 = q38_dot8(ty, base + r * rb, &vq);
                    const double ek = fabs(d8 - ref) / (mag + 1e-30), ea = fabs(d8 - refx) / (mag + 1e-30);
                    if (ek > worst_k) worst_k = ek;
                    if (ea > worst_a) worst_a = ea;
                    sum_a += ea;
                    nchk++;
                }
            }
        }
        printf("dot8 check: %d rows; kernel vs its own activation in double: worst %.2e; vs float32 activation: "
               "mean %.2e worst %.2e (relative to sum |w x|)\n", nchk, worst_k, sum_a / (nchk ? nchk : 1), worst_a);
    }
    if (xbench) {
        /* the routed experts on the CPU as in a generated token: 4 resident experts per layer,
         * all 48 layers, with and without a gap (the host waiting for the GPU) between layers */
        const int E = m.c.n_embd, L = m.c.n_layer;
        float *x = (float *)malloc((size_t)E * sizeof(float)), *y = (float *)malloc((size_t)E * sizeof(float));
        for (int i = 0; i < E; i++) x[i] = 0.01f * (float)((i * 7919) % 201 - 100);
        const int ids[4] = { 1, 2, 3, 4 };
        const float w[4] = { 0.4f, 0.3f, 0.2f, 0.1f };
        int64_t eb = 0;
        for (int il = 0; il < L; il++) {
            q38_model_experts(&m, il, x, ids, w, 4, y);       /* resident from now on */
            const q38_layer *l = &m.L[il];
            eb += (int64_t)m.c.ff_exp * (l->ex_gate.row_bytes + l->ex_up.row_bytes) + (int64_t)E * l->ex_down.row_bytes;
        }
        static const int ths[] = { 1, 4, 8, 12, 16 };
        for (int ti = 0; ti < 5; ti++) {
#ifdef _OPENMP
            omp_set_num_threads(ths[ti]);
#endif
            for (int gap = 0; gap < 2; gap++) {
                const int reps = ths[ti] == 1 ? 2 : 10;
                double tt = 0, tg = 0;
                for (int r = 0; r < reps; r++)
                    for (int il = 0; il < L; il++) {
                        if (gap) { const double g0 = now_s(); while (now_s() - g0 < 0.0005) { } tg += now_s() - g0; }
                        const double t0 = now_s();
                        q38_model_experts(&m, il, x, ids, w, 4, y);
                        tt += now_s() - t0;
                    }
                printf("cpu experts, %2d threads%s: %.2f ms per token of 48x4 experts, %.1f GB/s\n", ths[ti],
                       gap ? " (0.5 ms gap per layer)" : "", 1e3 * tt / reps, (double)eb * reps / tt / 1e9);
            }
        }
    }
    if (vbench && gen >= 64) {
        /* the cost of checking k tokens in one pass (speculative decoding) against one token:
         * the inputs are tokens this run generated, so the routing is a real one */
        static const int ks[] = { 1, 2, 4, 8 };
        double vb_t[4] = { 0 };
        for (int ki = 0; ki < 4; ki++) {
            const int k = ks[ki], reps = 12;
            double tt = 0;
            for (int r = 0; r < reps; r++) {
                const int *in = gtoks + (r * 5) % (gen - 8);
                const double t0 = now_s();
                const float *lg = k == 1 ? q38_model_forward(&m, in[0]) : q38_model_forward_batch(&m, in, k);
                if (!lg) return 1;
                if (r >= 2) tt += now_s() - t0;          /* the first two warm the experts */
            }
            vb_t[ki] = tt / (reps - 2);
            printf("verify %d tokens in one pass: %.4f s (%.2fx one token)\n", k, vb_t[ki], vb_t[ki] / vb_t[0]);
        }
    }
    printf("experts read %.2f GB\n", m.expert_bytes_read / 1e9);
    q38_model_close(&m);
    return 0;
}
