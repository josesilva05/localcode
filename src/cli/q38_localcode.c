/* q38_localcode.c - localcode on Qwen3.8-Flash-Next: the interactive session on this engine.
 *
 *   localcode [model.gguf] [--ctx N] [--gen N] [--reasoning xhigh|medium|low] [--threads N]
 *
 * The model is any shard of the split GGUF (or LOCALCODE_MODEL in the environment). The
 * session itself (screen, commands, transcript) is glm53f_chat.c / glm53f_tui.c; this file
 * opens the model and hands it over as a Glm53fEngine: the forward pass of q38_model.c, the
 * tokenizer of q38_tok.c and the chat template of the GGUF (tokenizer.chat_template, the
 * single-user-turn path with add_generation_prompt).
 */
#include "q38_model.h"
#include "q38_tok.h"
#ifdef Q38_CUDA
#include "q38_gpu.h"
#endif
#include "glm53f_chat.h"
#include "lc_tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <strings.h>
#define _stricmp strcasecmp
#endif

typedef struct {
    q38_model *m;
    q38_tok   *tok;
    int        eos[4], n_eos;
    int        threads;
    int        gpu;                     /* device holding the trunk, -1 = cpu */
} QwenEngine;

/* the template's reasoning_effort values; 'high' is an alias of xhigh there */
static const Glm53fLevel QWEN_LEVELS[] = {
    { "xhigh",  "think carefully, validate assumptions (the template's default)" },
    { "medium", "no reasoning instruction in the system prompt" },
    { "low",    "brief thinking: the answer starts soonest" },
};

static double now_s(void)
{
    struct timespec ts;
#ifdef _WIN32
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);          /* C11's timespec_get is not in gnu99's glibc headers */
#endif
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* LOCALCODE_TIMES=1: where the time of the generated tokens went, printed at exit */
static struct { int n; double all, read, experts, gpu, ple, tail; int64_t hits, misses; } g_dec;

static int qw_forward(Glm53fEngine *e, const int *ids, int n, float *logits)
{
    QwenEngine *q = (QwenEngine *)e->impl;
    const q38_model *mm = q->m;
    const double t0 = now_s(), r0 = mm->t_read, x0 = mm->t_experts, w0 = mm->t_gpu_wait, p0 = mm->t_ple, l0 = mm->t_tail;
    const int64_t h0 = mm->cache_hits, s0 = mm->cache_misses;
    const float *lg = q38_model_forward_batch(q->m, ids, n);
    if (n == 1) {
        g_dec.n++;
        g_dec.all += now_s() - t0;
        g_dec.read += mm->t_read - r0;
        g_dec.experts += mm->t_experts - x0;
        g_dec.gpu += mm->t_gpu_wait - w0;
        g_dec.ple += mm->t_ple - p0;
        g_dec.tail += mm->t_tail - l0;
        g_dec.hits += mm->cache_hits - h0;
        g_dec.misses += mm->cache_misses - s0;
    }
    if (!lg) { e->cached = q->m->n_past; return -1; }
    if (lg) memcpy(logits, lg, (size_t)q->m->c.n_vocab * sizeof(float));
    e->cached = q->m->n_past;
    return 0;
}

static int qw_pick(void *ctx, const float *logits, int n) { return glm53f_sample((Glm53fSampler *)ctx, logits, n); }

/* the MTP block's guesses (q38_model_forward_spec): tok and the right guesses in one pass */
static int qw_forward_spec(Glm53fEngine *e, int tok, int *extra, float *logits, Glm53fSampler *smp, int *next)
{
    QwenEngine *q = (QwenEngine *)e->impl;
    const double t0 = now_s();
    const int sampled = smp && smp->temp > 0.0f;
    const int k = q38_model_forward_spec(q->m, tok, q->eos, q->n_eos, extra, sampled ? qw_pick : NULL, smp, next);
    e->cached = q->m->n_past;
    if (k < 0) return -1;
    g_dec.n += 1 + k;
    g_dec.all += now_s() - t0;
    memcpy(logits, q->m->logits, (size_t)q->m->c.n_vocab * sizeof(float));
    return k;
}

static void qw_reset(Glm53fEngine *e)
{
    QwenEngine *q = (QwenEngine *)e->impl;
    q38_model_reset(q->m);
    e->cached = 0;
}

static int qw_encode(Glm53fEngine *e, const char *text, int len, int *out, int max)
{
    return q38_tok_encode(((QwenEngine *)e->impl)->tok, text, len, out, max);
}

static int qw_decode(Glm53fEngine *e, const int *ids, int n, char *out, int max)
{
    QwenEngine *q = (QwenEngine *)e->impl;
    int o = 0;
    char piece[1024];
    for (int i = 0; i < n; i++) {
        if (q38_tok_is_control(q->tok, ids[i])) continue;
        const int k = q38_tok_piece(q->tok, ids[i], piece, sizeof piece);
        if (o + k >= max) break;
        memcpy(out + o, piece, (size_t)k);
        o += k;
    }
    if (o < max) out[o] = 0;
    return o;
}

static int qw_is_eos(Glm53fEngine *e, int id)
{
    QwenEngine *q = (QwenEngine *)e->impl;
    for (int i = 0; i < q->n_eos; i++) if (id == q->eos[i]) return 1;
    return 0;
}

/* tokenizer.chat_template: a system block only for the xhigh/low instructions, then the
 * user turn and the generation prompt "<|im_start|>assistant\n<think>\n". The model's own
 * <|im_end|> is never fed (decoding stops on it), so a later turn closes the previous
 * answer first; its reasoning stays in the context, as preserve_thinking does. */
static void qw_turn_text(Glm53fEngine *e, int first, const char *level, const char *msg, char *out, size_t cap)
{
    (void)e;
    const char *sys = NULL;
    if (!strcmp(level, "xhigh"))
        sys = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
              "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    else if (!strcmp(level, "low"))
        sys = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
              "without unnecessary elaboration.";
    if (first)
        snprintf(out, cap, "%s%s%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n",
                 sys ? "<|im_start|>system\n" : "", sys ? sys : "", sys ? "<|im_end|>\n" : "", msg);
    else
        snprintf(out, cap, "<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n", msg);
}

#ifdef Q38_CUDA
static int qw_state_save(Glm53fEngine *e, const char *path)
{
    QwenEngine *q = (QwenEngine *)e->impl;
    return q->gpu >= 0 ? q38_gpu_state_file(q->m, path, 1) : -1;
}

static int qw_state_load(Glm53fEngine *e, const char *path)
{
    QwenEngine *q = (QwenEngine *)e->impl;
    const int n = q->gpu >= 0 ? q38_gpu_state_file(q->m, path, 0) : -1;
    e->cached = q->m->n_past;
    return n;
}
#endif

static void qw_describe(Glm53fEngine *e, int what, char *buf, size_t cap)
{
    QwenEngine *q = (QwenEngine *)e->impl;
    const q38_cfg *c = &q->m->c;
    int n_attn = 0;
    for (int i = 0; i < c->n_layer; i++) n_attn += !q->m->L[i].recurrent;
    if (what == GLM53F_DESC_BANNER)
        snprintf(buf, cap, "%d layers · %d DeltaNet + %d attention · %d experts top-%d", c->n_layer, c->n_layer - n_attn,
                 n_attn, c->n_exp, c->n_exp_used);
    else if (what == GLM53F_DESC_SETTINGS)
        snprintf(buf, cap, "GGUF weights as stored · expert cache %.1f GB · trunk on %s · %d threads",
                 q->m->cache_budget / 1e9, q->gpu >= 0 ? (q->gpu ? "gpu 1" : "gpu 0") : "the cpu", q->threads);
    else
        snprintf(buf, cap, "expert cache %.1f of %.1f GB (%lld experts) · trunk on %s · %d threads · disk busy %d%%%s",
                 q->m->cache_used / 1e9, q->m->cache_budget / 1e9, (long long)q->m->cache_count,
                 q->gpu >= 0 ? (q->gpu ? "gpu 1" : "gpu 0") : "the cpu", q->threads, q38_gguf_disk_busy(),
                 q38_gguf_disk_pause_ms() ? " (reads paused to keep it below the limit)" : "");
}

static void qw_counters(Glm53fEngine *e, Glm53fCounters *o)
{
    QwenEngine *q = (QwenEngine *)e->impl;
    memset(o, 0, sizeof *o);
    o->hits = (uint64_t)q->m->cache_hits;
    o->misses = (uint64_t)q->m->cache_misses;
    o->bytes = (uint64_t)q->m->expert_bytes_read;
    o->wait = q->m->t_read;
}

/* ------------------------------------------------------------------ loading ---- */
/* While the model opens (half a minute), one animated line instead of the engine's log; the
 * log goes to %LOCALAPPDATA%\localcode\engine.log, and comes back to the screen when the
 * opening fails. LOCALCODE_VERBOSE=1 keeps the log on the screen. */
static struct {
    volatile int on;
    volatile const char *stage;
    double t0;
    int saved_err;                      /* stderr's descriptor before the log file took it */
    char log[1024];
#ifdef _WIN32
    HANDLE th;
#endif
} LD = { 0, "", 0.0, -1, "" };

#ifdef _WIN32
static DWORD WINAPI load_anim(LPVOID arg)
{
    (void)arg;
    static const char *spin[] = { "·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢" };
    for (int i = 0; LD.on; i++) {
        printf("\r\033[2K  \033[38;2;72;222;128m%s \033[1mlocalcode\033[22m\033[38;2;138;145;140m · %s… %.0f s\033[0m",
               spin[i % 10], (const char *)LD.stage, now_s() - LD.t0);
        fflush(stdout);
        Sleep(120);
    }
    return 0;
}
#endif

static void load_begin(void)
{
    LD.t0 = now_s();
#ifdef _WIN32
    DWORD mode;
    const int console = GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &mode) != 0;
    const char *v = getenv("LOCALCODE_VERBOSE");
    const char *base = getenv("LOCALAPPDATA");
    if (console && !(v && *v && strcmp(v, "0")) && base) {
        char dir[900];
        snprintf(dir, sizeof dir, "%s\\localcode", base);
        CreateDirectoryA(dir, NULL);
        snprintf(LD.log, sizeof LD.log, "%s\\engine.log", dir);
        FILE *f = fopen(LD.log, "wb");
        if (f) {
            fflush(stderr);
            LD.saved_err = _dup(2);
            _dup2(_fileno(f), 2);
            fclose(f);
        }
        SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), mode | 0x0004);   /* VT colours */
        LD.on = 1;
        LD.th = CreateThread(NULL, 0, load_anim, NULL, 0, NULL);
        return;
    }
#endif
    printf("  loading the model…");
    fflush(stdout);
}

static void load_stage(const char *what) { LD.stage = what; }

static void load_end(int ok)
{
#ifdef _WIN32
    if (LD.on) {
        LD.on = 0;
        if (LD.th) { WaitForSingleObject(LD.th, 2000); CloseHandle(LD.th); LD.th = NULL; }
        printf("\r\033[2K");
        fflush(stdout);
    }
    if (LD.saved_err >= 0) {
        fflush(stderr);
        _dup2(LD.saved_err, 2);
        _close(LD.saved_err);
        LD.saved_err = -1;
        if (!ok) {                      /* the reason, from the log */
            FILE *f = fopen(LD.log, "rb");
            if (f) {
                char buf[4096];
                size_t n;
                while ((n = fread(buf, 1, sizeof buf, f)) > 0) fwrite(buf, 1, n, stderr);
                fclose(f);
            }
        }
    }
#endif
    (void)ok;
}

/* ----------------------------------------------------------------- variants ---- */
/* The models on this machine (list_variants: the folders of %USERPROFILE%\models), named by
 * their folders (Qwen3.6-35B-A3B-MTP, Qwen3.8-Flash-Next/UD-IQ4_XS). The one opened last is
 * remembered in %LOCALAPPDATA%\localcode\model.txt and opens by default. */
typedef struct { char name[96], path[1100], info[48]; double gb; } Variant;

#ifdef _WIN32
/* A folder holding a model's GGUF (its vision projector, mmproj*.gguf, aside) becomes variant
 * name; returns 1 when it held one. */
static int variant_dir(const char *dir, const char *name, Variant *x)
{
    char gp[1200];
    snprintf(gp, sizeof gp, "%s\\*.gguf", dir);
    WIN32_FIND_DATAA g;
    HANDLE hg = FindFirstFileA(gp, &g);
    if (hg == INVALID_HANDLE_VALUE) return 0;
    memset(x, 0, sizeof *x);
    snprintf(x->name, sizeof x->name, "%s", name);
    do {
        if (!_strnicmp(g.cFileName, "mmproj", 6)) continue;
        x->gb += ((double)g.nFileSizeHigh * 4294967296.0 + (double)g.nFileSizeLow) / 1073741824.0;   /* GiB, as Explorer shows */
        /* the first shard of a split, else the first file */
        if (!x->path[0] || strstr(g.cFileName, "-00001-of-")) snprintf(x->path, sizeof x->path, "%s\\%s", dir, g.cFileName);
    } while (FindNextFileA(hg, &g));
    FindClose(hg);
    snprintf(x->info, sizeof x->info, "%.0f GB", x->gb);
    return x->path[0] != 0;
}
#endif

/* The models on this machine: every folder of LOCALCODE_MODELS_DIR (default %USERPROFILE%\models)
 * holding a GGUF, or each of its subfolders that does (a model with several quantisations:
 * Qwen3.8-Flash-Next/UD-IQ4_XS), largest first. */
static int list_variants(Variant *v, int max)
{
    int n = 0;
#ifdef _WIN32
    char base[1024];
    const char *e = getenv("LOCALCODE_MODELS_DIR"), *home = getenv("USERPROFILE");
    if (e && *e) snprintf(base, sizeof base, "%s", e);
    else if (home) snprintf(base, sizeof base, "%s\\models", home);
    else return 0;
    char pat[1100];
    snprintf(pat, sizeof pat, "%s\\*", base);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0] == '.' || n >= max) continue;
        char dir[1100];
        snprintf(dir, sizeof dir, "%s\\%s", base, fd.cFileName);
        if (variant_dir(dir, fd.cFileName, &v[n])) { n++; continue; }
        char sp[1200];                           /* one level down: the quantisations */
        snprintf(sp, sizeof sp, "%s\\*", dir);
        WIN32_FIND_DATAA sd;
        HANDLE hs = FindFirstFileA(sp, &sd);
        if (hs == INVALID_HANDLE_VALUE) continue;
        do {
            if (!(sd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || sd.cFileName[0] == '.' || n >= max) continue;
            char sub[1400], nm[200];
            snprintf(sub, sizeof sub, "%s\\%s", dir, sd.cFileName);
            snprintf(nm, sizeof nm, "%s/%s", fd.cFileName, sd.cFileName);
            if (variant_dir(sub, nm, &v[n])) n++;
        } while (FindNextFileA(hs, &sd));
        FindClose(hs);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    for (int i = 0; i < n; i++)                  /* largest first */
        for (int j = i + 1; j < n; j++)
            if (v[j].gb > v[i].gb) { Variant t = v[i]; v[i] = v[j]; v[j] = t; }
#else
    (void)v;
    (void)max;
#endif
    return n;
}

static void remembered_path(char *out, size_t cap)
{
    const char *base = getenv("LOCALAPPDATA");
    snprintf(out, cap, "%s\\localcode\\model.txt", base ? base : ".");
}

/* the variant a name picks: exact, else the first whose name holds it (any case) */
static int find_variant(const Variant *v, int n, const char *name)
{
    for (int i = 0; i < n; i++) if (!_stricmp(v[i].name, name)) return i;
    for (int i = 0; i < n; i++) {
        for (const char *h = v[i].name; *h; h++) {
            const char *a = name, *b = h;
            while (*a && *b && (*a | 32) == (*b | 32)) { a++; b++; }
            if (!*a) return i;
        }
    }
    return -1;
}

/* The variant name of a model file: its folder's name, else the file's without extension. */
static void variant_of(const char *path, char *out, size_t cap)
{
    char p[1200];
    snprintf(p, sizeof p, "%s", path);
    char *slash = strrchr(p, '\\'), *s2 = strrchr(p, '/');
    if (s2 > slash) slash = s2;
    if (slash) {
        *slash = 0;
        char *up = strrchr(p, '\\'), *u2 = strrchr(p, '/');
        if (u2 > up) up = u2;
        snprintf(out, cap, "%s", up ? up + 1 : p);
    } else {
        snprintf(out, cap, "%s", p);
        char *dot = strrchr(out, '.');
        if (dot) *dot = 0;
    }
}

/* /model asked for a file that does not open: say why (load_end showed the log) and open the
 * previous one again. Returns 1 when there is one to go back to. */
static int back_to_previous(q38_model *m, const char **path, char *prev)
{
    q38_model_close(m);
    if (!prev[0]) return 0;
    char name[128];
    variant_of(prev, name, sizeof name);
    printf("localcode · esse modelo não abre neste motor ainda; voltando para %s…\n", name);
    fflush(stdout);
    static char back[1200];
    snprintf(back, sizeof back, "%s", prev);
    *path = back;
    prev[0] = 0;
    return 1;
}

static void usage(void)
{
    fprintf(stderr,
            "localcode - the Qwen hybrid models on this engine (Qwen3.8-Flash-Next, Qwen3.6-35B-A3B)\n\n"
            "  localcode [--model NAME|FILE] [--ctx N] [--gen N] [--level 0|1|2] [--continue] [--reasoning xhigh|medium|low]\n\n"
            "--model: a model by name (Qwen3.6-35B-A3B-MTP, or part of it: 35b, iq4) or any shard of a GGUF; the models\n"
            "         are the folders of %%USERPROFILE%%\\models (LOCALCODE_MODELS_DIR); without it, the one opened last\n"
            "         (/model switches inside localcode), else LOCALCODE_MODEL, else the largest\n"
            "--level: 0 chat, 1 quick (reads the files of this folder), 2 agent (edits files, runs commands; default)\n"
            "--continue: go back to the latest conversation of this folder (as /resume)\n"
            "The folder it starts in is the agent's working folder (install-cli.bat puts `localcode` on the PATH).\n");
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    const char *path = NULL, *want = NULL;
    int ctx = 0, gen = 2048, threads = 0, agent_level = 2;
    double cache_gb = 0;
    int gpu = -1;                       /* -1: try GPU 1 then 0; -2: --cpu */
    double preload_gb = 24.0;           /* experts read into RAM at start, most used first */
    const char *level = "xhigh";
    const char *resume = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reasoning") && i + 1 < argc) level = argv[++i];
        else if (!strcmp(argv[i], "--level") && i + 1 < argc) agent_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) gpu = -2;
        else if (!strcmp(argv[i], "--continue") || !strcmp(argv[i], "-c")) resume = "1";
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) want = argv[++i];
        else if (!strcmp(argv[i], "--preload-gb") && i + 1 < argc) preload_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
        else if (argv[i][0] != '-') path = argv[i];
        else { usage(); return 2; }
    }
    /* which file: --model or a path given, else the one opened last, else LOCALCODE_MODEL,
     * else the largest variant */
    static Variant vars[16];
    const int nvars = list_variants(vars, 16);
    if (!path && want) {
        FILE *f = fopen(want, "rb");
        if (f) { fclose(f); path = want; }
        else {
            const int k = find_variant(vars, nvars, want);
            if (k < 0) {
                fprintf(stderr, "no model matches \"%s\"; on this machine:\n", want);
                for (int i = 0; i < nvars; i++) fprintf(stderr, "  %s  %s\n", vars[i].name, vars[i].info);
                return 2;
            }
            path = vars[k].path;
        }
    }
    if (!path) {
        char rp[1100], name[128] = "";
        remembered_path(rp, sizeof rp);
        FILE *f = fopen(rp, "rb");
        if (f) {
            if (fgets(name, sizeof name, f)) name[strcspn(name, "\r\n")] = 0;
            fclose(f);
        }
        /* by name, or a name it holds (model.txt from before the models folder was one up) */
        const int k = name[0] ? find_variant(vars, nvars, name) : -1;
        if (k >= 0) path = vars[k].path;
    }
    if (!path) path = getenv("LOCALCODE_MODEL");
    if (!path && nvars) path = vars[0].path;
    if (!path) { usage(); return 2; }
    int li = -1;
    for (int i = 0; i < 3; i++) if (!strcmp(level, QWEN_LEVELS[i].name)) li = i;
    if (!strcmp(level, "high")) li = 0;
    if (li < 0) { fprintf(stderr, "--reasoning takes xhigh, medium or low\n"); return 2; }
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    threads = omp_get_max_threads();
#else
    threads = 1;
#endif

    static q38_model m;
    static char next_path[1200], prev_path[1200];
    int rc = 1;
    for (;;) {                          /* once per model: /model closes it and opens another */
    memset(&m, 0, sizeof m);
    memset(&g_dec, 0, sizeof g_dec);
    char variant[200];
    variant_of(path, variant, sizeof variant);
    for (int i = 0; i < nvars; i++) if (!_stricmp(vars[i].path, path)) snprintf(variant, sizeof variant, "%s", vars[i].name);
    const double t0 = now_s();
    load_begin();
    load_stage("abrindo o modelo");
    if (q38_model_open(&m, path, ctx)) { load_end(0); if (back_to_previous(&m, &path, prev_path)) continue; return 1; }
    load_stage("levando o tronco para a GPU");
    int on_gpu = -1;
#ifdef Q38_CUDA
    if (gpu >= 0) {
        if (q38_gpu_open(&m, gpu) == 0) on_gpu = gpu;
    } else if (gpu == -1) {
        if (q38_gpu_open(&m, 1) == 0) on_gpu = 1;
        else if (q38_gpu_open(&m, 0) == 0) on_gpu = 0;
    }
#endif
    if (cache_gb > 0) q38_model_set_cache(&m, cache_gb);
    q38_tok *tok = q38_tok_load(&m.g);
    if (!tok) { load_end(0); if (back_to_previous(&m, &path, prev_path)) continue; return 1; }

    QwenEngine q;
    memset(&q, 0, sizeof q);
    q.m = &m;
    q.tok = tok;
#ifdef _OPENMP
    q.threads = omp_get_max_threads();     /* after q38_model_open, which may have lowered it */
#else
    q.threads = threads;
#endif
    q.gpu = on_gpu;
    /* the usage profile of earlier sessions: warm the caches before the first question */
    char prof[1024] = "";
    {
        const char *base = getenv("LOCALAPPDATA");
        if (base) {
            char dir[900];
            snprintf(dir, sizeof dir, "%s\\localcode", base);
#ifdef _WIN32
            CreateDirectoryA(dir, NULL);
#endif
            /* one per model shape (the experts' numbering is the model's); Flash-Next's from
             * before keeps its old name */
            snprintf(prof, sizeof prof, "%s\\expert_profile-%s-%dx%d.bin", dir, m.c.arch, m.c.n_layer, m.c.n_exp);
            FILE *pf = fopen(prof, "rb");
            if (pf) fclose(pf);
            else if (!strcmp(m.c.arch, "qwen4exp")) snprintf(prof, sizeof prof, "%s\\qwen38_expert_profile.bin", dir);
        }
    }
    if (*prof && q38_model_profile_load(&m, prof) == 0 && preload_gb > 0) {
        const double tp = now_s();
        load_stage("aquecendo o cache de especialistas");
        const int nread = q38_model_preload(&m, preload_gb);
        int nv = 0;
#ifdef Q38_CUDA
        if (on_gpu >= 0) nv = q38_gpu_prefill(&m);
#endif
        fprintf(stderr, "\rq38: warmed %d experts into RAM (%.1f GB) and %d into VRAM in %.1f s\n", nread,
                m.cache_used / 1e9, nv, now_s() - tp);
    }
    load_end(1);
    const int ids[] = { (int)q38_gguf_int(&m.g, "tokenizer.ggml.eos_token_id", -1), q38_tok_id(tok, "<|im_end|>"),
                        q38_tok_id(tok, "<|endoftext|>") };
    for (int i = 0; i < 3; i++) {
        int dup = ids[i] < 0;
        for (int j = 0; j < q.n_eos; j++) if (q.eos[j] == ids[i]) dup = 1;
        if (!dup) q.eos[q.n_eos++] = ids[i];
    }

    Glm53fEngine e;
    memset(&e, 0, sizeof e);
    /* the name shown is the one the user controls (the model's folder, else its file), not the
     * GGUF's general.name, which the quantiser wrote and a rename does not change */
    e.name = variant;
    e.levels = QWEN_LEVELS;
    e.n_levels = 3;
    e.n_vocab = m.c.n_vocab;
    e.cap = m.max_ctx;
    e.impl = &q;
    e.forward = qw_forward;
    /* speculative decoding when the model has an MTP block on the GPU (LOCALCODE_SPEC=0: off) */
    if (m.mtp && on_gpu >= 0 && !(getenv("LOCALCODE_SPEC") && !atoi(getenv("LOCALCODE_SPEC")))) e.forward_spec = qw_forward_spec;
    e.reset = qw_reset;
    e.encode = qw_encode;
    e.decode = qw_decode;
    e.is_eos = qw_is_eos;
    e.turn_text = qw_turn_text;
    e.describe = qw_describe;
    e.counters = qw_counters;
    e.agent = 1;                        /* the GGUF template is Qwen's, with <tool_call> functions */
#ifdef Q38_CUDA
    if (on_gpu >= 0) { e.state_save = qw_state_save; e.state_load = qw_state_load; }
#endif
    e.variant = variant;
    e.n_variants = nvars;
    for (int i = 0; i < nvars; i++) { e.variant_name[i] = vars[i].name; e.variant_info[i] = vars[i].info; }
    {   /* the one that opens next time */
        char rp[1100];
        remembered_path(rp, sizeof rp);
        FILE *f = fopen(rp, "wb");
        if (f) { fprintf(f, "%s\n", variant); fclose(f); }
    }

    Glm53fChat s;
    memset(&s, 0, sizeof s);
    s.e = &e;
    s.reasoning = QWEN_LEVELS[li].name;
    s.agent_level = agent_level < 0 ? 0 : agent_level > 2 ? 2 : agent_level;
    if (lc_tools_init(NULL) != 0) s.agent_level = 0;   /* no working directory: chat only */
    if (s.agent_level && lc_auto_dir(lc_tools_root())) s.autonomous = s.auto_approve = 1;   /* /auto here before */
    if (gen >= e.cap) gen = e.cap / 2;
    s.gen = gen;
    {   /* Qwen's sampling with thinking; LOCALCODE_TEMP=0 is greedy */
        const char *tv = getenv("LOCALCODE_TEMP");
        s.smp.temp = tv ? (float)atof(tv) : 0.6f;
        s.smp.top_p = 0.95f;
        s.smp.top_k = 20;
        s.smp.rng = (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ull ^ (uint64_t)(uintptr_t)&s;
        if (getenv("LOCALCODE_SEED")) s.smp.rng = (uint64_t)strtoull(getenv("LOCALCODE_SEED"), NULL, 10);
    }
    s.resume = resume;
    s.load_s = now_s() - t0;
    printf("\r%*s\r", 40, "");
    s.ids_cap = 1 << 20;
    s.ids = (int *)malloc((size_t)s.ids_cap * sizeof(int));
    s.out = (int *)malloc((size_t)GLM53F_CHAT_MAX_GEN * sizeof(int));
    s.text_cap = GLM53F_CHAT_MAX_GEN * 64 + 1;
    s.text = (char *)malloc((size_t)s.text_cap);
    s.logits = (float *)malloc((size_t)m.c.n_vocab * sizeof(float));
    rc = 1;
    if (s.ids && s.out && s.text && s.logits) rc = glm53f_chat_run(&s);
    free(s.ids);
    free(s.out);
    free(s.text);
    free(s.logits);
    if (*prof) q38_model_profile_save(&m, prof);
    if (getenv("LOCALCODE_TIMES") && g_dec.n) {
        const double k = 1.0 / g_dec.n;
        fprintf(stderr, "q38: %d generated tokens, %.4f s each: disk %.4f, cpu experts %.4f, gpu wait %.4f, ple %.4f, tail %.4f,"
                        " other %.4f; cpu experts per token %.1f (%.1f from the disk)\n",
                g_dec.n, g_dec.all * k, g_dec.read * k, g_dec.experts * k, g_dec.gpu * k, g_dec.ple * k, g_dec.tail * k,
                (g_dec.all - g_dec.read - g_dec.experts - g_dec.gpu - g_dec.ple - g_dec.tail) * k,
                (g_dec.hits + g_dec.misses) * k, g_dec.misses * k);
    }
    q38_tok_free(tok);
    q38_model_close(&m);
    if (rc != 0 || !s.switch_to || s.switch_to > nvars) break;
    /* /model: the other file opens, and the conversation (when there was one) goes on in it
     * from its transcript */
    snprintf(prev_path, sizeof prev_path, "%s", path);
    snprintf(next_path, sizeof next_path, "%s", vars[s.switch_to - 1].path);
    path = next_path;
    resume = s.turns > 0 ? "1" : NULL;
    printf("localcode · trocando para %s…\n", vars[s.switch_to - 1].name);
    fflush(stdout);
    }
    return rc;
}
