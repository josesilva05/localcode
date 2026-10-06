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
    timespec_get(&ts, TIME_UTC);
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

static void usage(void)
{
    fprintf(stderr,
            "localcode - Qwen3.8-Flash-Next on this engine\n\n"
            "  localcode [model.gguf] [--ctx N] [--gen N] [--level 0|1|2] [--reasoning xhigh|medium|low] [--threads N]\n\n"
            "--level: 0 chat, 1 quick (reads the files of this folder), 2 agent (edits files, runs commands; default)\n"
            "The model is any shard of the split GGUF, or LOCALCODE_MODEL in the environment.\n");
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    const char *path = getenv("LOCALCODE_MODEL");
    int ctx = 0, gen = 2048, threads = 0, agent_level = 2;
    double cache_gb = 0;
    int gpu = -1;                       /* -1: try GPU 1 then 0; -2: --cpu */
    double preload_gb = 24.0;           /* experts read into RAM at start, most used first */
    const char *level = "xhigh";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reasoning") && i + 1 < argc) level = argv[++i];
        else if (!strcmp(argv[i], "--level") && i + 1 < argc) agent_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) gpu = -2;
        else if (!strcmp(argv[i], "--preload-gb") && i + 1 < argc) preload_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
        else if (argv[i][0] != '-') path = argv[i];
        else { usage(); return 2; }
    }
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
    const double t0 = now_s();
    printf("  loading Qwen3.8-Flash-Next…");
    fflush(stdout);
    if (q38_model_open(&m, path, ctx)) return 1;
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
    if (!tok) return 1;

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
            snprintf(prof, sizeof prof, "%s\\qwen38_expert_profile.bin", dir);
        }
    }
    if (*prof && q38_model_profile_load(&m, prof) == 0 && preload_gb > 0) {
        const double tp = now_s();
        printf("\r  warming the expert cache…");
        fflush(stdout);
        const int nread = q38_model_preload(&m, preload_gb);
        int nv = 0;
#ifdef Q38_CUDA
        if (on_gpu >= 0) nv = q38_gpu_prefill(&m);
#endif
        fprintf(stderr, "\rq38: warmed %d experts into RAM (%.1f GB) and %d into VRAM in %.1f s\n", nread,
                m.cache_used / 1e9, nv, now_s() - tp);
    }
    const int ids[] = { (int)q38_gguf_int(&m.g, "tokenizer.ggml.eos_token_id", -1), q38_tok_id(tok, "<|im_end|>"),
                        q38_tok_id(tok, "<|endoftext|>") };
    for (int i = 0; i < 3; i++) {
        int dup = ids[i] < 0;
        for (int j = 0; j < q.n_eos; j++) if (q.eos[j] == ids[i]) dup = 1;
        if (!dup) q.eos[q.n_eos++] = ids[i];
    }

    Glm53fEngine e;
    memset(&e, 0, sizeof e);
    e.name = "Qwen3.8-Flash-Next";
    e.levels = QWEN_LEVELS;
    e.n_levels = 3;
    e.n_vocab = m.c.n_vocab;
    e.cap = m.max_ctx;
    e.impl = &q;
    e.forward = qw_forward;
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

    Glm53fChat s;
    memset(&s, 0, sizeof s);
    s.e = &e;
    s.reasoning = QWEN_LEVELS[li].name;
    s.agent_level = agent_level < 0 ? 0 : agent_level > 2 ? 2 : agent_level;
    if (lc_tools_init(NULL) != 0) s.agent_level = 0;   /* no working directory: chat only */
    if (gen >= e.cap) gen = e.cap / 2;
    s.gen = gen;
    s.load_s = now_s() - t0;
    printf("\r%*s\r", 40, "");
    s.ids_cap = 1 << 20;
    s.ids = (int *)malloc((size_t)s.ids_cap * sizeof(int));
    s.out = (int *)malloc((size_t)GLM53F_CHAT_MAX_GEN * sizeof(int));
    s.text_cap = GLM53F_CHAT_MAX_GEN * 64 + 1;
    s.text = (char *)malloc((size_t)s.text_cap);
    s.logits = (float *)malloc((size_t)m.c.n_vocab * sizeof(float));
    int rc = 1;
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
    return rc;
}
