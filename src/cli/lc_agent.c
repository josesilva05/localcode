/* lc_agent.c - localcode's agent turn: the model works with tools until it answers.
 *
 * Levels (/level): 1 quick, the reading tools and low reasoning; 2 agent, every tool
 * (lc_tools.c) with medium reasoning. Writes and commands ask first unless /approve auto.
 *
 * The conversation is one token stream that only grows, in the chat template's own form
 * (tokenizer.chat_template of Qwen3.8-Flash-Next with tools): the system block lists the
 * functions in <tools>; the model answers with optional text and <tool_call> blocks and
 * stops on <|im_end|>; the results go back as one user turn of <tool_response> blocks; the
 * next assistant turn opens with <think>. The reasoning stays in the context, as the
 * template's preserve_thinking keeps it, so nothing already read is read again.
 */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "glm53f_chat.h"
#include "lc_tools.h"
#include "lc_hooks.h"

#define MAX_CALLS   8          /* tool calls carried out from one model answer */
#define ROOM        1024       /* positions kept free for the next step's start */

/* what /auto adds to each request of the folder */
static const char AUTONOMOUS_NOTE[] =
    "\n\n[localcode] Modo autônomo nesta pasta: não faça perguntas nem peça confirmação ao usuário. "
    "Quando algo for ambíguo, escolha a opção mais razoável, diga qual escolheu e siga até terminar a tarefa "
    "(com o teste passando, se houver).";

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

typedef struct { char *p; size_t n, cap; } Str;

static void s_add(Str *s, const char *p, size_t n)
{
    if (s->n + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap : 4096;
        while (s->n + n + 1 > cap) cap *= 2;
        char *g = (char *)realloc(s->p, cap);
        if (!g) return;
        s->p = g;
        s->cap = cap;
    }
    memcpy(s->p + s->n, p, n);
    s->n += n;
    s->p[s->n] = 0;
}

static void s_str(Str *s, const char *p) { s_add(s, p, strlen(p)); }

static void s_fmt(Str *s, const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) s_add(s, buf, (size_t)(n < (int)sizeof buf ? n : (int)sizeof buf - 1));
}

/* A tool result within what the context has left: at most half of the free positions
 * (about 3 characters each, after what this step already holds and the answer's budget),
 * never under 1500 characters; cut at a line start with a note on how to read the rest.
 * Three files of 4000 lines filled an 8k context in one step and stopped the turn. */
static void fit_result(const Glm53fEngine *e, size_t pending, int gen, const char *res, Str *t)
{
    const long free_tok = (long)e->cap - e->cached - ROOM - gen - (long)(pending / 3);
    size_t lim = free_tok > 0 ? (size_t)free_tok * 3 / 2 : 0;
    if (lim < 1500) lim = 1500;
    const size_t n = strlen(res);
    if (n <= lim) { s_add(t, res, n); return; }
    size_t cut = lim;
    while (cut > lim / 2 && res[cut - 1] != '\n') cut--;
    if (res[cut - 1] != '\n') { cut = lim; while (cut > 0 && ((unsigned char)res[cut] & 0xC0) == 0x80) cut--; }
    s_add(t, res, cut);
    s_fmt(t, "\n[localcode] cut to fit the context: %zu of %zu characters shown. Read the rest in smaller pieces "
             "(read_file with a line range, grep for what you need).", cut, n);
}

static int utf8_complete(const char *s, int n)
{
    int i = n, back = 0;
    while (i > 0 && back < 4 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) { i--; back++; }
    if (i == 0) return n;
    const unsigned char lead = (unsigned char)s[i - 1];
    const int need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return (back + 1 >= need) ? n : i - 1;
}

/* ------------------------------------------------------------- the prompt ---- */

/* the template's text around the <tools> block, as it renders it */
static const char FORMAT_NOTE[] =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

static const char LOW_EFFORT[] =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
    "unnecessary elaboration.";

static void system_block(Str *t, int level)
{
    s_str(t, "<|im_start|>system\n");
    if (level == 1) { s_str(t, LOW_EFFORT); s_str(t, "\n\n"); }
    s_str(t, "# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
    s_str(t, lc_tools_json(level));
    s_str(t, "\n</tools>");
    s_str(t, FORMAT_NOTE);
    s_str(t, "\n\n");
    if (level == 1) {
        s_fmt(t, "You are localcode, answering questions about the project in %s on the user's computer. "
                 "You can list, find, search and read its files with the functions above (paths relative to that "
                 "folder); you cannot change files or run programs. Read what you need, then answer briefly and "
                 "concretely in the user's language, naming the files and lines you rely on.", lc_tools_root());
    } else {
        s_fmt(t, "You are localcode, a coding agent working in the folder %s on the user's Windows computer, "
                 "through the functions above (paths relative to that folder; run uses PowerShell).\n", lc_tools_root());
        s_str(t, "For a task that changes code, work in this order:\n");
        if (*lc_skills_block())
            s_str(t, "0. Skills: when a skill listed below covers the task, or a method the user names (TDD, a spec, a plan, "
                     "a review, a design ...), call skill with its name before anything else, and follow it through the "
                     "steps below.\n");
        s_str(t, "1. Look: list, search and read the files involved.\n"
                 "2. Plan: call plan with the steps and a test_command that proves the task works (the project's "
                 "tests, a build, or running the program with a check; write a small check when there is none).\n"
                 "3. Code: the smallest correct change with edit_file; write_file only for new files or full rewrites.\n"
                 "4. Test: call test after the change.\n"
                 "5. If it fails: debug (read the error, find its cause in the code), fix, and call test again. "
                 "At most 5 failed attempts; then stop and say what still fails.\n"
                 "6. When the test passes, finish at once: a short answer with what you changed and the test that passed.\n"
                 "A question that changes nothing is answered directly, without plan or test. The user may decline a "
                 "change or a command; then take another way or ask. Answer in the user's language.");
    }
    if (*lc_skills_block()) {
        s_str(t, "\n\nSkills (instructions for kinds of tasks, from the plugins that are on):\n");
        s_str(t, lc_skills_block());
        s_str(t, "When a skill covers the task, call skill with its name first and follow it; where its steps differ "
                 "from the workflow above, the skill's win. When none listed fits but one might exist, skill_search "
                 "with a few words of the task finds more.");
    }
    s_str(t, "\nWhen the user writes in Portuguese, answer in Brazilian Portuguese (pt-BR: \"arquivo\", \"rodar\", "
             "\"você\", \"me diga\"), never European Portuguese (\"ficheiro\", \"correr\", \"tu\", \"diz-me\").");
    s_str(t, "<|im_end|>\n");
}

/* The system block read once and kept in a file: every new conversation of a level starts
 * with the same tokens, so they are loaded instead of read again (a minute on this engine).
 * The key covers the tokens, the context size and this build. Returns 1 when the block is
 * in the engine state (loaded or read and kept), 0 when the caller must read it. */
static int system_from_file(Glm53fChat *s, const Str *sys, Glm53fTurnSink *k)
{
    Glm53fEngine *e = s->e;
    if (!e->state_save || !e->state_load || e->cached) return 0;
    const int n = e->encode(e, sys->p, (int)sys->n, s->ids, s->ids_cap);
    if (n <= 0 || n + ROOM >= e->cap) return 0;
    s->base = n;
    unsigned long long h = 1469598103934665603ULL;            /* FNV-1a */
    const char *build = __DATE__ __TIME__;
    for (const char *p = build; *p; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
    h ^= (unsigned)e->cap; h *= 1099511628211ULL;
    for (const char *p = e->variant ? e->variant : ""; *p; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
    for (int i = 0; i < n; i++) { h ^= (unsigned)s->ids[i]; h *= 1099511628211ULL; }
    const char *base = getenv("LOCALAPPDATA");
    if (!base) return 0;
    char dir[1024], path[1200];
    snprintf(dir, sizeof dir, "%s/localcode/prefix", base);
    snprintf(path, sizeof path, "%s/agent-%016llx.q38s", dir, h);
    const double t0 = now_s();
    k->status(k, "Loading the agent's instructions", t0);
    if (e->state_load(e, path) == n) { k->status(k, NULL, 0.0); return 1; }
    e->reset(e);                                /* a missing or stale file */
    char line[96];
    snprintf(line, sizeof line, "Reading the agent's instructions (%d tokens, once)", n);
    k->status(k, line, t0);
    if (e->forward(e, s->ids, n, s->logits) != 0) { k->status(k, NULL, 0.0); e->reset(e); return 0; }
#ifdef _WIN32
    char top[1024];
    snprintf(top, sizeof top, "%s/localcode", base);
    CreateDirectoryA(top, NULL);
    CreateDirectoryA(dir, NULL);
#endif
    e->state_save(e, path);
    k->status(k, NULL, 0.0);
    return 1;
}

/* The text that opens a new context: the system block (loaded from its file when kept) and
 * the first user message, after what a compaction or a resume left in s->preface. */
static void open_context(Glm53fChat *s, Glm53fTurnSink *k, Str *t, int level, const char *msg)
{
    Str sys = { 0 };
    system_block(&sys, level);
    if (!system_from_file(s, &sys, k)) s_add(t, sys.p, sys.n);
    free(sys.p);
    s_str(t, "<|im_start|>user\n");
    if (s->preface) {
        s_str(t, s->preface);
        free(s->preface);
        s->preface = NULL;
    }
    s_str(t, msg);
    s_str(t, "<|im_end|>\n<|im_start|>assistant\n<think>\n");
}

/* A step's text that would not fit: the model summarises the context, which is emptied, and
 * the step goes on in a new one that opens with the summary and carries the step's own text
 * (the message, or the results of the last tool calls with the request they serve).
 * Returns 0 with *t rebuilt, 1 when there was no room even to summarise, -1 engine failure. */
static int compact_step(Glm53fChat *s, Glm53fTurnSink *k, Str *t, int level, const char *request)
{
    static const char HEAD[] = "<|im_end|>\n<|im_start|>user";
    static const char TAIL[] = "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    if (!t->p || strncmp(t->p, HEAD, sizeof HEAD - 1)) return 1;
    const char *body = t->p + sizeof HEAD - 1, *end = NULL;
    for (const char *q = body; (q = strstr(q, TAIL)); q++) end = q;
    if (!end) return 1;
    const int tools = !strncmp(body, "\n<tool_response>", 16);
    Str b = { 0 };
    if (tools) {
        s_str(&b, "The request in progress, word for word:\n");
        s_str(&b, request);
        s_str(&b, "\n\nYour last tool calls returned:");
        s_add(&b, body, (size_t)(end - body));
        s_str(&b, "\n\nContinue the task from where you were.");
    } else {
        if (*body == '\n') body++;
        s_add(&b, body, (size_t)(end - body));
    }
    const int rc = lc_compact(s, k, tools ? request : NULL, NULL);
    if (rc != 0) { free(b.p); return rc; }
    Str nt = { 0 };
    open_context(s, k, &nt, level, b.p ? b.p : "");
    free(b.p);
    free(t->p);
    *t = nt;
    return 0;
}

/* ----------------------------------------------------------- tool calls ---- */

/* The <tool_call> blocks of an answer (the text after </think>). Returns how many. */
static int parse_calls(const char *post, LcCall *calls, int max)
{
    int n = 0;
    const char *p = post;
    while (n < max && (p = strstr(p, "<tool_call>"))) {
        const char *f = strstr(p, "<function=");
        if (!f) break;
        const char *ne = strchr(f + 10, '>');
        if (!ne) break;
        const char *end = strstr(ne, "</function>");
        const char *close = strstr(ne, "</tool_call>");
        if (!end || (close && close < end)) end = close;
        if (!end) end = ne + strlen(ne);
        LcCall *c = &calls[n];
        memset(c, 0, sizeof *c);
        size_t nl = (size_t)(ne - (f + 10));
        if (nl >= sizeof c->name) nl = sizeof c->name - 1;
        memcpy(c->name, f + 10, nl);
        c->name[nl] = 0;
        const char *q = ne + 1;
        while (c->n < LC_MAX_PARAMS && (q = strstr(q, "<parameter=")) && q < end) {
            const char *ke = strchr(q + 11, '>');
            if (!ke || ke > end) break;
            const char *v = ke + 1;
            const char *ve = strstr(v, "</parameter>");
            if (!ve || ve > end) ve = end;
            if (*v == '\n') v++;
            const char *vz = ve;
            if (vz > v && vz[-1] == '\n') vz--;
            const size_t kl = (size_t)(ke - (q + 11)), vl = vz > v ? (size_t)(vz - v) : 0;
            c->key[c->n] = (char *)malloc(kl + 1);
            c->val[c->n] = (char *)malloc(vl + 1);
            if (!c->key[c->n] || !c->val[c->n]) { free(c->key[c->n]); free(c->val[c->n]); break; }
            memcpy(c->key[c->n], q + 11, kl); c->key[c->n][kl] = 0;
            memcpy(c->val[c->n], v, vl); c->val[c->n][vl] = 0;
            c->n++;
            q = ve + (*ve ? 12 : 0);
        }
        n++;
        p = end;
    }
    return n;
}

static int tool_bit(const char *name)
{
    static const char *names[] = { "write_file", "edit_file", "run" };
    if (!strcmp(name, "test")) return 1 << 2;           /* a command too: run's permission */
    for (int i = 0; i < 3; i++) if (!strcmp(name, names[i])) return 1 << i;
    return 0;
}

/* --------------------------------------------------------- the coding loop ---- */
/* plan -> code -> test -> (debug -> fix -> test)* -> finish, held by the code, not by the
 * model's word: test's exit code decides, a failure after a change is an attempt, MAX_FIX
 * attempts or the same failure three times in a row stop the loop, and an answer given
 * after a change that was never tested is sent back to test (twice at most). */
#define MAX_FIX 5

typedef struct {
    char *test_cmd;          /* the plan's */
    int   planned, plan_hint, edited, edits_since_test, attempts, same, stopped, stopped_calls, nudges, passed;
    unsigned long long last_fail;
} Flow;

/* a failure's output without its last line ("[exit code N · 1.2 s]": the time varies) */
static unsigned long long fail_hash(const char *out)
{
    const char *end = out + strlen(out), *tail = NULL;
    for (const char *p = out; (p = strstr(p, "[exit code")); p++) tail = p;
    if (tail) end = tail;
    unsigned long long h = 1469598103934665603ULL;
    for (const char *p = out; p < end; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
    return h;
}

/* plan: the steps for the screen (numbered) and the test command kept */
static char *flow_plan(Flow *fl, const LcCall *c, char *sum, size_t sc, int *ok)
{
    const char *steps = lc_call_arg(c, "steps"), *tc = lc_call_arg(c, "test_command");
    free(fl->test_cmd);
    fl->test_cmd = NULL;
    if (tc && *tc) {
        fl->test_cmd = (char *)malloc(strlen(tc) + 1);
        if (fl->test_cmd) strcpy(fl->test_cmd, tc);
    }
    fl->planned = 1;
    Str sm = { 0 };
    int k = 0;
    for (const char *p = steps ? steps : ""; *p;) {
        const char *nl = strchr(p, '\n');
        size_t n = nl ? (size_t)(nl - p) : strlen(p);
        const char *q = p;
        while (n && (*q == ' ' || *q == '-' || *q == '*' || (*q >= '0' && *q <= '9') || *q == '.' || *q == ')')) { q++; n--; }
        while (n && (q[n - 1] == '\r' || q[n - 1] == ' ')) n--;
        if (n) { s_fmt(&sm, "%s%d. ", k ? "\n" : "", ++k); s_add(&sm, q, n); }
        p = nl ? nl + 1 : p + strlen(p);
    }
    s_fmt(&sm, "%steste: %s", k ? "\n" : "", fl->test_cmd ? fl->test_cmd : "(nenhum)");
    snprintf(sum, sc, "%s", sm.p ? sm.p : "");
    free(sm.p);
    *ok = 1;
    Str r = { 0 };
    s_fmt(&r, "Plan recorded, %d step%s; test: %s.\n", k, k == 1 ? "" : "s", fl->test_cmd ? fl->test_cmd : "none given");
    s_str(&r, fl->test_cmd ? "Implement it step by step; after the change, call test."
                           : "Give a test_command (call plan again): the task is checked by it.");
    return r.p;
}

/* test: run the command, then the verdict that steers the next step */
static char *flow_test(Flow *fl, const char *cmd, volatile int *stop, char *sum, size_t sc, int *ok)
{
    LcCall rc;
    memset(&rc, 0, sizeof rc);
    snprintf(rc.name, sizeof rc.name, "run");
    rc.key[0] = (char *)"command";
    rc.val[0] = (char *)cmd;
    rc.key[1] = (char *)"timeout_s";
    rc.val[1] = (char *)"600";
    rc.n = 2;
    char rsum[300];
    int rok = 0;
    char *out = lc_tool_run(&rc, stop, rsum, sizeof rsum, &rok);
    Str r = { 0 };
    s_str(&r, out ? out : "(no output)\n");
    const int changed = fl->edits_since_test > 0;       /* an attempt is a test after a new change */
    fl->edits_since_test = 0;
    if (rok) {
        fl->passed = 1;
        s_str(&r, "[localcode] TEST PASSED. The task is done: finish now with a short answer (what you changed, and "
                  "this test passing). Do not change anything else.");
        snprintf(sum, sc, "passou ✓ · %s", rsum);
    } else {
        fl->passed = 0;
        const unsigned long long h = out ? fail_hash(out) : 0;
        fl->same = h == fl->last_fail ? fl->same + 1 : 0;
        fl->last_fail = h;
        if (changed) fl->attempts++;                 /* before any change it only shows the problem */
        if (fl->attempts >= MAX_FIX || fl->same >= 2) {
            fl->stopped = 1;
            if (fl->same >= 2) {
                s_str(&r, "[localcode] STOP: the same failure three times in a row. Do not call more functions: answer "
                          "now with what you tried, what still fails (the error), and what would be needed.");
                snprintf(sum, sc, "falhou · parou: a mesma falha 3 vezes · %s", rsum);
            } else {
                s_fmt(&r, "[localcode] STOP: the test still fails after %d attempts. Do not call more functions: answer "
                          "now with what you tried, what still fails (the error), and what you would try next.", fl->attempts);
                snprintf(sum, sc, "falhou · parou após %d tentativas · %s", fl->attempts, rsum);
            }
        } else if (fl->same == 1) {
            s_fmt(&r, "[localcode] TEST FAILED with the same output as before%s. Re-read the error, question your "
                      "assumption, and try a different fix (attempt %d of %d used).",
                  changed ? ": your last change did not affect it" : " (nothing changed since)", fl->attempts, MAX_FIX);
            snprintf(sum, sc, "falhou igual · %d de %d tentativas · %s", fl->attempts, MAX_FIX, rsum);
        } else if (!fl->edited) {
            s_str(&r, "[localcode] TEST FAILED (before any change: the problem, shown). Find its cause in the code, fix "
                      "it with the smallest change, then call test again.");
            snprintf(sum, sc, "falhou · antes da mudança · %s", rsum);
        } else {
            s_fmt(&r, "[localcode] TEST FAILED. Debug: read the error above, find its cause in the code, fix it with "
                      "the smallest change, then call test again (attempt %d of %d used).", fl->attempts, MAX_FIX);
            snprintf(sum, sc, "falhou · %d de %d tentativas · %s", fl->attempts, MAX_FIX, rsum);
        }
    }
    free(out);
    *ok = rok;
    return r.p;
}

/* What the user sees before allowing a call: the change or the command. */
static void call_detail(const LcCall *c, Str *d)
{
    const char *v;
    if ((!strcmp(c->name, "run") || !strcmp(c->name, "test")) && (v = lc_call_arg(c, "command"))) {
        s_str(d, v);
    } else if (!strcmp(c->name, "edit_file")) {
        const char *o = lc_call_arg(c, "old_string"), *nw = lc_call_arg(c, "new_string");
        s_str(d, "- ");
        s_add(d, o ? o : "", o && strlen(o) > 600 ? 600 : (o ? strlen(o) : 0));
        s_str(d, "\n+ ");
        s_add(d, nw ? nw : "", nw && strlen(nw) > 600 ? 600 : (nw ? strlen(nw) : 0));
    } else if (!strcmp(c->name, "write_file") && (v = lc_call_arg(c, "content"))) {
        const size_t n = strlen(v);
        s_add(d, v, n > 600 ? 600 : n);
        if (n > 600) s_fmt(d, "\n[... %zu more bytes]", n - 600);
    }
}

/* ------------------------------------------------------------- the turn ---- */

/* a hook event's runs on the screen: what blocked, what it says, or that it ran */
static void hook_shown(Glm53fTurnSink *k, const char *ev, const LcHookOut *o)
{
    if (!o->ran || !k->tool) return;
    char title[80], sum[600];
    snprintf(title, sizeof title, "hook %s", ev);
    const char *what = o->block ? o->reason : o->message ? o->message : o->context ? o->context : NULL;
    snprintf(sum, sizeof sum, "%s%.300s", o->block ? "blocked: " : "", what ? what : "ran");
    for (char *p = sum; *p; p++) if (*p == '\n' || *p == '\r') *p = ' ';
    k->tool(k, title, sum, !o->block && !(o->message && strstr(o->message, "failed")));
}

/* context from hooks, as Claude Code gives it to the model */
static void hook_context(Str *t, const LcHookOut *o)
{
    if (!o->context || !*o->context) return;
    s_str(t, "\n\n<system-reminder>\n");
    s_str(t, o->context);
    s_str(t, "\n</system-reminder>");
}

int lc_agent_turn(Glm53fChat *s, const Glm53fInput *in, Glm53fTurnSink *k)
{
    Glm53fEngine *e = s->e;
    volatile int *stop = glm53f_chat_stop_flag();
    const int level = s->agent_level, max_steps = level == 1 ? 12 : 60;
    Flow fl;
    memset(&fl, 0, sizeof fl);
    char line[512];
    *stop = 0;

    Str t = { 0 }, answer = { 0 }, actions = { 0 }, req = { 0 };
    lc_session_user(s, in->shown ? in->shown : in->prompt);
    /* /auto: the request carries the rule (the context's opening stays the same for every folder) */
    s_str(&req, in->prompt);
    if (s->autonomous && level == 2) s_str(&req, AUTONOMOUS_NOTE);
    /* the hooks of the plugins the user allowed and of the project (lc_hooks.c) */
    lc_hooks_load(lc_tools_root());
    if (s->first && lc_hooks_any(LC_HOOK_SESSION_START)) {
        LcHookOut ho;
        lc_hooks_run(LC_HOOK_SESSION_START, "startup", "\"source\": \"startup\"", &ho);
        hook_shown(k, "SessionStart", &ho);
        hook_context(&req, &ho);
        lc_hook_out_free(&ho);
    }
    if (lc_hooks_any(LC_HOOK_USER_PROMPT)) {
        LcHookOut ho;
        char *q = lc_json_quote(in->prompt);
        Str f = { 0 };
        s_str(&f, "\"prompt\": ");
        s_str(&f, q ? q : "\"\"");
        lc_hooks_run(LC_HOOK_USER_PROMPT, NULL, f.p, &ho);
        free(q);
        free(f.p);
        hook_shown(k, "UserPromptSubmit", &ho);
        if (ho.block) {                          /* the message does not go to the model */
            char m[700];
            snprintf(m, sizeof m, "a hook blocked the message: %.600s", ho.reason ? ho.reason : "");
            lc_hook_out_free(&ho);
            free(req.p);
            k->done(k, m, 2);
            return 0;
        }
        hook_context(&req, &ho);
        lc_hook_out_free(&ho);
    }
    const char *prompt = req.p ? req.p : in->prompt;
    if (s->first) {
        open_context(s, k, &t, level, prompt);
    } else {
        s_str(&t, "<|im_end|>\n<|im_start|>user\n");
        s_str(&t, prompt);
        s_str(&t, "<|im_end|>\n<|im_start|>assistant\n<think>\n");
    }

    int steps = 0, calls_done = 0, fatal = 0, failed = 0, interrupted = 0, compacted = 0, stop_hooks = 0;
    long pre_tok = 0, dec_tok = 0;
    double pre_s = 0.0, dec_s = 0.0;
    const double t_turn = now_s();
    const char *why_stop = NULL;

    for (;;) {
        /* read what goes before the model's next answer */
        const int n = e->encode(e, t.p, (int)t.n, s->ids, s->ids_cap);
        if (n <= 0) { why_stop = "could not tokenise the step"; failed = 1; break; }
        /* room kept for the answer before compacting: a quarter of the context up to 8192 for the
         * agent (a file written in one call), /gen up to 4096 for a chat answer */
        const int step_gen = level >= 1 ? (e->cap / 4 < 8192 ? e->cap / 4 : 8192) : (s->gen < 4096 ? s->gen : 4096);
        if (lc_compact_due(s, n + step_gen)) {
            const int rc = compact_step(s, k, &t, level, prompt);
            if (rc < 0) { why_stop = "the engine failed while compacting"; fatal = 1; break; }
            if (rc == 0) { compacted++; continue; }   /* an esc stops at the next answer's start */
        }
        if (e->cached + n + ROOM > e->cap) {
            snprintf(line, sizeof line, "the context is full (%d of %d positions): /reset to start a new conversation",
                     e->cached + n, e->cap);
            why_stop = line;
            failed = 1;
            break;
        }
        snprintf(line, sizeof line, "Reading %d token%s", n, n == 1 ? "" : "s");
        double t0 = now_s();
        k->status(k, line, t0);
        if (e->forward(e, s->ids, n, s->logits) != 0) { k->status(k, NULL, 0.0); why_stop = "the engine failed while reading"; fatal = 1; break; }
        s->first = 0;
        pre_tok += n;
        pre_s += now_s() - t0;
        k->status(k, NULL, 0.0);
        t.n = 0;
        if (t.p) t.p[0] = 0;

        /* the model's answer: reasoning, then text, then maybe tool calls */
        /* the agent's answer may take the whole free context: a tool call writing a file runs past
         * /gen (2048), and was cut in the middle; /gen stays the limit of a plain chat answer */
        int gen = level >= 1 ? e->cap - e->cached - 64 : s->gen;
        if (gen > e->cap - e->cached - 64) gen = e->cap - e->cached - 64;
        if (gen > GLM53F_CHAT_MAX_GEN - 32) gen = GLM53F_CHAT_MAX_GEN - 32;   /* s->out's room */
        int nout = 0, printed = 0, in_think = 1, eos = 0, closed = 0;
        /* the reasoning's budget (/gen, at least 4096): past it the engine closes the thinking itself
         * and the model goes on to answer, as llama.cpp's reasoning budget does; without one a
         * reasoning that loops runs until the context is full (20 minutes on the 27B) */
        const int think_cap = s->gen > 4096 ? s->gen : 4096;
        const double td = now_s();
        /* speculative decoding (forward_spec): the guesses the model agreed with are already fed;
         * they come out of pend one a step, as if generated, before the logits choose again */
        int pend[GLM53F_SPEC_MAX], npend = 0, ipend = 0;
        int next = -1;                                /* a token forward_spec drew, not fed yet */
        for (int step = 0; step < gen; step++) {
            if (*stop) { interrupted = 1; break; }
            const int fed = ipend < npend;            /* this step's token is a guess already fed */
            if (!fed && in_think && !closed && nout >= think_cap && nout + 16 < s->ids_cap) {
                int ids[16];
                const int k2 = e->encode(e, "\n</think>\n\n", 11, ids, 16);
                if (k2 > 0 && nout + k2 < GLM53F_CHAT_MAX_GEN) {
                    for (int i = 0; i < k2; i++) s->out[nout++] = ids[i];
                    closed = 1;
                    next = -1;                        /* its logits are gone */
                    if (e->forward(e, ids, k2, s->logits) != 0) { fatal = 1; break; }
                    if (k->tool) k->tool(k, "raciocínio", "chegou ao limite: o motor fechou o raciocínio e o modelo responde", 1);
                    continue;
                }
            }
            int best;
            if (fed) best = pend[ipend++];
            else if (next >= 0) { best = next; next = -1; }
            else best = glm53f_sample(&s->smp, s->logits, e->n_vocab);
            if (e->is_eos(e, best)) { eos = 1; break; }
            s->out[nout++] = best;
            const int len = e->decode(e, s->out, nout, s->text, s->text_cap - 1);
            s->text[len > 0 ? len : 0] = 0;
            const int ok = utf8_complete(s->text, len);
            if (in_think) {
                const char *close = strstr(s->text, "</think>");
                const int upto = close ? (int)(close - s->text) : ok;
                if (s->show_thinking) {
                    if (upto > printed) k->reasoning(k, s->text + printed, upto - printed);
                } else {
                    snprintf(line, sizeof line, "Thinking · %d token%s", nout, nout == 1 ? "" : "s");
                    k->status(k, line, td);
                }
                if (upto > printed) printed = upto;
                if (close) {
                    k->status(k, NULL, 0.0);
                    k->thought(k, now_s() - td, nout);
                    printed = (int)(close - s->text) + 8;
                    in_think = 0;
                }
            }
            if (!in_think) {
                /* the text before a tool call is shown; the call itself is not. A tail that
                 * could still become "<tool_call>" waits for the next token. */
                const char *tc = strstr(s->text + printed, "<tool_call>");
                int upto = tc ? (int)(tc - s->text) : ok - 11;
                if (upto > ok) upto = ok;
                if (upto > printed) { k->answer(k, s->text + printed, upto - printed); printed = upto; }
                if (tc) { snprintf(line, sizeof line, "Writing a tool call · %d tokens", nout); k->status(k, line, td); }
            }
            if (fed || step + 1 >= gen) continue;
            if (e->forward_spec && gen - step > GLM53F_SPEC_MAX + 1 && !(in_think && !closed && nout + GLM53F_SPEC_MAX >= think_cap)) {
                const int k2 = e->forward_spec(e, best, pend, s->logits, &s->smp, &next);
                if (k2 < 0) { fatal = 1; break; }
                npend = k2;
                ipend = 0;
            } else if (e->forward(e, &best, 1, s->logits) != 0) { fatal = 1; break; }
        }
        k->status(k, NULL, 0.0);
        dec_tok += nout;
        dec_s += now_s() - td;
        if (fatal) { why_stop = "the engine failed while writing"; break; }
        int len = nout ? e->decode(e, s->out, nout, s->text, s->text_cap - 1) : 0;
        if (len < 0) len = 0;
        s->text[len] = 0;
        const char *close = strstr(s->text, "</think>");
        const char *post = close ? close + 8 : NULL;
        LcCall calls[MAX_CALLS];
        const int nc = post ? parse_calls(post, calls, MAX_CALLS) : 0;
        if (!nc && post) {                       /* the rest of the text: the answer */
            if (len > printed) k->answer(k, s->text + printed, len - printed);
            while (*post == '\n' || *post == ' ') post++;
            s_str(&answer, post);
        }
        if (interrupted) { for (int i = 0; i < nc; i++) lc_call_free(&calls[i]); break; }
        if (!eos) {
            for (int i = 0; i < nc; i++) lc_call_free(&calls[i]);
            why_stop = post ? "the answer reached /gen tokens before it ended" : "the reasoning reached /gen tokens before an answer";
            failed = 1;
            break;
        }
        if (!nc) {                               /* answered */
            if (lc_hooks_any(LC_HOOK_STOP) && stop_hooks < 3) {
                /* a Stop hook may send the agent back to work, with its reason */
                LcHookOut ho;
                lc_hooks_run(LC_HOOK_STOP, NULL, stop_hooks ? "\"stop_hook_active\": true" : "\"stop_hook_active\": false", &ho);
                hook_shown(k, "Stop", &ho);
                const int again = ho.block && ho.reason && *ho.reason;
                if (again) {
                    stop_hooks++;
                    answer.n = 0;
                    if (answer.p) answer.p[0] = 0;
                    s_str(&t, "<|im_end|>\n<|im_start|>user\n[hook] ");
                    s_str(&t, ho.reason);
                    s_str(&t, "<|im_end|>\n<|im_start|>assistant\n<think>\n");
                }
                lc_hook_out_free(&ho);
                if (again) {
                    if (++steps >= max_steps) { why_stop = "stopped: too many steps"; break; }
                    continue;
                }
            }
            if (level >= 2 && fl.edits_since_test > 0 && fl.planned && !fl.stopped && fl.nudges < 2) {
                /* changed files, not tested since: back to test before finishing */
                fl.nudges++;
                answer.n = 0;
                if (answer.p) answer.p[0] = 0;
                if (k->tool) k->tool(k, "teste pendente", "mudou arquivos sem testar depois: o agente volta a testar", 1);
                s_str(&t, "<|im_end|>\n<|im_start|>user\n[localcode] You changed files and have not called test since. "
                          "Call test now (debug, fix and test again if it fails), then finish.<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\n");
                if (++steps >= max_steps) { why_stop = "stopped: too many steps"; break; }
                continue;
            }
            break;
        }
        if (fl.stopped && ++fl.stopped_calls > 1) {     /* past the limit, still calling: stop here */
            for (int i = 0; i < nc; i++) lc_call_free(&calls[i]);
            why_stop = "stopped: the fix limit was reached and the model kept calling functions";
            break;
        }

        /* carry out the calls; their results open the next step */
        s_str(&t, "<|im_end|>\n<|im_start|>user");
        for (int i = 0; i < nc; i++) {
            LcCall *c = &calls[i];
            char title[400], sum[600];
            lc_tool_title(c, title, sizeof title);
            char *res = NULL;
            int ok = 0;
            const char *test_cmd = !strcmp(c->name, "test") ? (lc_call_arg(c, "command") && *lc_call_arg(c, "command")
                                                                   ? lc_call_arg(c, "command") : fl.test_cmd) : NULL;
            if (!strcmp(c->name, "test")) snprintf(title, sizeof title, "test %s", test_cmd ? test_cmd : "");
            if (!strcmp(c->name, "plan")) snprintf(title, sizeof title, "plano");
            if (fl.stopped) {
                snprintf(sum, sizeof sum, "not run: the fix limit was reached");
                res = (char *)malloc(200);
                if (res) snprintf(res, 200, "[localcode] The fix limit was reached: no more functions. Answer now.");
            } else if (lc_tool_allowed(c->name, level) && !strcmp(c->name, "plan")) {
                res = flow_plan(&fl, c, sum, sizeof sum, &ok);
            } else if (lc_tool_allowed(c->name, level) && !strcmp(c->name, "test") && !test_cmd) {
                snprintf(sum, sizeof sum, "sem comando de teste");
                res = (char *)malloc(200);
                if (res) snprintf(res, 200, "No test command: call plan with a test_command, or give command.");
            } else if (!lc_tool_allowed(c->name, level)) {
                snprintf(sum, sizeof sum, "not available at level %d", level);
                res = (char *)malloc(200);
                if (res) snprintf(res, 200, "%s is not available at this level (%d); use the reading functions, or answer.", c->name, level);
            } else {
                int go = 1;
                const int bit = tool_bit(c->name);
                const char *hname = lc_hook_tool_name(c->name);
                char *hinput = NULL;
                LcHookOut pre;
                memset(&pre, 0, sizeof pre);
                if (lc_hooks_any(LC_HOOK_PRE_TOOL) || lc_hooks_any(LC_HOOK_POST_TOOL)) {
                    if (test_cmd) {
                        char *qc = lc_json_quote(test_cmd);
                        Str hi = { 0 };
                        s_str(&hi, "{\"command\": ");
                        s_str(&hi, qc ? qc : "\"\"");
                        s_str(&hi, "}");
                        free(qc);
                        hinput = hi.p;
                    } else {
                        hinput = lc_hook_tool_input(c);
                    }
                }
                if (lc_hooks_any(LC_HOOK_PRE_TOOL)) {
                    Str f = { 0 };
                    s_fmt(&f, "\"tool_name\": \"%s\", \"tool_input\": ", hname);
                    s_str(&f, hinput ? hinput : "{}");
                    lc_hooks_run(LC_HOOK_PRE_TOOL, hname, f.p, &pre);
                    free(f.p);
                    hook_shown(k, "PreToolUse", &pre);
                }
                if (pre.block) {                         /* a hook said no: the model reads why */
                    go = 0;
                    snprintf(sum, sizeof sum, "blocked by a hook");
                    Str b = { 0 };
                    s_fmt(&b, "A hook blocked this %s call: ", c->name);
                    s_str(&b, pre.reason ? pre.reason : "(no reason given)");
                    res = b.p;
                } else if (pre.decision != 1 && ((lc_tool_needs_approval(c->name) && !s->auto_approve && !(s->always & (unsigned)bit)) ||
                                                 pre.decision == 3)) {
                    Str d = { 0 };
                    if (test_cmd) s_str(&d, test_cmd); else call_detail(c, &d);
                    const int a = k->approve ? k->approve(k, title, d.p ? d.p : "") : 0;
                    free(d.p);
                    if (a == 2) s->always |= (unsigned)bit;
                    go = a > 0;
                    if (!go) {
                        snprintf(sum, sizeof sum, "declined by the user");
                        res = (char *)malloc(300);
                        if (res) snprintf(res, 300, "The user declined this %s call. Do not repeat it; take another way or ask the user.", c->name);
                    }
                }
                if (go && *stop) { interrupted = 1; go = 0; }
                if (go) {
                    snprintf(line, sizeof line, "%s", test_cmd ? "Testing" : title);
                    k->status(k, line, now_s());
                    if (test_cmd) {
                        res = flow_test(&fl, test_cmd, stop, sum, sizeof sum, &ok);
                    } else {
                        res = lc_tool_run(c, stop, sum, sizeof sum, &ok);
                        if (ok && (!strcmp(c->name, "edit_file") || !strcmp(c->name, "write_file"))) {
                            fl.edited = 1;
                            fl.edits_since_test++;
                            fl.passed = 0;
                            if (!fl.planned && !fl.plan_hint && res) {   /* once: the loop needs its test */
                                fl.plan_hint = 1;
                                const size_t rl = strlen(res);
                                static const char HINT[] = "\n[localcode] No plan recorded: call plan (steps and test_command) before more changes.";
                                char *g = (char *)realloc(res, rl + sizeof HINT);
                                if (g) { memcpy(g + rl, HINT, sizeof HINT); res = g; }
                            }
                        }
                    }
                    k->status(k, NULL, 0.0);
                    if (lc_hooks_any(LC_HOOK_POST_TOOL) && res) {
                        LcHookOut po;
                        char *qr = lc_json_quote(res);
                        Str f = { 0 };
                        s_fmt(&f, "\"tool_name\": \"%s\", \"tool_input\": ", hname);
                        s_str(&f, hinput ? hinput : "{}");
                        s_str(&f, ", \"tool_response\": {\"output\": ");
                        s_str(&f, qr ? qr : "\"\"");
                        s_str(&f, "}");
                        free(qr);
                        lc_hooks_run(LC_HOOK_POST_TOOL, hname, f.p, &po);
                        free(f.p);
                        hook_shown(k, "PostToolUse", &po);
                        if (po.block || po.context) {    /* the hook's word goes with the result */
                            Str r2 = { 0 };
                            s_str(&r2, res);
                            if (po.block && po.reason) { s_str(&r2, "\n[hook] "); s_str(&r2, po.reason); }
                            if (po.context) { s_str(&r2, "\n<system-reminder>\n"); s_str(&r2, po.context); s_str(&r2, "\n</system-reminder>"); }
                            free(res);
                            res = r2.p;
                        }
                        lc_hook_out_free(&po);
                    }
                }
                free(hinput);
                lc_hook_out_free(&pre);
            }
            if (!interrupted || res) {
                if (k->tool) k->tool(k, title, sum, ok);
                s_fmt(&actions, "- `%s` — %s\n", title, sum);
                s_str(&t, "\n<tool_response>\n");
                fit_result(e, t.n, step_gen, res ? res : "(out of memory)", &t);
                s_str(&t, "\n</tool_response>");
                calls_done++;
            }
            free(res);
            lc_call_free(c);
        }
        for (int i = 0; i < nc; i++) lc_call_free(&calls[i]);
        s_str(&t, "<|im_end|>\n<|im_start|>assistant\n<think>\n");
        if (interrupted) break;
        if (++steps >= max_steps) {
            snprintf(line, sizeof line, "stopped after %d steps without a final answer; say how to go on", steps);
            why_stop = line;
            break;
        }
    }
    /* results the model has not read yet (interrupted, or out of steps): feed them without
     * the assistant opening, so the next message's "<|im_end|>" closes their user turn and
     * the model still sees what its calls returned */
    if (!fatal && !failed && t.p && strstr(t.p, "<tool_response>")) {
        char *open = strstr(t.p, "<|im_end|>\n<|im_start|>assistant\n<think>\n");
        if (open) { *open = 0; t.n = (size_t)(open - t.p); }
        const int m = e->encode(e, t.p, (int)t.n, s->ids, s->ids_cap);
        if (m > 0 && e->cached + m + ROOM <= e->cap && e->forward(e, s->ids, m, s->logits) != 0) fatal = 1;
    }

    /* the session's measures and transcript */
    s->turns++;
    s->pre_tok += pre_tok;
    s->pre_s += pre_s;
    if (dec_tok) {
        s->dec_tok += dec_tok;
        s->dec_s += dec_s;
        s->last_spt = dec_s / (double)dec_tok;
        if (s->n_spt == 16) { memmove(s->spt_hist, s->spt_hist + 1, 15 * sizeof *s->spt_hist); s->n_spt--; }
        s->spt_hist[s->n_spt++] = s->last_spt;
    }
    char stats[400];
    const double total = now_s() - t_turn;
    snprintf(stats, sizeof stats, "level %d · %d step%s · %d tool call%s · wrote %ld tokens at %.1f tok/s · read %ld in %.1f s · %.0f s%s%s",
             level, steps + 1, steps ? "s" : "", calls_done, calls_done == 1 ? "" : "s", dec_tok,
             dec_s > 0 ? dec_tok / dec_s : 0.0, pre_tok, pre_s, total, compacted ? " · compacted" : "",
             interrupted ? " · interrupted" : "");
    lc_session_answer(s, answer.p ? answer.p : (why_stop ? why_stop : ""), actions.p, stats);
    glm53f_chat_log(s, "## Você\n\n");
    glm53f_chat_log(s, in->shown ? in->shown : in->prompt);
    glm53f_chat_log(s, "\n\n## localcode\n\n");
    if (actions.p) { glm53f_chat_log(s, actions.p); glm53f_chat_log(s, "\n"); }
    glm53f_chat_log(s, answer.p ? answer.p : (why_stop ? why_stop : "*(sem resposta)*"));
    glm53f_chat_log(s, "\n\n<sub>");
    glm53f_chat_log(s, stats);
    glm53f_chat_log(s, "</sub>\n\n");
    free(s->last_answer);
    s->last_answer = answer.p;
    answer.p = NULL;
    free(t.p);
    free(req.p);
    free(actions.p);
    free(fl.test_cmd);
    if (why_stop && failed) { k->done(k, why_stop, 2); return 0; }
    if (why_stop) { char m[600]; snprintf(m, sizeof m, "%s · %s", stats, why_stop); k->done(k, m, interrupted); return fatal ? -1 : 0; }
    k->done(k, stats, interrupted);
    return fatal ? -1 : 0;
}
