/* glm53f_chat.c - localcode: the interactive session, one loaded model, many turns.
 *
 * A one-shot run pays for loading the trunk and warming the expert cache every time, and
 * re-reads the whole conversation on every question. Here the model stays open: each turn
 * feeds only the new tokens, so the context the earlier turns left in the KV cache, the KDA
 * state and the DSA indexer is reused as the model itself would reuse it.
 *
 * This file is the session: commands, the chat template, one turn reported as events, and
 * the Markdown transcript behind /save. On a terminal the full-screen front end in
 * glm53f_tui.c shows it; otherwise (a pipe, NO_COLOR-style plain use) the line-by-line
 * front end at the bottom of this file does.
 */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "glm53f_chat.h"
#include "lc_tools.h"
#include "lc_plugins.h"
#include "lc_mods.h"

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void duration(char *buf, size_t cap, double sec)
{
    if (sec < 60.0) snprintf(buf, cap, "%.1fs", sec);
    else            snprintf(buf, cap, "%dm %02ds", (int)(sec / 60.0), (int)sec % 60);
}

/* Length of the longest prefix of s[0..n) that does not end inside a UTF-8 sequence. */
static int utf8_complete(const char *s, int n)
{
    int i = n, back = 0;
    while (i > 0 && back < 4 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) { i--; back++; }
    if (i == 0) return n;
    const unsigned char lead = (unsigned char)s[i - 1];
    const int need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return (back + 1 >= need) ? n : i - 1;
}

static char *dup_str(const char *p)
{
    const size_t n = strlen(p) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, p, n);
    return d;
}

/* The whole file, NUL-terminated (NULL if it cannot be read); *n gets its length. */
static char *read_file(const char *path, long *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *b = NULL;
    if (fseek(f, 0, SEEK_END) == 0) {
        const long len = ftell(f);
        if (len >= 0 && len < (1L << 30) && fseek(f, 0, SEEK_SET) == 0 && (b = (char *)malloc((size_t)len + 1))) {
            if (fread(b, 1, (size_t)len, f) != (size_t)len) { free(b); b = NULL; }
            else { b[len] = 0; *n = len; }
        }
    }
    fclose(f);
    return b;
}

static char *fmt_str(const char *fmt, const char *a, const char *b)
{
    const int n = snprintf(NULL, 0, fmt, a, b);
    char *d = (char *)malloc((size_t)n + 1);
    if (d) snprintf(d, (size_t)n + 1, fmt, a, b);
    return d;
}

/* ----------------------------------------------------------------- transcript ---- */

/* Append to the Markdown transcript; a failed allocation only costs the transcript. */
static void log_add(Glm53fChat *s, const char *p, size_t n)
{
    if (s->log_len + n + 1 > s->log_cap) {
        size_t cap = s->log_cap ? s->log_cap : 65536;
        while (s->log_len + n + 1 > cap) cap *= 2;
        char *g = (char *)realloc(s->log, cap);
        if (!g) return;
        s->log = g;
        s->log_cap = cap;
    }
    memcpy(s->log + s->log_len, p, n);
    s->log_len += n;
    s->log[s->log_len] = 0;
}

static void log_str(Glm53fChat *s, const char *p) { log_add(s, p, strlen(p)); }

/* One exchange: the message, the reasoning folded away, the answer. */
static void log_turn(Glm53fChat *s, const char *msg, const char *out, int len, const char *stats)
{
    log_str(s, "## Você\n\n");
    log_str(s, msg);
    log_str(s, "\n\n## ");
    log_str(s, s->e->name);
    log_str(s, "\n\n");
    const char *close = NULL;
    for (int i = 0; i + 8 <= len; i++)
        if (!memcmp(out + i, "</think>", 8)) { close = out + i; break; }
    const int think = close ? (int)(close - out) : len;
    if (think > 0) {
        log_str(s, "<details><summary>Raciocínio</summary>\n\n");
        log_add(s, out, (size_t)think);
        log_str(s, "\n\n</details>\n\n");
    }
    if (close) log_add(s, close + 8, (size_t)(len - think - 8));
    else       log_str(s, "*(a resposta não começou antes do fim da geração)*");
    log_str(s, "\n\n<sub>");
    log_str(s, stats);
    log_str(s, "</sub>\n\n");
}

static char *save(Glm53fChat *s, const char *arg)
{
    char name[512], msg[700];
    if (arg && *arg) {
        snprintf(name, sizeof name, "%s", arg);
    } else {
        const time_t now = time(NULL);
        char stamp[32];
        strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", localtime(&now));
        snprintf(name, sizeof name, "localcode-%s.md", stamp);
    }
    if (!s->log_len) return dup_str("nothing to save yet");
    FILE *f = fopen(name, "wb");
    if (!f) { snprintf(msg, sizeof msg, "cannot write %s", name); return dup_str(msg); }
    char settings[256];
    s->e->describe(s->e, GLM53F_DESC_SETTINGS, settings, sizeof settings);
    fprintf(f, "# Conversa com %s\n\n<sub>localcode · reasoning %s · %s</sub>\n\n", s->e->name, s->reasoning, settings);
    const int ok = fwrite(s->log, 1, s->log_len, f) == s->log_len;
    if (fclose(f) != 0 || !ok) { snprintf(msg, sizeof msg, "writing %s failed", name); return dup_str(msg); }
    snprintf(msg, sizeof msg, "saved %s", name);
    return dup_str(msg);
}

/* ------------------------------------------------------------------- commands ---- */

const Glm53fCommand glm53f_commands[] = {
    { "/file",      "PATH [question]", "send a file's text, then the question (a path with spaces in quotes)", 2 },
    { "/level",     "0|1|2",           "agent level: 0 chat, 1 quick (reads files), 2 agent (edits, runs commands); starts a new conversation", 2 },
    { "/model",     "[N|NAME]",        "the model files on this machine, or switch to one (the conversation goes on in it)", 2 },
    { "/approve",   "ask|auto",        "writes and commands: ask before each (default) or run without asking", 2 },
    { "/plugin",    "[on|off NAME]",   "the plugins (skills, commands, hooks, mods), or one plugin or skill (plugin:name) on or off; /plugin hooks on|off NAME allows its hooks", 2 },
    { "/auto",      "[off]",           "this folder, remembered: the agent goes from start to finish without asking anything", 1 },
    { "/reasoning", "LEVEL",           "reasoning effort of level 0 chat (set when a conversation opens)", 2 },
    { "/gen",       "N",               "tokens to generate per answer", 2 },
    { "/temp",      "T",               "sampling temperature: 0 always the most likely token; 0.6 Qwen's advice (top_p 0.95, top_k 20)", 2 },
    { "/thinking",  "",                "show or fold the model's reasoning", 0 },
    { "/stats",     "",                "measured speed, disk reads and expert cache of this session", 0 },
    { "/params",    "",                "the settings in force", 0 },
    { "/copy",      "",                "put the last answer on the clipboard", 0 },
    { "/save",      "[FILE]",          "write the conversation as Markdown (default localcode-DATE.md)", 1 },
    { "/compact",   "[FOCUS]",         "summarise the conversation and go on from the summary (it also happens by itself near the context's end)", 1 },
    { "/sessions",  "",                "this folder's earlier conversations", 0 },
    { "/resume",    "[N]",             "go back to an earlier conversation (default the latest; /sessions lists them)", 1 },
    { "/reset",     "",                "forget the conversation and start a new one", 0 },
    { "/help",      "",                "the commands and keys", 0 },
    { "/quit",      "",                "leave", 0 },
};
const int glm53f_ncommands = (int)(sizeof glm53f_commands / sizeof *glm53f_commands);

static const char KEYS[] =
    "/ opens the menu: up, down to choose, tab to complete, enter to run, esc to close\n"
    "esc, ctrl+c             stop the answer being written; ctrl+c on an empty box leaves\n"
    "pgup, pgdn, wheel       scroll the conversation; end follows it again\n"
    "up, down                earlier messages";

static char *help_note(void)
{
    char buf[3072];
    int n = 0;
    for (int i = 0; i < glm53f_ncommands && n < (int)sizeof buf; i++) {
        const Glm53fCommand *c = &glm53f_commands[i];
        char head[64];
        snprintf(head, sizeof head, "%s%s%s", c->name, *c->args ? " " : "", c->args);
        n += snprintf(buf + n, sizeof buf - (size_t)n, "%-24s %s\n", head, c->help);
    }
    if (n < (int)sizeof buf) snprintf(buf + n, sizeof buf - (size_t)n, "%s", KEYS);
    return dup_str(buf);
}

void glm53f_chat_command_value(const Glm53fChat *s, const Glm53fCommand *c, char *buf, size_t cap)
{
    const char *n = c->name;
    buf[0] = 0;
    if (!strcmp(n, "/reasoning"))     snprintf(buf, cap, "%s", s->reasoning);
    else if (!strcmp(n, "/level"))    snprintf(buf, cap, "%s", s->agent_level == 2 ? "2 agent" : s->agent_level == 1 ? "1 quick" : "0 chat");
    else if (!strcmp(n, "/approve"))  snprintf(buf, cap, "%s", s->auto_approve ? "auto" : "ask");
    else if (!strcmp(n, "/auto"))     snprintf(buf, cap, "%s", s->autonomous ? "on" : "off");
    else if (!strcmp(n, "/gen"))      snprintf(buf, cap, "%d", s->gen);
    else if (!strcmp(n, "/temp"))     snprintf(buf, cap, s->smp.temp > 0.0f ? "%.2g" : "0 greedy", s->smp.temp);
    else if (!strcmp(n, "/thinking")) snprintf(buf, cap, "%s", s->show_thinking ? "shown" : "folded");
    else if (!strcmp(n, "/stats") && s->last_spt > 0.0)
        snprintf(buf, cap, "%.1f tok/s · %.3f s/token", 1.0 / s->last_spt, s->last_spt);
    else if (!strcmp(n, "/copy"))     snprintf(buf, cap, "%s", s->last_answer ? "last answer" : "nothing yet");
    else if (!strcmp(n, "/save") && s->log_len) snprintf(buf, cap, "%.1f KB", (double)s->log_len / 1024.0);
    else if (!strcmp(n, "/reset"))    snprintf(buf, cap, "%d/%d", s->e->cached, s->e->cap);
    else if (!strcmp(n, "/compact") && s->compactions) snprintf(buf, cap, "%d so far", s->compactions);
    else if (!strcmp(n, "/model") && s->e->variant) snprintf(buf, cap, "%s", s->e->variant);
    else if (!strcmp(n, "/file"))     snprintf(buf, cap, "%d left", s->e->cap - s->e->cached);
}

/* A bar per answer, higher is slower: how the speed moved as the cache warmed up. */
static void spark(const Glm53fChat *s, char *buf, size_t cap)
{
    static const char *bars[] = { "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█" };
    double lo = 1e30, hi = 0.0;
    for (int i = 0; i < s->n_spt; i++) {
        if (s->spt_hist[i] < lo) lo = s->spt_hist[i];
        if (s->spt_hist[i] > hi) hi = s->spt_hist[i];
    }
    size_t n = 0;
    buf[0] = 0;
    for (int i = 0; i < s->n_spt && n + 4 < cap; i++) {
        const int k = hi > lo ? (int)(7.0 * (s->spt_hist[i] - lo) / (hi - lo) + 0.5) : 3;
        n += (size_t)snprintf(buf + n, cap - n, "%s", bars[k]);
    }
}

static char *stats_note(const Glm53fChat *s)
{
    char buf[1600], d1[32], d2[32], sp[80], eng[400];
    int n = 0;
    if (!s->dec_tok) {
        n += snprintf(buf + n, sizeof buf - (size_t)n, "nothing measured yet: the first answer fills this in\n");
    } else {
        const double spt = s->dec_s / (double)s->dec_tok;
        duration(d1, sizeof d1, s->dec_s);
        duration(d2, sizeof d2, s->pre_s);
        spark(s, sp, sizeof sp);
        n += snprintf(buf + n, sizeof buf - (size_t)n,
                      "writing   %ld tokens in %s over %d answer%s · %.1f tok/s · %.3f s/token %s\n",
                      s->dec_tok, d1, s->turns, s->turns == 1 ? "" : "s", 1.0 / spt, spt, sp);
        n += snprintf(buf + n, sizeof buf - (size_t)n, "reading   %ld prompt tokens in %s · %.1f tok/s · %.3f s/token\n",
                      s->pre_tok, d2, s->pre_s > 0.0 ? (double)s->pre_tok / s->pre_s : 0.0,
                      s->pre_tok ? s->pre_s / (double)s->pre_tok : 0.0);
        const double wait = s->dec_wait_s / (double)s->dec_tok;
        const double reqs = s->dec_reqs ? (double)s->dec_reqs : 1.0;
        const double ahead = (double)s->dec_ahead, demand = (double)s->dec_demand;
        const double ram = reqs - ahead - demand > 0.0 ? reqs - ahead - demand : 0.0;
        n += snprintf(buf + n, sizeof buf - (size_t)n,
                      "experts   %.0f per token: %.0f%% from RAM, %.0f%% read ahead from the SSD, "
                      "%.0f%% read on demand · %.2f GB read per token\n",
                      (double)s->dec_reqs / (double)s->dec_tok, 100.0 * ram / reqs, 100.0 * ahead / reqs,
                      100.0 * demand / reqs, s->dec_read_gb / (double)s->dec_tok);
        n += snprintf(buf + n, sizeof buf - (size_t)n, "waiting   %.0f ms per token on the disk, %.0f%% of the time\n",
                      1e3 * wait, spt > 0.0 ? 100.0 * wait / spt : 0.0);
        /* the finding the engine is built around (PERFORMANCE.md): decode time follows bytes read */
        n += snprintf(buf + n, sizeof buf - (size_t)n, "limit     %s\n",
                      wait > 0.5 * spt ? "the disk: tokens go as fast as experts arrive; more cache or int4 experts read less"
                                       : "compute: the experts mostly come from RAM");
    }
    s->e->describe(s->e, GLM53F_DESC_STATS, eng, sizeof eng);
    snprintf(buf + n, sizeof buf - (size_t)n, "engine    %s · context %d of %d", eng, s->e->cached, s->e->cap);
    return dup_str(buf);
}

/* The clipboard: the Windows one, or OSC 52, which most terminals pass to theirs. */
static char *copy_answer(const Glm53fChat *s)
{
    if (!s->last_answer || !*s->last_answer) return dup_str("no answer to copy yet");
    const char *t = s->last_answer;
    const size_t n = strlen(t);
#ifdef _WIN32
    const int wn = MultiByteToWideChar(CP_UTF8, 0, t, (int)n, NULL, 0);
    HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, ((size_t)wn + 1) * sizeof(wchar_t));
    if (!h) return dup_str("could not copy: out of memory");
    wchar_t *w = (wchar_t *)GlobalLock(h);
    MultiByteToWideChar(CP_UTF8, 0, t, (int)n, w, wn);
    w[wn] = 0;
    GlobalUnlock(h);
    if (!OpenClipboard(NULL)) { GlobalFree(h); return dup_str("could not open the clipboard"); }
    EmptyClipboard();
    if (!SetClipboardData(CF_UNICODETEXT, h)) { CloseClipboard(); GlobalFree(h); return dup_str("could not copy"); }
    CloseClipboard();
#else
    static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    fputs("\033]52;c;", stdout);
    for (size_t i = 0; i < n; i += 3) {
        const unsigned v = (unsigned)(unsigned char)t[i] << 16 |
                           (i + 1 < n ? (unsigned)(unsigned char)t[i + 1] << 8 : 0) |
                           (i + 2 < n ? (unsigned)(unsigned char)t[i + 2] : 0);
        const char q[4] = { b64[v >> 18 & 63], b64[v >> 12 & 63],
                            i + 1 < n ? b64[v >> 6 & 63] : '=', i + 2 < n ? b64[v & 63] : '=' };
        fwrite(q, 1, 4, stdout);
    }
    fputs("\a", stdout);
    fflush(stdout);
#endif
    char msg[96];
    snprintf(msg, sizeof msg, "copied the last answer (%zu byte%s)", n, n == 1 ? "" : "s");
    return dup_str(msg);
}

static char *params_note(const Glm53fChat *s)
{
    char line[600], settings[300];
    s->e->describe(s->e, GLM53F_DESC_SETTINGS, settings, sizeof settings);
    char smp[80];
    if (s->smp.temp > 0.0f) snprintf(smp, sizeof smp, "temp %.2g · top_p %.2g · top_k %d", s->smp.temp, s->smp.top_p, s->smp.top_k);
    else snprintf(smp, sizeof smp, "temp 0 (greedy)");
    snprintf(line, sizeof line, "context %d of %d used · %s · reasoning %s · thinking %s · gen %d · %s",
             s->e->cached, s->e->cap, settings, s->reasoning, s->show_thinking ? "shown" : "folded", s->gen, smp);
    return dup_str(line);
}

void glm53f_chat_input_free(Glm53fInput *in)
{
    free(in->prompt);
    free(in->shown);
    free(in->note);
    memset(in, 0, sizeof *in);
}

void glm53f_chat_input(Glm53fChat *s, const char *raw, Glm53fInput *in)
{
    memset(in, 0, sizeof *in);
    while (*raw == ' ' || *raw == '\t') raw++;
    size_t len = strlen(raw);
    while (len > 0 && (raw[len - 1] == '\n' || raw[len - 1] == '\r' || raw[len - 1] == ' ' || raw[len - 1] == '\t'))
        len--;
    if (!len) return;
    char *line = (char *)malloc(len + 1);
    if (!line) return;
    memcpy(line, raw, len);
    line[len] = 0;

    if (line[0] != '/') {
        in->kind = GLM53F_IN_MESSAGE;
        in->prompt = line;
        in->shown = dup_str(line);
        return;
    }
    in->kind = GLM53F_IN_NOTE;
    if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) {
        in->kind = GLM53F_IN_QUIT;
    } else if (!strcmp(line, "/help")) {
        in->note = help_note();
    } else if (!strcmp(line, "/stats")) {
        in->note = stats_note(s);
    } else if (!strcmp(line, "/copy")) {
        in->note = copy_answer(s);
    } else if (!strcmp(line, "/params")) {
        in->note = params_note(s);
    } else if (!strcmp(line, "/thinking")) {
        s->show_thinking = !s->show_thinking;
        in->note = dup_str(s->show_thinking ? "reasoning shown as it is written" : "reasoning folded into one line");
    } else if (!strcmp(line, "/compact") || !strncmp(line, "/compact ", 9)) {
        in->kind = GLM53F_IN_MESSAGE;
        in->compact = 1;
        in->prompt = dup_str(line[8] ? line + 9 : "");
        in->shown = dup_str(line);
    } else if (!strcmp(line, "/model") || !strncmp(line, "/model ", 7)) {
        Glm53fEngine *e = s->e;
        const char *arg = line[6] ? line + 7 : "";
        while (*arg == ' ') arg++;
        int pick = -1;
        if (*arg >= '1' && *arg <= '9' && !arg[1]) pick = *arg - '1';
        for (int i = 0; i < e->n_variants && pick < 0 && *arg; i++) {   /* a name, or part of one */
            const char *h = e->variant_name[i];
            for (; *h && pick < 0; h++) {
                const char *a = arg, *b = h;
                while (*a && *b && (*a | 32) == (*b | 32)) { a++; b++; }
                if (!*a) pick = i;
            }
        }
        char buf[1200];
        if (!*arg || pick < 0 || pick >= e->n_variants) {
            int k = snprintf(buf, sizeof buf, "%s on this machine (/model N to switch):", e->name);
            for (int i = 0; i < e->n_variants && k < (int)sizeof buf; i++)
                k += snprintf(buf + k, sizeof buf - (size_t)k, "\n%d  %s · %s%s", i + 1, e->variant_name[i],
                              e->variant_info[i], e->variant && !strcmp(e->variant, e->variant_name[i]) ? "  ← aberto" : "");
            if (!e->n_variants) snprintf(buf + k, sizeof buf - (size_t)k, " none found in the models folder");
            else if (*arg) snprintf(buf + k, sizeof buf - (size_t)k, "\nno model matches \"%s\"", arg);
            in->note = dup_str(buf);
        } else if (e->variant && !strcmp(e->variant, e->variant_name[pick])) {
            in->note = fmt_str("%s is the one open%s", e->variant_name[pick], "");
        } else {
            s->switch_to = pick + 1;              /* the caller closes this one and opens that */
            in->kind = GLM53F_IN_QUIT;
            in->note = fmt_str("switching to %s%s", e->variant_name[pick], "");
        }
    } else if (!strcmp(line, "/sessions")) {
        in->note = lc_session_list(s);
    } else if (!strcmp(line, "/resume") || !strncmp(line, "/resume ", 8)) {
        int ok = 0;
        in->note = lc_session_resume(s, line[7] ? line + 8 : NULL, &ok);
        if (ok) in->kind = GLM53F_IN_RESET;
    } else if (!strcmp(line, "/reset") || !strcmp(line, "/clear") || !strcmp(line, "/new")) {
        lc_session_end(s, 0);
        s->e->reset(s->e);
        s->first = 1;
        s->log_len = 0;                          /* a new conversation, a new transcript */
        in->kind = GLM53F_IN_RESET;
        in->note = dup_str("conversation cleared");
    } else if (!strncmp(line, "/level ", 7) || !strncmp(line, "/nivel ", 7)) {
        const int v = atoi(line + 7);
        if (v < 0 || v > 2 || line[7] < '0' || line[7] > '9') {
            in->note = dup_str("/level takes 0 (chat), 1 (quick: reads files) or 2 (agent: edits files, runs commands)");
        } else if (v > 0 && !s->e->agent) {
            in->note = dup_str("this model has no agent mode here; only /level 0");
        } else {
            lc_session_end(s, 0);
            s->agent_level = v;
            s->e->reset(s->e);                   /* the system prompt differs per level */
            s->first = 1;
            s->log_len = 0;
            in->kind = GLM53F_IN_RESET;
            in->note = dup_str(v == 2 ? "level 2 · agent: reads, edits, runs commands (asks first) · new conversation"
                               : v == 1 ? "level 1 · quick: reads the project's files to answer · new conversation"
                                        : "level 0 · chat, no tools · new conversation");
        }
    } else if (!strcmp(line, "/auto") || !strcmp(line, "/auto on") || !strcmp(line, "/auto off")) {
        /* the folder joins (or leaves) the autonomous list: no approvals, no questions */
        const int on = strcmp(line, "/auto off") != 0;
        const char *dir = lc_tools_root();
        s->autonomous = on;
        s->auto_approve = on;
        if (!on) s->always = 0;
        const int saved = dir && lc_auto_dir_set(dir, on) == 0;
        in->note = on ? fmt_str("modo autônomo em %s: o agente vai do início ao fim sem pedir aprovação nem fazer "
                                "perguntas%s · /auto off desliga", dir ? dir : "esta pasta",
                                saved ? " (fica lembrado para esta pasta)" : "")
                      : dup_str("modo autônomo desligado nesta pasta: escritas e comandos voltam a pedir aprovação");
    } else if (!strcmp(line, "/approve ask") || !strcmp(line, "/approve auto")) {
        s->auto_approve = !strcmp(line, "/approve auto");
        if (!s->auto_approve) { s->always = 0; s->autonomous = 0; }
        in->note = dup_str(s->auto_approve ? "writes and commands run without asking (inside the working directory)"
                                           : "writes and commands ask first");
    } else if (!strncmp(line, "/approve", 8)) {
        in->note = dup_str("/approve takes ask or auto");
    } else if (!strcmp(line, "/plugin") || !strcmp(line, "/plugins")) {
        lc_plugins_load(lc_tools_root());
        in->note = lc_plugins_report();
    } else if (!strncmp(line, "/plugin hooks on ", 17) || !strncmp(line, "/plugin hooks off ", 18)) {
        const int on = line[15] == 'n';
        const char *what = line + (on ? 17 : 18);
        lc_plugins_load(lc_tools_root());
        in->note = lc_plugin_trust(what, on)
                       ? fmt_str("no plugin named %s%s (/plugin lists them)", what, "")
                       : fmt_str(on ? "%s: its hooks may run from the next start of localcode%s"
                                    : "%s: its hooks no longer run (from the next start of localcode)%s", what, "");
    } else if (!strncmp(line, "/plugin on ", 11) || !strncmp(line, "/plugin off ", 12)) {
        const int on = line[9] == 'n';
        const char *what = line + (on ? 11 : 12);
        lc_plugins_load(lc_tools_root());
        in->note = lc_plugin_switch(what, on)
                       ? fmt_str("no plugin or skill named %s%s (/plugin lists them; a skill is plugin:name)", what, "")
                       : fmt_str("%s %s: the next conversation (/reset) sees it", what, on ? "on" : "off");
    } else if (!strcmp(line, "/save") || !strncmp(line, "/save ", 6)) {
        in->note = save(s, line[5] ? line + 6 : NULL);
    } else if (!strncmp(line, "/temp ", 6)) {
        char *end;
        const double v = strtod(line + 6, &end);
        if (end != line + 6 && v >= 0.0 && v <= 2.0) {
            s->smp.temp = (float)v;
            if (s->smp.temp > 0.0f && s->smp.top_p <= 0.0f) { s->smp.top_p = 0.95f; s->smp.top_k = 20; }
            in->note = params_note(s);
        } else in->note = dup_str("/temp takes a number from 0 (greedy) to 2");
    } else if (!strncmp(line, "/gen ", 5)) {
        const int v = atoi(line + 5);
        if (v > 0 && v <= GLM53F_CHAT_MAX_GEN) s->gen = v;
        in->note = params_note(s);
    } else if (!strncmp(line, "/reasoning ", 11)) {
        const char *v = line + 11;
        int hit = -1;
        for (int i = 0; i < s->e->n_levels && hit < 0; i++) {
            const char *a = s->e->levels[i].name, *b = v;
            while (*a && *b && (*a | 32) == (*b | 32)) { a++; b++; }
            if (!*a && !*b) hit = i;
        }
        if (hit >= 0) {
            s->reasoning = s->e->levels[hit].name;
            in->note = params_note(s);
        } else {
            char msg[200];
            int k = snprintf(msg, sizeof msg, "/reasoning takes");
            for (int i = 0; i < s->e->n_levels && k < (int)sizeof msg; i++)
                k += snprintf(msg + k, sizeof msg - (size_t)k, " %s", s->e->levels[i].name);
            in->note = dup_str(msg);
        }
    } else if (!strncmp(line, "/file ", 6)) {
        /* /file PATH [question]; a path with spaces goes in quotes */
        char path[1024];
        const char *p = line + 6, *q;
        while (*p == ' ') p++;
        if (*p == '"') { p++; q = strchr(p, '"'); } else { q = strchr(p, ' '); }
        size_t pl = q ? (size_t)(q - p) : strlen(p);
        if (pl >= sizeof path) pl = sizeof path - 1;
        memcpy(path, p, pl);
        path[pl] = 0;
        const char *question = q ? q + (*q == '"' ? 1 : 0) : "";
        while (*question == ' ') question++;
        FILE *f = fopen(path, "rb");
        if (!f) {
            in->note = fmt_str("cannot open %s%s", path, "");
        } else {
            fclose(f);
            long flen = 0;
            char *content = read_file(path, &flen);
            if (!content) { in->note = fmt_str("cannot read %s%s", path, ""); free(line); return; }
            const size_t bl = (size_t)flen + strlen(question) + 4;
            in->prompt = (char *)malloc(bl);
            if (in->prompt) {
                snprintf(in->prompt, bl, "%s%s%s", content, *question ? "\n\n" : "", question);
                in->kind = GLM53F_IN_MESSAGE;
                in->shown = fmt_str(*question ? "📄 %s\n\n%s" : "📄 %s%s", path, question);
                char note[1200];
                snprintf(note, sizeof note, "%s · %ld bytes", path, flen);
                in->note = dup_str(note);
            }
            free(content);
        }
    } else {
        /* a mod's command (/replay), a plugin's command, or a skill the user calls */
        lc_plugins_load(lc_tools_root());
        char *note = NULL;
        char *msg = NULL;
        int modc = -1;
        for (int i = 0; i < lc_mods_commands(); i++) {
            const char *nm = lc_mods_command_name(i);
            const size_t nl = strlen(nm);
            if (!strncmp(line + 1, nm, nl) && (line[1 + nl] == 0 || line[1 + nl] == ' ')) modc = i;
        }
        if (modc >= 0) {
            const char *nm = lc_mods_command_name(modc);
            const char *a = line + 1 + strlen(nm);
            while (*a == ' ') a++;
            in->note = lc_mods_command_run(nm, a);
        } else if ((msg = lc_command_expand(line, &note)) != NULL) {
            in->prompt = msg;
            in->kind = GLM53F_IN_MESSAGE;
            in->shown = dup_str(line);
            in->note = note;
        } else {
            in->note = dup_str("unknown command; /help lists them, /plugin the plugins'");
        }
    }
    free(line);
}

void glm53f_chat_gauge(const Glm53fChat *s, char *buf, size_t cap)
{
    const int n = snprintf(buf, cap, "context %d/%d (%d%%)", s->e->cached, s->e->cap,
                           s->e->cap ? (int)(100.0 * s->e->cached / s->e->cap + 0.5) : 0);
    if (s->compactions && n > 0 && (size_t)n < cap)
        snprintf(buf + n, cap - (size_t)n, " · compacted %d×", s->compactions);
}

/* ----------------------------------------------------------------------- turn ---- */

static volatile int g_stop = 0;

void glm53f_chat_stop(void) { g_stop = 1; }

volatile int *glm53f_chat_stop_flag(void) { return &g_stop; }

void glm53f_chat_log(Glm53fChat *s, const char *text) { log_str(s, text); }

int glm53f_chat_turn(Glm53fChat *s, const Glm53fInput *in, Glm53fTurnSink *k)
{
    Glm53fEngine *e = s->e;
    char line[320];
    if (in->compact) {
        g_stop = 0;
        const int rc = lc_compact(s, k, NULL, in->prompt);
        k->done(k, rc == 0 ? "the conversation goes on from the summary; the next message opens a new context"
                   : rc > 0 ? "nothing to compact yet" : "the engine failed while compacting", rc ? 2 : 0);
        return rc < 0 ? -1 : 0;
    }
    if (s->agent_level > 0 && e->agent) return lc_agent_turn(s, in, k);

    /* the chat template: the preamble only opens the conversation, every turn after that
     * appends its own user block, and the cache already holds what came before. A message
     * that would not fit first has the conversation summarised (lc_compact), and then opens
     * a new context after the summary. */
    lc_session_user(s, in->shown ? in->shown : in->prompt);
    char *text = NULL;
    int n = 0;
    for (int pass = 0; pass < 2; pass++) {
        const char *pre = s->first && s->preface ? s->preface : "";
        const size_t tl = strlen(pre) + strlen(in->prompt) + 1024;
        free(text);
        text = (char *)malloc(tl);
        char *msg = (char *)malloc(tl);
        if (!text || !msg) { free(msg); free(text); k->done(k, "out of memory", 2); return 0; }
        snprintf(msg, tl, "%s%s", pre, in->prompt);
        e->turn_text(e, s->first, s->reasoning, msg, text, tl);
        free(msg);
        n = e->encode(e, text, (int)strlen(text), s->ids, s->ids_cap);
        if (n <= 0 || !lc_compact_due(s, n + s->gen)) break;
        g_stop = 0;
        const int rc = lc_compact(s, k, NULL, NULL);
        if (rc < 0) { free(text); k->done(k, "the engine failed while compacting", 2); return -1; }
        if (rc > 0) break;
    }
    free(text);
    if (s->first && s->preface) { free(s->preface); s->preface = NULL; }
    if (n <= 0) { k->done(k, "could not tokenise the message (is it longer than the session?)", 2); return 0; }
    if (e->cached + n + s->gen > e->cap) {
        snprintf(line, sizeof line, "the session holds %d positions and this turn needs %d; "
                 "/reset to start over, or a smaller /gen", e->cap, e->cached + n + s->gen);
        k->done(k, line, 2);
        return 0;
    }
    s->first = 0;
    g_stop = 0;

    snprintf(line, sizeof line, "Reading %d token%s", n, n == 1 ? "" : "s");
    const double t0 = now_s();
    k->status(k, line, t0);
    if (e->forward(e, s->ids, n, s->logits) != 0) {
        k->status(k, NULL, 0.0);
        k->done(k, "the engine failed while reading the message", 2);
        return -1;
    }
    const double t_prefill = now_s() - t0;
    k->status(k, NULL, 0.0);

    int nout = 0, printed = 0, stopped = 0, in_think = 1, interrupted = 0;
    Glm53fCounters c0, c1;
    memset(&c0, 0, sizeof c0);
    if (e->counters) e->counters(e, &c0);
    const double td = now_s();
    int pend[GLM53F_SPEC_MAX], npend = 0, ipend = 0;   /* guesses already fed (forward_spec) */
    int next = -1;                                     /* a token forward_spec drew, not fed yet */
    for (int step = 0; step < s->gen; step++) {
        if (g_stop) { interrupted = 1; break; }
        const int fed = ipend < npend;
        int best;
        if (fed) best = pend[ipend++];
        else if (next >= 0) { best = next; next = -1; }
        else best = glm53f_sample(&s->smp, s->logits, e->n_vocab);
        if (e->is_eos(e, best)) { stopped = 1; break; }
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
        if (!in_think && ok > printed) {
            k->answer(k, s->text + printed, ok - printed);
            printed = ok;
        }
        if (fed || step + 1 >= s->gen) continue;
        int frc = 0;
        if (e->forward_spec && s->gen - step > GLM53F_SPEC_MAX + 1) {
            frc = e->forward_spec(e, best, pend, s->logits, &s->smp, &next);
            if (frc >= 0) { npend = frc; ipend = 0; }
        } else frc = e->forward(e, &best, 1, s->logits);
        if (frc < 0) {
            k->status(k, NULL, 0.0);
            k->done(k, "the engine failed while writing the answer", 2);
            return -1;
        }
    }
    k->status(k, NULL, 0.0);
    const double t_dec = now_s() - td;
    c1 = c0;
    if (e->counters) e->counters(e, &c1);

    s->turns++;
    s->pre_tok += n;
    s->pre_s += t_prefill;
    if (nout) {
        s->dec_tok += nout;
        s->dec_s += t_dec;
        s->dec_wait_s += c1.wait - c0.wait;
        s->dec_read_gb += (double)(c1.bytes - c0.bytes) / 1e9;
        const uint64_t hits = c1.hits - c0.hits, ahead = c1.ahead - c0.ahead;
        s->dec_ahead += ahead < hits ? ahead : hits;
        s->dec_demand += c1.misses - c0.misses;
        s->dec_reqs += hits + (c1.misses - c0.misses);
        s->last_spt = t_dec / nout;
        if (s->n_spt == 16) { memmove(s->spt_hist, s->spt_hist + 1, 15 * sizeof *s->spt_hist); s->n_spt--; }
        s->spt_hist[s->n_spt++] = s->last_spt;
    }

    char stats[256], d[32];
    duration(d, sizeof d, t_dec);
    snprintf(stats, sizeof stats, "%d tokens · %.1f tok/s · %.3f s/token · %s · read %d tokens in %.1f s%s",
             nout, t_dec > 0.0 ? nout / t_dec : 0.0, nout ? t_dec / nout : 0.0, d, n, t_prefill,
             interrupted ? " · interrupted" : stopped ? "" : " · reached /gen");
    int len = nout ? e->decode(e, s->out, nout, s->text, s->text_cap - 1) : 0;
    if (len < 0) len = 0;
    s->text[len] = 0;
    log_turn(s, in->shown ? in->shown : in->prompt, s->text, len, stats);
    {
        const char *close = strstr(s->text, "</think>");
        free(s->last_answer);
        s->last_answer = NULL;
        if (close) {
            close += 8;
            while (*close == '\n' || *close == ' ') close++;
            if (*close) s->last_answer = dup_str(close);
        }
    }
    lc_session_answer(s, s->last_answer ? s->last_answer : "", NULL, stats);
    k->done(k, stats, interrupted);
    return 0;
}

/* ------------------------------------------------------------ plain front end ---- */
/* Line by line, no colour: for pipes, captured sessions and terminals that cannot host
 * the full-screen front end. */

#ifdef _WIN32
static BOOL WINAPI on_ctrl(DWORD type)
{
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) { g_stop = 1; return TRUE; }
    return FALSE;
}
#endif

static void plain_status(Glm53fTurnSink *k, const char *text, double t0)
{
    (void)k; (void)t0;
    if (text && !strncmp(text, "Reading", 7)) printf("  %s…\n", text);
}
static void plain_reasoning(Glm53fTurnSink *k, const char *p, int n) { (void)k; fwrite(p, 1, (size_t)n, stdout); fflush(stdout); }
static void plain_thought(Glm53fTurnSink *k, double sec, int tokens)
{
    char d[32];
    int *shown = (int *)k->ctx;
    duration(d, sizeof d, sec);
    printf("%s  + Thought: %s · %d tokens\n\n", *shown ? "\n\n" : "", d, tokens);
}
static void plain_answer(Glm53fTurnSink *k, const char *p, int n) { (void)k; fwrite(p, 1, (size_t)n, stdout); fflush(stdout); }
static const char *g_name = "";

static void plain_tool(Glm53fTurnSink *k, const char *title, const char *summary, int ok)
{
    (void)k;
    printf("\n  %s %s · %s\n", ok ? "⚙" : "✗", title, summary);
    fflush(stdout);
}

static int read_line(char *buf, int cap);

static int plain_approve(Glm53fTurnSink *k, const char *title, const char *detail)
{
    (void)k;
    char a[64];
    printf("\n  ? allow %s\n", title);
    if (detail && *detail) printf("%s\n", detail);
    for (;;) {
        printf("  [y] yes  [a] always for this tool  [n] no: ");
        fflush(stdout);
        if (read_line(a, sizeof a) < 0) return 0;
        const char c = (char)(a[0] | 32);
        if (c == 'y' || c == 's') return 1;
        if (c == 'a' || c == 't') return 2;
        if (c == 'n') return 0;
    }
}

static void plain_done(Glm53fTurnSink *k, const char *stats, int state)
{
    (void)k;
    if (state == 2) printf("  %s\n", stats);
    else            printf("\n\n  ■ localcode · %s · %s\n", g_name, stats);
}

static int read_line(char *buf, int cap)
{
#ifdef _WIN32
    const HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(in, &mode)) {
        static wchar_t w[16384];
        DWORD n = 0;
        if (!ReadConsoleW(in, w, (DWORD)(sizeof w / sizeof *w) - 1, &n, NULL)) return -1;
        w[n] = 0;
        const int got = WideCharToMultiByte(CP_UTF8, 0, w, (int)n, buf, cap - 1, NULL, NULL);
        buf[got > 0 ? got : 0] = 0;
        return got > 0 ? got : 0;
    }
#endif
    if (!fgets(buf, cap, stdin)) return -1;
    return (int)strlen(buf);
}

/* A mod's pane while it holds a call (plain front end): drawn once, answered by a line (a button's
 * hotkey or the start of its label; an empty line or the end of the input stops what waits). */
static void plain_mod_pump(void)
{
    static int answered = 0;
    char title[160];
    int esc = 0;
    if (!lc_mods_pane(title, sizeof title, &esc)) { answered = 0; return; }
    if (answered) return;
    LcModView v;
    if (!lc_mods_view("Pane", 78, 40, &v)) return;
    printf("\n");
    for (int i = 0; i < v.n; i++) printf("  %s\n", v.lines[i]);
    printf("  ");
    for (int i = 0; i < v.nb; i++) printf("%s[%s] %s", i ? " · " : "", v.button[i].hotkey, v.button[i].label);
    printf("\n> ");
    fflush(stdout);
    char line[256];
    int pressed = 0;
    if (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        for (int i = 0; i < v.nb && !pressed && line[0]; i++)
            if (!strcmp(line, v.button[i].hotkey) || !strncmp(v.button[i].label, line, strlen(line))) {
                printf("  (%s)\n", v.button[i].label);
                lc_mods_press(v.button[i].id);
                pressed = 1;
            }
    }
    if (!pressed) {
        if (esc) lc_mods_pane_close();
        else lc_mods_abort();
    }
    answered = 1;
    lc_mods_view_free(&v);
}

static int plain_run(Glm53fChat *s)
{
#ifdef _WIN32
    SetConsoleCtrlHandler(on_ctrl, TRUE);
#endif
    const int cap = 65536;
    char *line = (char *)malloc((size_t)cap);
    if (!line) return 1;
    g_name = s->e->name;
    printf("localcode · %s · ready in %.1f s · /help for commands\n", s->e->name, s->load_s);
    if (s->start_note) printf("  %s\n", s->start_note);
    int rc = 0, shown = 0;
    Glm53fTurnSink k = { plain_status, plain_reasoning, plain_thought, plain_answer, plain_done, &shown, plain_tool, plain_approve,
                         plain_mod_pump };
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (read_line(line, cap) < 0) break;
        Glm53fInput in;
        glm53f_chat_input(s, line, &in);
        if (in.kind == GLM53F_IN_QUIT) { glm53f_chat_input_free(&in); break; }
        if (in.note) printf("  %s\n", in.note);
        if (in.kind == GLM53F_IN_MESSAGE) {
            shown = s->show_thinking;
            printf("\n");
            if (glm53f_chat_turn(s, &in, &k) != 0) rc = 1;
        }
        glm53f_chat_input_free(&in);
        if (rc) break;
    }
    free(line);
    return rc;
}

void (*glm53f_mods_changed)(void) = NULL;

static void mods_changed(void) { if (glm53f_mods_changed) glm53f_mods_changed(); }

int glm53f_chat_run(Glm53fChat *s)
{
    s->first = 1;
    /* the mods the user allowed (lc_mods.c), beside localcode while it runs */
    if (s->agent_level > 0) {
        lc_mods_usage(s->e->cached, s->e->cap);
        lc_mods_start(lc_tools_root(), mods_changed);
    }
    if (s->resume) {                             /* --continue: the latest conversation here */
        int ok = 0;
        s->start_note = lc_session_resume(s, s->resume, &ok);
    }
    /* LOCALCODE_TUI=1 forces the full-screen front end (captured sessions); LOCALCODE_PLAIN=1
     * or NO_COLOR keeps the plain one. */
    const char *force = getenv("LOCALCODE_TUI");
    const int forced = force && strcmp(force, "0") != 0;
    int rc = -1;
    if (!getenv("LOCALCODE_PLAIN") && !getenv("NO_COLOR")) rc = glm53f_tui_run(s, forced);
    if (rc < 0) rc = plain_run(s);
    if (s->sess && s->e->state_save && s->e->cached > 0) {
        printf("localcode · keeping this conversation's context for /resume…\n");
        fflush(stdout);
    }
    lc_mods_stop();
    lc_session_end(s, 1);                        /* the context too, for an exact /resume */
    free(s->start_note);
    s->start_note = NULL;
    free(s->log);
    s->log = NULL;
    free(s->last_answer);
    s->last_answer = NULL;
    return rc;
}
