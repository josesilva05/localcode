/* lc_session.c - localcode's conversations on disk, and compaction near the context's end.
 *
 * A conversation is a JSONL file in %LOCALAPPDATA%\localcode\sessions named by its start
 * time, one record per line:
 *   {"t":"meta","cwd":...,"model":...,"level":2,"ctx":32768}   when the first message comes
 *   {"t":"user","text":...}                                    each message, as shown
 *   {"t":"answer","text":...,"actions":...,"stats":...}        each answer (actions: tool list)
 *   {"t":"compact","text":...,"from":N}                        each summary that replaced the context
 *   {"t":"state","file":...,"pos":N,"level":L}                 the engine state kept on leaving
 *
 * /resume puts a conversation back: from its state file when one is kept (the context exactly
 * as it was, in a second), else from the text: the last summary and the messages after it
 * open the next message, within a quarter of the context.
 *
 * Compaction: when a step would leave less than lc_compact_room() positions, the model is
 * asked, in the context itself, for a summary written for its own continuation (no reasoning
 * block, at most cap/16 tokens, 768..8192); the context is emptied and the summary opens the
 * next one. A conversation goes on that way past any context size, as often as it needs.
 */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#endif

#include "glm53f_chat.h"
#include "lc_tools.h"

#define STATE_KEEP   3            /* engine state files kept (the latest conversations) */

typedef struct {
    FILE *f;
    char  path[1024];             /* the .jsonl */
    char  stamp[40];              /* its name without the extension */
} LcSess;

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

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static char *dup_str(const char *p)
{
    const size_t n = strlen(p) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, p, n);
    return d;
}

/* ------------------------------------------------------------------ files ---- */

static int sess_dir(char *dir, size_t cap)
{
    const char *base = getenv("LOCALAPPDATA");
    if (!base) base = getenv("HOME");
    if (!base) return -1;
    char top[1024];
    snprintf(top, sizeof top, "%s/localcode", base);
    if (snprintf(dir, cap, "%s/sessions", top) >= (int)cap) return -1;
#ifdef _WIN32
    CreateDirectoryA(top, NULL);
    CreateDirectoryA(dir, NULL);
#else
    mkdir(top, 0755);
    mkdir(dir, 0755);
#endif
    return 0;
}

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static char *read_all(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    Str s = { 0 };
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s_add(&s, buf, n);
    fclose(f);
    return s.p ? s.p : dup_str("");
}

/* free bytes on the drive holding path, -1 unknown */
static long long free_bytes(const char *path)
{
#ifdef _WIN32
    ULARGE_INTEGER avail;
    if (GetDiskFreeSpaceExA(path, &avail, NULL, NULL)) return (long long)avail.QuadPart;
#else
    (void)path;
#endif
    return -1;
}

/* ------------------------------------------------------------------- JSON ---- */

static void json_put(Str *o, const char *p)
{
    s_str(o, "\"");
    for (; p && *p; p++) {
        const unsigned char c = (unsigned char)*p;
        if (c == '"')       s_str(o, "\\\"");
        else if (c == '\\') s_str(o, "\\\\");
        else if (c == '\n') s_str(o, "\\n");
        else if (c == '\r') s_str(o, "\\r");
        else if (c == '\t') s_str(o, "\\t");
        else if (c < 0x20)  s_fmt(o, "\\u%04x", c);
        else                s_add(o, p, 1);
    }
    s_str(o, "\"");
}

/* The string value of "key" in one record line (our own records: values are escaped, so an
 * unescaped quote only ever closes one); NULL when absent. Malloc'd. */
static char *json_get(const char *line, const char *key)
{
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(line, pat);
    if (!p) return NULL;
    p += strlen(pat);
    Str o = { 0 };
    if (*p != '"') {                              /* a number */
        const char *q = p;
        while (*q && *q != ',' && *q != '}') q++;
        s_add(&o, p, (size_t)(q - p));
        return o.p ? o.p : dup_str("");
    }
    for (p++; *p && *p != '"'; p++) {
        if (*p != '\\') { s_add(&o, p, 1); continue; }
        p++;
        if (!*p) break;
        if (*p == 'n') s_str(&o, "\n");
        else if (*p == 'r') s_str(&o, "\r");
        else if (*p == 't') s_str(&o, "\t");
        else if (*p == 'u' && p[1] && p[2] && p[3] && p[4]) {
            char hex[5] = { p[1], p[2], p[3], p[4], 0 };
            const unsigned v = (unsigned)strtoul(hex, NULL, 16);
            if (v < 0x80) { const char c = (char)v; s_add(&o, &c, 1); }
            else if (v < 0x800) { const char b[2] = { (char)(0xC0 | v >> 6), (char)(0x80 | (v & 63)) }; s_add(&o, b, 2); }
            else { const char b[3] = { (char)(0xE0 | v >> 12), (char)(0x80 | (v >> 6 & 63)), (char)(0x80 | (v & 63)) }; s_add(&o, b, 3); }
            p += 4;
        } else s_add(&o, p, 1);
    }
    return o.p ? o.p : dup_str("");
}

static int json_int(const char *line, const char *key, int def)
{
    char *v = json_get(line, key);
    const int r = v ? atoi(v) : def;
    free(v);
    return r;
}

/* ---------------------------------------------------------------- writing ---- */

static void put_line(Glm53fChat *s, Str *o)
{
    LcSess *x = (LcSess *)s->sess;
    if (!x || !x->f || !o->p) { free(o->p); return; }
    s_str(o, "\n");
    fwrite(o->p, 1, o->n, x->f);
    fflush(x->f);
    free(o->p);
}

/* the session file, opened with the first message of a conversation */
static LcSess *sess_open(Glm53fChat *s)
{
    if (s->sess) return (LcSess *)s->sess;
    char dir[1024];
    if (sess_dir(dir, sizeof dir) != 0) return NULL;
    LcSess *x = (LcSess *)calloc(1, sizeof *x);
    if (!x) return NULL;
    const time_t now = time(NULL);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", localtime(&now));
    snprintf(x->stamp, sizeof x->stamp, "%s", stamp);
    if (snprintf(x->path, sizeof x->path, "%s/%s.jsonl", dir, x->stamp) >= (int)sizeof x->path) { free(x); return NULL; }
    for (int i = 2; file_exists(x->path) && i < 100; i++) {
        snprintf(x->stamp, sizeof x->stamp, "%s-%d", stamp, i);
        if (snprintf(x->path, sizeof x->path, "%s/%s.jsonl", dir, x->stamp) >= (int)sizeof x->path) { free(x); return NULL; }
    }
    x->f = fopen(x->path, "ab");
    if (!x->f) { free(x); return NULL; }
    s->sess = x;
    Str o = { 0 };
    s_str(&o, "{\"t\":\"meta\",\"cwd\":");
    json_put(&o, lc_tools_root() ? lc_tools_root() : "");
    s_str(&o, ",\"model\":");
    json_put(&o, s->e->name);
    s_str(&o, ",\"variant\":");
    json_put(&o, s->e->variant ? s->e->variant : "");
    s_fmt(&o, ",\"level\":%d,\"ctx\":%d,\"reasoning\":", s->agent_level, s->e->cap);
    json_put(&o, s->reasoning);
    s_str(&o, "}");
    put_line(s, &o);
    return x;
}

void lc_session_user(Glm53fChat *s, const char *shown)
{
    if (!sess_open(s)) return;
    Str o = { 0 };
    s_str(&o, "{\"t\":\"user\",\"text\":");
    json_put(&o, shown);
    s_str(&o, "}");
    put_line(s, &o);
}

void lc_session_answer(Glm53fChat *s, const char *answer, const char *actions, const char *stats)
{
    if (!s->sess) return;
    Str o = { 0 };
    s_str(&o, "{\"t\":\"answer\",\"text\":");
    json_put(&o, answer ? answer : "");
    s_str(&o, ",\"actions\":");
    json_put(&o, actions ? actions : "");
    s_str(&o, ",\"stats\":");
    json_put(&o, stats ? stats : "");
    s_fmt(&o, ",\"pos\":%d}", s->e->cached);
    put_line(s, &o);
}

/* Keep the STATE_KEEP newest engine state files of the sessions folder. */
static void prune_states(const char *dir)
{
#ifdef _WIN32
    char pat[1100];
    snprintf(pat, sizeof pat, "%s/*.q38s", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    char names[64][MAX_PATH];
    FILETIME times[64];
    int n = 0;
    do {
        if (n < 64) { snprintf(names[n], MAX_PATH, "%s", fd.cFileName); times[n] = fd.ftLastWriteTime; n++; }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    for (int i = 0; i < n; i++) {                 /* delete those with STATE_KEEP newer ones */
        int newer = 0;
        for (int j = 0; j < n; j++) if (j != i && CompareFileTime(&times[j], &times[i]) > 0) newer++;
        if (newer >= STATE_KEEP) {
            char p[1400];
            snprintf(p, sizeof p, "%s/%s", dir, names[i]);
            DeleteFileA(p);
        }
    }
#else
    (void)dir;
#endif
}

void lc_session_end(Glm53fChat *s, int keep_state)
{
    LcSess *x = (LcSess *)s->sess;
    Glm53fEngine *e = s->e;
    if (x && keep_state && e->state_save && e->cached > 0) {
        char dir[1024], path[1200];
        if (sess_dir(dir, sizeof dir) == 0) {
            /* about 75 KB per position on this model, and the fixed recurrent state */
            const long long need = (long long)e->cached * 80000LL + (64LL << 20);
            const long long avail = free_bytes(dir);
            if (avail < 0 || avail > need + 30000000000LL) {
                snprintf(path, sizeof path, "%s/%s.q38s", dir, x->stamp);
                const int pos = e->state_save(e, path);
                if (pos == e->cached) {
                    Str o = { 0 };
                    s_str(&o, "{\"t\":\"state\",\"file\":");
                    char name[64];
                    snprintf(name, sizeof name, "%s.q38s", x->stamp);
                    json_put(&o, name);
                    s_str(&o, ",\"variant\":");
                    json_put(&o, e->variant ? e->variant : "");
                    s_fmt(&o, ",\"pos\":%d,\"level\":%d,\"ctx\":%d}", pos, s->agent_level, e->cap);
                    put_line(s, &o);
                    prune_states(dir);
                } else {
                    remove(path);
                }
            }
        }
    }
    if (x) {
        if (x->f) fclose(x->f);
        free(x);
    }
    s->sess = NULL;
    free(s->preface);
    s->preface = NULL;
    s->compactions = 0;
}

/* ---------------------------------------------------------------- reading ---- */

typedef struct {
    char name[64];                /* stamp */
    char path[1100];
    char first[96];               /* the first message, cut */
    int  messages, compacts, level, state;
} Entry;

static int same_dir(const char *a, const char *b)
{
    for (; *a && *b; a++, b++) {
        char x = *a, y = *b;
        if (x == '\\') x = '/';
        if (y == '\\') y = '/';
        if (x >= 'A' && x <= 'Z') x = (char)(x + 32);
        if (y >= 'A' && y <= 'Z') y = (char)(y + 32);
        if (x != y) return 0;
    }
    return *a == *b;
}

static void scan_entry(const char *dir, const char *fname, const char *cwd, Entry *out, int *n, int max)
{
    if (*n >= max) return;
    Entry *en = &out[*n];
    memset(en, 0, sizeof *en);
    snprintf(en->path, sizeof en->path, "%s/%s", dir, fname);
    snprintf(en->name, sizeof en->name, "%.*s", (int)(strlen(fname) - 6), fname);
    char *all = read_all(en->path);
    if (!all) return;
    int ok = 0;
    for (char *line = all, *nl; line && *line; line = nl ? nl + 1 : NULL) {
        nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char *t = json_get(line, "t");
        if (!t) continue;
        if (!strcmp(t, "meta")) {
            char *c = json_get(line, "cwd");
            ok = c && cwd && same_dir(c, cwd);
            en->level = json_int(line, "level", 0);
            free(c);
        } else if (!strcmp(t, "user")) {
            if (!en->messages) {
                char *u = json_get(line, "text");
                if (u) {
                    for (char *q = u; *q; q++) if (*q == '\n' || *q == '\r') *q = ' ';
                    snprintf(en->first, sizeof en->first, "%.80s%s", u, strlen(u) > 80 ? "…" : "");
                }
                free(u);
            }
            en->messages++;
            en->state = 0;
        } else if (!strcmp(t, "compact")) {
            en->compacts++;
        } else if (!strcmp(t, "state")) {
            char *f = json_get(line, "file");
            char p[1200];
            snprintf(p, sizeof p, "%s/%s", dir, f ? f : "");
            en->state = f && file_exists(p);
            free(f);
        }
        free(t);
    }
    free(all);
    if (ok && en->messages) (*n)++;
}

/* This folder's conversations, newest first. */
static int list_entries(Glm53fChat *s, Entry *out, int max)
{
    char dir[1024];
    if (sess_dir(dir, sizeof dir) != 0) return 0;
    const char *cwd = lc_tools_root();
    const LcSess *cur = (const LcSess *)s->sess;
    int n = 0;
#ifdef _WIN32
    char pat[1100];
    snprintf(pat, sizeof pat, "%s/*.jsonl", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    char (*names)[MAX_PATH] = (char (*)[MAX_PATH])malloc(512 * sizeof *names);
    int nn = 0;
    if (names) {
        do { if (nn < 512) snprintf(names[nn++], MAX_PATH, "%s", fd.cFileName); } while (FindNextFileA(h, &fd));
    }
    FindClose(h);
    /* newest first: the names are start times */
    for (int i = 0; i < nn; i++)
        for (int j = i + 1; j < nn; j++)
            if (strcmp(names[j], names[i]) > 0) { char t[MAX_PATH]; memcpy(t, names[i], MAX_PATH); memcpy(names[i], names[j], MAX_PATH); memcpy(names[j], t, MAX_PATH); }
    for (int i = 0; i < nn && n < max; i++) {
        if (cur && !strncmp(names[i], cur->stamp, strlen(cur->stamp)) && names[i][strlen(cur->stamp)] == '.') continue;
        scan_entry(dir, names[i], cwd, out, &n, max);
    }
    free(names);
#else
    (void)cur;
    (void)cwd;
#endif
    return n;
}

char *lc_session_list(Glm53fChat *s)
{
    Entry en[10];
    const int n = list_entries(s, en, 10);
    if (!n) return dup_str("no earlier conversations in this folder");
    Str o = { 0 };
    s_str(&o, "conversations in this folder (/resume N):\n");
    for (int i = 0; i < n; i++) {
        const char *st = en[i].name;              /* 20261006-150212 */
        s_fmt(&o, "%2d  %.4s-%.2s-%.2s %.2s:%.2s  level %d · %d message%s%s%s · %s\n", i + 1, st, st + 4, st + 6,
              st + 9, st + 11, en[i].level, en[i].messages, en[i].messages == 1 ? "" : "s",
              en[i].compacts ? " · compacted" : "", en[i].state ? " · exact" : "", en[i].first);
    }
    if (o.n && o.p[o.n - 1] == '\n') o.p[--o.n] = 0;
    return o.p;
}

char *lc_session_recent(Glm53fChat *s, int max)
{
    Entry en[8];
    if (max > 8) max = 8;
    const int n = list_entries(s, en, max);
    if (!n) return NULL;
    Str o = { 0 };
    for (int i = 0; i < n; i++) {
        const char *st = en[i].name;
        s_fmt(&o, "%.2s/%.2s %.2s:%.2s  %s\n", st + 6, st + 4, st + 9, st + 11, en[i].first);
    }
    return o.p;
}

/* the conversation as text for a new context: the last summary and what came after it */
static char *text_preface(const char *all, int budget_chars, int *n_msgs)
{
    /* find the last summary */
    const char *from = all;
    char *summary = NULL;
    for (const char *p = all; p && *p;) {
        const char *nl = strchr(p, '\n');
        const size_t len = nl ? (size_t)(nl - p) : strlen(p);
        if (len > 16 && !strncmp(p, "{\"t\":\"compact\"", 14)) {
            char *line = (char *)malloc(len + 1);
            if (line) {
                memcpy(line, p, len);
                line[len] = 0;
                free(summary);
                summary = json_get(line, "text");
                free(line);
                from = nl ? nl + 1 : p + len;
            }
        }
        p = nl ? nl + 1 : NULL;
    }
    /* the messages after it, as lines; the oldest go first when over the budget */
    char **items = NULL;
    int ni = 0, cap = 0;
    for (const char *p = from; p && *p;) {
        const char *nl = strchr(p, '\n');
        const size_t len = nl ? (size_t)(nl - p) : strlen(p);
        char *line = (char *)malloc(len + 1);
        if (!line) break;
        memcpy(line, p, len);
        line[len] = 0;
        char *t = json_get(line, "t");
        Str it = { 0 };
        if (t && !strcmp(t, "user")) {
            char *u = json_get(line, "text");
            s_str(&it, "User: ");
            s_str(&it, u ? u : "");
            free(u);
        } else if (t && !strcmp(t, "answer")) {
            char *a = json_get(line, "text"), *ac = json_get(line, "actions");
            s_str(&it, "localcode: ");
            if (ac && *ac) { s_str(&it, "(actions)\n"); s_str(&it, ac); }
            s_str(&it, a ? a : "");
            free(a);
            free(ac);
        }
        free(t);
        free(line);
        if (it.p) {
            if (ni == cap) {
                cap = cap ? cap * 2 : 64;
                char **g = (char **)realloc(items, (size_t)cap * sizeof *items);
                if (!g) { free(it.p); break; }
                items = g;
            }
            items[ni++] = it.p;
        }
        p = nl ? nl + 1 : NULL;
    }
    int keep = ni;
    size_t total = summary ? strlen(summary) : 0;
    for (int i = ni - 1; i >= 0; i--) {
        if (total + strlen(items[i]) > (size_t)budget_chars) { keep = ni - 1 - i; break; }
        total += strlen(items[i]) + 2;
    }
    Str o = { 0 };
    s_str(&o, "[This conversation continues an earlier session of localcode in this folder. What came before:]\n");
    if (summary) {
        s_str(&o, "<summary>\n");
        const size_t sl = strlen(summary);
        s_add(&o, summary, sl < (size_t)budget_chars ? sl : (size_t)budget_chars);
        s_str(&o, "\n</summary>\n");
    }
    if (keep) {
        s_str(&o, "<earlier_messages>\n");
        if (keep < ni) s_fmt(&o, "(%d older messages left out)\n", ni - keep);
        for (int i = ni - keep; i < ni; i++) { s_str(&o, items[i]); s_str(&o, "\n\n"); }
        s_str(&o, "</earlier_messages>\n");
    }
    s_str(&o, "\n[The new message:]\n");
    for (int i = 0; i < ni; i++) free(items[i]);
    free(items);
    free(summary);
    *n_msgs = keep;
    return o.p;
}

char *lc_session_resume(Glm53fChat *s, const char *arg, int *ok)
{
    Glm53fEngine *e = s->e;
    *ok = 0;
    Entry en[20];
    const int n = list_entries(s, en, 20);
    int pick = arg && *arg ? atoi(arg) : 1;
    if (!n) return dup_str("no earlier conversations in this folder");
    if (pick < 1 || pick > n) {
        char m[96];
        snprintf(m, sizeof m, "/resume takes 1 to %d (/sessions lists them)", n);
        return dup_str(m);
    }
    const Entry *pe = &en[pick - 1];
    char *all = read_all(pe->path);
    if (!all) return dup_str("cannot read that conversation");

    /* the records that set the context back */
    int level = pe->level, pos = -1, last_pos = -1, ctx = 0;
    char *state = NULL, *last_user = NULL, *last_answer = NULL;
    Str log = { 0 };
    for (const char *p = all; p && *p;) {
        const char *nl = strchr(p, '\n');
        const size_t len = nl ? (size_t)(nl - p) : strlen(p);
        char *line = (char *)malloc(len + 1);
        if (!line) break;
        memcpy(line, p, len);
        line[len] = 0;
        char *t = json_get(line, "t");
        if (t && !strcmp(t, "user")) {
            free(last_user);
            last_user = json_get(line, "text");
            s_str(&log, "## Você\n\n");
            s_str(&log, last_user ? last_user : "");
            s_str(&log, "\n\n");
            free(state);
            state = NULL;                         /* a later message: the kept state is older */
        } else if (t && !strcmp(t, "answer")) {
            free(last_answer);
            last_answer = json_get(line, "text");
            last_pos = json_int(line, "pos", -1);
            char *ac = json_get(line, "actions");
            s_str(&log, "## localcode\n\n");
            if (ac && *ac) { s_str(&log, ac); s_str(&log, "\n"); }
            s_str(&log, last_answer ? last_answer : "");
            s_str(&log, "\n\n");
            free(ac);
        } else if (t && !strcmp(t, "state")) {
            free(state);
            state = json_get(line, "file");
            pos = json_int(line, "pos", -1);
            level = json_int(line, "level", level);
            ctx = json_int(line, "ctx", 0);
            /* a state of another model file (or of before variants were recorded) does not fit
             * this one, even with the same shapes: from the text instead */
            char *v = json_get(line, "variant");
            if (!v || !e->variant || strcmp(v, e->variant)) ctx = -1;
            free(v);
        }
        free(t);
        free(line);
        p = nl ? nl + 1 : NULL;
    }
    (void)last_pos;

    /* the current conversation ends; this one goes on in its own file */
    lc_session_end(s, 0);
    e->reset(e);
    s->first = 1;
    if (e->agent && level >= 0 && level <= 2) s->agent_level = level;
    s->compactions = pe->compacts;
    char dir[1024], spath[1200];
    int exact = 0;
    if (state && e->state_load && ctx == e->cap && sess_dir(dir, sizeof dir) == 0) {
        snprintf(spath, sizeof spath, "%s/%s", dir, state);
        const double t0 = now_s();
        if (file_exists(spath) && e->state_load(e, spath) == pos && pos > 0) {
            exact = 1;
            s->first = 0;
            (void)t0;
        } else {
            e->reset(e);
        }
        remove(spath);                            /* the conversation goes on: that state is old */
    }
    int n_msgs = 0;
    if (!exact) {
        int budget = e->cap / 4;
        if (budget > 16384) budget = 16384;
        s->preface = text_preface(all, budget * 3, &n_msgs);
    }
    LcSess *x = (LcSess *)calloc(1, sizeof *x);
    if (x) {
        snprintf(x->path, sizeof x->path, "%s", pe->path);
        snprintf(x->stamp, sizeof x->stamp, "%s", pe->name);
        x->f = fopen(x->path, "ab");
        if (x->f) s->sess = x; else free(x);
    }
    s->log_len = 0;
    if (log.p) glm53f_chat_log(s, log.p);

    Str o = { 0 };
    const char *st = pe->name;
    s_fmt(&o, "resumed the conversation of %.4s-%.2s-%.2s %.2s:%.2s · level %d · %d message%s · ", st, st + 4, st + 6,
          st + 9, st + 11, s->agent_level, pe->messages, pe->messages == 1 ? "" : "s");
    if (exact) s_fmt(&o, "context restored exactly (%d positions)", e->cached);
    else s_fmt(&o, "from its text: the next message carries %s%d earlier message%s",
               pe->compacts ? "the last summary and " : "", n_msgs, n_msgs == 1 ? "" : "s");
    if (last_user) {
        s_str(&o, "\n\nlast message: ");
        const size_t ul = strlen(last_user);
        s_add(&o, last_user, ul > 300 ? 300 : ul);
        if (ul > 300) s_str(&o, "…");
    }
    if (last_answer) {
        s_str(&o, "\nlast answer: ");
        const size_t al = strlen(last_answer);
        s_add(&o, last_answer, al > 500 ? 500 : al);
        if (al > 500) s_str(&o, "…");
    }
    free(state);
    free(last_user);
    free(last_answer);
    free(log.p);
    free(all);
    *ok = 1;
    return o.p;
}

/* ------------------------------------------------------------- compaction ---- */

static const char COMPACT_PROMPT[] =
    "The context window is nearly full, so this conversation will be cut here and continue from a summary that "
    "you write now. Write it for yourself, so that you can carry on without the earlier messages. Do not call any "
    "function. Use short headings:\n"
    "1. The user's requests and instructions, quoting the short ones word for word, with their preferences and "
    "what to avoid.\n"
    "2. What was done: files read, created or changed (paths and the gist of each change), commands run and "
    "their results.\n"
    "3. Decisions and why, and facts learned about the project (names, paths, line numbers, versions, errors).\n"
    "4. The current state: what works, what fails, what is in progress.\n"
    "5. The next steps, in order.\n"
    "Be complete but dense; leave out pleasantries and whatever no longer matters. Write in the language the "
    "user writes in (Portuguese: Brazilian Portuguese). Reply with the summary text only.";

static int summary_budget(int cap)
{
    int b = cap / 16;
    if (b < 768) b = 768;
    if (b > 8192) b = 8192;
    return b;
}

/* Room for the request and the summary; and nothing to gain while the conversation past the
 * opening is not even twice the summary (a big tool result in a young context: it is read as
 * is, and the compaction comes at a later step with what room is left). */
int lc_compact_due(const Glm53fChat *s, int need)
{
    const Glm53fEngine *e = s->e;
    const int budget = summary_budget(e->cap);
    if (s->first || e->cached + need + budget + 640 <= e->cap) return 0;
    return e->cached - s->base >= 2 * budget;
}

int lc_compact(Glm53fChat *s, Glm53fTurnSink *k, const char *request, const char *focus)
{
    Glm53fEngine *e = s->e;
    volatile int *stop = glm53f_chat_stop_flag();
    if (s->first || e->cached <= 0) return 1;
    const int before = e->cached;
    Str t = { 0 };
    s_str(&t, "<|im_end|>\n<|im_start|>user\n");
    s_str(&t, COMPACT_PROMPT);
    if (request && *request) {
        s_str(&t, "\n\nThe request in progress, word for word (keep it whole in the summary):\n");
        const size_t rl = strlen(request);
        s_add(&t, request, rl > 6000 ? 6000 : rl);
    }
    if (focus && *focus) { s_str(&t, "\n\nKeep in particular: "); s_str(&t, focus); }
    s_str(&t, "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    const int n = e->encode(e, t.p, (int)t.n, s->ids, s->ids_cap);
    free(t.p);
    int budget = summary_budget(e->cap);
    if (budget > GLM53F_CHAT_MAX_GEN) budget = GLM53F_CHAT_MAX_GEN;
    if (n <= 0 || e->cached + n + 256 > e->cap) return 1;
    if (e->cached + n + budget > e->cap) budget = e->cap - e->cached - n;

    char line[160];
    const double t0 = now_s();
    snprintf(line, sizeof line, "Compacting the conversation (%d of %d positions)", before, e->cap);
    k->status(k, line, t0);
    if (e->forward(e, s->ids, n, s->logits) != 0) { k->status(k, NULL, 0.0); return -1; }
    int nout = 0;
    const int was_stopped = *stop;
    for (int step = 0; step < budget; step++) {
        if (*stop && !was_stopped) break;         /* esc: keep what is written so far */
        int best = 0;
        for (int i = 1; i < e->n_vocab; i++) if (s->logits[i] > s->logits[best]) best = i;
        if (e->is_eos(e, best)) break;
        s->out[nout++] = best;
        if ((nout & 15) == 0) {
            snprintf(line, sizeof line, "Compacting the conversation · summary %d tokens", nout);
            k->status(k, line, t0);
        }
        if (step + 1 < budget && e->forward(e, &best, 1, s->logits) != 0) { k->status(k, NULL, 0.0); return -1; }
    }
    k->status(k, NULL, 0.0);
    int len = nout ? e->decode(e, s->out, nout, s->text, s->text_cap - 1) : 0;
    if (len < 0) len = 0;
    s->text[len] = 0;
    char *sum = s->text;
    char *call = strstr(sum, "<tool_call>");     /* a summary is text: a call is cut away */
    if (call) *call = 0;
    while (*sum == '\n' || *sum == ' ') sum++;
    if (!*sum) sum = (char *)"(no summary could be written; go on from the request and the files themselves)";

    /* the summary opens the next context */
    Str p = { 0 };
    s_str(&p, "[The earlier part of this conversation was compacted to make room; your summary of it:]\n<summary>\n");
    s_str(&p, sum);
    s_str(&p, "\n</summary>\n\n");
    free(s->preface);
    s->preface = p.p;
    s->compactions++;
    if (s->sess) {
        Str o = { 0 };
        s_str(&o, "{\"t\":\"compact\",\"text\":");
        json_put(&o, sum);
        s_fmt(&o, ",\"from\":%d,\"tokens\":%d}", before, nout);
        put_line(s, &o);
    }
    e->reset(e);
    s->first = 1;

    const double dt = now_s() - t0;
    char title[64], summary[200];
    snprintf(title, sizeof title, "compact");
    snprintf(summary, sizeof summary, "%d positions → a summary of %d tokens in %.0f s (compaction %d)", before, nout,
             dt, s->compactions);
    if (k->tool) k->tool(k, title, summary, nout > 0);
    char logl[260];
    snprintf(logl, sizeof logl, "*(conversa compactada: %s)*\n\n", summary);
    glm53f_chat_log(s, logl);
    return 0;
}

/* ------------------------------------------------------- autonomous folders ---- */
/* %LOCALAPPDATA%\localcode\auto_dirs.txt: one folder a line, where the agent works without
 * asking (/auto on). */
static int auto_path(char *path, size_t cap)
{
    char dir[1024];
    if (sess_dir(dir, sizeof dir)) return -1;            /* creates localcode\ */
    const char *base = getenv("LOCALAPPDATA");
    if (!base) base = getenv("HOME");
    snprintf(path, cap, "%s/localcode/auto_dirs.txt", base);
    return 0;
}

int lc_auto_dir(const char *dir)
{
    char path[1100], line[2048];
    if (!dir || auto_path(path, sizeof path)) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int hit = 0;
    while (!hit && fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        hit = line[0] && same_dir(line, dir);
    }
    fclose(f);
    return hit;
}

int lc_auto_dir_set(const char *dir, int on)
{
    char path[1100], line[2048];
    if (!dir || auto_path(path, sizeof path)) return -1;
    Str keep = { 0 };
    FILE *f = fopen(path, "rb");
    if (f) {
        while (fgets(line, sizeof line, f)) {
            line[strcspn(line, "\r\n")] = 0;
            if (line[0] && !same_dir(line, dir)) { s_str(&keep, line); s_str(&keep, "\n"); }
        }
        fclose(f);
    }
    if (on) { s_str(&keep, dir); s_str(&keep, "\n"); }
    f = fopen(path, "wb");
    if (!f) { free(keep.p); return -1; }
    if (keep.n) fwrite(keep.p, 1, keep.n, f);
    fclose(f);
    free(keep.p);
    return 0;
}
