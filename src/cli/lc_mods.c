/* lc_mods.c - see lc_mods.h and modhost.mjs. */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "json.h"
#include "lc_mods.h"
#include "lc_plugins.h"
#include "lc_tools.h"

#ifdef _WIN32
#define SEP "\\"
#else
#define SEP "/"

static void nap_ms(int ms)                     /* usleep is not in POSIX 2008 */
{
    const struct timespec t = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&t, NULL);
}
#endif

typedef struct { char *p; size_t n, cap; } Buf;

static void b_add(Buf *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 128;
        while (b->n + n + 1 > cap) cap *= 2;
        char *g = (char *)realloc(b->p, cap);
        if (!g) return;
        b->p = g;
        b->cap = cap;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void b_str(Buf *b, const char *s) { if (s) b_add(b, s, strlen(s)); }

static void b_fmt(Buf *b, const char *fmt, ...)
{
    char tmp[2048];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n > 0) b_add(b, tmp, (size_t)n < sizeof tmp ? (size_t)n : sizeof tmp - 1);
}

static char *xdup(const char *s)
{
    if (!s) return NULL;
    const size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

static void json_free(jval *v)
{
    if (!v) return;
    for (int i = 0; i < v->len; i++) {
        json_free(v->kids[i]);
        if (v->keys) free(v->keys[i]);
    }
    free(v->kids);
    free(v->keys);
    free(v->str);
    free(v);
}

static const char *jstr(jval *o, const char *key)
{
    jval *v = json_get(o, key);
    return v && v->t == J_STR ? v->str : NULL;
}

static double jnum(jval *o, const char *key, double def)
{
    jval *v = json_get(o, key);
    return v && v->t == J_NUM ? v->num : def;
}

static int jbool(jval *o, const char *key)
{
    jval *v = json_get(o, key);
    return v && ((v->t == J_BOOL && v->boolean) || (v->t == J_NUM && v->num != 0));
}

static char *quote(const char *s)
{
    Buf b = { 0 };
    b_add(&b, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        if (*p == '"') b_str(&b, "\\\"");
        else if (*p == '\\') b_str(&b, "\\\\");
        else if (*p == '\n') b_str(&b, "\\n");
        else if (*p == '\r') b_str(&b, "\\r");
        else if (*p == '\t') b_str(&b, "\\t");
        else if (*p < 0x20) b_fmt(&b, "\\u%04x", *p);
        else b_add(&b, (const char *)p, 1);
    }
    b_add(&b, "\"", 1);
    return b.p;
}

/* ------------------------------------------------------------ the state ---- */

#ifdef _WIN32
static CRITICAL_SECTION mu, wmu;
static CONDITION_VARIABLE cv;
#define LOCK()   EnterCriticalSection(&mu)
#define UNLOCK() LeaveCriticalSection(&mu)
#define WAKE()   WakeAllConditionVariable(&cv)
static HANDLE g_proc, g_to, g_from;
#else
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER, wmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
#define LOCK()   pthread_mutex_lock(&mu)
#define UNLOCK() pthread_mutex_unlock(&mu)
#define WAKE()   pthread_cond_broadcast(&cv)
static pid_t g_pid;
static int g_to = -1, g_from = -1;
#endif

#define MAX_REPLIES 64
static struct { int id; char *line; } g_rep[MAX_REPLIES];
static int  g_running, g_ready, g_seq, g_dead;
static int  g_pane, g_pane_esc, g_pane_focus;
static char g_pane_id[128], g_pane_title[160];
static char *g_toast;
static struct { char name[64], desc[200]; } g_cmd[32];
static int  g_ncmd, g_tokens, g_window;
static volatile int g_abort;                   /* lc_mods_abort: the waiting event is not wanted */
static char g_root[4096];
static void (*g_on_change)(void);

static void changed(void) { if (g_on_change) g_on_change(); }

static void send_line(const char *s)
{
    if (!g_running) return;
#ifdef _WIN32
    EnterCriticalSection(&wmu);
    DWORD w = 0;
    WriteFile(g_to, s, (DWORD)strlen(s), &w, NULL);
    WriteFile(g_to, "\n", 1, &w, NULL);
    LeaveCriticalSection(&wmu);
#else
    pthread_mutex_lock(&wmu);
    ssize_t r = write(g_to, s, strlen(s));
    r = write(g_to, "\n", 1);
    (void)r;
    pthread_mutex_unlock(&wmu);
#endif
}

static void mods_log(const char *s)
{
#ifdef _WIN32
    const char *b = getenv("LOCALAPPDATA");
    char path[1024];
    if (!b) return;
    snprintf(path, sizeof path, "%s\\localcode\\mods.log", b);
#else
    const char *b = getenv("HOME");
    char path[1024];
    if (!b) return;
    snprintf(path, sizeof path, "%s/.local/share/localcode/mods.log", b);
#endif
    FILE *f = fopen(path, "ab");
    if (!f) return;
    fprintf(f, "%s\n", s);
    fclose(f);
}

/* one line from the host */
static void on_line(char *line)
{
    jval *m = json_parse(line, NULL);
    if (!m) return;
    jval *id = json_get(m, "id"), *callk = json_get(m, "call");
    const char *note = jstr(m, "notify"), *lg = jstr(m, "log");
    if (json_get(m, "ready")) {
        LOCK();
        g_ready = 1;
        WAKE();
        UNLOCK();
    } else if (id && id->t == J_NUM) {
        LOCK();
        for (int i = 0; i < MAX_REPLIES; i++)
            if (!g_rep[i].line) { g_rep[i].id = (int)id->num; g_rep[i].line = xdup(line); break; }
        WAKE();
        UNLOCK();
    } else if (callk && callk->t == J_NUM) {
        const char *method = jstr(m, "method");
        jval *args = json_get(m, "args");
        char rep[200];
        if (method && !strcmp(method, "ui.open")) {
            LOCK();
            snprintf(g_pane_id, sizeof g_pane_id, "%s", jstr(args, "id") ? jstr(args, "id") : "pane");
            snprintf(g_pane_title, sizeof g_pane_title, "%s", jstr(args, "title") ? jstr(args, "title") : g_pane_id);
            g_pane_esc = jbool(args, "closeOnEscape");
            g_pane_focus = jbool(args, "focus");
            g_pane = 1;
            UNLOCK();
            snprintf(rep, sizeof rep, "{\"reply\": %d, \"value\": {\"isPlaced\": true}}", (int)callk->num);
            changed();
        } else {
            snprintf(rep, sizeof rep, "{\"reply\": %d, \"value\": null}", (int)callk->num);
        }
        send_line(rep);
    } else if (note) {
        jval *args = json_get(m, "args");
        if (!strcmp(note, "ui.toast")) {
            LOCK();
            free(g_toast);
            g_toast = xdup(jstr(args, "text"));
            UNLOCK();
        } else if (!strcmp(note, "ui.close")) {
            LOCK();
            if (!jstr(args, "id") || !strcmp(jstr(args, "id"), g_pane_id)) g_pane = 0;
            UNLOCK();
        } else if (!strcmp(note, "command.register") && jstr(args, "name") && g_ncmd < 32) {
            LOCK();
            snprintf(g_cmd[g_ncmd].name, sizeof g_cmd[g_ncmd].name, "%s", jstr(args, "name"));
            snprintf(g_cmd[g_ncmd].desc, sizeof g_cmd[g_ncmd].desc, "%s", jstr(args, "description") ? jstr(args, "description") : "");
            g_ncmd++;
            UNLOCK();
        }
        changed();
    } else if (lg) {
        mods_log(lg);
    }
    json_free(m);
}

#ifdef _WIN32
static DWORD WINAPI reader(LPVOID arg)
#else
static void *reader(void *arg)
#endif
{
    (void)arg;
    Buf acc = { 0 };
    char buf[8192];
    for (;;) {
#ifdef _WIN32
        DWORD got = 0;
        if (!ReadFile(g_from, buf, sizeof buf, &got, NULL) || !got) break;
#else
        ssize_t got = read(g_from, buf, sizeof buf);
        if (got <= 0) break;
#endif
        b_add(&acc, buf, (size_t)got);
        for (;;) {
            char *nl = acc.p ? memchr(acc.p, '\n', acc.n) : NULL;
            if (!nl) break;
            *nl = 0;
            if (nl > acc.p && nl[-1] == '\r') nl[-1] = 0;
            on_line(acc.p);
            const size_t used = (size_t)(nl - acc.p) + 1;
            memmove(acc.p, nl + 1, acc.n - used);
            acc.n -= used;
            acc.p[acc.n] = 0;
        }
    }
    free(acc.p);
    LOCK();
    g_dead = 1;
    g_pane = 0;
    WAKE();
    UNLOCK();
    mods_log("the mod host ended");
    changed();
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ------------------------------------------------------------- starting ---- */

/* modhost.mjs: beside localcode's program, or LOCALCODE_MODHOST */
static int host_path(char *out, size_t cap)
{
    const char *e = getenv("LOCALCODE_MODHOST");
    if (e && *e) { snprintf(out, cap, "%s", e); return 0; }
#ifdef _WIN32
    wchar_t w[4096];
    const DWORD n = GetModuleFileNameW(NULL, w, 4096);
    if (!n || n >= 4096) return -1;
    char exe[8192];
    WideCharToMultiByte(CP_UTF8, 0, w, -1, exe, sizeof exe, NULL, NULL);
    char *sl = strrchr(exe, '\\');
    if (!sl) return -1;
    *sl = 0;
    snprintf(out, cap, "%s\\modhost.mjs", exe);
#else
    char exe[4096];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return -1;
    exe[n] = 0;
    char *sl = strrchr(exe, '/');
    if (!sl) return -1;
    *sl = 0;
    snprintf(out, cap, "%s/modhost.mjs", exe);
#endif
    FILE *f = fopen(out, "rb");
    if (!f) return -1;
    fclose(f);
    return 0;
}

/* the modules of the mods that may run */
static int mod_modules(Buf *args)
{
    int n = 0;
    for (int i = 0; i < lc_plugins_n(); i++) {
        const LcPlugin *p = lc_plugin_at(i);
        if (!p->on || !p->mod || !p->trusted) continue;
        char path[4400];
        snprintf(path, sizeof path, "%s" SEP "hooks" SEP "hooks.json", p->dir);
        char *t = lc_slurp(path, NULL);
        jval *v = t ? json_parse(t, NULL) : NULL;
        free(t);
        jval *mods = v ? json_get(v, "modules") : NULL;
        for (int k = 0; mods && mods->t == J_ARR && k < mods->len; k++) {
            if (mods->kids[k]->t != J_STR) continue;
            const char *r = mods->kids[k]->str;
            while (r[0] == '.' && (r[1] == '/' || r[1] == '\\')) r += 2;
            char full[4600];
            snprintf(full, sizeof full, "%s" SEP "hooks" SEP "%s", p->dir, r);
            char *q = quote(full);
            b_str(args, " ");
            b_str(args, q);
            free(q);
            n++;
        }
        json_free(v);
    }
    return n;
}

int lc_mods_running(void) { return g_running && !g_dead; }

int lc_mods_start(const char *root, void (*on_change)(void))
{
    static int tried = 0;
    if (tried) return g_running ? 0 : -1;
    tried = 1;
    g_on_change = on_change;
    snprintf(g_root, sizeof g_root, "%s", root ? root : ".");
    lc_plugins_load(root);
    Buf args = { 0 };
    char host[4200];
    if (!mod_modules(&args) || host_path(host, sizeof host)) { free(args.p); return -1; }
#ifdef _WIN32
    InitializeCriticalSection(&mu);
    InitializeCriticalSection(&wmu);
    InitializeConditionVariable(&cv);
    Buf cl = { 0 };
    char *qh = quote(host);
    b_str(&cl, "node ");
    b_str(&cl, qh);
    b_str(&cl, args.p);
    free(qh);
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE in_r, in_w, out_r, out_w, nul;
    if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0)) { free(cl.p); free(args.p); return -1; }
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    nul = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);
    STARTUPINFOW si;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = nul;
    PROCESS_INFORMATION pi;
    const int wn = MultiByteToWideChar(CP_UTF8, 0, cl.p, -1, NULL, 0);
    wchar_t *wcl = (wchar_t *)malloc((size_t)wn * sizeof(wchar_t));
    const int wr = MultiByteToWideChar(CP_UTF8, 0, g_root, -1, NULL, 0);
    wchar_t *wcwd = (wchar_t *)malloc((size_t)wr * sizeof(wchar_t));
    if (wcl) MultiByteToWideChar(CP_UTF8, 0, cl.p, -1, wcl, wn);
    if (wcwd) MultiByteToWideChar(CP_UTF8, 0, g_root, -1, wcwd, wr);
    const BOOL ok = wcl && CreateProcessW(NULL, wcl, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, wcwd, &si, &pi);
    free(wcl);
    free(wcwd);
    free(cl.p);
    CloseHandle(in_r);
    CloseHandle(out_w);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) { CloseHandle(in_w); CloseHandle(out_r); free(args.p); mods_log("cannot start node (is it installed?)"); return -1; }
    CloseHandle(pi.hThread);
    g_proc = pi.hProcess;
    g_to = in_w;
    g_from = out_r;
    g_running = 1;
    HANDLE th = CreateThread(NULL, 0, reader, NULL, 0, NULL);
    if (th) CloseHandle(th);
#else
    int pin[2], pout[2];
    if (pipe(pin) || pipe(pout)) { free(args.p); return -1; }
    Buf sh = { 0 };
    char *qh = quote(host);
    b_str(&sh, "exec node ");
    b_str(&sh, qh);
    b_str(&sh, args.p);
    free(qh);
    g_pid = fork();
    if (g_pid == 0) {
        dup2(pin[0], 0); dup2(pout[1], 1);
        close(pin[1]); close(pout[0]);
        if (chdir(g_root)) _exit(127);
        execl("/bin/sh", "sh", "-c", sh.p, (char *)NULL);
        _exit(127);
    }
    free(sh.p);
    close(pin[0]); close(pout[1]);
    g_to = pin[1];
    g_from = pout[0];
    g_running = 1;
    pthread_t th;
    pthread_create(&th, NULL, reader, NULL);
    pthread_detach(th);
#endif
    free(args.p);
    /* ready, then session.start */
    int naps = 0;                                  /* 10 s at most (clock() is CPU time on POSIX) */
    LOCK();
    while (!g_ready && !g_dead && naps++ < 500) {
        UNLOCK();
#ifdef _WIN32
        Sleep(20);
#else
        nap_ms(20);
#endif
        LOCK();
    }
    const int ready = g_ready;
    UNLOCK();
    if (!ready) { mods_log("the mod host did not start"); return -1; }
    char *r = lc_mods_event("session.start", "{\"source\": \"startup\"}", 10000, NULL);
    free(r);
    return 0;
}

void lc_mods_stop(void)
{
    if (!g_running) return;
#ifdef _WIN32
    CloseHandle(g_to);
    if (WaitForSingleObject(g_proc, 1000) != WAIT_OBJECT_0) TerminateProcess(g_proc, 0);
#else
    close(g_to);
    kill(g_pid, SIGTERM);
#endif
    g_running = 0;
}

void lc_mods_usage(int tokens, int window)
{
    g_tokens = tokens;
    g_window = window;
}

/* --------------------------------------------------------------- events ---- */

char *lc_mods_event(const char *event, const char *data, int timeout_ms, void (*pump)(void))
{
    if (!lc_mods_running()) return NULL;
    LOCK();
    const int id = ++g_seq;
    UNLOCK();
    char *cwd = quote(g_root);
    Buf b = { 0 };
    b_fmt(&b, "{\"id\": %d, \"event\": \"%s\", \"data\": ", id, event);
    b_str(&b, data && *data ? data : "{}");
    b_fmt(&b, ", \"state\": {\"cwd\": %s, \"usage\": {\"tokens\": %d, \"window\": %d, \"percent\": %d}}}", cwd, g_tokens,
          g_window, g_window ? (int)(100.0 * g_tokens / g_window + 0.5) : 0);
    free(cwd);
    send_line(b.p);
    free(b.p);
    char *line = NULL;
    double waited = 0;
    for (;;) {
        LOCK();
        for (int i = 0; i < MAX_REPLIES; i++)
            if (g_rep[i].line && g_rep[i].id == id) { line = g_rep[i].line; g_rep[i].line = NULL; break; }
        const int dead = g_dead;
        UNLOCK();
        if (line || dead || waited > timeout_ms) break;
        if (g_abort) {                             /* the user stopped it: the mod hears so and answers */
            g_abort = 0;
            char ab[64];
            snprintf(ab, sizeof ab, "{\"abort\": %d}", id);
            send_line(ab);
        }
        if (pump) pump();
#ifdef _WIN32
        Sleep(20);
#else
        nap_ms(20);
#endif
        waited += 20;
    }
    if (!line) {                                   /* no answer: the host may stop working on it */
        char ab[64];
        snprintf(ab, sizeof ab, "{\"abort\": %d}", id);
        send_line(ab);
    }
    return line;
}

int lc_mods_tool_call(const char *tool, const char *input, char **why, void (*pump)(void))
{
    *why = NULL;
    if (!lc_mods_running()) return 0;
    Buf d = { 0 };
    b_fmt(&d, "{\"tool\": \"%s\"", tool);
    const char *in = input ? strchr(input, '{') : NULL;
    if (in) {
        in++;
        while (*in == ' ') in++;
        if (*in && *in != '}') { b_str(&d, ", "); b_str(&d, in); }
        else b_str(&d, "}");
    } else {
        b_str(&d, "}");
    }
    char *line = lc_mods_event("tool.call", d.p, 11 * 60 * 1000, pump);   /* a hold: up to 10 minutes */
    free(d.p);
    if (!line) return 0;
    jval *m = json_parse(line, NULL);
    free(line);
    jval *r = json_get(m, "result");
    const char *deny = jstr(r, "deny");
    int refused = 0;
    if (deny) { *why = xdup(deny); refused = 1; }
    json_free(m);
    return refused;
}

/* --------------------------------------------------------------- layout ---- */

/* visible columns of text with ANSI escapes */
static int vcols(const char *s, size_t n)
{
    int c = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\033') { while (i < n && !((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z' && s[i] != '[') || s[i] == 'm')) i++; continue; }
        if (((unsigned char)s[i] & 0xC0) != 0x80) c++;
    }
    return c;
}

typedef struct { Buf v[LC_MOD_LINES * 2]; int n; } Ls;

static void ls_free(Ls *l) { for (int i = 0; i < l->n; i++) free(l->v[i].p); l->n = 0; }
static Buf *ls_new(Ls *l) { if (l->n >= LC_MOD_LINES * 2) return NULL; memset(&l->v[l->n], 0, sizeof(Buf)); return &l->v[l->n++]; }

typedef struct { int bold, dim, italic, inverse, ul; const char *color; } Sty;

static const char *color_code(const char *c)
{
    static const struct { const char *n, *code; } t[] = {
        { "black", "30" }, { "red", "31" }, { "green", "32" }, { "yellow", "33" }, { "blue", "34" }, { "magenta", "35" },
        { "cyan", "36" }, { "white", "37" }, { "gray", "90" }, { "grey", "90" }, { "redBright", "91" }, { "greenBright", "92" },
        { "yellowBright", "93" }, { "blueBright", "94" }, { "magentaBright", "95" }, { "cyanBright", "96" }, { "whiteBright", "97" } };
    for (size_t i = 0; c && i < sizeof t / sizeof *t; i++) if (!strcmp(c, t[i].n)) return t[i].code;
    return NULL;
}

static void sty_open(Buf *b, const Sty *s)
{
    b_str(b, "\033[0");
    if (s->bold) b_str(b, ";1");
    if (s->dim) b_str(b, ";2");
    if (s->italic) b_str(b, ";3");
    if (s->ul) b_str(b, ";4");
    if (s->inverse) b_str(b, ";7");
    const char *cc = color_code(s->color);
    if (cc) { b_str(b, ";"); b_str(b, cc); }
    b_str(b, "m");
}

static Sty sty_of(jval *t, Sty in)
{
    if (jbool(t, "bold")) in.bold = 1;
    if (jbool(t, "dimColor")) in.dim = 1;
    if (jbool(t, "italic")) in.italic = 1;
    if (jbool(t, "inverse")) in.inverse = 1;
    if (jbool(t, "underline")) in.ul = 1;
    if (jstr(t, "color")) in.color = jstr(t, "color");
    return in;
}

/* a Text's spans on one line */
static void text_spans(jval *t, Sty s, Buf *out)
{
    jval *ch = json_get(t, "children");
    s = sty_of(t, s);
    for (int i = 0; ch && ch->t == J_ARR && i < ch->len; i++) {
        jval *c = ch->kids[i];
        if (c->t == J_STR) {
            sty_open(out, &s);
            for (const char *p = c->str; *p; p++) if (*p != '\n' && *p != '\r') b_add(out, p, 1); else b_add(out, " ", 1);
        } else if (c->t == J_OBJ) {
            text_spans(c, s, out);
        }
    }
}

/* the first n visible columns of a styled line */
static void cut_cols(Buf *b, int n)
{
    int c = 0;
    size_t i = 0;
    while (i < b->n) {
        if (b->p[i] == '\033') { while (i < b->n && b->p[i] != 'm') i++; i++; continue; }
        if (((unsigned char)b->p[i] & 0xC0) != 0x80) { if (c == n) break; c++; }
        i++;
    }
    while (i < b->n && ((unsigned char)b->p[i] & 0xC0) == 0x80) i++;
    if (i < b->n) {
        b->n = i;
        b->p[i] = 0;
        if (n > 0) { /* an ellipsis in the last column */ }
    }
    b_str(b, "\033[0m");
}

typedef struct { LcModView *v; int want_focus; } Ctx;

static void lay(jval *node, int width, Ls *out, Ctx *cx);

static void lay_button(jval *bt, Ls *out, Ctx *cx)
{
    Buf *l = ls_new(out);
    if (!l) return;
    LcModView *v = cx->v;
    const int k = v->nb;
    if (v->nb < LC_MOD_BUTTONS) {
        snprintf(v->button[k].id, sizeof v->button[k].id, "%s", jstr(bt, "id") ? jstr(bt, "id") : "");
        snprintf(v->button[k].hotkey, sizeof v->button[k].hotkey, "%s", jstr(bt, "hotkey") ? jstr(bt, "hotkey") : "");
        snprintf(v->button[k].label, sizeof v->button[k].label, "%s", jstr(bt, "label") ? jstr(bt, "label") : "");
        if (jbool(bt, "autoFocus") && v->focus < 0) v->focus = k;
        v->nb++;
    }
    const int focused = cx->want_focus >= 0 ? k == cx->want_focus : jbool(bt, "autoFocus");
    Sty s = { 0 };
    s.inverse = focused;
    s.bold = 1;
    sty_open(l, &s);
    if (jstr(bt, "hotkey")) b_fmt(l, " %s %s ", jstr(bt, "hotkey"), jstr(bt, "label") ? jstr(bt, "label") : "");
    else b_fmt(l, " %s ", jstr(bt, "label") ? jstr(bt, "label") : "");
    b_str(l, "\033[0m");
}

static void lay_text(jval *t, int width, Ls *out)
{
    Buf line = { 0 };
    Sty s = { 0 };
    text_spans(t, s, &line);
    b_str(&line, "\033[0m");
    const char *wrap = jstr(t, "wrap");
    if (wrap && !strcmp(wrap, "wrap") && vcols(line.p ? line.p : "", line.n) > width && width > 4) {
        /* wrapped: the plain text in pieces of width, in the Text's own style */
        Buf plain = { 0 };
        for (size_t i = 0; i < line.n; i++) {
            if (line.p[i] == '\033') { while (i < line.n && line.p[i] != 'm') i++; continue; }
            b_add(&plain, line.p + i, 1);
        }
        Sty ts = sty_of(t, s);
        const char *p = plain.p ? plain.p : "";
        while (*p) {
            Buf *l = ls_new(out);
            if (!l) break;
            sty_open(l, &ts);
            int c = 0;
            const char *q = p, *last_sp = NULL;
            while (*q && c < width) { if (*q == ' ') last_sp = q; q++; while (*q && ((unsigned char)*q & 0xC0) == 0x80) q++; c++; }
            if (*q && last_sp && last_sp > p) q = last_sp + 1;
            b_add(l, p, (size_t)(q - p));
            b_str(l, "\033[0m");
            p = q;
        }
        free(plain.p);
        free(line.p);
        return;
    }
    Buf *l = ls_new(out);
    if (l) { *l = line; if (vcols(l->p ? l->p : "", l->n) > width) cut_cols(l, width); }
    else free(line.p);
}

static void lay_box(jval *bx, int width, Ls *out, Ctx *cx)
{
    const int border = jstr(bx, "borderStyle") != NULL;
    const int px = (int)jnum(bx, "paddingX", jnum(bx, "padding", 0));
    const int gap = (int)jnum(bx, "gap", 0);
    const char *dir = jstr(bx, "flexDirection");
    const int row = !dir || !strcmp(dir, "row");
    const int inner = width - 2 * px - (border ? 2 : 0) > 1 ? width - 2 * px - (border ? 2 : 0) : 1;
    Ls body = { 0 };
    jval *ch = json_get(bx, "children");
    if (row) {
        /* each child's first line, side by side */
        Buf joined = { 0 };
        int used = 0, nkids = 0;
        Ls parts = { 0 };
        for (int i = 0; ch && ch->t == J_ARR && i < ch->len; i++) {
            Ls one = { 0 };
            lay(ch->kids[i], inner, &one, cx);
            if (one.n) { Buf *p = ls_new(&parts); if (p) { *p = one.v[0]; one.v[0].p = NULL; } nkids++; }
            ls_free(&one);
        }
        int total = 0;
        for (int i = 0; i < parts.n; i++) total += vcols(parts.v[i].p ? parts.v[i].p : "", parts.v[i].n);
        const char *jc = jstr(bx, "justifyContent");
        const int between = jc && !strcmp(jc, "space-between") && parts.n > 1;
        const int extra = between ? (inner - total - gap * (parts.n - 1)) : 0;
        for (int i = 0; i < parts.n; i++) {
            if (i) {
                int sp = gap + (between && extra > 0 ? extra / (parts.n - 1) + (i <= extra % (parts.n - 1) ? 1 : 0) : 0);
                for (int k = 0; k < sp; k++) b_add(&joined, " ", 1);
                used += sp;
            }
            b_str(&joined, parts.v[i].p);
            used += vcols(parts.v[i].p ? parts.v[i].p : "", parts.v[i].n);
        }
        (void)nkids;
        ls_free(&parts);
        Buf *l = ls_new(&body);
        if (l) { *l = joined; if (used > inner) cut_cols(l, inner); }
        else free(joined.p);
    } else {
        for (int i = 0; ch && ch->t == J_ARR && i < ch->len; i++) {
            jval *k = ch->kids[i];
            for (int m = 0; m < (int)jnum(k, "marginTop", 0); m++) ls_new(&body);
            lay(k, inner, &body, cx);
        }
    }
    /* padding and border around the body */
    Sty bs = { 0 };
    bs.color = jstr(bx, "borderColor");
    if (border) {
        Buf *l = ls_new(out);
        if (l) {
            sty_open(l, &bs);
            b_str(l, "╭");
            for (int i = 0; i < width - 2; i++) b_str(l, "─");
            b_str(l, "╮\033[0m");
        }
    }
    for (int i = 0; i < body.n; i++) {
        Buf *l = ls_new(out);
        if (!l) break;
        if (border) { sty_open(l, &bs); b_str(l, "│\033[0m"); }
        for (int k = 0; k < px; k++) b_add(l, " ", 1);
        b_str(l, body.v[i].p ? body.v[i].p : "");
        if (border) {
            const int vc = vcols(body.v[i].p ? body.v[i].p : "", body.v[i].n);
            for (int k = vc; k < inner; k++) b_add(l, " ", 1);
            for (int k = 0; k < px; k++) b_add(l, " ", 1);
            sty_open(l, &bs);
            b_str(l, "│\033[0m");
        }
    }
    if (border) {
        Buf *l = ls_new(out);
        if (l) {
            sty_open(l, &bs);
            b_str(l, "╰");
            for (int i = 0; i < width - 2; i++) b_str(l, "─");
            b_str(l, "╯\033[0m");
        }
    }
    ls_free(&body);
}

static void lay(jval *node, int width, Ls *out, Ctx *cx)
{
    if (!node || node->t != J_OBJ) {
        if (node && node->t == J_STR) { Buf *l = ls_new(out); if (l) b_str(l, node->str); }
        return;
    }
    const char *type = jstr(node, "type");
    if (!type) return;
    if (!strcmp(type, "Text")) lay_text(node, width, out);
    else if (!strcmp(type, "Button")) lay_button(node, out, cx);
    else lay_box(node, width, out, cx);
}

static int g_focus_keep = -1;                   /* the pane's focused button between renders */

int lc_mods_view(const char *component, int cols, int rows, LcModView *view)
{
    memset(view, 0, sizeof *view);
    view->focus = -1;
    if (!lc_mods_running()) return 0;
    char pid[160];
    LOCK();
    const int pane = g_pane;
    snprintf(pid, sizeof pid, "%s", g_pane_id);
    UNLOCK();
    const int is_pane = !strcmp(component, "Pane");
    if (is_pane && !pane) return 0;
    char *q = quote(pid);
    char data[600];
    snprintf(data, sizeof data, "{\"component\": \"%s\", \"requestId\": %s, \"bodyColumns\": %d, \"maxRows\": %d, \"hasSurvey\": false}",
             component, is_pane ? q : "null", cols, rows);
    free(q);
    char *line = lc_mods_event("ui.render", data, 2000, NULL);
    if (!line) return 0;
    jval *m = json_parse(line, NULL);
    free(line);
    jval *tree = json_get(m, "result");
    if (tree && tree->t == J_OBJ) {
        Ls out = { 0 };
        Ctx cx = { view, is_pane ? g_focus_keep : -1 };
        lay(tree, cols, &out, &cx);
        for (int i = 0; i < out.n && view->n < LC_MOD_LINES && view->n < rows; i++) {
            view->lines[view->n++] = out.v[i].p ? out.v[i].p : xdup("");
            out.v[i].p = NULL;
        }
        ls_free(&out);
        if (is_pane) {
            if (g_focus_keep >= 0 && g_focus_keep < view->nb) view->focus = g_focus_keep;
            else if (view->focus < 0 && view->nb) view->focus = 0;
            g_focus_keep = view->focus;
        } else if (view->focus < 0 && view->nb) {
            view->focus = 0;
        }
    }
    json_free(m);
    return view->n;
}

void lc_mods_view_free(LcModView *v)
{
    for (int i = 0; i < v->n; i++) free(v->lines[i]);
    memset(v, 0, sizeof *v);
}

int lc_mods_pane(char *title, int cap, int *esc_closes)
{
    LOCK();
    const int p = g_pane;
    if (title) snprintf(title, (size_t)cap, "%s", g_pane_title);
    if (esc_closes) *esc_closes = g_pane_esc;
    UNLOCK();
    return p;
}

void lc_mods_abort(void) { g_abort = 1; }

void lc_mods_press(const char *id)
{
    char *q = quote(id);
    char line[200];
    snprintf(line, sizeof line, "{\"press\": %s}", q);
    free(q);
    send_line(line);
}

void lc_mods_focus(int k) { g_focus_keep = k; }

void lc_mods_pane_close(void)
{
    char pid[160];
    LOCK();
    snprintf(pid, sizeof pid, "%s", g_pane_id);
    g_pane = 0;
    UNLOCK();
    g_focus_keep = -1;
    char *q = quote(pid), data[400];
    snprintf(data, sizeof data, "{\"id\": %s, \"requestId\": %s}", q, q);
    free(q);
    char *r = lc_mods_event("ui.close", data, 2000, NULL);
    free(r);
}

char *lc_mods_toast(void)
{
    LOCK();
    char *t = g_toast;
    g_toast = NULL;
    UNLOCK();
    return t;
}

int lc_mods_commands(void) { return g_ncmd; }
const char *lc_mods_command_name(int i) { return i >= 0 && i < g_ncmd ? g_cmd[i].name : ""; }
const char *lc_mods_command_desc(int i) { return i >= 0 && i < g_ncmd ? g_cmd[i].desc : ""; }

char *lc_mods_command_run(const char *name, const char *args)
{
    char *qn = quote(name), *qa = quote(args ? args : "");
    Buf d = { 0 };
    b_fmt(&d, "{\"command\": %s, \"args\": %s}", qn, qa);
    free(qn);
    free(qa);
    char *line = lc_mods_event("command.run", d.p, 30000, NULL);
    free(d.p);
    if (!line) return xdup("the mod did not answer");
    jval *m = json_parse(line, NULL);
    free(line);
    const char *t = jstr(json_get(m, "result"), "text");
    char *r = xdup(t ? t : "");
    json_free(m);
    return r;
}
