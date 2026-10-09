/* lc_hooks.c - see lc_hooks.h. */
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
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "json.h"
#include "lc_hooks.h"
#include "lc_plugins.h"

#ifdef _WIN32
#define SEP "\\"
#else
#define SEP "/"
#endif

typedef struct { char *p; size_t n, cap; } Buf;

static void b_add(Buf *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
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
    char tmp[4096];
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

char *lc_json_quote(const char *s)
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

/* -------------------------------------------------------------- the hooks ---- */

static const char *EV_NAME[LC_HOOK_N] = { "SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop",
                                          "SessionEnd", "PreCompact" };

typedef struct {
    int   ev, timeout;           /* timeout: seconds */
    char *matcher, *command;     /* command: a shell line, or the program when there are args */
    char **args;
    int   nargs;
    char *root;                  /* CLAUDE_PLUGIN_ROOT (forward slashes), "" for the project's */
    char *source;                /* the plugin's name, or the settings file */
} Hook;

static Hook *H;
static int   nH, capH, loaded;
static char  g_project[4096];

/* path with forward slashes (bash and node read them; Windows takes them too) */
static char *slashes(const char *p)
{
    char *d = xdup(p);
    for (char *c = d; c && *c; c++) if (*c == '\\') *c = '/';
    return d;
}

/* ${CLAUDE_PLUGIN_ROOT} and ${CLAUDE_PROJECT_DIR} put in */
static char *expand(const char *s, const char *root)
{
    Buf b = { 0 };
    char *proj = slashes(g_project);
    for (const char *p = s; *p;) {
        if (!strncmp(p, "${CLAUDE_PLUGIN_ROOT}", 21)) { b_str(&b, root); p += 21; }
        else if (!strncmp(p, "${CLAUDE_PROJECT_DIR}", 21)) { b_str(&b, proj); p += 21; }
        else if (!strncmp(p, "$CLAUDE_PROJECT_DIR", 19)) { b_str(&b, proj); p += 19; }
        else { b_add(&b, p, 1); p++; }
    }
    free(proj);
    return b.p ? b.p : xdup("");
}

/* {"Event": [{"matcher", "hooks": [...]}]} (or {"hooks": {...}} around it) */
static void hooks_from(jval *o, const char *root, const char *source)
{
    if (!o || o->t != J_OBJ) return;
    jval *inner = json_get(o, "hooks");
    if (inner && inner->t == J_OBJ) o = inner;
    for (int k = 0; k < o->len; k++) {
        int ev = -1;
        for (int e = 0; e < LC_HOOK_N; e++) if (!strcmp(o->keys[k], EV_NAME[e])) ev = e;
        jval *groups = o->kids[k];
        if (ev < 0 || groups->t != J_ARR) continue;
        for (int g = 0; g < groups->len; g++) {
            jval *grp = groups->kids[g];
            const char *matcher = jstr(grp, "matcher");
            jval *hs = json_get(grp, "hooks");
            for (int h = 0; hs && hs->t == J_ARR && h < hs->len; h++) {
                jval *hk = hs->kids[h];
                const char *type = jstr(hk, "type"), *cmd = jstr(hk, "command");
                if (!cmd || (type && strcmp(type, "command"))) continue;      /* prompt/agent hooks: not here */
                if (nH == capH) {
                    const int nc = capH ? capH * 2 : 32;
                    void *gr = realloc(H, (size_t)nc * sizeof *H);
                    if (!gr) return;
                    H = (Hook *)gr;
                    capH = nc;
                }
                Hook *x = &H[nH++];
                memset(x, 0, sizeof *x);
                x->ev = ev;
                x->matcher = xdup(matcher ? matcher : "");
                x->root = xdup(root);
                x->source = xdup(source);
                x->command = expand(cmd, root);
                jval *to = json_get(hk, "timeout");
                x->timeout = to && to->t == J_NUM && to->num > 0 ? (int)to->num : 60;
                jval *args = json_get(hk, "args");
                if (args && args->t == J_ARR && args->len) {
                    x->args = (char **)calloc((size_t)args->len, sizeof(char *));
                    for (int a = 0; x->args && a < args->len; a++)
                        if (args->kids[a]->t == J_STR) x->args[x->nargs++] = expand(args->kids[a]->str, root);
                }
            }
        }
    }
}

static void hooks_file(const char *path, const char *root, const char *source)
{
    char *t = lc_slurp(path, NULL);
    if (!t) return;
    jval *v = json_parse(t, NULL);
    free(t);
    hooks_from(v, root, source);
    json_free(v);
}

void lc_hooks_load(const char *root)
{
    if (loaded) return;
    loaded = 1;
    snprintf(g_project, sizeof g_project, "%s", root ? root : ".");
    lc_plugins_load(root);
    for (int i = 0; i < lc_plugins_n(); i++) {
        const LcPlugin *p = lc_plugin_at(i);
        if (!p->on || !p->hooks || !p->trusted || p->mod) continue;   /* mods: lc_mods */
        char *rootf = slashes(p->dir), path[4400];
        snprintf(path, sizeof path, "%s" SEP ".claude-plugin" SEP "plugin.json", p->dir);
        char *t = lc_slurp(path, NULL);
        jval *man = t ? json_parse(t, NULL) : NULL;
        free(t);
        jval *mh = man ? json_get(man, "hooks") : NULL;
        if (mh && mh->t == J_OBJ) hooks_from(mh, rootf, p->name);
        else {
            if (mh && mh->t == J_STR) {
                const char *r = mh->str;
                while (r[0] == '.' && (r[1] == '/' || r[1] == '\\')) r += 2;
                snprintf(path, sizeof path, "%s" SEP "%s", p->dir, r);
            } else {
                snprintf(path, sizeof path, "%s" SEP "hooks" SEP "hooks.json", p->dir);
            }
            hooks_file(path, rootf, p->name);
        }
        json_free(man);
        free(rootf);
    }
    static const char *settings[] = { ".claude" SEP "settings.json", ".claude" SEP "settings.local.json",
                                      ".localcode" SEP "settings.json" };
    for (int i = 0; i < 3; i++) {
        char path[4400];
        snprintf(path, sizeof path, "%s" SEP "%s", g_project, settings[i]);
        char *pr = slashes(g_project);
        hooks_file(path, pr, settings[i]);
        free(pr);
    }
}

int lc_hooks_any(int ev)
{
    int n = 0;
    for (int i = 0; i < nH; i++) n += H[i].ev == ev;
    return n;
}

/* "Write|Edit", "*", "" or ".*" against a name */
static int matches(const char *m, const char *name)
{
    if (!m[0] || !strcmp(m, "*") || !strcmp(m, ".*")) return 1;
    if (!name) return 0;
    const size_t nl = strlen(name);
    for (const char *p = m; *p;) {
        const char *bar = strchr(p, '|');
        const size_t l = bar ? (size_t)(bar - p) : strlen(p);
        if (l == nl && !strncmp(p, name, l)) return 1;
        if (!bar) break;
        p = bar + 1;
    }
    return 0;
}

/* ------------------------------------------------------- running a program ---- */

#ifdef _WIN32
static wchar_t *wide(const char *s)
{
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    wchar_t *w = n > 0 ? (wchar_t *)malloc((size_t)n * sizeof(wchar_t)) : NULL;
    if (w) MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    return w;
}

/* one argument quoted as CommandLineToArgvW reads it back */
static void quote_arg(Buf *b, const char *a)
{
    b_add(b, "\"", 1);
    for (const char *p = a;; p++) {
        int bs = 0;
        while (*p == '\\') { bs++; p++; }
        if (!*p) { for (int i = 0; i < bs * 2; i++) b_add(b, "\\", 1); break; }
        if (*p == '"') { for (int i = 0; i < bs * 2 + 1; i++) b_add(b, "\\", 1); b_add(b, "\"", 1); continue; }
        for (int i = 0; i < bs; i++) b_add(b, "\\", 1);
        b_add(b, p, 1);
    }
    b_add(b, "\"", 1);
}

/* Git's bash or sh (not WSL's bash in System32) */
static const char *git_tool(const char *exe)
{
    static char found[2][600];
    char *slot = found[!strcmp(exe, "sh.exe")];
    if (slot[0]) return slot;
    const char *bases[] = { getenv("ProgramW6432"), getenv("ProgramFiles"), getenv("LOCALAPPDATA") };
    const char *subs[] = { "\\Git\\bin\\", "\\Git\\usr\\bin\\", "\\Programs\\Git\\bin\\", "\\Programs\\Git\\usr\\bin\\" };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4 && bases[i]; j++) {
            char p[600];
            snprintf(p, sizeof p, "%s%s%s", bases[i], subs[j], exe);
            FILE *f = fopen(p, "rb");
            if (f) { fclose(f); snprintf(slot, 600, "%s", p); return slot; }
        }
    snprintf(slot, 600, "%s", exe);
    return slot;
}

static int run_proc(const Hook *h, const char *input, char **out, char **err)
{
    Buf cl = { 0 };
    if (h->nargs) {                                   /* the program and its arguments */
        const char *prog = h->command;
        if (!strcmp(prog, "sh")) prog = git_tool("sh.exe");
        else if (!strcmp(prog, "bash")) prog = git_tool("bash.exe");
        quote_arg(&cl, prog);
        for (int i = 0; i < h->nargs; i++) { b_add(&cl, " ", 1); quote_arg(&cl, h->args[i]); }
    } else {                                          /* a line for bash */
        quote_arg(&cl, git_tool("bash.exe"));
        b_str(&cl, " -c ");
        quote_arg(&cl, h->command);
    }
    char *proj = slashes(g_project);
    wchar_t *wr = wide(h->root), *wp = wide(proj);
    if (wr) SetEnvironmentVariableW(L"CLAUDE_PLUGIN_ROOT", wr);
    if (wp) SetEnvironmentVariableW(L"CLAUDE_PROJECT_DIR", wp);
    free(wr); free(wp); free(proj);
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE in_r = NULL, in_w = NULL, out_r = NULL, out_w = NULL, err_r = NULL, err_w = NULL;
    int code = -1;
    if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0) || !CreatePipe(&err_r, &err_w, &sa, 0)) {
        free(cl.p);
        return -1;
    }
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = err_w;
    PROCESS_INFORMATION pi;
    wchar_t *wcl = wide(cl.p ? cl.p : "");
    wchar_t *wcwd = wide(g_project);
    const BOOL okp = wcl && CreateProcessW(NULL, wcl, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, wcwd, &si, &pi);
    free(wcl); free(wcwd); free(cl.p);
    CloseHandle(in_r); CloseHandle(out_w); CloseHandle(err_w);
    Buf o = { 0 }, e = { 0 };
    if (okp) {
        DWORD wrote = 0;
        if (input) WriteFile(in_w, input, (DWORD)strlen(input), &wrote, NULL);
        CloseHandle(in_w);
        in_w = NULL;
        const ULONGLONG t0 = GetTickCount64();
        for (;;) {
            char buf[4096];
            DWORD avail = 0, got = 0;
            int any = 0;
            while (PeekNamedPipe(out_r, NULL, 0, NULL, &avail, NULL) && avail &&
                   ReadFile(out_r, buf, avail < sizeof buf ? avail : sizeof buf, &got, NULL) && got) { b_add(&o, buf, got); any = 1; }
            while (PeekNamedPipe(err_r, NULL, 0, NULL, &avail, NULL) && avail &&
                   ReadFile(err_r, buf, avail < sizeof buf ? avail : sizeof buf, &got, NULL) && got) { b_add(&e, buf, got); any = 1; }
            if (WaitForSingleObject(pi.hProcess, any ? 0 : 20) == WAIT_OBJECT_0) {
                /* what is left, without blocking: a child it left running may hold the pipes open */
                for (int k = 0; k < 10; k++) {
                    int more = 0;
                    while (PeekNamedPipe(out_r, NULL, 0, NULL, &avail, NULL) && avail &&
                           ReadFile(out_r, buf, avail < sizeof buf ? avail : sizeof buf, &got, NULL) && got) { b_add(&o, buf, got); more = 1; }
                    while (PeekNamedPipe(err_r, NULL, 0, NULL, &avail, NULL) && avail &&
                           ReadFile(err_r, buf, avail < sizeof buf ? avail : sizeof buf, &got, NULL) && got) { b_add(&e, buf, got); more = 1; }
                    if (!more) Sleep(10);
                }
                DWORD c = 0;
                GetExitCodeProcess(pi.hProcess, &c);
                code = (int)c;
                break;
            }
            if (GetTickCount64() - t0 > (ULONGLONG)h->timeout * 1000) {
                TerminateProcess(pi.hProcess, 1);
                b_fmt(&e, "\n(stopped after %d s)", h->timeout);
                code = -2;
                break;
            }
        }
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    } else {
        b_fmt(&e, "cannot start it (error %lu)", (unsigned long)GetLastError());
    }
    if (in_w) CloseHandle(in_w);
    CloseHandle(out_r);
    CloseHandle(err_r);
    *out = o.p ? o.p : xdup("");
    *err = e.p ? e.p : xdup("");
    return code;
}
#else
static int run_proc(const Hook *h, const char *input, char **out, char **err)
{
    int pin[2], pout[2], perr[2];
    if (pipe(pin) || pipe(pout) || pipe(perr)) return -1;
    setenv("CLAUDE_PLUGIN_ROOT", h->root, 1);
    setenv("CLAUDE_PROJECT_DIR", g_project, 1);
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(pin[0], 0); dup2(pout[1], 1); dup2(perr[1], 2);
        close(pin[1]); close(pout[0]); close(perr[0]);
        if (chdir(g_project)) _exit(127);
        if (h->nargs) {
            char **av = (char **)calloc((size_t)h->nargs + 2, sizeof(char *));
            av[0] = h->command;
            for (int i = 0; i < h->nargs; i++) av[i + 1] = h->args[i];
            execvp(h->command, av);
        } else {
            execl("/bin/sh", "sh", "-c", h->command, (char *)NULL);
        }
        _exit(127);
    }
    close(pin[0]); close(pout[1]); close(perr[1]);
    if (input) { ssize_t w = write(pin[1], input, strlen(input)); (void)w; }
    close(pin[1]);
    Buf o = { 0 }, e = { 0 };
    const time_t t0 = time(NULL);
    int open_o = 1, open_e = 1, code = -1, st = 0;
    while (open_o || open_e) {
        struct pollfd fds[2] = { { pout[0], POLLIN, 0 }, { perr[0], POLLIN, 0 } };
        poll(fds, 2, 100);
        char buf[4096];
        if (open_o && (fds[0].revents & (POLLIN | POLLHUP))) { ssize_t r = read(pout[0], buf, sizeof buf); if (r > 0) b_add(&o, buf, (size_t)r); else open_o = 0; }
        if (open_e && (fds[1].revents & (POLLIN | POLLHUP))) { ssize_t r = read(perr[0], buf, sizeof buf); if (r > 0) b_add(&e, buf, (size_t)r); else open_e = 0; }
        if (time(NULL) - t0 > h->timeout) { kill(pid, SIGKILL); code = -2; break; }
    }
    close(pout[0]); close(perr[0]);
    if (waitpid(pid, &st, 0) == pid && code != -2 && WIFEXITED(st)) code = WEXITSTATUS(st);
    *out = o.p ? o.p : xdup("");
    *err = e.p ? e.p : xdup("");
    return code;
}
#endif

/* ------------------------------------------------------------ an event ---- */

static void append(char **dst, const char *text)
{
    if (!text || !*text) return;
    Buf b = { 0 };
    if (*dst && **dst) { b_str(&b, *dst); b_str(&b, "\n"); }
    b_str(&b, text);
    free(*dst);
    *dst = b.p;
}

static char *trimmed(const char *s)
{
    while (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t') s++;
    char *d = xdup(s);
    size_t n = d ? strlen(d) : 0;
    while (n && (d[n - 1] == ' ' || d[n - 1] == '\n' || d[n - 1] == '\r' || d[n - 1] == '\t')) d[--n] = 0;
    return d;
}

int lc_hooks_run(int ev, const char *match, const char *fields, LcHookOut *o)
{
    memset(o, 0, sizeof *o);
    static char session[40];
    if (!session[0]) snprintf(session, sizeof session, "localcode-%lld", (long long)time(NULL));
    for (int i = 0; i < nH; i++) {
        const Hook *h = &H[i];
        if (h->ev != ev || !matches(h->matcher, match)) continue;
        char *cwd = lc_json_quote(g_project);
        Buf in = { 0 };
        b_fmt(&in, "{\"session_id\": \"%s\", \"transcript_path\": \"\", \"cwd\": %s, \"hook_event_name\": \"%s\"",
              session, cwd, EV_NAME[ev]);
        free(cwd);
        if (fields && *fields) { b_str(&in, ", "); b_str(&in, fields); }
        b_str(&in, "}");
        char *out = NULL, *err = NULL;
        const int code = run_proc(h, in.p, &out, &err);
        free(in.p);
        o->ran++;
        char *so = trimmed(out ? out : ""), *se = trimmed(err ? err : "");
        if (code == 2) {
            o->block = 1;
            append(&o->reason, *se ? se : so);
        } else if (code == 0 && so && so[0] == '{') {
            jval *j = json_parse(so, NULL);
            jval *cont = json_get(j, "continue");
            const char *dec = jstr(j, "decision"), *why = jstr(j, "reason");
            if (cont && cont->t == J_BOOL && !cont->boolean) { o->block = 1; append(&o->reason, jstr(j, "stopReason")); }
            if (dec && !strcmp(dec, "block")) { o->block = 1; append(&o->reason, why); }
            if (dec && (!strcmp(dec, "approve") || !strcmp(dec, "allow")) && o->decision == 0) o->decision = 1;
            append(&o->message, jstr(j, "systemMessage"));
            jval *hs = json_get(j, "hookSpecificOutput");
            append(&o->context, jstr(hs, "additionalContext"));
            const char *pd = jstr(hs, "permissionDecision");
            if (pd && !strcmp(pd, "deny")) { o->decision = 2; o->block = 1; append(&o->reason, jstr(hs, "permissionDecisionReason")); }
            else if (pd && !strcmp(pd, "ask") && o->decision != 2) o->decision = 3;
            else if (pd && !strcmp(pd, "allow") && o->decision == 0) o->decision = 1;
            json_free(j);
        } else if (code == 0) {
            if (ev == LC_HOOK_SESSION_START || ev == LC_HOOK_USER_PROMPT) append(&o->context, so);
        } else {
            char line[700];
            snprintf(line, sizeof line, "hook of %s failed (exit %d): %.400s", h->source, code, *se ? se : so);
            append(&o->message, line);
        }
        free(so); free(se); free(out); free(err);
    }
    return o->ran;
}

void lc_hook_out_free(LcHookOut *o)
{
    free(o->reason);
    free(o->context);
    free(o->message);
    memset(o, 0, sizeof *o);
}

/* ------------------------------------------------- localcode's tools by name ---- */

const char *lc_hook_tool_name(const char *t)
{
    static const char *map[][2] = { { "run", "Bash" }, { "test", "Bash" }, { "read_file", "Read" }, { "write_file", "Write" },
                                    { "edit_file", "Edit" }, { "apply_patch", "MultiEdit" }, { "glob", "Glob" }, { "grep", "Grep" }, { "list_dir", "LS" },
                                    { "skill", "Skill" }, { "plan", "TodoWrite" } };
    for (size_t i = 0; i < sizeof map / sizeof *map; i++) if (!strcmp(t, map[i][0])) return map[i][1];
    return t;
}

char *lc_hook_tool_input(const LcCall *c)
{
    Buf b = { 0 };
    b_str(&b, "{");
    int first = 1;
    for (int i = 0; i < c->n; i++) {
        const char *k = c->key[i];
        char *v = NULL;
        if (!strcmp(k, "path")) {                     /* Claude Code's tools take absolute paths */
            char abs[4400];
            snprintf(abs, sizeof abs, "%s" SEP "%s", g_project, c->val[i]);
            v = lc_json_quote(abs);
            k = !strcmp(c->name, "list_dir") || !strcmp(c->name, "grep") ? "path" : "file_path";
        } else if (!strcmp(k, "timeout_s")) {
            b_fmt(&b, "%s\"timeout\": %d", first ? "" : ", ", atoi(c->val[i]) * 1000);
            first = 0;
            continue;
        } else if (!strcmp(k, "name") && !strcmp(c->name, "skill")) {
            k = "skill";
            v = lc_json_quote(c->val[i]);
        } else if (!strcmp(k, "ignore_case")) {
            b_fmt(&b, "%s\"-i\": %s", first ? "" : ", ", !strcmp(c->val[i], "true") ? "true" : "false");
            first = 0;
            continue;
        } else {
            v = lc_json_quote(c->val[i]);
        }
        b_fmt(&b, "%s\"%s\": ", first ? "" : ", ", k);
        b_str(&b, v);
        free(v);
        first = 0;
    }
    b_str(&b, "}");
    return b.p;
}

char *lc_hooks_report(void)
{
    Buf b = { 0 };
    int per[LC_HOOK_N] = { 0 };
    for (int i = 0; i < nH; i++) per[H[i].ev]++;
    b_fmt(&b, "%d hooks ready:", nH);
    for (int e = 0; e < LC_HOOK_N; e++) if (per[e]) b_fmt(&b, " %s %d", EV_NAME[e], per[e]);
    if (!nH) b_str(&b, " none (a plugin's hooks run once allowed: /plugin hooks on NAME)");
    return b.p;
}
