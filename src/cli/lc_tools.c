/* lc_tools.c - see lc_tools.h. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE           /* realpath: XSI, not declared under POSIX alone by older glibc */

#include "lc_tools.h"
#include "lc_plugins.h"
#include "lc_regex.h"
#include "lc_web.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

/* caps on what one call hands back, so a call cannot fill the context */
#define OUT_CAP        24000     /* characters of any tool result                 */
#define READ_LINES     400       /* read_file lines when no limit is given        */
#define LINE_CAP       400       /* characters kept of one long line              */
#define LIST_CAP       300       /* list_dir / glob entries                       */
#define GREP_CAP       120       /* grep matching lines                           */
#define GLOB_SCAN      20000     /* glob matches gathered to order by date        */
#define FILE_CAP       (8 << 20) /* bytes a tool reads from one file              */
#define RUN_OUT_CAP    16000     /* characters of a command's output (head + tail) */

/* ------------------------------------------------------------------- strings ---- */

typedef struct { char *p; size_t n, cap; } Str;

static void s_add(Str *s, const char *p, size_t n)
{
    if (s->n + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap : 1024;
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
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) s_add(s, buf, (size_t)(n < (int)sizeof buf ? n : (int)sizeof buf - 1));
}

static char *s_take(Str *s)
{
    if (!s->p) { char *e = (char *)malloc(1); if (e) e[0] = 0; return e; }
    char *p = s->p;
    s->p = NULL;
    s->n = s->cap = 0;
    return p;
}

static char *dup_str(const char *p)
{
    const size_t n = strlen(p) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, p, n);
    return d;
}

static char *dup_fmt(const char *fmt, ...)
{
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return dup_str(buf);
}

/* A result past OUT_CAP keeps its start and says how much was left out. */
static char *capped(Str *s)
{
    if (s->n > OUT_CAP) {
        size_t cut = OUT_CAP;
        while (cut > 0 && ((unsigned char)s->p[cut] & 0xC0) == 0x80) cut--;
        char tail[160];
        snprintf(tail, sizeof tail, "\n[... %zu more characters not shown; narrow the request]", s->n - cut);
        s->n = cut;
        s->p[cut] = 0;
        s_str(s, tail);
    }
    return s_take(s);
}

void lc_call_free(LcCall *c)
{
    for (int i = 0; i < c->n; i++) { free(c->key[i]); free(c->val[i]); }
    memset(c, 0, sizeof *c);
}

const char *lc_call_arg(const LcCall *c, const char *key)
{
    for (int i = 0; i < c->n; i++) if (!strcmp(c->key[i], key)) return c->val[i];
    return NULL;
}

static int arg_int(const LcCall *c, const char *key, int def)
{
    const char *v = lc_call_arg(c, key);
    return v && *v ? atoi(v) : def;
}

static int arg_bool(const LcCall *c, const char *key)
{
    const char *v = lc_call_arg(c, key);
    return v && (!strcmp(v, "true") || !strcmp(v, "1") || !strcmp(v, "True"));
}

/* --------------------------------------------------------------------- paths ---- */

static char g_root[4096];

#ifdef _WIN32
static wchar_t *wide(const char *u)
{
    const int n = MultiByteToWideChar(CP_UTF8, 0, u, -1, NULL, 0);
    wchar_t *w = (wchar_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(wchar_t));
    if (!w) return NULL;
    if (n <= 0) { w[0] = 0; return w; }
    MultiByteToWideChar(CP_UTF8, 0, u, -1, w, n);
    return w;
}

static char *narrow(const wchar_t *w)
{
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    char *u = (char *)malloc((size_t)(n > 0 ? n : 1));
    if (!u) return NULL;
    if (n <= 0) { u[0] = 0; return u; }
    WideCharToMultiByte(CP_UTF8, 0, w, -1, u, n, NULL, NULL);
    return u;
}
#endif

int lc_tools_init(const char *root)
{
#ifdef _WIN32
    wchar_t buf[4096];
    if (root) {
        wchar_t *w = wide(root);
        if (!w) return -1;
        const DWORD n = GetFullPathNameW(w, 4096, buf, NULL);
        free(w);
        if (!n || n >= 4096) return -1;
    } else if (!GetCurrentDirectoryW(4096, buf)) {
        return -1;
    }
    size_t L = wcslen(buf);
    while (L > 3 && (buf[L - 1] == L'\\' || buf[L - 1] == L'/')) buf[--L] = 0;
    char *u = narrow(buf);
    if (!u) return -1;
    snprintf(g_root, sizeof g_root, "%s", u);
    free(u);
#else
    char buf[4096];
    if (!(root ? realpath(root, buf) : getcwd(buf, sizeof buf))) return -1;
    snprintf(g_root, sizeof g_root, "%s", buf);
#endif
    return 0;
}

const char *lc_tools_root(void) { return g_root; }

static int g_outside;        /* this call may reach outside the working directory (the user allowed it) */

void lc_tools_allow_outside(int on) { g_outside = on; }

/* The absolute path of a path the model gave, or NULL (with why) when it leaves the
 * working directory. out keeps '\\' on Windows. */
static int resolve(const char *rel, char *out, size_t cap, char *why, size_t wcap)
{
    if (!g_root[0]) lc_tools_init(NULL);
    if (!rel || !*rel) rel = ".";
#ifdef _WIN32
    char joined[8192];
    const int abs = (rel[0] && rel[1] == ':') || rel[0] == '\\' || rel[0] == '/';
    snprintf(joined, sizeof joined, abs ? "%s" : "%s\\%s", abs ? rel : g_root, rel);
    for (char *p = joined; *p; p++) if (*p == '/') *p = '\\';
    wchar_t *w = wide(joined);
    wchar_t full[4096];
    const DWORD n = w ? GetFullPathNameW(w, 4096, full, NULL) : 0;
    free(w);
    if (!n || n >= 4096) { snprintf(why, wcap, "invalid path: %s", rel); return -1; }
    char *u = narrow(full);
    if (!u) { snprintf(why, wcap, "out of memory"); return -1; }
    const size_t rl = strlen(g_root);
    const int inside = !_strnicmp(u, g_root, rl) && (u[rl] == 0 || u[rl] == '\\' || g_root[rl - 1] == '\\');
    if (!inside && !g_outside) {
        snprintf(why, wcap, "%s is outside the working directory %s", rel, g_root);
        free(u);
        return -1;
    }
    snprintf(out, cap, "%s", u);
    free(u);
#else
    char joined[8192], full[4096];
    snprintf(joined, sizeof joined, rel[0] == '/' ? "%s" : "%s/%s", rel[0] == '/' ? rel : g_root, rel);
    /* realpath needs an existing path: resolve the deepest existing parent */
    char probe[8192];
    snprintf(probe, sizeof probe, "%s", joined);
    char rest[8192] = "";
    while (!realpath(probe, full)) {
        char *sl = strrchr(probe, '/');
        if (!sl || sl == probe) { snprintf(why, wcap, "invalid path: %s", rel); return -1; }
        char tmp[8192];
        snprintf(tmp, sizeof tmp, "%s%s", sl, rest);
        snprintf(rest, sizeof rest, "%s", tmp);
        *sl = 0;
    }
    char whole[8192];
    snprintf(whole, sizeof whole, "%s%s", full, rest);
    const size_t rl = strlen(g_root);
    if ((strncmp(whole, g_root, rl) || (whole[rl] && whole[rl] != '/')) && !g_outside) {
        snprintf(why, wcap, "%.300s is outside the working directory %.250s", rel, g_root);
        return -1;
    }
    if (snprintf(out, cap, "%s", whole) >= (int)cap) { snprintf(why, wcap, "path too long: %.300s", rel); return -1; }
#endif
    return 0;
}

/* The path relative to the working directory, with '/' separators, for results. */
static const char *rel_of(const char *abs, char *buf, size_t cap)
{
    const size_t rl = strlen(g_root);
#ifdef _WIN32
    const int under = !_strnicmp(abs, g_root, rl);
#else
    const int under = !strncmp(abs, g_root, rl);
#endif
    const char *p = under ? abs + rl : abs;        /* outside (allowed by the user): the whole path */
    if (under) while (*p == '\\' || *p == '/') p++;
    snprintf(buf, cap, "%s", *p ? p : ".");
    for (char *q = buf; *q; q++) if (*q == '\\') *q = '/';
    return buf;
}

int lc_tools_where(const char *p, char *out, size_t cap)
{
    const int keep = g_outside;
    char abs[4096], why[600];
    g_outside = 1;
    const int r = resolve(p, abs, sizeof abs, why, sizeof why);
    g_outside = keep;
    if (r) { snprintf(out, cap, "%s", p ? p : ""); return -1; }
    const size_t rl = strlen(g_root);
#ifdef _WIN32
    const int inside = !_strnicmp(abs, g_root, rl) && (abs[rl] == 0 || abs[rl] == '\\' || g_root[rl - 1] == '\\');
#else
    const int inside = !strncmp(abs, g_root, rl) && (abs[rl] == 0 || abs[rl] == '/');
#endif
    if (inside) { rel_of(abs, out, cap); return 0; }
    snprintf(out, cap, "%s", abs);
    for (char *q = out; *q; q++) if (*q == '\\') *q = '/';
    return 1;
}

static FILE *open_file(const char *path, const char *mode)
{
#ifdef _WIN32
    wchar_t *w = wide(path), *m = wide(mode);
    FILE *f = (w && m) ? _wfopen(w, m) : NULL;
    free(w);
    free(m);
    return f;
#else
    return fopen(path, mode);
#endif
}

/* The whole file (NUL-terminated) or NULL; *n its length; *err why not. */
static char *slurp(const char *path, long *n, const char **err)
{
    FILE *f = open_file(path, "rb");
    if (!f) { *err = "cannot open it (missing, or a folder)"; return NULL; }
    char *b = NULL;
    if (fseek(f, 0, SEEK_END) == 0) {
        const long len = ftell(f);
        if (len > FILE_CAP) { *err = "it is larger than 8 MB"; fclose(f); return NULL; }
        if (len >= 0 && fseek(f, 0, SEEK_SET) == 0 && (b = (char *)malloc((size_t)len + 1))) {
            if (fread(b, 1, (size_t)len, f) != (size_t)len) { free(b); b = NULL; *err = "reading failed"; }
            else { b[len] = 0; *n = len; }
        }
    }
    fclose(f);
    if (!b && !*err) *err = "reading failed";
    return b;
}

char *lc_slurp(const char *path, long *n)
{
    const char *err = NULL;
    long len = 0;
    char *b = slurp(path, &len, &err);
    if (n) *n = len;
    return b;
}

static int write_all(const char *path, const char *data, size_t n)
{
#ifdef _WIN32
    /* the parent folders first */
    char dir[4096];
    snprintf(dir, sizeof dir, "%s", path);
    for (char *p = dir + 3; *p; p++) {
        if (*p != '\\') continue;
        *p = 0;
        wchar_t *w = wide(dir);
        if (w) { CreateDirectoryW(w, NULL); free(w); }
        *p = '\\';
    }
#else
    char dir[4096];
    snprintf(dir, sizeof dir, "%s", path);
    for (char *p = dir + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        mkdir(dir, 0755);
        *p = '/';
    }
#endif
    FILE *f = open_file(path, "wb");
    if (!f) return -1;
    const int ok = fwrite(data, 1, n, f) == n;
    return fclose(f) == 0 && ok ? 0 : -1;
}

static int is_binary(const char *p, long n)
{
    const long k = n < 8000 ? n : 8000;
    for (long i = 0; i < k; i++) if (p[i] == 0) return 1;
    return 0;
}

/* ------------------------------------------------------------------ the list ---- */

#define T_READ  "{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a text file of the working directory, with line numbers. Long files come in pieces: give offset (first line, 1-based) and limit (number of lines, default 400).\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"file path relative to the working directory\"}, \"offset\": {\"type\": \"integer\"}, \"limit\": {\"type\": \"integer\"}}, \"required\": [\"path\"]}}}"
#define T_LIST  "{\"type\": \"function\", \"function\": {\"name\": \"list_dir\", \"description\": \"List the files and folders of a folder of the working directory (folders end with /).\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"folder relative to the working directory, default .\"}}, \"required\": []}}}"
#define T_GLOB  "{\"type\": \"function\", \"function\": {\"name\": \"glob\", \"description\": \"Find files by name pattern, for example src/**/*.c or *.md. * matches within one folder name, ** any number of folders. The most recently changed files come first.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"pattern\": {\"type\": \"string\"}}, \"required\": [\"pattern\"]}}}"
#define T_GREP  "{\"type\": \"function\", \"function\": {\"name\": \"grep\", \"description\": \"Search the contents of text files with a regular expression (for example def \\\\w+\\\\( or TODO|FIXME; . [] * + ? {m,n} | () \\\\b \\\\d \\\\w \\\\s ^ $, no lookaround); lists path:line: text of each match. literal true searches the exact text instead.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"pattern\": {\"type\": \"string\", \"description\": \"the regular expression (or the text, with literal)\"}, \"literal\": {\"type\": \"boolean\"}, \"path\": {\"type\": \"string\", \"description\": \"folder or file to search, default .\"}, \"glob\": {\"type\": \"string\", \"description\": \"only file names matching this, for example *.c\"}, \"ignore_case\": {\"type\": \"boolean\"}}, \"required\": [\"pattern\"]}}}"
#define T_WRITE "{\"type\": \"function\", \"function\": {\"name\": \"write_file\", \"description\": \"Create a file or replace all of its content. Prefer edit_file to change part of an existing file.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"content\": {\"type\": \"string\", \"description\": \"the whole new content\"}}, \"required\": [\"path\", \"content\"]}}}"
#define T_EDIT  "{\"type\": \"function\", \"function\": {\"name\": \"edit_file\", \"description\": \"Replace an exact piece of text in a file. old_string must appear exactly once (copy it from read_file, without the line numbers, with enough surrounding lines to be unique), unless replace_all is true.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"old_string\": {\"type\": \"string\"}, \"new_string\": {\"type\": \"string\"}, \"replace_all\": {\"type\": \"boolean\"}}, \"required\": [\"path\", \"old_string\", \"new_string\"]}}}"
#define T_RUN   "{\"type\": \"function\", \"function\": {\"name\": \"run\", \"description\": \"Run a PowerShell command in the working directory (builds, tests, git, scripts) and get its output and exit code. Not for interactive programs.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}, \"timeout_s\": {\"type\": \"integer\", \"description\": \"seconds before it is stopped, default 120, at most 1800\"}}, \"required\": [\"command\"]}}}"
#define T_RUN_SH "{\"type\": \"function\", \"function\": {\"name\": \"run\", \"description\": \"Run a bash command (Git Bash) in the working directory (builds, tests, git, scripts) and get its output and exit code. Not for interactive programs.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}, \"timeout_s\": {\"type\": \"integer\", \"description\": \"seconds before it is stopped, default 120, at most 1800\"}}, \"required\": [\"command\"]}}}"

#define T_PLAN  "{\"type\": \"function\", \"function\": {\"name\": \"plan\", \"description\": \"Record the plan of a task that changes code, before changing files: the steps, and the command that proves the task works (the project's tests, a build, or running the program with a check; exit code 0 = works). Call it again to revise the plan or to show progress: begin a step with [x] when done, [>] while doing it.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"steps\": {\"type\": \"string\", \"description\": \"one step per line\"}, \"test_command\": {\"type\": \"string\", \"description\": \"PowerShell command run by test\"}}, \"required\": [\"steps\", \"test_command\"]}}}"
#define T_PLAN_SH "{\"type\": \"function\", \"function\": {\"name\": \"plan\", \"description\": \"Record the plan of a task that changes code, before changing files: the steps, and the command that proves the task works (the project's tests, a build, or running the program with a check; exit code 0 = works). Call it again to revise the plan or to show progress: begin a step with [x] when done, [>] while doing it.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"steps\": {\"type\": \"string\", \"description\": \"one step per line\"}, \"test_command\": {\"type\": \"string\", \"description\": \"bash command run by test\"}}, \"required\": [\"steps\", \"test_command\"]}}}"
#define T_TEST  "{\"type\": \"function\", \"function\": {\"name\": \"test\", \"description\": \"Run the plan's test command (or command, when given) and tell whether it passes (exit code 0), with its output. Call it after every change.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\", \"description\": \"another test command; default the plan's\"}}, \"required\": []}}}"
#define T_QUESTION "{\"type\": \"function\", \"function\": {\"name\": \"question\", \"description\": \"Ask the user a question and wait for the answer: only when a decision is theirs and the request, the code and sensible defaults do not settle it (which of two behaviours they want, a name, a missing value). Give the likely answers as options, the recommended one first.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"question\": {\"type\": \"string\"}, \"options\": {\"type\": \"string\", \"description\": \"the answers to choose from, one per line (at most 9); the user may also write another\"}}, \"required\": [\"question\"]}}}"
#define T_PATCH "{\"type\": \"function\", \"function\": {\"name\": \"apply_patch\", \"description\": \"Change several places or files in one call, all or nothing. The patch: *** Begin Patch, then per file *** Update File: path (hunks: a line @@ or @@ followed by a line just before the change, then lines starting with a space for unchanged context, - for removed, + for added; 3 context lines around each change), *** Add File: path (every line starting with +), or *** Delete File: path; *** Move to: new path after an Update renames; *** End Patch.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"patch\": {\"type\": \"string\"}}, \"required\": [\"patch\"]}}}"
#define T_WEB   "{\"type\": \"function\", \"function\": {\"name\": \"webfetch\", \"description\": \"Fetch a web page (http or https URL) and read it as text: documentation, an API reference, an issue, a changelog. The user approves each fetch unless approval is automatic.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"url\": {\"type\": \"string\"}}, \"required\": [\"url\"]}}}"
#define T_SKILL "{\"type\": \"function\", \"function\": {\"name\": \"skill\", \"description\": \"Load a skill listed in the instructions (or found by skill_search): its steps for a kind of task, to follow. name as listed (plugin:name when given so). file reads another file of the skill's folder.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"}, \"file\": {\"type\": \"string\"}}, \"required\": [\"name\"]}}}"
#define T_SEARCH "{\"type\": \"function\", \"function\": {\"name\": \"skill_search\", \"description\": \"Find skills that are not in the list of the instructions: give a few English words of the task (for example: security audit, slides, react performance), even when the user writes in another language; it returns matching skills to load with skill.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}}, \"required\": [\"query\"]}}}"

/* ------------------------------------------------------------------- the shell ---- */
/* run's shell: bash (Git Bash on Windows, as Claude Code and opencode use it, so the plugins' skills,
 * hooks and mods read their commands as written) by default where it is found, else PowerShell.
 * LOCALCODE_SHELL=bash|powershell at start, /shell during a session. */
static int g_shell = -1;                         /* 0 bash, 1 PowerShell */

#ifdef _WIN32
static const char *git_bash(void)
{
    static char found[600];
    static int looked = 0;
    if (looked) return found[0] ? found : NULL;
    looked = 1;
    const char *bases[] = { getenv("ProgramW6432"), getenv("ProgramFiles"), getenv("LOCALAPPDATA") };
    const char *subs[] = { "\\Git\\bin\\bash.exe", "\\Programs\\Git\\bin\\bash.exe" };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2 && bases[i]; j++) {
            char p[600];
            snprintf(p, sizeof p, "%s%s", bases[i], subs[j]);
            FILE *f = fopen(p, "rb");
            if (f) { fclose(f); snprintf(found, sizeof found, "%s", p); return found; }
        }
    return NULL;
}
#endif

int lc_tools_shell(void)
{
    if (g_shell < 0) {
        const char *e = getenv("LOCALCODE_SHELL");
#ifdef _WIN32
        g_shell = e && !_stricmp(e, "powershell") ? 1 : git_bash() ? 0 : 1;
#else
        g_shell = 0;
        (void)e;
#endif
    }
    return g_shell;
}

int lc_tools_set_shell(int ps)
{
#ifdef _WIN32
    if (!ps && !git_bash()) return -1;
#endif
    g_shell = ps ? 1 : 0;
    return 0;
}

const char *lc_tools_json(int level)
{
    static Str l1[2] = { { 0 } }, l2[2] = { { 0 } };
    const int sh = lc_tools_shell();
    Str *s = level <= 1 ? &l1[sh] : &l2[sh];
    if (s->p) return s->p;
    s_str(s, T_READ "\n" T_LIST "\n" T_GLOB "\n" T_GREP "\n" T_WEB "\n" T_QUESTION);
    if (level >= 2) s_str(s, sh ? "\n" T_WRITE "\n" T_EDIT "\n" T_PATCH "\n" T_RUN "\n" T_PLAN "\n" T_TEST
                                : "\n" T_WRITE "\n" T_EDIT "\n" T_PATCH "\n" T_RUN_SH "\n" T_PLAN_SH "\n" T_TEST);
    if (*lc_skills_block()) s_str(s, "\n" T_SKILL "\n" T_SEARCH);   /* only where there are skills */
    return s->p;
}

int lc_tool_allowed(const char *name, int level)
{
    static const char *reading[] = { "read_file", "list_dir", "glob", "grep", "skill", "skill_search", "webfetch", "question" };
    static const char *all[] = { "write_file", "edit_file", "apply_patch", "run", "plan", "test" };
    for (int i = 0; i < 8; i++) if (!strcmp(name, reading[i])) return level >= 1;
    for (int i = 0; i < 6; i++) if (!strcmp(name, all[i])) return level >= 2;
    return 0;
}

int lc_tool_needs_approval(const char *name)
{
    return !strcmp(name, "write_file") || !strcmp(name, "edit_file") || !strcmp(name, "apply_patch") || !strcmp(name, "run") ||
           !strcmp(name, "test") ||
           !strcmp(name, "webfetch");                /* it goes out to the network */
}

void lc_tool_title(const LcCall *c, char *buf, size_t cap)
{
    const char *a = lc_call_arg(c, "path");
    if (!strcmp(c->name, "run") || !strcmp(c->name, "test")) a = lc_call_arg(c, "command");
    else if (!strcmp(c->name, "glob") || !strcmp(c->name, "grep")) a = lc_call_arg(c, "pattern");
    else if (!strcmp(c->name, "skill")) a = lc_call_arg(c, "name");
    else if (!strcmp(c->name, "skill_search")) a = lc_call_arg(c, "query");
    else if (!strcmp(c->name, "webfetch")) a = lc_call_arg(c, "url");
    else if (!strcmp(c->name, "plan")) a = "";
    else if (!strcmp(c->name, "question")) a = lc_call_arg(c, "question");
    char files[300] = "";
    if (!strcmp(c->name, "apply_patch")) {        /* the files it names */
        const char *p = lc_call_arg(c, "patch");
        for (; p && (p = strstr(p, "*** ")); p += 4) {
            const char *q = !strncmp(p, "*** Add File:", 13) ? p + 13 : !strncmp(p, "*** Update File:", 16) ? p + 16 :
                            !strncmp(p, "*** Delete File:", 16) ? p + 16 : NULL;
            if (!q) continue;
            while (*q == ' ') q++;
            const size_t k = strcspn(q, "\r\n"), fl = strlen(files);
            if (fl < sizeof files - 8) snprintf(files + fl, sizeof files - fl, "%s%.*s", fl ? ", " : "", (int)k, q);
        }
        a = files;
    }
    char one[300];
    snprintf(one, sizeof one, "%s", a ? a : "");
    for (char *p = one; *p; p++) if (*p == '\n' || *p == '\r') *p = ' ';
    if (!strcmp(c->name, "grep") && lc_call_arg(c, "path"))
        snprintf(buf, cap, "grep \"%s\" in %s", one, lc_call_arg(c, "path"));
    else if (!strcmp(c->name, "grep"))
        snprintf(buf, cap, "grep \"%s\"", one);
    else
        snprintf(buf, cap, "%s %s", c->name, one);
}

/* ----------------------------------------------------------------- read_file ---- */

static char *t_read(const LcCall *c, char *sum, size_t sc, int *ok)
{
    char path[4096], why[600], rel[1024];
    if (resolve(lc_call_arg(c, "path"), path, sizeof path, why, sizeof why)) { snprintf(sum, sc, "%s", why); return dup_str(why); }
    long n = 0;
    const char *err = NULL;
    char *b = slurp(path, &n, &err);
    rel_of(path, rel, sizeof rel);
    if (!b) { snprintf(sum, sc, "%s: %s", rel, err); return dup_fmt("cannot read %s: %s", rel, err); }
    if (is_binary(b, n)) { free(b); snprintf(sum, sc, "%s is binary", rel); return dup_fmt("%s is a binary file", rel); }
    int total = n ? 1 : 0;
    for (long i = 0; i < n; i++) if (b[i] == '\n' && i + 1 < n) total++;
    int first = arg_int(c, "offset", 1), lim = arg_int(c, "limit", READ_LINES);
    if (first < 1) first = 1;
    if (lim < 1) lim = READ_LINES;
    Str s = { 0 };
    int line = 1, shown = 0;
    const char *p = b, *end = b + n;
    while (p < end && line < first) { const char *q = memchr(p, '\n', (size_t)(end - p)); p = q ? q + 1 : end; line++; }
    while (p < end && shown < lim && s.n < OUT_CAP) {
        const char *q = memchr(p, '\n', (size_t)(end - p));
        size_t len = (size_t)((q ? q : end) - p);
        if (len && p[len - 1] == '\r') len--;
        s_fmt(&s, "%6d\t", line);
        if (len > LINE_CAP) { s_add(&s, p, LINE_CAP); s_str(&s, " [line cut]"); }
        else s_add(&s, p, len);
        s_str(&s, "\n");
        p = q ? q + 1 : end;
        line++;
        shown++;
    }
    free(b);
    const int last = first + shown - 1;
    if (!shown) s_fmt(&s, "(%s has %d lines; nothing at line %d)\n", rel, total, first);
    else if (first > 1 || last < total) s_fmt(&s, "(lines %d-%d of %d; read more with offset %d)\n", first, last, total, last + 1);
    snprintf(sum, sc, "%s · %d line%s%s", rel, shown, shown == 1 ? "" : "s", last < total ? " (more)" : "");
    *ok = 1;
    return capped(&s);
}

/* ------------------------------------------------------------ directory walks ---- */

typedef LcEnt Ent;

static int ent_cmp(const void *a, const void *b)
{
    const Ent *x = (const Ent *)a, *y = (const Ent *)b;
    if (x->dir != y->dir) return y->dir - x->dir;
    return strcmp(x->name, y->name);
}

/* The entries of one folder (absolute path), without . and ..; *n their count. */
static Ent *read_dir(const char *dir, int *n)
{
    int cap = 64, k = 0;
    Ent *v = (Ent *)malloc((size_t)cap * sizeof *v);
    if (!v) { *n = 0; return NULL; }
#ifdef _WIN32
    char pat[4200];
    snprintf(pat, sizeof pat, "%s\\*", dir);
    wchar_t *w = wide(pat);
    WIN32_FIND_DATAW fd;
    HANDLE h = w ? FindFirstFileW(w, &fd) : INVALID_HANDLE_VALUE;
    free(w);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
            if (k == cap) { Ent *g = (Ent *)realloc(v, (size_t)(cap *= 2) * sizeof *v); if (!g) break; v = g; }
            v[k].name = narrow(fd.cFileName);
            v[k].dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
            v[k].size = ((long long)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
            v[k].mtime = (long long)((((unsigned long long)fd.ftLastWriteTime.dwHighDateTime << 32) |
                                      fd.ftLastWriteTime.dwLowDateTime) / 10000000ULL);
            if (v[k].name) k++;
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
#else
    DIR *d = opendir(dir);
    struct dirent *de;
    while (d && (de = readdir(d))) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (k == cap) { Ent *g = (Ent *)realloc(v, (size_t)(cap *= 2) * sizeof *v); if (!g) break; v = g; }
        char full[8192];
        struct stat st;
        snprintf(full, sizeof full, "%s/%s", dir, de->d_name);
        v[k].name = dup_str(de->d_name);
        v[k].dir = !stat(full, &st) && S_ISDIR(st.st_mode);
        v[k].size = v[k].dir ? 0 : (long long)st.st_size;
        v[k].mtime = (long long)st.st_mtime;
        if (v[k].name) k++;
    }
    if (d) closedir(d);
#endif
    qsort(v, (size_t)k, sizeof *v, ent_cmp);
    *n = k;
    return v;
}

static void ents_free(Ent *v, int n)
{
    for (int i = 0; i < n; i++) free(v[i].name);
    free(v);
}

LcEnt *lc_read_dir(const char *dir, int *n) { return read_dir(dir, n); }
void   lc_ents_free(LcEnt *v, int n) { ents_free(v, n); }

#ifdef _WIN32
#define SEP "\\"
#else
#define SEP "/"
#endif

/* folders a walk does not enter unless asked for by name */
static int noisy(const char *name)
{
    return !strcmp(name, ".git") || !strcmp(name, "node_modules") || !strcmp(name, ".venv") || !strcmp(name, "__pycache__");
}

static char *t_list(const LcCall *c, char *sum, size_t sc, int *ok)
{
    char path[4096], why[600], rel[1024];
    if (resolve(lc_call_arg(c, "path"), path, sizeof path, why, sizeof why)) { snprintf(sum, sc, "%s", why); return dup_str(why); }
    int n = 0;
    Ent *v = read_dir(path, &n);
    rel_of(path, rel, sizeof rel);
    if (!v || !n) {
        if (v) ents_free(v, n);
        snprintf(sum, sc, "%s: empty or not a folder", rel);
        return dup_fmt("%s is empty or is not a folder", rel);
    }
    Str s = { 0 };
    for (int i = 0; i < n && i < LIST_CAP; i++) {
        if (v[i].dir) s_fmt(&s, "%s/\n", v[i].name);
        else          s_fmt(&s, "%s  (%lld bytes)\n", v[i].name, v[i].size);
    }
    if (n > LIST_CAP) s_fmt(&s, "[... %d more entries]\n", n - LIST_CAP);
    snprintf(sum, sc, "%s · %d entr%s", rel, n, n == 1 ? "y" : "ies");
    ents_free(v, n);
    *ok = 1;
    return capped(&s);
}

/* Shell-style match of one name: * any run, ? one character (case-insensitive on Windows). */
static int match_name(const char *pat, const char *s)
{
    for (; *pat; pat++, s++) {
        if (*pat == '*') {
            while (pat[1] == '*') pat++;
            if (!pat[1]) return 1;
            for (; *s; s++) if (match_name(pat + 1, s)) return 1;
            return match_name(pat + 1, s);
        }
        if (!*s) return 0;
#ifdef _WIN32
        const int a = (*pat >= 'A' && *pat <= 'Z') ? *pat + 32 : *pat, b = (*s >= 'A' && *s <= 'Z') ? *s + 32 : *s;
#else
        const int a = *pat, b = *s;
#endif
        if (*pat != '?' && a != b) return 0;
    }
    return !*s;
}

/* Match a '/'-separated relative path against a pattern with ** for any folders. */
static int match_path(const char *pat, const char *path)
{
    if (!*pat) return !*path;
    const char *ps = strchr(pat, '/');
    const size_t pl = ps ? (size_t)(ps - pat) : strlen(pat);
    if (pl == 2 && !strncmp(pat, "**", 2)) {
        const char *rest = ps ? ps + 1 : "";
        if (!*rest) return 1;
        for (const char *p = path;;) {
            if (match_path(rest, p)) return 1;
            const char *sl = strchr(p, '/');
            if (!sl) return 0;
            p = sl + 1;
        }
    }
    const char *ss = strchr(path, '/');
    const size_t sl = ss ? (size_t)(ss - path) : strlen(path);
    char a[512], b[512];
    if (pl >= sizeof a || sl >= sizeof b) return 0;
    memcpy(a, pat, pl); a[pl] = 0;
    memcpy(b, path, sl); b[sl] = 0;
    if (!match_name(a, b)) return 0;
    if (!ps) return !ss;
    if (!ss) return 0;
    return match_path(ps + 1, ss + 1);
}

typedef struct { char *rel; long long mtime; } Hit;

typedef struct {
    const char *pattern;      /* glob: the path pattern; grep: the name filter or NULL */
    const char *needle;       /* grep: the text */
    int         icase;
    Str        *out;
    int         found, files, cap_hit;
    LcRegex    *re;           /* grep: the regular expression, or NULL for the literal text */
    Hit        *hits;         /* glob: the files found, ordered once the walk is done */
    int         nhits, chits;
} Walk;

static void grep_file(Walk *w, const char *abs, const char *rel);

/* Walk a folder: dir is absolute, rel relative to the root ('/'). */
static void walk(Walk *w, const char *dir, const char *rel, int depth)
{
    if (depth > 24 || w->cap_hit) return;
    int n = 0;
    Ent *v = read_dir(dir, &n);
    if (!v) return;
    for (int i = 0; i < n && !w->cap_hit; i++) {
        char abs[8192], r[8192];
        snprintf(abs, sizeof abs, "%s" SEP "%s", dir, v[i].name);
        snprintf(r, sizeof r, "%s%s%s", rel, *rel ? "/" : "", v[i].name);
        if (v[i].dir) {
            if (noisy(v[i].name) && !(w->pattern && strstr(w->pattern, v[i].name))) continue;
            walk(w, abs, r, depth + 1);
        } else if (w->needle) {
            if (!w->pattern || match_name(w->pattern, v[i].name)) grep_file(w, abs, r);
        } else if (match_path(w->pattern, r)) {
            if (w->nhits == w->chits) {
                const int c = w->chits ? w->chits * 2 : 64;
                Hit *g = (Hit *)realloc(w->hits, (size_t)c * sizeof *g);
                if (!g) { w->cap_hit = 1; break; }
                w->hits = g;
                w->chits = c;
            }
            w->hits[w->nhits].rel = dup_str(r);
            w->hits[w->nhits].mtime = v[i].mtime;
            if (w->hits[w->nhits].rel) w->nhits++;
            if (++w->found >= GLOB_SCAN) w->cap_hit = 1;
        }
    }
    ents_free(v, n);
}

static int hit_cmp(const void *a, const void *b)
{
    const Hit *x = (const Hit *)a, *y = (const Hit *)b;
    if (x->mtime != y->mtime) return x->mtime > y->mtime ? -1 : 1;
    return strcmp(x->rel, y->rel);
}

static char *t_glob(const LcCall *c, char *sum, size_t sc, int *ok)
{
    const char *pat = lc_call_arg(c, "pattern");
    if (!pat || !*pat) { snprintf(sum, sc, "glob: no pattern"); return dup_str("glob needs a pattern"); }
    char p[2048];
    snprintf(p, sizeof p, "%s", pat);
    for (char *q = p; *q; q++) if (*q == '\\') *q = '/';
    char *start = p;
    while (start[0] == '.' && start[1] == '/') start += 2;
    /* the folders before the first wildcard are walked directly */
    char base[2048] = "";
    const char *wild = strpbrk(start, "*?");
    const char *cut = NULL;
    for (const char *q = start; *q && (!wild || q < wild); q++) if (*q == '/') cut = q;
    if (cut) { memcpy(base, start, (size_t)(cut - start)); base[cut - start] = 0; }
    if (!strchr(start, '/') && !strstr(start, "**")) {   /* a bare name pattern matches in any folder */
        char any[2100];
        snprintf(any, sizeof any, "**/%s", start);
        snprintf(p, sizeof p, "%.*s", (int)sizeof p - 1, any);
        start = p;
    }
    char dir[4096], why[600];
    if (resolve(*base ? base : ".", dir, sizeof dir, why, sizeof why)) { snprintf(sum, sc, "%s", why); return dup_str(why); }
    Str s = { 0 };
    Walk w;
    memset(&w, 0, sizeof w);
    w.pattern = start;
    w.out = &s;
    walk(&w, dir, base, 0);
    /* the newest first (what was just being worked on), then by path */
    qsort(w.hits, (size_t)w.nhits, sizeof *w.hits, hit_cmp);
    for (int i = 0; i < w.nhits; i++) {
        if (i < LIST_CAP) s_fmt(&s, "%s\n", w.hits[i].rel);
        free(w.hits[i].rel);
    }
    free(w.hits);
    if (!w.found) s_fmt(&s, "no file matches %s\n", pat);
    if (w.cap_hit) s_fmt(&s, "[stopped after %d files; narrow the pattern]\n", GLOB_SCAN);
    else if (w.nhits > LIST_CAP) s_fmt(&s, "[%d more files, older; narrow the pattern]\n", w.nhits - LIST_CAP);
    snprintf(sum, sc, "%s · %d file%s", pat, w.found, w.found == 1 ? "" : "s");
    *ok = 1;
    return capped(&s);
}

static const char *find_text(const char *hay, size_t n, const char *needle, size_t m, int icase)
{
    if (!m || m > n) return NULL;
    for (size_t i = 0; i + m <= n; i++) {
        size_t j = 0;
        if (icase) {
            for (; j < m; j++) {
                int a = (unsigned char)hay[i + j], b = (unsigned char)needle[j];
                if (a >= 'A' && a <= 'Z') a += 32;
                if (b >= 'A' && b <= 'Z') b += 32;
                if (a != b) break;
            }
        } else {
            for (; j < m && hay[i + j] == needle[j]; j++) {}
        }
        if (j == m) return hay + i;
    }
    return NULL;
}

static int utf8_valid(const char *s, long n)
{
    const unsigned char *p = (const unsigned char *)s, *e = p + n;
    while (p < e) {
        const int c = *p;
        const int len = c < 0x80 ? 1 : c >= 0xC2 && c <= 0xDF ? 2 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xF0 && c <= 0xF4 ? 4 : 0;
        if (!len || p + len > e) return 0;
        for (int k = 1; k < len; k++) if ((p[k] & 0xC0) != 0x80) return 0;
        p += len;
    }
    return 1;
}

/* each byte of s as the code point of the same number (Latin-1), in UTF-8; *n the new length */
static char *latin1_to_utf8(const char *s, long *n)
{
    char *u = (char *)malloc((size_t)*n * 2 + 1);
    if (!u) return NULL;
    long k = 0;
    for (long i = 0; i < *n; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) u[k++] = (char)c;
        else { u[k++] = (char)(0xC0 | c >> 6); u[k++] = (char)(0x80 | (c & 63)); }
    }
    u[k] = 0;
    *n = k;
    return u;
}

static void grep_file(Walk *w, const char *abs, const char *rel)
{
    long n = 0;
    const char *err = NULL;
    char *b = slurp(abs, &n, &err);
    if (!b) return;
    if (n > (2 << 20) || is_binary(b, n)) { free(b); return; }
    w->files++;
    if (!utf8_valid(b, n)) {                       /* a Latin-1 / Windows-1252 file: read as UTF-8 */
        char *u = latin1_to_utf8(b, &n);
        if (u) { free(b); b = u; }
    }
    const size_t m = strlen(w->needle);
    int line = 1;
    const char *p = b, *end = b + n;
    while (p < end && !w->cap_hit) {
        const char *q = memchr(p, '\n', (size_t)(end - p));
        size_t len = (size_t)((q ? q : end) - p);
        if (len && p[len - 1] == '\r') len--;
        if (w->re ? lc_regex_search(w->re, p, len) : find_text(p, len, w->needle, m, w->icase) != NULL) {
            s_fmt(w->out, "%s:%d: ", rel, line);
            s_add(w->out, p, len > 240 ? 240 : len);
            s_str(w->out, len > 240 ? " [...]\n" : "\n");
            if (++w->found >= GREP_CAP) w->cap_hit = 1;
        }
        p = q ? q + 1 : end;
        line++;
    }
    free(b);
}

static char *t_grep(const LcCall *c, char *sum, size_t sc, int *ok)
{
    const char *needle = lc_call_arg(c, "pattern");
    if (!needle || !*needle) { snprintf(sum, sc, "grep: no pattern"); return dup_str("grep needs a pattern"); }
    char path[4096], why[600], rel[1024];
    if (resolve(lc_call_arg(c, "path"), path, sizeof path, why, sizeof why)) { snprintf(sum, sc, "%s", why); return dup_str(why); }
    Str s = { 0 };
    Walk w;
    memset(&w, 0, sizeof w);
    w.pattern = lc_call_arg(c, "glob");
    w.needle = needle;
    w.icase = arg_bool(c, "ignore_case");
    w.out = &s;
    if (w.pattern && !*w.pattern) w.pattern = NULL;
    if (!arg_bool(c, "literal")) {
        char err[200];
        w.re = lc_regex_new(needle, w.icase, err, sizeof err);
        if (!w.re) s_fmt(&s, "(\"%s\" is not a regular expression this grep takes: %s; searched as literal text)\n", needle, err);
    }
    rel_of(path, rel, sizeof rel);
    int n = 0;
    Ent *probe = read_dir(path, &n);
    if (probe && n) walk(&w, path, strcmp(rel, ".") ? rel : "", 0);
    else            grep_file(&w, path, rel);
    if (probe) ents_free(probe, n);
    if (!w.found) s_fmt(&s, "no match for \"%s\" in %d file%s\n", needle, w.files, w.files == 1 ? "" : "s");
    if (w.cap_hit) s_fmt(&s, "[stopped at %d matches; narrow the search]\n", GREP_CAP);
    lc_regex_free(w.re);
    snprintf(sum, sc, "%d match%s in %d file%s", w.found, w.found == 1 ? "" : "es", w.files, w.files == 1 ? "" : "s");
    *ok = 1;
    return capped(&s);
}

/* ------------------------------------------------------------- write / edit ---- */

static char *t_write(const LcCall *c, char *sum, size_t sc, int *ok)
{
    const char *content = lc_call_arg(c, "content");
    char path[4096], why[600], rel[1024];
    if (!content) { snprintf(sum, sc, "write_file: no content"); return dup_str("write_file needs content"); }
    if (resolve(lc_call_arg(c, "path"), path, sizeof path, why, sizeof why)) { snprintf(sum, sc, "%s", why); return dup_str(why); }
    rel_of(path, rel, sizeof rel);
    FILE *f = open_file(path, "rb");
    const int existed = f != NULL;
    if (f) fclose(f);
    const size_t n = strlen(content);
    if (write_all(path, content, n)) { snprintf(sum, sc, "%s: write failed", rel); return dup_fmt("could not write %s", rel); }
    int lines = n ? 1 : 0;
    for (size_t i = 0; i + 1 < n; i++) lines += content[i] == '\n';
    snprintf(sum, sc, "%s · %s, %d lines", rel, existed ? "replaced" : "created", lines);
    *ok = 1;
    return dup_fmt("%s %s: %zu bytes, %d lines", existed ? "replaced" : "created", rel, n, lines);
}

static char *t_edit(const LcCall *c, char *sum, size_t sc, int *ok)
{
    const char *old = lc_call_arg(c, "old_string"), *neu = lc_call_arg(c, "new_string");
    char path[4096], why[600], rel[1024];
    if (!old || !*old || !neu) { snprintf(sum, sc, "edit_file: needs old_string and new_string"); return dup_str("edit_file needs old_string and new_string"); }
    if (resolve(lc_call_arg(c, "path"), path, sizeof path, why, sizeof why)) { snprintf(sum, sc, "%s", why); return dup_str(why); }
    rel_of(path, rel, sizeof rel);
    long n = 0;
    const char *err = NULL;
    char *b = slurp(path, &n, &err);
    if (!b) { snprintf(sum, sc, "%s: %s", rel, err); return dup_fmt("cannot edit %s: %s", rel, err); }
    /* a file with CRLF line ends: the model writes LF, so match and write in the file's style */
    const int crlf = strstr(b, "\r\n") != NULL;
    Str o = { 0 }, nw = { 0 };
    for (const char *p = old; *p; p++) { if (crlf && *p == '\n' && (p == old || p[-1] != '\r')) s_str(&o, "\r\n"); else s_add(&o, p, 1); }
    for (const char *p = neu; *p; p++) { if (crlf && *p == '\n' && (p == neu || p[-1] != '\r')) s_str(&nw, "\r\n"); else s_add(&nw, p, 1); }
    if (!nw.p) s_str(&nw, "");
    int count = 0;
    for (const char *p = b; (p = strstr(p, o.p)); p += o.n) count++;
    const int all = arg_bool(c, "replace_all");
    char *res;
    if (!count) {
        snprintf(sum, sc, "%s: old_string not found", rel);
        res = dup_fmt("old_string was not found in %s; read the file again and copy the text exactly (without line numbers)", rel);
    } else if (count > 1 && !all) {
        snprintf(sum, sc, "%s: old_string found %d times", rel, count);
        res = dup_fmt("old_string appears %d times in %s; include more surrounding lines to make it unique, or set replace_all", count, rel);
    } else {
        Str out = { 0 };
        int first_line = 0;
        const char *p = b, *q;
        while ((q = strstr(p, o.p))) {
            if (!first_line) { first_line = 1; for (const char *r = b; r < q; r++) first_line += *r == '\n'; }
            s_add(&out, p, (size_t)(q - p));
            s_add(&out, nw.p, nw.n);
            p = q + o.n;
            if (!all) break;
        }
        s_str(&out, p);
        if (write_all(path, out.p ? out.p : "", out.n)) {
            snprintf(sum, sc, "%s: write failed", rel);
            res = dup_fmt("could not write %s", rel);
        } else {
            const int k = all ? count : 1;
            snprintf(sum, sc, "%s · %d change%s at line %d", rel, k, k == 1 ? "" : "s", first_line);
            res = dup_fmt("edited %s: %d replacement%s, the first at line %d", rel, k, k == 1 ? "" : "s", first_line);
            *ok = 1;
        }
        free(out.p);
    }
    free(o.p);
    free(nw.p);
    free(b);
    return res;
}

/* --------------------------------------------------------------- apply_patch ---- */
/* opencode's (and Codex's) patch format: several files changed, added, deleted or moved in one call,
 * all or nothing: every hunk is located before the first file is written.
 *     *** Begin Patch
 *     *** Add File: path            then its lines, each after +
 *     *** Delete File: path
 *     *** Update File: path         [*** Move to: new path]
 *     @@ [a line that comes before the hunk, to find it]
 *      context line / -removed line / +added line
 *     *** End Patch
 * A hunk's lines are found from where the previous one ended: exactly, then ignoring the spaces at
 * the ends of lines, then ignoring the spaces at both ends. */

typedef struct { const char *p; size_t n; } Ln;

typedef struct {
    int    op;               /* 'A' add, 'D' delete, 'U' update */
    char   path[1024], to[1024];
    char  *text;             /* the new content (add, update) */
    size_t len;
    int    crlf;
} PFile;

static int split_lines(const char *s, size_t n, Ln **out)
{
    int cap = 64, k = 0;
    Ln *v = (Ln *)malloc((size_t)cap * sizeof *v);
    if (!v) return -1;
    for (size_t i = 0; i < n;) {
        size_t j = i;
        while (j < n && s[j] != '\n') j++;
        size_t len = j - i;
        if (len && s[i + len - 1] == '\r') len--;
        if (k == cap) { Ln *g = (Ln *)realloc(v, (size_t)(cap *= 2) * sizeof *v); if (!g) { free(v); return -1; } v = g; }
        v[k].p = s + i;
        v[k].n = len;
        k++;
        i = j + 1;
    }
    *out = v;
    return k;
}

/* a line equal to another at a level: 0 exactly, 1 but for the spaces at its end, 2 at both ends */
static int line_eq(Ln a, Ln b, int level)
{
    if (level >= 1) {
        while (a.n && (a.p[a.n - 1] == ' ' || a.p[a.n - 1] == '\t')) a.n--;
        while (b.n && (b.p[b.n - 1] == ' ' || b.p[b.n - 1] == '\t')) b.n--;
    }
    if (level >= 2) {
        while (a.n && (*a.p == ' ' || *a.p == '\t')) a.p++, a.n--;
        while (b.n && (*b.p == ' ' || *b.p == '\t')) b.p++, b.n--;
    }
    return a.n == b.n && !memcmp(a.p, b.p, a.n);
}

/* where old[0..m) first appears in f[from..n), or -1 */
static int find_lines(const Ln *f, int n, int from, const Ln *old, int m)
{
    for (int level = 0; level < 3; level++)
        for (int i = from; i + m <= n; i++) {
            int j = 0;
            while (j < m && line_eq(f[i + j], old[j], level)) j++;
            if (j == m) return i;
        }
    return -1;
}

static int starts(Ln l, const char *pfx)
{
    const size_t k = strlen(pfx);
    return l.n >= k && !memcmp(l.p, pfx, k);
}

static void ln_copy(char *dst, size_t cap, Ln l, size_t skip)
{
    size_t n = l.n > skip ? l.n - skip : 0;
    const char *p = l.p + skip;
    while (n && (*p == ' ' || *p == '\t')) p++, n--;
    while (n && (p[n - 1] == ' ' || p[n - 1] == '\t')) n--;
    if (n >= cap) n = cap - 1;
    memcpy(dst, p, n);
    dst[n] = 0;
}

/* the hunks of one Update (lines h[0..nh)) applied to the file's text; NULL and why on failure */
static char *patch_update(const char *rel, const char *body, size_t blen, const Ln *h, int nh, size_t *olen, Str *why)
{
    Ln *f = NULL;
    const int nf = split_lines(body, blen, &f);
    if (nf < 0) { s_str(why, "out of memory"); return NULL; }
    const int crlf = strstr(body, "\r\n") != NULL;
    const int had_nl = blen && body[blen - 1] == '\n';
    Str out = { 0 };
    int cur = 0, i = 0, hunk = 0, bad = 0;
    Ln *old = (Ln *)malloc((size_t)(nh + 1) * sizeof *old), *neu = (Ln *)malloc((size_t)(nh + 1) * sizeof *neu);
    if (!old || !neu) { s_str(why, "out of memory"); bad = 1; }
    while (!bad && i < nh) {
        int from = cur;
        hunk++;
        if (starts(h[i], "@@")) {                  /* the header: a line to find first */
            char head[600];
            ln_copy(head, sizeof head, h[i], 2);
            i++;
            if (*head) {
                const Ln hl = { head, strlen(head) };
                const int at = find_lines(f, nf, from, &hl, 1);
                if (at >= 0) from = at + 1;        /* not found: the hunk's own lines still decide */
            }
        }
        int no = 0, nn = 0;
        while (i < nh && !starts(h[i], "@@")) {
            Ln l = h[i++];
            if (starts(l, "*** End of File")) continue;
            const char c = l.n ? l.p[0] : ' ';
            Ln t = l;
            if (c == ' ' || c == '-' || c == '+') { t.p++; t.n--; }
            if (!l.n) t.n = 0;
            if (c == '-') old[no++] = t;
            else if (c == '+') neu[nn++] = t;
            else { old[no++] = t; neu[nn++] = t; }   /* context (a missing space taken as one) */
        }
        int at;
        if (!no) at = from > cur ? from : nf;      /* only additions: after the header, or at the end */
        else at = find_lines(f, nf, from, old, no);
        if (at < 0 && from > cur) at = find_lines(f, nf, cur, old, no);
        if (at < 0) {
            s_fmt(why, "%s: hunk %d not found; its lines (context and -) must be in the file as it is now:\n", rel, hunk);
            for (int k = 0; k < no && k < 12; k++) { s_str(why, "  "); s_add(why, old[k].p, old[k].n); s_str(why, "\n"); }
            bad = 1;
            break;
        }
        for (int k = cur; k < at; k++) { s_add(&out, f[k].p, f[k].n); s_str(&out, crlf ? "\r\n" : "\n"); }
        for (int k = 0; k < nn; k++) { s_add(&out, neu[k].p, neu[k].n); s_str(&out, crlf ? "\r\n" : "\n"); }
        cur = at + no;
    }
    if (!bad) {
        for (int k = cur; k < nf; k++) { s_add(&out, f[k].p, f[k].n); s_str(&out, crlf ? "\r\n" : "\n"); }
        if (!had_nl && out.n) out.n -= crlf ? 2 : 1, out.p[out.n] = 0;   /* as the file ended */
    }
    free(old);
    free(neu);
    free(f);
    if (bad) { free(out.p); return NULL; }
    *olen = out.n;
    return out.p ? out.p : dup_str("");
}

static char *t_patch(const LcCall *c, char *sum, size_t sc, int *ok)
{
    const char *text = lc_call_arg(c, "patch");
    if (!text || !*text) { snprintf(sum, sc, "apply_patch: no patch"); return dup_str("apply_patch needs patch"); }
    Ln *l = NULL;
    const int nl = split_lines(text, strlen(text), &l);
    if (nl < 0) return dup_str("out of memory");
    PFile files[64];
    int nfiles = 0;
    Str why = { 0 };
    int i = 0;
    while (i < nl && !starts(l[i], "*** Begin Patch")) i++;
    if (i == nl) i = 0;                            /* no Begin line: taken as if it were there */
    else i++;
    while (i < nl && !why.n) {
        Ln x = l[i];
        if (starts(x, "*** End Patch")) break;
        if (!x.n || (x.n == 1 && x.p[0] == ' ')) { i++; continue; }
        int op = starts(x, "*** Add File:") ? 'A' : starts(x, "*** Delete File:") ? 'D' : starts(x, "*** Update File:") ? 'U' : 0;
        if (!op) { s_fmt(&why, "line %d: expected *** Add File:, *** Update File: or *** Delete File:, not: %.*s\n", i + 1, (int)(x.n > 120 ? 120 : x.n), x.p); break; }
        if (nfiles == 64) { s_str(&why, "more than 64 files in one patch; split it\n"); break; }
        PFile *pf = &files[nfiles];
        memset(pf, 0, sizeof *pf);
        pf->op = op;
        ln_copy(pf->path, sizeof pf->path, x, op == 'A' ? 13 : op == 'D' ? 16 : 16);
        i++;
        char abs[4096], w[600], rel[1024];
        if (resolve(pf->path, abs, sizeof abs, w, sizeof w)) { s_fmt(&why, "%s\n", w); break; }
        rel_of(abs, rel, sizeof rel);
        snprintf(pf->path, sizeof pf->path, "%s", rel);
        int twice = 0;
        for (int k = 0; k < nfiles; k++) twice |= !strcmp(files[k].path, rel);
        if (twice) { s_fmt(&why, "%s appears twice; put all of its hunks under one *** Update File:\n", rel); break; }
        if (op == 'A') {
            Str body = { 0 };
            while (i < nl && !starts(l[i], "*** ")) {
                Ln t = l[i++];
                if (t.n && t.p[0] == '+') { t.p++; t.n--; }
                s_add(&body, t.p, t.n);
                s_str(&body, "\n");
            }
            FILE *e = open_file(abs, "rb");
            if (e) { fclose(e); s_fmt(&why, "%s already exists; use *** Update File: to change it\n", rel); free(body.p); break; }
            pf->text = body.p ? body.p : dup_str("");
            pf->len = body.n;
        } else {
            long n = 0;
            const char *err = NULL;
            char *b = slurp(abs, &n, &err);
            if (!b) { s_fmt(&why, "%s: %s\n", rel, err); break; }
            if (op == 'U') {
                if (i < nl && starts(l[i], "*** Move to:")) {
                    char to[1024], ta[4096];
                    ln_copy(to, sizeof to, l[i], 12);
                    i++;
                    if (resolve(to, ta, sizeof ta, w, sizeof w)) { s_fmt(&why, "%s\n", w); free(b); break; }
                    rel_of(ta, pf->to, sizeof pf->to);
                }
                const int h0 = i;
                while (i < nl && !(starts(l[i], "*** ") && !starts(l[i], "*** End of File"))) i++;
                pf->text = patch_update(rel, b, (size_t)n, l + h0, i - h0, &pf->len, &why);
                if (!pf->text) { free(b); break; }
            }
            free(b);
        }
        nfiles++;
    }
    if (!why.n && !nfiles) s_str(&why, "the patch changes no file\n");
    Str r = { 0 };
    if (why.n) {
        s_str(&r, "the patch was not applied (no file changed):\n");
        s_add(&r, why.p, why.n);
        snprintf(sum, sc, "apply_patch: not applied");
    } else {
        int written = 0;
        char line[200] = "";
        for (int k = 0; k < nfiles; k++) {
            PFile *pf = &files[k];
            char abs[4096], w[600];
            resolve(pf->to[0] ? pf->to : pf->path, abs, sizeof abs, w, sizeof w);
            if (pf->op == 'D') {
                char src[4096];
                resolve(pf->path, src, sizeof src, w, sizeof w);
#ifdef _WIN32
                wchar_t *ws = wide(src);
                const int gone = ws && DeleteFileW(ws);
                free(ws);
#else
                const int gone = unlink(src) == 0;
#endif
                s_fmt(&r, "%s %s\n", gone ? "deleted" : "could not delete", pf->path);
            } else if (write_all(abs, pf->text, pf->len)) {
                s_fmt(&r, "could not write %s\n", pf->to[0] ? pf->to : pf->path);
            } else {
                if (pf->to[0]) {
                    char src[4096];
                    resolve(pf->path, src, sizeof src, w, sizeof w);
                    remove(src);
                    s_fmt(&r, "updated %s, moved to %s\n", pf->path, pf->to);
                } else {
                    s_fmt(&r, "%s %s\n", pf->op == 'A' ? "added" : "updated", pf->path);
                }
                written++;
            }
            const size_t ll = strlen(line);
            if (ll < 120) snprintf(line + ll, sizeof line - ll, "%s%c %s", k ? ", " : "", pf->op == 'U' ? 'M' : pf->op, pf->path);
        }
        snprintf(sum, sc, "%s", line);
        *ok = 1;
    }
    for (int k = 0; k < nfiles; k++) free(files[k].text);
    free(why.p);
    free(l);
    return r.p ? r.p : dup_str("");
}

/* ------------------------------------------------------------------------ run ---- */

static double now_s(void)
{
    struct timespec t;
#ifdef _WIN32
    timespec_get(&t, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &t);          /* C11's timespec_get is not in gnu99's glibc headers */
#endif
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Keep the first and the last part of a long output. */
static void keep_ends(Str *s)
{
    if (s->n <= RUN_OUT_CAP) return;
    const size_t head = RUN_OUT_CAP / 3, tail = RUN_OUT_CAP - head;
    Str o = { 0 };
    s_add(&o, s->p, head);
    s_fmt(&o, "\n[... %zu characters of output left out ...]\n", s->n - head - tail);
    s_add(&o, s->p + s->n - tail, tail);
    free(s->p);
    *s = o;
}

static char *t_run(const LcCall *c, volatile int *stop, char *sum, size_t sc, int *ok)
{
    const char *cmd = lc_call_arg(c, "command");
    if (!cmd || !*cmd) { snprintf(sum, sc, "run: no command"); return dup_str("run needs a command"); }
    int timeout = arg_int(c, "timeout_s", 120);
    if (timeout < 1) timeout = 120;
    if (timeout > 1800) timeout = 1800;
    if (!g_root[0]) lc_tools_init(NULL);
    Str out = { 0 };
    int code = -1, timed_out = 0, stopped = 0;
    const double t0 = now_s();
#ifdef _WIN32
    /* the output as UTF-8, no profile, no prompt for input; the script goes base64-encoded
     * (UTF-16LE), so no quote or backslash in it is reinterpreted on the way */
    Str line = { 0 };
    const int bash = !lc_tools_shell();
    if (bash) {                                  /* Git Bash: the command as one quoted argument */
        s_str(&line, "\"");
        s_str(&line, git_bash());
        s_str(&line, "\" --noprofile --norc -c \"");
        for (const char *p = cmd;; p++) {        /* quoted as CommandLineToArgvW reads it back */
            int bs = 0;
            while (*p == '\\') { bs++; p++; }
            if (!*p) { for (int i = 0; i < bs * 2; i++) s_str(&line, "\\"); break; }
            if (*p == '"') { for (int i = 0; i < bs * 2 + 1; i++) s_str(&line, "\\"); s_str(&line, "\""); continue; }
            for (int i = 0; i < bs; i++) s_str(&line, "\\");
            s_add(&line, p, 1);
        }
        s_str(&line, "\"");
    } else {
        Str script = { 0 };
        s_str(&script, "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $ProgressPreference='SilentlyContinue'; ");
        s_str(&script, cmd);
        wchar_t *ws = wide(script.p);
        free(script.p);
        const size_t nb = ws ? wcslen(ws) * 2 : 0;
        const unsigned char *bytes = (const unsigned char *)ws;
        static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        s_str(&line, "pwsh -NoProfile -NonInteractive -EncodedCommand ");
        for (size_t i = 0; i < nb; i += 3) {
            const unsigned v = (unsigned)bytes[i] << 16 | (i + 1 < nb ? (unsigned)bytes[i + 1] << 8 : 0) | (i + 2 < nb ? bytes[i + 2] : 0);
            const char q[4] = { b64[v >> 18 & 63], b64[v >> 12 & 63], i + 1 < nb ? b64[v >> 6 & 63] : '=', i + 2 < nb ? b64[v & 63] : '=' };
            s_add(&line, q, 4);
        }
        free(ws);
    }
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 0)) { free(line.p); snprintf(sum, sc, "run: no pipe"); return dup_str("could not start the command (pipe)"); }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);
    STARTUPINFOW si;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = wr;
    si.hStdError = wr;
    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof pi);
    wchar_t *wl = wide(line.p), *wd = wide(g_root);
    BOOL started = wl && wd && CreateProcessW(NULL, wl, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, wd, &si, &pi);
    if (!started && wl && !bash) {              /* no pwsh: Windows PowerShell */
        free(wl);
        Str l2 = { 0 };
        s_str(&l2, "powershell");
        s_str(&l2, line.p + 4);
        wl = wide(l2.p);
        free(l2.p);
        started = wl && wd && CreateProcessW(NULL, wl, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, wd, &si, &pi);
    }
    free(wl);
    free(wd);
    free(line.p);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!started) {
        CloseHandle(rd);
        snprintf(sum, sc, "run: cannot start %s", bash ? "bash" : "PowerShell");
        return dup_str(bash ? "could not start Git Bash" : "could not start PowerShell");
    }
    char buf[8192];
    for (;;) {
        DWORD avail = 0;
        if (PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL) && avail) {
            DWORD got = 0;
            if (ReadFile(rd, buf, avail < sizeof buf ? avail : sizeof buf, &got, NULL) && got) s_add(&out, buf, got);
            if (out.n > 4 * RUN_OUT_CAP) keep_ends(&out);
            continue;
        }
        if (WaitForSingleObject(pi.hProcess, 50) == WAIT_OBJECT_0) {
            DWORD got = 0;
            while (PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL) && avail && ReadFile(rd, buf, avail < sizeof buf ? avail : sizeof buf, &got, NULL) && got)
                s_add(&out, buf, got);
            break;
        }
        if (stop && *stop) { stopped = 1; break; }
        if (now_s() - t0 > timeout) { timed_out = 1; break; }
    }
    if (timed_out || stopped) {
        /* the whole tree: pwsh's children too */
        wchar_t kill[64];
        swprintf(kill, 64, L"taskkill /T /F /PID %lu", (unsigned long)pi.dwProcessId);
        STARTUPINFOW ks;
        PROCESS_INFORMATION kp;
        memset(&ks, 0, sizeof ks);
        ks.cb = sizeof ks;
        if (CreateProcessW(NULL, kill, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &ks, &kp)) {
            WaitForSingleObject(kp.hProcess, 5000);
            CloseHandle(kp.hProcess);
            CloseHandle(kp.hThread);
        }
        TerminateProcess(pi.hProcess, 1);
    }
    DWORD ec = 0;
    if (GetExitCodeProcess(pi.hProcess, &ec)) code = (int)ec;
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(rd);
#else
    char line[16384];
    snprintf(line, sizeof line, "cd '%s' && (%s) 2>&1", g_root, cmd);
    FILE *p = popen(line, "r");
    if (!p) { snprintf(sum, sc, "run: cannot start the shell"); return dup_str("could not start the shell"); }
    char buf[8192];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, p)) > 0) {
        s_add(&out, buf, got);
        if (stop && *stop) { stopped = 1; break; }
    }
    const int st = pclose(p);
    code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
#endif
    keep_ends(&out);
    const double dt = now_s() - t0;
    Str res = { 0 };
    if (out.n) { s_add(&res, out.p, out.n); if (out.p[out.n - 1] != '\n') s_str(&res, "\n"); }
    else s_str(&res, "(no output)\n");
    if (timed_out)    s_fmt(&res, "[stopped after %d s: the timeout]\n", timeout);
    else if (stopped) s_str(&res, "[stopped by the user]\n");
    else              s_fmt(&res, "[exit code %d · %.1f s]\n", code, dt);
    free(out.p);
    snprintf(sum, sc, "%s", timed_out ? "timed out" : stopped ? "stopped" : code == 0 ? "exit 0" : "failed");
    if (!timed_out && !stopped) snprintf(sum + strlen(sum), sc - strlen(sum), " · %.1f s", dt);
    if (!timed_out && !stopped && code != 0) snprintf(sum, sc, "exit %d · %.1f s", code, dt);
    *ok = !timed_out && !stopped && code == 0;
    return s_take(&res);
}

/* -------------------------------------------------------------------- skills ---- */
/* The plugins' skills (lc_plugins.c): the model sees those that are on, as many as fit; skill_search
 * finds the others; skill(name) gives the instructions, skill(name, file) another file of the
 * skill's folder. */
#define SKILLS_BUDGET 7000                       /* characters of the prompt's list (~1800 tokens) */

static void plugins_ready(void)
{
    if (!g_root[0]) lc_tools_init(NULL);
    lc_plugins_load(g_root);
}

const char *lc_skills_block(void)
{
    plugins_ready();
    return lc_skills_prompt(SKILLS_BUDGET);
}

static char *t_skill_search(const LcCall *c, char *sum, size_t sc, int *ok)
{
    plugins_ready();
    const char *q = lc_call_arg(c, "query");
    char *r = lc_skill_search(q ? q : "", 8);
    snprintf(sum, sc, "%s", q ? q : "");
    *ok = 1;
    return r;
}

static char *t_skill(const LcCall *c, char *sum, size_t sc, int *ok)
{
    plugins_ready();
    const char *name = lc_call_arg(c, "name"), *file = lc_call_arg(c, "file");
    const LcSkill *sk = lc_skill_find(name);
    if (!sk) {
        snprintf(sum, sc, "no skill %s", name ? name : "");
        return dup_fmt("There is no skill named \"%s\" that is on. The skills are listed in the instructions; "
                       "skill_search finds others.", name ? name : "");
    }
    if (file && (strstr(file, "..") || file[0] == '/' || file[0] == '\\' || strchr(file, ':'))) {
        snprintf(sum, sc, "outside the skill");
        return dup_str("file must be a path inside the skill's folder");
    }
    char path[4400];
    snprintf(path, sizeof path, "%s" SEP "%s", sk->dir, file && *file ? file : "SKILL.md");
    for (char *p = path + strlen(sk->dir); *p; p++) if (*p == '/' || *p == '\\') *p = SEP[0];
    long len = 0;
    const char *err = NULL;
    char *text = slurp(path, &len, &err);
    if (!text) { snprintf(sum, sc, "%s: %s", file ? file : "SKILL.md", err ? err : "unreadable"); return dup_fmt("cannot read it: %s", err ? err : "?"); }
    Str r = { 0 };
    const char *body = text;
    if (!file && !strncmp(text, "---", 3)) {            /* the instructions after the header */
        const char *end = strstr(text + 3, "\n---");
        if (end) { body = end + 4; while (*body == '\r' || *body == '\n') body++; }
    }
    s_fmt(&r, "Skill %s%s%s. Follow it for this task; where it differs from your usual workflow, it wins. "
              "Where it names tools of another agent (Bash, Read, Edit, Write, Grep, Glob, the Skill tool), use yours: "
              "run, read_file, edit_file, write_file, grep, glob, skill.\n\n",
          sk->name, file ? ", file " : "", file ? file : "");
    s_str(&r, body);
    if (!file) {                                          /* what else the folder holds */
        int n = 0;
        Ent *v = read_dir(sk->dir, &n);
        int any = 0;
        for (int i = 0; i < n; i++) {
            if (!strcmp(v[i].name, "SKILL.md")) continue;
            s_fmt(&r, "%s%s%s", any ? ", " : "\n\nOther files of this skill (skill with file=...): ", v[i].name, v[i].dir ? "/" : "");
            any = 1;
        }
        ents_free(v, n);
    }
    free(text);
    snprintf(sum, sc, "%s%s%s", sk->name, file ? " · " : "", file ? file : "");
    *ok = 1;
    return capped(&r);
}

char *lc_tool_run(const LcCall *c, volatile int *stop, char *summary, size_t scap, int *ok)
{
    *ok = 0;
    summary[0] = 0;
    if (!strcmp(c->name, "skill"))      return t_skill(c, summary, scap, ok);
    if (!strcmp(c->name, "skill_search")) return t_skill_search(c, summary, scap, ok);
    if (!strcmp(c->name, "webfetch"))   return lc_webfetch(lc_call_arg(c, "url"), 60000, summary, scap, ok);
    if (!strcmp(c->name, "read_file"))  return t_read(c, summary, scap, ok);
    if (!strcmp(c->name, "list_dir"))   return t_list(c, summary, scap, ok);
    if (!strcmp(c->name, "glob"))       return t_glob(c, summary, scap, ok);
    if (!strcmp(c->name, "grep"))       return t_grep(c, summary, scap, ok);
    if (!strcmp(c->name, "write_file")) return t_write(c, summary, scap, ok);
    if (!strcmp(c->name, "edit_file"))  return t_edit(c, summary, scap, ok);
    if (!strcmp(c->name, "apply_patch")) return t_patch(c, summary, scap, ok);
    if (!strcmp(c->name, "run"))        return t_run(c, stop, summary, scap, ok);
    snprintf(summary, scap, "unknown tool %s", c->name);
    return dup_fmt("there is no tool named %s", c->name);
}
