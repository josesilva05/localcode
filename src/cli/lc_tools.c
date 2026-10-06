/* lc_tools.c - see lc_tools.h. */
#define _POSIX_C_SOURCE 200809L

#include "lc_tools.h"

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
    if (!inside) {
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
    if (strncmp(whole, g_root, rl) || (whole[rl] && whole[rl] != '/')) {
        snprintf(why, wcap, "%s is outside the working directory %s", rel, g_root);
        return -1;
    }
    snprintf(out, cap, "%s", whole);
#endif
    return 0;
}

/* The path relative to the working directory, with '/' separators, for results. */
static const char *rel_of(const char *abs, char *buf, size_t cap)
{
    const size_t rl = strlen(g_root);
    const char *p = strlen(abs) >= rl ? abs + rl : abs;
    while (*p == '\\' || *p == '/') p++;
    snprintf(buf, cap, "%s", *p ? p : ".");
    for (char *q = buf; *q; q++) if (*q == '\\') *q = '/';
    return buf;
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
#define T_GLOB  "{\"type\": \"function\", \"function\": {\"name\": \"glob\", \"description\": \"Find files by name pattern, for example src/**/*.c or *.md. * matches within one folder name, ** any number of folders.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"pattern\": {\"type\": \"string\"}}, \"required\": [\"pattern\"]}}}"
#define T_GREP  "{\"type\": \"function\", \"function\": {\"name\": \"grep\", \"description\": \"Search text files for a literal string (not a regular expression); lists path:line: text of each match.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"pattern\": {\"type\": \"string\", \"description\": \"the text to find\"}, \"path\": {\"type\": \"string\", \"description\": \"folder or file to search, default .\"}, \"glob\": {\"type\": \"string\", \"description\": \"only file names matching this, for example *.c\"}, \"ignore_case\": {\"type\": \"boolean\"}}, \"required\": [\"pattern\"]}}}"
#define T_WRITE "{\"type\": \"function\", \"function\": {\"name\": \"write_file\", \"description\": \"Create a file or replace all of its content. Prefer edit_file to change part of an existing file.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"content\": {\"type\": \"string\", \"description\": \"the whole new content\"}}, \"required\": [\"path\", \"content\"]}}}"
#define T_EDIT  "{\"type\": \"function\", \"function\": {\"name\": \"edit_file\", \"description\": \"Replace an exact piece of text in a file. old_string must appear exactly once (copy it from read_file, without the line numbers, with enough surrounding lines to be unique), unless replace_all is true.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}, \"old_string\": {\"type\": \"string\"}, \"new_string\": {\"type\": \"string\"}, \"replace_all\": {\"type\": \"boolean\"}}, \"required\": [\"path\", \"old_string\", \"new_string\"]}}}"
#define T_RUN   "{\"type\": \"function\", \"function\": {\"name\": \"run\", \"description\": \"Run a PowerShell command in the working directory (builds, tests, git, scripts) and get its output and exit code. Not for interactive programs.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}, \"timeout_s\": {\"type\": \"integer\", \"description\": \"seconds before it is stopped, default 120, at most 1800\"}}, \"required\": [\"command\"]}}}"

const char *lc_tools_json(int level)
{
    if (level <= 1) return T_READ "\n" T_LIST "\n" T_GLOB "\n" T_GREP;
    return T_READ "\n" T_LIST "\n" T_GLOB "\n" T_GREP "\n" T_WRITE "\n" T_EDIT "\n" T_RUN;
}

int lc_tool_allowed(const char *name, int level)
{
    static const char *reading[] = { "read_file", "list_dir", "glob", "grep" };
    static const char *all[] = { "write_file", "edit_file", "run" };
    for (int i = 0; i < 4; i++) if (!strcmp(name, reading[i])) return level >= 1;
    for (int i = 0; i < 3; i++) if (!strcmp(name, all[i])) return level >= 2;
    return 0;
}

int lc_tool_needs_approval(const char *name)
{
    return !strcmp(name, "write_file") || !strcmp(name, "edit_file") || !strcmp(name, "run");
}

void lc_tool_title(const LcCall *c, char *buf, size_t cap)
{
    const char *a = lc_call_arg(c, "path");
    if (!strcmp(c->name, "run")) a = lc_call_arg(c, "command");
    else if (!strcmp(c->name, "glob") || !strcmp(c->name, "grep")) a = lc_call_arg(c, "pattern");
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

typedef struct { char *name; int dir; long long size; } Ent;

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

typedef struct {
    const char *pattern;      /* glob: the path pattern; grep: the name filter or NULL */
    const char *needle;       /* grep: the text */
    int         icase;
    Str        *out;
    int         found, files, cap_hit;
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
            s_fmt(w->out, "%s\n", r);
            if (++w->found >= LIST_CAP) w->cap_hit = 1;
        }
    }
    ents_free(v, n);
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
        snprintf(p, sizeof p, "%s", any);
        start = p;
    }
    char dir[4096], why[600];
    if (resolve(*base ? base : ".", dir, sizeof dir, why, sizeof why)) { snprintf(sum, sc, "%s", why); return dup_str(why); }
    Str s = { 0 };
    Walk w = { start, NULL, 0, &s, 0, 0, 0 };
    walk(&w, dir, base, 0);
    if (!w.found) s_fmt(&s, "no file matches %s\n", pat);
    if (w.cap_hit) s_fmt(&s, "[stopped at %d files; narrow the pattern]\n", LIST_CAP);
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

static void grep_file(Walk *w, const char *abs, const char *rel)
{
    long n = 0;
    const char *err = NULL;
    char *b = slurp(abs, &n, &err);
    if (!b) return;
    if (n > (2 << 20) || is_binary(b, n)) { free(b); return; }
    w->files++;
    const size_t m = strlen(w->needle);
    int line = 1;
    const char *p = b, *end = b + n;
    while (p < end && !w->cap_hit) {
        const char *q = memchr(p, '\n', (size_t)(end - p));
        size_t len = (size_t)((q ? q : end) - p);
        if (len && p[len - 1] == '\r') len--;
        if (find_text(p, len, w->needle, m, w->icase)) {
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
    Walk w = { lc_call_arg(c, "glob"), needle, arg_bool(c, "ignore_case"), &s, 0, 0, 0 };
    if (w.pattern && !*w.pattern) w.pattern = NULL;
    rel_of(path, rel, sizeof rel);
    int n = 0;
    Ent *probe = read_dir(path, &n);
    if (probe && n) walk(&w, path, strcmp(rel, ".") ? rel : "", 0);
    else            grep_file(&w, path, rel);
    if (probe) ents_free(probe, n);
    if (!w.found) s_fmt(&s, "no match for \"%s\" in %d file%s\n", needle, w.files, w.files == 1 ? "" : "s");
    if (w.cap_hit) s_fmt(&s, "[stopped at %d matches; narrow the search]\n", GREP_CAP);
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

/* ------------------------------------------------------------------------ run ---- */

static double now_s(void)
{
    struct timespec t;
    timespec_get(&t, TIME_UTC);
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
    {
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
    if (!started && wl) {                       /* no pwsh: Windows PowerShell */
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
    if (!started) { CloseHandle(rd); snprintf(sum, sc, "run: cannot start PowerShell"); return dup_str("could not start PowerShell"); }
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

char *lc_tool_run(const LcCall *c, volatile int *stop, char *summary, size_t scap, int *ok)
{
    *ok = 0;
    summary[0] = 0;
    if (!strcmp(c->name, "read_file"))  return t_read(c, summary, scap, ok);
    if (!strcmp(c->name, "list_dir"))   return t_list(c, summary, scap, ok);
    if (!strcmp(c->name, "glob"))       return t_glob(c, summary, scap, ok);
    if (!strcmp(c->name, "grep"))       return t_grep(c, summary, scap, ok);
    if (!strcmp(c->name, "write_file")) return t_write(c, summary, scap, ok);
    if (!strcmp(c->name, "edit_file"))  return t_edit(c, summary, scap, ok);
    if (!strcmp(c->name, "run"))        return t_run(c, stop, summary, scap, ok);
    snprintf(summary, scap, "unknown tool %s", c->name);
    return dup_fmt("there is no tool named %s", c->name);
}
