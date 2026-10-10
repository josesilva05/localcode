/* lc_snap.c - see lc_snap.h. */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/stat.h>
#endif

#include "lc_snap.h"
#include "lc_tools.h"

#ifdef _WIN32
#include <windows.h>
#else
#define popen_  popen
#define pclose_ pclose
#endif

#define MAX_TURNS 64
#define SNAP_FILES 30000
#define SNAP_BYTES (1LL << 30)

static char g_dir[4400];                      /* the shadow repository's git directory */
static char g_undo[MAX_TURNS][48], g_redo[MAX_TURNS][48];
static int  g_nundo, g_nredo, g_state;        /* g_state: 0 not tried, 1 ready, -1 no git */
static int  g_note;                           /* 1 undone, 2 redone since the model last heard */

static char *xdup(const char *s)
{
    const size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

const char *lc_snap_note(void)
{
    const int n = g_note;
    g_note = 0;
    if (n == 1) return "[localcode] The user undid the changes of your last turn (/undo): the files are as they were before it.";
    if (n == 2) return "[localcode] The user redid the changes of your last turn (/redo): the files are as you left them.";
    return NULL;
}

/* a git command on the shadow repository and the working directory; its output (malloc'd), *code its exit */
static char *git(int *code, const char *fmt, ...)
{
    char args[3000], cmd[8000];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(args, sizeof args, fmt, ap);
    va_end(ap);
#ifdef _WIN32
    /* git started directly (cmd.exe's quoting rules mangle the nested quotes) */
    snprintf(cmd, sizeof cmd, "git \"--git-dir=%s\" \"--work-tree=%s\" %s", g_dir, lc_tools_root(), args);
    const int wn = MultiByteToWideChar(CP_UTF8, 0, cmd, -1, NULL, 0);
    wchar_t *wcl = (wchar_t *)malloc((size_t)wn * sizeof(wchar_t));
    if (!wcl) { if (code) *code = -1; return NULL; }
    MultiByteToWideChar(CP_UTF8, 0, cmd, -1, wcl, wn);
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 0)) { free(wcl); if (code) *code = -1; return NULL; }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = NULL;
    si.hStdOutput = wr;
    si.hStdError = wr;
    PROCESS_INFORMATION pi;
    const BOOL ok = CreateProcessW(NULL, wcl, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    free(wcl);
    CloseHandle(wr);
    if (!ok) { CloseHandle(rd); if (code) *code = -1; return NULL; }
    size_t n = 0, cap = 4096;
    char *out = (char *)malloc(cap);
    for (;;) {
        if (n + 4096 > cap) { cap *= 2; char *g = (char *)realloc(out, cap); if (!g) break; out = g; }
        DWORD got = 0;
        if (!ReadFile(rd, out + n, 4095, &got, NULL) || !got) break;
        n += got;
    }
    if (out) out[n] = 0;
    WaitForSingleObject(pi.hProcess, 60000);
    DWORD ec = 1;
    GetExitCodeProcess(pi.hProcess, &ec);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(rd);
    if (code) *code = (int)ec;
    return out;
#else
    snprintf(cmd, sizeof cmd, "git --git-dir='%s' --work-tree='%s' %s 2>&1", g_dir, lc_tools_root(), args);
    FILE *p = popen_(cmd, "r");
    if (!p) { if (code) *code = -1; return NULL; }
    size_t n = 0, cap = 4096;
    char *out = (char *)malloc(cap);
    for (;;) {
        if (n + 2048 > cap) { cap *= 2; char *g = (char *)realloc(out, cap); if (!g) break; out = g; }
        const size_t got = fread(out + n, 1, 2047, p);
        if (!got) break;
        n += got;
    }
    if (out) out[n] = 0;
    const int rc = pclose_(p);
    if (code) *code = rc;
    return out;
#endif
}

static int ready(void)
{
    if (g_state) return g_state > 0;
    g_state = -1;
    const char *root = lc_tools_root();
    if (!root || !*root) return 0;
    unsigned long long h = 1469598103934665603ULL;
    for (const char *p = root; *p; p++) { h ^= (unsigned char)(*p >= 'A' && *p <= 'Z' ? *p + 32 : *p); h *= 1099511628211ULL; }
#ifdef _WIN32
    const char *base = getenv("LOCALAPPDATA");
    if (!base) return 0;
    snprintf(g_dir, sizeof g_dir, "%s\\localcode\\snapshots\\%016llx", base, h);
#else
    const char *base = getenv("HOME");
    if (!base) return 0;
    snprintf(g_dir, sizeof g_dir, "%s/.local/share/localcode/snapshots/%016llx", base, h);
#endif
    int rc = 0;
    char *o = git(&rc, "rev-parse --git-dir");
    free(o);
    if (rc != 0) {                                 /* a new one, with what is never worth keeping */
        char parent[4400];                         /* git init does not make the folders above it */
        snprintf(parent, sizeof parent, "%s", g_dir);
        for (char *q = parent + 3; *q; q++)
            if (*q == '\\' || *q == '/') {
                const char keep = *q;
                *q = 0;
#ifdef _WIN32
                CreateDirectoryA(parent, NULL);
#else
                mkdir(parent, 0755);
#endif
                *q = keep;
            }
        o = git(&rc, "init -q");
        free(o);
        if (rc != 0) return 0;
        free(git(NULL, "config user.email localcode@localhost"));
        free(git(NULL, "config user.name localcode"));
        free(git(NULL, "config core.autocrlf false"));
        char ex[4500];
        snprintf(ex, sizeof ex, "%s/info/exclude", g_dir);
        FILE *f = fopen(ex, "ab");
        if (f) {
            fputs("node_modules/\n.venv/\nvenv/\n__pycache__/\n*.gguf\n*.q38s\n.git/\n", f);
            fclose(f);
        }
    }
    /* never a copy of a huge folder: what git would take (the .gitignore's rules kept), at most
     * SNAP_FILES files and SNAP_BYTES bytes, or no snapshots here */
    char *list = git(&rc, "ls-files -c -o --exclude-standard");
    long long files = 0, bytes = 0;
    for (char *line = list; line && *line && files <= SNAP_FILES && bytes <= SNAP_BYTES;) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (*line) {
            char path[4400];
            snprintf(path, sizeof path, "%s/%s", lc_tools_root(), line);
            FILE *f = fopen(path, "rb");
            if (f) {
#ifdef _WIN32
                _fseeki64(f, 0, SEEK_END);
                bytes += _ftelli64(f);
#else
                fseeko(f, 0, SEEK_END);
                bytes += ftello(f);
#endif
                fclose(f);
            }
            files++;
        }
        line = nl ? nl + 1 : NULL;
    }
    free(list);
    if (files > SNAP_FILES || bytes > SNAP_BYTES) return 0;
    g_state = 1;
    return 1;
}

/* the folder as it is: a commit; its id into id (40 hex), 0 or -1 */
static int take(char *id)
{
    int rc = 0;
    free(git(&rc, "add -A"));
    if (rc != 0) return -1;
    free(git(&rc, "commit -q --allow-empty --no-verify -m turn"));
    if (rc != 0) return -1;
    char *o = git(&rc, "rev-parse HEAD");
    if (rc != 0 || !o || strlen(o) < 40) { free(o); return -1; }
    snprintf(id, 41, "%.40s", o);
    free(o);
    return 0;
}

int lc_snap_turn(void)
{
    if (!ready()) return -1;
    char id[48];
    if (take(id)) return -1;
    if (g_nundo == MAX_TURNS) { memmove(g_undo, g_undo + 1, sizeof g_undo[0] * (MAX_TURNS - 1)); g_nundo--; }
    snprintf(g_undo[g_nundo++], sizeof g_undo[0], "%s", id);
    g_nredo = 0;                                   /* a new turn: what was undone stays undone */
    return 0;
}

/* the folder put back as it was at id: files added since removed, the others as they were; now: the
 * current state's id (taken here) */
static int restore(const char *id, char *now, int *changed)
{
    if (take(now)) return -1;
    int rc = 0;
    char *names = git(&rc, "diff --name-only %s %s", id, now);
    int n = 0;
    for (char *p = names; p && *p; p++) if (*p == '\n') n++;
    free(names);
    *changed = n;
    if (!n) return 0;
    char *added = git(&rc, "diff --name-only --diff-filter=A %s %s", id, now);
    for (char *line = added; line && *line;) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (*line) {
            char path[4400];
            snprintf(path, sizeof path, "%s/%s", lc_tools_root(), line);
            remove(path);
        }
        line = nl ? nl + 1 : NULL;
    }
    free(added);
    /* the files the snapshot had, back (none: an empty folder, only the removals above) */
    char *tree = git(&rc, "ls-tree -r --name-only %s", id);
    const int empty = !tree || !*tree;
    free(tree);
    if (empty) return 0;
    char *o = git(&rc, "checkout -q %s -- .", id);
    free(o);
    return rc == 0 ? 0 : -1;
}

char *lc_snap_undo(int *changed)
{
    *changed = 0;
    char buf[300];
    if (!ready()) return xdup("undo is off here: no git, or the folder is too large (more than 30000 files or 1 GB outside .gitignore)");
    if (!g_nundo) return xdup("nothing to undo: no agent turn yet in this session");
    char now[48];
    const char *id = g_undo[g_nundo - 1];
    if (restore(id, now, changed)) return xdup("undo failed (git)");
    g_nundo--;
    snprintf(g_redo[g_nredo++ % MAX_TURNS], sizeof g_redo[0], "%s", now);
    if (*changed) {
        snprintf(buf, sizeof buf, "undone: %d file%s back as before the last agent turn · /redo puts them back", *changed,
                 *changed == 1 ? "" : "s");
        g_note = 1;
    } else {
        snprintf(buf, sizeof buf, "the last agent turn changed no file");
    }
    return xdup(buf);
}

char *lc_snap_redo(int *changed)
{
    *changed = 0;
    char buf[300];
    if (!ready() || !g_nredo) return xdup("nothing to redo");
    char now[48];
    const char *id = g_redo[(g_nredo - 1) % MAX_TURNS];
    if (restore(id, now, changed)) return xdup("redo failed (git)");
    g_nredo--;
    if (g_nundo < MAX_TURNS) snprintf(g_undo[g_nundo++], sizeof g_undo[0], "%s", now);
    snprintf(buf, sizeof buf, "redone: %d file%s back as the agent left them", *changed, *changed == 1 ? "" : "s");
    if (*changed) g_note = 2;
    return xdup(buf);
}
