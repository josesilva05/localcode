/* glm53f_tui.c - localcode's full-screen front end.
 *
 * The layout of the terminal coding tools it follows: a welcome box (who, which model on
 * this machine, the folder, tips, this folder's recent conversations), then the conversation,
 * which fills the screen and scrolls above an input line between two rules that never
 * leaves the bottom, with the model, the folder and the agent's mode under it. The whole screen is redrawn from a model of the conversation (blocks of
 * text) whenever something changes, inside the terminal's alternate screen.
 *
 * The model runs on a worker thread, so the screen stays alive while it reads or writes:
 * the conversation scrolls (wheel, PgUp/PgDn), the next message can be typed and is queued,
 * and Esc or Ctrl+C stops an answer. Input is read raw, as the VT sequences the terminal
 * sends (keys, SGR mouse reports, bracketed paste), on Windows and POSIX alike.
 */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <dirent.h>
#include <poll.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>
#endif

#include "glm53f_chat.h"

static Glm53fChat *g_s;   /* the session (set by glm53f_tui_run) */

/* Palette, 24-bit colour: localcode's green for what belongs to it (the welcome box, the
 * prompt, the marks of its answers and tools), neutral greys for the text */
#define C_RESET  "\033[0m"
#define C_BOLD   "\033[1m"
#define C_NOBOLD "\033[22m"
#define C_BRIGHT "\033[38;2;72;222;128m"    /* localcode green: accents, marks, bold text */
#define C_GREEN  "\033[38;2;46;160;94m"     /* its darker shade: borders, rules           */
#define C_DIM    "\033[38;2;138;145;140m"   /* hints, numbers, notes                      */
#define C_FAINT  "\033[38;2;98;104;100m"    /* reasoning text, rules                      */
#define C_TEXT   "\033[38;2;232;235;232m"   /* answer and input text                      */
#define C_CODE   "\033[38;2;136;206;230m"   /* inline code and code blocks                */
#define C_PANEL  "\033[48;2;40;43;41m"      /* the user's messages, the menu's choice     */
#define C_YELLOW "\033[38;2;232;196;90m"    /* the agent's mode, questions                */
#define C_RED    "\033[38;2;238;108;98m"    /* failures, removed lines                    */
#define C_ADD    "\033[38;2;110;214;140m"   /* added lines                                */

#define LOCALCODE_VERSION "0.4"

#define EDITOR_ROWS 8                        /* input rows shown before the box scrolls */

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void duration(char *buf, size_t cap, double sec)
{
    if (sec < 60.0)        snprintf(buf, cap, "%.1fs", sec);
    else if (sec < 3600.0) snprintf(buf, cap, "%dm %02ds", (int)(sec / 60.0), (int)sec % 60);
    else                   snprintf(buf, cap, "%dh %02dm", (int)(sec / 3600.0), (int)(sec / 60.0) % 60);
}

static int is_cont(char c) { return ((unsigned char)c & 0xC0) == 0x80; }

/* Columns of n bytes of UTF-8, one per code point (no wide glyphs are drawn). */
static int cols_of(const char *p, int n)
{
    int c = 0;
    for (int i = 0; i < n; i++) c += !is_cont(p[i]);
    return c;
}

/* Columns a string occupies on screen: code points, not counting ANSI sequences. */
static int vis_cols(const char *p)
{
    int c = 0;
    while (*p) {
        if (*p == 0x1b) {
            p++;
            if (*p == '[') {
                p++;
                while (*p && !((unsigned char)*p >= 0x40 && (unsigned char)*p <= 0x7e)) p++;
                if (*p) p++;
            }
            continue;
        }
        c += !is_cont(*p);
        p++;
    }
    return c;
}

/* ------------------------------------------------------------------ buffers ---- */

typedef struct { char *p; size_t n, cap; } Buf;

static void buf_add(Buf *b, const char *s, size_t n)
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

static void buf_str(Buf *b, const char *s) { buf_add(b, s, strlen(s)); }

static void buf_fmt(Buf *b, const char *fmt, ...)
{
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n > 0) buf_add(b, tmp, (size_t)(n < (int)sizeof tmp ? n : (int)sizeof tmp - 1));
}

static void spaces(Buf *b, int n)
{
    for (int i = 0; i < n; i++) buf_add(b, " ", 1);
}

/* ----------------------------------------------------------------- terminal ---- */

static struct {
    int w, h, forced;
#ifdef _WIN32
    HANDLE in, out;
    DWORD  in_mode, out_mode;
    int    in_console, out_console;
    wchar_t high;                            /* first half of a surrogate pair */
#else
    struct termios saved;
    int    raw;
#endif
} T;

static void term_size(void)
{
    int w = 0, h = 0;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (T.out_console && GetConsoleScreenBufferInfo(T.out, &info)) {
        w = info.srWindow.Right - info.srWindow.Left + 1;
        h = info.srWindow.Bottom - info.srWindow.Top + 1;
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) { w = ws.ws_col; h = ws.ws_row; }
#endif
    if (w <= 0) { const char *e = getenv("COLUMNS"); w = e ? atoi(e) : 100; }
    if (h <= 0) { const char *e = getenv("LINES"); h = e ? atoi(e) : 40; }
    T.w = w < 40 ? 40 : w;
    T.h = h < 12 ? 12 : h;
}

static void out_str(const char *s, size_t n)
{
    fwrite(s, 1, n, stdout);
    fflush(stdout);
}

static int term_open(int forced)
{
    T.forced = forced;
#ifdef _WIN32
    T.in = GetStdHandle(STD_INPUT_HANDLE);
    T.out = GetStdHandle(STD_OUTPUT_HANDLE);
    T.in_console = GetConsoleMode(T.in, &T.in_mode) != 0;
    T.out_console = GetConsoleMode(T.out, &T.out_mode) != 0;
    if (!forced && (!T.in_console || !T.out_console)) return -1;
    if (T.out_console &&
        !SetConsoleMode(T.out, T.out_mode | 0x0004 /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */))
        return -1;
    /* keys arrive as VT sequences, ctrl+c as a byte, no echo, no line editing, no
     * quick-edit selection stealing the mouse */
    if (T.in_console &&
        !SetConsoleMode(T.in, 0x0200 /* ENABLE_VIRTUAL_TERMINAL_INPUT */ | 0x0008 /* WINDOW_INPUT */ |
                              0x0080 /* ENABLE_EXTENDED_FLAGS */)) {
        SetConsoleMode(T.out, T.out_mode);
        return -1;
    }
#else
    const int in_tty = isatty(STDIN_FILENO), out_tty = isatty(STDOUT_FILENO);
    if (!forced && (!in_tty || !out_tty)) return -1;
    if (in_tty && tcgetattr(STDIN_FILENO, &T.saved) == 0) {
        struct termios raw = T.saved;
        raw.c_iflag &= ~(tcflag_t)(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        raw.c_lflag &= ~(tcflag_t)(ECHO | ICANON | IEXTEN | ISIG);
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == 0) T.raw = 1;
    }
#endif
    static const char enter[] = "\033[?1049h\033[?1000h\033[?1006h\033[?2004h\033[H\033[2J";
    out_str(enter, sizeof enter - 1);
    term_size();
    return 0;
}

static void term_close(void)
{
    static const char leave[] = "\033[?2004l\033[?1006l\033[?1000l\033[0m\033[?25h\033[?1049l";
    out_str(leave, sizeof leave - 1);
#ifdef _WIN32
    if (T.in_console) SetConsoleMode(T.in, T.in_mode);
    if (T.out_console) SetConsoleMode(T.out, T.out_mode);
#else
    if (T.raw) tcsetattr(STDIN_FILENO, TCSAFLUSH, &T.saved);
#endif
}

/* Up to cap bytes of input as UTF-8: >0 bytes, 0 nothing within timeout_ms, -1 end of input. */
static int in_read(char *buf, int cap, int timeout_ms)
{
#ifdef _WIN32
    if (T.in_console) {
        if (WaitForSingleObject(T.in, (DWORD)timeout_ms) != WAIT_OBJECT_0) return 0;
        INPUT_RECORD rec[64];
        DWORD n = 0, avail = 0;
        if (!GetNumberOfConsoleInputEvents(T.in, &avail) || !avail) return 0;
        if (!ReadConsoleInputW(T.in, rec, avail < 64 ? avail : 64, &n)) return -1;
        int got = 0;
        for (DWORD i = 0; i < n; i++) {
            if (rec[i].EventType != KEY_EVENT || !rec[i].Event.KeyEvent.bKeyDown) continue;
            const wchar_t ch = rec[i].Event.KeyEvent.uChar.UnicodeChar;
            if (!ch) continue;
            for (WORD r = 0; r < rec[i].Event.KeyEvent.wRepeatCount && got + 8 < cap; r++) {
                if (ch >= 0xD800 && ch <= 0xDBFF) { T.high = ch; continue; }
                wchar_t pair[2];
                int np = 0;
                if (ch >= 0xDC00 && ch <= 0xDFFF && T.high) pair[np++] = T.high;
                pair[np++] = ch;
                T.high = 0;
                got += WideCharToMultiByte(CP_UTF8, 0, pair, np, buf + got, cap - got, NULL, NULL);
            }
        }
        return got;
    }
    {   /* a pipe or a file (a captured session) */
        DWORD avail = 0, rd = 0;
        if (GetFileType(T.in) == FILE_TYPE_PIPE) {
            if (!PeekNamedPipe(T.in, NULL, 0, NULL, &avail, NULL)) return -1;
            if (!avail) { Sleep((DWORD)timeout_ms); return 0; }
            if (avail > (DWORD)cap) avail = (DWORD)cap;
        } else {
            avail = (DWORD)cap;
        }
        if (!ReadFile(T.in, buf, avail, &rd, NULL) || rd == 0) return -1;
        return (int)rd;
    }
#else
    struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 };
    const int r = poll(&pfd, 1, timeout_ms);
    if (r <= 0) return 0;
    const ssize_t n = read(STDIN_FILENO, buf, (size_t)cap);
    return n > 0 ? (int)n : -1;
#endif
}

/* ------------------------------------------------------------------- blocks ---- */
/* The conversation as the screen shows it. The worker thread appends, the UI thread draws;
 * both under g_mu. */

enum { B_SPLASH, B_USER, B_STATUS, B_REASON, B_THOUGHT, B_ANSWER, B_STATS, B_NOTE, B_ERROR, B_TOOL, B_ASK };

typedef struct {
    int    kind;
    Buf    text;
    double t0;
} Block;

static pthread_mutex_t g_mu;
static Block *g_blocks;
static int    g_nblocks, g_capblocks;
static int    g_dirty = 1;

static Block *block_new(int kind)
{
    if (g_nblocks == g_capblocks) {
        const int cap = g_capblocks ? g_capblocks * 2 : 32;
        Block *g = (Block *)realloc(g_blocks, (size_t)cap * sizeof *g);
        if (!g) return NULL;
        g_blocks = g;
        g_capblocks = cap;
    }
    Block *b = &g_blocks[g_nblocks++];
    memset(b, 0, sizeof *b);
    b->kind = kind;
    g_dirty = 1;
    return b;
}

static void block_add(int kind, const char *text)
{
    Block *b = block_new(kind);
    if (b && text) buf_str(&b->text, text);
}

static void blocks_clear(void)
{
    for (int i = 0; i < g_nblocks; i++) free(g_blocks[i].text.p);
    g_nblocks = 0;
    g_dirty = 1;
}

static Block *last_block(int kind)
{
    return g_nblocks && g_blocks[g_nblocks - 1].kind == kind ? &g_blocks[g_nblocks - 1] : NULL;
}

/* ----------------------------------------------------------------- wrapping ---- */
/* Lines of the screen: ANSI text and the columns it occupies. Text is wrapped at word
 * boundaries; every line starts with the prefix, and the style in force is reapplied. */

typedef struct { char *s; } Line;
typedef struct { Line *v; int n, cap; } Lines;

static void lines_push(Lines *L, Buf *b)
{
    if (L->n == L->cap) {
        const int cap = L->cap ? L->cap * 2 : 256;
        Line *g = (Line *)realloc(L->v, (size_t)cap * sizeof *g);
        if (!g) return;
        L->v = g;
        L->cap = cap;
    }
    L->v[L->n++].s = b->p ? b->p : (char *)calloc(1, 1);
    memset(b, 0, sizeof *b);
}

static void lines_free(Lines *L)
{
    for (int i = 0; i < L->n; i++) free(L->v[i].s);
    free(L->v);
    memset(L, 0, sizeof *L);
}

typedef struct {
    Lines      *out;
    Buf         cur;
    const char *prefix;          /* ANSI + text at the start of each line      */
    int         prefix_cols;
    const char *hang;            /* prefix of continuation lines (NULL: prefix) */
    int         hang_cols;
    int         width;           /* last usable column                          */
    int         pad_to;          /* pad with spaces up to this column (bubbles) */
    int         col, content;
    char        style[96];
} Wrap;

static void wr_begin(Wrap *w, int first)
{
    const char *p = (!first && w->hang) ? w->hang : w->prefix;
    buf_str(&w->cur, p);
    w->col = (!first && w->hang) ? w->hang_cols : w->prefix_cols;
    buf_str(&w->cur, w->style);
    w->content = 0;
}

static void wr_flush(Wrap *w)
{
    if (w->pad_to > w->col) spaces(&w->cur, w->pad_to - w->col);
    buf_str(&w->cur, C_RESET);
    lines_push(w->out, &w->cur);
    wr_begin(w, 0);
}

static void wr_style(Wrap *w, const char *ansi)
{
    snprintf(w->style, sizeof w->style, "%s", ansi);
    buf_str(&w->cur, ansi);
}

static void wr_text(Wrap *w, const char *p, int n)
{
    int i = 0;
    while (i < n) {
        if (p[i] == '\n') { wr_flush(w); i++; continue; }
        if (p[i] == ' ' || p[i] == '\t') {
            if (w->content && w->col < w->width) { buf_add(&w->cur, " ", 1); w->col++; }
            i++;
            continue;
        }
        int j = i;
        while (j < n && p[j] != ' ' && p[j] != '\t' && p[j] != '\n') j++;
        int wc = cols_of(p + i, j - i);
        const int room = w->width - (w->hang ? w->hang_cols : w->prefix_cols);
        if (w->col + wc > w->width && w->content) wr_flush(w);
        if (wc > room) {                         /* longer than a line: cut by code points */
            int k = i;
            while (k < j) {
                int step = 1;
                while (k + step < j && is_cont(p[k + step])) step++;
                if (w->col + 1 > w->width) wr_flush(w);
                buf_add(&w->cur, p + k, (size_t)step);
                w->col++;
                w->content = 1;
                k += step;
            }
        } else {
            buf_add(&w->cur, p + i, (size_t)(j - i));
            w->col += wc;
            w->content = 1;
        }
        i = j;
    }
}

static void wr_end(Wrap *w)
{
    if (w->content) {
        if (w->pad_to > w->col) spaces(&w->cur, w->pad_to - w->col);
        buf_str(&w->cur, C_RESET);
        lines_push(w->out, &w->cur);
    } else {
        free(w->cur.p);
        memset(&w->cur, 0, sizeof w->cur);
    }
}

static void push_text(Lines *L, const char *ansi)
{
    Buf b = { 0 };
    buf_str(&b, ansi);
    buf_str(&b, C_RESET);
    lines_push(L, &b);
}

/* Inline Markdown: **bold** and `code` inside one line of text. */
static void wr_inline(Wrap *w, const char *p, int n, const char *base)
{
    int bold = 0, code = 0, i = 0, start = 0;
    while (i < n) {
        if (!code && p[i] == '*' && i + 1 < n && p[i + 1] == '*') {
            wr_text(w, p + start, i - start);
            bold = !bold;
            char st[96];
            snprintf(st, sizeof st, "%s%s", bold ? C_BOLD : C_NOBOLD, bold ? C_BRIGHT : base);
            wr_style(w, st);
            i += 2;
            start = i;
            continue;
        }
        if (p[i] == '`') {
            wr_text(w, p + start, i - start);
            code = !code;
            wr_style(w, code ? C_CODE : (bold ? C_BRIGHT : base));
            i++;
            start = i;
            continue;
        }
        i++;
    }
    wr_text(w, p + start, n - start);
    char st[96];
    snprintf(st, sizeof st, "%s%s", C_NOBOLD, base);
    wr_style(w, st);
}

static void render_markdown(Lines *L, const char *text, size_t len, int W)
{
    int fence = 0;
    size_t i = 0;
    while (i <= len) {
        size_t j = i;
        while (j < len && text[j] != '\n') j++;
        const char *line = text + i;
        int n = (int)(j - i);
        if (n && line[n - 1] == '\r') n--;
        int ls = 0;
        while (ls < n && line[ls] == ' ') ls++;
        const char *body = line + ls;
        int bn = n - ls;

        Wrap w = { 0 };
        w.out = L;
        w.width = W - 2;
        char prefix[64], hang[64];
        if (bn >= 3 && !strncmp(body, "```", 3)) {
            fence = !fence;
        } else if (fence) {
            Buf b = { 0 };                           /* code keeps its spaces and is not wrapped */
            buf_str(&b, "    " C_CODE);
            buf_add(&b, line, (size_t)n);
            buf_str(&b, C_RESET);
            lines_push(L, &b);
        } else if (bn == 0) {
            if (i < len) push_text(L, "");
        } else if (body[0] == '#') {
            int h = 0;
            while (h < bn && body[h] == '#') h++;
            int k = h;
            while (k < bn && body[k] == ' ') k++;
            w.prefix = "  "; w.prefix_cols = 2;
            snprintf(w.style, sizeof w.style, "%s%s", C_BOLD, C_BRIGHT);
            wr_begin(&w, 1);
            wr_text(&w, body + k, bn - k);
            wr_end(&w);
        } else if ((body[0] == '-' || body[0] == '*' || body[0] == '+') && bn > 1 && body[1] == ' ') {
            snprintf(prefix, sizeof prefix, "%*s%s•%s ", 2 + ls, "", C_BRIGHT, C_TEXT);
            snprintf(hang, sizeof hang, "%*s", 4 + ls, "");
            w.prefix = prefix; w.prefix_cols = 4 + ls;
            w.hang = hang; w.hang_cols = 4 + ls;
            snprintf(w.style, sizeof w.style, "%s", C_TEXT);
            wr_begin(&w, 1);
            wr_inline(&w, body + 2, bn - 2, C_TEXT);
            wr_end(&w);
        } else if (bn > 2 && body[0] == '>' ) {
            snprintf(prefix, sizeof prefix, "  %s│ ", C_DIM);
            w.prefix = prefix; w.prefix_cols = 4;
            snprintf(w.style, sizeof w.style, "%s", C_DIM);
            wr_begin(&w, 1);
            wr_inline(&w, body + 1, bn - 1, C_DIM);
            wr_end(&w);
        } else {
            int num = 0;
            while (num < bn && body[num] >= '0' && body[num] <= '9') num++;
            const int numbered = num > 0 && num + 1 < bn && body[num] == '.' && body[num + 1] == ' ';
            if (numbered) {
                snprintf(prefix, sizeof prefix, "%*s%s%.*s %s", 2 + ls, "", C_BRIGHT, num + 1, body, C_TEXT);
                snprintf(hang, sizeof hang, "%*s", 2 + ls + num + 2, "");
                w.prefix = prefix; w.prefix_cols = 2 + ls + num + 2;
                w.hang = hang; w.hang_cols = 2 + ls + num + 2;
                snprintf(w.style, sizeof w.style, "%s", C_TEXT);
                wr_begin(&w, 1);
                wr_inline(&w, body + num + 2, bn - num - 2, C_TEXT);
            } else {
                snprintf(prefix, sizeof prefix, "%*s", 2 + ls, "");
                w.prefix = prefix; w.prefix_cols = 2 + ls;
                snprintf(w.style, sizeof w.style, "%s", C_TEXT);
                wr_begin(&w, 1);
                wr_inline(&w, body, bn, C_TEXT);
            }
            wr_end(&w);
        }
        i = j + 1;
    }
}

/* ------------------------------------------------------------------- render ---- */

/* s cut to cols columns with "…", or padded with spaces to them */
static void put_cols(Buf *b, const char *s, int cols)
{
    if (cols <= 0) return;
    const int have = cols_of(s, (int)strlen(s));
    if (have <= cols) { buf_str(b, s); spaces(b, cols - have); return; }
    int i = 0, c = 0;
    while (s[i] && c < cols - 1) { i++; while (s[i] && is_cont(s[i])) i++; c++; }
    buf_add(b, s, (size_t)i);
    buf_str(b, "…");
}

static void rule(Buf *b, int n) { for (int i = 0; i < n; i++) buf_str(b, "─"); }

/* The user's first name: LOCALCODE_USER, else the account's full name, else its login. */
static const char *who(void)
{
    static char name[128];
    if (name[0]) return name;
    const char *e = getenv("LOCALCODE_USER");
    if (e && *e) snprintf(name, sizeof name, "%s", e);
#ifdef _WIN32
    if (!name[0]) {                              /* NetUserGetInfo, loaded when needed */
        typedef DWORD (WINAPI *GetInfo)(LPCWSTR, LPCWSTR, DWORD, LPBYTE *);
        typedef DWORD (WINAPI *FreeBuf)(LPVOID);
        HMODULE h = LoadLibraryA("netapi32.dll");
        wchar_t user[257];
        DWORD un = 257;
        if (h && GetUserNameW(user, &un)) {
            GetInfo gi = (GetInfo)(void (*)(void))GetProcAddress(h, "NetUserGetInfo");
            FreeBuf fb = (FreeBuf)(void (*)(void))GetProcAddress(h, "NetApiBufferFree");
            LPBYTE buf = NULL;
            /* USER_INFO_2: usri2_full_name follows the name, password, password age,
             * privilege, home dir, comment, flags and script path (lm.h) */
            if (gi && fb && gi(NULL, user, 2, &buf) == 0 && buf) {
                typedef struct { LPWSTR name, password; DWORD age, priv; LPWSTR home, comment; DWORD flags;
                                 LPWSTR script; DWORD auth; LPWSTR full; } Info2;
                const wchar_t *full = ((Info2 *)buf)->full;
                if (full && *full) WideCharToMultiByte(CP_UTF8, 0, full, -1, name, (int)sizeof name, NULL, NULL);
                fb(buf);
            }
        }
    }
#endif
    if (!name[0]) {
        e = getenv("USERNAME");
        if (!e) e = getenv("USER");
        snprintf(name, sizeof name, "%s", e && *e ? e : "");
    }
    char *sp = strchr(name, ' ');
    if (sp) *sp = 0;
    return name;
}

/* The working folder, with the home folder as ~ */
static void short_cwd(char *out, size_t cap)
{
    char cwd[1024] = "";
#ifdef _WIN32
    if (!_getcwd(cwd, (int)sizeof cwd)) cwd[0] = 0;
    const char *home = getenv("USERPROFILE");
#else
    if (!getcwd(cwd, sizeof cwd)) cwd[0] = 0;
    const char *home = getenv("HOME");
#endif
    const size_t hl = home ? strlen(home) : 0;
    if (hl && !strncmp(cwd, home, hl) && (cwd[hl] == 0 || cwd[hl] == '\\' || cwd[hl] == '/'))
        snprintf(out, cap, "~%s", cwd + hl);
    else
        snprintf(out, cap, "%s", cwd);
}

/* The git branch of the working folder (its .git/HEAD, or a parent's), "" outside a
 * repository; read again every 2 seconds. */
static void git_branch(char *out, size_t cap)
{
    static char cached[96];
    static double when = -10.0;
    const double t = now_s();
    if (t - when > 2.0) {
        when = t;
        cached[0] = 0;
        char dir[1024] = "";
#ifdef _WIN32
        if (!_getcwd(dir, (int)sizeof dir)) dir[0] = 0;
#else
        if (!getcwd(dir, sizeof dir)) dir[0] = 0;
#endif
        for (int up = 0; up < 32 && dir[0]; up++) {
            char p[1100], head[256];
            snprintf(p, sizeof p, "%s/.git/HEAD", dir);
            FILE *fh = fopen(p, "rb");
            if (fh) {
                const size_t n = fread(head, 1, sizeof head - 1, fh);
                fclose(fh);
                head[n] = 0;
                char *nl = strpbrk(head, "\r\n");
                if (nl) *nl = 0;
                if (!strncmp(head, "ref: refs/heads/", 16)) snprintf(cached, sizeof cached, "%s", head + 16);
                else if (n >= 7) snprintf(cached, sizeof cached, "%.7s", head);   /* detached */
                break;
            }
            char *cut = strrchr(dir, '\\');
            char *cut2 = strrchr(dir, '/');
            if (cut2 > cut) cut = cut2;
            if (!cut || cut == dir || (cut == dir + 2 && dir[1] == ':')) break;
            *cut = 0;
        }
    }
    snprintf(out, cap, "%s", cached);
}

/* 1994 → "2.0k", 131072 → "131k" */
static void fmt_k(char *out, size_t cap, int n)
{
    if (n < 1000)        snprintf(out, cap, "%d", n);
    else if (n < 100000) snprintf(out, cap, "%.1fk", n / 1000.0);
    else                 snprintf(out, cap, "%dk", (n + 500) / 1000);
}

/* The window's title: "localcode · folder", then the conversation's first message. */
static void set_title(const char *what)
{
    char cwd[512], t[600];
    short_cwd(cwd, sizeof cwd);
    const char *base = cwd + strlen(cwd);
    while (base > cwd && base[-1] != '\\' && base[-1] != '/') base--;
    if (what && *what) {
        char m[200];
        snprintf(m, sizeof m, "%.48s%s", what, strlen(what) > 48 ? "…" : "");
        for (char *c = m; *c; c++) if (*c == '\n' || *c == '\r' || *c == '\033' || *c == '\007') *c = ' ';
        snprintf(t, sizeof t, "\033]0;localcode | %s\007", m);
    } else {
        snprintf(t, sizeof t, "\033]0;localcode · %s\007", *base ? base : cwd);
    }
    out_str(t, strlen(t));
}

static char *g_recent;                               /* this folder's latest conversations */

/* localcode's mascot: the computer it runs on, 12 by 8 pixels in half blocks */
static const char *MASCOT[8] = {
    ".##########.",
    ".##########.",
    ".###.##.###.",
    ".##########.",
    ".###....###.",
    ".##########.",
    ".....##.....",
    "...######...",
};

static void mascot_row(char *out, size_t cap, int row)
{
    size_t n = 0;
    out[0] = 0;
    for (int x = 0; x < 12 && n + 4 < cap; x++) {
        const int top = MASCOT[2 * row][x] == '#', bot = MASCOT[2 * row + 1][x] == '#';
        n += (size_t)snprintf(out + n, cap - n, "%s", top && bot ? "█" : top ? "▀" : bot ? "▄" : " ");
    }
}

typedef struct { const char *style; char text[256]; int centre, divider, tail; } Cell;

static void cell_put(Buf *b, const Cell *c, int width)
{
    if (c->divider) { buf_str(b, C_GREEN); rule(b, width); buf_str(b, C_RESET); return; }
    const int have = cols_of(c->text, (int)strlen(c->text));
    if (c->tail && have > width && width > 1) {      /* a path: its end matters, "…" before */
        const char *p = c->text;
        for (int skip = have - (width - 1); skip > 0 && *p; skip--) { p++; while (*p && is_cont(*p)) p++; }
        buf_str(b, c->style ? c->style : "");
        buf_str(b, "…");
        buf_str(b, p);
        buf_str(b, C_RESET);
        return;
    }
    const int pad = c->centre && have < width ? (width - have) / 2 : 0;
    spaces(b, pad);
    buf_str(b, c->style ? c->style : "");
    put_cols(b, c->text, width - pad);
    buf_str(b, C_RESET);
}

/* The welcome box: who and where on the left, tips and recent conversations on the right
 * (one column under the other in a narrow window), then the line about staying local. */
static void render_splash(Lines *L, const Glm53fChat *s, int W)
{
    Cell left[16], right[16];
    int nl = 0, nr = 0;
    memset(left, 0, sizeof left);
    memset(right, 0, sizeof right);
#define ADD(arr, n, st, ctr, ...) do { arr[n].style = st; arr[n].centre = ctr; \
                                       snprintf(arr[n].text, sizeof arr[n].text, __VA_ARGS__); n++; } while (0)
    ADD(left, nl, "", 1, "%s", "");
    if (*who()) ADD(left, nl, C_BOLD C_TEXT, 1, "Bem-vindo de volta, %s!", who());
    else        ADD(left, nl, C_BOLD C_TEXT, 1, "Bem-vindo ao localcode!");
    ADD(left, nl, "", 1, "%s", "");
    for (int r = 0; r < 4; r++) { char m[64]; mascot_row(m, sizeof m, r); ADD(left, nl, C_BRIGHT, 1, "%s", m); }
    ADD(left, nl, "", 1, "%s", "");
    static const char *lv[] = { "chat", "nível 1 · rápido", "nível 2 · agente" };
    ADD(left, nl, C_DIM, 1, "%s · %s", s->e->name, lv[s->agent_level > 0 && s->e->agent ? s->agent_level : 0]);
    {
        char cap[24];
        fmt_k(cap, sizeof cap, s->e->cap);
        ADD(left, nl, C_DIM, 1, "contexto de %s tokens · pronto em %.0f s", cap, s->load_s);
    }
    {
        char cwd[512];
        short_cwd(cwd, sizeof cwd);
        ADD(left, nl, C_DIM, 1, "%s", cwd);
        left[nl - 1].tail = 1;
    }

    ADD(right, nr, C_BOLD C_BRIGHT, 0, "Primeiros passos");
    ADD(right, nr, C_TEXT, 0, "Peça uma mudança: o agente lê, edita e testa");
    ADD(right, nr, C_TEXT, 0, "/level 0 chat · 1 rápido · 2 agente");
    ADD(right, nr, C_TEXT, 0, "/resume volta a uma conversa · /compact resume");
    if (s->e->n_variants > 1) ADD(right, nr, C_TEXT, 0, "/model troca o modelo (%d nesta máquina)", s->e->n_variants);
    right[nr++].divider = 1;
    ADD(right, nr, C_BOLD C_BRIGHT, 0, "Conversas recentes nesta pasta");
    int nrec = 0;
    for (const char *p = g_recent; p && *p && nrec < 3; nrec++) {
        const char *nl2 = strchr(p, '\n');
        const int len = nl2 ? (int)(nl2 - p) : (int)strlen(p);
        ADD(right, nr, C_TEXT, 0, "%.*s", len, p);
        p = nl2 ? nl2 + 1 : NULL;
    }
    if (!nrec) ADD(right, nr, C_DIM, 0, "nenhuma ainda; a primeira começa aqui");
    else       ADD(right, nr, C_DIM, 0, "/resume N para voltar · /sessions lista todas");
#undef ADD

    /* the frame */
    const int two = W >= 76;
    const int inner = W - 5;                            /* " │ " ... " │" */
    int lw = two ? (W - 8) * 44 / 100 : inner, rw = two ? W - 8 - lw : 0;
    if (lw < 1) lw = 1;
    push_text(L, "");
    {
        Buf b = { 0 };
        buf_str(&b, " " C_GREEN "╭─── " C_BOLD C_BRIGHT "localcode" C_NOBOLD C_DIM " v" LOCALCODE_VERSION " · local " C_GREEN);
        const int used = 6 + 9 + (int)strlen(" v" LOCALCODE_VERSION " · local ") - 1;
        rule(&b, W - 2 - used);
        buf_str(&b, "╮" C_RESET);
        lines_push(L, &b);
    }
    if (two) {
        const int n = nl > nr ? nl : nr;
        for (int i = 0; i < n; i++) {
            Buf b = { 0 };
            buf_str(&b, " " C_GREEN "│" C_RESET " ");
            if (i < nl) cell_put(&b, &left[i], lw); else spaces(&b, lw);
            buf_str(&b, " " C_GREEN "│" C_RESET " ");
            if (i < nr) cell_put(&b, &right[i], rw - 1); else spaces(&b, rw - 1);
            buf_str(&b, " " C_GREEN "│" C_RESET);
            lines_push(L, &b);
        }
    } else {
        for (int i = 0; i < nl + nr + 1; i++) {
            Buf b = { 0 };
            buf_str(&b, " " C_GREEN "│" C_RESET " ");
            if (i < nl) cell_put(&b, &left[i], inner - 1);
            else if (i == nl) spaces(&b, inner - 1);
            else cell_put(&b, &right[i - nl - 1], inner - 1);
            buf_str(&b, " " C_GREEN "│" C_RESET);
            lines_push(L, &b);
        }
    }
    {
        Buf b = { 0 };
        buf_str(&b, " " C_GREEN "╰");
        rule(&b, W - 4);
        buf_str(&b, "╯" C_RESET);
        lines_push(L, &b);
    }
    push_text(L, "");
    push_text(L, "  " C_GREEN "▎ " C_TEXT "Tudo roda nesta máquina: " C_BRIGHT "o modelo, o contexto e as conversas" C_TEXT
                 " ficam aqui." C_DIM " Nada vai para a nuvem.");
}

static void render_bubble(Lines *L, const char *text, size_t len, int W)
{
    char prefix[96], hang[64];
    snprintf(prefix, sizeof prefix, "%s %s❯ %s", C_PANEL, C_DIM, C_TEXT);
    snprintf(hang, sizeof hang, "%s   ", C_PANEL);
    Wrap w = { 0 };
    w.out = L;
    w.prefix = prefix; w.prefix_cols = 3;
    w.hang = hang; w.hang_cols = 3;
    w.width = W - 2;
    w.pad_to = W - 1;
    snprintf(w.style, sizeof w.style, "%s", C_TEXT);
    wr_begin(&w, 1);
    wr_text(&w, text, (int)len);
    w.content = 1;
    wr_end(&w);
}

/* The first line of what was just rendered from index from on gets the mark (an answer's
 * green dot) in place of its two leading spaces. */
static void mark_first(Lines *L, int from, const char *mark)
{
    for (int i = from; i < L->n; i++) {
        char *s = L->v[i].s;
        if (strncmp(s, "  ", 2) || !s[2] || s[2] == '\033' && !strncmp(s + 2, C_RESET, strlen(C_RESET))) continue;
        Buf b = { 0 };
        buf_str(&b, mark);
        buf_str(&b, C_RESET " ");
        buf_str(&b, s + 2);
        free(s);
        L->v[i].s = b.p;
        return;
    }
}

/* The question before a write or a command: a frame with the change or the command, the
 * removed lines red and the added green, and the keys under it. */
static void render_ask(Lines *L, const char *text, size_t len, int W)
{
    const char *nl = memchr(text, '\n', len);
    const size_t tl = nl ? (size_t)(nl - text) : len;
    const int inner = W - 5;
    Buf b = { 0 };
    buf_str(&b, " " C_YELLOW "╭");
    rule(&b, W - 4);
    buf_str(&b, "╮" C_RESET);
    lines_push(L, &b);
    char title[600];
    snprintf(title, sizeof title, "%.*s?", (int)(tl < 500 ? tl : 500), text);
    if (title[0] >= 'a' && title[0] <= 'z') title[0] = (char)(title[0] - 32);
    buf_str(&b, " " C_YELLOW "│ " C_BOLD C_TEXT);
    put_cols(&b, title, inner - 1);
    buf_str(&b, C_RESET " " C_YELLOW "│" C_RESET);
    lines_push(L, &b);
    if (nl) {
        buf_str(&b, " " C_YELLOW "│" C_RESET);
        spaces(&b, inner + 1);
        buf_str(&b, C_YELLOW "│" C_RESET);
        lines_push(L, &b);
        int shown = 0;
        for (const char *p = nl + 1, *end = text + len; p < end && shown < 14; shown++) {
            const char *q = memchr(p, '\n', (size_t)(end - p));
            const size_t n = q ? (size_t)(q - p) : (size_t)(end - p);
            char line[1024];
            snprintf(line, sizeof line, "%.*s", (int)(n < 1000 ? n : 1000), p);
            for (char *c = line; *c; c++) if (*c == '\t' || *c == '\r') *c = ' ';
            const char *st = line[0] == '-' && line[1] == ' ' ? C_RED : line[0] == '+' && line[1] == ' ' ? C_ADD : C_CODE;
            buf_str(&b, " " C_YELLOW "│ " C_RESET "  ");
            buf_str(&b, st);
            put_cols(&b, line, inner - 3);
            buf_str(&b, C_RESET " " C_YELLOW "│" C_RESET);
            lines_push(L, &b);
            p = q ? q + 1 : end;
            if (shown == 13 && p < end) {
                buf_str(&b, " " C_YELLOW "│ " C_DIM "  ");
                put_cols(&b, "…", inner - 3);
                buf_str(&b, C_RESET " " C_YELLOW "│" C_RESET);
                lines_push(L, &b);
            }
        }
    }
    buf_str(&b, " " C_YELLOW "│" C_RESET);
    spaces(&b, inner + 1);
    buf_str(&b, C_YELLOW "│" C_RESET);
    lines_push(L, &b);
    {
        const char *keys = C_BRIGHT "❯ s" C_TEXT " sim   " C_BRIGHT "a" C_TEXT " sim, sempre para esta ferramenta   "
                           C_BRIGHT "n" C_TEXT " não, faça de outro jeito";
        const int kc = vis_cols(keys);
        buf_str(&b, " " C_YELLOW "│ " C_RESET);
        buf_str(&b, keys);
        spaces(&b, inner - 1 - kc > 0 ? inner - 1 - kc : 0);
        buf_str(&b, C_RESET " " C_YELLOW "│" C_RESET);
        lines_push(L, &b);
    }
    buf_str(&b, " " C_YELLOW "╰");
    rule(&b, W - 4);
    buf_str(&b, "╯" C_RESET);
    lines_push(L, &b);
}

static void render_plain(Lines *L, const char *text, size_t len, int W, const char *style, const char *prefix)
{
    Wrap w = { 0 };
    w.out = L;
    w.prefix = prefix; w.prefix_cols = vis_cols(prefix);
    if (w.prefix_cols > 4) { w.hang = "    "; w.hang_cols = 4; }   /* long prefixes: wrap under the text */
    w.width = W - 2;
    snprintf(w.style, sizeof w.style, "%s", style);
    wr_begin(&w, 1);
    wr_text(&w, text, (int)len);
    wr_end(&w);
}

static void render_blocks(Lines *L, const Glm53fChat *s, int W)
{
    static const char *spin[] = { "·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢" };
    for (int i = 0; i < g_nblocks; i++) {
        const Block *b = &g_blocks[i];
        const char *t = b->text.p ? b->text.p : "";
        if (b->kind == B_ANSWER || b->kind == B_REASON) {   /* nothing to show: no gap either */
            const char *q = t;
            while (*q == ' ' || *q == '\n' || *q == '\r' || *q == '\t') q++;
            if (!*q) continue;
        }
        /* consecutive tools stay together */
        if (i > 0 && !(b->kind == B_TOOL && g_blocks[i - 1].kind == B_TOOL)) push_text(L, "");
        switch (b->kind) {
        case B_SPLASH: render_splash(L, s, W); break;
        case B_USER:   render_bubble(L, t, b->text.n, W); break;
        case B_STATUS: {
            char d[32], line[512];
            duration(d, sizeof d, now_s() - b->t0);
            snprintf(line, sizeof line, "  %s%s %s%s… %s(%s · esc para interromper)", C_BRIGHT,
                     spin[(int)(now_s() * 8.0) % 10], C_BRIGHT, t, C_DIM, d);
            push_text(L, line);
            break;
        }
        case B_REASON:  render_plain(L, t, b->text.n, W, C_FAINT, "  "); break;
        case B_THOUGHT: {
            char line[256];
            snprintf(line, sizeof line, "  %s✻ %s%s", C_DIM, C_DIM, t);
            push_text(L, line);
            break;
        }
        case B_ANSWER: {
            const int from = L->n;
            render_markdown(L, t, b->text.n, W);
            mark_first(L, from, C_BRIGHT "●");
            break;
        }
        case B_STATS:
            render_plain(L, t, b->text.n, W, C_FAINT, "  " C_FAINT "└ ");
            break;
        case B_NOTE:  render_plain(L, t, b->text.n, W, C_DIM, "  " C_FAINT "⎿  " C_DIM); break;
        case B_TOOL: {                           /* t0: 1 the call worked, 0 it failed; title \n summary */
            const char *nl = strchr(t, '\n');
            const int tl = nl ? (int)(nl - t) : (int)strlen(t);
            char title[512];
            snprintf(title, sizeof title, "%.*s", tl, t);
            Buf h = { 0 };
            buf_str(&h, b->t0 > 0.5 ? C_BRIGHT "●" : C_RED "●");
            buf_str(&h, C_RESET " " C_BOLD C_TEXT);
            put_cols(&h, title, W - 5);
            buf_str(&h, C_RESET);
            lines_push(L, &h);
            if (nl && nl[1]) render_plain(L, nl + 1, strlen(nl + 1), W, b->t0 > 0.5 ? C_DIM : C_RED, "  " C_FAINT "⎿  ");
            break;
        }
        case B_ASK: render_ask(L, t, b->text.n, W); break;
        case B_ERROR: {
            render_plain(L, t, b->text.n, W, C_TEXT, "  " C_RED "✗ ");
            break;
        }
        default: break;
        }
    }
}

/* ------------------------------------------------------------------- editor ---- */

static struct { Buf b; int cur; } E;                 /* the text being typed, cursor in bytes */
static char *g_hist[64];
static int   g_nhist, g_hpos;

static void menu_touch(void);                        /* an edit reopens the menu */

static void ed_set(const char *p)
{
    menu_touch();
    E.b.n = 0;
    if (E.b.p) E.b.p[0] = 0;
    buf_str(&E.b, p);
    E.cur = (int)E.b.n;
}

static void ed_insert(const char *p, int n)
{
    menu_touch();
    buf_add(&E.b, "", 0);
    Buf nb = { 0 };
    buf_add(&nb, E.b.p ? E.b.p : "", (size_t)E.cur);
    buf_add(&nb, p, (size_t)n);
    buf_add(&nb, E.b.p ? E.b.p + E.cur : "", E.b.n - (size_t)E.cur);
    free(E.b.p);
    E.b = nb;
    E.cur += n;
}

static void ed_erase(int from, int to)
{
    if (from < 0 || to > (int)E.b.n || from >= to) return;
    menu_touch();
    memmove(E.b.p + from, E.b.p + to, E.b.n - (size_t)to + 1);
    E.b.n -= (size_t)(to - from);
    E.cur = from;
}

static int ed_prev(int i) { if (i <= 0) return 0; i--; while (i > 0 && is_cont(E.b.p[i])) i--; return i; }
static int ed_next(int i) { if (i >= (int)E.b.n) return (int)E.b.n; i++; while (i < (int)E.b.n && is_cont(E.b.p[i])) i++; return i; }

/* --------------------------------------------------------------------- menu ---- */
/* Typing "/" opens a menu above the input box, built from what has been typed: the
 * commands (by prefix, then by the letters in order), then a command's values - Reasoning
 * Effort, answer lengths with the time they would take at the speed this session measured,
 * the files of a directory with the tokens they would take out of the context left. */

#define MENU_MAX  64
#define MENU_ROWS 8

typedef struct {
    char tab[1100];              /* the editor text choosing it leaves               */
    char label[160], hint[200], value[48];
    int  submits;                /* enter sends `tab`; otherwise enter only fills it */
} MenuItem;

static struct {
    MenuItem it[MENU_MAX];
    int      n, sel, closed;
    int      raw_enter;          /* enter sends what was typed (/gen 300)            */
    char     title[160], right[96];
    char    *key;                /* the editor text the items were built for         */
    double   built;
} M;

static void menu_touch(void) { M.closed = 0; M.sel = 0; }

static MenuItem *menu_add(void)
{
    if (M.n == MENU_MAX) return NULL;
    MenuItem *it = &M.it[M.n++];
    memset(it, 0, sizeof *it);
    return it;
}

static int lower(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

static int starts_with_ci(const char *s, const char *pre)
{
    while (*pre) if (lower((unsigned char)*s++) != lower((unsigned char)*pre++)) return 0;
    return 1;
}

/* the letters of `typed` appear in `name` in order */
static int subsequence(const char *name, const char *typed)
{
    for (; *typed; typed++) {
        while (*name && lower((unsigned char)*name) != lower((unsigned char)*typed)) name++;
        if (!*name) return 0;
        name++;
    }
    return 1;
}

static void menu_commands(const char *typed)
{
    snprintf(M.title, sizeof M.title, "commands");
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < glm53f_ncommands; i++) {
            const Glm53fCommand *c = &glm53f_commands[i];
            const int pre = starts_with_ci(c->name, typed);
            if (pass == 0 ? !pre : (pre || !subsequence(c->name + 1, typed + 1))) continue;
            MenuItem *it = menu_add();
            if (!it) return;
            snprintf(it->tab, sizeof it->tab, "%s%s", c->name, c->arg ? " " : "");
            snprintf(it->label, sizeof it->label, "%s%s%s", c->name, *c->args ? " " : "", c->args);
            snprintf(it->hint, sizeof it->hint, "%s", c->help);
            glm53f_chat_command_value(g_s, c, it->value, sizeof it->value);
            it->submits = c->arg != 2;
        }
}

static void menu_reasoning(const char *typed)
{
    const Glm53fEngine *e = g_s->e;
    snprintf(M.title, sizeof M.title, "Reasoning effort (applies when a conversation opens)");
    for (int i = 0; i < e->n_levels; i++) {
        const char *name = e->levels[i].name;
        if (!starts_with_ci(name, typed)) continue;
        MenuItem *it = menu_add();
        snprintf(it->tab, sizeof it->tab, "/reasoning %s", name);
        snprintf(it->label, sizeof it->label, "%s", name);
        snprintf(it->hint, sizeof it->hint, "%s", e->levels[i].help);
        if (g_s->reasoning == name) snprintf(it->value, sizeof it->value, "now");
        it->submits = 1;
    }
}

/* /model: the models on this machine; enter opens the chosen one */
static void menu_models(const char *typed)
{
    const Glm53fEngine *e = g_s->e;
    snprintf(M.title, sizeof M.title, "Models on this machine (enter switches; the conversation goes on in it)");
    for (int i = 0; i < e->n_variants; i++) {
        const char *name = e->variant_name[i];
        int hit = !*typed, digits = *typed != 0;     /* the typed text anywhere in the name, or its number */
        for (const char *d = typed; *d; d++) if (*d < '0' || *d > '9') digits = 0;
        if (digits) hit = atoi(typed) == i + 1;
        for (const char *h = name; *h && !hit && !digits; h++) {
            const char *a = typed, *b = h;
            while (*a && *b && (*a | 32) == (*b | 32)) { a++; b++; }
            if (!*a) hit = 1;
        }
        if (!hit) continue;
        MenuItem *it = menu_add();
        if (!it) return;
        snprintf(it->tab, sizeof it->tab, "/model %d", i + 1);
        snprintf(it->label, sizeof it->label, "%s", name);
        snprintf(it->hint, sizeof it->hint, "%s", e->variant_info[i]);
        if (e->variant && !strcmp(e->variant, name)) snprintf(it->value, sizeof it->value, "open");
        it->submits = 1;
    }
}

static void menu_gen(const char *typed)
{
    static const int v[] = { 256, 512, 1024, 2048, 4096 };
    const int left = g_s->e->cap - g_s->e->cached;
    snprintf(M.title, sizeof M.title, "tokens per answer");
    int numeric = *typed != 0;
    for (const char *p = typed; *p; p++) if (*p < '0' || *p > '9') numeric = 0;
    M.raw_enter = numeric;
    for (int i = 0; i < (int)(sizeof v / sizeof *v); i++) {
        if (v[i] > GLM53F_CHAT_MAX_GEN || v[i] > g_s->e->cap) break;
        char num[16];
        snprintf(num, sizeof num, "%d", v[i]);
        if (!starts_with_ci(num, typed)) continue;
        MenuItem *it = menu_add();
        snprintf(it->tab, sizeof it->tab, "/gen %d", v[i]);
        snprintf(it->label, sizeof it->label, "%d", v[i]);
        if (g_s->last_spt > 0.0) {
            char d[32];
            duration(d, sizeof d, v[i] * g_s->last_spt);
            snprintf(it->hint, sizeof it->hint, "up to %s at the last answer's %.1f tok/s (%.3f s/token)", d,
                     1.0 / g_s->last_spt, g_s->last_spt);
        } else {
            snprintf(it->hint, sizeof it->hint, "its time shows once an answer has been measured");
        }
        if (v[i] > left) {
            const size_t k = strlen(it->hint);
            snprintf(it->hint + k, sizeof it->hint - k, " · more than the %d positions left", left);
        }
        if (v[i] == g_s->gen) snprintf(it->value, sizeof it->value, "now");
        it->submits = 1;
    }
}

/* ---- files, for /file ---- */

typedef struct { char name[400]; int dir; long long size; } DirEnt;

static int dirent_cmp(const void *a, const void *b)
{
    const DirEnt *x = (const DirEnt *)a, *y = (const DirEnt *)b;
    if (x->dir != y->dir) return y->dir - x->dir;
    const char *p = x->name, *q = y->name;
    while (*p && lower((unsigned char)*p) == lower((unsigned char)*q)) { p++; q++; }
    return lower((unsigned char)*p) - lower((unsigned char)*q);
}

/* Up to cap entries of directory dir ("" is the working directory) whose name starts with pre. */
static int list_dir(const char *dir, const char *pre, DirEnt *out, int cap)
{
    int n = 0;
    const int hidden = pre[0] == '.';
#ifdef _WIN32
    char pat[1100];
    wchar_t wpat[1100];
    snprintf(pat, sizeof pat, "%s*", *dir ? dir : ".\\");
    if (!MultiByteToWideChar(CP_UTF8, 0, pat, -1, wpat, 1100)) return 0;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(wpat, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        DirEnt *e = &out[n];
        if (!WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, e->name, (int)sizeof e->name, NULL, NULL)) continue;
        if (!strcmp(e->name, ".") || !strcmp(e->name, "..")) continue;
        if ((e->name[0] == '.' || (fd.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN)) && !hidden) continue;
        if (!starts_with_ci(e->name, pre)) continue;
        e->dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        e->size = (long long)fd.nFileSizeHigh << 32 | fd.nFileSizeLow;
        n++;
    } while (n < cap && FindNextFileW(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(*dir ? dir : ".");
    if (!d) return 0;
    struct dirent *de;
    while (n < cap && (de = readdir(d)) != NULL) {
        DirEnt *e = &out[n];
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (de->d_name[0] == '.' && !hidden) continue;
        if (!starts_with_ci(de->d_name, pre)) continue;
        snprintf(e->name, sizeof e->name, "%s", de->d_name);
        char full[1500];
        snprintf(full, sizeof full, "%s%s", dir, de->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        e->dir = S_ISDIR(st.st_mode);
        e->size = (long long)st.st_size;
        n++;
    }
    closedir(d);
#endif
    qsort(out, (size_t)n, sizeof *out, dirent_cmp);
    return n;
}

static void menu_files(const char *typed)
{
    const int quoted = typed[0] == '"';
    if (quoted) typed++;
    if (quoted ? strchr(typed, '"') != NULL : strchr(typed, ' ') != NULL) return;   /* the question */
    const char *slash = NULL;
    for (const char *p = typed; *p; p++) if (*p == '/' || *p == '\\') slash = p;
    char dir[1024] = "";
    const char *pre = typed;
    if (slash) {
        const size_t dl = (size_t)(slash - typed) + 1;
        if (dl >= sizeof dir) return;
        memcpy(dir, typed, dl);
        dir[dl] = 0;
        pre = slash + 1;
    }
    static DirEnt ents[256];
    const int n = list_dir(dir, pre, ents, 256);
    const int left = g_s->e->cap - g_s->e->cached;
    snprintf(M.title, sizeof M.title, "files in %s", *dir ? dir : "the working directory");
    snprintf(M.right, sizeof M.right, "%d positions left", left);
    for (int i = 0; i < n; i++) {
        const DirEnt *e = &ents[i];
        /* a path cut short would complete to the wrong file: leave out what does not fit */
        char path[1500], tab[sizeof ((MenuItem *)0)->tab];
        if (snprintf(path, sizeof path, "%s%s%s", dir, e->name, e->dir ? "/" : "") >= (int)sizeof path) continue;
        const int q = quoted || strchr(path, ' ') != NULL;
        const int tl = e->dir ? snprintf(tab, sizeof tab, "/file %s%s", q ? "\"" : "", path)
                              : snprintf(tab, sizeof tab, q ? "/file \"%s\" " : "/file %s ", path);
        if (tl >= (int)sizeof tab) continue;
        MenuItem *it = menu_add();
        if (!it) break;
        memcpy(it->tab, tab, (size_t)tl + 1);
        /* the label is only shown: a long name is cut, on a UTF-8 boundary, and ends in "..." */
        if (snprintf(it->label, sizeof it->label, "%s%s", e->name, e->dir ? "/" : "") >= (int)sizeof it->label) {
            size_t cut = sizeof it->label - 4;
            while (cut > 0 && ((unsigned char)it->label[cut] & 0xC0) == 0x80) cut--;
            memcpy(it->label + cut, "...", 4);
        }
        if (e->dir) {
            snprintf(it->hint, sizeof it->hint, "folder");
        } else {
            /* ~3.5 bytes per token for prose and code; the real count comes when it is sent */
            const long long tok = (long long)((double)e->size / 3.5 + 0.5);
            if (tok > left) snprintf(it->hint, sizeof it->hint, "≈ %lld tokens · more than the %d positions left", tok, left);
            else            snprintf(it->hint, sizeof it->hint, "≈ %lld tokens · fits", tok);
            if (e->size < 1024)          snprintf(it->value, sizeof it->value, "%lld B", e->size);
            else if (e->size < 1 << 20)  snprintf(it->value, sizeof it->value, "%.1f KB", (double)e->size / 1024.0);
            else                         snprintf(it->value, sizeof it->value, "%.1f MB", (double)e->size / 1048576.0);
        }
        it->submits = 0;
    }
}

/* Rebuild the items for the editor's text; no menu unless it is one line starting with "/". */
static void menu_build(void)
{
    const char *t = E.b.p ? E.b.p : "";
    if (M.key && !strcmp(M.key, t) && now_s() - M.built < 1.0) return;
    free(M.key);
    M.key = (char *)malloc(E.b.n + 1);
    if (M.key) memcpy(M.key, t, E.b.n + 1);
    M.built = now_s();
    M.n = 0;
    M.raw_enter = 0;
    M.title[0] = M.right[0] = 0;
    if (t[0] != '/' || strchr(t, '\n')) return;
    if (g_s->last_spt > 0.0)
        snprintf(M.right, sizeof M.right, "last answer %.1f tok/s · %.3f s/token", 1.0 / g_s->last_spt, g_s->last_spt);
    else
        snprintf(M.right, sizeof M.right, "local · %s", g_s->e->name);
    const char *sp = strchr(t, ' ');
    if (!sp)                              menu_commands(t);
    else if (!strncmp(t, "/reasoning ", 11)) menu_reasoning(t + 11);
    else if (!strncmp(t, "/model ", 7))     menu_models(t + 7);
    else if (!strncmp(t, "/gen ", 5))       menu_gen(t + 5);
    else if (!strncmp(t, "/file ", 6))      menu_files(t + 6);
    if (M.sel >= M.n) M.sel = M.n ? M.n - 1 : 0;
}

static int menu_open(void) { menu_build(); return M.n > 0 && !M.closed; }

/* Exactly cols columns of s: cut with an ellipsis, or padded with spaces. */
/* The menu's rows (0 when closed), drawn from screen row `top`. */
static int menu_rows(void) { return M.n && !M.closed ? (M.n < MENU_ROWS ? M.n : MENU_ROWS) + 2 : 0; }

static void menu_draw(Buf *f, int top, int W)
{
    const int rows = menu_rows() - 2, inner = W - 6;     /* between "  │" and "│" */
    int first = M.sel - rows + 1;
    if (first < 0) first = 0;
    int lw = 0, vw = 0;
    for (int i = 0; i < M.n; i++) {
        const int l = cols_of(M.it[i].label, (int)strlen(M.it[i].label));
        const int v = cols_of(M.it[i].value, (int)strlen(M.it[i].value));
        if (l > lw) lw = l;
        if (v > vw) vw = v;
    }
    if (lw > inner / 2) lw = inner / 2;
    int hw = inner - 3 - lw - 2 - vw - (vw ? 2 : 1);
    if (hw < 0) hw = 0;

    char left[200];
    snprintf(left, sizeof left, " %s ", M.title);
    int lc = cols_of(left, (int)strlen(left)), rc = cols_of(M.right, (int)strlen(M.right)) + 2;
    if (lc + rc + 2 > inner) rc = 0;
    if (lc + 2 > inner) lc = inner - 2;
    buf_fmt(f, "\033[%d;1H\033[2K  %s╭─%s", top, C_DIM, C_BRIGHT);
    put_cols(f, left, lc);
    buf_str(f, C_DIM);
    rule(f, inner - 2 - lc - rc);
    if (rc) buf_fmt(f, " %s%s%s ", C_GREEN, M.right, C_DIM);
    buf_str(f, "─╮" C_RESET);

    for (int r = 0; r < rows; r++) {
        const int i = first + r;
        const MenuItem *it = &M.it[i];
        const int on = i == M.sel;
        buf_fmt(f, "\033[%d;1H\033[2K  %s│%s", top + 1 + r, C_DIM, on ? C_PANEL : "");
        buf_str(f, on ? C_BRIGHT " ▸ " : "   ");
        buf_str(f, on ? C_BOLD C_BRIGHT : C_GREEN);
        put_cols(f, it->label, lw);
        buf_str(f, C_NOBOLD "  ");
        buf_str(f, on ? C_TEXT : C_DIM);
        put_cols(f, it->hint, hw);
        buf_str(f, " ");
        if (vw) {
            const int v = cols_of(it->value, (int)strlen(it->value));
            spaces(f, vw - v);
            buf_str(f, on ? C_BRIGHT : C_DIM);
            buf_str(f, it->value);
            buf_str(f, " ");
        }
        buf_str(f, C_RESET C_DIM "│" C_RESET);
    }

    char keys[160], count[32];
    snprintf(keys, sizeof keys, " ↑↓ choose · tab complete · enter %s · esc close ",
             M.raw_enter ? "send as typed" : M.it[M.sel].submits ? "run" : "complete");
    snprintf(count, sizeof count, " %d/%d ", M.sel + 1, M.n);
    int kc = cols_of(keys, (int)strlen(keys));
    const int cc = cols_of(count, (int)strlen(count));
    if (kc + cc + 2 > inner) kc = 0;
    buf_fmt(f, "\033[%d;1H\033[2K  %s╰─", top + 1 + rows, C_DIM);
    if (kc) buf_str(f, keys);
    rule(f, inner - 2 - kc - cc);
    buf_str(f, count);
    buf_str(f, "─╯" C_RESET);
}

/* ------------------------------------------------------------------ session ---- */
static volatile int g_busy, g_worker_done, g_fatal;
static pthread_t g_worker;
static Glm53fInput g_job;
static char *g_queue[16];
static int   g_nqueue;
static int   g_offset;                               /* lines scrolled up from the bottom */
static int   g_paste;

static void sink_status(Glm53fTurnSink *k, const char *text, double t0)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    Block *b = last_block(B_STATUS);
    if (!text) {
        if (b) { free(b->text.p); g_nblocks--; g_dirty = 1; }
    } else {
        if (!b) b = block_new(B_STATUS);
        if (b) { b->text.n = 0; buf_str(&b->text, text); b->t0 = t0; g_dirty = 1; }
    }
    pthread_mutex_unlock(&g_mu);
}

static void sink_reasoning(Glm53fTurnSink *k, const char *p, int n)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    Block *b = last_block(B_REASON);
    if (!b) b = block_new(B_REASON);
    if (b) buf_add(&b->text, p, (size_t)n);
    g_dirty = 1;
    pthread_mutex_unlock(&g_mu);
}

static void sink_thought(Glm53fTurnSink *k, double sec, int tokens)
{
    (void)k;
    char d[32], line[96];
    duration(d, sizeof d, sec);
    snprintf(line, sizeof line, "Pensou por %s · %d tokens", d, tokens);
    pthread_mutex_lock(&g_mu);
    block_add(B_THOUGHT, line);
    pthread_mutex_unlock(&g_mu);
}

static void sink_answer(Glm53fTurnSink *k, const char *p, int n)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    Block *b = last_block(B_ANSWER);
    if (!b) b = block_new(B_ANSWER);
    if (b) {
        if (!b->text.n) while (n > 0 && (*p == '\n' || *p == ' ')) { p++; n--; }
        buf_add(&b->text, p, (size_t)n);
    }
    g_dirty = 1;
    pthread_mutex_unlock(&g_mu);
}

static void sink_done(Glm53fTurnSink *k, const char *stats, int state)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    block_add(state == 2 ? B_ERROR : B_STATS, stats);
    pthread_mutex_unlock(&g_mu);
}

/* the agent's tool calls, and its question before one that writes or runs: the worker
 * waits on g_ask_cv until the UI thread reads s, a or n (or esc) */
static pthread_cond_t g_ask_cv;
static volatile int   g_ask_state, g_ask_answer;    /* state 1: a question is open */
static int            g_ask_block;                  /* its block */

static void sink_tool(Glm53fTurnSink *k, const char *title, const char *summary, int ok)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    Block *b = block_new(B_TOOL);
    if (b) { buf_str(&b->text, title); buf_str(&b->text, "\n"); buf_str(&b->text, summary); b->t0 = ok ? 1.0 : 0.0; }
    pthread_mutex_unlock(&g_mu);
}

static int sink_approve(Glm53fTurnSink *k, const char *title, const char *detail)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    Block *b = block_new(B_ASK);
    if (b) {
        buf_str(&b->text, "permitir ");
        buf_str(&b->text, title);
        if (detail && *detail) { buf_str(&b->text, "\n"); buf_str(&b->text, detail); }
    }
    g_ask_block = g_nblocks - 1;
    g_ask_answer = 0;
    g_ask_state = 1;
    g_offset = 0;
    g_dirty = 1;
    while (g_ask_state == 1) pthread_cond_wait(&g_ask_cv, &g_mu);
    const int a = g_ask_answer;
    pthread_mutex_unlock(&g_mu);
    return a;
}

/* The UI thread's answer to the open question (under g_mu). */
static void ask_answer(int a)
{
    if (g_ask_state != 1) return;
    g_ask_answer = a;
    g_ask_state = 0;
    if (g_ask_block < g_nblocks && g_blocks[g_ask_block].kind == B_ASK) {
        Block *b = &g_blocks[g_ask_block];
        char *nl = b->text.p ? strchr(b->text.p, '\n') : NULL;   /* keep the title line */
        if (nl) { b->text.n = (size_t)(nl - b->text.p); *nl = 0; }
        buf_str(&b->text, a == 2 ? " — sempre" : a == 1 ? " — sim" : " — não");
        b->kind = B_NOTE;
    }
    g_dirty = 1;
    pthread_cond_broadcast(&g_ask_cv);
}

static void *worker_main(void *arg)
{
    (void)arg;
    Glm53fTurnSink k = { sink_status, sink_reasoning, sink_thought, sink_answer, sink_done, NULL, sink_tool, sink_approve };
    if (glm53f_chat_turn(g_s, &g_job, &k) != 0) g_fatal = 1;
    pthread_mutex_lock(&g_mu);
    g_worker_done = 1;
    g_dirty = 1;
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

/* Carry out one submitted line on the UI thread; a message starts the worker. */
static int handle_line(const char *line)
{
    Glm53fInput in;
    glm53f_chat_input(g_s, line, &in);
    int quit = 0;
    pthread_mutex_lock(&g_mu);
    switch (in.kind) {
    case GLM53F_IN_QUIT:  quit = 1; break;
    case GLM53F_IN_NOTE:  block_add(B_NOTE, in.note); break;
    case GLM53F_IN_RESET:
        free(g_recent);
        g_recent = lc_session_recent(g_s, 3);
        blocks_clear();
        block_add(B_SPLASH, NULL);
        block_add(B_NOTE, in.note);
        g_offset = 0;
        set_title(NULL);
        break;
    case GLM53F_IN_MESSAGE:
        if (g_fatal) { block_add(B_ERROR, "the engine failed earlier in this session; restart localcode"); break; }
        {
            int users = 0;
            for (int i = 0; i < g_nblocks; i++) users += g_blocks[i].kind == B_USER;
            if (!users && !in.compact) set_title(in.shown);
        }
        block_add(B_USER, in.shown);
        if (in.note) block_add(B_NOTE, in.note);
        g_job = in;
        memset(&in, 0, sizeof in);
        g_busy = 1;
        g_worker_done = 0;
        g_offset = 0;
        if (pthread_create(&g_worker, NULL, worker_main, NULL) != 0) {
            g_busy = 0;
            block_add(B_ERROR, "could not start the model thread");
            glm53f_chat_input_free(&g_job);
        }
        break;
    default: break;
    }
    g_dirty = 1;
    pthread_mutex_unlock(&g_mu);
    glm53f_chat_input_free(&in);
    return quit;
}

static void submit(void)
{
    if (!E.b.n) return;
    char *line = (char *)malloc(E.b.n + 1);
    if (!line) return;
    memcpy(line, E.b.p, E.b.n + 1);
    if (g_nhist == 64) { free(g_hist[0]); memmove(g_hist, g_hist + 1, 63 * sizeof *g_hist); g_nhist--; }
    g_hist[g_nhist] = (char *)malloc(E.b.n + 1);
    if (g_hist[g_nhist]) memcpy(g_hist[g_nhist++], line, E.b.n + 1);
    g_hpos = g_nhist;
    ed_set("");
    if (g_nqueue < 16) g_queue[g_nqueue++] = line;
    else free(line);
    g_dirty = 1;
}

/* ------------------------------------------------------------------- frame ---- */

static void draw(Glm53fChat *s)
{
    term_size();
    const int W = T.w, H = T.h;
    Lines L = { 0 };
    pthread_mutex_lock(&g_mu);
    render_blocks(&L, s, W);
    g_dirty = 0;
    pthread_mutex_unlock(&g_mu);

    /* the input box: its text wrapped at EW columns, the cursor's row and column */
    const int EW = W - 7;
    int *rows_start = (int *)malloc((E.b.n + 2) * sizeof(int));
    int nrows = 0, cur_row = 0, cur_col = 0;
    if (!rows_start) { lines_free(&L); return; }
    {
        int col = 0;
        rows_start[nrows++] = 0;
        for (int i = 0; i <= (int)E.b.n; i++) {
            if (i == (int)E.b.n) { if (i == E.cur) { cur_row = nrows - 1; cur_col = col; } break; }
            if (E.b.p[i] == '\n') {
                if (i == E.cur) { cur_row = nrows - 1; cur_col = col; }
                rows_start[nrows++] = i + 1;
                col = 0;
                continue;
            }
            if (is_cont(E.b.p[i])) continue;
            if (col == EW) { rows_start[nrows++] = i; col = 0; }
            if (i == E.cur) { cur_row = nrows - 1; cur_col = col; }
            col++;
        }
    }
    const int shown_rows = nrows < EDITOR_ROWS ? nrows : EDITOR_ROWS;
    int first_row = cur_row - shown_rows + 1;
    if (first_row < 0) first_row = 0;
    if (first_row + shown_rows > nrows) first_row = nrows - shown_rows;

    const int box = shown_rows + 3, footer = 1;
    menu_build();
    int mr = menu_rows();
    if (H - box - footer - 1 - mr < 3) mr = 0;          /* too short a window for it */
    int view = H - box - footer - 1 - mr;
    if (view < 1) view = 1;
    int maxoff = L.n - view;
    if (maxoff < 0) maxoff = 0;
    if (g_offset > maxoff) g_offset = maxoff;
    if (g_offset < 0) g_offset = 0;

    Buf f = { 0 };
    buf_str(&f, "\033[?2026h\033[?25l");
    /* a conversation shorter than the screen starts at the top; a longer one ends at the bottom */
    const int top_pad = L.n < view ? 0 : -1;
    for (int r = 0; r < view; r++) {
        const int idx = top_pad == 0 ? r : L.n - view - g_offset + r;
        buf_fmt(&f, "\033[%d;1H\033[2K", r + 1);
        if (idx >= 0 && idx < L.n) buf_str(&f, L.v[idx].s);
    }
    if (mr) menu_draw(&f, view + 1, W);
    buf_fmt(&f, "\033[%d;1H\033[2K", view + mr + 1);

    int row = view + mr + 2;
    const char *bar = C_PANEL C_BRIGHT "▌" C_TEXT "  ";
    buf_fmt(&f, "\033[%d;1H\033[2K  %s", row++, bar);
    spaces(&f, W - 7);
    buf_str(&f, C_RESET);
    const int editor_top = row;
    for (int r = first_row; r < first_row + shown_rows; r++) {
        buf_fmt(&f, "\033[%d;1H\033[2K  %s", row++, bar);
        int a = rows_start[r], z = r + 1 < nrows ? rows_start[r + 1] : (int)E.b.n;
        if (z > a && E.b.p[z - 1] == '\n') z--;
        int used = 0;
        if (E.b.n == 0 && r == 0) {
            const char *ph = g_busy ? "digite a próxima mensagem; ela vai quando esta resposta terminar"
                                    : "Peça algo…  \"O que este projeto faz?\"  ·  / comandos";
            buf_str(&f, C_DIM);
            buf_str(&f, ph);
            used = cols_of(ph, (int)strlen(ph));
            buf_str(&f, C_TEXT);
        } else if (z > a) {
            buf_add(&f, E.b.p + a, (size_t)(z - a));
            used = cols_of(E.b.p + a, z - a);
        }
        spaces(&f, W - 7 - used);
        buf_str(&f, C_RESET);
    }
    buf_fmt(&f, "\033[%d;1H\033[2K  %s", row++, bar);    /* a breath under the text */
    spaces(&f, W - 7);
    buf_str(&f, C_RESET);
    {   /* the mode row: what answers, with which model, here; the state on the right */
        char left[400], right[160];
        static const char *mode[] = { "Chat", "Rápido", "Agente" };
        const int lvl = s->agent_level > 0 && s->e->agent ? s->agent_level : 0;
        snprintf(left, sizeof left, C_BOLD C_BRIGHT "%s" C_NOBOLD C_DIM " · " C_TEXT "%s" C_DIM " · " C_YELLOW "%s",
                 mode[lvl], s->e->name,
                 lvl ? (lvl == 1 ? "low" : "medium") : s->reasoning);
        if (g_ask_state == 1)  snprintf(right, sizeof right, C_YELLOW "s sim · a sempre · n não");
        else if (g_busy)       snprintf(right, sizeof right, C_DIM "esc interrompe%s", g_nqueue ? " · na fila" : "");
        else if (lvl == 2)     snprintf(right, sizeof right, s->auto_approve ? C_YELLOW "⏵⏵ aprovação automática"
                                                                           : C_DIM "pede antes de editar ou rodar");
        else                   snprintf(right, sizeof right, C_DIM "%s", g_nqueue ? "na fila" : "");
        const int lc = vis_cols(left), rc = vis_cols(right);
        buf_fmt(&f, "\033[%d;1H\033[2K  %s%s", row++, bar, left);
        spaces(&f, W - 7 - lc - rc - 1 > 0 ? W - 7 - lc - rc - 1 : 1);
        buf_str(&f, C_PANEL);
        buf_str(&f, right);
        buf_str(&f, " " C_RESET);
    }
    {   /* footer: the folder and its git branch; the context, local, the commands */
        char cwd[512], br[96], where[640], right[200];
        short_cwd(cwd, sizeof cwd);
        git_branch(br, sizeof br);
        snprintf(where, sizeof where, "%s%s%s", cwd, br[0] ? ":" : "", br);
        const int pct = s->e->cap ? (int)(100.0 * s->e->cached / s->e->cap + 0.5) : 0;
        char used[24], cap[24];
        fmt_k(used, sizeof used, s->e->cached);
        fmt_k(cap, sizeof cap, s->e->cap);
        if (g_offset)
            snprintf(right, sizeof right, C_DIM "↑ %d linhas · end volta · " C_TEXT "%s/%s" C_DIM " (%d%%)%s · " C_BRIGHT "local",
                     g_offset, used, cap, pct, s->compactions ? " · compactado" : "");
        else
            snprintf(right, sizeof right, C_TEXT "%s/%s" C_DIM " (%d%%)%s · " C_BRIGHT "local" C_DIM " · " C_TEXT "/" C_DIM " comandos",
                     used, cap, pct, s->compactions ? " · compactado" : "");
        const int rc = vis_cols(right);
        int room = W - 4 - rc - 2;
        if (room < 1) room = 1;
        const int wl = (int)strlen(where);
        const char *cp = where;
        int cut = 0;
        if (cols_of(where, wl) > room) {
            cut = 1;
            for (int skip = cols_of(where, wl) - (room - 1); skip > 0 && *cp; skip--) { cp++; while (*cp && is_cont(*cp)) cp++; }
        }
        buf_fmt(&f, "\033[%d;1H\033[2K  %s%s%s", row, C_DIM, cut ? "…" : "", cp);
        const int pc = cols_of(cp, (int)strlen(cp)) + cut;
        spaces(&f, W - 4 - pc - rc > 0 ? W - 4 - pc - rc : 1);
        buf_str(&f, right);
        buf_str(&f, C_RESET);
    }
    buf_fmt(&f, "\033[%d;%dH\033[?25h\033[?2026l", editor_top + (cur_row - first_row), 6 + cur_col);
    out_str(f.p, f.n);
    free(f.p);
    free(rows_start);
    lines_free(&L);
}

/* -------------------------------------------------------------------- input ---- */

/* Handle a chunk of input bytes; returns 1 to leave. */
static int on_input(const char *p, int n, int view_rows)
{
    int i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)p[i];
        if (g_ask_state == 1) {                          /* an open question takes the keys */
            if (c == 0x1b && i + 1 < n && (p[i + 1] == '[' || p[i + 1] == 'O')) {   /* arrows, mouse: not an answer */
                int j = i + 2;
                while (j < n && !((unsigned char)p[j] >= 0x40 && (unsigned char)p[j] <= 0x7e)) j++;
                i = j + 1;
                continue;
            }
            int a = -1;
            if (c == 's' || c == 'S' || c == 'y' || c == 'Y') a = 1;
            else if (c == 'a' || c == 'A') a = 2;
            else if (c == 'n' || c == 'N' || c == 0x1b || c == 0x03) a = 0;
            if (c == 0x1b || c == 0x03) glm53f_chat_stop();
            if (a >= 0) { pthread_mutex_lock(&g_mu); ask_answer(a); pthread_mutex_unlock(&g_mu); }
            i++;
            continue;
        }
        if (g_paste) {
            if (c == 0x1b && i + 5 < n && !memcmp(p + i, "\033[201~", 6)) { g_paste = 0; i += 6; continue; }
            if (c == '\r') { ed_insert("\n", 1); if (i + 1 < n && p[i + 1] == '\n') i++; }
            else if (c >= 0x20 || c == '\n' || c == '\t') ed_insert(p + i, 1);
            i++;
            continue;
        }
        if (c == 0x1b) {
            if (i + 1 < n && (p[i + 1] == '[' || p[i + 1] == 'O')) {
                const int ss3 = p[i + 1] == 'O';
                int j = i + 2;
                while (j < n && !((unsigned char)p[j] >= 0x40 && (unsigned char)p[j] <= 0x7e)) j++;
                if (j >= n) return 0;                    /* an incomplete sequence: drop it */
                const char fin = p[j];
                char par[32];
                int pl = j - (i + 2);
                if (pl > 31) pl = 31;
                memcpy(par, p + i + 2, (size_t)pl);
                par[pl] = 0;
                i = j + 1;
                if (!ss3 && par[0] == '<' && (fin == 'M' || fin == 'm')) {     /* SGR mouse */
                    const int b = atoi(par + 1);
                    if (b == 64) g_offset += 3;
                    if (b == 65) g_offset -= 3;
                    g_dirty = 1;
                    continue;
                }
                if (!strcmp(par, "200") && fin == '~') { g_paste = 1; continue; }
                if ((fin == 'A' || fin == 'B') && menu_open()) {   /* up, down: through the menu */
                    M.sel = (M.sel + (fin == 'A' ? M.n - 1 : 1)) % M.n;
                    g_dirty = 1;
                    continue;
                }
                if (fin == 'Z' && menu_open()) {                   /* shift+tab: back through it */
                    M.sel = (M.sel + M.n - 1) % M.n;
                    g_dirty = 1;
                    continue;
                }
                switch (fin) {
                case 'A':
                    if (g_hpos > 0) { g_hpos--; ed_set(g_hist[g_hpos]); }
                    break;
                case 'B':
                    if (g_hpos < g_nhist - 1) { g_hpos++; ed_set(g_hist[g_hpos]); }
                    else { g_hpos = g_nhist; ed_set(""); }
                    break;
                case 'C': E.cur = ed_next(E.cur); break;
                case 'D': E.cur = ed_prev(E.cur); break;
                case 'H': E.cur = 0; break;
                case 'F': E.cur = (int)E.b.n; g_offset = 0; break;
                case '~': {
                    const int k = atoi(par);
                    if (k == 3) ed_erase(E.cur, ed_next(E.cur));
                    else if (k == 5) g_offset += view_rows - 2;
                    else if (k == 6) g_offset -= view_rows - 2;
                    else if (k == 1 || k == 7) E.cur = 0;
                    else if (k == 4 || k == 8) { E.cur = (int)E.b.n; g_offset = 0; }
                    break;
                }
                default: break;
                }
                g_dirty = 1;
                continue;
            }
            if (menu_open()) M.closed = 1;               /* esc: close the menu first, */
            else if (g_busy) glm53f_chat_stop();         /* then stop the answer       */
            i++;
            g_dirty = 1;
            continue;
        }
        if (c == 0x03) {                                 /* ctrl+c */
            if (g_busy) glm53f_chat_stop();
            else if (E.b.n) ed_set("");
            else return 1;
            i++;
            g_dirty = 1;
            continue;
        }
        if (c == 0x04 && !E.b.n && !g_busy) return 1;   /* ctrl+d on an empty box */
        if (c == '\r' || c == '\n') {
            int next = i + 1;
            if (c == '\r' && next < n && p[next] == '\n') next++;
            /* an unbracketed paste: more text right behind this line break */
            if (c == '\r' && next < n && (unsigned char)p[next] >= 0x20) ed_insert("\n", 1);
            else if (menu_open() && !M.raw_enter) {
                const MenuItem *it = &M.it[M.sel];
                const int send = it->submits;
                ed_set(it->tab);
                if (send) submit();                     /* otherwise it only fills the box */
            } else {
                submit();
            }
            i = next;
            g_dirty = 1;
            continue;
        }
        if (c == 0x7f || c == 0x08) { ed_erase(ed_prev(E.cur), E.cur); i++; g_dirty = 1; continue; }
        if (c == 0x01) { E.cur = 0; i++; g_dirty = 1; continue; }
        if (c == 0x05) { E.cur = (int)E.b.n; i++; g_dirty = 1; continue; }
        if (c == 0x15) { ed_set(""); i++; g_dirty = 1; continue; }
        if (c == 0x17) {                                 /* ctrl+w: the word before the cursor */
            int a = E.cur;
            while (a > 0 && E.b.p[a - 1] == ' ') a--;
            while (a > 0 && E.b.p[a - 1] != ' ') a--;
            ed_erase(a, E.cur);
            i++;
            g_dirty = 1;
            continue;
        }
        if (c == '\t') {
            if (menu_open()) ed_set(M.it[M.sel].tab);   /* tab: complete the choice */
            else ed_insert("    ", 4);
            i++;
            g_dirty = 1;
            continue;
        }
        if (c < 0x20) { i++; continue; }
        int j = i;
        while (j < n && (unsigned char)p[j] >= 0x20 && p[j] != 0x7f) j++;
        ed_insert(p + i, j - i);
        i = j;
        g_dirty = 1;
    }
    return 0;
}

/* --------------------------------------------------------------------- main ---- */

int glm53f_tui_run(Glm53fChat *s, int forced)
{
    if (term_open(forced) != 0) return -1;
    g_s = s;
    pthread_mutex_init(&g_mu, NULL);
    pthread_cond_init(&g_ask_cv, NULL);
    g_recent = lc_session_recent(s, 3);
    set_title(NULL);
    block_add(B_SPLASH, NULL);
    if (s->start_note) block_add(B_NOTE, s->start_note);

    char in[4096];
    int quit = 0, eof = 0;
    double last_draw = 0.0;
    while (!quit) {
        const int got = eof ? 0 : in_read(in, sizeof in, g_busy ? 100 : 400);
        if (got < 0) eof = 1;
        if (got > 0) {
            const int view = T.h - 6;
            if (on_input(in, got, view > 3 ? view : 3)) quit = 1;
        }
        if (g_busy) {
            pthread_mutex_lock(&g_mu);
            const int done = g_worker_done;
            pthread_mutex_unlock(&g_mu);
            if (done) {
                pthread_join(g_worker, NULL);
                glm53f_chat_input_free(&g_job);
                g_busy = 0;
                g_dirty = 1;
            }
        }
        if (!g_busy && g_nqueue && !quit) {
            char *line = g_queue[0];
            memmove(g_queue, g_queue + 1, (size_t)(--g_nqueue) * sizeof *g_queue);
            if (handle_line(line)) quit = 1;
            free(line);
        }
        if (eof && !g_busy && !g_nqueue) quit = 1;
        const int old_w = T.w, old_h = T.h;
        term_size();
        if (T.w != old_w || T.h != old_h) { out_str("\033[2J", 4); g_dirty = 1; }
        const double t = now_s();
        if (g_dirty || (g_busy && t - last_draw > 0.12)) {
            draw(s);
            last_draw = t;
        }
    }
    if (g_busy) {                                        /* leaving mid-answer: stop it first */
        glm53f_chat_stop();
        pthread_mutex_lock(&g_mu);
        ask_answer(0);
        pthread_mutex_unlock(&g_mu);
        pthread_join(g_worker, NULL);
        glm53f_chat_input_free(&g_job);
        g_busy = 0;
    }
    term_close();
    pthread_mutex_lock(&g_mu);
    blocks_clear();
    pthread_mutex_unlock(&g_mu);
    free(g_blocks);
    g_blocks = NULL;
    free(g_recent);
    g_recent = NULL;
    g_capblocks = 0;
    for (int i = 0; i < g_nhist; i++) free(g_hist[i]);
    for (int i = 0; i < g_nqueue; i++) free(g_queue[i]);
    free(E.b.p);
    free(M.key);
    printf("localcode · bye\n");
    return g_fatal ? 1 : 0;
}
