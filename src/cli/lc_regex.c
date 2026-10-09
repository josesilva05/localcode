/* lc_regex.c - see lc_regex.h. The pattern is parsed into a tree, the tree compiled into a program
 * (Thompson's construction: byte sets, split, jump, assertions, match), and the program run over the
 * text as a Pike VM: every live thread advances one byte at a time, at most one per instruction.
 *
 * The text is UTF-8 and every character-wide piece of a pattern takes a whole character: a literal
 * ã is its two bytes in a row, and ., [...], \w \s \d and their negations take a lead byte with
 * exactly its continuation bytes, so a quantifier, a class or a \b never splits a character. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lc_regex.h"

#define MAX_PROG 60000         /* instructions, counted after {m,n} copies     */
#define MAX_REP  1000          /* the largest m or n of {m,n}                   */
#define MAX_CPS  70000         /* code points above 127 that one class may name */

enum { N_SET, N_BOL, N_EOL, N_WB, N_NWB, N_CAT, N_ALT, N_REP, N_EMPTY };
enum { I_SET, I_SPLIT, I_JMP, I_ASSERT, I_MATCH };

typedef struct { int type, a, b, min, max, set; } Node;
typedef struct { unsigned char op, kind; int x, y; } Ins;
typedef unsigned char Set[32];

struct LcRegex {
    Ins *prog;
    int  n;
    Set *sets;
    int  nsets;
    unsigned *mark;
    unsigned  gen;
    int *l1, *l2, *stack;
};

typedef struct {
    const char *p;
    int   icase, bad;
    char *err;
    size_t ecap;
    Node *nodes;
    int   nn, cn;
    Set  *sets;
    int   ns, cs;
} Parse;

/* a class: its ASCII members as bits, whether it takes every character above 127, and the ones
 * above 127 it names */
typedef struct {
    Set lo;
    int allhigh;
    int *cp;
    int ncp, ccp;
} Cls;

static void fail(Parse *ps, const char *why)
{
    if (!ps->bad) snprintf(ps->err, ps->ecap, "%s", why);
    ps->bad = 1;
}

static int node(Parse *ps, int type, int a, int b)
{
    if (ps->nn == ps->cn) {
        const int c = ps->cn ? ps->cn * 2 : 64;
        Node *g = (Node *)realloc(ps->nodes, (size_t)c * sizeof *g);
        if (!g) { fail(ps, "out of memory"); return 0; }
        ps->nodes = g;
        ps->cn = c;
    }
    Node *x = &ps->nodes[ps->nn];
    memset(x, 0, sizeof *x);
    x->type = type;
    x->a = a;
    x->b = b;
    return ps->nn++;
}

static void set_add(Set s, int c) { s[c >> 3] |= (unsigned char)(1u << (c & 7)); }
static int  set_has(const Set s, int c) { return (s[c >> 3] >> (c & 7)) & 1; }

/* a node taking one byte of s */
static int set_node(Parse *ps, const Set s)
{
    if (ps->ns == ps->cs) {
        const int c = ps->cs ? ps->cs * 2 : 16;
        Set *g = (Set *)realloc(ps->sets, (size_t)c * sizeof *g);
        if (!g) { fail(ps, "out of memory"); return 0; }
        ps->sets = g;
        ps->cs = c;
    }
    memcpy(ps->sets[ps->ns], s, sizeof(Set));
    const int x = node(ps, N_SET, 0, 0);
    if (!ps->bad) ps->nodes[x].set = ps->ns++;
    return x;
}

static int range_node(Parse *ps, int lo, int hi)
{
    Set s;
    memset(s, 0, sizeof s);
    for (int k = lo; k <= hi; k++) set_add(s, k);
    return set_node(ps, s);
}

static int byte_node(Parse *ps, int c, int icase)
{
    Set s;
    memset(s, 0, sizeof s);
    set_add(s, c);
    if (icase && c >= 'a' && c <= 'z') set_add(s, c - 32);
    if (icase && c >= 'A' && c <= 'Z') set_add(s, c + 32);
    return set_node(ps, s);
}

/* a then b; -1 stands for nothing on either side */
static int cat(Parse *ps, int a, int b) { return a < 0 ? b : b < 0 ? a : node(ps, N_CAT, a, b); }
static int alt(Parse *ps, int a, int b) { return a < 0 ? b : b < 0 ? a : node(ps, N_ALT, a, b); }

static int is_word(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c >= 128; }
static int is_space(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static int is_digit(int c) { return c >= '0' && c <= '9'; }

static int utf8_enc(int cp, unsigned char *b)
{
    if (cp < 0x80) { b[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) { b[0] = (unsigned char)(0xC0 | cp >> 6); b[1] = (unsigned char)(0x80 | (cp & 63)); return 2; }
    if (cp < 0x10000) {
        b[0] = (unsigned char)(0xE0 | cp >> 12); b[1] = (unsigned char)(0x80 | ((cp >> 6) & 63)); b[2] = (unsigned char)(0x80 | (cp & 63));
        return 3;
    }
    b[0] = (unsigned char)(0xF0 | cp >> 18); b[1] = (unsigned char)(0x80 | ((cp >> 12) & 63));
    b[2] = (unsigned char)(0x80 | ((cp >> 6) & 63)); b[3] = (unsigned char)(0x80 | (cp & 63));
    return 4;
}

/* the character at *pp (UTF-8; a byte that starts no valid sequence stands for itself), *pp moved past */
static int utf8_dec(const char **pp)
{
    const unsigned char *s = (const unsigned char *)*pp;
    const int c = s[0];
    int len = c >= 0xF0 && c <= 0xF4 ? 4 : c >= 0xE0 ? 3 : c >= 0xC2 && c <= 0xDF ? 2 : 1;
    if (c < 0x80 || c > 0xF4) len = 1;
    int cp = len == 1 ? c : len == 2 ? c & 31 : len == 3 ? c & 15 : c & 7;
    for (int k = 1; k < len; k++) {
        if ((s[k] & 0xC0) != 0x80) { *pp += 1; return c; }
        cp = cp << 6 | (s[k] & 63);
    }
    *pp += len;
    return cp;
}

/* the bytes of one character in a row */
static int char_seq(Parse *ps, int cp, int icase)
{
    unsigned char b[4];
    const int n = utf8_enc(cp, b);
    int r = -1;
    for (int k = 0; k < n; k++) r = cat(ps, r, byte_node(ps, b[k], icase));
    return r;
}

/* any character above 127: a lead byte and exactly its continuation bytes */
static int any_high(Parse *ps)
{
    const int c = range_node(ps, 0x80, 0xBF);
    int two = cat(ps, range_node(ps, 0xC2, 0xDF), c);
    int three = cat(ps, cat(ps, range_node(ps, 0xE0, 0xEF), c), c);
    int four = cat(ps, cat(ps, cat(ps, range_node(ps, 0xF0, 0xF4), c), c), c);
    return alt(ps, alt(ps, two, three), four);
}

typedef struct { unsigned char b[4]; int n; } Seq;

static int seq_cmp(const void *x, const void *y)
{
    const Seq *a = (const Seq *)x, *b = (const Seq *)y;
    if (a->n != b->n) return a->n - b->n;
    return memcmp(a->b, b->b, (size_t)a->n);
}

/* the sequences v[0..n) (sorted, one length, the same bytes before depth): a tree that takes any of them */
static int seq_any(Parse *ps, const Seq *v, int n, int depth)
{
    int r = -1;
    Set leaf;
    memset(leaf, 0, sizeof leaf);
    int leaves = 0;
    for (int i = 0; i < n && !ps->bad;) {
        int j = i;
        while (j < n && v[j].b[depth] == v[i].b[depth]) j++;
        if (depth == v[i].n - 1) { set_add(leaf, v[i].b[depth]); leaves = 1; }
        else r = alt(ps, r, cat(ps, byte_node(ps, v[i].b[depth], 0), seq_any(ps, v + i, j - i, depth + 1)));
        i = j;
    }
    if (leaves) r = alt(ps, r, set_node(ps, leaf));
    return r;
}

/* every sequence of length len whose byte at depth is in [lo, hi], except v[0..n) (sorted, the same
 * bytes before depth): -1 when there is none */
static int seq_but(Parse *ps, const Seq *v, int n, int depth, int len, int lo, int hi)
{
    Set plain;
    memset(plain, 0, sizeof plain);
    int any = 0;
    for (int b = lo; b <= hi; b++) set_add(plain, b), any = 1;
    int r = -1;
    for (int i = 0; i < n && !ps->bad;) {
        int j = i;
        const int b = v[i].b[depth];
        while (j < n && v[j].b[depth] == b) j++;
        plain[b >> 3] &= (unsigned char)~(1u << (b & 7));
        if (depth < len - 1) {
            const int sub = seq_but(ps, v + i, j - i, depth + 1, len, 0x80, 0xBF);
            if (sub >= 0) r = alt(ps, r, cat(ps, byte_node(ps, b, 0), sub));
        }
        i = j;
    }
    any = 0;
    for (int b = lo; b <= hi; b++) any |= set_has(plain, b);
    if (any) {
        int t = set_node(ps, plain);
        for (int k = depth + 1; k < len; k++) t = cat(ps, t, range_node(ps, 0x80, 0xBF));
        r = alt(ps, r, t);
    }
    return r;
}

/* the node of a class, negated or not */
static int cls_node(Parse *ps, Cls *c, int neg)
{
    Set lo;
    int has_lo = 0;
    for (int k = 0; k < 32; k++) lo[k] = (unsigned char)((neg ? ~c->lo[k] : c->lo[k]) & (k < 16 ? 0xFF : 0));
    for (int k = 0; k < 16; k++) has_lo |= lo[k];
    int r = has_lo ? set_node(ps, lo) : -1;
    Seq *v = NULL;
    if (c->ncp && !c->allhigh) {
        v = (Seq *)malloc((size_t)c->ncp * sizeof *v);
        if (!v) { fail(ps, "out of memory"); return 0; }
        for (int i = 0; i < c->ncp; i++) v[i].n = utf8_enc(c->cp[i], v[i].b);
        qsort(v, (size_t)c->ncp, sizeof *v, seq_cmp);
    }
    if (!neg) {
        if (c->allhigh) r = alt(ps, r, any_high(ps));
        else if (v) r = alt(ps, r, seq_any(ps, v, c->ncp, 0));
    } else if (!c->allhigh) {
        static const int lead[3][2] = { { 0xC2, 0xDF }, { 0xE0, 0xEF }, { 0xF0, 0xF4 } };
        for (int len = 2; len <= 4; len++) {
            int i = 0, j;
            while (v && i < c->ncp && v[i].n < len) i++;
            for (j = i; v && j < c->ncp && v[j].n == len; j++) {}
            r = alt(ps, r, seq_but(ps, v ? v + i : NULL, v ? j - i : 0, 0, len, lead[len - 2][0], lead[len - 2][1]));
        }
    }
    free(v);
    if (r < 0) {                                   /* a class that takes nothing: [^\s\S] */
        Set none;
        memset(none, 0, sizeof none);
        r = set_node(ps, none);
    }
    return r;
}

static void cls_byte(Parse *ps, Cls *c, int b)
{
    set_add(c->lo, b);
    if (ps->icase && b >= 'a' && b <= 'z') set_add(c->lo, b - 32);
    if (ps->icase && b >= 'A' && b <= 'Z') set_add(c->lo, b + 32);
}

static void cls_cp(Parse *ps, Cls *c, int cp)
{
    if (cp < 128) { cls_byte(ps, c, cp); return; }
    if (c->ncp == c->ccp) {
        const int n = c->ccp ? c->ccp * 2 : 16;
        int *g = (int *)realloc(c->cp, (size_t)n * sizeof *g);
        if (!g) { fail(ps, "out of memory"); return; }
        c->cp = g;
        c->ccp = n;
    }
    c->cp[c->ncp++] = cp;
}

/* \d \w \s and their negations into c; 0 when e is none of them */
static int class_escape(Cls *c, int e)
{
    int (*f)(int);
    switch (e) {
    case 'd': case 'D': f = is_digit; break;
    case 'w': case 'W': f = is_word; break;
    case 's': case 'S': f = is_space; break;
    default: return 0;
    }
    const int neg = e >= 'A' && e <= 'Z';
    for (int k = 0; k < 128; k++) if (f(k) != neg) set_add(c->lo, k);
    if (neg != (e == 'w' || e == 'W')) c->allhigh = 1;    /* \w and \D \S take the characters above 127 */
    return 1;
}

static int hexval(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* the character of an escape that stands for one (after the backslash, at ps->p), or -1 (an error) */
static int escape_char(Parse *ps)
{
    const int e = (unsigned char)*ps->p;
    if (!e) { fail(ps, "a pattern cannot end with \\"); return -1; }
    if (e >= 0x80) return utf8_dec(&ps->p);      /* \ before a non-ASCII character: itself */
    ps->p++;
    switch (e) {
    case 'n': return '\n';
    case 't': return '\t';
    case 'r': return '\r';
    case 'f': return '\f';
    case 'v': return '\v';
    case '0': return 0;
    case 'x': {
        const int h = hexval(ps->p[0]), l = h < 0 ? -1 : hexval(ps->p[1]);
        if (l < 0) { fail(ps, "\\x needs two hex digits"); return -1; }
        ps->p += 2;
        return h * 16 + l;
    }
    }
    if ((e >= 'a' && e <= 'z') || (e >= 'A' && e <= 'Z') || (e >= '0' && e <= '9')) {
        fail(ps, "unknown escape (\\ before a letter or digit other than d w s b n t r f v x)");
        return -1;
    }
    return e;                                      /* \. \( \\ ...: the character itself */
}

static int parse_alt(Parse *ps);

static int parse_class(Parse *ps)
{
    Cls c;
    memset(&c, 0, sizeof c);
    int neg = 0, first = 1;
    if (*ps->p == '^') { neg = 1; ps->p++; }
    for (;;) {
        int ch = (unsigned char)*ps->p;
        if (!ch) { fail(ps, "a [ without its ]"); break; }
        if (ch == ']' && !first) { ps->p++; break; }
        first = 0;
        if (ch == '\\') {
            ps->p++;
            if (class_escape(&c, *ps->p)) { ps->p++; continue; }
            if (*ps->p == 'b') { ps->p++; ch = '\b'; }
            else if ((ch = escape_char(ps)) < 0) break;
        } else {
            ch = utf8_dec(&ps->p);
        }
        int hi = ch;
        if (ps->p[0] == '-' && ps->p[1] && ps->p[1] != ']') {
            ps->p++;
            if (*ps->p == '\\') {
                ps->p++;
                if ((hi = escape_char(ps)) < 0) break;
            } else {
                hi = utf8_dec(&ps->p);
            }
            if (hi < ch) { fail(ps, "a range [b-a] out of order"); break; }
            if (hi - ch > MAX_CPS) { fail(ps, "a range of more than 70000 characters"); break; }
        }
        for (int k = ch; k <= hi && !ps->bad; k++) cls_cp(ps, &c, k);
    }
    const int r = ps->bad ? 0 : cls_node(ps, &c, neg);
    free(c.cp);
    return r;
}

static int parse_atom(Parse *ps)
{
    const int c = (unsigned char)*ps->p;
    switch (c) {
    case '(': {
        ps->p++;
        if (ps->p[0] == '?') {
            if (ps->p[1] != ':') { fail(ps, "only (?: ) groups are supported, not lookaround or flags"); return 0; }
            ps->p += 2;
        }
        const int a = parse_alt(ps);
        if (*ps->p != ')') { fail(ps, "a ( without its )"); return 0; }
        ps->p++;
        return a;
    }
    case '[':
        ps->p++;
        return parse_class(ps);
    case '.': {
        ps->p++;
        Cls d;
        memset(&d, 0, sizeof d);
        for (int k = 0; k < 128; k++) if (k != '\n') set_add(d.lo, k);
        d.allhigh = 1;
        return cls_node(ps, &d, 0);
    }
    case '^': ps->p++; return node(ps, N_BOL, 0, 0);
    case '$': ps->p++; return node(ps, N_EOL, 0, 0);
    case '*': case '+': case '?':
        fail(ps, "nothing to repeat before * + or ?");
        return 0;
    case '\\': {
        ps->p++;
        const int e = (unsigned char)*ps->p;
        if (e == 'b' || e == 'B') { ps->p++; return node(ps, e == 'b' ? N_WB : N_NWB, 0, 0); }
        Cls d;
        memset(&d, 0, sizeof d);
        if (class_escape(&d, e)) { ps->p++; return cls_node(ps, &d, 0); }
        const int ch = escape_char(ps);
        return ch < 0 ? 0 : char_seq(ps, ch, ps->icase);
    }
    }
    return char_seq(ps, utf8_dec(&ps->p), ps->icase);
}

/* {m}, {m,} or {m,n} at ps->p: 1 and the bounds, or 0 (then { is a literal, as in Python) */
static int braces(Parse *ps, int *mn, int *mx)
{
    const char *q = ps->p + 1;
    if (*q < '0' || *q > '9') return 0;
    long a = 0, b;
    while (*q >= '0' && *q <= '9') a = a * 10 + (*q++ - '0'), a = a > 100000 ? 100000 : a;
    if (*q == '}') b = a;
    else if (*q == ',') {
        q++;
        if (*q == '}') b = -1;
        else {
            if (*q < '0' || *q > '9') return 0;
            b = 0;
            while (*q >= '0' && *q <= '9') b = b * 10 + (*q++ - '0'), b = b > 100000 ? 100000 : b;
            if (*q != '}') return 0;
        }
    } else return 0;
    ps->p = q + 1;
    *mn = (int)a;
    *mx = (int)b;
    return 1;
}

static int parse_rep(Parse *ps)
{
    int a = parse_atom(ps);
    for (;;) {
        int mn, mx;
        const char q = *ps->p;
        if (q == '*') { mn = 0; mx = -1; ps->p++; }
        else if (q == '+') { mn = 1; mx = -1; ps->p++; }
        else if (q == '?') { mn = 0; mx = 1; ps->p++; }
        else if (q == '{' && braces(ps, &mn, &mx)) {
            if (mx >= 0 && mx < mn) { fail(ps, "{m,n} with n < m"); return 0; }
            if (mn > MAX_REP || mx > MAX_REP) { fail(ps, "a repeat count above 1000"); return 0; }
        } else break;
        if (*ps->p == '?') ps->p++;                /* lazy: the same lines match */
        if (ps->bad) return 0;
        const int t = ps->nodes[a].type;
        if (t == N_BOL || t == N_EOL || t == N_WB || t == N_NWB) {   /* an anchor repeated is itself, or nothing */
            if (!mn) a = node(ps, N_EMPTY, 0, 0);
            continue;
        }
        const int r = node(ps, N_REP, a, 0);
        if (ps->bad) return 0;
        ps->nodes[r].min = mn;
        ps->nodes[r].max = mx;
        a = r;
    }
    return a;
}

static int parse_seq(Parse *ps)
{
    int r = -1;
    while (*ps->p && *ps->p != '|' && *ps->p != ')' && !ps->bad) r = cat(ps, r, parse_rep(ps));
    return r < 0 ? node(ps, N_EMPTY, 0, 0) : r;
}

static int parse_alt(Parse *ps)
{
    int a = parse_seq(ps);
    while (*ps->p == '|' && !ps->bad) {
        ps->p++;
        a = node(ps, N_ALT, a, parse_seq(ps));
    }
    return a;
}

/* --------------------------------------------------------------------------- compile ---- */

typedef struct { Ins *v; int n, cap, bad; const Node *nodes; } Emit;

static int emit(Emit *e, int op, int kind, int x, int y)
{
    if (e->n >= MAX_PROG) { e->bad = 1; return 0; }
    if (e->n == e->cap) {
        const int c = e->cap ? e->cap * 2 : 64;
        Ins *g = (Ins *)realloc(e->v, (size_t)c * sizeof *g);
        if (!g) { e->bad = 1; return 0; }
        e->v = g;
        e->cap = c;
    }
    e->v[e->n].op = (unsigned char)op;
    e->v[e->n].kind = (unsigned char)kind;
    e->v[e->n].x = x;
    e->v[e->n].y = y;
    return e->n++;
}

static void gen(Emit *e, int ni)
{
    if (e->bad) return;
    const Node *nd = &e->nodes[ni];
    switch (nd->type) {
    case N_SET: emit(e, I_SET, 0, nd->set, 0); break;
    case N_BOL: case N_EOL: case N_WB: case N_NWB: emit(e, I_ASSERT, nd->type, 0, 0); break;
    case N_EMPTY: break;
    case N_CAT: gen(e, nd->a); gen(e, nd->b); break;
    case N_ALT: {
        const int s = emit(e, I_SPLIT, 0, 0, 0);
        gen(e, nd->a);
        const int j = emit(e, I_JMP, 0, 0, 0);
        const int y = e->n;
        gen(e, nd->b);
        if (e->bad) return;
        e->v[s].x = s + 1;
        e->v[s].y = y;
        e->v[j].x = e->n;
        break;
    }
    case N_REP: {
        for (int i = 0; i < nd->min && !e->bad; i++) gen(e, nd->a);
        if (nd->max < 0) {
            const int s = emit(e, I_SPLIT, 0, 0, 0);
            gen(e, nd->a);
            emit(e, I_JMP, 0, s, 0);
            if (e->bad) return;
            e->v[s].x = s + 1;
            e->v[s].y = e->n;
        } else {
            const int first = e->n;
            for (int i = nd->min; i < nd->max && !e->bad; i++) {
                const int s = emit(e, I_SPLIT, 0, 0, -1);
                gen(e, nd->a);
                if (!e->bad) e->v[s].x = s + 1;
            }
            if (e->bad) return;
            for (int k = first; k < e->n; k++)    /* the optional copies' exits: past the last */
                if (e->v[k].op == I_SPLIT && e->v[k].y == -1) e->v[k].y = e->n;
        }
        break;
    }
    }
}

LcRegex *lc_regex_new(const char *pattern, int icase, char *err, size_t ecap)
{
    char dummy[8];
    if (!err || !ecap) { err = dummy; ecap = sizeof dummy; }
    *err = 0;
    Parse ps;
    memset(&ps, 0, sizeof ps);
    ps.p = pattern ? pattern : "";
    ps.icase = icase;
    ps.err = err;
    ps.ecap = ecap;
    const int root = parse_alt(&ps);
    if (!ps.bad && *ps.p == ')') fail(&ps, "a ) without its (");
    if (ps.bad) { free(ps.nodes); free(ps.sets); return NULL; }
    Emit e;
    memset(&e, 0, sizeof e);
    e.nodes = ps.nodes;
    gen(&e, root);
    emit(&e, I_MATCH, 0, 0, 0);
    free(ps.nodes);
    if (e.bad) {
        snprintf(err, ecap, "the pattern is too large once its repeats are written out");
        free(e.v);
        free(ps.sets);
        return NULL;
    }
    LcRegex *re = (LcRegex *)calloc(1, sizeof *re);
    if (!re) { free(e.v); free(ps.sets); return NULL; }
    re->prog = e.v;
    re->n = e.n;
    re->sets = ps.sets;
    re->nsets = ps.ns;
    re->mark = (unsigned *)calloc((size_t)e.n, sizeof *re->mark);
    re->l1 = (int *)malloc((size_t)e.n * sizeof(int));
    re->l2 = (int *)malloc((size_t)e.n * sizeof(int));
    re->stack = (int *)malloc((size_t)e.n * sizeof(int));
    if (!re->mark || !re->l1 || !re->l2 || !re->stack) { lc_regex_free(re); return NULL; }
    return re;
}

void lc_regex_free(LcRegex *re)
{
    if (!re) return;
    free(re->prog);
    free(re->sets);
    free(re->mark);
    free(re->l1);
    free(re->l2);
    free(re->stack);
    free(re);
}

/* -------------------------------------------------------------------------- the VM ---- */

static int assert_ok(int kind, const unsigned char *s, size_t n, size_t i)
{
    switch (kind) {
    case N_BOL: return i == 0;
    case N_EOL: return i == n;
    case N_WB: case N_NWB: {
        const int a = i > 0 && is_word(s[i - 1]), b = i < n && is_word(s[i]);
        if (kind == N_NWB && !n) return 0;         /* as Python: \B never matches an empty line */
        return (a != b) == (kind == N_WB);
    }
    }
    return 0;
}

/* the threads reached from pc at position i into list (the byte-consuming ones); 1 on a match */
static int add(LcRegex *re, int *list, int *ln, int pc, const unsigned char *s, size_t n, size_t i)
{
    int sp = 0;
    if (re->mark[pc] == re->gen) return 0;
    re->mark[pc] = re->gen;
    re->stack[sp++] = pc;
    while (sp) {
        const int p = re->stack[--sp];
        const Ins *in = &re->prog[p];
        switch (in->op) {
        case I_MATCH: return 1;
        case I_SET: list[(*ln)++] = p; break;
        case I_JMP:
            if (re->mark[in->x] != re->gen) { re->mark[in->x] = re->gen; re->stack[sp++] = in->x; }
            break;
        case I_SPLIT:
            if (re->mark[in->y] != re->gen) { re->mark[in->y] = re->gen; re->stack[sp++] = in->y; }
            if (re->mark[in->x] != re->gen) { re->mark[in->x] = re->gen; re->stack[sp++] = in->x; }
            break;
        case I_ASSERT:
            if (assert_ok(in->kind, s, n, i) && re->mark[p + 1] != re->gen) { re->mark[p + 1] = re->gen; re->stack[sp++] = p + 1; }
            break;
        }
    }
    return 0;
}

static void next_gen(LcRegex *re)
{
    if (++re->gen == 0) {                          /* wrapped: the marks start again */
        memset(re->mark, 0, (size_t)re->n * sizeof *re->mark);
        re->gen = 1;
    }
}

int lc_regex_search(LcRegex *re, const char *str, size_t n)
{
    const unsigned char *s = (const unsigned char *)str;
    int *cl = re->l1, *nl = re->l2, cn = 0, nn;
    next_gen(re);
    if (add(re, cl, &cn, 0, s, n, 0)) return 1;
    for (size_t i = 0; i < n; i++) {
        next_gen(re);
        nn = 0;
        const int c = s[i];
        for (int k = 0; k < cn; k++) {
            const Ins *in = &re->prog[cl[k]];
            if (set_has(re->sets[in->x], c) && add(re, nl, &nn, cl[k] + 1, s, n, i + 1)) return 1;
        }
        /* a match may start at any character (not inside one) */
        if ((i + 1 == n || (s[i + 1] & 0xC0) != 0x80) && add(re, nl, &nn, 0, s, n, i + 1)) return 1;
        int *t = cl; cl = nl; nl = t;
        cn = nn;
    }
    return 0;
}
