/* lc_perm.c - see lc_perm.h. */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "lc_perm.h"
#include "lc_tools.h"

#ifdef _WIN32
#define SEP "\\"
#define PATH_ICASE 1
#else
#define SEP "/"
#define PATH_ICASE 0
#endif

typedef struct {
    int   claude;            /* 0 opencode's format, 1 Claude Code's */
    char  key[40];           /* opencode: read edit bash ... or *; Claude Code: Read Edit Bash ... */
    char *pat;               /* NULL: any */
    int   act;
    char  src[48];           /* the file it came from, for /permissions and why */
} Rule;

static Rule *R;
static int   nR, cR;
static char  g_root[4096];

static void add_rule(int claude, const char *key, const char *pat, int act, const char *src)
{
    if (nR == cR) {
        const int c = cR ? cR * 2 : 32;
        Rule *g = (Rule *)realloc(R, (size_t)c * sizeof *g);
        if (!g) return;
        R = g;
        cR = c;
    }
    Rule *r = &R[nR];
    memset(r, 0, sizeof *r);
    r->claude = claude;
    snprintf(r->key, sizeof r->key, "%s", key);
    if (pat) {
        const size_t n = strlen(pat) + 1;
        r->pat = (char *)malloc(n);
        if (!r->pat) return;
        memcpy(r->pat, pat, n);
    }
    r->act = act;
    snprintf(r->src, sizeof r->src, "%s", src);
    nR++;
}

static int act_of(const char *s)
{
    if (!s) return LC_PERM_NONE;
    if (!strcmp(s, "allow")) return LC_PERM_ALLOW;
    if (!strcmp(s, "deny")) return LC_PERM_DENY;
    if (!strcmp(s, "ask")) return LC_PERM_ASK;
    return LC_PERM_NONE;
}

static const char *act_name(int a) { return a == LC_PERM_ALLOW ? "allow" : a == LC_PERM_DENY ? "deny" : a == LC_PERM_ASK ? "ask" : "-"; }

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

/* the text without // and block comments outside strings (opencode.jsonc), in place */
static void strip_comments(char *t)
{
    char *w = t;
    int str = 0;
    for (char *p = t; *p;) {
        if (str) {
            if (*p == '\\' && p[1]) { *w++ = *p++; *w++ = *p++; continue; }
            if (*p == '"') str = 0;
            *w++ = *p++;
        } else if (*p == '"') {
            str = 1;
            *w++ = *p++;
        } else if (p[0] == '/' && p[1] == '/') {
            while (*p && *p != '\n') p++;
        } else if (p[0] == '/' && p[1] == '*') {
            p += 2;
            while (*p && !(p[0] == '*' && p[1] == '/')) p++;
            if (*p) p += 2;
        } else {
            *w++ = *p++;
        }
    }
    *w = 0;
}

static void load_file(const char *path, const char *src)
{
    char *t = lc_slurp(path, NULL);
    if (!t) return;
    strip_comments(t);
    jval *v = json_parse(t, NULL);
    free(t);
    if (!v || v->t != J_OBJ) { json_free(v); return; }
    jval *oc = json_get(v, "permission");          /* opencode */
    if (oc && oc->t == J_STR) add_rule(0, "*", NULL, act_of(oc->str), src);
    if (oc && oc->t == J_OBJ)
        for (int i = 0; i < oc->len; i++) {
            jval *x = oc->kids[i];
            if (x->t == J_STR) add_rule(0, oc->keys[i], NULL, act_of(x->str), src);
            else if (x->t == J_OBJ)
                for (int j = 0; j < x->len; j++)
                    if (x->kids[j]->t == J_STR) add_rule(0, oc->keys[i], strcmp(x->keys[j], "*") ? x->keys[j] : NULL, act_of(x->kids[j]->str), src);
        }
    jval *cc = json_get(v, "permissions");         /* Claude Code */
    static const char *lists[] = { "allow", "ask", "deny" };
    for (int l = 0; cc && cc->t == J_OBJ && l < 3; l++) {
        jval *a = json_get(cc, lists[l]);
        for (int i = 0; a && a->t == J_ARR && i < a->len; i++) {
            if (a->kids[i]->t != J_STR) continue;
            char key[40], pat[2048];
            const char *s = a->kids[i]->str, *open = strchr(s, '(');
            if (open && s[strlen(s) - 1] == ')') {
                snprintf(key, sizeof key, "%.*s", (int)(open - s), s);
                snprintf(pat, sizeof pat, "%.*s", (int)(strlen(open) - 2), open + 1);
                add_rule(1, key, *pat ? pat : NULL, act_of(lists[l]), src);
            } else {
                add_rule(1, s, NULL, act_of(lists[l]), src);
            }
        }
    }
    json_free(v);
}

void lc_perm_load(const char *root)
{
    for (int i = 0; i < nR; i++) free(R[i].pat);
    nR = 0;
    snprintf(g_root, sizeof g_root, "%s", root ? root : ".");
    for (char *p = g_root; *p; p++) if (*p == '\\') *p = '/';
    add_rule(0, "read", "*.env", LC_PERM_ASK, "built-in");
    add_rule(0, "read", "*.env.*", LC_PERM_ASK, "built-in");
    add_rule(0, "read", "*.env.example", LC_PERM_ALLOW, "built-in");
    add_rule(0, "external_directory", NULL, LC_PERM_ASK, "built-in");
    add_rule(0, "doom_loop", NULL, LC_PERM_ASK, "built-in");
    char path[4400];
    const char *home = getenv("USERPROFILE");
    if (!home) home = getenv("HOME");
    if (home) {
        snprintf(path, sizeof path, "%s" SEP ".localcode" SEP "settings.json", home);
        load_file(path, "~/.localcode/settings.json");
    }
    static const char *files[] = { ".claude/settings.json", ".claude/settings.local.json", ".localcode/settings.json",
                                   "opencode.json", "opencode.jsonc" };
    for (int i = 0; i < 5; i++) {
        snprintf(path, sizeof path, "%s/%s", root ? root : ".", files[i]);
        load_file(path, files[i]);
    }
}

/* ------------------------------------------------------------------------ matching ---- */

static int low(int c, int icase) { return icase && c >= 'A' && c <= 'Z' ? c + 32 : c == '\\' ? '/' : c; }

/* a wildcard pattern against s: ? one character, * any text (slash: not across '/', and ** across
 * folders, **\/ any number of them) */
static int wm(const char *p, const char *s, int slash, int icase)
{
    for (; *p; p++, s++) {
        if (*p == '*') {
            int cross = !slash;
            if (p[1] == '*') {
                cross = 1;
                while (p[1] == '*') p++;
                if (slash && p[1] == '/' && wm(p + 2, s, slash, icase)) return 1;   /* **\/: no folder at all */
            }
            if (!p[1]) return cross || !strchr(s, '/');
            for (const char *t = s;; t++) {
                if (wm(p + 1, t, slash, icase)) return 1;
                if (!*t || (!cross && (*t == '/' || *t == '\\'))) return 0;
            }
        }
        if (!*s) return 0;
        if (*p == '?') {
            if (slash && (*s == '/' || *s == '\\')) return 0;
            continue;
        }
        if (low((unsigned char)*p, icase) != low((unsigned char)*s, icase)) return 0;
    }
    return !*s;
}

static const char *okey(const char *tool)
{
    if (!strcmp(tool, "read_file")) return "read";
    if (!strcmp(tool, "list_dir")) return "list";
    if (!strcmp(tool, "write_file") || !strcmp(tool, "edit_file") || !strcmp(tool, "apply_patch")) return "edit";
    if (!strcmp(tool, "run") || !strcmp(tool, "test")) return "bash";
    if (!strcmp(tool, "skill_search")) return "skill";
    return tool;                                   /* glob grep webfetch skill external_directory doom_loop */
}

static int is_path_tool(const char *t)
{
    return !strcmp(t, "read_file") || !strcmp(t, "list_dir") || !strcmp(t, "glob") || !strcmp(t, "grep") ||
           !strcmp(t, "write_file") || !strcmp(t, "edit_file") || !strcmp(t, "apply_patch") || !strcmp(t, "external_directory");
}

/* a Claude Code rule's tool name covers a localcode tool */
static int cc_covers(const char *k, const char *tool)
{
    static const char *map[][2] = { { "read_file", "Read" }, { "list_dir", "LS" }, { "glob", "Glob" }, { "grep", "Grep" },
                                    { "write_file", "Write" }, { "edit_file", "Edit" }, { "apply_patch", "MultiEdit" },
                                    { "run", "Bash" }, { "test", "Bash" }, { "webfetch", "WebFetch" }, { "skill", "Skill" },
                                    { "skill_search", "Skill" } };
    for (size_t i = 0; i < sizeof map / sizeof *map; i++)
        if (!strcmp(map[i][0], tool) && !strcmp(map[i][1], k)) return 1;
    const int edits = !strcmp(tool, "write_file") || !strcmp(tool, "edit_file") || !strcmp(tool, "apply_patch");
    const int reads = !strcmp(tool, "read_file") || !strcmp(tool, "list_dir") || !strcmp(tool, "glob") || !strcmp(tool, "grep");
    return (edits && !strcmp(k, "Edit")) || (reads && !strcmp(k, "Read"));
}

/* a Claude Code pattern against the subject of a tool */
static int cc_match(const char *pat, const char *tool, const char *subj)
{
    if (!strcmp(tool, "run") || !strcmp(tool, "test")) {
        const size_t n = strlen(pat);
        if (n >= 2 && !strcmp(pat + n - 2, ":*")) {  /* Bash(npm run test:*): the command and anything after */
            const size_t k = n - 2;
            return !strncmp(subj, pat, k) && (!subj[k] || subj[k] == ' ');
        }
        return wm(pat, subj, 0, 0);
    }
    if (!strcmp(tool, "webfetch")) {
        if (strncmp(pat, "domain:", 7)) return wm(pat, subj, 0, 1);
        const char *h = strstr(subj, "://");
        h = h ? h + 3 : subj;
        char host[256];
        snprintf(host, sizeof host, "%.*s", (int)strcspn(h, "/:?#"), h);
        return wm(pat + 7, host, 0, 1);
    }
    if (!is_path_tool(tool)) return wm(pat, subj, 0, 0);
    /* gitignore's way: //abs, ~/home, /project-relative, ./relative; a bare name at any depth */
    char p[2400], s[4400];
    const char *home = getenv("USERPROFILE");
    if (!home) home = getenv("HOME");
    const int abs_subj = subj[0] == '/' || (subj[0] && subj[1] == ':');
    if (abs_subj) snprintf(s, sizeof s, "%s", subj);
    else snprintf(s, sizeof s, "%s/%s", g_root, subj);
    if (!strncmp(pat, "//", 2)) {
        const char *q = pat + 2;                   /* //c/Users/x: a drive on Windows */
        if (q[0] && q[1] == '/') snprintf(p, sizeof p, "%c:/%s", q[0], q + 2);
        else snprintf(p, sizeof p, "/%s", q);
    } else if (!strncmp(pat, "~/", 2) && home) {
        snprintf(p, sizeof p, "%s/%s", home, pat + 2);
    } else {
        const char *q = pat;
        if (q[0] == '.' && q[1] == '/') q += 2;
        else if (q[0] == '/') q++;
        if (!strchr(q, '/')) snprintf(p, sizeof p, "%s/**/%s", g_root, q);
        else snprintf(p, sizeof p, "%s/%s", g_root, q);
    }
    for (char *c = p; *c; c++) if (*c == '\\') *c = '/';
    for (char *c = s; *c; c++) if (*c == '\\') *c = '/';
    const size_t pl = strlen(p);                   /* a folder: everything in it */
    if (pl && p[pl - 1] == '/' && pl + 2 < sizeof p) memcpy(p + pl, "**", 3);
    return wm(p, s, 1, PATH_ICASE);
}

static int check_one(const char *tool, const char *subj, char *why, size_t wcap)
{
    const char *k = okey(tool);
    int oc = LC_PERM_NONE, ocr = -1, cc[4] = { 0 }, ccr[4] = { -1, -1, -1, -1 };
    for (int i = 0; i < nR; i++) {
        const Rule *r = &R[i];
        if (!r->act) continue;
        if (!r->claude) {
            if (strcmp(r->key, k) && strcmp(r->key, "*")) continue;
            if (r->pat && !wm(r->pat, subj, 0, is_path_tool(tool) ? PATH_ICASE : 0)) continue;
            oc = r->act;
            ocr = i;
        } else {
            if (!cc_covers(r->key, tool)) continue;
            if (r->pat && !cc_match(r->pat, tool, subj)) continue;
            if (!cc[r->act]) { cc[r->act] = 1; ccr[r->act] = i; }
        }
    }
    int act = LC_PERM_NONE, ri = -1;
    if (oc == LC_PERM_DENY) { act = oc; ri = ocr; }
    else if (cc[LC_PERM_DENY]) { act = LC_PERM_DENY; ri = ccr[LC_PERM_DENY]; }
    else if (cc[LC_PERM_ASK]) { act = LC_PERM_ASK; ri = ccr[LC_PERM_ASK]; }
    else if (cc[LC_PERM_ALLOW]) { act = LC_PERM_ALLOW; ri = ccr[LC_PERM_ALLOW]; }
    else if (oc) { act = oc; ri = ocr; }
    if (why && wcap) {
        if (ri < 0) *why = 0;
        else if (R[ri].claude) snprintf(why, wcap, "%s(%s) in %s", R[ri].key, R[ri].pat ? R[ri].pat : "", R[ri].src);
        else snprintf(why, wcap, "%s \"%s\": %s in %s", R[ri].key, R[ri].pat ? R[ri].pat : "*", act_name(R[ri].act), R[ri].src);
    }
    return act;
}

int lc_perm_check(const char *tool, const char *subject, char *why, size_t wcap)
{
    if (why && wcap) *why = 0;
    if (!subject) subject = "";
    if (strcmp(tool, "run") && strcmp(tool, "test")) return check_one(tool, subject, why, wcap);
    /* a command line: each command of it (split at && || ; | and newlines, outside quotes) */
    const size_t n = strlen(subject);
    char *buf = (char *)malloc(n + 1);
    if (!buf) return LC_PERM_ASK;
    int any_deny = 0, any_ask = 0, all_allow = 1, parts = 0;
    char w[300] = "", wa[300] = "";
    size_t i = 0;
    while (i <= n) {
        size_t k = 0;
        char q = 0;
        for (; i < n; i++) {
            const char c = subject[i];
            if (q) { if (c == q) q = 0; buf[k++] = c; continue; }
            if (c == '\'' || c == '"') { q = c; buf[k++] = c; continue; }
            if (c == ';' || c == '\n' || c == '|' || (c == '&' && subject[i + 1] == '&')) {
                i += (c == '|' && subject[i + 1] == '|') || c == '&' ? 2 : 1;
                break;
            }
            buf[k++] = c;
        }
        if (i >= n) i = n + 1;
        buf[k] = 0;
        char *s = buf;
        while (*s == ' ' || *s == '\t' || *s == '\r') s++;
        size_t e = strlen(s);
        while (e && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) s[--e] = 0;
        if (!*s) continue;
        parts++;
        char one[300];
        const int a = check_one(tool, s, one, sizeof one);
        if (a == LC_PERM_DENY && !any_deny) { any_deny = 1; snprintf(w, sizeof w, "%s", one); }
        if (a == LC_PERM_ASK && !any_ask) { any_ask = 1; snprintf(wa, sizeof wa, "%s", one); }
        if (a != LC_PERM_ALLOW) all_allow = 0;
        else if (!*wa && !*w) snprintf(wa, sizeof wa, "%s", one);
    }
    free(buf);
    if (!parts) return check_one(tool, subject, why, wcap);
    const int act = any_deny ? LC_PERM_DENY : any_ask ? LC_PERM_ASK : all_allow ? LC_PERM_ALLOW : LC_PERM_NONE;
    if (why && wcap && act != LC_PERM_NONE) snprintf(why, wcap, "%s", act == LC_PERM_DENY ? w : wa);
    return act;
}

char *lc_perm_list(void)
{
    size_t cap = 256 + (size_t)nR * 200, n = 0;
    char *s = (char *)malloc(cap);
    if (!s) return NULL;
    n += (size_t)snprintf(s + n, cap - n, "permission rules (a deny wins; then Claude Code's; then opencode's, the last match):\n");
    for (int i = 0; i < nR && n < cap; i++) {
        const Rule *r = &R[i];
        if (r->claude) n += (size_t)snprintf(s + n, cap - n, "  %-5s %s(%s)  · %s\n", act_name(r->act), r->key, r->pat ? r->pat : "", r->src);
        else           n += (size_t)snprintf(s + n, cap - n, "  %-5s %s %s  · %s\n", act_name(r->act), r->key, r->pat ? r->pat : "*", r->src);
    }
    return s;
}
