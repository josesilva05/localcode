/* lc_plugins.c - see lc_plugins.h. */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "lc_plugins.h"
#include "lc_tools.h"

#ifdef _WIN32
#define SEP "\\"
#else
#define SEP "/"
#endif

/* ------------------------------------------------------------------- helpers ---- */

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

static void b_str(Buf *b, const char *s) { b_add(b, s, strlen(s)); }

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

static char *xfmt(const char *fmt, ...)
{
    char tmp[4400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    return xdup(tmp);
}

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
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

static jval *json_file(const char *path)
{
    char *t = lc_slurp(path, NULL);
    if (!t) return NULL;
    jval *v = json_parse(t, NULL);
    free(t);
    return v;
}

static const char *jstr(jval *o, const char *key)
{
    jval *v = json_get(o, key);
    return v && v->t == J_STR ? v->str : NULL;
}

/* a relative path of a manifest ("./skills/x/", "skills") joined to dir, with the native separators */
static char *join_rel(const char *dir, const char *rel)
{
    while (rel[0] == '.' && (rel[1] == '/' || rel[1] == '\\')) rel += 2;
    char *p = xfmt("%s" SEP "%s", dir, rel);
    if (!p) return NULL;
    size_t n = strlen(p);
    for (size_t i = 0; i < n; i++) if (p[i] == '/' || p[i] == '\\') p[i] = SEP[0];
    while (n > 1 && p[n - 1] == SEP[0]) p[--n] = 0;
    if (n >= 2 && p[n - 1] == '.' && p[n - 2] == SEP[0]) p[n - 2] = 0;   /* "./" itself */
    return p;
}

/* ------------------------------------------------- the header of a .md file ---- */

/* The value of key in the YAML header between the first two "---" lines (malloc'd), or NULL. The
 * forms the skills use: plain, quoted ('...' or "...", over several lines too), block scalars (| and
 * >) and plain values continued on indented lines; the lines are joined with spaces. */
static char *fm_get(const char *text, const char *key)
{
    if (strncmp(text, "---", 3)) return NULL;
    const char *p = strchr(text, '\n');
    if (!p) return NULL;
    p++;
    const size_t kl = strlen(key);
    while (*p) {
        const char *nl = strchr(p, '\n');
        const char *end = nl ? nl : p + strlen(p);
        if (!strncmp(p, "---", 3)) return NULL;
        if (!strncmp(p, key, kl) && p[kl] == ':') {
            const char *v = p + kl + 1;
            while (v < end && (*v == ' ' || *v == '\t')) v++;
            size_t n = (size_t)(end - v);
            while (n && (v[n - 1] == '\r' || v[n - 1] == ' ' || v[n - 1] == '\t')) n--;
            Buf b = { 0 };
            if (n && (v[0] == '"' || v[0] == '\'')) {          /* quoted, maybe over lines */
                const char q = v[0];
                for (const char *c = v + 1; *c; c++) {
                    if (q == '"' && c[0] == '\\' && c[1]) { b_add(&b, c + 1, 1); c++; continue; }
                    if (*c == q) {
                        if (q == '\'' && c[1] == '\'') { b_add(&b, "'", 1); c++; continue; }
                        break;
                    }
                    if (*c == '\r') continue;
                    if (*c == '\n') { b_add(&b, " ", 1); while (c[1] == ' ' || c[1] == '\t') c++; continue; }
                    b_add(&b, c, 1);
                }
                return b.p ? b.p : xdup("");
            }
            const int block = n && (v[0] == '|' || v[0] == '>');
            if (!block && n) b_add(&b, v, n);
            for (const char *q = nl ? nl + 1 : end; *q;) {      /* the indented lines after it */
                const char *nl2 = strchr(q, '\n');
                const char *e2 = nl2 ? nl2 : q + strlen(q);
                const char *t = q;
                while (t < e2 && (*t == ' ' || *t == '\t')) t++;
                const int blank = t >= e2 || *t == '\r';
                if (!blank && t == q) break;                    /* the next key, or the end */
                size_t m = (size_t)(e2 - t);
                while (m && (t[m - 1] == '\r' || t[m - 1] == ' ')) m--;
                if (m) { if (b.n) b_add(&b, " ", 1); b_add(&b, t, m); }
                q = nl2 ? nl2 + 1 : e2;
            }
            return b.p ? b.p : xdup("");
        }
        p = nl ? nl + 1 : end;
    }
    return NULL;
}

/* -------------------------------------------------------------- the registry ---- */

static LcPlugin  *P;
static LcSkill   *S;
static LcCommand *C;
static int nP, capP, nS, capS, nC, capC;
static int loaded;

#define GROW(arr, n, cap) do { if ((n) == (cap)) { int nc_ = (cap) ? (cap) * 2 : 32; \
    void *g_ = realloc((arr), (size_t)nc_ * sizeof *(arr)); if (!g_) return -1; (arr) = g_; (cap) = nc_; } } while (0)

static int plugin_find(const char *name)
{
    for (int i = 0; i < nP; i++) if (!strcmp(P[i].name, name)) return i;
    return -1;
}

static int plugin_new(const char *name, const char *desc, const char *version, const char *dir, const char *origin)
{
    if (plugin_find(name) >= 0) return -1;                 /* the first of a name wins */
    GROW(P, nP, capP);
    LcPlugin *p = &P[nP];
    memset(p, 0, sizeof *p);
    p->name = xdup(name);
    p->desc = xdup(desc ? desc : "");
    p->version = xdup(version ? version : "");
    p->dir = xdup(dir);
    p->origin = xdup(origin);
    p->on = 1;
    return nP++;
}

/* the skill whose SKILL.md is in dir, for plugin pi (one name once a plugin) */
static int skill_add(int pi, const char *dir, const char *folder)
{
    char *path = xfmt("%s" SEP "SKILL.md", dir);
    char *text = path ? lc_slurp(path, NULL) : NULL;
    free(path);
    if (!text) return -1;
    char *name = fm_get(text, "name"), *desc = fm_get(text, "description");
    char *dmi = fm_get(text, "disable-model-invocation");
    free(text);
    if (!name || !*name) { free(name); name = xdup(folder); }
    for (int i = 0; i < nS; i++)
        if (S[i].plugin == pi && !strcmp(S[i].name, name)) { free(name); free(desc); free(dmi); return -1; }
    GROW(S, nS, capS);
    LcSkill *s = &S[nS++];
    memset(s, 0, sizeof *s);
    s->name = name;
    s->desc = desc ? desc : xdup("");
    s->dir = xdup(dir);
    s->plugin = pi;
    s->model = !(dmi && !strcmp(dmi, "true"));
    free(dmi);
    P[pi].n_skills++;
    return 0;
}

/* the .md files of a commands folder; a subfolder's are "sub:name" */
static int commands_add(int pi, const char *dir, const char *prefix, int depth)
{
    int n = 0;
    LcEnt *v = lc_read_dir(dir, &n);
    for (int i = 0; i < n; i++) {
        char *path = xfmt("%s" SEP "%s", dir, v[i].name);
        if (!path) continue;
        if (v[i].dir) {
            if (depth < 2 && v[i].name[0] != '.') {
                char *pre = prefix[0] ? xfmt("%s:%s", prefix, v[i].name) : xdup(v[i].name);
                if (pre) commands_add(pi, path, pre, depth + 1);
                free(pre);
            }
            free(path);
            continue;
        }
        const size_t len = strlen(v[i].name);
        if (len < 4 || strcmp(v[i].name + len - 3, ".md") || !strcmp(v[i].name, "README.md")) { free(path); continue; }
        char *text = lc_slurp(path, NULL);
        if (nC == capC) {
            const int nc = capC ? capC * 2 : 64;
            void *g = realloc(C, (size_t)nc * sizeof *C);
            if (!g) { free(text); free(path); break; }
            C = (LcCommand *)g;
            capC = nc;
        }
        LcCommand *c = &C[nC++];
        memset(c, 0, sizeof *c);
        c->name = prefix[0] ? xfmt("%s:%.*s", prefix, (int)(len - 3), v[i].name) : xfmt("%.*s", (int)(len - 3), v[i].name);
        c->plugin = xdup(P[pi].name);
        c->pi = pi;
        c->file = path;
        c->desc = text ? fm_get(text, "description") : NULL;
        c->hint = text ? fm_get(text, "argument-hint") : NULL;
        if (!c->desc) c->desc = xdup("");
        if (!c->hint) c->hint = xdup("");
        free(text);
        P[pi].n_commands++;
    }
    lc_ents_free(v, n);
    return 0;
}

static int count_md(const char *dir)
{
    int n = 0, k = 0;
    LcEnt *v = lc_read_dir(dir, &n);
    for (int i = 0; i < n; i++) {
        const size_t len = strlen(v[i].name);
        if (!v[i].dir && len > 3 && !strcmp(v[i].name + len - 3, ".md") && strcmp(v[i].name, "README.md")) k++;
    }
    lc_ents_free(v, n);
    return k;
}

/* folders a plugin's walk does not enter: other agents' copies of the same skills, tests, packages */
static int skip_dir(const char *name)
{
    if (name[0] == '.') return strcmp(name, ".claude") != 0;
    static const char *no[] = { "node_modules", "tests", "test", "testdata", "fixtures", "__pycache__", "dist", "examples",
                                "benchmarks", "evals", "media", "assets", "screenshots" };
    for (size_t i = 0; i < sizeof no / sizeof *no; i++) if (!strcmp(name, no[i])) return 1;
    return 0;
}

/* every skill (a folder with SKILL.md, not entered further), commands/ and agents/ folder below dir */
static void walk(int pi, const char *dir, int depth)
{
    int n = 0;
    LcEnt *v = lc_read_dir(dir, &n);
    for (int i = 0; i < n; i++)
        if (!v[i].dir && !strcmp(v[i].name, "SKILL.md")) {
            const char *base = strrchr(dir, SEP[0]);
            skill_add(pi, dir, base ? base + 1 : dir);
            lc_ents_free(v, n);
            return;
        }
    for (int i = 0; i < n && depth < 7; i++) {
        if (!v[i].dir || skip_dir(v[i].name)) continue;
        char *path = xfmt("%s" SEP "%s", dir, v[i].name);
        if (!path) continue;
        if (!strcmp(v[i].name, "commands")) commands_add(pi, path, "", 0);
        else if (!strcmp(v[i].name, "agents")) P[pi].n_agents += count_md(path);
        else walk(pi, path, depth + 1);
        free(path);
    }
    lc_ents_free(v, n);
}

/* hooks: hooks/hooks.json, or the manifest's "hooks" (a path or the hooks themselves); a mod's are
 * JavaScript modules ({"modules": [...]}) */
static void hooks_note(int pi, const char *dir, jval *man)
{
    jval *h = man ? json_get(man, "hooks") : NULL;
    char *path = NULL;
    if (h && h->t == J_STR) path = join_rel(dir, h->str);
    else if (h && h->t == J_OBJ) P[pi].hooks = 1;
    if (!path) path = xfmt("%s" SEP "hooks" SEP "hooks.json", dir);
    jval *hj = path ? json_file(path) : NULL;
    if (hj) {
        P[pi].hooks = 1;
        if (json_get(hj, "modules")) P[pi].mod = 1;
        json_free(hj);
    }
    free(path);
}

/* one plugin at dir: its manifest (plugin.json, or the collection's entry), its parts */
static void plugin_load(const char *dir, const char *origin, jval *entry, const char *fallback)
{
    char *mp = xfmt("%s" SEP ".claude-plugin" SEP "plugin.json", dir);
    jval *man = mp ? json_file(mp) : NULL;
    free(mp);
    const char *name = entry && jstr(entry, "name") ? jstr(entry, "name") : man && jstr(man, "name") ? jstr(man, "name") : fallback;
    const char *desc = entry && jstr(entry, "description") ? jstr(entry, "description") : man ? jstr(man, "description") : NULL;
    const char *ver = man ? jstr(man, "version") : entry ? jstr(entry, "version") : NULL;
    const int pi = name ? plugin_new(name, desc, ver, dir, origin) : -1;
    if (pi < 0) { json_free(man); return; }
    jval *list = entry ? json_get(entry, "skills") : NULL;    /* a collection's own list: only those */
    if (list && list->t == J_ARR) {
        for (int i = 0; i < list->len; i++) {
            if (list->kids[i]->t != J_STR) continue;
            char *d = join_rel(dir, list->kids[i]->str);
            if (d) walk(pi, d, 0);
            free(d);
        }
    } else {
        jval *sk = man ? json_get(man, "skills") : NULL;     /* the manifest's paths, then its folders */
        for (int i = 0; sk && i < (sk->t == J_ARR ? sk->len : 1); i++) {
            jval *e = sk->t == J_ARR ? sk->kids[i] : sk;
            if (e->t != J_STR) continue;
            char *d = join_rel(dir, e->str);
            if (d) walk(pi, d, 0);
            free(d);
        }
        walk(pi, dir, 0);
    }
    hooks_note(pi, dir, man);
    json_free(man);
}

/* a collection: .claude-plugin/marketplace.json's plugins (a source outside the folder, as a
 * repository on GitHub, is this folder: the clone it was found in) */
static int marketplace_load(const char *dir, const char *origin)
{
    char *mp = xfmt("%s" SEP ".claude-plugin" SEP "marketplace.json", dir);
    jval *m = mp ? json_file(mp) : NULL;
    free(mp);
    jval *pl = m ? json_get(m, "plugins") : NULL;
    int k = 0;
    for (int i = 0; pl && pl->t == J_ARR && i < pl->len; i++) {
        jval *e = pl->kids[i];
        const char *src = jstr(e, "source");
        char *sd = src ? join_rel(dir, src) : xdup(dir);
        if (!sd) continue;
        plugin_load(sd, origin, e, NULL);
        free(sd);
        k++;
    }
    json_free(m);
    return k;
}

/* the folders of a root: collections, plugins, or folders of skills (named after the folder,
 * "owner__repo" as "owner/repo") */
static void root_load(const char *root, const char *origin)
{
    int n = 0;
    LcEnt *v = lc_read_dir(root, &n);
    for (int i = 0; i < n; i++) {
        if (!v[i].dir || v[i].name[0] == '.') continue;
        char *d = xfmt("%s" SEP "%s", root, v[i].name);
        if (!d) continue;
        char *mk = xfmt("%s" SEP ".claude-plugin" SEP "marketplace.json", d);
        char *pj = xfmt("%s" SEP ".claude-plugin" SEP "plugin.json", d);
        int done = 0;
        if (mk && file_exists(mk)) done = marketplace_load(d, origin) > 0;
        if (!done && pj && file_exists(pj)) { plugin_load(d, origin, NULL, v[i].name); done = 1; }
        if (!done) {
            /* a collection one level down (claude-code/mods/.claude-plugin/marketplace.json) */
            int m = 0;
            LcEnt *w = lc_read_dir(d, &m);
            for (int j = 0; j < m && !done; j++) {
                if (!w[j].dir || skip_dir(w[j].name)) continue;
                char *sub = xfmt("%s" SEP "%s", d, w[j].name);
                for (int k = 0; sub && k < 2 && !done; k++) {
                    int q = 0;
                    LcEnt *x = k ? lc_read_dir(sub, &q) : NULL;
                    char *cand = k == 0 ? xdup(sub) : NULL;
                    for (int t = 0; k == 1 && t < q && !done; t++) {
                        if (!x[t].dir || skip_dir(x[t].name)) continue;
                        char *c2 = xfmt("%s" SEP "%s", sub, x[t].name);
                        char *mk2 = c2 ? xfmt("%s" SEP ".claude-plugin" SEP "marketplace.json", c2) : NULL;
                        if (mk2 && file_exists(mk2)) done = marketplace_load(c2, origin) > 0;
                        free(mk2);
                        free(c2);
                    }
                    if (cand) {
                        char *mk2 = xfmt("%s" SEP ".claude-plugin" SEP "marketplace.json", cand);
                        if (mk2 && file_exists(mk2)) done = marketplace_load(cand, origin) > 0;
                        free(mk2);
                    }
                    free(cand);
                    if (x) lc_ents_free(x, q);
                }
                free(sub);
            }
            lc_ents_free(w, m);
        }
        if (!done) {
            char name[512];
            snprintf(name, sizeof name, "%s", v[i].name);
            char *u = strstr(name, "__");
            if (u) { *u = '/'; memmove(u + 1, u + 2, strlen(u + 2) + 1); }
            plugin_load(d, origin, NULL, name);
        }
        free(mk);
        free(pj);
        free(d);
    }
    lc_ents_free(v, n);
}

/* a folder whose subfolders are skills (.claude/skills): one plugin of them */
static void skills_root_load(const char *dir, const char *name, const char *origin)
{
    int n = 0;
    LcEnt *v = lc_read_dir(dir, &n);
    lc_ents_free(v, n);
    if (n <= 0) return;
    const int pi = plugin_new(name, "", "", dir, origin);
    if (pi >= 0) walk(pi, dir, 0);
}

/* ------------------------------------------------------------- on and off ---- */

/* The guide's items (docs/localcode-plugins-plan.md): the skills listed first in the prompt (and every
 * skill of the featured plugins), and the collections of which only some skills are on by default
 * (the rest is there for /plugin on, or other languages and services). */
static const char *FEATURED_PLUGINS[] = { "superpowers", NULL };
static const char *FEATURED[] = { "find-skills", "skill-creator", "grill-me", "git-guardrails-claude-code", "caveman",
                                  "last30days", "planning-with-files", "openspec-propose", "openspec-explore",
                                  "openspec-apply-change", "spec-driven-development", "allium", "improve-codebase-architecture",
                                  "frontend-design", "ui-ux-pro-max", "theme-factory", "canvas-design", "web-artifacts-builder",
                                  "webapp-testing", "playwright-skill", "mcp-builder", "claude-api", "brand-guidelines",
                                  "internal-comms", "algorithmic-art", "deckset-presentation-expert", "printing-press",
                                  "security-reviewer", "audit-context-building", NULL };
static const char *DEFAULT_OFF[] = { "academy-guide", "discernment-nudge", "document-skills", NULL };
static const struct { const char *plugin, *skills; } DEFAULT_ONLY[] = {
    { "mattpocock-skills", " grill-me grilling git-guardrails-claude-code improve-codebase-architecture " },
    { "example-skills", " skill-creator frontend-design theme-factory canvas-design web-artifacts-builder webapp-testing "
                        "mcp-builder brand-guidelines internal-comms algorithmic-art " },
    { "planning-with-files", " planning-with-files " },
    { "caveman", " caveman caveman-commit caveman-review caveman-compress caveman-help " },
};

static int in_list(const char **l, const char *s)
{
    for (int i = 0; l[i]; i++) if (!strcmp(l[i], s)) return 1;
    return 0;
}

static char *state_path(void)
{
#ifdef _WIN32
    const char *b = getenv("LOCALAPPDATA");
    return b ? xfmt("%s\\localcode\\plugins.txt", b) : NULL;
#else
    const char *h = getenv("HOME");
    return h ? xfmt("%s/.local/share/localcode/plugins.txt", h) : NULL;
#endif
}

static void defaults_apply(void)
{
    for (int i = 0; i < nP; i++) P[i].on = !in_list(DEFAULT_OFF, P[i].name);
    for (int i = 0; i < nS; i++) {
        const char *pn = P[S[i].plugin].name;
        S[i].on = 1;
        for (size_t k = 0; k < sizeof DEFAULT_ONLY / sizeof *DEFAULT_ONLY; k++)
            if (!strcmp(pn, DEFAULT_ONLY[k].plugin)) {
                char key[160];
                snprintf(key, sizeof key, " %s ", S[i].name);
                S[i].on = strstr(DEFAULT_ONLY[k].skills, key) != NULL;
            }
        S[i].featured = (in_list(FEATURED_PLUGINS, pn) || in_list(FEATURED, S[i].name)) && S[i].on;
    }
    char *sp = state_path();
    char *t = sp ? lc_slurp(sp, NULL) : NULL;
    free(sp);
    for (char *line = t; line && *line;) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char kind[16], what[600], val[8];
        if (sscanf(line, "%15s %599s %7s", kind, what, val) == 3) {
            const int on = !strcmp(val, "on");
            if (!strcmp(kind, "hooks")) {
                const int pi = plugin_find(what);
                if (pi >= 0) P[pi].trusted = on;
            } else if (!strcmp(kind, "plugin")) {
                const int pi = plugin_find(what);
                if (pi >= 0) P[pi].on = on;
            } else if (!strcmp(kind, "skill")) {
                char *colon = strrchr(what, ':');
                if (colon) {
                    *colon = 0;
                    const int pi = plugin_find(what);
                    for (int i = 0; pi >= 0 && i < nS; i++) if (S[i].plugin == pi && !strcmp(S[i].name, colon + 1)) S[i].on = on;
                }
            }
        }
        line = nl ? nl + 1 : NULL;
    }
    free(t);
}

void lc_plugins_load(const char *root)
{
    if (loaded) return;
    loaded = 1;
    char *d;
    if (root && *root) {
        if ((d = xfmt("%s" SEP ".localcode" SEP "plugins", root))) { root_load(d, "project"); free(d); }
        if ((d = xfmt("%s" SEP ".localcode" SEP "skills", root))) { skills_root_load(d, "project-skills", "project"); free(d); }
        if ((d = xfmt("%s" SEP ".claude" SEP "skills", root))) { skills_root_load(d, "project-claude-skills", "project"); free(d); }
    }
#ifdef _WIN32
    const char *home = getenv("USERPROFILE"), *sys = getenv("LOCALAPPDATA");
#else
    const char *home = getenv("HOME"), *sys = NULL;
#endif
    if (home) {
        if ((d = xfmt("%s" SEP ".localcode" SEP "plugins", home))) { root_load(d, "user"); free(d); }
        if ((d = xfmt("%s" SEP ".localcode" SEP "skills", home))) { skills_root_load(d, "user-skills", "user"); free(d); }
    }
    if (sys) {
        if ((d = xfmt("%s" SEP "localcode" SEP "plugins", sys))) { root_load(d, "system"); free(d); }
        if ((d = xfmt("%s" SEP "localcode" SEP "skills" SEP "vendor", sys))) { root_load(d, "system"); free(d); }
    }
    defaults_apply();
}

int             lc_plugins_n(void) { return nP; }
const LcPlugin *lc_plugin_at(int i) { return i >= 0 && i < nP ? &P[i] : NULL; }
int             lc_skills_n(void) { return nS; }
const LcSkill  *lc_skill_at(int i) { return i >= 0 && i < nS ? &S[i] : NULL; }
int             lc_commands_n(void) { return nC; }
const LcCommand *lc_command_at(int i) { return i >= 0 && i < nC ? &C[i] : NULL; }

static int skill_live(int i) { return S[i].on && P[S[i].plugin].on; }

/* a plugin's short name, its prefix in plugin:name ("ChristopherKahler/paul" -> "paul") */
static const char *pshort(int pi)
{
    const char *s = strrchr(P[pi].name, '/');
    return s ? s + 1 : P[pi].name;
}

/* the n characters at s name plugin pi, by its name or its short name */
static int pmatch(int pi, const char *s, size_t n)
{
    const char *sh = pshort(pi);
    return (strlen(P[pi].name) == n && !strncmp(P[pi].name, s, n)) || (strlen(sh) == n && !strncmp(sh, s, n));
}

const LcSkill *lc_skill_find(const char *name)
{
    if (!name) return NULL;
    const char *colon = strrchr(name, ':');
    for (int pass = 0; pass < 2; pass++)        /* "plugin:name" first, then a skill named all of it */
        for (int i = 0; i < nS; i++) {
            if (!skill_live(i)) continue;
            if (pass == 0 && colon) {
                const size_t pl = (size_t)(colon - name);
                if (pmatch(S[i].plugin, name, pl) && !strcmp(S[i].name, colon + 1))
                    return &S[i];
            } else if (pass == 1 && !strcmp(S[i].name, name)) {
                return &S[i];
            }
        }
    return NULL;
}

/* the name the model calls it by: plugin:name when another skill that is on has the same name */
static void skill_label(int i, char *out, size_t cap)
{
    int dup = 0;
    for (int j = 0; j < nS && !dup; j++) dup = j != i && skill_live(j) && !strcmp(S[j].name, S[i].name);
    if (dup) snprintf(out, cap, "%s:%s", pshort(S[i].plugin), S[i].name);
    else snprintf(out, cap, "%s", S[i].name);
}

const char *lc_skills_prompt(int budget)
{
    static Buf b = { 0 };
    static int built = 0;
    if (built) return b.p ? b.p : "";
    built = 1;
    int shown = 0, left = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < nS; i++) {
            if (!skill_live(i) || !S[i].model || S[i].featured != (pass == 0)) continue;
            char lab[700], line[1200];
            skill_label(i, lab, sizeof lab);
            const char *d = S[i].desc[0] ? S[i].desc : "(no description)";
            snprintf(line, sizeof line, "- %s: %.*s%s\n", lab, 130, d, strlen(d) > 130 ? "..." : "");
            if ((int)(b.n + strlen(line)) > budget) { left++; continue; }
            b_str(&b, line);
            shown++;
        }
    if (left) b_fmt(&b, "- ... and %d more skills: find them with skill_search (words of the task)\n", left);
    return b.p ? b.p : "";
}

/* lowercase words of 3+ letters */
static int words(const char *s, char w[][32], int max)
{
    /* words every description has */
    static const char *stop[] = { "the", "and", "for", "use", "using", "when", "with", "this", "that", "from", "your",
                                  "you", "any", "are", "into", "how", "what", "about", "make", "need", "want", "skill" };
    int n = 0;
    while (*s && n < max) {
        while (*s && !isalnum((unsigned char)*s)) s++;
        int k = 0;
        while (*s && isalnum((unsigned char)*s)) { if (k < 31) w[n][k++] = (char)tolower((unsigned char)*s); s++; }
        w[n][k] = 0;
        int sw = 0;
        for (size_t i = 0; i < sizeof stop / sizeof *stop; i++) sw |= !strcmp(w[n], stop[i]);
        if (k >= 3 && !sw) {
            /* a crude stem: tests -> test, presentations -> prese(ntation...), writing -> writi */
            if (k > 4 && w[n][k - 1] == 's') w[n][--k] = 0;
            if (k > 5) w[n][5] = 0;
            n++;
        }
    }
    return n;
}

static int has_word(const char *text, const char *w)
{
    const size_t n = strlen(w);
    for (const char *p = text; *p; p++) {
        size_t i = 0;
        while (i < n && p[i] && tolower((unsigned char)p[i]) == w[i]) i++;
        if (i == n) return 1;
    }
    return 0;
}

char *lc_skill_search(const char *query, int max)
{
    char w[16][32];
    const int nw = words(query ? query : "", w, 16);
    double *score = (double *)calloc((size_t)(nS ? nS : 1), sizeof(double));
    Buf b = { 0 };
    if (!score) return xdup("out of memory");
    /* a word found in few skills counts more (its inverse document frequency) */
    double idf[16];
    int live = 0;
    for (int i = 0; i < nS; i++) live += skill_live(i) && S[i].model;
    for (int k = 0; k < nw; k++) {
        int df = 0;
        for (int i = 0; i < nS; i++)
            if (skill_live(i) && S[i].model && (has_word(S[i].name, w[k]) || has_word(S[i].desc, w[k]))) df++;
        idf[k] = df ? log(1.0 + (double)live / df) : 0.0;
    }
    for (int i = 0; i < nS; i++) {
        if (!skill_live(i) || !S[i].model) continue;
        for (int k = 0; k < nw; k++)
            score[i] += idf[k] * (3 * has_word(S[i].name, w[k]) + 2 * has_word(S[i].desc, w[k]) + has_word(P[S[i].plugin].name, w[k]));
    }
    for (int r = 0; r < max; r++) {
        int best = -1;
        for (int i = 0; i < nS; i++) if (score[i] > 0 && (best < 0 || score[i] > score[best])) best = i;
        if (best < 0) break;
        char lab[700];
        skill_label(best, lab, sizeof lab);
        b_fmt(&b, "- %s: %.*s\n", lab, 400, S[best].desc);
        score[best] = 0;
    }
    free(score);
    if (!b.p) b_str(&b, "no skill matches those words; try others, or go on without one");
    return b.p;
}

int lc_plugin_switch(const char *what, int on)
{
    char kind[8];
    const char *colon = strrchr(what, ':');
    int found = 0;
    if (colon && plugin_find(what) < 0) {
        snprintf(kind, sizeof kind, "skill");
        const size_t pl = (size_t)(colon - what);
        for (int i = 0; i < nS; i++)
            if (pmatch(S[i].plugin, what, pl) && !strcmp(S[i].name, colon + 1)) {
                S[i].on = on;
                found = 1;
            }
    } else {
        snprintf(kind, sizeof kind, "plugin");
        const int pi = plugin_find(what);
        if (pi >= 0) { P[pi].on = on; found = 1; }
    }
    if (!found) return -1;
    /* plugins.txt: this line replaced or added */
    char *sp = state_path();
    if (!sp) return 0;
    char *t = lc_slurp(sp, NULL);
    Buf b = { 0 };
    char head[700];
    snprintf(head, sizeof head, "%s %s ", kind, what);
    for (char *line = t; line && *line;) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (strncmp(line, head, strlen(head)) && *line) { b_str(&b, line); b_str(&b, "\n"); }
        line = nl ? nl + 1 : NULL;
    }
    free(t);
    b_fmt(&b, "%s%s\n", head, on ? "on" : "off");
    FILE *f = fopen(sp, "wb");
    if (f) { fwrite(b.p, 1, b.n, f); fclose(f); }
    free(sp);
    free(b.p);
    return 0;
}

static int state_line(const char *kind, const char *what, int on)
{
    char *sp = state_path();
    if (!sp) return 0;
    char *t = lc_slurp(sp, NULL);
    Buf b = { 0 };
    char head[700];
    snprintf(head, sizeof head, "%s %s ", kind, what);
    for (char *line = t; line && *line;) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (strncmp(line, head, strlen(head)) && *line) { b_str(&b, line); b_str(&b, "\n"); }
        line = nl ? nl + 1 : NULL;
    }
    free(t);
    b_fmt(&b, "%s%s\n", head, on ? "on" : "off");
    FILE *f = fopen(sp, "wb");
    if (f) { fwrite(b.p, 1, b.n, f); fclose(f); }
    free(sp);
    free(b.p);
    return 0;
}

int lc_plugin_trust(const char *name, int on)
{
    const int pi = plugin_find(name);
    if (pi < 0) return -1;
    P[pi].trusted = on;
    return state_line("hooks", name, on);
}

char *lc_plugins_report(void)
{
    Buf b = { 0 };
    int on = 0, sk = 0, skon = 0;
    for (int i = 0; i < nP; i++) on += P[i].on;
    for (int i = 0; i < nS; i++) { sk++; skon += skill_live(i); }
    b_fmt(&b, "%d plugins (%d on), %d skills (%d on), %d commands\n", nP, on, sk, skon, nC);
    for (int i = 0; i < nP; i++) {
        int son = 0;
        for (int j = 0; j < nS; j++) son += S[j].plugin == i && S[j].on;
        b_fmt(&b, "%s %-34s %3d skills (%d on)", P[i].on ? "on " : "off", P[i].name, P[i].n_skills, son);
        if (P[i].n_commands) b_fmt(&b, ", %d commands", P[i].n_commands);
        if (P[i].n_agents) b_fmt(&b, ", %d agents", P[i].n_agents);
        if (P[i].hooks) b_fmt(&b, ", %s %s", P[i].mod ? "mod" : "hooks", P[i].trusted ? "allowed" : "not allowed");
        b_fmt(&b, " · %s\n", P[i].origin);
    }
    return b.p ? b.p : xdup("");
}

/* ------------------------------------------------------------- commands ---- */

int lc_command_label(int i, char *out, int cap)
{
    if (i < 0 || i >= nC || !P[C[i].pi].on) return 0;
    int dup = 0;
    for (int j = 0; j < nC && !dup; j++) dup = j != i && P[C[j].pi].on && !strcmp(C[j].name, C[i].name);
    if (dup) snprintf(out, (size_t)cap, "%s:%s", pshort(C[i].pi), C[i].name);
    else snprintf(out, (size_t)cap, "%s", C[i].name);
    return 1;
}

/* the text of a .md after its header */
static const char *md_body(const char *t)
{
    if (strncmp(t, "---", 3)) return t;
    const char *e = strstr(t + 3, "\n---");
    if (!e) return t;
    e += 4;
    while (*e && *e != '\n') e++;
    while (*e == '\n' || *e == '\r') e++;
    return e;
}

/* the arguments one by one ("quoted" ones kept whole) */
static int split_args(const char *a, char out[][256], int max)
{
    int n = 0;
    while (*a && n < max) {
        while (*a == ' ') a++;
        if (!*a) break;
        int k = 0;
        if (*a == '"') { a++; while (*a && *a != '"') { if (k < 255) out[n][k++] = *a; a++; } if (*a) a++; }
        else while (*a && *a != ' ') { if (k < 255) out[n][k++] = *a; a++; }
        out[n++][k] = 0;
    }
    return n;
}

char *lc_command_expand(const char *line, char **note)
{
    if (note) *note = NULL;
    if (!line || line[0] != '/') return NULL;
    const char *sp = strchr(line, ' ');
    char name[300];
    snprintf(name, sizeof name, "%.*s", (int)(sp ? (size_t)(sp - line - 1) : strlen(line + 1)), line + 1);
    const char *args = sp ? sp + 1 : "";
    while (*args == ' ') args++;
    int ci = -1;
    for (int i = 0; i < nC && ci < 0; i++) {           /* plugin:name, then the label */
        if (!P[C[i].pi].on) continue;
        char full[600], sh[600], lab[600];
        snprintf(full, sizeof full, "%s:%s", C[i].plugin, C[i].name);
        snprintf(sh, sizeof sh, "%s:%s", pshort(C[i].pi), C[i].name);
        if (!strcmp(full, name) || !strcmp(sh, name) || (lc_command_label(i, lab, sizeof lab) && !strcmp(lab, name))) ci = i;
    }
    if (ci >= 0) {
        char *text = lc_slurp(C[ci].file, NULL);
        if (!text) return NULL;
        char av[9][256];
        const int na = split_args(args, av, 9);
        Buf b = { 0 };
        for (const char *p = md_body(text); *p;) {
            if (!strncmp(p, "$ARGUMENTS", 10)) { b_str(&b, args); p += 10; continue; }
            if (p[0] == '$' && p[1] >= '1' && p[1] <= '9') {
                const int k = p[1] - '1';
                if (k < na) b_str(&b, av[k]);
                p += 2;
                continue;
            }
            if (!strncmp(p, "${CLAUDE_PLUGIN_ROOT}", 21)) { b_str(&b, P[C[ci].pi].dir); p += 21; continue; }
            b_add(&b, p, 1);
            p++;
        }
        free(text);
        if (note) *note = xfmt("/%s from the plugin %s", C[ci].name, C[ci].plugin);
        return b.p ? b.p : xdup("");
    }
    const LcSkill *sk = lc_skill_find(name);              /* a skill the user calls */
    if (!sk) return NULL;
    if (note) *note = xfmt("the skill %s of the plugin %s", sk->name, P[sk->plugin].name);
    return xfmt("Call skill with name \"%s:%s\" first and follow it%s%s", pshort(sk->plugin), sk->name,
                *args ? " for this: " : ".", args);
}
