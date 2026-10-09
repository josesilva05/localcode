/* lc_web.c - the agent's webfetch: a URL's page as text (opencode's webfetch). HTTP(S) GET through
 * WinHTTP on Windows (curl elsewhere), redirects followed, at most WEB_MAX bytes read; HTML turned into
 * plain text (scripts, styles and markup dropped; headings, list items, paragraphs and links kept). */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#endif

#include "lc_web.h"

#ifndef _WIN32
#include <strings.h>
#define _strnicmp strncasecmp
#endif

#define WEB_MAX (4 << 20)

typedef struct { char *p; size_t n, cap; } Buf;

static void b_add(Buf *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
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

#ifdef _WIN32
static wchar_t *wide(const char *s)
{
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    wchar_t *w = n > 0 ? (wchar_t *)malloc((size_t)n * sizeof(wchar_t)) : NULL;
    if (w) MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    return w;
}

static int fetch(const char *url, Buf *body, char *ctype, size_t ccap, int *status, char *err, size_t ecap)
{
    wchar_t *wu = wide(url);
    URL_COMPONENTS uc;
    memset(&uc, 0, sizeof uc);
    uc.dwStructSize = sizeof uc;
    wchar_t host[512], path[4096];
    uc.lpszHostName = host; uc.dwHostNameLength = 512;
    uc.lpszUrlPath = path; uc.dwUrlPathLength = 4096;
    wchar_t extra[4096];
    uc.lpszExtraInfo = extra; uc.dwExtraInfoLength = 4096;
    if (!wu || !WinHttpCrackUrl(wu, 0, 0, &uc)) { free(wu); snprintf(err, ecap, "not a valid URL"); return -1; }
    free(wu);
    wchar_t full[8192];
    _snwprintf(full, 8192, L"%s%s", path[0] ? path : L"/", extra);
    full[8191] = 0;
    HINTERNET s = WinHttpOpen(L"localcode/0.4 (webfetch)", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                              WINHTTP_NO_PROXY_BYPASS, 0);
    if (!s) { snprintf(err, ecap, "WinHTTP failed to start"); return -1; }
    WinHttpSetTimeouts(s, 10000, 10000, 20000, 30000);
    HINTERNET c = WinHttpConnect(s, host, uc.nPort, 0);
    HINTERNET r = c ? WinHttpOpenRequest(c, L"GET", full, NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                         uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0) : NULL;
    int rc = -1;
    if (r && WinHttpSendRequest(r, L"Accept: text/html,text/plain,application/json,*/*\r\n", (DWORD)-1L, WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
        && WinHttpReceiveResponse(r, NULL)) {
        DWORD code = 0, sz = sizeof code;
        WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code, &sz, WINHTTP_NO_HEADER_INDEX);
        *status = (int)code;
        wchar_t ct[256];
        DWORD cts = sizeof ct;
        ctype[0] = 0;
        if (WinHttpQueryHeaders(r, WINHTTP_QUERY_CONTENT_TYPE, WINHTTP_HEADER_NAME_BY_INDEX, ct, &cts, WINHTTP_NO_HEADER_INDEX))
            WideCharToMultiByte(CP_UTF8, 0, ct, -1, ctype, (int)ccap, NULL, NULL);
        for (;;) {
            DWORD avail = 0, got = 0;
            if (!WinHttpQueryDataAvailable(r, &avail) || !avail) break;
            char buf[16384];
            if (!WinHttpReadData(r, buf, avail < sizeof buf ? avail : sizeof buf, &got) || !got) break;
            b_add(body, buf, got);
            if (body->n > WEB_MAX) break;
        }
        rc = 0;
    } else {
        snprintf(err, ecap, "the request failed (error %lu)", (unsigned long)GetLastError());
    }
    if (r) WinHttpCloseHandle(r);
    if (c) WinHttpCloseHandle(c);
    WinHttpCloseHandle(s);
    return rc;
}
#else
static int fetch(const char *url, Buf *body, char *ctype, size_t ccap, int *status, char *err, size_t ecap)
{
    char cmd[4400];
    for (const char *p = url; *p; p++) if (*p == '\'') { snprintf(err, ecap, "not a valid URL"); return -1; }
    snprintf(cmd, sizeof cmd, "curl -sL --max-time 30 --max-filesize %d -w '\\n%%{http_code} %%{content_type}' '%s'", WEB_MAX, url);
    FILE *p = popen(cmd, "r");
    if (!p) { snprintf(err, ecap, "curl failed"); return -1; }
    char buf[16384];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, p)) > 0) b_add(body, buf, got);
    pclose(p);
    char *nl = body->p ? strrchr(body->p, '\n') : NULL;
    if (!nl) { snprintf(err, ecap, "no answer"); return -1; }
    *status = atoi(nl + 1);
    const char *sp = strchr(nl + 1, ' ');
    snprintf(ctype, ccap, "%s", sp ? sp + 1 : "");
    *nl = 0;
    body->n = (size_t)(nl - body->p);
    return 0;
}
#endif

/* &amp; and the like */
static size_t entity(const char *p, Buf *o)
{
    static const struct { const char *e, *t; } E[] = { { "&amp;", "&" }, { "&lt;", "<" }, { "&gt;", ">" }, { "&quot;", "\"" },
                                                       { "&#39;", "'" }, { "&apos;", "'" }, { "&nbsp;", " " }, { "&mdash;", "—" },
                                                       { "&ndash;", "–" }, { "&hellip;", "…" }, { "&copy;", "©" } };
    for (size_t i = 0; i < sizeof E / sizeof *E; i++) {
        const size_t l = strlen(E[i].e);
        if (!strncmp(p, E[i].e, l)) { b_str(o, E[i].t); return l; }
    }
    if (p[1] == '#') {                            /* &#NNN; &#xHH; */
        const char *q = p + 2;
        unsigned long cp = q[0] == 'x' || q[0] == 'X' ? strtoul(q + 1, (char **)&q, 16) : strtoul(q, (char **)&q, 10);
        if (*q == ';' && cp > 0 && cp < 0x110000) {
            char u[5];
            int n = 0;
            if (cp < 0x80) u[n++] = (char)cp;
            else if (cp < 0x800) { u[n++] = (char)(0xC0 | cp >> 6); u[n++] = (char)(0x80 | (cp & 63)); }
            else if (cp < 0x10000) { u[n++] = (char)(0xE0 | cp >> 12); u[n++] = (char)(0x80 | (cp >> 6 & 63)); u[n++] = (char)(0x80 | (cp & 63)); }
            else { u[n++] = (char)(0xF0 | cp >> 18); u[n++] = (char)(0x80 | (cp >> 12 & 63)); u[n++] = (char)(0x80 | (cp >> 6 & 63)); u[n++] = (char)(0x80 | (cp & 63)); }
            b_add(o, u, (size_t)n);
            return (size_t)(q - p) + 1;
        }
    }
    b_add(o, "&", 1);
    return 1;
}

static int tag_is(const char *t, size_t n, const char *name)
{
    const size_t l = strlen(name);
    return n >= l && !_strnicmp(t, name, l) && (n == l || !isalnum((unsigned char)t[l]));
}

/* HTML as text: markup dropped, blocks on lines of their own, headings as "# ", items as "- ", links as
 * "text (url)" */
static char *html_text(const char *h, size_t n)
{
    Buf o = { 0 }, href = { 0 };
    int skip = 0, in_link = 0;
    for (size_t i = 0; i < n;) {
        if (h[i] == '<') {
            size_t j = i + 1;
            if (!strncmp(h + i, "<!--", 4)) { const char *e = strstr(h + i + 4, "-->"); i = e ? (size_t)(e - h) + 3 : n; continue; }
            while (j < n && h[j] != '>') j++;
            const char *t = h + i + 1;
            const int close = *t == '/';
            if (close) t++;
            const size_t tl = (size_t)(h + j - t);
            if (tag_is(t, tl, "script") || tag_is(t, tl, "style") || tag_is(t, tl, "noscript") || tag_is(t, tl, "svg") || tag_is(t, tl, "head"))
                skip = close ? 0 : 1;
            else if (!skip) {
                if (tag_is(t, tl, "br") || tag_is(t, tl, "p") || tag_is(t, tl, "div") || tag_is(t, tl, "tr") || tag_is(t, tl, "section") ||
                    tag_is(t, tl, "article") || tag_is(t, tl, "pre") || tag_is(t, tl, "table") || tag_is(t, tl, "ul") || tag_is(t, tl, "ol"))
                    b_str(&o, "\n");
                else if (!close && (tag_is(t, tl, "h1") || tag_is(t, tl, "h2") || tag_is(t, tl, "h3") || tag_is(t, tl, "h4")))
                    b_str(&o, "\n\n# ");
                else if (close && (tag_is(t, tl, "h1") || tag_is(t, tl, "h2") || tag_is(t, tl, "h3") || tag_is(t, tl, "h4")))
                    b_str(&o, "\n");
                else if (!close && tag_is(t, tl, "li")) b_str(&o, "\n- ");
                else if ((tag_is(t, tl, "td") || tag_is(t, tl, "th")) && !close) b_str(&o, " | ");
                else if (tag_is(t, tl, "a")) {
                    if (!close) {
                        href.n = 0;
                        const char *hr = NULL;
                        for (const char *q = t; q + 5 < h + j; q++) if (!_strnicmp(q, "href=", 5)) { hr = q + 5; break; }
                        if (hr) {
                            const char qch = *hr == '"' || *hr == '\'' ? *hr++ : ' ';
                            const char *e = hr;
                            while (e < h + j && *e != qch && *e != '>') e++;
                            if (e > hr && *hr != '#' && strncmp(hr, "javascript:", 11)) b_add(&href, hr, (size_t)(e - hr));
                        }
                        in_link = 1;
                    } else if (in_link) {
                        if (href.n) { b_str(&o, " ("); b_add(&o, href.p, href.n); b_str(&o, ")"); }
                        in_link = 0;
                    }
                }
            }
            i = j < n ? j + 1 : n;
            continue;
        }
        if (skip) { i++; continue; }
        if (h[i] == '&') { i += entity(h + i, &o); continue; }
        if (h[i] == '\r' || h[i] == '\t' || h[i] == '\n') { if (o.n && o.p[o.n - 1] != ' ' && o.p[o.n - 1] != '\n') b_str(&o, " "); i++; continue; }
        if (h[i] == ' ' && o.n && (o.p[o.n - 1] == ' ' || o.p[o.n - 1] == '\n')) { i++; continue; }
        b_add(&o, h + i, 1);
        i++;
    }
    free(href.p);
    /* blank lines folded: at most one empty line in a row, lines trimmed */
    Buf f = { 0 };
    int blank = 0;
    for (char *line = o.p; line && *line;) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        while (*line == ' ') line++;
        size_t l = strlen(line);
        while (l && line[l - 1] == ' ') l--;
        if (l) { b_add(&f, line, l); b_str(&f, "\n"); blank = 0; }
        else if (!blank && f.n) { b_str(&f, "\n"); blank = 1; }
        line = nl ? nl + 1 : NULL;
    }
    free(o.p);
    return f.p ? f.p : calloc(1, 1);
}

char *lc_webfetch(const char *url, int cap, char *sum, size_t sc, int *ok)
{
    *ok = 0;
    if (!url || (strncmp(url, "http://", 7) && strncmp(url, "https://", 8))) {
        snprintf(sum, sc, "not an http(s) URL");
        return strdup("webfetch takes an http:// or https:// URL");
    }
    Buf body = { 0 };
    char ctype[256] = "", err[300] = "";
    int status = 0;
    if (fetch(url, &body, ctype, sizeof ctype, &status, err, sizeof err)) {
        free(body.p);
        snprintf(sum, sc, "%s", err);
        char *r = (char *)malloc(400);
        if (r) snprintf(r, 400, "could not fetch %s: %s", url, err);
        return r;
    }
    const int html = strstr(ctype, "html") != NULL || (body.p && body.n > 15 && (strstr(body.p, "<html") || strstr(body.p, "<!DOCTYPE") || strstr(body.p, "<!doctype")));
    char *text = html ? html_text(body.p ? body.p : "", body.n) : (body.p ? body.p : calloc(1, 1));
    if (html) free(body.p);
    const size_t tl = strlen(text);
    Buf r = { 0 };
    char head[600];
    snprintf(head, sizeof head, "%s · HTTP %d · %s%s\n\n", url, status, ctype[0] ? ctype : "unknown type", html ? " (as text)" : "");
    b_str(&r, head);
    if ((int)tl > cap) {
        b_add(&r, text, (size_t)cap);
        char tail[160];
        snprintf(tail, sizeof tail, "\n[... %zu more characters not shown]", tl - (size_t)cap);
        b_str(&r, tail);
    } else {
        b_str(&r, text);
    }
    free(text);
    snprintf(sum, sc, "HTTP %d · %zu chars", status, tl);
    *ok = status >= 200 && status < 400;
    return r.p;
}
