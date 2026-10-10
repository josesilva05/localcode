/* lc_web.h - the agent's webfetch (lc_web.c). */
#ifndef LC_WEB_H
#define LC_WEB_H

#include <stddef.h>

/* The page at url (http or https) as text, at most cap characters of it, with a first line naming the
 * URL, the HTTP status and the type (malloc'd); sum gets a line for the screen; *ok 0 on failure. */
char *lc_webfetch(const char *url, int cap, char *sum, size_t sc, int *ok);

#endif
