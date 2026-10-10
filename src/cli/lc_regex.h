/* lc_regex.h - the regular expressions of the agent's grep (lc_regex.c): the common syntax of
 * ripgrep and Python's re, matched by a Pike VM (time linear in the text, no catastrophic
 * backtracking):
 *   .  [abc] [^a-z]  \d \D \w \W \s \S  \b \B  ^ $  ( ) (?: )  |  * + ? {m} {m,} {m,n} (and lazy forms)
 *   \n \t \r \xHH, \ before any other punctuation for itself
 * Bytes, not characters: a UTF-8 letter is a run of bytes (it matches itself literally; . matches
 * one byte of it). No backreferences or lookaround. */
#ifndef LC_REGEX_H
#define LC_REGEX_H

#include <stddef.h>

typedef struct LcRegex LcRegex;

/* NULL when the pattern is not one (err gets why). icase: ASCII letters match either case. */
LcRegex *lc_regex_new(const char *pattern, int icase, char *err, size_t ecap);
/* 1 when the regex matches somewhere in s[0..n) (one line: ^ and $ are its ends). */
int  lc_regex_search(LcRegex *re, const char *s, size_t n);
void lc_regex_free(LcRegex *re);

#endif
