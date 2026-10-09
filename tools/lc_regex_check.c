/* lc_regex_check - lc_regex against a reference: each case of the file is
 *     pattern \x1f icase \x1f text \x1f expected \n
 * (expected 1 match, 0 no match, E not a pattern), written by tools/lc_regex_cases.py from Python's
 * re. Prints the cases that differ and a count; exit 1 on any difference.
 *     lc_regex_check cases.txt */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lc_regex.h"

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: lc_regex_check cases.txt\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) return 2;
    b[n] = 0;
    fclose(f);
    int cases = 0, bad = 0, errs = 0;
    for (char *line = b; line && *line;) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char *fld[4] = { line, NULL, NULL, NULL };
        int k = 1;
        for (char *q = line; *q && k < 4; q++) if (*q == '\x1f') { *q = 0; fld[k++] = q + 1; }
        if (k == 4) {
            cases++;
            char err[200];
            LcRegex *re = lc_regex_new(fld[0], fld[1][0] == '1', err, sizeof err);
            const char want = fld[3][0];
            char got;
            if (!re) got = 'E';
            else got = lc_regex_search(re, fld[2], strlen(fld[2])) ? '1' : '0';
            if (want == 'E') errs++;
            /* a pattern Python refuses may be refused here for another reason, or (for a few
             * Python-only refusals) accepted: only accepted-by-both cases must agree exactly */
            if (got != want && !(want == 'E' && got == 'E')) {
                if (bad < 40) printf("DIFF pattern=[%s] icase=%s text=[%s] want=%c got=%c%s%s\n", fld[0], fld[1], fld[2], want,
                                     got, re ? "" : " err=", re ? "" : err);
                bad++;
            }
            lc_regex_free(re);
        }
        line = nl ? nl + 1 : NULL;
    }
    printf("%d cases, %d refused by both or expected refused, %d differ\n", cases, errs, bad);
    free(b);
    return bad ? 1 : 0;
}
