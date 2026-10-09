/* lc_plugins_check.c - what localcode finds of the plugins (src/cli/lc_plugins.c), for a folder:
 * the time to find them, /plugin's report, the skills list of the prompt and skill_search's answers.
 *
 *     lc_plugins_check [folder] [query ...]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lc_plugins.h"
#include "lc_tools.h"

int main(int argc, char **argv)
{
    if (lc_tools_init(argc > 1 ? argv[1] : NULL) != 0) { fprintf(stderr, "no such folder\n"); return 1; }
    const clock_t t0 = clock();
    lc_plugins_load(lc_tools_root());
    printf("found in %.2f s\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
    char *r = lc_plugins_report();
    printf("%s\n", r);
    free(r);
    const char *p = lc_skills_block();
    printf("---- the prompt's list (%zu characters):\n%s\n", strlen(p), p);
    int cmds = lc_commands_n();
    printf("---- %d commands, the first:\n", cmds);
    for (int i = 0; i < cmds && i < 12; i++) {
        const LcCommand *c = lc_command_at(i);
        printf("/%s:%s  %s %.80s\n", c->plugin, c->name, c->hint, c->desc);
    }
    for (int i = 2; i < argc; i++) {
        char *s = lc_skill_search(argv[i], 6);
        printf("---- skill_search(%s):\n%s\n", argv[i], s);
        free(s);
    }
    return 0;
}
