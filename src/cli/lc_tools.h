/* lc_tools.h - localcode's agent tools: what the model can do in the working directory.
 *
 * Every path a tool takes is relative to the working directory (the folder localcode was
 * started in) and must stay inside it. Results are plain text for the model, capped so one
 * call cannot fill the context. The tools are described to the model in the chat template's
 * <tools> block (lc_tools_json) and called as <tool_call><function=NAME><parameter=...>.
 */
#ifndef LC_TOOLS_H
#define LC_TOOLS_H
#include <stddef.h>

#define LC_MAX_PARAMS 12

/* One parsed call: the function name and its parameters as text. */
typedef struct {
    char  name[64];
    char *key[LC_MAX_PARAMS];
    char *val[LC_MAX_PARAMS];
    int   n;
} LcCall;

void        lc_call_free(LcCall *c);
/* The parameter's text, or NULL when the call does not give it. */
const char *lc_call_arg(const LcCall *c, const char *key);

/* The working directory the tools are confined to; NULL keeps the current one. */
int  lc_tools_init(const char *root);
const char *lc_tools_root(void);

/* The tools of an agent level, one JSON object per line, for the <tools> block:
 * level 1 the reading tools, level 2 and up every tool. */
const char *lc_tools_json(int level);
/* 1 when a level offers the tool. */
int  lc_tool_allowed(const char *name, int level);
/* 1 when the tool changes files or runs a program, so it asks before it does. */
int  lc_tool_needs_approval(const char *name);
/* One line saying what a call will do ("edit_file src/main.c", "run: cmake --build build"). */
void lc_tool_title(const LcCall *c, char *buf, size_t cap);

/* Carry out a call. Returns the text for the model (malloc'd, never NULL unless out of
 * memory); summary gets one line for the screen; *ok 0 when the call failed. stop is
 * polled by long calls (run) and ends them early. */
char *lc_tool_run(const LcCall *c, volatile int *stop, char *summary, size_t scap, int *ok);

#endif
