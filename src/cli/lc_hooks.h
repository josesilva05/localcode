/* lc_hooks.h - Claude Code's hooks in localcode: programs run at the agent's events.
 *
 * Where they come from: the plugins the user allowed (/plugin hooks on NAME: their hooks/hooks.json
 * or the manifest's "hooks"), and the project's .claude/settings.json, .claude/settings.local.json
 * and .localcode/settings.json ("hooks"). The format is Claude Code's: {"EventName": [{"matcher":
 * "Write|Edit", "hooks": [{"type": "command", "command": "...", "args": [...], "timeout": 10}]}]}.
 *
 * A hook runs through Git Bash (as Claude Code does on Windows), or as its program with "args",
 * with the event as JSON on its input and CLAUDE_PLUGIN_ROOT / CLAUDE_PROJECT_DIR set. Exit code 2
 * blocks (its stderr says why); on exit 0 its output may be JSON (decision, reason, systemMessage,
 * hookSpecificOutput.additionalContext / permissionDecision) or, for SessionStart and
 * UserPromptSubmit, plain text for the model. Matchers name localcode's tools by Claude Code's names
 * (Bash, Read, Write, Edit, Glob, Grep, LS, Skill).
 */
#ifndef LC_HOOKS_H
#define LC_HOOKS_H

#include "lc_tools.h"

enum { LC_HOOK_SESSION_START, LC_HOOK_USER_PROMPT, LC_HOOK_PRE_TOOL, LC_HOOK_POST_TOOL, LC_HOOK_STOP,
       LC_HOOK_SESSION_END, LC_HOOK_PRE_COMPACT, LC_HOOK_N };

typedef struct {
    int   ran;               /* hooks that ran */
    int   block;             /* one blocked: exit 2, decision "block", permissionDecision "deny", continue false */
    int   decision;          /* PreToolUse: 0 none, 1 allow (no question), 2 deny, 3 ask (a question even under /auto) */
    char *reason;            /* why it blocks */
    char *context;           /* text for the model */
    char *message;           /* for the user: systemMessage, a hook that failed */
} LcHookOut;

/* Finds the hooks (once; root: the working directory). */
void lc_hooks_load(const char *root);
/* How many hooks an event has. */
int  lc_hooks_any(int ev);
/* Runs the event's hooks whose matcher takes match (a tool's Claude Code name, a session's source).
 * fields: the event's own JSON members ("\"prompt\": \"...\""), added to the common ones. Fills o
 * (lc_hook_out_free); returns how many ran. */
int  lc_hooks_run(int ev, const char *match, const char *fields, LcHookOut *o);
void lc_hook_out_free(LcHookOut *o);

/* A call's tool as Claude Code names it ("run" -> "Bash"), and its tool_input as JSON (malloc'd). */
const char *lc_hook_tool_name(const char *tool);
char *lc_hook_tool_input(const LcCall *c);
/* s as a JSON string, quotes included (malloc'd). */
char *lc_json_quote(const char *s);
/* A line about the hooks for /plugin. */
char *lc_hooks_report(void);

#endif
