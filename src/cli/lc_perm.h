/* lc_perm.h - permission rules for the agent's calls (lc_perm.c): allow, ask or deny by tool and by
 * pattern, in opencode's format and in Claude Code's.
 *
 * opencode ("permission" in opencode.json, or in .localcode/settings.json):
 *     "permission": { "edit": "ask", "bash": { "git *": "allow", "rm *": "deny", "*": "ask" },
 *                     "read": { "*.env": "deny" }, "webfetch": "allow",
 *                     "external_directory": "ask", "doom_loop": "ask" }
 *   keys: read list glob grep edit bash webfetch skill external_directory doom_loop, "*" for all;
 *   in a pattern * is any text; the last rule that matches wins.
 * Claude Code ("permissions" in .claude/settings.json, .claude/settings.local.json or
 * .localcode/settings.json):
 *     "permissions": { "allow": ["Bash(npm run test:*)", "Edit(src/**)"], "ask": [...],
 *                      "deny": ["Read(./.env)", "WebFetch(domain:example.com)"] }
 *   deny wins over ask, ask over allow; Edit rules cover every tool that changes files.
 * Read from ~/.localcode/settings.json, then the project's .claude/settings.json,
 * .claude/settings.local.json, .localcode/settings.json and opencode.json(c) (never ~/.claude: those
 * rules were given to Claude Code). Built in, below everything: reading .env files asks (but
 * .env.example), a path outside the working directory asks, and the same call three times in a row
 * asks. A deny from any side wins; then Claude Code's rules; then opencode's. A command with
 * && || ; | is allowed only when every part is. */
#ifndef LC_PERM_H
#define LC_PERM_H

#include <stddef.h>

enum { LC_PERM_NONE, LC_PERM_ALLOW, LC_PERM_DENY, LC_PERM_ASK };

/* The rules read again (the files may have changed); root: the project. */
void lc_perm_load(const char *root);
/* What the rules say of a call: tool is localcode's name (or external_directory, doom_loop),
 * subject the command, the path relative to the project (absolute outside it) or the URL.
 * why gets the rule that decided ("" for none). */
int  lc_perm_check(const char *tool, const char *subject, char *why, size_t wcap);
/* The rules, one per line, for /permissions (malloc'd). */
char *lc_perm_list(void);

#endif
