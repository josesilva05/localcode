/* lc_plugins.h - localcode's plugins: Claude Code's plugin format (skills, commands, agents, hooks),
 * found in the system, user and project folders and switched on or off by the user.
 *
 * A plugin is a folder with .claude-plugin/plugin.json, a plugin entry of a collection's
 * .claude-plugin/marketplace.json, or a plain folder of skills. Its parts are found where the
 * manifest says and, as Claude Code does, also in its own folders: every SKILL.md below it (a skill:
 * a header with name and description, then the instructions), the .md files of its commands/ folders,
 * of its agents/ folders, and hooks/hooks.json. Other agents' copies (.codex, .cursor ...), tests and
 * node_modules are not searched. See docs/localcode-plugins-plan.md.
 *
 * Where (the first of a name wins): the project's .localcode/plugins, .localcode/skills and
 * .claude/skills; the user's %USERPROFILE%\.localcode\plugins and \skills; the system's
 * %LOCALAPPDATA%\localcode\plugins and \skills\vendor (the skills localcode ships with).
 * On or off: %LOCALAPPDATA%\localcode\plugins.txt, lines "plugin NAME on|off" and
 * "skill PLUGIN:NAME on|off"; without a line, the defaults (lc_plugins.c).
 */
#ifndef LC_PLUGINS_H
#define LC_PLUGINS_H

typedef struct {
    char *name, *desc, *dir;     /* dir: the folder holding SKILL.md */
    int   plugin;                /* index into the plugins */
    int   model;                 /* 0: disable-model-invocation (the user calls it, as /name) */
    int   on, featured;          /* featured: listed first in the prompt */
} LcSkill;

typedef struct {
    char *name, *plugin, *desc, *hint, *file;   /* a command's .md file; hint: argument-hint */
    int   pi;
} LcCommand;

typedef struct {
    char *name, *desc, *version, *dir, *origin; /* origin: "system", "user", "project" */
    int   on;
    int   n_skills, n_commands, n_agents, hooks, mod;   /* mod: hooks are JavaScript modules */
    int   trusted;               /* its hooks (or mod) may run: the user said so (/plugin hooks on) */
} LcPlugin;

/* Finds everything (once; root: the working directory). */
void lc_plugins_load(const char *root);

int             lc_plugins_n(void);
const LcPlugin *lc_plugin_at(int i);
int             lc_skills_n(void);
const LcSkill  *lc_skill_at(int i);
int             lc_commands_n(void);
const LcCommand *lc_command_at(int i);

/* A skill that is on, by "name" or "plugin:name"; NULL when none. */
const LcSkill *lc_skill_find(const char *name);
/* The model's list of skills that are on, within a budget of characters (the featured first), and
 * a last line about skill_search when some did not fit; "" when no skill is on. */
const char *lc_skills_prompt(int budget);
/* Up to max skills that are on matching the words of query, one "plugin:name: description" a line. */
char *lc_skill_search(const char *query, int max);
/* A plugin (by name) or one skill ("plugin:name") on or off, kept in plugins.txt. 0, or -1 when
 * there is no such plugin or skill. */
int lc_plugin_switch(const char *what, int on);
/* A plugin's hooks (programs it runs at the agent's events) allowed or not, kept in plugins.txt; they
 * are not allowed until the user says so. 0, or -1 when there is no such plugin. */
int lc_plugin_trust(const char *name, int on);
/* /plugin's text: every plugin with its parts, on or off, where it is. */
char *lc_plugins_report(void);

/* What /name calls command i of a plugin that is on: name, or plugin:name when another command has
 * the name; 0 when its plugin is off. */
int lc_command_label(int i, char *out, int cap);
/* A typed "/name args" as the message it stands for: a plugin's command (its text with $ARGUMENTS,
 * $1.. and ${CLAUDE_PLUGIN_ROOT} put in), or a skill called by the user (the model is asked to load
 * it and follow it with the arguments). malloc'd, or NULL when no command or skill is named so;
 * *note gets one line for the screen. */
char *lc_command_expand(const char *line, char **note);

#endif
