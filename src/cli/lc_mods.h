/* lc_mods.h - localcode's mods: Claude Code mod modules (a plugin whose hooks.json is {"modules":
 * [...]}: claude-code-playground's blast-radius, replay-theater, token-weather) run by modhost.mjs in
 * Node, beside localcode, talking by JSON lines (see modhost.mjs).
 *
 * Only the mods of the plugins the user allowed run (/plugin hooks on NAME). Their events: session.start
 * (when localcode opens), turn.start / turn.complete (each answer), tool.call (before each tool: a mod
 * may hold it, ask the user in a pane, and refuse it), command.run (the commands they register), and
 * ui.render: what they draw above the prompt (AbovePrompt) and in a pane, as lines with buttons.
 */
#ifndef LC_MODS_H
#define LC_MODS_H

#define LC_MOD_LINES 64
#define LC_MOD_BUTTONS 16

typedef struct {
    char *lines[LC_MOD_LINES];  /* with ANSI colours, each at most the width asked for */
    int   n;
    struct { char id[24], hotkey[8], label[80]; } button[LC_MOD_BUTTONS];
    int   nb, focus;           /* focus: the button Enter presses */
} LcModView;

/* Starts the mods the user allowed (once; root: the working directory) and sends session.start.
 * on_change is called (from another thread) when what they draw changed. 0, or -1 when none run. */
int  lc_mods_start(const char *root, void (*on_change)(void));
int  lc_mods_running(void);
void lc_mods_stop(void);

/* The context's use, for $.session.usage(). */
void lc_mods_usage(int tokens, int window);

/* An event and its result (the whole reply line, malloc'd; NULL on no reply within timeout_ms).
 * data: a JSON object. pump, when given, is called every ~50 ms while waiting (the plain front end
 * answers a pane there). */
char *lc_mods_event(const char *event, const char *data, int timeout_ms, void (*pump)(void));

/* What a tool call becomes: 0 go on, 1 refused (*why: the mod's reason, malloc'd). tool: Claude Code's
 * name, input: its tool_input JSON. */
int  lc_mods_tool_call(const char *tool, const char *input, char **why, void (*pump)(void));

/* What the mods draw: "AbovePrompt" or the open pane ("Pane"), cols wide, at most rows; 0 when
 * nothing. view is lc_mods_view_free'd by the caller. */
int  lc_mods_view(const char *component, int cols, int rows, LcModView *view);
void lc_mods_view_free(LcModView *v);

/* The open pane: 1 and its title when one is open; *esc_closes: Escape closes it (else Escape stops
 * what waits on it: lc_mods_abort). */
int  lc_mods_pane(char *title, int cap, int *esc_closes);
/* The event being waited on is not wanted any more (Escape): the mod is told, and answers. */
void lc_mods_abort(void);
/* A button pressed (its id), the pane closed (Escape). */
void lc_mods_press(const char *id);
/* The pane's focused button (the one Enter presses), by its place among the pane's buttons. */
void lc_mods_focus(int k);
void lc_mods_pane_close(void);
/* A toast the mods showed (malloc'd), or NULL. */
char *lc_mods_toast(void);

/* The commands the mods registered (/replay): count, name and description; command.run's text. */
int  lc_mods_commands(void);
const char *lc_mods_command_name(int i);
const char *lc_mods_command_desc(int i);
char *lc_mods_command_run(const char *name, const char *args);

#endif
