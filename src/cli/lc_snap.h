/* lc_snap.h - /undo and /redo of the agent's changes, as opencode does them: before each agent turn the
 * working directory is committed into a shadow git repository of its own
 * (%LOCALAPPDATA%\localcode\snapshots\<hash of the folder>), outside the project's git, honouring the
 * project's .gitignore. /undo puts the files back as they were before the last turn (what run's
 * commands changed too: files changed, added or deleted), /redo puts that turn's result back.
 * Nothing happens without git. */
#ifndef LC_SNAP_H
#define LC_SNAP_H

/* Before an agent turn: the folder as it is now. 0, or -1 (no git, or it failed). */
int  lc_snap_turn(void);
/* /undo, /redo: a line for the screen (malloc'd); *changed: files put back. */
char *lc_snap_undo(int *changed);
char *lc_snap_redo(int *changed);
/* What the model is told at its next turn after an undo or a redo (once), or NULL. */
const char *lc_snap_note(void);

#endif
