/* glm53f_chat.h - localcode, the interactive session (glm53f_chat.c, glm53f_tui.c).
 *
 * The caller opens a model and hands it over as a Glm53fEngine with the settings it chose;
 * the session owns the screen until the user leaves. Nothing here changes what the model
 * computes: a turn feeds the new tokens to the same forward pass a one-shot run uses.
 *
 * glm53f_chat.c holds the session itself (commands, a turn, the transcript) and a plain
 * line-by-line front end for pipes; glm53f_tui.c is the full-screen front end used on a
 * terminal. Both drive a turn through the same events. The engine supplies everything
 * model-specific: the forward pass, the tokenizer, the chat template and the reasoning
 * levels it offers (glm53f_run.c wraps GLM-5.3-Flash, q38_localcode.c Qwen3.8-Flash-Next).
 */
#ifndef GLM53F_CHAT_H
#define GLM53F_CHAT_H

#include <stddef.h>
#include <stdint.h>

#define GLM53F_CHAT_MAX_GEN 32768     /* an agent answer: a whole file in one tool call */

/* Expert-cache counters (they only grow in a session); zero when the engine has none. */
typedef struct { uint64_t hits, misses, ahead, bytes; double wait; } Glm53fCounters;

typedef struct { const char *name, *help; } Glm53fLevel;

typedef struct Glm53fEngine {
    const char *name;                    /* "Qwen3.8-Flash-Next" */
    const Glm53fLevel *levels;           /* reasoning levels, as /reasoning takes them; the
                                          * first is the default */
    int         n_levels;
    int         n_vocab;
    int         cached, cap;             /* context positions used / available */
    void       *impl;

    /* Feed n tokens; logits of the last one. Returns 0, or -1 on failure. */
    int  (*forward)(struct Glm53fEngine *e, const int *ids, int n, float *logits);
    void (*reset)(struct Glm53fEngine *e);
    int  (*encode)(struct Glm53fEngine *e, const char *text, int len, int *out, int max);
    int  (*decode)(struct Glm53fEngine *e, const int *ids, int n, char *out, int max);
    int  (*is_eos)(struct Glm53fEngine *e, int id);
    /* The chat-template text of one user message, ending where the assistant starts
     * thinking. first: the message opens the conversation; level: one of levels[].name. */
    void (*turn_text)(struct Glm53fEngine *e, int first, const char *level, const char *msg,
                      char *out, size_t cap);
    /* One line about the engine: GLM53F_DESC_BANNER (model shape), _SETTINGS (cache, device,
     * precision), _STATS (the "engine" line of /stats). */
    void (*describe)(struct Glm53fEngine *e, int what, char *buf, size_t cap);
    void (*counters)(struct Glm53fEngine *e, Glm53fCounters *c);
    /* 1 when the engine's chat template is Qwen's with <tool_call> functions, which the
     * agent (lc_agent.c) writes directly; 0 keeps the session to plain chat. */
    int  agent;
    /* The whole sequence state to a file and back (NULL when the engine cannot): the agent
     * keeps its system prompt read once. Both return the position, or -1; after a failed
     * load the state is undefined (reset). */
    int  (*state_save)(struct Glm53fEngine *e, const char *path);
    int  (*state_load)(struct Glm53fEngine *e, const char *path);
    /* which file of the model is open ("UD-IQ4_XS"): saved states belong to it alone; and
     * the variants on this machine that /model offers (name, "87 GB"), the open one first
     * found by name */
    const char *variant;
    int         n_variants;
    const char *variant_name[16], *variant_info[16];
} Glm53fEngine;

enum { GLM53F_DESC_BANNER, GLM53F_DESC_SETTINGS, GLM53F_DESC_STATS };

typedef struct {
    Glm53fEngine *e;
    const char  *reasoning;      /* one of e->levels[].name */
    int          gen;            /* tokens per answer            */
    int          show_thinking;  /* 1: show the reasoning text, 0: fold it into one line */
    double       load_s;         /* seconds spent opening the model */

    int         *ids, ids_cap;   /* scratch owned by the caller */
    int         *out;
    char        *text;
    int          text_cap;
    float       *logits;

    /* owned by the session */
    int          first;          /* the next message opens the conversation */
    char        *log;            /* the conversation as Markdown, for /save */
    size_t       log_len, log_cap;
    char        *last_answer;    /* the last answer without its reasoning, for /copy */

    /* measured over the session's answers, for /stats and the estimates in the menu */
    int          turns;
    long         pre_tok, dec_tok;
    double       pre_s, dec_s;
    double       dec_wait_s;     /* decode time spent waiting on expert reads  */
    double       dec_read_gb;    /* experts read during decode, both paths     */
    uint64_t     dec_reqs;       /* experts the answers used ...               */
    uint64_t     dec_ahead;      /* ... read from the SSD ahead of their layer */
    uint64_t     dec_demand;     /* ... read from the SSD when asked for       */
    double       last_spt;       /* s/token of the last answer, 0 before one   */
    double       spt_hist[16];   /* s/token of the last answers, oldest first  */
    int          n_spt;
    /* the agent (lc_agent.c): 0 plain chat, 1 quick (reading tools), 2 agent (every tool) */
    int          agent_level;
    int          auto_approve;   /* 1: writes and commands run without asking  */
    int          autonomous;     /* 1: /auto in this folder: no approvals, no questions, to the end */
    unsigned     always;         /* tools the user allowed for the session (bit per tool) */
    /* the conversation on disk and its compaction (lc_session.c) */
    void        *sess;           /* the session file being written; NULL until the first message */
    char        *preface;        /* opens the next message of a new context: a summary, or a
                                  * conversation resumed from its transcript */
    int          compactions;    /* summaries that replaced the context in this conversation */
    int          base;           /* positions the context's opening takes (the agent's system block) */
    const char  *resume;         /* set by the caller: resume this session (as /resume takes it) at start */
    char        *start_note;     /* what that resume says, shown when the screen opens */
    int          switch_to;      /* /model: 1 + the variant to open next (the session ends), 0 none */
} Glm53fChat;

int glm53f_chat_run(Glm53fChat *s);

/* ---- shared by the front ends ---- */

/* Events of one turn, in order: status (reading, thinking; NULL text turns it off), the
 * reasoning text if shown, the thought summary, answer text, and done. */
typedef struct Glm53fTurnSink {
    void (*status)(struct Glm53fTurnSink *k, const char *text, double t0);
    void (*reasoning)(struct Glm53fTurnSink *k, const char *bytes, int n);
    void (*thought)(struct Glm53fTurnSink *k, double seconds, int tokens);
    void (*answer)(struct Glm53fTurnSink *k, const char *bytes, int n);
    void (*done)(struct Glm53fTurnSink *k, const char *stats, int interrupted);
    void *ctx;
    /* agent turns: a tool call carried out (title "edit_file src/a.c", one-line summary, ok),
     * and the question before one that changes files or runs a program: returns 0 no,
     * 1 yes, 2 yes for this tool until the session ends. NULL approve: always no. */
    void (*tool)(struct Glm53fTurnSink *k, const char *title, const char *summary, int ok);
    int  (*approve)(struct Glm53fTurnSink *k, const char *title, const char *detail);
} Glm53fTurnSink;

/* What a line typed by the user asks for. */
typedef struct {
    int   kind;                  /* GLM53F_IN_*                                   */
    char *prompt;                /* MESSAGE: the text for the model, templated    */
    char *shown;                 /* MESSAGE: what the transcript shows            */
    char *note;                  /* NOTE, or a remark before a MESSAGE (/file)    */
    int   compact;               /* MESSAGE from /compact: prompt is what to keep in focus */
} Glm53fInput;

enum { GLM53F_IN_NONE, GLM53F_IN_MESSAGE, GLM53F_IN_NOTE, GLM53F_IN_QUIT, GLM53F_IN_RESET };

/* Interpret one line: a command is carried out (settings, /save, /reset) and answered with
 * a note; a message is wrapped in the chat template. Free with glm53f_chat_input_free. */
void glm53f_chat_input(Glm53fChat *s, const char *line, Glm53fInput *in);
void glm53f_chat_input_free(Glm53fInput *in);

/* Run one turn. g_stop (glm53f_chat_stop) ends it early; returns -1 if the engine failed. */
int  glm53f_chat_turn(Glm53fChat *s, const Glm53fInput *in, Glm53fTurnSink *k);
void glm53f_chat_stop(void);
/* the flag glm53f_chat_stop raises (cleared when a turn starts), for long tool calls */
volatile int *glm53f_chat_stop_flag(void);
/* Append to the session's Markdown transcript (/save). */
void glm53f_chat_log(Glm53fChat *s, const char *text);
/* An agent turn (lc_agent.c); glm53f_chat_turn hands over to it when agent_level > 0. */
int  lc_agent_turn(Glm53fChat *s, const Glm53fInput *in, Glm53fTurnSink *k);

/* ---- lc_session.c: conversations kept on disk, and compaction ----
 * Every conversation is a JSONL file in %LOCALAPPDATA%\localcode\sessions (messages, answers,
 * summaries); leaving localcode also keeps the engine state beside it, so /resume of the last
 * conversations puts the context back exactly, and of older ones rebuilds it from the text.
 * Near the end of the context the model writes a summary of the conversation, the context is
 * emptied and the summary opens the next one: a conversation runs on past the context size,
 * whatever that size is (32k, 128k, 256k). */
void  lc_session_user(Glm53fChat *s, const char *shown);
void  lc_session_answer(Glm53fChat *s, const char *answer, const char *actions, const char *stats);
/* The conversation ends (/reset, /level, leaving); keep_state also keeps the engine state. */
void  lc_session_end(Glm53fChat *s, int keep_state);
char *lc_session_list(Glm53fChat *s);
/* The latest conversations of this folder, one line each ("06/10 14:33  first message"). */
char *lc_session_recent(Glm53fChat *s, int max);
/* /resume [N]: the N-th latest conversation of this folder (default 1); returns the note.
 * *ok is 1 when the conversation changed. */
char *lc_session_resume(Glm53fChat *s, const char *arg, int *ok);
/* 1 when a step of need positions calls for a compaction first: it would leave fewer free
 * than a compaction needs, and the conversation holds enough to be worth summarising. */
int   lc_compact_due(const Glm53fChat *s, int need);
/* Autonomous folders (/auto): the agent works there without asking, remembered across runs. */
int   lc_auto_dir(const char *dir);
int   lc_auto_dir_set(const char *dir, int on);
/* The model summarises the context (request: the request in progress, kept word for word;
 * focus: what /compact asked to keep), the context is emptied and the summary waits in
 * s->preface. Returns 0, 1 when there is no room even for the summary, -1 engine failure. */
int   lc_compact(Glm53fChat *s, Glm53fTurnSink *k, const char *request, const char *focus);

/* "context 88/2048 (4%)" and the settings line, for status rows. */
void glm53f_chat_gauge(const Glm53fChat *s, char *buf, size_t cap);

/* The commands, in the order /help and the menu list them. arg: 0 none, 1 optional, 2
 * required. */
typedef struct {
    const char *name, *args, *help;
    int         arg;
} Glm53fCommand;

extern const Glm53fCommand glm53f_commands[];
extern const int           glm53f_ncommands;

/* What a command's setting stands at ("Max", "2048", "folded"), or "" when it has none. */
void glm53f_chat_command_value(const Glm53fChat *s, const Glm53fCommand *c, char *buf, size_t cap);

/* Full-screen front end; returns 0, or -1 when the terminal cannot host it. */
int  glm53f_tui_run(Glm53fChat *s, int forced);

#endif /* GLM53F_CHAT_H */
