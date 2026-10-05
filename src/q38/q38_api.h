/* q38_api.h - the engine behind a C ABI, for servers in other languages (tools/q38_server.py).
 *
 * One model per process: open it, turn text into tokens, feed tokens and read the logits of
 * the last one, and save / restore the sequence state so a later request that repeats the
 * same prompt continues from where it ended instead of reading it again.
 */
#ifndef Q38_API_H
#define Q38_API_H

#ifdef _WIN32
#define Q38API __declspec(dllexport)
#else
#define Q38API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q38api q38api;

/* model: any shard of the split GGUF. ctx: tokens the sequence may hold (0: 8192).
 * gpu: device for the trunk (-1: try 1 then 0; -2: CPU only). preload_gb: experts read into
 * RAM at start from the usage profile (%LOCALAPPDATA%\localcode\qwen38_expert_profile.bin).
 * Returns NULL with the reason printed. */
Q38API q38api *q38api_open(const char *model, int ctx, int gpu, double preload_gb);
/* Saves the usage profile, frees everything. */
Q38API void    q38api_close(q38api *h);

Q38API int     q38api_n_ctx(q38api *h);
Q38API int     q38api_n_vocab(q38api *h);
Q38API int     q38api_n_past(q38api *h);
/* A string from the GGUF metadata (e.g. "tokenizer.chat_template") into out; returns its
 * length, or -1 when absent (out gets what fits, NUL-terminated). */
Q38API int     q38api_meta_str(q38api *h, const char *key, char *out, int cap);
Q38API long long q38api_meta_int(q38api *h, const char *key, long long def);

/* text -> ids (control tokens such as <|im_start|> are matched in the text); returns the count. */
Q38API int     q38api_tokenize(q38api *h, const char *text, int len, int *out, int max);
/* The bytes of one token; returns the length. */
Q38API int     q38api_piece(q38api *h, int id, char *out, int cap);
Q38API int     q38api_is_control(q38api *h, int id);
/* Id of a control token by its text, or -1. */
Q38API int     q38api_token_id(q38api *h, const char *text);

/* Feeds n tokens at the current position (a batched pass when n > 1); the logits of the last
 * one go to logits (n_vocab floats) when it is not NULL. Returns 0 or -1. */
Q38API int     q38api_eval(q38api *h, const int *toks, int n, float *logits);
/* Back to an empty sequence. */
Q38API void    q38api_reset(q38api *h);
/* Remembers the sequence state at the current position; returns that position, or -1 (no
 * GPU). q38api_restore goes back to it: returns the position, or -1 when nothing is saved. */
Q38API int     q38api_snapshot(q38api *h);
Q38API int     q38api_restore(q38api *h);
/* Writes the usage profile now (also done by q38api_close). Returns 0 or -1. */
Q38API int     q38api_save_profile(q38api *h);

#ifdef __cplusplus
}
#endif

#endif
