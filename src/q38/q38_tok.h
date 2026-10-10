/* q38_tok.h - the Qwen3.8-Flash-Next tokenizer, built from the GGUF's own vocabulary.
 *
 * tokenizer.ggml.tokens (byte-level strings), tokenizer.ggml.merges ("left right", rank =
 * position) and tokenizer.ggml.token_type fill third_party/tok.h's structures; encoding is
 * tok.h's byte-level BPE with the qwen35 pre-tokenizer and ignore_merges = false, as
 * llama.cpp does for tokenizer.ggml.pre = "qwen35". Control and user-defined tokens
 * (<|im_start|>, <think>, ...) are matched atomically in the text, like llama.cpp's
 * parse_special = true. tools/q38_tok_check.c compares the result with llama.cpp.
 */
#ifndef Q38_TOK_H
#define Q38_TOK_H

#include "q38_gguf.h"

typedef struct q38_tok q38_tok;

q38_tok *q38_tok_load(const q38_gguf *g);
void     q38_tok_free(q38_tok *t);

int  q38_tok_n_vocab(const q38_tok *t);
/* text -> ids; returns the count (at most max). */
int  q38_tok_encode(q38_tok *t, const char *text, int len, int *out, int max);
/* The bytes of one token (control tokens included); returns the length, NUL-terminated. */
int  q38_tok_piece(q38_tok *t, int id, char *out, int max);
/* Id of a control/user-defined token by its text ("<|im_end|>"), or -1. */
int  q38_tok_id(q38_tok *t, const char *text);
/* 1 for a control token (never part of an answer's text). */
int  q38_tok_is_control(const q38_tok *t, int id);

#endif
