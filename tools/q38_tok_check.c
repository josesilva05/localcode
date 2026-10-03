/* q38_tok_check - compare the engine's tokenizer with llama.cpp's on a text file.
 *
 *   q38_tok_check <model.gguf> <corpus.txt>
 *
 * Tokenizes every line of the corpus (with its newline) and then the whole file with both
 * tokenizers (special tokens parsed), and requires identical ids. Also checks that
 * decoding the ids gives the text back. Built when CMake gets -DQ38_LLAMA_CPP_DIR.
 */
#include "llama.h"
#include "q38_gguf.h"
#include "q38_tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(q38_tok *qt, const struct llama_vocab *v, const char *s, int len, int *bad, int show)
{
    static int a[1 << 20], b[1 << 20];
    int na = q38_tok_encode(qt, s, len, a, 1 << 20);
    int nb = llama_tokenize(v, s, len, b, 1 << 20, false, true);
    int same = na == nb && !memcmp(a, b, (size_t)na * sizeof(int));
    /* round trip through the engine's pieces */
    static char back[1 << 22];
    int o = 0;
    char piece[512];
    for (int i = 0; i < na; i++) {
        int n = q38_tok_piece(qt, a[i], piece, sizeof piece);
        if (o + n < (int)sizeof back) { memcpy(back + o, piece, (size_t)n); o += n; }
    }
    int round = o == len && !memcmp(back, s, (size_t)len);
    if (!same || !round) {
        (*bad)++;
        if (show) {
            printf("MISMATCH%s%s: \"%.*s\"\n  engine:", same ? "" : " ids", round ? "" : " roundtrip", len > 120 ? 120 : len, s);
            for (int i = 0; i < na && i < 40; i++) printf(" %d", a[i]);
            printf("\n  llama: ");
            for (int i = 0; i < nb && i < 40; i++) printf(" %d", b[i]);
            printf("\n");
        }
    }
    return na;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <corpus.txt>\n", argv[0]); return 2; }
    q38_gguf g;
    if (q38_gguf_open(&g, argv[1])) return 1;
    q38_tok *qt = q38_tok_load(&g);
    if (!qt) return 1;

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "llama.cpp could not load the vocabulary\n"); return 1; }
    const struct llama_vocab *v = llama_model_get_vocab(model);

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = (char *)malloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) return 1;
    s[n] = 0;
    fclose(f);

    int bad = 0, lines = 0;
    long toks = 0;
    for (long i = 0; i < n;) {
        long j = i;
        while (j < n && s[j] != '\n') j++;
        if (j < n) j++;
        toks += check(qt, v, s + i, (int)(j - i), &bad, 1);
        lines++;
        i = j;
    }
    int whole_bad = 0;
    long wt = check(qt, v, s, (int)n, &whole_bad, 1);
    printf("%d lines (%ld tokens): %d mismatches; whole file (%ld tokens): %s\n", lines, toks, bad, wt,
           whole_bad ? "MISMATCH" : "identical");
    llama_model_free(model);
    q38_tok_free(qt);
    q38_gguf_close(&g);
    return bad || whole_bad;
}
