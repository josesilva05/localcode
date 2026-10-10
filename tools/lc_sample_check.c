/* lc_sample_check.c - glm53f_sample (src/cli/lc_sample.c) against probabilities worked out here
 * independently (the whole vocabulary sorted, softmax over the top_k, the top_p and min_p cuts),
 * by drawing many times. Exit code 0 when every token's share is within 5 standard deviations of
 * its probability and no token outside the kept set is ever drawn.
 *
 *     lc_sample_check [draws]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm53f_chat.h"

#define N 5000

static const float *g_lg;
static int by_logit(const void *a, const void *b)
{
    const int i = *(const int *)a, j = *(const int *)b;
    if (g_lg[i] != g_lg[j]) return g_lg[i] > g_lg[j] ? -1 : 1;
    return i - j;                                   /* an earlier id first among equals */
}

/* the probability of every id under (temp, top_k, top_p, min_p) */
static void expected(const float *lg, int n, const Glm53fSampler *p, double *prob)
{
    int *ord = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) ord[i] = i;
    g_lg = lg;
    qsort(ord, (size_t)n, sizeof(int), by_logit);
    const int k = p->top_k > 0 && p->top_k < n ? p->top_k : n;
    double *w = (double *)calloc((size_t)k, sizeof(double)), sum = 0.0;
    for (int j = 0; j < k; j++) sum += w[j] = exp(((double)lg[ord[j]] - lg[ord[0]]) / p->temp);
    int m = k;
    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        double c = 0.0;
        for (m = 0; m < k; m++) if ((c += w[m]) >= p->top_p * sum) { m++; break; }
    }
    if (p->min_p > 0.0f) while (m > 1 && w[m - 1] < p->min_p * w[0]) m--;
    double kept = 0.0;
    for (int j = 0; j < m; j++) kept += w[j];
    memset(prob, 0, (size_t)n * sizeof(double));
    for (int j = 0; j < m; j++) prob[ord[j]] = w[j] / kept;
    free(w);
    free(ord);
}

static int run(const char *name, const float *lg, Glm53fSampler p, long draws)
{
    static double prob[N];
    static long hits[N];
    expected(lg, N, &p, prob);
    memset(hits, 0, sizeof hits);
    for (long d = 0; d < draws; d++) hits[glm53f_sample(&p, lg, N)]++;
    double worst = 0.0;
    int bad = 0, kept = 0;
    for (int i = 0; i < N; i++) {
        if (prob[i] > 0.0) kept++;
        if (prob[i] == 0.0) { if (hits[i]) bad = 1; continue; }
        const double sd = sqrt(prob[i] * (1.0 - prob[i]) / (double)draws) + 1e-12;
        const double z = fabs((double)hits[i] / (double)draws - prob[i]) / sd;
        if (z > worst) worst = z;
    }
    const int ok = !bad && worst < 5.0;
    printf("%-34s kept %4d  worst %.2f sd%s  %s\n", name, kept, worst, bad ? "  DRAWN OUTSIDE THE SET" : "", ok ? "ok" : "FAIL");
    return ok;
}

int main(int argc, char **argv)
{
    const long draws = argc > 1 ? atol(argv[1]) : 2000000;
    static float lg[N];
    uint64_t s = 12345;
    for (int i = 0; i < N; i++) {                   /* logits like a model's: a few high, a long tail */
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const double u = (double)(s >> 11) / 9007199254740992.0;
        lg[i] = (float)(8.0 * pow(u, 6.0) - 2.0);
    }
    lg[17] = lg[4000] = 7.5f;                       /* a tie near the top */
    int ok = 1;
    {   /* greedy: the first highest, as the old loop chose */
        Glm53fSampler g = { 0 };
        int best = 0;
        for (int i = 1; i < N; i++) if (lg[i] > lg[best]) best = i;
        const int got = glm53f_sample(&g, lg, N);
        printf("%-34s %d (expected %d)  %s\n", "temp 0 (greedy)", got, best, got == best ? "ok" : "FAIL");
        ok &= got == best;
    }
    ok &= run("temp 0.6 top_p 0.95 top_k 20", lg, (Glm53fSampler){ 0.6f, 0.95f, 0.0f, 20, 1 }, draws);
    ok &= run("temp 1.0 top_k 40", lg, (Glm53fSampler){ 1.0f, 1.0f, 0.0f, 40, 2 }, draws);
    ok &= run("temp 1.5 top_p 0.8 top_k 256", lg, (Glm53fSampler){ 1.5f, 0.8f, 0.0f, 256, 3 }, draws);
    ok &= run("temp 0.8 top_k 100 min_p 0.05", lg, (Glm53fSampler){ 0.8f, 1.0f, 0.05f, 100, 4 }, draws);
    ok &= run("temp 0.3 top_p 0.5 top_k 20", lg, (Glm53fSampler){ 0.3f, 0.5f, 0.0f, 20, 5 }, draws);
    printf("%s\n", ok ? "all ok" : "FAILED");
    return ok ? 0 : 1;
}
