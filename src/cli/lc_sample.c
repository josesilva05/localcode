/* lc_sample.c - how localcode chooses the next token (glm53f_sample, see glm53f_chat.h). */
#include <math.h>
#include <stdint.h>

#include "glm53f_chat.h"

#define SAMPLE_K_MAX 256

/* splitmix64: the sampler's random numbers */
static uint64_t rng_next(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int glm53f_sample(Glm53fSampler *p, const float *logits, int n)
{
    if (!p || p->temp <= 0.0f) {                 /* greedy: the first highest */
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    /* the K highest, highest first (an earlier id before a later one of the same logit) */
    const int K = p->top_k > 0 && p->top_k < SAMPLE_K_MAX ? p->top_k : SAMPLE_K_MAX;
    int id[SAMPLE_K_MAX], k = 0;
    float v[SAMPLE_K_MAX];
    for (int i = 0; i < n; i++) {
        const float x = logits[i];
        if (x != x || (k == K && x <= v[K - 1])) continue;
        int j = k < K ? k++ : K - 1;
        while (j > 0 && v[j - 1] < x) { v[j] = v[j - 1]; id[j] = id[j - 1]; j--; }
        v[j] = x;
        id[j] = i;
    }
    if (k == 0) return 0;
    double pr[SAMPLE_K_MAX], sum = 0.0;
    for (int j = 0; j < k; j++) sum += pr[j] = exp((double)(v[j] - v[0]) / p->temp);
    /* top_p: the smallest prefix holding that much; min_p: against the best one */
    int m = k;
    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        double c = 0.0;
        for (m = 0; m < k; ) { c += pr[m++]; if (c >= p->top_p * sum) break; }
    }
    if (p->min_p > 0.0f) while (m > 1 && pr[m - 1] < p->min_p * pr[0]) m--;
    double kept = 0.0;
    for (int j = 0; j < m; j++) kept += pr[j];
    double u = (double)(rng_next(&p->rng) >> 11) * (1.0 / 9007199254740992.0) * kept;
    for (int j = 0; j < m - 1; j++) { if (u < pr[j]) return id[j]; u -= pr[j]; }
    return id[m - 1];
}
