/* q38_gpu_tp.cuh - the dense models (Qwen3.8-27B) on two devices by tensor parallel (included by
 * q38_gpu.cu after q38_gpu_dense_batch.cuh, whose layer function it runs).
 *
 * The layer split (q38_gpu_dense.cuh) runs the first device's layers, then the second's: a token
 * reads the weights at one device's speed and each card waits half the time. Here every layer is
 * on both devices, each with half of it, and both work at once:
 *   - Gated DeltaNet: value head h reads key head h % Hk, so a device takes the value heads whose
 *     h % Hk falls in its half of the key heads (8 of 16 key heads, 24 of 48 value heads); qkv,
 *     the convolution, z, beta, alpha, dt_bias and a are cut to them, ssm_out to its inputs;
 *   - attention: half the query heads with their K/V heads (12 and 2), wo cut to their inputs;
 *   - FFN: half the rows of gate and up, those inputs of down;
 *   - the head: half the vocabulary each, the logits gathered on the host.
 * The norms, the residual and the MTP block's eh_proj are on both. After a layer's mixer and after
 * its FFN each device holds a partial sum of the output: both add theirs and the other's
 * (res += mine + other, the same sum in the same order on both, so the residuals stay equal). The
 * partials cross through mapped host memory: a kernel writes its rows there and raises a flag, the
 * other device's kernel waits for the flag and reads them; no host thread in between, so a layer is
 * one CUDA graph per device and N. All-reduce k of pass p carries the number p * 256 + k.
 *
 * Q38_TP=1 turns it on; the layer split stays the default while a prompt pass is slower here (466 against
 * 302 tok/s on the 27B): its sums are 20 MB a layer, and the second card sits in a PCIe 3.0 x4 slot. A
 * token (22 against 14 tok/s) and a verification (MTP 53 against 37 tok/s) are faster. */

static float *g_ar[2];                  /* mapped pinned: device r's partial rows, 2 slots */
static int   *g_arf[2];                 /* mapped pinned: device r's flags, one per slot */
static int64_t g_ar_slot;               /* floats a slot: the largest pass's rows */
static float *g_tp_lg;                  /* pinned: logits of up to 4 tokens, both halves */

struct TpRange { int64_t a, b; };

/* the rows of w in the ranges, packed into a new matrix (its data malloc'd) */
static int tp_rows(const q38_mat *w, const TpRange *r, int nr, q38_mat *o)
{
    *o = *w;
    int64_t n = 0;
    for (int i = 0; i < nr; i++) n += r[i].b - r[i].a;
    o->rows = n;
    if (!(o->data = (uint8_t *)malloc((size_t)(n * w->row_bytes)))) return -1;
    uint8_t *p = o->data;
    for (int i = 0; i < nr; i++) {
        memcpy(p, w->data + r[i].a * w->row_bytes, (size_t)((r[i].b - r[i].a) * w->row_bytes));
        p += (r[i].b - r[i].a) * w->row_bytes;
    }
    return 0;
}

/* elements [a, b) of every row (whole blocks), packed */
static int tp_cols(const q38_mat *w, const TpRange *r, int nr, q38_mat *o)
{
    *o = *w;
    const int bs = q38_type_block(w->type);
    int64_t n = 0;
    for (int i = 0; i < nr; i++) {
        if (bs <= 0 || r[i].a % bs || r[i].b % bs) return 1;
        n += r[i].b - r[i].a;
    }
    o->ne0 = n;
    o->row_bytes = q38_row_bytes(w->type, n);
    if (!(o->data = (uint8_t *)malloc((size_t)(o->rows * o->row_bytes)))) return -1;
    for (int64_t row = 0; row < w->rows; row++) {
        uint8_t *p = o->data + row * o->row_bytes;
        for (int i = 0; i < nr; i++) {
            const int64_t off = q38_row_bytes(w->type, r[i].a), len = q38_row_bytes(w->type, r[i].b - r[i].a);
            memcpy(p, w->data + row * w->row_bytes + off, (size_t)len);
            p += len;
        }
    }
    return 0;
}

#define TP_OFF(f) ((int)(offsetof(q38_layer, f) / sizeof(q38_mat)))

/* matrix i of layer il, device r's part of it: 0 uploaded, 1 a matrix tensor parallel does not know */
static int tp_upload(q38_model *m, Gpu *g, int il, int i, int r)
{
    const q38_cfg *c = &m->c;
    const q38_mat *w = (const q38_mat *)&m->L[il] + i;
    if (!w->data) { memset(&g->L[il][i], 0, sizeof(DMat)); return 0; }
    const int S = c->d_state, Hk = c->n_kh, Hv = c->n_vh, D = c->head_dim;
    const int hk = Hk / 2, kh0 = r * hk;                /* its key heads: [kh0, kh0 + hk) */
    TpRange vr[16], qr[18];                              /* its value heads (in units of a head), q k v rows */
    int nv = 0;
    for (int b = 0; b < Hv / Hk && nv < 16; b++) vr[nv++] = { (int64_t)b * Hk + kh0, (int64_t)b * Hk + kh0 + hk };
    int nq = 0;
    qr[nq++] = { (int64_t)kh0 * S, (int64_t)(kh0 + hk) * S };
    qr[nq++] = { (int64_t)(Hk + kh0) * S, (int64_t)(Hk + kh0 + hk) * S };
    for (int k = 0; k < nv; k++) qr[nq++] = { 2 * (int64_t)Hk * S + vr[k].a * S, 2 * (int64_t)Hk * S + vr[k].b * S };
    TpRange vs[16];                                      /* value heads in elements of S */
    for (int k = 0; k < nv; k++) vs[k] = { vr[k].a * S, vr[k].b * S };
    const int h0 = r * c->n_head / 2, h1 = h0 + c->n_head / 2;
    const int kv0 = r * c->n_head_kv / 2, kv1 = kv0 + c->n_head_kv / 2;
    const int64_t f0 = (int64_t)r * c->ff_shexp / 2, f1 = f0 + c->ff_shexp / 2;
    q38_mat o;
    int rc;
    if (i == TP_OFF(qkv) || i == TP_OFF(conv)) rc = tp_rows(w, qr, nq, &o);
    else if (i == TP_OFF(gate)) rc = tp_rows(w, vs, nv, &o);
    else if (i == TP_OFF(beta) || i == TP_OFF(alpha)) rc = tp_rows(w, vr, nv, &o);
    else if (i == TP_OFF(dt_bias) || i == TP_OFF(a)) rc = tp_cols(w, vr, nv, &o);
    else if (i == TP_OFF(ssm_out)) rc = tp_cols(w, vs, nv, &o);
    else if (i == TP_OFF(wq)) { const TpRange x = { (int64_t)h0 * 2 * D, (int64_t)h1 * 2 * D }; rc = tp_rows(w, &x, 1, &o); }
    else if (i == TP_OFF(wk) || i == TP_OFF(wv)) { const TpRange x = { (int64_t)kv0 * D, (int64_t)kv1 * D }; rc = tp_rows(w, &x, 1, &o); }
    else if (i == TP_OFF(wo)) { const TpRange x = { (int64_t)h0 * D, (int64_t)h1 * D }; rc = tp_cols(w, &x, 1, &o); }
    else if (i == TP_OFF(sh_gate) || i == TP_OFF(sh_up)) { const TpRange x = { f0, f1 }; rc = tp_rows(w, &x, 1, &o); }
    else if (i == TP_OFF(sh_down)) { const TpRange x = { f0, f1 }; rc = tp_cols(w, &x, 1, &o); }
    else if (i == TP_OFF(ssm_norm) || i == TP_OFF(q_norm) || i == TP_OFF(k_norm) || i == TP_OFF(attn_norm) ||
             i == TP_OFF(post_norm))
        return upload(&g->L[il][i], w, g);          /* on both */
    else return 1;
    if (rc) return rc;
    rc = upload(&g->L[il][i], &o, g);
    free(o.data);
    return rc;
}

/* 1 when the model's shape cuts in two (heads, FFN rows in whole blocks, only the matrices above) */
static int tp_fits(const q38_model *m)
{
    const q38_cfg *c = &m->c;
    if (c->n_kh % 2 || c->n_vh % c->n_kh || c->n_head % 2 || c->n_head_kv % 2 || (c->n_head / 2) % (c->n_head / c->n_head_kv) ||
        c->ff_shexp % 512 || (c->n_kh / 2 * c->d_state) % 256 || (c->n_head / 2 * c->head_dim) % 256 || c->n_vh / c->n_kh > 16)
        return 0;
    static const int known[] = { TP_OFF(qkv), TP_OFF(conv), TP_OFF(gate), TP_OFF(beta), TP_OFF(alpha), TP_OFF(dt_bias), TP_OFF(a),
                                 TP_OFF(ssm_out), TP_OFF(wq), TP_OFF(wk), TP_OFF(wv), TP_OFF(wo), TP_OFF(sh_gate), TP_OFF(sh_up),
                                 TP_OFF(sh_down), TP_OFF(ssm_norm), TP_OFF(q_norm), TP_OFF(k_norm), TP_OFF(attn_norm),
                                 TP_OFF(post_norm) };
    for (int il = 0; il < c->n_layer + (m->mtp ? 1 : 0); il++)
        for (int i = 0; i < NMAT; i++) {
            if (!((const q38_mat *)&m->L[il])[i].data) continue;
            int ok = 0;
            for (int k = 0; k < (int)(sizeof known / sizeof *known); k++) ok |= known[k] == i;
            if (!ok) return 0;
        }
    return 1;
}

/* device r's half of the model */
static int tp_part(q38_model *m, int dev, int r, int kvb, Gpu **out)
{
    const q38_cfg *c = &m->c;
    CK(cudaSetDevice(dev));
    Gpu *g = (Gpu *)calloc(1, sizeof(Gpu));
    if (!g) return -1;
    *out = g;
    G = g;
    g->dev = dev;
    g->tp = 1;
    g->tp_rank = r;
    g->tp_H = c->n_head / 2; g->tp_Hkv = c->n_head_kv / 2; g->tp_Hk = c->n_kh / 2; g->tp_Hv = c->n_vh / 2;
    g->tp_F = c->ff_shexp / 2;
    g->tp_V0 = r ? c->n_vocab / 2 : 0;
    g->tp_V = r ? c->n_vocab - c->n_vocab / 2 : c->n_vocab / 2;
    g->kv_half = kvb == 2;
    g->kv_q8 = kvb == 1;
    g->kvh_n = g->tp_Hkv;
    CK(cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&g->ev, cudaEventDisableTiming | cudaEventBlockingSync));
    const int nl = c->n_layer + (m->mtp ? 1 : 0);
    for (int il = 0; il < nl; il++)
        for (int i = 0; i < NMAT; i++)
            if (tp_upload(m, g, il, i, r)) { fprintf(stderr, "q38_gpu: tensor parallel upload failed (layer %d)\n", il); return -1; }
    {
        q38_mat h = m->output;                       /* its half of the vocabulary: rows, no copy */
        h.data = m->output.data + (int64_t)g->tp_V0 * m->output.row_bytes;
        h.rows = g->tp_V;
        if (upload(&g->output, &h, g) || upload(&g->output_norm, &m->output_norm, g)) return -1;
    }
    if (m->mtp && (upload(&g->mtp_eh, &m->mtp_eh, g) || upload(&g->mtp_enorm, &m->mtp_enorm, g) ||
                   upload(&g->mtp_hnorm, &m->mtp_hnorm, g) || upload(&g->mtp_shnorm, &m->mtp_shnorm, g))) return -1;
    const int E = c->n_embd, S = c->d_state, ch = gdn_ch(g, c), Hv = g->tp_Hv;
    g->res = dalloc(E, g); g->mixed = dalloc(E, g); g->y = dalloc(E, g);
    g->fa_part = dalloc((size_t)4 * (FA_SEG_MMA > FA_SEG ? FA_SEG_MMA : FA_SEG) * g->tp_H * (c->head_dim + 2), g);
    fa_setup(c->head_dim);
    CK(cudaMalloc(&g->q8x, 4 * 32768));
    g->q8d = dalloc(4 * 1024, g); g->q8s = dalloc(4 * 1024, g);
    g->logits = dalloc((size_t)4 * g->tp_V, g);          /* 4 rows: a verification's */
    for (int il = 0; il < nl; il++) {
        if (m->L[il].recurrent) {
            g->conv[il] = dalloc((size_t)(c->conv_k - 1) * ch, g);
            g->ssm[il] = dalloc((size_t)Hv * S * S, g);
            if (!g->conv[il] || !g->ssm[il]) { fprintf(stderr, "q38_gpu: out of device memory (state)\n"); return -1; }
        } else {
            const size_t kv_f = ((size_t)m->max_ctx * g->tp_Hkv * c->head_dim * kvb + 3) / 4;
            g->kc[il] = dalloc(kv_f, g);
            g->vc[il] = dalloc(kv_f, g);
            if (kvb == 1) {
                g->ksc[il] = dalloc((size_t)m->max_ctx * g->tp_Hkv * (c->head_dim / 32), g);
                g->vsc[il] = dalloc((size_t)m->max_ctx * g->tp_Hkv * (c->head_dim / 32), g);
            }
            if (!g->kc[il] || !g->vc[il] || (kvb == 1 && (!g->ksc[il] || !g->vsc[il]))) {
                fprintf(stderr, "q38_gpu: out of device memory (K/V)\n");
                return -1;
            }
        }
    }
    if (m->mtp) {
        g->mtp_hprev = dalloc(E, g);             /* zero: no hidden before position 0 */
        g->mtp_hchain = dalloc(E, g);
        if (!g->mtp_hprev || !g->mtp_hchain) return -1;
    }
    if (!g->logits || !g->fa_part || !g->q8s) { fprintf(stderr, "q38_gpu: out of device memory\n"); return -1; }
    CK(cudaMallocHost(&g->h_pos, sizeof(int)));
    CK(cudaMalloc(&g->d_pos, sizeof(int)));
    CK(cudaMalloc(&g->d_pass, sizeof(int)));
    CK(cudaMemset(g->d_pass, 0, sizeof(int)));
    const char *eg = getenv("Q38_GRAPH");
    g->use_graphs = !eg || atoi(eg) != 0;
    return 0;
}

static int  dense_bat_ready(q38_model *m);
static void dense_free_host(q38_model *m);

/* Both devices, each with half of every layer. 0 opened, 1 the model does not cut in two (the caller
 * splits the layers), -1 failed. room: each device's free bytes less the reserve. */
static int tp_open(q38_model *m, int kvb, const int *devs, const int64_t *room)
{
    const q38_cfg *c = &m->c;
    if (!(getenv("Q38_TP") && atoi(getenv("Q38_TP")))) return 1;   /* opt-in, see the top */
    if (!tp_fits(m)) return 1;
    const int L = c->n_layer, nl = L + (m->mtp ? 1 : 0);
    const int asked = m->max_ctx;
    int64_t need = 0;
    for (;;) {                                   /* a device's share; the context shrinks until it fits */
        need = (m->output.row_bytes * m->output.rows) / 2 + (int64_t)c->n_vocab * 8;
        if (m->mtp) need += m->mtp_eh.row_bytes * m->mtp_eh.rows;
        for (int il = 0; il < nl; il++) need += (dense_layer_bytes(m, il) + dense_state_bytes(m, il, kvb)) / 2;
        const int64_t room_min = room[0] < room[1] ? room[0] : room[1];
        if (need <= room_min || m->max_ctx <= 8192) break;
        m->max_ctx = m->max_ctx * 3 / 4 / 1024 * 1024;
    }
    if (need > (room[0] < room[1] ? room[0] : room[1])) return 1;
    if (m->max_ctx < asked) fprintf(stderr, "q38_gpu: context %d -> %d positions, to fit the devices\n", asked, m->max_ctx);
    for (int r = 0; r < 2; r++) if (tp_part(m, devs[r], r, kvb, &GD[r])) return -1;
    GD_n = 2;
    GD_split = L;
    G = GD[0];
    m->gpu = GD[0];
    /* the partials' slots: the largest pass is a prompt piece */
    const char *e = getenv("Q38_BATCH");
    int nb = e ? atoi(e) : 1024;
    if (nb < 16) nb = 16;
    if (nb > 2048) nb = 2048;
    g_ar_slot = (int64_t)nb * c->n_embd;
    for (int r = 0; r < 2; r++) {
        CK(cudaSetDevice(devs[r]));
        CK(cudaHostAlloc((void **)&g_ar[r], (size_t)(2 * g_ar_slot) * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc((void **)&g_arf[r], 2 * sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
        g_arf[r][0] = g_arf[r][1] = 0;
    }
    CK(cudaHostAlloc((void **)&g_tp_lg, (size_t)4 * c->n_vocab * sizeof(float), cudaHostAllocPortable));
    dense_free_host(m);
    if (dense_bat_ready(m)) return -1;
    cudaDeviceProp p0, p1;
    cudaGetDeviceProperties(&p0, devs[0]);
    cudaGetDeviceProperties(&p1, devs[1]);
    fprintf(stderr, "q38: dense trunk by tensor parallel, half of every layer on GPU %d (%s, %.2f GB) and GPU %d (%s, %.2f GB) "
                    "(context %d, K/V %s)\n", devs[0], p0.name, GD[0]->bytes / 1e9, devs[1], p1.name, GD[1]->bytes / 1e9,
            m->max_ctx, kvb == 1 ? "int8" : kvb == 2 ? "fp16" : "fp32");
    return 0;
}

/* ---- the all-reduce ---- */

/* this device's partial rows (n floats) to its slot, then its flag: one block (a token, a verification) */
__global__ void k_ar_put1(const float *y, int n, float *slots, int64_t slot, int *flag, const int *pass, int k)
{
    const int s = *pass * 256 + k;
    float *dst = slots + (int64_t)(s & 1) * slot;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = y[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile int *)(flag + (s & 1)) = s + 1;
}

/* many blocks (a prompt piece); k_ar_flag after it */
__global__ void k_ar_putn(const float *y, int64_t n, float *slots, int64_t slot, const int *pass, int k)
{
    const int s = *pass * 256 + k;
    float4 *dst = (float4 *)(slots + (int64_t)(s & 1) * slot);
    const float4 *src = (const float4 *)y;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n / 4; i += (int64_t)gridDim.x * blockDim.x) dst[i] = src[i];
    __threadfence_system();
}

__global__ void k_ar_flag(int *flag, const int *pass, int k)
{
    const int s = *pass * 256 + k;
    __threadfence_system();
    *(volatile int *)(flag + (s & 1)) = s + 1;
}

/* res += y + the other device's partial, once its flag says the partial is there */
__global__ void k_ar_add(float *res, const float *y, int64_t n, const float *oslots, int64_t slot, const int *oflag,
                         const int *pass, int k)
{
    const int s = *pass * 256 + k;
    if (threadIdx.x == 0) {
        /* at most 10 s: a device that never comes (its host failed) must not keep this one spinning
         * after the process ends; the trap fails the context instead */
        unsigned long long t0, t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        while (*(const volatile int *)(oflag + (s & 1)) < s + 1) {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            if (t - t0 > 10000000000ull) {
                printf("q38_gpu: tensor parallel sum %d waited 10 s for the other device (its flag %d)\n", s,
                       *(const volatile int *)(oflag + (s & 1)));
                __trap();
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    const float *src = oslots + (int64_t)(s & 1) * slot;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        res[i] += y[i] + __ldcv(src + i);
}

__global__ void k_tp_pass(int *pass, int *pos, int p0) { (*pass)++; *pos = p0; }

static int tp_add(q38_model *m, DBat *d, int N, int k, cudaStream_t st)
{
    const int r = G->tp_rank;
    const int64_t n = (int64_t)N * m->c.n_embd;
    if (n > g_ar_slot) { fprintf(stderr, "q38_gpu: tensor parallel pass larger than its slots\n"); return -1; }
    if (n <= 32768) {
        k_ar_put1<<<1, 1024, 0, st>>>(d->y, (int)n, g_ar[r], g_ar_slot, g_arf[r], G->d_pass, k);
        k_ar_add<<<(int)((n + 1023) / 1024), 1024, 0, st>>>(d->res, d->y, n, g_ar[r ^ 1], g_ar_slot, g_arf[r ^ 1], G->d_pass, k);
    } else {
        k_ar_putn<<<64, 256, 0, st>>>(d->y, n, g_ar[r], g_ar_slot, G->d_pass, k);
        k_ar_flag<<<1, 1, 0, st>>>(g_arf[r], G->d_pass, k);
        k_ar_add<<<256, 256, 0, st>>>(d->res, d->y, n, g_ar[r ^ 1], g_ar_slot, g_arf[r ^ 1], G->d_pass, k);
    }
    return 0;
}

/* ---- passes ---- */

/* layer il for N tokens on device p: a graph per N up to 4 */
static int tp_layer(q38_model *m, int p, int il, int N)
{
    G = GD[p];
    DBat *d = GDB[p];
    cudaStream_t st = G->st;
    if (N > 4 || !G->use_graphs || m->tap) return dense_layer_b(m, d, il, N, st);
    cudaGraphExec_t *ge = &G->tpg[N - 1][il];
    if (!*ge) {
        cudaGraph_t gr;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        const int rc = dense_layer_b(m, d, il, N, st);
        CK(cudaStreamEndCapture(st, &gr));
        if (rc) return -1;
        CK(cudaGraphInstantiate(ge, gr, 0));
        cudaGraphDestroy(gr);
    }
    CK(cudaGraphLaunch(*ge, st));
    return 0;
}

/* Q38_TP_SYNC=1: both devices synchronised and checked after each step (where a fault happened) */
static int tp_check(const char *what, int il)
{
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("Q38_TP_SYNC") && atoi(getenv("Q38_TP_SYNC"));
    for (int p = 0; dbg && p < 2; p++) {
        CK(cudaSetDevice(GD[p]->dev));
        const cudaError_t e = cudaStreamSynchronize(GD[p]->st);
        if (e != cudaSuccess) {
            fprintf(stderr, "q38_gpu: tensor parallel %s %d failed on device %d: %s\n", what, il, GD[p]->dev, cudaGetErrorString(e));
            return -1;
        }
    }
    return 0;
}

/* layers [l0, l1) for the N rows in both devices' d->res: layer by layer, each device's work handed to
 * it at once (WDDM batches submissions; the other device waits on it) */
static int tp_layers(q38_model *m, int l0, int l1, int N)
{
    for (int il = l0; il < l1; il++) {
        for (int p = 0; p < 2; p++) {
            CK(cudaSetDevice(GD[p]->dev));
            if (tp_layer(m, p, il, N)) return -1;
            cudaStreamQuery(GD[p]->st);
        }
        if (tp_check("layer", il)) return -1;
    }
    return 0;
}

/* a pass's start on both devices: the rows (embeddings of toks, or hidden rows already on the
 * devices when toks is NULL), the position, the pass number */
static int tp_start(q38_model *m, const int *toks, int N, int p0)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd;
    for (int p = 0; p < 2; p++) {
        G = GD[p];
        DBat *d = GDB[p];
        CK(cudaSetDevice(G->dev));
        if (toks) {
            CK(cudaEventSynchronize(d->ev_in[0]));
            if (p == 0)
                for (int t = 0; t < N; t++)
                    q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)toks[t] * m->tok_embd.row_bytes,
                                    d->h_in[0] + (int64_t)t * E, E);
            else memcpy(d->h_in[0], GDB[0]->h_in[0], (size_t)N * E * sizeof(float));
            CK(cudaMemcpyAsync(d->res, d->h_in[0], (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, G->st));
            CK(cudaEventRecord(d->ev_in[0], G->st));
        }
        k_tp_pass<<<1, 1, 0, G->st>>>(G->d_pass, G->d_pos, p0);
    }
    return 0;
}

/* the head on rows [t0, t0 + nr) of res (normed with nrm into mixed): each device its half of the
 * vocabulary into g_tp_lg (row stride n_vocab) */
static int tp_head(q38_model *m, int t0, int nr)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, V = c->n_vocab;
    for (int p = 0; p < 2; p++) {
        G = GD[p];
        DBat *d = GDB[p];
        cudaStream_t st = G->st;
        CK(cudaSetDevice(G->dev));
        k_rmsnorm_rows_w<<<nr, 256, 0, st>>>(d->res + (int64_t)t0 * E, (const float *)G->output_norm.d, d->mixed, E, c->eps, 1);
        if (dmm(d, &G->output, d->mixed, E, G->logits, G->tp_V, nr, st)) return -1;
        CK(cudaMemcpy2DAsync(g_tp_lg + G->tp_V0, (size_t)V * sizeof(float), G->logits, (size_t)G->tp_V * sizeof(float),
                             (size_t)G->tp_V * sizeof(float), (size_t)nr, cudaMemcpyDeviceToHost, st));
    }
    return tp_check("head rows", nr);
}

static int tp_sync(void)
{
    for (int p = 1; p >= 0; p--) {
        CK(cudaSetDevice(GD[p]->dev));
        CK(cudaStreamSynchronize(GD[p]->st));
        CK(cudaGetLastError());
    }
    G = GD[0];
    return 0;
}

/* The MTP block's K/V over the N rows just through the layers (d->res holds their hidden rows): the
 * pairs (h of the position before, token), hprev the last row after; on both devices */
static int tp_mtp_catchup(q38_model *m, const int *toks, int N)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd;
    for (int p = 0; p < 2; p++) {
        G = GD[p];
        DBat *d = GDB[p];
        cudaStream_t st = G->st;
        CK(cudaSetDevice(G->dev));
        CK(cudaEventSynchronize(d->ev_mtp));
        if (p == 0)
            for (int t = 0; t < N; t++)
                q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)toks[t] * m->tok_embd.row_bytes,
                                d->h_mtp + (int64_t)t * E, E);
        else memcpy(d->h_mtp, GDB[0]->h_mtp, (size_t)N * E * sizeof(float));
        CK(cudaMemcpyAsync(d->emb2, d->h_mtp, (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, st));
        CK(cudaEventRecord(d->ev_mtp, st));
        k_mtp_cat<<<N, 256, 0, st>>>(d->emb2, d->res, G->mtp_hprev, (const float *)G->mtp_enorm.d, (const float *)G->mtp_hnorm.d,
                                     d->cat2, E, c->eps);
        CK(cudaMemcpyAsync(G->mtp_hprev, d->res + (int64_t)(N - 1) * E, (size_t)E * sizeof(float), cudaMemcpyDeviceToDevice, st));
        if (dmm(d, &G->mtp_eh, d->cat2, 2 * E, d->res, E, N, st)) return -1;
    }
    if (tp_check("mtp input rows", N)) return -1;
    return tp_layers(m, c->n_layer, c->n_layer + 1, N);
}

/* One token: logits into m->logits (and the MTP block's K/V at its position) */
static int tp_forward(q38_model *m, int token)
{
    const q38_cfg *c = &m->c;
    if (dense_bat_ready(m)) return -1;
    const double t0 = now_s();
    if (tp_start(m, &token, 1, m->n_past) || tp_layers(m, 0, c->n_layer, 1) || tp_head(m, 0, 1))
        return -1;
    if (GD[0]->mtp_eh.d && tp_mtp_catchup(m, &token, 1)) return -1;
    const int mtp_tap = m->tap && GD[0]->mtp_eh.d;
    for (int p = 0; mtp_tap && p < 2; p++) {    /* a dump: the block's logits on this pair too */
        G = GD[p];
        DBat *d = GDB[p];
        CK(cudaSetDevice(G->dev));
        k_rmsnorm_rows<<<1, 256, 0, G->st>>>(d->res, (const float *)G->mtp_shnorm.d, d->mixed, c->n_embd, c->eps);
        if (mv(&G->output, d->mixed, G->logits + G->tp_V, G->st)) return -1;
        CK(cudaMemcpyAsync(g_tp_lg + c->n_vocab + G->tp_V0, G->logits + G->tp_V, (size_t)G->tp_V * sizeof(float),
                           cudaMemcpyDeviceToHost, G->st));
    }
    const double tw = now_s();
    if (tp_sync()) return -1;
    m->t_gpu_wait += now_s() - tw;
    memcpy(m->logits, g_tp_lg, (size_t)c->n_vocab * sizeof(float));
    if (m->tap) m->tap(m->tap_ctx, "logits", m->logits, c->n_vocab);
    if (mtp_tap) {
        memcpy(m->mtp_logits, g_tp_lg + c->n_vocab, (size_t)c->n_vocab * sizeof(float));
        m->tap(m->tap_ctx, "mtp_logits", m->mtp_logits, c->n_vocab);
    }
    G->t_launch += tw - t0;
    G->tokens_fwd++;
    return 0;
}

/* n tokens in pieces of nb (a prompt): both devices on every piece */
static int tp_forward_batch(q38_model *m, const int *toks, int n)
{
    const q38_cfg *c = &m->c;
    if (dense_bat_ready(m)) return -1;
    const int nb = GDB[0]->nb, p00 = m->n_past;
    const double t0 = now_s();
    for (int i = 0; i < n; i += nb) {
        const int N = n - i < nb ? n - i : nb, last = i + N == n;
        if (tp_start(m, toks + i, N, p00 + i) || tp_layers(m, 0, c->n_layer, N)) return -1;
        if (last && tp_head(m, N - 1, 1)) return -1;
        if (GD[0]->mtp_eh.d && tp_mtp_catchup(m, toks + i, N)) return -1;
    }
    if (tp_sync()) return -1;
    memcpy(m->logits, g_tp_lg, (size_t)c->n_vocab * sizeof(float));
    for (int t = 0; t < n; t++) m->toks[p00 + t] = toks[t];
    m->n_past = p00 + n;
    if (getenv("Q38_BATCH_VERBOSE"))
        fprintf(stderr, "q38: dense prompt of %d tokens (tensor parallel, pieces of %d): %.2f s (%.0f tok/s)\n", n, nb,
                now_s() - t0, n / (now_s() - t0));
    return 0;
}

static int keep_open(q38_model *m);

/* A verification (q38_gpu_verify): N <= 4 tokens, the logits of each */
static int tp_verify(q38_model *m, const int *toks, int N, float *logits)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, p0 = m->n_past;
    if (N < 1 || N > 4 || dense_bat_ready(m)) return -1;
    static int keep = -1;
    if (keep < 0) keep = !(getenv("Q38_SPEC_KEEP") && !atoi(getenv("Q38_SPEC_KEEP"))) && keep_open(m) == 0;
    for (int p = 0; p < 2; p++) GDB[p]->keeping = keep && N > 1;
    if (tp_start(m, toks, N, p0) || tp_layers(m, 0, c->n_layer, N) || tp_head(m, 0, N)) return -1;
    for (int p = 0; p < 2; p++)
        if (GDB[p]->keeping) {
            CK(cudaSetDevice(GD[p]->dev));
            CK(cudaMemcpyAsync(GDB[p]->keep_h, GDB[p]->res, (size_t)N * E * sizeof(float), cudaMemcpyDeviceToDevice, GD[p]->st));
        }
    if (GD[0]->mtp_eh.d && tp_mtp_catchup(m, toks, N)) return -1;
    if (tp_sync()) return -1;
    memcpy(logits, g_tp_lg, (size_t)N * c->n_vocab * sizeof(float));
    for (int t = 0; t < N; t++) m->toks[p0 + t] = toks[t];
    m->n_past = p0 + N;
    for (int p = 0; p < 2; p++) GDB[p]->keeping = 0;
    return keep && N > 1 ? 1 : 0;
}

/* The MTP draft (q38_gpu_mtp_draft): the block at n_past + j on (hprev, tok), or on (the previous
 * draft's normed output, tok) for a chained one; its logits into m->mtp_logits */
static int tp_mtp_draft(q38_model *m, int tok, int j)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd;
    if (dense_bat_ready(m)) return -1;
    for (int p = 0; p < 2; p++) {
        G = GD[p];
        DBat *d = GDB[p];
        cudaStream_t st = G->st;
        CK(cudaSetDevice(G->dev));
        CK(cudaEventSynchronize(d->ev_mtp));
        if (p == 0) q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)tok * m->tok_embd.row_bytes, d->h_mtp, E);
        else memcpy(d->h_mtp, GDB[0]->h_mtp, (size_t)E * sizeof(float));
        CK(cudaMemcpyAsync(d->emb2, d->h_mtp, (size_t)E * sizeof(float), cudaMemcpyHostToDevice, st));
        CK(cudaEventRecord(d->ev_mtp, st));
        k_tp_pass<<<1, 1, 0, st>>>(G->d_pass, G->d_pos, m->n_past + j);
        k_mtp_cat<<<1, 256, 0, st>>>(d->emb2, NULL, j ? G->mtp_hchain : G->mtp_hprev, (const float *)G->mtp_enorm.d,
                                     (const float *)G->mtp_hnorm.d, d->cat2, E, c->eps);
        if (mv(&G->mtp_eh, d->cat2, d->res, st)) return -1;
    }
    if (tp_layers(m, c->n_layer, c->n_layer + 1, 1)) return -1;
    /* the normed output: the next chained draft's h (both devices), and the head */
    for (int p = 0; p < 2; p++) {
        G = GD[p];
        DBat *d = GDB[p];
        cudaStream_t st = G->st;
        CK(cudaSetDevice(G->dev));
        k_rmsnorm_rows<<<1, 256, 0, st>>>(d->res, (const float *)G->mtp_shnorm.d, d->mixed, E, c->eps);
        CK(cudaMemcpyAsync(G->mtp_hchain, d->mixed, (size_t)E * sizeof(float), cudaMemcpyDeviceToDevice, st));
        if (mv(&G->output, d->mixed, G->logits, st)) return -1;
        CK(cudaMemcpyAsync(g_tp_lg + G->tp_V0, G->logits, (size_t)G->tp_V * sizeof(float), cudaMemcpyDeviceToHost, st));
    }
    if (tp_sync()) return -1;
    memcpy(m->mtp_logits, g_tp_lg, (size_t)c->n_vocab * sizeof(float));
    return 0;
}

static void tp_close(void)
{
    for (int r = 0; r < 2; r++) {
        if (g_ar[r]) cudaFreeHost(g_ar[r]);
        if (g_arf[r]) cudaFreeHost(g_arf[r]);
        g_ar[r] = NULL;
        g_arf[r] = NULL;
    }
    if (g_tp_lg) cudaFreeHost(g_tp_lg);
    g_tp_lg = NULL;
}
