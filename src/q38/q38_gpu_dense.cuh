/* q38_gpu_dense.cuh - the dense models (qwen35: Qwen3.8-27B) on one or two devices (included
 * by q38_gpu.cu after the segments).
 *
 * The trunk is the whole model (19.4 GB for the 27B in Q5_K_M), more than one 12 GB card: the
 * layers are split between the devices, each a Gpu of its own (G points at the one working)
 * holding a run of layers with their weights, K/V, recurrent state and scratch; the last one
 * also holds the head. A token goes through the first device's layers, its residual (n_embd
 * floats) to the host and on to the second's. Each layer is one CUDA graph, mixer then FFN:
 * nothing goes to the host inside a device's run (the MoE models route on the host per layer).
 * The weights stay in their GGUF types (lane_dot reads them all); only Q8_0 is repacked. */

static int  dense_forward_batch(q38_model *m, const int *toks, int n);   /* q38_gpu_dense_batch.cuh */
static void dense_batch_close(void);

static int64_t dense_layer_bytes(const q38_model *m, int il)
{
    const q38_mat *w = (const q38_mat *)&m->L[il];
    int64_t b = 0;
    for (int i = 0; i < NMAT; i++) if (w[i].data) b += w[i].row_bytes * w[i].rows;
    return b;
}

/* K/V (attention layers; kvb bytes an element, int8 with a float scale per row) or recurrent state of
 * a layer on the device */
static int64_t dense_state_bytes(const q38_model *m, int il, int kvb)
{
    const q38_cfg *c = &m->c;
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    if (m->L[il].recurrent) return ((int64_t)(c->conv_k - 1) * ch + (int64_t)c->n_vh * S * S) * 4;
    return 2 * (int64_t)m->max_ctx * c->n_head_kv * ((int64_t)c->head_dim * kvb + (kvb == 1 ? c->head_dim / 32 * 4 : 0));
}

/* one device's part: layers [l0, l1), the head when last */
static int dense_part(q38_model *m, int dev, int l0, int l1, int last, int kvb, Gpu **out)
{
    const q38_cfg *c = &m->c;
    CK(cudaSetDevice(dev));
    Gpu *g = (Gpu *)calloc(1, sizeof(Gpu));
    if (!g) return -1;
    *out = g;
    G = g;
    g->dev = dev;
    g->kv_half = kvb == 2;
    g->kv_q8 = kvb == 1;
    g->kvh_n = c->n_head_kv;
    CK(cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&g->ev, cudaEventDisableTiming | cudaEventBlockingSync));
    for (int il = l0; il < l1; il++) {
        const q38_mat *w = (const q38_mat *)&m->L[il];
        for (int i = 0; i < NMAT; i++)
            if (upload(&g->L[il][i], &w[i], g)) return -1;
    }
    if (last && (upload(&g->output, &m->output, g) || upload(&g->output_norm, &m->output_norm, g))) return -1;
    const int mtp = last && m->mtp;
    if (mtp) {
        const q38_mat *w = (const q38_mat *)&m->L[c->n_layer];
        for (int i = 0; i < NMAT; i++)
            if (upload(&g->L[c->n_layer][i], &w[i], g)) return -1;
        if (upload(&g->mtp_eh, &m->mtp_eh, g) || upload(&g->mtp_enorm, &m->mtp_enorm, g) ||
            upload(&g->mtp_hnorm, &m->mtp_hnorm, g) || upload(&g->mtp_shnorm, &m->mtp_shnorm, g)) return -1;
    }
    const int E = c->n_embd, S = c->d_state, F = c->ff_shexp;
    const int ch = 2 * c->n_kh * S + c->n_vh * S;
    g->res = dalloc(E, g); g->xn = dalloc(E, g); g->mixed = dalloc(E, g); g->lo = dalloc(1, g);
    g->gate = dalloc(E, g); g->inject = dalloc(64, g); g->y = dalloc(E, g); g->routed = dalloc(E, g);
    g->qkv = dalloc(ch, g); g->z = dalloc((size_t)c->n_vh * S, g); g->o = dalloc((size_t)c->n_vh * S, g);
    g->beta = dalloc(256, g); g->g = dalloc(256, g);
    g->qf = dalloc((size_t)c->n_head * c->head_dim * 2, g); g->aout = dalloc((size_t)c->n_head * c->head_dim, g);
    g->knew = dalloc((size_t)c->n_head_kv * c->head_dim, g); g->vnew = dalloc((size_t)c->n_head_kv * c->head_dim, g);
    g->fa_part = dalloc((size_t)(FA_SEG_MMA > FA_SEG ? FA_SEG_MMA : FA_SEG) * c->n_head * (c->head_dim + 2), g);
    fa_setup(c->head_dim);
    g->sh_gu = dalloc((size_t)2 * F, g); g->sh_h = dalloc(F, g);
    CK(cudaMalloc(&g->q8x, 4 * 32768));           /* x in 8 bits: up to 4 rows (verification) of 32768 */
    g->q8d = dalloc(4 * 1024, g); g->q8s = dalloc(4 * 1024, g);
    g->logits = dalloc(last ? c->n_vocab : 1, g);
    for (int il = l0; il < l1; il++) {
        if (m->L[il].recurrent) {
            g->conv[il] = dalloc((size_t)(c->conv_k - 1) * ch, g);
            g->ssm[il] = dalloc((size_t)c->n_vh * S * S, g);
            if (!g->conv[il] || !g->ssm[il]) { fprintf(stderr, "q38_gpu: out of device memory (state)\n"); return -1; }
        } else {
            const size_t kv_f = ((size_t)m->max_ctx * c->n_head_kv * c->head_dim * kvb + 3) / 4;
            g->kc[il] = dalloc(kv_f, g);
            g->vc[il] = dalloc(kv_f, g);
            if (kvb == 1) {
                g->ksc[il] = dalloc((size_t)m->max_ctx * c->n_head_kv * (c->head_dim / 32), g);
                g->vsc[il] = dalloc((size_t)m->max_ctx * c->n_head_kv * (c->head_dim / 32), g);
            }
            if (!g->kc[il] || !g->vc[il] || (kvb == 1 && (!g->ksc[il] || !g->vsc[il]))) {
                fprintf(stderr, "q38_gpu: out of device memory (K/V)\n");
                return -1;
            }
        }
    }
    if (mtp) {
        const int il = c->n_layer;
        const size_t kv_f = ((size_t)m->max_ctx * c->n_head_kv * c->head_dim * kvb + 3) / 4;
        g->kc[il] = dalloc(kv_f, g);
        g->vc[il] = dalloc(kv_f, g);
        if (kvb == 1) {
            g->ksc[il] = dalloc((size_t)m->max_ctx * c->n_head_kv * (c->head_dim / 32), g);
            g->vsc[il] = dalloc((size_t)m->max_ctx * c->n_head_kv * (c->head_dim / 32), g);
        }
        g->mtp_hprev = dalloc(E, g);             /* zero: no hidden before position 0 */
        g->mtp_hchain = dalloc(E, g);
        g->mtp_cat = dalloc((size_t)2 * E, g);
        g->mtp_emb = dalloc(E, g);
        g->mtp_logits = dalloc(c->n_vocab, g);
        CK(cudaMallocHost(&g->h_mtp_logits, (size_t)c->n_vocab * sizeof(float)));
        if (!g->kc[il] || !g->vc[il] || !g->mtp_logits) { fprintf(stderr, "q38_gpu: out of device memory (MTP)\n"); return -1; }
    }
    if (!g->sh_h || !g->logits || !g->fa_part) { fprintf(stderr, "q38_gpu: out of device memory\n"); return -1; }
    CK(cudaMallocHost(&g->h_pos, sizeof(int)));
    CK(cudaMalloc(&g->d_pos, sizeof(int)));
    CK(cudaMallocHost(&g->h_emb, (size_t)E * sizeof(float)));
    CK(cudaMallocHost(&g->h_res, (size_t)E * sizeof(float)));
    CK(cudaMallocHost(&g->h_logits, (size_t)(last ? c->n_vocab : 1) * sizeof(float)));
    g->hcE = E;
    const char *eg = getenv("Q38_GRAPH");
    g->use_graphs = !eg || atoi(eg) != 0;
    return 0;
}

/* The layers split so that each device keeps ~0.8 GB free (cuBLAS, the batch, the driver):
 * the most even split that fits, the device asked for first. One device when it holds it all. */
static int dense_open(q38_model *m, int device)
{
    const q38_cfg *c = &m->c;
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    /* the trunk's products with 8-bit activations (llama.cpp's mmvq); Q38_ACT8=0: float, exact */
    g_mvq = !(getenv("Q38_ACT8") && !atoi(getenv("Q38_ACT8")));
    /* K/V: Q38_KV=8|16|32; by default int8 above 32k positions (llama.cpp's -ctk q8_0: 64k of the 27B
     * fit beside its weights), fp16 below. int8 goes through the mma kernels only. */
    const char *ek = getenv("Q38_KV");
    int kvb = ek ? (atoi(ek) == 8 ? 1 : atoi(ek) == 16 ? 2 : 4) : m->max_ctx > 32768 ? 1 : 2;
    if (kvb == 1 && !((c->head_dim == 256 || c->head_dim == 128) && c->n_head % c->n_head_kv == 0 &&
                      c->n_head / c->n_head_kv <= 16)) kvb = 2;
    int devs[2] = { device, -1 }, nd = 1;
    if (n_dev >= 2 && !(getenv("Q38_ONE_GPU") && atoi(getenv("Q38_ONE_GPU")))) devs[nd++] = device == 0 ? 1 : 0;
    int64_t room[2] = { 0, 0 };
    for (int d = 0; d < nd; d++) {
        size_t fr = 0, tot = 0;
        cudaSetDevice(devs[d]);
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) return -1;
        room[d] = (int64_t)fr - (int64_t)800 * 1000 * 1000;
    }
    const int L = c->n_layer;
    int64_t lay[Q38_MAX_LAYERS], all = 0;
    int64_t head = m->output.row_bytes * m->output.rows + (int64_t)c->n_vocab * 4 + (int64_t)c->n_embd * 4;
    if (m->mtp)                                  /* the MTP block and its nextn tensors go with the head */
        head += dense_layer_bytes(m, c->n_layer) + m->mtp_eh.row_bytes * m->mtp_eh.rows + (int64_t)c->n_embd * 3 * 4;
    int split = -1;
    const int asked = m->max_ctx;
    for (;;) {                                              /* the context shrinks until it fits */
        all = 0;
        for (int il = 0; il < L; il++) { lay[il] = dense_layer_bytes(m, il) + dense_state_bytes(m, il, kvb); all += lay[il]; }
        if (m->mtp) all += dense_state_bytes(m, L, kvb);   /* the MTP block's K/V (an attention layer) */
        if (all + head <= room[0]) split = L;               /* all on the first device */
        else if (nd == 2) {
            int64_t best = INT64_MAX, a = 0;
            for (int k = 0; k <= L; k++) {
                if (k) a += lay[k - 1];
                const int64_t b = all - a + head;
                if (a <= room[0] && b <= room[1]) {
                    const int64_t d = a > b ? a - b : b - a;
                    if (d < best) { best = d; split = k; }
                }
            }
        }
        if (split >= 0 || m->max_ctx <= 8192) break;
        m->max_ctx = m->max_ctx * 3 / 4 / 1024 * 1024;
    }
    if (split >= 0 && m->max_ctx < asked)
        fprintf(stderr, "q38_gpu: context %d -> %d positions, to fit the devices\n", asked, m->max_ctx);
    if (split < 0) {
        fprintf(stderr, "q38_gpu: the model (%.2f GB with its K/V) does not fit the devices (%.2f + %.2f GB free); "
                        "a smaller context (--ctx) or the CPU\n", (all + head) / 1e9, room[0] / 1e9, nd > 1 ? room[1] / 1e9 : 0.0);
        return -1;
    }
    GD_n = split < L ? 2 : 1;
    GD_split = split;
    if (dense_part(m, devs[0], 0, split, GD_n == 1, kvb, &GD[0])) return -1;
    if (GD_n == 2 && dense_part(m, devs[1], split, L, 1, kvb, &GD[1])) return -1;
    G = GD[0];
    m->gpu = GD[0];
    /* the host keeps the embedding table only (looked up per token) */
    int64_t freed = 0;
    for (int il = 0; il < L; il++) {
        q38_mat *w = (q38_mat *)&m->L[il];
        for (int i = 0; i < NMAT; i++) {
            if (!w[i].data) continue;
            freed += w[i].row_bytes * w[i].rows;
            free(w[i].data);
            w[i].data = NULL;
        }
        free(m->kcache[il]); free(m->vcache[il]);
        m->kcache[il] = m->vcache[il] = NULL;
    }
    freed += m->output.row_bytes * m->output.rows;
    free(m->output.data);
    m->output.data = NULL;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, devs[0]);
    fprintf(stderr, "q38: dense trunk on GPU %d (%s): layers 0-%d, %.2f GB", devs[0], p.name, split - 1, GD[0]->bytes / 1e9);
    if (GD_n == 2) fprintf(stderr, "; GPU %d: layers %d-%d and the head, %.2f GB", devs[1], split, L - 1, GD[1]->bytes / 1e9);
    fprintf(stderr, " (context %d, K/V %s, %.2f GB of RAM freed)\n", m->max_ctx, kvb == 1 ? "int8" : kvb == 2 ? "fp16" : "fp32",
            freed / 1e9);
    return 0;
}

__global__ void k_set_int(int *p, int v) { *p = v; }

/* the MTP block's input rows for N tokens: [enorm(emb_t); hnorm(h_{t-1})], h_{-1} = hprev (the last
 * hidden before them), h_{t-1} = hres row t-1 after; a block per token */
__global__ void k_mtp_cat(const float *emb, const float *hres, const float *hprev, const float *enorm, const float *hnorm,
                          float *cat, int E, float eps)
{
    __shared__ float part[8];
    const int t = blockIdx.x;
    const float *e = emb + (int64_t)t * E, *h = t ? hres + (int64_t)(t - 1) * E : hprev;
    float se = 0.0f, sh = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { se += e[i] * e[i]; sh += h[i] * h[i]; }
    se = block_sum(se, part);
    sh = block_sum(sh, part);
    const float ie = rsqrtf(se / E + eps), ih = rsqrtf(sh / E + eps);
    float *o = cat + (int64_t)t * 2 * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { o[i] = e[i] * ie * enorm[i]; o[E + i] = h[i] * ih * hnorm[i]; }
}

/* the MTP block on the last part for one token at *d_pos (its embedding in h_emb on the host side):
 * res = eh_proj(cat), then the block (seg_front of layer n_layer: its K/V written); hprev becomes
 * hcur when given (the catch-up after a forward) */
static int mtp_one(q38_model *m, int token, const float *hsrc, const float *hcur)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd;
    cudaStream_t st = G->st;
    q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)token * m->tok_embd.row_bytes, G->h_emb, E);
    CK(cudaMemcpyAsync(G->mtp_emb, G->h_emb, (size_t)E * sizeof(float), cudaMemcpyHostToDevice, st));
    k_mtp_cat<<<1, 256, 0, st>>>(G->mtp_emb, NULL, hsrc ? hsrc : G->mtp_hprev, (const float *)G->mtp_enorm.d,
                                 (const float *)G->mtp_hnorm.d, G->mtp_cat, E, c->eps);
    if (hcur) CK(cudaMemcpyAsync(G->mtp_hprev, hcur, (size_t)E * sizeof(float), cudaMemcpyDeviceToDevice, st));
    if (mv(&G->mtp_eh, G->mtp_cat, G->res, st)) return -1;
    return seg_front(m, c->n_layer, 0);
}

/* the block's output through shared_head_norm and the head: m->mtp_logits */
static int mtp_logits(q38_model *m)
{
    const q38_cfg *c = &m->c;
    cudaStream_t st = G->st;
    k_rmsnorm_rows<<<1, 256, 0, st>>>(G->res, (const float *)G->mtp_shnorm.d, G->mixed, c->n_embd, c->eps);
    if (mv(&G->output, G->mixed, G->mtp_logits, st)) return -1;
    CK(cudaMemcpyAsync(G->mtp_hchain, G->mixed, (size_t)c->n_embd * sizeof(float), cudaMemcpyDeviceToDevice, st));
    CK(cudaMemcpyAsync(G->h_mtp_logits, G->mtp_logits, (size_t)c->n_vocab * sizeof(float), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    memcpy(m->mtp_logits, G->h_mtp_logits, (size_t)c->n_vocab * sizeof(float));
    return 0;
}

/* One token through the parts; logits into m->logits. */
static int dense_forward(q38_model *m, int token)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd;
    Gpu *a = GD[0];
    G = a;
    CK(cudaSetDevice(a->dev));
    *a->h_pos = m->n_past;
    q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)token * m->tok_embd.row_bytes, a->h_emb, E);
    const double t0 = now_s();
    if (run_segment(m, 0)) return -1;
    for (int il = 0; il < GD_split; il++) if (run_segment(m, GF(il))) return -1;
    if (GD_n == 2) {
        Gpu *b = GD[1];
        CK(cudaMemcpyAsync(a->h_res, a->res, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost, a->st));
        CK(cudaStreamSynchronize(a->st));
        G = b;
        CK(cudaSetDevice(b->dev));
        *b->h_pos = m->n_past;
        CK(cudaMemcpyAsync(b->d_pos, b->h_pos, sizeof(int), cudaMemcpyHostToDevice, b->st));
        CK(cudaMemcpyAsync(b->res, a->h_res, (size_t)E * sizeof(float), cudaMemcpyHostToDevice, b->st));
        for (int il = GD_split; il < c->n_layer; il++) if (run_segment(m, GF(il))) return -1;
    }
    if (run_segment(m, GT)) return -1;
    if (G->mtp_eh.d) {                            /* the MTP block's K/V at this position */
        CK(cudaMemcpyAsync(G->mixed, G->res, (size_t)E * sizeof(float), cudaMemcpyDeviceToDevice, G->st));   /* h_t */
        if (mtp_one(m, token, NULL, G->mixed)) return -1;
        if (m->tap && mtp_logits(m)) return -1;
    }
    const double tw = now_s();
    CK(cudaStreamSynchronize(G->st));
    m->t_gpu_wait += now_s() - tw;
    memcpy(m->logits, G->h_logits, (size_t)c->n_vocab * sizeof(float));
    if (m->tap) m->tap(m->tap_ctx, "logits", m->logits, c->n_vocab);
    if (m->tap && G->mtp_eh.d) m->tap(m->tap_ctx, "mtp_logits", m->mtp_logits, c->n_vocab);
    CK(cudaGetLastError());
    G = a;
    G->t_launch += tw - t0;
    G->tokens_fwd++;
    return 0;
}

static void dense_reset(q38_model *m)
{
    const q38_cfg *c = &m->c;
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    for (int p = 0; p < GD_n; p++) {
        Gpu *g = GD[p];
        cudaSetDevice(g->dev);
        for (int il = 0; il < c->n_layer; il++) {
            if (g->conv[il]) cudaMemset(g->conv[il], 0, (size_t)(c->conv_k - 1) * ch * sizeof(float));
            if (g->ssm[il]) cudaMemset(g->ssm[il], 0, (size_t)c->n_vh * S * S * sizeof(float));
        }
        if (g->mtp_hprev) cudaMemset(g->mtp_hprev, 0, (size_t)c->n_embd * sizeof(float));
        cudaDeviceSynchronize();
    }
    G = GD[0];
}

static void dense_close(q38_model *m)
{
    dense_batch_close();
    for (int p = 0; p < GD_n; p++) {
        cudaSetDevice(GD[p]->dev);
        cudaDeviceSynchronize();
        cudaDeviceReset();
        free(GD[p]);
        GD[p] = NULL;
    }
    GD_n = 0;
    G = NULL;
    m->gpu = NULL;
}

/* the MTP draft on the device: the block at n_past + j on (hprev, tok) for j = 0, on (the previous
 * draft's output, tok) for a chained one; its logits into m->mtp_logits */
extern "C" int q38_gpu_mtp_draft(void *model, int tok, int j)
{
    q38_model *m = (q38_model *)model;
    Gpu *g = GD[GD_n - 1];
    if (!g || !g->mtp_eh.d) return -1;
    G = g;
    CK(cudaSetDevice(g->dev));
    k_set_int<<<1, 1, 0, g->st>>>(g->d_pos, m->n_past + j);
    const int rc = mtp_one(m, tok, j ? g->mtp_hchain : NULL, NULL) || mtp_logits(m) ? -1 : 0;
    G = GD[0];
    return rc;
}
