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

/* K/V (attention layers) or recurrent state of a layer on the device */
static int64_t dense_state_bytes(const q38_model *m, int il, int kv_half)
{
    const q38_cfg *c = &m->c;
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    if (m->L[il].recurrent) return ((int64_t)(c->conv_k - 1) * ch + (int64_t)c->n_vh * S * S) * 4;
    return 2 * (int64_t)m->max_ctx * c->n_head_kv * c->head_dim * (kv_half ? 2 : 4);
}

/* one device's part: layers [l0, l1), the head when last */
static int dense_part(q38_model *m, int dev, int l0, int l1, int last, int kv_half, Gpu **out)
{
    const q38_cfg *c = &m->c;
    CK(cudaSetDevice(dev));
    Gpu *g = (Gpu *)calloc(1, sizeof(Gpu));
    if (!g) return -1;
    *out = g;
    G = g;
    g->dev = dev;
    g->kv_half = kv_half;
    CK(cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&g->ev, cudaEventDisableTiming | cudaEventBlockingSync));
    for (int il = l0; il < l1; il++) {
        const q38_mat *w = (const q38_mat *)&m->L[il];
        for (int i = 0; i < NMAT; i++)
            if (upload(&g->L[il][i], &w[i], g)) return -1;
    }
    if (last && (upload(&g->output, &m->output, g) || upload(&g->output_norm, &m->output_norm, g))) return -1;
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
    CK(cudaMalloc(&g->q8x, 32768));               /* x in 8 bits: rows up to 32768 wide */
    g->q8d = dalloc(1024, g); g->q8s = dalloc(1024, g);
    g->logits = dalloc(last ? c->n_vocab : 1, g);
    for (int il = l0; il < l1; il++) {
        if (m->L[il].recurrent) {
            g->conv[il] = dalloc((size_t)(c->conv_k - 1) * ch, g);
            g->ssm[il] = dalloc((size_t)c->n_vh * S * S, g);
            if (!g->conv[il] || !g->ssm[il]) { fprintf(stderr, "q38_gpu: out of device memory (state)\n"); return -1; }
        } else {
            const size_t kv_f = ((size_t)m->max_ctx * c->n_head_kv * c->head_dim * (kv_half ? 2 : 4) + 3) / 4;
            g->kc[il] = dalloc(kv_f, g);
            g->vc[il] = dalloc(kv_f, g);
            if (!g->kc[il] || !g->vc[il]) { fprintf(stderr, "q38_gpu: out of device memory (K/V)\n"); return -1; }
        }
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
    const char *ek = getenv("Q38_KV");
    const int kv_half = ek ? atoi(ek) == 16 : 1;            /* fp16 K/V by default: VRAM is tight */
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
    for (int il = 0; il < L; il++) { lay[il] = dense_layer_bytes(m, il) + dense_state_bytes(m, il, kv_half); all += lay[il]; }
    const int64_t head = m->output.row_bytes * m->output.rows + (int64_t)c->n_vocab * 4 + (int64_t)c->n_embd * 4;
    int split = -1;
    if (all + head <= room[0]) split = L;                   /* all on the first device */
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
    if (split < 0) {
        fprintf(stderr, "q38_gpu: the model (%.2f GB with its K/V) does not fit the devices (%.2f + %.2f GB free); "
                        "a smaller context (--ctx) or the CPU\n", (all + head) / 1e9, room[0] / 1e9, nd > 1 ? room[1] / 1e9 : 0.0);
        return -1;
    }
    GD_n = split < L ? 2 : 1;
    GD_split = split;
    if (dense_part(m, devs[0], 0, split, GD_n == 1, kv_half, &GD[0])) return -1;
    if (GD_n == 2 && dense_part(m, devs[1], split, L, 1, kv_half, &GD[1])) return -1;
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
    fprintf(stderr, " (K/V %s, %.2f GB of RAM freed)\n", kv_half ? "fp16" : "fp32", freed / 1e9);
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
    const double tw = now_s();
    CK(cudaStreamSynchronize(G->st));
    m->t_gpu_wait += now_s() - tw;
    memcpy(m->logits, G->h_logits, (size_t)c->n_vocab * sizeof(float));
    if (m->tap) m->tap(m->tap_ctx, "logits", m->logits, c->n_vocab);
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
