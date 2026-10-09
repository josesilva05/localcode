/* q38_gpu_dense_batch.cuh - prompt passes of the dense models (included by q38_gpu.cu after
 * q38_gpu_batch.cuh, whose kernels it uses).
 *
 * N tokens a layer at a time on each part (q38_gpu_dense.cuh): the Gated DeltaNet walks them in
 * order inside its kernel, attention is the batch's flash kernel, and every matrix product is one
 * GEMM on the tensor cores: the weights dequantised to fp16 (whatever their GGUF type), the
 * activations converted to fp16, fp32 accumulation (cuBLAS). llama.cpp reads the 27B's prompt at
 * ~490 tok/s on these cards; a token at a time it was ~15. */

typedef struct {
    int     nb;                                  /* tokens per pass */
    float  *res, *mixed, *inject, *y, *qkv, *z, *o, *beta, *g, *qf, *knew, *vnew, *aout, *shgu, *shh;
    float  *h_in[2], *h_out[2];                  /* [nb][E] pinned, two slots each: the embeddings (part 0),
                                                  * the residual for the next part */
    cudaEvent_t ev_in[2], ev_out[2];             /* this part read slot k / wrote its result for the next */
    cublasHandle_t blas;
    __half *wh, *xh;                             /* a matrix in fp16; the activations in fp16 */
    int64_t wcap, xcap;
} DBat;

static DBat *GDB[2];

/* chunk idx of a matrix kept in its GGUF type, as fp16 into out [rows][ne0] */
__global__ void k_deq_nat(const uint8_t *__restrict__ W, int64_t rb, int ne0, int rows, int type, __half *__restrict__ out)
{
    const int kq = type == Q38_Q4_K || type == Q38_Q5_K, five = type == Q38_Q5_K;
    const int nch = kq ? (ne0 / 256) * 32 : iq_chunks(type, ne0);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (int64_t)rows * nch; i += (int64_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / nch), c = (int)(i % nch);
        float wt[8];
        int o0, o1;
        if (kq) kq_chunk(five, W + (int64_t)r * rb, c, wt, &o0, &o1);
        else iq_chunk(type, W + (int64_t)r * rb, c, wt, &o0, &o1);
        __half *o = out + (int64_t)r * ne0;
#pragma unroll
        for (int k = 0; k < 4; k++) { o[o0 + k] = __float2half(wt[k]); o[o1 + k] = __float2half(wt[4 + k]); }
    }
}

__global__ void k_deq_q8r_h(const int8_t *__restrict__ Q, const float *__restrict__ D, int64_t n, __half *__restrict__ W)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        W[i] = __float2half((float)Q[i] * D[i >> 5]);
}

/* N rows of ne0 (row stride ldx) to fp16, packed */
__global__ void k_rows_h(const float *__restrict__ X, int ldx, int ne0, int N, __half *__restrict__ out)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (int64_t)N * ne0; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t t = i / ne0, k = i % ne0;
        out[i] = __float2half(fminf(fmaxf(X[t * ldx + k], -65504.0f), 65504.0f));
    }
}

/* Y [N][rows] (row stride ldy) = W . X for N tokens */
static int dmm(DBat *d, const DMat *w, const float *X, int ldx, float *Y, int ldy, int N, cudaStream_t st)
{
    const int64_t n = w->rows * w->ne0;
    if (w->type == Q38_F32) {
        const dim3 blk(32, ROWS_PER_BLOCK), grid(nblocks(w->rows, ROWS_PER_BLOCK), nblocks(N, TB));
        k_mm_f32<<<grid, blk, 0, st>>>((const float *)w->d, (int)w->ne0, (int)w->rows, X, ldx, Y, ldy, N);
        return 0;
    }
    if (n > d->wcap || (int64_t)N * w->ne0 > d->xcap) { fprintf(stderr, "q38_gpu: dense batch scratch too small\n"); return -1; }
    if (w->type == Q38_Q8R) k_deq_q8r_h<<<1024, 256, 0, st>>>((const int8_t *)w->d, w->scale, n, d->wh);
    else if (w->type == Q38_Q4_K || w->type == Q38_Q5_K || w->type == Q38_Q6_K || w->type == Q38_IQ4_XS ||
             w->type == Q38_IQ4_NL || w->type == Q38_IQ3_S)
        k_deq_nat<<<1024, 256, 0, st>>>((const uint8_t *)w->d, w->row_bytes, (int)w->ne0, (int)w->rows, w->type, d->wh);
    else { fprintf(stderr, "q38_gpu: no batched product for type %s\n", q38_type_name(w->type)); return -1; }
    k_rows_h<<<512, 256, 0, st>>>(X, ldx, (int)w->ne0, N, d->xh);
    const float one = 1.0f, zero = 0.0f;
    cublasSetStream(d->blas, st);
    if (cublasGemmEx(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)w->rows, N, (int)w->ne0, &one, d->wh, CUDA_R_16F, (int)w->ne0,
                     d->xh, CUDA_R_16F, (int)w->ne0, &zero, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "q38_gpu: cublasGemmEx failed (dense)\n");
        return -1;
    }
    return 0;
}

/* a part's batch buffers, on its device (G) */
static int dbat_open(q38_model *m, int p, int l0, int l1, int nb)
{
    const q38_cfg *c = &m->c;
    Gpu *g = GD[p];
    CK(cudaSetDevice(g->dev));
    DBat *d = (DBat *)calloc(1, sizeof(DBat));
    if (!d) return -1;
    GDB[p] = d;
    d->nb = nb;
    const int E = c->n_embd, S = c->d_state, F = c->ff_shexp, ch = 2 * c->n_kh * S + c->n_vh * S;
#define DA2(ptr, cnt) do { if (!((ptr) = dalloc((size_t)(cnt), g))) return -1; } while (0)
    DA2(d->res, (int64_t)nb * E); DA2(d->mixed, (int64_t)nb * E); DA2(d->inject, nb); DA2(d->y, (int64_t)nb * E);
    DA2(d->qkv, (int64_t)nb * ch); DA2(d->z, (int64_t)nb * c->n_vh * S); DA2(d->o, (int64_t)nb * c->n_vh * S);
    DA2(d->beta, (int64_t)nb * c->n_vh); DA2(d->g, (int64_t)nb * c->n_vh);
    DA2(d->qf, (int64_t)nb * c->n_head * c->head_dim * 2);
    DA2(d->knew, (int64_t)nb * c->n_head_kv * c->head_dim); DA2(d->vnew, (int64_t)nb * c->n_head_kv * c->head_dim);
    DA2(d->aout, (int64_t)nb * c->n_head * c->head_dim);
    DA2(d->shgu, (int64_t)nb * 2 * F); DA2(d->shh, (int64_t)nb * F);
#undef DA2
    CK(cudaMemset(d->inject, 0, (size_t)nb * sizeof(float)));       /* plain blocks: res += out */
    for (int k = 0; k < 2; k++) {
        CK(cudaMallocHost(&d->h_in[k], (size_t)nb * E * sizeof(float)));
        CK(cudaMallocHost(&d->h_out[k], (size_t)nb * E * sizeof(float)));
        CK(cudaEventCreateWithFlags(&d->ev_in[k], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&d->ev_out[k], cudaEventDisableTiming));
    }
    int64_t wmax = 0, xmax = 0;
    for (int il = l0; il < l1; il++)
        for (int i = 0; i < NMAT; i++) {
            const DMat *w = &g->L[il][i];
            if (!w->d || w->type == Q38_F32) continue;
            if (w->rows * w->ne0 > wmax) wmax = w->rows * w->ne0;
            if (w->ne0 > xmax) xmax = w->ne0;
        }
    d->wcap = wmax;
    d->xcap = (int64_t)nb * xmax;
    if (cudaMalloc(&d->wh, (size_t)d->wcap * sizeof(__half)) != cudaSuccess ||
        cudaMalloc(&d->xh, (size_t)d->xcap * sizeof(__half)) != cudaSuccess ||
        cublasCreate(&d->blas) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "q38_gpu: no room for the dense batch on GPU %d\n", g->dev);
        return -1;
    }
    g->bytes += (d->wcap + d->xcap) * 2;
    return 0;
}

/* layer il for the N tokens in d->res (positions *d_pos ..) */
static int dense_layer_b(q38_model *m, DBat *d, int il, int N, cudaStream_t st)
{
    const q38_cfg *c = &m->c;
    const q38_layer *l = &m->L[il];
    const int E = c->n_embd, S = c->d_state, Hk = c->n_kh, Hv = c->n_vh, F = c->ff_shexp;
    const int ch = 2 * Hk * S + Hv * S;
    k_rmsnorm_rows_w<<<N, 256, 0, st>>>(d->res, (const float *)LM(il, attn_norm).d, d->mixed, E, c->eps, 1);
    if (l->recurrent) {
        if (dmm(d, &LM(il, qkv), d->mixed, E, d->qkv, ch, N, st) || dmm(d, &LM(il, gate), d->mixed, E, d->z, Hv * S, N, st) ||
            dmm(d, &LM(il, beta), d->mixed, E, d->beta, Hv, N, st) || dmm(d, &LM(il, alpha), d->mixed, E, d->g, Hv, N, st))
            return -1;
        k_gdn_gates_b<<<nblocks(Hv * N, 256), 256, 0, st>>>(d->beta, d->g, (const float *)LM(il, dt_bias).d,
                                                           (const float *)LM(il, a).d, Hv, N);
        k_conv_silu_b<<<nblocks(ch, 256), 256, 0, st>>>(d->qkv, ch, G->conv[il], (const float *)LM(il, conv).d, ch, c->conv_k, N);
        k_l2norm_b<<<N * 2 * Hk, 32, 0, st>>>(d->qkv, S, 2 * Hk, ch, c->eps);
        k_gdn_step_b<<<Hv, dim3(S, 4), 6 * S * sizeof(float), st>>>(G->ssm[il], d->qkv, ch, d->g, d->beta, d->o, S, Hk, Hv, N);
        k_norm_gate<<<N * Hv, 128, 0, st>>>(d->o, d->z, (const float *)LM(il, ssm_norm).d, S, c->eps, c->gdn_silu);
        if (dmm(d, &LM(il, ssm_out), d->o, Hv * S, d->y, E, N, st)) return -1;
    } else {
        const int H = c->n_head, Hkv = c->n_head_kv, D = c->head_dim;
        if (dmm(d, &LM(il, wq), d->mixed, E, d->qf, H * 2 * D, N, st) || dmm(d, &LM(il, wk), d->mixed, E, d->knew, Hkv * D, N, st) ||
            dmm(d, &LM(il, wv), d->mixed, E, d->vnew, Hkv * D, N, st)) return -1;
        k_norm_rope_b<<<N * H, 256, 0, st>>>(d->qf, 2 * D, H, (int64_t)H * 2 * D, (const float *)LM(il, q_norm).d, D,
                                            c->n_rot, G->d_pos, c->rope_base, c->eps);
        k_norm_rope_b<<<N * Hkv, 256, 0, st>>>(d->knew, D, Hkv, (int64_t)Hkv * D, (const float *)LM(il, k_norm).d, D,
                                              c->n_rot, G->d_pos, c->rope_base, c->eps);
        kv_store(G, il, d->knew, d->vnew, Hkv * D, N, st);
        attn(G, il, d->qf, d->aout, H, Hkv, D, NULL, c->max_dense_ctx, N, st);
        if (dmm(d, &LM(il, wo), d->aout, H * D, d->y, E, N, st)) return -1;
    }
    k_hc_combine_b<<<nblocks((int64_t)N * E, 256), 256, 0, st>>>(d->res, d->y, d->inject, E, 1, N);
    k_rmsnorm_rows_w<<<N, 256, 0, st>>>(d->res, (const float *)LM(il, post_norm).d, d->mixed, E, c->eps, 1);
    if (dmm(d, &LM(il, sh_gate), d->mixed, E, d->shgu, 2 * F, N, st) ||
        dmm(d, &LM(il, sh_up), d->mixed, E, d->shgu + F, 2 * F, N, st)) return -1;
    k_swiglu_b<<<nblocks((int64_t)N * F, 256), 256, 0, st>>>(d->shgu, d->shh, F, N);
    if (dmm(d, &LM(il, sh_down), d->shh, F, d->y, E, N, st)) return -1;
    k_hc_combine_b<<<nblocks((int64_t)N * E, 256), 256, 0, st>>>(d->res, d->y, d->inject, E, 1, N);
    return 0;
}

__global__ void k_set_int(int *p, int v) { *p = v; }

/* n tokens in pieces of nb, pipelined over the parts: part 0 starts piece k+1 while part 1 runs
 * piece k (each its own device and stream; the residual goes through pinned slots, the next part's
 * stream waiting on the event of the copy). Positions are set by a kernel argument, so queued
 * pieces each keep theirs. The last piece's last token gets the logits. */
static int dense_forward_batch(q38_model *m, const int *toks, int n)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd;
    if (!GDB[0]) {
        const char *e = getenv("Q38_BATCH");
        int nb = e ? atoi(e) : 1024;            /* 1024: 435 tok/s on the 27B, 512: 421, 256: 379 */
        if (nb < 16) nb = 16;
        if (nb > 2048) nb = 2048;
        if (dbat_open(m, 0, 0, GD_split, nb) || (GD_n == 2 && dbat_open(m, 1, GD_split, c->n_layer, nb))) {
            G = GD[0];
            return -1;
        }
    }
    const int nb = GDB[0]->nb, p00 = m->n_past;
    const double t0 = now_s();
    for (int i = 0, k = 0; i < n; i += nb, k ^= 1) {
        const int N = n - i < nb ? n - i : nb, p0 = p00 + i, last = i + N == n;
        for (int p = 0; p < GD_n; p++) {
            G = GD[p];
            DBat *d = GDB[p];
            cudaStream_t st = G->st;
            CK(cudaSetDevice(G->dev));
            if (p == 0) {
                CK(cudaEventSynchronize(d->ev_in[k]));            /* slot k free: its last copy is done */
                for (int t = 0; t < N; t++)
                    q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)toks[i + t] * m->tok_embd.row_bytes,
                                    d->h_in[k] + (int64_t)t * E, E);
                CK(cudaMemcpyAsync(d->res, d->h_in[k], (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, st));
            } else {
                DBat *prev = GDB[p - 1];
                CK(cudaStreamWaitEvent(st, prev->ev_out[k], 0));  /* the part before wrote slot k */
                CK(cudaMemcpyAsync(d->res, prev->h_out[k], (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, st));
            }
            CK(cudaEventRecord(d->ev_in[k], st));
            k_set_int<<<1, 1, 0, st>>>(G->d_pos, p0);
            const int l0 = p == 0 ? 0 : GD_split, l1 = p == 0 ? GD_split : c->n_layer;
            for (int il = l0; il < l1; il++) if (dense_layer_b(m, d, il, N, st)) { G = GD[0]; return -1; }
            if (p + 1 < GD_n) {
                /* slot k of this part's output: the next part has read the previous piece in it */
                CK(cudaStreamWaitEvent(st, GDB[p + 1]->ev_in[k], 0));
                CK(cudaMemcpyAsync(d->h_out[k], d->res, (size_t)N * E * sizeof(float), cudaMemcpyDeviceToHost, st));
                CK(cudaEventRecord(d->ev_out[k], st));
            } else if (last) {                                    /* the head, for the last token */
                k_rmsnorm_rows<<<1, 256, 0, st>>>(d->res + (int64_t)(N - 1) * E, (const float *)G->output_norm.d, G->mixed, E, c->eps);
                if (mv(&G->output, G->mixed, G->logits, st)) { G = GD[0]; return -1; }
                CK(cudaMemcpyAsync(G->h_logits, G->logits, (size_t)c->n_vocab * sizeof(float), cudaMemcpyDeviceToHost, st));
            }
        }
    }
    for (int p = GD_n - 1; p >= 0; p--) {
        CK(cudaSetDevice(GD[p]->dev));
        CK(cudaStreamSynchronize(GD[p]->st));
        CK(cudaGetLastError());
    }
    memcpy(m->logits, GD[GD_n - 1]->h_logits, (size_t)c->n_vocab * sizeof(float));
    for (int t = 0; t < n; t++) m->toks[p00 + t] = toks[t];
    m->n_past = p00 + n;
    G = GD[0];
    if (getenv("Q38_BATCH_VERBOSE"))
        fprintf(stderr, "q38: dense prompt of %d tokens in pieces of %d: %.2f s (%.0f tok/s)\n", n, nb, now_s() - t0,
                n / (now_s() - t0));
    return 0;
}

static void dense_batch_close(void)
{
    for (int p = 0; p < 2; p++) {
        if (!GDB[p]) continue;
        for (int k = 0; k < 2; k++) {
            if (GDB[p]->h_in[k]) cudaFreeHost(GDB[p]->h_in[k]);
            if (GDB[p]->h_out[k]) cudaFreeHost(GDB[p]->h_out[k]);
        }
        free(GDB[p]);
        GDB[p] = NULL;
    }
}
