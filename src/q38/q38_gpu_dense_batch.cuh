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
    float  *h_mtp, *emb2, *cat2;                 /* the MTP catch-up (last part): embeddings, block input */
    cudaEvent_t ev_mtp;
    float  *lgv, *h_lgv;                         /* a verification's logits, 4 rows (last part) */
    float  *snap_conv[Q38_MAX_LAYERS], *snap_ssm[Q38_MAX_LAYERS], *snap_h;   /* the recurrent state before it */
    /* a verification keeps the recurrent state after each of its tokens (but the last) and the hidden
     * rows: a wrong guess then costs a copy back, not a second pass */
    float  *keep_conv[Q38_MAX_LAYERS], *keep_ssm[Q38_MAX_LAYERS], *keep_h;
    int     keeping, keep_ok;
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

/* repacked Q8_0 times NC float vectors (rows ldx apart): a warp per row, each 16 quants and their
 * scale read once for all the vectors (a vector at a time, the Q8_0 matrices cost 4x in a verification) */
template <int NC>
__global__ void k_mv_q8r_nc(const int8_t *__restrict__ Q, const float *__restrict__ D, int ne0, int rows,
                            const float *__restrict__ X, int ldx, float *__restrict__ Y, int ldy)
{
    const int r = blockIdx.x * 8 + threadIdx.y;
    if (r >= rows) return;
    const int nb = ne0 / 32;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) acc[c] = 0.0f;
    for (int i = threadIdx.x * 16; i < ne0; i += 32 * 16) {
        const int4 v = __ldg((const int4 *)(Q + (int64_t)r * ne0 + i));
        const int8_t *b = (const int8_t *)&v;
        const float sc = __ldg(D + (int64_t)r * nb + i / 32);
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const float *x = X + (int64_t)c * ldx + i;
            const float4 x0 = __ldg((const float4 *)x), x1 = __ldg((const float4 *)(x + 4));
            const float4 x2 = __ldg((const float4 *)(x + 8)), x3 = __ldg((const float4 *)(x + 12));
            float t = b[0] * x0.x + b[1] * x0.y + b[2] * x0.z + b[3] * x0.w;
            t += b[4] * x1.x + b[5] * x1.y + b[6] * x1.z + b[7] * x1.w;
            t += b[8] * x2.x + b[9] * x2.y + b[10] * x2.z + b[11] * x2.w;
            t += b[12] * x3.x + b[13] * x3.y + b[14] * x3.z + b[15] * x3.w;
            acc[c] += sc * t;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const float s = warp_sum(acc[c]);
        if (threadIdx.x == 0) Y[(int64_t)c * ldy + r] = s;
    }
}

/* Y [N][rows] for N <= 4 tokens (a verification): the 8-bit products, weights read once for all;
 * other types a vector at a time */
static int mvn(const DMat *w, const float *X, int ldx, float *Y, int ldy, int N, cudaStream_t st)
{
    if (w->type == Q38_Q8R && N > 1 && w->ne0 % 512 == 0 && ldx % 4 == 0) {
        const dim3 b8(32, 8);
        const int g8 = (int)((w->rows + 7) / 8);
        const int8_t *Q = (const int8_t *)w->d;
        if (N == 2) k_mv_q8r_nc<2><<<g8, b8, 0, st>>>(Q, w->scale, (int)w->ne0, (int)w->rows, X, ldx, Y, ldy);
        else if (N == 3) k_mv_q8r_nc<3><<<g8, b8, 0, st>>>(Q, w->scale, (int)w->ne0, (int)w->rows, X, ldx, Y, ldy);
        else k_mv_q8r_nc<4><<<g8, b8, 0, st>>>(Q, w->scale, (int)w->ne0, (int)w->rows, X, ldx, Y, ldy);
        return 0;
    }
    if (mvq_ok(w->type) && G->q8x && w->ne0 % 256 == 0 && w->ne0 <= 32768) {
        const int nb = (int)(w->ne0 / 32);
        k_q8x_nc<<<(nb * N + 3) / 4, 128, 0, st>>>(X, ldx, (int)w->ne0, N, G->q8x, G->q8d, G->q8s);
        const dim3 b4(32, 4), b8(32, 8);
        const int g4 = (int)((w->rows + 3) / 4), g32 = (int)((w->rows + 31) / 32);
        const uint8_t *W = (const uint8_t *)w->d;
        /* x in shared memory (up to the 99 KB mvq_setup allows; Q38_MVN_SH=0: never) */
        static int shx = -1;
        if (shx < 0) shx = !(getenv("Q38_MVN_SH") && !atoi(getenv("Q38_MVN_SH")));
        const size_t smem = (size_t)N * w->ne0 + (size_t)2 * N * (w->ne0 / 32) * sizeof(float);
        const int sh = shx && N > 1 && smem <= 99 * 1024;
#define MVN_CASE(T, NC) do { if (sh) k_mvq_ncs<T, NC><<<g32, b8, smem, st>>>(W, w->row_bytes, (int)w->ne0, (int)w->rows, G->q8x, G->q8d, G->q8s, Y, ldy); \
                             else k_mvq_nc<T, NC><<<g4, b4, 0, st>>>(W, w->row_bytes, (int)w->ne0, (int)w->rows, G->q8x, G->q8d, G->q8s, Y, ldy); } while (0)
#define MVN_TYPE(T) switch (N) { case 1: MVN_CASE(T, 1); break; case 2: MVN_CASE(T, 2); break; \
                                 case 3: MVN_CASE(T, 3); break; default: MVN_CASE(T, 4); break; }
        switch (w->type) {
        case Q38_Q4_K:   MVN_TYPE(Q38_Q4_K); break;
        case Q38_Q5_K:   MVN_TYPE(Q38_Q5_K); break;
        case Q38_Q6_K:   MVN_TYPE(Q38_Q6_K); break;
        case Q38_IQ4_XS: MVN_TYPE(Q38_IQ4_XS); break;
        default:         MVN_TYPE(Q38_IQ4_NL); break;
        }
#undef MVN_TYPE
#undef MVN_CASE
        return 0;
    }
    for (int t = 0; t < N; t++) if (mv(w, X + (int64_t)t * ldx, Y + (int64_t)t * ldy, st)) return -1;
    return 0;
}

/* Y [N][rows] (row stride ldy) = W . X for N tokens */
static int dmm(DBat *d, const DMat *w, const float *X, int ldx, float *Y, int ldy, int N, cudaStream_t st)
{
    if (N <= 4) return mvn(w, X, ldx, Y, ldy, N, st);   /* a verification: the decode's products */
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
    if (g->mtp_eh.d) {
        CK(cudaMallocHost(&d->h_mtp, (size_t)nb * E * sizeof(float)));
        if (!(d->emb2 = dalloc((size_t)nb * E, g)) || !(d->cat2 = dalloc((size_t)nb * 2 * E, g))) return -1;
        CK(cudaEventCreateWithFlags(&d->ev_mtp, cudaEventDisableTiming));
    }
    int64_t wmax = g->mtp_eh.d ? g->mtp_eh.rows * g->mtp_eh.ne0 : 0, xmax = g->mtp_eh.d ? g->mtp_eh.ne0 : 0;
    for (int il = l0; il < l1 + (g->mtp_eh.d ? 1 : 0); il++)
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
        k_conv_silu_b<<<nblocks(ch, 256), 256, 0, st>>>(d->qkv, ch, G->conv[il], (const float *)LM(il, conv).d, ch, c->conv_k, N,
                                                       d->keeping ? d->keep_conv[il] : NULL);
        k_l2norm_b<<<N * 2 * Hk, 32, 0, st>>>(d->qkv, S, 2 * Hk, ch, c->eps);
        k_gdn_step_b<<<Hv, dim3(S, 4), 6 * S * sizeof(float), st>>>(G->ssm[il], d->qkv, ch, d->g, d->beta, d->o, S, Hk, Hv, N,
                                                                   d->keeping ? d->keep_ssm[il] : NULL);
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


/* n tokens in pieces of nb, pipelined over the parts: part 0 starts piece k+1 while part 1 runs
 * piece k (each its own device and stream; the residual goes through pinned slots, the next part's
 * stream waiting on the event of the copy). Positions are set by a kernel argument, so queued
 * pieces each keep theirs. The last piece's last token gets the logits. */
static int mtp_catchup_b(q38_model *m, DBat *d, const int *toks, int N, cudaStream_t st);   /* below */

static int dense_bat_ready(q38_model *m)
{
    if (GDB[0]) return 0;
    const char *e = getenv("Q38_BATCH");
    int nb = e ? atoi(e) : 1024;                 /* 1024: 435 tok/s on the 27B, 512: 421, 256: 379 */
    if (nb < 16) nb = 16;
    if (nb > 2048) nb = 2048;
    const int rc = dbat_open(m, 0, 0, GD_split, nb) || (GD_n == 2 && dbat_open(m, 1, GD_split, m->c.n_layer, nb)) ? -1 : 0;
    G = GD[0];
    return rc;
}

static int dense_forward_batch(q38_model *m, const int *toks, int n)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd;
    if (dense_bat_ready(m)) return -1;
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
            if (p + 1 == GD_n && G->mtp_eh.d && mtp_catchup_b(m, d, toks + i, N, st)) {   /* the MTP block's K/V */
                G = GD[0];
                return -1;
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

/* The MTP block's K/V over N tokens just through the last part (d->res holds their hidden rows):
 * pairs (h of the position before, token), hprev the last row after */
static int mtp_catchup_b(q38_model *m, DBat *d, const int *toks, int N, cudaStream_t st)
{
    const q38_cfg *c = &m->c;
    const int E = c->n_embd;
    CK(cudaEventSynchronize(d->ev_mtp));
    for (int t = 0; t < N; t++)
        q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)toks[t] * m->tok_embd.row_bytes, d->h_mtp + (int64_t)t * E, E);
    CK(cudaMemcpyAsync(d->emb2, d->h_mtp, (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, st));
    CK(cudaEventRecord(d->ev_mtp, st));
    k_mtp_cat<<<N, 256, 0, st>>>(d->emb2, d->res, G->mtp_hprev, (const float *)G->mtp_enorm.d, (const float *)G->mtp_hnorm.d,
                                 d->cat2, E, c->eps);
    CK(cudaMemcpyAsync(G->mtp_hprev, d->res + (int64_t)(N - 1) * E, (size_t)E * sizeof(float), cudaMemcpyDeviceToDevice, st));
    return dmm(d, &G->mtp_eh, d->cat2, 2 * E, d->res, E, N, st) || dense_layer_b(m, d, c->n_layer, N, st) ? -1 : 0;
}

/* A verification: N <= 4 tokens at n_past through the parts (the decode's 8-bit products, weights read
 * once for all of them), the logits of every one into logits [N][n_vocab], the MTP catch-up. */
/* the buffers a verification keeps its states in (Q38_SPEC_MAX copies a recurrent layer); 0 when they fit */
static int keep_open(q38_model *m)
{
    const q38_cfg *c = &m->c;
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    for (int p = 0; p < GD_n; p++) {
        DBat *d = GDB[p];
        if (d->keep_ok) continue;
        CK(cudaSetDevice(GD[p]->dev));
        for (int il = 0; il < c->n_layer; il++) {
            if (!GD[p]->conv[il]) continue;
            if (cudaMalloc(&d->keep_conv[il], (size_t)Q38_SPEC_MAX * (c->conv_k - 1) * ch * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d->keep_ssm[il], (size_t)Q38_SPEC_MAX * c->n_vh * S * S * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                return -1;
            }
        }
        if (cudaMalloc(&d->keep_h, (size_t)4 * c->n_embd * sizeof(float)) != cudaSuccess) { cudaGetLastError(); return -1; }
        d->keep_ok = 1;
    }
    return 0;
}

extern "C" int q38_gpu_verify(void *model, const int *toks, int N, float *logits)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    const int E = c->n_embd, p0 = m->n_past;
    if (N < 1 || N > 4 || dense_bat_ready(m)) return -1;
    static int keep = -1;                        /* the states kept per token: when their buffers fit */
    if (keep < 0) keep = !(getenv("Q38_SPEC_KEEP") && !atoi(getenv("Q38_SPEC_KEEP"))) && keep_open(m) == 0;
    for (int p = 0; p < GD_n; p++) GDB[p]->keeping = keep && N > 1;
    for (int p = 0; p < GD_n; p++) {
        G = GD[p];
        DBat *d = GDB[p];
        cudaStream_t st = G->st;
        CK(cudaSetDevice(G->dev));
        if (p == 0) {
            CK(cudaEventSynchronize(d->ev_in[0]));
            for (int t = 0; t < N; t++)
                q38_dequant_row(m->tok_embd.type, m->tok_embd.data + (int64_t)toks[t] * m->tok_embd.row_bytes,
                                d->h_in[0] + (int64_t)t * E, E);
            CK(cudaMemcpyAsync(d->res, d->h_in[0], (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, st));
            CK(cudaEventRecord(d->ev_in[0], st));
        } else {
            CK(cudaMemcpyAsync(d->res, GDB[p - 1]->h_out[0], (size_t)N * E * sizeof(float), cudaMemcpyHostToDevice, st));
        }
        k_set_int<<<1, 1, 0, st>>>(G->d_pos, p0);
        const int l0 = p == 0 ? 0 : GD_split, l1 = p == 0 ? GD_split : c->n_layer;
        for (int il = l0; il < l1; il++) if (dense_layer_b(m, d, il, N, st)) { G = GD[0]; return -1; }
        if (p + 1 < GD_n) {
            CK(cudaMemcpyAsync(d->h_out[0], d->res, (size_t)N * E * sizeof(float), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            continue;
        }
        if (!d->lgv) {
            if (!(d->lgv = dalloc((size_t)4 * c->n_vocab, G))) { G = GD[0]; return -1; }
            CK(cudaMallocHost(&d->h_lgv, (size_t)4 * c->n_vocab * sizeof(float)));
        }
        k_rmsnorm_rows<<<N, 256, 0, st>>>(d->res, (const float *)G->output_norm.d, d->mixed, E, c->eps);
        if (mvn(&G->output, d->mixed, E, d->lgv, c->n_vocab, N, st)) { G = GD[0]; return -1; }
        CK(cudaMemcpyAsync(d->h_lgv, d->lgv, (size_t)N * c->n_vocab * sizeof(float), cudaMemcpyDeviceToHost, st));
        if (d->keeping) CK(cudaMemcpyAsync(d->keep_h, d->res, (size_t)N * E * sizeof(float), cudaMemcpyDeviceToDevice, st));
        if (G->mtp_eh.d && mtp_catchup_b(m, d, toks, N, st)) { G = GD[0]; return -1; }
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        memcpy(logits, d->h_lgv, (size_t)N * c->n_vocab * sizeof(float));
    }
    for (int t = 0; t < N; t++) m->toks[p0 + t] = toks[t];
    m->n_past = p0 + N;
    for (int p = 0; p < GD_n; p++) GDB[p]->keeping = 0;
    G = GD[0];
    return keep && N > 1 ? 1 : 0;                /* 1: the states after each token were kept */
}

/* After a verification of N tokens that kept its states (q38_gpu_verify returned 1): only its first
 * a + 1 tokens stand. The recurrent state and the MTP block's hidden go back to those after token a;
 * n_past to after it (the K/V past it is written again later). */
extern "C" int q38_gpu_spec_keep(void *model, int a, int N)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    const size_t conv_b = (size_t)(c->conv_k - 1) * ch * sizeof(float), ssm_b = (size_t)c->n_vh * S * S * sizeof(float);
    if (a >= N - 1) return 0;
    for (int p = 0; p < GD_n; p++) {
        Gpu *g = GD[p];
        DBat *d = GDB[p];
        CK(cudaSetDevice(g->dev));
        for (int il = 0; il < c->n_layer; il++) {
            if (!g->conv[il]) continue;
            CK(cudaMemcpyAsync(g->conv[il], (const uint8_t *)d->keep_conv[il] + (size_t)a * conv_b, conv_b, cudaMemcpyDeviceToDevice, g->st));
            CK(cudaMemcpyAsync(g->ssm[il], (const uint8_t *)d->keep_ssm[il] + (size_t)a * ssm_b, ssm_b, cudaMemcpyDeviceToDevice, g->st));
        }
        if (g->mtp_hprev)
            CK(cudaMemcpyAsync(g->mtp_hprev, d->keep_h + (size_t)a * c->n_embd, (size_t)c->n_embd * sizeof(float),
                               cudaMemcpyDeviceToDevice, g->st));
    }
    m->n_past -= N - 1 - a;
    G = GD[0];
    return 0;
}

/* The recurrent state (GDN and convolution of every layer, the MTP block's hidden) copied aside before
 * a verification, and put back when its draft was wrong (the K/V past n_past is just written again). */
extern "C" int q38_gpu_spec_state(void *model, int restore)
{
    q38_model *m = (q38_model *)model;
    const q38_cfg *c = &m->c;
    const int S = c->d_state, ch = 2 * c->n_kh * S + c->n_vh * S;
    const size_t conv_b = (size_t)(c->conv_k - 1) * ch * sizeof(float), ssm_b = (size_t)c->n_vh * S * S * sizeof(float);
    if (dense_bat_ready(m)) return -1;
    for (int p = 0; p < GD_n; p++) {
        Gpu *g = GD[p];
        DBat *d = GDB[p];
        CK(cudaSetDevice(g->dev));
        for (int il = 0; il < c->n_layer; il++) {
            if (!g->conv[il]) continue;
            if (!d->snap_conv[il]) {
                CK(cudaMalloc(&d->snap_conv[il], conv_b));
                CK(cudaMalloc(&d->snap_ssm[il], ssm_b));
            }
            CK(cudaMemcpyAsync(restore ? g->conv[il] : d->snap_conv[il], restore ? d->snap_conv[il] : g->conv[il], conv_b,
                               cudaMemcpyDeviceToDevice, g->st));
            CK(cudaMemcpyAsync(restore ? g->ssm[il] : d->snap_ssm[il], restore ? d->snap_ssm[il] : g->ssm[il], ssm_b,
                               cudaMemcpyDeviceToDevice, g->st));
        }
        if (g->mtp_hprev) {
            if (!d->snap_h) CK(cudaMalloc(&d->snap_h, (size_t)c->n_embd * sizeof(float)));
            CK(cudaMemcpyAsync(restore ? g->mtp_hprev : d->snap_h, restore ? d->snap_h : g->mtp_hprev,
                               (size_t)c->n_embd * sizeof(float), cudaMemcpyDeviceToDevice, g->st));
        }
    }
    G = GD[0];
    return 0;
}

static void dense_batch_close(void)
{
    for (int p = 0; p < 2; p++) {
        if (!GDB[p]) continue;
        if (GDB[p]->h_mtp) cudaFreeHost(GDB[p]->h_mtp);
        for (int k = 0; k < 2; k++) {
            if (GDB[p]->h_in[k]) cudaFreeHost(GDB[p]->h_in[k]);
            if (GDB[p]->h_out[k]) cudaFreeHost(GDB[p]->h_out[k]);
        }
        free(GDB[p]);
        GDB[p] = NULL;
    }
}
