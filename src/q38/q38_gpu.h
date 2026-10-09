/* q38_gpu.h - the dense part of Qwen3.8-Flash-Next on one CUDA device.
 *
 * Everything but the routed experts runs on the GPU: hyper-connections, Gated DeltaNet,
 * gated attention, the router, the shared expert and the output head. Per MoE layer the
 * device sends the mixed activation and the router logits to the host, the CPU computes the
 * routed experts from its RAM cache (q38_model_moe_routed) while the device computes the
 * shared expert, and the routed sum comes back. The PLE layer runs on the host too (one
 * layer, its table is read from the SSD).
 *
 * The arithmetic is the CPU's, in float32: results agree with tools/q38_reference.py to the
 * same tolerance as the CPU path (the order of additions differs).
 */
#ifndef Q38_GPU_H
#define Q38_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

struct q38_model_s;

/* Uploads the trunk to `device`. Returns 0, or -1 (the model then stays on the CPU). */
int  q38_gpu_open(void *model, int device);
void q38_gpu_close(void *model);
void q38_gpu_reset(void *model);
/* Keep the device's clocks up (1) or let them fall (0); q38_gpu_forward turns it on, the
 * caller turns it off when it stops decoding (end of an answer). */
void q38_gpu_warm(void *model, int on);
/* After q38_model_preload: upload the most used resident experts into the VRAM pool.
 * Returns how many. */
int  q38_gpu_prefill(void *model);
/* n consecutive tokens in one pass per layer (prompt reading); the logits of the last one
 * land in the model's host buffer. Returns 0 or -1. */
int  q38_gpu_forward_batch(void *model, const int *toks, int n);
int  q38_gpu_mvbench(void *model);
int  q38_gpu_launchbench(void *model);
/* One token; the logits land in the model's host buffer. Returns 0 or -1. */
int  q38_gpu_forward(void *model, int token);
/* The MTP block's draft on the device (see q38_model_mtp_draft): logits into m->mtp_logits. */
int  q38_gpu_mtp_draft(void *model, int tok, int j);
/* N <= 4 tokens at n_past, the logits of every one into logits [N][n_vocab] (dense models) */
int  q38_gpu_verify(void *model, const int *toks, int n, float *logits);
/* the recurrent state copied aside (restore 0) or put back (restore 1) around a verification */
int  q38_gpu_spec_state(void *model, int restore);
/* after a verification that kept its states (q38_gpu_verify returned 1): only its first a + 1 tokens stand */
int  q38_gpu_spec_keep(void *model, int a, int n);
/* The sequence state that does not follow from the position alone (Gated DeltaNet states,
 * convolution windows, the PLE ring) copied to a host buffer (save = 1) or back (save = 0).
 * With n_past set back to the position of the save, the model continues from there: the
 * attention and indexer caches are indexed by position. Returns 0 or -1. */
int  q38_gpu_state(void *model, int save);
/* The whole sequence state at n_past (also the caches up to it and the token ids) to a file
 * (save = 1) or back; returns the position, or -1 (after a failed load: reset). */
int  q38_gpu_state_file(void *model, const char *path, int save);

#ifdef __cplusplus
}
#endif
#endif
