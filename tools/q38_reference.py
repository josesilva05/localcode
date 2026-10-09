#!/usr/bin/env python3
"""Float64 numpy reference of Qwen3.8-Flash-Next (GGUF arch qwen4exp), one token at a time.

    python tools/q38_reference.py <model dir or shard> <ref_dir> [--tokens N] [--out dir]
        [--gguf-py C:/Users/JoseS/App/llama.cpp/gguf-py]

ref_dir is a q38_ref_dump output (tokens.txt plus llama.cpp's intermediate tensors). The
script runs the same token ids through a plain numpy forward pass written from llama.cpp's
graph (src/models/qwen4exp.cpp, delta-net-base.cpp): weights are dequantised with
llama.cpp's own gguf-py, every product and sum is float64. For every token it compares
each layer's output (l_last, the 4 hyper-connection streams) and the logits with what
llama.cpp computed, and prints the relative error per layer and the top-5 tokens of both.

llama.cpp quantises activations to 8 bits inside its matrix products, so the two cannot
agree to the last bit; the comparison is a check of the semantics (a misread formula is
off by O(1), activation quantisation by ~1e-2). The engine is then checked against this
reference, which has no such approximation (--out writes its tensors for that).

Simplifications, all exact for what they cover: attention is dense, which is what the
indexer selects while the context is at most indexer_top_k + compress_ratio - 1 = 2051
tokens; text positions only (M-RoPE then reduces to NeoX RoPE on the first n_rot dims).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np


class Model:
    def __init__(self, model, gguf_py):
        sys.path.insert(0, gguf_py)
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.deq = dequantize
        paths = sorted(glob.glob(os.path.join(model, "*.gguf"))) if os.path.isdir(model) else \
            sorted(glob.glob(model.rsplit("-", 3)[0] + "-*-of-*.gguf")) or [model]
        self.t = {}
        self.kv = {}
        for i, p in enumerate(paths):
            r = GGUFReader(p)
            for t in r.tensors:
                self.t[t.name] = t
            if i == 0:
                for k, f in r.fields.items():
                    self.kv[k] = f.contents()
        self.arch = str(self.kv.get("general.architecture", "qwen4exp"))
        a = self.arch + "."
        kv = self.kv
        # qwen35moe (Qwen3.5/3.6 MoE): one residual stream, pre-norm blocks, silu DeltaNet gate,
        # its MTP block(s) after the decoder ones
        self.plain = self.arch != "qwen4exp"
        self.n_layer = kv[a + "block_count"] - (kv.get(a + "nextn_predict_layers", 0) if self.plain else 0)
        self.E = kv[a + "embedding_length"]
        self.n_head = kv[a + "attention.head_count"]
        self.n_head_kv = kv[a + "attention.head_count_kv"]
        self.head_dim = kv[a + "attention.key_length"]
        self.eps = float(kv[a + "attention.layer_norm_rms_epsilon"])
        self.n_exp = kv.get(a + "expert_count", 0)       # qwen35 (dense): none, one FFN a layer
        self.k_exp = kv.get(a + "expert_used_count", 0)
        self.n_rot = kv[a + "rope.dimension_count"]
        self.rope_base = float(kv[a + "rope.freq_base"])
        self.interval = kv[a + "full_attention_interval"]
        self.hc = 1 if self.plain else kv[a + "hyper_connection.count"]
        self.conv_k = kv[a + "ssm.conv_kernel"]
        self.d_state = kv[a + "ssm.state_size"]          # head dim of keys and values
        self.n_kh = kv[a + "ssm.group_count"]            # key heads
        self.n_vh = kv[a + "ssm.time_step_rank"]         # value heads
        self.ple_layers = list(kv.get(a + "ple.layers", []))
        self.ple_ngram = kv.get(a + "ple.ngram_size", 0)
        self.ple_per_gram = kv.get(a + "ple.heads_per_ngram", 0)
        self.ple_kern = kv.get(a + "ple.conv_kernel", 0)
        self.ple_eos = kv.get(a + "ple.eos_token_id", 0)
        self.ple_dim = kv.get(a + "embedding_length_per_layer_input", 0)
        self.ple_mult = [int(x) for x in kv.get(a + "ple.layer_multipliers", [])]
        self.ple_off = [int(x) for x in kv.get(a + "ple.head_offsets", [])]
        self.ple_voc = [int(x) for x in kv.get(a + "ple.head_vocab_sizes", [])]
        self.cache = {}

    def is_recr(self, il):
        return (il + 1) % self.interval != 0

    def w(self, name, keep=True):
        """Dequantised 2-D weight as float64, shape (rows = ne1, cols = ne0)."""
        if name in self.cache:
            return self.cache[name]
        t = self.t[name]
        x = self.deq(np.asarray(t.data), t.tensor_type).astype(np.float64)
        x = x.reshape(int(np.prod(t.shape[1:])) if len(t.shape) > 1 else 1, int(t.shape[0]))
        if keep and x.size <= 4 << 20:    # small tensors only: the trunk in float64 would be ~30 GB
            self.cache[name] = x
        return x

    def vec(self, name):
        return self.w(name).reshape(-1)

    def rows(self, name, idx):
        """Selected rows of a 2-D tensor (embedding lookups)."""
        t = self.t[name]
        data = np.asarray(t.data)
        out = []
        for i in idx:
            out.append(self.deq(data[i:i + 1], t.tensor_type).astype(np.float64).reshape(-1))
        return np.stack(out)

    def expert(self, name, e):
        t = self.t[name]
        x = self.deq(np.asarray(t.data)[e], t.tensor_type).astype(np.float64)
        return x.reshape(int(t.shape[1]), int(t.shape[0]))


def rms(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


class State:
    def __init__(self, m):
        self.conv = {}     # il -> [conv_k-1, channels]
        self.ssm = {}      # il -> [n_vh, d_state(key i), d_state(value j)]
        self.k = {}        # il -> list of [n_head_kv, head_dim]
        self.v = {}
        self.ple_hist = {}  # il -> [hist, hc*E]
        self.toks = []


class Ref:
    def __init__(self, m):
        self.m = m
        self.s = State(m)
        self.taps = {}

    def tap(self, name, x):
        self.taps[name] = np.asarray(x, dtype=np.float64).reshape(-1)

    def hc_mix(self, res, pre, inject):
        m = self.m
        hc, E = m.hc, m.E
        wn = m.vec(pre + "_norm.weight").reshape(hc, E)
        xn = rms(res, m.eps) * wn                         # [hc, E]
        flat = xn.reshape(-1)
        lo = m.w(pre + "_down.weight") @ flat             # [low_rank]
        lo = silu(lo / hc)
        gate = m.w(pre + "_up.weight") @ lo               # [hc*E]
        gated = (flat * sigmoid(gate)).reshape(hc, E)
        mixed = gated.sum(axis=0) / hc
        inj = m.w(pre + "_inject.weight") @ flat if inject else None
        return mixed, inj

    def hc_combine(self, res, out, inj):
        w = 2.0 * sigmoid(inj / self.m.hc)
        return res + w[:, None] * out[None, :]

    def ple(self, il, res, tok):
        m, st = self.m, self.s
        hc, E = m.hc, m.E
        # n-gram context: the token, then its predecessors; EOS or the sequence start cuts it
        ctx = [tok]
        cut = False
        for s in range(1, m.ple_ngram):
            t = st.toks[-1 - s] if len(st.toks) - 1 - s >= 0 else -1
            cut = cut or t < 0 or t == m.ple_eos
            ctx.append(m.ple_eos if cut else t)
        rows = []
        mask = (1 << 64) - 1
        for n in range(2, m.ple_ngram + 1):
            mixed = (ctx[0] * m.ple_mult[0]) & mask
            for j in range(1, n):
                mixed ^= (ctx[j] * m.ple_mult[j]) & mask
            base = (n - 2) * m.ple_per_gram
            for g in range(m.ple_per_gram):
                h = base + g
                rows.append(mixed % m.ple_voc[h] + m.ple_off[h])
        emb = m.rows("per_layer_token_embd.weight", rows).reshape(-1)      # [n_heads*ple_dim]
        p = f"blk.{il}."
        key = (m.w(p + "ple_key.weight") @ emb).reshape(hc, E)
        value = m.w(p + "ple_value.weight") @ emb                            # [E]
        key = rms(key, m.eps) * m.vec(p + "ple_norm_key.weight").reshape(hc, E)
        query = rms(res, m.eps) * m.vec(p + "ple_norm_query.weight").reshape(hc, E)
        s = (key * query).sum(axis=1) / np.sqrt(E)
        gate = sigmoid(np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6)))
        gated = value[None, :] * gate[:, None]                              # [hc, E]
        normed = (rms(gated, m.eps) * m.vec(p + "ple_norm_conv.weight").reshape(hc, E)).reshape(-1)
        hist = (m.ple_kern - 1) * m.ple_ngram
        hbuf = st.ple_hist.get(il)
        if hbuf is None:
            hbuf = np.zeros((hist, hc * E))
        xs = np.vstack([hbuf, normed[None, :]])                              # [hist+1, hc*E]
        wk = m.w(p + "ple_conv1d.weight")                                    # [hc*E, kern]
        conv = np.zeros(hc * E)
        for k in range(m.ple_kern):
            conv += wk[:, k] * xs[hist - (m.ple_kern - 1 - k) * m.ple_ngram]
        st.ple_hist[il] = xs[1:]
        conv = silu(conv).reshape(hc, E)
        self.tap(f"ple_gate-{il}", gate)
        return res + gated + conv

    def gdn(self, il, x):
        m, st = self.m, self.s
        p = f"blk.{il}."
        S, Hk, Hv = m.d_state, m.n_kh, m.n_vh
        qkv = m.w(p + "attn_qkv.weight") @ x                      # [2*Hk*S + Hv*S]
        z = m.w(p + "attn_gate.weight") @ x                       # [Hv*S]
        beta = sigmoid(m.w(p + "ssm_beta.weight") @ x)            # [Hv]
        alpha = m.w(p + "ssm_alpha.weight") @ x
        g = softplus(alpha + m.vec(p + "ssm_dt.bias")) * m.vec(p + "ssm_a")
        ch = qkv.shape[0]
        cs = st.conv.get(il)
        if cs is None:
            cs = np.zeros((m.conv_k - 1, ch))
        xin = np.vstack([cs, qkv[None, :]])                        # [conv_k, ch], oldest first
        wc = m.w(p + "ssm_conv1d.weight")                          # [ch, conv_k]
        conv = (xin.T * wc).sum(axis=1)
        st.conv[il] = xin[1:]
        conv = silu(conv)
        q = conv[:Hk * S].reshape(Hk, S)
        k = conv[Hk * S:2 * Hk * S].reshape(Hk, S)
        v = conv[2 * Hk * S:].reshape(Hv, S)
        q = q / np.sqrt((q * q).sum(axis=1, keepdims=True) + m.eps)
        k = k / np.sqrt((k * k).sum(axis=1, keepdims=True) + m.eps)
        state = st.ssm.get(il)
        if state is None:
            state = np.zeros((Hv, S, S))                           # [h, i (key), j (value)]
        o = np.zeros((Hv, S))
        scale = 1.0 / np.sqrt(S)
        for h in range(Hv):
            kh = h % Hk                                            # ggml repeats (tiles) the key heads
            Sm = state[h] * np.exp(g[h])
            sk = Sm.T @ k[kh]
            d = (v[h] - sk) * beta[h]
            Sm = Sm + np.outer(k[kh], d)
            o[h] = (Sm.T @ q[kh]) * scale
            state[h] = Sm
        st.ssm[il] = state
        self.tap(f"attn_output-{il}", o)
        gate = silu(z.reshape(Hv, S)) if m.plain else sigmoid(z.reshape(Hv, S))
        o = rms(o, m.eps) * m.vec(p + "ssm_norm.weight")[None, :] * gate
        return m.w(p + "ssm_out.weight") @ o.reshape(-1)

    def rope(self, x, pos):
        m = self.m
        half = m.n_rot // 2
        inv = m.rope_base ** (-np.arange(half) * 2.0 / m.n_rot)
        ang = pos * inv
        c, s = np.cos(ang), np.sin(ang)
        y = x.copy()
        x0, x1 = x[..., :half], x[..., half:m.n_rot]
        y[..., :half] = x0 * c - x1 * s
        y[..., half:m.n_rot] = x0 * s + x1 * c
        return y

    def attn(self, il, x, pos):
        m, st = self.m, self.s
        p = f"blk.{il}."
        H, Hkv, D = m.n_head, m.n_head_kv, m.head_dim
        qf = (m.w(p + "attn_q.weight") @ x).reshape(H, 2, D)
        q, gate = qf[:, 0], qf[:, 1]
        q = rms(q, m.eps) * m.vec(p + "attn_q_norm.weight")
        k = rms((m.w(p + "attn_k.weight") @ x).reshape(Hkv, D), m.eps) * m.vec(p + "attn_k_norm.weight")
        v = (m.w(p + "attn_v.weight") @ x).reshape(Hkv, D)
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        st.k.setdefault(il, []).append(k)
        st.v.setdefault(il, []).append(v)
        K = np.stack(st.k[il])                                     # [T, Hkv, D]
        V = np.stack(st.v[il])
        out = np.zeros((H, D))
        r = H // Hkv
        for h in range(H):
            kv = h // r
            sc = K[:, kv] @ q[h] / np.sqrt(D)
            sc = np.exp(sc - sc.max())
            sc /= sc.sum()
            out[h] = sc @ V[:, kv]
        out = out * sigmoid(gate)
        o = m.w(p + "attn_output.weight") @ out.reshape(-1)
        self.tap(f"attn_output-{il}", o)
        return o

    def ffn(self, il, x):
        m = self.m
        p = f"blk.{il}."
        if m.n_exp == 0:                  # dense: down(silu(gate x) * up x)
            g = m.w(p + "ffn_gate.weight") @ x
            u = m.w(p + "ffn_up.weight") @ x
            return m.w(p + "ffn_down.weight") @ (silu(g) * u)
        logits = m.w(p + "ffn_gate_inp.weight") @ x
        pr = np.exp(logits - logits.max())
        pr /= pr.sum()
        top = np.argsort(-pr, kind="stable")[:m.k_exp]
        wts = pr[top] / pr[top].sum()
        out = np.zeros(m.E)
        for e, wt in zip(top, wts):
            g = m.expert(p + "ffn_gate_exps.weight", e) @ x
            u = m.expert(p + "ffn_up_exps.weight", e) @ x
            out += wt * (m.expert(p + "ffn_down_exps.weight", e) @ (silu(g) * u))
        g = m.w(p + "ffn_gate_shexp.weight") @ x
        u = m.w(p + "ffn_up_shexp.weight") @ x
        sh = m.w(p + "ffn_down_shexp.weight") @ (silu(g) * u)
        sg = sigmoid(m.vec(p + "ffn_gate_inp_shexp.weight") @ x)
        self.tap(f"experts-{il}", top.astype(np.float64))
        return out + sg * sh

    def forward(self, tok, pos):
        m, st = self.m, self.s
        self.taps = {}
        x = m.rows("token_embd.weight", [tok])[0]
        st.toks.append(tok)
        if m.plain:
            for il in range(m.n_layer):
                p = f"blk.{il}."
                h = rms(x, m.eps) * m.vec(p + "attn_norm.weight")
                out = self.gdn(il, h) if m.is_recr(il) else self.attn(il, h, pos)
                if m.is_recr(il):
                    self.tap(f"linear_attn_out-{il}", out)
                x = x + out
                h = rms(x, m.eps) * m.vec(p + "post_attention_norm.weight")
                out = self.ffn(il, h)
                self.tap(f"ffn_out-{il}", out)
                x = x + out
                self.tap(f"l_out-{il}", x)
            final = rms(x, m.eps) * m.vec("output_norm.weight")
            self.tap("result_norm", final)
            logits = m.w("output.weight") @ final
            self.tap("logits", logits)
            return logits
        res = np.tile(x, (m.hc, 1))
        for il in range(m.n_layer):
            if il in m.ple_layers:
                res = self.ple(il, res, tok)
            mixed, inj = self.hc_mix(res, f"blk.{il}.hc_attn", True)
            self.tap(f"hc_mixed_attn-{il}", mixed)
            out = self.gdn(il, mixed) if m.is_recr(il) else self.attn(il, mixed, pos)
            res = self.hc_combine(res, out, inj)
            mixed, inj = self.hc_mix(res, f"blk.{il}.hc_ffn", True)
            self.tap(f"hc_mixed_ffn-{il}", mixed)
            out = self.ffn(il, mixed)
            self.tap(f"ffn_out-{il}", out)
            res = self.hc_combine(res, out, inj)
            self.tap(f"l_last-{il}", res)
        final, _ = self.hc_mix(res, "output_hc", False)
        self.tap("result_norm", final)
        logits = m.w("output.weight") @ final
        self.tap("logits", logits)
        return logits


def load_ref(ref_dir):
    idx = {}
    with open(os.path.join(ref_dir, "index.txt")) as f:
        for line in f:
            name, n = line.split()
            idx[name] = int(n)
    return idx


def ref_row(ref_dir, idx, name, tok_i, per_token=1, which=0):
    n = idx[name]
    a = np.memmap(os.path.join(ref_dir, name + ".f32"), dtype=np.float32, mode="r")
    off = (tok_i * per_token + which) * n
    return np.asarray(a[off:off + n], dtype=np.float64)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref_dir")
    ap.add_argument("--tokens", type=int, default=0, help="stop after N tokens (0 = all)")
    ap.add_argument("--out", default="", help="write this reference's taps here (float32, per token)")
    ap.add_argument("--gguf-py", default="C:/Users/JoseS/App/llama.cpp/gguf-py")
    ap.add_argument("--layers", default="0,1,2,3,7,23,47", help="layers printed per token")
    args = ap.parse_args()

    m = Model(args.model, args.gguf_py)
    toks = [int(x) for x in open(os.path.join(args.ref_dir, "tokens.txt")).read().split()]
    if args.tokens:
        toks = toks[:args.tokens]
    idx = load_ref(args.ref_dir)
    show = [int(x) for x in args.layers.split(",")]
    r = Ref(m)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "tokens.txt"), "w") as f:
            f.write("\n".join(map(str, toks)) + "\n")
    worst = 0.0
    for i, tok in enumerate(toks):
        t0 = time.time()
        logits = r.forward(tok, i)
        dt = time.time() - t0
        errs = []
        lname = "l_out" if m.plain else "l_last"
        for il in range(m.n_layer):
            e = rel(r.taps[f"{lname}-{il}"], ref_row(args.ref_dir, idx, f"{lname}-{il}", i))
            errs.append(e)
        lref = ref_row(args.ref_dir, idx, "logits", i)
        el = rel(logits, lref)
        worst = max(worst, el)
        top_r, top_l = np.argsort(-logits)[:5], np.argsort(-lref)[:5]
        lay = " ".join(f"L{il}:{errs[il]:.1e}" for il in show if il < m.n_layer)
        print(f"tok {i:3d} id {tok:6d} {dt:5.1f}s | {lay} | logits {el:.2e} | top5 ref {list(top_r)} llama {list(top_l)}"
              f"{'' if top_r[0] == top_l[0] else '  <-- argmax differs'}", flush=True)
        if args.out:
            for name, x in r.taps.items():
                with open(os.path.join(args.out, name + ".f32"), "ab") as f:
                    f.write(x.astype(np.float32).tobytes())
    print(f"worst logits relative error vs llama.cpp: {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
