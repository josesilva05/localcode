#!/usr/bin/env python3
"""Replay a Qwen3.8-Flash-Next routing trace through an expert cache.

    python tools/q38_route_sim.py trace.txt <model dir or shard> [--gb 20,30,38,45]
        [--warm 32] [--ssd 3.2] [--k 10]

The trace comes from q38_route_trace (one line per token and MoE layer: P|D, position,
layer, then the router's 16 best experts as id:prob, best first; the first --k are used).

An expert is (layer, id); its size is read from the GGUF (gate + up + down slices of that
layer, so the Q8_0/Q5_K layers count as bigger). Prompt tokens pass through the cache
(they warm it) but only generated tokens are measured, after --warm of them. For every
cache budget it reports, per generated token:

  lru     least-recently-used eviction, the engine's current policy
  opt     Belady (evict the expert whose next use is farthest), the bound no policy
          without knowledge of the future can beat
  pin     an oracle static set: the most used experts of the whole trace pinned, nothing
          else cached (what a fixed placement such as llama.cpp's -ot can reach at best)

as SSD reads, GB read, and the disk-only time at --ssd GB/s with the tok/s that time
alone would allow. Everything else per token (trunk, compute) comes on top.
"""
import argparse
import glob
import os
import sys
from collections import Counter, OrderedDict, defaultdict


def expert_bytes(model, gguf_py):
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    from gguf.constants import GGML_QUANT_SIZES
    paths = sorted(glob.glob(os.path.join(model, "*.gguf"))) if os.path.isdir(model) else \
        sorted(glob.glob(model.rsplit("-", 3)[0] + "-*-of-*.gguf")) or [model]
    size = defaultdict(int)
    for p in paths:
        for t in GGUFReader(p).tensors:
            n = t.name
            if not (n.startswith("blk.") and "_exps" in n):
                continue
            layer = int(n.split(".")[1])
            block, tbytes = GGML_QUANT_SIZES[t.tensor_type]
            n_exp = int(t.shape[2])
            size[layer] += int(t.n_bytes) // n_exp
            assert t.n_elements // n_exp // block * tbytes == int(t.n_bytes) // n_exp
    return dict(size)


def load(path, k):
    toks = []          # list of (phase, [(layer, [ids])...]) in token order
    cur_pos, cur = None, None
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3 + k:
                continue
            phase, pos, layer = p[0], int(p[1]), int(p[2])
            ids = [int(x.split(":")[0]) for x in p[3:3 + k]]
            if pos != cur_pos:
                cur = (phase, [])
                toks.append(cur)
                cur_pos = pos
            cur[1].append((layer, ids))
    return toks


def accesses(toks):
    """Flat access list [(token_index, key)], in execution order."""
    seq = []
    for ti, (_, layers) in enumerate(toks):
        for layer, ids in layers:
            for e in ids:
                seq.append((ti, (layer, e)))
    return seq


def sim_lru(seq, size, budget, measured):
    cache, used = OrderedDict(), 0
    reads = Counter()
    rbytes = Counter()
    for ti, key in seq:
        if key in cache:
            cache.move_to_end(key)
            continue
        b = size[key[0]]
        if ti in measured:
            reads[ti] += 1
            rbytes[ti] += b
        while used + b > budget and cache:
            old, ob = cache.popitem(last=False)
            used -= ob
        cache[key] = b
        used += b
    return reads, rbytes


def sim_opt(seq, size, budget, measured):
    import heapq
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        key = seq[i][1]
        nxt[i] = last.get(key, 1 << 60)
        last[key] = i
    cache = {}          # key -> next use index
    heap = []           # (-next_use, key) lazy
    used = 0
    reads, rbytes = Counter(), Counter()
    for i, (ti, key) in enumerate(seq):
        b = size[key[0]]
        if key in cache:
            cache[key] = nxt[i]
            heapq.heappush(heap, (-nxt[i], key))
            continue
        if ti in measured:
            reads[ti] += 1
            rbytes[ti] += b
        if nxt[i] >= 1 << 60:
            continue            # never used again: do not cache
        while used + b > budget and heap:
            negn, old = heapq.heappop(heap)
            if old in cache and cache[old] == -negn:
                if -negn < nxt[i]:
                    # every resident is needed sooner than this one: bypass the cache
                    heapq.heappush(heap, (negn, old))
                    break
                del cache[old]
                used -= size[old[0]]
        if used + b <= budget:
            cache[key] = nxt[i]
            heapq.heappush(heap, (-nxt[i], key))
            used += b
    return reads, rbytes


def sim_pin(seq, size, budget, measured):
    freq = Counter(key for _, key in seq)
    pinned, used = set(), 0
    for key, _ in freq.most_common():
        b = size[key[0]]
        if used + b <= budget:
            pinned.add(key)
            used += b
    reads, rbytes = Counter(), Counter()
    for ti, key in seq:
        if key not in pinned and ti in measured:
            reads[ti] += 1
            rbytes[ti] += size[key[0]]
    return reads, rbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("model")
    ap.add_argument("--gb", default="20,30,38,45", help="cache budgets in GiB")
    ap.add_argument("--warm", type=int, default=32, help="generated tokens not measured")
    ap.add_argument("--ssd", type=float, default=3.2, help="SSD read speed, GB/s")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--gguf-py", default="C:/Users/JoseS/App/llama.cpp/gguf-py")
    args = ap.parse_args()

    size = expert_bytes(args.model, args.gguf_py)
    toks = load(args.trace, args.k)
    seq = accesses(toks)
    gen = [i for i, (ph, _) in enumerate(toks) if ph == "D"]
    measured = set(gen[args.warm:])
    n_meas = len(measured)
    all_bytes = sum(size.values()) * 512
    print(f"experts: {len(size)} layers, {min(size.values())/1e6:.2f}-{max(size.values())/1e6:.2f} MB each, "
          f"{all_bytes/2**30:.1f} GiB in all")
    print(f"trace: {len(toks)} tokens ({len(toks) - len(gen)} prompt, {len(gen)} generated), "
          f"measuring {n_meas} after {args.warm} warm-up")
    if not n_meas:
        return 1
    per_tok = sum(len(ids) for _, ids in toks[gen[-1]][1])
    gen_keys = Counter(key for ti, key in seq if ti in measured)
    print(f"per generated token: {per_tok} routed experts; {len(gen_keys)} distinct experts used while measuring "
          f"({sum(size[k[0]] for k in gen_keys)/2**30:.1f} GiB)")
    nocache = sum(size[l] * len(ids) for l, ids in toks[gen[-1]][1])
    print(f"no cache: {nocache/1e9:.3f} GB/token -> {nocache/1e9/args.ssd:.3f} s/token at {args.ssd} GB/s\n")

    print(f"{'GiB':>5} {'policy':6} {'reads/tok':>9} {'hit %':>6} {'GB/tok':>7} {'disk s/tok':>10} {'disk-only tok/s':>15}")
    for gb in [float(x) for x in args.gb.split(",")]:
        budget = gb * 2**30
        for name, fn in (("lru", sim_lru), ("opt", sim_opt), ("pin", sim_pin)):
            reads, rbytes = fn(seq, size, budget, measured)
            r = sum(reads.values()) / n_meas
            gbt = sum(rbytes.values()) / n_meas / 1e9
            hit = 100.0 * (1 - r / per_tok)
            t = gbt / args.ssd
            print(f"{gb:5.0f} {name:6} {r:9.1f} {hit:6.1f} {gbt:7.3f} {t:10.3f} {(1/t if t else float('inf')):15.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
