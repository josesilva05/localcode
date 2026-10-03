#!/usr/bin/env python3
"""Compare two tap dumps of the same token ids, tensor by tensor.

    python tools/q38_compare.py <dir A> <dir B> [--names l_last,logits] [--tol 1e-4]

Both directories hold <name>.f32 files with one row per token (q38_run --dump writes the
engine's, q38_reference.py --out the float64 reference's, q38_ref_dump llama.cpp's; for
llama.cpp, names it taps twice per layer are skipped). For each tensor name present in
both it prints the worst relative error over the tokens (|a-b| / |b|, B is the reference),
and for the logits whether the argmax agrees on every token. Exit status 1 when a tensor
in --names exceeds --tol.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np


def rows(path, n_tok):
    a = np.fromfile(path, dtype=np.float32)
    if n_tok == 0 or a.size % n_tok:
        return None
    return a.reshape(n_tok, -1).astype(np.float64)


def key(name):
    m = re.match(r"(.*)-(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", default="l_last,logits", help="tensors held to --tol")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--all", action="store_true", help="print every layer, not a summary")
    args = ap.parse_args()

    tok_a = open(os.path.join(args.a, "tokens.txt")).read().split() if os.path.exists(os.path.join(args.a, "tokens.txt")) else None
    tok_b = open(os.path.join(args.b, "tokens.txt")).read().split()
    n = len(tok_b)
    names = sorted({os.path.basename(p)[:-4] for p in glob.glob(os.path.join(args.a, "*.f32"))} &
                   {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(args.b, "*.f32"))}, key=key)
    if tok_a and tok_a[:n] != tok_b[:len(tok_a)]:
        print("warning: the token lists differ")
    held = set(args.names.split(","))
    fail = 0
    groups = {}
    for nm in names:
        ra = rows(os.path.join(args.a, nm + ".f32"), n)
        rb = rows(os.path.join(args.b, nm + ".f32"), n)
        if ra is None or rb is None or ra.shape != rb.shape:
            continue
        err = np.linalg.norm(ra - rb, axis=1) / np.maximum(np.linalg.norm(rb, axis=1), 1e-30)
        base, layer = key(nm)
        groups.setdefault(base, []).append((layer, float(err.max()), int(err.argmax())))
        if base == "logits":
            agree = int((ra.argmax(axis=1) == rb.argmax(axis=1)).sum())
            print(f"logits: worst rel {err.max():.3e} (token {err.argmax()}), argmax agrees on {agree}/{n} tokens")
        if base in held and err.max() > args.tol:
            fail += 1
    for base, lst in groups.items():
        if base == "logits":
            continue
        worst = max(lst, key=lambda t: t[1])
        if args.all:
            print(base + ": " + " ".join(f"L{l}:{e:.1e}" for l, e, _ in lst))
        else:
            print(f"{base:16s} {len(lst):3d} tensors, worst rel {worst[1]:.3e} (layer {worst[0]}, token {worst[2]})")
    if not groups:
        print("FAIL: no tensor in common (did the run produce its dump?)")
        return 1
    print("PASS" if not fail else f"FAIL: {fail} tensor(s) in {args.names} above {args.tol}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
