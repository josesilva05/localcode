#!/usr/bin/env python3
"""Seed the localcode expert-usage profile from q38_route_trace traces.

    python tools/q38_profile_from_trace.py trace1.txt [trace2.txt ...] [--out PATH]

Counts how often each (layer, expert) was among the router's top 10 in the traces and
writes the profile localcode reads at start (%LOCALAPPDATA%/localcode/qwen38_expert_profile.bin
by default): magic "Q8P1", n_layer, n_exp, then n_layer * n_exp float32 counts.
localcode adds its own counts at every exit, so this only matters for the first sessions.
"""
import argparse
import os
import struct
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.environ.get("LOCALAPPDATA", "."), "localcode",
                                                  "qwen38_expert_profile.bin"))
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    use = np.zeros(args.layers * args.experts, dtype=np.float32)
    for path in args.traces:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) < 3 + args.k:
                    continue
                il = int(p[2])
                for t in p[3:3 + args.k]:
                    use[il * args.experts + int(t.split(":")[0])] += 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<III", 0x31503851, args.layers, args.experts))
        f.write(use.tobytes())
    print(f"{args.out}: {int((use > 0).sum())} experts seen, {int(use.sum())} choices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
