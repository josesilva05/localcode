#!/usr/bin/env python3
"""Check the engine's GGUF reader + dequantisation against llama.cpp's gguf-py.

    python tools/q38_check_dequant.py <model dir or any shard> [--exe build/Release/q38_gguf_info.exe]
        [--gguf-py C:/Users/JoseS/App/llama.cpp/gguf-py]

For one tensor of every quant type in the model it dequantises the same rows twice -- with
q38_gguf_info (C) and with gguf.quants.dequantize (numpy, llama.cpp's own Python port of
the reference dequantisers) -- and requires the two float32 arrays to be bit-identical.
Rows are taken at the start, middle and end of the tensor so a wrong offset or a wrong
expert stride shows up as a mismatch.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--exe", default=os.path.join("build", "Release", "q38_gguf_info.exe"))
    ap.add_argument("--gguf-py", default="C:/Users/JoseS/App/llama.cpp/gguf-py")
    ap.add_argument("--rows", type=int, default=4, help="rows per sample point")
    args = ap.parse_args()

    sys.path.insert(0, args.gguf_py)
    from gguf import GGUFReader
    from gguf.quants import dequantize

    shards = sorted(glob.glob(os.path.join(args.model, "*.gguf"))) if os.path.isdir(args.model) else \
        sorted(glob.glob(args.model.rsplit("-", 3)[0] + "-*-of-*.gguf")) or [args.model]
    first = shards[0]

    # one tensor per type, preferring the big expert tensors (they exercise the 3-D offsets)
    picked = {}
    readers = []
    for path in shards:
        r = GGUFReader(path)
        readers.append(r)
        for t in r.tensors:
            key = t.tensor_type.name
            big = t.n_elements
            if key not in picked or big > picked[key][1].n_elements:
                picked[key] = (path, t)

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for qtype, (path, t) in sorted(picked.items()):
            ne0 = int(t.shape[0])
            n_rows = int(t.n_elements // ne0)
            raw = np.asarray(t.data).reshape(n_rows, -1) if t.data.ndim > 1 or qtype not in ("F32",) else \
                np.asarray(t.data).reshape(n_rows, ne0)
            points = sorted({0, n_rows // 2, max(0, n_rows - args.rows)})
            worst = 0
            for p in points:
                n = min(args.rows, n_rows - p)
                out = os.path.join(tmp, "rows.f32")
                cmd = [args.exe, first, "dump", t.name, str(p), str(n), out]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode:
                    print(f"FAIL {qtype:8s} {t.name}: {res.stderr.strip()}")
                    failures += 1
                    break
                c = np.fromfile(out, dtype=np.float32).reshape(n, ne0)
                ref = dequantize(raw[p:p + n], t.tensor_type).astype(np.float32).reshape(n, ne0)
                same = np.array_equal(c.view(np.uint32), ref.view(np.uint32))
                if not same:
                    diff = np.abs(c.astype(np.float64) - ref.astype(np.float64))
                    worst = max(worst, float(diff.max()))
                    nbad = int((c.view(np.uint32) != ref.view(np.uint32)).sum())
                    print(f"FAIL {qtype:8s} {t.name} rows {p}..{p + n - 1}: {nbad} values differ, max |d| {diff.max():.3g}")
                    failures += 1
            else:
                print(f"ok   {qtype:8s} {t.name} {list(map(int, t.shape))}: rows {points} x{args.rows} bit-identical")
    print("PASS" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
