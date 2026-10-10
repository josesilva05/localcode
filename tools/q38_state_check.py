"""q38_state_check.py - checks the saved prompt beginning (q38api_state_save / q38api_state_load)
the server reads a harness's long system prompt from instead of reading it again.

    python tools/q38_state_check.py [--model GGUF] [--chars 9000]

Builds a prompt whose first turn (a system prompt of --chars characters of this engine's own
source, past the 2051 tokens where the indexer takes over from dense attention) the server
keeps in a file, then:
  1. reads it as a new conversation: the first turn, the file written, the rest; greedy tokens
  2. empties the state and reads it again: the first turn now comes from the file
The logits after the prompt and at every generated step must agree with the first pass (float
rounding: the VRAM pool takes in experts between the passes, so some sums change order).
The prompt read whole in one pass, without the file, is shown for comparison (other batch
boundaries, so not bit for bit).
"""
import argparse
import os
import sys
import time

import numpy as np

os.environ["Q38_ACT8"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q38_server as qs  # noqa: E402


def rel(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def run(eng, prompt, steps):
    """A new conversation: empty state, the prompt, then greedy tokens. Logits after each."""
    eng.lib.q38api_reset(eng.h)
    eng.live, eng.snap = [], -1
    t0 = time.time()
    reused = eng.prepare(prompt)
    t_prompt = time.time() - t0
    out = [eng.logits.copy()]
    for _ in range(steps):
        eng.step(int(np.argmax(out[-1])))
        out.append(eng.logits.copy())
    return reused, t_prompt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LOCALCODE_MODEL", qs.DEFAULT_MODEL))
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chars", type=int, default=9000)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    dll = os.path.join(qs.ROOT, "build", "Release", "q38api.dll")
    eng = qs.Engine(dll, args.model, args.ctx, -1, 0.0)
    tpl = qs.compile_template(eng.chat_template)
    with open(os.path.join(qs.ROOT, "src", "q38", "q38_model.c"), encoding="utf-8", errors="replace") as f:
        system = "You review C code. The file under review:\n\n" + f.read()[:args.chars]
    text = tpl.render(messages=[{"role": "system", "content": system},
                                {"role": "user", "content": "In one sentence: what does this file do?"}],
                      add_generation_prompt=True, reasoning_effort="low")
    prompt = eng.tokenize(text)
    P = eng.prefix_len(prompt)
    print("prompt %d tokens, first turn %d" % (len(prompt), P), flush=True)
    if P < eng.PREFIX_MIN or P >= len(prompt):
        sys.exit("the first turn is too short to be kept (Q38_PREFIX_MIN %d)" % eng.PREFIX_MIN)
    path = eng._prefix_path(prompt[:P])
    if os.path.exists(path):
        os.remove(path)

    r1, t1, a = run(eng, prompt, args.steps)
    print("1. new conversation: %d reused, prompt in %.1f s, file %s (%.0f MB)" %
          (r1, t1, os.path.exists(path), os.path.getsize(path) / 1e6 if os.path.exists(path) else 0), flush=True)
    r2, t2, b = run(eng, prompt, args.steps)
    print("2. the same again:   %d reused, prompt in %.1f s" % (r2, t2), flush=True)
    ok = r1 == 0 and r2 == P
    worst = 0.0
    for i, (x, y) in enumerate(zip(b, a)):
        d = rel(x, y)
        worst = max(worst, d)
        if d > 1e-4 or x.argmax() != y.argmax():
            ok = False
            print("   step %d: rel %.2e, argmax %d vs %d" % (i, d, int(x.argmax()), int(y.argmax())))
    print("   from the file vs read: worst rel %.2e over %d logits, argmax %s" %
          (worst, len(a), "same" if all(x.argmax() == y.argmax() for x, y in zip(a, b)) else "DIFFERENT"))

    saved_min = eng.PREFIX_MIN
    eng.PREFIX_MIN = 0                     # the whole prompt in one pass, no file
    _, t3, c = run(eng, prompt, 0)
    eng.PREFIX_MIN = saved_min
    print("3. read whole (other batch boundaries): prompt in %.1f s, rel %.2e, argmax %s" %
          (t3, rel(c[0], a[0]), "same" if c[0].argmax() == a[0].argmax() else "different"))
    os.remove(path)
    eng.close()
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
