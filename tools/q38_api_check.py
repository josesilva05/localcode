"""q38_api_check.py - checks the engine library's saved-state round trip (q38api_snapshot /
q38api_restore) the server's prompt reuse rests on.

    python tools/q38_api_check.py [--model GGUF]

Reads a prompt, saves the state, generates 12 tokens greedily (one at a time), goes back to
the saved state and feeds the same tokens again one at a time: every step's logits must agree
with the first pass to float32 rounding. Then it goes back once more and feeds them in one
batch: the last logits must agree too (the batched kernels sum in another order).

The CPU experts run with float32 activations here (Q38_ACT8=0): between the two passes the
VRAM pool takes in experts, and with 8-bit activations on the CPU an expert that moved to a
device would change the logits by about 1% on its own, hiding what is being checked.
"""
import argparse
import os
import sys

import numpy as np

os.environ["Q38_ACT8"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q38_server as qs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LOCALCODE_MODEL", qs.DEFAULT_MODEL))
    ap.add_argument("--ctx", type=int, default=32768)
    args = ap.parse_args()
    dll = os.path.join(qs.ROOT, "build", "Release", "q38api.dll")
    eng = qs.Engine(dll, args.model, args.ctx, -1, 0.0)
    tpl = qs.compile_template(eng.chat_template)
    text = tpl.render(messages=[{"role": "user", "content": "Write a haiku about the sea."}], add_generation_prompt=True,
                      reasoning_effort="low")
    prompt = eng.tokenize(text)
    eng.prepare(prompt)
    gen, first = [], []
    for _ in range(12):
        tok = int(np.argmax(eng.logits))
        gen.append(tok)
        eng.step(tok)
        first.append(eng.logits.copy())
    ok = True
    # back to the end of the prompt, the same tokens one by one
    assert eng.lib.q38api_restore(eng.h) == len(prompt)
    del eng.live[len(prompt):]
    worst = 0.0
    for i, tok in enumerate(gen):
        eng.step(tok)
        a, b = eng.logits.astype(np.float64), first[i].astype(np.float64)
        rel = np.linalg.norm(a - b) / np.linalg.norm(b)
        worst = max(worst, rel)
        if rel > 1e-5 or a.argmax() != b.argmax():
            ok = False
            print("step %d: rel %.2e, argmax %d vs %d" % (i, rel, int(a.argmax()), int(b.argmax())))
    print("one token at a time after restore: worst rel %.2e over 12 steps: %s" % (worst, "OK" if ok else "DIFFERENT"))
    # and in one batch
    assert eng.lib.q38api_restore(eng.h) == len(prompt)
    del eng.live[len(prompt):]
    eng._eval(gen)
    a, b = eng.logits.astype(np.float64), first[-1].astype(np.float64)
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    print("one batch after restore: rel %.2e, argmax %d vs %d" % (rel, int(a.argmax()), int(b.argmax())))
    eng.close()
    sys.exit(0 if ok and rel < 1e-3 and a.argmax() == b.argmax() else 1)


if __name__ == "__main__":
    main()
