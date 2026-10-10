<div align="center">

<h1>localcode</h1>

<h3>A coding agent and its own inference engines, in C, running entirely on your machine.</h3>

<img src="docs/media/demo.gif" width="760" alt="localcode answering in the terminal: GLM-5.3-Flash running locally">

<sub><code>localcode</code>, the interactive session: the 314B checkpoint on a PC with 64 GB of RAM and
two RTX 3060, at 1.10 s/token with the int4 container.</sub>

</div>

localcode is one program and the engines under it, with no framework and no cloud:

| part | what it is |
|---|---|
| **localcode** (`src/cli`) | the terminal session and its agent: plan, edit, test and fix loop; grep with regular expressions, `apply_patch`, `/undo` and `/redo`, allow/ask/deny permissions, `AGENTS.md` rules; Claude Code's plugins, skills, commands, hooks and mods. See [docs/localcode-agent-plan.md](docs/localcode-agent-plan.md) and [docs/localcode-vs-opencode.md](docs/localcode-vs-opencode.md) |
| **q38** (`src/q38`) | the engine for the Qwen3.8 family from GGUF files: Flash-Next (35B-A3B, MoE) and the dense 27B, on the CPU and on one or two NVIDIA GPUs (MTP speculative decoding, tensor parallelism, 128k contexts) |
| **glm53f** (`src/core`, `src/model`, ...) | the engine for GLM-5.3-Flash (~314B total, ~18B active), validated against transformers; described below |
| **the API** (`tools/q38_server.py`, [docs/API.md](docs/API.md)) | an OpenAI / llama-server compatible server over the engine, for other tools and other machines |

The repository was called glm-5.3-flash-in-c, then qwen3.8-flash-next-in-c; the old GitHub
links redirect here.

## The GLM-5.3-Flash engine

This engine runs the **released GLM-5.3-Flash checkpoint** (FP8, 62 shards, ~306 GB) and is
validated against the official transformers implementation (`modeling_glm5_next.py`):

| check | result |
|---|---|
| full 45-layer prefill, real weights, 18-token chat prompt | same next token and top-10 as the reference; logits within 1.5e-5 (7e-7 relative) |
| real-weights decode (prompt + 5 generated tokens) | reference predicts the same 6th token as the engine |
| tiny random model with the same architecture (weightless, in CI) | prefill logits, 6/6 incremental greedy tokens, 18/18 teacher-forced positions |
| mutation check of that test | removing the SwiGLU clamp, transposing the mHC mixer or skipping Sinkhorn each fail it |
| C tokenizer vs official tokenizer | 12/12 cases token-identical (Portuguese, code, CJK, emoji, whitespace) |
| config reader | loads the released config; rejects 12 mutated configs it would otherwise misread |

Measured on this machine (Ryzen 7 5700X3D, 64 GB RAM, NVMe, Windows 11, MSVC build):

```console
$ build/Release/glm53f.exe <model_dir> --prompt-file prompt.txt --gen 400 --cache-gb 30
...
</think>A capital do Brasil é **Brasília**.

Ela foi inaugurada em 21 de abril de 1960, substituindo o Rio de Janeiro como capital. ...

prefill: 18 tokens in 54.7 s | decode: 234 tokens in 678.0 s (2.90 s/token) | stopped at EOS
expert bytes read: 954.57 GB | peak RSS 45.71 GB | expert drops 0
```

Trunk (everything but routed experts) is resident: 12.6 GB + 2.5 GB embeddings/lm_head.
Routed experts (25.2 MB each, FP8) stream from the SSD through an LRU cache, and each MoE
layer predicts the next layer's experts so they are read while compute continues. With a
24 GB cache, decode takes **2.22 s/token** (2.96 without prefetch) and an 18-token prompt
prefills in **~31 s** (46 without), with identical output. Longer prompts go layer by layer,
so each layer's experts are read once per prompt: a 372-token document prefills in 119 s
instead of 193 with `--gpu`.

With the optional CUDA backend (`-DGLM53F_CUDA=ON`, `--gpu`) the trunk runs on two RTX 3060
12 GB cards, its RAM goes to the expert cache, and decode takes **1.80 s/token** at the
same peak RSS (`--cache-gb 38`), with the same generated tokens; see
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

With the optional int4 expert container (`--write-int4`, all 42 MoE layers, ~171 GB; an
approximation of the checkpoint, not the checkpoint) decode takes **0.69-0.72 s/token**
with `--gpu --cache-gb 38`. On this machine that is close to the ceiling the SSD sets
(~1.6 GB of expert reads per token): moving expert products to the second GPU
(`GLM53F_GPU_EXPERTS`) cut compute but not the step. Development is paused there; the
measurements and what was tried are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

### Architecture implemented

- **mHC** residual: 4 streams; per sub-block, learned collapse weights and a Sinkhorn-normalised 4x4 mixer.
- **KDA** (34 layers): gated delta rule with per-channel decay, short conv, low-rank output gate.
- **MLA** (11 layers): NoPE, no output gate, a KV cache that can hold the compressed latent
  (22 KB per position instead of 1.44 MB), with the model's **DeepSeek Sparse Attention**:
  the indexer scores pools of 4 positions and keeps the 2048 best plus the tail, so a query
  attends to at most 2051 positions of any context. Up to 2051 positions that selection is
  everything, and the engine runs the dense path it always did.
- **MoE** (42 layers): 288 experts, top-8, sigmoid router with correction bias, 1 shared expert;
  3 dense layers; SwiGLU clamped at 10.
- **FP8 E4M3** block-scaled matrices multiplied straight from the codes; BF16 elsewhere.
- Not used: the MTP layer (45) and the vision encoder.
- Optional CUDA backend: trunk on NVIDIA GPUs (cuBLAS + custom kernels), routed experts on the CPU.

### Build, test, run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

```bash
build/Release/glm53f <model_dir> --prompt-file prompt.txt --gen 400 --quiet
```

The expert cache is sized from the free RAM unless `--cache-gb` is given. For a conversation
that keeps the model loaded between messages, `localcode` opens a terminal chat in the
style of terminal coding tools (green theme, folded reasoning, rendered Markdown, `/file`,
`/save`):

```bash
build/Release/localcode <model_dir>
```

`install-cli.bat` puts a `localcode` command on the user PATH (`%USERPROFILE%\.local\bin`):
then `cd` to any project and type `localcode`. That folder is the agent's working folder,
and the model is found in `%USERPROFILE%\models\Qwen3.8-Flash-Next\UD-Q4_K_XL` when none is
given. The screen opens with a welcome box (the model, the folder, this folder's recent
conversations) and keeps the input panel, the agent's mode and the context at the bottom.

With NVIDIA GPUs:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGLM53F_CUDA=ON
cmake --build build --config Release -j
build/Release/glm53f <model_dir> --prompt-file prompt.txt --gen 400 --gpu --quiet
```

## Documentation

| document | contents |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | requirements, build, every CLI option, output format, Windows notes, memory tuning |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the model (mHC, KDA, MLA/DSA, MoE, storage formats) and the engine (files, data layout, forward pass, expert streaming, failure policy) |
| [docs/VALIDATION.md](docs/VALIDATION.md) | tests, reference tools, how to compare with transformers, recorded results |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | measured speed, disk/compute split, per-component cost, hardware limits |
| [docs/API.md](docs/API.md) | the OpenAI / llama-server compatible API over the q38 engine |
| [docs/localcode-agent-plan.md](docs/localcode-agent-plan.md) | localcode's agent: levels, the engineering loop, sampling |
| [docs/localcode-plugins-plan.md](docs/localcode-plugins-plan.md) | Claude Code's plugins, skills, hooks and mods in localcode |
| [docs/localcode-vs-opencode.md](docs/localcode-vs-opencode.md) | the comparison with opencode v2, the roadmap and the benchmark (`tools/bench`) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | development rules and verification checklist |
| [CHANGELOG.md](CHANGELOG.md) | release history |

## License

Apache-2.0 for this code (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The model weights are
not part of this repository and are covered by their own license.
