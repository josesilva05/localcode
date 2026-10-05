# The HTTP API (llama-server / OpenAI compatible)

`start-api.bat` serves Qwen3.8-Flash-Next from this engine on `http://127.0.0.1:8080`:

| Endpoint | What it answers |
| --- | --- |
| `GET /props` | llama-server's description: `model_path`, `total_slots` (1), `default_generation_settings.n_ctx`, `modalities.vision` (false), `chat_template` |
| `GET /health` | `{"status":"ok"}`, or 503 while the model loads |
| `GET /v1/models` | the one model |
| `POST /v1/chat/completions` | OpenAI chat completions, streamed (`stream: true`, server-sent events) or not |
| `POST /tokenize`, `POST /apply-template` | the prompt's tokens / text |

All inference runs in the engine (`build/Release/q38api.dll`, `src/q38/q38_api.c`);
`tools/q38_server.py` renders the model's own chat template (from the GGUF, with jinja2 as
transformers does), samples, and writes the OpenAI wire format:

- reasoning streams as `reasoning_content`, the answer as `content`;
- the model's `<tool_call><function=…><parameter=…>` blocks become `tool_calls` (arguments as
  JSON, typed by each tool's schema), with `finish_reason: "tool_calls"`;
- `stream_options.include_usage` adds the usage chunk (`prompt_tokens_details.cached_tokens`
  is how much of the prompt was not read again); the last chunk also carries llama-server's
  `timings`.

## deepseek-harness

The `llamacpp` route (`@deepseek-ai/dsh-llm-llamacpp`) looks at `http://127.0.0.1:8080` by
default, so with `start-api.bat` running, select `llamacpp/loaded`. Nothing else to configure.
Do not run llama-server, localcode and the API at the same time: each loads the model.

## Options (`start-api.bat --help`)

| Option | Default | |
| --- | --- | --- |
| `--ctx` | 32768 | tokens of context (the device indexer allows up to 32768); more context leaves less VRAM for experts |
| `--reasoning-effort` | low | the template's effort when a request names none (`low`, `medium`, `xhigh`) |
| `--no-thinking` | off | answer without a reasoning block unless a request asks for one |
| `--temperature`, `--top-k`, `--top-p`, `--min-p` | 0.6, 20, 0.95, 0 | sampling when a request names none (greedy at temperature 0) |
| `--port`, `--host` | 8080, 127.0.0.1 | |
| `--preload-gb` | 24 | experts read into RAM at start from the usage profile |

A request may set `reasoning_effort` (or `chat_template_kwargs.reasoning_effort` /
`enable_thinking`) to override the effort for itself.

## Conversation state between requests

One conversation is kept: the engine holds the tokens of the last request (prompt and answer)
and a copy of the recurrent state (Gated DeltaNet, convolution windows, PLE ring) saved at the
end of that prompt. A request that extends what the engine holds continues from there; one
that shares the last prompt continues from the saved state; anything else is read from the
start. An agent's next turn repeats the conversation, so usually only the new messages are
read. `tools/q38_api_check.py` checks the save/restore round trip (logits agree to 1e-6).
