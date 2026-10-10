"""q38_server.py - Qwen3.8-Flash-Next on this engine behind an HTTP API that llama.cpp clients
understand: llama-server's GET /props and /health, and OpenAI's GET /v1/models and
POST /v1/chat/completions (streamed as server-sent events, with reasoning_content, tool calls
and usage). deepseek-harness's llama.cpp route (@deepseek-ai/dsh-llm-llamacpp) works with it
unchanged.

    python tools/q38_server.py [--model GGUF] [--host 127.0.0.1] [--port 8080] [--ctx 32768]
                               [--reasoning-effort low|medium|xhigh] [--gpu N] [--preload-gb 24]

All inference runs in the engine (build/Release/q38api.dll); this file only renders the
model's own chat template (tokenizer.chat_template, with jinja2 as transformers does), keeps
one conversation's state between requests, samples, and turns the model's output into the
OpenAI wire format.

The state between requests: the engine keeps the tokens of the last request (prompt and
answer) and a saved state at the end of that prompt. A new request whose tokens extend what
the engine holds continues from there; one that shares the old prompt continues from the
saved state; anything else is read from the start. An agent's next turn repeats the whole
conversation, so usually only the new messages are read.
"""
import argparse
import codecs
import contextlib
import ctypes
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(os.path.expanduser("~"), "models", "Qwen3.8-Flash-Next", "UD-Q4_K_XL",
                             "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- the engine
class Engine:
    def __init__(self, dll, model, ctx, gpu, preload_gb):
        # Python resolves a DLL's own dependencies without PATH: name the CUDA and DLL folders
        if hasattr(os, "add_dll_directory"):
            dirs = [os.path.dirname(os.path.abspath(dll))]
            cuda = os.environ.get("CUDA_PATH")
            if cuda:
                dirs += [os.path.join(cuda, "bin"), os.path.join(cuda, "bin", "x64")]
            for d in dirs:
                if os.path.isdir(d):
                    os.add_dll_directory(d)
        lib = ctypes.CDLL(dll, winmode=0) if os.name == "nt" else ctypes.CDLL(dll)
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
        sig = {
            "q38api_open": (P, [ctypes.c_char_p, I, I, F]),
            "q38api_close": (None, [P]),
            "q38api_n_ctx": (I, [P]), "q38api_n_vocab": (I, [P]), "q38api_n_past": (I, [P]),
            "q38api_meta_str": (I, [P, ctypes.c_char_p, ctypes.c_char_p, I]),
            "q38api_meta_int": (ctypes.c_longlong, [P, ctypes.c_char_p, ctypes.c_longlong]),
            "q38api_tokenize": (I, [P, ctypes.c_char_p, I, ctypes.POINTER(I), I]),
            "q38api_piece": (I, [P, I, ctypes.c_char_p, I]),
            "q38api_is_control": (I, [P, I]),
            "q38api_token_id": (I, [P, ctypes.c_char_p]),
            "q38api_eval": (I, [P, ctypes.POINTER(I), I, ctypes.POINTER(ctypes.c_float)]),
            "q38api_reset": (None, [P]),
            "q38api_snapshot": (I, [P]), "q38api_restore": (I, [P]),
            "q38api_state_save": (I, [P, ctypes.c_char_p]), "q38api_state_load": (I, [P, ctypes.c_char_p]),
            "q38api_save_profile": (I, [P]),
        }
        for name, (res, args) in sig.items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        self.lib = lib
        # saved prompt beginnings (see _from_prefix): only valid for this model and this build
        self.prefix_dir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "localcode", "prefix")
        self.prefix_tag = "%s|%d|%d|%d" % (os.path.abspath(model), os.path.getsize(model), ctx,
                                           int(os.path.getmtime(dll)))
        self.h = lib.q38api_open(model.encode("utf-8"), ctx, gpu, preload_gb)
        if not self.h:
            raise RuntimeError("the engine could not open " + model)
        self.n_ctx = lib.q38api_n_ctx(self.h)
        self.n_vocab = lib.q38api_n_vocab(self.h)
        self.logits = np.empty(self.n_vocab, dtype=np.float32)
        self._lp = self.logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n = lib.q38api_meta_str(self.h, b"tokenizer.chat_template", None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        lib.q38api_meta_str(self.h, b"tokenizer.chat_template", buf, n + 1)
        self.chat_template = buf.value.decode("utf-8")
        self.pieces = []
        b = ctypes.create_string_buffer(1024)
        for i in range(self.n_vocab):
            k = lib.q38api_piece(self.h, i, b, 1024)
            self.pieces.append(b.raw[:max(k, 0)])
        self.control = set(i for i in range(self.n_vocab) if lib.q38api_is_control(self.h, i))
        eos = {int(lib.q38api_meta_int(self.h, b"tokenizer.ggml.eos_token_id", -1))}
        for t in ("<|im_end|>", "<|endoftext|>"):
            eos.add(lib.q38api_token_id(self.h, t.encode()))
        self.eos = {e for e in eos if e >= 0}
        self.im_start = lib.q38api_token_id(self.h, b"<|im_start|>")
        self.parked = []         # [(tokens, file)]: conversations set aside for another prompt (_park)
        self.park_seq = 0
        if os.path.isdir(self.prefix_dir):
            for f in os.listdir(self.prefix_dir):
                if f.endswith(".park"):
                    try:
                        os.remove(os.path.join(self.prefix_dir, f))
                    except OSError:
                        pass
        self.live = []           # the tokens the engine state holds (n_past == len(live))
        self.snap = -1           # len(live) at the saved state, -1 = none
        self.snap_logits = None

    def tokenize(self, text):
        raw = text.encode("utf-8")
        cap = len(raw) + 16
        out = (ctypes.c_int * cap)()
        n = self.lib.q38api_tokenize(self.h, raw, len(raw), out, cap)
        return list(out[:n])

    def _eval(self, toks):
        if not toks:
            return
        arr = (ctypes.c_int * len(toks))(*toks)
        if self.lib.q38api_eval(self.h, arr, len(toks), self._lp) != 0:
            raise RuntimeError("the engine failed while reading tokens")
        self.live.extend(toks)

    # a prompt's first turn (a harness's system prompt with its tools) at least this long is kept
    # in a file once read: every new conversation starts with it (Q38_PREFIX_MIN, 0 = never)
    PREFIX_MIN = int(os.environ.get("Q38_PREFIX_MIN", "1024"))
    PREFIX_FILES = 4

    def prefix_len(self, prompt):
        """Where the prompt's second turn starts (its second <|im_start|>), or 0."""
        seen = 0
        for i, t in enumerate(prompt):
            if t == self.im_start:
                seen += 1
                if seen == 2:
                    return i
        return 0

    def prepare(self, prompt):
        """Bring the state to the end of prompt with the next-token logits in self.logits.
        Returns how many prompt tokens did not have to be read."""
        live, n = self.live, len(prompt)
        common = 0
        lim = min(len(live), n)
        while common < lim and live[common] == prompt[common]:
            common += 1
        P = self.prefix_len(prompt) if self.PREFIX_MIN > 0 else 0
        if P < self.PREFIX_MIN or n <= P:
            P = 0
        parked = self._parked_for(prompt)
        best = max(P, len(parked[0]) if parked else 0)   # what a file would spare
        if common == len(live) and n > common and common >= best:    # it extends what the engine holds
            self._eval(prompt[common:])
            reused = common
        elif self.snap >= 0 and best <= self.snap <= common and self.snap_logits is not None:
            if self.snap < len(live) or self.lib.q38api_n_past(self.h) != self.snap:
                if self.lib.q38api_restore(self.h) < 0:
                    return self._from_start(prompt, P)
                del self.live[self.snap:]
            if n > self.snap:
                self._eval(prompt[self.snap:])
            else:
                self.logits[:] = self.snap_logits
            reused = self.snap
        else:
            self._park()
            if parked and len(parked[0]) >= P and self._unpark(parked):
                self._eval(prompt[len(parked[0]):])
                self._save()
                return len(parked[0])
            return self._from_start(prompt, P)
        self._save()
        return reused

    # conversations set aside while another prompt (a session title, another session) used the
    # engine: their next turn continues from the file instead of reading everything again
    PARKED = 2

    def _parked_for(self, prompt):
        """The longest set-aside conversation the prompt continues, or None."""
        best = None
        for toks, path in self.parked:
            k = len(toks)
            if k < len(prompt) and (best is None or k > len(best[0])) and prompt[:k] == toks:
                best = (toks, path)
        return best

    def _park(self):
        """The conversation the engine holds to a file, before another prompt takes its place."""
        live = self.live
        if len(live) < max(self.PREFIX_MIN, 1) or any(t == live for t, _ in self.parked):
            return
        try:
            os.makedirs(self.prefix_dir, exist_ok=True)
            if shutil.disk_usage(self.prefix_dir).free < 20e9:
                return
        except OSError:
            return
        path = os.path.join(self.prefix_dir, "%d-%d.park" % (os.getpid(), self.park_seq))
        self.park_seq += 1
        t0 = time.time()
        if self.lib.q38api_state_save(self.h, path.encode("mbcs" if os.name == "nt" else "utf-8")) != len(live):
            return
        self.parked.insert(0, (list(live), path))
        for _, old in self.parked[self.PARKED:]:
            try:
                os.remove(old)
            except OSError:
                pass
        del self.parked[self.PARKED:]
        log("conversation of %d tokens set aside in %.1f s" % (len(live), time.time() - t0))

    def _unpark(self, parked):
        toks, path = parked
        t0 = time.time()
        if self.lib.q38api_state_load(self.h, path.encode("mbcs" if os.name == "nt" else "utf-8")) == len(toks):
            self.live = list(toks)
            self.snap = -1
            log("conversation of %d tokens taken back in %.1f s" % (len(toks), time.time() - t0))
            return True
        self.parked = [p for p in self.parked if p[1] != path]
        try:
            os.remove(path)
        except OSError:
            pass
        self.lib.q38api_reset(self.h)
        self.live, self.snap = [], -1
        return False

    def _prefix_path(self, toks):
        key = hashlib.sha1(self.prefix_tag.encode("utf-8") + np.asarray(toks, dtype=np.int32).tobytes()).hexdigest()
        return os.path.join(self.prefix_dir, key[:20] + ".q38s")

    def _keep_prefix_files(self):
        try:
            files = sorted((os.path.join(self.prefix_dir, f) for f in os.listdir(self.prefix_dir) if f.endswith(".q38s")),
                           key=os.path.getmtime, reverse=True)
            for f in files[self.PREFIX_FILES:]:
                os.remove(f)
        except OSError:
            pass

    def _from_start(self, prompt, P=0):
        """Read prompt into an empty state. Its first P tokens come from their file when an
        earlier prompt began the same way, and go into one otherwise."""
        path = self._prefix_path(prompt[:P]) if P else None
        if path and os.path.exists(path):
            t0 = time.time()
            got = self.lib.q38api_state_load(self.h, path.encode("mbcs" if os.name == "nt" else "utf-8"))
            if got == P:
                os.utime(path)
                self.live = list(prompt[:P])
                self.snap = -1
                log("prompt beginning: %d tokens from %s in %.1f s" % (P, os.path.basename(path), time.time() - t0))
                self._eval(prompt[P:])
                self._save()
                return P
            log("prompt beginning %s does not load; reading the prompt" % os.path.basename(path))
            try:
                os.remove(path)
            except OSError:
                pass
        self.lib.q38api_reset(self.h)
        self.live = []
        self.snap = -1
        if path:
            self._eval(prompt[:P])
            free = shutil.disk_usage(os.path.dirname(self.prefix_dir)).free if os.path.isdir(
                os.path.dirname(self.prefix_dir)) else 0
            if free > 20e9:                    # never fill the disk (the file is a few hundred MB)
                os.makedirs(self.prefix_dir, exist_ok=True)
                tmp = path + ".tmp"
                t0 = time.time()
                if self.lib.q38api_state_save(self.h, tmp.encode("mbcs" if os.name == "nt" else "utf-8")) == P:
                    os.replace(tmp, path)
                    self._keep_prefix_files()
                    log("prompt beginning: %d tokens kept in %s (%.0f MB, %.1f s)" %
                        (P, os.path.basename(path), os.path.getsize(path) / 1e6, time.time() - t0))
            prompt = prompt[P:]
        self._eval(prompt)
        self._save()
        return 0

    def _save(self):
        if self.lib.q38api_snapshot(self.h) >= 0:
            self.snap = len(self.live)
            self.snap_logits = self.logits.copy()
        else:
            self.snap = -1

    def step(self, tok):
        self._eval([tok])

    def close(self):
        if self.h:
            self.lib.q38api_close(self.h)
            self.h = None


# --------------------------------------------------------------------------- the chat template
def compile_template(source):
    """As transformers' apply_chat_template compiles it."""
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env.from_string(source)


class BadRequest(Exception):
    pass


def normalize_messages(messages):
    """The OpenAI request's messages in the shape the template reads: tool call arguments as
    objects, image parts as text (the model is served without vision)."""
    out = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise BadRequest("every message needs a role")
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image"):
                    parts.append({"type": "text", "text": "[image]"})
                elif isinstance(p, str):
                    parts.append({"type": "text", "text": p})
            m["content"] = parts
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a) if a.strip() else {}
                    except ValueError:
                        fn["arguments"] = {"arguments": a}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        if m.get("reasoning_content") is None and isinstance(m.get("reasoning"), str):
            m["reasoning_content"] = m["reasoning"]
        out.append(m)
    return out


# --------------------------------------------------------------------------- sampling
def sample(logits, temperature, top_k, top_p, min_p, rng):
    if temperature <= 0:
        return int(np.argmax(logits))
    v = logits.shape[0]
    if 0 < top_k < v:
        idx = np.argpartition(logits, -top_k)[-top_k:]
    else:
        idx = np.arange(v)
    vals = logits[idx].astype(np.float64)
    order = np.argsort(-vals)
    idx, vals = idx[order], vals[order]
    p = np.exp((vals - vals[0]) / temperature)
    p /= p.sum()
    if min_p > 0:
        keep = p >= min_p * p[0]
        idx, p = idx[keep], p[keep]
        p /= p.sum()
    if top_p < 1.0:
        n = int(np.searchsorted(np.cumsum(p), top_p)) + 1
        idx, p = idx[:n], p[:n]
        p /= p.sum()
    return int(rng.choice(idx, p=p))


# --------------------------------------------------------------------------- the model's output
TOOL_OPEN, TOOL_CLOSE, THINK_CLOSE = "<tool_call>", "</tool_call>", "</think>"


def _schema_types(tools):
    types = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        types[fn.get("name")] = {k: (v or {}).get("type") for k, v in props.items()}
    return types


def parse_tool_call(block, types):
    """<function=NAME><parameter=P>value</parameter>...</function> -> (name, arguments JSON)."""
    m = re.search(r"<function=([^>\n]+)>(.*?)</function>", block, re.S)
    if not m:
        return None
    name = m.group(1).strip()
    known = types.get(name, {})
    args = {}
    for pm in re.finditer(r"<parameter=([^>\n]+)>(.*?)</parameter>", m.group(2), re.S):
        key, raw = pm.group(1).strip(), pm.group(2)
        if raw.startswith("\n"):
            raw = raw[1:]
        if raw.endswith("\n"):
            raw = raw[:-1]
        t = known.get(key)
        is_string = t == "string" or (isinstance(t, list) and "string" in t and len(t) == 1)
        value = raw
        if not is_string:
            try:
                value = json.loads(raw)
            except ValueError:
                value = raw
        args[key] = value
    return name, json.dumps(args, ensure_ascii=False)


class OutputParser:
    """Splits the generated text into reasoning (before </think>), content and tool calls
    (<tool_call> blocks after </think>), holding back what could be the start of a tag."""

    def __init__(self, thinking, tools):
        self.mode = "think" if thinking else "content"
        self.buf = ""
        self.types = _schema_types(tools)
        self.content_started = False
        self.n_tools = 0

    @staticmethod
    def _holdback(text, tag):
        """Length of the longest suffix of text that is a proper prefix of tag."""
        for k in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:k]):
                return k
        return 0

    def feed(self, text, final=False):
        self.buf += text
        events = []
        while True:
            if self.mode == "think":
                i = self.buf.find(THINK_CLOSE)
                if i >= 0:
                    r = self.buf[:i].rstrip()
                    if r:
                        events.append(("reasoning", r))
                    self.buf = self.buf[i + len(THINK_CLOSE):]
                    self.mode = "content"
                    continue
                keep = len(self.buf) if final else self._holdback(self.buf, THINK_CLOSE)
                # trailing whitespace waits too: the reasoning ends without it
                body = self.buf[:len(self.buf) - keep] if not final else self.buf.rstrip()
                if not final:
                    stripped = body.rstrip()
                    keep += len(body) - len(stripped)
                    body = stripped
                if body:
                    events.append(("reasoning", body))
                self.buf = self.buf[len(self.buf) - keep:] if keep and not final else ""
                break
            if self.mode == "content":
                if not self.content_started:
                    self.buf = self.buf.lstrip("\n")
                i = self.buf.find(TOOL_OPEN)
                if i >= 0:
                    before = self.buf[:i].rstrip()
                    if before:
                        events.append(("content", before))
                        self.content_started = True
                    self.buf = self.buf[i + len(TOOL_OPEN):]
                    self.mode = "tool"
                    continue
                keep = 0 if final else self._holdback(self.buf, TOOL_OPEN)
                body = self.buf[:len(self.buf) - keep]
                if not final:
                    stripped = body.rstrip()
                    keep += len(body) - len(stripped)
                    body = stripped
                if body:
                    events.append(("content", body))
                    self.content_started = True
                self.buf = self.buf[len(self.buf) - keep:] if keep else ""
                break
            if self.mode == "tool":
                i = self.buf.find(TOOL_CLOSE)
                if i < 0 and not final:
                    break
                block = self.buf if i < 0 else self.buf[:i]
                self.buf = "" if i < 0 else self.buf[i + len(TOOL_CLOSE):]
                parsed = parse_tool_call(block, self.types)
                if parsed:
                    events.append(("tool", parsed))
                    self.n_tools += 1
                elif block.strip():
                    events.append(("content", TOOL_OPEN + block + ("" if i < 0 else TOOL_CLOSE)))
                self.mode = "content"
                if i < 0:
                    break
                continue
        return events


# --------------------------------------------------------------------------- the server
class Server:
    def __init__(self, args):
        self.args = args
        self.engine = None
        self.template = None
        self.lock = threading.Lock()
        self.cv = threading.Condition()
        self.turns_waiting = 0              # requests with tools (a conversation's turns) waiting or running
        self.model_path = os.path.abspath(args.model)
        self.model_name = os.path.basename(self.model_path)
        if self.model_name.lower().endswith(".gguf"):
            self.model_name = self.model_name[:-5]
        self.loading = True
        self.error = None

    @contextlib.contextmanager
    def turn(self, side):
        """The engine for one request. With --turns-first a side request (no tools: a harness
        asking for a session title as the conversation starts) lets a conversation's turn sent
        with it, or waiting, go first: the person waits for the answer, not for the title."""
        side = side and self.args.turns_first
        if side:
            time.sleep(0.3)
            with self.cv:
                self.cv.wait_for(lambda: self.turns_waiting == 0, timeout=900)
        else:
            with self.cv:
                self.turns_waiting += 1
        try:
            with self.lock:
                yield
        finally:
            if not side:
                with self.cv:
                    self.turns_waiting -= 1
                    self.cv.notify_all()

    def load(self):
        try:
            t0 = time.time()
            dll = self.args.dll or os.path.join(ROOT, "build", "Release", "q38api.dll")
            self.engine = Engine(dll, self.model_path, self.args.ctx, self.args.gpu, self.args.preload_gb)
            self.template = compile_template(self.engine.chat_template)
            log("model ready in %.1f s: context %d tokens, reasoning effort %s" %
                (time.time() - t0, self.engine.n_ctx, self.args.reasoning_effort))
        except Exception as e:                      # noqa: BLE001 - reported on /health
            self.error = str(e)
            log("loading failed: " + self.error)
        finally:
            self.loading = False

    def props(self):
        n_ctx = self.engine.n_ctx if self.engine else 0
        return {
            "model_path": self.model_path,
            "model_alias": self.model_name,
            "total_slots": 1,
            "modalities": {"vision": False, "audio": False},
            "default_generation_settings": {
                "n_ctx": n_ctx,
                "params": {"temperature": self.args.temperature, "top_k": self.args.top_k, "top_p": self.args.top_p,
                           "min_p": self.args.min_p, "n_predict": -1},
            },
            "chat_template": self.engine.chat_template if self.engine else "",
            "build_info": "localcode (q38 engine)",
            "is_sleeping": False,
        }

    def render(self, req):
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BadRequest("messages must be a non-empty list")
        kwargs = dict(req.get("chat_template_kwargs") or {})
        effort = kwargs.get("reasoning_effort", req.get("reasoning_effort", self.args.reasoning_effort))
        if effort == "high":
            effort = "xhigh"
        if effort in ("none", "minimal"):
            kwargs["enable_thinking"] = False
        elif effort in ("low", "medium", "xhigh"):
            kwargs["reasoning_effort"] = effort
        if self.args.no_thinking and "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = False
        tools = req.get("tools") or None
        if req.get("tool_choice") == "none":
            tools = None
        try:
            text = self.template.render(messages=normalize_messages(messages), tools=tools, add_generation_prompt=True,
                                        **kwargs)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest("chat template: " + str(e))
        thinking = kwargs.get("enable_thinking", True) is not False
        return text, tools, thinking


def make_handler(srv):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "q38-server"

        def log_message(self, fmt, *a):          # quiet: requests are logged by the server
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, message, etype="invalid_request_error", **extra):
            err = {"code": code, "message": message, "type": etype}
            err.update(extra)
            self._json(code, {"error": err})

        def _not_ready(self):
            if srv.loading:
                self._error(503, "Loading model", "unavailable_error")
                return True
            if srv.engine is None:
                self._error(500, "the model could not be loaded: %s" % srv.error, "server_error")
                return True
            return False

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/health" or path == "/v1/health":
                if self._not_ready():
                    return
                return self._json(200, {"status": "ok"})
            if path == "/props":
                if self._not_ready():
                    return
                return self._json(200, srv.props())
            if path in ("/v1/models", "/models"):
                if self._not_ready():
                    return
                return self._json(200, {"object": "list", "data": [{
                    "id": srv.model_name, "object": "model", "created": int(time.time()), "owned_by": "q38",
                    "meta": {"n_ctx": srv.engine.n_ctx, "n_vocab": srv.engine.n_vocab}}]})
            if path == "/slots":
                if self._not_ready():
                    return
                return self._json(200, [{"id": 0, "n_ctx": srv.engine.n_ctx, "is_processing": srv.lock.locked()}])
            return self._error(404, "not found: " + self.path, "not_found_error")

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n > 0 else b""
            try:
                req = json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                return self._error(400, "the body is not JSON")
            if path in ("/v1/chat/completions", "/chat/completions"):
                if self._not_ready():
                    return
                return self.chat(req)
            if path == "/tokenize":
                if self._not_ready():
                    return
                return self._json(200, {"tokens": srv.engine.tokenize(req.get("content", ""))})
            if path == "/apply-template":
                if self._not_ready():
                    return
                try:
                    text, _, _ = srv.render(req)
                except BadRequest as e:
                    return self._error(400, str(e))
                return self._json(200, {"prompt": text})
            return self._error(404, "not found: " + self.path, "not_found_error")

        # ----------------------------------------------------------- chat completions
        def _sse(self, obj):
            data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        def chat(self, req):
            try:
                prompt_text, tools, thinking = srv.render(req)
            except BadRequest as e:
                return self._error(400, str(e))
            eng = srv.engine
            prompt = eng.tokenize(prompt_text)
            n_ctx = eng.n_ctx
            if len(prompt) >= n_ctx:
                return self._error(400, "the request (%d prompt tokens) does not fit in the context (%d tokens)" %
                                   (len(prompt), n_ctx), "exceed_context_size_error",
                                   n_prompt_tokens=len(prompt), n_ctx=n_ctx)
            want = req.get("max_completion_tokens") or req.get("max_tokens") or req.get("n_predict")
            limit = n_ctx - len(prompt)
            max_tokens = min(int(want), limit) if want and int(want) > 0 else limit
            stream = bool(req.get("stream"))
            usage_wanted = stream and bool((req.get("stream_options") or {}).get("include_usage"))
            temp = float(req.get("temperature", srv.args.temperature))
            top_k = int(req.get("top_k", srv.args.top_k))
            top_p = float(req.get("top_p", srv.args.top_p))
            min_p = float(req.get("min_p", srv.args.min_p))
            seed = req.get("seed")
            rng = np.random.default_rng(seed if isinstance(seed, int) else None)
            stops = req.get("stop") or []
            if isinstance(stops, str):
                stops = [stops]
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())

            def chunk(delta, finish=None):
                return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": srv.model_name,
                        "system_fingerprint": "q38", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            with srv.turn(not tools):
                t0 = time.time()
                alive = True
                if stream:
                    # headers at once and a comment every few seconds while the prompt is read:
                    # a long prompt takes minutes, and HTTP clients give up on silent servers
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.flush()
                result = {}

                def read_prompt():
                    try:
                        result["reused"] = eng.prepare(prompt)
                    except Exception as e:                 # noqa: BLE001
                        result["error"] = e

                worker = threading.Thread(target=read_prompt)
                worker.start()
                while worker.is_alive():
                    worker.join(5.0)
                    if worker.is_alive() and stream and alive:
                        try:
                            note = (": reading the prompt, %d s\n\n" % int(time.time() - t0)).encode()
                            self.wfile.write(b"%x\r\n" % len(note) + note + b"\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            alive = False
                if "error" in result:
                    eng.live, eng.snap = [], -1
                    msg = "the engine failed while reading the prompt: %s" % result["error"]
                    if not stream:
                        return self._error(500, msg, "server_error")
                    try:
                        self._sse({"error": {"code": 500, "message": msg, "type": "server_error"}})
                        done = b"data: [DONE]\n\n"
                        self.wfile.write(b"%x\r\n" % len(done) + done + b"\r\n0\r\n\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                    return
                reused = result["reused"]
                t1 = time.time()
                log("prompt %d tokens (%d reused, %d read in %.1f s)" % (len(prompt), reused, len(prompt) - reused, t1 - t0))
                parser = OutputParser(thinking, tools)
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                reasoning, content, calls = [], [], []
                finish, n_out = "stop", 0
                text_so_far = ""

                def emit(events):
                    nonlocal alive
                    for kind, val in events:
                        if kind == "reasoning":
                            reasoning.append(val)
                            d = {"reasoning_content": val}
                        elif kind == "content":
                            content.append(val)
                            d = {"content": val}
                        else:
                            name, args_json = val
                            call = {"index": len(calls), "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                    "function": {"name": name, "arguments": args_json}}
                            calls.append(call)
                            d = {"tool_calls": [call]}
                        if stream and alive:
                            try:
                                self._sse(chunk(d))
                            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                                alive = False

                if stream:
                    try:
                        self._sse(chunk({"role": "assistant", "content": None}))
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        alive = False
                try:
                    while alive:
                        if n_out >= max_tokens:
                            finish = "length"
                            break
                        tok = sample(eng.logits, temp, top_k, top_p, min_p, rng)
                        if tok in eng.eos:
                            break
                        n_out += 1
                        piece = "" if tok in eng.control else decoder.decode(eng.pieces[tok])
                        if piece:
                            text_so_far += piece
                            hit = next((s for s in stops if s and s in text_so_far), None)
                            if hit:
                                cut = text_so_far.index(hit)
                                piece = piece[:max(0, len(piece) - (len(text_so_far) - cut))]
                                emit(parser.feed(piece))
                                break
                            emit(parser.feed(piece))
                        if n_out < max_tokens and len(eng.live) + 1 < n_ctx:
                            eng.step(tok)
                        else:
                            finish = "length"
                            break
                except Exception as e:                     # noqa: BLE001
                    log("generation failed: %s" % e)
                    eng.live, eng.snap = [], -1
                    finish = "stop"
                emit(parser.feed(decoder.decode(b"", final=True), final=True))
                t2 = time.time()
                if calls and finish == "stop":
                    finish = "tool_calls"
                gen_s = max(t2 - t1, 1e-9)
                log("answer %d tokens in %.1f s (%.1f tok/s), %s" % (n_out, gen_s, n_out / gen_s, finish))
                usage = {"prompt_tokens": len(prompt), "completion_tokens": n_out, "total_tokens": len(prompt) + n_out,
                         "prompt_tokens_details": {"cached_tokens": reused}}
                timings = {"prompt_n": len(prompt) - reused, "prompt_ms": 1e3 * (t1 - t0), "cache_n": reused,
                           "predicted_n": n_out, "predicted_ms": 1e3 * gen_s,
                           "predicted_per_second": n_out / gen_s,
                           "prompt_per_second": (len(prompt) - reused) / max(t1 - t0, 1e-9)}
                if stream:
                    if alive:
                        try:
                            last = chunk({}, finish)
                            last["timings"] = timings
                            self._sse(last)
                            if usage_wanted:
                                u = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                     "model": srv.model_name, "system_fingerprint": "q38", "choices": [], "usage": usage}
                                self._sse(u)
                            done = b"data: [DONE]\n\n"
                            self.wfile.write(b"%x\r\n" % len(done) + done + b"\r\n0\r\n\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            pass
                    return
                msg = {"role": "assistant", "content": "".join(content) if content or not calls else None}
                if reasoning:
                    msg["reasoning_content"] = "".join(reasoning)
                if calls:
                    msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
                return self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": srv.model_name,
                                        "system_fingerprint": "q38",
                                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                        "usage": usage, "timings": timings})

    return Handler


# --------------------------------------------------------------------------- deepseek-harness
def _answers(url):
    try:
        urllib.request.urlopen(url, timeout=2).close()
        return True
    except urllib.error.HTTPError:
        return True                                    # it is serving, whatever the page says
    except Exception:                                  # noqa: BLE001
        return False


def _find_browser():
    pf, pf86, local = os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", ""), os.environ.get("LOCALAPPDATA", "")
    for p in (os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
              os.path.join(pf86, "Google", "Chrome", "Application", "chrome.exe"),
              os.path.join(local, "Google", "Chrome", "Application", "chrome.exe"),
              os.path.join(pf86, "Microsoft", "Edge", "Application", "msedge.exe"),
              os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe")):
        if p and os.path.exists(p):
            return p
    return None


def open_app_window(url, browser):
    """The page as an app window (no tabs, no address bar) of a browser profile of its own,
    without extensions, sync or background services: the lightest way to keep it open."""
    exe = browser or _find_browser()
    if not exe:
        log("no Chrome or Edge found; open %s yourself" % url)
        return
    profile = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "localcode", "harness-browser")
    flags = ["--app=" + url, "--user-data-dir=" + profile, "--no-first-run", "--no-default-browser-check",
             "--disable-extensions", "--disable-component-extensions-with-background-pages",
             "--disable-background-networking", "--disable-sync", "--disable-default-apps",
             "--disable-features=Translate,MediaRouter,OptimizationHints", "--window-size=1320,920"]
    kw = {"creationflags": 0x00000008} if os.name == "nt" else {}       # DETACHED_PROCESS
    subprocess.Popen([exe] + flags, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    log("opened %s in %s (app window)" % (url.split("?")[0], os.path.basename(exe)))


_JOB = None


def _tie_to_this_process(proc):
    """Windows: put proc (and what it starts) in a job that ends when this process ends,
    however it ends, so closing the console does not leave the harness running."""
    global _JOB
    if os.name != "nt":
        return
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    class IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = 0x2000                      # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) and \
            k32.AssignProcessToJobObject(job, ctypes.c_void_p(int(proc._handle))):
        _JOB = job                                                      # closed when this process ends


def start_harness(args):
    """deepseek-harness's web UI (dsh web --no-open) pointed at this API, then its app window.
    Returns the harness process this server started (stopped with it), or None."""
    url = "http://127.0.0.1:%d" % args.harness_port
    if _answers(url):
        log("deepseek-harness already answers at %s" % url)
        open_app_window(url, args.browser)
        return None
    bin_ts = os.path.join(args.harness_dir, "apps", "cli", "src", "bin.ts")
    if not os.path.exists(bin_ts):
        log("deepseek-harness not found in %s (give --harness-dir)" % args.harness_dir)
        return None
    env = dict(os.environ)
    host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    # the address of this API for the harness's localcode route (and for a profile still on the llamacpp route)
    env["LOCALCODE_BASE_URL"] = env["LLAMACPP_BASE_URL"] = "http://%s:%d" % (host, args.port)
    logdir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "localcode")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, "harness.log")
    start_at = os.path.getsize(logpath) if os.path.exists(logpath) else 0
    logf = open(logpath, "ab")
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}       # CREATE_NO_WINDOW
    proc = subprocess.Popen(["node", "--import", "tsx/esm", bin_ts, "web", "--no-open", "--port", str(args.harness_port)],
                            cwd=args.harness_dir, env=env, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, **kw)
    try:
        _tie_to_this_process(proc)
    except Exception as e:                                              # noqa: BLE001
        log("could not tie deepseek-harness to this process (%s); stop it yourself if the API is closed" % e)
    log("starting deepseek-harness (log: %s)" % logpath)
    # the UI's address carries an access token: "dsh web: http://127.0.0.1:3080/?token=..."
    t0, page = time.time(), None
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            log("deepseek-harness exited with code %d; see %s" % (proc.returncode, logpath))
            return None
        if page is None:
            try:
                with open(logpath, "rb") as f:
                    f.seek(start_at)
                    m = re.search(r"dsh web: (https?://\S+)", f.read().decode("utf-8", "replace"))
                if m:
                    page = m.group(1)
            except OSError:
                pass
        if page and _answers(url):
            log("deepseek-harness ready at %s in %.0f s" % (url, time.time() - t0))
            open_app_window(page, args.browser)
            return proc
        time.sleep(1)
    if _answers(url):
        log("deepseek-harness answers at %s but printed no address; opening it without one" % url)
        open_app_window(url, args.browser)
    else:
        log("deepseek-harness did not answer at %s in 3 minutes; see %s" % (url, logpath))
    return proc


def stop_process_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=os.environ.get("LOCALCODE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=32768, help="tokens of context (the device indexer allows up to 32768)")
    ap.add_argument("--gpu", type=int, default=-1, help="trunk device; -1 tries 1 then 0, -2 the CPU only")
    ap.add_argument("--preload-gb", type=float, default=24.0)
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "xhigh"],
                    help="the template's reasoning effort when a request names none")
    ap.add_argument("--no-thinking", action="store_true", help="answer without a reasoning block by default")
    ap.add_argument("--turns-first", action="store_true",
                    help="requests without tools (session titles) wait for the conversation's turns; the answer "
                         "comes sooner, but a title can outlast the harness's 60 s title timeout")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--dll", default=None)
    ap.add_argument("--harness", action="store_true",
                    help="once the model is ready, start deepseek-harness's web UI and open it as an app window")
    ap.add_argument("--harness-dir", default=os.environ.get("DSH_DIR", os.path.join(os.path.dirname(ROOT), "deepseek-harness")))
    ap.add_argument("--harness-port", type=int, default=3080)
    ap.add_argument("--browser", default=None, help="Chrome or Edge executable (found by itself when omitted)")
    args = ap.parse_args()
    if not os.path.exists(args.model):
        sys.exit("no model at %s (give --model or set LOCALCODE_MODEL)" % args.model)
    srv = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(srv))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log("listening on http://%s:%d (loading the model; /health answers 503 until it is ready)" % (args.host, args.port))
    srv.load()
    harness = start_harness(args) if args.harness and srv.engine else None
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log("stopping")
    finally:
        stop_process_tree(harness)
        httpd.shutdown()
        with srv.lock:
            if srv.engine:
                srv.engine.close()


if __name__ == "__main__":
    main()
