// modhost.mjs - runs localcode's mods: Claude Code mod modules (claude-code-playground's format)
// loaded unchanged, talking to localcode by JSON lines on stdin/stdout (src/cli/lc_mods.c).
//
//     node modhost.mjs <module.mjs> ...
//
// A module exports register(on); on(event, [filter], handler) adds a handler; a handler is
// handler($, e, next): next(e) runs the following handlers and then localcode's default.
//
// localcode -> host:   {"id": N, "event": "tool.call", "data": {...}, "state": {...}}
//                      {"press": "<button id>"}        a button pressed
//                      {"abort": N}                    the request N is no longer wanted
//                      {"reply": K, "value": ...}      the answer to the host's call K
// host -> localcode:   {"id": N, "result": ...}        the event's result
//                      {"call": K, "method": "ui.open", "args": {...}}   (an answer is awaited)
//                      {"notify": "ui.invalidate" | "ui.toast" | "ui.close" | "command.register", "args": {...}}
//                      {"log": "..."}                  for localcode's log
//
// Events: session.start, turn.start, turn.complete, tool.call (the result: {"deny": "why"} or
// whatever the default gives, {"proceed": true}), command.run (its result {"text"}), ui.render
// ({"component": "AbovePrompt" | "Pane", "requestId", "bodyColumns", "maxRows"}: the result is a
// tree of {"type": "Box" | "Text" | "Button", ...props, "children": [...]}), ui.close.

import { spawn } from "node:child_process";
import { promises as fsp, existsSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import readline from "node:readline";

const handlers = [];             // { event, filter, fn, mod }
const state = { cwd: process.cwd(), usage: { tokens: 0, window: 0, percent: 0 } };
const waiting = new Map();        // the host's calls to localcode: K -> resolve
const aborts = new Map();         // request N -> its signal
const buttons = new Map();        // button id -> onPress
let callSeq = 0;
let buttonSeq = 0;

function send(o) { process.stdout.write(JSON.stringify(o) + "\n"); }
function log(s) { send({ log: String(s) }); }

function call(method, args) {
  const k = ++callSeq;
  return new Promise((resolve) => { waiting.set(k, resolve); send({ call: k, method, args }); });
}
function notify(name, args) { send({ notify: name, args: args ?? {} }); }

// Git's bash on Windows (not WSL's), as for localcode's hooks
function gitTool(exe) {
  if (process.platform !== "win32") return exe;
  const bases = [process.env.ProgramW6432, process.env.ProgramFiles, process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "Programs")];
  for (const b of bases) {
    if (!b) continue;
    for (const sub of ["Git/bin", "Git/usr/bin"]) {
      const p = path.join(b, sub, exe + ".exe");
      if (existsSync(p)) return p;
    }
  }
  return exe;
}

// The host object a handler gets: what it may do and ask
const $ = {
  session: {
    cwd: async () => state.cwd,
    usage: async () => ({ context: { ...state.usage } }),
  },
  process: {
    run: (argv, opts = {}) => new Promise((resolve) => {
      if (argv[0] === "sleep") {              // no sleep program on Windows' PATH: a timer
        setTimeout(() => resolve({ code: 0, exitCode: 0, stdout: "", stderr: "" }), Number(argv[1]) * 1000);
        return;
      }
      const prog = argv[0] === "bash" || argv[0] === "sh" ? gitTool(argv[0]) : argv[0];
      let out = "", err = "", done = false;
      let p;
      try {
        p = spawn(prog, argv.slice(1), { cwd: opts.cwd ?? state.cwd, windowsHide: true, env: { ...process.env, ...(opts.env ?? {}) } });
      } catch (e) {
        resolve({ code: -1, exitCode: -1, stdout: "", stderr: String(e) });
        return;
      }
      const t = opts.timeoutMs ? setTimeout(() => { if (!done) p.kill(); }, opts.timeoutMs) : null;
      p.stdout.on("data", (d) => { out += d; });
      p.stderr.on("data", (d) => { err += d; });
      if (opts.input !== undefined) p.stdin.end(String(opts.input)); else p.stdin.end();
      p.on("error", (e) => { if (!done) { done = true; if (t) clearTimeout(t); resolve({ code: -1, exitCode: -1, stdout: out, stderr: String(e) }); } });
      p.on("close", (code) => { if (!done) { done = true; if (t) clearTimeout(t); resolve({ code: code ?? -1, exitCode: code ?? -1, stdout: out, stderr: err }); } });
    }),
  },
  fs: {
    exists: async (p) => { try { await fsp.access(p); return true; } catch { return false; } },
    read: async (p) => fsp.readFile(p, "utf8"),
    readFile: async (p) => fsp.readFile(p, "utf8"),
  },
  clock: {
    now: async () => Date.now(),
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
  },
  ui: {
    open: async (o) => (await call("ui.open", o)) ?? { isPlaced: false },
    close: async (o) => { notify("ui.close", o); },
    invalidate: () => { notify("ui.invalidate"); },
    toast: (text) => { notify("ui.toast", { text: String(text) }); },
    resolve: () => elements,
  },
  command: {
    register: async (c) => { notify("command.register", c); },
  },
};

// The elements a render returns: plain objects; a Button's onPress kept under an id
function kids(c) {
  if (c === undefined || c === null || c === false) return [];
  return (Array.isArray(c) ? c : [c]).flatMap((x) => (Array.isArray(x) ? kids(x) : x === null || x === false || x === undefined ? [] : [x]));
}
const elements = {
  Box: (p = {}) => ({ ...p, type: "Box", children: kids(p.children) }),
  Text: (p = {}) => ({ ...p, type: "Text", children: kids(p.children).map((x) => (typeof x === "object" ? x : String(x))) }),
  Button: (p = {}) => {
    const id = `b${++buttonSeq}`;
    if (typeof p.onPress === "function") buttons.set(id, p.onPress);
    const { onPress, ...rest } = p;
    return { ...rest, type: "Button", id };
  },
  Svg: () => ({ type: "Text", children: [] }),
};

function filterMatches(filter, e) {
  if (!filter) return true;
  for (const [k, v] of Object.entries(filter)) {
    if (k === "tool" && e.tool !== v) return false;
    if (k === "component" && e.component !== v) return false;
    if (k === "command" && e.command !== v) return false;
  }
  return true;
}

function defaultResult(event, e) {
  if (event === "tool.call") return { proceed: true };
  if (event === "ui.render") return null;
  if (event === "command.run") return { text: "" };
  return undefined;
}

async function dispatch(event, e, signal) {
  const chain = handlers.filter((h) => h.event === event && filterMatches(h.filter, e));
  const run = async (i, ev) => {
    if (i >= chain.length) return defaultResult(event, ev);
    const next = (ev2) => run(i + 1, ev2 ?? ev);
    next.signal = signal;
    try {
      return await chain[i].fn($, ev, next);
    } catch (err) {
      log(`${chain[i].mod}: ${event} failed: ${err && err.stack ? err.stack : err}`);
      return next(ev);
    }
  };
  return run(0, e);
}

// Loading the modules
for (const m of process.argv.slice(2)) {
  try {
    const mod = await import(pathToFileURL(path.resolve(m)).href);
    const name = path.basename(path.dirname(path.dirname(path.resolve(m))));
    if (typeof mod.register === "function") {
      mod.register((event, a, b) => {
        const filter = typeof a === "function" ? null : a;
        const fn = typeof a === "function" ? a : b;
        handlers.push({ event, filter, fn, mod: name });
      }, {});
    }
  } catch (err) {
    log(`cannot load ${m}: ${err && err.stack ? err.stack : err}`);
  }
}
send({ ready: true, handlers: handlers.length });

// localcode's lines
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", async (line) => {
  let msg;
  try { msg = JSON.parse(line); } catch { return; }
  if (msg.reply !== undefined) {
    const r = waiting.get(msg.reply);
    waiting.delete(msg.reply);
    if (r) r(msg.value);
    return;
  }
  if (msg.press !== undefined) {
    const f = buttons.get(msg.press);
    if (f) { try { await f(); } catch (err) { log(`button failed: ${err}`); } }
    return;
  }
  if (msg.abort !== undefined) {
    const s = aborts.get(msg.abort);
    if (s) s.aborted = true;
    return;
  }
  if (msg.event) {
    if (msg.state) Object.assign(state, msg.state);
    if (buttons.size > 2000) buttons.clear();          // old renders' buttons (ids are never reused)
    const signal = { aborted: false };
    aborts.set(msg.id, signal);
    let result;
    try {
      result = await dispatch(msg.event, msg.data ?? {}, signal);
    } catch (err) {
      log(`${msg.event}: ${err}`);
    }
    aborts.delete(msg.id);
    send({ id: msg.id, result: result === undefined ? null : result });
  }
});
rl.on("close", () => process.exit(0));
