"""The agent benchmark: the tasks of tools/bench/tasks (make_tasks.py) done by localcode and by opencode
with the same model on this machine, and checked.

    python tools/bench/bench.py [--agents localcode,opencode] [--tasks a,b] [--tag NAME] [--model GGUF]

For each task and agent: the project is copied into a fresh folder under
%LOCALAPPDATA%\\localcode\\bench\\runs\\<tag>\\<agent>\\<task>, the agent is given prompt.txt with every
permission granted, then check.py runs there (the agents never see it). localcode runs its own engine;
opencode talks to the same engine through tools/q38_server.py (OpenAI-compatible, started here) with
the same sampling. The table goes to runs\\<tag>\\results.md and results.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "localcode", "bench")
LOCALCODE = os.path.join(REPO, "build", "Release", "localcode.exe")
OPENCODE = os.path.join(BASE, "opencode", "node_modules", ".bin", "opencode.cmd")
DEFAULT_MODEL = os.path.join(os.path.expanduser("~"), "models", "Qwen3.8-27B", "Qwen3.8-27B-UD-Q5_K_M.gguf")
PORT = 8080


def server_python():
    """a Python with what tools/q38_server.py needs (jinja2, numpy): BENCH_PY, or the one start-api.bat finds"""
    if os.environ.get("BENCH_PY"):
        return os.environ["BENCH_PY"]
    venv = os.path.join(REPO, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv):
        return venv
    for v in ("Python313", "Python312", "Python311", "Python310"):
        p = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", v, "python.exe")
        if os.path.exists(p):
            return p
    return sys.executable
TIMEOUT = 20 * 60


def fresh(task, agent, tag):
    d = os.path.join(BASE, "runs", tag, agent, task)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join(HERE, "tasks", task, "project"), d)
    return d


def check(task, d):
    shutil.copy(os.path.join(HERE, "tasks", task, "check.py"), os.path.join(d, "check.py"))
    r = subprocess.run([sys.executable, "check.py"], cwd=d, capture_output=True, text=True, timeout=120)
    os.remove(os.path.join(d, "check.py"))
    return r.returncode == 0


def run_localcode(prompt, d, log):
    env = dict(os.environ, LOCALCODE_PLAIN="1", MSYS_NO_PATHCONV="1", PWD=d)
    inp = "/approve auto\n" + prompt.replace("\n", " ") + "\n/quit\n"
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([LOCALCODE, "--level", "2"], cwd=d, input=inp, stdout=f, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", env=env, timeout=TIMEOUT)
    return time.time() - t0, p.returncode


def opencode_config():
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "q38": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "q38 (this machine)",
                "options": {"baseURL": f"http://127.0.0.1:{PORT}/v1"},
                "models": {"qwen3.8-27b": {"name": "Qwen3.8-27B", "limit": {"context": 49152, "output": 16384}}},
            }
        },
        "model": "q38/qwen3.8-27b",
        "small_model": "q38/qwen3.8-27b",
        "permission": "allow",
        "autoupdate": False,
        "share": "disabled",
    }
    p = os.path.join(BASE, "opencode.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return p


def server_up(timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def run_opencode(prompt, d, log, cfg):
    # opencode takes its folder from PWD when it is set (a bash parent sets it): this task's folder
    env = dict(os.environ, OPENCODE_CONFIG=cfg, OPENCODE_DISABLE_AUTOUPDATE="1", OPENCODE_DISABLE_SHARE="1", PWD=d)
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([OPENCODE, "run", "--model", "q38/qwen3.8-27b", prompt], cwd=d, stdout=f,
                           stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=env, timeout=TIMEOUT, shell=False)
    return time.time() - t0, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="localcode,opencode")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    # the tasks written again: a run must never start from a task an agent changed in place
    subprocess.run([sys.executable, os.path.join(HERE, "make_tasks.py")], check=True, capture_output=True)
    tasks = a.tasks.split(",") if a.tasks else sorted(os.listdir(os.path.join(HERE, "tasks")))
    agents = a.agents.split(",")
    out = os.path.join(BASE, "runs", a.tag)
    os.makedirs(out, exist_ok=True)
    results = []
    for agent in agents:
        server = None
        cfg = None
        if agent == "opencode":
            cfg = opencode_config()
            srv_log = open(os.path.join(out, "q38_server.log"), "w", encoding="utf-8", errors="replace")
            server = subprocess.Popen([server_python(), os.path.join(REPO, "tools", "q38_server.py"), "--model", a.model,
                                       "--port", str(PORT), "--ctx", "49152", "--reasoning-effort", "medium", "--gpu", "1"],
                                      cwd=REPO, stdout=srv_log, stderr=subprocess.STDOUT)
            if not server_up(600):
                print("the q38 server did not start; see", srv_log.name)
                server.kill()
                continue
        for task in tasks:
            prompt = open(os.path.join(HERE, "tasks", task, "prompt.txt"), encoding="utf-8").read().strip()
            d = fresh(task, agent, a.tag)
            log = os.path.join(out, f"{agent}-{task}.log")
            try:
                secs, rc = run_localcode(prompt, d, log) if agent == "localcode" else run_opencode(prompt, d, log, cfg)
                timed_out = False
            except subprocess.TimeoutExpired:
                secs, rc, timed_out = TIMEOUT, -1, True
            ok = check(task, d)
            r = {"agent": agent, "task": task, "solved": ok, "seconds": round(secs, 1), "exit": rc, "timeout": timed_out}
            results.append(r)
            print(json.dumps(r), flush=True)
        if server:
            server.terminate()
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    lines = ["| tarefa | " + " | ".join(agents) + " |", "|---|" + "---|" * len(agents)]
    for task in tasks:
        cells = []
        for agent in agents:
            r = next((x for x in results if x["agent"] == agent and x["task"] == task), None)
            cells.append("—" if not r else f"{'resolvida' if r['solved'] else 'falhou'} · {r['seconds']:.0f} s")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    tot = []
    for agent in agents:
        rs = [x for x in results if x["agent"] == agent]
        tot.append(f"{sum(x['solved'] for x in rs)}/{len(rs)} · {sum(x['seconds'] for x in rs):.0f} s")
    lines.append("| **total** | " + " | ".join(tot) + " |")
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
