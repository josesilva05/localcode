"""The cases of tools/lc_regex_check.c: random patterns in the syntax lc_regex supports, random texts,
and what Python's re says of each (the reference).

    python tools/lc_regex_cases.py OUT.txt [N] [SEED]

The texts use letters (with lowercase accented ones), digits, spaces and punctuation, never a newline,
and accented letters only in lowercase: on those, Python's Unicode rules and lc_regex's bytes agree.
A set of realistic grep patterns (code searches) is added to the random ones.
"""
import random
import re
import sys

ALPHA = "abcxyzAB_019 .-()[]{}*+?|^$\\/\"':;=,#ãéç"
LITS = "abcxyzAB019_ -=:,#ãé"
ESCAPED = r"\. \( \) \[ \] \{ \} \* \+ \? \| \^ \$ \\ \/ \- \#".split()
CLASSES = [r"\d", r"\w", r"\s", r"\D", r"\W", r"\S", "."]
BRACKETS = ["[abc]", "[^abc]", "[a-z]", "[^a-z]", "[A-Z0-9_]", r"[\d.]", r"[^\s]", "[x-]", "[]a]", r"[\]]", "[ãé]", r"[\w-]", "[^ã]"]
QUANT = ["*", "+", "?", "{2}", "{1,3}", "{0,2}", "{2,}", "*?", "+?", "??", "{1,2}?"]
ANCHORS = ["^", "$", r"\b", r"\B"]

REAL = [
    r"def \w+\(", r"^import ", r"^\s*#", r"TODO|FIXME", r"\bfoo\b", r"foo\(\s*\)", r"\d+\.\d+", r"[A-Z_]{3,}",
    r"class \w+(?:\(\w*\))?:", r"return\s+None", r"^\s*$", r"\$\{?\w+\}?", r"https?://\S+", r"\w+@\w+\.\w+",
    r"(?:get|set)_\w+", r"x{2,}", r"a.c", r"a.*c", r"^.{5}$", r"[^ ]+ [^ ]+", r"ç[aã]o", r"\.py$", r"\\n",
]
REAL_TEXTS = [
    "def main(args):", "import os", "    # a comment", "x = 1  # TODO later", "foo()", "foo( )", "barfoo baz",
    "version 3.14 out", "MAX_SIZE = 10", "class Thing(Base):", "class Other:", "    return None", "",
    "   ", "echo ${HOME} $PATH", "see https://example.com/x?y=1 now", "mail jose@host.com", "get_value(x)",
    "xxx", "abc", "a-c", "aãc", "a long line with c", "12345", "123456", "ação", "configuração", "main.py",
    "a\\nb", "set_x = 2", "ab cd", "FOO_BAR", "çao",
]


def atom(r, d):
    k = r.random()
    if d > 0 and k < 0.18:
        inner = alt(r, d - 1)
        return "(" + ("?:" if r.random() < 0.3 else "") + inner + ")"
    if k < 0.32:
        return r.choice(CLASSES)
    if k < 0.45:
        return r.choice(BRACKETS)
    if k < 0.52:
        return r.choice(ESCAPED)
    if k < 0.58:
        return r.choice(ANCHORS)
    return r.choice(LITS)


def piece(r, d):
    a = atom(r, d)
    if a in ANCHORS:
        return a
    if r.random() < 0.3:
        a += r.choice(QUANT)
    return a


def seq(r, d):
    return "".join(piece(r, d) for _ in range(r.randint(1, 4)))


def alt(r, d):
    s = seq(r, d)
    while r.random() < 0.2:
        s += "|" + seq(r, d)
    return s


def text(r):
    return "".join(r.choice(ALPHA) for _ in range(r.randint(0, 14)))


def verdict(p, icase, t):
    try:
        rx = re.compile(p, re.IGNORECASE if icase else 0)
    except re.error:
        return "E"
    return "1" if rx.search(t) else "0"


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    r = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    cases = []
    for p in REAL:
        for t in REAL_TEXTS:
            for ic in (0, 1):
                cases.append((p, ic, t))
    for _ in range(n):
        p = alt(r, 2)
        ic = 1 if r.random() < 0.25 else 0
        for _ in range(3):
            cases.append((p, ic, text(r)))
    # patterns both must refuse
    for p in ["(", "a)", "[a", "*a", "a{2,1}", "\\q","a|*", "\\", "[b-a]", "(?<n>x)"]:
        cases.append((p, 0, "abc"))
    # a pattern that would take exponential time in a backtracking engine, on a long text
    cases.append(("(a*)*b", 0, "a" * 5000))
    cases.append(("(a|aa)+$", 0, "a" * 3000 + "!"))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for p, ic, t in cases:
            v = verdict(p, ic, t) if len(t) < 1000 else "0"
            f.write(f"{p}\x1f{ic}\x1f{t}\x1f{v}\n")
    print(len(cases), "cases")


main()
