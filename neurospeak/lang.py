"""The .ns front end: each statement parses to one library call, and its result prints as English."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .core import DEV, Brain, Expectation, Pair, as_addr, as_addrs
from .reports import BOLD, RED, RESET

STR = re.compile(r'"""(.*?)"""|"((?:[^"\\]|\\.)*)"', re.S)


def lex(src: str):
    """Pull string literals out (so they can span lines), then return (lines, strings)."""
    strings = []

    def sub(m):
        strings.append(m[1] if m[1] is not None else bytes(m[2], "utf-8").decode("unicode_escape"))
        return f"§{len(strings) - 1}§"

    body = STR.sub(sub, src)
    lines = []
    for raw in body.split("\n"):
        line = raw.split("#")[0].rstrip()
        if line.strip():
            lines.append((len(line) - len(line.lstrip()), line.strip()))
    return lines, strings


def _match(pattern: str, line: str, usage: str):
    m = re.fullmatch(pattern, line)
    if not m:
        raise SyntaxError(f"expected: {usage}")
    return m


class Interpreter:
    def __init__(self):
        self.b: Brain | None = None
        self.expect_fail = 0

    @property
    def brain(self) -> Brain:
        if self.b is None:
            self.b = Brain()
        return self.b

    def say(self, *a):
        print(*a, flush=True)

    def run_source(self, src: str) -> int:
        lines, S = lex(src)
        i = 0
        while i < len(lines):
            indent, line = lines[i]
            block = []
            if line.endswith(":"):
                j = i + 1
                while j < len(lines) and lines[j][0] > indent:
                    block.append(lines[j][1])
                    j += 1
                i = j
            else:
                i += 1
            self.say(f"\n{BOLD}» {self._render(line, S)}{RESET}")
            try:
                self.exec(line.rstrip(":"), block, S)
            except (SyntaxError, KeyError, ValueError) as e:
                self.say(f"{RED}  error: {e}{RESET}")
        if self.expect_fail:
            self.say(f"\n{RED}{self.expect_fail} expectation(s) failed{RESET}")
        return self.expect_fail

    def _render(self, line, S):
        return re.sub(r"§(\d+)§", lambda m: '"' + (S[int(m[1])][:40].replace("\n", "⏎")) + '…"', line)

    def _strs(self, s, S):
        return [S[int(k)] for k in re.findall(r"§(\d+)§", s)]

    def exec(self, line, block, S):
        w = line.split()
        head = w[0]
        if head == "model":
            self.b = Brain(w[1])
            self.say(f"  loaded {w[1]}: {self.b.n_layers} layers, {self.b.anatomy.n_neurons(0)} MLP neurons/layer, on {DEV}")
        elif head == "suffix":
            self.brain.ask_suffix = self._strs(line, S)[0]
            self.say("  problems passed to ask now end with this text")
        elif head == "concept":
            self._define(line, block, S)
        elif head == "where":
            self.say(self.brain.get(w[1]).where())
        elif head == "describe":
            self._describe(w[1])
        elif head in ("why", "trace"):
            m = _match(rf"{head} (§\d+§)(?: at L(\d+))?", line, f'{head} "text" [at L19]')
            text = self._strs(m[1], S)[0]
            at = int(m[2]) if m[2] else 19
            self.say(self.brain.why(text, at=at) if head == "why" else self.brain.trace(text, at=at))
        elif head == "word":
            m = _match(r"word L(\d+)\.f(\d+)", line, "word L19.f3686")
            self.say(self.brain.features[int(m[1])][int(m[2])].record() or f"  no name saved for L{m[1]}.f{m[2]}")
        elif head == "writers":
            m = _match(r"writers of (\w+) in (.+?)(?: top (\d+))?", line, "writers of NAME in L15.mlp, L19.mlp [top K]")
            c = self.brain.get(m[1])
            for a in as_addrs(m[2].split(",")):
                self.say(c.writers(a, int(m[3] or 10)))
        elif head == "read":
            m = _match(r"read (\w+) on (§\d+§)(?: at (L\d+))?", line, 'read NAME on "text" [at L16]')
            self.say(self.brain.get(m[1]).read(self._strs(m[2], S)[0], as_addr(m[3]).layer if m[3] else None))
        elif head == "erase":
            m = _match(r"erase (\w+) from (.+?)(?: writers top (\d+))?(?: strength ([\d.]+))?", line,
                       "erase NAME from L15.mlp, L19.mlp [writers top K] [strength S]")
            self.say(self.brain.erase(m[1], m[2].split(","), int(m[3]) if m[3] else None, float(m[4] or 1)))
        elif head == "silence":
            self.say(self.brain.silence(w[1]))
        elif head == "scale":
            self.say(self.brain.scale(w[1], float(w[3])))
        elif head == "measure":
            self.say(self.brain.get(w[1]).measure())
        elif head == "expect":
            m = _match(r"expect (\w+)\.(is|not) (falls|rises|steady)(?: (?:by|within) ([\d.]+))?", line,
                       "expect NAME.is|not falls|rises|steady [by|within N]")
            c = self.brain.get(m[1])
            e = Expectation(c.is_ if m[2] == "is" else c.not_)
            check = getattr(e, m[3])
            p = check(float(m[4])) if m[4] else check()
            self.expect_fail += not p
            self.say(p)
        elif head == "explain":
            self.say(self.brain.explain_edit(self._strs(line, S)[0]))
        elif head in ("compare", "ask"):
            m = re.search(r"max (\d+)", line)
            n, problem = int(m[1]) if m else 600, self._strs(line, S)[0]
            self.say(self.brain.compare(problem, n) if head == "compare" else self.brain.ask(problem, n))
        elif head == "inject":
            m = _match(r"inject (-?)(\w+) at (L\d+) strength ([\d.]+) into (§\d+§)(?: max (\d+))?", line,
                       'inject [-]NAME at L15 strength K into "problem" [max N]')
            self.say(self.brain.get(m[2]).inject(as_addr(m[3]).layer, float(m[4]) * (-1 if m[1] else 1),
                                                 self._strs(m[5], S)[0], int(m[6] or 400)))
        elif head == "save":
            path = self._strs(line, S)[0]
            self.say(f"  saved {self.brain.save(path)} edited tensor(s) to {path}")
        elif head == "load":
            path = self._strs(line, S)[0]
            self.brain.load_edit(path)
            self.say(f"  applied edit file {path}")
        elif head == "undo":
            label = self.brain.undo()
            self.say(f"  undid: {label}" if label else "  nothing to undo")
        elif head == "reset":
            self.brain.reset()
            self.say("  model restored to original weights")
        elif head == "history":
            for k, (lbl, ch) in enumerate(self.brain.history):
                self.say(f"  {k + 1}. {lbl}  ({len(ch)} tensors)")
        else:
            raise SyntaxError(f"unknown statement {head!r}")

    def _define(self, line, block, S):
        m = _match(r"concept (\w+)(?: at (.+))?", line.rstrip(":"), "concept NAME [at L15, L19]:")
        layers = [as_addr(a).layer for a in m[2].split(",")] if m[2] else []
        pairs, ask, given, cur_is = [], None, "", None
        for b in block:
            kw, vals = b.split()[0], self._strs(b, S)
            if kw == "given":
                ask, given = None, vals[0]
            elif kw == "ask":
                ask, given = vals[0], ""
            elif kw == "is":
                cur_is = vals[0]
            elif kw == "not":
                pairs.append(Pair(cur_is, vals[0], ask, given))
                cur_is = None
            else:
                raise SyntaxError(f"in concept body: expected given/ask/is/not, got {kw!r}")
        self.brain.concept(m[1], pairs, layers)
        where = ", ".join(f"L{l}" for l in layers) or "all layers"
        self.say(f"  learned {BOLD}{m[1]}{RESET} from {len(pairs)} contrast pair(s); anchored at {where}")

    def _describe(self, target):
        b = self.brain
        if target in b.concepts:
            self.say(b.get(target).describe())
            return
        a = as_addr(target)
        if a.kind == "neuron":
            self.say(b.L[a.layer].n[a.index].describe())
        elif a.kind == "head":
            self.say(b.L[a.layer].h[a.index].describe())
        else:
            raise SyntaxError("describe takes a concept, a neuron (L19.n11) or a head (L15.h7)")


def main():
    it = Interpreter()
    if len(sys.argv) > 1:
        sys.exit(1 if it.run_source(Path(sys.argv[1]).read_text()) else 0)
    print("NeuroSpeak REPL — one statement per line; concepts need a script file.")
    while True:
        try:
            line = input("ns> ")
        except EOFError:
            break
        it.run_source(line)
