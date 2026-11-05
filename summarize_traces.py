"""Summarize trace files: accuracy, share cut at the token cap, repeat-rate. Usage: summarize_traces.py a.jsonl b.jsonl ..."""
import json, re, sys
from neurospeak import loopiness


def boxed(t):
    b = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t)
    return b[-1].strip() if b else None


def norm(x):
    return re.sub(r"[\s$,]|\\!|dfrac|tfrac|\\text\{[^}]*\}", "", x or "").replace("frac", "").rstrip(".").lower()


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    acc = sum(norm(boxed(r["completion"])) == norm(r["answer"]) for r in rows)
    cut = sum(r["cut"] for r in rows)
    loop = sum(loopiness(r["completion"]) for r in rows) / n
    toks = sum(r["n_tokens"] for r in rows) / n
    print(f"{path:45s} n={n:3d}  correct {acc:3d} ({acc / n:5.1%})  cut-at-cap {cut / n:5.1%}  "
          f"mean repeat-rate {loop:5.1%}  mean tokens {toks:6.0f}")
