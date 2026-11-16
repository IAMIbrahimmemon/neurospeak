"""Check on study 10, no bars: how often does a different problem's answer appear in a run's text by coincidence?"""
import json, random, re, sys
sys.path[:0] = ["whyfix"]
from sensitivity_replicate_rules import value
NUM = re.compile(r"(?<![\d.,])-?\d[\d,]*(?:\.\d+)?")
vals = lambda t: {float(m.group().replace(",", "").rstrip(".")) for m in NUM.finditer(t)}
out = {}
for a in ("none", "dir_self_doubt", "dir_self_doubt_matched"):
    R = [json.loads(l) for f in ("gsm64", "gsm64b") for l in open(f"runs/whyfix/{f}_{a}_traces.jsonl")]
    ans = [value(r["answer"]) for r in R]
    tv = [vals(r["completion"]) for r in R]; pv = [vals(r["prompt"]) for r in R]
    rng = random.Random(0); rates = []
    for _ in range(200):
        hit = n = 0
        for i in range(len(R)):
            j = rng.randrange(len(R))
            if j == i or ans[j] is None or ans[j] in pv[i] or ans[j] == ans[i]: continue
            n += 1; hit += ans[j] in tv[i]
        rates.append(100 * hit / n)
    out[a] = dict(placebo_reached_pct=round(sum(rates) / len(rates), 1))
json.dump(out, open("runs/research/commit_audit.json", "w"), indent=1); print(out)
