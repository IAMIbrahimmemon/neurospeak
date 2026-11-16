"""Judges bars B1-B4 in PLAN_BOTH.md."""
import json, sys
from neurospeak import loopiness
from neurospeak.investigate import collapsed
sys.path[:0] = ["investigate", "whyfix", "research"]
from behaviors import DOUBT
from gsm_score import boxed, ci
from loop_onset import HIGH, windows
from sensitivity_replicate_rules import right
A = ("none", "doubt", "rep", "both", "random2")
R = {a: [json.loads(l) for l in open(f"runs/research/both_{a}_traces.jsonl")] for a in A}
assert all(len(v) == 64 and [r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
ok = {a: [100 * right(r) for r in v] for a, v in R.items()}
broken = lambda r: collapsed(r["completion"]) or len(set(r["completion"].split()[:200])) < 0.2 * min(200, max(len(r["completion"].split()), 1))
p = lambda r: 100 * len(DOUBT.findall(" ".join(r["completion"].split()[:500]))) / max(min(500, len(r["completion"].split())), 1)
arms = {a: dict(correct_of_64=sum(x > 0 for x in ok[a]), boxed_of_64=sum(boxed(r["completion"]) is not None for r in v), finished_of_64=sum(not r["cut"] for r in v), broken_of_64=sum(broken(r) for r in v),
                repeats=sum(100 * loopiness(r["completion"]) for r in v) / 64, loops_of_64=sum(any(x >= HIGH for x in windows(r["completion"].split())) for r in v), doubt_first_500=sum(p(r) for r in v) / 64) for a, v in R.items()}
out = dict(arms=arms, B1=ci(ok["both"], ok["doubt"]), B2=ci(ok["both"], ok["rep"]), B3=ci(ok["both"], ok["random2"]))
bar = lambda k, f: out[k]["diff"] >= f and out[k]["low"] > 0
out["bars"] = dict(B1=bar("B1", 5), B2=bar("B2", 5), B3=bar("B3", 10), B4=arms["both"]["broken_of_64"] <= arms["none"]["broken_of_64"] + 2)
json.dump(out, open("runs/research/both_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
