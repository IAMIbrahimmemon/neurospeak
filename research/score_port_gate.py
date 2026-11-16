"""Judges the gate in PLAN_PORT.md."""
import json, sys
sys.path[:0] = ["investigate", "whyfix"]
from behaviors import DOUBT
from gsm_score import boxed
from sensitivity_replicate_rules import right
R = [json.loads(l) for l in open("runs/research/port_none_traces.jsonl")]
assert len(R) == 32
per = [100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in R]
words = [len(r["completion"].split()) for r in R]
out = dict(doubt_per_100=sum(per) / 32, runs_with_doubt=sum(x > 0 for x in per), runs_100_words=sum(w >= 100 for w in words), median_words=sorted(words)[16],
           cut=sum(r["cut"] for r in R), boxed=sum(boxed(r["completion"]) is not None for r in R), correct=sum(right(r) for r in R))
out["gate"] = out["doubt_per_100"] >= 0.3 and out["runs_with_doubt"] >= 10 and out["runs_100_words"] >= 20
json.dump(out, open("runs/research/port_gate.json", "w"), indent=1)
print(out)
print(R[0]["completion"][:900]); print("-----"); print(R[5]["completion"][:600])
