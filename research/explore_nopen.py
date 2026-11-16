"""Exploratory checks on study 19, not pre-registered: finished-and-right, loops, and cut runs that already had the answer."""
import json, sys
sys.path[:0] = ["whyfix"]
from neurospeak import loopiness
from gsm_score import ci
from sensitivity_replicate_rules import right

A = ("doubt", "none", "none_pp", "doubt_pp", "astra_pp")
R = {a: [json.loads(l) for l in open(f"runs/research/head_{a}_traces.jsonl")] for a in A}
fin = {a: [100 * (right(r) and not r["cut"]) for r in v] for a, v in R.items()}
out = dict(
    finished_right={a: sum(x > 0 for x in fin[a]) for a in A},
    loopy={a: sum(loopiness(r["completion"]) > 0.5 for r in R[a]) for a in A},
    cut_but_right={a: sum(right(r) and r["cut"] for r in R[a]) for a in A},
    strict_doubt_vs_none=ci(fin["doubt"], fin["none"]),
    strict_doubt_vs_astra=ci(fin["doubt"], fin["astra_pp"]),
    strict_doubtpp_vs_doubt=ci(fin["doubt_pp"], fin["doubt"]),
    strict_interaction=ci([p - q for p, q in zip(fin["doubt_pp"], fin["doubt"])], [p - q for p, q in zip(fin["none_pp"], fin["none"])]),
    finished_wrong={a: sum(not r["cut"] and not right(r) for r in R[a]) for a in A},
    overlap_doubt_astra=dict(both=sum(right(x) and right(y) for x, y in zip(R["doubt"], R["astra_pp"])), only_doubt=sum(right(x) and not right(y) for x, y in zip(R["doubt"], R["astra_pp"])), only_astra=sum(right(y) and not right(x) for x, y in zip(R["doubt"], R["astra_pp"]))),
)
json.dump(out, open("runs/research/nopen_explore.json", "w"), indent=1)
print(json.dumps(out, indent=1))
