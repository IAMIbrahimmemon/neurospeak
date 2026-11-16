"""Judges bars P1-P5 in PLAN_PAIRS.md. With a directory and arm names as arguments, it dry-runs on older traces and saves nothing."""
import json, sys
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", "whyfix"]
from behaviors import DOUBT
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right

dry = len(sys.argv) > 1
pattern = sys.argv[1] if dry else "runs/research/pairs_{}_traces.jsonl"
N, M = "none", "dir_self_doubt_matched"
W = sys.argv[2:] if dry else ["dir_writer_2", "dir_writer_3"]
R = {a: [json.loads(l) for l in open(pattern.format(a))] for a in (N, *W, M)}
assert all(len(v) == 64 and [(r["id"], r["prompt"]) for r in v] == [(r["id"], r["prompt"]) for r in R[N]] for v in R.values())
mean = lambda v: sum(v) / len(v)
ok = {a: [100 * right(r) for r in v] for a, v in R.items()}
doubt = {a: mean([100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in v]) for a, v in R.items()}
arms = {a: dict(correct_of_64=sum(x > 0 for x in ok[a]), boxed_of_64=sum(boxed(r["completion"]) is not None for r in R[a]), finished_of_64=sum(not r["cut"] for r in R[a]),
                collapsed_of_64=sum(collapsed(r["completion"]) for r in R[a]), doubt_per_100=doubt[a]) for a in R}
cos = json.load(open("runs/research/pair_cosines.json"))
out = dict(arms=arms, vs_unedited={w: ci(ok[w], ok[N]) for w in W}, vs_matched={w: ci(ok[w], ok[M]) for w in W})
out["bars"] = {w: dict(P1=cos[w]["original"] >= 0.3 if w in cos else None,
                       P2=out["vs_unedited"][w]["diff"] >= 5 and out["vs_unedited"][w]["low"] > 0,
                       P3=out["vs_matched"][w]["low"] > 0,
                       P4=arms[w]["collapsed_of_64"] <= arms[N]["collapsed_of_64"] + 2,
                       P5=doubt[w] <= doubt[N] * 2 / 3) for w in W}
if not dry:
    json.dump(out, open("runs/research/pairs_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
