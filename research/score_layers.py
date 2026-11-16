"""Judges bars Y1-Y5 in PLAN_LAYERS.md for whichever arms have 64 rows; controls are scored when present."""
import json, os, re, sys
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", "whyfix"]
from behaviors import DOUBT
from gsm_score import ci
from sensitivity_replicate_rules import right

LAYERS = (3, 7, 11, 15, 19, 23)
path = lambda a: f"runs/research/layers_{a}_traces.jsonl"
names = ["none"] + [f"dir_layer_{l}{s}" for l in LAYERS for s in ("", "_matched")]
R = {a: [json.loads(l) for l in open(path(a))] for a in names if os.path.exists(path(a))}
R = {a: v for a, v in R.items() if len(v) == 64}
assert all([r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
mean = lambda v: sum(v) / len(v)
ok = {a: [100 * right(r) for r in v] for a, v in R.items()}
fin = {a: [100 * (right(r) and not r["cut"]) for r in v] for a, v in R.items()}
last_box = lambda t: (re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t) or ["x"])[-1].strip()
strict_box = {a: [100 * (right(r) and last_box(r["completion"]) != "") for r in v] for a, v in R.items()}  # exploratory, after Fable's review
doubt = {a: mean([100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in v]) for a, v in R.items()}
bad = lambda r: bool(re.search(r" {200,}", r["completion"])) or collapsed(r["completion"])
arms = {a: dict(correct=sum(x > 0 for x in ok[a]), finished=sum(not r["cut"] for r in R[a]), finished_right=sum(x > 0 for x in fin[a]), right_last_box_nonempty=sum(x > 0 for x in strict_box[a]),
                doubt_per_100=round(doubt[a], 3), bad=sum(bad(r) for r in R[a]), tokens=sum(r["n_tokens"] for r in R[a]) // 64) for a in R}
out = dict(arms=arms, layers={})
for l in LAYERS:
    a, m = f"dir_layer_{l}", f"dir_layer_{l}_matched"
    if a not in R:
        continue
    y = dict(Y1=ci(ok[a], ok["none"]), Y3=doubt[a] <= doubt["none"] * 2 / 3, Y5=arms[a]["bad"] <= arms["none"]["bad"] + 2,
             strict_vs_none=ci(fin[a], fin["none"]), nonempty_vs_none=ci(strict_box[a], strict_box["none"]))
    if m in R:
        y["Y2"] = ci(ok[a], ok[m])
        y["nonempty_vs_control"] = ci(strict_box[a], strict_box[m])
    if l != 19 and "dir_layer_19" in R:
        y["Y4"] = ci(ok[a], ok["dir_layer_19"])
    y["pass"] = dict(Y1=y["Y1"]["low"] > 0, Y2=y["Y2"]["low"] > 0 if "Y2" in y else None, Y3=y["Y3"], Y5=y["Y5"],
                     Y4=None if "Y4" not in y else "higher" if y["Y4"]["low"] > 0 else "lower" if y["Y4"]["high"] < 0 else "none shown")
    out["layers"][l] = y
json.dump(out, open("runs/research/layers_results.json", "w"), indent=1)
print(json.dumps(out["arms"], indent=1))
print(json.dumps({l: y["pass"] for l, y in out["layers"].items()}, indent=1))
