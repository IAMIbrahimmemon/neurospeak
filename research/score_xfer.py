"""Judges bars T1-T5 in PLAN_TRANSFER.md for whichever arms have 64 rows; compares against study 20's traces."""
import json, os, re, sys
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", "whyfix"]
from behaviors import DOUBT
from gsm_score import ci
from sensitivity_replicate_rules import right

LAYERS = (7, 11, 15, 23)
P = "runs/research/"
files = {"none": P + "layers_none_traces.jsonl", "L19": P + "layers_dir_layer_19_traces.jsonl"}
for l in LAYERS:
    files[f"vec_at{l}"] = P + f"xfer_dir_self_doubt_at{l}_traces.jsonl"
    files[f"ctl_at{l}"] = P + f"xfer_dir_xfer_{l}_matched_at{l}_traces.jsonl"
    files[f"own_{l}"] = P + f"layers_dir_layer_{l}_traces.jsonl"
R = {a: [json.loads(x) for x in open(f)] for a, f in files.items() if os.path.exists(f)}
R = {a: v for a, v in R.items() if len(v) == 64}
assert all([r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
for l in LAYERS:
    for a in (f"vec_at{l}", f"ctl_at{l}"):
        if a in R:
            assert all(r["run"]["layer_used"] == l for r in R[a]), a
mean = lambda v: sum(v) / len(v)
ok = {a: [100 * right(r) for r in v] for a, v in R.items()}
fin = {a: [100 * (right(r) and not r["cut"]) for r in v] for a, v in R.items()}
last_box = lambda t: (re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t) or ["x"])[-1].strip()
nonempty = {a: [100 * (right(r) and last_box(r["completion"]) != "") for r in v] for a, v in R.items()}
doubt = {a: mean([100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in v]) for a, v in R.items()}
wait = {a: sum(len(re.findall(r"\bwait\b", r["completion"], re.I)) for r in v) for a, v in R.items()}
bad = lambda r: bool(re.search(r" {200,}", r["completion"])) or collapsed(r["completion"])
arms = {a: dict(correct=sum(x > 0 for x in ok[a]), right_last_box_nonempty=sum(x > 0 for x in nonempty[a]),
                finished=sum(not r["cut"] for r in R[a]), finished_right=sum(x > 0 for x in fin[a]),
                doubt_per_100=round(doubt[a], 3), wait=wait[a], bad=sum(bad(r) for r in R[a]),
                tokens=sum(r["n_tokens"] for r in R[a]) // 64) for a in R}
verdict = lambda c: "higher" if c["low"] > 0 else "lower" if c["high"] < 0 else "none shown"
out = dict(arms=arms, layers={})
for l in LAYERS:
    v, c, o = f"vec_at{l}", f"ctl_at{l}", f"own_{l}"
    if v not in R:
        continue
    t = dict(T1=ci(ok[v], ok["none"]), T5=arms[v]["bad"] <= arms["none"]["bad"] + 2,
             nonempty_vs_none=ci(nonempty[v], nonempty["none"]), finished_right_vs_none=ci(fin[v], fin["none"]))
    if c in R:
        t["T2"] = ci(ok[v], ok[c])
        t["nonempty_vs_control"] = ci(nonempty[v], nonempty[c])
    if o in R:
        t["T3"] = ci(ok[v], ok[o])
    t["T4"] = ci(ok[v], ok["L19"])
    t["pass"] = dict(T1=t["T1"]["low"] > 0, T2=t["T2"]["low"] > 0 if "T2" in t else None,
                     T3=verdict(t["T3"]) if "T3" in t else None, T4=verdict(t["T4"]), T5=t["T5"])
    out["layers"][l] = t
json.dump(out, open(P + "xfer_results.json", "w"), indent=1)
print(json.dumps(out["arms"], indent=1))
print(json.dumps({l: {**t["pass"], **{k: t[k] for k in t if k[:1] == "T" and isinstance(t[k], dict)}} for l, t in out["layers"].items()}, indent=1))
