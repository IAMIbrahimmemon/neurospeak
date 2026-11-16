"""Judges bars R1-R5 in PLAN_REPORTH.md (study 27). Unedited and self-doubt arms are study 20's traces on the same problems."""
import json, os, re, sys
from neurospeak import loopiness
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", "whyfix", "research"]
from behaviors import DOUBT
from gsm_score import ci
from sensitivity_replicate_rules import right
from loop_onset import HIGH, windows

P = "runs/research/"
OTHER = ["dir_both_rep", "dir_neurospeak"]
files = {"none": P + "layers_none_traces.jsonl", "doubt": P + "layers_dir_layer_19_traces.jsonl",
         **{a: P + f"otherdir_{a}_traces.jsonl" for a in OTHER}}
R = {a: [json.loads(x) for x in open(f)] for a, f in files.items() if os.path.exists(f)}
R = {a: v for a, v in R.items() if len(v) == 64}
assert all([r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
for a in OTHER:
    if a in R:
        assert all(r["run"]["arm"] == a and r["run"]["layer_used"] == 19 and r["run"]["max"] == 3000 and r["run"]["seed"] == 20
                   and not r["run"]["add"] and r["run"]["gain"] == 1.0 and r["run"]["mode"] == "full" for r in R[a]), a
mean = lambda v: sum(v) / len(v)
ok = {a: [100 * right(r) for r in v] for a, v in R.items()}
last_box = lambda t: (re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t) or ["x"])[-1].strip()
nonempty = {a: [100 * (right(r) and last_box(r["completion"]) != "") for r in v] for a, v in R.items()}
fin = {a: [100 * (right(r) and not r["cut"]) for r in v] for a, v in R.items()}
doubt = {a: mean([100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in v]) for a, v in R.items()}
wait_k = {a: mean([1000 * len(re.findall(r"\bwait\b", r["completion"], re.I)) / max(r["n_tokens"], 1) for r in v]) for a, v in R.items()}
bad = lambda r: bool(re.search(r" {200,}", r["completion"])) or collapsed(r["completion"])
arms = {a: dict(correct=sum(x > 0 for x in ok[a]), right_last_box_nonempty=sum(x > 0 for x in nonempty[a]),
                finished=sum(not r["cut"] for r in R[a]), finished_right=sum(x > 0 for x in fin[a]),
                doubt_per_100=round(doubt[a], 3), wait_per_1k=round(wait_k[a], 2), bad=sum(bad(r) for r in R[a]),
                tokens=sum(r["n_tokens"] for r in R[a]) // 64,
                repeats=round(sum(100 * loopiness(r["completion"]) for r in R[a]) / 64, 1),
                loops=sum(any(x >= HIGH for x in windows(r["completion"].split())) for r in R[a])) for a in R}
done = [a for a in OTHER if a in R]
rep = {a: [100 * loopiness(r["completion"]) for r in v] for a, v in R.items()}
out = dict(arms=arms, done=done)
if "dir_both_rep" in R:
    o = "dir_both_rep"
    out["R1"] = dict(repeats=arms[o]["repeats"], unedited=arms["none"]["repeats"], cut_by_third=arms[o]["repeats"] <= arms["none"]["repeats"] * 2 / 3, diff=ci(rep[o], rep["none"]))
    out["R2"] = ci(ok[o], ok["none"])
    out["R3"] = ci(ok["doubt"], ok[o])
    out["R2_finished_right"] = ci(fin[o], fin["none"])
if "dir_both_rep" in R and "dir_neurospeak" in R:
    out["R4"] = ci(ok["dir_neurospeak"], ok["dir_both_rep"])
    out["raw_minus_unedited"] = ci(ok["dir_neurospeak"], ok["none"])
out["R5"] = {a: dict(bad=arms[a]["bad"], unedited_bad=arms["none"]["bad"], ok=arms[a]["bad"] <= arms["none"]["bad"] + 2) for a in done}
out["pass"] = {k: v for k, v in dict(R1=out["R1"]["cut_by_third"] if "R1" in out else None, R3=out["R3"]["low"] > 0 if "R3" in out else None,
                                     R5=all(v["ok"] for v in out["R5"].values())).items() if v is not None}
json.dump(out, open(P + "reporth_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
print(json.dumps({a: arms[a] for a in done}, indent=1))
