"""Judges bars N1-N4 in PLAN_OTHERDIR.md (study 25). Unedited and self-doubt arms are study 20's traces on the same problems."""
import json, os, re, sys
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", "whyfix"]
from behaviors import DOUBT
from gsm_score import ci
from sensitivity_replicate_rules import right

P = "runs/research/"
OTHER = ["dir_bullets", "dir_chinese", "dir_restating"]
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
                tokens=sum(r["n_tokens"] for r in R[a]) // 64) for a in R}
done = [a for a in OTHER if a in R]
line = doubt["none"] * 2 / 3
out = dict(arms=arms, done=done,
           N1={a: ci(ok["doubt"], ok[a]) for a in done},
           N2={a: ci(ok[a], ok["none"]) for a in done},
           N2_finished_right={a: ci(fin[a], fin["none"]) for a in done},
           N3=dict(line=round(line, 3), doubt_below=doubt["doubt"] <= line, others_above={a: doubt[a] > line for a in done}),
           N4={a: dict(bad=arms[a]["bad"], unedited_bad=arms["none"]["bad"], ok=arms[a]["bad"] <= arms["none"]["bad"] + 2) for a in done})
out["pass"] = dict(N1={a: out["N1"][a]["low"] > 0 for a in done}, N2_helps={a: out["N2"][a]["low"] > 0 for a in done},
                   N3=out["N3"]["doubt_below"] and all(out["N3"]["others_above"].values()), N4=all(v["ok"] for v in out["N4"].values()))
json.dump(out, open(P + "otherdir_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
print(json.dumps({a: arms[a] for a in done}, indent=1))
