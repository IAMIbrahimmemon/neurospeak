"""Judges bars D1-D5 in PLAN_DOSECTL.md; self-doubt and unedited arms are study 20's traces on the same problems."""
import json, os, re, sys
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", "whyfix"]
from behaviors import DOUBT
from gsm_score import ci
from sensitivity_replicate_rules import right

P = "runs/research/"
SEEDS = (101, 102, 103, 104)
files = {"none": P + "layers_none_traces.jsonl", "doubt": P + "layers_dir_layer_19_traces.jsonl",
         **{f"ctl{s}": P + f"dosectl_dir_dosectl_{s}_at19_traces.jsonl" for s in SEEDS}}
R = {a: [json.loads(x) for x in open(f)] for a, f in files.items() if os.path.exists(f)}
R = {a: v for a, v in R.items() if len(v) == 64}
assert all([r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
for a in R:
    if a.startswith("ctl"):
        assert all(r["run"]["layer_used"] == 19 and r["run"]["arm"] == f"dir_dosectl_{a[3:]}" for r in R[a]), a
ctls = [a for a in R if a.startswith("ctl")]
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
out = dict(arms=arms, controls=ctls)
out["D1"] = {a: ci(ok["doubt"], ok[a]) for a in ctls}
out["D5"] = {a: dict(bad=arms[a]["bad"], unedited_bad=arms["none"]["bad"]) for a in ctls}
line = doubt["none"] * 2 / 3
out["D4"] = dict(line=round(line, 3), doubt_below=doubt["doubt"] <= line, controls_above={a: doubt[a] > line for a in ctls})
if len(ctls) == len(SEEDS):
    pool = lambda s: [mean([s[a][i] for a in ctls]) for i in range(64)]
    out["D2"] = ci(ok["doubt"], pool(ok))
    out["D3"] = ci(pool(ok), ok["none"])
    out["D2_nonempty"] = ci(nonempty["doubt"], pool(nonempty))
    out["D2_finished_right"] = ci(fin["doubt"], pool(fin))
    out["pass"] = dict(D1=all(c["low"] > 0 for c in out["D1"].values()), D2=out["D2"]["low"] > 0,
                       D3="holds" if out["D3"]["low"] <= 0 else "fails, controls help",
                       D4=out["D4"]["doubt_below"] and all(out["D4"]["controls_above"].values()))
json.dump(out, open(P + "dosectl_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
print(json.dumps(arms, indent=1))
