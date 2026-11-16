"""Judges bars G1-G5 in PLAN_AXIS.md, with the escape rule from study 7."""
import json
from neurospeak.investigate import collapsed

src = open("research/score_rescue.py").read()
exec(src[src.index("SIZE, STEP"):src.index("R = {a:")])
import sys
sys.path[:0] = ["whyfix"]
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right

load = lambda p: [json.loads(l) for l in open(p)]
R = dict(none=load("runs/research/rescue_none_traces.jsonl"), repetition=load("runs/research/rescue_dir_neurospeak_traces.jsonl"), only_100=load("runs/research/dose_rep_only_traces.jsonl"))
NEW = ("rep_25", "rep_50", "rep_75", "only_109", "both")
for a in NEW:
    R[a] = load(f"runs/research/axis_{a}_traces.jsonl")
assert all(len(v) == 56 and [(r["id"], r["prefix"]) for r in v] == [(r["id"], r["prefix"]) for r in R["none"]] for v in R.values())
esc = {a: [100 * escaped(r) for r in v] for a, v in R.items()}
arms = {a: dict(escaped_of_56=sum(x > 0 for x in esc[a]), finished_of_56=sum(not r["cut"] for r in R[a]), collapsed_of_56=sum(collapsed(r["completion"]) for r in R[a]),
                boxed_of_56=sum(boxed(r["completion"]) is not None for r in R[a]), correct_of_56=sum(right(r) for r in R[a])) for a in R}
out = dict(arms=arms, G1=ci(esc["rep_75"], esc["rep_25"]), G2=ci(esc["rep_25"], esc["none"]), G3=ci(esc["repetition"], esc["only_109"]), G4=ci(esc["both"], esc["only_109"]))
ok = lambda *n: all(arms[a]["collapsed_of_56"] <= arms["none"]["collapsed_of_56"] + 2 for a in n)
bar = lambda k, f, *n: (out[k]["diff"] >= f and out[k]["low"] > 0) if ok(*n) else "void"
out["bars"] = dict(G1=bar("G1", 20, "rep_75", "rep_25"), G2=bar("G2", 10, "rep_25"), G3=bar("G3", 15, "only_109"), G4=bar("G4", 15, "both", "only_109"), G5=ok(*NEW))
json.dump(out, open("runs/research/axis_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
