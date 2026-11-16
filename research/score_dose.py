"""Judges bars F1-F5 in PLAN_DOSE.md, with the escape rule from study 7."""
import json
from neurospeak.investigate import collapsed

src = open("research/score_rescue.py").read()
exec(src[src.index("SIZE, STEP"):src.index("R = {a:")].replace('N, S, P, M = "none", "dir_self_doubt", "dir_neurospeak", "dir_self_doubt_matched"', ""))
import sys
sys.path[:0] = ["whyfix"]
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right

load = lambda p: [json.loads(l) for l in open(p)]
R = dict(none=load("runs/research/rescue_none_traces.jsonl"), repetition=load("runs/research/rescue_dir_neurospeak_traces.jsonl"))
G = json.load(open("runs/research/dose_gains.json"))
for a in ("matched_mean", "doubt_dose", "random_dose", "rep_only", "doubt_only"):
    R[a] = load(f"runs/research/dose_{a}_traces.jsonl")
assert all(len(v) == 56 and [(r["id"], r["prefix"]) for r in v] == [(r["id"], r["prefix"]) for r in R["none"]] for v in R.values())
esc = {a: [100 * escaped(r) for r in v] for a, v in R.items()}
arms = {a: dict(escaped_of_56=sum(x > 0 for x in esc[a]), finished_of_56=sum(not r["cut"] for r in R[a]), collapsed_of_56=sum(collapsed(r["completion"]) for r in R[a]),
                boxed_of_56=sum(boxed(r["completion"]) is not None for r in R[a]), correct_of_56=sum(right(r) for r in R[a]),
                gain=G["arms"].get(a, {}).get("gain"), dose=G["arms"].get(a, {}).get("dose")) for a in R}
C = lambda a, b: ci(esc[a], esc[b])
out = dict(target_dose=G["target"], arms=arms, F1=C("repetition", "matched_mean"), F2=C("repetition", "random_dose"), F3=C("doubt_dose", "random_dose"), F4=C("rep_only", "random_dose"), F5=C("doubt_only", "random_dose"))
reached = lambda a: G["arms"][a].get("reached", True)
out["bars"] = {k: out[k]["diff"] >= 20 and out[k]["low"] > 0 for k in ("F1", "F2", "F3", "F4", "F5")}
out["void"] = [a for a in G["arms"] if not reached(a)]
json.dump(out, open("runs/research/dose_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
