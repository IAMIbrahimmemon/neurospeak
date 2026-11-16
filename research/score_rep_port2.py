"""Judges bars R1-R4 in PLAN_REP_PORT2.md."""
import json, re, sys
from neurospeak import loopiness
from neurospeak.investigate import collapsed
sys.path[:0] = ["whyfix"]
from gsm_score import ci
from sensitivity_replicate_rules import right
A = ("none", "rep", "rep_matched")
R = {a: [json.loads(l) for l in open(f"runs/research/rport2_{a}_traces.jsonl")] for a in A}
assert all(len(v) == 128 and [r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
def first_real(r):      # a box with no digit echoes the instruction, as in "\\boxed{answer}"
    return next((m for m in BOX.finditer(r["completion"]) if re.search(r"\d", m.group(1))), None)
def before(r):
    m = first_real(r)
    return r["completion"][:m.start()] if m else r["completion"]
def first(r):
    m = first_real(r)
    return bool(m) and right({"completion": m.group(0), "answer": r["answer"]})
broken = lambda r: collapsed(r["completion"]) or len(set(r["completion"].split()[:200])) < 0.2 * min(200, max(len(r["completion"].split()), 1))
fb = {a: [100 * first(r) for r in v] for a, v in R.items()}
pre = {a: [100 * loopiness(before(r)) for r in v] for a, v in R.items()}
arms = {a: dict(first_box_correct=sum(x > 0 for x in fb[a]), last_box_correct=sum(right(r) for r in R[a]), any_box=sum(bool(BOX.search(r["completion"])) for r in R[a]), repeats_before_box=sum(pre[a]) / 128,
                median_words_before=sorted(len(before(r).split()) for r in R[a])[64], broken=sum(broken(dict(completion=before(r))) for r in R[a]), broken_whole_run=sum(broken(r) for r in R[a])) for a in A}
out = dict(arms=arms, R1=ci(fb["rep"], fb["rep_matched"]), R2=ci(fb["rep"], fb["none"]), R3=ci(pre["rep"], pre["rep_matched"]), matched_vs_none=ci(fb["rep_matched"], fb["none"]))
out["bars"] = dict(R1=out["R1"]["diff"] >= 5 and out["R1"]["low"] > 0, R2=out["R2"]["diff"] >= 5 and out["R2"]["low"] > 0, R3=out["R3"]["diff"] <= -5 and out["R3"]["high"] < 0, R4=arms["rep"]["broken"] <= arms["none"]["broken"] + 3)
json.dump(out, open("runs/research/rport2_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
