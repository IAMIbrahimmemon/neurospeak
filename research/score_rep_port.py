"""Judges the gate and bars Q1-Q5 in PLAN_REP_PORT.md."""
import json, sys
from neurospeak import loopiness
from neurospeak.investigate import collapsed
sys.path[:0] = ["whyfix", "research"]
from gsm_score import boxed, ci
from loop_onset import HIGH, windows
from sensitivity_replicate_rules import right
A = ("none", "rep", "rep_matched")
R = {a: [json.loads(l) for l in open(f"runs/research/rport_{a}_traces.jsonl")] for a in A}
assert all(len(v) == 64 and [r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
broken = lambda r: collapsed(r["completion"]) or len(set(r["completion"].split()[:200])) < 0.2 * min(200, max(len(r["completion"].split()), 1))
rep = {a: [100 * loopiness(r["completion"]) for r in v] for a, v in R.items()}
loop = {a: [100 * any(x >= HIGH for x in windows(r["completion"].split())) for r in v] for a, v in R.items()}
ok = {a: [100 * right(r) for r in v] for a, v in R.items()}
mean = lambda v: sum(v) / len(v)
arms = {a: dict(repeats=mean(rep[a]), loops_of_64=sum(x > 0 for x in loop[a]), correct_of_64=sum(x > 0 for x in ok[a]), boxed_of_64=sum(boxed(r["completion"]) is not None for r in R[a]), broken_of_64=sum(broken(r) for r in R[a])) for a in A}
out = dict(arms=arms, rep_vs_matched=ci(rep["rep"], rep["rep_matched"]), loop_vs_matched=ci(loop["rep"], loop["rep_matched"]), correct_vs_none=ci(ok["rep"], ok["none"]), matched_vs_none=ci(rep["rep_matched"], rep["none"]), rep_vs_none=ci(rep["rep"], rep["none"]))
out["gate"] = arms["none"]["repeats"] >= 20 and arms["none"]["loops_of_64"] >= 16
M, V = out["matched_vs_none"], out["rep_vs_none"]
out["bars"] = dict(Q1=arms["rep"]["repeats"] <= arms["none"]["repeats"] * 2 / 3 and out["rep_vs_matched"]["high"] < 0,
                   Q2=out["loop_vs_matched"]["diff"] <= -20 and out["loop_vs_matched"]["high"] < 0,
                   Q3=out["correct_vs_none"]["diff"] >= 5 and out["correct_vs_none"]["low"] > 0,
                   Q4=(M["low"] <= 0 <= M["high"]) or abs(M["diff"]) < abs(V["diff"]) / 3,
                   Q5=arms["rep"]["broken_of_64"] <= arms["none"]["broken_of_64"] + 2)
json.dump(out, open("runs/research/rport_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))

# Amendment: text before each run's first box
import re
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
def first_real(r):      # a box with no digit echoes the instruction, as in "\\boxed{answer}"
    return next((m for m in BOX.finditer(r["completion"]) if re.search(r"\d", m.group(1))), None)
def before(r):
    m = first_real(r)
    return r["completion"][:m.start()] if m else r["completion"]
def first_box_right(r):
    m = first_real(r)
    return bool(m) and right({"completion": m.group(0), "answer": r["answer"]})
pre = {a: [100 * loopiness(before(r)) for r in v] for a, v in R.items()}
fb = {a: [100 * first_box_right(r) for r in v] for a, v in R.items()}
words = {a: sorted(len(before(r).split()) for r in v)[32] for a, v in R.items()}
am = dict(repeats_before_box={a: mean(pre[a]) for a in A}, first_box_correct={a: sum(x > 0 for x in fb[a]) for a in A}, median_words_before_box=words,
          gate=mean(pre["none"]) >= 20, pre_rep_vs_matched=ci(pre["rep"], pre["rep_matched"]), first_box_rep_vs_none=ci(fb["rep"], fb["none"]), first_box_rep_vs_matched=ci(fb["rep"], fb["rep_matched"]))
out["amended"] = am
json.dump(out, open("runs/research/rport_results.json", "w"), indent=1)
print(json.dumps(am, indent=1))
