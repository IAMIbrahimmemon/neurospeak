"""Pools studies 13 and 15 on first-box correct and checks whether broken runs drive study 15. No bars; see PLAN_REP_PORT2.md."""
import json, re, sys
from neurospeak.investigate import collapsed
sys.path[:0] = ["whyfix"]
from gsm_score import ci
from sensitivity_replicate_rules import right
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
real = lambda r: next((m for m in BOX.finditer(r["completion"]) if re.search(r"\d", m.group(1))), None)
fb = lambda r: 100 * (lambda m: bool(m) and right({"completion": m.group(0), "answer": r["answer"]}))(real(r))
broken = lambda r: collapsed(r["completion"]) or len(set(r["completion"].split()[:200])) < 0.2 * min(200, max(len(r["completion"].split()), 1))
load = lambda s, a: [json.loads(l) for l in open(f"runs/research/{s}_{a}_traces.jsonl")]
A = ("none", "rep", "rep_matched")
R15 = {a: load("rport2", a) for a in A}
R13 = {a: load("rport", a) for a in A}
P = {a: R13[a] + R15[a] for a in A}
s = {a: [fb(r) for r in P[a]] for a in A}
pre = lambda r: dict(completion=r["completion"][:real(r).start()] if real(r) else r["completion"])
un = {a: [not broken(pre(r)) for r in R15[a]] for a in A}
keep = [i for i in range(128) if un["none"][i] and un["rep"][i] and un["rep_matched"][i]]
f15 = {a: [fb(R15[a][i]) for i in keep] for a in A}
out = dict(pooled_n=len(s["none"]), pooled_counts={a: sum(x > 0 for x in s[a]) for a in A}, pooled_rep_none=ci(s["rep"], s["none"]), pooled_rep_matched=ci(s["rep"], s["rep_matched"]),
           pooled_matched_none=ci(s["rep_matched"], s["none"]), s15_none_broken_before_box_n=len(keep), s15_none_broken_before_box_counts={a: sum(x > 0 for x in f15[a]) for a in A},
           s15_none_broken_before_box_rep_none=ci(f15["rep"], f15["none"]), s15_none_broken_before_box_rep_matched=ci(f15["rep"], f15["rep_matched"]),
           real_box={a: sum(real(r) is not None for r in R15[a]) for a in A}, right_when_real_box={a: round(sum(x > 0 for x in map(fb, R15[a])) / max(sum(real(r) is not None for r in R15[a]), 1), 3) for a in A},
           correct_first_box_broken_whole_run={a: sum(fb(r) > 0 and broken(r) for r in R15[a]) for a in A})
json.dump(out, open("runs/research/rport2_audit.json", "w"), indent=1)
print(json.dumps(out, indent=1))
