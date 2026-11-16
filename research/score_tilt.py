"""Judges bars T1-T6 in PLAN_TILT.md, with the escape rule from study 7."""
import json, sys
from neurospeak.investigate import collapsed

src = open("research/score_rescue.py").read()
exec(src[src.index("SIZE, STEP"):src.index("R = {a:")])
sys.path[:0] = ["whyfix"]
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right

load = lambda p: [json.loads(l) for l in open(p)]
ARMS = ("none", "only", "only_eq", "tilt", "both", "perp_eq", "rand_eq")
R = {a: load(f"runs/research/tilt_{a}_traces.jsonl") for a in ARMS}
PX = load("runs/research/tilt_prefixes.jsonl")
N = len(PX)
P = {r["id"]: r["source"] for r in PX}
assert all(len(v) == N and [(r["id"], r["prefix"]) for r in v] == [(r["id"], r["prefix"]) for r in R["none"]] for v in R.values())
broken = lambda r: collapsed(r["completion"]) or len(set(r["completion"].split()[:200])) < 0.2 * min(200, max(len(r["completion"].split()), 1))
esc = {a: [100 * escaped(r) for r in v] for a, v in R.items()}
arms = {a: dict(escaped=sum(x > 0 for x in esc[a]), finished=sum(not r["cut"] for r in R[a]), broken=sum(broken(r) for r in R[a]), boxed=sum(boxed(r["completion"]) is not None for r in R[a]), correct=sum(right(r) for r in R[a])) for a in ARMS}
out = dict(n=N, arms=arms, T1=ci(esc["tilt"], esc["only"]), T2=ci(esc["tilt"], esc["only_eq"]), T3=ci(esc["both"], esc["tilt"]), T4=ci(esc["both"], esc["only"]), T5=ci(esc["tilt"], esc["perp_eq"]), T6=ci(esc["tilt"], esc["rand_eq"]))
out["by_source"] = {s: {a: sum(x > 0 for x, r in zip(esc[a], R[a]) if P[r["id"]] == s) for a in ARMS} for s in sorted(set(P.values()))}
ok = lambda *n: all(arms[a]["broken"] <= arms["none"]["broken"] + 3 for a in n)
gate = lambda v, *n: v if ok(*n) else "void"
out["bars"] = dict(
    T1=gate(out["T1"]["low"] > 0, "tilt", "only"),
    T2=gate(abs(out["T2"]["diff"]) <= 5 and out["T2"]["high"] < 15, "tilt", "only_eq"),
    T3=gate(out["T3"]["diff"] <= 5 and out["T3"]["high"] < 15, "both", "tilt"),
    T4=gate(out["T4"]["low"] > 0, "both", "only"),
    T5=gate(out["T5"]["diff"] >= 20 and out["T5"]["low"] > 0 and arms["perp_eq"]["escaped"] <= 8, "tilt", "perp_eq"),
    T6=gate(out["T6"]["diff"] >= 20 and out["T6"]["low"] > 0, "tilt", "rand_eq"),
    T7=ok(*ARMS[1:]))
json.dump(out, open("runs/research/tilt_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
