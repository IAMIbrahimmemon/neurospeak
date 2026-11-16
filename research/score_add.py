"""Judges bars A1-A6 in PLAN_ADD.md."""
import json, sys
from neurospeak.investigate import collapsed
sys.path[:0] = ["investigate", "whyfix", "research"]
from behaviors import DOUBT
from gsm_score import boxed, ci
from loop_onset import HIGH, windows
from sensitivity_replicate_rules import right

A = ("none", "doubt_low", "doubt_high", "random_low", "random_high", "rep_high")
R = {a: [json.loads(l) for l in open(f"runs/research/add_{a}_traces.jsonl")] for a in A}
assert all(len(v) == 48 and [(r["id"], r["prompt"]) for r in v] == [(r["id"], r["prompt"]) for r in R["none"]] for v in R.values())
per100 = lambda t: 100 * len(DOUBT.findall(t)) / max(len(t.split()), 1)
mean = lambda v: sum(v) / len(v)
whole = {a: [per100(r["completion"]) for r in v] for a, v in R.items()}
doubt = {a: [per100(" ".join(r["completion"].split()[:500])) for r in v] for a, v in R.items()}
loop = {a: [100 * (not collapsed(r["completion"]) and any(x >= HIGH for x in windows(r["completion"].split()))) for r in v] for a, v in R.items()}
G = json.load(open("runs/research/add_gains.json"))["arms"]
arms = {a: dict(doubt_first_500=mean(doubt[a]), doubt_whole_run=mean(whole[a]), loops_of_48=sum(x > 0 for x in loop[a]), correct_of_48=sum(right(r) for r in R[a]),
                boxed_of_48=sum(boxed(r["completion"]) is not None for r in R[a]), finished_of_48=sum(not r["cut"] for r in R[a]), collapsed_of_48=sum(collapsed(r["completion"]) for r in R[a]),
                amount=G.get(a, {}).get("amount"), dose=G.get(a, {}).get("dose")) for a in A}
out = dict(arms=arms, A1=ci(doubt["doubt_high"], doubt["random_high"]), A2=ci(doubt["doubt_low"], doubt["random_low"]), A3=ci(doubt["doubt_high"], doubt["doubt_low"]),
           A4=ci(doubt["doubt_high"], doubt["rep_high"]), A6=ci(loop["rep_high"], loop["random_high"]))
ok = lambda *names: all(arms[n]["collapsed_of_48"] <= arms["none"]["collapsed_of_48"] + 2 for n in names)
bar = lambda k, floor: out[k]["diff"] >= floor and out[k]["low"] > 0
uses = dict(A1=("doubt_high", "random_high"), A2=("doubt_low", "random_low"), A3=("doubt_high", "doubt_low"), A4=("doubt_high", "rep_high"), A6=("rep_high", "random_high"))
floors = dict(A1=1.0, A2=0.5, A3=1e-9, A4=0.5, A6=20)
out["bars"] = {k: bar(k, floors[k]) if ok(*uses[k]) else "void" for k in uses}
out["bars"]["A5"] = ok("doubt_low", "doubt_high", "random_low", "random_high", "rep_high")
json.dump(out, open("runs/research/add_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
