"""Judges bars B1-B5 in PLAN_BAN.md (study 28). Unedited and self-doubt arms are study 20's traces on the same problems."""
import json, os, re, sys
from neurospeak import loopiness
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", "whyfix", "research"]
from behaviors import DOUBT
from gsm_score import ci
from sensitivity_replicate_rules import right
from loop_onset import HIGH, windows

P = "runs/research/"
BAN = "13784,13428,11158,3655,77264,85152,84485,50821,32645,70546,3404"
files = {"none": P + "layers_none_traces.jsonl", "doubt": P + "layers_dir_layer_19_traces.jsonl", "ban": P + "ban_traces.jsonl"}
R = {a: [json.loads(x) for x in open(f)] for a, f in files.items()}
assert all(len(v) == 64 for v in R.values()), {a: len(v) for a, v in R.items()}
assert all([r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
assert all(r["run"]["arm"] == "none" and r["run"]["ban"] == BAN and r["run"]["max"] == 3000 and r["run"]["seed"] == 20 and not r["run"]["presence"] for r in R["ban"])
SUB = re.compile(r"let me (check|verify|confirm|re-?examine|re-?calculate)|double-check|re-?read|is there a trick|hold on|but let me", re.I)
mean = lambda v: sum(v) / len(v)
ok = {a: [100 * right(r) for r in v] for a, v in R.items()}
fin = {a: [100 * (right(r) and not r["cut"]) for r in v] for a, v in R.items()}
stuck = lambda r: "\\boxed{" not in r["completion"] and bool(re.search(r"=\s*\$?" + re.escape(r["answer"]) + r"\b", r["completion"]))
bad = lambda r: bool(re.search(r" {200,}", r["completion"])) or collapsed(r["completion"])
arms = {a: dict(correct=sum(x > 0 for x in ok[a]), finished=sum(not r["cut"] for r in v), finished_right=sum(x > 0 for x in fin[a]),
                doubt_per_100=round(mean([100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in v]), 3),
                substitutes_per_1k=round(mean([1000 * len(SUB.findall(r["completion"])) / max(r["n_tokens"], 1) for r in v]), 2),
                stuck=sum(stuck(r) for r in v), bad=sum(bad(r) for r in v), tokens=sum(r["n_tokens"] for r in v) // 64,
                repeats=round(mean([100 * loopiness(r["completion"]) for r in v]), 1),
                loops=sum(any(x >= HIGH for x in windows(r["completion"].split())) for r in v)) for a, v in R.items()}
out = dict(arms=arms,
           B1=dict(doubt_per_100=arms["ban"]["doubt_per_100"], ok=arms["ban"]["doubt_per_100"] < 0.5),
           B2=ci(ok["ban"], ok["none"]), B3=ci(ok["doubt"], ok["ban"]),
           B4={a: arms[a]["stuck"] for a in arms},
           B5=dict(bad=arms["ban"]["bad"], unedited_bad=arms["none"]["bad"], ok=arms["ban"]["bad"] <= arms["none"]["bad"] + 2),
           B2_finished_right=ci(fin["ban"], fin["none"]))
out["pass"] = dict(B1=out["B1"]["ok"], B3=out["B3"]["low"] > 0, B5=out["B5"]["ok"])
json.dump(out, open(P + "ban_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
