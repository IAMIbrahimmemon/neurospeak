"""Judges bars F1-F5 in PLAN_FACTOR.md (study 30). Unedited and d-removed arms are study 20's traces on the same problems."""
import json, os, re, sys
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", "whyfix", "research"]
from behaviors import DOUBT
from gsm_score import ci
from sensitivity_replicate_rules import right

P = "runs/research/"
A, B = "dir_factor_doubt_only", "dir_factor_conclude_only"
files = {"none": P + "layers_none_traces.jsonl", "doubt": P + "layers_dir_layer_19_traces.jsonl", **{a: P + f"otherdir_{a}_traces.jsonl" for a in (A, B)}}
R = {a: [json.loads(x) for x in open(f)] for a, f in files.items() if os.path.exists(f)}
R = {a: v for a, v in R.items() if len(v) == 64}
assert all([r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
for a in (A, B):
    if a in R:
        assert all(r["run"]["arm"] == a and r["run"]["layer_used"] == 19 and r["run"]["max"] == 3000 and r["run"]["seed"] == 20
                   and not r["run"]["add"] and r["run"]["gain"] == 1.0 and r["run"]["mode"] == "full" and not r["run"].get("ban") for r in R[a]), a
mean = lambda v: sum(v) / len(v)
ok = {a: [100 * right(r) for r in v] for a, v in R.items()}
bad = lambda r: bool(re.search(r" {200,}", r["completion"])) or collapsed(r["completion"])
cutans = lambda r: r["cut"] and bool(re.search(r"=\s*\$?" + re.escape(r["answer"]) + r"\b", r["completion"]))
arms = {a: dict(correct=sum(x > 0 for x in ok[a]), finished=sum(not r["cut"] for r in v), closes_thinking=sum("</think>" in r["completion"] for r in v),
                doubt_per_100=round(mean([100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in v]), 3),
                i_will_per_1k=round(mean([1000 * len(re.findall(r"\bI will\b", r["completion"])) / max(r["n_tokens"], 1) for r in v]), 2),
                answer_but_cut=sum(cutans(r) for r in v), bad=sum(bad(r) for r in v), tokens=sum(r["n_tokens"] for r in v) // 64) for a, v in R.items()}
out = dict(arms=arms, done=[a for a in (A, B) if a in R])
if A in R:
    out["F1"] = ci(ok[A], ok["none"]); out["F4_doubt"] = ci(ok["doubt"], ok[A])
if B in R:
    out["F2"] = ci(ok[B], ok["none"]); out["F4_conclude"] = ci(ok["doubt"], ok[B])
if A in R and B in R:
    out["F3"] = ci(ok[A], ok[B])
out["F5"] = {a: dict(bad=arms[a]["bad"], unedited_bad=arms["none"]["bad"], ok=arms[a]["bad"] <= arms["none"]["bad"] + 2) for a in out["done"]}
out["pass"] = {k: out[k]["low"] > 0 for k in ("F1", "F2") if k in out}
json.dump(out, open(P + "factor_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
