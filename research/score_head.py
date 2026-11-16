"""Judges bars H1-H5 and H8 in PLAN_HEAD.md, and reports H6 as an estimate, scoring each run at 9,000 tokens and as if cut at 3,000. H7 comes from the reader's file."""
import json, sys
from transformers import AutoTokenizer
from neurospeak.investigate import collapsed
from neurospeak import loopiness
sys.path[:0] = ["whyfix"]
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
A = ("none", "none_pp", "astra_pp", "doubt_pp", "astra_doubt_pp", "matched_pp")
R = {a: [json.loads(l) for l in open(f"runs/research/head_{a}_traces.jsonl")] for a in A}
assert all(len(v) == 96 and [r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
cut = lambda r: {**r, "completion": tok.decode(tok(r["completion"], add_special_tokens=False).input_ids[:3000])}
full = {a: [100 * right(r) for r in v] for a, v in R.items()}
short = {a: [100 * right(cut(r)) for r in v] for a, v in R.items()}
broken = lambda r: collapsed(r["completion"]) or len(set(r["completion"].split()[:200])) < 0.2 * min(200, max(len(r["completion"].split()), 1))
arms = {a: dict(correct_3000=sum(x > 0 for x in short[a]), correct_9000=sum(x > 0 for x in full[a]), finished_3000=sum(r["n_tokens"] < 3000 for r in R[a]), finished_9000=sum(not r["cut"] for r in R[a]),
                boxed_9000=sum(boxed(r["completion"]) is not None for r in R[a]), broken=sum(broken(r) for r in R[a]), loopy=sum(loopiness(r["completion"]) > 0.5 for r in R[a]), tokens=sum(r["n_tokens"] for r in R[a]) // len(R[a])) for a in A}
out = dict(n=96, arms=arms, H1=ci(full["astra_pp"], full["none_pp"]), H2=ci(full["astra_doubt_pp"], full["astra_pp"]), H3=ci(short["doubt_pp"], short["astra_pp"]), H4=ci(full["doubt_pp"], full["astra_pp"]), H5=ci(short["doubt_pp"], short["none_pp"]), H6=ci(full["none_pp"], full["none"]), H8=ci(short["doubt_pp"], short["matched_pp"]))
out["bars"] = dict(H1=out["H1"]["diff"] >= 10 and out["H1"]["low"] > 0, H2=out["H2"]["low"] > 0, H2_useful_size=out["H2"]["diff"] >= 10, H3=out["H3"]["low"] > 0,
                   H4="doubt" if out["H4"]["low"] > 0 else "astra" if out["H4"]["high"] < 0 else "no winner", H5=out["H5"]["diff"] >= 15 and out["H5"]["low"] > 0, H8=out["H8"]["low"] > 0)
json.dump(out, open("runs/research/head_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
