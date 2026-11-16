"""Judges bars E1-E5 in PLAN_FINISH.md, scoring each run at 9,000 tokens and as if cut at 3,000."""
import json, sys
from transformers import AutoTokenizer
from neurospeak.investigate import collapsed
sys.path[:0] = ["whyfix"]
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
A = ("rep", "doubt", "both")
R = {a: [json.loads(l) for l in open(f"runs/research/finish_{a}_traces.jsonl")] for a in A}
assert all(len(v) == 64 and [r["id"] for r in v] == [r["id"] for r in R["doubt"]] for v in R.values())
cut = lambda r: {**r, "completion": tok.decode(tok(r["completion"], add_special_tokens=False).input_ids[:3000])}
full = {a: [100 * right(r) for r in v] for a, v in R.items()}
short = {a: [100 * right(cut(r)) for r in v] for a, v in R.items()}
fin3 = {a: [100 * (r["n_tokens"] < 3000) for r in v] for a, v in R.items()}
broken = lambda r: collapsed(r["completion"]) or len(set(r["completion"].split()[:200])) < 0.2 * min(200, max(len(r["completion"].split()), 1))
arms = {a: dict(correct_3000=sum(x > 0 for x in short[a]), correct_9000=sum(x > 0 for x in full[a]), finished_3000=sum(x > 0 for x in fin3[a]), finished_9000=sum(not r["cut"] for r in R[a]),
                boxed_9000=sum(boxed(r["completion"]) is not None for r in R[a]), broken=sum(broken(r) for r in R[a]), correct_among_finished=sum(right(r) for r in R[a] if not r["cut"]), tokens=sum(r["n_tokens"] for r in R[a]) / 64) for a in A}
out = dict(arms=arms, E1=ci(fin3["both"], fin3["doubt"]), E2=ci(full["both"], full["doubt"]), E3=ci(full["both"], short["both"]), E6=ci(fin3["rep"], fin3["doubt"]), E7=ci(full["both"], full["rep"]), tie_3000=ci(short["both"], short["doubt"]), doubt_growth=ci(full["doubt"], short["doubt"]))
out["bars"] = dict(E1=out["E1"]["diff"] >= 10 and out["E1"]["low"] > 0, E2=out["E2"]["diff"] >= 10 and out["E2"]["low"] > 0, E3=out["E3"]["diff"] >= 10 and out["E3"]["low"] > 0, E4=arms["both"]["finished_9000"] >= 32, E5=arms["both"]["broken"] <= arms["doubt"]["broken"] + 2, E6=out["E6"]["diff"] >= 10 and out["E6"]["low"] > 0, E7=out["E7"]["diff"] >= 10 and out["E7"]["low"] > 0)
json.dump(out, open("runs/research/finish_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
