"""Judges bars N1-N5 in PLAN_NOPEN.md, scoring each run at 9,000 tokens and as if cut at 3,000."""
import json, re, sys
from transformers import AutoTokenizer
from neurospeak.investigate import collapsed
sys.path[:0] = ["whyfix"]
from gsm_score import ci
from sensitivity_replicate_rules import right

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
A = ("doubt", "none", "none_pp", "doubt_pp", "astra_pp")
R = {a: [json.loads(l) for l in open(f"runs/research/head_{a}_traces.jsonl")] for a in A}
assert all(len(v) == 96 and [r["id"] for r in v] == [r["id"] for r in R["none"]] for v in R.values())
cut = lambda r: {**r, "completion": tok.decode(tok(r["completion"], add_special_tokens=False).input_ids[:3000])}
full = {a: [100 * right(r) for r in v] for a, v in R.items()}
short = {a: [100 * right(cut(r)) for r in v] for a, v in R.items()}
bad = lambda r: bool(re.search(r" {200,}", r["completion"])) or collapsed(r["completion"])
arms = {a: dict(correct_3000=sum(x > 0 for x in short[a]), correct_9000=sum(x > 0 for x in full[a]), finished_9000=sum(not r["cut"] for r in R[a]),
                right_when_finished=sum(right(r) and not r["cut"] for r in R[a]), bad=sum(bad(r) for r in R[a]), tokens=sum(r["n_tokens"] for r in R[a]) // 96) for a in A}
pen_doubt = [p - q for p, q in zip(full["doubt_pp"], full["doubt"])]
pen_none = [p - q for p, q in zip(full["none_pp"], full["none"])]
out = dict(n=96, arms=arms, N1=ci(full["doubt"], full["none"]), N2=ci(full["doubt_pp"], full["doubt"]), N3=ci(pen_doubt, pen_none), N4=ci(full["doubt"], full["astra_pp"]))
side = lambda c: "positive" if c["low"] > 0 else "negative" if c["high"] < 0 else "none shown"
out["bars"] = dict(N1=out["N1"]["low"] > 0, N2=out["N2"]["low"] > 0, N3=side(out["N3"]), N4=side(out["N4"]), N5=arms["doubt"]["bad"] <= arms["none"]["bad"] + 2)
json.dump(out, open("runs/research/nopen_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
