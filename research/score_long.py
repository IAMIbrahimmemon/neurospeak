"""Judges bars L1-L5 in PLAN_LONG_CAP.md, scoring each run at the full cap and as if cut at 3,000 tokens."""
import json, sys
from transformers import AutoTokenizer
sys.path.insert(0, "whyfix")
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
N, S, M = "none", "dir_self_doubt", "dir_self_doubt_matched"
R = {a: [json.loads(l) for l in open(f"runs/research/long_{a}_traces.jsonl")] for a in (N, S, M)}
assert all(len(v) == 32 and [r["prompt"] for r in v] == [r["prompt"] for r in R[N]] for v in R.values())
cut = lambda r: {**r, "completion": tok.decode(tok(r["completion"], add_special_tokens=False).input_ids[:3000])}
full = {a: [100 * right(r) for r in v] for a, v in R.items()}
short = {a: [100 * right(cut(r)) for r in v] for a, v in R.items()}
count = lambda v: sum(x > 0 for x in v)
arms = {a: dict(correct_3000=count(short[a]), correct_10000=count(full[a]), finished=sum(not r["cut"] for r in R[a]),
                boxed=sum(boxed(r["completion"]) is not None for r in R[a]), tokens=sum(r["n_tokens"] for r in R[a]) / 32) for a in R}
both = [(right(x), right(y)) for x, y in zip(R[N], R[S]) if boxed(x["completion"]) is not None and boxed(y["completion"]) is not None]
out = dict(arms=arms, L1=ci(short[S], short[N]), L2=ci(full[S], full[N]), L3=ci(full[S], full[M]), L4=ci(full[N], short[N]),
           both_boxed=dict(n=len(both), unedited_right=sum(a for a, _ in both), self_doubt_right=sum(b for _, b in both)))
out["bars"] = dict(L1=out["L1"]["low"] > 0, L2=out["L2"]["diff"] >= 15 and out["L2"]["low"] > 0, L3=out["L3"]["low"] > 0, L4=out["L4"]["low"] > 0, L5=arms[N]["finished"] >= 16)
json.dump(out, open("runs/research/long_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
