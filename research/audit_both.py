"""Checks study 14's counts by a second route and describes runs that never finish. See PLAN_BOTH.md."""
import json, re, statistics as st, sys
sys.path[:0] = ["whyfix", "research"]
from gsm_score import boxed
from sensitivity_replicate_rules import right
A = ("none", "doubt", "rep", "both", "random2")
R = {a: [json.loads(l) for l in open(f"runs/research/both_{a}_traces.jsonl")] for a in A}
out = {}
BOX = re.compile(r"\\boxed\{[^}\s]")
for a, v in R.items():
    first = [BOX.search(r["completion"]) for r in v]
    words = [len(r["completion"][:m.start()].split()) for r, m in zip(v, first) if m]
    rv = [r for r in v if not boxed(r["completion"])]
    has = sum(re.search(r"(?<![\d.])" + re.escape(str(r["answer"]).replace(",", "")) + r"(?![\d])", r["completion"].replace(",", "")) is not None for r in rv)
    out[a] = dict(correct=sum(right(r) for r in v), boxed=len(words), median_words_to_box=st.median(words) if words else None,
                  ok_when_boxed=round(sum(right(r) for r in v if boxed(r["completion"])) / max(len(words), 1), 3), unboxed=len(rv), unboxed_has_value=has,
                  cut=sum(r["cut"] for r in v), median_words=st.median(len(r["completion"].split()) for r in v), thinking_closed=sum("</think>" in r["completion"] for r in v))
ok = {a: {r["id"] for r in v if right(r)} for a, v in R.items()}
out["overlap"] = dict(both_doubt=len(ok["both"] & ok["doubt"]), both_rep=len(ok["both"] & ok["rep"]), doubt_rep=len(ok["doubt"] & ok["rep"]), any_of_three=len(ok["both"] | ok["doubt"] | ok["rep"]))
json.dump(out, open("runs/research/both_audit.json", "w"), indent=1)
print(json.dumps(out, indent=1))
