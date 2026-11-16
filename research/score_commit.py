"""Judges bars C1-C4 in PLAN_COMMIT.md, CPU only."""
import json, re, sys
sys.path[:0] = ["whyfix"]
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right, value

NUM = re.compile(r"(?<![\d.,])-?\d[\d,]*(?:\.\d+)?")
vals = lambda t: {float(m.group().replace(",", "").rstrip(".")) for m in NUM.finditer(t)}
N, S, M = "none", "dir_self_doubt", "dir_self_doubt_matched"
R = {a: [json.loads(l) for f in ("gsm64", "gsm64b") for l in open(f"runs/whyfix/{f}_{a}_traces.jsonl")] for a in (N, S, M)}
assert all(len(v) == 128 and [r["id"] for r in v] == [r["id"] for r in R[N]] for v in R.values())
keep = [i for i, r in enumerate(R[N]) if value(r["answer"]) is not None and value(r["answer"]) not in vals(r["prompt"])]
reach = {a: [100 * (value(R[a][i]["answer"]) in vals(R[a][i]["completion"])) for i in keep] for a in R}
ok = {a: [100 * right(R[a][i]) for i in keep] for a in R}
ans = {a: [100 * (boxed(R[a][i]["completion"]) is not None) for i in keep] for a in R}
pct = lambda v: sum(v) / len(v)
both_s = [j for j in range(len(keep)) if reach[N][j] and reach[S][j]]
both_m = [j for j in range(len(keep)) if reach[N][j] and reach[M][j]]
pick = lambda v, idx: [v[j] for j in idx]
out = dict(problems=len(keep), left_out=128 - len(keep),
           arms={a: dict(reached=sum(x > 0 for x in reach[a]), correct=sum(x > 0 for x in ok[a]), answered=sum(x > 0 for x in ans[a]),
                         correct_given_reached=sum(1 for r, o in zip(reach[a], ok[a]) if r and o)) for a in R},
           reach_diff=ci(reach[S], reach[N]), reach_diff_matched=ci(reach[M], reach[N]),
           C3=dict(n=len(both_s), **ci(pick(ok[S], both_s), pick(ok[N], both_s))),
           C4=dict(n=len(both_m), **ci(pick(ok[M], both_m), pick(ok[N], both_m))))
A = out["arms"]
out["bars"] = dict(C1=A[N]["reached"] >= 2 * A[N]["correct"], C2=out["reach_diff"]["diff"] <= 10,
                   C3=out["C3"]["diff"] >= 20 and out["C3"]["low"] > 0, C4=out["C4"]["low"] <= 0 <= out["C4"]["high"])
json.dump(out, open("runs/research/commit_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
