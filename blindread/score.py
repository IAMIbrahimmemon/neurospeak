"""Step 7 — score. Top-1 accuracy (chance 20%), mean rank (chance 3.0), Wilson CI, exact binomial p vs 0.2."""
import glob, json, math
key = json.load(open("trials_key.json"))
ans = {a["trial"]: a for f in sorted(glob.glob("scores_*.json")) for a in json.load(open(f))}
assert len(ans) == 180, len(ans)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def p_above(k, n, p=0.2):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

rows = []
for tid, k in key.items():
    r = ans[tid]["ranking"]
    rows.append({**k, "hit": r[0] == k["answer"], "rank": r.index(k["answer"]) + 1})
print(f"{'condition':10s} {'band':7s} {'n':>3s} {'top-1':>6s}  {'95% CI':>13s}  {'mean rank':>9s}  p(vs 20%)")
for cond in ("blind", "ceiling", "shuffled"):
    for band in ("early", "middle", "late", "ALL"):
        s = [x for x in rows if x["cond"] == cond and (band == "ALL" or x["band"] == band)]
        k, n = sum(x["hit"] for x in s), len(s)
        lo, hi = wilson(k, n)
        print(f"{cond:10s} {band:7s} {n:3d} {k / n:6.0%}  [{lo:4.0%}, {hi:4.0%}]  {sum(x['rank'] for x in s) / n:9.2f}  {p_above(k, n):.4f}")
    print()
b = [x for x in rows if x["cond"] == "blind"]
for c in sorted({x["reader_confidence"] for x in b}):
    s = [x for x in b if x["reader_confidence"] == c]
    print(f"blind, reader confidence {c}: {sum(x['hit'] for x in s)}/{len(s)}")
json.dump(rows, open("results_rows.json", "w"), indent=1)
