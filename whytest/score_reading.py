"""Scores R1 and R2.  python3 whytest/score_reading.py"""
import collections, glob, json, random, statistics as st

OUT = "runs/whytest"
key = json.load(open(f"{OUT}/read_key.json"))
picks = {}
for p in glob.glob(f"{OUT}/read_scores_*.json"):
    for row in json.load(open(p)):
        picks[row["trial"]] = row["pick"]
missing = [t for t in key if t not in picks]
print("missing", len(missing))


def ci(xs, n=4000, seed=2):
    r = random.Random(seed)
    m = sorted(sum(r.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return round(st.mean(xs), 3), round(m[int(.025 * n)], 3), round(m[int(.975 * n)], 3)


out = {}
by = collections.defaultdict(dict)
for t, k in key.items():
    if t in picks:
        by[(k["test"], k["arm"])][(k["chunk"], k["base"])] = (float(picks[t] == k["right"]), picks[t])
for (test, arm), rows in sorted(by.items()):
    xs = [v[0] for v in rows.values()]
    out[f"{test}_{arm}"] = dict(n=len(xs), acc=ci(xs), said_A=round(st.mean(v[1] == "A" for v in rows.values()), 3))
diff = lambda a, b: [by[("R2", a)][d][0] - by[("R2", b)][d][0] for d in by[("R2", a)] if d in by[("R2", b)]]
out["R2_real_minus_none"] = ci(diff("real", "none"))
out["R2_other_minus_none"] = ci(diff("other", "none"))
out["R2_real_minus_other"] = ci(diff("real", "other"))
json.dump(out, open(f"{OUT}/reading_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
