"""Scores R1 and R2.  python3 whytest/score_reading.py"""
import collections, glob, json, random, statistics as st

OUT = "runs/whytest/round2"
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
diff = lambda a, b: [by[("R3", a)][d][0] - by[("R3", b)][d][0] for d in by[("R3", a)] if d in by[("R3", b)]]
out["R3_lists_minus_none"] = ci(diff("lists", "none"))
out["R3_other_lists_minus_none"] = ci(diff("other_lists", "none"))
out["R3_sentences_minus_none"] = ci(diff("sentences", "none"))
exact = [float(picks[t] == k["right"]) for t, k in key.items() if k["test"] == "R1b" and k["arm"] == "real" and k["exact"] and t in picks]
out["R1b_real_exact_match"] = dict(n=len(exact), acc=ci(exact))
json.dump(out, open(f"{OUT}/reading_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
