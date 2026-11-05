"""Scores C4b.  python3 tracetest/score_reading.py"""
import collections, glob, json, random, statistics as st

key = json.load(open("runs/tracetest/b/read_key.json"))
picks = {r["trial"]: r["pick"] for p in glob.glob("runs/tracetest/b/read_scores_*.json") for r in json.load(open(p))}
print("missing", len([t for t in key if t not in picks]))
by = collections.defaultdict(list)
said = collections.defaultdict(list)
for t, k in key.items():
    if t in picks:
        by[k["arm"]].append(float(picks[t] == k["right"]))
        said[k["arm"]].append(picks[t] == "A")


def ci(xs, n=4000):
    r = random.Random(6)
    m = sorted(sum(r.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return round(st.mean(xs), 3), round(m[int(.025 * n)], 3), round(m[int(.975 * n)], 3)


out = {a: dict(n=len(x), acc=ci(x), said_A=round(st.mean(said[a]), 3)) for a, x in by.items()}
json.dump(out, open("runs/tracetest/b/reading_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
