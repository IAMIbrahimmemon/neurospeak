"""Scores bars E1 to E4 in PLAN_READER.md. Readers are Claude models."""
import glob, json, random, statistics as st

D = "runs/investigate/reader"
key = json.load(open(f"{D}/key.json"))
ans = {a["id"]: a for f in glob.glob(f"{D}/answers_*.json") for a in json.load(open(f))}
missing = [k for k in key if k not in ans]
rows = [dict(**key[k], ok=ans[k]["pick"] == key[k]["right"], pick=ans[k]["pick"]) for k in key if k in ans]
by = {}
for r in rows:
    by.setdefault((r["behavior"], r["index"]), {})[r["condition"]] = float(r["ok"])
trials = list(by.values())
rng = random.Random(0)


def diff(a, b):
    use = [t for t in trials if a in t and b in t]
    d = [t[a] - t[b] for t in use]
    m = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return round(100 * st.mean(d), 1), round(100 * m[249], 1), round(100 * m[9749], 1)


conds = ["none", "wrong", "name", "evidence"]
acc = {c: round(100 * st.mean(r["ok"] for r in rows if r["condition"] == c), 1) for c in conds}
per = {b: {c: round(100 * st.mean(r["ok"] for r in rows if r["condition"] == c and r["behavior"] == b), 1) for c in conds} for b in {r["behavior"] for r in rows}}
said_a = {c: round(100 * st.mean(r["pick"] == "A" for r in rows if r["condition"] == c), 1) for c in conds}
e2, e3, e4 = diff("evidence", "none"), diff("evidence", "wrong"), diff("evidence", "name")
out = dict(n=len(rows), missing=missing, accuracy=acc, per_behavior=per, said_a=said_a, evidence_minus_none=e2, evidence_minus_wrong=e3, evidence_minus_name=e4,
           E1=acc["evidence"] >= 75, E2=e2[0] >= 10 and e2[1] > 0, E3=e3[0] >= 10 and e3[1] > 0)
json.dump(out, open(f"{D}/results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
