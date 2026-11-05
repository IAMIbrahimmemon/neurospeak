"""Score the steering replication (see the amendment in PLAN_EFFECT.md). Writes replication_results.json."""
import glob, json, random, statistics as st, sys

D = sys.argv[1] if len(sys.argv) > 1 else "runs/wordtest"
key = json.load(open(f"{D}/replication_key.json"))
ans = {a["trial"]: a for f in sorted(glob.glob(f"{D}/replication_scores_*.json")) for a in json.load(open(f))}
missing = [t for t in key if t not in ans]
assert not missing, (len(missing), missing[:5])
records = {r["id"]: r for r in json.load(open(f"{D}/units.json"))["records"]}
labels = {x["id"]: x for f in sorted(glob.glob(f"{D}/labels_*.json")) for x in json.load(open(f))}
rows = [{"cond": k["cond"], "unit": k["unit"], "ok": ans[t]["list"] == k["answer"], "said": ans[t]["list"],
         "kind": records[k["unit"]]["kind"], "layer": records[k["unit"]]["layer"]} for t, k in key.items()]


def unit_ci(rows_, n=4000):
    """Bootstrap by unit, because each unit has two trials."""
    by = {}
    for r in rows_:
        by.setdefault(r["unit"], []).append(float(r["ok"]))
    means = [st.mean(v) for v in by.values()]
    rng = random.Random(0)
    b = sorted(st.mean(rng.choices(means, k=len(means))) for _ in range(n))
    return b[int(.025 * n)], b[int(.975 * n)]


out = {}
print(f"{'condition':16s} {'kind':8s} {'trials':>6s} {'correct':>8s}  95% CI (by unit)   said A")
for cond in ("steer_real", "steer_null", "steer_shuffled"):
    for kind in ("feature", "neuron"):
        s = [r for r in rows if r["cond"] == cond and r["kind"] == kind]
        m = st.mean(float(r["ok"]) for r in s)
        lo, hi = unit_ci(s)
        out[f"{cond}/{kind}"] = {"n": len(s), "correct": m, "ci": [lo, hi]}
        print(f"{cond:16s} {kind:8s} {len(s):6d} {m:8.1%}  [{lo:.1%}, {hi:.1%}]   {st.mean(r['said'] == 'A' for r in s):.0%}")

s = [r for r in rows if r["cond"] == "steer_real" and r["kind"] == "feature"]
for L in sorted({r["layer"] for r in s}):
    print(f"steer real, features, L{L}: {st.mean(float(r['ok']) for r in s if r['layer'] == L):.1%}")
for c in range(1, 6):
    x = [float(r["ok"]) for r in s if labels[r["unit"]]["confidence"] == c]
    if x: print(f"confidence {c}: trials={len(x)} {st.mean(x):.1%}")
by = {}
for r in s:
    by.setdefault(r["unit"], []).append(r["ok"])
print("features right on both / one / neither of two trials:", [sum(sum(v) == k for v in by.values()) for k in (2, 1, 0)])
g = lambda c: out[f"{c}/feature"]["correct"]
bars = {"gate": .45 <= g("steer_null") <= .55 and .45 <= g("steer_shuffled") <= .55, "E1": g("steer_real") >= .65}
print("bars:", bars)
json.dump({"summary": out, "bars": bars, "rows": rows}, open(f"{D}/replication_results.json", "w"), indent=1)
