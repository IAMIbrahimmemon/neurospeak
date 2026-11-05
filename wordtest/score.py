"""Score the word test against the bars in PLAN.md. Writes results.json next to the scores."""
import glob, json, random, statistics as st, sys

D = sys.argv[1] if len(sys.argv) > 1 else "runs/wordtest"
records = {r["id"]: r for r in json.load(open(f"{D}/units.json"))["records"]}
key = json.load(open(f"{D}/trials_key.json"))
ans = {a["trial"]: a for f in sorted(glob.glob(f"{D}/scores_*.json")) for a in json.load(open(f))}
missing = [t for t in key if t not in ans]
assert not missing, missing[:5]
labels = {x["id"]: x for f in sorted(glob.glob(f"{D}/labels_*.json")) for x in json.load(open(f))}

rows = []
for tid, k in key.items():
    a = ans[tid]
    assert len(a["fires"]) == len(k["types"]), tid
    by = {}
    for said, kind in zip(a["fires"], k["types"]):
        correct = said if kind.startswith("positive") else not said
        by.setdefault(kind, []).append(correct)
    pos = by.get("positive_top", []) + by.get("positive_weak", [])
    neg = by.get("same_token", []) + by.get("random", [])
    r = records[k["unit"]]
    rows.append({"cond": k["cond"], "unit": k["unit"], "kind": r["kind"], "layer": r["layer"],
                 "balanced": (st.mean(pos) + st.mean(neg)) / 2, **{t: st.mean(v) for t, v in by.items()},
                 "list_ok": (a.get("list") == k["answer"]) if k["answer"] else None})


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def ci(xs, n=4000):
    rng = random.Random(0)
    xs = [x for x in xs if x is not None]
    b = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return b[int(.025 * n)], b[int(.975 * n)]


out = {}
print(f"{'group':26s} {'n':>4s} {'balanced':>9s} {'95% CI':>15s} {'top':>5s} {'weak':>5s} {'sameTok':>8s} {'random':>7s} {'list':>5s}")
for cond in ("real", "shuffled"):
    for kind in ("feature", "neuron"):
        for layer in (None, *sorted({x["layer"] for x in rows})):
            s = [x for x in rows if x["cond"] == cond and x["kind"] == kind and layer in (None, x["layer"])]
            lo, hi = ci([x["balanced"] for x in s])
            d = {m: mean([x.get(m) for x in s]) for m in ("balanced", "positive_top", "positive_weak", "same_token", "random", "list_ok")}
            name = f"{cond}/{kind}/{'all' if layer is None else 'L' + str(layer)}"
            out[name] = {**d, "n": len(s), "ci": [lo, hi]}
            print(f"{name:26s} {len(s):4d} {d['balanced']:9.1%}   [{lo:5.1%},{hi:5.1%}] {d['positive_top']:5.0%} {d['positive_weak']:5.0%} {d['same_token']:8.0%} {d['random']:7.0%} {d['list_ok']:5.0%}")

f = [x["balanced"] for x in rows if x["cond"] == "real" and x["kind"] == "feature"]
n = [x["balanced"] for x in rows if x["cond"] == "real" and x["kind"] == "neuron"]
rng = random.Random(1)
diffs = sorted(st.mean(rng.choices(f, k=len(f))) - st.mean(rng.choices(n, k=len(n))) for _ in range(4000))
out["feature_minus_neuron"] = {"diff": st.mean(f) - st.mean(n), "ci": [diffs[100], diffs[3900]]}
print(f"\nfeature minus neuron, real labels: {st.mean(f) - st.mean(n):+.1%} [{diffs[100]:+.1%}, {diffs[3900]:+.1%}]")

print("\ncausal strength (removing the unit vs same-size controls)")
for kind in ("feature", "neuron"):
    c = [r["causal"] for r in records.values() if r["kind"] == kind and r["causal"]]
    d = {"n": len(c), "median_kl_remove": st.median(x["kl_remove"] for x in c), "median_kl_random": st.median(x["kl_random"] for x in c),
         "median_kl_other_word": st.median(x["kl_other_word"] for x in c),
         "beats_random": mean([x["kl_remove"] > x["kl_random"] for x in c]), "beats_other_word": mean([x["kl_remove"] > x["kl_other_word"] for x in c])}
    out["causal/" + kind] = d
    print(kind, {k: round(v, 5) if isinstance(v, float) else v for k, v in d.items()})

print("\nreader confidence, real feature trials")
for conf in range(1, 6):
    s = [x for x in rows if x["cond"] == "real" and x["kind"] == "feature" and labels[x["unit"]]["confidence"] == conf]
    if s: print(f"confidence {conf}: n={len(s):3d} balanced {mean([x['balanced'] for x in s]):.1%} list {mean([x['list_ok'] for x in s]):.0%}")

passed = {
    "read": out["real/feature/all"]["balanced"] >= .75,
    "hard_read": out["real/feature/all"]["same_token"] >= .65,
    "beats_neurons": out["feature_minus_neuron"]["ci"][0] > 0,
    "causal_strength": out["causal/feature"]["beats_random"] >= .8 and out["causal/feature"]["beats_other_word"] >= .5,
    "causal_reading": out["real/feature/all"]["list_ok"] >= .65,
}
print("\nbars:", passed)
out["bars"] = passed
json.dump({"summary": out, "rows": rows}, open(f"{D}/results.json", "w"), indent=1)
