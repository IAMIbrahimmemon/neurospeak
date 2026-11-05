"""Score PLAN_EFFECT.md. Writes effect_results.json."""
import glob, json, random, re, statistics as st, sys

D = sys.argv[1] if len(sys.argv) > 1 else "runs/wordtest"
key = json.load(open(f"{D}/effect_key.json"))
ans = {a["trial"]: a for f in sorted(glob.glob(f"{D}/effect_scores_*.json")) for a in json.load(open(f))}
assert not [t for t in key if t not in ans]
effect = json.load(open(f"{D}/effect.json"))
records = {r["id"]: r for r in json.load(open(f"{D}/units.json"))["records"]}
labels = {x["id"]: x for f in sorted(glob.glob(f"{D}/labels_*.json")) for x in json.load(open(f))}


def ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    b = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return b[int(.025 * n)], b[int(.975 * n)]


rows = [{**k, "trial": t, "ok": ans[t]["list"] == k["answer"], "kind": records[k["unit"]]["kind"], "layer": records[k["unit"]]["layer"]}
        for t, k in key.items()]
out = {}
print(f"{'condition':18s} {'kind':8s} {'n':>4s} {'correct':>8s}  95% CI")
for cond in ("steer_real", "steer_null", "steer_shuffled", "remove_real", "remove_shuffled"):
    for kind in ("feature", "neuron"):
        s = [float(r["ok"]) for r in rows if r["cond"] == cond and r["kind"] == kind]
        lo, hi = ci(s)
        out[f"{cond}/{kind}"] = {"n": len(s), "correct": st.mean(s), "ci": [lo, hi]}
        print(f"{cond:18s} {kind:8s} {len(s):4d} {st.mean(s):8.1%}  [{lo:.1%}, {hi:.1%}]")

for L in (15, 19):
    s = [float(r["ok"]) for r in rows if r["cond"] == "steer_real" and r["kind"] == "feature" and r["layer"] == L]
    print(f"steer real, features, L{L}: {st.mean(s):.1%} (n={len(s)})")

print("\nstability (tokens shared by lists from two halves of the contexts, of 8)")
for kind in ("feature", "neuron"):
    x = [e for e in effect.values() if e["kind"] == kind]
    d = [e["split_half"] - e["split_half_random"] for e in x]
    lo, hi = ci(d)
    out[f"split_half/{kind}"] = {"unit": st.mean(e["split_half"] for e in x), "random": st.mean(e["split_half_random"] for e in x), "diff_ci": [lo, hi]}
    print(kind, {k: round(v, 2) if not isinstance(v, list) else [round(q, 2) for q in v] for k, v in out[f"split_half/{kind}"].items()})


def fired_tokens(r):
    found = set()
    for c in r["label_view"]["strongest"]:
        found |= {m.strip().lower() for m in re.findall(r"\[\[(.*?)\]\]", c["text"])}
    return found - {""}


echo = {}
for x, e in effect.items():
    echo[x] = len({t.strip().lower() for t in e["steer"]} & fired_tokens(records[x]))
print("\nsteered lists that repeat tokens the unit fires on")
for kind in ("feature", "neuron"):
    ids = [x for x in effect if effect[x]["kind"] == kind]
    share = st.mean(echo[x] >= 1 for x in ids)
    out[f"echo/{kind}"] = share
    print(kind, f"{share:.0%} have at least one, mean {st.mean(echo[x] for x in ids):.2f} of 8")
for name, keep in (("no echo", lambda x: echo[x] == 0), ("echo", lambda x: echo[x] >= 1)):
    s = [float(r["ok"]) for r in rows if r["cond"] == "steer_real" and r["kind"] == "feature" and keep(r["unit"])]
    n = [float(r["ok"]) for r in rows if r["cond"] == "steer_null" and r["kind"] == "feature" and keep(r["unit"])]
    out[f"steer_real_{name}"] = {"n": len(s), "real": st.mean(s), "null": st.mean(n), "ci": ci(s)}
    print(f"steer real, features, {name}: {st.mean(s):.1%} n={len(s)} (null {st.mean(n):.1%})")

print("\nby labeler confidence, steer real, features")
for c in range(1, 6):
    s = [float(r["ok"]) for r in rows if r["cond"] == "steer_real" and r["kind"] == "feature" and labels[r["unit"]]["confidence"] == c]
    if s: print(f"confidence {c}: n={len(s)} {st.mean(s):.1%}")

read = {r["unit"]: r["balanced"] for r in json.load(open(f"{D}/results.json"))["rows"] if r["cond"] == "real" and r["kind"] == "feature"}
steer = {r["unit"]: r["ok"] for r in rows if r["cond"] == "steer_real" and r["kind"] == "feature"}
both = [(read[u], steer[u]) for u in steer if u in read]
hi_ = [s for b, s in both if b >= .75]; lo_ = [s for b, s in both if b < .75]
print(f"\nsteer real by read score: read at least 75%: {st.mean(hi_):.1%} (n={len(hi_)}); below: {st.mean(lo_):.1%} (n={len(lo_)})")

g = lambda c: out[f"{c}/feature"]["correct"]
bars = {"gate_steer": .45 <= g("steer_null") <= .55 and .45 <= g("steer_shuffled") <= .55, "gate_remove": .45 <= g("remove_shuffled") <= .55,
        "E1_steer_real": g("steer_real") >= .65, "E2_remove_real": g("remove_real") >= .6, "E3_stability": out["split_half/feature"]["diff_ci"][0] > 0}
out["bars"] = bars
print("\nbars:", bars)
json.dump({"summary": out, "rows": rows}, open(f"{D}/effect_results.json", "w"), indent=1)
