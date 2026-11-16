"""Independent recount of study 16's exits, per-arm dose, and a blinded excerpt file for a Claude reader. See PLAN_TILT.md."""
import json, random, re, sys
from neurospeak.investigate import collapsed
from neurospeak import loopiness

src = open("research/score_rescue.py").read()
exec(src[src.index("SIZE, STEP"):src.index("R = {a:")])
sys.path[:0] = ["whyfix"]
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right

OUT = sys.argv[1]
load = lambda p: [json.loads(l) for l in open(p)]
ARMS = ("none", "only", "only_eq", "tilt", "both", "perp_eq", "rand_eq")
R = {a: load(f"runs/research/tilt_{a}_traces.jsonl") for a in ARMS}
res = json.load(open("runs/research/tilt_results.json"))
for a in ARMS:
    assert sum(escaped(r) for r in R[a]) == res["arms"][a]["escaped"], a


def first_free(r):
    """The first 150 words of the first freed window, or the last 150 words of a finished run that has none."""
    w = (r["prefix"] + r["completion"]).split()
    first, old = {}, []
    for i in range(len(w) - 7):
        k = tuple(w[i:i + 8])
        old.append(first.setdefault(k, i) < i)
    for i in range(r["prefix_words"], len(old) - SIZE + 1, STEP):
        if sum(old[i:i + SIZE]) / SIZE < LOW:
            return " ".join(w[i:i + 150]), i - r["prefix_words"]
    return " ".join(w[-150:]), None


rng = random.Random(16)
items = []
for a in ARMS[1:]:
    ex = [r for r in R[a] if escaped(r)]
    take = ex if a in ("perp_eq", "rand_eq") else rng.sample(ex, 20)
    for r in take:
        t, at = first_free(r)
        items.append(dict(arm=a, id=r["id"], text=t, words_after_prefix=at, finished=not r["cut"]))
rng.shuffle(items)
for i, x in enumerate(items):
    x["item"] = i
json.dump(items, open(f"{OUT}/tilt_key.json", "w"), indent=1)
with open(f"{OUT}/tilt_blind.txt", "w") as f:
    for x in items:
        f.write(f"=== ITEM {x['item']} ===\n{x['text']}\n\n")
stats = {}
for a in ARMS:
    ex = [r for r in R[a] if escaped(r)]
    ats = sorted(v for v in (first_free(r)[1] for r in ex) if v is not None)
    stats[a] = dict(exits=len(ex), median_words_to_exit=ats[len(ats) // 2] if ats else None, median_tokens=sorted(r.get("new_tokens", 0) for r in R[a])[len(R[a]) // 2])
ids = {a: {r["id"]: escaped(r) for r in R[a]} for a in ARMS}
# post hoc pairs not in the bars
out = dict(stats=stats, posthoc=dict(only_eq_vs_only=ci([100 * escaped(r) for r in R["only_eq"]], [100 * escaped(r) for r in R["only"]]),
    only_eq_vs_both=ci([100 * escaped(r) for r in R["only_eq"]], [100 * escaped(r) for r in R["both"]]),
    perp_vs_none=ci([100 * escaped(r) for r in R["perp_eq"]], [100 * escaped(r) for r in R["none"]]),
    rand_vs_none=ci([100 * escaped(r) for r in R["rand_eq"]], [100 * escaped(r) for r in R["none"]]),
    correct_tilt_vs_only_eq=ci([100 * right(r) for r in R["tilt"]], [100 * right(r) for r in R["only_eq"]])))
json.dump(out, open("runs/research/tilt_audit.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print(len(items), "excerpts")
