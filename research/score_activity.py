"""Judges bars D1-D5 from the saved per-trace AUROCs."""
import json, random

T = json.load(open("runs/research/activity_traces.json"))
rng = random.Random(0)
vals = lambda a, k: [t[a][k] for t in T if t[a][k] is not None]


def boot(x):
    m = sorted(sum(rng.choices(x, k=len(x))) / len(x) for _ in range(10000))
    return dict(mean=sum(x) / len(x), low=m[249], high=m[9749], n=len(x))


def diff(a, b, k):
    return boot([t[a][k] - t[b][k] for t in T if t[a][k] is not None and t[b][k] is not None])


names = ["original", "writer_2", "writer_3", "repetition", "matched"]
out = {n: {k: boot(vals(n, k)) for k in ("lag1", "lag5", "after_break")} for n in names}
out["minus_matched"], out["minus_repetition"] = diff("original", "matched", "lag1"), diff("original", "repetition", "lag1")
o = out["original"]
out["bars"] = dict(D1=o["lag1"]["mean"] >= 0.70 and o["lag1"]["low"] > 0.5, D2=out["minus_matched"]["mean"] >= 0.10 and out["minus_matched"]["low"] > 0,
                   D3=o["after_break"]["n"] >= 20 and o["after_break"]["mean"] >= 0.60 and o["after_break"]["low"] > 0.5, D4=o["lag5"]["mean"] >= 0.60 and o["lag5"]["low"] > 0.5, D5=out["minus_repetition"]["low"] > 0)
json.dump(out, open("runs/research/activity_results.json", "w"), indent=1)
for n in names:
    print(n, {k: (round(v["mean"], 3), round(v["low"], 3), round(v["high"], 3), v["n"]) for k, v in out[n].items()})
print(out["minus_matched"], out["minus_repetition"]); print(out["bars"])
