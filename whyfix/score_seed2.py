"""Judge bars F9 to F11 on the corrected loop test, with paired bootstrap intervals over problems."""
import json, random
from neurospeak import loopiness

rows = {a: [json.loads(l) for l in open(f"runs/whyfix/seed2_{a}_traces.jsonl")] for a in "none why2 active2 contrast2 random2".split()}
rate = {a: [loopiness(r["completion"]) * 100 for r in v] for a, v in rows.items()}
mean = {a: sum(v) / len(v) for a, v in rate.items()}
assert all(r["prompt"] == b["prompt"] for a in rows for r, b in zip(rows[a], rows["none"]))
rng = random.Random(0)


def ci(x, y):
    d = [p - q for p, q in zip(x, y)]
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(10000))
    return sum(d) / len(d), means[249], means[9749], sum(v < 0 for v in d), sum(v > 0 for v in d)


line = mean["none"] * 2 / 3
out = dict(mean=mean, line=line, arms={})
for a in rows:
    if a != "none":
        out["arms"][a] = dict(vs_none=ci(rate[a], rate["none"]), vs_random=None if a == "random2" else ci(rate[a], rate["random2"]))
for bar, a in (("F9", "why2"), ("F10", "active2"), ("F11", "contrast2")):
    out[bar] = dict(arm=a, under_line=mean[a] <= line, below_random=mean["random2"] - mean[a], passed=mean[a] <= line and mean["random2"] - mean[a] >= 10)
json.dump(out, open("runs/whyfix/seed2_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
