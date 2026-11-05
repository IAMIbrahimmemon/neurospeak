"""Judge bars A1 and A2 on 64 new GSM8K test problems, with paired bootstrap intervals over problems."""
import json, random, re
from neurospeak import loopiness


def boxed(t):
    b = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t)
    return b[-1].strip() if b else None


def norm(x):
    return re.sub(r"[\s$,]|\\!|dfrac|tfrac|\\text\{[^}]*\}", "", x or "").replace("frac", "").rstrip(".").lower()


rows = {a: [json.loads(l) for l in open(f"runs/whyfix/gsm64_{a}_traces.jsonl")] for a in ("none", "dir_neurospeak")}
assert [r["id"] for r in rows["none"]] == [r["id"] for r in rows["dir_neurospeak"]]
right = {a: [100 * (norm(boxed(r["completion"])) == norm(r["answer"])) for r in v] for a, v in rows.items()}
rep = {a: [100 * loopiness(r["completion"]) for r in v] for a, v in rows.items()}
rng = random.Random(0)


def ci(x, y):
    d = [p - q for p, q in zip(x, y)]
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(10000))
    return dict(diff=sum(d) / len(d), low=m[249], high=m[9749], better=sum(v > 0 for v in d), worse=sum(v < 0 for v in d))


mean = lambda v: sum(v) / len(v)
out = dict(n=len(rows["none"]),
           correct={a: mean(v) for a, v in right.items()}, repeats={a: mean(v) for a, v in rep.items()},
           finished={a: mean([100 * (not r["cut"]) for r in v]) for a, v in rows.items()},
           tokens={a: mean([r["n_tokens"] for r in v]) for a, v in rows.items()},
           correct_diff=ci(right["dir_neurospeak"], right["none"]), repeats_diff=ci(rep["dir_neurospeak"], rep["none"]))
d = out["correct_diff"]
out["A1"] = d["diff"] >= 5 and d["low"] > 0
out["A2"] = out["repeats"]["dir_neurospeak"] <= out["repeats"]["none"] * 2 / 3
json.dump(out, open("runs/whyfix/gsm64_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
