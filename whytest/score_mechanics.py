"""Scores the bars in whytest/PLAN.md for the mechanics test.  .venv/bin/python whytest/score_mechanics.py"""
import json, os, random, statistics as st


def boot(xs, n=2000, seed=1):
    r = random.Random(seed)
    m = sorted(sum(r.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return m[int(.025 * n)], m[int(.975 * n)]


def spearman(a, b):
    rank = lambda v: [sorted(v).index(x) for x in v]
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
    return num / den if den else 0.0


out = {}
FILES = {19: "runs/whytest/L19.jsonl", 15: "runs/whytest/L15.jsonl", "distilgpt2_L4": "runs/whytest/distil_L4.jsonl"}
for L, path in FILES.items():
    if not os.path.exists(path):
        continue
    R = [json.loads(l) for l in open(path)]
    res = {"n": len(R), "median_lead": st.median(r["gap"] for r in R)}
    for other in ("active_last", "active_any", "random_word"):
        wins = [float(r["arms"]["why"]["drop"] > r["arms"][other]["drop"]) for r in R]
        res[f"M1_beats_{other}"] = (st.mean(wins), *boot(wins))
    res["arm_median_drop"] = {m: st.median(r["arms"][m]["drop"] for r in R) for m in R[0]["arms"]}
    res["arm_flip"] = {m: st.mean(r["arms"][m]["flip"] for r in R) for m in R[0]["arms"]}
    words = [s for r in R for s in r["single"]]
    res["M3_words_real"] = st.mean(s["measured"] < 0 for s in words)
    res["M4_spearman_median"] = st.median(spearman([s["est"] for s in r["single"]], [-s["measured"] for s in r["single"]]) for r in R if len(r["single"]) >= 3)
    res["share_top_word_on_last_token"] = st.mean(r["single"][0]["pos"] == 127 for r in R if r["single"])
    res["curve"] = {n: dict(drop=st.median(c["drop"] for r in R for c in r["curve"] if c["n"] == n),
                            flip=st.mean(c["flip"] for r in R for c in r["curve"] if c["n"] == n)) for n in (1, 2, 5, 10, 20, 50)}
    res["by_source"] = {s: dict(n=sum(r["source"] == s for r in R), flip=st.mean(r["arms"]["why"]["flip"] for r in R if r["source"] == s)) for s in ("wiki", "gsm8k", "trace")}
    bins = [(0, .5), (.5, 1), (1, 2), (2, 99)]
    res["by_lead"] = {f"{a}-{b}": dict(n=sum(a <= r["gap"] < b for r in R), flip5=st.mean([r["arms"]["why"]["flip"] for r in R if a <= r["gap"] < b] or [0]),
                                       flip_random=st.mean([r["arms"]["random_dir"]["flip"] for r in R if a <= r["gap"] < b] or [0])) for a, b in bins}
    out[L] = res
json.dump(out, open("runs/whytest/mechanics.json", "w"), indent=1)
print(json.dumps(out, indent=1))
