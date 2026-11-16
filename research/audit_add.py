"""Checks on study 9, no bars: is the first-500 doubt rise just repetition? Early-loop runs left out, a shorter window, and new-text counts."""
import json, sys
sys.path[:0] = ["investigate", "research", "whyfix"]
from behaviors import DOUBT
from gsm_score import ci
from loop_onset import HIGH, STEP, windows
A = ("none", "doubt_low", "doubt_high", "random_low", "random_high", "rep_high")
R = {a: [json.loads(l) for l in open(f"runs/research/add_{a}_traces.jsonl")] for a in A}
per100 = lambda w: 100 * len(DOUBT.findall(" ".join(w))) / max(len(w), 1)
def onset(r):
    s = windows(r["completion"].split())
    i = next((i for i, x in enumerate(s) if x >= HIGH), None)
    return None if i is None else i * STEP
def novel(w):
    seen, n = set(), 0                      # doubt words whose surrounding 8 words haven't been seen before in this run
    for i, x in enumerate(w):
        k = tuple(w[max(i - 4, 0):i + 4])
        if DOUBT.search(x) and k not in seen: n += 1
        seen.add(k)
    return 100 * n / max(len(w), 1)
out = {}
for a in A:
    w = [r["completion"].split() for r in R[a]]
    late = [i for i, r in enumerate(R[a]) if (onset(r) or 10**9) >= 500]
    out[a] = dict(first_200=sum(per100(x[:200]) for x in w) / 48, first_500_novel=sum(novel(x[:500]) for x in w) / 48,
                  no_loop_before_500=len(late), first_500_those=sum(per100(w[i][:500]) for i in late) / max(len(late), 1), median_words=sorted(len(x) for x in w)[24])
d = {a: [per100(r["completion"].split()[:200]) for r in R[a]] for a in A}
n = {a: [novel(r["completion"].split()[:500]) for r in R[a]] for a in A}
out["high_vs_random_first_200"] = ci(d["doubt_high"], d["random_high"])
out["high_vs_random_novel_500"] = ci(n["doubt_high"], n["random_high"])
out["low_vs_random_novel_500"] = ci(n["doubt_low"], n["random_low"])
json.dump(out, open("runs/research/add_audit.json", "w"), indent=1)
for k, v in out.items(): print(k, {a: round(b, 2) if isinstance(b, float) else b for a, b in v.items()})
