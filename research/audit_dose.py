"""Checks on study 8, no bars: thresholds, whether freed runs stay out, text identity, and repeat share."""
import json, sys
sys.path[:0] = ["investigate"]
from behaviors import DOUBT
from neurospeak.investigate import collapsed
SIZE, STEP = 200, 50
load = lambda p: [json.loads(l) for l in open(p)]
R = dict(none=load("runs/research/rescue_none_traces.jsonl"), repetition=load("runs/research/rescue_dir_neurospeak_traces.jsonl"))
for a in ("matched_mean", "doubt_dose", "random_dose", "rep_only", "doubt_only"):
    R[a] = load(f"runs/research/dose_{a}_traces.jsonl")
def shares(r):
    w = (r["prefix"] + r["completion"]).split()
    first = {}
    for i in range(len(w) - 7):
        first.setdefault(tuple(w[i:i + 8]), i)
    old = [first[tuple(w[i:i + 8])] < i for i in range(len(w) - 7)]
    return [sum(old[i:i + SIZE]) / SIZE for i in range(r["prefix_words"], len(old) - SIZE + 1, STEP)]
out = {}
for a, v in R.items():
    s = [shares(r) for r in v]; full = [x for x in s if x]
    w = [r["completion"].split() for r in v]
    out[a] = dict(any_window_under={str(t): sum(any(y < t for y in x) for x in s) for t in (0.3, 0.5, 0.7, 0.9)},
        last_under_half=sum(x[-1] < 0.5 for x in full), last_above_90=sum(x[-1] >= 0.9 for x in full),
        mean_share=round(sum(sum(x) / len(x) for x in full) / max(len(full), 1), 3),
        doubt_per_100=round(sum(100 * len(DOUBT.findall(r["completion"])) / max(len(u), 1) for r, u in zip(v, w)) / len(v), 2),
        new_words_share=round(sum(len(set(u) - set(r["prefix"].split())) / max(len(u), 1) for r, u in zip(v, w)) / len(v), 3),
        same_as_unedited=sum(r["completion"] == q["completion"] for r, q in zip(v, R["none"])),
        first100_same=sum(r["completion"][:100] == q["completion"][:100] for r, q in zip(v, R["none"])),
        collapsed=sum(collapsed(r["completion"]) for r in v), finished=sum(not r["cut"] for r in v))
json.dump(out, open("runs/research/dose_audit.json", "w"), indent=1)
for a, o in out.items(): print(a, o)
