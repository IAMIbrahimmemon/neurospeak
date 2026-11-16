"""Checks on study 7, no bars: other thresholds, whether freed runs stay free, what the self-doubt edit changed, and whether arms differ from unedited at all."""
import json, sys
sys.path[:0] = ["investigate"]
from behaviors import DOUBT

SIZE, STEP = 200, 50
N, S, P, M = "none", "dir_self_doubt", "dir_neurospeak", "dir_self_doubt_matched"
R = {a: [json.loads(l) for l in open(f"runs/research/rescue_{a}_traces.jsonl")] for a in (N, S, P, M)}


def shares(r):
    w = (r["prefix"] + r["completion"]).split()
    first = {}
    for i in range(len(w) - 7):
        first.setdefault(tuple(w[i:i + 8]), i)
    old = [first[tuple(w[i:i + 8])] < i for i in range(len(w) - 7)]
    return [sum(old[i:i + SIZE]) / SIZE for i in range(r["prefix_words"], len(old) - SIZE + 1, STEP)]


out = {}
for a, v in R.items():
    s = [shares(r) for r in v]
    full = [x for x in s if x]
    w = [r["completion"].split() for r in v]
    out[a] = dict(
        any_window_under={str(t): sum(any(y < t for y in x) for x in s) for t in (0.3, 0.5, 0.7, 0.9)},
        last_window_still_under_half=sum(x[-1] < 0.5 for x in full),
        last_window_back_above_90=sum(x[-1] >= 0.9 for x in full),
        mean_share=sum(sum(x) / len(x) for x in full) / max(len(full), 1),
        doubt_per_100=sum(100 * len(DOUBT.findall(r["completion"])) / max(len(u), 1) for r, u in zip(v, w)) / len(v),
        new_words_share=sum(len(set(u) - set(r["prefix"].split())) / max(len(u), 1) for r, u in zip(v, w)) / len(v),
        exact_same_as_unedited=sum(r["completion"] == q["completion"] for r, q in zip(v, R[N])),
        first_100_chars_same_as_unedited=sum(r["completion"][:100] == q["completion"][:100] for r, q in zip(v, R[N])))
json.dump(out, open("runs/research/rescue_audit.json", "w"), indent=1)
print(json.dumps(out, indent=1))
