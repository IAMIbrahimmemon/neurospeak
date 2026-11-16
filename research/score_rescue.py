"""Judges bars E1-E5 in PLAN_RESCUE.md."""
import json, sys
from neurospeak.investigate import collapsed

sys.path[:0] = ["whyfix"]
from gsm_score import boxed, ci
from sensitivity_replicate_rules import right

SIZE, STEP, LOW = 200, 50, 0.5
N, S, P, M = "none", "dir_self_doubt", "dir_neurospeak", "dir_self_doubt_matched"


def freed_window(r):
    w = (r["prefix"] + r["completion"]).split()
    first = {}
    for i in range(len(w) - 7):
        first.setdefault(tuple(w[i:i + 8]), i)
    old = [first[tuple(w[i:i + 8])] < i for i in range(len(w) - 7)]
    return any(sum(old[i:i + SIZE]) / SIZE < LOW for i in range(r["prefix_words"], len(old) - SIZE + 1, STEP))


def escaped(r):
    return not collapsed(r["completion"]) and (not r["cut"] or freed_window(r))


R = {a: [json.loads(l) for l in open(f"runs/research/rescue_{a}_traces.jsonl")] for a in (N, S, P, M)}
assert all(len(v) == 56 and [(r["id"], r["prefix"]) for r in v] == [(r["id"], r["prefix"]) for r in R[N]] for v in R.values())
esc = {a: [100 * escaped(r) for r in v] for a, v in R.items()}
# an answer given in the continuation, judged with the prefix's own boxes left out
ans = {a: [100 * right(r) for r in v] for a, v in R.items()}
count = lambda v: sum(x > 0 for x in v)
arms = {a: dict(escaped_of_56=count(esc[a]), finished_of_56=sum(not r["cut"] for r in R[a]), window_freed_of_56=sum(freed_window(r) for r in R[a]),
                collapsed_of_56=sum(collapsed(r["completion"]) for r in R[a]), boxed_in_continuation_of_56=sum(boxed(r["completion"]) is not None for r in R[a]),
                correct_in_continuation_of_56=count(ans[a])) for a in R}
out = dict(arms=arms, self_doubt_vs_unedited=ci(esc[S], esc[N]), self_doubt_vs_matched=ci(esc[S], esc[M]), repetition_vs_unedited=ci(esc[P], esc[N]), self_doubt_vs_repetition=ci(esc[S], esc[P]))
c = lambda a: arms[a]["collapsed_of_56"]
out["bars"] = dict(E1=arms[N]["escaped_of_56"] / 56 <= 0.10,
                   E2=out["self_doubt_vs_unedited"]["diff"] >= 20 and out["self_doubt_vs_unedited"]["low"] > 0,
                   E3=out["self_doubt_vs_matched"]["low"] > 0,
                   E4=out["repetition_vs_unedited"]["diff"] >= 20 and out["repetition_vs_unedited"]["low"] > 0,
                   E5=c(S) <= c(N) + 2 and c(P) <= c(N) + 2)
json.dump(out, open("runs/research/rescue_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
