"""Scores C1-C3 from tracetest/PLAN.md, plus the no-bar source report.  python3 tracetest/score.py"""
import json, random, statistics as st

R = [json.loads(l) for l in open("runs/tracetest/mech.jsonl")]


def ci(xs, n=4000, seed=4):
    r = random.Random(seed)
    m = sorted(sum(r.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return round(st.mean(xs), 3), round(m[int(.025 * n)], 3), round(m[int(.975 * n)], 3)


def median_ci(xs, n=4000, seed=5):
    r = random.Random(seed)
    m = sorted(st.median(r.choices(xs, k=len(xs))) for _ in range(n))
    return round(st.median(xs), 4), round(m[int(.025 * n)], 4), round(m[int(.975 * n)], 4)


out = dict(n=len(R))
for other in ("active_same", "random_word"):
    out[f"C1_beats_{other}"] = ci([float(r["drops"]["parents"] > r["drops"][other]) for r in R])
out["median_child_drop"] = {a: round(st.median(r["drops"][a] / max(r["child_base"], 1e-6) for r in R), 3) for a in ("parents", "active_same", "random_word")}
singles = [s for r in R for s in r["singles"]]
out["C2_parents_real"] = round(st.mean(s > 0 for s in singles), 3)
# how much of the lead drop returns when the word is held at its original value
undo = [r["lead_held"] - r["lead_cut"] for r in R]
undo_other = [r["lead_other"] - r["lead_cut"] for r in R]
out["lead_drop_from_parents_median"] = round(st.median(r["gap"] - r["lead_cut"] for r in R), 3)
out["held_child_returns_median"] = median_ci(undo)
out["held_random_word_returns_median"] = median_ci(undo_other)
out["C3_paired_median"] = median_ci([a - b for a, b in zip(undo, undo_other)])
out["C3_child_wins"] = ci([float(a > b) for a, b in zip(undo, undo_other)])
out["top_source_not_last_token"] = ci([float(r["top_source"][0] != r["child_pos"]) for r in R])
out["child_on_last_token"] = round(st.mean(r["child_pos"] == r["last_token"] for r in R), 3)
out["parent_on_child_token"] = round(st.mean(r["parents"][0][0] == r["child_pos"] for r in R), 3)
json.dump(out, open("runs/tracetest/mechanics.json", "w"), indent=1)
print(json.dumps(out, indent=1))
