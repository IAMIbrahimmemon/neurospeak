"""Judges bars R1-R7 in PLAN_REPLICATE.md. Run with gsm64 as the argument to test it on the seed 3 arms; only gsm64b saves results."""
import json, os, sys
from neurospeak import loopiness
from neurospeak.investigate import collapsed

sys.path[:0] = ["investigate", os.path.dirname(os.path.abspath(__file__))]
from behaviors import DOUBT
from gsm_score import boxed, ci, norm

prefix = sys.argv[1] if len(sys.argv) > 1 else "gsm64b"
N, S, P, M = "none", "dir_self_doubt", "dir_neurospeak", "dir_self_doubt_matched"
rows = {a: [json.loads(l) for l in open(f"runs/whyfix/{prefix}_{a}_traces.jsonl")] for a in (N, S, P, M)}
base = rows[N]
assert len(base) == 64 and all([(r["id"], r["prompt"]) for r in v] == [(r["id"], r["prompt"]) for r in base] for v in rows.values())
ok = lambda r: norm(boxed(r["completion"])) == norm(r["answer"])
mean = lambda v: sum(v) / len(v)
right = {a: [100 * ok(r) for r in v] for a, v in rows.items()}
done = {a: [100 * (ok(r) and not r["cut"]) for r in v] for a, v in rows.items()}
rep = {a: [100 * loopiness(r["completion"]) for r in v] for a, v in rows.items()}
doubt = {a: [100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in v] for a, v in rows.items()}
arms = {a: dict(correct_of_64=sum(x > 0 for x in right[a]), boxed_of_64=sum(boxed(r["completion"]) is not None for r in rows[a]),
                finished_of_64=sum(not r["cut"] for r in rows[a]), finished_and_correct_of_64=sum(x > 0 for x in done[a]),
                repeats=mean(rep[a]), doubt_per_100=mean(doubt[a]), collapsed_of_64=sum(collapsed(r["completion"]) for r in rows[a]),
                tokens=mean([r["n_tokens"] for r in rows[a]])) for a in rows}
out = dict(arms=arms, correct_vs_unedited=ci(right[S], right[N]), repeats_vs_unedited=ci(rep[S], rep[N]),
           correct_vs_matched=ci(right[S], right[M]), repeats_vs_matched=ci(rep[S], rep[M]),
           finished_correct_vs_unedited=ci(done[S], done[N]), correct_vs_repetition=ci(right[S], right[P]), repeats_vs_repetition=ci(rep[S], rep[P]),
           matched_correct_vs_unedited=ci(right[M], right[N]), repetition_correct_vs_unedited=ci(right[P], right[N]),
           wrong_but_boxed={a: [[r["id"], boxed(r["completion"]), r["answer"]] for r in v if not ok(r) and boxed(r["completion"]) is not None] for a, v in rows.items()})
out["bars"] = dict(R1=out["correct_vs_unedited"]["diff"] >= 5 and out["correct_vs_unedited"]["low"] > 0,
                   R2=arms[S]["repeats"] <= arms[N]["repeats"] * 2 / 3,
                   R3=out["correct_vs_matched"]["low"] > 0 and out["repeats_vs_matched"]["high"] < 0,
                   R4=arms[S]["collapsed_of_64"] <= arms[N]["collapsed_of_64"] + 2,
                   R5=arms[S]["doubt_per_100"] <= arms[N]["doubt_per_100"] * 2 / 3,
                   R6=out["finished_correct_vs_unedited"]["low"] > 0, R7=out["correct_vs_repetition"]["low"] > 0)
if prefix == "gsm64b":
    json.dump(out, open("runs/whyfix/replicate_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
