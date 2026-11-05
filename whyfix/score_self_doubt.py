"""Judges bars S1-S5 and O1 from PLAN_SELF_DOUBT.md. --check rescores only the two old arms, to test the scorer."""
import json, random, re, sys
from neurospeak import loopiness
from neurospeak.investigate import collapsed

sys.path.insert(0, "investigate")
from behaviors import DOUBT

OLD, NEW = ["none", "dir_neurospeak"], ["dir_self_doubt", "dir_self_doubt_matched", "dir_self_doubt_orth"]


def boxed(t):
    b = [x.strip() for x in re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t)]
    b = [x for x in b if x]  # the model sometimes echoes the instruction's empty box
    return b[-1] if b else None


def norm(x):
    return re.sub(r"\\?[$%]|\s|,|\\!|dfrac|tfrac|\\text\{[^}]*\}", "", x or "").replace("frac", "").rstrip(".").lower()


def ci(x, y, seed=0):
    d = [p - q for p, q in zip(x, y)]
    rng = random.Random(seed)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(10000))
    return dict(diff=sum(d) / len(d), low=m[249], high=m[9749], better=sum(v > 0 for v in d), worse=sum(v < 0 for v in d))


mean = lambda v: sum(v) / len(v)
check = "--check" in sys.argv
arms = OLD if check else OLD + NEW
rows = {a: [json.loads(l) for l in open(f"runs/whyfix/gsm64_{a}_traces.jsonl")] for a in arms}
ids = [r["id"] for r in rows["none"]]
assert len(ids) == 64 and all([r["id"] for r in v] == ids for v in rows.values())
right = {a: [100 * (norm(boxed(r["completion"])) == norm(r["answer"])) for r in v] for a, v in rows.items()}
rep = {a: [100 * loopiness(r["completion"]) for r in v] for a, v in rows.items()}
doubt = {a: [100 * len(DOUBT.findall(r["completion"])) / max(len(r["completion"].split()), 1) for r in v] for a, v in rows.items()}
table = {a: dict(correct_of_64=round(mean(right[a]) * 64 / 100), repeats=mean(rep[a]), doubt_per_100=mean(doubt[a]),
                 finished_of_64=sum(not r["cut"] for r in rows[a]), boxed_of_64=sum(boxed(r["completion"]) is not None for r in rows[a]),
                 collapsed_of_64=sum(collapsed(r["completion"]) for r in rows[a]), tokens=mean([r["n_tokens"] for r in rows[a]])) for a in arms}
out = dict(arms=table, wrong_but_boxed={a: [[r["id"], boxed(r["completion"]), r["answer"]] for r, ok in zip(rows[a], right[a]) if not ok and boxed(r["completion"]) is not None] for a in arms})
if not check:
    S, M, O, N = "dir_self_doubt", "dir_self_doubt_matched", "dir_self_doubt_orth", "none"
    out["vs_unedited"] = {a: dict(correct=ci(right[a], right[N]), repeats=ci(rep[a], rep[N])) for a in arms if a != N}
    out["vs_matched"] = dict(correct=ci(right[S], right[M]), repeats=ci(rep[S], rep[M]))
    out["vs_repetition"] = dict(correct=ci(right[S], right["dir_neurospeak"]), repeats=ci(rep[S], rep["dir_neurospeak"]))
    s1 = lambda a: out["vs_unedited"][a]["correct"]["diff"] >= 5 and out["vs_unedited"][a]["correct"]["low"] > 0
    s2 = lambda a: table[a]["repeats"] <= table[N]["repeats"] * 2 / 3
    out["bars"] = dict(S1=s1(S), S2=s2(S), S3=out["vs_matched"]["correct"]["low"] > 0 and out["vs_matched"]["repeats"]["high"] < 0,
                       S4=table[S]["collapsed_of_64"] <= table[N]["collapsed_of_64"] + 2, S5=table[S]["doubt_per_100"] <= table[N]["doubt_per_100"] * 2 / 3,
                       O1=s1(O) and s2(O))
    json.dump(out, open("runs/whyfix/self_doubt_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
