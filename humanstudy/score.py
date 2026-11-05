"""Scores returned files against the bars in PLAN.md.  python3 humanstudy/score.py"""
import glob, json, random, statistics as st

key = json.load(open("runs/humanstudy/key.json"))
files = [json.load(open(p)) for p in sorted(glob.glob("humanstudy/responses/*.json"))]
if not files:
    raise SystemExit("no files in humanstudy/responses/")
people = []
for f in files:
    by = {"none": [], "real": [], "other": [], "cause": []}
    for a in f["answers"]:
        by[a["condition"]].append(float(a["pick"] == key[a["trial"]]["right"]))
    people.append({k: st.mean(v) if v else None for k, v in by.items()})


def ci(fn, n=4000):
    r = random.Random(7)
    m = sorted(fn(r.choices(people, k=len(people))) for _ in range(n))
    return round(fn(people), 3), round(m[int(.025 * n)], 3), round(m[int(.975 * n)], 3)


mean = lambda k: lambda ps: st.mean(p[k] for p in ps)
diff = lambda a, b: lambda ps: st.mean(p[a] - p[b] for p in ps)
out = dict(n=len(people), none=ci(mean("none")), real=ci(mean("real")), other=ci(mean("other")), cause=ci(mean("cause")),
           H1_real_minus_none=ci(diff("real", "none")), H2_real_minus_other=ci(diff("real", "other")), H3_other_minus_none=ci(diff("other", "none")))
print(json.dumps(out, indent=1))
