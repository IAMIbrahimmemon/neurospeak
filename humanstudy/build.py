"""Builds the study page from real trials. The answers stay out of the page.  python3 humanstudy/build.py"""
import json, os, random

SRC = "runs/whytest/reading.json"
WORDS = "runs/words/Qwen__Qwen3.5-0.8B/L19.json"
N_PREDICT, N_CAUSE = 30, 10
names = json.load(open(WORDS))
D = json.load(open(SRC))
note = lambda f: dict(fires_on=names[str(f)]["fires_on"], pushes=names[str(f)]["effect"])
rng = random.Random(91)
pred = [t for t in D["r2"] if all(str(f) in names for f in t["words"])]
cause = [t for t in D["r1"] if str(t["driver"]) in names and str(t["decoy"]) in names]
rng.shuffle(pred)
rng.shuffle(cause)
pred, cause = pred[:N_PREDICT], cause[:N_CAUSE]
key, trials = {}, []
others = pred[1:] + pred[:1]
for i, (t, o) in enumerate(zip(pred, others)):
    right = "A" if i % 2 == 0 else "B"
    a, b = (t["answer"], t["against"]) if right == "A" else (t["against"], t["answer"])
    tid = f"P{i:02d}"
    key[tid] = dict(right=right, kind="predict", slot=i % 3)
    trials.append(dict(id=tid, kind="predict", context=t["context"], A=a, B=b, slot=i % 3, real=[note(f) for f in t["words"]], other=[note(f) for f in o["words"]]))
for i, t in enumerate(cause):
    right = "A" if i % 2 == 0 else "B"
    a, b = (t["driver"], t["decoy"]) if right == "A" else (t["decoy"], t["driver"])
    tid = f"C{i:02d}"
    key[tid] = dict(right=right, kind="cause")
    trials.append(dict(id=tid, kind="cause", context=t["context"], answer=t["answer"], against=t["against"], A=note(a), B=note(b)))
os.makedirs("runs/humanstudy", exist_ok=True)
json.dump(key, open("runs/humanstudy/key.json", "w"), indent=1)
page = open("humanstudy/template.html", encoding="utf-8").read().replace("/*TRIALS*/[]", json.dumps(trials, ensure_ascii=False).replace("</", "<\\/"))
open("humanstudy/study.html", "w", encoding="utf-8").write(page)
print("predict", len(pred), "cause", len(cause), "page bytes", len(page))
