"""Merges round 2's labels, then writes reader batches and the key.  python3 whytest/make_round2_trials.py"""
import glob, json, random

OUT, WORDS = "runs/whytest/round2", "runs/words/Qwen__Qwen3.5-0.8B/L19.json"
names = json.load(open(WORDS))
asked = sum(len(json.load(open(p))) for p in glob.glob(f"{OUT}/label_batch_*.json"))
got = 0
for p in glob.glob(f"{OUT}/labels_*.json"):
    for row in json.load(open(p)):
        got += 1
        names[row["id"][1:]] = dict(fires_on=row["fires_on"], effect=row["effect"], confidence=row["confidence"])
json.dump(names, open(WORDS, "w"), indent=1, ensure_ascii=False)
print("asked", asked, "labeled", got, "names", len(names))

effects = json.load(open(f"{OUT}/word_effects.json"))
note = lambda f: {"fires_on": names[str(f)]["fires_on"], "tends_to_make_it_say": names[str(f)]["effect"]}
lists = lambda f: {"raises": effects[str(f)]["raises"], "lowers": effects[str(f)]["lowers"]}
D = json.load(open(f"{OUT}/reading.json"))
rng = random.Random(61)
r1 = []
for t in D["r1"]:
    d = names.get(str(t["driver"]))
    near = [f for f in t["decoys"] if d and str(f) in names and abs(names[str(f)]["confidence"] - d["confidence"]) <= 1]
    near.sort(key=lambda f: abs(names[str(f)]["confidence"] - d["confidence"]))
    if near:
        r1.append(dict(t, decoy=near[0], exact=names[str(near[0])]["confidence"] == d["confidence"]))
print("R1b trials within 1 confidence point", len(r1), "of", len(D["r1"]), "exact", sum(t["exact"] for t in r1))
r3 = [dict(t, sent=[f for f in t["words"] if str(f) in names], lists=[f for f in t["words"] if str(f) in effects]) for t in D["r2"]]
key, n = {}, 0


def tid():
    global n
    n += 1
    return f"T{n:03d}"


def balanced(k):
    v = ["A"] * (k // 2) + ["B"] * (k - k // 2)
    rng.shuffle(v)
    return v


def cause(t, src, arm, right):
    a, b = (src["driver"], src["decoy"]) if right == "A" else (src["decoy"], src["driver"])
    i = tid()
    key[i] = dict(test="R1b", arm=arm, right=right, base=t["pos"], chunk=t["chunk"], exact=src["exact"])
    return dict(trial=i, kind="cause", context=t["context"], answer=t["answer"], against=t["against"], A=note(a), B=note(b))


def predict(t, words, arm, right):
    a, b = (t["answer"], t["against"]) if right == "A" else (t["against"], t["answer"])
    i = tid()
    key[i] = dict(test="R3", arm=arm, right=right, base=t["pos"], chunk=t["chunk"])
    out = dict(trial=i, kind="predict", context=t["context"], A=a, B=b)
    if words:
        out["words"] = words
    return out


rng.shuffle(r1)
groups = [r1[i::4] for i in range(4)]
other_of = {(t["chunk"], t["pos"]): o for t, o in zip(r1, r1[1:] + r1[:1])}
batches = {}
for r in range(4):
    real, floor = groups[r], groups[(r + 1) % 4]
    trials = [cause(t, t, "real", s) for t, s in zip(real, balanced(len(real)))]
    trials += [cause(t, other_of[(t["chunk"], t["pos"])], "floor", s) for t, s in zip(floor, balanced(len(floor)))]
    rng.shuffle(trials)
    batches[f"r1b_{r}"] = trials

rng.shuffle(r3)
groups = [r3[i::8] for i in range(8)]
list_of = {(t["chunk"], t["pos"]): s["lists"] for t, s in zip(r3, r3[9:] + r3[:9])}
for r in range(8):
    trials = []
    arms = (("none", lambda t: None), ("sentences", lambda t: [note(f) for f in t["sent"]]), ("lists", lambda t: [lists(f) for f in t["lists"]]),
            ("other_lists", lambda t: [lists(f) for f in list_of[(t["chunk"], t["pos"])]]))
    for j, (arm, words) in enumerate(arms):
        g = groups[(r + 2 * j) % 8]
        trials += [predict(t, words(t), arm, s) for t, s in zip(g, balanced(len(g)))]
    rng.shuffle(trials)
    batches[f"r3_{r}"] = trials

for name, trials in batches.items():
    json.dump(trials, open(f"{OUT}/read_{name}.json", "w"), indent=1, ensure_ascii=False)
    print(name, len(trials))
json.dump(key, open(f"{OUT}/read_key.json", "w"))
