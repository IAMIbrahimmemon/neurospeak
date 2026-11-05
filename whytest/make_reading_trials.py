"""Merges labels into the word file, then writes reader batches and the key.  python3 whytest/make_reading_trials.py"""
import glob, json, random

OUT, WORDS = "runs/whytest", "runs/words/Qwen__Qwen3.5-0.8B/L19.json"
names = json.load(open(WORDS))
asked = sum(len(json.load(open(p))) for p in glob.glob(f"{OUT}/label_batch_*.json"))
got = 0
for p in glob.glob(f"{OUT}/labels_*.json"):
    for row in json.load(open(p)):
        got += 1
        names[row["id"][1:]] = dict(fires_on=row["fires_on"], effect=row["effect"], confidence=row["confidence"])
json.dump(names, open(WORDS, "w"), indent=1, ensure_ascii=False)
print("asked", asked, "labeled", got, "total names", len(names))

note = lambda f: {"fires_on": names[str(f)]["fires_on"], "tends_to_make_it_say": names[str(f)]["effect"]}
D = json.load(open(f"{OUT}/reading.json"))
rng = random.Random(41)
r1 = [t for t in D["r1"] if str(t["driver"]) in names and str(t["decoy"]) in names]
r2 = [dict(t, words=[f for f in t["words"] if str(f) in names]) for t in D["r2"]]
print("R1 trials", len(r1), "of", len(D["r1"]))
key, n = {}, 0


def tid():
    global n
    n += 1
    return f"T{n:03d}"


def balanced(k):
    v = ["A"] * (k // 2) + ["B"] * (k - k // 2)
    rng.shuffle(v)
    return v


def cause(t, other, arm, right):
    a, b = (t["driver"], t["decoy"]) if right == "A" else (t["decoy"], t["driver"])
    if other:
        a, b = (other["driver"], other["decoy"]) if right == "A" else (other["decoy"], other["driver"])
    i = tid()
    key[i] = dict(test="R1", arm=arm, right=right, base=t["pos"], chunk=t["chunk"])
    return dict(trial=i, kind="cause", context=t["context"], answer=t["answer"], against=t["against"], A=note(a), B=note(b))


def predict(t, words, arm, right):
    a, b = (t["answer"], t["against"]) if right == "A" else (t["against"], t["answer"])
    i = tid()
    key[i] = dict(test="R2", arm=arm, right=right, base=t["pos"], chunk=t["chunk"])
    out = dict(trial=i, kind="predict", context=t["context"], A=a, B=b)
    if words is not None:
        out["words"] = [note(f) for f in words]
    return out


# R1: four readers. A reader never sees the same trial under both arms.
rng.shuffle(r1)
groups = [r1[i::4] for i in range(4)]
others = r1[1:] + r1[:1]
other_of = {(t["chunk"], t["pos"]): o for t, o in zip(r1, others)}
batches = {}
for r in range(4):
    real, floor = groups[r], groups[(r + 1) % 4]
    sides = balanced(len(real)) + balanced(len(floor))
    trials = [cause(t, None, "real", s) for t, s in zip(real, sides)]
    trials += [cause(t, other_of[(t["chunk"], t["pos"])], "floor", s) for t, s in zip(floor, sides[len(real):])]
    rng.shuffle(trials)
    batches[f"r1_{r}"] = trials

# R2: six readers. Each decision goes to three readers, one per arm.
rng.shuffle(r2)
groups = [r2[i::6] for i in range(6)]
shift = r2[7:] + r2[:7]
words_of = {(t["chunk"], t["pos"]): s["words"] for t, s in zip(r2, shift)}
for r in range(6):
    none, real, other = groups[r], groups[(r + 2) % 6], groups[(r + 4) % 6]
    trials = []
    for arm, g, words in (("none", none, lambda t: None), ("real", real, lambda t: t["words"]), ("other", other, lambda t: words_of[(t["chunk"], t["pos"])])):
        trials += [predict(t, words(t), arm, s) for t, s in zip(g, balanced(len(g)))]
    rng.shuffle(trials)
    batches[f"r2_{r}"] = trials

for name, trials in batches.items():
    json.dump(trials, open(f"{OUT}/read_{name}.json", "w"), indent=1, ensure_ascii=False)
    print(name, len(trials))
json.dump(key, open(f"{OUT}/read_key.json", "w"))
