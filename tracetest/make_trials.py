"""Merges labels and writes C4 batches and the key.  python3 tracetest/make_trials.py"""
import glob, json, random, re

OUT = "runs/tracetest"
W = {L: f"runs/words/Qwen__Qwen3.5-0.8B/L{L}.json" for L in (15, 19)}
names = {L: json.load(open(p)) for L, p in W.items()}
got = 0
for p in glob.glob(f"{OUT}/labels_*.json"):
    for row in json.load(open(p)):
        m = re.fullmatch(r"L(\d+)F(\d+)", row["id"])
        names[int(m[1])][m[2]] = dict(fires_on=row["fires_on"], effect=row["effect"], confidence=row["confidence"])
        got += 1
for L, p in W.items():
    json.dump(names[L], open(p, "w"), indent=1, ensure_ascii=False)
print("labeled", got, {L: len(n) for L, n in names.items()})
T = json.load(open(f"{OUT}/reading.json"))
print("trials", len(T))
note = lambda L, f: {"fires_on": names[L][str(f)]["fires_on"], "tends_to_make_it_say": names[L][str(f)]["effect"]}
rng = random.Random(81)
rng.shuffle(T)
shift = T[1:] + T[:1]
other = {(t["chunk"], t["pos"]): s for t, s in zip(T, shift)}
groups = [T[i::6] for i in range(6)]
key, n = {}, 0


def balanced(k):
    v = ["A"] * (k // 2) + ["B"] * (k - k // 2)
    rng.shuffle(v)
    return v


for r in range(6):
    trials = []
    for j, arm in enumerate(("real", "floor", "child_swapped")):
        g = groups[(r + 2 * j) % 6]
        for t, side in zip(g, balanced(len(g))):
            o = other[(t["chunk"], t["pos"])]
            src = o if arm == "floor" else t
            child = o if arm in ("floor", "child_swapped") else t
            a, b = (src["parent"], src["decoy"]) if side == "A" else (src["decoy"], src["parent"])
            n += 1
            i = f"C{n:03d}"
            key[i] = dict(arm=arm, right=side, chunk=t["chunk"], base=t["pos"])
            trials.append(dict(trial=i, context=t["context"], later=note(19, child["child"]), A=note(15, a), B=note(15, b)))
    rng.shuffle(trials)
    json.dump(trials, open(f"{OUT}/read_{r}.json", "w"), indent=1, ensure_ascii=False)
    print(r, len(trials))
json.dump(key, open(f"{OUT}/read_key.json", "w"))
