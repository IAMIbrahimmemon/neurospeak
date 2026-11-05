"""C4b: names only.  python3 tracetest/make_trials_b.py"""
import json, os, random

OUT = "runs/tracetest/b"
os.makedirs(OUT, exist_ok=True)
names = {L: json.load(open(f"runs/words/Qwen__Qwen3.5-0.8B/L{L}.json")) for L in (15, 19)}
T = json.load(open("runs/tracetest/reading.json"))
note = lambda L, f: {"fires_on": names[L][str(f)]["fires_on"], "tends_to_make_it_say": names[L][str(f)]["effect"]}
rng = random.Random(83)
rng.shuffle(T)
other = {(t["chunk"], t["pos"]): s for t, s in zip(T, T[1:] + T[:1])}
groups = [T[i::4] for i in range(4)]
key, n = {}, 0


def balanced(k):
    v = ["A"] * (k // 2) + ["B"] * (k - k // 2)
    rng.shuffle(v)
    return v


for r in range(4):
    trials = []
    for arm, g in (("real", groups[r]), ("swapped", groups[(r + 1) % 4])):
        for t, side in zip(g, balanced(len(g))):
            child = t if arm == "real" else other[(t["chunk"], t["pos"])]
            a, b = (t["parent"], t["decoy"]) if side == "A" else (t["decoy"], t["parent"])
            n += 1
            i = f"C{n:03d}"
            key[i] = dict(arm=arm, right=side, chunk=t["chunk"], base=t["pos"])
            trials.append(dict(trial=i, later=note(19, child["child"]), A=note(15, a), B=note(15, b)))
    rng.shuffle(trials)
    json.dump(trials, open(f"{OUT}/read_{r}.json", "w"), indent=1, ensure_ascii=False)
    print(r, len(trials))
json.dump(key, open(f"{OUT}/read_key.json", "w"))
