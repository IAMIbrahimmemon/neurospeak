"""Builds reader batches from saved runs. See PLAN_READER.md."""
import json, os, random

D, OUT, TRIALS, READERS = "runs/investigate", "runs/investigate/reader", 14, 8
BEHAVIORS = {"calibration": "repetition", "chinese": "answering in Chinese", "self_doubt": "self-doubt"}
CONDITIONS = ["none", "wrong", "name", "evidence"]
rng = random.Random(2026)
tail = lambda t: " ".join(t.split(" "))[-500:]
runs, sheets, picks = {}, {}, {}
for b in BEHAVIORS:
    r = runs[b] = json.load(open(f"{D}/{b}.json"))
    n = len(r["prompts"])
    moved = sorted(range(n), key=lambda i: r["scores"]["removed"][i] - r["scores"]["unedited"][i])[:2]
    sheets[b] = "\n\n".join(f"Example {k + 1}. Prompt: {r['prompts'][i][:300]}\nWith the direction, score {r['scores']['unedited'][i]:.3g}, ending: ...{tail(r['texts']['unedited'][i])[-300:]}\nWithout it, score {r['scores']['removed'][i]:.3g}, ending: ...{tail(r['texts']['removed'][i])[-300:]}" for k, i in enumerate(moved))
    picks[b] = rng.sample([i for i in range(n) if i not in moved], TRIALS)

names = list(BEHAVIORS)
trials = []
for b in names:
    for i in picks[b]:
        trials.append(dict(behavior=b, index=i))
rng.shuffle(trials)
batches, key = [[] for _ in range(READERS)], {}
tid = 0
for k, t in enumerate(trials):
    readers = rng.sample(range(READERS), len(CONDITIONS))    # four different readers for one trial's four conditions
    for reader, cond in zip(readers, CONDITIONS):
        batches[reader].append((t, cond))
sides = {c: [] for c in CONDITIONS}
for c in CONDITIONS:
    sides[c] = ["A", "B"] * (len(trials) // 2) + (["A"] if len(trials) % 2 else [])
    rng.shuffle(sides[c])
pos = {c: 0 for c in CONDITIONS}
os.makedirs(OUT, exist_ok=True)
for reader, batch in enumerate(batches):
    rng.shuffle(batch)
    rows = []
    for t, cond in batch:
        tid += 1
        r = runs[t["behavior"]]
        i = t["index"]
        side = sides[cond][pos[cond]]; pos[cond] += 1
        a, b = (r["texts"]["removed"][i], r["texts"]["unedited"][i]) if side == "A" else (r["texts"]["unedited"][i], r["texts"]["removed"][i])
        other = names[(names.index(t["behavior"]) + 1) % len(names)]
        note = {"none": None, "name": f"This direction was built to capture {BEHAVIORS[t['behavior']]}.", "wrong": "Evidence about this direction:\n" + sheets[other], "evidence": "Evidence about this direction:\n" + sheets[t["behavior"]]}[cond]
        id_ = f"T{tid:03d}"
        rows.append(dict(id=id_, note=note, prompt=r["prompts"][i][:300], A="..." + tail(a), B="..." + tail(b)))
        key[id_] = dict(right=side, condition=cond, behavior=t["behavior"], index=i, reader=reader)
    json.dump(rows, open(f"{OUT}/batch_{reader}.json", "w"), ensure_ascii=False, indent=1)
json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
print({i: len(b) for i, b in enumerate(batches)}, {c: sum(1 for k in key.values() if k['condition'] == c and k['right'] == 'A') for c in CONDITIONS})
