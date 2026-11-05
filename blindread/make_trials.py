"""Step 5 — build anonymized multiple-choice trials (blind / ceiling / shuffled), 9 scorer batches + a separate key."""
import glob, json, random
rng = random.Random(42)
truth = {x["id"]: x for x in json.load(open("truth.json"))}
load = lambda pat: {p["id"]: p for f in sorted(glob.glob(pat)) for p in json.load(open(f))}
blind, ceil = load("pred_blind_*.json"), load("pred_ceiling_*.json")
assert len(blind) == len(ceil) == 60
band = {}
for i, t in truth.items(): band.setdefault(t["band"], []).append(i)

def option(i):
    t = truth[i]["test_half"]
    return {"strong_positive_contexts": [c["text"] for c in t["positive"]],
            "strong_negative_contexts": [c["text"] for c in t["negative"]]}

trials, key = [], {}
for x, t in truth.items():
    others = [o for o in band[t["band"]] if o != x]
    distract = rng.sample(others, 4)
    opts = [x] + distract; rng.shuffle(opts)
    letters = "ABCDE"
    outside = [o for o in others if o not in distract]
    z = rng.choice(outside)
    for cond, pred in (("blind", blind[x]), ("ceiling", ceil[x]), ("shuffled", blind[z])):
        trials.append({"cond": cond, "neuron": x, "band": t["band"], "answer": letters[opts.index(x)],
                       "desc": {"positive": pred["positive"], "negative": pred["negative"]},
                       "options": {letters[k]: option(o) for k, o in enumerate(opts)},
                       "reader_confidence": pred.get("confidence")})
rng.shuffle(trials)
for k, tr in enumerate(trials):
    tid = f"T{k:03d}"
    key[tid] = {f: tr[f] for f in ("cond", "neuron", "band", "answer", "reader_confidence")}
    tr["tid"] = tid
for b in range(9):
    json.dump([{"trial": tr["tid"], "description": tr["desc"], "options": tr["options"]} for tr in trials[b*20:(b+1)*20]],
              open(f"score_batch_{b}.json", "w"), ensure_ascii=False, indent=1)
json.dump(key, open("trials_key.json", "w"), indent=1)
print(len(trials), "trials in 9 batches")
