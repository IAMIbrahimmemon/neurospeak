"""Anonymous trials for PLAN_EFFECT.md. Writes effect_batch_*.json and effect_key.json."""
import glob, json, random, sys

D = sys.argv[1] if len(sys.argv) > 1 else "runs/wordtest"
N_BATCHES, MIN_LEFT = 6, 3
rng = random.Random(41)
records = {r["id"]: r for r in json.load(open(f"{D}/units.json"))["records"]}
effect = json.load(open(f"{D}/effect.json"))
control = json.load(open(f"{D}/control.json"))
labels = {x["id"]: x for f in sorted(glob.glob(f"{D}/labels_*.json")) for x in json.load(open(f))}

groups = {}
for x, e in effect.items():
    groups.setdefault((e["kind"], e["layer"]), []).append(x)
other, third = {}, {}
for ids in groups.values():
    rng.shuffle(ids)
    for i, x in enumerate(ids):
        other[x], third[x] = ids[(i + 1) % len(ids)], ids[(i + 2) % len(ids)]

trials = []
def add(cond, unit, sentence_of, a, b, offset):
    answer = rng.choice("AB")
    lists = {"A": a, "B": b} if answer == "A" else {"A": b, "B": a}
    trials.append({"cond": cond, "unit": unit, "answer": answer, "batch": (order[unit] + offset) % N_BATCHES,
                   "effect": labels[sentence_of]["effect"], "token_lists": lists})

order = {x: i for i, x in enumerate(rng.sample(sorted(effect), len(effect)))}
skipped = 0
for x, e in effect.items():
    d = other[x]
    add("steer_real", x, x, e["steer"], effect[d]["steer"], 0)
    add("steer_null", x, x, e["steer_random"], effect[d]["steer_random"], 1)
    add("steer_shuffled", x, third[x], e["steer"], effect[d]["steer"], 2)
    if x in control:
        real, rand = records[x]["causal"]["promotes"], control[x]["random_removal"]
        a, b = [t for t in real if t not in rand], [t for t in rand if t not in real]
        if min(len(a), len(b)) >= MIN_LEFT:
            add("remove_real", x, x, a, b, 3)
            add("remove_shuffled", x, third[x], a, b, 4)
        else:
            skipped += 1
rng.shuffle(trials)
key, out = {}, {b: [] for b in range(N_BATCHES)}
for i, t in enumerate(trials):
    tid = f"E{i:04d}"
    key[tid] = {"cond": t["cond"], "unit": t["unit"], "answer": t["answer"]}
    out[t["batch"]].append({"trial": tid, "effect": t["effect"], "token_lists": t["token_lists"]})
for b, part in out.items():
    json.dump(part, open(f"{D}/effect_batch_{b}.json", "w"), ensure_ascii=False, indent=0)
json.dump(key, open(f"{D}/effect_key.json", "w"), indent=1)
print(len(trials), "trials; skipped", skipped, {b: len(p) for b, p in out.items()})
