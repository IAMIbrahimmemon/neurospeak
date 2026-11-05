"""Replication of the three steering conditions from PLAN_EFFECT.md, with new pairings and balanced A/B."""
import glob, json, random, sys

D = sys.argv[1] if len(sys.argv) > 1 else "runs/wordtest"
N_BATCHES = 7
rng = random.Random(77)
effect = json.load(open(f"{D}/effect.json"))
labels = {x["id"]: x for f in sorted(glob.glob(f"{D}/labels_*.json")) for x in json.load(open(f))}
groups = {}
for x, e in effect.items():
    groups.setdefault((e["kind"], e["layer"]), []).append(x)

trials = []
order = {x: i for i, x in enumerate(rng.sample(sorted(effect), len(effect)))}
for ids in groups.values():
    for x in ids:
        others = [o for o in ids if o != x]
        for rep, d in enumerate(rng.sample(others, 2)):
            first = rng.choice("AB") if rep == 0 else ("B" if first == "A" else "A")
            third = rng.choice([o for o in others if o != d])
            for offset, (cond, source, key_) in enumerate((("steer_real", x, "steer"), ("steer_null", x, "steer_random"), ("steer_shuffled", third, "steer"))):
                a, b = effect[x][key_], effect[d][key_]
                lists = {"A": a, "B": b} if first == "A" else {"A": b, "B": a}
                trials.append({"cond": cond, "unit": x, "answer": first, "batch": (order[x] + offset * 2 + rep * 3) % N_BATCHES,
                               "effect": labels[source]["effect"], "token_lists": lists})
rng.shuffle(trials)
key, out = {}, {b: [] for b in range(N_BATCHES)}
for i, t in enumerate(trials):
    tid = f"R{i:04d}"
    key[tid] = {"cond": t["cond"], "unit": t["unit"], "answer": t["answer"]}
    out[t["batch"]].append({"trial": tid, "effect": t["effect"], "token_lists": t["token_lists"]})
for b, part in out.items():
    json.dump(part, open(f"{D}/replication_batch_{b}.json", "w"), ensure_ascii=False, indent=0)
json.dump(key, open(f"{D}/replication_key.json", "w"), indent=1)
print(len(trials), {b: len(p) for b, p in out.items()})
