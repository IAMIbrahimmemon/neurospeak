"""Build anonymous scoring trials from finished labels. Each unit gets a real-label trial and a shuffled-label trial, in
different scorer batches. Writes score_batch_*.json and trials_key.json."""
import glob, json, random, sys

D = sys.argv[1] if len(sys.argv) > 1 else "runs/wordtest"
N_BATCHES = 8
rng = random.Random(11)
records = json.load(open(f"{D}/units.json"))["records"]
labels = {x["id"]: x for f in sorted(glob.glob(f"{D}/labels_*.json")) for x in json.load(open(f))}
missing = [r["id"] for r in records if r["id"] not in labels]
assert not missing, missing

groups = {}
for r in records:
    groups.setdefault((r["kind"], r["layer"]), []).append(r)
floor_label, distractor = {}, {}
for g in groups.values():
    ids = [r["id"] for r in g]
    rng.shuffle(ids)
    with_causal = [r for r in g if r["causal"]]
    for i, x in enumerate(ids):
        floor_label[x] = ids[(i + 1) % len(ids)]
    rng.shuffle(with_causal)
    for i, r in enumerate(with_causal):
        distractor[r["id"]] = with_causal[(i + 1) % len(with_causal)]

trials = []
for u, r in enumerate(rng.sample(records, len(records))):
    t = r["test"]
    items = [(s, "positive_top") for s in t["positive_top"]] + [(s, "positive_weak") for s in t["positive_weak"]] \
        + [(s, "same_token") for s in t["same_token"]] + [(s, "random") for s in t["random"]]
    rng.shuffle(items)
    lists, answer = None, None
    if r["causal"]:
        true_list, other = r["causal"]["promotes"], distractor[r["id"]]["causal"]["promotes"]
        answer = rng.choice("AB")
        lists = {"A": true_list, "B": other} if answer == "A" else {"A": other, "B": true_list}
    for cond, source, b in (("real", r["id"], u % N_BATCHES), ("shuffled", floor_label[r["id"]], (u + N_BATCHES // 2) % N_BATCHES)):
        L = labels[source]
        trials.append({"cond": cond, "unit": r["id"], "batch": b, "answer": answer, "types": [k for _, k in items],
                       "label": {"fires_on": L["fires_on"], "effect": L["effect"]},
                       "snippets": [s for s, _ in items], "token_lists": lists})
rng.shuffle(trials)
key = {}
for b in range(N_BATCHES):
    json.dump([], open(f"{D}/score_batch_{b}.json", "w"))
out = {b: [] for b in range(N_BATCHES)}
for i, t in enumerate(trials):
    tid = f"T{i:03d}"
    key[tid] = {k: t[k] for k in ("cond", "unit", "answer", "types")}
    item = {"trial": tid, "label": t["label"], "snippets": t["snippets"]}
    if t["token_lists"]: item["token_lists"] = t["token_lists"]
    out[t["batch"]].append(item)
for b, part in out.items():
    json.dump(part, open(f"{D}/score_batch_{b}.json", "w"), ensure_ascii=False, indent=1)
json.dump(key, open(f"{D}/trials_key.json", "w"), indent=1)
print(len(trials), "trials in", N_BATCHES, "batches", {b: len(p) for b, p in out.items()})
