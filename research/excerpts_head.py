"""H7 for study 18: 20 runs per arm, 150 words from the middle of each completion, shuffled and blinded for a Claude reader. See PLAN_HEAD.md.

Usage: python research/excerpts_head.py OUTDIR   (writes head_key.json and head_blind.txt; the key stays out of the reader's sight)
After the reader: python research/excerpts_head.py OUTDIR LABELS.json   (LABELS maps item number to coherent/repetitive/garbled)
"""
import json, random, sys

ARMS = ("astra_pp", "none_pp", "doubt_pp", "astra_doubt_pp", "matched_pp", "none")
OUT = sys.argv[1]

if len(sys.argv) > 2:
    key = json.load(open(f"{OUT}/head_key.json"))
    labels = {int(k): v for k, v in json.load(open(sys.argv[2])).items()}
    assert set(labels) == {x["item"] for x in key}, "every item needs exactly one label"
    out = {}
    for a in ARMS:
        got = [labels[x["item"]] for x in key if x["arm"] == a]
        out[a] = dict(coherent=got.count("coherent"), repetitive=got.count("repetitive"), garbled=got.count("garbled"), n=len(got))
        out[a]["H7_pass"] = out[a]["garbled"] <= 2
    json.dump(out, open("runs/research/head_reader.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    sys.exit()

rng = random.Random(18)
items = []
for a in ARMS:
    runs = [json.loads(l) for l in open(f"runs/research/head_{a}_traces.jsonl")]
    for r in rng.sample(runs, 20):
        w = r["completion"].split()
        start = max(0, len(w) // 2 - 75)
        items.append(dict(arm=a, id=r["id"], text=" ".join(w[start:start + 150])))
rng.shuffle(items)
for i, x in enumerate(items):
    x["item"] = i
json.dump(items, open(f"{OUT}/head_key.json", "w"), indent=1)
with open(f"{OUT}/head_blind.txt", "w") as f:
    for x in items:
        f.write(f"=== ITEM {x['item']} ===\n{x['text']}\n\n")
print(len(items), "excerpts")
