"""With no bar: can each word's general effect pick the model's choice with no reader?  .venv/bin/python whytest/auto_reader.py"""
import json, random, statistics as st
import torch

import neurospeak as ns
from neurospeak.core import DEV
from neurospeak.why import attribute

S = torch.load("runs/wordtest/scan.pt", map_location="cpu")
brain = ns.load(dtype=torch.float32)
D = json.load(open("runs/whytest/round2/reading.json"))["r2"]
V = torch.load("runs/whytest/round2/word_effect_vectors.pt")
by_key = {(r["chunk"], r["pos"]): r for r in map(json.loads, open("runs/whytest/L19.jsonl"))}
right, shuffled = [], []
rng = random.Random(3)
order = list(D)
rng.shuffle(order)
for t, other in zip(D, order):
    r = by_key[(t["chunk"], t["pos"])]
    ids = S["tokens"][t["chunk"], t["pos"] - 127:t["pos"] + 1].long().to(DEV)[None]
    vals, idx, scores = attribute(brain, ids, 19, r["answer"], r["against"])
    k = scores.shape[1]
    best = scores.view(-1).abs().topk(5).indices.tolist()
    words = [(int(idx[i // k, i % k]), float(vals[i // k, i % k])) for i in best]
    score = sum(v * float(V[f][r["answer"]] - V[f][r["against"]]) for f, v in words if f in V)
    fake = sum(float(V[f][r["answer"]] - V[f][r["against"]]) for f in other["words"] if f in V)
    right.append(float(score > 0))
    shuffled.append(float(fake > 0))
out = dict(n=len(right), auto_real=st.mean(right), auto_other_decisions_words=st.mean(shuffled))
json.dump(out, open("runs/whytest/round2/auto_reader.json", "w"))
print(out)
