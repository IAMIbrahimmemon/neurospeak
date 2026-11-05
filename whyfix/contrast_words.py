"""Adds contrast words from the mining traces: before sustained loops vs earlier in the same traces.  .venv/bin/python whyfix/contrast_words.py"""
import json

import neurospeak as ns
from whyfix.mine import onsets

brain = ns.load()
rows = [json.loads(l) for l in open("runs/base_traces.jsonl")][16:] + [json.loads(l) for l in open("runs/whyfix/mine_traces.jsonl")]
loops, other = [], []
for r in rows:
    ids = brain.tok(r["prompt"] + r["completion"]).input_ids
    prompt = len(brain.tok(r["prompt"]).input_ids)
    points = [p for p in onsets(ids) if p > prompt]
    if not points:
        continue
    for p in points:
        loops.append(brain.tok.decode(ids[max(prompt, p - 128):p]))
    if points[0] - prompt >= 256:
        ends = points[0] - 128
        other.append(brain.tok.decode(ids[max(prompt, ends - 512):ends]))
print("loop windows", len(loops), "earlier windows", len(other))
report = brain.contrast(loops, other)
W = json.load(open("runs/whyfix/words.json"))
W["contrast2"] = report.change.topk(16).indices.tolist()
json.dump(W, open("runs/whyfix/words.json", "w"), indent=1)
print(W["contrast2"], "f976 in it:", 976 in W["contrast2"], "overlap active2:", len(set(W["contrast2"]) & set(W["active2"])), "overlap why2:", len(set(W["contrast2"]) & set(W["why2"])))
