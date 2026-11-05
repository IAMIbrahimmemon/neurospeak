"""Adds the words that fall most under Astra's edit, on mining traces only.  .venv/bin/python whyfix/diff_words.py"""
import json

import neurospeak as ns

brain = ns.load()
texts = []
for l in open("runs/whyfix/mine_traces.jsonl"):
    r = json.loads(l)
    ids = brain.tok(r["prompt"] + r["completion"]).input_ids
    mid = len(ids) // 2
    texts += [brain.tok.decode(ids[:256]), brain.tok.decode(ids[mid:mid + 256])]
report = brain.diff("data/astra_p50c_edit.npz", texts)
W = json.load(open("runs/whyfix/words.json"))
W["diff"] = (-report.change).topk(16).indices.tolist()
json.dump(W, open("runs/whyfix/words.json", "w"), indent=1)
print(W["diff"], "f976 in it:", 976 in W["diff"])
