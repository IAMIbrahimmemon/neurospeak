"""Scores difftest/PLAN.md.  .venv/bin/python difftest/run.py"""
import json
import torch

import neurospeak as ns

brain = ns.load()
rows = [json.loads(l) for l in open("runs/base_traces.jsonl")][:16]
texts = []
for r in rows:
    ids = brain.tok(r["prompt"] + r["completion"]).input_ids
    mid = len(ids) // 2
    texts += [brain.tok.decode(ids[:256]), brain.tok.decode(ids[mid:mid + 256])]
edits = {"astra": "data/astra_p50c_edit.npz", "neurospeak": "runs/neurospeak_repetition_edit.pt", "random": "runs/control_random_edit.pt"}
reports = {k: brain.diff(p, texts) for k, p in edits.items()}


def spearman(a, b):
    rank = lambda v: v.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([rank(a), rank(b)]))[0, 1])


a, n, r = (reports[k].change for k in ("astra", "neurospeak", "random"))
top = (a.abs() + n.abs()).topk(200).indices
out = dict(tokens=reports["astra"].tokens, D1_astra_vs_neurospeak=spearman(a[top], n[top]), D2_random_vs_astra=spearman(r[top], a[top]), D2_random_vs_neurospeak=spearman(r[top], n[top]),
           top20_share={k: reports[k].top20_share for k in edits}, total_abs_change={k: float(reports[k].change.abs().sum()) for k in edits})
json.dump(out, open("runs/difftest/results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
with open("runs/difftest/report.txt", "w") as f:
    for k in edits:
        f.write(f"== {k}\n{reports[k]}\n")
