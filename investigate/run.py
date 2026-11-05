"""Runs one behavior through brain.investigate and saves the report.  .venv/bin/python investigate/run.py hedging"""
import dataclasses, json, sys
import torch
import neurospeak as ns
sys.path.insert(0, "investigate")
from behaviors import behavior, learned

name = sys.argv[1]
second = len(sys.argv) > 2 and sys.argv[2] == "--learned"
fixed = len(sys.argv) > 2 and sys.argv[2] == "--fixed"
model, args = learned(name) if second else behavior(name)
brain = ns.load(model)
r = brain.investigate(name, **args)
out = f"runs/investigate/{name}{'_learned' if second else '_fixed' if fixed else ''}"
print(r)
open(f"{out}.txt", "w").write(str(r))
torch.save(dict(direction=r.direction, control=r.control_direction, layer=r.layer), f"{out}.pt")
json.dump(dict(behavior=name, model=model, layer=r.layer, n=r.n, status=r.status, unedited=r.unedited, removed=r.removed, control=r.control, collapsed=r.collapsed,
               checks=[dataclasses.asdict(c) for c in r.checks], texts=r.texts, scores=r.scores, prompts=args["prompts"]), open(f"{out}.json", "w"), ensure_ascii=False, indent=1)
