"""Builds the repetition direction and its control on MobileLLM, see PLAN_REP_PORT.md."""
import json, sys
import torch
import neurospeak as ns
from neurospeak import core
from neurospeak.investigate import matched_control
sys.path.insert(0, "investigate")
from behaviors import REPETITION_PAIRS

L = 12
brain = ns.load("facebook/MobileLLM-R1-360M-base")
same = [core.Pair(p.is_, p.not_, None, brain.chat_prompt(p.ask, True)) for p in REPETITION_PAIRS]
d = brain.concept("repetition", same, at=[L]).dirs[L].cpu().float()
total, n = torch.zeros_like(d), 0
for line in open("runs/research/rport_none_traces.jsonl"):
    r = json.loads(line)
    _, hs, _, _ = brain.run(r["prompt"] + " ".join(r["completion"].split()[:300]))
    h = hs[L + 1].cpu().float()
    total += h.sum(0); n += len(h)
mean = total / n
m = matched_control(d, mean, seed=1)
torch.save(dict(rep=d, rep_matched=m, mean=mean), "runs/research/port_dirs.pt")
out = dict(cos_mean=(torch.nn.functional.normalize(mean, dim=0) @ d).item(), control_cos_mean=(torch.nn.functional.normalize(mean, dim=0) @ m).item(), control_cos_rep=(m @ d).item())
json.dump(out, open("runs/research/rport_dirs.json", "w"), indent=1)
print(out)
