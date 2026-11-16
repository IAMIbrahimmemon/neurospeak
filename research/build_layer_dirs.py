"""Builds the self-doubt direction and a mean-matched random control at each layer for PLAN_LAYERS.md, after checking L19 rebuilds the original."""
import json, sys
import torch
import neurospeak as ns
from neurospeak import core
from neurospeak.investigate import matched_control, mean_residual
sys.path.insert(0, "investigate")
from behaviors import PAIRS

LAYERS = (3, 7, 11, 15, 19, 23)
brain = ns.load("Qwen/Qwen3.5-0.8B")
same = [core.Pair(p.is_, p.not_, None, brain.chat_prompt(p.ask, True)) for p in PAIRS["self_doubt"]]
c = brain.concept("self_doubt_layers", same, at=list(LAYERS))
D = torch.load("runs/whyfix/directions.pt")
orig = D["dir_self_doubt"].float()
rebuild = (c.dirs[19].cpu().float() @ orig).item()
print("rebuild cosine", round(rebuild, 5))
if rebuild < 0.999:
    raise SystemExit("the build doesn't reproduce the original direction")
normal = core._default_corpus()[:24]
out, info = {}, {}
for l in LAYERS:
    u = c.dirs[l].cpu().float()
    m = mean_residual(brain, l, normal).cpu().float()
    out[f"dir_layer_{l}"] = u
    out[f"dir_layer_{l}_matched"] = matched_control(u, m, seed=1)
    info[l] = dict(mean_overlap=(u @ torch.nn.functional.normalize(m, dim=0)).item(), cos_with_L19=(u @ orig).item(), control_cos=(out[f"dir_layer_{l}_matched"] @ u).item())
info["mean_L19_vs_dir_mean"] = (torch.nn.functional.normalize(mean_residual(brain, 19, normal).cpu().float(), dim=0) @ D["dir_mean"].float()).item()
torch.save(out, "runs/research/layer_dirs.pt")
json.dump(info, open("runs/research/layer_dirs.json", "w"), indent=1)
print(json.dumps(info, indent=1))
