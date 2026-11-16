"""Mean of block 19's output over the default corpus, for the centered and offset arms of PLAN_SPLIT.md (study 23)."""
import json
import torch
import torch.nn.functional as F
import neurospeak as ns
from neurospeak import core

L = 19
brain = ns.load("Qwen/Qwen3.5-0.8B")
u = F.normalize(torch.load("runs/whyfix/directions.pt")["dir_self_doubt"].float(), dim=0)
acts = []
h = brain.anatomy.blocks[L].register_forward_hook(lambda _m, _i, o: acts.append((o[0] if isinstance(o, tuple) else o)[0, 1:].detach().float().cpu()))
with torch.no_grad():
    for t in core._default_corpus()[:24]:
        brain.run(t[:2000], capture=False)
h.remove()
H = torch.cat(acts)
mu = H.mean(0)
p = H @ u
info = dict(tokens=len(H), mu_dot_d=(mu @ u).item(), mu_norm=mu.norm().item(), cos_mu_d=F.cosine_similarity(mu, u, dim=0).item(),
            proj_mean=p.mean().item(), proj_abs_mean=p.abs().mean().item(), proj_std=p.std().item())
torch.save({"mean": mu}, "runs/research/split_mean.pt")
json.dump(info, open("runs/research/split_mean.json", "w"), indent=1)
print(json.dumps(info, indent=1))
