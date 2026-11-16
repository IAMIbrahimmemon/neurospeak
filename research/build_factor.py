"""Study 30 Part A, see PLAN_FACTOR.md. Builds the doubt-only and conclude-only directions after checking the same code rebuilds d, then reports cosines and a fit of d on the two."""
import json, sys
import torch
import neurospeak as ns
from neurospeak import core
sys.path.insert(0, "investigate")
from behaviors import PAIRS

brain = ns.load("Qwen/Qwen3.5-0.8B")
L = 19


def build(name, pairs):
    same = [core.Pair(p.is_, p.not_, None, brain.chat_prompt(p.ask, True)) for p in pairs]
    return brain.concept(name, same, at=[L]).dirs[L].cpu().float()


D = torch.load("runs/whyfix/directions.pt")
d = D["dir_self_doubt"].float()
c = (build("check", PAIRS["self_doubt"]) @ d).item()
print("rebuild cosine", round(c, 5))
if c < 0.999:
    raise SystemExit("the build doesn't reproduce the original direction")
P = json.load(open("research/factor_pairs.json"))
new = {f"dir_factor_{k}": build(k, [ns.pair(ask=r["ask"], is_=r["is_"], not_=r["not_"]) for r in v]) for k, v in P.items()}
torch.save(new, "runs/research/factor_dirs.pt")
a, b = new["dir_factor_doubt_only"], new["dir_factor_conclude_only"]
X = torch.stack([a, b], 1)
w = torch.linalg.lstsq(X, d[:, None]).solution[:, 0]
fit = X @ w
out = dict(rebuild=c, norms={k: v.norm().item() for k, v in new.items()},
           cos_d={k: (v @ d / v.norm() / d.norm()).item() for k, v in new.items()},
           cos_rep={k: (v @ D["dir_neurospeak"].float() / v.norm()).item() for k, v in new.items()},
           cos_each_other=(a @ b / a.norm() / b.norm()).item(),
           fit_weights=dict(doubt_only=w[0].item(), conclude_only=w[1].item()), fit_r2=1 - ((d - fit).norm() ** 2 / d.norm() ** 2).item())
json.dump(out, open("runs/research/factor_cosines.json", "w"), indent=1)
print(json.dumps(out, indent=1))
