"""Dose-matched random controls for the L19 self-doubt direction, for PLAN_DOSECTL.md (study 22)."""
import json
import torch
import torch.nn.functional as F
import neurospeak as ns
from neurospeak import core

L, WANT, TOL, CHECK_TOL = 19, 4, 0.05, 0.25
brain = ns.load("Qwen/Qwen3.5-0.8B")
u = F.normalize(torch.load("runs/whyfix/directions.pt")["dir_self_doubt"].float(), dim=0)


def block_out(texts):
    acts = []
    h = brain.anatomy.blocks[L].register_forward_hook(lambda _m, _i, o: acts.append((o[0] if isinstance(o, tuple) else o)[0, 1:].detach().float().cpu()))
    with torch.no_grad():
        for t in texts:
            brain.run(t, capture=False)
    h.remove()
    return torch.cat(acts)


dose = lambda H, d: ((H @ d).abs() / H.norm(dim=-1)).mean().item()
H = block_out([t[:2000] for t in core._default_corpus()[:24]])
gsm = [json.loads(x) for x in open("runs/research/layers_none_traces.jsonl")][:6]
G = block_out([r["prompt"] + r["completion"][:3000] for r in gsm])
mean = F.normalize(H.mean(0), dim=0)
evals, evecs = torch.linalg.eigh(torch.cov(H.T))
evals = evals.clamp(min=0)
target, target_gsm = dose(H, u), dose(G, u)


def control(seed, alpha):
    z = torch.randn(len(u), generator=torch.Generator().manual_seed(seed))
    d = evecs @ (evals ** alpha * (evecs.T @ z))
    for b in (u, mean):
        d = d - (d @ b) * b
    d = F.normalize(d, dim=0)
    return F.normalize(d - (d @ u) * u, dim=0)


out, info, seed = {}, dict(target_corpus=target, target_gsm=target_gsm, tokens_corpus=len(H), tokens_gsm=len(G), skipped=[]), 101
while len(out) < WANT:
    lo, hi = 0.0, 2.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dose(H, control(seed, mid)) < target else (lo, mid)
    d = control(seed, (lo + hi) / 2)
    row = dict(seed=seed, alpha=(lo + hi) / 2, dose_corpus=dose(H, d), dose_gsm=dose(G, d), cos_doubt=(d @ u).item(), cos_mean=(d @ mean).item())
    ok = abs(row["dose_corpus"] / target - 1) <= TOL and abs(row["dose_gsm"] / target_gsm - 1) <= CHECK_TOL
    if ok:
        out[f"dir_dosectl_{seed}"] = d
        info[f"dir_dosectl_{seed}"] = row
    else:
        info["skipped"].append(row)
    seed += 1
    assert seed < 140, "could not build 4 controls"
torch.save(out, "runs/research/dosectl_dirs.pt")
json.dump(info, open("runs/research/dosectl_dirs.json", "w"), indent=1)
print(json.dumps(info, indent=1))
