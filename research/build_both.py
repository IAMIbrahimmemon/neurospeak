"""Directions for PLAN_BOTH.md: repetition orthogonal to self-doubt, and two orthogonal controls."""
import json
import torch, torch.nn.functional as F
from neurospeak.investigate import matched_control
D = {k: v.float() for k, v in torch.load("runs/whyfix/directions.pt", map_location="cpu").items()}
r, s, m = D["dir_neurospeak"], D["dir_self_doubt"], D["dir_mean"]
rep = F.normalize(r - (r @ s) * s, dim=0)
m1 = matched_control(rep, m, seed=2)
c2 = matched_control(s, m, seed=3)
m2 = F.normalize(c2 - (c2 @ m1) * m1, dim=0)
torch.save(dict(dir_both_rep=rep, dir_both_m1=m1, dir_both_m2=m2), "runs/research/both_dirs.pt")
out = dict(rep_s=(rep @ s).item(), m1_m2=(m1 @ m2).item(), mean_rep=(rep @ m).item(), mean_s=(s @ m).item(), mean_m1=(m1 @ m).item(), mean_m2=(m2 @ m).item(), rep_m1=(rep @ m1).item(), s_m2=(s @ m2).item())
json.dump(out, open("runs/research/both_dirs.json", "w"), indent=1)
print(out)
