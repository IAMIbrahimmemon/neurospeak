"""Adds the control and the overlap-removed arm for PLAN_SELF_DOUBT.md to directions.pt, keeping every existing direction."""
import torch, torch.nn.functional as F
from neurospeak.investigate import matched_control

p = "runs/whyfix/directions.pt"
D = torch.load(p)
s, n, m = (D[k].float() for k in ("dir_self_doubt", "dir_neurospeak", "dir_mean"))
D["dir_self_doubt_matched"] = matched_control(s, m, seed=1)
D["dir_self_doubt_orth"] = F.normalize(s - (s @ n) * n, dim=0)
torch.save(D, p)
for k in ("dir_self_doubt", "dir_self_doubt_matched", "dir_self_doubt_orth"):
    print(k, "norm", round(D[k].norm().item(), 4), "mean", round((D[k] @ m).item(), 4), "repetition", round((D[k] @ n).item(), 4), "self_doubt", round((D[k] @ s).item(), 4))
