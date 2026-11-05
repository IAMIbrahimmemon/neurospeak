"""Saves each loop edit's top L19 direction, and one random direction, for the direction arms of generate.py."""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from neurospeak.anatomy import discover
from neurospeak.core import load_edit_file

m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype="auto")
an = discover(m)
params = dict(m.named_parameters())
out = {}
for arm, path in (("dir_neurospeak", "runs/neurospeak_repetition_edit.pt"), ("dir_astra", "data/astra_p50c_edit.npz")):
    for name, Wn in load_edit_file(m, path).items():
        if an.locate(name)[0] == 19:
            W0 = an.columns(name, params[name].detach()).float().cpu()
            D = an.columns(name, Wn).float().cpu() - W0
            cols = (D.abs().sum(0) > 0).nonzero()[:, 0]
            out[arm] = F.normalize(torch.linalg.svd(D[:, cols], full_matrices=False)[0][:, 0], dim=0)
torch.manual_seed(1)
out["dir_random"] = F.normalize(torch.randn(an.hidden_size), dim=0)
torch.save(out, "runs/whyfix/directions.pt")
print({k: round(v.norm().item(), 3) for k, v in out.items()}, round((out["dir_astra"] @ out["dir_neurospeak"]).abs().item(), 3))
