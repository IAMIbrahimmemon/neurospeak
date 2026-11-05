"""Next-token loss on held-out text with a direction removed at L19, against no removal. Bars N1 and N2 in PLAN_SPREAD.md."""
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from neurospeak.anatomy import discover
from neurospeak.core import DEV

S = torch.load("runs/wordtest/scan.pt", map_location="cpu")
D = torch.load("runs/whyfix/directions.pt")
m = AutoModelForCausalLM.from_pretrained(S["model"], dtype=torch.float32).to(DEV).eval()
block = discover(m).blocks[19]
BATCH, SKIP = 8, 2


def losses(T, name):
    handle = None
    if name:
        d = D[name].to(DEV)

        def hook(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            new = h - (h @ d).unsqueeze(-1) * d
            return (new, *o[1:]) if isinstance(o, tuple) else new
        handle = block.register_forward_hook(hook)
    out = []
    with torch.no_grad():
        for s in range(0, len(T), BATCH):
            b = T[s:s + BATCH].long().to(DEV)
            l = F.cross_entropy(m(b, use_cache=False).logits[:, SKIP:-1].flatten(0, 1).float(), b[:, SKIP + 1:].flatten(), reduction="none")
            out.append(l.view(len(b), -1).mean(1).cpu())
    if handle:
        handle.remove()
    return torch.cat(out)


res = {}
for src in ("wiki", "gsm8k"):
    T = S["tokens"][[i for i, s in enumerate(S["source"]) if s == src]][:160]
    base = losses(T, None)
    res[src] = dict(chunks=len(T), base=base.mean().item())
    for arm in ("dir_neurospeak", "dir_astra", "dir_random", "dir_matched", "dir_mean"):
        d = losses(T, arm) - base
        boot = torch.stack([d[torch.randint(len(d), (len(d),))].mean() for _ in range(5000)]).sort().values
        res[src][arm] = dict(rise=d.mean().item(), low=boot[125].item(), high=boot[4875].item())
json.dump(res, open("runs/whyfix/harm.json", "w"), indent=1)
print(json.dumps(res, indent=1))
