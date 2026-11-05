"""FVU of a dictionary on unseen traces and held-out Wikipedia, with the formula wordtest/scan.py uses.

    .venv/bin/python reasoning/score.py runs/sae/Qwen__Qwen3.5-0.8B/L19.pt
"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurospeak.anatomy import discover
from neurospeak.core import DEV
from neurospeak.sae import TopKSAE

SEQ, SKIP, BATCH, L = 512, 2, 8, 19
path = sys.argv[1]
c = torch.load(path, map_location=DEV)
sae = TopKSAE(c["config"]["d"], c["config"]["n"], c["config"]["k"]).to(DEV)
sae.load_state_dict(c["state"]); sae.requires_grad_(False)
model = c["config"]["model"]
tok = AutoTokenizer.from_pretrained(model)
m = AutoModelForCausalLM.from_pretrained(model, dtype="auto").to(DEV).eval()
block = discover(m).blocks[L]
grab = {}
block.register_forward_hook(lambda _m, _i, out: grab.__setitem__("x", out[0] if isinstance(out, tuple) else out))

S = torch.load("runs/wordtest/scan.pt", map_location="cpu")
wiki = S["tokens"][[i for i, s in enumerate(S["source"]) if s == "wiki"]]
ids = []
for l in open("runs/whyfix/mine_traces.jsonl"):
    r = json.loads(l)
    t = tok(r["prompt"] + r["completion"]).input_ids
    ids += [t[i:i + SEQ] for i in range(0, len(t) - SEQ + 1, SEQ)]
traces = torch.tensor(ids)

out = dict(dictionary=path, tokens_trained=c["tokens"])
for name, T in (("traces", traces), ("wiki", wiki)):
    e, v = 0.0, 0.0
    for s in range(0, len(T), BATCH):
        with torch.no_grad():
            m(T[s:s + BATCH].to(DEV), use_cache=False)
            x = grab["x"][:, SKIP:].reshape(-1, c["config"]["d"]).float() * c["scale"]
            recon, *_ = sae(x)
            e += (x - recon).pow(2).sum().item()
            v += (x - x.mean(0)).pow(2).sum().item()
    out[name] = dict(fvu=e / v, chunks=len(T))
print(json.dumps(out, indent=1))
