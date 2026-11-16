"""Exploratory, no bars, run on CPU. Inside locked loops: how sure is the next token, does each edit change it, and how active is each direction compared with earlier in the same run?"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover

mid, K = "Qwen/Qwen3.5-0.8B", 30
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).eval()
D = torch.load("runs/whyfix/directions.pt", map_location="cpu")
names = ["dir_self_doubt", "dir_neurospeak", "dir_self_doubt_matched"]
block = discover(m).blocks[19]
runs = [json.loads(l) for l in open("runs/research/rescue_none_traces.jsonl")][::8]
out = {n: dict(top_changed=[], kl=[], p_top=[]) for n in names}
base_stats = dict(p_top=[], over_95=[])
activity = {n: dict(early=[], loop=[]) for n in names}
torch.set_num_threads(4)
for r in runs:
    ids = tok(r["prompt"] + r["prefix"], return_tensors="pt", add_special_tokens=False).input_ids
    plen = len(tok(r["prompt"], add_special_tokens=False).input_ids)
    grabbed = {}
    h0 = block.register_forward_hook(lambda _m, _i, o: grabbed.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float().detach()))
    with torch.no_grad():
        base = m(ids, use_cache=False).logits[0, -K:].float().log_softmax(-1)
    h0.remove()
    base_stats["p_top"].append(base.exp().max(-1).values.mean().item())
    base_stats["over_95"].append((base.exp().max(-1).values > 0.95).float().mean().item())
    h = grabbed["h"][0]
    for n in names:
        d = D[n].float()
        p = h @ d
        activity[n]["early"].append(p[plen:plen + 200].abs().mean().item())
        activity[n]["loop"].append(p[-K:].abs().mean().item())

        def edit(_m, _i, o, d=d):
            x = o[0] if isinstance(o, tuple) else o
            e = d.to(x.dtype)
            new = x.clone()
            new[:, -K:] = x[:, -K:] - (x[:, -K:] @ e).unsqueeze(-1) * e
            return (new, *o[1:]) if isinstance(o, tuple) else new

        hk = block.register_forward_hook(edit)
        with torch.no_grad():
            lp = m(ids, use_cache=False).logits[0, -K:].float().log_softmax(-1)
        hk.remove()
        out[n]["top_changed"].append((lp.argmax(-1) != base.argmax(-1)).float().mean().item())
        out[n]["kl"].append((base.exp() * (base - lp)).sum(-1).mean().item())
        out[n]["p_top"].append(lp.exp().max(-1).values.mean().item())
    print(r["id"], ids.shape[1], flush=True)
mean = lambda v: sum(v) / len(v)
res = dict(runs=len(runs), unedited=dict(p_top=mean(base_stats["p_top"]), over_95=mean(base_stats["over_95"])),
           edits={n: {k: mean(v) for k, v in s.items()} for n, s in out.items()},
           activity={n: {k: mean(v) for k, v in s.items()} for n, s in activity.items()})
json.dump(res, open("runs/research/loop_confidence.json", "w"), indent=1)
print(json.dumps(res, indent=1))
