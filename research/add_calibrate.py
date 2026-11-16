"""Fixes the amounts for PLAN_ADD.md. Run on the GPU, alone, before any arm."""
import json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover

K, LOW, HIGH = 128, 0.0176, 0.088
GRID = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
mid = "Qwen/Qwen3.5-0.8B"
D = {k: v.float() for k, v in torch.load("runs/whyfix/directions.pt", map_location="cpu").items()}
texts = []
for l in open("runs/research/long_none_traces.jsonl"):
    x = json.loads(l)
    ws = list(re.finditer(r"\S+", x["completion"]))
    if len(ws) >= 400 and len(texts) < 8:
        texts.append(x["prompt"] + x["completion"][:ws[399].end()])
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("mps").eval()
block = discover(m).blocks[19]
ids = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to("mps") for t in texts]
with torch.no_grad():
    base = [m(i, use_cache=False).logits[0, -K:].float().log_softmax(-1) for i in ids]


def dose(direction, amount):
    d = direction.to("mps")

    def edit(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        new = h + amount * d.to(h.dtype)
        return (new, *o[1:]) if isinstance(o, tuple) else new

    hook = block.register_forward_hook(edit)
    out = []
    with torch.no_grad():
        for i, b in zip(ids, base):
            lp = m(i, use_cache=False).logits[0, -K:].float().log_softmax(-1)
            out.append((b.exp() * (b - lp)).sum(-1).mean().item())
    hook.remove()
    return sum(out) / len(out)


def closest(curve, target):
    return min(curve, key=lambda a: abs(torch.tensor(curve[a] / target).log().item()))


curves = {k: {a: dose(D[k], a) for a in GRID} for k in ("dir_self_doubt", "dir_self_doubt_matched", "dir_neurospeak")}
print(json.dumps(curves, indent=1), flush=True)
pick = [("doubt_low", "dir_self_doubt", LOW), ("doubt_high", "dir_self_doubt", HIGH), ("random_high", "dir_self_doubt_matched", HIGH), ("rep_high", "dir_neurospeak", HIGH)]
arms = {n: dict(direction=k, amount=closest(curves[k], t), dose=curves[k][closest(curves[k], t)], target=t) for n, k, t in pick}
json.dump(dict(arms=arms, curves=curves, runs=len(texts)), open("runs/research/add_gains.json", "w"), indent=1)
print(json.dumps(arms, indent=1))
