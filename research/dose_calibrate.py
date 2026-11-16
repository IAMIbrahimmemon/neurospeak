"""Fixes the gains for PLAN_DOSE.md and saves the two new directions. Run on the GPU, alone, before any study 8 arm."""
import json, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover

sys.path[:0] = ["research"]
from loop_onset import HIGH, SIZE, STEP, windows

K, GRID = 128, [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
mid = "Qwen/Qwen3.5-0.8B"
D = {k: v.float() for k, v in torch.load("runs/whyfix/directions.pt", map_location="cpu").items()}
r, s = D["dir_neurospeak"], D["dir_self_doubt"]
rep_only = r - (r @ s) * s
new = {"dir_rep_only": rep_only / rep_only.norm()}
torch.save(new, "runs/research/dose_dirs.pt")
D.update(new)
runs = []
for l in open("runs/research/long_none_traces.jsonl"):
    x = json.loads(l)
    ws = list(re.finditer(r"\S+", x["completion"]))
    sh = windows([m.group() for m in ws])
    on = next((i for i, v in enumerate(sh) if v >= HIGH), None)
    if on is not None and len(runs) < 8:
        end = min(on * STEP + SIZE + 7 + 100, len(ws)) - 1      # keep some loop text past the window to edit
        runs.append(x["prompt"] + x["completion"][:ws[end].end()])
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("mps").eval()
block = discover(m).blocks[19]


def dose(direction, gain):
    d = direction.to("mps")

    def edit(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        e = d.to(h.dtype)
        new = h.clone()
        new[:, -K:] = h[:, -K:] - gain * (h[:, -K:] @ e).unsqueeze(-1) * e
        return (new, *o[1:]) if isinstance(o, tuple) else new

    hook = block.register_forward_hook(edit)
    out = []
    with torch.no_grad():
        for ids, base in zip(all_ids, bases):
            lp = m(ids, use_cache=False).logits[0, -K:].float().log_softmax(-1)
            out.append((base.exp() * (base - lp)).sum(-1).mean().item())
    hook.remove()
    return sum(out) / len(out)


all_ids = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to("mps") for t in runs]
with torch.no_grad():
    bases = [m(i, use_cache=False).logits[0, -K:].float().log_softmax(-1) for i in all_ids]
target = dose(D["dir_neurospeak"], 1)
res = dict(target=target, runs=len(runs), arms={})
import math
for arm, name in dict(doubt_dose="dir_self_doubt", random_dose="dir_self_doubt_matched", rep_only="dir_rep_only", doubt_only="dir_self_doubt_orth").items():
    curve = {g: dose(D[name], g) for g in GRID}
    g = min(GRID, key=lambda x: abs(math.log(max(curve[x], 1e-12) / target)))
    res["arms"][arm] = dict(direction=name, gain=g, dose=curve[g], reached=curve[max(GRID)] >= target / 3, curve=curve)
    print(arm, g, curve[g], flush=True)
res["arms"]["matched_mean"] = dict(direction="dir_matched", gain=1, dose=dose(D["dir_matched"], 1))
json.dump(res, open("runs/research/dose_gains.json", "w"), indent=1)
print(json.dumps({k: {a: b for a, b in v.items() if a != "curve"} for k, v in res["arms"].items()}, indent=1), target)
