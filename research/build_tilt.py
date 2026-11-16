"""Freezes the directions, prefixes and gains for PLAN_TILT.md. CPU only, no edited arm.
    .venv/bin/python research/build_tilt.py"""
import json, math, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover

sys.path[:0] = ["research"]
from loop_onset import HIGH, SIZE, STEP, windows

mid, K, THETA = "Qwen/Qwen3.5-0.8B", 64, math.radians(42)
D = torch.load("runs/whyfix/directions.pt", map_location="cpu")
only = torch.load("runs/research/dose_dirs.pt", map_location="cpu")["dir_rep_only"].float()
s = D["dir_self_doubt"].float()
assert abs(only @ s) < 1e-3
tilt = math.cos(THETA) * only + math.sin(THETA) * s
perp = -math.sin(THETA) * only + math.cos(THETA) * s
g = torch.Generator().manual_seed(2026)
rand = torch.randn(only.shape, generator=g)
for e in (only, s):
    rand -= (rand @ e) * e
rand /= rand.norm()
dirs = dict(dir_tilt=tilt / tilt.norm(), dir_perp=perp / perp.norm(), dir_rand=rand)
torch.save(dirs, "runs/research/tilt_dirs.pt")


def onset_prefix(r):
    ws = list(re.finditer(r"\S+", r["completion"]))
    sh = windows([m.group() for m in ws])
    on = next((i for i, x in enumerate(sh) if x >= HIGH), None)
    if on is None:
        return None
    end = min(on * STEP + SIZE + 7, len(ws)) - 1
    return {**{k: r[k] for k in ("id", "answer", "prompt")}, "prefix": r["completion"][:ws[end].end()], "prefix_words": end + 1}


calib = 0
runs = []
for name in ("pairs", "add", "both", "long"):
    for l in open(f"runs/research/{name}_none_traces.jsonl"):
        r = json.loads(l)
        ws = list(re.finditer(r"\S+", r["completion"]))
        sh = windows([m.group() for m in ws])
        looped = any(x >= HIGH for x in sh)
        if name == "long" and looped and calib < 8:      # same first 8 as dose_calibrate.py
            calib += 1
            continue
        p = onset_prefix(r)
        if p:
            runs.append({**p, "source": name})
assert len({r["id"] for r in runs}) == len(runs)
with open("runs/research/tilt_prefixes.jsonl", "w") as f:
    for r in runs:
        f.write(json.dumps(r) + "\n")
print(len(runs), {n: sum(r["source"] == n for r in runs) for n in ("pairs", "add", "both", "long")})

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
got = []
hook = discover(m).blocks[19].register_forward_hook(lambda _m, _i, o: got.append((o[0] if isinstance(o, tuple) else o)[0, -K:].detach().float()))
sample = runs[::max(1, len(runs) // 24)][:24]
with torch.no_grad():
    for r in sample:
        ids = tok(r["prompt"] + r["prefix"], return_tensors="pt", add_special_tokens=False)
        m.model(**ids, use_cache=False)
hook.remove()
h = torch.cat(got)
axes = dict(only=only, s=s, **dirs)
act = {k: dict(mean_abs=(h @ v).abs().mean().item(), mean=(h @ v).mean().item()) for k, v in axes.items()}
gains = dict(perp_eq=act["dir_tilt"]["mean_abs"] / act["dir_perp"]["mean_abs"], rand_eq=act["dir_tilt"]["mean_abs"] / act["dir_rand"]["mean_abs"])
json.dump(dict(n_prefixes=len(runs), sample=len(sample), tokens=len(h), activity=act, gains=gains), open("runs/research/tilt_gains.json", "w"), indent=1)
print(json.dumps(dict(activity=act, gains=gains), indent=1))
