"""Study 8's dose (mean next-token KL, edit on the last 128 teacher-forced tokens) for each tilt-study edit, on 8 tilt prefixes, CPU. See PLAN_TILT.md."""
import json, os, sys, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover

OUT = sys.argv[1]
mid, K = "Qwen/Qwen3.5-0.8B", 128
D = {k: v.float() for k, v in torch.load("runs/whyfix/directions.pt", map_location="cpu").items()}
only = torch.load("runs/research/dose_dirs.pt", map_location="cpu")["dir_rep_only"].float()
T = {k: v.float() for k, v in torch.load("runs/research/tilt_dirs.pt", map_location="cpu").items()}
G = json.load(open("runs/research/tilt_gains.json"))
s = D["dir_self_doubt"]
P = [json.loads(l) for l in open("runs/research/tilt_prefixes.jsonl")]
sample = P[::max(1, len(P) // 24)][:24][::3][:8]      # 8 of the 24 calibration prefixes, spread over sources
torch.set_num_threads(int(os.environ.get("THREADS", "5")))
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
block = discover(m).blocks[19]
TILTMEAN = G["activity"]["dir_tilt"]["mean_abs"]

# edit kinds: ("proj", d, gain) = h - gain (h.d) d ; ("proj2", d, e) = both removed ; ("shift", d, amount) = h - amount d  (constant translation)
A = G["activity"]
edits = {
    "only": ("proj", only, 1.0),
    "only_eq": ("proj", only, TILTMEAN / (A["only"]["mean_abs"] * math.cos(math.radians(42)))),
    "tilt": ("proj", T["dir_tilt"], 1.0),
    "both": ("proj2", only, s),
    "perp_eq": ("proj", T["dir_perp"], G["gains"]["perp_eq"]),
    "rand_eq": ("shift", T["dir_rand"], TILTMEAN),
}


def make(kind, d, x):
    def edit(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        new = h.clone()
        hk = h[:, -K:]
        if kind == "proj":
            new[:, -K:] = hk - x * (hk @ d).unsqueeze(-1) * d
        elif kind == "proj2":
            new[:, -K:] = hk - (hk @ d).unsqueeze(-1) * d - (hk @ x).unsqueeze(-1) * x
        else:
            new[:, -K:] = hk - x * d
        return (new, *o[1:]) if isinstance(o, tuple) else new
    return edit


res = {}
for r in sample:
    ids = tok(r["prompt"] + r["prefix"], return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        base = m(ids, use_cache=False).logits[0, -K:].float().log_softmax(-1)
    row = dict(p_top=base.exp().max(-1).values.mean().item())
    for name, (kind, d, x) in edits.items():
        hk = block.register_forward_hook(make(kind, d, x))
        with torch.no_grad():
            lp = m(ids, use_cache=False).logits[0, -K:].float().log_softmax(-1)
        hk.remove()
        row[name] = dict(kl=(base.exp() * (base - lp)).sum(-1).mean().item(), top_changed=(lp.argmax(-1) != base.argmax(-1)).float().mean().item())
    res[r["id"]] = row
    print(r["id"], r["source"], ids.shape[1], {k: round(v["kl"], 4) if isinstance(v, dict) else round(v, 3) for k, v in row.items()}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
mean = lambda k, f: sum(v[k][f] for v in res.values()) / len(res)
print("\nmean over %d prefixes:" % len(res))
for k in edits:
    print("  %-15s kl %.4f  top_changed %.3f" % (k, mean(k, "kl"), mean(k, "top_changed")))
