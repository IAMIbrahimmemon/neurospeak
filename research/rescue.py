"""Continues stuck unedited runs, removing a direction only from the prefix's last token on. See PLAN_RESCUE.md.
    .venv/bin/python research/rescue.py --arm none --out runs/research/rescue_none_traces.jsonl"""
import argparse, json, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurospeak.anatomy import discover

sys.path[:0] = ["research"]
from loop_onset import HIGH, SIZE, STEP, windows

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max", type=int, default=1500)
ap.add_argument("--gain", type=float, default=1.0, help="1 removes the direction; above 1 overshoots")
ap.add_argument("--also", default="", help="also remove this direction, which must be orthogonal to --arm")
ap.add_argument("--limit", type=int, default=0, help="smoke test on this many runs")
ap.add_argument("--runs", default="", help="prefix file from build_tilt.py, in place of the study 7 prefixes")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--shift", type=float, default=0.0, help="subtract this constant times the direction, in place of removing it")
a = ap.parse_args()
mid = "Qwen/Qwen3.5-0.8B"

runs = []
for f in () if a.runs else ("runs/whyfix/gsm64_none_traces.jsonl", "runs/whyfix/gsm64b_none_traces.jsonl"):
    for l in open(f):
        r = json.loads(l)
        ws = list(re.finditer(r"\S+", r["completion"]))
        s = windows([m.group() for m in ws])
        on = next((i for i, x in enumerate(s) if x >= HIGH), None)
        if on is not None:
            end = min(on * STEP + SIZE + 7, len(ws)) - 1      # the window's last 8-gram ends here
            runs.append({**{k: r[k] for k in ("id", "answer")}, "prompt": r["prompt"], "prefix": r["completion"][:ws[end].end()], "prefix_words": end + 1})
runs = [json.loads(l) for l in open(a.runs)] if a.runs else runs
assert a.runs or len(runs) == 56, len(runs)
tok = AutoTokenizer.from_pretrained(mid)
tok.padding_side = "left"
runs.sort(key=lambda r: (r["prefix_words"], r["id"]))
runs = runs[:a.limit] if a.limit else runs
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("mps").eval()
if a.arm != "none":
    extra = torch.load("runs/research/dose_dirs.pt") if a.arm == "dir_rep_only" else {}
    tilted = torch.load("runs/research/tilt_dirs.pt") if a.runs else {}
    all_dirs = {**torch.load("runs/whyfix/directions.pt"), **extra, **tilted}
    d = all_dirs[a.arm].to("mps")
    also = all_dirs[a.also].to("mps") if a.also else None

    def project_out(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        e = d.to(h.dtype)
        new = h.clone()
        new[:, -1:] = h[:, -1:] - (a.shift * e if a.shift else a.gain * (h[:, -1:] @ e).unsqueeze(-1) * e)      # after the prefix pass, each step is one token
        if also is not None:
            f = also.to(h.dtype)
            new[:, -1:] = new[:, -1:] - (h[:, -1:] @ f).unsqueeze(-1) * f
        return (new, *o[1:]) if isinstance(o, tuple) else new

    discover(m).blocks[19].register_forward_hook(project_out)
torch.manual_seed(a.seed)
out = open(a.out, "w")
for i in range(0, len(runs), 8):
    batch = runs[i:i + 8]
    enc = tok([r["prompt"] + r["prefix"] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
    t = time.time()
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=a.max, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0)
    for r, row in zip(batch, g):
        new = row[enc.input_ids.shape[1]:]
        n = int((new != tok.pad_token_id).sum())
        out.write(json.dumps({**r, "completion": tok.decode(new, skip_special_tokens=True), "n_tokens": n, "cut": n >= a.max}) + "\n")
    out.flush()
    print(i, round(time.time() - t), flush=True)
