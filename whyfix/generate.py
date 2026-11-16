"""Generates traces with a set of L19 words, or one direction (arm dir_*), removed at every token.  .venv/bin/python whyfix/generate.py --arm why --out runs/whyfix/why_traces.jsonl"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from neurospeak.anatomy import discover
from neurospeak.sae import TopKSAE, checkpoint_dir

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gsm-test", type=int, default=0, help="use this many GSM8K test problems instead of the 16 practice problems")
ap.add_argument("--gsm-start", type=int, default=0, help="first GSM8K test row to use, after the fixed shuffle")
ap.add_argument("--also", default="", help="also remove this direction, which must be orthogonal to --arm; a key in runs/research/both_dirs.pt or directions.pt")
ap.add_argument("--add", type=float, default=0, help="with a dir_* arm: add this much of the direction at every token instead of removing it")
ap.add_argument("--edit", default="", help="apply this weight edit file before generating")
ap.add_argument("--resume", action="store_true", help="keep finished whole batches already in --out and generate only the rest")
ap.add_argument("--layer", type=int, default=None, help="edit at this layer instead of the words file's layer (19)")
ap.add_argument("--gain", type=float, default=1.0, help="with a dir_* arm: remove only this fraction of the projection")
ap.add_argument("--mode", default="full", choices=["full", "centered", "offset"], help="centered removes only (h - mean)·d; offset removes only mean·d; mean from runs/research/split_mean.pt")
ap.add_argument("--level", type=float, default=None, help="with --mode centered or offset: use this value for the constant part along d instead of mean·d")
ap.add_argument("--ban", default="", help="comma-separated token ids whose logits are set to -inf at every step")
ap.add_argument("--presence", type=float, default=0, help="subtract this from the logits of every token already generated, before temperature")
a = ap.parse_args()
mid = "Qwen/Qwen3.5-0.8B"
W = json.load(open("runs/whyfix/words.json"))
ALL = torch.load("runs/whyfix/directions.pt")
if a.arm.startswith("dir_xfer"):
    ALL = {**ALL, **torch.load("runs/research/xfer_dirs.pt")}
if a.arm.startswith("dir_dosectl"):
    ALL = {**ALL, **torch.load("runs/research/dosectl_dirs.pt")}
if a.arm.startswith("dir_layer"):
    ALL = {**ALL, **torch.load("runs/research/layer_dirs.pt")}
if a.arm.startswith("dir_factor"):
    ALL = {**ALL, **torch.load("runs/research/factor_dirs.pt")}
if a.also or a.arm.startswith("dir_both"):
    ALL = {**ALL, **torch.load("runs/research/both_dirs.pt")}
direction = ALL[a.arm].to("mps").float() if a.arm.startswith("dir_") else None
mu = torch.load("runs/research/split_mean.pt")["mean"].to("mps").float() if a.mode != "full" else None
level = (a.level if a.level is not None else float(mu @ direction)) if a.mode != "full" else None
also = ALL[a.also].to("mps").float() if a.also else None
L, words = W["layer"] if a.layer is None else a.layer, ([976] if a.arm == "f976" else [] if a.arm == "none" or direction is not None else W[a.arm])
tok = AutoTokenizer.from_pretrained(mid)
tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("mps").eval()
if a.edit:
    from neurospeak import apply_edit_file
    apply_edit_file(m, a.edit)
if words:
    c = torch.load(f"{checkpoint_dir(mid)}/L{L}.pt", map_location="mps")
    sae = TopKSAE(c["config"]["d"], c["config"]["n"], c["config"]["k"]).to("mps")
    sae.load_state_dict(c["state"])
    sae.requires_grad_(False)
    scale = c["scale"]


def hook(_m, _i, o):
    h = o[0] if isinstance(o, tuple) else o
    x = h.float() * scale
    vals, idx = sae.encode(x)
    mask = sum((idx == w) for w in words).bool()
    if not mask.any():
        return o
    full = sae.W_dec.T[idx] * (vals * mask).unsqueeze(-1)
    new = h - (full.sum(-2) / scale).to(h.dtype)
    return (new, *o[1:]) if isinstance(o, tuple) else new


def project_out(_m, _i, o):
    h = o[0] if isinstance(o, tuple) else o
    d = direction.to(h.dtype)
    if a.add:
        new = h + a.add * d
    elif a.mode == "centered":
        new = h - ((h @ d).unsqueeze(-1) - level) * d
    elif a.mode == "offset":
        new = h - level * d
    elif a.gain != 1:
        new = h - a.gain * (h @ d).unsqueeze(-1) * d
    else:
        new = h - (h @ d).unsqueeze(-1) * d
    if also is not None:
        f = also.to(h.dtype)
        new = new - (h @ f).unsqueeze(-1) * f
    return (new, *o[1:]) if isinstance(o, tuple) else new


if words:
    discover(m).blocks[L].register_forward_hook(hook)
if direction is not None:
    discover(m).blocks[L].register_forward_hook(project_out)
probs = [json.loads(l) for l in open("data/practice_sample.jsonl")][:16]
if a.gsm_test:
    import random
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    used = {json.loads(l)["problem"].strip() for l in open("data/practice_sample.jsonl")}
    rows = [r for r in pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist() if r["question"].strip() not in used]
    random.Random(2026).shuffle(rows)
    probs = [dict(id=f"gsm-test-{i}", source="gsm8k-test", problem=r["question"], answer=r["answer"].split("####")[-1].strip().replace(",", "")) for i, r in enumerate(rows[a.gsm_start:a.gsm_start + a.gsm_test], a.gsm_start)]


class Presence(LogitsProcessor):
    """vLLM-style presence penalty: tokens generated so far lose a constant. Prompt tokens are not counted."""

    def __init__(self, start, penalty):
        self.start, self.penalty, self.seen = start, penalty, None

    def __call__(self, input_ids, scores):
        if self.seen is None:
            self.seen = torch.zeros_like(scores, dtype=torch.bool)
        if input_ids.shape[1] > self.start:
            self.seen[torch.arange(len(input_ids)), input_ids[:, -1]] = True
        return scores - self.penalty * self.seen.to(scores.dtype)


class Ban(LogitsProcessor):
    """Sets the logits of fixed token ids to -inf at every step."""

    def __init__(self, ids):
        self.ids = ids

    def __call__(self, input_ids, scores):
        scores[:, self.ids] = float("-inf")
        return scores


class Progress(LogitsProcessor):
    """Prints tokens generated and seconds elapsed every 500 steps, so a slow batch shows where it slows."""

    def __init__(self, start):
        self.start, self.t = start, time.time()

    def __call__(self, input_ids, scores):
        n = input_ids.shape[1] - self.start
        if n % 500 == 0:
            print(f"  step {n} {round(time.time() - self.t)}s", flush=True)
        return scores


SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
torch.manual_seed(a.seed)
done = 0
if a.resume and os.path.exists(a.out):
    kept = [l for l in open(a.out) if l.strip()]
    done = len(kept) // 8 * 8
    assert [json.loads(l)["id"] for l in kept[:done]] == [p["id"] for p in probs[:done]], "existing output does not match these problems"
    open(a.out, "w").writelines(kept[:done])
    torch.manual_seed(a.seed + done)
out = open(a.out, "a" if done else "w")
for i in range(done, len(probs), 8):
    batch = probs[i:i + 8]
    texts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + SUFFIX}], add_generation_prompt=True, tokenize=False, enable_thinking=True) for p in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to("mps")
    t = time.time()
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=a.max, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0, logits_processor=LogitsProcessorList(([Presence(enc.input_ids.shape[1], a.presence)] if a.presence else []) + ([Ban([int(x) for x in a.ban.split(",")])] if a.ban else []) + [Progress(enc.input_ids.shape[1])]))
    for p, row, prompt in zip(batch, g, texts):
        new = row[enc.input_ids.shape[1]:]
        n_tok = int((new != tok.pad_token_id).sum())
        out.write(json.dumps({**p, "prompt": prompt, "completion": tok.decode(new, skip_special_tokens=True), "n_tokens": n_tok, "cut": n_tok >= a.max, "run": {**vars(a), "layer_used": L}}) + "\n")
    out.flush()
    print(i, round(time.time() - t), flush=True)
