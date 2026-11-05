"""Generates traces with a set of L19 words, or one direction (arm dir_*), removed at every token.  .venv/bin/python whyfix/generate.py --arm why --out runs/whyfix/why_traces.jsonl"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurospeak.anatomy import discover
from neurospeak.sae import TopKSAE, checkpoint_dir

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gsm-test", type=int, default=0, help="use this many GSM8K test problems instead of the 16 practice problems")
ap.add_argument("--gsm-start", type=int, default=0, help="first GSM8K test row to use, after the fixed shuffle")
a = ap.parse_args()
mid = "Qwen/Qwen3.5-0.8B"
W = json.load(open("runs/whyfix/words.json"))
direction = torch.load("runs/whyfix/directions.pt")[a.arm].to("mps") if a.arm.startswith("dir_") else None
L, words = W["layer"], ([976] if a.arm == "f976" else [] if a.arm == "none" or direction is not None else W[a.arm])
tok = AutoTokenizer.from_pretrained(mid)
tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("mps").eval()
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
    new = h - (h @ d).unsqueeze(-1) * d
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
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
torch.manual_seed(a.seed)
out = open(a.out, "w")
for i in range(0, len(probs), 8):
    batch = probs[i:i + 8]
    texts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + SUFFIX}], add_generation_prompt=True, tokenize=False, enable_thinking=True) for p in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to("mps")
    t = time.time()
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=a.max, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0)
    for p, row, prompt in zip(batch, g, texts):
        new = row[enc.input_ids.shape[1]:]
        n_tok = int((new != tok.pad_token_id).sum())
        out.write(json.dumps({**p, "prompt": prompt, "completion": tok.decode(new, skip_special_tokens=True), "n_tokens": n_tok, "cut": n_tok >= a.max}) + "\n")
    out.flush()
    print(i, round(time.time() - t), flush=True)
