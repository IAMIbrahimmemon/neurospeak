"""Generates traces on another model, with an optional direction removed at every token. See PLAN_PORT.md."""
import argparse, json, random, time
import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, help="none, or a key in runs/research/port_dirs.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=32)
ap.add_argument("--start", type=int, default=352)
ap.add_argument("--seed", type=int, default=9)
a = ap.parse_args()
mid, L, MAX, SEED, START = "facebook/MobileLLM-R1-360M-base", 12, 1500, a.seed, a.start
tok = AutoTokenizer.from_pretrained(mid)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("mps").eval()
if a.arm != "none":
    d = torch.load("runs/research/port_dirs.pt")[a.arm].float().to("mps")

    def project_out(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        e = d.to(h.dtype)
        new = h - (h @ e).unsqueeze(-1) * e
        return (new, *o[1:]) if isinstance(o, tuple) else new

    discover(m).blocks[L].register_forward_hook(project_out)
used = {json.loads(l)["problem"].strip() for l in open("data/practice_sample.jsonl")}
rows = [r for r in pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist() if r["question"].strip() not in used]
random.Random(2026).shuffle(rows)
probs = [dict(id=f"gsm-test-{i}", problem=r["question"], answer=r["answer"].split("####")[-1].strip().replace(",", "")) for i, r in enumerate(rows[START:START + a.limit], START)]
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
torch.manual_seed(SEED)
out = open(a.out, "w")
for i in range(0, len(probs), 8):
    batch = probs[i:i + 8]
    texts = [p["problem"] + SUFFIX for p in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to("mps")
    t = time.time()
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=MAX, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0)
    for p, row, prompt in zip(batch, g, texts):
        new = row[enc.input_ids.shape[1]:]
        n = int((new != tok.pad_token_id).sum())
        out.write(json.dumps({**p, "prompt": prompt, "completion": tok.decode(new, skip_special_tokens=True), "n_tokens": n, "cut": n >= MAX}) + "\n")
    out.flush()
    print(i, round(time.time() - t), flush=True)
