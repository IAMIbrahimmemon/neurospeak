"""Sample thinking-mode traces from Qwen3.5-0.8B (optionally with a weight edit) to find real loops."""
import json, sys, time, torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=24); ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--max", type=int, default=3000); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--edit", default=None, help="npz edit file (astra format) or .pt state from neurospeak")
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
mid = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("mps").eval()
if a.edit:
    from neurospeak import apply_edit_file
    apply_edit_file(m, a.edit)
probs = [json.loads(l) for l in open("data/practice_sample.jsonl")][a.offset:a.offset + a.n]
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
torch.manual_seed(a.seed)
out = open(a.out, "w")
for i in range(0, len(probs), a.bs):
    batch = probs[i:i + a.bs]
    texts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + SUFFIX}], add_generation_prompt=True,
                                     tokenize=False, enable_thinking=True) for p in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to("mps")
    t = time.time()
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=a.max, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                       repetition_penalty=1.0)
    for p, row, prompt in zip(batch, g, texts):
        new = row[enc.input_ids.shape[1]:]
        n_tok = int((new != tok.pad_token_id).sum())
        txt = tok.decode(new, skip_special_tokens=True)
        out.write(json.dumps({**p, "prompt": prompt, "completion": txt, "n_tokens": n_tok,
                              "cut": n_tok >= a.max}) + "\n"); out.flush()
    print(f"batch {i//a.bs} {time.time()-t:.0f}s", flush=True)
