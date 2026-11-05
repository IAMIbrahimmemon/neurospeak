"""Step 3 — ground truth. Run only AFTER blind predictions exist. Output: blindread/truth.json"""
import glob, json, random, sys, torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
assert len(glob.glob("blindread/pred_blind_*.json")) == 6, "blind predictions must exist first"
mid = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("mps").eval()
dossiers = json.load(open("blindread/dossiers.json"))

# corpus: ~70% Wikipedia (general), ~30% math problems + reasoning traces
CHUNK = 512
gen_text = []
for split in ("validation", "test"):
    p = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    gen_text.append("".join(pq.read_table(p).column("text").to_pylist()))
gen_ids = tok("".join(gen_text)).input_ids[:280_000]
chunks = [gen_ids[i:i + CHUNK] for i in range(0, len(gen_ids), CHUNK)]
math_ids = []
for f in sorted(glob.glob("runs/*traces*.jsonl")):
    for l in open(f):
        r = json.loads(l); math_ids.append(tok(r["prompt"] + r["completion"]).input_ids[:CHUNK * 2])
for l in open("data/practice_sample.jsonl"):
    math_ids.append(tok(json.loads(l)["problem"]).input_ids)
random.Random(0).shuffle(math_ids)
budget = 120_000
for ids in math_ids:
    for i in range(0, len(ids), CHUNK):
        if budget <= 0: break
        chunks.append(ids[i:i + CHUNK]); budget -= len(ids[i:i + CHUNK])
print(len(chunks), "chunks", sum(map(len, chunks)), "tokens", flush=True)

by_layer = {}
for d in dossiers:
    by_layer.setdefault(d["layer"], []).append(d["neuron"])
store = {L: [] for L in by_layer}
def mk(L):
    idx = torch.tensor(by_layer[L])
    return lambda _m, inp: store[L].append(inp[0][0][:, idx.to(inp[0].device)].float().cpu())
hooks = [m.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L)) for L in by_layer]
toks = []
with torch.no_grad():
    for k, c in enumerate(chunks):
        m(torch.tensor([c], device="mps")); toks.append(torch.tensor(c))
        if k % 200 == 0: print(k, flush=True)
for h in hooks: h.remove()
T = torch.cat(toks)
chunk_id = torch.cat([torch.full((len(c),), i) for i, c in enumerate(chunks)])
acts = {L: torch.cat(v) for L, v in store.items()}

def ctx(i):
    s = max(0, i - 16)
    while chunk_id[s] != chunk_id[i]: s += 1
    e = i + 4
    while e > i + 1 and (e > len(T) or chunk_id[e - 1] != chunk_id[i]): e -= 1
    f = lambda a, b: tok.decode(T[a:b]).replace("\n", "⏎")
    return f"…{f(s, i)}[[{f(i, i + 1)}]]{f(i + 1, e)}"

def top_contexts(a, n=10):
    out, seen_txt, seen_pos = [], set(), []
    for i in a.topk(400).indices.tolist():
        if any(abs(i - j) < 10 for j in seen_pos): continue
        c = ctx(i)
        if c in seen_txt: continue
        seen_txt.add(c); seen_pos.append(i); out.append({"act": round(a[i].item(), 3), "text": c})
        if len(out) == n: break
    return out

truth = []
for d in dossiers:
    a = acts[d["layer"]][:, by_layer[d["layer"]].index(d["neuron"])]
    pos, neg = top_contexts(a), top_contexts(-a)
    for c in neg: c["act"] = -c["act"]
    truth.append({"id": d["id"], "band": d["band"], "layer": d["layer"], "neuron": d["neuron"],
                  "stats": {"max": round(a.max().item(), 3), "min": round(a.min().item(), 3), "mean": round(a.mean().item(), 4)},
                  "seen_half": {"positive": pos[0::2], "negative": neg[0::2]},     # for ceiling readers
                  "test_half": {"positive": pos[1::2], "negative": neg[1::2]}})    # candidate sets for scoring
json.dump(truth, open("blindread/truth.json", "w"), ensure_ascii=False, indent=1)
print("wrote truth for", len(truth))
