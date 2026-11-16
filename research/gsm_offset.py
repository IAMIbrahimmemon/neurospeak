"""Study 23 audit: signed mean of h·d (self-doubt, block 19 output) on unedited GSM traces, by position in the completion, CPU. Compares with the corpus mean mu·d used by the offset and centered arms."""
import json, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover
torch.set_num_threads(6)
N = int(sys.argv[1])
mid = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
d = F.normalize(torch.load("runs/whyfix/directions.pt")["dir_self_doubt"].float(), dim=0)
mu = torch.load("runs/research/split_mean.pt")["mean"].float().cpu()
acts = []
discover(m).blocks[19].register_forward_hook(lambda _m, _i, o: acts.append((o[0] if isinstance(o, tuple) else o)[0].float()))
BINS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000)]
proj = {b: [] for b in BINS}
for r in [json.loads(x) for x in open("runs/research/layers_none_traces.jsonl")][:N]:
    pl = len(tok(r["prompt"]).input_ids)
    ids = tok(r["prompt"] + r["completion"], return_tensors="pt").input_ids
    acts.clear()
    with torch.no_grad():
        m(ids)
    p = (acts[0] @ d)[pl:]
    for b in BINS:
        proj[b] += p[b[0]:b[1]].tolist()
mean = lambda v: sum(v) / max(len(v), 1)
out = dict(traces=N, mu_dot_d=round((mu @ d).item(), 3),
           bins={f"{a}-{b}": dict(tokens=len(v), signed_mean=round(mean(v), 3), abs_mean=round(mean([abs(x) for x in v]), 3)) for (a, b), v in proj.items()})
allv = sum(proj.values(), [])
out["all"] = dict(signed_mean=round(mean(allv), 3), abs_mean=round(mean([abs(x) for x in allv]), 3))
json.dump(out, open("runs/research/split_gsm_offset.json", "w"), indent=1)
print(json.dumps(out, indent=1))
