"""Study 25 (PLAN_OTHERDIR.md): mean |h·d|/|h| at L19 on unedited GSM traces for self-doubt and the three other learned directions, CPU."""
import json, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover
torch.set_num_threads(4)
N = int(sys.argv[1])
mid = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
D = torch.load("runs/whyfix/directions.pt")
keys = sys.argv[3].split(",") if len(sys.argv) > 3 else ["dir_self_doubt", "dir_bullets", "dir_chinese", "dir_restating"]
D = {**D, **torch.load("runs/research/both_dirs.pt")}
import os
if os.path.exists("runs/research/factor_dirs.pt"):
    D = {**D, **torch.load("runs/research/factor_dirs.pt")}
U = torch.stack([F.normalize(D[k].float(), dim=0) for k in keys])
acts = []
discover(m).blocks[19].register_forward_hook(lambda _m, _i, o: acts.append((o[0] if isinstance(o, tuple) else o)[0].float()))
H = []
for r in [json.loads(x) for x in open("runs/research/layers_none_traces.jsonl")][:N]:
    pl = len(tok(r["prompt"]).input_ids)
    acts.clear()
    with torch.no_grad():
        m(tok(r["prompt"] + r["completion"], return_tensors="pt").input_ids)
    H.append(acts[0][pl:])
H = torch.cat(H)
P = H @ U.T
dose = (P.abs() / H.norm(dim=1, keepdim=True)).mean(0)
out = dict(traces=N, tokens=len(H), dose={k: round(v, 4) for k, v in zip(keys, dose.tolist())},
           dose_vs_self_doubt={k: round(v / dose[0].item(), 2) for k, v in zip(keys, dose.tolist())},
           signed_mean={k: round(v, 3) for k, v in zip(keys, P.mean(0).tolist())})
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "runs/research/otherdir_dose.json", "w"), indent=1)
print(json.dumps(out, indent=1))
