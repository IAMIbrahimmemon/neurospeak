"""Study 26 Part A (PLAN_WRITERS.md): split h·d at block 19's output into the embedding and each block's mixer and MLP outputs, on unedited GSM traces, CPU."""
import json, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover
sys.path[:0] = ["investigate"]
from behaviors import DOUBT
torch.set_num_threads(6)
N, L = int(sys.argv[1]), 19
mid = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
B = discover(m).blocks
d = F.normalize(torch.load("runs/whyfix/directions.pt")["dir_self_doubt"].float(), dim=0)
cur = {}
first = lambda o: o[0] if isinstance(o, tuple) else o
B[0].register_forward_pre_hook(lambda _m, a, k: cur.__setitem__("embed", (a[0] if a else k["hidden_states"]) @ d), with_kwargs=True)
for l in range(L + 1):
    mixer = getattr(B[l], "self_attn", None) or B[l].linear_attn
    kind = "attn" if hasattr(B[l], "self_attn") else "lin"
    mixer.register_forward_hook(lambda _m, _i, o, n=f"{kind}{l}": cur.__setitem__(n, first(o) @ d))
    B[l].mlp.register_forward_hook(lambda _m, _i, o, n=f"mlp{l}": cur.__setitem__(n, first(o) @ d))
B[L].register_forward_hook(lambda _m, _i, o: cur.__setitem__("total", first(o) @ d))
cols, near, worst = {}, [], 0.0
for r in [json.loads(x) for x in open("runs/research/layers_none_traces.jsonl")][:N]:
    text = r["prompt"] + r["completion"]
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    pl = len(tok(r["prompt"]).input_ids)
    cur.clear()
    with torch.no_grad():
        m(enc.input_ids)
    parts = {k: v[0] for k, v in cur.items()}
    total = parts.pop("total")
    s = sum(parts.values())
    worst = max(worst, ((s - total).abs() / total.abs().mean()).max().item())
    for k, v in {**parts, "total": total}.items():
        cols.setdefault(k, []).append(v[pl:])
    spans = [mt.span() for mt in DOUBT.finditer(text)]
    off = enc.offset_mapping[0].tolist()
    hit = torch.tensor([any(a < e and b > s0 for s0, e in spans) for a, b in off])
    nxt = torch.zeros_like(hit)
    for j in (1, 2, 3):
        nxt[:-j] |= hit[j:]
    near.append(nxt[pl:])
X = {k: torch.cat(v) for k, v in cols.items()}
near = torch.cat(near)
T = X.pop("total")
var = T.var()
rows = {k: dict(mean=round(v.mean().item(), 4), var_share=round((((v - v.mean()) * (T - T.mean())).mean() / var).item(), 4),
                doubt_minus_other=round((v[near].mean() - v[~near].mean()).item(), 4)) for k, v in X.items()}
order = sorted(rows, key=lambda k: -rows[k]["var_share"])
top3 = order[:3]
out = dict(traces=N, tokens=len(T), doubt_near_tokens=int(near.sum()), sum_check_max_rel_err=round(worst, 5),
           total=dict(mean=round(T.mean().item(), 4), var=round(var.item(), 4), doubt_minus_other=round((T[near].mean() - T[~near].mean()).item(), 4)),
           var_share_sum=round(sum(r["var_share"] for r in rows.values()), 4), top3=top3,
           top3_share=round(sum(rows[k]["var_share"] for k in top3), 4), ranked=[[k, rows[k]] for k in order])
json.dump(out, open("runs/research/writers.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "ranked"}, indent=1))
for k, v in out["ranked"][:12]: print(k, v)
print("...")
for k, v in out["ranked"][-4:]: print(k, v)
