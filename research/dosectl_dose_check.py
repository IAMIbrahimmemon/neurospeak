"""Study 22 review: dose of self-doubt and the dose-matched controls on early and late GSM text, CPU. Written by Claude (Fable) during review."""
"""Review of study 22 (Claude Fable): dose of self-doubt and the 4 dose-matched controls at L19 on
(a) unedited GSM text, head and tail, (b) each arm's own edited-run text, plus structure of the controls."""
import json, time, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover
torch.set_num_threads(8)
OUT = sys.argv[1]
NTR, HEAD, TAIL = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
mid = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to("cpu").eval()
A = discover(m)
D = torch.load("runs/whyfix/directions.pt"); C = torch.load("runs/research/dosectl_dirs.pt"); Lr = torch.load("runs/research/layer_dirs.pt")
dirs = {"doubt": F.normalize(D["dir_self_doubt"].float(), dim=0), "old_ctl": F.normalize(Lr["dir_layer_19_matched"].float(), dim=0),
        **{f"ctl{s}": F.normalize(C[f"dir_dosectl_{s}"].float(), dim=0) for s in (101, 102, 103, 104)}}
P = "runs/research/"
arms = {"none": "layers_none_traces.jsonl", "doubt": "layers_dir_layer_19_traces.jsonl",
        **{f"ctl{s}": f"dosectl_dir_dosectl_{s}_at19_traces.jsonl" for s in (101, 102, 103, 104)}}
acts = []
hd = A.blocks[19].register_forward_hook(lambda _m, _i, o: acts.append((o[0] if isinstance(o, tuple) else o)[0].detach().float()))


def run(ids):
    acts.clear()
    with torch.no_grad():
        m(ids)
    return acts[0]


groups = {}
t0 = time.time()
for arm, f in arms.items():
    rows = [json.loads(x) for x in open(P + f)][:NTR]
    head, tail = [], []
    for r in rows:
        ids = tok(r["prompt"] + r["completion"], return_tensors="pt").input_ids
        pl = len(tok(r["prompt"]).input_ids)
        head.append(run(ids[:, :HEAD])[1:])  # drop position 0
        if TAIL and ids.shape[1] > HEAD + TAIL:
            tail.append(run(ids[:, -TAIL:])[1:])  # tail of the completion (no prompt in context; approximate)
    groups[arm + "_head"] = torch.cat(head)
    if tail:
        groups[arm + "_tail"] = torch.cat(tail)
    print(arm, "done", round(time.time() - t0), "s", flush=True)
hd.remove()

res = {"tokens": {g: int(H.shape[0]) for g, H in groups.items()}, "groups": {}}
Hu = groups["none_head"]
mu = Hu.mean(0)
res["mean_norm_unedited_head"] = round(mu.norm().item(), 3)
cov = torch.cov(Hu.T)
evals, evecs = torch.linalg.eigh(cov)
evals = evals.clamp(min=0).flip(0); evecs = evecs.flip(1)
tr = evals.sum().item()
res["top_eigs_share_pct"] = [round(100 * e / tr, 2) for e in evals[:10].tolist()]
absmean = Hu.abs().mean(0)
massive = absmean.topk(8)
res["massive_dims"] = dict(idx=massive.indices.tolist(), mean_abs=[round(v, 2) for v in massive.values.tolist()], median_mean_abs=round(absmean.median().item(), 3))
res["dirs"] = {}
for n, d in dirs.items():
    w = (evecs.T @ d) ** 2
    k = ((d - d.mean()) ** 4).mean() / ((d - d.mean()) ** 2).mean() ** 2
    top = d.abs().topk(5)
    res["dirs"][n] = dict(cos_mean=round(float(d @ F.normalize(mu, dim=0)), 4), var_share_pct=round(100 * float(d @ cov @ d) / tr, 3),
                          mass_top10_eig_pct=round(100 * w[:10].sum().item(), 2), mass_top50_eig_pct=round(100 * w[:50].sum().item(), 2),
                          kurtosis=round(k.item(), 2), max_abs_coord=round(top.values[0].item(), 3), top_coords=top.indices.tolist(),
                          mass_on_massive_dims_pct=round(100 * (d[massive.indices] ** 2).sum().item(), 3))
res["cos_between"] = {f"{a}|{b}": round(float(dirs[a] @ dirs[b]), 4) for a in dirs for b in dirs if a < b}
for g, H in groups.items():
    nrm = H.norm(dim=-1)
    mg = H.mean(0)
    row = dict(mean_norm=round(nrm.mean().item(), 3))
    for n, d in dirs.items():
        p = H @ d
        pc = (H - mu) @ d  # centered against the unedited-head mean
        r = p.abs() / nrm
        row[n] = dict(dose=round(r.mean().item(), 4), median=round(r.median().item(), 4), p90=round(r.quantile(0.9).item(), 4),
                      signed_mean_over_abs=round((p.mean() / p.abs().mean()).item(), 3), centered_dose=round((pc.abs() / nrm).mean().item(), 4),
                      share_pct=round(100 * (p ** 2 / nrm ** 2).mean().item(), 3), frac_gt_0p1=round((r > 0.1).float().mean().item(), 3))
    res["groups"][g] = row
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
