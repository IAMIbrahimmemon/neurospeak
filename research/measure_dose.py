"""Study 21 review: how much of the residual each direction removes per layer, CPU, 6 unedited prompts. Written by Claude (Fable) during review."""
import json, torch, time, statistics as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from neurospeak.anatomy import discover
torch.set_num_threads(8)
mid="Qwen/Qwen3.5-0.8B"
tok=AutoTokenizer.from_pretrained(mid)
m=AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to("cpu").eval()
A=discover(m); print("blocks", len(A.blocks))
D=torch.load("runs/whyfix/directions.pt"); X=torch.load("runs/research/xfer_dirs.pt"); Lr=torch.load("runs/research/layer_dirs.pt")
u=D["dir_self_doubt"].float()
dirs={}
for l in (7,11,15,19,23):
    dirs[l]={"L19vec":u}
    if l!=19: dirs[l]["xfer_ctl"]=X[f"dir_xfer_{l}_matched"].float()
    dirs[l]["own"]=Lr[f"dir_layer_{l}"].float(); dirs[l]["own_ctl"]=Lr[f"dir_layer_{l}_matched"].float()
acts={l:[] for l in dirs}
hs=[]
def mk(l):
    def f(_m,_i,o):
        h=o[0] if isinstance(o,tuple) else o
        acts[l].append(h[0].detach().float())
    return f
hands=[A.blocks[l].register_forward_hook(mk(l)) for l in dirs]
rows=[json.loads(x) for x in open("runs/research/layers_none_traces.jsonl")][:6]
t0=time.time()
with torch.no_grad():
    for r in rows:
        text=r["prompt"]+r["completion"][:3000]
        ids=tok(text,return_tensors="pt").input_ids[:, :448]
        out=m(ids, output_hidden_states=True)
        hs.append([h[0].float() for h in out.hidden_states])
print("forward done", round(time.time()-t0,1),"s, tokens", sum(a.shape[0] for a in acts[7]))
res={}
for l in dirs:
    H=torch.cat([a[1:] for a in acts[l]])  # drop position 0
    n=H.norm(dim=-1)
    res[l]={"mean_resid_norm":round(n.mean().item(),2),"median_resid_norm":round(n.median().item(),2)}
    for nm,d in dirs[l].items():
        p=H@d
        share=(p**2/n**2)
        res[l][nm]=dict(mean_abs_proj=round(p.abs().mean().item(),3), mean_proj=round(p.mean().item(),3), mean_abs_ratio=round((p.abs()/n).mean().item(),4), median_abs_ratio=round((p.abs()/n).median().item(),4), mean_share_pct=round(100*share.mean().item(),3), frac_tokens_ratio_gt_0p1=round((p.abs()/n>0.1).float().mean().item(),3))
    print(l, json.dumps(res[l]))
# post-norm check at L23
H23=torch.cat([a[1:] for a in acts[23]]); HS24=torch.cat([h[24][1:] for h in hs]); HS23=torch.cat([h[23][1:] for h in hs])
print("block23 out vs hidden_states[24]: same?", torch.allclose(H23,HS24,atol=1e-3), "norm block23", round(H23.norm(dim=-1).mean().item(),2), "norm hs24", round(HS24.norm(dim=-1).mean().item(),2))
print("block22 out == hidden_states[23]?", torch.allclose(torch.cat([a[1:] for a in acts[19]]), torch.cat([h[20][1:] for h in hs]), atol=1e-3))
m23=H23.mean(0); m24=HS24.mean(0)
print("cos(mean block23, mean hs24)", round(float(torch.nn.functional.normalize(m23,dim=0)@torch.nn.functional.normalize(m24,dim=0)),3))
print("position-0 norms by layer:", {l: round(acts[l][0][0].norm().item(),1) for l in dirs})
json.dump({str(k):v for k,v in res.items()}, open("runs/research/xfer_dose.json","w"), indent=1)
