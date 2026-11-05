"""Measures how many L19 dictionary words carry the direction a loop edit removes. See PLAN_SPREAD.md."""
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurospeak.anatomy import discover
from neurospeak.core import DEV, load_edit_file
from neurospeak.sae import TopKSAE

SEQ, SKIP, BATCH, L, MODEL = 512, 2, 8, 19, "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto").to(DEV).eval()
an = discover(m)
c = torch.load(f"runs/sae/{MODEL.replace('/', '__')}/L{L}.pt", map_location=DEV)
sae = TopKSAE(c["config"]["d"], c["config"]["n"], c["config"]["k"]).to(DEV)
sae.load_state_dict(c["state"]); sae.requires_grad_(False)
W = sae.W_dec.detach()


def edit_direction(path):
    params = dict(m.named_parameters())
    for name, Wn in load_edit_file(m, path).items():
        if an.locate(name)[0] != L:
            continue
        W0 = an.columns(name, params[name].detach()).float().cpu()
        D = an.columns(name, Wn).float().cpu() - W0
        cols = (D.abs().sum(0) > 0).nonzero()[:, 0]
        U, S, _ = torch.linalg.svd(D[:, cols], full_matrices=False)
        return F.normalize(U[:, 0], dim=0).to(DEV), (S[0] ** 2 / (S ** 2).sum()).item()
    raise KeyError(path)


dirs, rank1 = {}, {}
for name, path in (("neurospeak", "runs/neurospeak_repetition_edit.pt"), ("astra", "data/astra_p50c_edit.npz")):
    dirs[name], rank1[name] = edit_direction(path)
torch.manual_seed(0)
for i in range(20):
    dirs[f"random{i}"] = F.normalize(torch.randn(c["config"]["d"], device=DEV), dim=0)
names = list(dirs)
D = torch.stack([dirs[n] for n in names])

grab = {}
an.blocks[L].register_forward_hook(lambda _m, _i, out: grab.__setitem__("x", out[0] if isinstance(out, tuple) else out))
ids = []
for l in open("runs/whyfix/mine_traces.jsonl"):
    r = json.loads(l)
    t = tok(r["prompt"] + r["completion"]).input_ids
    ids += [t[i:i + SEQ] for i in range(0, len(t) - SEQ + 1, SEQ)]
T = torch.tensor(ids)

word = torch.zeros(len(names), sae.n, device=DEV)
err = torch.zeros(len(names), device=DEV)
coef = D @ W        # dirs x words
for s in range(0, len(T), BATCH):
    with torch.no_grad():
        m(T[s:s + BATCH].to(DEV), use_cache=False)
        x = grab["x"][:, SKIP:].reshape(-1, c["config"]["d"]).float() * c["scale"]
        recon, vals, idx, _ = sae(x)
        e = (x - recon) @ D.T
        err += e.abs().sum(0)
        for k in range(len(names)):
            word[k].index_add_(0, idx.view(-1), (vals * coef[k][idx].abs()).view(-1))

out = dict(tokens=int((T.shape[1] - SKIP) * T.shape[0]), rank1=rank1, dirs={})
W_lists = json.load(open("runs/whyfix/words.json"))
for k, n in enumerate(names):
    total = word[k].sum().item() + err[k].item()
    order = word[k].argsort(descending=True)
    cum = word[k][order].cumsum(0) / word[k].sum()
    rec = dict(top16_share_of_words=cum[15].item(), top16_share_of_all=word[k][order[:16]].sum().item() / total, error_share=err[k].item() / total,
               words_for_50=int((cum < .5).sum()) + 1, words_for_90=int((cum < .9).sum()) + 1,
               top16=order[:16].tolist(), nearest_cos=coef[k].abs().max().item(), nearest=coef[k].abs().argmax().item())
    if not n.startswith("random"):
        rec["overlap"] = {a: len(set(rec["top16"]) & set(W_lists[a])) for a in ("active", "active2", "why2", "contrast2", "diff")}
    out["dirs"][n] = rec
out["cos_astra_neurospeak"] = (dirs["astra"] @ dirs["neurospeak"]).abs().item()
json.dump(out, open("runs/whyfix/spread.json", "w"), indent=1)
rnd = [v for n, v in out["dirs"].items() if n.startswith("random")]
print(json.dumps({n: {k: v for k, v in r.items() if k != "top16"} for n, r in out["dirs"].items() if not n.startswith("random")}, indent=1))
for f in ("top16_share_of_all", "top16_share_of_words", "error_share", "words_for_50", "words_for_90", "nearest_cos"):
    v = sorted(r[f] for r in rnd)
    print("random", f, "min", round(v[0], 3), "median", round(v[10], 3), "max", round(v[-1], 3))
print("cos", out["cos_astra_neurospeak"], "tokens", out["tokens"])
