"""Exploratory, added after the first results and not in PLAN.md. Does the causal list test measure the unit or just the context?
For the same contexts, lists the tokens that fall when a random same-size direction is removed, and the tokens that were
simply most likely. Writes control.json and anonymous trials that use the random-removal lists in place of the real ones."""
import json, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurospeak.anatomy import discover
from neurospeak.core import DEV
from neurospeak.sae import TopKSAE, checkpoint_dir

D = sys.argv[1] if len(sys.argv) > 1 else "runs/wordtest"
WINDOW, GAP = 64, 10
S = torch.load(f"{D}/scan.pt", map_location="cpu")
model, SEQ, SKIP = S["model"], S["seq"], S["skip"]
tok = AutoTokenizer.from_pretrained(model)
T = S["tokens"].view(-1)
half = S["half"].repeat_interleave(SEQ)
usable = (torch.arange(len(T)) % SEQ) >= SKIP
m = AutoModelForCausalLM.from_pretrained(model, dtype="auto").to(DEV).eval()
m.requires_grad_(False)
an = discover(m)
saes = {}
for L in S["layers"]:
    c = torch.load(f"{checkpoint_dir(model)}/L{L}.pt", map_location=DEV)
    saes[L] = TopKSAE(c["config"]["d"], c["config"]["n"], c["config"]["k"]).to(DEV)
    saes[L].load_state_dict(c["state"]); saes[L].requires_grad_(False)
cols = {L: an.columns(an.mlp_out[L] + ".weight", an.mlp_module(L).weight).detach().float() for L in S["layers"]}
gen = torch.Generator().manual_seed(5)
records = json.load(open(f"{D}/units.json"))["records"]


def values(r):
    U = S["units"][r["layer"]]
    if r["kind"] == "neuron":
        return U["neuron_acts"][:, S["neurons"][r["layer"]].index(r["index"])]
    mask = U["idx"].long() == r["index"]
    rows = mask.any(1)
    v = torch.zeros(len(T))
    v[rows] = (U["vals"][rows].float() * mask[rows].float()).sum(1)
    return v


out = {}
for k, r in enumerate(records):
    if not r["causal"]: continue
    L, ix = r["layer"], r["index"]
    v = values(r)
    act = ((v > r["hi"]) & usable & (half == 1)).nonzero().squeeze(1)
    order = act[v[act].argsort(descending=True)].tolist()
    chosen = []
    for p in order:
        if p % SEQ >= WINDOW - 1 and all(abs(p - q) >= GAP for q in chosen):
            chosen.append(p)
        if len(chosen) == 6: break
    ids = torch.stack([T[p - WINDOW + 1:p + 1] for p in chosen]).to(DEV).long().repeat_interleave(2, 0)
    grabbed = {}
    pre = an.mlp_module(L).register_forward_pre_hook(lambda _m, inp: grabbed.__setitem__("x", inp[0]))
    used = []
    def hook(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        last = h[0::2, -1].float()
        if r["kind"] == "feature":
            vals, idc = saes[L].encode(last * S["scale"][L])
            amount = (vals * (idc == ix).float()).sum(1) / S["scale"][L]
        else:
            amount = grabbed["x"][0::2, -1, ix].float().relu() * cols[L][:, ix].norm()
        used.append(amount.cpu())
        rnd = torch.randn(len(chosen), an.hidden_size, generator=gen).to(DEV)
        rnd = rnd / rnd.norm(dim=1, keepdim=True)
        new = h.clone()
        new[1::2, -1] -= (amount.unsqueeze(1) * rnd).to(h.dtype)
        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
    hk = an.blocks[L].register_forward_hook(hook)
    with torch.no_grad():
        p = m(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().softmax(-1)
    hk.remove(); pre.remove()
    ok = used[0].to(DEV) > 0
    if not ok.any(): continue
    base, rand = p[0::2][ok], p[1::2][ok]
    random_list = [tok.decode([t]) for t in (base - rand).mean(0).topk(8).indices.tolist()]
    likely_list = [tok.decode([t]) for t in base.mean(0).topk(8).indices.tolist()]
    promotes = r["causal"]["promotes"]
    out[r["id"]] = {"random_removal": random_list, "most_likely": likely_list,
                    "overlap_random": len(set(promotes) & set(random_list)), "overlap_likely": len(set(promotes) & set(likely_list))}
    if k % 40 == 0: print(k, flush=True)
json.dump(out, open(f"{D}/control.json", "w"), ensure_ascii=False, indent=1)

labels = {x["id"]: x for f in sorted(__import__("glob").glob(f"{D}/labels_*.json")) for x in json.load(open(f))}
rng = random.Random(21)
groups = {}
for r in records:
    if r["id"] in out: groups.setdefault((r["kind"], r["layer"]), []).append(r["id"])
key, trials = {}, []
for ids_ in groups.values():
    rng.shuffle(ids_)
    for i, x in enumerate(ids_):
        o = ids_[(i + 1) % len(ids_)]
        answer = rng.choice("AB")
        a, b = (x, o) if answer == "A" else (o, x)
        trials.append((x, answer, {"label": {"fires_on": labels[x]["fires_on"], "effect": labels[x]["effect"]},
                                   "token_lists": {"A": out[a]["random_removal"], "B": out[b]["random_removal"]}}))
rng.shuffle(trials)
part = [[], []]
for i, (x, answer, t) in enumerate(trials):
    tid = f"C{i:03d}"
    key[tid] = {"unit": x, "answer": answer}
    part[i % 2].append({"trial": tid, **t})
for i, p_ in enumerate(part):
    json.dump(p_, open(f"{D}/control_batch_{i}.json", "w"), ensure_ascii=False, indent=1)
json.dump(key, open(f"{D}/control_key.json", "w"), indent=1)
print(len(trials), "control trials")
