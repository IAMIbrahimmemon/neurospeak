"""Effect lists: add each unit's direction at the same neutral contexts as every other unit in its layer, so the context can't
identify the unit. Also adds a random direction of the same size, which supplies the null lists.

    .venv/bin/python wordtest/effect.py --strength 3
"""
import argparse, json, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurospeak.anatomy import discover
from neurospeak.core import DEV
from neurospeak.sae import TopKSAE, checkpoint_dir

WINDOW, N_CONTEXTS, TOP, MIN_COUNT = 64, 20, 8, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/wordtest")
    ap.add_argument("--strength", type=float, required=True, help="multiple of the unit's typical strong firing")
    ap.add_argument("--limit", type=int, default=None, help="pilot: only this many units of each kind")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = torch.load(f"{a.dir}/scan.pt", map_location="cpu")
    model, SEQ, SKIP = S["model"], S["seq"], S["skip"]
    tok = AutoTokenizer.from_pretrained(model)
    T = S["tokens"].view(-1)
    half = S["half"].repeat_interleave(SEQ)
    m = AutoModelForCausalLM.from_pretrained(model, dtype="auto").to(DEV).eval()
    m.requires_grad_(False)
    an = discover(m)
    dec, cols = {}, {}
    for L in S["layers"]:
        c = torch.load(f"{checkpoint_dir(model)}/L{L}.pt", map_location="cpu")
        dec[L] = c["state"]["W_dec"].to(DEV)
        cols[L] = an.columns(an.mlp_out[L] + ".weight", an.mlp_module(L).weight).detach().float()

    rng = random.Random(31)
    gen = torch.Generator().manual_seed(32)
    pool = ((half == 1) & (torch.arange(len(T)) % SEQ >= WINDOW - 1)).nonzero().squeeze(1).tolist()
    contexts = {L: rng.sample(pool, N_CONTEXTS) for L in S["layers"]}
    batch = {L: torch.stack([T[p - WINDOW + 1:p + 1] for p in contexts[L]]).long().to(DEV) for L in S["layers"]}
    base = {}
    with torch.no_grad():
        for L in S["layers"]:
            base[L] = m(batch[L], use_cache=False, logits_to_keep=1).logits[:, -1].float().softmax(-1)

    def values(r):
        U = S["units"][r["layer"]]
        if r["kind"] == "neuron":
            return U["neuron_acts"][:, S["neurons"][r["layer"]].index(r["index"])]
        mask = U["idx"].long() == r["index"]
        rows = mask.any(1)
        v = torch.zeros(len(T))
        v[rows] = (U["vals"][rows].float() * mask[rows].float()).sum(1)
        return v

    records = [r for r in json.load(open(f"{a.dir}/units.json"))["records"] if r["causal"]]
    if a.limit:
        records = [r for r in records if r["kind"] == "feature"][:a.limit] + [r for r in records if r["kind"] == "neuron"][:a.limit]

    # A consistent push across contexts, in common words. Raw probability change is dominated by one context's likely word.
    common = (torch.bincount(T.long(), minlength=base[S["layers"][0]].shape[1]) >= MIN_COUNT).to(DEV)

    def top(d, k=TOP):
        return d.mean(0).masked_fill(~common, -1e9).topk(k).indices.tolist()

    out = {}
    for k, r in enumerate(records):
        L, ix = r["layer"], r["index"]
        u = dec[L][:, ix] if r["kind"] == "feature" else cols[L][:, ix] / cols[L][:, ix].norm()
        rnd = torch.randn(an.hidden_size, generator=gen).to(DEV)
        rnd = rnd / rnd.norm()
        amount = a.strength * r["causal"]["amount"]
        quiet = torch.tensor([bool(x) for x in (values(r)[contexts[L]] <= r["hi"]).tolist()], device=DEV)
        ids = batch[L].repeat(2, 1)

        def hook(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            new = h.clone()
            new[:N_CONTEXTS, -1] += (amount * u).to(h.dtype)
            new[N_CONTEXTS:, -1] += (amount * rnd).to(h.dtype)
            return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
        hk = an.blocks[L].register_forward_hook(hook)
        with torch.no_grad():
            p = m(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().softmax(-1)
        hk.remove()
        b = base[L][quiet]
        lp = lambda q: q.clamp_min(1e-12).log()
        du, dr = lp(p[:N_CONTEXTS][quiet]) - lp(b), lp(p[N_CONTEXTS:][quiet]) - lp(b)
        kl = lambda q: (q * (q.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(-1).mean().item()
        likely = set(b.mean(0).topk(TOP).indices.tolist())
        overlap = lambda d: len(set(top(d[0::2])) & set(top(d[1::2])))
        out[r["id"]] = {"kind": r["kind"], "layer": L, "strength": a.strength, "contexts": int(quiet.sum()),
                        "steer": [tok.decode([t]) for t in top(du)], "steer_random": [tok.decode([t]) for t in top(dr)],
                        "split_half": overlap(du), "split_half_random": overlap(dr),
                        "kl": kl(p[:N_CONTEXTS][quiet]), "kl_random": kl(p[N_CONTEXTS:][quiet]),
                        "likely_overlap": len(set(top(du)) & likely), "likely_overlap_random": len(set(top(dr)) & likely)}
        if k % 40 == 0: print(f"effect {k}/{len(records)}", flush=True)
    json.dump(out, open(a.out or f"{a.dir}/effect.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
