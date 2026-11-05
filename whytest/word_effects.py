"""Measured effect of every word in round 2's R3 decisions and of a sample's fixed-context contexts. Context comes from the labeling half.

    .venv/bin/python whytest/word_effects.py
"""
import json, random
import torch

import neurospeak as ns
from neurospeak.core import DEV

OUT, L, WINDOW, N_CONTEXTS, TOP, MIN_COUNT, STRENGTH = "runs/whytest/round2", 19, 64, 20, 8, 20, 3


def main():
    S = torch.load("runs/wordtest/scan.pt", map_location="cpu")
    brain = ns.load(dtype=torch.float32)
    tok, SEQ = brain.tok, S["seq"]
    T = S["tokens"].view(-1)
    half = S["half"].repeat_interleave(SEQ)
    rng = random.Random(51)
    pool = ((half == 0) & (torch.arange(len(T)) % SEQ >= WINDOW - 1)).nonzero().squeeze(1).tolist()
    contexts = rng.sample(pool, N_CONTEXTS)
    ids = torch.stack([T[p - WINDOW + 1:p + 1] for p in contexts]).long().to(DEV)
    with torch.no_grad():
        base = brain.m(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
    common = (torch.bincount(T.long(), minlength=base.shape[1]) >= MIN_COUNT).to(DEV)

    D = json.load(open(f"{OUT}/reading.json"))
    words = sorted({f for t in D["r2"] for f in t["words"]})
    F = brain.features[L]
    dec = F.sae.W_dec.detach()
    U = S["units"][L]
    idx, vals = U["idx"].long(), U["vals"].float()
    show = lambda t: repr(tok.decode([t]))[1:-1]
    out, vectors = {}, {}
    for w in words:
        m = idx == w
        fired = vals[m]
        fired = fired[fired > 0]
        if len(fired) == 0:
            continue
        typical = float(fired.quantile(.9)) / F.scale
        direction = dec[:, w]

        def hook(_m, _i, o):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[:, -1] += (STRENGTH * typical * direction).to(h.dtype)
            return (h, *o[1:]) if isinstance(o, tuple) else h

        handle = brain.layers[L].register_forward_hook(hook)
        with torch.no_grad():
            lp = brain.m(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
        handle.remove()
        delta = (lp - base).mean(0)
        up = delta.masked_fill(~common, -1e9).topk(TOP).indices.tolist()
        down = (-delta).masked_fill(~common, -1e9).topk(TOP).indices.tolist()
        out[str(w)] = dict(raises=[show(t) for t in up], lowers=[show(t) for t in down])
        vectors[w] = delta.cpu()
    json.dump(out, open(f"{OUT}/word_effects.json", "w"), indent=1, ensure_ascii=False)
    torch.save(vectors, f"{OUT}/word_effect_vectors.pt")
    print("words", len(out), "of", len(words))


if __name__ == "__main__":
    main()
