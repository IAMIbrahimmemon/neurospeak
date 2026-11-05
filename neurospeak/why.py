"""Why did the model say this? Finds the dictionary words that pushed one next-token decision, then removes each word and
reruns the model to check. A word's `predicted` is a linear estimate; its `measured` is what removal actually did."""
from __future__ import annotations

import torch

from . import core, reports as R

BATCH = 16


def _residual(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h, *out[1:]) if isinstance(out, tuple) else h


def _objective(logits, answer: int, against: int | None):
    lg = logits[0, -1].float()
    return lg[answer] - lg[against] if against is not None else lg.log_softmax(-1)[answer]


def attribute(brain, ids, layer: int, answer: int, against: int | None = None):
    """Gradient times activation of every active word for the answer's lead. Returns (values, indices, scores), each [T, k]."""
    return attribute_to(brain, ids, layer, lambda logits: _objective(logits, answer, against))


def attribute_to(brain, ids, layer: int, objective):
    """Gradient times activation for any `objective(logits)`. The reconstruction error stays in the model as a constant, so the
    model's output doesn't change."""
    F = brain.features[layer]
    sae, dtype = F.sae, brain.dtype
    box = {}

    def hook(_m, _i, out):
        h = _residual(out)[0].float()
        x = h.detach() * F.scale
        vals, idx = sae.encode(x)
        vals = vals.detach().requires_grad_(True)
        recon = (sae.W_dec.T[idx] * vals.unsqueeze(-1)).sum(-2) + sae.b_dec
        err = (x - recon).detach()
        box.update(vals=vals, idx=idx)
        return _replace(out, ((recon + err) / F.scale).to(dtype)[None])

    handle = brain.layers[layer].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = brain.m(ids, use_cache=False, logits_to_keep=1).logits
            grad = torch.autograd.grad(objective(logits), box["vals"])[0]
    finally:
        handle.remove()
    return box["vals"].detach(), box["idx"], (grad * box["vals"]).detach()


@torch.no_grad()
def removed_logits(brain, ids, layer: int, removals: list[list[tuple[int, int, float]]]) -> torch.Tensor:
    """Final-token logits with each list of (position, feature or direction, value) subtracted from the residual, one run per list."""
    F = brain.features[layer]
    dec = F.sae.W_dec.detach()
    out = []
    for s in range(0, len(removals), BATCH):
        part = removals[s:s + BATCH]

        def hook(_m, _i, o):
            h = _residual(o).clone()
            for row, words in enumerate(part):
                for pos, f, v in words:
                    direction = f if torch.is_tensor(f) else dec[:, f]
                    h[row, pos] -= (v / F.scale * direction).to(h.dtype)
            return _replace(o, h)

        handle = brain.layers[layer].register_forward_hook(hook)
        try:
            out.append(brain.m(ids.repeat(len(part), 1), use_cache=False, logits_to_keep=1).logits[:, -1].float())
        finally:
            handle.remove()
    return torch.cat(out)


def _show(brain, token_id: int) -> str:
    return repr(brain.tok.decode([token_id]))[1:-1]


def why(brain, text: str, at: int = 19, answer: str | int | None = None, against: str | int | None = None, top: int = 8,
        check: bool = True) -> R.Why:
    """Explains the next token after `text`. `answer` defaults to the model's top choice, `against` to its runner-up."""
    def token_id(t):
        if isinstance(t, int):
            return t
        got = brain.tok(t, add_special_tokens=False).input_ids
        if len(got) != 1:
            raise ValueError(f"{t!r} is {len(got)} tokens; why() explains a single next token")
        return got[0]

    ids = brain.tok(text, return_tensors="pt").input_ids.to(core.DEV)
    with torch.no_grad():
        base = brain.m(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    order = base.topk(2).indices.tolist()
    a = token_id(answer) if answer is not None else order[0]
    b = token_id(against) if against is not None else (order[1] if order[0] == a else order[0])

    vals, idx, scores = attribute(brain, ids, at, a, b)
    chosen = scores.view(-1).topk(min(top, scores.numel())).indices.tolist()
    k = scores.shape[1]
    words = [(i // k, int(idx[i // k, i % k]), float(vals[i // k, i % k]), float(scores[i // k, i % k])) for i in chosen]
    words = [w for w in words if w[3] > 0]

    gap0 = float(base[a] - base[b])
    measured, flipped_all, gap_all = [None] * len(words), None, None
    if check and words:
        lists = [[(p, f, v)] for p, f, v, _ in words] + [[(p, f, v) for p, f, v, _ in words]]
        lg = removed_logits(brain, ids, at, lists)
        gaps = (lg[:, a] - lg[:, b]).tolist()
        measured = [g - gap0 for g in gaps[:-1]]
        gap_all = gaps[-1]
        flipped_all = int(lg[-1].argmax()) != a

    F = brain.features[at]
    toks = [brain.tok.decode([t]) for t in ids[0].tolist()]
    records = core.word_records(brain, at)
    return R.Why(answer=_show(brain, a), against=_show(brain, b), p_answer=float(base.softmax(-1)[a]), gap=gap0,
                 words=[R.WhyWord(str(F[f]), p, toks[p], v, s, m, records.get(f))
                        for (p, f, v, s), m in zip(words, measured)],
                 gap_without=gap_all, flipped=flipped_all, context=toks[-12:],
                 rounding=0.25 if brain.dtype in (torch.bfloat16, torch.float16) else 0.0)
