"""What did this weight edit change? Compares each dictionary word's firing with and without an edit, on the same text."""
from __future__ import annotations

import torch

from . import core, reports as R


@torch.no_grad()
def word_rates(brain, texts: list[str], layer: int, seq: int = 256) -> tuple[torch.Tensor, int]:
    """Total activation per word over the texts, and the token count."""
    F = brain.features[layer]
    box = {}
    handle = brain.layers[layer].register_forward_hook(lambda _m, _i, o: box.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    total, tokens = torch.zeros(F.sae.n), 0
    try:
        for t in texts:
            ids = brain.tok(t, return_tensors="pt").input_ids[:, :seq].to(core.DEV)
            brain.m(ids, use_cache=False, logits_to_keep=1)
            vals, idx = F.sae.encode(box["h"][0, 2:].float() * F.scale)
            total.index_add_(0, idx.view(-1).cpu(), vals.view(-1).cpu())
            tokens += vals.shape[0]
    finally:
        handle.remove()
    return total, tokens


def diff(brain, edit_path: str, texts: list[str], at: int = 19, top: int = 10) -> R.WordDiff:
    before, tokens = word_rates(brain, texts, at)
    with brain.edit(strict=False) as e:
        e.load(edit_path)
        after, _ = word_rates(brain, texts, at)
    change = (after - before) / tokens
    records = core.word_records(brain, at)
    F = brain.features[at]
    pick = lambda order: [R.WordChange(str(F[i]), float(before[i] / tokens), float(after[i] / tokens), records.get(i)) for i in order.indices.tolist()]
    return R.WordDiff(edit_path, tokens, pick((-change).topk(top)), pick(change.topk(top)), float(change.abs().topk(20).values.sum() / change.abs().sum().clamp(min=1e-9)), change)


def contrast(brain, a_texts: list[str], b_texts: list[str], at: int = 19, top: int = 10) -> R.WordDiff:
    """Which dictionary words fire more on texts A than on texts B? No edit involved: this compares two kinds of behavior."""
    a, n_a = word_rates(brain, a_texts, at)
    b, n_b = word_rates(brain, b_texts, at)
    change = a / n_a - b / n_b
    records = core.word_records(brain, at)
    F = brain.features[at]
    pick = lambda order: [R.WordChange(str(F[i]), float(b[i] / n_b), float(a[i] / n_a), records.get(i)) for i in order.indices.tolist()]
    return R.WordDiff("A vs B", n_a + n_b, pick((-change).topk(top)), pick(change.topk(top)), float(change.abs().topk(20).values.sum() / change.abs().sum().clamp(min=1e-9)), change)
