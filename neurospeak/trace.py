"""How one dictionary word feeds another: from text, through a lower layer, to a word that pushes the answer."""
from __future__ import annotations

import torch

from . import core, reports as R
from .why import _replace, _residual, attribute_to, removed_logits, why


def _pre(F, residual, pos: int, f: int):
    """The word's activation before the top-k cut, so it stays meaningful when the word drops out of the top k."""
    x = residual[0, pos].float() * F.scale
    return (x - F.sae.b_dec) @ F.sae.W_enc[f] + F.sae.b_enc[f]


def parents(brain, ids, parent_layer: int, child_layer: int, pos: int, f: int):
    """Gradient times activation of every lower-layer word for one higher-layer word. Returns (values, indices, scores)."""
    child = brain.features[child_layer]
    box = {}
    handle = brain.layers[child_layer].register_forward_hook(lambda _m, _i, o: box.__setitem__("h", _residual(o)))
    try:
        return attribute_to(brain, ids, parent_layer, lambda _logits: _pre(child, box["h"], pos, f))
    finally:
        handle.remove()


@torch.no_grad()
def child_after(brain, ids, parent_layer: int, child_layer: int, pos: int, f: int, removals: list[list[tuple]]) -> list[float]:
    """The child's activation with each list of parent words removed."""
    child = brain.features[child_layer]
    out = []
    handle = brain.layers[child_layer].register_forward_hook(lambda _m, _i, o: out.append(float(_pre(child, _residual(o)[:1], pos, f))))
    try:
        for words in removals:
            removed_logits(brain, ids, parent_layer, [words])
    finally:
        handle.remove()
    return out


@torch.no_grad()
def lead_with_held(brain, ids, parent_layer: int, child_layer: int, words: list[tuple], held: tuple[int, int] | None, answer: int, against: int) -> float:
    """The lead with parent `words` removed while the held (pos, feature) word is kept at its original activation."""
    child = brain.features[child_layer]
    box = {}
    handle = None
    if held is not None:
        pos, f = held
        direction = child.sae.W_dec.detach()[:, f]

        def hook(_m, _i, o):
            h = _residual(o)
            now = _pre(child, h, pos, f)
            h = h.clone()
            h[0, pos] += ((box["orig"] - now) / child.scale * direction).to(h.dtype)
            return _replace(o, h)

        grab = brain.layers[child_layer].register_forward_hook(lambda _m, _i, o: box.__setitem__("orig", _pre(child, _residual(o), pos, f)))
        brain.m(ids, use_cache=False, logits_to_keep=1)
        grab.remove()
        handle = brain.layers[child_layer].register_forward_hook(hook)
    try:
        lg = removed_logits(brain, ids, parent_layer, [words])[0]
    finally:
        if handle:
            handle.remove()
    return float(lg[answer] - lg[against])


@torch.no_grad()
def sources(brain, ids, child_layer: int, pos: int, f: int, window: int = 32) -> list[tuple[int, float]]:
    """Replace each recent token with a neutral one and rerun: how much does the word fall? Returns (position, fall), biggest first."""
    child = brain.features[child_layer]
    box = {}
    handle = brain.layers[child_layer].register_forward_hook(lambda _m, _i, o: box.__setitem__("h", _residual(o)))
    neutral = brain.tok.eos_token_id if brain.tok.eos_token_id is not None else 0
    falls = []
    try:
        brain.m(ids, use_cache=False, logits_to_keep=1)
        base = float(_pre(child, box["h"], pos, f))
        for p in range(max(0, pos - window + 1), pos + 1):
            changed = ids.clone()
            changed[0, p] = neutral
            brain.m(changed, use_cache=False, logits_to_keep=1)
            falls.append((p, base - float(_pre(child, box["h"], pos, f))))
    finally:
        handle.remove()
    return sorted(falls, key=lambda x: -x[1])


def trace(brain, text: str, at: int = 19, down_to: int = 15, top: int = 3, parents_each: int = 3, **options) -> R.Trace:
    """The explanation chain for the next token: which words pushed it, what fed them, and which text fed those."""
    report = why(brain, text, at=at, top=top, **options)
    ids = brain.tok(text, return_tensors="pt").input_ids.to(core.DEV)
    toks = [brain.tok.decode([t]) for t in ids[0].tolist()]
    lower = brain.features[down_to]
    records = core.word_records(brain, down_to)
    links = []
    for w in report.words:
        f = int(w.feature.split(".f")[1])
        vals, idx, scores = parents(brain, ids, down_to, at, w.position, f)
        k = scores.shape[1]
        chosen = [i for i in scores.view(-1).topk(parents_each).indices.tolist() if scores.view(-1)[i] > 0]
        words = [(i // k, int(idx[i // k, i % k]), float(vals[i // k, i % k])) for i in chosen]
        base = child_after(brain, ids, down_to, at, w.position, f, [[]])[0]
        after = child_after(brain, ids, down_to, at, w.position, f, [[x] for x in words])
        feeders = [R.TraceParent(str(lower[g]), p, toks[p], a - base, records.get(g)) for (p, g, _), a in zip(words, after)]
        text_sources = [(toks[p], fall) for p, fall in sources(brain, ids, at, w.position, f)[:3] if fall > 0]
        links.append(R.TraceLink(w, feeders, text_sources))
    return R.Trace(report, links)
