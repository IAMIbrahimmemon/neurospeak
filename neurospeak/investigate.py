"""Debug a behavior: build a direction from examples, remove it, and measure what changes against controls."""
from __future__ import annotations

import inspect
import random
import re

import torch
import torch.nn.functional as F

from . import core, reports as R

TEMPERATURE, TOP_P, TOP_K = 0.6, 0.95, 20
MAX_COST = 0.05          # nats per token on normal text
COLLAPSE = re.compile(r"(\S.{0,3}?)\1{24,}", re.S)


def collapsed(text: str) -> bool:
    """True if the text has degenerated into one short unit repeated. Word-based loop measures can't see this."""
    return bool(COLLAPSE.search(text))


def complete(brain, prompts: list[str], max_new: int, seed: int, thinking: bool = True, batch: int = 8) -> list[str]:
    """Sampled continuations, in the same order as prompts."""
    tok = brain.tok
    texts = [brain.chat_prompt(p, thinking) for p in prompts]
    device = next(brain.m.parameters()).device
    can_pad = hasattr(tok, "pad")
    size = batch if can_pad else 1
    if can_pad:
        side = tok.padding_side
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    torch.manual_seed(seed)
    out = []
    try:
        for s in range(0, len(texts), size):
            part = texts[s:s + size]
            enc = tok(part, return_tensors="pt", padding=True).to(device) if can_pad else None
            ids = enc.input_ids if can_pad else tok(part[0], return_tensors="pt").input_ids.to(device)
            extra = dict(attention_mask=enc.attention_mask) if can_pad else {}
            with torch.no_grad():
                g = brain.m.generate(ids, **extra, max_new_tokens=max_new, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
            out += [tok.decode(row[ids.shape[1]:], skip_special_tokens=True) for row in g]
    finally:
        if can_pad:
            tok.padding_side = side
    return out


def mean_residual(brain, layer: int, texts: list[str]) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([brain.run(t[:2000])[1][layer + 1] for t in texts]).mean(0)


def matched_control(direction: torch.Tensor, mean: torch.Tensor, seed: int) -> torch.Tensor:
    """A random unit direction with the same overlap with the residual's mean, so removing it shrinks the mean as much."""
    m = F.normalize(mean, dim=0)
    u = F.normalize(direction, dim=0)
    overlap = (u @ m).clamp(-0.999, 0.999)
    r = torch.randn(len(u), generator=torch.Generator().manual_seed(seed))
    r = F.normalize(r - (r @ m) * m, dim=0)
    return F.normalize(overlap * m + (1 - overlap ** 2).sqrt() * r, dim=0)


def paired(x: list[float], y: list[float], seed: int = 0, n: int = 10000) -> tuple[float, float, float]:
    """Mean of x - y, with a 95% bootstrap interval over items."""
    d = [a - b for a, b in zip(x, y)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
    return sum(d) / len(d), means[int(.025 * n)], means[int(.975 * n)]


def text_loss(brain, texts: list[str]) -> float:
    with torch.no_grad():
        return torch.cat([-brain.run(t[:2000], capture=False)[3] for t in texts]).mean().item()


def judge(base: float, removed: float, change: tuple, versus: tuple, collapsed_base: int, collapsed_removed: int, cost: float | None, least: float) -> list[R.Check]:
    checks = [R.Check("The behavior is present", f"at least {least:g} unedited", f"{base:.3g}", base >= least),
              R.Check("It falls", "at most two thirds of unedited, interval below 0", f"{removed:.3g} from {base:.3g}, change {change[0]:+.3g} ({change[1]:+.3g} to {change[2]:+.3g})",
                      removed <= base * 2 / 3 and change[2] < 0),
              R.Check("It beats a matched random direction", "interval below 0", f"{versus[0]:+.3g} ({versus[1]:+.3g} to {versus[2]:+.3g})", versus[2] < 0),
              R.Check("The output stays intact", "no more collapsed texts than unedited", f"{collapsed_removed} against {collapsed_base}", collapsed_removed <= collapsed_base)]
    if cost is not None:
        checks.append(R.Check("Normal text stays cheap", f"at most {MAX_COST} nats per token", f"{cost:+.3f}", cost <= MAX_COST))
    return checks


def learn_direction(brain, prompts: list[str], score, layer: int, max_new: int, seed: int, thinking: bool) -> torch.Tensor:
    """Generate unedited, then take mean residual over the top third of outputs minus the bottom third."""
    texts = complete(brain, prompts, max_new, seed, thinking)
    scores = [score(t, q) for t, q in zip(texts, prompts)]
    order = sorted(range(len(texts)), key=lambda i: scores[i])
    third = max(len(order) // 3, 1)

    def mean_over(ids):
        vecs = []
        with torch.no_grad():
            for i in ids:
                s, _, hs, _, _ = brain._continuation(brain.chat_prompt(prompts[i], thinking), texts[i])
                if len(hs[layer + 1][s:]):
                    vecs.append(hs[layer + 1][s:].mean(0))
        return torch.stack(vecs).mean(0)
    return F.normalize(mean_over(order[-third:]) - mean_over(order[:third]), dim=0)


def investigate(brain, name: str, pairs: list | None, prompts: list[str], measure, at: int | None = None, max_new: int = 200, seed: int = 0, thinking: bool = True,
                normal: list[str] | None = None, least: float = 0.0, show: int = 2, learn: list[str] | None = None) -> R.Investigation:
    """`measure(text)` or `measure(text, prompt)` scores a text, higher meaning more of the behavior. The direction comes from example `pairs`, or from
    the model's own outputs on `learn` prompts. `prompts` test it, and shouldn't overlap with either."""
    takes_prompt = len(inspect.signature(measure).parameters) == 2
    score = (lambda t, q: float(measure(t, q))) if takes_prompt else (lambda t, q: float(measure(t)))
    if learn is not None:
        if at is None:
            raise ValueError("learning a direction needs a layer: pass at=")
        layer, u = at, learn_direction(brain, learn, score, at, max_new, seed, thinking).cpu().float()
    else:
        # build the direction in the same context it will be tested in
        same_context = [core.Pair(p.is_, p.not_, None, brain.chat_prompt(p.ask, thinking)) if p.ask else p for p in pairs]
        concept = brain.concept(name, same_context, at=[at] if at is not None else None)
        layer = at if at is not None else concept.where().best
        u = concept.dirs[layer].cpu().float()
    normal = normal if normal is not None else core._default_corpus()[:24]
    control = matched_control(u, mean_residual(brain, layer, normal), seed)

    def run(direction=None):
        if direction is None:
            return complete(brain, prompts, max_new, seed, thinking)
        with brain.without(direction, at=layer):
            return complete(brain, prompts, max_new, seed, thinking)

    texts = dict(unedited=run(), removed=run(u), control=run(control))
    scores = {k: [score(t, q) for t, q in zip(v, prompts)] for k, v in texts.items()}
    mean = {k: sum(v) / len(v) for k, v in scores.items()}
    dead = {k: sum(collapsed(t) for t in v) for k, v in texts.items()}
    base_loss = text_loss(brain, normal) if normal else None
    cost = None
    if base_loss is not None:
        with brain.without(u, at=layer):
            cost = text_loss(brain, normal) - base_loss
    change, versus = paired(scores["removed"], scores["unedited"]), paired(scores["removed"], scores["control"])
    gaps = sorted(range(len(prompts)), key=lambda i: scores["removed"][i] - scores["unedited"][i])[:show]
    examples = [R.BeforeAfter(prompts[i], texts["unedited"][i], texts["removed"][i], scores["unedited"][i], scores["removed"][i]) for i in gaps]
    words = []
    try:
        records = core.word_records(brain, layer)
        words = [R.DirectionWord(str(h.feature), h.value, records.get(h.feature.index)) for h in brain.features[layer].nearest(u, top=5)]
    except (KeyError, ValueError):
        pass
    return R.Investigation(name, layer, len(prompts), mean["unedited"], mean["removed"], mean["control"], dead,
                           judge(mean["unedited"], mean["removed"], change, versus, dead["unedited"], dead["removed"], cost, least), examples, words, u, control, texts, scores)
