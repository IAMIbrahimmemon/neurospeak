"""The library: a Brain holds a model, Concepts are words, Neurons and Heads are addresses."""
from __future__ import annotations

from contextlib import contextmanager
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import reports as R
from .anatomy import discover

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ───────────────────────────── edits on disk ─────────────────────────────

def _param_name_for(model, name: str) -> str:
    """Map checkpoint names ('model.language_model.layers.15...') onto this model's names."""
    names = dict(model.named_parameters())
    for cand in (name, name.replace("language_model.", "")):
        if cand in names:
            return cand
    raise KeyError(name)


def load_edit_file(model, path: str) -> dict[str, torch.Tensor]:
    """Return {param_name: edited full tensor}. Supports NeuroSpeak .pt and the Astra (exp8) .npz format."""
    path = str(path)
    if path.endswith(".pt"):
        return dict(torch.load(path))
    z = np.load(path)
    manifest = json.load(open(Path(path).with_name(Path(path).name.replace("_edit.npz", "_manifest.json"))))
    params = dict(model.named_parameters())
    out = {}
    for i, t in enumerate(manifest["tensors"]):
        name = _param_name_for(model, t["name"])
        w = params[name].detach().to("cpu").clone()
        idx = torch.from_numpy(z[f"t{i}_idx"].astype(np.int64))
        val = torch.from_numpy(z[f"t{i}_val"].astype(np.int16)).view(torch.bfloat16)
        w.view(-1)[idx] = val.to(w.dtype)
        out[name] = w
    return out


def apply_edit_file(model, path: str):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in load_edit_file(model, path).items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))


def loopiness(text: str, n: int = 8) -> float:
    """Share of word 8-grams in the last half of the text that already appeared earlier."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n)]
    if len(grams) < 20:
        return 0.0
    half = len(grams) // 2
    seen = set(grams[:half])
    rep = 0
    for g in grams[half:]:
        rep += g in seen
        seen.add(g)
    return rep / (len(grams) - half)


# ───────────────────────────── addresses ─────────────────────────────

@dataclass(frozen=True)
class Addr:
    layer: int
    kind: str          # "resid" | "mlp" | "attn" | "neuron" | "head"
    index: int | None = None

    def __str__(self):
        return {"resid": f"L{self.layer}", "mlp": f"L{self.layer}.mlp", "attn": f"L{self.layer}.attn",
                "neuron": f"L{self.layer}.n{self.index}", "head": f"L{self.layer}.h{self.index}"}[self.kind]


def parse_addr(s: str) -> Addr:
    m = re.fullmatch(r"L(\d+)(?:\.(mlp|attn|n(\d+)|h(\d+)))?", s.strip())
    if not m:
        raise SyntaxError(f"not an address: {s!r} (try L19, L19.mlp, L19.n11, L15.h7)")
    layer, part, n, h = int(m[1]), m[2], m[3], m[4]
    if part is None:
        return Addr(layer, "resid")
    if n is not None:
        return Addr(layer, "neuron", int(n))
    if h is not None:
        return Addr(layer, "head", int(h))
    return Addr(layer, part)


def as_addr(x) -> Addr:
    """Everything that takes an address also takes its written form, or a Neuron/Head handle."""
    if isinstance(x, str):
        return parse_addr(x)
    return getattr(x, "addr", x)


def as_addrs(x) -> list[Addr]:
    return [as_addr(a) for a in x] if isinstance(x, (list, tuple)) else [as_addr(x)]


@dataclass(frozen=True)
class Neuron:
    brain: "Brain"
    layer: int
    index: int

    @property
    def addr(self) -> Addr:
        return Addr(self.layer, "neuron", self.index)

    def __str__(self):
        return str(self.addr)

    __repr__ = __str__

    def output(self) -> torch.Tensor:
        return self.brain.weights(self.brain.mlp_param(self.layer))[:, self.index].float().cpu()

    def describe(self) -> R.NeuronReport:
        b = self.brain
        col = self.output()
        up, down = b.lens(col)
        acts, toks = b.scan(self.layer)
        a = acts[:, self.index].float()

        def contexts(order):
            found, seen = [], set()
            for i in order.topk(60).indices.tolist():
                if any(abs(i - s) < 12 for s in seen):
                    continue
                seen.add(i)
                found.append(R.Context(a[i].item(),
                                       b.tok.decode(toks[max(0, i - 14):i]).replace("\n", "⏎")[-60:],
                                       b.tok.decode(toks[i:i + 1]).replace("\n", "⏎"),
                                       b.tok.decode(toks[i + 1:i + 4]).replace("\n", "⏎")))
                if len(found) == 4:
                    break
            return found

        cos = {n: torch.nn.functional.cosine_similarity(col, c.dirs[self.layer], 0).item()
               for n, c in b.concepts.items() if self.layer in c.dirs}
        return R.NeuronReport(str(self), len(a), a.min().item(), a.max().item(), a.mean().item(),
                              up, down, contexts(a), contexts(-a), cos)


@dataclass(frozen=True)
class Head:
    brain: "Brain"
    layer: int
    index: int

    @property
    def addr(self) -> Addr:
        return Addr(self.layer, "head", self.index)

    def __str__(self):
        return str(self.addr)

    __repr__ = __str__

    def describe(self) -> R.HeadReport:
        b = self.brain
        O = b.weights(b.head_param(self.layer))[:, b.head_slice(self.index)].float().cpu()
        U, S, _ = torch.linalg.svd(O, full_matrices=False)
        up, down = b.lens(U[:, 0] * S[0])
        shares = {n: ((c.dirs[self.layer] @ O).norm() / O.norm()).item() for n, c in b.concepts.items()}
        return R.HeadReport(str(self), (S[0] ** 2 / (S ** 2).sum()).item(), up, down, shares)


class _Indexer:
    def __init__(self, make):
        self.make = make

    def __getitem__(self, i):
        return self.make(i)


class LayerRef:
    """brain.L[19].mlp, brain.L[19].n[11], brain.L[15].h[7], brain.L[15, 19].mlp"""

    def __init__(self, brain, layers: tuple[int, ...]):
        self.brain, self.layers = brain, layers

    def _addrs(self, kind):
        out = [Addr(l, kind) for l in self.layers]
        return out[0] if len(out) == 1 else out

    @property
    def mlp(self):
        return self._addrs("mlp")

    @property
    def attn(self):
        return self._addrs("attn")

    @property
    def resid(self):
        return self._addrs("resid")

    def _one(self) -> int:
        if len(self.layers) != 1:
            raise ValueError("a neuron or head belongs to one layer: use brain.L[19].n[11]")
        return self.layers[0]

    @property
    def n(self):
        return _Indexer(lambda i: Neuron(self.brain, self._one(), i))

    @property
    def h(self):
        return _Indexer(lambda i: Head(self.brain, self._one(), i))


class _Layers:
    def __init__(self, brain):
        self.brain = brain

    def __getitem__(self, key):
        return LayerRef(self.brain, key if isinstance(key, tuple) else (key,))


# ───────────────────────────── concepts ─────────────────────────────

@dataclass
class Pair:
    """One contrast. `ask` chat-wraps a problem with thinking on; `given` is a raw shared prefix."""
    is_: str
    not_: str
    ask: str | None = None
    given: str = ""


def pair(is_: str, not_: str, ask: str | None = None, given: str = "") -> Pair:
    return Pair(is_, not_, ask, given)


@dataclass(frozen=True)
class Branch:
    concept: "Concept"
    key: str     # "is" | "not"


@dataclass
class Concept:
    """A word: a unit direction per layer, defined by what separates `is` texts from `not` texts."""
    name: str
    brain: "Brain"
    pairs: list[Pair]
    layers: list[int] = field(default_factory=list)                  # anchored layers
    dirs: dict[int, torch.Tensor] = field(default_factory=dict)      # layer -> unit vector (fp32, cpu)
    act_diff: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> mean MLP act(is) − act(not)
    tokens: dict | None = None                                       # per-token residuals, None if derived

    @property
    def is_(self) -> Branch:
        return Branch(self, "is")

    @property
    def not_(self) -> Branch:
        return Branch(self, "not")

    def __repr__(self):
        return f"Concept({self.name!r}, {len(self.pairs)} pairs, anchored {self.layers})"

    def _examples(self):
        if self.tokens is None:
            raise ValueError(f"{self.name} was built by algebra and has no examples of its own")
        return self.tokens

    def where(self) -> R.Separability:
        """Leave-one-pair-out d′ per layer: how cleanly a direction learned from the other pairs splits is/not tokens."""
        toks, n = self._examples(), len(self.pairs)
        if n < 2:
            raise ValueError("need ≥2 pairs for held-out separability")
        res = []
        for l in range(self.brain.n_layers):
            scores = []
            for hold in range(n):
                mi = torch.stack([toks["is"][l][j].mean(0) for j in range(n) if j != hold]).mean(0)
                mn = torch.stack([toks["not"][l][j].mean(0) for j in range(n) if j != hold]).mean(0)
                u = (mi - mn) / (mi - mn).norm()
                a, b = toks["is"][l][hold] @ u, toks["not"][l][hold] @ u
                pooled = torch.cat([a - a.mean(), b - b.mean()]).std() + 1e-6
                scores.append(((a.mean() - b.mean()) / pooled).item())
            res.append(float(np.mean(scores)))
        return R.Separability(self.name, res, self.layers)

    def _writer_scores(self, layer: int):
        """score = Δact × (out-vector · u): how much each MLP neuron writes this concept."""
        b = self.brain
        W = b.weights(b.mlp_param(layer), original=True)
        return self.act_diff[layer] * (self.dirs[layer].to(DEV) @ W.float()).cpu()

    def writers(self, where, top: int = 10) -> R.Writers:
        a = as_addr(where)
        if a.kind != "mlp":
            raise SyntaxError("writers … in Lx.mlp")
        score = self._writer_scores(a.layer)
        pos, idx = score.clamp(min=0), score.topk(top).indices
        diff = self.act_diff[a.layer]
        items = [R.Writer(Neuron(self.brain, a.layer, i), score[i].item(), (pos[i] / pos.sum()).item(), diff[i].item())
                 for i in idx.tolist()]
        return R.Writers(self.name, a.layer, items, (pos[idx].sum() / pos.sum()).item(), len(score))

    def _default_layers(self) -> list[int]:
        return self.layers or [self.where().best]

    def describe(self) -> R.ConceptReport:
        b, out = self.brain, []
        for l in self._default_layers():
            up, down = b.lens(self.dirs[l])
            out.append(R.ConceptAtLayer(l, up, down, self._writer_scores(l).topk(5).indices.tolist(),
                                        l < b.n_layers - 4))
        return R.ConceptReport(self.name, out)

    def read(self, text: str, at: int | None = None) -> R.Heatmap:
        """Per-token projection onto the concept, scaled against the concept's own examples."""
        b, toks = self.brain, self._examples()
        layer = at if at is not None else self._default_layers()[0]
        ids, hs, _, _ = b.run(text)
        u = self.dirs[layer]
        proj = hs[layer + 1] @ u
        ref = torch.cat([t @ u for k in ("is", "not") for t in toks[k][layer]])
        mid, scale = ref.median(), (ref - ref.median()).abs().max()
        return R.Heatmap(self.name, layer, [b.tok.decode([t]).replace("\n", "⏎") for t in ids.tolist()],
                         ((proj - mid) / scale).tolist())

    def _branch_logps(self) -> dict[str, float]:
        b, out = self.brain, {"is": [], "not": []}
        for p in self.pairs:
            for k, br in (("is", p.is_), ("not", p.not_)):
                s, _, _, _, logp = b._continuation(p.given, br)
                out[k].append(logp[max(s - 1, 0):].mean().item())
        return {k: float(np.mean(v)) for k, v in out.items()}

    def measure(self) -> R.Measurement:
        self._examples()
        with self.brain.original():
            base = self._branch_logps()
        return R.Measurement(self.name, base, self._branch_logps())

    def inject(self, at: int, strength: float, into: str, max_new: int = 400) -> R.Generation:
        """Causal readout: add strength × 5% of the residual norm × u after a layer at every position, then generate."""
        b, toks = self.brain, self._examples()
        u = self.dirs[at].to(DEV, b.dtype)
        norm = torch.cat([t for k in ("is", "not") for t in toks[k][at]]).norm(dim=-1).mean().item()

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + strength * norm * 0.05 * u
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        sign = "-" if strength < 0 else "+"
        hd = b.layers[at].register_forward_hook(hook)
        try:
            gen = b.ask(into, max_new, label=f"{'−' if strength < 0 else '+'}{self.name}")
        finally:
            hd.remove()
        gen.note = f"  injecting {sign}{self.name} at L{at} (strength {abs(strength):g} × 5% of residual norm)"
        return gen

    def cos(self, other: "Concept", at: int) -> float:
        return torch.dot(self.dirs[at], other.dirs[at]).item()

    def _combine(self, other: "Concept", sign: int, symbol: str) -> "Concept":
        dirs = {l: self.dirs[l] + sign * other.dirs[l] for l in self.dirs}
        return Concept(f"({self.name}{symbol}{other.name})", self.brain, [], sorted({*self.layers, *other.layers}),
                       {l: v / v.norm() for l, v in dirs.items()},
                       {l: self.act_diff[l] + sign * other.act_diff[l] for l in self.act_diff})

    def __sub__(self, other: "Concept") -> "Concept":
        return self._combine(other, -1, "-")

    def __add__(self, other: "Concept") -> "Concept":
        return self._combine(other, 1, "+")


# ───────────────────────────── promises and edit blocks ─────────────────────────────

class ExpectationFailed(AssertionError):
    def __init__(self, broken: list[R.Promise]):
        self.broken = broken
        super().__init__("; ".join(p.summary for p in broken))


class Expectation:
    def __init__(self, branch: Branch, record=None):
        self.branch, self.record = branch, record

    def _check(self, verb: str, tolerance: float) -> R.Promise:
        c = self.branch.concept
        p = R.Promise(c.name, self.branch.key, verb, tolerance, c.measure())
        if self.record is not None:
            self.record.append(p)
        return p

    def falls(self, by: float = 0.01) -> R.Promise:
        return self._check("falls", by)

    def rises(self, by: float = 0.01) -> R.Promise:
        return self._check("rises", by)

    def steady(self, within: float = 0.02) -> R.Promise:
        return self._check("steady", within)


class Edit:
    """`with brain.edit() as e:` Changes are undone when the block ends unless keep=True.
    With strict=True (the default), any broken promise raises ExpectationFailed at the end of the block."""

    def __init__(self, brain: "Brain", keep: bool, strict: bool):
        self.brain, self.keep, self.strict = brain, keep, strict
        self.promises: list[R.Promise] = []

    def __enter__(self):
        self._mark = len(self.brain.history)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.keep:
            self.brain._rewind(self._mark)
        broken = [p for p in self.promises if not p]
        if exc_type is None and self.strict and broken:
            raise ExpectationFailed(broken)

    def erase(self, concept, from_, writers: int | None = None, strength: float = 1) -> R.EraseResult:
        return self.brain.erase(concept, from_, writers, strength)

    def scale(self, target, by: float) -> str:
        return self.brain.scale(target, by)

    def silence(self, target) -> str:
        return self.brain.scale(target, 0.0)

    def load(self, path: str):
        self.brain.load_edit(path)

    def save(self, path: str) -> int:
        return self.brain.save(path)

    def expect(self, branch: Branch) -> Expectation:
        return Expectation(branch, self.promises)


# ───────────────────────────── the brain ─────────────────────────────

def word_records(brain, layer: int) -> dict[int, R.WordRecord]:
    """What a labeling run and the word test have saved about a layer's dictionary words."""
    import json, os
    path = f"runs/words/{brain.model_id.replace('/', '__')}/L{layer}.json"
    if not os.path.exists(path):
        return {}
    return {int(k): R.WordRecord(f"L{layer}.f{k}", v["fires_on"], v["effect"], v["confidence"], v.get("tests")) for k, v in json.load(open(path)).items()}


class Brain:
    def __init__(self, model=DEFAULT_MODEL, tokenizer=None, anatomy: dict | None = None, ask_suffix: str = "",
                 corpus: list[str] | None = None, dtype="auto", **from_pretrained_args):
        """`model` is a Hugging Face id or an already-built model (pass `tokenizer` with it)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if isinstance(model, str):
            self.model_id = model
            self.tok = tokenizer or AutoTokenizer.from_pretrained(model)
            model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **from_pretrained_args)
        else:
            self.model_id = model.config.name_or_path or type(model).__name__
            self.tok = tokenizer
        self.m = model.to(DEV).eval()
        self.cfg = self.m.config.get_text_config()
        self.dtype = next(self.m.parameters()).dtype
        self.anatomy = discover(self.m, anatomy)
        self.layers = self.anatomy.blocks
        self.n_layers = len(self.layers)
        self.ask_suffix = ask_suffix
        self.corpus_texts = corpus
        self.params = dict(self.m.named_parameters())
        self.originals: dict[str, torch.Tensor] = {}     # param name -> untouched copy
        self.history: list[tuple[str, dict[str, torch.Tensor]]] = []
        self.concepts: dict[str, Concept] = {}
        self.L = _Layers(self)
        self._features: dict = {}
        self._corpus = None
        self._scan: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._mlp_acts: dict[int, torch.Tensor] = {}
        self._capture = False
        for i, layer in enumerate(self.layers):
            self.anatomy.mlp_module(i).register_forward_pre_hook(self._make_hook(i))

    def __repr__(self):
        return f"Brain({self.model_id!r}, {self.n_layers} layers, {len(self.history)} edits applied)"

    @property
    def features(self):
        """brain.features[19] is that layer's SAE dictionary; brain.features[19][4021] is one word in it."""
        from .sae import Features
        cache = self._features
        return _Indexer(lambda L: cache.setdefault(L, Features(self, L)) if L not in cache else cache[L])

    def why(self, text: str, at: int = 19, **options):
        """Which dictionary words pushed the next token after `text`, each checked by removing it. See neurospeak/why.py."""
        from .why import why
        return why(self, text, at, **options)

    def trace(self, text: str, at: int = 19, down_to: int = 15, **options):
        """why(), then what fed each word: lower-layer words and tokens of text, each checked by removing it."""
        from .trace import trace
        return trace(self, text, at, down_to, **options)

    def diff(self, edit_path: str, texts: list[str], at: int = 19, **options):
        """Which dictionary words fire more or less under a weight edit, on these texts."""
        from .diff import diff
        return diff(self, edit_path, texts, at, **options)

    def contrast(self, a_texts: list[str], b_texts: list[str], at: int = 19, **options):
        """Which dictionary words fire more on one kind of text than another."""
        from .diff import contrast
        return contrast(self, a_texts, b_texts, at, **options)

    def _make_hook(self, i):
        def hook(_mod, inp):
            if self._capture:
                self._mlp_acts[i] = inp[0][0].float().cpu()
        return hook

    @torch.no_grad()
    def run(self, text: str, capture: bool = True):
        """One forward pass. Returns (ids, hidden[L+1] per layer, mlp acts per layer, token logprobs)."""
        ids = self.tok(text, return_tensors="pt").input_ids.to(DEV)
        self._capture, self._mlp_acts = capture, {}
        out = self.m(ids, output_hidden_states=capture)
        self._capture = False
        logp = torch.log_softmax(out.logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None])[:, 0].cpu()
        hs = [h[0].float().cpu() for h in out.hidden_states] if capture else None
        return ids[0].cpu(), hs, dict(self._mlp_acts), logp

    def lens(self, vec: torch.Tensor, k: int = 8) -> tuple[list[str], list[str]]:
        """Logit lens: which next-tokens does writing `vec` into the residual stream push up / down?"""
        with torch.no_grad():
            v = vec.to(DEV, self.dtype)[None]
            logits = self.anatomy.unembed(self.anatomy.to_residual(v))[0].float()
            logits = logits - logits.mean()
        show = lambda i: repr(self.tok.decode([i]))[1:-1].replace(" ", "·") or "∅"
        return ([show(i) for i in logits.topk(k).indices.tolist()],
                [show(i) for i in (-logits).topk(k).indices.tolist()])

    def mlp_param(self, layer: int) -> str:
        return self.anatomy.mlp_out[layer] + ".weight"

    def attn_param(self, layer: int) -> str:
        path = self.anatomy.attn_out[layer]
        if path is None:
            raise KeyError(f"layer {layer} has no attention output projection I can find")
        return path + ".weight"

    def head_param(self, layer: int) -> str:
        """Heads exist only where attention output splits evenly by head, and not on linear-attention layers."""
        types = getattr(self.cfg, "layer_types", None)
        if types and types[layer] != "full_attention":
            raise KeyError(f"L{layer} is a linear-attention layer; heads are addressable only on full-attention layers "
                           f"({[i for i, t in enumerate(types) if t == 'full_attention']})")
        return self.attn_param(layer)

    def head_slice(self, head: int) -> slice:
        d = getattr(self.cfg, "head_dim", None) or self.cfg.hidden_size // self.cfg.num_attention_heads
        return slice(head * d, (head + 1) * d)

    def weights(self, param_name: str, original: bool = False) -> torch.Tensor:
        """A weight in (residual, neuron) layout, whatever layout the model stores it in."""
        w = self.originals.get(param_name, self.params[param_name]) if original else self.params[param_name]
        return self.anatomy.columns(param_name, w.detach())

    # -- words --
    def chat_prompt(self, problem: str, thinking: bool = True) -> str:
        """Wrap a problem in the model's chat format. A model with no chat template gets the plain text."""
        text = problem + self.ask_suffix
        if not getattr(self.tok, "chat_template", None):
            return text
        return self.tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                            tokenize=False, enable_thinking=thinking)

    def _continuation(self, given: str, branch: str):
        """Run given+branch; return (span start, outputs). Span = tokens after the shared prefix."""
        ids, hs, acts, logp = self.run(given + branch)
        start = 0
        if given:
            gids = self.tok(given).input_ids
            start = next((i for i, (a, b) in enumerate(zip(gids, ids.tolist())) if a != b), len(gids))
        return start, ids, hs, acts, logp

    def concept(self, name: str, pairs: list[Pair], at: list[int] | int | None = None) -> Concept:
        """Learn a word. Direction = mean residual(is) − mean residual(not) over continuation tokens, per layer."""
        if not pairs:
            raise SyntaxError("a concept needs at least one is/not pair")
        layers = [at] if isinstance(at, int) else list(at or [])
        resolved = [Pair(p.is_, p.not_, p.ask, self.chat_prompt(p.ask) if p.ask else p.given) for p in pairs]
        L = self.n_layers
        diff_h = [torch.zeros(self.anatomy.hidden_size) for _ in range(L)]
        diff_a = [torch.zeros(self.anatomy.n_neurons(l)) for l in range(L)]
        tokens = {"is": [[] for _ in range(L)], "not": [[] for _ in range(L)]}
        for p in resolved:
            for sign, br, key in ((1, p.is_, "is"), (-1, p.not_, "not")):
                s, _, hs, acts, _ = self._continuation(p.given, br)
                for l in range(L):
                    diff_h[l] += sign * hs[l + 1][s:].mean(0) / len(resolved)
                    diff_a[l] += sign * acts[l][s:].mean(0) / len(resolved)
                    tokens[key][l].append(hs[l + 1][s:])
        c = Concept(name, self, resolved, layers, {l: diff_h[l] / diff_h[l].norm() for l in range(L)},
                    dict(enumerate(diff_a)), tokens)
        self.concepts[name] = c
        return c

    def get(self, name: str) -> Concept:
        if name not in self.concepts:
            raise KeyError(f"unknown concept {name!r}")
        return self.concepts[name]

    # -- changing weights --
    def edit(self, keep: bool = False, strict: bool = True) -> Edit:
        return Edit(self, keep, strict)

    def expect(self, branch: Branch) -> Expectation:
        return Expectation(branch)

    def apply(self, label: str, changes: dict[str, torch.Tensor]):
        with torch.no_grad():
            for name, new in changes.items():
                if name not in self.originals:
                    self.originals[name] = self.params[name].detach().clone()
                self.params[name].copy_(new.to(self.params[name].device, self.params[name].dtype))
        self.history.append((label, changes))

    def reset(self):
        with torch.no_grad():
            for name, w in self.originals.items():
                self.params[name].copy_(w)
        self.history.clear()

    def _rewind(self, mark: int):
        kept = self.history[:mark]
        self.reset()
        for label, changes in kept:
            self.apply(label, changes)

    def undo(self) -> str | None:
        """Drop the latest edit. Returns its label, or None if there was nothing to undo."""
        if not self.history:
            return None
        label = self.history[-1][0]
        self._rewind(len(self.history) - 1)
        return label

    class _Original:
        def __init__(self, brain):
            self.b = brain

        def __enter__(self):
            self.saved = {n: self.b.params[n].detach().clone() for n in self.b.originals}
            with torch.no_grad():
                for n, w in self.b.originals.items():
                    self.b.params[n].copy_(w)

        def __exit__(self, *a):
            with torch.no_grad():
                for n, w in self.saved.items():
                    self.b.params[n].copy_(w)

    def original(self):
        """Context manager: temporarily run the untouched model."""
        return Brain._Original(self)

    def edit_directions(self, path: str) -> dict[int, torch.Tensor]:
        """The main direction a weight edit changes at each layer, as a unit vector in residual space."""
        found = {}
        for name, Wn in load_edit_file(self.m, path).items():
            layer, _ = self.anatomy.locate(name)
            W0 = self.weights(name, original=True).float().cpu()
            D = self.anatomy.columns(name, Wn).float().cpu() - W0
            cols = (D.abs().sum(0) > 0).nonzero()[:, 0]
            found[layer] = F.normalize(torch.linalg.svd(D[:, cols], full_matrices=False)[0][:, 0], dim=0)
        return found

    def investigate(self, name: str, pairs: list[Pair] | None, prompts: list[str], measure, **options) -> R.Investigation:
        """Debug a behavior: build its direction from examples, remove it, and measure the change against controls."""
        from .investigate import investigate
        return investigate(self, name, pairs, prompts, measure, **options)

    @contextmanager
    def without(self, direction, at: int):
        """While the block runs, remove one direction from a layer's output at every token. Weights don't change.
        Takes a concept, a dictionary word, or a raw vector."""
        if hasattr(direction, "as_concept"):
            direction = direction.as_concept()
        v = direction.dirs[at] if isinstance(direction, Concept) else direction
        v = F.normalize(v.to(DEV).float(), dim=0)

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            d = v.to(h.dtype)
            h = h - (h @ d).unsqueeze(-1) * d
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        handle = self.layers[at].register_forward_hook(hook)
        try:
            yield self
        finally:
            handle.remove()

    def erase(self, concept: Concept | str, from_, writers: int | None = None, strength: float = 1) -> R.EraseResult:
        """W' = W − strength · u (mask ⊙ uᵀW). strength 1 removes the direction; above 1 also writes its opposite."""
        c = self.get(concept) if isinstance(concept, str) else concept
        addrs, changes, notes = as_addrs(from_), {}, []
        for a in addrs:
            u = c.dirs[a.layer].to(DEV)
            if a.kind == "mlp":
                pname = self.mlp_param(a.layer)
            elif a.kind == "attn":
                pname = self.attn_param(a.layer)
            else:
                raise SyntaxError("erase … from Lx.mlp or Lx.attn")
            W0 = self.weights(pname, original=True).float()
            Wcur = self.weights(pname).float()
            coef = u @ Wcur
            if writers:
                if a.kind != "mlp":
                    raise SyntaxError("'writers top K' only applies to MLPs")
                mask = torch.zeros_like(coef)
                mask[c._writer_scores(a.layer).topk(writers).indices.to(DEV)] = 1
                coef = coef * mask
            Wn = Wcur - strength * torch.outer(u, coef)
            changes[pname] = self.anatomy.columns(pname, Wn)
            notes.append(f"{a}: {int((coef != 0).sum())} columns, relative change "
                         f"{((Wn - W0).norm() / W0.norm() * 100):.2f}%")
        self.apply(f"erase {c.name} from {', '.join(map(str, addrs))}" + (f" top {writers}" if writers else "")
                   + f" strength {strength:g}", changes)
        verb = ("removed" if strength == 1 else
                "reversed and amplified ×%g" % (strength - 1) if strength > 1 else "weakened")
        return R.EraseResult(c.name, verb, notes)

    def scale(self, target, by: float) -> str:
        a = as_addr(target)
        if a.kind == "neuron":
            pname, sl = self.mlp_param(a.layer), (slice(None), slice(a.index, a.index + 1))
        elif a.kind == "head":
            pname, sl = self.head_param(a.layer), (slice(None), self.head_slice(a.index))
        else:
            raise SyntaxError("silence/scale takes a neuron (L19.n11) or head (L15.h7)")
        W = self.weights(pname).clone()
        W[sl] *= by
        self.apply(f"scale {a} by {by:g}", {pname: self.anatomy.columns(pname, W)})
        return f"  {a} output scaled by {by:g}"

    def silence(self, target) -> str:
        return self.scale(target, 0.0)

    def load_edit(self, path: str):
        self.apply(f"load {path}", load_edit_file(self.m, path))

    def save(self, path: str) -> int:
        torch.save({n: self.params[n].detach().cpu().clone() for n in self.originals}, path)
        return len(self.originals)

    def explain_edit(self, path: str) -> R.EditExplanation:
        """Translate a raw weight edit back into this language."""
        changes = []
        for name, Wn in load_edit_file(self.m, path).items():
            W0 = self.weights(name, original=True).float().cpu()
            D = self.anatomy.columns(name, Wn).float().cpu() - W0
            cols = (D.abs().sum(0) > 0).nonzero()[:, 0]
            U, S, _ = torch.linalg.svd(D[:, cols], full_matrices=False)
            layer, kind = self.anatomy.locate(name)
            u = U[:, 0]
            ref = next((c.dirs[layer] for c in self.concepts.values()), None)
            if ref is not None and torch.dot(u, ref) < 0:
                u = -u      # erasing is sign-free; orient toward a known concept
            removed = -(u @ D[:, cols]) / (u @ W0[:, cols])
            up, down = self.lens(u)
            matches = []
            for cname, c in self.concepts.items():
                top = c._writer_scores(layer).topk(len(cols)).indices.tolist()
                matches.append(R.ConceptMatch(cname, torch.dot(u, c.dirs[layer]).item(),
                                              len(set(top) & set(cols.tolist()))))
            changes.append(R.TensorChange(layer, kind, cols.tolist(),
                                          (S[0] ** 2 / (S ** 2).sum()).item(), removed.median().item(),
                                          up, down, matches))
        return R.EditExplanation(path, changes)

    # -- talking --
    def _generate(self, ids, max_new: int) -> str:
        torch.manual_seed(0)
        with torch.no_grad():
            o = self.m.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        return self.tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)

    def _prompt_ids(self, problem: str):
        return self.tok(self.chat_prompt(problem), return_tensors="pt").input_ids.to(DEV)

    def ask(self, problem: str, max_new: int = 600, label: str | None = None) -> R.Generation:
        text = self._generate(self._prompt_ids(problem), max_new)
        return R.Generation([(label or ("edited" if self.history else "model"), text, loopiness(text))])

    def compare(self, problem: str, max_new: int = 600) -> R.Generation:
        ids = self._prompt_ids(problem)
        with self.original():
            base = self._generate(ids, max_new)
        now = self._generate(ids, max_new)
        return R.Generation([("original", base, loopiness(base)),
                             ("edited" if self.history else "model", now, loopiness(now))])

    # -- corpus scan for neuron descriptions --
    def corpus(self):
        if self._corpus is None:
            texts = self.corpus_texts if self.corpus_texts is not None else _default_corpus()
            self._corpus = []
            for t in texts:
                ids = self.tok(t).input_ids
                for i in range(0, len(ids), 768):
                    self._corpus.append(ids[i:i + 768])
        return self._corpus

    @torch.no_grad()
    def scan(self, layer: int):
        if layer not in self._scan:
            acts, toks = [], []
            hook = self.anatomy.mlp_module(layer).register_forward_pre_hook(
                lambda _m, inp: acts.append(inp[0][0].to(torch.float16).cpu()))
            try:
                for ids in self.corpus():
                    self.m(torch.tensor([ids], device=DEV))
                    toks.append(torch.tensor(ids))
            finally:
                hook.remove()
            self._scan[layer] = (torch.cat(acts), torch.cat(toks))
        return self._scan[layer]


def _default_corpus() -> list[str]:
    """This project's traces if present, otherwise Wikipedia, so `describe` works for any model anywhere."""
    texts = []
    for p in Path("runs").glob("*traces*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            texts.append(r["prompt"] + r["completion"][:6000])
    practice = Path("data/practice_sample.jsonl")
    if practice.exists():
        texts += [json.loads(l)["problem"] for l in open(practice)][:150]
    if texts:
        return texts
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/validation-00000-of-00001.parquet", repo_type="dataset")
    return ["".join(pq.read_table(path).column("text").to_pylist())[:200_000]]


def load(model=DEFAULT_MODEL, **kwargs) -> Brain:
    return Brain(model, **kwargs)
