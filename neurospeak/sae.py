"""Auto-vocabulary: sparse-autoencoder features as words. brain.features[19][4021] is one feature."""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import core, reports as R


def checkpoint_dir(model_id: str) -> str:
    return "runs/sae/" + model_id.replace("/", "__")


class TopKSAE(torch.nn.Module):
    def __init__(self, d: int, n: int, k: int, k_aux: int = 512):
        super().__init__()
        self.d, self.n, self.k, self.k_aux = d, n, k, k_aux
        dec = F.normalize(torch.randn(d, n), dim=0)
        self.W_dec = torch.nn.Parameter(dec)
        self.W_enc = torch.nn.Parameter(dec.T.clone())
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.b_enc = torch.nn.Parameter(torch.zeros(n))

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc.T + self.b_enc

    def encode(self, x):
        vals, idx = self.pre(x).topk(self.k, dim=-1)
        return vals.relu(), idx

    def forward(self, x, dead=None):
        pre = self.pre(x)
        vals, idx = pre.topk(self.k, dim=-1)
        vals = vals.relu()
        recon = (self.W_dec.T[idx] * vals.unsqueeze(-1)).sum(-2) + self.b_dec
        aux = None
        if dead is not None and dead.any():
            # AuxK: dead latents model the leftover error, which brings them back to life
            v, i = pre.masked_fill(~dead, -1e9).topk(min(self.k_aux, int(dead.sum())), dim=-1)
            aux = (self.W_dec.T[i] * v.relu().unsqueeze(-1)).sum(-2)
        return recon, vals, idx, aux

    @torch.no_grad()
    def norm_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


@dataclass(frozen=True)
class Feature:
    """One dictionary entry: a direction in the residual stream after `layer`."""
    features: "Features"
    index: int

    def __str__(self):
        return f"L{self.features.layer}.f{self.index}"

    __repr__ = __str__

    @property
    def direction(self) -> torch.Tensor:
        return self.features.sae.W_dec[:, self.index].detach().cpu()

    def pushes(self, k: int = 8) -> tuple[list[str], list[str]]:
        return self.features.brain.lens(self.direction, k)

    def record(self) -> R.WordRecord | None:
        """This word's name and test scores, if any run has saved them."""
        return core.word_records(self.features.brain, self.features.layer).get(self.index)

    def as_concept(self) -> core.Concept:
        """Usable anywhere that needs only a direction, such as brain.erase(f.as_concept(), "L19.mlp")."""
        L = self.features.layer
        b = self.features.brain
        return core.Concept(str(self), b, [], [L], {L: self.direction},
                            {L: torch.zeros(b.anatomy.n_neurons(L))})


@dataclass
class FeatureHit:
    feature: Feature
    value: float


class Features:
    """The SAE dictionary for one layer, loaded from a sae_train.py checkpoint."""

    def __init__(self, brain, layer: int, path: str | None = None):
        path = path or f"{checkpoint_dir(brain.model_id)}/L{layer}.pt"
        if not os.path.exists(path):
            raise KeyError(f"no SAE for layer {layer} at {path}; train one with sae_train.py")
        c = torch.load(path, map_location=core.DEV)
        if c["config"]["model"] != brain.model_id:
            raise ValueError(f"{path} was trained on {c['config']['model']}, not {brain.model_id}")
        self.brain, self.layer, self.scale, self.tokens_trained = brain, layer, c["scale"], c["tokens"]
        self.sae = TopKSAE(c["config"]["d"], c["config"]["n"], c["config"]["k"]).to(core.DEV)
        self.sae.load_state_dict(c["state"])
        self.sae.requires_grad_(False)

    def __getitem__(self, i: int) -> Feature:
        return Feature(self, i)

    def __len__(self):
        return self.sae.n

    def encode(self, text: str) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Returns (tokens, values[T, k], indices[T, k]) for the layer's residual on this text."""
        ids, hs, _, _ = self.brain.run(text)
        with torch.no_grad():
            vals, idx = self.sae.encode(hs[self.layer + 1].to(core.DEV) * self.scale)
        return [self.brain.tok.decode([t]) for t in ids.tolist()], vals.cpu(), idx.cpu()

    def active(self, text: str, top: int = 10) -> list[FeatureHit]:
        """The features carrying the most total activation across the text."""
        _, vals, idx = self.encode(text)
        total = torch.zeros(self.sae.n).index_add_(0, idx.view(-1), vals.view(-1))
        best = total.topk(top)
        return [FeatureHit(self[i], v) for v, i in zip(best.values.tolist(), best.indices.tolist())]

    def nearest(self, target, top: int = 10) -> list[FeatureHit]:
        """Dictionary words whose direction best matches a concept or raw vector (by cosine)."""
        v = target.dirs[self.layer] if isinstance(target, core.Concept) else target
        cos = (F.normalize(v.to(core.DEV).float(), dim=0) @ self.sae.W_dec.detach()).cpu()
        best = cos.topk(top)
        return [FeatureHit(self[i], c) for c, i in zip(best.values.tolist(), best.indices.tolist())]
