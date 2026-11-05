"""Find the parts NeuroSpeak needs in any transformer, so nothing else knows a model's layout.

A neuron is one input to a block's MLP output projection. A concept lives in a block's residual output.
Discovery goes by structure and common names. Models it can't read take explicit paths through `overrides`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

MLP_NAME = re.compile(r"(^|\.)(mlp|ffn?|feed_?forward)(\.|$)", re.I)
ATTN_NAME = re.compile(r"(^|\.)\w*(attn|attention)\w*(\.|$)", re.I)
NORM_NAME = re.compile(r"(norm|ln_f|^ln$)", re.I)


def _is_conv1d(mod) -> bool:
    return type(mod).__name__ == "Conv1D"


def _is_projection(mod) -> bool:
    return isinstance(mod, nn.Linear) or _is_conv1d(mod)


def _out_features(mod) -> int:
    return mod.nf if _is_conv1d(mod) else mod.out_features


def _last_output(block: nn.Module, name: re.Pattern, hidden: int) -> str | None:
    """The last matching projection that writes to the residual width. By convention it is the module's output."""
    found = [n for n, mod in block.named_modules() if name.search(n) and _is_projection(mod) and _out_features(mod) == hidden]
    return found[-1] if found else None


@dataclass
class Anatomy:
    model: nn.Module
    blocks_path: str
    blocks: nn.ModuleList
    norm: nn.Module | None
    unembed: nn.Module
    hidden_size: int
    mlp_out: list[str]             # per block, full module path
    attn_out: list[str | None]     # per block, full module path

    def module(self, path: str) -> nn.Module:
        return self.model.get_submodule(path)

    def mlp_module(self, layer: int) -> nn.Module:
        return self.module(self.mlp_out[layer])

    def n_neurons(self, layer: int) -> int:
        return self.mlp_module(layer).weight.shape[0 if _is_conv1d(self.mlp_module(layer)) else 1]

    def columns(self, param_name: str, w: torch.Tensor) -> torch.Tensor:
        """Convert between stored and (residual, neuron) layout. GPT-2's Conv1D stores weights transposed.
        A transpose is its own inverse, so this converts both ways."""
        mod = self.module(param_name.rsplit(".", 1)[0])
        return w.T if _is_conv1d(mod) else w

    def locate(self, param_name: str) -> tuple[int, str]:
        """Which block and part a weight belongs to: (19, "mlp")."""
        path = param_name.rsplit(".", 1)[0]
        for layer, (m, a) in enumerate(zip(self.mlp_out, self.attn_out)):
            if path == m:
                return layer, "mlp"
            if path == a:
                return layer, "attn"
        raise KeyError(f"{param_name} is not an MLP or attention output weight")

    def to_residual(self, vec: torch.Tensor) -> torch.Tensor:
        return self.norm(vec) if self.norm is not None else vec

    def summary(self) -> str:
        return (f"{len(self.blocks)} blocks at {self.blocks_path}, residual width {self.hidden_size}, "
                f"{self.n_neurons(0)} MLP neurons per layer (e.g. {self.mlp_out[0]}), "
                f"attention output {self.attn_out[0] or 'not found'}, final norm {'yes' if self.norm is not None else 'none'}")


def discover(model: nn.Module, overrides: dict | None = None) -> Anatomy:
    """overrides may set blocks ("model.layers"), mlp_out ("mlp.fc2"), attn_out ("attn.proj") and norm ("model.norm")."""
    o = overrides or {}
    unembed = model.get_output_embeddings()
    if unembed is None:
        raise ValueError("this model has no output head; NeuroSpeak needs a model that predicts next tokens")
    hidden = unembed.weight.shape[1]

    def has_mlp_out(block):
        return (_last_output(block, re.compile(re.escape(o["mlp_out"]) + "$"), hidden) if "mlp_out" in o
                else _last_output(block, MLP_NAME, hidden)) is not None

    if "blocks" in o:
        blocks_path = o["blocks"]
    else:
        candidates = [(sum(p.numel() for p in mod.parameters()), n) for n, mod in model.named_modules()
                      if isinstance(mod, nn.ModuleList) and len(mod) > 1 and all(has_mlp_out(b) for b in mod)]
        if not candidates:
            raise ValueError("could not find the transformer blocks. Name them: "
                             'ns.load(id, anatomy={"blocks": "model.layers", "mlp_out": "mlp.fc2"})')
        blocks_path = max(candidates)[1]
    blocks = model.get_submodule(blocks_path)

    if "norm" in o:
        norm = model.get_submodule(o["norm"])
    else:
        parent = model.get_submodule(blocks_path.rsplit(".", 1)[0]) if "." in blocks_path else model
        norm = next((m for n, m in parent.named_children() if NORM_NAME.search(n)), None)

    mlp_name = re.compile(re.escape(o["mlp_out"]) + "$") if "mlp_out" in o else MLP_NAME
    attn_name = re.compile(re.escape(o["attn_out"]) + "$") if "attn_out" in o else ATTN_NAME
    mlp_out, attn_out = [], []
    for i, block in enumerate(blocks):
        m = _last_output(block, mlp_name, hidden)
        if m is None:
            raise ValueError(f"block {i} has no MLP output projection; name it with anatomy={{'mlp_out': ...}}")
        a = _last_output(block, attn_name, hidden)
        mlp_out.append(f"{blocks_path}.{i}.{m}")
        attn_out.append(f"{blocks_path}.{i}.{a}" if a else None)
    return Anatomy(model, blocks_path, blocks, norm, unembed, hidden, mlp_out, attn_out)
