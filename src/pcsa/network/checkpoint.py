"""Checkpoints carry the contract version, the feature-list hash, the widths and the vocabulary sizes they were trained
with (and CG's own contract record when the layout came from CG), and refuse to load against a different one.

Source: pcsa-former/pcsa_former/checkpoint.py, copied verbatim; only the imports are package-relative.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from . import layout as L
from .model import Config, Net


class LayoutMismatch(RuntimeError):
    pass


def save(path: Path | str, net: Net, optimizer: torch.optim.Optimizer | None = None, extra: dict | None = None) -> None:
    torch.save(
        {
            "model": net.state_dict(),
            "config": asdict(net.cfg),
            "layout": net.layout.describe(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "extra": extra or {},
        },
        path,
    )


def load(path: Path | str, layout: L.Layout, learnset: torch.Tensor | None = None, device="cpu",
         allow_contract: bool = False) -> tuple[Net, dict]:  # fmt: skip
    """(net, the checkpoint dict). Raises LayoutMismatch when the feature list, widths, vocabularies or the contract
    version differ. `allow_contract` lets another contract version through when the rest matches (to go on training
    a net made under an earlier version)."""
    ck = torch.load(path, map_location=device, weights_only=True)
    want, have = ck["layout"], layout.describe()
    for key in ("hash", "widths", "vocab"):
        if want[key] != have[key]:
            raise LayoutMismatch(f"checkpoint {key} {want[key]!r} does not match the current layout {have[key]!r}")
    if want.get("contract") != have["contract"] and not allow_contract:
        raise LayoutMismatch(f"checkpoint contract {want.get('contract')!r} is not the current {have['contract']!r} (allow_contract to load anyway)")
    net = Net(layout, Config(**ck["config"]), learnset).to(device)
    net.load_state_dict(ck["model"])
    return net, ck
