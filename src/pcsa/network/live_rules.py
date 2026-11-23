"""pcsa's own rule masks, applied after training's (rules.py, which stays a verbatim copy).

The mirror rule: when our active and theirs are the same species and that species is listed in
`cg.rules.status_free_mirrors`, our status moves (and their Mega copies) are closed at turn requests, only while
another action stays open.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pcsa.common.config_cg import RuleSettings

from . import layout as L
from . import rules as masks
from . import vocab

LIVE_ONLY = ("status_free_mirrors",)  # RuleSettings fields training's Rules does not take


class MirrorRule:
    """Closes our status moves in a mirror of one of `species` (CG species ids)."""

    def __init__(self, layout: L.Layout, species: Sequence[str]) -> None:
        names = vocab.tables()["species"]
        unknown = [s for s in species if s not in names]
        if unknown:
            raise ValueError(f"cg.rules.status_free_mirrors: unknown species {unknown}")
        self.species = list(species)
        self.ids = np.array([names.index(s) for s in species], np.int64)
        self.active_col, self.species_col = layout.col(L.F_IS_ACTIVE), layout.col(L.F_SPECIES)
        by = {f.name: f for f in layout.features.values()}
        self.c_cat = np.asarray(by["pokemon.move.fact.category"].columns())  # physical, special, status per slot
        self.closed = 0  # decisions the rule closed something for

    def mask(self, obs: dict, m: np.ndarray) -> np.ndarray:
        """m [1, 11] (one game's mask after training's rules) with the mirror rule applied."""
        rows, ids = np.asarray(obs["rows"])[0], np.asarray(obs["row_ids"])[0]
        row = int(rows[: L.OWN_PICKS, self.active_col].argmax())
        ours, theirs = int(ids[row, self.species_col]), int(ids[L.OPP_ACTIVE, self.species_col])
        if ours != theirs or ours not in self.ids:
            return m
        status = rows[row, self.c_cat].reshape(4, 3)[:, 2] > 0
        close = np.concatenate([status, status]) & (m[0, :8] > 0)
        rest = m[0] > 0
        rest[:8] &= ~close
        if not close.any() or not rest.any():
            return m
        out = m.copy()
        out[0, :8][close] = 0
        self.closed += 1
        return out


def build(rules: Mapping[str, Any] | None, layout: L.Layout) -> tuple[masks.RuleConfig | None, MirrorRule | None]:
    """The policy's `rules` argument as (training's RuleConfig, the mirror rule or None); (None, None) when off."""
    if rules is None:
        return None, None
    limits = RuleSettings.model_validate({"enabled": True, **rules})
    cfg = masks.RuleConfig(**limits.model_dump(exclude={"enabled", *LIVE_ONLY}))
    mirrors = limits.status_free_mirrors
    return cfg, (MirrorRule(layout, mirrors) if mirrors else None)


def apply(mirror: MirrorRule | None, obs: dict, m: np.ndarray) -> np.ndarray:
    return m if mirror is None else mirror.mask(obs, m)


def info(cfg: masks.RuleConfig | None, mirror: MirrorRule | None) -> dict | None:
    """The rule settings as the policy reports them."""
    if cfg is None:
        return None
    return {**dataclasses.asdict(cfg), "status_free_mirrors": [] if mirror is None else mirror.species}
