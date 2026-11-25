"""pcsa's own rule masks, applied after training's (rules.py, which stays a verbatim copy).

The mirror rule: when our active and theirs are the same species and that species is listed in
`cg.rules.status_free_mirrors`, our status moves (and their Mega copies) are closed at turn requests, only while
another action stays open.

The Mega rule (`cg.rules.always_mega`): at turn requests, a plain move is closed while its Mega copy is open, so
we Mega Evolve whenever we can; switches stay open.

The lineup rule (`cg.rules.one_mega_lineup`): at team preview, our picks that bring two or more Mega stone holders
are closed (only one can Mega Evolve), only while another pick stays open.
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

LIVE_ONLY = ("status_free_mirrors", "always_mega", "one_mega_lineup")  # not fields of training's Rules


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
        """m [N, 11] (masks after training's rules) with the mirror rule applied to the turn rows."""
        rows, ids, n = np.asarray(obs["rows"]), np.asarray(obs["row_ids"]), len(m)
        e = np.arange(n)
        row = rows[:, : L.OWN_PICKS, self.active_col].argmax(-1)
        ours, theirs = ids[e, row, self.species_col], ids[:, L.OPP_ACTIVE, self.species_col]
        hit = (ours == theirs) & np.isin(ours, self.ids)
        if "kind" in obs:
            hit &= np.asarray(obs["kind"]).reshape(-1) == L.KIND_TURN
        status = rows[e, row][:, self.c_cat].reshape(n, 4, 3)[..., 2] > 0
        close = np.concatenate([status, status], 1) & (m[:, :8] > 0) & hit[:, None]
        rest = m > 0
        rest[:, :8] &= ~close
        close &= rest.any(1)[:, None]
        if not close.any():
            return m
        out = m.copy()
        out[:, :8][close] = 0
        self.closed += int(close.any(1).sum())
        return out


class MegaRule:
    """Closes our plain moves whose Mega copy is open."""

    def __init__(self) -> None:
        self.closed = 0  # decisions the rule closed something for

    def mask(self, obs: dict, m: np.ndarray) -> np.ndarray:
        """m [N, 11] with the rule applied to the turn rows."""
        close = (m[:, :4] > 0) & (m[:, 4:8] > 0)
        if "kind" in obs:
            close &= (np.asarray(obs["kind"]).reshape(-1) == L.KIND_TURN)[:, None]
        if not close.any():
            return m
        out = m.copy()
        out[:, :4][close] = 0
        self.closed += int(close.any(1).sum())
        return out


class TurnRules:
    """pcsa's turn-request rules in order: the mirror rule, then the Mega rule (either may be None)."""

    def __init__(self, mirror: MirrorRule | None, mega: MegaRule | None) -> None:
        self.mirror, self.mega = mirror, mega

    def mask(self, obs: dict, m: np.ndarray) -> np.ndarray:
        for rule in (self.mirror, self.mega):
            if rule is not None:
                m = rule.mask(obs, m)
        return m


class LineupRule:
    """Closes our team-preview picks that bring two or more Mega stone holders."""

    def __init__(self, layout: L.Layout) -> None:
        items = vocab.tables()["items"]
        self.stones = np.array([items.index(s) for s in vocab.mega_stones()], np.int64)
        self.item_col = layout.col(L.F_ITEM)
        self.combos = np.asarray(L.preview_combos())  # [60, 3] team indices
        self.closed = 0  # preview decisions the rule closed something for

    def mask(self, row_ids: Any, kind: Any, m: np.ndarray) -> np.ndarray:
        """m [N, 60] (the preview picks) with the rule applied to the preview rows."""
        mega = np.isin(np.asarray(row_ids)[:, L.OWN_ROWS, self.item_col], self.stones)  # our six, in team order
        close = (mega[:, self.combos].sum(-1) > 1) & (m > 0)
        close &= (np.asarray(kind).reshape(-1) == L.KIND_PREVIEW)[:, None]
        close &= ((m > 0) & ~close).any(1)[:, None]
        if not close.any():
            return m
        out = m.copy()
        out[close] = 0
        self.closed += int(close.any(1).sum())
        return out


Rules = tuple[masks.RuleConfig | None, TurnRules | None, LineupRule | None]


def build(rules: Mapping[str, Any] | None, layout: L.Layout) -> Rules:
    """The policy's `rules` argument as (training's RuleConfig, pcsa's turn rules, the lineup rule); all None when
    off, and each of pcsa's rules None when not set."""
    if rules is None:
        return None, None, None
    limits = RuleSettings.model_validate({"enabled": True, **rules})
    cfg = masks.RuleConfig(**limits.model_dump(exclude={"enabled", *LIVE_ONLY}))
    mirrors = limits.status_free_mirrors
    mirror = MirrorRule(layout, mirrors) if mirrors else None
    mega = MegaRule() if limits.always_mega else None
    turn = TurnRules(mirror, mega) if mirror or mega else None
    return cfg, turn, (LineupRule(layout) if limits.one_mega_lineup else None)


def apply(turn: TurnRules | None, obs: dict, m: np.ndarray) -> np.ndarray:
    return m if turn is None else turn.mask(obs, m)


def close_preview(lineup: LineupRule | None, arrays: Any, b: dict, closed: list[list[int]]) -> None:
    """The lineup rule on b["preview_mask"] (a tensor, replaced); the picks it closed go into `closed` per row,
    as actions (11 + pick)."""
    if lineup is None:
        return
    before = b["preview_mask"].cpu().numpy()
    new = lineup.mask(arrays.row_ids, arrays.kind, before)
    if new is before:
        return
    for r in range(len(closed)):
        closed[r] += [L.PREVIEW_OFFSET + int(a) for a in np.flatnonzero((before[r] > 0) & (new[r] == 0))]
    b["preview_mask"] = b["preview_mask"].new_tensor(new)


def info(cfg: masks.RuleConfig | None, turn: TurnRules | None, lineup: LineupRule | None) -> dict | None:
    """The rule settings as the policy reports them."""
    if cfg is None:
        return None
    mirror, mega = (None, None) if turn is None else (turn.mirror, turn.mega)
    return {**dataclasses.asdict(cfg), "status_free_mirrors": [] if mirror is None else mirror.species,
            "always_mega": mega is not None, "one_mega_lineup": lineup is not None}  # fmt: skip
