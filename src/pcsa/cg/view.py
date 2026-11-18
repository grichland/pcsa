"""A snapshot of CG's Battle as plain JSON: the record contract's `notes.cg.view` (CG string ids throughout).

A pure function of a `champions_gym.Battle` (any object with its attributes); nothing here imports CG.
"""

from __future__ import annotations

from typing import Any

KINDS = {0: "preview", 1: "turn", 2: "replace"}
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
PICKS = 3
TEAM = 6


def _padded(values: Any, size: int) -> list:
    out = list(values or [])[:size]
    return out + [None] * (size - len(out))


def _counts(values: Any) -> dict[str, int]:
    return {str(k): int(v) for k, v in dict(values or {}).items()}


def mon(p: Any) -> dict:
    """One Pokémon: `species` (theirs unseen: the preview species), hp 0..1, item None unknown and "" gone."""
    boosts = dict(p.boosts or {})
    return {
        "species": p.species,
        "form": p.form,
        "hp": float(p.current_hp_fraction),
        "fainted": bool(p.fainted),
        "status": p.status or "",
        "boosts": {k: int(boosts.get(k, 0)) for k in BOOSTS},
        "effects": [str(e) for e in p.effects or ()],
        "moves": [[str(m), None if pp is None else int(pp)] for m, pp in p.moves or ()],
        "item": p.item,
        "ability": p.ability,
        "types": [str(t) for t in p.types or ()],
        "active": bool(p.active),
        "revealed": bool(p.revealed),
    }


def view(battle: Any) -> dict:
    """The Battle at the request the player decided on."""
    return {
        "turn": int(battle.turn),
        "kind": KINDS.get(int(battle.kind), str(battle.kind)),
        "picks": [None if p is None else int(p) for p in _padded(battle.picks, PICKS)],
        "opp_preview": _padded(battle.opponent_preview, TEAM),
        "anomalies": int(battle.anomalies),
        "own": [mon(p) for p in battle.team],
        "opp": [mon(p) for p in battle.opponent_team],
        "weather": _counts(battle.weather),
        "fields": _counts(battle.fields),
        "side": _counts(battle.side_conditions),
        "opp_side": _counts(battle.opponent_side_conditions),
    }
