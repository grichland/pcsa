"""The decision notes of the former player (pcsa.rl_training.former_player): the pair matrix as plain lists, the
belief's summary and the matrix blend's values. Split from former_player.py for pcsa's file limit."""

from __future__ import annotations

from functools import lru_cache

import torch

from pcsa.rl_training import former_features as fx
from pcsa.rl_training.features_vocab import tables
from pcsa.rl_training.spec import VOCAB

MOVES, SWITCH_AT = 4, 8  # action rows: 0-3 move, 4-7 move with Mega, 8-10 switch to bench
THEIR_ACTIVE, THEIR_BENCH = 6, 7  # observation rows of their active and of their first bench member
MASKED = -1e8  # a logit below this is a masked one


@lru_cache(maxsize=4)
def _names(kind: str) -> dict[int, str]:
    return {i: name for name, i in tables()[kind].items()}


def matrix_notes(obs: fx.Obs, out: dict, probs: torch.Tensor, index: int) -> dict:
    """The pair matrix of one decision as plain lists: our 11 actions (rows) against their kept actions (columns),
    with each column's probability and our final distribution. Names are vocabulary ids ("" for an empty slot)."""
    move, species = _names("move"), _names("species")
    ours = [move.get(int(i), "") for i in obs.ids[0, 3 : 3 + MOVES]]
    bench = [species.get(int(obs.ids[1 + k, 0]), "") for k in range(len(obs.mask) - SWITCH_AT)]
    shown = {int(i) for i in obs.ids[THEIR_ACTIVE, 3 : 3 + MOVES] if int(i)}
    kept = [int(i) for i in out["kept_idx"].tolist()]
    v = VOCAB["move"]
    cols = [move.get(i, "") if i < v else species.get(int(obs.ids[THEIR_BENCH + i - v, 0]), "") for i in kept]
    mega = float(out["opp_mega"])
    return {
        "rows": [*ours, *(f"{m} + mega" if m else "" for m in ours), *bench],
        "legal": [bool(x) for x in obs.mask],
        "cols": cols,
        "col_kind": ["move" if i < v else "switch" for i in kept],
        "shown": [i in shown for i in kept],
        "col_legal": [bool(out["opp_logits"][i] > MASKED) for i in kept],  # False: fainted or absent
        "p_cols": [round(float(x), 3) for x in out["kept_p"].tolist()],
        "cells": [[round(float(x), 3) for x in row] for row in out["matrix"].tolist()],
        "probs": [round(float(x), 3) for x in probs[0].tolist()],
        "chosen": index,
        "mega_p": round(float(torch.sigmoid(out["opp_mega"])), 3) if mega > MASKED else None,
    }


def belief_notes(belief) -> dict:
    """A short summary of the battle's belief for the decision record (feature modes other than "base")."""
    t = belief.tracker
    return {
        "members": len(t.members),
        "updates": int(belief.updates),
        "rebuilds": int(belief.rebuilds),
        "evidence": {k: int(v) for k, v in sorted(t.stats.items())},
    }


def blend_notes(out: dict, blend: float) -> dict:
    """The matrix blend of one decision (stage 3): its weight, Q per action row, and the action the network alone
    would have taken (greedy over its own logits; `chosen` in the matrix notes is the blended one)."""
    if "matrix_error" in out:
        return {"blend": blend, "error": out["matrix_error"]}
    if "matrix_q" not in out:
        return {}
    info = out.get("matrix_info") or {}
    return {
        "blend": blend,
        "q": [round(float(x), 4) for x in out["matrix_q"].tolist()],
        "network_index": int(torch.argmax(out["network_logits"])),
        "computed": {k: int(v) for k, v in (info.get("counts") or {}).items()},
    }


def their_actions(out: dict) -> tuple[tuple[str, str, float], ...]:
    """Their kept actions as the matrix weighs them (the notes' cols / col_kind / p_cols): (kind "move" or "switch",
    move id or "", probability), for the neural player's sucker_punch guard rule."""
    move, v = _names("move"), VOCAB["move"]
    kept = [int(i) for i in out["kept_idx"].reshape(-1).tolist()]
    probs = [float(p) for p in out["kept_p"].reshape(-1).tolist()]
    return tuple(
        ("move", move.get(i, ""), p) if i < v else ("switch", "", p) for i, p in zip(kept, probs, strict=True)
    )
