"""The only module of the network that touches champions_gym (CG), contract 0.6.

CG is imported from the installed wheel.

Source: pcsa-former/pcsa_former/cg_adapter.py, copied verbatim without the training parts (the source-tree
fallback import, write_teams and CGEnv).
"""

from __future__ import annotations

import numpy as np
import torch

from . import layout as L

VOCAB_KEYS = {"species": "species", "moves": "move", "items": "item", "abilities": "ability"}  # CG's name -> ours
BLOCKS = ("rows", "row_ids", "field", "last", "last_ids")
OBS = ("rows", "row_ids", "field", "last", "last_ids", "mask", "opp_prior", "kind", "seat", "decision")


def _cg():
    import champions_gym as cg

    return cg


def layout() -> L.Layout:
    """The layout from champions_gym.FEATURES, with the contract's sizes and action numbering checked."""
    cg = _cg()
    feats = {}
    for fid, f in cg.FEATURES.items():
        if f.block not in BLOCKS:  # the action mask and the learnset table are not input columns
            continue
        cols = [int(c) for c in f.columns()]
        if f.slot:
            mine = L.Feature(fid, f.name, f.block, cols[0], f.size, L.MOVE_SLOTS, cg.obs.MOVE_STRIDE, f.encoding, f.lo, f.hi)
        else:
            mine = L.Feature(fid, f.name, f.block, cols[0], f.size, f.repeats, 0, f.encoding, f.lo, f.hi)
        if sorted(mine.columns()) != sorted(cols):
            raise RuntimeError(f"{fid}: columns read from FEATURES do not match its own columns()")
        feats[fid] = mine
    out = L.Layout(feats)
    vocab = {VOCAB_KEYS[k]: int(v) for k, v in cg.VOCAB.items()}
    if vocab != L.VOCAB:
        raise RuntimeError(f"vocabulary sizes: champions_gym has {vocab}, the contract said {L.VOCAB}")
    shapes = {k: int(cg.SHAPES[k][-1]) for k in BLOCKS}
    if shapes != out.widths:
        raise RuntimeError(f"array widths: champions_gym has {shapes}, its feature list gives {out.widths}")
    if (cg.KIND_PREVIEW, cg.KIND_TURN, cg.KIND_REPLACE, cg.PREVIEW0, cg.N_ACTIONS) != (
        L.KIND_PREVIEW, L.KIND_TURN, L.KIND_REPLACE, L.PREVIEW_OFFSET, L.PREVIEW_OFFSET + L.PREVIEW_ACTIONS):  # fmt: skip
        raise RuntimeError("request kinds or action numbering differ from the contract")
    combos = [tuple(cg.preview_picks(L.PREVIEW_OFFSET + a)) for a in range(L.PREVIEW_ACTIONS)]
    if combos != L.preview_combos():
        raise RuntimeError("preview action numbering differs from layout.preview_combos")
    out.cg = contract()
    if out.cg["layout_hash"] != out.hash:
        raise RuntimeError(f"layout hash: champions_gym has {out.cg['layout_hash']}, ours is {out.hash}")
    return out


def contract() -> dict:
    """What this CG build gives a model (contract version, layout hash, shapes, vocabulary sizes), plus the build."""
    cg = _cg()
    import champions_gym.neural as neural

    return {**neural.contract(), "cg_version": cg.__version__}


def learnset() -> torch.Tensor:
    return torch.from_numpy(np.asarray(_cg().learnset_table())).bool()
