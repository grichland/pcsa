"""Their active's move options narrowed by the builds it could be (inference only, not trained).
With `opponent_moves_from_builds` on, the opponent head is masked to the moves of the species' seen-ladder builds that
agree
with every revealed move (else its listed builds), plus the revealed moves, and renormalised (switches untouched).
No agreeing build, or none listed: no mask (the full learnset, as trained).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.rl_training.features_vocab import index
from pcsa.rl_training.spec import VOCAB

NOT_MOVES = frozenset({"struggle", "recharge"})


@lru_cache(maxsize=1)
def _catalog():
    from pcsa.battle.builds import Catalog

    return Catalog.load(seen=True)


def revealed(mon) -> frozenset[str]:
    return frozenset(to_id_str(m) for m in (mon.moves or {})) - NOT_MOVES


def _fitting(builds, shown: frozenset[str]) -> list:
    return [b for b in builds if shown <= {to_id_str(m) for m in b.moves}]


def build_moves(species: str, shown: frozenset[str], catalog=None) -> frozenset[str] | None:
    """The moves of `species`' builds holding every move in `shown`, plus `shown`: its builds seen on the live
    ladder when one of them fits, else its listed builds (seen and usage catalog); None: no build fits."""
    catalog = catalog or _catalog()
    seen = [s.build for s in catalog.seen.ranked(species)] if catalog.seen is not None else []
    if not seen and catalog.seen is not None:
        seen = [s.build for s in catalog.seen.ranked(to_id_str(game.base_name(species)))]
    fitting = _fitting(seen, shown) or _fitting(catalog.builds(species), shown)
    if not fitting:
        return None
    return frozenset(to_id_str(m) for b in fitting for m in b.moves) | shown


def allowed_mask(battle, catalog=None) -> np.ndarray | None:
    """[move vocab] bool over their active's possible moves, or None (no narrowing)."""
    mon = getattr(battle, "opponent_active_pokemon", None)
    if mon is None:
        return None
    shown = revealed(mon)
    moves = build_moves(mon.species, shown, catalog)
    if not moves:
        return None
    mask = np.zeros(VOCAB["move"], bool)
    for m in moves:
        mask[index("move", m)] = True
    mask[0] = False
    return mask if mask.any() else None
