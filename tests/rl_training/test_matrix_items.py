"""The matrix's one-turn hits read the defender's item from the belief: resist berries halve a super-effective hit
of their type, Focus Sash stops a KO from full HP (pcsa.belief grids; 13 Nov 2026 check after a live loss)."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")


def _hit(item: str | None, hp: float = 1.0):
    from pcsa.rl_training.belief_features import new_battle_belief
    from pcsa.rl_training.matrix import Lookahead
    from tests.players.test_disciplined import battle

    b = battle("Charizard", foe="Scizor")
    foe = b.opponent_active_pokemon
    if item is not None:
        foe._item = item
    bb = new_battle_belief(crit_reported=False, empty_unknown=True)
    bb.update_from(b)
    return Lookahead(bb).hit(b, b.active_pokemon, foe, "flamethrower", False, coming=True, defender_hp=hp)


def test_a_resist_berry_halves_the_super_effective_hit() -> None:
    plain, berry = _hit("leftovers", 0.5), _hit("occaberry", 0.5)
    assert berry.mean == pytest.approx(plain.mean / 2, rel=0.05)
    assert plain.ko > 0.9 and berry.ko < plain.ko


def test_focus_sash_survives_from_full_hp() -> None:
    assert _hit("leftovers").ko > 0.9
    assert _hit("focussash").ko == 0.0
