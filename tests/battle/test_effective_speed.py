"""Speed as the engine would use it: Choice Scarf, stages and paralysis change who moves first."""

from __future__ import annotations

import pytest

from pcsa.battle.stats import effective_speed, stage
from tests.battle.conftest import translator_for


def test_scarf_stage_and_paralysis_change_the_speed() -> None:
    t = translator_for(("Garchomp", "Garchomp", 185), [("Snorlax", "Snorlax")])
    mine, theirs = t.battle.active_pokemon, t.battle.opponent_active_pokemon
    assert mine is not None and theirs is not None
    base = effective_speed(theirs, False)
    theirs._item = "choicescarf"
    scarfed = effective_speed(theirs, False)
    assert scarfed == pytest.approx(base * 1.5, abs=1.5)
    t.apply("|-boost|p2a: Snorlax|spe|2")
    assert effective_speed(theirs, False) == pytest.approx(scarfed * 2, abs=2)
    t.apply("|-status|p2a: Snorlax|par")
    assert effective_speed(theirs, False) == pytest.approx(scarfed, abs=2)  # +2 then paralysis halves it
    assert effective_speed(mine) > base  # Garchomp outspeeds an unboosted Snorlax


def test_stages() -> None:
    assert stage(100, 0) == 100
    assert stage(100, 2) == 200
    assert stage(100, -1) == 66
