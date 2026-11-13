"""The team-preview rules over the neural player's critic: no Golisopod lead into a Fire six, at most one of
Golisopod and Gholdengo brought into one, every pick kept when no rule applies or none can be met."""

from __future__ import annotations

import pytest

from pcsa.battle.state import replay
from pcsa.common.errors import ConfigError
from pcsa.players.neural import NeuralParams
from pcsa.players.preview_rules import check, keep_mask, keeper, types_of
from pcsa.rl_training.preview import all_picks
from tests.conftest import REPO

GOLD = ["Gholdengo", "Garchomp", "Golisopod", "Primarina", "Mamoswine", "Cinderace"]
FIRE_SIX = ["Charizard", "Excadrill", "Basculegion-F", "Espeon", "Absol", "Araquanid"]
NO_FIRE = ["Tyranitar", "Corviknight", "Gardevoir", "Dragonite", "Rotom-Wash", "Garchomp"]
LEAD = {"golisopod": ["Fire"]}
PAIR = [{"types": ["Fire"], "species": ["golisopod", "gholdengo"]}]
GHOLDENGO, GOLISOPOD = 0, 2


def preview(theirs: list[str], ours: list[str] = GOLD):
    lines = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|clearpoke"]
    lines += [f"|poke|p1|{name}, L50|" for name in ours] + [f"|poke|p2|{name}, L50|" for name in theirs]
    lines.append("|teampreview|3")
    side = [
        {
            "ident": f"p1: {n}",
            "details": f"{n}, L50",
            "condition": "150/150",
            "active": False,
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100},
            "moves": ["tackle"],
            "baseAbility": "",
            "item": "",
        }
        for n in ours
    ]
    return replay(lines, {"teamPreview": True, "maxTeamSize": 3, "side": {"name": "me", "id": "p1", "pokemon": side}})


def kept(theirs: list[str], lead=LEAD, pair=PAIR) -> tuple[list[tuple[int, ...]], dict]:
    picks = all_picks()
    keep, notes = keep_mask(preview(theirs), picks, lead, pair)
    return [p for p, k in zip(picks, keep, strict=True) if k], notes


def test_the_repo_settings_turn_both_rules_on_and_the_rollback_leaves_them_off() -> None:
    params = NeuralParams.load(REPO / "config" / "players" / "neural.yaml")
    assert params.preview_lead_avoid == LEAD and params.preview_bring_max_one == PAIR
    gold940 = NeuralParams.load(REPO / "config" / "players" / "neural-gold940.yaml")
    assert not gold940.preview_lead_avoid and not gold940.preview_bring_max_one
    assert keeper({}, []) is None


def test_fire_types_from_the_pokedex_forms_included() -> None:
    assert "fire" in types_of("Charizard") and "fire" in types_of("Rotom-Heat")
    assert "fire" not in types_of("Rotom-Wash") and "fire" not in types_of("Basculegion-F")
    assert types_of("Missingname") == frozenset()


def test_no_golisopod_lead_into_a_fire_six() -> None:
    picks, notes = kept(FIRE_SIX, pair=[])
    assert picks and all(p[0] != GOLISOPOD for p in picks)
    assert any(GOLISOPOD in p[1:] for p in picks)  # it may still come, just not lead
    assert notes["why"] == ["no golisopod lead (charizard)"] and notes["kept"] == len(picks) == 50


def test_at_most_one_of_golisopod_and_gholdengo_into_a_fire_six() -> None:
    picks, notes = kept(FIRE_SIX, lead={})
    assert picks and not any(GOLISOPOD in p and GHOLDENGO in p for p in picks)
    assert any(GOLISOPOD in p for p in picks) and any(GHOLDENGO in p for p in picks)
    assert len(picks) == 48  # the 4 trios with both, 3 leads each, are out
    both, _ = kept(FIRE_SIX)  # 60 - 10 Golisopod leads - 12 with both + 4 counted twice
    assert len(both) == 42 and all(p[0] != GOLISOPOD and not {GOLISOPOD, GHOLDENGO} <= set(p) for p in both)


def test_nothing_changes_without_a_fire_type() -> None:
    picks, notes = kept(NO_FIRE)
    assert picks == all_picks() and notes == {}


def test_every_pick_is_kept_when_no_pick_meets_the_rules() -> None:
    every = {name.lower(): ["Fire"] for name in GOLD}
    keep, notes = keep_mask(preview(FIRE_SIX), all_picks(), every, [])
    assert not any(keep) and notes["unsatisfiable"] and notes["kept"] == 0


@pytest.mark.parametrize(
    ("lead", "pair"),
    [({"golisopod": "Fire"}, []), ({"golisopod": ["Lava"]}, []), ({}, [{"types": ["Fire"]}]), ({}, {"x": 1})],
)
def test_bad_settings_are_rejected(lead, pair) -> None:
    with pytest.raises(ConfigError):
        check(lead, pair, "test")
