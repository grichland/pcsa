"""Hints from LIVE events: PP counted, Mega used, Scarf lock, Encore, Taunt, first turn, trapped. Advice only."""

from __future__ import annotations

from pcsa.battle import hints
from tests.live.turn_rig import TEAM

GHOLDENGO, MAMOSWINE = TEAM.members[0], TEAM.members[4]
START = {"v": "0.3", "type": "battle_start"}


def own(k: str, mon: str = "mamoswine", **fields) -> dict:
    return {"k": k, "side": "own", "mon": mon, **fields}


def opp(k: str, mon: str = "glimmora", **fields) -> dict:
    return {"k": k, "side": "opp", "mon": mon, **fields}


def test_pp_counts_uses_pressure_and_pp_rows() -> None:
    events = [START, own("enter"), opp("enter"), own("move", move="earthquake"),
              opp("ability", ability="pressure"), own("move", move="earthquake"),
              {"k": "pp", "side": "own", "mon": "mamoswine", "move": "iceshard", "lost": 4},
              opp("enter", mon="lucario"), own("move", move="earthquake")]  # fmt: skip
    got = hints.build(events, "mamoswine", MAMOSWINE)
    assert got.pp_counted["earthquake"] == (12 - 1 - 2 - 1, 12)
    assert got.pp_counted["iceshard"][0] == got.pp_counted["iceshard"][1] - 4
    assert hints.build(events, "mamoswine", MAMOSWINE, offsets={"earthquake": -2}).pp_counted["earthquake"][0] == 6


def test_choice_scarf_lock_resets_on_entry_and_ignores_struggle() -> None:
    events = [own("enter", "gholdengo"), own("move", "gholdengo", move="makeitrain"),
              own("move", "gholdengo", move="struggle")]  # fmt: skip
    assert hints.build(events, "gholdengo", GHOLDENGO).choice_locked == "makeitrain"
    events += [own("withdraw", "gholdengo"), own("enter", "gholdengo")]
    assert hints.build(events, "gholdengo", GHOLDENGO).choice_locked is None
    knocked = [
        *events[:2],
        {"k": "item", "side": "own", "mon": "gholdengo", "item": "choicescarf", "what": "removed"},
    ]
    assert hints.build(knocked, "gholdengo", GHOLDENGO).choice_locked is None


def test_encore_taunt_first_turn_and_trapped() -> None:
    events = [own("enter"), own("move", move="iceshard"), own("volatile", effect="encore"),
              own("volatile", effect="taunt"), own("volatile", effect="firespin")]  # fmt: skip
    got = hints.build(events, "mamoswine", MAMOSWINE)
    assert (got.encore, got.taunted, got.first_turn_over, got.trapped) == ("iceshard", True, True, True)
    events += [own("volatile_end", effect="encore"), own("volatile_end", effect="taunt")]
    got = hints.build(events, "mamoswine", MAMOSWINE)
    assert (got.encore, got.taunted, got.trapped) == (None, False, True)
    fresh = hints.build([*events, own("withdraw"), own("enter")], "mamoswine", MAMOSWINE)
    assert not fresh.first_turn_over and not fresh.trapped


def test_mega_used_from_the_log_or_the_screen() -> None:
    assert hints.build([own("mega", "garchomp")], "mamoswine", MAMOSWINE).mega_used
    assert hints.build([], "mamoswine", MAMOSWINE, mega_spent=True).mega_used
    assert not hints.build([opp("mega")], "mamoswine", MAMOSWINE).mega_used
    unknown = hints.build([own("mega")], None, None, refused=(2,))
    assert unknown.mega_used and unknown.refused == (2,) and unknown.pp_counted == {}
