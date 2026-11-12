"""pcsa.belief: the belief tracker's live fixes on hand-made battles.

- live battle logs print no critical hits, so with crit_reported=False every hit may have been one (the damage
  likelihood mixes in a x1.5 part) instead of crits being left out;
- a decision whose protocol does not extend the one read before (a battle log read differently) rebuilds the
  tracker from the whole protocol, and gives what a fresh tracker gives;
- a seen build with an empty item or ability holds an unknown one (empty_unknown) instead of "no item";
- the outputs the features read (stat intervals, speed, damage ranges) are finite and ordered.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcsa.battle.builds import DEFAULT_PATH, Catalog
from pcsa.battle.state import replay
from pcsa.belief import fast
from pcsa.belief.member import FastMember
from pcsa.common import paths

HEAD = [
    "|player|p1|me|",
    "|player|p2|opponent|",
    "|gen|9",
    "|poke|p1|Garchomp, L50|",
    "|poke|p1|Gholdengo, L50|",
    "|poke|p2|Kingambit, L50|",
    "|poke|p2|Dragonite, L50|",
    "|start",
    "|switch|p1a: Garchomp|Garchomp, L50|183/183",
    "|switch|p2a: Kingambit|Kingambit, L50|100/100",
    "|turn|1",
]
TURN1 = [
    "|move|p1a: Garchomp|Earthquake|p2a: Kingambit",
    "|-supereffective|p2a: Kingambit",
    "|-damage|p2a: Kingambit|38/100",
    "|move|p2a: Kingambit|Iron Head|p1a: Garchomp",
    "|-damage|p1a: Garchomp|130/183",
    "|turn|2",
]
TURN2 = [
    "|move|p1a: Garchomp|Earthquake|p2a: Kingambit",
    "|-supereffective|p2a: Kingambit",
    "|-damage|p2a: Kingambit|0 fnt",
    "|faint|p2a: Kingambit",
]


def request(hp: int = 183) -> dict:
    mons = [
        (
            "Garchomp",
            hp,
            183,
            ["earthquake", "dragonclaw", "swordsdance", "stoneedge"],
            "choicescarf",
            "roughskin",
            {"atk": 182, "def": 115, "spa": 90, "spd": 105, "spe": 169},
            True,
        ),
        (
            "Gholdengo",
            162,
            162,
            ["makeitrain", "shadowball", "nastyplot", "thunderbolt"],
            "airballoon",
            "goodasgold",
            {"atk": 72, "def": 115, "spa": 185, "spd": 112, "spe": 136},
            False,
        ),
    ]
    side = [
        {
            "ident": f"p1: {n}",
            "details": f"{n}, L50",
            "condition": f"{cur}/{top}",
            "active": active,
            "stats": st,
            "moves": mv,
            "baseAbility": ab,
            "ability": ab,
            "item": it,
        }
        for n, cur, top, mv, it, ab, st, active in mons
    ]
    moves = [
        {"move": m, "id": m, "pp": 8, "maxpp": 8, "target": "normal", "disabled": False} for m in side[0]["moves"]
    ]
    return {"active": [{"moves": moves}], "side": {"name": "me", "id": "p1", "pokemon": side}, "rqid": 2}


@pytest.fixture(scope="module")
def spaces():
    catalog = Catalog.load(seen=True)
    return catalog, fast.FastSpaces(catalog, fast.FastPriors(scarf_floor=0.15, empty_unknown=True))


def belief(spaces, crit_reported: bool = False) -> fast.BattleBelief:
    catalog, sp = spaces
    return fast.BattleBelief(catalog, spaces=sp, crit_reported=crit_reported)


def test_live_hits_may_be_crits_and_simulator_crit_lines_are_left_out(spaces, monkeypatch) -> None:
    calls: list[bool] = []
    real = FastMember.observe_hit

    def spy(self, *args, crit_possible: bool) -> None:
        calls.append(crit_possible)
        real(self, *args, crit_possible=crit_possible)

    monkeypatch.setattr(FastMember, "observe_hit", spy)
    lines = HEAD + TURN1
    live = belief(spaces, crit_reported=False)
    live.update(lines, None, request(130))
    assert calls == [True, True]  # our hit and theirs, each possibly a crit
    assert live.tracker.stats["hit_used_dealt"] == 1 and live.tracker.stats["hit_used_taken"] == 1
    calls.clear()
    sim = belief(spaces, crit_reported=True)
    sim.update(lines, None, request(130))
    assert calls == [False, False]
    w_live = live.tracker.member("p2: Kingambit").weights()
    w_sim = sim.tracker.member("p2: Kingambit").weights()
    assert w_live.shape == w_sim.shape and not np.allclose(w_live, w_sim)
    # a printed crit (the simulator) is not evidence about the build at all
    calls.clear()
    crit = belief(spaces, crit_reported=True)
    crit.update(HEAD + TURN1[:2] + ["|-crit|p2a: Kingambit"] + TURN1[2:], None, request(130))
    assert crit.tracker.stats["hit_skip_crit"] == 1 and calls == [False]


def test_a_protocol_that_does_not_extend_the_last_one_rebuilds(spaces) -> None:
    first = HEAD + TURN1
    reread = HEAD + [x.replace("38/100", "41/100") for x in TURN1] + TURN2  # turn 1 read differently
    b = belief(spaces)
    b.update(HEAD, None, request())  # the decisions of turns 1 and 2, then turn 2 again with the log re-read
    b.update(first, None, request(130))
    b.update(reread, None, request(130))
    assert b.rebuilds == 1 and b.counts["rebuilds"] == 1
    # the same as a belief that saw the re-read protocol all along, decision by decision
    fresh = belief(spaces)
    fresh.update(HEAD, None, request())
    fresh.update(reread[: len(first)], None, request(130))
    fresh.update(reread, None, request(130))
    key = "p2: Kingambit"
    assert fresh.rebuilds == 0
    assert np.array_equal(b.tracker.member(key).weights(), fresh.tracker.member(key).weights())
    assert b.tracker.stats == fresh.tracker.stats
    # a protocol that does extend the last one is read incrementally; the same protocol again changes nothing
    inc = belief(spaces)
    inc.update(HEAD, None, request())
    inc.update(first, None, request(130))
    inc.update(first + TURN2, None, request(130))
    version = inc.tracker.member(key).version
    inc.update(first + TURN2, None, request(130))
    assert inc.rebuilds == 0 and inc.tracker.member(key).version == version


def test_a_shorter_protocol_rebuilds_too(spaces) -> None:
    b = belief(spaces)
    b.update(HEAD + TURN1, None, request(130))
    b.update(HEAD, None, request())
    assert b.rebuilds == 1 and b.tracker.stats["hit_used_dealt"] == 0


def _seen(species: str, item: str, ability: str, moves: list[str]) -> dict:
    return {
        "source": "added",
        "count": 1,
        "first": "2026-11-02",
        "last": "2026-11-02",
        "species": species,
        "item": item,
        "ability": ability,
        "nature": "Modest",
        "points": {"hp": 32, "atk": 0, "def": 0, "spa": 32, "spd": 2, "spe": 0},
        "moves": moves,
        "known_moves": [],
        "known_item": bool(item),
        "known_ability": bool(ability),
    }


@pytest.fixture
def empty_seen(tmp_path) -> Catalog:
    """A seen-builds store like the ladder's: a Stunfisk with neither item nor ability read, a Floette whose
    ability was never shown (seen_builds.json has a dozen such builds)."""
    import json

    from pcsa.battle import seen_builds

    path = tmp_path / "seen_builds.json"
    data = {
        "schema": seen_builds.SCHEMA,
        "note": "test",
        "battles": [],
        "species": {
            "stunfisk": [_seen("Stunfisk", "", "", ["Earth Power", "Discharge", "Stealth Rock", "Foul Play"])],
            "floette": [_seen("Floette", "Floettite", "", ["Moonblast", "Calm Mind", "Psychic", "Protect"])],
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    plain = json.loads((paths.assets_dir() / DEFAULT_PATH).read_text(encoding="utf-8"))
    return Catalog(plain, seen=seen_builds.SeenBuilds.load(path))


@pytest.mark.parametrize(("species", "what"), [("stunfisk", "item"), ("stunfisk", "ability"), ("floette", "ability")])
def test_an_empty_seen_item_or_ability_is_unknown(empty_seen, species: str, what: str) -> None:
    old = fast.FastSpaces(empty_seen, fast.FastPriors(scarf_floor=0.15)).get(species)
    new = fast.FastSpaces(empty_seen, fast.FastPriors(scarf_floor=0.15, empty_unknown=True)).get(species)
    k = next(i for i, lb in enumerate(new.builds) if lb.seen and not getattr(lb, what))
    rows_old, rows_new = (old.r_build == k) & (old.r_comp == 0), (new.r_build == k) & (new.r_comp == 0)
    if what == "item":
        assert "" in old.items and "" not in new.items  # research: "no item"; pcsa live: unknown
        assert len(set(old.r_item[rows_old].tolist())) == 1
        assert len(set(new.r_item[rows_new].tolist())) == len(new.items)
    else:
        assert set(old.r_ability[rows_old].tolist()) == {0}  # research: the species' first ability
        assert len(set(new.r_ability[rows_new].tolist())) == len(new.abilities) > 1
    # the components keep their prior masses
    for sp in (old, new):
        assert np.exp(sp.r_lp).sum() + np.exp(sp.free_lp).sum() == pytest.approx(1.0, abs=1e-9)
        assert np.exp(sp.r_lp[sp.r_comp == 0]).sum() == pytest.approx(0.45, rel=1e-6)


def test_outputs_are_finite_and_ordered(spaces) -> None:
    lines = HEAD + TURN1
    battle = replay(lines, request(130), "outputs")
    b = belief(spaces)
    b.update_from(battle)
    theirs, ours = battle.opponent_active_pokemon, battle.active_pokemon
    lo, hi = b.stats(battle, theirs)
    assert all(0 < a <= c for a, c in zip(lo, hi, strict=True))
    sp = b.speed(battle, theirs, 169.0, coming=True)
    assert sp["faster"] + sp["tie"] + sp["slower"] == pytest.approx(1.0) and 0 < sp["lo"] <= sp["hi"]
    for move, theirs_attacking in (("earthquake", False), ("ironhead", True), ("kowtowcleave", True)):
        r = b.damage(battle, ours, theirs, move, theirs_attacking, coming=True)
        assert 0 <= r["lo"] <= r["point"] <= r["hi"] < 5
    # a member seen only at team preview reads its species' prior
    dragonite = next(m for m in battle.teampreview_opponent_team if m.species == "dragonite")
    lo, hi = b.stats(battle, dragonite)
    assert all(0 < a <= c for a, c in zip(lo, hi, strict=True))
