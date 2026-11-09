"""style_hazard_lead: the setter-led preview, the turn-1 Stealth Rock, the setter's pivot and the Rocks-aware pick.

The threat estimates are test_disciplined's FakeThreat table; the opponent's assumed sets are SETS / ABILITIES.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
from poke_env.battle import PokemonType

from pcsa.battle.state import replay
from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config
from pcsa.players import disciplined, free_turn, mirror, opponent
from pcsa.players.base import DecisionView, Degrade, ScreenFacts, TeamPreviewView
from pcsa.players.factory import KINDS, LIVE_KINDS, SIM_KINDS, build_player
from pcsa.players.free_turn import FreeTurnPlayer
from pcsa.players.style_hazard_lead import HazardLeadParams, StyleHazardLeadPlayer, rocks_chip
from tests.players.test_disciplined import FakeThreat, FakeTracker

PLAYERS = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9"]
PRELUDE = [*PLAYERS, "|teamsize|p1|3", "|teamsize|p2|3", "|start"]
TEAM = {  # the three brought: the Stealth Rock setter and two attackers
    "Garchomp": (183, ["stealthrock", "earthquake", "swordsdance", "dragonclaw"]),
    "Azumarill": (175, ["playrough", "aquajet", "swordsdance"]),
    "Charizard": (153, ["airslash", "flamethrower"]),
}
BENCH = {  # the rest of the six: one attacking move each, so no attacker role
    "Snorlax": (235, ["bodyslam", "curse"]),
    "Blissey": (325, ["seismictoss", "softboiled"]),
    "Slowbro": (170, ["scald", "slackoff"]),
}
THEIRS = ["Metagross", "Espeon", "Corviknight", "Volcarona", "Dragonite", "Milotic"]
SETS = {
    "metagross": ("meteormash", "zenheadbutt", "earthquake", "bulletpunch"),
    "espeon": ("psychic", "shadowball", "dazzlinggleam", "morningsun"),
    "corviknight": ("bravebird", "bodypress", "defog", "roost"),
    "volcarona": ("fierydance", "bugbuzz", "gigadrain", "morningsun"),
}
ABILITIES = {"espeon": "magicbounce", "metagross": "clearbody"}
ROCKS_UP = "|-sidestart|p2: opponent|move: Stealth Rock"
# Garchomp's own hit when a test threatens it: at or above `trade_hit_min`, so it attacks rather than trades out
HIT_BACK = {("garchomp", "metagross"): (0.3, "earthquake")}


@pytest.fixture
def fake(monkeypatch):
    def install(hits: dict | None = None) -> FakeThreat:
        estimates = FakeThreat(hits or {})
        for module in (disciplined, free_turn):
            monkeypatch.setattr(module, "threat", estimates)
        monkeypatch.setattr(disciplined, "opponent", SimpleNamespace(Tracker=FakeTracker))
        sets = SimpleNamespace(
            assumed=lambda battle, mon, *known: SimpleNamespace(
                moves=SETS.get(mon.species, ()), ability=ABILITIES.get(mon.species, "")
            ),
            move_object=opponent.move_object,
        )
        monkeypatch.setattr(free_turn, "opponent", sets)
        return estimates

    return install


def _member(name: str, full: int, moves: list[str], condition: str, active: bool = False) -> dict:
    stats = {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100}
    return {"ident": f"p1: {name}", "details": f"{name}, L50", "condition": condition, "active": active,
            "stats": stats, "moves": moves, "baseAbility": "", "item": ""}  # fmt: skip


def preview(six: dict, theirs: list[str] = THEIRS):
    """Team preview of our `six` (name -> (HP, moves)) against `theirs`."""
    lines = [*PLAYERS, *(f"|poke|p2|{name}, L50|" for name in theirs), "|teampreview"]
    pokemon = [_member(name, full, moves, f"{full}/{full}") for name, (full, moves) in six.items()]
    b = replay(lines, {"side": {"name": "me", "id": "p1", "pokemon": pokemon}, "rqid": 1, "teamPreview": True})
    b._max_team_size = 3
    return b


def battle(foe: str = "Metagross", *, turn: int = 1, rocks: bool = False, hp: float = 1.0, **kw):
    """Garchomp (at `hp`) against `foe`; kw: `bench_foe` revealed before `foe`, `fainted` for a forced pick."""
    bench_foe, fainted = kw.get("bench_foe"), kw.get("fainted", False)
    lines = [*PRELUDE, "|switch|p1a: Garchomp|Garchomp, L50|183/183"]
    if bench_foe:
        lines.append(f"|switch|p2a: {bench_foe}|{bench_foe}, L50|100/100")
    lines += [f"|switch|p2a: {foe}|{foe}, L50|100/100", *([ROCKS_UP] if rocks else []), "|turn|1"]
    lines += [f"|turn|{t}" for t in range(2, turn + 1)]
    if fainted:
        lines.append("|faint|p1a: Garchomp")
    pokemon = []
    for name, (full, moves) in TEAM.items():
        here = round(full * hp) if name == "Garchomp" else full
        condition = "0 fnt" if fainted and name == "Garchomp" else f"{here}/{full}"
        pokemon.append(_member(name, full, moves, condition, name == "Garchomp"))
    request: dict = {"side": {"name": "me", "id": "p1", "pokemon": pokemon}, "rqid": 1}
    if fainted:
        request["forceSwitch"] = [True]
    else:
        request["active"] = [{"moves": [{"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal"}
                                        for m in TEAM["Garchomp"][1]]}]  # fmt: skip
    return replay(lines, request)


def decide(player, b) -> tuple[str, dict]:
    view = DecisionView(mirror.legal_actions(b), ScreenFacts(), Degrade.FULL, random.Random(0), b)
    return player.choose_action(view, Deadline.after(FakeClock(), 30.0)).message, player.notes()


def pick(player, b) -> tuple[list[int], dict]:
    return player.choose_team(TeamPreviewView(b, (None,) * 6, random.Random(0)), Deadline.after(FakeClock(), 30.0)), (
        player.notes()
    )


def planned() -> StyleHazardLeadPlayer:
    """A player whose preview chose Garchomp + Azumarill + Charizard."""
    me = StyleHazardLeadPlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]
    picks, _ = pick(me, preview({**TEAM, **BENCH}))
    assert picks[0] == 0 and sorted(picks) == [0, 1, 2]
    return me


# Preview --------------------------------------------------------------------------------------------------
def test_fit_leads_the_setter_with_two_attackers(fake) -> None:
    fake()
    me = StyleHazardLeadPlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]
    six = {**BENCH, "Charizard": TEAM["Charizard"], "Garchomp": TEAM["Garchomp"], "Azumarill": TEAM["Azumarill"]}
    picks, notes = pick(me, preview(six))
    assert picks[0] == 4 and sorted(picks[1:]) == [3, 5]
    assert notes == {"style": "hazard_lead", "lead": "garchomp", "attackers": notes["attackers"]}
    assert sorted(notes["attackers"]) == ["azumarill", "charizard"]


def test_attackers_by_summed_matchup(fake) -> None:
    fake()
    me = StyleHazardLeadPlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]
    six = {
        **TEAM,
        "Snorlax": (235, ["bodyslam", "earthquake"]),
        "Blissey": BENCH["Blissey"],
        "Slowbro": BENCH["Slowbro"],
    }
    fire = ["Arcanine", "Torkoal", "Heatran", "Volcarona", "Chandelure", "Cinderace"]
    b = preview(six, fire)
    mons, foes = list(b.team.values()), b.teampreview_opponent_team
    score = {i: sum(me.heuristic.matchup(mons[i], f) for f in foes) for i in (1, 2, 3)}
    picks, _ = pick(me, b)
    assert picks[0] == 0 and picks[1:] == sorted(score, key=lambda i: -score[i])[:2]


def test_no_fit_plays_free_turn(fake) -> None:
    fake({("metagross", "garchomp"): (0.6, "meteormash"), **HIT_BACK})
    no_rocks = {name: (full, [m for m in moves if m != "stealthrock"]) for name, (full, moves) in TEAM.items()}
    me, base = StyleHazardLeadPlayer(catalog=SimpleNamespace()), FreeTurnPlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]
    picks, notes = pick(me, preview({**no_rocks, **BENCH}))
    assert notes == {"style": "no fit"} and picks == pick(base, preview({**no_rocks, **BENCH}))[0]
    chosen, notes = decide(me, battle())
    assert (chosen, notes) == decide(base, battle()) and chosen != "/choose move stealthrock"
    assert notes["free"] == "not free: hit 0.60" and "hazard" not in notes


def test_a_choice_setter_is_no_fit(fake) -> None:
    fake()
    b = preview({**TEAM, **BENCH})
    next(iter(b.team.values()))._item = "choicescarf"  # Garchomp
    picks, notes = pick(StyleHazardLeadPlayer(catalog=SimpleNamespace()), b)  # type: ignore[arg-type]
    assert notes["style"].startswith("no fit") and len(picks) == 3  # the style rules reject a Choice setter


# Rocks lead -----------------------------------------------------------------------------------------------
def test_turn_one_rocks_though_not_free(fake) -> None:
    fake({("metagross", "garchomp"): (0.6, "meteormash")})
    assert decide(planned(), battle()) == ("/choose move stealthrock", {"rule": "hazard:rocks"})
    assert decide(FreeTurnPlayer(catalog=SimpleNamespace()), battle())[0] != "/choose move stealthrock"  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("foe", "hit", "hp", "why"),
    [
        ("Metagross", 1.0, 1.0, "they KO"),
        ("Metagross", 0.8, 1.0, "hit 0.80"),
        ("Metagross", 0.6, 0.7, "hit 0.60"),  # 0.75 of the current HP
        ("Espeon", 0.6, 1.0, "Magic Bounce"),
        ("Corviknight", 0.6, 1.0, "foe has defog"),
    ],
)
def test_no_rocks_when_unsafe_or_wasted(fake, foe: str, hit: float, hp: float, why: str) -> None:
    fake({(foe.lower(), "garchomp"): (hit, "x")})
    chosen, notes = decide(planned(), battle(foe, hp=hp))
    assert chosen != "/choose move stealthrock" and notes["hazard"] == f"no rocks: {why}"


@pytest.mark.parametrize(("item", "hp", "chosen"), [("focussash", 1.0, "stealthrock"), ("focussash", 0.9, "x")])
def test_a_sash_at_full_hp_survives_a_ko(fake, item: str, hp: float, chosen: str) -> None:
    fake({("metagross", "garchomp"): (1.2, "meteormash")})
    b = battle(hp=hp)
    b.active_pokemon._item = item  # type: ignore[union-attr]
    assert (decide(planned(), b)[0] == "/choose move stealthrock") == (chosen == "stealthrock")


def test_no_rocks_from_a_choice_holder(fake) -> None:
    fake()
    b = battle()
    b.active_pokemon._item = "choicescarf"  # type: ignore[union-attr]
    chosen, notes = decide(planned(), b)
    assert chosen != "/choose move stealthrock" and notes["hazard"] == "no rocks: choice item choicescarf"


def test_rocks_once_and_turn_one_only(fake) -> None:
    fake({("metagross", "garchomp"): (0.6, "meteormash")})
    me = planned()
    assert decide(me, battle())[0] == "/choose move stealthrock"
    chosen, notes = decide(me, battle())
    assert chosen != "/choose move stealthrock" and notes["hazard"] == "no rocks: already used"
    assert "hazard" not in decide(planned(), battle(turn=2))[1]  # later turns are free_turn's


def test_the_setter_never_sets_up_attackers_may(fake) -> None:
    fake()
    chosen, notes = decide(planned(), battle("Espeon", turn=2))  # a free turn, Rocks bounced
    assert chosen.startswith("/choose move") and "swordsdance" not in chosen and notes["rule"] == "attack"
    assert decide(FreeTurnPlayer(catalog=SimpleNamespace()), battle("Espeon", turn=2))[0] == (  # type: ignore[arg-type]
        "/choose move swordsdance"
    )
    me = planned()
    assert not me._setter_active(
        SimpleNamespace(battle_tag="replay", active_pokemon=SimpleNamespace(species="azumarill"))
    )


# Setter after Rocks ---------------------------------------------------------------------------------------
PIVOT = {
    ("metagross", "garchomp"): (0.5, "meteormash"),
    ("metagross", "azumarill"): (0.3, "meteormash"),
    ("metagross", "charizard"): (0.45, "earthquake"),
    **HIT_BACK,
}


def test_threatened_setter_goes_to_an_attacker_after_rocks(fake) -> None:
    fake(PIVOT)
    chosen, notes = decide(planned(), battle(turn=2, rocks=True))
    assert chosen == "/choose switch Azumarill" and notes["rule"] == "hazard:to_attacker"
    chosen, notes = decide(planned(), battle(turn=2))  # no Rocks up: disciplined's stricter switch says stay
    assert chosen.startswith("/choose move") and notes["rule"] == "attack"


def test_pivot_respects_back_to_back_and_a_ko(fake) -> None:
    fake(PIVOT)
    me = planned()
    me._tag, me._turn, me._last_switch, me._streak = "replay", 1, 1, 1
    chosen, notes = decide(me, battle(turn=2, rocks=True))
    assert (
        chosen.startswith("/choose move")
        and notes["hazard"] == "no pivot: switched last turn and the opponent did not"
    )
    fake({**PIVOT, ("garchomp", "metagross"): (1.0, "earthquake")})
    assert decide(planned(), battle(turn=2, rocks=True))[0] == "/choose move earthquake"


# Rocks chip -----------------------------------------------------------------------------------------------
def test_rocks_chip() -> None:
    assert rocks_chip([PokemonType.FIRE, PokemonType.FLYING]) == 0.5
    assert rocks_chip([PokemonType.DRAGON, PokemonType.GROUND]) == 1 / 16
    assert rocks_chip([PokemonType.NORMAL, None]) == 1 / 8


def test_forced_pick_counts_the_chip_on_incoming_foes(fake) -> None:
    fake({("azumarill", "volcarona"): (0.6, "playrough")})  # 0.6 + Rocks 0.5 on Volcarona: a KO as it comes in
    chosen, notes = decide(planned(), battle(rocks=True, fainted=True, bench_foe="Volcarona"))
    assert chosen == "/choose switch Azumarill"
    assert notes == {"rule": "hazard:forced", "ko_after_rocks": ["volcarona"]}
    chosen, notes = decide(planned(), battle(fainted=True, bench_foe="Volcarona"))  # no Rocks: max_damage's pick
    assert chosen == "/choose switch Charizard" and notes["rule"] == "forced"


# Factory --------------------------------------------------------------------------------------------------
def test_factory_builds_it_with_the_config_thresholds(config: Config, fake) -> None:
    fake()
    built = build_player(config, "style_hazard_lead")
    assert isinstance(built, StyleHazardLeadPlayer) and built.name == "style_hazard_lead"
    assert built.hazard == HazardLeadParams()  # the yaml holds the defaults
    assert built.params()["rocks_max_hit"] == 0.75 and built.params()["free_hit_max"] == 0.4
    assert "style_hazard_lead" in SIM_KINDS and "style_hazard_lead" in KINDS and "style_hazard_lead" not in LIVE_KINDS
