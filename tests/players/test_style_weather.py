"""style_weather: the weather trio at preview and the weather rules on top of free_turn.

The threat estimates are test_disciplined's FakeThreat table plus `fractions` for hit_fraction; sixes are hand-built.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from pcsa.battle.state import replay
from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config
from pcsa.players import disciplined, free_turn, mirror, style_weather
from pcsa.players.base import DecisionView, Degrade, ScreenFacts, TeamPreviewView
from pcsa.players.factory import KINDS, LIVE_KINDS, SIM_KINDS, build_player
from pcsa.players.free_turn import FreeTurnPlayer
from pcsa.players.style_weather import Plan, StyleWeatherPlayer, WeatherParams
from tests.players.test_disciplined import FakeThreat, FakeTracker

PRELUDE = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9"]
SUN_SIX = {
    "Torkoal": ("drought", ["lavaplume", "earthpower"]),
    "Venusaur": ("chlorophyll", ["gigadrain", "sludgebomb", "solarbeam"]),
    "Charizard": ("blaze", ["flamethrower", "airslash"]),
    "Garchomp": ("roughskin", ["earthquake", "dragonclaw"]),
    "Clefable": ("unaware", ["moonblast"]),
    "Snorlax": ("thickfat", ["bodyslam"]),
}
RAIN_SIX = {
    "Garchomp": ("roughskin", ["earthquake", "dragonclaw"]),
    "Politoed": ("drizzle", ["scald", "icebeam"]),
    "Kingdra": ("swiftswim", ["dracometeor", "dragonpulse"]),
    "Barraskewda": ("swiftswim", ["closecombat", "psychicfangs"]),
    "Clefable": ("unaware", ["moonblast"]),
    "Gengar": ("cursedbody", ["shadowball", "sludgebomb"]),
}
DRY_SIX = {
    k: SUN_SIX.get(k) or RAIN_SIX[k] for k in ("Garchomp", "Venusaur", "Charizard", "Clefable", "Snorlax", "Gengar")
}
TRIO = {k: SUN_SIX[k] for k in ("Torkoal", "Venusaur", "Charizard")}
SUN = "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal"
PLAN = Plan("replay", style_weather.styles.load().weathers["sun"], "torkoal", frozenset({"venusaur", "charizard"}))


class WeatherThreat(FakeThreat):
    """FakeThreat plus `fractions[(attacker, move id)]` for hit_fraction (0.3 otherwise)."""

    def __init__(self, hits: dict, fractions: dict | None = None, **kwargs) -> None:
        super().__init__(hits, **kwargs)
        self.fractions = fractions or {}

    def hit_fraction(self, battle, attacker, defender, move, catalog, tracker=None):
        return self.fractions.get((attacker.species, move.id), 0.3)


@pytest.fixture
def fake(monkeypatch):
    def install(hits: dict | None = None, fractions: dict | None = None) -> WeatherThreat:
        estimates = WeatherThreat(hits or {}, fractions)
        for module in (disciplined, free_turn, style_weather):
            monkeypatch.setattr(module, "threat", estimates)
        monkeypatch.setattr(disciplined, "opponent", SimpleNamespace(Tracker=FakeTracker))
        sets = SimpleNamespace(assumed=lambda *a: SimpleNamespace(moves=(), ability=""), move_object=lambda i: None)
        monkeypatch.setattr(free_turn, "opponent", sets)
        return estimates

    return install


def _pokemon(team: dict, active: str = "", fainted: str = "") -> list[dict]:
    return [
        {
            "ident": f"p1: {name}",
            "details": f"{name}, L50",
            "condition": "0 fnt" if name == fainted else "150/150",
            "active": name == active,
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100},
            "moves": moves,
            "baseAbility": ability,
            "ability": ability,
            "item": "",
        }
        for name, (ability, moves) in team.items()
    ]


def preview(team: dict, foes: tuple[str, ...] = ("Garchomp", "Clefable")) -> TeamPreviewView:
    lines = [*PRELUDE, "|clearpoke", *(f"|poke|p2|{foe}, L50|" for foe in foes), "|teampreview|3"]
    b = replay(lines, {"side": {"name": "me", "id": "p1", "pokemon": _pokemon(team)}, "rqid": 1, "teamPreview": True})
    return TeamPreviewView(b, tuple(team), random.Random(0))


def battle(ours: str, foe: str = "Metagross", *, sun: bool = False, forced: bool = False):
    """`ours` (of TRIO) in front of `foe`, sun up or not; `forced`: `ours` has just fainted."""
    lines = [*PRELUDE, "|teamsize|p1|3", "|teamsize|p2|3", "|start", f"|switch|p1a: {ours}|{ours}, L50|150/150"]
    lines += [f"|switch|p2a: {foe}|{foe}, L50|100/100", *([SUN] if sun else []), "|turn|1"]
    request: dict = {"side": {"name": "me", "id": "p1", "pokemon": _pokemon(TRIO, ours, ours if forced else "")}}
    request["rqid"] = 1
    if forced:
        request["forceSwitch"] = [True]
    else:
        request["active"] = [{"moves": [{"move": m, "id": m, "pp": 10, "maxpp": 10} for m in TRIO[ours][1]]}]
    return replay(lines, request)


def pick(player, view: TeamPreviewView) -> tuple[list[int], dict]:
    return player.choose_team(view, Deadline.after(FakeClock(), 30.0)), player.notes()


def decide(player, b) -> tuple[str, dict]:
    view = DecisionView(mirror.legal_actions(b), ScreenFacts(), Degrade.FULL, random.Random(0), b)
    return player.choose_action(view, Deadline.after(FakeClock(), 30.0)).message, player.notes()


def player(plan: Plan | None = PLAN) -> StyleWeatherPlayer:
    me = StyleWeatherPlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]
    me._plan = plan
    return me


# Team preview ---------------------------------------------------------------------------------------------
def test_a_sun_six_brings_the_setter_first_and_two_abusers(fake) -> None:
    fake()
    picks, notes = pick(player(None), preview(SUN_SIX))
    assert picks == [0, 1, 2] and notes == {"style": "weather:sun"}  # Torkoal, Venusaur, Charizard


def test_politoed_goes_second_behind_an_abuser(fake) -> None:
    fake()
    picks, notes = pick(player(None), preview(RAIN_SIX))
    assert picks[1] == 1 and {picks[0], picks[2]} == {2, 3} and notes["style"] == "weather:rain"


def test_a_rival_setter_is_noted_and_our_setter_still_leads(fake) -> None:
    fake()
    picks, notes = pick(player(None), preview(SUN_SIX, ("Pelipper", "Garchomp", "Torkoal")))
    assert picks[0] == 0 and notes == {"style": "weather:sun", "weather_clash": "rain"}


def test_a_setter_of_another_weather_is_not_brought_as_an_abuser(fake) -> None:
    fake()
    snow = {
        "Abomasnow": ("snowwarning", ["blizzard", "woodhammer"]),
        "Pelipper": ("drizzle", ["weatherball", "hurricane"]),  # abuses snow by Weather Ball, but sets rain
        "Weavile": ("pressure", ["iciclecrash", "knockoff"]),
        "Glaceon": ("icebody", ["icebeam", "shadowball"]),
        "Garchomp": ("roughskin", ["earthquake"]),
        "Clefable": ("unaware", ["moonblast"]),
    }
    picks, notes = pick(player(None), preview(snow))
    assert picks == [0, 2, 3] and notes["style"] == "weather:snow"


def test_no_fit_plays_exactly_like_free_turn(fake) -> None:
    fake({("metagross", "venusaur"): (0.6, "meteormash")})
    me = player(None)
    picks, notes = pick(me, preview(DRY_SIX))
    assert picks == pick(FreeTurnPlayer(catalog=SimpleNamespace()), preview(DRY_SIX))[0]  # type: ignore[arg-type]
    assert notes == {"style": "no fit"} and me._plan is None
    assert decide(me, battle("Venusaur")) == decide(FreeTurnPlayer(catalog=SimpleNamespace()), battle("Venusaur"))  # type: ignore[arg-type]


def test_a_plan_from_another_battle_is_not_used(fake) -> None:
    fake({("metagross", "venusaur"): (0.6, "meteormash")})
    assert decide(player(Plan("other", PLAN.weather, "torkoal", PLAN.abusers)), battle("Venusaur"))[1]["rule"] != (
        "weather:reset"
    )


# Battle rules ---------------------------------------------------------------------------------------------
def test_weather_down_and_threatened_brings_the_setter_back(fake) -> None:
    fake({("metagross", "venusaur"): (0.6, "meteormash")})
    chosen, notes = decide(player(), battle("Venusaur"))
    assert chosen == "/choose switch Torkoal" and notes["rule"] == "weather:reset"
    fake()  # not threatened: no reset
    chosen, notes = decide(player(), battle("Venusaur"))
    assert chosen.startswith("/choose move") and notes["rule"] == "attack"


def test_an_abuser_in_its_weather_never_switches_voluntarily(fake) -> None:
    hits = {("metagross", "venusaur"): (0.6, "meteormash"), ("metagross", "torkoal"): (0.1, "meteormash")}
    fake({**hits, ("torkoal", "metagross"): (0.5, "lavaplume")})
    freely = decide(FreeTurnPlayer(catalog=SimpleNamespace()), battle("Venusaur", sun=True))  # type: ignore[arg-type]
    assert freely[0] == "/choose switch Torkoal"  # free_turn would leave
    chosen, notes = decide(player(), battle("Venusaur", sun=True))
    assert chosen.startswith("/choose move") and "switch" not in notes


def test_the_weather_move_within_the_slack_is_preferred(fake) -> None:
    fake(fractions={("charizard", "flamethrower"): 0.5, ("charizard", "airslash"): 0.54})
    assert decide(player(), battle("Charizard", sun=True)) == (
        "/choose move flamethrower",
        {"rule": "weather:boosted"},
    )
    fake(fractions={("charizard", "flamethrower"): 0.5, ("charizard", "airslash"): 0.7})
    assert decide(player(), battle("Charizard", sun=True))[1] == {"rule": "attack"}  # max_damage's own pick


def test_solar_beam_only_in_sun(fake) -> None:
    fake(fractions={("venusaur", "solarbeam"): 0.8})
    assert decide(player(), battle("Venusaur", sun=True)) == ("/choose move solarbeam", {"rule": "weather:boosted"})
    chosen, _ = decide(player(), battle("Venusaur"))
    assert chosen.startswith("/choose move") and chosen != "/choose move solarbeam"


def test_a_threatened_setter_pivots_to_the_safest_abuser(fake) -> None:
    hits = {("metagross", "torkoal"): (0.6, "earthquake"), ("metagross", "venusaur"): (0.2, "meteormash")}
    fake({**hits, ("metagross", "charizard"): (0.3, "meteormash")})
    chosen, notes = decide(player(), battle("Torkoal", sun=True))
    assert chosen == "/choose switch Venusaur" and notes["rule"] == "weather:to_abuser"
    fake({**hits, ("metagross", "venusaur"): (0.55, "meteormash"), ("metagross", "charizard"): (0.6, "x")})
    chosen, notes = decide(player(), battle("Torkoal", sun=True))
    assert chosen.startswith("/choose move") and notes["rule"] == "attack"


def test_forced_switch_prefers_the_setter_out_of_weather_else_the_hardest_hitting_abuser(fake) -> None:
    fake({("venusaur", "metagross"): (0.2, "gigadrain"), ("charizard", "metagross"): (0.5, "flamethrower")})
    chosen, notes = decide(player(), battle("Venusaur", forced=True))
    assert chosen == "/choose switch Torkoal" and notes["rule"] == "weather:forced_setter"
    chosen, notes = decide(player(), battle("Torkoal", sun=True, forced=True))  # the sun outlasts its setter
    assert chosen == "/choose switch Charizard" and notes["rule"] == "weather:forced_abuser"


# Factory --------------------------------------------------------------------------------------------------
def test_factory_builds_it_with_the_config_thresholds(config: Config, fake) -> None:
    fake()
    built = build_player(config, "style_weather")
    assert isinstance(built, StyleWeatherPlayer) and built.name == "style_weather"
    assert built.weather == WeatherParams()  # the yaml holds the defaults
    assert built.params()["pivot_max_hit"] == 0.5 and built.params()["free_hit_max"] == 0.4
    assert "style_weather" in SIM_KINDS and "style_weather" in KINDS and "style_weather" not in LIVE_KINDS
