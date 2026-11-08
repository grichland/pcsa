"""endgame: disciplined until <= 2 members each, then the 1v1 chain picks stay / switch / the replacement.

Damage and speed are a table here (FakeThreat): per (attacker, defender, move) expected shares of max HP, and a
speed per species; disciplined's threat helpers read the same table.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from pcsa.battle.damage import Fighter
from pcsa.battle.state import replay
from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config
from pcsa.players import disciplined, endgame, mirror
from pcsa.players.base import DecisionView, Degrade, ScreenFacts
from pcsa.players.disciplined import DisciplinedPlayer
from pcsa.players.endgame import EndgameParams, EndgamePlayer
from pcsa.players.factory import KINDS, LIVE_KINDS, SIM_KINDS, build_player
from pcsa.players.opponent import move_object

PRELUDE = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|teamsize|p1|3", "|teamsize|p2|3"]
TEAM = {
    "Azumarill": (175, ["playrough", "aquajet"]),
    "Slowbro": (170, ["surf", "slackoff"]),
    "Charizard": (153, ["airslash", "flamethrower"]),
}
FOE_MOVES = {
    "dragonite": ["outrage", "extremespeed"],
    "gengar": ["shadowball"],
    "garchomp": ["earthquake"],
    "clefable": ["moonblast"],
}
SPEED = {"azumarill": 50, "slowbro": 30, "charizard": 100, "dragonite": 80, "gengar": 110, "clefable": 60}
# Dragonite outspeeds and KOs Azumarill, which barely scratches it; Slowbro walls Dragonite and beats Gengar.
HITS = {
    ("dragonite", "azumarill", "outrage"): 1.0,
    ("azumarill", "dragonite", "playrough"): 0.3,
    ("dragonite", "slowbro", "outrage"): 0.15,
    ("slowbro", "dragonite", "surf"): 0.25,
    ("gengar", "slowbro", "shadowball"): 0.2,
    ("slowbro", "gengar", "surf"): 0.5,
    ("gengar", "azumarill", "shadowball"): 0.4,
    ("azumarill", "gengar", "playrough"): 0.6,
    ("dragonite", "charizard", "outrage"): 0.5,
    ("charizard", "dragonite", "airslash"): 0.2,
}


class FakeThreat:
    """endgame's and disciplined's threat helpers on the `hits` table."""

    def __init__(self, hits: dict, fail: bool = False) -> None:
        self.hits, self.fail = hits, fail

    def hit_fraction(self, battle, attacker, defender, move, catalog, tracker=None):
        if self.fail:
            raise RuntimeError("damage table broke")
        return self.hits.get((attacker.species, defender.species, move.id), 0.0)

    def known_moves(self, battle, mon, catalog, tracker=None):
        if any(m is mon for m in battle.team.values()):
            return list(mon.moves.values())
        return [move_object(m) for m in FOE_MOVES.get(mon.species, [])]

    def fighters(self, battle, mon, catalog, tracker=None):
        return Fighter({"spe": SPEED.get(mon.species, 50)}, (), "", "", {}, "", {})

    def best_hit(self, battle, attacker, defender, catalog, tracker=None):
        shares = [(v, m) for (a, d, m), v in self.hits.items() if (a, d) == (attacker.species, defender.species)]
        return max(shares, default=(0.1, None))

    def can_ko(self, battle, attacker, defender, catalog, tracker=None):
        return self.best_hit(battle, attacker, defender, catalog)[0] >= defender.current_hp_fraction

    def outspeeds(self, battle, a, b, catalog, tracker=None):
        return SPEED.get(a.species, 50) > SPEED.get(b.species, 50)

    def threatened(self, battle, mine, theirs, catalog, tracker=None, threshold=0.4):
        return self.best_hit(battle, theirs, mine, catalog)[0] >= threshold

    def priority_ko(self, battle, attacker, defender, catalog, tracker=None):
        return None


class FakeTracker:
    def __init__(self, catalog=None) -> None:
        pass

    def reset(self) -> None:
        pass

    def observe(self, battle) -> None:
        pass


@pytest.fixture
def fake(monkeypatch):
    def install(hits: dict | None = None, fail: bool = False) -> FakeThreat:
        estimates = FakeThreat({**HITS, **(hits or {})}, fail)
        monkeypatch.setattr(disciplined, "threat", estimates)
        monkeypatch.setattr(endgame, "threat", estimates)
        monkeypatch.setattr(disciplined, "opponent", SimpleNamespace(Tracker=FakeTracker))
        return estimates

    return install


def battle(hp: dict[str, float], active: str, foes: list[tuple[str, int]], preview: tuple[str, ...] = ()):
    """Our members at `hp` (0 fainted; a fainted `active` means a forced replacement) against `foes` (species,
    HP%) revealed in order, 0 fainted, the last one active; `preview`: their team-preview species."""
    lines = [*PRELUDE, *(f"|poke|p2|{name}, L50|" for name in preview), "|start"]
    full = TEAM[active][0]
    lines.append(f"|switch|p1a: {active}|{active}, L50|{max(1, round(full * hp[active]))}/{full}")
    if hp[active] <= 0:
        lines.append(f"|faint|p1a: {active}")
    for name, percent in foes:
        lines.append(f"|switch|p2a: {name}|{name}, L50|{max(1, percent)}/100")
        if percent <= 0:
            lines.append(f"|faint|p2a: {name}")
    lines.append("|turn|1")
    pokemon = [
        {
            "ident": f"p1: {name}",
            "details": f"{name}, L50",
            "condition": f"{round(full * hp[name])}/{full}" if hp[name] > 0 else "0 fnt",
            "active": name == active,
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100},
            "moves": moves,
            "baseAbility": "",
            "item": "",
        }
        for name, (full, moves) in TEAM.items()
    ]
    request: dict = {"side": {"name": "me", "id": "p1", "pokemon": pokemon}, "rqid": 1}
    if hp[active] <= 0:
        request["forceSwitch"] = [True]
    else:
        moves = [{"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal"} for m in TEAM[active][1]]
        request["active"] = [{"moves": moves}]
    return replay(lines, request)


def decide(player, b) -> tuple[str, dict]:
    view = DecisionView(mirror.legal_actions(b), ScreenFacts(), Degrade.FULL, random.Random(0), b)
    return player.choose_action(view, Deadline.after(FakeClock(), 30.0)).message, player.notes()


def player() -> EndgamePlayer:
    return EndgamePlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]


ALL_UP = {"Azumarill": 1.0, "Slowbro": 1.0, "Charizard": 1.0}
TWO_UP = {"Azumarill": 1.0, "Slowbro": 1.0, "Charizard": 0.0}
TWO_FOES = [("Gengar", 100), ("Garchomp", 0), ("Dragonite", 100)]  # Gengar on their bench, Garchomp down


def test_three_against_three_plays_exactly_disciplined(fake) -> None:
    fake()
    b = battle(ALL_UP, "Azumarill", [("Dragonite", 100)])
    expected = decide(DisciplinedPlayer(catalog=SimpleNamespace()), b)  # type: ignore[arg-type]
    assert decide(player(), b) == expected and "endgame" not in expected[1]["rule"]


def test_two_against_two_switches_when_the_chain_says_so(fake) -> None:
    fake()
    b = battle(TWO_UP, "Azumarill", TWO_FOES)
    chosen, notes = decide(player(), b)
    assert chosen == "/choose switch Slowbro" and notes["rule"] == "endgame:switch"
    assert notes["wins"] is True and notes["line"].startswith("slowbro")
    # disciplined stays: Slowbro hits Dragonite for 0.25, under its switch-in threat bar
    assert decide(DisciplinedPlayer(catalog=SimpleNamespace()), b)[0].startswith("/choose move")  # type: ignore[arg-type]


def test_one_against_one_takes_the_priority_ko(fake) -> None:
    fake({("azumarill", "dragonite", "playrough"): 0.8, ("azumarill", "dragonite", "aquajet"): 0.35})
    alone = {"Azumarill": 0.5, "Slowbro": 0.0, "Charizard": 0.0}
    b = battle(alone, "Azumarill", [("Gengar", 0), ("Garchomp", 0), ("Dragonite", 30)])
    chosen, notes = decide(player(), b)
    assert chosen == "/choose move aquajet"
    assert notes["rule"] == "endgame:stay" and notes["wins"] is True


def test_forced_replacement_sends_the_chain_s_member(fake) -> None:
    fake()
    b = battle({"Azumarill": 1.0, "Slowbro": 1.0, "Charizard": 0.0}, "Charizard", TWO_FOES)
    chosen, notes = decide(player(), b)
    assert chosen == "/choose switch Slowbro"
    assert notes["rule"] == "endgame:switch" and notes["forced"] is True


def test_an_unseen_member_is_its_worst_preview_candidate(fake) -> None:
    fake({("clefable", "slowbro", "moonblast"): 0.6, ("slowbro", "clefable", "surf"): 0.1})
    preview = ("Dragonite", "Garchomp", "Gengar", "Clefable", "Pikachu", "Eevee")
    b = battle(TWO_UP, "Azumarill", [("Garchomp", 0), ("Dragonite", 100)], preview)
    _, notes = decide(player(), b)
    assert notes["rule"].startswith("endgame:") and "worst unseen: clefable" in notes["line"]


def test_an_error_falls_back_to_disciplined_with_a_note(fake) -> None:
    fake(fail=True)
    b = battle(TWO_UP, "Azumarill", TWO_FOES)
    chosen, notes = decide(player(), b)
    assert chosen == decide(DisciplinedPlayer(catalog=SimpleNamespace()), b)[0]  # type: ignore[arg-type]
    assert notes["rule"] == "endgame:fallback" and "damage table broke" in notes["why"]


def test_factory_builds_it(config: Config, fake) -> None:
    fake()
    built = build_player(config, "endgame")
    assert isinstance(built, EndgamePlayer) and built.name == "endgame"
    assert built.endgame == EndgameParams() and built.params()["max_members"] == 2
    assert "endgame" in SIM_KINDS and "endgame" in KINDS and "endgame" not in LIVE_KINDS
