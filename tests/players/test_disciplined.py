"""disciplined: the sweeper denier, Mega first, disciplined switching, the recovery threshold, the phase rule.

The threat estimates are a table here (FakeThreat): who hits whom for what share of max HP, who is faster.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from pcsa.battle.state import replay
from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config
from pcsa.players import disciplined, mirror
from pcsa.players.base import DecisionView, Degrade, ScreenFacts, TeamPreviewView
from pcsa.players.disciplined import DisciplinedParams, DisciplinedPlayer
from pcsa.players.factory import KINDS, LIVE_KINDS, SIM_KINDS, build_player
from pcsa.players.max_damage import MaxDamagePlayer

PRELUDE = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|teamsize|p1|3", "|teamsize|p2|3", "|start"]
TEAM = {
    "Azumarill": (175, ["playrough", "aquajet"], ""),
    "Slowbro": (170, ["surf", "slackoff", "haze"], ""),
    "Charizard": (153, ["airslash", "flamethrower"], "charizarditex"),
}
FOE = "Dragonite"
BOOST = ["|-boost|p2a: Dragonite|atk|1"]
FOE_DOWN = ["|faint|p2a: Garchomp", "|switch|p2a: Dragonite|Dragonite, L50|100/100"]


class FakeThreat:
    """`hits[(attacker, defender)] = (share of the defender's max HP, move id)`; `faster`: who outspeeds whom."""

    def __init__(
        self, hits: dict, faster: set | None = None, priority: dict | None = None, move_hits: dict | None = None
    ) -> None:
        self.hits, self.faster, self.priority = hits, faster or set(), priority or {}
        self.move_hits = move_hits or {}  # (attacker, defender, move id) -> share, for the legal-move checks

    def hit_fraction(self, battle, attacker, defender, move, catalog, tracker=None):
        key = (attacker.species, defender.species, move.id)
        return self.move_hits.get(key, self.best_hit(battle, attacker, defender, catalog)[0])

    def best_hit(self, battle, attacker, defender, catalog, tracker=None):
        return self.hits.get((attacker.species, defender.species), (0.1, None))

    def can_ko(self, battle, attacker, defender, catalog, tracker=None):
        return self.best_hit(battle, attacker, defender, catalog)[0] >= defender.current_hp_fraction

    def outspeeds(self, battle, a, b, catalog, tracker=None):
        return (a.species, b.species) in self.faster

    def threatened(self, battle, mine, theirs, catalog, tracker=None, threshold=0.4):
        return self.best_hit(battle, theirs, mine, catalog)[0] >= threshold

    def priority_ko(self, battle, attacker, defender, catalog, tracker=None):
        return self.priority.get((attacker.species, defender.species))


class FakeTracker:
    def __init__(self, catalog=None) -> None:
        self.observed, self.resets = 0, 0

    def reset(self) -> None:
        self.resets += 1

    def observe(self, battle) -> None:
        self.observed += 1


# The base table: Dragonite threatens Azumarill (0.6) which cannot KO it; Slowbro takes 0.2 and hits back 0.3.
HITS = {
    ("dragonite", "azumarill"): (0.6, "outrage"),
    ("azumarill", "dragonite"): (0.5, "playrough"),
    ("dragonite", "slowbro"): (0.2, "outrage"),
    ("slowbro", "dragonite"): (0.3, "surf"),
    ("dragonite", "charizard"): (0.5, "outrage"),
    ("charizard", "dragonite"): (0.2, "airslash"),
}


@pytest.fixture
def fake(monkeypatch):
    def install(hits: dict | None = None, **kwargs) -> FakeThreat:
        estimates = FakeThreat({**HITS, **(hits or {})}, **kwargs)
        monkeypatch.setattr(disciplined, "threat", estimates)
        monkeypatch.setattr(disciplined, "opponent", SimpleNamespace(Tracker=FakeTracker))
        return estimates

    return install


def battle(ours: str, turns: list[list[str]] | None = None, *, hp: float = 1.0, foe: str = FOE, mega: bool = False):
    """`ours` leads against `foe`; `turns` holds each finished turn's events; our active is at `hp`."""
    lines = [*PRELUDE, f"|switch|p1a: {ours}|{ours}, L50|{TEAM[ours][0]}/{TEAM[ours][0]}"]
    lines += [f"|switch|p2a: {foe}|{foe}, L50|100/100", "|turn|1"]
    for turn, events in enumerate(turns or [], start=2):
        lines += [*events, f"|turn|{turn}"]
    active = [line.split("|")[2][5:] for line in lines if line.startswith("|switch|p1a")][-1]
    pokemon = [
        {
            "ident": f"p1: {name}",
            "details": f"{name}, L50",
            "condition": f"{round(full * hp) if name == active else full}/{full}",
            "active": name == active,
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100},
            "moves": moves,
            "baseAbility": "",
            "item": item,
        }
        for name, (full, moves, item) in TEAM.items()
    ]
    request: dict = {"side": {"name": "me", "id": "p1", "pokemon": pokemon}, "rqid": 1}
    moves: dict = {
        "moves": [{"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal"} for m in TEAM[active][1]]
    }
    request["active"] = [{**moves, "canMegaEvo": True} if mega else moves]
    return replay(lines, request)


def decide(player, b) -> tuple[str, dict]:
    view = DecisionView(mirror.legal_actions(b), ScreenFacts(), Degrade.FULL, random.Random(0), b)
    return player.choose_action(view, Deadline.after(FakeClock(), 30.0)).message, player.notes()


def player() -> DisciplinedPlayer:
    return DisciplinedPlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]


# Sweeper denier -------------------------------------------------------------------------------------------
def test_unboosted_threat_switches_but_a_boosted_one_is_met_with_a_priority_ko(fake) -> None:
    fake(priority={("azumarill", "dragonite"): "aquajet"})
    chosen, notes = decide(player(), battle("Azumarill"))
    assert chosen == "/choose switch Slowbro" and notes["rule"] == "switch"
    chosen, notes = decide(player(), battle("Azumarill", [BOOST]))
    assert chosen == "/choose move aquajet" and notes["rule"] == "denier:priority_ko"


def test_a_used_setup_move_triggers_the_denier_without_a_boost(fake) -> None:
    fake(priority={("azumarill", "dragonite"): "aquajet"})
    used = ["|move|p2a: Dragonite|Dragon Dance|p2a: Dragonite", "|-clearallboost"]
    chosen, notes = decide(player(), battle("Azumarill", [used]))
    assert chosen == "/choose move aquajet" and notes["rule"] == "denier:priority_ko"


def test_denier_prefers_a_ko_now_then_haze(fake) -> None:
    fake({("azumarill", "dragonite"): (1.0, "playrough")}, faster={("azumarill", "dragonite")})
    assert decide(player(), battle("Azumarill", [BOOST])) == ("/choose move playrough", {"rule": "denier:ko"})
    fake({("azumarill", "dragonite"): (1.0, "playrough"), ("dragonite", "azumarill"): (1.0, "outrage")})
    chosen, notes = decide(player(), battle("Azumarill", [BOOST]))  # slower and KOed first: no KO now
    assert chosen.startswith("/choose move") and notes["rule"] == "denier:hardest_hit"
    fake({("dragonite", "slowbro"): (0.7, "outrage")})
    assert decide(player(), battle("Slowbro", [BOOST])) == ("/choose move haze", {"rule": "denier:anti_setup"})


def test_denier_switches_only_to_a_member_that_takes_little_and_kos(fake) -> None:
    fake({("slowbro", "dragonite"): (1.0, "surf")})
    chosen, notes = decide(player(), battle("Azumarill", [BOOST]))
    assert chosen == "/choose switch Slowbro" and notes["rule"] == "denier:switch"


# Disciplined switching ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("change", "blocked"),
    [
        ({("dragonite", "azumarill"): (0.3, "outrage")}, False),  # not threatened
        ({("azumarill", "dragonite"): (1.0, "playrough")}, False),  # we KO it
        ({("dragonite", "slowbro"): (0.3, "outrage")}, True),  # the switch-in takes too much
        ({("slowbro", "dragonite"): (0.2, "surf")}, True),  # the switch-in does not threaten
    ],
)
def test_a_voluntary_switch_needs_all_four_conditions(fake, change: dict, blocked: bool) -> None:
    fake(change)
    chosen, notes = decide(player(), battle("Azumarill"))
    assert chosen.startswith("/choose move") and notes["rule"] == "attack"
    assert ("switch_blocked" in notes) == blocked


def test_switch_in_damage_is_a_share_of_current_hp(fake) -> None:
    fake()
    b = battle("Azumarill")
    b.team["p1: Slowbro"]._current_hp = 85  # half HP: 0.2 of max is 0.4 of what is left
    chosen, notes = decide(player(), b)
    assert chosen.startswith("/choose move") and "switch_blocked" in notes


def test_back_to_back_switch_blocked_when_the_opponent_stayed(fake) -> None:
    fake({("dragonite", "slowbro"): (0.6, "outrage"), ("dragonite", "azumarill"): (0.2, "outrage")})
    me = player()
    assert decide(me, battle("Slowbro"))[0] == "/choose switch Azumarill"
    stayed = ["|switch|p1a: Azumarill|Azumarill, L50|175/175", "|move|p2a: Dragonite|Outrage|p1a: Azumarill"]
    fake({("dragonite", "slowbro"): (0.2, "outrage")})  # now Azumarill is threatened and Slowbro is safe
    chosen, notes = decide(me, battle("Slowbro", [stayed]))
    assert chosen.startswith("/choose move") and "the opponent did not" in notes["switch_blocked"]


# Recovery and the phase rule ------------------------------------------------------------------------------
def test_recovery_only_below_the_threshold(fake) -> None:
    fake()
    assert decide(player(), battle("Slowbro", hp=0.5)) == ("/choose move slackoff", {"rule": "recover"})
    assert decide(player(), battle("Slowbro", hp=0.7)) == ("/choose move surf", {"rule": "attack"})
    fake({("dragonite", "slowbro"): (1.0, "outrage")})  # still KOed from full HP: attack
    assert decide(player(), battle("Slowbro", hp=0.5))[0] == "/choose move surf"


def test_ahead_in_members_no_recovery_but_the_denier_s_haze(fake) -> None:
    fake()
    ahead = battle("Slowbro", [FOE_DOWN], hp=0.5, foe="Garchomp")
    chosen, notes = decide(player(), ahead)
    assert chosen == "/choose move surf" and "ahead" in notes["recover_blocked"]
    fake({("dragonite", "slowbro"): (0.7, "outrage")})
    boosted = battle("Slowbro", [FOE_DOWN, BOOST], hp=0.5, foe="Garchomp")
    assert decide(player(), boosted) == ("/choose move haze", {"rule": "denier:anti_setup"})


# Mega, bookkeeping, factory -------------------------------------------------------------------------------
def test_mega_is_taken_the_first_turn_it_is_offered(fake) -> None:
    fake({("dragonite", "charizard"): (0.2, "outrage")})
    b = battle("Charizard", mega=True)
    chosen, notes = decide(player(), b)
    assert chosen.endswith(" mega") and notes["mega"] == "first turn offered"
    # max_damage scores Air Slash higher before Mega X drops the Flying type, and so would hold the Mega
    assert decide(MaxDamagePlayer(), b)[0] == "/choose move airslash"


def test_the_tracker_observes_every_decision_and_resets_each_game(fake) -> None:
    fake()
    me = player()
    decide(me, battle("Azumarill"))
    decide(me, battle("Azumarill", [[]]))
    assert me.tracker.observed == 2  # type: ignore[attr-defined]
    resets = me.tracker.resets  # type: ignore[attr-defined]
    me.choose_team(TeamPreviewView(None, (None,) * 6, random.Random(0)), Deadline.after(FakeClock(), 30.0))
    assert me.tracker.resets == resets + 1  # type: ignore[attr-defined]


def test_factory_builds_it_with_the_config_thresholds(config: Config, fake) -> None:
    fake()
    built = build_player(config, "disciplined")
    assert isinstance(built, DisciplinedPlayer) and built.name == "disciplined"
    assert built.rules == DisciplinedParams()  # the yaml holds the defaults
    assert built.params()["recover_below"] == 0.6
    assert "disciplined" in LIVE_KINDS and "disciplined" in KINDS and "disciplined" not in SIM_KINDS  # promoted 8 Nov


# Futile attack and bad trade --------------------------------------------------------------------------------
def test_a_choice_lock_into_a_useless_move_switches_to_the_hardest_hitter(fake) -> None:
    fake(move_hits={("azumarill", "dragonite", "aquajet"): 0.02})
    b = battle("Azumarill")
    b._available_moves = [m for m in b.available_moves if m.id == "aquajet"]  # locked into Aqua Jet
    chosen, notes = decide(player(), b)
    assert chosen == "/choose switch Slowbro" and notes["rule"] == "switch"
    assert notes["switch"].startswith("futile:")


def test_a_bad_trade_switches_to_the_member_taking_least(fake) -> None:
    # Azumarill hits for 0.1 and takes 0.6; Slowbro hits only 0.1 (no disciplined switch) but takes 0.2 (< 0.6).
    fake({("azumarill", "dragonite"): (0.1, "playrough"), ("slowbro", "dragonite"): (0.1, "surf")})
    chosen, notes = decide(player(), battle("Azumarill"))
    assert chosen == "/choose switch Slowbro" and notes["rule"] == "switch"
    assert notes["switch"].startswith("bad trade:")


def test_a_fair_trade_keeps_attacking(fake) -> None:
    fake({("azumarill", "dragonite"): (0.2, "playrough"), ("slowbro", "dragonite"): (0.1, "surf")})
    chosen, notes = decide(player(), battle("Azumarill"))
    assert chosen == "/choose move playrough" and notes["rule"] == "attack"
