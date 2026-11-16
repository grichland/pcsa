"""15 Nov 2026 (run 20261115-034548, battles 31-32): a communication error during team selection came back to the same
selection screen with 2 seconds left (frame 02036); the session took it for a new battle (31 ended "unknown"), the
lineup taps came too late and the game chose Gholdengo, Garchomp and Golisopod itself (frame 02037), while every
decision of battle 32 was made for the requested Mamoswine, Cinderace and Primarina.
Fixture: tests/fixtures/live_20261115/battle32_team.json (the roster, the request, the log rows, the party read)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pcsa.common import paths
from pcsa.common.types import Flag, Observation, Screen
from pcsa.live import handlers
from pcsa.live.session import Session
from pcsa.live.team_check import check_team
from pcsa.phone.log_merge import BattleLog
from pcsa.run.battle_state import BattleRecord

FIX = json.loads(
    (paths.repo_root() / "tests" / "fixtures" / "live_20261115" / "battle32_team.json").read_text(encoding="utf-8")
)
SIZE = (2340, 1080)


def _record(auto: bool) -> BattleRecord:
    b = BattleRecord(32, "id", tuple(FIX["roster"]), BattleLog("id"), requested=list(FIX["requested"]))
    b.team_submitted, b.team_auto = True, auto
    b.log.turns[1] = list(FIX["log_rows"]["1"])
    b.observed["lead"] = FIX["first_action_active_name"]
    return b


def test_the_automatic_lineup_is_played_from_the_first_action_screen() -> None:
    b = _record(auto=True)
    assert check_team(b, active=FIX["first_action_active_name"])
    assert list(b.names) == ["Gholdengo", "Garchomp", "Golisopod"]  # the first three roster rows, Gholdengo led
    assert b.slot_of("Garchomp") == 1 and b.visible_order == [0, 1, 2]
    assert not check_team(b, party=FIX["party_names"])  # the party screen agrees: nothing more to change
    summary = b.summary(0, True)
    assert summary["team_requested_names"] == ["Mamoswine", "Cinderace", "Primarina"]
    assert summary["team_played_names"] == ["Gholdengo", "Garchomp", "Golisopod"]
    assert b.failures and "team mismatch" in b.failures[0]


def test_without_the_notice_the_guess_is_corrected_by_the_party_rows() -> None:
    b = _record(auto=False)
    assert check_team(b, active="Gholdengo")  # seen: Gholdengo (screen), Golisopod (log); the third from the request
    assert b.names[0] == "Gholdengo" and set(b.names) == {"Gholdengo", "Golisopod", "Mamoswine"}
    assert check_team(b, party=FIX["party_names"])
    assert set(b.names) == {"Gholdengo", "Garchomp", "Golisopod"} and b.names[0] == "Gholdengo"
    assert b.requested_at_preview == FIX["requested"]


def test_the_requested_team_is_left_alone() -> None:
    b = _record(auto=False)
    b.requested = [0, 2, 1]
    assert not check_team(b, active="Gholdengo", party=FIX["party_names"])
    assert b.requested == [0, 2, 1] and b.requested_at_preview is None


def _notice(b: BattleRecord) -> None:
    obs = Observation(Screen.WAIT, SIZE, frozenset({Flag.AUTO_NOTICE}), {"notice": FIX["notice"]})
    Session.flags(SimpleNamespace(battle=b), obs)  # type: ignore[arg-type]


def test_a_timeout_notice_before_any_turn_marks_the_lineup_automatic() -> None:
    b = _record(auto=False)
    _notice(b)
    assert b.team_auto
    later = _record(auto=False)
    later.action_visits = 3  # a move timer running out is not the lineup
    _notice(later)
    assert not later.team_auto


@pytest.mark.parametrize(("shown", "same"), [(FIX["opponent_preview"], True), (["Garchomp"] * 6, False)])
def test_the_selection_back_after_a_disconnect_is_the_same_battle(monkeypatch, shown: list[str], same: bool) -> None:
    b = _record(auto=False)
    b.opponent_preview = tuple(SimpleNamespace(name=n) for n in FIX["opponent_preview"])
    calls: list[str] = []

    def new_battle(s, roster):
        calls.append("new")
        s.battle = SimpleNamespace(team_submitted=True)

    monkeypatch.setattr(handlers, "to_reference", lambda image, viewport: image)
    monkeypatch.setattr(handlers, "match_team_preview", lambda image: [SimpleNamespace(name=n) for n in shown])
    monkeypatch.setattr(handlers, "new_battle", new_battle)
    monkeypatch.setattr(handlers, "repair_team_selection", lambda s, obs, battle: calls.append("repair"))
    s = SimpleNamespace(
        battle=b,
        previous=Observation(Screen.COMM_ERROR, SIZE),
        phone=SimpleNamespace(profile=SimpleNamespace(viewport=None)),
    )
    obs = Observation(Screen.TEAM_SELECT, SIZE, fields={"timer": "00:02", "markers": {}, "roster": FIX["roster"]})
    handlers.on_team_select(s, obs, SimpleNamespace(image=None), 0.0)
    assert calls == (["repair"] if same else ["new", "repair"])
