"""The battle's growing LIVE event list from a BattleLog: turns, rows once each, gaps, unread rows, rebuilds."""

from __future__ import annotations

from dataclasses import replace

from pcsa.decide.api import Hints, MoveOffer, Offer, Order, OwnTeam, Request
from pcsa.decide.runner import Decision
from pcsa.events.live_check import message_problems, problems
from pcsa.live.battle_events import BattleEvents, preview_ids, set_of
from pcsa.live.decision_record import decision_request, history_entry, live_messages
from pcsa.phone.log_merge import BattleLog
from pcsa.run.battle_state import BattleRecord
from tests.live.turn_rig import PREVIEW, TEAM


def battle() -> BattleRecord:
    return BattleRecord(1, "b-1", TEAM.names, BattleLog("b-1"), requested=[4, 1, 2], opponent_preview=PREVIEW)


def read(b: BattleRecord, turn: int, rows: list[str], warnings=(), sides=None) -> None:
    b.log.turns.setdefault(turn, []).extend(rows)
    b.log.row_notes[turn] = [(None, s, ()) for s in (sides or [None] * len(b.log.turns[turn]))]
    b.log.visits.append({"visit": len(b.log.visits) + 1, "turn": turn, "appended": len(rows),
                         "warnings": list(warnings)})  # fmt: skip


def test_battle_start_sets_and_preview() -> None:
    events = BattleEvents("b-1", TEAM, PREVIEW)
    start = events.all()[0]
    assert start["type"] == "battle_start" and start["opp_preview"] == ["glimmora", "lucario", "garchomp"]
    assert start["own_team"][4] == set_of(TEAM.members[4])
    assert start["own_team"][4]["moves"] == ["iceshard", "earthquake", "iciclespear", "endeavor"]
    assert start["own_team"][4]["item"] == "focussash" and "points" not in start["own_team"][4]
    assert preview_ids([{"top": []}]) == (None,)


def test_rows_once_each_turns_and_a_gap_for_a_turn_never_read() -> None:
    b, events = battle(), BattleEvents("b-1", TEAM, PREVIEW)
    read(b, 1, ["Go! Mamoswine!", "U- sent out Glimmora!"])
    assert [e["k"] for e in events.update(b)] == ["turn", "enter", "enter"]
    assert events.update(b) == []
    events.to_turn(2, b)
    read(b, 1, ["Move: Mamoswine used Earthquake!"])
    read(b, 3, ["Move: Mamoswine used Ice Shard!"])
    new = events.update(b)
    assert [e["k"] for e in new] == ["move", "gap", "turn", "move"]  # turn 1's last row late; turn 2 never read
    assert new[1] == {"k": "gap", "what": "turn", "n": 2}
    assert events.late_rows == 1 and events.quality(b) == "gaps"
    assert [r.row_id for r in events.rows(b)] == ["1-0", "1-1", "1-2", "3-0"]
    assert events.rows(b)[2].kind == "move"
    assert not [p for e in events.events for p in problems(e)]


def test_visit_warnings_become_gaps_after_their_turn() -> None:
    b, events = battle(), BattleEvents("b-1", TEAM, PREVIEW)
    read(b, 1, ["Go! Mamoswine!"], warnings=["start of log is not captured"])
    assert events.update(b) == [{"k": "turn", "n": 1}, {"k": "gap", "what": "rows", "n": 1},
                                {"k": "enter", "side": "own", "mon": "mamoswine"}]  # fmt: skip


def test_pill_side_and_unread_rows() -> None:
    b, events = battle(), BattleEvents("b-1", TEAM, PREVIEW)
    read(b, 1, ["Go! Mamoswine!", "Held Item: Glimmora’s Focus Sash", "Until Battle Ends"],
         sides=[None, "opp", None])  # fmt: skip
    new = events.update(b)
    assert new[2] == {"k": "item", "side": "opp", "mon": "glimmora", "item": "focussash", "what": "activated"}
    assert new[3] == {"k": "unread", "text": "Until Battle Ends"}


def test_a_rebuild_gives_the_same_rows() -> None:
    b, first = battle(), BattleEvents("b-1", TEAM, PREVIEW)
    read(b, 1, ["Go! Mamoswine!"])
    read(b, 2, ["Move: Mamoswine used Earthquake!"])
    first.update(b)
    again = BattleEvents("b-1", TEAM, PREVIEW)
    again.update(b)
    assert again.all() == first.all()
    first.refused({"kind": "move", "move": "earthquake", "mega": False})
    first.end("win")
    assert [e["k"] for e in first.events[-2:]] == ["refused", "end"]


def request(events: BattleEvents, seq: int, read: bool = False) -> Request:
    offer = Offer((MoveOffer(0, "earthquake"), MoveOffer(1, "iceshard")), read, (), None)
    return Request("b-1", seq, (), "full", OwnTeam("", ()), (), offer, Hints(), 0.0, events.all())


def test_live_request_refused_retry_and_fallback() -> None:
    b, events = battle(), BattleEvents("b-1", TEAM, PREVIEW)
    b.battle_id = "b-live"
    read(b, 1, ["Go! Mamoswine!"])
    events.update(b)
    player = Decision(1, Order.move(0), "/choose move earthquake", "player", "")
    first = live_messages("action", request(events, 1), player, events.converter)
    assert first == {"decision_request": {"v": "0.7", "battle": "b-1", "type": "decision_request", "seq": 1,
                                          "kind": "turn"},
                     "played": {"kind": "move", "move": "earthquake", "mega": False}}  # fmt: skip
    refused = {"kind": "move", "move": "earthquake", "mega": False}
    events.refused(refused)
    second_request = request(events, 2, read=True)
    second_request = replace(second_request, live=decision_request(second_request, "action", refused=refused,
                                                                   retry=True))  # fmt: skip
    second = live_messages("action", second_request, player, events.converter)
    assert second["decision_request"]["refused"] == {"order": refused} and "retry" not in second["decision_request"]
    assert second["decision_request"]["screen_offer"] == {"moves": ["earthquake", "iceshard"]}
    builtin = Decision(3, Order.switch("Garchomp"), "/choose switch Garchomp", "builtin", "no answer")
    third_request = request(events, 3)
    third_request = replace(third_request, live=decision_request(third_request, "forced", retry=True))
    third = live_messages("forced", third_request, builtin, events.converter)
    assert third["decision_request"]["retry"] is True and "refused" not in third["decision_request"]
    assert third["decision_request"]["kind"] == "replace"
    assert third["fallback"] == {
        "seq": 3,
        "order": {"kind": "switch", "mon": "garchomp"},
        "reason": "builtin: no answer",
    }
    for live in (first, second, third):
        assert message_problems(live["decision_request"]) == []
    fallback = {"v": "0.7", "battle": "b-1", "type": "events", "events": [], "fallback": third["fallback"]}
    assert message_problems(fallback) == []
    record = {"live": third}
    assert history_entry(record, third_request, builtin) == {"events": len(third_request.events), **third}
    automatic = Decision(4, Order.move(0), "/choose move struggle", "automatic", "automatic turn")
    assert history_entry({"live": first}, request(events, 4), automatic) is None
    assert "played" not in live_messages("action", request(events, 4), automatic, events.converter)
    assert third["played"] == third["fallback"]["order"]
    unnamed = replace(request(events, 5), offer=Offer((MoveOffer(0, None),), False, (), None))
    blind = Decision(5, Order.move(0), "/choose move 1", "screen", "late")
    fifth = live_messages("action", unnamed, blind, events.converter)
    assert fifth["fallback"] == {"seq": 5, "reason": "screen: late"}  # no order: the row had no id (LIVE 0.6)
    assert message_problems({**fallback, "fallback": fifth["fallback"]}) == []
