"""The stream bookkeeping: each event once, the fallbacks on the next events message, a whole resend for a new CG
(picks in battle_start, replayed requests with what was played), the preview fallback."""

from __future__ import annotations

import pytest

from pcsa.cg.stream import Stream
from pcsa.cg.transport import CGError
from pcsa.events.live_check import message_problems
from tests.cg.helpers import START, TAG, TURN1, TURN2, entry, request

FALLBACK = {"seq": 0, "order": {"kind": "move", "move": "makeitrain", "mega": False}, "reason": "builtin: late"}


def kinds(messages: list[dict]) -> list[str]:
    return [m["type"] + (f":{m['seq']}" if "seq" in m else "") for m in messages]


def test_preview_then_decisions_send_each_event_once() -> None:
    stream = Stream(TAG)
    first = stream.preview(START, 45000)
    assert first == [START, {"v": "0.7", "battle": TAG, "type": "preview_request", "deadline_ms": 45000}]
    assert stream.preview(START, 45000) == []
    r0 = request(0, TURN1)
    sent = stream.decision(r0, r0.live)
    assert kinds(sent) == ["events", "decision_request:0"] and sent[0]["events"] == list(TURN1)
    assert stream.decision(r0, r0.live) == []  # asked again: nothing twice
    r1 = request(1, TURN1 + TURN2, history=(entry(r0),))
    sent = stream.decision(r1, r1.live)
    assert kinds(sent) == ["events", "decision_request:1"] and sent[0]["events"] == list(TURN2)
    assert "fallback" not in sent[0]
    assert all(message_problems(m) == [] for m in first + sent)


def test_the_fallback_rides_on_the_next_events_message_once() -> None:
    stream = Stream(TAG)
    stream.preview(START, None)
    r0 = request(0, TURN1)
    stream.decision(r0, r0.live)
    r1 = request(1, TURN1, history=(entry(r0, FALLBACK),), offer=None)
    sent = stream.decision(r1, r1.live)
    assert sent[0] == {"v": "0.7", "battle": TAG, "type": "events", "events": [], "fallback": FALLBACK}
    assert message_problems(sent[0]) == []
    r2 = request(2, TURN1 + TURN2, history=(entry(r0, FALLBACK), entry(r1)))
    sent = stream.decision(r2, r2.live)
    assert kinds(sent) == ["events", "decision_request:2"] and "fallback" not in sent[0]


def test_a_new_cg_gets_the_whole_stream_in_place() -> None:
    r0 = request(0, TURN1[:1])
    r1 = request(1, TURN1, history=(entry(r0, FALLBACK, FALLBACK["order"]),))
    played = {"kind": "switch", "mon": "garchomp"}
    r2 = request(2, TURN1 + TURN2, history=(entry(r0, FALLBACK, FALLBACK["order"]), entry(r1, None, played)),
                 retry=True, picks=(3, 1, 2))  # fmt: skip
    sent = Stream(TAG).decision(r2, r2.live)
    assert kinds(sent) == ["battle_start", "preview_request", "events", "decision_request:0", "events",
                           "decision_request:1", "events", "decision_request:2"]  # fmt: skip
    assert sent[0] == {**START, "picks": [3, 1, 2]} and sent[1]["replay"] is True
    assert sent[2]["events"] == list(TURN1[:1]) and "fallback" not in sent[2]
    assert sent[3] == {**r0.live, "replay": True, "played": FALLBACK["order"]}
    assert sent[4]["events"] == list(TURN1[1:]) and sent[4]["fallback"] == FALLBACK
    assert sent[5] == {**r1.live, "replay": True, "played": played}
    assert sent[6]["events"] == list(TURN2) and sent[7]["retry"] is True and "replay" not in sent[7]
    events = [e for m in sent if m["type"] == "events" for e in m["events"]]
    assert events == list(TURN1 + TURN2)
    assert all(message_problems(m) == [] for m in sent)


def test_a_replay_without_played_has_no_order() -> None:
    r0 = request(0, TURN1)
    r1 = request(1, TURN1 + TURN2, history=(entry(r0),))
    sent = Stream(TAG).decision(r1, r1.live)
    assert sent[3] == {**r0.live, "replay": True}


def test_two_fallbacks_go_on_one_message_before_its_events() -> None:
    stream = Stream(TAG)
    r0 = request(0, TURN1)
    second = {"seq": 1, "reason": "builtin: late"}
    r2 = request(2, TURN1 + TURN2, history=(entry(r0, FALLBACK), entry(request(1, TURN1), second)))
    stream.decision(r0, r0.live)
    stream.seq = 1  # seq 1 went out, its answer was late
    sent = stream.decision(r2, r2.live)
    assert kinds(sent) == ["events", "decision_request:2"]
    assert sent[0]["fallback"] == [FALLBACK, second] and sent[0]["events"] == list(TURN2)
    assert message_problems(sent[0]) == []


def test_the_preview_fallback_when_the_team_brought_is_not_cgs() -> None:
    stream = Stream(TAG)
    stream.preview(START, None)
    stream.picks = (0, 1, 2)  # CG's answer
    r0 = request(0, TURN1, picks=(3, 1, 2))
    sent = stream.decision(r0, r0.live)
    assert sent[0]["fallback"] == {"picks": [3, 1, 2], "reason": "overruled"} and message_problems(sent[0]) == []
    r1 = request(1, TURN1 + TURN2, history=(entry(r0),), picks=(3, 1, 2))
    assert "fallback" not in stream.decision(r1, r1.live)[0]
    same = Stream(TAG)
    same.preview(START, None)
    same.picks = (0, 1, 2)
    assert "fallback" not in same.decision(request(0, TURN1), request(0, TURN1).live)[0]
    unanswered = Stream(TAG)
    unanswered.preview(START, None)  # CG's answer never came: the runner picked
    sent = unanswered.decision(request(0, TURN1), request(0, TURN1).live)
    assert sent[0]["fallback"] == {"picks": [0, 1, 2], "reason": "no preview answer from CG"}


def test_without_battle_start_nothing_is_sent() -> None:
    with pytest.raises(CGError):
        Stream(TAG).preview(None, None)


def test_a_changed_list_is_seen() -> None:
    stream = Stream(TAG)
    r1 = request(1, TURN1)
    stream.decision(r1, r1.live)
    assert not stream.changed(request(2, TURN1 + TURN2))
    assert stream.changed(request(2, TURN1[:2] + TURN2))
