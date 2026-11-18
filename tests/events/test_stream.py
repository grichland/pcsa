"""The full LIVE 0.6 message stream: sets, orders, refusals, fallbacks, gaps, and the order messages go out in."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcsa.events import decisions as dec
from pcsa.events.gaps import final_turn, turn_gaps, visit_gaps
from pcsa.events.live_check import event_problems, message_problems, order_problems, set_problems
from pcsa.events.recorded import Recorded, load_battle
from pcsa.events.start import own_sets
from pcsa.events.stream import build_stream

STATS = {"hp": 185, "atk": 135, "def": 115, "spa": 132, "spd": 105, "spe": 169}
POINTS = {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}
MOVES = ("Draco Meteor", "Earth Power", "Power Gem", "Flamethrower")
REAL = Path(__file__).resolve().parents[2] / "output" / "runs" / "20261116-133224-run" / "battle-9"


def sheet_member(name: str, inferences: tuple[str, ...] = ()) -> dict:
    return {"name": name, "ability": "Rough Skin", "ability_status": "known", "item": "Leftovers",
            "item_status": "known", "moves": [{"slot": i, "name": m, "status": "known"} for i, m in enumerate(MOVES)],
            "stats": STATS, "stat_points": POINTS, "nature": "Timid", "stat_inferences": list(inferences)}


NAMES = ("Garchomp", "Primarina", "Mamoswine", "Gholdengo", "Golisopod", "Cinderace")
OWN_TEAM = {"sheet": {"members": [sheet_member(n) for n in NAMES]},
            "team": {"members": [{"name": n, "species": n} for n in NAMES]}}
CTX = dec.OrderContext({"garchomp": "garchomp", "primarina": "primarina"},
                       {"garchomp": ["dracometeor", "earthpower", "powergem", "flamethrower"]},
                       ["Garchomp", "Primarina", "Mamoswine"])


# sets -------------------------------------------------------------------------------------------
def test_set_has_every_read_field() -> None:
    sets, notes = own_sets(OWN_TEAM, list(NAMES))
    assert notes == [] and len(sets) == 6
    assert sets[0] == {"species": "garchomp", "item": "leftovers", "ability": "roughskin",
                       "moves": ["dracometeor", "earthpower", "powergem", "flamethrower"], "nature": "timid",
                       "points": POINTS, "stats": STATS}
    assert set_problems(sets[0]) == []


def test_points_inferred_from_the_total_are_left_out() -> None:
    team = {"sheet": {"members": [sheet_member("Garchomp", ("hp=2: inferred from the 66-point total",))]},
            "team": {"members": [{"name": "Garchomp", "species": "Garchomp"}]}}
    sets, notes = own_sets(team, ["Garchomp"])
    assert "points" not in sets[0] and "stats" in sets[0] and "inferred" in notes[0]


def test_confirmed_low_confidence_points_are_kept() -> None:
    team = {"sheet": {"members": [sheet_member("Garchomp", ("hp=2: low-confidence read confirmed by the displayed stat",))]},
            "team": {"members": [{"name": "Garchomp", "species": "Garchomp"}]}}
    assert own_sets(team, ["Garchomp"])[0][0]["points"] == POINTS


def test_no_sheet_no_sets() -> None:
    assert own_sets(None, None) == (None, ["no own_team.json sheet"])


# orders -----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("order", "out"),
    [
        ("/choose move earthquake", {"kind": "move", "move": "earthquake", "mega": False}),
        ("/choose move flamethrower mega", {"kind": "move", "move": "flamethrower", "mega": True}),
        ("/choose switch Primarina", {"kind": "switch", "mon": "primarina"}),
        ("move2", {"kind": "move", "move": "powergem", "mega": False}),
        ("mega+move0", {"kind": "move", "move": "dracometeor", "mega": True}),
        ("switch1", {"kind": "switch", "mon": "primarina"}),
    ],
)
def test_orders(order: str, out: dict) -> None:
    assert dec.order_of({"order": order}, CTX, active_hint="garchomp") == out
    assert order_problems(out) == []


def test_screen_label_uses_the_request_active() -> None:
    record = {"order": "move1", "request": {"side": {"pokemon": [{"ident": "p1: Garchomp", "active": True}]}}}
    assert dec.order_of(record, CTX)["move"] == "earthpower"


def test_unresolved_order_is_noted() -> None:
    ctx = dec.OrderContext({}, {}, [])
    assert dec.order_of({"order": "move3"}, ctx) is None and ctx.problems


def test_refusal_and_replace_retry() -> None:
    a, b = {"kind": "action"}, {"kind": "action", "same_turn": True}
    assert dec.refused_by(b, a) and not dec.refused_by(a, None)
    assert dec.refused_by({"kind": "forced", "attempt": 2}, {"kind": "forced", "attempt": 1})
    assert not dec.refused_by({"kind": "forced", "attempt": 1}, {"kind": "action"})


def test_repeat_is_a_refusal_only_where_pcsa_marked_the_move_row() -> None:
    move, switch = {"kind": "move", "move": "earthquake", "mega": False}, {"kind": "switch", "mon": "primarina"}
    refused = {"kind": "action", "same_turn": True, "refused": [0]}
    assert dec.repeat_of(refused, {"kind": "action"}, move) == "refused"
    assert dec.repeat_of({"kind": "action", "same_turn": True}, {"kind": "action"}, move) == "retry"
    assert dec.repeat_of({**refused, "refused": []}, {"kind": "action"}, switch) == "retry"
    assert dec.repeat_of({"kind": "forced", "attempt": 2}, {"kind": "forced"}, switch) == "retry"
    assert dec.repeat_of({"kind": "action"}, {"kind": "action"}, move) is None


def test_request_retry_and_refused_shapes() -> None:
    retry = dec.decision_request("b", {"sequence": 4, "kind": "action"}, None, retry=True)
    assert retry["retry"] is True and "refused" not in retry and message_problems(retry) == []
    both = {**retry, "refused": {"order": {"kind": "switch", "mon": "x"}}}
    assert message_problems(both) == ["decision_request:retry"]


def test_fallbacks() -> None:
    order = {"kind": "switch", "mon": "primarina"}
    deadline = dec.fallback_of({"sequence": 3, "degrade": "emergency", "reason": "0.4s left"}, order)
    assert deadline == {"seq": 3, "order": order, "reason": "deadline"}
    player = {"sequence": 2, "degrade": "legal_random", "notes": {"rule": "neural:fallback"}}
    assert dec.fallback_reason(player) == "player_fallback" and dec.fallback_of(player, None) is None
    assert dec.fallback_of({"degrade": "full", "notes": {"rule": "neural"}}, order) is None
    events = {"v": "0.7", "battle": "b", "type": "events", "events": []}
    assert message_problems({**events, "fallback": deadline}) == []
    assert message_problems({**events, "fallback": {**deadline, "by": "game"}}) == ["fallback:extra:by"]
    assert message_problems({**events, "fallback": {"seq": 3, "reason": "x"}}) == []  # no order: LIVE 0.6
    picks = {"picks": [4, 1, 2], "reason": "overruled"}
    assert message_problems({**events, "fallback": [picks, deadline, {"seq": 4, "reason": "x"}]}) == []  # 0.7
    assert message_problems({**events, "fallback": {**picks, "picks": [4, 4, 2]}}) == ["fallback:picks"]
    assert message_problems({**events, "fallback": []}) == ["fallback:empty"]


def test_replay_and_picks_shapes() -> None:
    request = dec.decision_request("b", {"sequence": 4, "kind": "action"}, None)
    move = {"kind": "move", "move": "surf"}  # mega may be left out (0.7)
    assert message_problems({**request, "replay": True, "played": move}) == []
    assert message_problems({**request, "replay": False}) == ["decision_request:replay"]
    assert message_problems({**request, "played": {"kind": "fly"}}) == ["played:order:kind"]
    preview = {"v": "0.7", "battle": "b", "type": "preview_request", "replay": True}
    assert message_problems(preview) == [] and message_problems({**preview, "played": {"picks": [0, 1, 2]}}) == []
    start = {"v": "0.7", "battle": "b", "type": "battle_start", "own_team": [], "opp_preview": [], "picks": [0, 1]}
    assert "battle_start:picks" in message_problems(start)


# gaps -------------------------------------------------------------------------------------------
def test_visit_gaps() -> None:
    visit = {"turn": 4, "warnings": ["Turn4: start of log is not captured", "Turn4: re-visit could not be anchored; rows appended",
                                     "x.png: low OCR confidence 0.8 at y=1: Move"]}
    assert visit_gaps(visit) == [{"k": "gap", "what": "rows", "n": 4}, {"k": "gap", "what": "rows", "n": 4}]


def test_turn_gaps() -> None:
    rows = [{"turn": 1}, {"turn": 1}, {"turn": 3}]
    audit = {"visits": [{"visit": 1, "turn": 1}, {"visit": 3, "turn": 3}]}
    decisions = [{"kind": "action", "visit": 1, "turn": 1, "log_turn": 1}, {"kind": "action", "visit": 3, "turn": 4, "log_turn": 3}]
    last = final_turn({"coverage": [{"kind": "final_turn", "turn": 4}]}, rows, decisions)
    assert last == 4
    assert turn_gaps(rows, audit, decisions, last) == {1: {"k": "gap", "what": "rows", "n": 1},
                                                       2: {"k": "gap", "what": "turn", "n": 2},
                                                       4: {"k": "gap", "what": "turn", "n": 4}}


def test_gap_shape() -> None:
    assert event_problems({"k": "gap", "what": "turn", "n": 2}) == []
    assert event_problems({"k": "gap", "what": "rows"}) == []
    assert event_problems({"k": "gap", "what": "turn"}) == ["gap:n"]
    assert event_problems({"k": "gap", "what": "late", "n": 2}) == ["value:what"]


# the stream -------------------------------------------------------------------------------------
def synthetic() -> Recorded:
    rows = [{"turn": 1, "index": 0, "text": "Go! Garchomp!"}, {"turn": 1, "index": 1, "text": "U- sent out Glimmora!"},
            {"turn": 1, "index": 2, "text": "Move: Garchomp used Earthquake!"},
            {"turn": 1, "index": 3, "text": "Fainted: Garchomp fainted!"}, {"turn": 1, "index": 4, "text": "Go! Primarina!"}]
    audit = {"visits": [{"visit": 1, "turn": 1, "appended": 2, "warnings": []},
                        {"visit": 2, "turn": 1, "appended": 3, "warnings": []},
                        {"visit": 3, "turn": 1, "appended": 0, "warnings": []}]}
    decisions = [
        {"sequence": 0, "kind": "action", "turn": 1, "visit": 1, "log_turn": 1, "order": "/choose move earthquake",
         "degrade": "full", "notes": {"rule": "neural"}},
        {"sequence": 1, "kind": "forced", "turn": 1, "order": "/choose switch Primarina", "degrade": "full",
         "attempt": 1, "pivot": False},
        {"sequence": 2, "kind": "action", "turn": 2, "visit": 2, "log_turn": 1, "order": "/choose move moonblast",
         "degrade": "full", "same_turn": False},
        {"sequence": 3, "kind": "action", "turn": 2, "visit": 3, "log_turn": 1, "order": "switch2",
         "degrade": "emergency", "reason": "0.2s left", "same_turn": True},
    ]
    meta = {"roster": list(NAMES), "team_requested": [0, 1, 2], "team_requested_names": ["Garchomp", "Primarina", "Mamoswine"],
            "outcome": "loss", "battle_id": "b-1",
            "opponent_preview": [{"species": [["Glimmora", 0.9]]}] * 6}
    return Recorded("run", "battle-1", 1, meta, rows, decisions, audit, {"coverage": [{"kind": "final_turn", "turn": 2}]},
                    OWN_TEAM, [])


def shape_of(stream) -> list:
    shape = []
    for item in stream.messages:
        msg = item["msg"]
        if msg.get("type") == "events":
            shape.append(("events", tuple(e["k"] for e in msg["events"]), "fallback" in msg))
        else:
            shape.append(msg.get("type") or ("picks" if "picks" in msg else "order"))
    return shape


def test_stream_order() -> None:
    stream = build_stream(synthetic())
    assert shape_of(stream) == [
        "battle_start", "preview_request", "picks",
        ("events", ("turn", "enter", "enter"), False), "decision_request", "order",
        ("events", ("move", "faint"), False), "decision_request", "order",  # the replace after its faint row
        ("events", ("enter", "turn"), False), "decision_request", "order",
        "decision_request",  # a repeated tap (retry); emergency: no CG answer
        ("events", ("gap", "end"), True), "battle_end",
    ]
    assert stream.messages[7]["msg"]["kind"] == "replace"
    retry = stream.messages[12]["msg"]
    assert retry["retry"] is True and "refused" not in retry
    last_events = stream.messages[-2]["msg"]
    assert last_events["fallback"] == {"seq": 3, "order": {"kind": "switch", "mon": "mamoswine"}, "reason": "deadline"}
    assert last_events["events"][0] == {"k": "gap", "what": "turn", "n": 2}
    assert stream.row_ids == ["T1#0", "T1#1", "T1#2", "T1#3", "T1#4"]
    assert stream.counts["replace_placed:after_trigger"] == 1
    assert [p for m in stream.messages if m["from"] == "controller" for p in message_problems(m["msg"])] == []


def test_a_marked_refusal_sends_refused() -> None:
    rec = synthetic()
    rec.decisions[3]["refused"] = [1]
    stream = build_stream(rec)
    assert ("events", ("refused",), False) in shape_of(stream)
    request = stream.messages[13]["msg"]
    assert request["refused"] == {"order": {"kind": "move", "move": "moonblast", "mega": False}}
    assert "retry" not in request


def test_replace_without_its_row_goes_at_the_end_of_its_turn() -> None:
    rec = synthetic()
    rec.rows = [r for r in rec.rows if "Fainted" not in r["text"]]
    rec.audit["visits"][1]["appended"] = 2
    stream = build_stream(rec)
    assert stream.counts["replace_placed:turn_end"] == 1
    assert shape_of(stream)[6:9] == [("events", ("move", "enter"), False), "decision_request", "order"]


def test_game_timeout_is_counted_not_sent() -> None:
    rec = synthetic()
    rec.run_events = [{"event": "turn_timed_out", "ts": "2", "notice": "The server did not receive your input"}]
    stream = build_stream(rec)
    assert stream.counts["fallback_not_sent:game_timeout"] == 1
    assert all(m["msg"]["fallback"]["reason"] != "game_timeout" for m in stream.messages if "fallback" in m["msg"])


@pytest.mark.skipif(not REAL.exists(), reason="pcsa recordings not present")
def test_real_battle() -> None:
    rec = load_battle(REAL)
    stream = build_stream(rec)
    types = [m["msg"].get("type") for m in stream.messages if m["from"] == "controller"]
    assert types[0] == "battle_start" and types[-1] == "battle_end"
    assert types.count("decision_request") == len(rec.decisions)
    assert len(stream.row_ids) == len(rec.rows)
    start = stream.messages[0]["msg"]
    assert all(set_problems(s) == [] for s in start["own_team"]) and len(start["own_team"]) == 6
    problems = {p for m in stream.messages if m["from"] == "controller" for p in message_problems(m["msg"])}
    assert problems <= {"ability:missing:side", "item:missing:side"}  # CT-1: a pill not read


def test_back_paged_turn_has_no_gap() -> None:
    # turn 2 had no action screen of ours; visit 3 paged back to it before reading turn 3
    rows = [{"turn": 1}, {"turn": 2, "back_paged": True}, {"turn": 3}]
    audit = {"visits": [{"visit": 1, "turn": 1}, {"visit": 3, "turn": 2, "page": "back"}, {"visit": 3, "turn": 3}]}
    decisions = [{"kind": "action", "visit": 1, "turn": 2, "log_turn": 1},
                 {"kind": "action", "visit": 3, "turn": 4, "log_turn": 3}]
    assert turn_gaps(rows, audit, decisions, 4) == {4: {"k": "gap", "what": "turn", "n": 4}}
