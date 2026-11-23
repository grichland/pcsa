"""Full battles end to end on the fake device (real Session, TurnPipeline, Runner, built-in Player): one battle,
restarts, forfeit and unread results, a battle already on at start."""

from __future__ import annotations

from pathlib import Path

from pcsa.common.config import Config
from pcsa.common.types import MoveInfo, Screen
from pcsa.decide.api import Order
from pcsa.live.fake import Step
from tests.live.battle_rig import (
    Script,
    action,
    end_of_battle,
    forced,
    live_problems,
    lobby_to_team_select,
    party,
    peeked_action,
    row_events,
    run_battles,
)

START = ["Kai sent out Glimmora!", "Go! Gholdengo!"]
TURN1 = [
    "Move: Gholdengo used Make It Rain!",
    "HP Lost: Glimmora’s HP: 100% → 40%",
    "Move: The opposing Glimmora used Earth Power!",
    "HP Lost: Gholdengo’s HP: 150 → 60",
    "It’s super effective!",
]
TURN2 = [
    "Move: The opposing Glimmora used Earth Power!",
    "HP Lost: Gholdengo’s HP: 60 → 0",
    "It’s super effective!",
    "Fainted: Gholdengo fainted!",
    "Go! Garchomp!",
]
VISITS = [{1: START}, {1: START + TURN1}, {1: START + TURN1, 2: TURN2}]
CHOMP_MOVES = tuple(MoveInfo(k, n, (16, 16)) for k, n in enumerate(("Earthquake", "Dragon Claw", "Swords Dance",
                                                                    "Protect")))  # fmt: skip
FAINT = forced(("Gholdengo", True), ("Garchomp", False), ("Golisopod", False))
BOX = {"name": "x", "center": (1400, 300), "box": (1300, 280, 1500, 320)}


def one_battle() -> list[Step]:
    return (
        lobby_to_team_select()
        + action("Gholdengo")
        + action("Gholdengo")
        + FAINT
        + peeked_action("Garchomp", CHOMP_MOVES, mega=True)
        + end_of_battle("win")
    )


def rows_of(request) -> list[tuple[str, str]]:
    return [(r.row_id, r.text) for r in request.rows]


def assert_whole(request, rows: list[str]) -> None:
    """The Request carries every row read so far once, in turn order, and one event per row."""
    ids = [r.row_id for r in request.rows]
    assert len(ids) == len(set(ids)) and [r.text for r in request.rows] == rows
    assert len(row_events(request.events)) == len(rows)
    turns = [e["n"] for e in request.events if e.get("k") == "turn"]
    assert turns == list(range(1, len(turns) + 1))
    assert live_problems(request.events) == []


# One battle -------------------------------------------------------------------------------------------
def test_a_full_battle(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = one_battle()[:-1] + lobby_to_team_select() + action("Gholdengo") + end_of_battle("loss")
    played = run_battles(config, steps, tmp_path, monkeypatch, target=2, visits=[*VISITS, {1: START}])
    assert played.ending == "completed" and played.state.outcomes == ["win", "loss"] and played.state.restarts == 0
    assert played.device.index == len(steps) - 2 and played.device.rejected == []  # done on the options screen
    tag, tag2 = played.battle(1)["battle_id"], played.battle(2)["battle_id"]
    assert played.player.calls == [
        ("start", tag), ("preview", tag), ("end", tag, "win"), ("start", tag2), ("preview", tag2), ("end", tag2, "loss")
    ]  # fmt: skip
    next_lobby = len(one_battle()) - 1
    assert [n for i, n in played.device.taps if i in (next_lobby - 1, next_lobby)] == [
        "continue_battling", "begin_matchmaking"
    ]  # fmt: skip
    assert [r["kind"] for r in played.records(2)] == ["action"] and played.records(2)[0]["battle_id"] == tag2
    preview = played.json("battle-1/team_preview.json")
    assert preview["decision"]["picks"] == [0, 1, 2] and preview["decision"]["source"] == "player"
    assert preview["request"]["battle_tag"] == tag and preview["request"]["opponent_preview"][0] == "Glimmora"
    assert [n for i, n in played.device.taps if i == 2] == ["team_rows0", "team_rows1", "team_rows2", "team_done"]
    battle = played.battle(1)
    assert (battle["outcome"], battle["outcome_source"]) == ("win", "result screen")
    assert battle["team_played_names"] == ["Gholdengo", "Garchomp", "Golisopod"]
    final = played.json("battle-1/events.json")  # the whole event list, end included
    assert final["battle"] == tag and final["events"][0]["type"] == "battle_start"
    assert final["events"][-1] == {"k": "end", "result": "win"}

    records = played.records()
    assert [(r["kind"], r["turn"], r["sequence"]) for r in records] == [
        ("action", 1, 0), ("action", 2, 1), ("forced", 2, 2), ("action", 3, 3)
    ]  # fmt: skip
    assert all(r["schema"] == "pcsa-decision/4" and r["battle_id"] == tag for r in records)
    assert all("protocol" not in r and "request" not in r for r in records)
    assert [r["order"] for r in records] == [
        "/choose move makeitrain", "/choose move makeitrain", "/choose switch Garchomp", "/choose move earthquake"
    ]  # fmt: skip
    assert [r["peek"] for r in records if r["kind"] == "action"] == ["skipped", "skipped", "mega_button"]
    assert [r["degrade"] for r in records] == ["player"] * 4
    first, second, replace, peeked = records
    assert first["offer"]["moves_read"] is False and first["offer"]["can_mega_evolve"] is None
    assert [m["id"] for m in first["offer"]["moves"]] == ["makeitrain", "shadowball", "trick", "nastyplot"]
    assert [s["name"] for s in first["offer"]["switches"]] == ["Garchomp", "Golisopod"]
    assert second["hints"]["pp_counted"]["makeitrain"][0] < second["hints"]["pp_counted"]["makeitrain"][1]
    assert second["hints"]["first_turn_over"] is True
    assert replace["offer"]["forced"] == "faint" and [s["name"] for s in replace["offer"]["switches"]] == [
        "Garchomp", "Golisopod"
    ]  # fmt: skip
    assert peeked["offer"]["moves_read"] is True and peeked["offer"]["can_mega_evolve"] is True
    assert [m["pp"] for m in peeked["offer"]["moves"]] == [[16, 16]] * 4
    assert [s["name"] for s in peeked["offer"]["switches"]] == ["Golisopod"]  # Gholdengo fainted
    assert [len(r["rows"]) for r in records] == [2, 7, 7, 12]
    assert [r["text"] for r in peeked["rows"]] == START + TURN1 + TURN2

    requests = played.requests()[:4]
    for request, rows in zip(requests, ([START, START + TURN1, START + TURN1, START + TURN1 + TURN2]), strict=True):
        assert request.battle_tag == tag
        assert_whole(request, rows)
    for before, after in zip(requests, requests[1:], strict=False):
        assert after.events[: len(before.events)] == before.events  # the list only grows
    assert requests[0].events[0]["type"] == "battle_start"
    assert [e["k"] for e in requests[-1].events[1:]].count("faint") == 1
    tag_end, result, events = played.ends[0]
    assert (tag_end, result) == (tag, "win") and events[-1] == {"k": "end", "result": "win"}
    assert live_problems(events) == []
    assert (played.recorder.dir / "battle-1" / "battle_log.txt").is_file()


# Restarts ---------------------------------------------------------------------------------------------
def test_restart_at_team_preview(config: Config, tmp_path: Path, monkeypatch) -> None:
    played = run_battles(config, one_battle(), tmp_path, monkeypatch, visits=VISITS, faults={("teampreview", 1)})
    assert played.ending == "completed" and played.state.restarts == 1 and played.state.outcomes == ["win"]
    tag = played.battle(1)["battle_id"]
    assert played.player.calls == [("start", tag), ("preview", tag), ("end", tag, "win")]
    asked = played.requests("teampreview")
    assert [r.battle_tag for r in asked] == [tag, tag] and asked[0].opponent_preview == asked[1].opponent_preview
    assert [n for i, n in played.device.taps if i == 2] == ["team_rows0", "team_rows1", "team_rows2", "team_done"]
    assert any(e["event"] == "team_select_after_restart" and e["same_match"] for e in played.log_events())
    assert [r["kind"] for r in played.records()] == ["action", "action", "forced", "action"]
    assert_whole(played.requests()[0], START)


def test_restart_mid_battle_on_an_action_turn(config: Config, tmp_path: Path, monkeypatch) -> None:
    visits = [VISITS[0], VISITS[1], VISITS[1], VISITS[2]]  # the visit after the restart reads the same log
    played = run_battles(config, one_battle(), tmp_path, monkeypatch, visits=visits, faults={("decide", 2)})
    assert played.ending == "completed" and played.state.restarts == 1 and played.state.outcomes == ["win"]
    tag = played.battle(1)["battle_id"]
    assert [c[0] for c in played.player.calls] == ["start", "preview", "end"]
    lost, again = played.requests()[1:3]
    assert {lost.battle_tag, again.battle_tag} == {tag} and lost.request_id == again.request_id == 1
    assert rows_of(again) == rows_of(lost) and again.events == lost.events  # nothing lost, nothing added twice
    assert_whole(again, START + TURN1)
    assert again.hints.refused == () and not any(m.disabled for m in again.offer.moves)
    records = played.records()
    assert [(r["kind"], r["turn"]) for r in records] == [("action", 1), ("action", 2), ("forced", 2), ("action", 3)]
    assert records[1]["refused"] == [] and records[1]["same_turn"] is False


def test_restart_at_a_forced_switch(config: Config, tmp_path: Path, monkeypatch) -> None:
    played = run_battles(config, one_battle(), tmp_path, monkeypatch, visits=VISITS, faults={("decide", 3)})
    assert played.ending == "completed" and played.state.restarts == 1 and played.state.outcomes == ["win"]
    lost, again = played.requests()[2:4]
    assert lost.offer.forced == again.offer.forced == "faint" and lost.battle_tag == again.battle_tag
    assert rows_of(again) == rows_of(lost) and again.events == lost.events
    assert [s.name for s in again.offer.switches] == ["Garchomp", "Golisopod"]
    assert_whole(again, START + TURN1)
    records = played.records()
    assert [(r["kind"], r["order"]) for r in records][2:] == [
        ("forced", "/choose switch Garchomp"), ("action", "/choose move earthquake")
    ]  # fmt: skip
    assert len({r["battle_id"] for r in records}) == 1


PIVOT_START = ["Kai sent out Glimmora!", "Go! Cinderace!"]
PIVOT_TURN = [
    "Move: Cinderace used U-turn!",
    "HP Lost: Glimmora’s HP: 100% → 70%",
    "Cinderace, come back!",
    "Go! Primarina!",
    "Move: The opposing Glimmora used Power Gem!",
    "HP Lost: Primarina’s HP: 180 → 150",
]


def test_restart_after_a_pivot(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = (
        lobby_to_team_select()
        + action("Cinderace")
        + forced(("Cinderace", False), ("Primarina", False), ("Mamoswine", False))
        + action("Primarina")
        + end_of_battle("loss")
    )
    visits = [{1: PIVOT_START}, {1: PIVOT_START + PIVOT_TURN}, {1: PIVOT_START + PIVOT_TURN}]
    player = Script((5, 3, 4), [Order.move(1)])  # Cinderace leads with U-turn
    played = run_battles(config, steps, tmp_path, monkeypatch, visits=visits, player=player,
                         faults={("decide", 3)})  # fmt: skip
    assert played.ending == "completed" and played.state.restarts == 1 and played.state.outcomes == ["loss"]
    requests = played.requests()
    pivot, lost, again = requests[1:4]
    assert pivot.offer.forced == "pivot" and pivot.offer.leaving == "Cinderace"
    assert [s.name for s in pivot.offer.switches] == ["Primarina", "Mamoswine"]
    assert len({r.battle_tag for r in requests}) == 1
    assert rows_of(again) == rows_of(lost) and again.events == lost.events
    assert_whole(again, PIVOT_START + PIVOT_TURN)
    assert [s.name for s in again.offer.switches] == ["Cinderace", "Mamoswine"]  # Primarina is in
    assert [m.id for m in again.offer.moves][:2] == ["moonblast", "surf"]
    assert again.hints.refused == () and not any(e.get("k") == "refused" for e in again.events)
    records = played.records()
    assert [(r["kind"], r["order"]) for r in records] == [
        ("action", "/choose move uturn"), ("forced", "/choose switch Primarina"), ("action", "/choose move moonblast")
    ]  # fmt: skip
    assert records[1]["pivot"] is True


# Forfeit and unread results ---------------------------------------------------------------------------
def test_opponent_forfeit_and_an_unread_result(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = (
        lobby_to_team_select()
        + action("Gholdengo")
        + [Step(Screen.FORFEIT_MSG, after_s=2), Step(Screen.OPTIONS, accept=("continue_battling",))]
        + lobby_to_team_select()
        + action("Gholdengo")
        + [Step(Screen.COMM_ERROR, {"kind": "comm_error", "dismiss": BOX}, accept=("@dismiss",))]
        + lobby_to_team_select()
        + action("Gholdengo")
        + end_of_battle("loss")
    )
    played = run_battles(config, steps, tmp_path, monkeypatch, visits=[{1: START}] * 3)
    assert played.state.outcomes == ["opponent_forfeit", "unknown", "loss"]
    assert played.state.completed == 1 and played.state.played == 3
    assert played.battle(1)["outcome_source"] == "forfeit message" and played.battle(2)["outcome"] == "unknown"
    tags = [played.battle(i)["battle_id"] for i in (1, 2, 3)]
    ends = [c for c in played.player.calls if c[0] == "end"]
    assert ends == [("end", tags[0], "opponent_forfeit"), ("end", tags[1], "unknown"), ("end", tags[2], "loss")]
    assert [(t, r) for t, r, _ in played.ends] == list(
        zip(tags, ["opponent_forfeit", "unknown", "loss"], strict=True)
    )
    assert [events[-1] for _, _, events in played.ends] == [
        {"k": "end", "result": r} for r in ("forfeit", "unknown", "loss")
    ]  # LIVE `end` mirrors battle_end: the opponent's forfeit is "forfeit"
    assert all(live_problems(events) == [] for _, _, events in played.ends)


# A battle already on at start -------------------------------------------------------------------------
def test_a_battle_on_at_start_is_not_adopted(config: Config, tmp_path: Path, monkeypatch) -> None:
    steps = (
        [
            Step(Screen.ACTION, {"active_name": "Garchomp", "move_time": 40}, after_s=30),
            Step(Screen.WAIT, after_s=3),
            Step(
                Screen.FORCED, {"party": party(("Garchomp", True), ("Primarina", False)), "no_back": True}, after_s=20
            ),  # fmt: skip
            Step(Screen.MOVES, {"moves": CHOMP_MOVES}, after_s=5),
        ]
        + end_of_battle("loss")[:-1]
        + lobby_to_team_select()
        + action("Gholdengo")
        + end_of_battle("win")
    )
    played = run_battles(config, steps, tmp_path, monkeypatch, visits=[{1: START}])
    assert played.ending == "completed" and played.state.outcomes == ["win"] and played.state.played == 1
    lobby = next(i for i, s in enumerate(steps) if s.screen == Screen.LOBBY)
    assert [(i, n) for i, n in played.device.taps if i < lobby] == [(lobby - 1, "continue_battling")]
    assert [s.kind for s in played.sent] == ["teampreview", "decide"]
    assert played.player.calls[0][0] == "start" and len(played.records()) == 1
    assert any(e["event"] == "battle_not_adopted" for e in played.log_events())
    assert_whole(played.requests()[0], START)
