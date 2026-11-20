"""The turn pipeline on the scripted device: Request rows and events, the Offer, peeks, refusals, switches,
forced replacements, the /3 record."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcsa.common.config import Config
from pcsa.common.types import MoveInfo, Observation, PartyMember, Screen
from pcsa.decide.api import Answer, Order
from pcsa.decide.builtin import FirstPlayer
from pcsa.live.fake import Step
from tests.live.turn_rig import Rig

START = {1: ["U- sent out Glimmora!", "Go! Mamoswine!"]}
TURN1 = {
    1: [
        "U- sent out Glimmora!",
        "Go! Mamoswine!",
        "Move: Mamoswine used Earthquake!",
        "HP Lost: Glimmora’s HP: 100% → 40%",
    ]
}
ACTION = Step(Screen.ACTION, {"active_name": "Mamoswine", "move_time": 40}, accept=("fight", "move_rows"),
              advance_on=())  # fmt: skip


def action_obs(name: str = "Mamoswine") -> Observation:
    return Observation(Screen.ACTION, (2340, 1080), fields={"active_name": name, "move_time": 40})


class Choose(FirstPlayer):
    name = "choose"

    def __init__(self, order: Order) -> None:
        self.order, self.seen = order, []

    def choose_move(self, request):
        self.seen.append(request)
        return Answer(self.order, {"why": "test"})


@pytest.fixture
def rigs():
    made = []
    yield made
    for rig in made:
        rig.close()


def make(rigs, *args, **kw) -> Rig:
    rig = Rig(*args, **kw)
    rigs.append(rig)
    return rig


def test_turn_without_a_peek_sends_rows_events_and_the_sheet_moves(config: Config, tmp_path: Path, rigs) -> None:
    player = Choose(Order.move(1))
    rig = make(rigs, config, tmp_path, [ACTION], [START, TURN1], player, move_peek="when_needed")
    b = rig.battle
    rig.pipeline.action(b, action_obs(), rig.clock.now())
    rig.pipeline.action(b, action_obs(), rig.clock.now())
    first, second = player.seen
    assert first.events[0]["type"] == "battle_start" and len(first.events[0]["own_team"]) == 6
    assert [e["k"] for e in first.events[1:]] == ["turn", "enter", "enter"]
    assert [e["k"] for e in second.events[1:]] == ["turn", "enter", "enter", "move", "hp", "turn"]
    assert second.events[1:] == (*first.events[1:], *second.events[4:])  # the list only grows
    assert [r.row_id for r in second.rows] == ["1-0", "1-1", "1-2", "1-3"]
    offer = second.offer
    assert not offer.moves_read and offer.can_mega_evolve is None
    assert [m.id for m in offer.moves] == ["iceshard", "earthquake", "iciclespear", "endeavor"]
    assert all(m.pp is None and m.disabled is None for m in offer.moves)
    assert [s.name for s in offer.switches] == ["Garchomp", "Golisopod"]
    assert second.hints.pp_counted["earthquake"] == (11, 12) and second.hints.first_turn_over
    assert second.team.picks == (4, 1, 2) and second.team.brought() == ("Mamoswine", "Garchomp", "Golisopod")
    assert second.opponent_preview == ("glimmora", "lucario", "garchomp")
    assert len(rig.taps()) == 6 and rig.taps()[:2] == ["fight", "move_rows1"]  # Fight, then the row twice
    records = rig.records()
    assert [r["schema"] for r in records] == ["pcsa-decision/4"] * 2
    rec = records[1]
    assert (rec["sequence"], rec["kind"], rec["turn"], rec["choice"], rec["order"]) == (
        1, "action", 2, "move1", "/choose move earthquake"
    )  # fmt: skip
    assert rec["peek"] == "skipped" and rec["degrade"] == "player" and rec["player"] == "choose"
    assert (
        len(rec["rows"]) == 4 and rec["events"][0]["type"] == "battle_start" and rec["offer"]["moves_read"] is False
    )
    assert "protocol" not in rec and "request" not in rec
    assert rig.saver.calls == 2 and b.last_action["move_id"] == "earthquake"


def test_a_refused_move_is_offered_disabled_and_the_player_is_asked_again(
    config: Config, tmp_path: Path, rigs
) -> None:
    player = Choose(Order.move(0))
    rig = make(rigs, config, tmp_path, [ACTION], [START, TURN1, TURN1], player, move_peek="when_needed")
    b = rig.battle
    for _ in range(2):
        rig.pipeline.action(b, action_obs(), rig.clock.now())
    rig.pipeline.action(b, action_obs(), rig.clock.now(), repeated=True)  # same log turn, nothing new
    last = player.seen[-1]
    assert last.hints.refused == (0,)
    assert last.offer.moves[0].disabled is True
    assert last.events[-1] == {"k": "refused", "order": {"kind": "move", "move": "iceshard", "mega": False}}
    rec = rig.records()[-1]
    assert rec["refused"] == [0] and rec["same_turn"] is True and rec["peek"] == "retry"
    assert rec["degrade"] == "screen" and rec["choice"] == "move1"  # row 0 shown refused: the next usable row


def test_low_counted_pp_opens_the_move_list(config: Config, tmp_path: Path, rigs) -> None:
    moves = (MoveInfo(0, "Ice Shard", (20, 20)), MoveInfo(1, "Earthquake", (1, 12)),
             MoveInfo(2, "Icicle Spear", (20, 20)), MoveInfo(3, "Endeavor", (8, 8)))  # fmt: skip
    steps = [
        Step(Screen.ACTION, accept=("fight",)),
        Step(Screen.MOVES, {"moves": moves, "mega_available": False}, accept=("move_back",)),
        ACTION,
    ]
    used = ["Move: Mamoswine used Earthquake!"] * 8
    visits = [{1: ["Go! Mamoswine!", *used]}]
    player = Choose(Order.move(1))
    rig = make(rigs, config, tmp_path, steps, visits, player, move_peek="when_needed")
    rig.battle.last_turn = None
    rig.pipeline.action(rig.battle, action_obs(), rig.clock.now())
    request = player.seen[0]
    assert request.hints.pp_counted["earthquake"] == (1, 12)  # 12 - 8 counted, 3 more the look showed
    assert request.offer.moves_read and request.offer.moves[1].pp == (1, 12)
    assert request.offer.can_mega_evolve is False
    assert rig.records()[0]["peek"] == "low_pp"
    assert rig.battle.pp_offsets == {"0:earthquake": -3}  # the look showed three uses the log missed


def test_a_mega_stone_holder_looks_for_the_button(config: Config, tmp_path: Path, rigs) -> None:
    moves = tuple(MoveInfo(k, n, (10, 10)) for k, n in enumerate(("Earthquake", "Dragon Claw", "Swords Dance",
                                                                   "Protect")))  # fmt: skip
    steps = [
        Step(Screen.ACTION, accept=("fight",)),
        Step(Screen.MOVES, {"moves": moves, "mega_available": True}, accept=("move_back",)),
        Step(Screen.ACTION, accept=("fight", "mega", "move_rows"), advance_on=()),
    ]
    player = Choose(Order.move(0, mega=True))
    rig = make(rigs, config, tmp_path, steps, [{1: ["Go! Garchomp!"]}], player, move_peek="when_needed")
    rig.battle.requested = [1, 4, 2]
    rig.pipeline.action(rig.battle, action_obs("Garchomp"), rig.clock.now())
    assert rig.records()[0]["peek"] == "mega_button"
    assert player.seen[0].offer.can_mega_evolve is True
    assert rig.records()[0]["order"] == "/choose move earthquake mega" and "mega" in rig.taps()
    assert rig.battle.mega_pending


def test_a_switch_is_tapped_by_name_on_the_party_screen(config: Config, tmp_path: Path, rigs) -> None:
    party = (PartyMember(0, "Mamoswine", (195, 195)), PartyMember(1, "Golisopod", (160, 160)),
             PartyMember(2, "Garchomp", (180, 180)))  # fmt: skip
    steps = [
        Step(Screen.ACTION, accept=("pokemon",)),
        Step(Screen.PARTY, {"party": party}, accept=("party_rows", "switch_in"), advance_on=("switch_in",)),
        Step(Screen.WAIT),
    ]
    player = Choose(Order.switch("Garchomp"))
    rig = make(rigs, config, tmp_path, steps, [START], player, move_peek="when_needed")
    rig.pipeline.action(rig.battle, action_obs(), rig.clock.now())
    assert rig.taps() == ["pokemon", "party_rows2", "party_rows2", "switch_in"]
    assert rig.records()[0]["choice"] == "switch1" and rig.records()[0]["order"] == "/choose switch Garchomp"
    assert rig.battle.visible_order[0] == 1


def test_a_forced_replacement_after_a_faint(config: Config, tmp_path: Path, rigs) -> None:
    party = (PartyMember(0, "Mamoswine", (0, 195), fainted=True), PartyMember(1, "Garchomp", (180, 180)),
             PartyMember(2, "Golisopod", (160, 160)))  # fmt: skip
    steps = [ACTION, Step(Screen.FORCED, {"party": party}, accept=("party_rows", "switch_in"),
                          advance_on=("switch_in",)), Step(Screen.WAIT)]  # fmt: skip
    rig = make(rigs, config, tmp_path, steps, [START], move_peek="when_needed")
    b = rig.battle
    rig.pipeline.action(b, action_obs(), rig.clock.now())
    rig.device.index = 1
    obs = Observation(Screen.FORCED, (2340, 1080), fields={"party": party})
    rig.pipeline.forced(b, obs, previous_forced=False)
    rec = rig.records()[-1]
    assert rec["kind"] == "forced" and rec["offer"]["forced"] == "faint" and rec["pivot"] is False
    assert [s["name"] for s in rec["offer"]["switches"]] == ["Garchomp", "Golisopod"]
    assert rec["order"] == "/choose switch Garchomp" and rec["attempt"] == 1
    assert rig.taps()[-3:] == ["party_rows1", "party_rows1", "switch_in"]
    assert 0 in b.fainted and b.visible_order[0] == 1


def test_a_new_pipeline_sends_the_full_rows_and_events_again(config: Config, tmp_path: Path, rigs) -> None:
    player = Choose(Order.move(1))
    rig = make(rigs, config, tmp_path, [ACTION], [START, TURN1], player, move_peek="when_needed")
    b = rig.battle
    for _ in range(2):
        rig.pipeline.action(b, action_obs(), rig.clock.now())
    before = player.seen[-1]
    again = make(rigs, config, tmp_path / "again", [ACTION], [{}], player, move_peek="when_needed")
    again.pipeline.action(b, action_obs(), again.clock.now(), repeated=True)
    after = player.seen[-1]
    assert after.rows == before.rows
    rows = [e for e in after.events[1:] if e["k"] not in ("turn", "refused", "gap")]
    assert rows == [e for e in before.events[1:] if e["k"] not in ("turn", "refused", "gap")]


def test_a_pivot_offers_the_others_and_a_repeat_is_a_retry(config: Config, tmp_path: Path, rigs) -> None:
    party = (PartyMember(0, "Mamoswine", (120, 195)), PartyMember(1, "Garchomp", (180, 180)),
             PartyMember(2, "Golisopod", (160, 160)))  # fmt: skip
    forced = Step(Screen.FORCED, {"party": party}, accept=("party_rows", "switch_in"), advance_on=())
    rig = make(rigs, config, tmp_path, [forced], [START], move_peek="when_needed")
    b = rig.battle
    obs = Observation(Screen.FORCED, (2340, 1080), fields={"party": party})
    rig.pipeline.forced(b, obs, previous_forced=False)
    b.visible_order = [0, 1, 2]  # the screen is still up: the replacement did not go through
    rig.pipeline.forced(b, obs, previous_forced=True)
    first, second = rig.records()
    assert first["offer"]["forced"] == "pivot" and first["offer"]["leaving"] == "Mamoswine" and first["pivot"]
    assert [s["name"] for s in first["offer"]["switches"]] == ["Garchomp", "Golisopod"]
    assert second["attempt"] == 2
    assert not any(e.get("k") == "refused" for e in second["events"])  # the game cannot refuse a healthy member
    assert second["live"]["decision_request"]["retry"] is True


def test_without_a_team_the_move_list_is_read(config: Config, tmp_path: Path, rigs) -> None:
    rig = make(rigs, config, tmp_path, [ACTION], [START], move_peek="when_needed")
    rig.pipeline.team = None
    rig.pipeline.action(rig.battle, action_obs(), rig.clock.now())
    rec = rig.records()[0]
    assert rec["peek"] == "untracked" and rec["offer"]["moves"] == [] and rec["own_export"] == ""
    assert rec["order"] == "move0" and rec["degrade"] == "player"


def test_the_log_read_leaves_the_search_its_time() -> None:
    from types import SimpleNamespace

    from pcsa.live.turn import SEARCH_OVERHEAD_S, TurnPipeline

    def reserve(enabled: bool, ms: float) -> float:
        search = SimpleNamespace(enabled=enabled, ms=ms)
        cfg = SimpleNamespace(cg=SimpleNamespace(search=search), degrade=SimpleNamespace(decide_reserve_s=3.0))
        return TurnPipeline._decide_reserve(SimpleNamespace(cfg=cfg))  # type: ignore[arg-type]

    assert reserve(False, 5000.0) == 3.0 and reserve(True, 1500.0) == 3.0
    assert reserve(True, 5000.0) == 5.0 + SEARCH_OVERHEAD_S
