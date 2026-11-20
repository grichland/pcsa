"""T6: the runner's contract (fallbacks, late and stale answers, automatic turns, replay)."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from pcsa.common import log
from pcsa.common.clock import FakeClock
from pcsa.common.config import DegradeConfig
from pcsa.decide.api import Answer, MoveOffer, Offer, Order, TeamPreviewRequest
from pcsa.decide.builtin import FirstPlayer
from pcsa.decide.load import load_player
from pcsa.decide.runner import Runner
from tests.decide.examples import forced_offer, gold_team, read_offer, request

WAIT_S = 5.0


class Scripted(FirstPlayer):
    """A Player whose choose_move / teampreview are given functions; remembers every Request it was shown."""

    name = "scripted"

    def __init__(self, move=None, preview=None):
        self.move, self.preview, self.seen = move, preview, []

    def choose_move(self, request):
        self.seen.append(request)
        return self.move(request) if self.move else super().choose_move(request)

    def teampreview(self, request):
        return self.preview(request) if self.preview else super().teampreview(request)


@pytest.fixture
def clock():
    return FakeClock(start=1000.0)


def runner_for(player, clock):
    return Runner(player, clock, DegradeConfig(tap_reserve_s=2.5))


def test_valid_answer_is_used(clock):
    decision = runner_for(Scripted(lambda r: Answer(Order.move(1), {"why": "x"})), clock).decide(request())
    assert (decision.source, decision.text, decision.notes, decision.rejected) == (
        "player",
        "/choose move earthquake",
        {"why": "x"},
        (),
    )
    assert decision.player_ms is not None


def test_a_mega_the_screen_does_not_offer_is_dropped_and_the_move_kept(clock):
    decision = runner_for(Scripted(lambda r: Answer(Order.move(1, mega=True), {"why": "x"})), clock).decide(request())
    assert (decision.source, decision.text) == ("player", "/choose move earthquake")
    assert decision.notes == {"why": "x", "repaired": "mega dropped: not offered on screen"}


def _boom(request):
    raise RuntimeError("bad state")


@pytest.mark.parametrize(
    ("move", "offer", "reason", "text"),
    [
        (_boom, None, "Player raised RuntimeError: bad state", "/choose move iceshard"),
        (lambda r: Order.move(1), None, "wrong answer type Order", "/choose move iceshard"),
        (
            lambda r: Answer(Order.move(1)),
            forced_offer(),
            "move on a forced (faint) screen",
            "/choose switch Garchomp",
        ),
        (
            lambda r: Answer(Order.switch("Primarina")),
            None,
            "switch target Primarina not on the party screen",
            "/choose move iceshard",
        ),
        (
            lambda r: Answer(Order.move(1)),
            read_offer(disabled=(1,)),
            "move row 1 shown disabled",
            "/choose move iceshard",
        ),
        (
            lambda r: Answer(Order.switch("Mamoswine")),
            forced_offer("Mamoswine", "pivot"),
            "switch target Mamoswine is leaving (pivot)",
            "/choose switch Garchomp",
        ),
    ],
)
def test_rejected_answers_fall_back_to_builtin(clock, move, offer, reason, text):
    decision = runner_for(Scripted(move), clock).decide(request(offer))
    assert (decision.source, decision.reason, decision.text) == ("builtin", reason, text)
    assert decision.rejected[-1]["reason"] == reason
    assert decision.notes["fallback"] == reason and "reason" in decision.notes["builtin"]


def test_late_answer_by_the_fake_clock(clock):
    def slow(r):
        clock.advance(60)
        return Answer(Order.move(1))

    decision = runner_for(Scripted(slow), clock).decide(request())
    assert decision.source == "builtin"
    assert decision.reason.startswith("answer late by")
    assert decision.rejected == ({"order": "/choose move earthquake", "reason": decision.reason},)


def test_hung_player_then_stale_answer_dropped(clock):
    release = threading.Event()

    def hang_once(r):
        if r.request_id == 2:
            clock.advance(60)
            release.wait(WAIT_S)
            return Answer(Order.move(3))
        return Answer(Order.move(2))

    runner = runner_for(Scripted(hang_once), clock)
    first = runner.decide(request(request_id=2))
    assert (first.source, first.reason) == ("builtin", "no answer within 37.5s")
    release.set()
    assert runner.worker.idle(WAIT_S)
    second = runner.decide(request(request_id=3, deadline=clock.now() + 40))
    assert (second.source, second.text) == ("player", "/choose move iciclespear")
    assert [(e["request_id"], e["order"], e["outcome"]) for e in second.late] == [(2, "move3", "dropped")]


def test_no_time_left_skips_the_player(clock):
    player = Scripted()
    decision = runner_for(player, clock).decide(request(deadline=1001.0))
    assert decision.source == "builtin" and decision.reason.startswith("no time left")
    assert player.seen == []


def test_builtin_invalid_falls_to_screen_then_blind(clock):
    all_disabled = replace(read_offer(disabled=(0, 1, 2, 3)), switches=())
    decision = runner_for(Scripted(_boom), clock).decide(request(all_disabled))
    assert (decision.source, decision.order) == ("blind", Order.move(0))
    assert decision.rejected[-1]["reason"] == "builtin: nothing on offer"
    only_pp = Offer((MoveOffer(0, None, (0, 5)), MoveOffer(1, None, (3, 5))), True, (), True)
    decision = runner_for(Scripted(lambda r: Answer(Order.move(0))), clock).decide(request(only_pp))
    assert (decision.source, decision.order) == ("builtin", Order.move(1))


def test_automatic_turn_asks_nothing(clock):
    player = Scripted(_boom)
    offer = Offer((MoveOffer(0, "struggle"),), True, (), False, automatic=True)
    decision = runner_for(player, clock).decide(request(offer))
    assert (decision.source, decision.order, decision.text) == ("automatic", Order.move(0), "/choose move struggle")
    assert player.seen == []


@pytest.mark.parametrize("name", ["first"])  # cg: tests/cg
def test_same_offer_whichever_player(clock, name):
    inner = load_player(name)
    spy = Scripted(lambda r: inner.choose_move(r))
    req = request(read_offer(disabled=(0,), mega=True))
    decision = runner_for(spy, clock).decide(req)
    assert spy.seen == [req] and spy.seen[0].offer is req.offer
    assert decision.source == "player"


def test_builtin_answers_the_same_when_asked_twice(clock):
    runner = runner_for(FirstPlayer(), clock)
    for offer in (None, read_offer(disabled=(0,)), forced_offer()):
        first, again = runner.decide(request(offer)), runner.decide(request(offer))
        assert (first.order, first.text, first.notes) == (again.order, again.text, again.notes)


def _preview(deadline=1040.0):
    return TeamPreviewRequest("tag", replace(gold_team(), picks=None), (None,) * 6, deadline)


@pytest.mark.parametrize(
    ("answer", "picks", "source"),
    [
        (lambda r: ((4, 1, 2), {"a": 1}), (4, 1, 2), "player"),
        (lambda r: ([4, 1, 2], {}), (4, 1, 2), "player"),
        (lambda r: ((0, 0, 1), {}), (0, 1, 2), "builtin"),
        (lambda r: (4, 1, 2), (0, 1, 2), "builtin"),
        (_boom, (0, 1, 2), "builtin"),
    ],
)
def test_teampreview(clock, answer, picks, source):
    decision = runner_for(Scripted(preview=answer), clock).teampreview(_preview())
    assert (decision.picks, decision.source) == (picks, source)
    if source == "builtin":
        assert decision.notes == {"fallback": decision.reason}


def test_battle_start_and_end_errors_are_logged(clock, monkeypatch):
    seen = []
    monkeypatch.setattr(log, "warning", lambda name, **data: seen.append((name, data)))

    class Broken(FirstPlayer):
        def battle_start(self, battle_tag, team):
            raise RuntimeError("start")

        def battle_end(self, battle_tag, outcome, events=(), history=()):
            raise RuntimeError("end")

    runner = runner_for(Broken(), clock)
    runner.battle_start("tag", gold_team())
    runner.battle_end("tag", "win")
    assert runner.worker.idle(WAIT_S)
    assert [(n, d["call"]) for n, d in seen] == [("player_failed", "battle_start"), ("player_failed", "battle_end")]
    assert runner.decide(request()).source == "player"


def test_unknown_player_name():
    from pcsa.common.errors import ConfigError

    with pytest.raises(ConfigError):
        load_player("random")
