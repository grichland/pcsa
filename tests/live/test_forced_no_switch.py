"""14 Nov 2026 (run 20261114-173647, battle 5): our last member went in by itself, the replacement screen's party
rows read as all fainted, no switch was legal and the session crashed (`b.names[None]`); the restart then played
with the first three of the roster instead of the trio brought."""

from types import SimpleNamespace

import pytest

from pcsa.live import restore
from pcsa.live.decide import Decision
from pcsa.live.orders import SemanticAction
from pcsa.live.turn import TurnPipeline
from pcsa.live.turn_sync import resync_party
from pcsa.phone.log_merge import BattleLog
from pcsa.players.base import Degrade, LegalAction
from pcsa.run.battle_state import BattleRecord

ROSTER = ("Gholdengo", "Garchomp", "Golisopod", "Primarina", "Mamoswine", "Cinderace")


def _record() -> BattleRecord:
    b = BattleRecord(5, "id", ROSTER, BattleLog("id"), requested=[3, 2, 4])  # Primarina, Golisopod, Mamoswine
    b.visible_order = [1, 0, 2]
    return b


def _row(row: int, name: str, fainted: bool) -> SimpleNamespace:
    return SimpleNamespace(row=row, name=name, fainted=fainted, hp=(0, 1) if fainted else (1, 1))


def test_a_replacement_screen_without_a_switch_taps_nothing() -> None:
    b = _record()
    b.forced_choice = {"action": None, "first_seen": 0.0, "turn": 15}
    move = LegalAction(SemanticAction("move", move=0), None)
    decision = Decision(move, (move,), Degrade.LEGAL_RANDOM, "nothing readable", [], {})
    slept = []

    def boom(*_a, **_k):
        raise AssertionError("nothing may be saved or tapped")

    pipeline = SimpleNamespace(
        _begin_forced=lambda *_a: ((), None, None),
        _build=lambda *_a, **_k: (None, None),
        _request=lambda *_a, **_k: None,
        decider=SimpleNamespace(decide=lambda *_a, **_k: decision),
        _save_decision=boom,
        phone=SimpleNamespace(executor=SimpleNamespace(plan_forced=boom, execute=boom)),
        clock=SimpleNamespace(sleep=slept.append),
        cfg=SimpleNamespace(timing=SimpleNamespace(post_action_wait_s=1.5)),
    )
    TurnPipeline.forced(pipeline, b, SimpleNamespace(fields={}), False)  # type: ignore[arg-type]
    assert slept == [1.5] and b.forced_choice["action"] is None and b.visible_order == [1, 0, 2]


def test_party_rows_that_all_read_fainted_mark_nobody() -> None:
    b = _record()
    party = (_row(0, "Golisopod", True), _row(1, "Primarina", True), _row(2, "Mamoswine", True))
    resync_party(b, party)
    assert b.fainted == set()
    resync_party(b, (*party[:2], _row(2, "Mamoswine", False)))
    assert b.fainted == {1, 0}  # Golisopod (slot 1) and Primarina (slot 0)


def _decision(names: list[str], conditions: list[str], degrade: str = "full", details: str = "") -> dict:
    side = [
        {"ident": f"p1: {n}", "condition": c, "details": details if i == 0 else n}
        for i, (n, c) in enumerate(zip(names, conditions, strict=True))
    ]
    return {"degrade": degrade, "request": {"side": {"pokemon": side}}}


def test_the_lineup_comes_back_from_the_earlier_records() -> None:
    records = [
        _decision(["Primarina", "Golisopod", "Mamoswine"], ["187/187", "182/182", "187/187"]),
        _decision(
            ["Golisopod", "Primarina", "Mamoswine"], ["36/182", "0 fnt", "154/187 tox"], details="golisopodmega, L50"
        ),
        _decision(["Primarina", "Golisopod", "Mamoswine"], ["0 fnt", "0 fnt", "0 fnt"], degrade="legal_random"),
    ]
    found = restore.lineup(records, ROSTER)
    assert found == {"requested": [3, 2, 4], "visible_order": [1, 0, 2], "fainted": {0}, "mega_used": True}
    b = BattleRecord(5, "id", ROSTER, BattleLog("id"))
    restore._restore_lineup(b, records)
    assert b.names == ("Primarina", "Golisopod", "Mamoswine") and b.visible_order == [1, 0, 2] and b.fainted == {0}


@pytest.mark.parametrize(
    "records",
    [[], [{"degrade": "full", "request": {}}], [_decision(["Pikachu", "Golisopod", "Mamoswine"], ["1/1"] * 3)]],
)
def test_records_that_do_not_name_the_trio_leave_the_battle_alone(records) -> None:
    assert restore.lineup(records, ROSTER) is None
    b = BattleRecord(5, "id", ROSTER, BattleLog("id"))
    restore._restore_lineup(b, records)
    assert b.requested is None and b.visible_order == [0, 1, 2]
