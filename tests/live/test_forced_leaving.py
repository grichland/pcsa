"""Which of our mons is leaving on a replacement screen (live/turn.py leaving_slot)."""

from types import SimpleNamespace

from pcsa.live.turn import leaving_slot
from pcsa.phone.log_merge import BattleLog
from pcsa.run.battle_state import BattleRecord


def _row(row: int, name: str, fainted: bool = False) -> SimpleNamespace:
    return SimpleNamespace(row=row, name=name, fainted=fainted, hp=(0, 1) if fainted else (1, 1))


def _record(active: int) -> BattleRecord:
    b = BattleRecord(1, "id", ("Mamoswine", "Golisopod", "Primarina"), BattleLog("id"), requested=[0, 1, 2])
    b.visible_order = [active] + [s for s in (0, 1, 2) if s != active]
    return b


def test_emergency_exit_marks_the_standing_active_as_leaving() -> None:
    # 14 Nov battle 21: Golisopod came in, dropped below half and Emergency Exit asked for a replacement before the
    # log showed it; the party screen lists Primarina first, the active Golisopod second (alive), Mamoswine fainted.
    b = _record(active=1)
    party = (_row(0, "Primarina"), _row(1, "Golisopod"), _row(2, "Mamoswine", fainted=True))
    assert leaving_slot(b, party) == 1


def test_a_fainted_active_is_a_plain_replacement() -> None:
    b = _record(active=0)
    party = (_row(0, "Mamoswine", fainted=True), _row(1, "Golisopod"), _row(2, "Primarina"))
    assert leaving_slot(b, party) is None


def test_no_team_yet_means_no_leaving_slot() -> None:
    b = BattleRecord(1, "id", (), BattleLog("id"))
    assert leaving_slot(b, (_row(0, "Primarina"),)) is None
