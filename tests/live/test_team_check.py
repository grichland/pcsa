"""The team brought when the game chose the lineup (the selection timed out): slots 1-3, lead first."""

from __future__ import annotations

from pcsa.live.team_check import check_team
from pcsa.phone.log_merge import BattleLog
from pcsa.run.battle_state import BattleRecord
from tests.live.turn_rig import TEAM


def battle(requested, auto: bool) -> BattleRecord:
    return BattleRecord(1, "b-1", TEAM.names, BattleLog("b-1"), requested=list(requested), team_submitted=True,
                        team_auto=auto)  # fmt: skip


def test_an_automatic_lineup_is_taken_even_when_the_lead_fits_the_request() -> None:
    b = battle([0, 4, 5], auto=True)  # the lead the game sent, Gholdengo, was also requested
    assert check_team(b, active="Gholdengo") and b.requested == [0, 1, 2]
    assert b.requested_at_preview == [0, 4, 5] and not check_team(b, active="Gholdengo")


def test_an_automatic_lineup_that_the_battle_contradicts() -> None:
    b = battle([4, 5, 3], auto=True)
    assert check_team(b, active="Gholdengo") and b.requested == [0, 1, 2]


def test_without_the_notice_a_fitting_lead_changes_nothing() -> None:
    b = battle([0, 4, 5], auto=False)
    assert not check_team(b, active="Gholdengo") and b.requested == [0, 4, 5]
