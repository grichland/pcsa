"""Our export text equals every recorded own_team.txt and decision own_export."""

from __future__ import annotations

from pcsa.battle.teams import export_text, team_from_record, teambuilder_team
from tests.battle.data_files import load


def test_every_recorded_own_team_round_trips() -> None:
    cases = load("own_exports.json.gz")["cases"]
    assert len(cases) >= 48
    wrong = []
    for case in cases:
        team = team_from_record(case["team"])
        text = export_text(teambuilder_team(team))
        expected = [case["own_team_txt"], *case["decision_exports"]]
        wrong += [case["run"] for want in expected if want is not None and want != text]
    assert wrong == []


def test_nickname_and_unread_moves() -> None:
    record = {
        "name": None,
        "members": [
            {
                "roster_index": 0,
                "name": "Chompy",
                "species": "Garchomp",
                "ability": None,
                "item": None,
                "moves": ["Earthquake", None],
                "stats": None,
                "points": None,
                "nature": None,
            }
        ],
    }
    text = export_text(teambuilder_team(team_from_record(record)))
    assert text == "Chompy (Garchomp)\nLevel: 50\n- Earthquake\n"
