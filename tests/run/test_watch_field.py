"""The battle-state box: field effects and each side's conditions with turns left, from the protocol."""

from __future__ import annotations

from pcsa.run.watch_field import field_view

PROTOCOL = [
    "|switch|p1a: Charizard|Charizard, L50|100/100",
    "|turn|1",
    "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Charizard",
    "|-sidestart|p1: me|move: Stealth Rock",
    "|-sidestart|p2: them|Spikes",
    "|-sidestart|p2: them|Spikes",
    "|turn|2",
    "|-weather|SunnyDay|[upkeep]",
    "|-fieldstart|move: Trick Room|[of] p2a: Hatterene",
    "|-sidestart|p2: them|move: Reflect",
    "|turn|3",
]


def test_field_view_counts_turns_left_and_layers() -> None:
    view = field_view(PROTOCOL, 3)
    assert view["turn"] == 3
    assert view["weather"] == {"name": "Sun", "left": 3}  # set on turn 1; the upkeep line does not restart it
    assert view["fields"] == [{"name": "Trick Room", "left": 4}]
    assert view["ours"] == [{"name": "Stealth Rock", "layers": None, "left": None}]
    assert view["theirs"] == [
        {"name": "Spikes", "layers": 2, "left": None},
        {"name": "Reflect", "layers": None, "left": 4},
    ]


def test_conditions_end_and_weather_clears() -> None:
    lines = [*PROTOCOL, "|-weather|none", "|-fieldend|move: Trick Room", "|-sideend|p2: them|Reflect", "|turn|4"]
    view = field_view(lines)
    assert view["turn"] == 4 and view["weather"] is None and view["fields"] == []
    assert [c["name"] for c in view["theirs"]] == ["Spikes"]


def test_missing_or_odd_protocol_gives_an_empty_field() -> None:
    assert field_view(None) == {"turn": None, "weather": None, "fields": [], "ours": [], "theirs": []}
    assert field_view(["|-sidestart|p3: x|Spikes", 7, "|turn|x"])["ours"] == []
