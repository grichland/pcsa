"""Recorded messages given the form ids of their Pokémon."""

from __future__ import annotations

from pcsa.events.forms import with_forms


def _messages() -> list[dict]:
    return [
        {"type": "battle_start", "own_team": [], "opp_preview": ["rotom", "ninetales", "garchomp", None]},
        {"type": "events", "events": [
            {"k": "enter", "side": "opp", "mon": "rotom"},
            {"k": "enter", "side": "own", "mon": "ninetales"},
            {"k": "item", "side": "own", "mon": "ninetales", "item": "leftovers", "what": "removed",
             "of": {"side": "opp", "mon": "rotom"}},
        ]},  # fmt: skip
        {"seq": 0, "order": {"kind": "switch", "mon": "ninetales"}},
    ]


def test_the_preview_names_give_the_forms() -> None:
    sent = _messages()
    out = with_forms(sent, ["Rotom Wash", "Alolan Ninetales", "Garchomp", None])
    assert out[0]["opp_preview"] == ["rotomwash", "ninetalesalola", "garchomp", None]
    assert [e["mon"] for e in out[1]["events"]] == ["rotomwash", "ninetales", "ninetales"]  # ours stay
    assert out[1]["events"][2]["of"] == {"side": "opp", "mon": "rotomwash"} and out[2] == sent[2]
    assert sent[0]["opp_preview"][0] == "rotom"  # the recorded messages are not changed


def test_a_form_only_move_names_the_form_and_no_form_changes_nothing() -> None:
    sent = _messages()
    assert with_forms(sent, ["Rotom", "Ninetales", "Garchomp", None]) is sent
    sent[1]["events"].append({"k": "move", "side": "opp", "mon": "rotom", "move": "overheat"})
    out = with_forms(sent, ["Rotom", "Ninetales", "Garchomp", None])
    assert out[0]["opp_preview"][0] == "rotomheat" and out[1]["events"][0]["mon"] == "rotomheat"
