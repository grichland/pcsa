"""The teams and field panels from CG's Battle view, the network's belief about their builds, display names, and
records of decision schemas before pcsa-decision/4."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pcsa.common import paths
from pcsa.run.watch import RunReader
from pcsa.run.watch_belief import belief_index, belief_view
from pcsa.run.watch_matrix import matrix_view, row_of_order
from pcsa.run.watch_names import game_names, max_pp, names_of, pretty, shown
from pcsa.run.watch_view import battle_text, field_view, view_panels

FIXTURE = paths.repo_root() / "tests" / "fixtures" / "watch_run"
EXPORT = "Garchomp @ Choice Scarf\nAbility: Rough Skin\n- Earthquake\n- Scale Shot\n\nRaichu @ Raichunite Y\n- Zap Cannon\n"
ZERO = dict.fromkeys(("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"), 0)


def mon(species, **extra) -> dict:
    base = {
        "species": species, "form": None, "hp": 1.0, "fainted": False, "status": "", "boosts": dict(ZERO),
        "effects": [], "moves": [], "item": None, "ability": None, "types": [], "active": False, "revealed": True,
    }  # fmt: skip
    return {**base, **extra}


def view(**extra) -> dict:
    own = [
        mon("garchomp", active=True, hp=0.536, item="choicescarf", ability="roughskin", types=["dragon", "ground"],
            moves=[["earthquake", 9], ["scaleshot", None]], boosts={**ZERO, "atk": 1, "spe": -1}),
        mon("raichu", item="", moves=[["zapcannon", 5]]),
        mon("lucario", fainted=True, hp=0.0),
    ]  # fmt: skip
    opp = [
        mon("basculegion", hp=0.87, moves=[["flipturn", None]], types=["water", "ghost"]),
        mon("raichu", active=True, hp=0.004, status="par", item="raichunitey", moves=[["fakeout", None]]),
        mon("blaziken", revealed=False),
        mon("meowscarada", revealed=False),
        mon(None, revealed=False),
    ]
    base = {
        "turn": 4, "kind": "turn", "picks": [0, 1, 2], "opp_preview": [], "anomalies": 0, "own": own, "opp": opp,
        "weather": {"sunnyday": 2}, "fields": {"trickroom": 1}, "side": {"stealthrock": 1, "reflect": 6},
        "opp_side": {"spikes": 2, "tailwind": 1},
    }  # fmt: skip
    return {**base, **extra}


BELIEF = [
    {"species": "raichu", "brought": 1.0, "lead": None, "item": [["raichunitey", 0.9], ["", 0.1]],
     "ability": [["lightningrod", 0.6], ["static", 0.4]],
     "moves": [["fakeout", 0.25], ["thunderbolt", 0.2], ["nastyplot", 0.15], ["surf", 0.1], ["voltswitch", 0.05]]},
    {"species": "basculegion", "brought": 1.0, "lead": None, "item": [["choicescarf", 0.5], ["", 0.3], ["lifeorb", 0.2]],
     "ability": [["adaptability", 0.8]], "moves": [["flipturn", 0.25], ["wavecrash", 0.24]]},
    {"species": "blaziken", "brought": 0.62, "lead": None, "item": [["blazikenite", 0.7]], "ability": [["speedboost", 0.9]],
     "moves": [["closecombat", 0.25], ["flareblitz", 0.24]]},
    {"species": None, "brought": 0.1, "lead": None, "item": [], "ability": [], "moves": []},
    {"species": "garchomp", "brought": 0.5, "item": "odd", "ability": [["x"], [1, 2], ["roughskin", "p"]], "moves": 7},
]  # fmt: skip


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURE, tmp_path / "20261118-162900-neural-148"))


def test_our_team_from_the_view_and_the_team_sheet() -> None:
    panels = view_panels(view(), EXPORT, names_of({"own_export": EXPORT}), BELIEF)
    chomp, raichu, lucario = panels["your_team"]
    assert chomp["name"] == "Garchomp" and chomp["hp"] == "54%" and chomp["fraction"] == 0.536
    assert chomp["types"] == "Dragon/Ground" and chomp["item"] == "Choice Scarf" and chomp["ability"] == "Rough Skin"
    assert chomp["boosts"] == "atk+1 spe-1" and chomp["active"] and not chomp["fainted"]
    assert chomp["moves"] == [
        {"name": "Earthquake", "pp": [9, max_pp("earthquake")]},
        {"name": "Scale Shot", "pp": None},
    ]
    assert raichu["item"] == "" and raichu["moves"] == [
        {"name": "Zap Cannon", "pp": None}
    ]  # used up; PP: active only
    assert raichu["types"] == "Electric"  # the view gave none: CG's species table
    assert lucario["fainted"] and lucario["hp"] == "0%" and lucario["fraction"] == 0.0 and lucario["status"] == ""
    assert panels["active"] == {
        "name": "Garchomp",
        "hp": "54%",
        "fraction": 0.536,
        "status": "",
        "boosts": "atk+1 spe-1",
    }


def test_their_cards_the_unseen_ones_and_the_field() -> None:
    panels = view_panels(view(), EXPORT, names_of({"own_export": EXPORT}), BELIEF)
    basculegion, raichu = panels["foes"]
    assert raichu["hp"] == "1%" and raichu["status"] == "par" and raichu["item"] == "Raichunite Y"
    assert raichu["moves"] == ["Fake Out"] and basculegion["ability"] == "" and basculegion["types"] == "Water/Ghost"
    assert panels["opponent"]["name"] == "Raichu" and panels["unseen"] == ["Blaziken", "Meowscarada"]
    field = panels["field"]
    assert field["turn"] == 4 and field["weather"] == {"name": "Sun", "left": 3}
    assert field["fields"] == [{"name": "Trick Room", "left": 4}]
    assert field["ours"] == [
        {"name": "Stealth Rock", "layers": None, "left": None},
        {"name": "Reflect", "layers": None, "left": 1},  # past its default duration: at least one turn left
    ]
    assert field["theirs"] == [
        {"name": "Spikes", "layers": 2, "left": None},
        {"name": "Tailwind", "layers": None, "left": 3},
    ]
    text = battle_text(panels)
    assert (
        text.startswith("TURN 4   weather: Sun (3 left)") and "preview, not yet seen: Blaziken, Meowscarada" in text
    )
    assert "Earthquake 9/" in text and "Spikes x2" in text
    empty = field_view({}, {})
    assert empty == {"turn": None, "weather": None, "fields": [], "ours": [], "theirs": []}


def test_a_revealed_fact_replaces_its_belief() -> None:
    panels = view_panels(view(), EXPORT, {}, BELIEF)
    basculegion, raichu = panels["foes"]
    assert raichu["belief"]["item"] == []  # revealed: Raichunite Y
    assert [a["name"] for a in raichu["belief"]["ability"]] == ["Lightning Rod", "Static"]
    assert [m["name"] for m in raichu["belief"]["moves"]] == ["Thunderbolt", "Nasty Plot", "Surf", "Volt Switch"]
    assert basculegion["belief"]["item"][1] == {"id": "", "name": "No item", "p": 0.3}
    assert [m["name"] for m in basculegion["belief"]["moves"]] == ["Wave Crash"]  # Flip Turn was revealed
    gone = view_panels(view(opp=[mon("raichu", active=True, item="", ability="static")]), None, {}, BELIEF)
    assert gone["foes"][0]["belief"]["item"] == [] and gone["foes"][0]["belief"]["ability"] == []


def test_unseen_members_carry_their_belief_beside_the_names() -> None:
    panels = view_panels(view(), EXPORT, {}, BELIEF)
    assert panels["unseen"] == ["Blaziken", "Meowscarada"]
    blaziken, meowscarada = panels["unseen_belief"]
    assert (
        blaziken["name"] == "Blaziken"
        and blaziken["belief"]["brought"] == 0.62
        and blaziken["belief"]["lead"] is None
    )
    assert blaziken["belief"]["item"][0]["name"] == "Blazikenite" and meowscarada == {
        "name": "Meowscarada",
        "belief": None,
    }
    no_belief = view_panels(view(), EXPORT, {}, None)
    assert all(f["belief"] is None for f in no_belief["foes"]) and no_belief["unseen_belief"][0]["belief"] is None


def test_malformed_belief_entries_are_skipped() -> None:
    index = belief_index([*BELIEF, "x", {"species": ""}, {"species": "raichu", "brought": 0.0}])
    assert sorted(index) == ["basculegion", "blaziken", "garchomp", "raichu"] and index["raichu"]["brought"] == 1.0
    odd = belief_view(index["garchomp"], None, {})
    assert odd == {"brought": 0.5, "lead": None, "item": [], "ability": [], "moves": []}
    assert belief_index("nope") == {} and belief_view(None, None, {}) is None


def test_the_preview_shows_brought_and_lead(run: Path) -> None:
    for path in (run / "battle-1" / "decisions").glob("*.json"):
        path.unlink()
    state = RunReader(run).state(battle=1)
    assert state["decision"] is None and state["legacy"] is False and state["foes"] == []
    assert [m["name"] for m in state["your_team"]] == ["Hippowdon", "Metagross", "Primarina"]  # the picks sent
    assert state["unseen"] == ["Blaziken", "Meganium", "Gholdengo", "Swampert", "Hydreigon", "Glimmora"]
    leads = [u["belief"]["lead"] for u in state["unseen_belief"]]
    assert all(isinstance(x, float) for x in leads) and sum(leads) == pytest.approx(1.0, abs=1e-3)
    assert all(0 <= u["belief"]["brought"] <= 1 for u in state["unseen_belief"])


def test_the_newest_belief_is_used_and_the_preview_when_no_record_has_one(run: Path) -> None:
    decisions = run / "battle-1" / "decisions"
    for path in decisions.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["notes"]["cg"]["net"]
        path.write_text(json.dumps(data), encoding="utf-8")
    state = RunReader(run).state(battle=1)
    assert state["decision"]["notes"]["matrix"] is None and state["decision"]["notes"]["v"] is not None
    preview = json.loads((run / "battle-1" / "team_preview.json").read_text(encoding="utf-8"))
    leads = {e["species"]: e["lead"] for e in preview["cg"]["net"]["belief"]}
    assert all(u["belief"]["lead"] == leads[u["name"].lower()] for u in state["unseen_belief"])


def test_display_names_come_from_cg_tables_and_the_team_sheet() -> None:
    assert len(game_names()) > 1000 and shown("choicescarf", {}) == "Choice Scarf"
    assert shown("garchompmega", {}) == "Garchomp-Mega" and shown("unknownthing", {}) == "unknownthing"
    assert pretty("unknownthing") == "Unknownthing" and pretty("") == ""
    names = names_of({"own_export": "Mr. Mime @ Leftovers\n- Psychic\n", "own_names": ["Mr. Mime", 3]})
    assert shown("mrmime", names) == "Mr. Mime" and names_of({"own_export": 7, "own_names": "x"}) == {}
    assert max_pp("earthquake") and max_pp("notamove") is None


# Records of decision schemas before pcsa-decision/4 -------------------------------------------------


def _old_run(tmp_path: Path, schema: str, notes: dict, extra: dict) -> Path:
    run = tmp_path / f"old-{schema[-1]}"
    (run / "battle-1" / "decisions").mkdir(parents=True)
    lines = [
        {"event": "battle_start", "battle": 1, "ts": "2026-11-09T20:00:00+00:00"},
        {"event": "decision", "battle": 1, "kind": "action", "ts": "2026-11-09T20:00:10+00:00"},
    ]
    (run / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    record = {"schema": schema, "sequence": 0, "kind": "action", "turn": 3, "order": "/choose move earthquake",
              "degrade": "full", "legal": ["/choose move earthquake"], "notes": notes, **extra}  # fmt: skip
    (run / "battle-1" / "decisions" / "0000-action.json").write_text(json.dumps(record), encoding="utf-8")
    (run / "battle-1" / "battle_log.txt").write_text("=== Turn 1 ===\n[Move] Garchomp used Earthquake!\n", "utf-8")
    return run


@pytest.mark.parametrize(
    ("schema", "notes", "extra"),
    [
        (
            "pcsa-decision/2",
            {"rule": "neural", "p": 0.7, "v": 0.2, "matrix": {"rows": ["a"] * 11, "cols": ["b"] * 9}},
            {"request": {"side": {"pokemon": []}}, "protocol": ["|turn|3", "|-weather|SunnyDay"]},
        ),
        (
            "pcsa-decision/3",
            {"cg_ms": 16.0, "cg": {"action": 0, "player": "neural:x.pt", "model": {"value": 0.4}}},
            {},
        ),
    ],
)
def test_an_old_run_opens_with_what_it_has(tmp_path: Path, schema: str, notes: dict, extra: dict) -> None:
    state = RunReader(_old_run(tmp_path, schema, notes, extra)).state()
    assert state["legacy"] is True and state["battle"]["index"] == 1 and state["turn"] == 3
    assert state["decision"]["label"] == "Earthquake" and state["decision"]["notes"]["matrix"] is None
    assert state["decision"]["notes"]["v"] in (0.2, 0.4) and len(state["decisions"]) == 1
    assert state["your_team"] == [] and state["foes"] == [] and state["unseen_belief"] == []
    assert state["field"]["weather"] is None and [t["turn"] for t in state["log"]["turns"]] == [1]
    assert "records carry no battle view" in state["battle_state"]["battle_object"]
    json.dumps(state)


# The real replayed records: what they hold that the contract text did not say -----------------------


def real(name: str) -> dict:
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def panels_of(name: str) -> dict:
    data = real(name)
    cg = data["notes"]["cg"]
    return view_panels(cg["view"], data["own_export"], names_of(data), cg["net"]["belief"])


@pytest.mark.parametrize("name", ["battle-3/decisions/0002-action.json", "battle-3/decisions/0010-action.json"])
def test_a_revealed_life_orb_replaces_the_believed_mega_stone(name: str) -> None:
    belief = next(e for e in real(name)["notes"]["cg"]["net"]["belief"] if e["species"] == "salamence")
    assert belief["item"][0][0] == "salamencite"  # the network still believes the stone
    salamence = next(f for f in panels_of(name)["foes"] if f["name"] == "Salamence")
    assert salamence["item"] == "Life Orb" and salamence["belief"]["item"] == []


def test_mega_form_ids_in_rows_and_columns_are_named_by_the_base_species() -> None:
    data = real("battle-2/decisions/0011-action.json")
    net = data["notes"]["cg"]["net"]
    assert net["rows"][9] == "dragonitemega" and net["cols"][8] == "staraptormega"
    view = matrix_view(net, names_of(data), data["order"], data["notes"]["cg"]["action"])
    assert view is not None and view["row_labels"][9] == "Dragonite" and view["col_labels"][4] == "Staraptor"
    assert (
        view["row_labels"][4] == "Earthquake + Mega" and row_of_order("/choose switch Dragonite", view["rows"]) == 9
    )


def test_their_form_equal_to_the_species_reads_like_no_form() -> None:
    data = real("battle-2/decisions/0011-action.json")
    opp = data["notes"]["cg"]["view"]["opp"]
    assert opp[0]["species"] == opp[0]["form"] == "basculegion" and opp[1]["form"] == "staraptormega"
    foes = {f["name"]: f for f in panels_of("battle-2/decisions/0011-action.json")["foes"]}
    assert foes["Basculegion"]["types"] == "Water/Ghost" and foes["Staraptor"]["types"] == "Fighting/Flying"
    assert set(foes) == {"Basculegion", "Staraptor", "Metagross"}


def test_a_used_item_shows_as_gone_with_no_belief() -> None:
    swampert = next(f for f in panels_of("battle-1/decisions/0006-action.json")["foes"] if f["name"] == "Swampert")
    assert swampert["item"] == "" and swampert["belief"]["item"] == []  # its Sitrus Berry: "" in the view
    assert [a["name"] for a in swampert["belief"]["ability"]][0] == "Torrent"  # the ability is still believed
    hippowdon = panels_of("battle-2/decisions/0011-action.json")["your_team"][0]
    assert hippowdon["name"] == "Hippowdon" and hippowdon["item"] == ""  # ours too: not the team sheet's
