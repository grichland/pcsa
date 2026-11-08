"""Phone requests must not reverse observed form changes; HP-only rows never cure a status."""

import copy

import pytest
from poke_env.battle import Battle
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.adapter import current_request
from pcsa.battle.moves import observed_form
from pcsa.battle.state import LOGGER, USERNAME, replay
from pcsa.battle.stats import hp_value, stat_value
from tests.battle.conftest import PRELUDE


def request_for(species: str, name: str = "member") -> dict:
    entry = game.species_entry(to_id_str(species))
    points = {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
    stats = {
        s: hp_value(base, points[s]) if s == "hp" else stat_value(base, points[s], 100)
        for s, base in entry["baseStats"].items()
    }
    ability = to_id_str(entry["abilities"]["0"])
    member = {
        "ident": f"p1: {name}",
        "details": f"{species}, L50",
        "condition": f"{stats['hp']}/{stats['hp']}",
        "active": False,
        "stats": stats,
        "moves": ["protect"],
        "item": "",
        "baseAbility": ability,
        "ability": ability,
    }
    return {"side": {"id": "p1", "name": "me", "pokemon": [member]}}


def protocol_for(base: str, form: str, *, return_as_base: bool = False) -> list[str]:
    return [
        *PRELUDE,
        f"|switch|p1a: member|{base}, L50|100/100",
        f"|detailschange|p1a: member|{form}, L50",
        "|switch|p1a: bench|Primarina, L50|100/100",
        f"|switch|p1a: member|{base if return_as_base else form}, L50|100/100",
    ]


@pytest.mark.parametrize("status", ["par", "tox", "brn"])
def test_hp_only_rows_do_not_cure_a_status(status: str) -> None:
    lines = [
        *PRELUDE,
        "|switch|p2a: member|Dragonite, L50|100/100",
        f"|-status|p2a: member|{status}",
        "|-damage|p2a: member|63/100",
        "|-heal|p2a: member|80/100",
        "|switch|p2a: bench|Primarina, L50|100/100",
        "|switch|p2a: member|Dragonite, L50|80/100",
    ]
    assert replay(lines, None).opponent_team["p2: member"].status.name.lower() == status
    assert (
        replay([*lines, f"|-curestatus|p2a: member|{status}", "|-heal|p2a: member|90/100"], None)
        .opponent_team["p2: member"]
        .status
        is None
    )


def test_item_ocr_is_normalized_only_when_unambiguous() -> None:
    assert game.canonical_item("Rockv Helmet") == "rockyhelmet"
    assert game.canonical_item("Sitrus Berrv") == "sitrusberry"
    assert game.canonical_item("unreadable item") is None
    assert game.canonical_item("Life Orb") == "Life Orb"


def test_observed_stones_resolve_all_unambiguous_mega_variants() -> None:
    checked = 0
    for entry in game.dex().values():
        stone = entry.get("requiredItem")
        if "Mega" not in entry.get("forme", "") or not stone:
            continue
        form = game.mega_form_for_stone(entry["name"], stone)
        if form is not None:
            assert form == entry["name"]
            checked += 1
    assert checked > 80
    prefix = [
        *PRELUDE,
        "|switch|p2a: Lucario|Lucario, L50|100/100",
        "|-item|p2a: Lucario|Lucarionite Z",
        "|detailschange|p2a: Lucario|Lucario-Mega, L50",
        "|-mega|p2a: Lucario|Lucario|Lucarionite Z",
    ]
    battle = replay(prefix, None)
    mon = battle.opponent_team["p2: Lucario"]
    assert observed_form(battle, mon) == "lucariomegaz"
    assert mon.ability == to_id_str(game.species_entry("lucariomegaz")["abilities"]["0"])
    assert not battle.replay_errors


def test_unknown_field_end_is_recorded_without_inventing_a_start() -> None:
    battle = replay([*PRELUDE, "|-fieldend|move: Grassy Terrain"], None)
    assert not battle.fields and not battle.replay_errors
    assert len(battle.compatibility_repairs) == 1


@pytest.mark.parametrize("side", ["p1", "p2"])
def test_base_details_and_switch_cannot_restore_a_busted_disguise(side: str) -> None:
    lines = [*PRELUDE, f"|switch|{side}a: member|Mimikyu, L50|100/100", "|turn|1"]
    battle = replay(lines, None)
    team = battle.team if side == "p1" else battle.opponent_team
    assert observed_form(battle, team[f"{side}: member"]) == "mimikyu"
    lines += [
        f"|-activate|{side}a: member|ability: Disguise",
        f"|detailschange|{side}a: member|Mimikyu-Busted, L50",
        f"|detailschange|{side}a: member|Mimikyu, L50",
        f"|switch|{side}a: bench|Primarina, L50|100/100",
        f"|switch|{side}a: member|Mimikyu, L50|88/100",
    ]
    battle = replay(lines, request_for("Mimikyu") if side == "p1" else None)
    team = battle.team if side == "p1" else battle.opponent_team
    assert observed_form(battle, team[f"{side}: member"]) == "mimikyubusted"


def test_all_champions_mega_forms_keep_form_ability_and_current_stats() -> None:
    checked = 0
    for entry in game.dex().values():
        if "Mega" not in entry.get("forme", ""):
            continue
        base, form = entry["baseSpecies"], entry["name"]
        request = request_for(base)
        original = copy.deepcopy(request)
        lines = protocol_for(base, form)
        battle = replay(lines, None)
        normalized = current_request(battle, request)
        member = normalized["side"]["pokemon"][0]
        assert member["details"] == f"{form}, L50", form
        assert member["ability"] == to_id_str(entry["abilities"]["0"]) == member["baseAbility"]
        assert member["stats"]["atk"] == stat_value(entry["baseStats"]["atk"], 32, 100)
        assert request == original
        assert current_request(battle, normalized) == normalized
        restored = replay(lines, request).team["p1: member"]
        assert restored.species == entry["id"] and restored.ability == member["ability"]
        returned = replay(protocol_for(base, form, return_as_base=True), request).team["p1: member"]
        assert returned.ability == member["ability"]
        checked += 1
    assert checked > 40


def test_base_species_sendout_does_not_undo_a_persistent_mega() -> None:
    battle = replay(protocol_for("Raichu", "Raichu-Mega-Y", return_as_base=True), request_for("Raichu"))
    assert battle.team["p1: member"].species == "raichumegay"
    assert battle.team["p1: member"].ability == "noguard"


def test_temporary_ability_takes_priority_over_the_mega_default() -> None:
    request = request_for("Raichu")
    lines = [*protocol_for("Raichu", "Raichu-Mega-Y"), "|-ability|p1a: member|Levitate"]
    member = current_request(replay(lines, None), request)["side"]["pokemon"][0]
    assert (member["baseAbility"], member["ability"]) == ("noguard", "levitate")
    assert replay(lines, request).team["p1: member"].ability == "levitate"


def test_base_form_request_is_unchanged_without_an_observed_mega() -> None:
    request = request_for("Raichu")
    assert current_request(replay(protocol_for("Raichu", "Raichu")[:5], None), request) == request


ROUND = [
    *PRELUDE[:3],
    "|teamsize|p1|3",
    "|teamsize|p2|3",
    "|start",
    "|switch|p1a: Ditto|Ditto, L50|155/155",
    "|switch|p2a: Snorlax|Snorlax, L50|100/100",
    "|turn|1",
    "|move|p1a: Ditto|Round|p2a: Snorlax|[from]move: Round",
    "|-damage|p2a: Snorlax|80/100",
    "|turn|2",
]


def test_replay_skips_a_line_poke_env_cannot_apply_and_keeps_the_rest() -> None:
    raw = Battle("raw", USERNAME, LOGGER, 9)
    raw.player_role = "p1"
    with pytest.raises(KeyError):
        for line in ROUND:
            raw.parse_message(line.split("|"))
    battle = replay(ROUND, None)
    assert len(battle.replay_errors) == 1 and "KeyError" in battle.replay_errors[0]
    assert battle.turn == 2 and battle.opponent_active_pokemon.current_hp == 80
