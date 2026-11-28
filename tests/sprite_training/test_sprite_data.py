"""Labels for the sprite network: species named in log text, forms, and the opponent's sent-out species from either
label file a battle folder can hold."""

from __future__ import annotations

import json

import numpy as np

from pcsa.phone.sprites import SIZE
from pcsa.sprite_training import sprite_data as sd
from pcsa.sprite_training.sprite_baseline import SourceIndex


def source(names: list[str], bases: list[str]) -> SourceIndex:
    n = len(names)
    return SourceIndex(
        names, bases, np.zeros((n, SIZE, SIZE), np.float32), np.zeros((n, SIZE, SIZE, 3), np.float32), np.ones(n)
    )


SOURCES = {
    "preview": source(["Alolan Ninetales", "Farfetch'd", "Tauros"], ["Ninetales", "Farfetch'd", "Tauros"]),
    "log": source(
        ["Ninetales", "Ninetales-Alola", "Farfetch’d", "Tauros", "Tauros-Paldea-Blaze", "Kommo-o"],
        ["Ninetales", "Ninetales", "Farfetch’d", "Tauros", "Tauros", "Kommo-o"],
    ),
}
CLASSES = sd.Classes.from_sources(
    {**SOURCES, "megas": source(["Garchomp", "Garchomp-Mega", "Kommo-o"], ["Garchomp", "Garchomp", "Kommo-o"])}
)


def test_both_domains_share_form_ids() -> None:
    classes = sd.Classes.from_sources(SOURCES)
    assert classes.base("ninetalesalola") == "ninetales"
    assert sorted(classes.forms_of["tauros"]) == ["tauros", "taurospaldeablaze"]
    assert "farfetchd" in classes.ids and len([c for c in classes.ids if c.startswith("farfetch")]) == 1


def test_log_text_names_the_species_whatever_the_apostrophe_or_prefix() -> None:
    lexicon = sd.Lexicon(SOURCES)
    assert lexicon.species("Move: The opposing Tauros used Bitter Malice!") == {"tauros"}
    assert lexicon.species("HP Lost: Farfetch’d’s HP: 100% → 42%") == {"farfetchd"}
    assert lexicon.species("Ability: Kommo-o’s Bulletproof") == {"kommoo"}
    assert lexicon.species("It’s super effective!") == set()


def showdown_battle(folder, lines: list[str]):
    folder.mkdir(parents=True)
    (folder / "showdown.log").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return folder


def events_battle(folder, events: list[dict]):
    folder.mkdir(parents=True)
    start = {"v": "0.7", "battle": "b", "type": "battle_start", "own_team": [], "opp_preview": []}
    body = {"battle": "b", "events": [start, *({"v": "0.7", "battle": "b", **e} for e in events)]}
    (folder / "events.json").write_text(json.dumps(body), encoding="utf-8")
    return folder


def test_the_opponents_sent_out_forms_come_from_the_showdown_log(tmp_path) -> None:
    folder = showdown_battle(
        tmp_path / "battle-1",
        [
            "|switch|p1a: Tauros|Tauros, L50|100/100",
            "|switch|p2a: Tauros|Tauros-Paldea-Blaze, L50|100/100",
            "|drag|p2a: Ninetales|Ninetales-Alola, L50|100/100",
        ],
    )
    sent, labels = sd.opponent_sent(folder, sd.Classes.from_sources(SOURCES))
    assert labels == "showdown"
    assert sent == {"tauros": {"taurospaldeablaze"}, "ninetales": {"ninetalesalola"}}


def test_events_and_showdown_log_give_the_same_labels_for_the_same_battle(tmp_path) -> None:
    old = showdown_battle(
        tmp_path / "old",
        [
            "|switch|p1a: Kommo-o|Kommo-o, L50|100/100",
            "|switch|p2a: Farfetch’d|Farfetch’d, L50|100/100",
            "|switch|p2a: Ninetales|Ninetales-Alola, L50|100/100",
            "|drag|p2a: Kommo-o|Kommo-o, L50|100/100",
        ],
    )
    new = events_battle(
        tmp_path / "new",
        [
            {"k": "turn", "n": 1},
            {"k": "enter", "side": "own", "mon": "kommoo"},
            {"k": "enter", "side": "opp", "mon": "farfetchd"},
            {"k": "withdraw", "side": "opp", "mon": "farfetchd"},
            {"k": "enter", "side": "opp", "mon": "ninetalesalola"},
            {"k": "enter", "side": "opp", "mon": "kommoo", "forced": True},
        ],
    )
    from_log, from_events = sd.opponent_sent(old, CLASSES), sd.opponent_sent(new, CLASSES)
    sent = {"farfetchd": {"farfetchd"}, "ninetales": {"ninetalesalola"}, "kommoo": {"kommoo"}}
    assert from_log == (sent, "showdown")
    assert from_events == (from_log[0], "events")


def test_an_event_without_the_form_allows_every_form_but_the_megas(tmp_path) -> None:
    folder = events_battle(
        tmp_path / "battle-1",
        [
            {"k": "enter", "side": "opp", "mon": "tauros"},
            {"k": "enter", "side": "opp", "mon": "garchomp"},
            {"k": "mega", "side": "opp", "mon": "garchomp", "form": "garchompmega"},
            {"k": "enter", "side": "own", "mon": "ninetales"},
        ],
    )
    sent, _ = sd.opponent_sent(folder, CLASSES)
    assert sent == {"tauros": {"tauros", "taurospaldeablaze"}, "garchomp": {"garchomp"}}


def test_events_take_precedence_and_a_folder_with_neither_file_has_no_labels(tmp_path) -> None:
    both = events_battle(tmp_path / "both", [{"k": "enter", "side": "opp", "mon": "kommoo"}])
    (both / "showdown.log").write_text("|switch|p2a: Tauros|Tauros, L50|100/100\n", encoding="utf-8")
    assert sd.opponent_sent(both, CLASSES) == ({"kommoo": {"kommoo"}}, "events")
    (tmp_path / "none").mkdir()
    assert sd.opponent_sent(tmp_path / "none", CLASSES) == ({}, "")


def test_save_and_load_keep_the_sent_out_species(tmp_path) -> None:
    crops = sd.Crops()
    variants = [sd.sprites.Feature(np.ones((SIZE, SIZE), np.float32), np.zeros((SIZE, SIZE, 3), np.float32), 1.0)]
    sd._add(crops, "preview", "run/battle-1", [*variants, None], "tauros", {"tauros"}, {"slot": 0}, 0)
    battles = [sd.Battle("run/battle-1", {"tauros": {"tauros", "taurospaldeablaze"}}, "events")]
    sd.save(crops, battles, tmp_path / "data.npz")
    data = sd.load(tmp_path / "data.npz")
    assert data.species == ["tauros"] and data.allowed == [["tauros"]] and list(data.variant[0]) == [0, 1]
    assert data.battles == {"run/battle-1": {"tauros": {"tauros", "taurospaldeablaze"}}}


def _cards(species: list[str], scores: list[float], sent: dict[str, set[str]]) -> sd.Dataset:
    n = len(species)
    return sd.Dataset(
        np.zeros(n, np.int8), np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)),
        np.arange(n, dtype=np.int8), ["b"] * n, list(species), [[s] if s else [] for s in species],
        [{"top": [["X", s]]} for s in scores], {"b": sent},
    )  # fmt: skip


def test_the_one_unsure_card_takes_the_one_sent_out_species_no_card_claims() -> None:
    sent = {"garchomp": {"garchomp"}, "milotic": {"milotic"}, "floette": {"floetteeternal"}}
    data = _cards(["", "garchomp", "milotic", "", "", ""], [0.34, 0.99, 0.96, 0.99, 0.94, 0.98], sent)
    assert sd.by_elimination(data) == 1
    assert data.species[0] == "floette" and data.allowed[0] == ["floetteeternal"]


def test_two_unsure_cards_or_no_free_species_label_nothing() -> None:
    sent = {"garchomp": {"garchomp"}, "milotic": {"milotic"}, "floette": {"floetteeternal"}}
    data = _cards(["", "garchomp", "milotic", "", "", ""], [0.34, 0.99, 0.96, 0.45, 0.94, 0.98], sent)
    assert sd.by_elimination(data) == 0
    data = _cards(["floette", "garchomp", "milotic", "", "", ""], [0.9, 0.99, 0.96, 0.4, 0.94, 0.98], sent)
    assert sd.by_elimination(data) == 0
