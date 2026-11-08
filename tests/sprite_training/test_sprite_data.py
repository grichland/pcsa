"""Labels for the sprite network: species named in log text, forms, and the opponent's sent-out species."""

from __future__ import annotations

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


def test_both_domains_share_showdown_form_ids() -> None:
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


def test_the_opponents_sent_out_forms_come_from_the_showdown_log(tmp_path) -> None:
    log = tmp_path / "showdown.log"
    log.write_text(
        "|switch|p1a: Tauros|Tauros, L50|100/100\n"
        "|switch|p2a: Tauros|Tauros-Paldea-Blaze, L50|100/100\n"
        "|drag|p2a: Ninetales|Ninetales-Alola, L50|100/100\n",
        encoding="utf-8",
    )
    sent = sd.opponent_sent(log, sd.Classes.from_sources(SOURCES))
    assert sent == {"tauros": {"taurospaldeablaze"}, "ninetales": {"ninetalesalola"}}
