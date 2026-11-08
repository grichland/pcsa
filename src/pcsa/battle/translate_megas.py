"""Mega Evolution rows: the stone reacting, the evolution itself, and forms resolved later by an ability."""

from __future__ import annotations

from pcsa.battle import game
from pcsa.battle.log_rows import Event, RawRow
from pcsa.battle.moves import observed_form
from pcsa.battle.names import showdown_species
from pcsa.battle.translate import LEVEL, Member, Translator


def resolve_pending_mega(t: Translator, side: str, member: Member, ability: str) -> None:
    """A Mega whose form the text left ambiguous is settled when its ability is revealed."""
    pending = t.pending_megas.pop((side, member.name), None)
    if pending is None:
        return
    pending_row, mega_name = pending
    mega_form = game.mega_form_for_ability(member.species, ability)
    if mega_form is None:
        return t.unmapped(
            pending_row, f"ambiguous Mega form {mega_name!r}; ability {ability!r} did not pick one form"
        )
    form, stone = mega_form
    ident = t.ident(side, member)
    t.megas[ident] = stone
    note = {
        "kind": "mega_form_from_ability",
        "row": pending_row.row_id,
        "ability": ability,
        "form": form,
        "item": stone,
    }
    t.out.inferred.append(note)
    t.apply(f"|-item|{ident}|{stone}")
    t.apply(f"|detailschange|{ident}|{form}, L{LEVEL}")
    t.apply(f"|-mega|{ident}|{member.species}|{stone}")


def mega_reacting(t: Translator, row: RawRow, e: Event) -> None:
    if (member := t.require(row, e)) is None or e.side is None:
        return
    t.ensure_active(e.side, member, row)
    t.megas[t.ident(e.side, member)] = e.data["stone"]
    t.apply(f"|-item|{t.ident(e.side, member)}|{e.data['stone']}")


def mega(t: Translator, row: RawRow, e: Event) -> None:
    if (member := t.require(row, e)) is None or e.side is None:
        return
    t.ensure_active(e.side, member, row)
    ident = t.ident(e.side, member)
    result = showdown_species(e.data["mega"])
    stone = t.megas.get(ident, "")
    stone_form = game.mega_form_for_stone(member.species, stone)
    sprite_form = row.sprite.get("name") if row.sprite and row.sprite.get("status") == "matched" else None
    if result.status != "mapped" and not sprite_form and not stone_form:
        if result.status == "ambiguous" and not stone:
            t.pending_megas[(e.side, member.name)] = (row, e.data["mega"])
            return
        return t.unmapped(row, f"unknown Mega form {e.data['mega']!r} (stone {stone!r})")
    form = sprite_form or stone_form or result.species
    t.apply(f"|detailschange|{ident}|{form}, L{LEVEL}")
    t.apply(f"|-mega|{ident}|{member.species}|{stone}")


def mega_from_sprite(t: Translator, side: str, member: Member, row: RawRow) -> None:
    """A row sprite already in a Mega form the protocol has not reached: the Mega Evolution rows were not read
    (the start of the turn's log was not captured). The evolution, which happens before any move, is written
    before this row, as the stone reacting and the evolution rows would have written it."""
    mon = t.tracked(side, member)
    entry = game.species_entry(member.species)
    stone = entry.get("requiredItem")
    if mon is None or not stone or not game.is_mega(member.species) or game.is_mega(observed_form(t.battle, mon)):
        return
    if any(line.startswith(f"|-mega|{side}a: ") for line in t.out.lines):
        return  # one Mega Evolution per side and battle: a second Mega sprite is not this
    ident = t.ident(side, member)
    t.megas[ident] = stone
    t.out.inferred.append({"kind": "mega_from_sprite", "row": row.row_id, "form": member.species, "item": stone})
    t.apply(f"|-item|{ident}|{stone}")
    t.apply(f"|detailschange|{ident}|{member.species}, L{LEVEL}")
    t.apply(f"|-mega|{ident}|{member.species}|{stone}")
    t.pending_megas.pop((side, member.name), None)
