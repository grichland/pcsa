"""Battle-log rows -> Showdown protocol, applied to a poke-env Battle as they are written.

The Battle is the only state. A recognized row sprite decides species and form before the text
event is applied; preview sprites and names are the fallback. Rows that map to nothing are listed
as unmapped; informational rows (effectiveness, Protect blocking a move) write nothing.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from functools import lru_cache

from poke_env.battle import Battle
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.log_rows import Event, RawRow, parse_row, rows_from_texts
from pcsa.battle.moves import observed_form, reconcile_form

LEVEL = game.LEVEL
# Events that only happen once a turn has begun; they end Turn 1's opening send-outs.
TURN_ACTION_KINDS = frozenset(
    """move withdraw mega_reacting mega item cant trailing fail miss effectiveness hitcount knock_off prepare
    faint drag copyboost""".split()
)
INFORMATIONAL_KINDS = frozenset("effectiveness fail prepare cause trailing boost_blocked withdraw".split())
CONSUMED_ITEMS = frozenset(
    "Focus Sash|White Herb|Mental Herb|Power Herb|Mirror Herb|Weakness Policy|Booster Energy|Throat Spray|"
    "Room Service|Eject Button|Eject Pack|Red Card|Adrenaline Orb|Blunder Policy|Absorb Bulb|Cell Battery|"
    "Luminous Moss|Snowball|Grassy Seed|Electric Seed|Psychic Seed|Misty Seed".split("|")
)
CONSUMED_IDS = frozenset(to_id_str(v) for v in CONSUMED_ITEMS)
NAME_MATCH = 0.7


@lru_cache(maxsize=1)
def sprite_forms() -> frozenset[str]:
    """Forms that have their own battle-log sprite (the log reference names, read without the network)."""
    import json

    import numpy as np

    from pcsa.common import paths

    folder = paths.assets_dir() / "sprites"
    metadata = json.loads((folder / "sprite_net.json").read_text(encoding="utf-8"))
    with np.load(folder / metadata["references"]["log"]["file"], allow_pickle=False) as archive:
        return frozenset(to_id_str(name) for name in archive["names"].tolist())


def letters(text: str) -> str:
    return "".join(c for c in (text or "").lower() if c.isalpha())


def best_match(name: str, candidates) -> str | None:
    if not name:
        return None
    exact = [c for c in candidates if letters(c) == letters(name)]
    if exact:
        return exact[0]
    scored = sorted(
        ((difflib.SequenceMatcher(None, letters(name), letters(c)).ratio(), c) for c in candidates), reverse=True
    )
    return scored[0][1] if scored and scored[0][0] >= NAME_MATCH else None


@dataclass
class Member:
    name: str  # as the game writes it in the log
    species: str  # Showdown species for the details field
    max_hp: int | None = None  # ours only


@dataclass
class Translation:
    lines: list[str] = field(default_factory=list)
    unmapped: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    revealed_abilities: dict[str, str] = field(default_factory=dict)
    inferred: list[dict] = field(default_factory=list)
    illusions: dict[str, str] = field(default_factory=dict)  # send-out row id -> Illusion user
    unpreviewed: list[Member] = field(default_factory=list)


def has_illusion(species: str) -> bool:
    return "Illusion" in game.species_entry(species).get("abilities", {}).values()


class Translator:
    def __init__(
        self, battle: Battle, ours: list[Member], theirs: list[Member], *, disguises: dict[str, str] | None = None
    ) -> None:
        self.battle = battle
        self.disguises = dict(disguises or {})
        self.alias: dict[str, tuple[str, Member]] = {}  # side -> (shown name, Illusion user)
        self.sent_out: dict[str, str] = {}  # side -> row id of its latest send-out
        self.ours = {m.name: m for m in ours}
        self.theirs = {m.name: m for m in theirs}
        self.out = Translation()
        self.megas: dict[str, str] = {}  # identifier -> stone
        self.pending_megas: dict[tuple[str, str], tuple[RawRow, str]] = {}
        self.next_row: RawRow | None = None
        self.passed_substitute: set[str] = set()
        self.previous_event: Event | None = None

    # Writing -----------------------------------------------------------------------------------------
    def apply(self, line: str) -> None:
        self.out.lines.append(line)
        try:
            self.battle.parse_message(line.split("|"))
        except Exception as error:  # a line poke-env cannot apply is recorded, never fatal
            self.out.warnings.append(f"poke-env rejected {line!r}: {type(error).__name__}: {error}")

    def unmapped(self, row: RawRow, reason: str) -> None:
        self.out.unmapped.append({"row": row.row_id, "text": row.text, "reason": reason})

    # Identity ------------------------------------------------------------------------------------------
    def member(self, side: str | None, name: str | None) -> Member | None:
        if side is None:
            return None
        if (alias := self.alias.get(side)) and best_match(name or "", [alias[0]]):
            return alias[1]
        pool = self.ours if side == "p1" else self.theirs
        match = best_match(name or "", pool)
        return pool[match] if match else None

    def ident(self, side: str, member: Member) -> str:
        return f"{side}a: {member.name}"

    def require(self, row: RawRow, e: Event) -> Member | None:
        """The event's member, or None after recording the row as unmapped."""
        member = self.member(e.side, e.name)
        if member is None:
            self.unmapped(row, f"unknown {e.side} Pokemon {e.name!r}")
        return member

    def tracked(self, side: str, member: Member):
        team = self.battle.team if side == "p1" else self.battle.opponent_team
        return team.get(f"{side}: {member.name}")

    def side_of(self, name: str | None) -> tuple[str, Member] | None:
        """The side of a row that does not say it: the side with that name; in a mirror, the active one."""
        found = [(side, m) for side in ("p1", "p2") if (m := self.member(side, name))]
        if len(found) == 1:
            return found[0]
        active = [(side, m) for side, m in found if (mon := self.tracked(side, m)) is not None and mon.active]
        alive = [(side, m) for side, m in active if not getattr(self.tracked(side, m), "fainted", False)] or active
        return alive[0] if len(alive) == 1 else None

    def located(self, e: Event) -> tuple[str, Member] | None:
        """(side, member) for an event, using side_of for rows that do not name the side."""
        if e.side is None:
            return self.side_of(e.name)
        member = self.member(e.side, e.name)
        return (e.side, member) if member else None

    def sprite_fits(self, side: str, member: Member, sprite_name: str) -> bool:
        mon = self.tracked(side, member)
        if mon is None:
            return True
        current = observed_form(self.battle, mon)
        if current not in sprite_forms():
            return True
        return reconcile_form(current, sprite_name) == current

    def active(self, side: str) -> str | None:
        mon = self.battle.active_pokemon if side == "p1" else self.battle.opponent_active_pokemon
        if mon is None:
            return None
        for member in (self.ours if side == "p1" else self.theirs).values():
            if self.tracked(side, member) is mon:
                return self.ident(side, member)
        return None

    def other(self, side: str) -> str:
        return "p2" if side == "p1" else "p1"

    def hp_field(self, side: str, member: Member, value: int | None = None) -> str:
        mon = self.tracked(side, member)
        if side == "p1":
            maximum = member.max_hp or (mon.max_hp if mon is not None and mon.max_hp else None) or 100
            current = (
                value
                if value is not None
                else (mon.current_hp if mon is not None and mon.current_hp is not None else maximum)
            )
            return "0 fnt" if current <= 0 else f"{min(current, maximum)}/{maximum}"
        current = (
            value
            if value is not None
            else (mon.current_hp if mon is not None and mon.current_hp is not None else 100)
        )
        return "0 fnt" if current <= 0 else f"{current}/100"

    def ensure_active(self, side: str, member: Member, row: RawRow) -> None:
        """A Pokémon that acts is on the field; a missed send-out row becomes an inferred switch."""
        mon = self.tracked(side, member)
        if mon is not None and mon.active:
            return
        self.out.inferred.append({"row": row.row_id, "side": side, "name": member.name, "text": row.text})
        self.apply(f"|switch|{self.ident(side, member)}|{member.species}, L{LEVEL}|{self.hp_field(side, member)}")

    # Rows ------------------------------------------------------------------------------------------------
    def _side_from_next_row(self, e: Event) -> None:
        """Item and Ability rows do not say whose; the next row's subject does (needed in mirrors)."""
        if e.kind in ("item", "ability") and e.side is None and self.next_row is not None:
            after = parse_row(self.next_row)
            if after.side is not None and after.name and to_id_str(after.name) == to_id_str(e.name or ""):
                e.side = after.side

    def _sprite_candidates(self, sprite: dict, sides: list[str]) -> list[tuple[str, Member]]:
        dex = GenData.from_gen(9).pokedex
        base = to_id_str(sprite["base_name"])

        def same_base(m: Member) -> bool:
            return to_id_str(dex.get(to_id_str(m.species), {}).get("baseSpecies", m.species)) == base

        found = [(s, m) for s in sides for m in (self.ours if s == "p1" else self.theirs).values() if same_base(m)]
        if len(found) > 1:
            active = [(s, m) for s, m in found if (mon := self.tracked(s, m)) is not None and mon.active]
            alive = [(s, m) for s, m in active if not getattr(self.tracked(s, m), "fainted", False)]
            found = alive or active
        if len(found) > 1:
            found = [(s, m) for s, m in found if self.sprite_fits(s, m, sprite["name"])]
        return found

    def _apply_sprite(self, row: RawRow, e: Event) -> tuple[str, Member] | None:
        """Row-local sprite identity: the sprite decides species and form; returns the member it named."""
        sprite = row.sprite
        if (
            sprite
            and e.side != "p1"
            and (alias := self.alias.get("p2"))
            and best_match(sprite.get("base_name") or "", [alias[0]])
        ):
            sprite = None  # the disguise's icon; the name resolves to the Illusion user
        if not (sprite and sprite.get("status") == "matched" and e.name):
            return None
        found = self._sprite_candidates(sprite, [e.side] if e.side is not None else ["p1", "p2"])
        if len(found) != 1:
            self.unmapped(row, f"cannot assign sprite {sprite['name']!r} to one roster member")
            return "skip"  # type: ignore[return-value]
        side, member = found[0]
        name = sprite["name"]
        if (mon := self.tracked(side, member)) is not None:
            form = reconcile_form(observed_form(self.battle, mon), name)
            name = game.species_entry(form).get("name", name)
        if e.kind == "form_change" and e.data.get("ability") == "Disguise":
            name = e.data["form"]
        self.out.inferred.append(
            {
                "kind": "row_sprite_identity",
                "row": row.row_id,
                "text_name": e.name,
                "form": name,
                "side": side,
                "sprite_form": sprite["name"],
                "score": sprite.get("score"),
            }
        )
        member.species = name
        e.name, e.side = member.name, side
        if e.kind == "form_change":
            e.data["form"] = name
        return side, member

    def row(self, row: RawRow) -> None:
        from pcsa.battle.translate_events import dispatch
        from pcsa.battle.translate_megas import mega_from_sprite

        e = parse_row(row)
        self._side_from_next_row(e)
        located = self._apply_sprite(row, e)
        if located == "skip":
            return
        if located is not None and e.kind != "mega":
            mega_from_sprite(self, *located, row)
        dispatch(self, row, e)
        if located is not None:
            side, member = located
            mon = self.tracked(side, member)
            if mon is not None:
                name = reconcile_form(observed_form(self.battle, mon), member.species)
                if observed_form(self.battle, mon) != to_id_str(name):
                    self.apply(f"|detailschange|{self.ident(side, member)}|{name}, L{LEVEL}")
                self.pending_megas.pop((side, member.name), None)

    def turns(self, rows_by_turn: dict[int, list[str]], decision_turn: int, *, sprites_by_turn=None) -> None:
        """Every collected turn in order, then `|turn|` for the turn being decided now."""
        for turn in sorted(rows_by_turn):
            sprites = (sprites_by_turn or {}).get(turn)
            if sprites is not None and len(sprites) != len(rows_by_turn[turn]):
                raise ValueError(f"turn {turn}: sprite count does not match text row count")
            raw = rows_from_texts(turn, rows_by_turn[turn], list(sprites) if sprites else None)
            if turn == 1:  # opening send-outs (and entry abilities or weather) happen before turn 1 starts
                split = next((i for i, r in enumerate(raw) if parse_row(r).kind in TURN_ACTION_KINDS), len(raw))
                self._rows(raw[:split])
                if split < len(raw):
                    self.apply("|turn|1")
                raw = raw[split:]
            elif self.battle.turn < turn:
                self.apply(f"|turn|{turn}")
            self._rows(raw)
        if self.battle.turn < decision_turn:
            self.apply(f"|turn|{decision_turn}")
        for row, mega_name in self.pending_megas.values():
            self.unmapped(row, f"ambiguous Mega form {mega_name!r}; no disambiguating ability was observed")
        self.pending_megas.clear()

    def _rows(self, raw: list[RawRow]) -> None:
        for index, r in enumerate(raw):
            self.next_row = raw[index + 1] if index + 1 < len(raw) else None
            self.row(r)
        self.next_row = None
