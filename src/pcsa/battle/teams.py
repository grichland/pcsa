"""Our team as the battle knows it: species resolved from the sheet, stat points, Showdown export."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from poke_env.teambuilder import TeambuilderPokemon

from pcsa.battle import game

LEVEL = game.LEVEL
STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")
EXPORT_STATS = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}


@dataclass(frozen=True)
class OwnMember:
    roster_index: int
    name: str  # as the game writes it ("Rotom" for Rotom-Wash)
    species: str  # Showdown species, form resolved
    ability: str | None
    item: str | None
    moves: tuple[str | None, ...]  # by slot, None where unread
    stats: dict[str, int] | None
    points: dict[str, int] | None
    nature: str | None

    def record(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "roster_index",
                "name",
                "species",
                "ability",
                "item",
                "moves",
                "stats",
                "points",
                "nature",
            )
        }


@dataclass(frozen=True)
class Team:
    name: str | None
    members: tuple[OwnMember, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.members)

    def record(self) -> dict:
        return {"name": self.name, "members": [m.record() for m in self.members]}


def teambuilder_team(team: Team) -> list[TeambuilderPokemon]:
    """Six TeambuilderPokemon in roster order; Champions stat points go in the EVs field."""
    out = []
    for m in team.members:
        out.append(
            TeambuilderPokemon(
                nickname=m.name if m.name != m.species else None,
                species=m.species,
                item=m.item,
                ability=m.ability,
                moves=[move for move in m.moves if move],
                nature=m.nature,
                evs=[(m.points or {}).get(s, 0) for s in STAT_ORDER] if m.points else None,
                level=LEVEL,
            )
        )
    return out


def export_text(team: list[TeambuilderPokemon]) -> str:
    """Showdown export format."""
    blocks = []
    for mon in team:
        name = f"{mon.nickname} ({mon.species})" if mon.nickname else str(mon.species)
        lines = [name + (f" @ {mon.item}" if mon.item else "")]
        if mon.ability:
            lines.append(f"Ability: {mon.ability}")
        lines.append(f"Level: {mon.level}")
        if mon.evs and any(mon.evs):
            lines.append(
                "EVs: "
                + " / ".join(
                    f"{value} {EXPORT_STATS[stat]}" for stat, value in zip(STAT_ORDER, mon.evs, strict=True) if value
                )
            )
        if mon.nature:
            lines.append(f"{mon.nature} Nature")
        lines += [f"- {move}" for move in mon.moves]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def team_from_record(record: dict) -> Team:
    """Inverse of Team.record(); keys OwnMember no longer has (older records) are ignored."""
    known = {f.name for f in fields(OwnMember)}
    members = []
    for m in record["members"]:
        kept: dict[str, Any] = {k: v for k, v in m.items() if k in known}
        kept["moves"] = tuple(m["moves"])
        members.append(OwnMember(**kept))
    return Team(record["name"], tuple(members))
