"""The one interface between the controller and whatever decides. Standard library only.

Terms follow poke-env and CG: Player, Order, choose_move, teampreview. Unread screen fields are None, never a
guess. Legal choices are the Offer (what the screen shows); Hints are the controller's reading of the log and
never narrow the Offer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

MOVE_ROWS = 4
TEAM_SIZE = 6
PICKS = 3
OUTCOMES = ("win", "loss", "opponent_forfeit", "unknown")


@dataclass(frozen=True)
class LogRow:
    """One row as the game's battle log shows it."""

    row_id: str  # "<turn>-<index>"
    turn: int
    index: int
    text: str
    sprite: str | None = None  # matched sprite form
    sprite_score: float | None = None
    kind: str | None = None  # the row parser's kind; None when unparsed


@dataclass(frozen=True)
class MoveOffer:
    row: int  # 0-3, top to bottom
    id: str | None  # read from the screen, or the sheet's move for this row when the list was not opened
    pp: tuple[int, int] | None = None  # None: not read
    disabled: bool | None = None  # None: not read


@dataclass(frozen=True)
class SwitchOffer:
    name: str  # switches are tapped by name: the party screen's order varies
    hp: tuple[int, int] | None = None  # None: not read


@dataclass(frozen=True)
class Offer:
    """The legal choices: only what the screen shows."""

    moves: tuple[MoveOffer, ...]
    moves_read: bool
    switches: tuple[SwitchOffer, ...]
    can_mega_evolve: bool | None  # the button when the move list was read
    forced: Literal["faint", "pivot"] | None = None
    leaving: str | None = None  # the member leaving on a pivot
    automatic: bool = False  # Struggle or recharge: the game takes no choice


@dataclass(frozen=True)
class Hints:
    """The controller's reading of the log. Advice only."""

    pp_counted: dict[str, tuple[int, int]] = field(default_factory=dict)  # move id -> (left, max) by the log
    mega_used: bool = False
    choice_locked: str | None = None  # Choice Scarf and the last move since entry
    encore: str | None = None
    taunted: bool = False
    first_turn_over: bool = False  # Fake Out, First Impression, Mat Block no longer work
    trapped: bool = False
    refused: tuple[int, ...] = ()  # move rows the game refused this turn


@dataclass(frozen=True)
class OwnTeam:
    """Our team as the game's team screens show it."""

    export: str  # Showdown export of all six, moves in set order
    names: tuple[str, ...]  # the six in team-select order (the export's order)
    picks: tuple[int, ...] | None = None  # indices into names, lead first; None before team preview
    unread: tuple[str, ...] = ()  # fields the sheet did not read, e.g. "Charizard.points"

    def brought(self) -> tuple[str, ...]:
        """The three brought, lead first: CG's switch slots."""
        return tuple(self.names[i] for i in self.picks or ())


@dataclass(frozen=True)
class Request:
    """One decision."""

    battle_tag: str  # stable across a restart
    request_id: int  # the decision's sequence number; an answer to an older id is dropped
    rows: tuple[LogRow, ...]  # the whole battle log so far
    log_quality: Literal["full", "gaps", "none"]
    team: OwnTeam
    opponent_preview: tuple[str | None, ...]  # the six species from the preview sprites
    offer: Offer
    hints: Hints
    deadline: float  # clock time by which the answer must be back
    events: tuple[dict, ...] = ()  # the battle's LIVE events so far, battle_start first
    live: dict | None = None  # the LIVE decision_request this decision stands for
    history: tuple[dict, ...] = ()  # earlier decisions the Player was asked: {events, decision_request, [fallback]}
    showdown_lines: tuple[str, ...] | None = None  # optional, controller-internal


@dataclass(frozen=True)
class TeamPreviewRequest:
    battle_tag: str
    team: OwnTeam
    opponent_preview: tuple[str | None, ...]
    deadline: float
    events: tuple[dict, ...] = ()  # the LIVE battle_start, when the team sheet was read


@dataclass(frozen=True)
class Order:
    """A move row (optionally with Mega Evolution) or a switch target by name."""

    kind: Literal["move", "switch"]
    row: int | None = None
    name: str | None = None
    mega: bool = False

    def __post_init__(self) -> None:
        if self.kind == "move":
            if self.row is None or not 0 <= self.row < MOVE_ROWS or self.name is not None:
                raise ValueError(f"a move order needs a row 0-{MOVE_ROWS - 1} and no name: {self}")
        elif self.kind == "switch":
            if not self.name or self.row is not None or self.mega:
                raise ValueError(f"a switch order needs a name, no row and no Mega: {self}")
        else:
            raise ValueError(f"unknown order kind {self.kind!r}")

    @staticmethod
    def move(row: int, mega: bool = False) -> Order:
        return Order("move", row=row, mega=mega)

    @staticmethod
    def switch(name: str) -> Order:
        return Order("switch", name=name)

    def label(self) -> str:
        return f"{'mega+' if self.mega else ''}move{self.row}" if self.kind == "move" else f"switch:{self.name}"


@dataclass(frozen=True)
class Answer:
    order: Order
    notes: dict = field(default_factory=dict)  # recorded verbatim


def valid_picks(picks: tuple[int, ...]) -> bool:
    return len(picks) == PICKS and len(set(picks)) == PICKS and all(0 <= i < TEAM_SIZE for i in picks)


class Player(Protocol):
    name: str

    def battle_start(self, battle_tag: str, team: OwnTeam) -> None: ...

    def teampreview(self, request: TeamPreviewRequest) -> tuple[tuple[int, int, int], dict]:
        """Three indices into team.names, lead first, and notes."""
        ...

    def choose_move(self, request: Request) -> Answer:
        """May be asked twice for the same decision (taps that did not go through); answer the same way."""
        ...

    def battle_end(
        self, battle_tag: str, outcome: str, events: tuple[dict, ...] = (), history: tuple[dict, ...] = ()
    ) -> None:
        """outcome is one of OUTCOMES; `events` the battle's whole LIVE list (`end` last) and `history` as in
        Request.history, so a Player can pass on what came after its last decision."""
        ...
