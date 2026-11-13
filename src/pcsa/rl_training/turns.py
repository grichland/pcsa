# From research/pcsa-rl/rl2/turns.py (sha256 3f6c7577066d, 2026-11-10 20:44); keep diffable.
"""One pass over a battle protocol: what each side did in each turn.

Used for the last-turn token (the previous turn's record) and the opponent-action labels (the record of the turn a
decision was made in). A turn's segment is everything after `|turn|t` up to `|turn|t+1`; the decision made while
`battle.turn == t` (not a forced switch) acts in segment t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from poke_env.data.normalize import to_id_str

FROM = "[from]"
REVEAL_ITEM = ("-item", "-enditem")
REVEAL_ABILITY = ("-ability",)


@dataclass
class SideTurn:
    """One side's turn."""

    action: str = ""  # "move" | "switch" | "" (no choice of its own: fainted, recharging, dragged only)
    move: str = ""  # move id of its first own move
    mega: bool = False  # Mega Evolved this turn
    switch_to: str = ""  # species id switched in by choice at the start of the turn
    moved_first: bool = False
    damage_taken: float = 0.0  # HP fraction lost by its members this turn (direct and indirect)
    fainted: int = 0
    revealed_item: bool = False
    revealed_ability: bool = False


@dataclass
class TurnRecord:
    turn: int
    sides: dict[str, SideTurn] = field(default_factory=lambda: {"p1": SideTurn(), "p2": SideTurn()})


def _side(ident: str) -> str:
    return ident[:2]


def _species(details: str) -> str:
    return to_id_str(details.split(",", 1)[0])


def _fraction(text: str) -> float | None:
    token = text.split(" ", 1)[0]
    if token == "0" or token.endswith("fnt"):
        return 0.0
    if "/" not in token:
        return None
    a, b = token.split("/", 1)
    try:
        return float(a) / float(b) if float(b) > 0 else None
    except ValueError:
        return None


class _Reader:
    def __init__(self) -> None:
        self.turns: dict[int, TurnRecord] = {}
        self.turn = 0
        self.hp: dict[str, float] = {}
        self.first_mover: str | None = None

    def record(self) -> TurnRecord:
        return self.turns.setdefault(self.turn, TurnRecord(self.turn))

    def line(self, parts: list[str]) -> None:
        kind = parts[1] if len(parts) > 1 else ""
        if kind == "turn" and len(parts) > 2:
            self.turn = int(parts[2]) if parts[2].isdigit() else self.turn + 1
            self.first_mover = None
            self.record()
            return
        if len(parts) < 3 or parts[2][:2] not in ("p1", "p2"):
            return
        side = self.record().sides[_side(parts[2])]
        called = any(p.startswith(FROM) for p in parts[3:])
        if kind == "move" and len(parts) > 3 and not called:
            self._move(parts, side)
        elif kind == "switch":
            self._switch(parts, side)
        elif kind == "-mega":
            side.mega = True
        elif kind in ("-damage", "-heal", "-sethp"):
            self._hp(parts, side)
        elif kind == "faint":
            side.fainted += 1
        if kind in REVEAL_ITEM or any(p.startswith(FROM + " item:") for p in parts[3:]):
            side.revealed_item = True
        if kind in REVEAL_ABILITY or any(p.startswith(FROM + " ability:") for p in parts[3:]):
            side.revealed_ability = True

    def _move(self, parts: list[str], side: SideTurn) -> None:
        if not side.action:
            side.action, side.move = "move", to_id_str(parts[3])
        if self.first_mover is None:
            self.first_mover = _side(parts[2])
            side.moved_first = True

    def _switch(self, parts: list[str], side: SideTurn) -> None:
        if len(parts) > 3:
            hp = _fraction(parts[4]) if len(parts) > 4 else 1.0
            if hp is not None:  # an unreadable HP leaves the last known one (as a damage line does)
                self.hp[parts[2][:2] + ": " + parts[2].split(": ", 1)[-1]] = hp
        if not side.action and not side.fainted and self.turn > 0:  # chosen at turn start (not a faint replacement)
            side.action, side.switch_to = "switch", _species(parts[3]) if len(parts) > 3 else ""

    def _hp(self, parts: list[str], side: SideTurn) -> None:
        key = parts[2][:2] + ": " + parts[2].split(": ", 1)[-1]
        new = _fraction(parts[3]) if len(parts) > 3 else None
        if new is None:
            return
        old = self.hp.get(key)
        if old is not None and new < old:
            side.damage_taken += old - new
        self.hp[key] = new


def read(protocol: list[list[str]]) -> dict[int, TurnRecord]:
    """turn -> what each side did in it, from a list of split protocol lines (battle._replay_data)."""
    reader = _Reader()
    for parts in protocol:
        reader.line(parts)
    return reader.turns


def from_text(text: str) -> dict[int, TurnRecord]:
    return read([line.split("|") for line in text.splitlines() if line.startswith("|")])
