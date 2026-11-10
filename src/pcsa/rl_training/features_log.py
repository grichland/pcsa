# From research/pcsa-rl/rl/features_log.py (sha256 6bc4a7ebf2a8, 2026-11-08 20:13); keep diffable.
"""One pass over a Battle's protocol: per-member evidence and per-side tendencies for the feature rows/field."""

from __future__ import annotations

from dataclasses import dataclass, field

from poke_env.data.normalize import to_id_str

from pcsa.players.opponent import move_object

PROTECTS = frozenset(
    [
        "protect",
        "detect",
        "kingsshield",
        "spikyshield",
        "banefulbunker",
        "silktrap",
        "burningbulwark",
        "obstruct",
        "endure",
    ]
)


@dataclass
class Member:
    """What the log showed about one member (key 'p1: Name')."""

    entries: int = 0
    entered_turn: int = -1
    lead: bool = False
    biggest_hit: float = 0.0  # largest single drop as a fraction of max HP
    moved_first: int = 0
    moved_second: int = 0
    hp: float | None = None  # last fraction seen


@dataclass
class Turn:
    """One side's actions in one turn."""

    switched: bool = False
    attacked: bool = False
    status_move: bool = False
    protected: bool = False
    setup: bool = False
    mega: bool = False
    dealt: float = 0.0
    fainted: bool = False


@dataclass
class Side:
    turns: int = 0
    switches: int = 0
    protects: int = 0
    setups: int = 0
    attacks: int = 0
    faints: int = 0
    last: Turn = field(default_factory=Turn)
    current: Turn = field(default_factory=Turn)
    last_switch_turn: int = 0


@dataclass
class Log:
    members: dict[str, Member]
    sides: dict[str, Side]


def key(ident: str) -> str:
    """'p1a: Name' -> 'p1: Name'."""
    side, _, name = ident.partition(": ")
    return f"{side[:2]}: {name}"


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


def _is_setup(move) -> bool:
    boosts = move.boosts or {}
    self_boost = move.self_boost or {}
    target_self = str(getattr(move.target, "name", "")).upper() in ("SELF", "ADJACENT_ALLY_OR_SELF")
    gains = sum(v for v in (boosts if target_self else {}).values() if v > 0)
    gains += sum(v for v in self_boost.values() if v > 0) if move.category.name == "STATUS" else 0
    return gains > 0


class _Reader:
    def __init__(self) -> None:
        self.log = Log({}, {"p1": Side(), "p2": Side()})
        self.turn = 0
        self.order: list[str] = []
        self.last_mover: str | None = None

    def member(self, k: str) -> Member:
        return self.log.members.setdefault(k, Member())

    def end_turn(self) -> None:
        if len(self.order) == 2 and self.order[0][:2] != self.order[1][:2]:
            self.member(self.order[0]).moved_first += 1
            self.member(self.order[1]).moved_second += 1
        for side in self.log.sides.values():
            side.turns += self.turn > 0
            side.last, side.current = side.current, Turn()
        self.order = []

    def switch(self, parts: list[str]) -> None:
        k = key(parts[2])
        m = self.member(k)
        m.entries += 1
        m.entered_turn = self.turn
        m.lead = m.lead or self.turn == 0
        m.hp = _fraction(parts[4]) if len(parts) > 4 else m.hp
        side = self.log.sides[k[:2]]
        if parts[1] == "switch" and self.turn > 0:
            side.current.switched = True
            side.switches += 1
            side.last_switch_turn = self.turn

    def move(self, parts: list[str]) -> None:
        k = key(parts[2])
        called = any(p.startswith("[from]") for p in parts[4:])
        move = move_object(to_id_str(parts[3])) if len(parts) > 3 else None
        self.last_mover = k
        if called or move is None:
            return
        if k not in self.order:
            self.order.append(k)
        side = self.log.sides[k[:2]]
        cur = side.current
        if move.id in PROTECTS:
            cur.protected, side.protects = True, side.protects + 1
        elif move.category.name == "STATUS":
            cur.status_move = True
            if _is_setup(move):
                cur.setup, side.setups = True, side.setups + 1
        else:
            cur.attacked, side.attacks = True, side.attacks + 1

    def damage(self, parts: list[str]) -> None:
        k = key(parts[2])
        m = self.member(k)
        new = _fraction(parts[3]) if len(parts) > 3 else None
        if new is None:
            return
        direct = parts[1] == "-damage" and not any(p.startswith("[from]") for p in parts[4:])
        if direct and m.hp is not None and new < m.hp:
            drop = m.hp - new
            m.biggest_hit = max(m.biggest_hit, drop)
            if self.last_mover is not None and self.last_mover[:2] != k[:2]:
                self.log.sides[self.last_mover[:2]].current.dealt += drop
        m.hp = new

    def line(self, parts: list[str]) -> None:
        kind = parts[1] if len(parts) > 1 else ""
        if kind == "turn":
            self.end_turn()
            self.turn = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else self.turn + 1
        elif kind in ("switch", "drag", "replace") and len(parts) > 2:
            self.switch(parts)
        elif kind == "move" and len(parts) > 2:
            self.move(parts)
        elif kind in ("-damage", "-heal", "-sethp") and len(parts) > 2:
            self.damage(parts)
        elif kind == "-mega" and len(parts) > 2:
            self.log.sides[key(parts[2])[:2]].current.mega = True
        elif kind == "faint" and len(parts) > 2:
            k = key(parts[2])
            self.member(k).hp = 0.0
            side = self.log.sides[k[:2]]
            side.faints += 1
            side.current.fainted = True


def read(battle) -> Log:
    """Evidence and tendencies from everything the Battle received so far (p1 = us)."""
    reader = _Reader()
    for parts in battle._replay_data:
        reader.line(parts)
    for side in reader.log.sides.values():
        if side.current != Turn():  # a decision inside a turn (forced switch, pivot): the turn so far
            side.last = side.current
    return reader.log
