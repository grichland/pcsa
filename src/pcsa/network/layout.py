"""The input layout: constants of Champions Gym's model contract (MODEL.md) and the feature list, looked up by
feature id.

The model never uses a column number of its own: everything comes from a Layout, built from
champions_gym.FEATURES through cg_adapter.

Copied from pcsa-former (pcsa_former/layout.py) without its feature-CSV reader; the rest behaves identically to it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

CONTRACT = "0.6"

# vocabulary sizes of the model contract; cg_adapter checks them against champions_gym.VOCAB
VOCAB = {"species": 393, "item": 168, "ability": 319, "move": 516}
OWN_ACTIONS = 11  # 0-3 move, 4-7 move + Mega, 8-10 switch to row 0-2
OPP_ACTIONS = 13  # 0-3 move, 4-7 move + Mega, 8-12 switch to row 7-11
PREVIEW_ACTIONS = 60  # lead * 10 + pair; the environment's action is 11 + this
PREVIEW_OFFSET = 11
MOVE_SLOTS = 4
ROWS, OWN_ROWS, OPP_ROWS = 12, slice(0, 6), slice(6, 12)
OWN_PICKS = 3  # switch action 8 + p -> row p
OPP_ACTIVE, OPP_BENCH = 6, slice(7, 12)
KIND_PREVIEW, KIND_TURN, KIND_REPLACE = 0, 1, 2

# feature ids the model needs by meaning
F_IS_ACTIVE = "F3-9"
F_SPECIES, F_ITEM, F_ABILITY, F_MOVE_ID, F_POSSIBLE_ABILITY = "F1-3", "F2-6", "F2-7", "F7-13", "F8-10"
F_LAST_MOVES, F_LAST_SPECIES = ("L2-1.1", "L2-1.2"), ("L2-2.1", "L2-2.2")


@dataclass(frozen=True)
class Feature:
    id: str
    name: str
    block: str  # rows | row_ids | field | last | last_ids
    offset: int  # first column (slot features: within slot 0, absolute)
    size: int  # columns of one repeat
    repeats: int
    stride: int  # columns between repeats (0 when repeats == 1)
    encoding: str
    low: float
    high: float

    @property
    def width(self) -> int:
        return self.size * self.repeats if self.block.endswith("ids") else self.size

    def columns(self) -> list[int]:
        """Every column of this feature, repeat by repeat."""
        if self.block.endswith("ids"):
            return list(range(self.offset, self.offset + self.size * self.repeats))
        return [self.offset + r * self.stride + c for r in range(self.repeats) for c in range(self.size)]


@dataclass
class Layout:
    features: dict[str, Feature]
    hash: str = ""  # sha256 of where every feature sits: the same from a feature CSV or champions_gym.FEATURES
    source: str = ""  # sha256 of the feature-list file, when built from one
    widths: dict[str, int] = field(default_factory=dict)
    slot_base: int = 0  # move slots in `rows`: slot s occupies slot_base + slot_stride * s ...
    slot_stride: int = 0
    cg: dict = field(default_factory=dict)  # champions_gym.neural.contract() of the build the layout was read from

    def __post_init__(self) -> None:
        if not self.hash:
            where = sorted((f.id, f.block, f.offset, f.size, f.repeats, f.stride) for f in self.features.values())
            self.hash = hashlib.sha256(repr(where).encode()).hexdigest()
        for block in ("rows", "row_ids", "field", "last", "last_ids"):
            cols = sorted(c for f in self.features.values() if f.block == block for c in f.columns())
            if cols != list(range(len(cols))):
                raise ValueError(f"{block}: the feature list leaves gaps or overlaps")
            self.widths[block] = len(cols)
        slots = [f for f in self.features.values() if f.block == "rows" and f.repeats > 1]
        if not slots or {f.repeats for f in slots} != {MOVE_SLOTS} or len({f.stride for f in slots}) != 1:
            raise ValueError("move slots: expected one group of features repeated per slot")
        self.slot_base, self.slot_stride = min(f.offset for f in slots), slots[0].stride
        first = sorted(c for f in slots for c in range(f.offset, f.offset + f.size))
        if first != list(range(self.slot_base, self.slot_base + self.slot_stride)):
            raise ValueError("move slots: a slot is not one contiguous run of columns")

    def __getitem__(self, feature_id: str) -> Feature:
        return self.features[feature_id]

    def col(self, feature_id: str) -> int:
        f = self.features[feature_id]
        if f.width != 1:
            raise ValueError(f"{feature_id} is {f.width} columns wide")
        return f.offset

    def row_plain_columns(self) -> list[int]:
        """Columns of `rows` outside the move slots."""
        lo, hi = self.slot_base, self.slot_base + self.slot_stride * MOVE_SLOTS
        return [c for c in range(self.widths["rows"]) if not lo <= c < hi]

    def describe(self) -> dict:
        out = {"hash": self.hash, "source": self.source, "widths": dict(self.widths), "vocab": dict(VOCAB),
               "contract": self.cg.get("contract", CONTRACT)}  # fmt: skip
        if self.cg:
            out["cg"] = dict(self.cg)  # saved in every checkpoint as CG gave it
        return out


def preview_combos() -> list[tuple[int, int, int]]:
    """Preview action a (0-59) -> (lead, back 1, back 2) as team indices: a = lead * 10 + pair, pair numbering the two
    back picks among the other five in team order, lexicographic. (The model contract's numbering; cg_adapter checks
    it against champions_gym.preview_picks.)"""
    out = []
    for lead in range(6):
        rest = [i for i in range(6) if i != lead]
        out += [(lead, rest[i], rest[j]) for i in range(5) for j in range(i + 1, 5)]
    return out
