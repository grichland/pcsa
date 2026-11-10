# From research/pcsa-rl/rl/features_rows.py (sha256 50bc9d07403e, 2026-11-08 20:13); keep diffable.
"""The 318-float Pokémon row: identity 24 | build 10 | condition 110 | knowledge 4 | matchup 6 | evidence 4 |
moves 4 x 40 (27 facts, PP, disabled, locked, known, 9 damage columns vs the opposing active)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from poke_env.battle import Effect, Field, Move, PokemonType, SideCondition, Status
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.stats import (
    NATURE_DOWN,
    NATURE_UP,
    POINTS_MAX,
    current_stats,
    form_of,
    known_ability,
    stage,
    stat_value,
)
from pcsa.players.opponent import AssumedSet, revealed_item, speed
from pcsa.players.tactics import mega_form_of
from pcsa.rl_training import features_moves as fm
from pcsa.rl_training.features_vocab import index

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STATUSES = (Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX)
VOLATILES = tuple(
    Effect[name]
    for name in [
        "CONFUSION",
        "ENCORE",
        "TAUNT",
        "TORMENT",
        "DISABLE",
        "SUBSTITUTE",
        "LEECH_SEED",
        "YAWN",
        "PERISH0",
        "PERISH1",
        "PERISH2",
        "PERISH3",
        "ATTRACT",
        "CURSE",
        "NIGHTMARE",
        "DESTINY_BOND",
        "PROTECT",
        "ENDURE",
        "FLINCH",
        "FOCUS_ENERGY",
        "LASER_FOCUS",
        "CHARGE",
        "MAGNET_RISE",
        "TELEKINESIS",
        "SMACK_DOWN",
        "INGRAIN",
        "AQUA_RING",
        "HEAL_BLOCK",
        "EMBARGO",
        "GASTRO_ACID",
        "THROAT_CHOP",
        "SALT_CURE",
        "SYRUP_BOMB",
        "TAR_SHOT",
        "OCTOLOCK",
        "NO_RETREAT",
        "TRAPPED",
        "PARTIALLY_TRAPPED",
        "BIND",
        "WRAP",
        "FIRE_SPIN",
        "WHIRLPOOL",
        "SAND_TOMB",
        "INFESTATION",
        "MAGMA_STORM",
        "CLAMP",
        "SNAP_TRAP",
        "THUNDER_CAGE",
        "LOCKED_MOVE",
        "UPROAR",
        "STOCKPILE1",
        "STOCKPILE2",
        "STOCKPILE3",
        "MINIMIZE",
        "DEFENSE_CURL",
        "ROOST",
        "POWER_TRICK",
        "TYPECHANGE",
        "TYPEADD",
        "DISGUISE",
        "ICE_FACE",
        "FLASH_FIRE",
        "PROTOSYNTHESIS",
        "PROTOSYNTHESISATK",
        "PROTOSYNTHESISDEF",
        "PROTOSYNTHESISSPA",
        "PROTOSYNTHESISSPD",
        "PROTOSYNTHESISSPE",
        "QUARK_DRIVE",
        "QUARKDRIVEATK",
        "QUARKDRIVEDEF",
        "QUARKDRIVESPA",
        "QUARKDRIVESPD",
        "QUARKDRIVESPE",
        "SLOW_START",
        "ILLUSION",
        "KINGS_SHIELD",
        "SPIKY_SHIELD",
        "BANEFUL_BUNKER",
        "SILK_TRAP",
        "BURNING_BULWARK",
        "OBSTRUCT",
        "GLAIVE_RUSH",
        "FOLLOW_ME",
        "RAGE_POWDER",
        "MAGIC_COAT",
        "IMPRISON",
    ]
)
VOLATILE_INDEX = {e: i for i, e in enumerate(VOLATILES)}
DERIVED_VOLATILES = 3  # first turn out, preparing a two-turn move, must recharge
assert len(VOLATILES) + DERIVED_VOLATILES == 90, len(VOLATILES)
BASE_DIV, STAT_DIV = 255.0, 400.0
SPEED_CLIP = 2.0
SCARF, PARALYSIS, TAILWIND = 1.5, 0.5, 2.0
ROW = 318


@dataclass
class Row:
    """One canonical row: who it is and what we assume about it."""

    mon: Any  # a poke-env Pokemon, or None for an empty row (Any for pyright; research: object | None)
    ours: bool
    key: str | None = None  # log key 'p1: Name'
    chosen: AssumedSet | None = None
    brought: float = 0.0
    not_brought: float = 0.0
    active: bool = False
    moves: list[Move | None] = field(default_factory=list)
    known: list[bool] = field(default_factory=list)
    fighter: Any = None  # pcsa.players.threat fighter (research: object | None)
    form: str = ""  # dex key of the current form
    mega: bool = False
    ability: str = ""  # ours: the ability; theirs: the one the log showed ('' when not shown)

    def resolve(self, battle) -> Row:
        """Fill the form and the known ability once (both scan the protocol)."""
        if self.mon is not None:
            self.form, self.mega = form_of(self.mon, battle)
            self.ability = to_id_str(self.mon.ability or "") if self.ours else known_ability(battle, self.mon)
        return self

    @property
    def hp(self) -> float:
        if self.mon is None:
            return 0.0
        if not self.ours and self.key is None:
            return 1.0  # seen only at preview
        return max(0.0, min(1.0, float(self.mon.current_hp_fraction or 0.0)))


def boost_scalar(s: int) -> float:
    mult = (2 + s) / 2 if s >= 0 else 2 / (2 - s)
    return math.log2(mult) / 2


def _types(mon, form: str) -> list[PokemonType]:
    if getattr(mon, "_temporary_types", None):
        return list(mon.types)
    names = game.species_entry(form).get("types")
    if not names:
        return [t for t in mon.types if t is not None]
    return [PokemonType.from_name(t) for t in names]


def identity(f: np.ndarray, mon, form: str) -> None:
    for t in _types(mon, form):
        if t in fm.TYPE_INDEX:
            f[fm.TYPE_INDEX[t]] = 1.0
    base = game.species_entry(form).get("baseStats") or mon.base_stats or {}
    f[18:24] = [min(1.0, base.get(s, 0) / BASE_DIV) for s in STATS]


def build(f: np.ndarray, row: Row) -> None:
    mon, form = row.mon, row.form
    if row.ours:
        f[24], f[25], f[26], f[33] = 1.0, float(mon.item == ""), 1.0, 1.0
        stats = row.fighter.stats if row.fighter is not None else current_stats(mon, True, form)
    else:
        known, item = revealed_item(mon)
        f[24], f[25], f[26] = float(known), float(known and item is None), float(bool(row.ability))
        stats = row.fighter.stats if row.fighter is not None else current_stats(mon, False, form)
    f[27:33] = [min(2.0, float(stats.get(s) or 0) / STAT_DIV) for s in STATS]


def condition(f: np.ndarray, battle, row: Row, ctx) -> None:
    mon = row.mon
    f[34] = row.hp
    if mon.status in STATUSES:
        f[35 + STATUSES.index(mon.status)] = 1.0
        f[41] = min(1.0, (mon.status_counter or 0) / 5.0)
    boosts = mon.boosts or {}
    f[42:49] = [boost_scalar(int(boosts.get(s, 0))) for s in BOOSTS]
    for effect in mon.effects or ():
        if effect in VOLATILE_INDEX:
            f[49 + VOLATILE_INDEX[effect]] = 1.0
    at = 49 + len(VOLATILES)
    f[at : at + 3] = [
        float(bool(mon.first_turn and row.active)),
        float(bool(mon.preparing)),
        float(mon.must_recharge),
    ]
    f[139] = min(1.0, (mon.protect_counter or 0) / 3.0)
    member = ctx.log.members.get(row.key) if row.key else None
    if row.active and member is not None and member.entered_turn >= 0:
        f[140] = min(1.0, max(0, battle.turn - member.entered_turn) / 10.0)
    f[141] = float(row.active)
    f[142] = _can_mega(battle, row)
    f[143] = float(row.mega)


def _can_mega(battle, row: Row) -> float:
    mon = row.mon
    if mon.fainted or row.mega:
        return 0.0
    if row.ours:
        return float(not battle.used_mega_evolve and mega_form_of(mon) is not None)
    if battle.opponent_used_mega_evolve:
        return 0.0
    item = row.chosen.item if row.chosen is not None else None
    return float(bool(item) and game.mega_form_for_stone(mon.species, to_id_str(item)) is not None)


def knowledge(f: np.ndarray, row: Row, ctx) -> None:
    member = ctx.log.members.get(row.key) if row.key else None
    f[144], f[145] = row.brought, row.not_brought
    f[146] = float(member is not None and member.entries > 0)
    f[147] = float(member is not None and member.lead)


def speed_bounds(battle, row: Row) -> tuple[float, float]:
    """(low, high) speed: ours exact, theirs from 0 points -nature to 32 points +nature (x1.5 if no item seen)."""
    mon = row.mon
    if row.ours:
        exact = speed(battle, mon, row.fighter)
        return exact, exact
    base = (game.species_entry(row.form).get("baseStats") or mon.base_stats)["spe"]
    low, high = float(stat_value(base, 0, NATURE_DOWN)), float(stat_value(base, POINTS_MAX, NATURE_UP))
    known, item = revealed_item(mon)
    scarf = SCARF if known and to_id_str(item or "") == "choicescarf" else 1.0
    low, high = low * scarf, high * (scarf if known else SCARF)
    boost = int((mon.boosts or {}).get("spe", 0))
    low, high = float(stage(int(low), boost)), float(stage(int(high), boost))
    scale = PARALYSIS if mon.status == Status.PAR else 1.0
    scale *= TAILWIND if SideCondition.TAILWIND in battle.opponent_side_conditions else 1.0
    return low * scale, high * scale


def _ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return max(-SPEED_CLIP, min(SPEED_CLIP, math.log2(a / b))) / SPEED_CLIP


def matchup(f: np.ndarray, battle, row: Row, opp: Row | None, ctx) -> None:
    if opp is None or opp.mon is None or opp.mon.fainted or row.mon.fainted:
        return
    mine, theirs = ctx.speed(row), ctx.speed(opp)
    faster, slower = mine[0] > theirs[1], mine[1] < theirs[0]
    trick_room = Field.TRICK_ROOM in battle.fields
    f[148], f[149] = (float(slower), float(faster)) if trick_room else (float(faster), float(slower))
    mid = math.sqrt(max(mine[0], 1e-6) * max(mine[1], 1e-6))
    f[150], f[151] = _ratio(mid, theirs[0]), _ratio(mid, theirs[1])
    hits = ctx.hits(opp, row)
    hp = row.hp
    if hits and hp > 0:
        f[152] = min(2.0, max(h[1] for h in hits) / hp)
        f[153] = min(2.0, max(h[2] for h in hits) / hp)


def evidence(f: np.ndarray, row: Row, ctx) -> None:
    member = ctx.log.members.get(row.key) if row.key else None
    if member is None:
        return
    f[154] = min(1.0, member.biggest_hit)
    f[155] = min(1.0, member.moved_first / 5.0)
    f[156] = min(1.0, member.moved_second / 5.0)
    f[157] = min(1.0, member.entries / 4.0)


def moves(f: np.ndarray, row: Row, opp: Row | None, ctx) -> None:
    hits = ctx.hits(row, opp) if opp is not None and opp.mon is not None and not opp.mon.fainted else []
    hp = opp.hp if opp is not None else 1.0
    for j, move in enumerate(row.moves[:4]):
        if move is None:
            continue
        state = ctx.move_state(row, j, move)
        dmg = fm.damage_columns(*hits[j], hp) if j < len(hits) else [0.0] * fm.DAMAGE  # pyright: ignore[reportCallIssue]
        at = 158 + j * fm.MOVE_WIDTH
        fm.block(f[at : at + fm.MOVE_WIDTH], move, state, dmg)


def encode(battle, row: Row, opp: Row | None, ctx) -> tuple[np.ndarray, np.ndarray]:
    """(318 floats, 7 ids) for one row; zeros for an empty row."""
    f = np.zeros(ROW, dtype=np.float32)
    ids = np.zeros(7, dtype=np.int64)
    if row.mon is None:
        return f, ids
    identity(f, row.mon, row.form)
    build(f, row)
    condition(f, battle, row, ctx)
    knowledge(f, row, ctx)
    matchup(f, battle, row, opp, ctx)
    evidence(f, row, ctx)
    moves(f, row, opp, ctx)
    moves_ids = [index("move", m.id if m else None) for m in (row.moves + [None] * 4)[:4]]
    ids[:] = [index("species", row.form), *_build_ids(row), *moves_ids]
    return f, ids


def _build_ids(row: Row) -> tuple[int, int]:
    mon = row.mon
    if row.ours:
        return index("item", mon.item), index("ability", mon.ability)
    known, item = revealed_item(mon)
    item = item if known else (row.chosen.item if row.chosen else None)
    ability = row.ability or (row.chosen.ability if row.chosen else None)
    return index("item", item), index("ability", ability)
