# From research/pcsa-rl/rl/features_field.py (sha256 0a99fa3b8994, 2026-11-08 20:13); keep diffable.
"""The 126-float field vector: global 16 | weather 6 | terrain 5 | rooms 5 | side conditions 2 x 15 |
last turn 2 x 8 | their active's last move 21 | tendencies 2 x 4 | threat summary 6 | misc 13."""

from __future__ import annotations

import numpy as np
from poke_env.battle import Effect, Field, SideCondition, Weather
from poke_env.data.normalize import to_id_str

from pcsa.rl_training import features_moves as fm
from pcsa.rl_training.features_log import Side, Turn

FIELD = 126
WEATHER_INDEX = {
    Weather.SUNNYDAY: 0,
    Weather.DESOLATELAND: 0,
    Weather.RAINDANCE: 1,
    Weather.PRIMORDIALSEA: 1,
    Weather.SANDSTORM: 2,
    Weather.SNOWSCAPE: 3,
    Weather.HAIL: 3,
}
STRONG = frozenset({Weather.DESOLATELAND, Weather.PRIMORDIALSEA, Weather.DELTASTREAM})
TERRAINS = (Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN)
ROOMS = (Field.TRICK_ROOM, Field.GRAVITY, Field.MAGIC_ROOM, Field.WONDER_ROOM)
FLAGS = (
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
    SideCondition.AURORA_VEIL,
    SideCondition.TAILWIND,
    SideCondition.SAFEGUARD,
    SideCondition.MIST,
    SideCondition.STEALTH_ROCK,
    SideCondition.SPIKES,
    SideCondition.TOXIC_SPIKES,
    SideCondition.STICKY_WEB,
)
LAYERS = {SideCondition.SPIKES: 3.0, SideCondition.TOXIC_SPIKES: 2.0}
TIMED = (
    (SideCondition.REFLECT, 8.0),
    (SideCondition.LIGHT_SCREEN, 8.0),
    (SideCondition.AURORA_VEIL, 8.0),
    (SideCondition.TAILWIND, 4.0),
    (SideCondition.SAFEGUARD, 5.0),
)
CHOICE = frozenset({"choicescarf", "choiceband", "choicespecs"})


def _elapsed(battle, start, div: float) -> float:
    try:
        return max(0.0, min(1.0, (battle.turn - int(start)) / div))
    except (TypeError, ValueError):
        return 0.0


def _global(f: np.ndarray, battle, mine: list, theirs: list, legal, log) -> None:
    moves = sum(1 for a in legal if a.action.kind == "move" and not a.action.mega)
    switches = sum(1 for a in legal if a.action.kind == "switch")
    my_alive = [r for r in mine[:3] if r.mon is not None and not r.mon.fainted]
    seen = [r for r in theirs if r.key is not None]
    faints = log.sides["p2"].faints
    their_hp = sum(r.hp for r in seen if not r.mon.fainted) + max(0, 3 - len(seen))
    f[0] = min(1.0, battle.turn / 30.0)
    f[1], f[2] = float(bool(battle.force_switch)), float(bool(battle.trapped or battle.maybe_trapped))
    f[3], f[4] = float(bool(battle.can_mega_evolve)), float(bool(battle.used_mega_evolve))
    f[5] = float(bool(battle.opponent_used_mega_evolve))
    f[6], f[7] = len(my_alive) / 3.0, max(0.0, 3 - faints) / 3.0
    f[8] = min(1.0, len(seen) / 3.0)
    f[9], f[10] = sum(r.hp for r in my_alive) / 3.0, min(1.0, their_hp / 3.0)
    f[11], f[12] = min(1.0, moves / 4.0), min(1.0, switches / 2.0)
    f[13] = float(mine[0].mon is None or mine[0].mon.fainted)
    f[14] = float(theirs[0].mon is None or theirs[0].mon.fainted)
    f[15] = float(theirs[0].mon is not None and theirs[0].key is not None and bool(theirs[0].mon.first_turn))


def _weather_rooms(f: np.ndarray, battle) -> None:
    for w, start in battle.weather.items():
        if w in WEATHER_INDEX:
            f[16 + WEATHER_INDEX[w]] = 1.0
        f[20] = max(f[20], float(w in STRONG))
        # pcsa deviation from research (9 Nov): capped at one turn. In the simulator Showdown's end-of-turn
        # `|-weather|X|[upkeep]` line resets poke-env's start turn, so training only ever saw 0 or 0.125 here;
        # the phone's battle log has no upkeep line and the value grew to 0.875 (run 20261109-205551).
        f[21] = min(_elapsed(battle, start, 8.0), 0.125)
    for i, t in enumerate(TERRAINS):
        if t in battle.fields:
            f[22 + i] = 1.0
            f[26] = _elapsed(battle, battle.fields[t], 8.0)
    for i, r in enumerate(ROOMS):
        if r in battle.fields:
            f[27 + i] = 1.0
    if Field.TRICK_ROOM in battle.fields:
        f[31] = _elapsed(battle, battle.fields[Field.TRICK_ROOM], 5.0)


def _side(f: np.ndarray, battle, conditions: dict) -> None:
    for i, c in enumerate(FLAGS):
        if c in conditions:
            f[i] = min(1.0, conditions[c] / LAYERS[c]) if c in LAYERS else 1.0
    for i, (c, div) in enumerate(TIMED):
        if c in conditions:
            f[10 + i] = _elapsed(battle, conditions[c], div)


def _turn(f: np.ndarray, t: Turn) -> None:
    f[:] = [t.switched, t.attacked, t.status_move, t.protected, t.setup, t.mega, min(1.0, t.dealt), t.fainted]


def _tendency(f: np.ndarray, s: Side) -> None:
    turns = max(1, s.turns)
    f[:] = [min(1.0, x / turns) for x in (s.switches, s.protects, s.setups, s.attacks)]


def _last_move(f: np.ndarray, mon) -> None:
    move = mon.last_move if mon is not None else None
    if move is None:
        return
    facts = fm.facts(move.id)
    f[:21] = facts[:21]


def _threat(f: np.ndarray, rows: np.ndarray) -> None:
    """From the encoded rows: my active's best mean hit, theirs, speed flags, KO flags."""
    mine = rows[0, 158:].reshape(4, fm.MOVE_WIDTH)[:, fm.FACTS + fm.STATE :]
    theirs = rows[6, 158:].reshape(4, fm.MOVE_WIDTH)[:, fm.FACTS + fm.STATE :]
    f[:] = [
        mine[:, 1].max() / 2,
        theirs[:, 1].max() / 2,
        rows[0, 148],
        rows[0, 149],
        mine[:, 3].max(),
        theirs[:, 4].max(),
    ]


def _misc(f: np.ndarray, battle, mine: list, theirs: list, legal, log, rows: np.ndarray) -> None:
    me = mine[0].mon
    legal_moves = {
        a.order.order.id
        for a in legal
        if a.action.kind == "move" and a.order is not None and hasattr(a.order.order, "id")
    }
    item = to_id_str(getattr(me, "item", "") or "") if me is not None else ""
    effects = (me.effects or {}) if me is not None else {}
    my_side, their_side = log.sides["p1"], log.sides["p2"]
    f[0] = float(item in CHOICE and len(legal_moves) == 1 and len(getattr(me, "moves", {})) > 1)
    f[1], f[2] = float(Effect.ENCORE in effects), float(Effect.TAUNT in effects)
    f[3], f[4] = rows[0, 140], rows[6, 140]
    f[5], f[6] = min(1.0, my_side.switches / 10.0), min(1.0, their_side.switches / 10.0)
    f[7], f[8] = min(1.0, my_side.faints / 3.0), min(1.0, their_side.faints / 3.0)
    f[9], f[10] = rows[0, 34], rows[6, 34]
    f[11] = min(1.0, max(0, battle.turn - my_side.last_switch_turn) / 10.0)
    f[12] = min(1.0, max(0, battle.turn - their_side.last_switch_turn) / 10.0)


def encode(battle, mine: list, theirs: list, legal, log, rows: np.ndarray) -> np.ndarray:
    f = np.zeros(FIELD, dtype=np.float32)
    _global(f[0:16], battle, mine, theirs, legal, log)
    _weather_rooms(f, battle)
    _side(f[32:47], battle, battle.side_conditions)
    _side(f[47:62], battle, battle.opponent_side_conditions)
    _turn(f[62:70], log.sides["p1"].last)
    _turn(f[70:78], log.sides["p2"].last)
    _last_move(f[78:99], theirs[0].mon if theirs[0].key is not None else None)
    _tendency(f[99:103], log.sides["p1"])
    _tendency(f[103:107], log.sides["p2"])
    _threat(f[107:113], rows)
    _misc(f[113:126], battle, mine, theirs, legal, log, rows)
    return f
