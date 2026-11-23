"""The optional rule masks (rules.RuleConfig fields after the first two): hand-built observations per rule, the ids
against CG's data, and real CG observations (never all closed; with every new rule off, the old rules exactly).

Ported from pcsa-former (tests/test_rules_extra.py) for pcsa's copy of rules.py."""

import json
import os
import re
from dataclasses import fields

import numpy as np
import pytest

cg = pytest.importorskip("champions_gym")
pytest.importorskip("torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pcsa.common.paths import repo_root  # noqa: E402
from pcsa.network import cg_adapter, rules  # noqa: E402
from pcsa.network import layout as L  # noqa: E402

ROOT = repo_root().parent  # the folder holding champions-gym next to this repository
ALL_ON = rules.RuleConfig(max_boosts=2, max_heals=2, no_null_moves=True, no_boost_when_ko=True, no_failing_status=True,
                          no_redundant_field=True, no_double_protect=True, no_wasted_self=True)  # fmt: skip


class Obs:
    """One turn request: our active in row 0, theirs in row 6, every action open."""

    def __init__(self, lay, moves=("tackle",)):
        self.lay = lay
        self.by = {f.name: f for f in lay.features.values()}
        self.o = {"rows": np.zeros((1, 12, lay.widths["rows"]), np.float32), "row_ids": np.zeros((1, 12, 10), np.int32),
                  "field": np.zeros((1, lay.widths["field"]), np.float32), "mask": np.ones((1, 11), np.uint8),
                  "kind": np.array([L.KIND_TURN], np.uint8), "seat": np.array([0], np.uint8)}  # fmt: skip
        self.row(0, "pokemon.condition.is_active", 1)
        self.row(0, "pokemon.condition.hp_fraction", 1)
        self.o["row_ids"][0, 0, lay.col(L.F_SPECIES)] = 5
        for j in range(4):
            self.move(j, 200 + j, cat=0, power=0.5)

    def _col(self, name, k=0):
        return self.by[name].columns()[k]

    def row(self, r, name, v, k=0):
        self.o["rows"][0, r, self._col(name, k)] = v

    def field(self, name, v, k=0):
        self.o["field"][0, self._col(name, k)] = v

    def move(self, j, mid, cat=2, power=0.0, ty=None, prio=0.0, setup=0, heal=0):
        """Slot j of our active: move id, category (0 physical, 1 special, 2 status), type name."""
        size = 3
        for c in range(size):
            self.row(0, "pokemon.move.fact.category", float(c == cat), j * size + c)
        for t in range(18):
            self.row(0, "pokemon.move.fact.type", 0.0, j * 18 + t)
        if ty:
            self.row(0, "pokemon.move.fact.type", 1.0, j * 18 + rules.TYPES.index(ty))
        self.row(0, "pokemon.move.fact.priority", prio, j)
        self.row(0, "pokemon.move.fact.is_setup", setup, j)
        self.row(0, "pokemon.move.fact.is_heal_or_drain", heal, j)
        self.row(0, "pokemon.move.damage.effectiveness", 0.25 if cat < 2 else 0.0, j)
        self.row(0, "pokemon.move.damage.max", power if cat < 2 else 0.0, j)
        self.o["row_ids"][0, 0, self.lay[L.F_MOVE_ID].columns()[j]] = mid

    def foe_type(self, *types):
        for t in types:
            self.row(6, "pokemon.identity.type", 1, rules.TYPES.index(t))

    def foe_ability(self, a, possible_only=False):
        ids = self.o["row_ids"][0, 6]
        if possible_only:
            ids[self.lay[L.F_POSSIBLE_ABILITY].columns()[0]] = a
        else:
            ids[self.lay.col(L.F_ABILITY)] = a

    def foe_vol(self, name):
        self.row(6, "pokemon.condition.volatile", 1, rules.VOLATILE[name])


@pytest.fixture(scope="module")
def lay():
    return cg_adapter.layout()


def run(cfg, obs, seq=()):
    """The mask after the actions `seq` were taken from this same request."""
    r = rules.Rules(cfg, obs.lay, 1)
    for a in seq:
        r.mask(obs.o)
        r.record(obs.o, np.array([a]))
    return r.mask(obs.o)[0], r


def only(**kw):
    return rules.RuleConfig(max_same_move=0, max_switches=0, **kw)


def test_defaults_are_unchanged():
    d = rules.RuleConfig()
    assert (d.max_same_move, d.max_switches) == (4, 3)
    assert all(not getattr(d, f.name) for f in fields(d)[2:])


def test_ids_match_cg_data():
    moves = ROOT / "champions-gym" / "data" / "moves.json"
    gen = ROOT / "champions-gym" / "engine" / "src" / "sim" / "data" / "generated.rs"
    if not moves.exists() or not gen.exists():
        pytest.skip("champions-gym is not next to this repository")
    names = [""] + list(json.loads(moves.read_text(encoding="utf-8")))
    assert all(names[i] == n for n, i in rules.MOVE.items())
    assert set(rules.SETUP_BOOSTS) <= set(rules.MOVE)
    text = gen.read_text(encoding="utf-8")
    abilities = re.findall(r'AbilityData \{ id: "([a-z0-9]*)"', text)
    items = re.findall(r'ItemData \{ id: "([a-z0-9]*)"', text)
    assert all(abilities[i] == n for n, i in rules.ABILITY.items())
    assert all(items[i - 1] == n for n, i in rules.ITEM.items())
    assert "curse" not in [names[i] for i in rules.FOE_STATUS]
    vols = re.findall(r'^\s+\("([a-z0-9]+)",', (ROOT / "champions-gym" / "obs" / "src" / "vols.rs").read_text(encoding="utf-8"), re.M)
    assert all(vols[i] == n for n, i in rules.VOLATILE.items())


def test_m1_boost_runs(lay):
    o = Obs(lay)
    o.move(0, rules.MOVE["swordsdance"], setup=1)
    m, r = run(only(max_boosts=2), o, [0, 0])
    assert m[0] == 0 and m[4] == 0 and m[1] and m[8]
    assert r.closed["boosts"] == 1
    m, _ = run(only(max_boosts=2), o, [0, 1, 0])  # broken by another move
    assert m[0]
    m, _ = run(only(max_boosts=2), o, [0, 8, 0])  # and by a switch
    assert m[0]


def test_m2_heal_runs_not_drain(lay):
    o = Obs(lay)
    o.move(0, rules.MOVE["recover"], heal=1)
    o.move(1, 150, cat=1, power=0.4, heal=1)  # a drain attack
    m, _ = run(only(max_heals=1), o, [0])
    assert m[0] == 0 and m[1]
    m, _ = run(only(max_heals=1), o, [1, 1])
    assert m[1]


def test_m4_null_moves(lay):
    o = Obs(lay)
    o.move(0, 210, cat=0, power=0.0)  # computed 0
    o.move(1, rules.MOVE["counter"], cat=0, power=0.0)  # not computed: left alone
    o.move(2, 211, cat=0, power=0.5, ty="ground")
    o.move(3, 212, cat=0, power=0.3, prio=0.2)
    m, _ = run(only(no_null_moves=True), o)
    assert m[0] == 0 and m[4] == 0 and m[1] and m[2] and m[3]
    o.foe_ability(rules.ABILITY["levitate"], possible_only=True)
    assert run(only(no_null_moves=True), o)[0][2] == 0
    o2 = Obs(lay)
    o2.move(2, 211, cat=0, power=0.5, ty="ground")
    o2.o["row_ids"][0, 6, lay.col(L.F_ITEM)] = rules.ITEM["airballoon"]
    assert run(only(no_null_moves=True), o2)[0][2] == 0
    o2.o["row_ids"][0, 0, lay.col(L.F_ABILITY)] = rules.ABILITY["moldbreaker"]  # balloon still holds
    assert run(only(no_null_moves=True), o2)[0][2] == 0
    o.field("field.terrain.kind", 1, 3)  # Psychic Terrain
    assert run(only(no_null_moves=True), o)[0][3]  # a levitating target is not grounded
    o3 = Obs(lay)
    o3.move(3, 212, cat=0, power=0.3, prio=0.2)
    o3.field("field.terrain.kind", 1, 3)
    assert run(only(no_null_moves=True), o3)[0][3] == 0
    o3.foe_type("flying")
    assert run(only(no_null_moves=True), o3)[0][3]


def test_m5_no_boost_into_ko(lay):
    o = Obs(lay)
    o.move(0, rules.MOVE["nastyplot"], setup=1)
    o.row(0, "pokemon.matchup.incoming_hit.mean", 0.9)
    assert run(only(no_boost_when_ko=True), o)[0][0]
    o.row(0, "pokemon.matchup.incoming_hit.mean", 1.2)
    assert run(only(no_boost_when_ko=True), o)[0][0] == 0


@pytest.mark.parametrize("move,setup", [
    ("thunderwave", lambda o: o.foe_type("ground")), ("thunderwave", lambda o: o.foe_type("electric")),
    ("willowisp", lambda o: o.foe_type("fire")), ("toxic", lambda o: o.foe_type("steel")),
    ("toxic", lambda o: o.foe_type("poison")), ("spore", lambda o: o.foe_type("grass")),
    ("leechseed", lambda o: o.foe_type("grass")), ("leechseed", lambda o: o.foe_vol("leechseed")),
    ("toxic", lambda o: o.row(6, "pokemon.condition.status", 1, 2)), ("yawn", lambda o: o.row(6, "pokemon.condition.status", 1, 0)),
    ("thunderwave", lambda o: o.foe_ability(rules.ABILITY["goodasgold"])),
    ("thunderwave", lambda o: o.foe_ability(rules.ABILITY["goodasgold"], possible_only=True)),
    ("willowisp", lambda o: o.foe_vol("substitute")), ("toxic", lambda o: o.field("field.side.opp.safeguard", 1)),
    ("spore", lambda o: o.field("field.terrain.kind", 1, 0)),
])  # fmt: skip
def test_m7_failing_status(lay, move, setup):
    o = Obs(lay)
    o.move(0, rules.MOVE[move])
    assert run(only(no_failing_status=True), o)[0][0]
    setup(o)
    m, r = run(only(no_failing_status=True), o)
    assert m[0] == 0 and m[4] == 0 and r.closed["status"] == 1


def test_m7_exceptions(lay):
    o = Obs(lay)
    o.move(0, rules.MOVE["sing"])
    o.foe_vol("substitute")  # a sound move goes through
    assert run(only(no_failing_status=True), o)[0][0]
    o = Obs(lay)
    o.move(0, rules.MOVE["thunderwave"])
    o.foe_ability(rules.ABILITY["goodasgold"])
    o.o["row_ids"][0, 0, lay.col(L.F_ABILITY)] = rules.ABILITY["moldbreaker"]
    assert run(only(no_failing_status=True), o)[0][0]
    o = Obs(lay)
    o.move(0, rules.MOVE["spikes"])  # a hazard is not stopped by Good as Gold
    o.foe_ability(rules.ABILITY["goodasgold"])
    assert run(only(no_failing_status=True), o)[0][0]


def test_m8_redundant_field(lay):
    o = Obs(lay)
    for j, n in enumerate(("stealthrock", "spikes", "sunnyday", "psychicterrain")):
        o.move(j, rules.MOVE[n])
    assert run(only(no_redundant_field=True), o)[0][:4].all()
    o.field("field.side.opp.stealth_rock", 1)
    o.field("field.side.opp.spikes_layers", 2 / 3)
    o.field("field.weather.kind", 1, 1)  # rain: Sunny Day still changes it
    o.field("field.terrain.kind", 1, 3)
    m = run(only(no_redundant_field=True), o)[0]
    assert list(m[:4]) == [0, 1, 1, 0]
    o.field("field.side.opp.spikes_layers", 1)
    o.field("field.weather.kind", 1, 0)
    assert run(only(no_redundant_field=True), o)[0][:4].sum() == 0  # switches stay open


def test_m9_double_protect(lay):
    o = Obs(lay)
    o.move(0, rules.MOVE["protect"])
    m, _ = run(only(no_double_protect=True), o, [0])
    assert m[0] == 0 and m[4] == 0
    assert run(only(no_double_protect=True), o, [0, 1])[0][0]
    assert run(only(no_double_protect=True), o, [0, 8])[0][0]


def test_m10_wasted_self(lay):
    o = Obs(lay)
    o.move(0, rules.MOVE["recover"], heal=1)
    o.move(1, rules.MOVE["dragondance"], setup=1)
    o.move(2, rules.MOVE["substitute"])
    o.move(3, rules.MOVE["shellsmash"], setup=1)
    o.row(0, "pokemon.condition.boost.atk", 1.0)
    o.row(0, "pokemon.condition.boost.spa", 1.0)
    m = run(only(no_wasted_self=True), o)[0]
    assert list(m[:4]) == [0, 1, 1, 1]  # full HP; Speed not yet +6
    o.row(0, "pokemon.condition.boost.spe", 1.0)
    o.row(0, "pokemon.condition.hp_fraction", 0.25)
    assert list(run(only(no_wasted_self=True), o)[0][:4]) == [1, 0, 0, 0]
    o.row(0, "pokemon.condition.hp_fraction", 0.8)
    o.row(0, "pokemon.condition.volatile", 1, rules.VOLATILE["substitute"])
    assert run(only(no_wasted_self=True), o)[0][2] == 0


def test_never_all_closed_and_only_turns(lay):
    o = Obs(lay)
    for j in range(4):
        o.move(j, 210 + j, cat=0, power=0.0)
    o.o["mask"][0, 8:] = 0  # no switch: nothing may be closed
    m, r = run(only(no_null_moves=True), o)
    assert m.sum() == 8 and r.closed["null"] == 0
    o.o["mask"][0, 8:] = 1
    assert run(only(no_null_moves=True), o)[0].tolist() == [0] * 8 + [1] * 3
    o.o["kind"][0] = L.KIND_REPLACE
    assert run(only(no_null_moves=True), o)[0].sum() == 11


class OldRules:
    """rules.Rules before the optional rules, kept here to check the defaults change nothing."""

    def __init__(self, cfg, layout, games):
        self.cfg = cfg
        self.active_col, self.species_col = layout.col(L.F_IS_ACTIVE), layout.col(L.F_SPECIES)
        self.move_cols = layout[L.F_MOVE_ID].columns()
        self.key = np.zeros((games, 2, 2), np.int64)
        self.same = np.zeros((games, 2), np.int64)
        self.switches = np.zeros((games, 2), np.int64)

    def _active(self, obs):
        e = np.arange(len(obs["kind"]))
        row = np.asarray(obs["rows"])[:, : L.OWN_PICKS, self.active_col].argmax(-1)
        ids = np.asarray(obs["row_ids"])[e, row]
        return ids[:, self.species_col].astype(np.int64), ids[:, self.move_cols].astype(np.int64)

    def mask(self, obs):
        m = np.array(obs["mask"], copy=True)
        e, s = np.arange(len(m)), np.asarray(obs["seat"]).astype(np.int64)
        turn = np.asarray(obs["kind"]) == L.KIND_TURN
        moves_open = m[:, :8] > 0
        if self.cfg.max_same_move:
            species, moves = self._active(obs)
            due = turn & (self.same[e, s] >= self.cfg.max_same_move) & (self.key[e, s, 0] == species)
            slot = (moves == self.key[e, s, 1][:, None]) & (moves != 0) & due[:, None]
            close = np.concatenate([slot, slot], 1)
            close &= (moves_open & ~close).any(1)[:, None]
            m[:, :8][close] = 0
            moves_open = m[:, :8] > 0
        if self.cfg.max_switches:
            due = turn & (self.switches[e, s] >= self.cfg.max_switches) & moves_open.any(1)
            m[due, 8:] = 0
        return m

    def record(self, obs, action):
        action = np.asarray(action).astype(np.int64)
        e, s = np.arange(len(action)), np.asarray(obs["seat"]).astype(np.int64)
        turn = np.asarray(obs["kind"]) == L.KIND_TURN
        move, switch = turn & (action < 8), turn & (action >= 8)
        species, moves = self._active(obs)
        used = moves[e, np.clip(action, 0, 7) % L.MOVE_SLOTS]
        again = move & (self.key[e, s, 0] == species) & (self.key[e, s, 1] == used)
        self.same[e, s] = np.where(move, np.where(again, self.same[e, s] + 1, 1), np.where(switch, 0, self.same[e, s]))
        self.key[e, s, 0] = np.where(move, species, self.key[e, s, 0])
        self.key[e, s, 1] = np.where(move, used, self.key[e, s, 1])
        self.switches[e, s] = np.where(switch, self.switches[e, s] + 1, np.where(move, 0, self.switches[e, s]))


def observe(env) -> dict:
    t = env.observe_tensors()
    return {k: np.asarray(getattr(t, k)) for k in ("rows", "row_ids", "field", "mask", "kind", "seat")}


def test_real_cg_observations(lay):
    games = 48
    env = cg.BatchEnv(None, games, seed=3, threads=2)
    old, same = OldRules(rules.RuleConfig(3, 2), lay, games), rules.Rules(rules.RuleConfig(3, 2), lay, games)
    full = rules.Rules(rules.RuleConfig(3, 2, **{f.name: getattr(ALL_ON, f.name) for f in fields(ALL_ON)[2:]}), lay, games)
    rng = np.random.default_rng(0)
    turns = 0
    for _ in range(300):
        obs = observe(env)
        kind, legal = np.asarray(obs["kind"]), np.asarray(obs["mask"])
        a, b, c = old.mask(obs), same.mask(obs), full.mask(obs)
        assert np.array_equal(a, b)
        turn = kind == L.KIND_TURN
        turns += int(turn.sum())
        assert (c <= legal).all() and (c[turn].sum(1) > 0).all()
        assert np.array_equal(c[~turn], legal[~turn])
        act = np.zeros(games, np.int64)
        for i in range(games):
            if kind[i] == L.KIND_PREVIEW:
                act[i] = L.PREVIEW_OFFSET + rng.integers(L.PREVIEW_ACTIONS)
            else:  # a seat plays from the full mask, so all three see the same game
                act[i] = rng.choice(np.flatnonzero(c[i]))
        for r in (old, same, full):
            r.record(obs, act)
        env.act(act)
        done = np.flatnonzero(np.asarray(env.results().done))
        for r in (old, same, full):
            r.key[done], r.same[done], r.switches[done] = 0, 0, 0
        full.reset(done)
    assert turns > 1000
    assert sum(full.closed.values()) > 0


def test_defensive_setup_is_open_under_ko_threat(lay):
    R = rules
    r = R.Rules(R.RuleConfig(no_boost_when_ko=True), lay, 1)
    t = r.t_boost
    defensive = lambda name: bool(t[R.MOVE[name]].any() and not (t[R.MOVE[name]] & ~r.t_def_stats).any())  # noqa: E731
    assert defensive("irondefense") and defensive("cosmicpower") and defensive("amnesia") and defensive("minimize")
    assert not defensive("swordsdance") and not defensive("bulkup") and not defensive("calmmind") and not defensive("dragondance")
