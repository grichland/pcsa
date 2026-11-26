"""pcsa's own rules (pcsa.network.live_rules): no status moves when both actives are a listed species, and no
team-preview pick that brings two Mega stone holders, and Mega Evolution whenever it is open."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cg = pytest.importorskip("champions_gym")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pcsa.network import cg_adapter, live_rules, vocab  # noqa: E402
from pcsa.network import layout as L  # noqa: E402
from tests import configs  # noqa: E402
from tests.cg import neural  # noqa: E402
from tests.cg.neural import CHECKPOINT  # noqa: E402
from tests.network.test_preview_search import preview_request  # noqa: E402
from tests.network.test_rules import scores_of, turn_request  # noqa: E402

PRIMARINA = vocab.tables()["species"].index("primarina")
STATUS, PHYSICAL = 2, 0


@pytest.fixture(scope="module")
def layout() -> L.Layout:
    return cg_adapter.layout()


def categories(layout: L.Layout) -> np.ndarray:
    return np.asarray({f.name: f for f in layout.features.values()}["pokemon.move.fact.category"].columns())


def obs_of(layout: L.Layout, legal: list[int], ours: int, theirs: int) -> dict:
    """Our active (row 0) with a status move in slot 0 and a physical one in slot 1, against `theirs`."""
    w = layout.widths
    rows, ids = np.zeros((1, L.ROWS, w["rows"]), np.float32), np.zeros((1, L.ROWS, w["row_ids"]), np.int64)
    rows[0, 0, layout.col(L.F_IS_ACTIVE)] = 1.0
    ids[0, 0, layout.col(L.F_SPECIES)], ids[0, L.OPP_ACTIVE, layout.col(L.F_SPECIES)] = ours, theirs
    cat = categories(layout).reshape(4, 3)
    rows[0, 0, cat[0, STATUS]], rows[0, 0, cat[1, PHYSICAL]] = 1.0, 1.0
    mask = np.zeros((1, L.OWN_ACTIONS), np.uint8)
    mask[0, legal] = 1
    return {"mask": mask, "rows": rows, "row_ids": ids}


def open_of(rule: live_rules.MirrorRule, obs: dict) -> list[int]:
    return [int(a) for a in np.flatnonzero(rule.mask(obs, obs["mask"])[0])]


def test_a_mirror_closes_status_moves_and_their_mega_copies(layout) -> None:
    rule = live_rules.MirrorRule(layout, ["primarina"])
    assert open_of(rule, obs_of(layout, [0, 1, 4, 5, 8], PRIMARINA, PRIMARINA)) == [1, 5, 8]
    assert rule.closed == 1


def test_other_matchups_and_unlisted_mirrors_are_left_alone(layout) -> None:
    rule = live_rules.MirrorRule(layout, ["primarina"])
    legal = [0, 1, 8]
    assert open_of(rule, obs_of(layout, legal, PRIMARINA, PRIMARINA + 1)) == legal
    assert open_of(rule, obs_of(layout, legal, PRIMARINA + 1, PRIMARINA + 1)) == legal
    assert rule.closed == 0


def test_a_status_move_stays_when_nothing_else_is_open(layout) -> None:
    rule = live_rules.MirrorRule(layout, ["primarina"])
    assert open_of(rule, obs_of(layout, [0, 4], PRIMARINA, PRIMARINA)) == [0, 4]


def test_an_unknown_species_is_refused(layout) -> None:
    with pytest.raises(ValueError, match="unknown species"):
        live_rules.MirrorRule(layout, ["notamon"])


def test_the_policy_applies_it_after_trainings_masks_and_reports_it() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip(f"no test checkpoint at {CHECKPOINT}")
    policy = neural.policy()  # the shipped rules, with status_free_mirrors [primarina]
    assert policy.info["rule_masks"]["status_free_mirrors"] == ["primarina"]
    arrays: Any = turn_request(cg.BatchEnv(None, 1, seed=3, threads=1))
    lay = policy.layout
    rows, ids = np.asarray(arrays.rows).copy(), np.asarray(arrays.row_ids).copy()
    me = int(rows[0, : L.OWN_PICKS, lay.col(L.F_IS_ACTIVE)].argmax())
    ids[0, me, lay.col(L.F_SPECIES)] = ids[0, L.OPP_ACTIVE, lay.col(L.F_SPECIES)] = PRIMARINA
    slot = int(np.flatnonzero(np.asarray(arrays.mask)[0, :4])[0])
    cat = categories(lay).reshape(4, 3)
    rows[0, me, cat[slot]] = 0.0
    rows[0, me, cat[slot, STATUS]] = 1.0
    policy.reset([0])
    scores, extras = scores_of(policy, arrays._replace(rows=rows, row_ids=ids), 0)
    assert slot in extras["closed"] and scores[slot] == float("-inf")


STONES = {1, 2, 3}  # CG's test team: Gyarados, Golisopod and Salamence hold their stones
TWO_STONE_PICKS = [a for a, c in enumerate(L.preview_combos()) if len(STONES & set(c)) >= 2]


def preview_open(rule: live_rules.LineupRule, arrays: Any, m: np.ndarray) -> list[int]:
    return [int(a) for a in np.flatnonzero(rule.mask(arrays.row_ids, arrays.kind, m)[0])]


def test_the_lineup_rule_closes_every_pick_with_two_stone_holders(layout) -> None:
    arrays, rule = preview_request(), live_rules.LineupRule(layout)
    held = [vocab.name("items", int(i)) for i in np.asarray(arrays.row_ids)[0, :6, layout.col(L.F_ITEM)]]
    assert {i for i, item in enumerate(held) if item in vocab.mega_stones()} == STONES
    m = np.ones((1, L.PREVIEW_ACTIONS), np.uint8)
    assert len(TWO_STONE_PICKS) == 30  # 10 sets of three, 3 leads each
    assert preview_open(rule, arrays, m) == sorted(set(range(L.PREVIEW_ACTIONS)) - set(TWO_STONE_PICKS))
    assert rule.closed == 1


def test_the_lineup_rule_keeps_a_pick_when_nothing_else_is_open_and_skips_turns(layout) -> None:
    arrays, rule = preview_request(), live_rules.LineupRule(layout)
    m = np.zeros((1, L.PREVIEW_ACTIONS), np.uint8)
    m[0, TWO_STONE_PICKS[:2]] = 1
    assert preview_open(rule, arrays, m) == TWO_STONE_PICKS[:2]
    turn = arrays._replace(kind=np.full_like(np.asarray(arrays.kind), L.KIND_TURN))
    assert preview_open(rule, turn, np.ones((1, L.PREVIEW_ACTIONS), np.uint8)) == list(range(L.PREVIEW_ACTIONS))
    assert rule.closed == 0
    assert live_rules.build({**configs.rules(), "one_mega_lineup": False}, layout)[2] is None


def test_the_policy_never_picks_two_stone_holders_and_lists_the_closed_picks() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip(f"no test checkpoint at {CHECKPOINT}")
    policy = neural.policy()  # the shipped rules, with one_mega_lineup on
    assert policy.info["rule_masks"]["one_mega_lineup"] is True
    out = policy.decide(preview_request(), [0])
    scores, extras = out["scores"][0, L.PREVIEW_OFFSET :], out["extras"][0]
    assert int(scores.argmax()) not in TWO_STONE_PICKS
    assert sorted(extras["closed"]) == [L.PREVIEW_OFFSET + a for a in TWO_STONE_PICKS]
    assert all(scores[a] == float("-inf") for a in TWO_STONE_PICKS)


def test_the_mega_rule_closes_plain_moves_whose_mega_copy_is_open() -> None:
    rule = live_rules.MegaRule()
    m = np.zeros((2, L.OWN_ACTIONS), np.uint8)
    m[0, [0, 1, 4, 8]] = 1  # move 0 can Mega Evolve, move 1 has no Mega copy open
    m[1, [0, 9]] = 1  # no Mega open
    obs = {"kind": np.array([L.KIND_TURN, L.KIND_TURN])}
    out = rule.mask(obs, m)
    assert list(np.flatnonzero(out[0])) == [1, 4, 8] and list(np.flatnonzero(out[1])) == [0, 9]
    assert rule.closed == 1
    replace = {"kind": np.array([L.KIND_REPLACE, L.KIND_TURN])}
    assert (rule.mask(replace, m)[0] == m[0]).all()


def test_the_policy_mega_evolves_whenever_it_can() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip(f"no test checkpoint at {CHECKPOINT}")
    policy = neural.policy()  # the shipped rules, with always_mega on
    assert policy.info["rule_masks"]["always_mega"] is True
    arrays: Any = turn_request(cg.BatchEnv(None, 1, seed=3, threads=1))
    mask = np.asarray(arrays.mask).copy()
    mask[0, 4:8] = mask[0, :4]  # as if our active could Mega Evolve
    policy.reset([0])
    scores, extras = scores_of(policy, arrays._replace(mask=mask), 0)
    plain = [int(a) for a in np.flatnonzero(mask[0, :4])]
    assert plain and all(scores[a] == float("-inf") for a in plain) and set(plain) <= set(extras["closed"])
    assert int(np.argmax(scores)) >= 4


def _guarded(open_: list[int], action):  # noqa: ANN001, ANN202


    m = np.zeros((1, 11), np.float32)
    m[0, open_] = 1.0
    rules = SimpleNamespace(save=lambda g: (), load=lambda g, saved: None, mask=lambda obs: m)
    seat = SimpleNamespace(decision=7, obs={"mask": m}, before=(), rules=rules)
    policy = SimpleNamespace(rule_cfg=object(), turn_rules=None, seats={0: seat})
    return live_rules.guard(policy, 7, action)


def test_the_guard_keeps_the_action_that_goes_out_inside_the_masks() -> None:
    assert _guarded([4, 5, 8], 0) == 4  # a plain move with its Mega copy open: the Mega copy
    assert _guarded([2, 9], 1) == 2  # a closed move: another move
    assert _guarded([1, 10], 8) == 10  # a closed switch: another switch
    assert _guarded([1], 9) == 1  # no switch open: a move
    assert _guarded([0, 8], 8) == 8 and _guarded([0], 12) == 12 and _guarded([0], None) is None
    assert live_rules.guard(SimpleNamespace(rule_cfg=None), 7, 3) == 3  # no masks: unchanged
