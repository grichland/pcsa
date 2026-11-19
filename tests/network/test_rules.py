"""The rule masks in pcsa.network.policy (the same move at most 4 turns in a row, at most 3 switches by choice in a
row) and the belief-named opponent slots, on the shipped test checkpoint."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cg = pytest.importorskip("champions_gym")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pcsa.network import cg_adapter, rules  # noqa: E402
from pcsa.network import layout as L  # noqa: E402
from pcsa.network.policy import Policy  # noqa: E402
from tests.cg.neural import CHECKPOINT  # noqa: E402

MOVE_A, MOVE_B, SPECIES = 11, 22, 5


@pytest.fixture(scope="module")
def layout() -> L.Layout:
    return cg_adapter.layout()


@pytest.fixture(scope="module")
def policy() -> Policy:
    if not CHECKPOINT.is_file():
        pytest.skip(f"no test checkpoint at {CHECKPOINT}")
    return Policy(str(CHECKPOINT), threads=2)


def obs_of(layout: L.Layout, legal: list[int], kind: int = L.KIND_TURN, species: int = SPECIES) -> dict:
    """One game's request: row 0 on the field with moves A and B in slots 0 and 1."""
    w = layout.widths
    rows, ids = np.zeros((1, L.ROWS, w["rows"]), np.float32), np.zeros((1, L.ROWS, w["row_ids"]), np.int64)
    rows[0, 0, layout.col(L.F_IS_ACTIVE)] = 1.0
    ids[0, 0, layout.col(L.F_SPECIES)] = species
    cols = layout[L.F_MOVE_ID].columns()
    ids[0, 0, cols[0]], ids[0, 0, cols[1]] = MOVE_A, MOVE_B
    mask = np.zeros((1, L.OWN_ACTIONS), np.uint8)
    mask[0, legal] = 1
    return {"mask": mask, "seat": np.zeros(1, np.int64), "kind": np.full(1, kind), "rows": rows, "row_ids": ids}


def play(r: rules.Rules, obs: dict, action: int) -> list[int]:
    """The open actions under the rules, then `action` recorded."""
    open_ = [int(a) for a in np.flatnonzero(r.mask(obs)[0])]
    r.record(obs, np.array([action]))
    return open_


def test_the_same_move_is_closed_after_four_turns_in_a_row(layout) -> None:
    r = rules.Rules(rules.RuleConfig(), layout, 1)
    legal = [0, 1, 4, 5, 8]
    for _ in range(4):
        assert play(r, obs_of(layout, legal), 0) == legal
    assert play(r, obs_of(layout, legal), 4) == [1, 5, 8]  # plain and with Mega; Mega of the same move still counts
    assert r.mask(obs_of(layout, legal))[0].tolist()[:2] == [0, 1]
    r.record(obs_of(layout, legal), np.array([1]))
    assert play(r, obs_of(layout, legal), 0) == legal  # another move ends the run


def test_a_pokemon_with_one_usable_move_is_left_alone(layout) -> None:
    r = rules.Rules(rules.RuleConfig(), layout, 1)
    for _ in range(6):
        assert play(r, obs_of(layout, [0, 8]), 0) == [0, 8]


def test_switches_are_closed_after_three_in_a_row_and_a_replacement_does_not_count(layout) -> None:
    r = rules.Rules(rules.RuleConfig(), layout, 1)
    legal = [0, 1, 8, 9]
    for _ in range(2):
        assert play(r, obs_of(layout, legal), 8) == legal
    assert play(r, obs_of(layout, [8, 9], kind=L.KIND_REPLACE), 9) == [8, 9]  # neither counts nor resets
    assert play(r, obs_of(layout, legal), 8) == legal
    assert play(r, obs_of(layout, legal), 0) == [0, 1]
    assert play(r, obs_of(layout, legal), 8) == legal
    assert r.mask(obs_of(layout, [8, 9]))[0].tolist()[8:10] == [1, 1]  # no move open: the switches stay


def test_the_limits_are_settings_and_a_battle_end_clears_the_state(layout) -> None:
    r = rules.Rules(rules.RuleConfig(max_same_move=2, max_switches=0), layout, 1)
    legal = [0, 1, 8]
    play(r, obs_of(layout, legal), 0), play(r, obs_of(layout, legal), 0)
    assert play(r, obs_of(layout, legal), 8) == [1, 8]
    for _ in range(5):
        assert 8 in play(r, obs_of(layout, legal), 8)  # no switch rule
    play(r, obs_of(layout, legal), 0), play(r, obs_of(layout, legal), 0)
    r.reset(np.array([0]))
    assert play(r, obs_of(layout, legal), 0) == legal


def turn_request(env) -> object:
    """Play the fixed game on with max_damage until seat 0 has a turn request with two moves open."""
    while True:
        t = env.observe_tensors()
        if int(t.kind[0]) == cg.KIND_TURN and int(t.seat[0]) == 0 and int(np.asarray(t.mask)[0, :4].sum()) >= 2:
            return t
        env.act(env.builtin_actions("max_damage"))


def scores_of(policy: Policy, arrays, slot: int) -> tuple[list[float], dict]:
    out = policy.decide(arrays, [slot])
    return out["scores"][0, :11].tolist(), out["extras"][0]


def test_the_policy_closes_a_repeated_move_and_counts_a_decision_once(policy: Policy) -> None:
    env = cg.BatchEnv(None, 1, seed=3, threads=1)
    arrays = turn_request(env)
    slot, move = 7, int(np.flatnonzero(np.asarray(arrays.mask)[0, :4])[0])
    policy.reset([slot])
    decision = int(np.asarray(arrays.decision).reshape(-1)[0])
    seat_obs = lambda d: arrays._replace(decision=np.array([d]))  # noqa: E731
    for n in range(4):
        scores, extras = scores_of(policy, seat_obs(decision + n), slot)
        assert extras["closed"] == [] and scores[move] > float("-inf")
        policy.played(decision + n, move)
        policy.played(decision + n, move)  # told twice (a refusal answered again): counted once
        scores, extras = scores_of(policy, seat_obs(decision + n), slot)  # asked again: as before the request
        assert extras["closed"] == []
        policy.played(decision + n, move)
    scores, extras = scores_of(policy, seat_obs(decision + 4), slot)
    mega = move + 4
    assert move in extras["closed"] and scores[move] == float("-inf") and extras["pi"][move] == 0.0
    assert mega not in [a for a in range(11) if np.asarray(arrays.mask)[0, a]] or mega in extras["closed"]
    assert max(scores[:8]) > float("-inf") and sum(extras["pi"]) == pytest.approx(1.0, abs=1e-5)
    policy.reset([slot])  # the next battle starts clean
    assert scores_of(policy, seat_obs(decision + 5), slot)[1]["closed"] == []


def test_a_replayed_request_counts_its_action_and_the_setting_turns_the_rules_off(policy: Policy) -> None:
    env = cg.BatchEnv(None, 1, seed=3, threads=1)
    arrays = turn_request(env)
    move = int(np.flatnonzero(np.asarray(arrays.mask)[0, :4])[0])
    at = lambda d: arrays._replace(decision=np.array([d]))  # noqa: E731
    policy.reset([8])
    for n in range(4):
        policy.observe(at(100 + n), [8], [move])
    assert move in scores_of(policy, at(104), 8)[1]["closed"]
    off = Policy(str(CHECKPOINT), threads=2, rule_masks=False)
    for n in range(6):
        scores, extras = scores_of(off, at(200 + n), 0)
        assert "closed" not in extras and scores[move] > float("-inf")
        off.played(200 + n, move)
    two = Policy(str(CHECKPOINT), threads=2, max_same_move=2)
    for n in range(2):
        scores_of(two, at(300 + n), 0)
        two.played(300 + n, move)
    assert move in scores_of(two, at(302), 0)[1]["closed"]


def test_belief_slots_name_their_unseen_columns(policy: Policy) -> None:
    env = cg.BatchEnv(None, 1, seed=3, threads=1)
    arrays = turn_request(env)
    plain = scores_of(policy, arrays, 20)[1]
    assert "opp_seen" not in plain and sum(c == "" for c in plain["cols"][:4]) == 1
    named = Policy(str(CHECKPOINT), threads=2)
    named.net.cfg = dataclasses.replace(named.net.cfg, belief_slots=True)  # the same weights: no new parameters
    e = scores_of(named, arrays, 0)[1]
    seen = e["opp_seen"]
    assert len(seen) == 4 and all(isinstance(x, bool) for x in seen) and not all(seen)
    cols = e["cols"][:4]
    assert all(isinstance(c, str) and c for c in cols) and len(set(cols)) == 4 and e["cols"][4:8] == cols
    assert [c for c, s in zip(cols, seen, strict=True) if s] == [c for c in plain["cols"][:4] if c]  # shown ones kept
    assert sum(e["p_opp"]) == pytest.approx(1.0, abs=1e-5)
    for a in range(11):
        assert e["score"][a] == pytest.approx(sum(m * p for m, p in zip(e["matrix"][a], e["p_opp"], strict=True)), abs=1e-5)
