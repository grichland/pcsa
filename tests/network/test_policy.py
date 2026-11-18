"""pcsa.network.policy on the shipped test checkpoint, on CPU: the extras' shapes, score = matrix x p_opp, and the
belief heads changing nothing else (pi, value and the chosen action with beliefs on and off)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
cg = pytest.importorskip("champions_gym")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pcsa.network import decide  # noqa: E402
from pcsa.network.policy import Policy  # noqa: E402
from tests.cg.neural import CHECKPOINT  # noqa: E402


@pytest.fixture(scope="module")
def policy() -> Policy:
    if not CHECKPOINT.is_file():
        pytest.skip(f"no test checkpoint at {CHECKPOINT}")
    return Policy(str(CHECKPOINT), threads=2)


def observation(kind: int, turns: int = 0):
    """A fixed game's arrays at its first request of `kind` (after `turns` more turn requests)."""
    env = cg.BatchEnv(None, 1, seed=1, threads=1)
    while True:
        k = int(env.observe()[1][0])
        if k == kind and turns <= 0:
            return env.observe_tensors()
        if k == kind:
            turns -= 1
        env.act(env.builtin_actions("max_damage"))


def check_belief(belief: list, preview: bool) -> None:
    assert len(belief) == 6
    for e in belief:
        assert set(e) == {"species", "brought", "lead", "item", "ability", "moves"} and 0.0 <= e["brought"] <= 1.0
        assert (e["lead"] is not None) == preview
        assert len(e["item"]) <= 3 and len(e["ability"]) <= 3 and len(e["moves"]) <= 8
        assert all(isinstance(i, str) and 0.0 < p <= 1.0 for i, p in e["item"] + e["ability"] + e["moves"])
        assert e["species"] is None or e["moves"], e
    if preview:
        assert abs(sum(e["lead"] for e in belief) - 1.0) < 1e-4


def test_turn_extras(policy: Policy) -> None:
    arrays = observation(cg.KIND_TURN, turns=2)
    out = policy.decide(arrays, [0])
    e = out["extras"][0]
    assert len(e["pi"]) == len(e["score"]) == len(e["rows"]) == 11 and len(e["p_opp"]) == len(e["cols"]) == 13
    assert len(e["matrix"]) == 11 and all(len(r) == 13 for r in e["matrix"]) and e["tau"] > 0
    assert abs(e["tau"] - float(policy.net.temperature().detach())) < 1e-9
    for a in range(11):
        assert e["score"][a] == pytest.approx(sum(m * p for m, p in zip(e["matrix"][a], e["p_opp"], strict=True)), abs=1e-5)
    legal = [a for a in range(11) if arrays.mask[0][a]]
    assert all(e["pi"][a] == 0.0 for a in range(11) if a not in legal) and sum(e["pi"]) == pytest.approx(1.0, abs=1e-5)
    assert all(e["rows"][a] is not None for a in legal)
    assert e["rows"][:4] == e["rows"][4:8] and e["cols"][:4] == e["cols"][4:8]
    assert sum(c == "" for c in e["cols"][:4]) <= 1
    check_belief(e["belief"], preview=False)


def test_preview_extras(policy: Policy) -> None:
    out = policy.decide(observation(cg.KIND_PREVIEW), [1])
    e = out["extras"][0]
    assert set(e) == {"belief", "picks_top"}
    picks = e["picks_top"]
    assert 1 <= len(picks) <= 5 and all(len(set(t)) == 3 and all(0 <= i < 6 for i in t) for t, _ in picks)
    assert [p for _, p in picks] == sorted((p for _, p in picks), reverse=True)
    check_belief(e["belief"], preview=True)


@pytest.mark.parametrize("kind", [cg.KIND_PREVIEW, cg.KIND_TURN])
def test_beliefs_change_nothing_else(policy: Policy, kind: int) -> None:
    b = policy.batch(observation(kind, turns=1))
    with torch.no_grad():
        on, off = policy.net.forward(b, beliefs=True), policy.net.forward(b, beliefs=False)
    for key in ("pi", "value", "preview_pi", "score", "matrix", "p_opp"):
        assert torch.equal(on[key], off[key]), key
    assert "belief_item" in on and "belief_item" not in off
    for pi, legal in (("pi", "legal"), ("preview_pi", "preview_legal")):
        assert torch.equal(decide.choose(on[pi], on[legal], True)[0], decide.choose(off[pi], off[legal], True)[0])
