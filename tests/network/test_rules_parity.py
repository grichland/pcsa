"""pcsa's rules.py is pcsa-former's: the same source apart from its header, and the same masks and counts at every
step of seeded simulator games, for the shipped settings and for the settings of the newer checkpoints.

Runs when pcsa-former is checked out next to this repository (or at PCSA_FORMER) and skips otherwise."""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

cg = pytest.importorskip("champions_gym")
pytest.importorskip("torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pcsa.common.paths import repo_root  # noqa: E402
from pcsa.network import cg_adapter, rules  # noqa: E402
from pcsa.network import layout as L  # noqa: E402
from tests import configs  # noqa: E402

FORMER = Path(os.environ.get("PCSA_FORMER") or repo_root().parent / "pcsa-former")
OURS = repo_root() / "src" / "pcsa" / "network" / "rules.py"
# what training uses for the scratch line and its gold fine-tunes
NEWER = {"max_same_move": 3, "max_switches": 2, "max_boosts": 3, "max_heals": 3, "no_null_moves": True,
         "no_boost_when_ko": True, "no_failing_status": True, "no_redundant_field": True, "no_double_protect": True,
         "no_wasted_self": True}  # fmt: skip
OBS = ("rows", "row_ids", "field", "mask", "kind", "seat")


@pytest.fixture(scope="module")
def former():
    """pcsa-former's rules and layout modules."""
    if not (FORMER / "pcsa_former" / "rules.py").is_file():
        pytest.skip(f"no pcsa-former checkout at {FORMER}")
    sys.path.insert(0, str(FORMER))
    try:
        return importlib.import_module("pcsa_former.rules"), importlib.import_module("pcsa_former.cg_adapter")
    finally:
        sys.path.remove(str(FORMER))


def _body(text: str, import_line: str) -> str:
    """The source after the module docstring, with the layout import as pcsa writes it."""
    rest = text.split('"""', 2)[2]
    return rest.replace(import_line, "from . import layout as L", 1)


def test_the_source_is_the_same(former) -> None:
    theirs = (FORMER / "pcsa_former" / "rules.py").read_text(encoding="utf-8")
    ours = _body(OURS.read_text(encoding="utf-8"), "from . import layout as L")
    assert ours == _body(theirs, "from pcsa_former import layout as L")


def test_the_settings_are_the_same(former) -> None:
    ours = [(f.name, f.default) for f in dataclasses.fields(rules.RuleConfig)]
    theirs = [(f.name, f.default) for f in dataclasses.fields(former[0].RuleConfig)]
    assert ours == theirs
    assert set(configs.rules()) == {name for name, _ in ours}


@pytest.mark.parametrize("settings", [configs.rules(), NEWER], ids=["shipped", "newer"])
def test_the_same_masks_on_simulator_games(former, settings) -> None:
    their_rules, their_cg = former
    games, rng = 32, np.random.default_rng(5)
    ours = rules.Rules(rules.RuleConfig(**settings), cg_adapter.layout(), games)
    theirs = their_rules.Rules(their_rules.RuleConfig(**settings), their_cg.layout(), games)
    env = cg.BatchEnv(None, games, seed=7, threads=2)
    turns = 0
    for step in range(400):
        t = env.observe_tensors()
        obs = {k: np.asarray(getattr(t, k)).copy() for k in OBS}
        a, b = ours.mask(obs), theirs.mask(obs)
        assert np.array_equal(a, b), f"step {step}"
        kind = obs["kind"]
        turns += int((kind == L.KIND_TURN).sum())
        act = np.array([L.PREVIEW_OFFSET + rng.integers(L.PREVIEW_ACTIONS) if kind[i] == L.KIND_PREVIEW
                        else rng.choice(np.flatnonzero(a[i])) for i in range(games)], np.int64)  # fmt: skip
        ours.record(obs, act)
        theirs.record(obs, act)
        for x, y in zip(ours.save(step % games), theirs.save(step % games), strict=True):
            assert np.array_equal(x, y)
        env.act(act)
        done = np.flatnonzero(np.asarray(env.results().done))
        ours.reset(done)
        theirs.reset(done)
    assert turns > 1000
    assert ours.counts() == theirs.counts()
    if settings is NEWER:
        assert sum(ours.closed.values()) > 0
