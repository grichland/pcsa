"""The network copied from pcsa-former gives the same outputs as pcsa-former's own code, and a checkpoint's own
settings and layout checks hold.

The parity tests run when pcsa-former is checked out next to this repository (or at PCSA_FORMER) and skip
otherwise; PCSA_PARITY_CHECKPOINT selects another checkpoint than the test one."""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cg = pytest.importorskip("champions_gym")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pcsa.common.paths import repo_root  # noqa: E402
from pcsa.network import cg_adapter, checkpoint, model  # noqa: E402
from pcsa.network import layout as L  # noqa: E402
from pcsa.network.policy import INPUTS  # noqa: E402
from tests.cg.neural import CHECKPOINT  # noqa: E402

FORMER = Path(os.environ.get("PCSA_FORMER") or repo_root().parent / "pcsa-former")
PARITY_CHECKPOINT = Path(os.environ.get("PCSA_PARITY_CHECKPOINT") or CHECKPOINT)


@pytest.fixture(scope="module")
def former():
    """pcsa-former's checkpoint, cg_adapter and model modules."""
    if not (FORMER / "pcsa_former" / "model.py").is_file():
        pytest.skip(f"no pcsa-former checkout at {FORMER}")
    sys.path.insert(0, str(FORMER))
    try:
        return tuple(importlib.import_module(f"pcsa_former.{m}") for m in ("checkpoint", "cg_adapter", "model"))
    finally:
        sys.path.remove(str(FORMER))


@pytest.fixture(scope="module")
def batch() -> tuple[dict, np.ndarray]:
    """A fixed batch of real observations (previews and turns) from seeded simulator games, and their kinds."""
    env = cg.BatchEnv(None, 4, seed=11, threads=1)
    steps = []
    for _ in range(8):
        t = env.observe_tensors()
        steps.append({k: np.asarray(getattr(t, k)).copy() for k in (*INPUTS, "action_mask", "kind")})
        env.act(env.builtin_actions("max_damage"))
    cat = {k: np.concatenate([s[k] for s in steps]) for k in steps[0]}
    b = {k: torch.from_numpy(np.ascontiguousarray(cat[k])) for k in INPUTS}
    b["preview_mask"] = torch.from_numpy(np.ascontiguousarray(cat["action_mask"]))[:, L.PREVIEW_OFFSET :]
    return b, cat["kind"]


def _outputs(net, b: dict) -> dict:
    net.eval()
    with torch.no_grad():
        return net.forward(dict(b), beliefs=True)


def test_the_copied_network_gives_pcsa_formers_outputs(former, batch) -> None:
    if not PARITY_CHECKPOINT.is_file():
        pytest.skip(f"no checkpoint at {PARITY_CHECKPOINT}")
    f_checkpoint, f_adapter, _ = former
    ours, _ = checkpoint.load(str(PARITY_CHECKPOINT), cg_adapter.layout(), cg_adapter.learnset(), "cpu")
    theirs, _ = f_checkpoint.load(str(PARITY_CHECKPOINT), f_adapter.layout(), f_adapter.learnset(), "cpu")
    a, b = _outputs(ours, batch[0]), _outputs(theirs, batch[0])
    assert set(a) == set(b) and {"opp_logits", "matrix", "p_opp", "pi", "score", "value"} <= set(a)
    for key in a:
        if torch.is_tensor(a[key]):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True, msg=key)


def test_the_config_takes_every_field_pcsa_former_saves(former) -> None:
    f_model = former[2]
    assert {f.name for f in dataclasses.fields(model.Config)} == {f.name for f in dataclasses.fields(f_model.Config)}


def _saved(tmp_path: Path, name: str, change) -> Path:
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    change(ck)
    path = tmp_path / name
    torch.save(ck, path)
    return path


def test_a_checkpoint_carries_its_own_policy_cap(tmp_path, batch) -> None:
    if not CHECKPOINT.is_file():
        pytest.skip(f"no test checkpoint at {CHECKPOINT}")
    path = _saved(tmp_path, "cap.pt", lambda ck: ck["config"].update(max_prob=0.8))
    net, _ = checkpoint.load(str(path), cg_adapter.layout(), cg_adapter.learnset(), "cpu")
    assert net.cfg.max_prob == 0.8
    out = _outputs(net, batch[0])
    turns = (batch[1] == L.KIND_TURN) & (out["legal"].sum(-1) >= 2).numpy()
    assert turns.any() and float(out["pi"][torch.from_numpy(turns)].max()) <= 0.8 + 1e-5


def test_a_layout_or_contract_mismatch_is_refused(tmp_path) -> None:
    if not CHECKPOINT.is_file():
        pytest.skip(f"no test checkpoint at {CHECKPOINT}")
    for key, value in (("hash", "not-this-layout"), ("contract", "0.0")):
        path = _saved(tmp_path, f"{key}.pt", lambda ck, k=key, v=value: ck["layout"].update({k: v}))
        with pytest.raises(checkpoint.LayoutMismatch):
            checkpoint.load(str(path), cg_adapter.layout(), cg_adapter.learnset(), "cpu")
