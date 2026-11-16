"""Value calibration (pcsa.rl_training.value_calibration) and its neural player setting."""

from __future__ import annotations

import numpy as np
import pytest

from pcsa.common.errors import ConfigError
from pcsa.players.neural import NeuralParams
from pcsa.rl_training import value_calibration as vc


def test_off_by_default() -> None:
    assert NeuralParams().value_calibration == {}
    assert vc.mapper({}) is None


def test_map_is_two_sigmoid_minus_one_and_keeps_order() -> None:
    f = vc.mapper({"a": 1.93, "b": -1.16})
    v = np.array([-1.0, -0.5, 0.0, 0.5, 0.9, 1.0])
    got = f(v)
    want = 2 / (1 + np.exp(-(1.93 * v - 1.16))) - 1
    assert np.allclose(got, want)
    assert np.all(np.diff(got) > 0)
    assert np.all((got > -1) & (got < 1))
    # the live fit lowers an optimistic 0.9 (raw win chance 0.95) to about 0.64
    assert (f(np.array([0.9]))[0] + 1) / 2 == pytest.approx(0.640, abs=0.005)


@pytest.mark.parametrize(
    "bad",
    [
        {"a": 1.0},
        {"a": 1.0, "b": 0.0, "c": 1.0},
        {"a": 0.0, "b": 0.0},
        {"a": -1.0, "b": 0.0},
        {"a": True, "b": 0.0},
        {"a": float("nan"), "b": 0.0},
        {"a": "1", "b": 0.0},
    ],
)
def test_bad_settings_fail_at_load(bad: dict) -> None:
    with pytest.raises(ConfigError):
        NeuralParams(value_calibration=bad).check("test")


def test_good_setting_passes() -> None:
    NeuralParams(value_calibration={"a": 1.9287, "b": -1.15785}).check("test")


def test_apply_and_notes() -> None:
    f = vc.mapper({"a": 1.93, "b": -1.16})
    v = np.array([0.0, 0.9], dtype=np.float32)
    assert vc.apply(None, v) is v
    got = vc.apply(f, v)
    assert got.dtype == np.float32 and np.allclose(got, f(v), atol=1e-6)
    assert vc.notes(None, 0.9) == {"v": 0.9}
    n = vc.notes(f, 0.9)
    assert n["v_raw"] == 0.9 and n["v"] == pytest.approx(float(f(np.float64(0.9))), abs=1e-4)
