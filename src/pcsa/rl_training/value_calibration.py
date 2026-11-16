"""Value calibration: the critic's value mapped so (v + 1) / 2 reads as a live win chance.

The critic learned in the simulator, which it wins far more than the ladder. A Platt fit p = sigmoid(a * v + b), per
checkpoint, keeps the ranking: v' = 2 * sigmoid(a * v + b) - 1. Set as `value_calibration: {a, b}` in neural.yaml.
"""

from __future__ import annotations

import math

import numpy as np

from pcsa.common.errors import ConfigError

KEYS = ("a", "b")


def check(setting: dict, where: str) -> None:
    """{} = off, else {a, b}: finite numbers, a > 0 (the map must keep the critic's order)."""
    if not setting:
        return
    if not isinstance(setting, dict) or set(setting) != set(KEYS):
        raise ConfigError(f"{where}: value_calibration must be {{}} or {{a: <number>, b: <number>}}, not {setting!r}")
    for key in KEYS:
        x = setting[key]
        if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x):
            raise ConfigError(f"{where}: value_calibration.{key} must be a finite number, not {x!r}")
    if setting["a"] <= 0:
        raise ConfigError(f"{where}: value_calibration.a must be above 0 (it keeps the critic's order)")


def mapper(setting: dict):
    """The value map of a checked setting: None when off."""
    if not setting:
        return None
    a, b = float(setting["a"]), float(setting["b"])

    def calibrated(v):
        z = np.clip(a * np.asarray(v, dtype=np.float64) + b, -30.0, 30.0)
        return 2.0 / (1.0 + np.exp(-z)) - 1.0

    return calibrated


def apply(value_map, values: np.ndarray) -> np.ndarray:
    """`values` through `value_map` (a mapper's result), unchanged when it is None."""
    return values if value_map is None else np.asarray(value_map(values), dtype=values.dtype)


def notes(value_map, value: float) -> dict:
    """A decision's value notes: v (calibrated when a map is set) and v_raw (the critic's own, only with a map)."""
    if value_map is None:
        return {"v": round(float(value), 4)}
    return {"v": round(float(value_map(np.float64(value))), 4), "v_raw": round(float(value), 4)}
