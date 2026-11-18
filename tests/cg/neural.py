"""CG's model player (champions_gym.neural:load with pcsa.network.policy) behind pcsa.cg.serve, for the tests: this
venv's Python (CG_NEURAL_PYTHON to use another), the test checkpoint in assets/models (CG_NEURAL_CHECKPOINT), on CPU.
CG_NEURAL_TRANSPORT=inprocess runs CG in this process instead."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from functools import cache
from pathlib import Path

from pcsa.cg.startup import NEURAL
from pcsa.common.config import CGConfig
from pcsa.common.paths import repo_root

PYTHON = Path(os.environ.get("CG_NEURAL_PYTHON") or sys.executable)
CHECKPOINT = Path(
    os.environ.get("CG_NEURAL_CHECKPOINT") or repo_root() / "assets" / "models" / "league-20261118-g-iter-00148.pt"
)
ENV = {"CUDA_VISIBLE_DEVICES": "-1"}  # an empty value does not hide the GPU on Windows
POLICY = "pcsa.network.policy:Policy"
SERVE = "pcsa.cg.serve"
TRANSPORT = os.environ.get("CG_NEURAL_TRANSPORT", "subprocess")


def player_args(checkpoint: Path | str = CHECKPOINT) -> dict:
    return {"policy": POLICY, "checkpoint": str(checkpoint), "device": "cpu", "policy_args": {"threads": 4},
            "fallback": "tower", "timeout_s": 2.0, "hard_timeout_s": 6.0}  # fmt: skip


def subprocess_config(checkpoint: Path | str = CHECKPOINT, timeout_s: float = 8.0) -> CGConfig:
    command = [str(PYTHON), "-m", SERVE, "--player", NEURAL, "--player-args", json.dumps(player_args(checkpoint))]
    return CGConfig(transport="subprocess", command=command, env=ENV, run_log_dir="cg", timeout_s=timeout_s)


def inprocess_config(
    checkpoint: Path | str = CHECKPOINT, timeout_s: float = 8.0, entry: str | None = None
) -> CGConfig:
    args = {"player": NEURAL, "player_args": player_args(checkpoint)}
    return CGConfig(transport="inprocess", entry=entry or "champions_gym.live:Session", args=args,
                    run_log_dir="cg", timeout_s=timeout_s)  # fmt: skip


def config(checkpoint: Path | str = CHECKPOINT, timeout_s: float = 8.0) -> CGConfig:
    """The connection CG_NEURAL_TRANSPORT names."""
    make = inprocess_config if TRANSPORT == "inprocess" else subprocess_config
    return make(checkpoint, timeout_s)


@cache
def missing() -> str | None:
    """Why the model player cannot run here, or None."""
    if importlib.util.find_spec("torch") is None and Path(sys.executable) == PYTHON:
        return "the model player needs torch (the torch extra)"
    if TRANSPORT == "inprocess" and os.environ.get("CUDA_VISIBLE_DEVICES") != ENV["CUDA_VISIBLE_DEVICES"]:
        return "in process, set CUDA_VISIBLE_DEVICES=-1 first"
    if not PYTHON.is_file():
        return f"no Python at {PYTHON}"
    if not CHECKPOINT.is_file():
        return f"no test checkpoint at {CHECKPOINT}"
    return None


def slow_config(seq: int, slow_s: float, timeout_s: float) -> CGConfig:
    """The model player behind a Session that answers live request `seq` `slow_s` late (tests/slow_cg.py)."""
    cfg = config(timeout_s=timeout_s)
    if cfg.transport == "inprocess":
        args = {**cfg.args, "slow_seq": seq, "slow_s": slow_s}
        return cfg.model_copy(update={"entry": "tests.cg.slow_cg:SlowSession", "args": args})
    command = list(cfg.command or [])
    command[1:3] = ["-m", "tests.cg.slow_cg", "--slow-seq", str(seq), "--slow-s", str(slow_s)]
    return cfg.model_copy(update={"command": command, "python_path": [*cfg.python_path, str(repo_root())]})
