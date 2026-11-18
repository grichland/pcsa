"""CG's model player (champions_gym.neural:load) for the tests and the step-11 checks: a Python with torch (the pcsa
venv, CG_NEURAL_PYTHON), the test checkpoint (CG_NEURAL_CHECKPOINT; never the shipped config's), on CPU.
CG_NEURAL_TRANSPORT=inprocess runs CG in this process instead (only in a Python with torch: the pcsa venv's)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from functools import cache
from pathlib import Path

from pcsa.cg.startup import NEURAL
from pcsa.common.config import CGConfig
from pcsa.common.paths import repo_root

ROOT = repo_root().parent
PYTHON = Path(os.environ.get("CG_NEURAL_PYTHON") or ROOT / "pcsa" / ".venv" / "Scripts" / "python.exe")
CHECKPOINT = Path(
    os.environ.get("CG_NEURAL_CHECKPOINT") or ROOT / "pcsa-former" / "runs" / "league-20261118-g" / "iter-00125.pt"
)
PATHS = [ROOT / "champions-gym" / "python", ROOT / "pcsa-former", ROOT / "champions-gym" / "docs" / "examples"]
ENV = {"CUDA_VISIBLE_DEVICES": "-1"}  # an empty value does not hide the GPU on Windows
POLICY = "pcsa_former_policy:PcsaFormerPolicy"  # pcsa_former.cg_policy once it is copied there
TRANSPORT = os.environ.get("CG_NEURAL_TRANSPORT", "subprocess")


def player_args(checkpoint: Path | str = CHECKPOINT) -> dict:
    return {"policy": POLICY, "checkpoint": str(checkpoint), "device": "cpu", "policy_args": {"threads": 4},
            "fallback": "tower", "timeout_s": 2.0, "hard_timeout_s": 6.0}  # fmt: skip


def subprocess_config(checkpoint: Path | str = CHECKPOINT, timeout_s: float = 8.0) -> CGConfig:
    command = [str(PYTHON), "-m", "champions_gym.live", "--player", NEURAL,
               "--player-args", json.dumps(player_args(checkpoint))]  # fmt: skip
    return CGConfig(transport="subprocess", command=command, python_path=[str(p) for p in PATHS], env=ENV,
                    run_log_dir="cg", timeout_s=timeout_s)  # fmt: skip


def inprocess_config(
    checkpoint: Path | str = CHECKPOINT, timeout_s: float = 8.0, entry: str | None = None
) -> CGConfig:
    """For a controller running in a Python with torch (the step-11 in-process check)."""
    args = {"player": NEURAL, "player_args": player_args(checkpoint)}
    return CGConfig(transport="inprocess", entry=entry or "champions_gym.live:Session", args=args,
                    python_path=[str(p) for p in PATHS], run_log_dir="cg", timeout_s=timeout_s)  # fmt: skip


def config(checkpoint: Path | str = CHECKPOINT, timeout_s: float = 8.0) -> CGConfig:
    """The connection CG_NEURAL_TRANSPORT names."""
    make = inprocess_config if TRANSPORT == "inprocess" else subprocess_config
    return make(checkpoint, timeout_s)


@cache
def missing() -> str | None:
    """Why the model player cannot run here, or None."""
    if TRANSPORT == "inprocess":
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        if importlib.util.find_spec("torch") is None:
            return "CG_NEURAL_TRANSPORT=inprocess needs torch in this Python"
        return None if cuda == ENV["CUDA_VISIBLE_DEVICES"] else "in process, set CUDA_VISIBLE_DEVICES=-1 first"
    if not PYTHON.is_file():
        return f"no Python with torch at {PYTHON}"
    if not CHECKPOINT.is_file():
        return f"no test checkpoint at {CHECKPOINT}"
    if not all(p.is_dir() for p in PATHS):
        return f"missing one of {PATHS}"
    probe = "import torch, champions_gym.neural, pcsa_former, pcsa_former_policy"
    env = {**os.environ, **ENV, "PYTHONPATH": os.pathsep.join(str(p) for p in PATHS)}
    done = subprocess.run([str(PYTHON), "-c", probe], env=env, capture_output=True, text=True, timeout=120)
    return None if done.returncode == 0 else f"the model's Python cannot import it: {done.stderr.strip()[-300:]}"


def slow_config(seq: int, slow_s: float, timeout_s: float) -> CGConfig:
    """The model player behind a Session that answers live request `seq` `slow_s` late (tests/slow_cg.py)."""
    cfg = config(timeout_s=timeout_s)
    if cfg.transport == "inprocess":
        args = {**cfg.args, "slow_seq": seq, "slow_s": slow_s}
        return cfg.model_copy(update={"entry": "tests.cg.slow_cg:SlowSession", "args": args})
    command = list(cfg.command or [])
    command[1:3] = ["-m", "tests.cg.slow_cg", "--slow-seq", str(seq), "--slow-s", str(slow_s)]
    return cfg.model_copy(update={"command": command, "python_path": [*cfg.python_path, str(repo_root())]})
