"""Making CG for a run (LIVE 0.9 section 3): its log_dir in the run folder, and the start-up status check.

The model player (`champions_gym.neural:load`) must report a clean contract check, no fallback hand-over and the
configured checkpoint in `status.player_status`; anything else stops the run before a battle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pcsa.common.config import CGConfig

NEURAL = "champions_gym.neural:load"
STATUS = {"type": "status"}


def _flag(command: list[str], name: str) -> int | None:
    return command.index(name) if name in command[:-1] else None


def _player(cfg: CGConfig) -> tuple[str | None, dict]:
    """The Session's player and player_args, from cg.args or cg.command."""
    if cfg.transport == "inprocess":
        return cfg.args.get("player"), dict(cfg.args.get("player_args") or {})
    command = list(cfg.command or [])
    at, args_at = _flag(command, "--player"), _flag(command, "--player-args")
    player = command[at + 1] if at is not None else None
    try:
        args = json.loads(command[args_at + 1]) if args_at is not None else {}
    except ValueError:
        args = {}
    return player, args if isinstance(args, dict) else {}


def expects_model(cfg: CGConfig) -> bool:
    return _player(cfg)[0] == NEURAL


def checkpoint(cfg: CGConfig) -> str | None:
    return _player(cfg)[1].get("checkpoint")


def with_log_dir(cfg: CGConfig, folder: Path) -> CGConfig:
    """The config with CG's log_dir, and a model player's, set to `folder`."""
    player, player_args = _player(cfg)
    model = player == NEURAL or "log_dir" in player_args
    if cfg.transport == "inprocess":
        args = {**cfg.args, "log_dir": str(folder)}
        if model:
            args["player_args"] = {**player_args, "log_dir": str(folder)}
        return cfg.model_copy(update={"args": args})
    command = list(cfg.command or [])
    at = _flag(command, "--log-dir")
    if at is None:
        command += ["--log-dir", str(folder)]
    else:
        command[at + 1] = str(folder)
    args_at = _flag(command, "--player-args")
    if model and args_at is not None:
        command[args_at + 1] = json.dumps({**player_args, "log_dir": str(folder)})
    return cfg.model_copy(update={"command": command})


def _same_file(a: Any, b: str) -> bool:
    if not isinstance(a, str):
        return False
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def status_problems(reply: Any, cfg: CGConfig) -> list[str]:
    """Why CG's answer to {"type":"status"} does not allow a battle; [] when it does."""
    status = reply.get("status") if isinstance(reply, dict) else None
    if not isinstance(status, dict):
        return [f"no status answer: {reply}"]
    found = status.get("player_status")
    if found is None:
        return ["no player_status: the model player is not in place"] if expects_model(cfg) else []
    if not isinstance(found, dict):
        return [f"player_status is not an object: {found}"]
    out = []
    if found.get("layout_ok") is not True:
        out.append(f"layout_ok is {found.get('layout_ok')}")
    if found.get("mismatch") != []:
        out.append(f"mismatch {found.get('mismatch')}")
    if found.get("allow_mismatch") is not False:
        out.append(f"allow_mismatch is {found.get('allow_mismatch')}")
    if found.get("handed_to_fallback") is not False:
        out.append(f"handed_to_fallback is {found.get('handed_to_fallback')}")
    wanted = checkpoint(cfg)
    if wanted is not None and not _same_file(found.get("checkpoint"), wanted):
        out.append(f"checkpoint {found.get('checkpoint')} is not the configured {wanted}")
    return out


def stderr_tail(path: Path | None, lines: int = 12) -> str:
    """The end of the subprocess's stderr, for a clear start-up error."""
    if path is None or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return "\n".join(text[-lines:])
