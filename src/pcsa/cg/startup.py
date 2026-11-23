"""Making CG for a run: the player's arguments resolved, its log_dir in the run folder, and the start-up status
check.

The neural player (`champions_gym.neural:load`) must report a clean contract check, no fallback hand-over and the
configured checkpoint in `status.player_status`; anything else stops the run before a battle.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from pcsa.common.config import CGConfig
from pcsa.common.paths import repo_root

NEURAL = "champions_gym.neural:load"
STATUS = {"type": "status"}
PYTHON = "{python}"  # in cg.command: the interpreter running pcsa
SEARCH_SLOW_S, SEARCH_HARD_S = 1.0, 4.0  # over cg.search.ms: the shell's slow and hard limits


def expects_model(cfg: CGConfig) -> bool:
    return cfg.player == NEURAL


def checkpoint(cfg: CGConfig) -> str | None:
    return (cfg.player_args or {}).get("checkpoint")


def _from_repo(path: str) -> str:
    return str(Path(path) if Path(path).is_absolute() else (repo_root() / path).resolve())


def resolved(cfg: CGConfig) -> CGConfig:
    """The config as CG is made from it: `{python}` in cg.command is this interpreter, a relative checkpoint is
    taken from the repo root, and the neural player's policy_args carry cg.rules, cg.search and cg.choice."""
    args = dict(cfg.player_args or {})
    if isinstance(args.get("checkpoint"), str):
        args["checkpoint"] = _from_repo(args["checkpoint"])
    if cfg.player == NEURAL:
        args["policy_args"] = {**(args.get("policy_args") or {}), **policy_settings(cfg)}
        if cfg.search.enabled:  # the shell hands the battle to its fallback after slow answers
            seconds = cfg.search.ms / 1000.0
            args["timeout_s"] = max(float(args.get("timeout_s") or 0.0), seconds + SEARCH_SLOW_S)
            args["hard_timeout_s"] = max(float(args.get("hard_timeout_s") or 0.0), seconds + SEARCH_HARD_S)
    command = [sys.executable if part == PYTHON else part for part in cfg.command] if cfg.command else None
    return cfg.model_copy(update={"command": command, "player_args": args or cfg.player_args})


def policy_settings(cfg: CGConfig) -> dict:
    """cg.rules, cg.search and cg.choice as the policy's keyword arguments (None: off)."""
    return {"rules": cfg.rules.args(), "search": cfg.search.args(), "choice": cfg.choice.args()}


def policy_kwargs(cfg: CGConfig) -> dict:
    """The neural player's Policy keyword arguments as the live player gets them, for tools that build one."""
    args = resolved(cfg).player_args or {}
    return {"device": args["device"], **args.get("policy_args", {})}


def with_log_dir(cfg: CGConfig, folder: Path) -> CGConfig:
    """The config with CG's log_dir, and a model player's, set to `folder`."""
    player_args = cfg.player_args
    if expects_model(cfg) or "log_dir" in (player_args or {}):
        player_args = {**(player_args or {}), "log_dir": str(folder)}
    if cfg.transport == "inprocess":
        return cfg.model_copy(update={"args": {**cfg.args, "log_dir": str(folder)}, "player_args": player_args})
    command = list(cfg.command or [])
    at = command.index("--log-dir") if "--log-dir" in command[:-1] else None
    if at is None:
        command += ["--log-dir", str(folder)]
    else:
        command[at + 1] = str(folder)
    return cfg.model_copy(update={"command": command, "player_args": player_args})


def session_args(cfg: CGConfig) -> dict:
    """The in-process Session's keyword arguments: cg.args with the player."""
    out = dict(cfg.args)
    if cfg.player:
        out["player"] = cfg.player
    if cfg.player_args:
        out["player_args"] = cfg.player_args
    return out


def command_line(cfg: CGConfig) -> list[str]:
    """The subprocess's argv: cg.command with the player."""
    out = list(cfg.command or [])
    if cfg.player:
        out += ["--player", cfg.player]
    if cfg.player_args:
        out += ["--player-args", json.dumps(cfg.player_args)]
    return out


def _same_file(a: Any, b: str) -> bool:
    if not isinstance(a, str):
        return False
    a, b = (os.path.normcase(os.path.abspath(p.replace("\\", "/"))) for p in (a, b))  # either separator, any OS
    return a == b


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
