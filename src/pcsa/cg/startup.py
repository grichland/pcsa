"""Making CG for a run (LIVE 0.9 section 3): the command resolved, its log_dir in the run folder, and the start-up
status check.

The model player (`champions_gym.neural:load`) must report a clean contract check, no fallback hand-over and the
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


def _from_repo(path: str) -> str:
    return str(Path(path) if Path(path).is_absolute() else (repo_root() / path).resolve())


SEARCH_SLOW_S, SEARCH_HARD_S = 1.0, 4.0  # over cg.search.ms: the shell's slow and hard limits


def resolved(cfg: CGConfig) -> CGConfig:
    """The config as CG is made from it: `{python}` in cg.command is this interpreter, a relative checkpoint in
    the player_args is taken from the repo root, and with cg.search.enabled the neural player's policy_args carry
    the search settings (and cg.choice its `choice`, unless that is the network's own pick)."""
    player, args = _player(cfg)
    if isinstance(args.get("checkpoint"), str):
        args = {**args, "checkpoint": _from_repo(args["checkpoint"])}
    choice = cfg.choice.args() if player == NEURAL else {}
    if choice:
        args = {**args, "policy_args": {**(args.get("policy_args") or {}), "choice": choice}}
    if cfg.search.enabled and player == NEURAL:
        args = {**args, "policy_args": {**(args.get("policy_args") or {}), **search_args(cfg)}}
        # the shell hands the battle to its fallback after slow answers: its limits must clear the search time
        seconds = cfg.search.ms / 1000.0
        args["timeout_s"] = max(float(args.get("timeout_s") or 0.0), seconds + SEARCH_SLOW_S)
        args["hard_timeout_s"] = max(float(args.get("hard_timeout_s") or 0.0), seconds + SEARCH_HARD_S)
    if cfg.transport == "inprocess":
        if "player_args" not in cfg.args:
            return cfg
        return cfg.model_copy(update={"args": {**cfg.args, "player_args": args}})
    command = [sys.executable if part == PYTHON else part for part in cfg.command or []]
    args_at = _flag(command, "--player-args")
    if args_at is not None and ("checkpoint" in args or cfg.search.enabled or choice):
        command[args_at + 1] = json.dumps(args)
    return cfg.model_copy(update={"command": command})


def search_args(cfg: CGConfig) -> dict:
    """cg.search as the policy's keyword arguments."""
    s = cfg.search
    return {"search": True, "search_ms": s.ms, "worlds": s.worlds, "depth": s.depth, "outcomes": s.outcomes,
            "c": s.c, "n0": s.n0, "batch": s.batch, "min_sims": s.min_sims, "top_ours": s.top_ours,
            "top_theirs": s.top_theirs, "settle": [s.settle_sims, s.settle_share, s.settle_gap],
            "from_turn": s.from_turn}  # fmt: skip


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
