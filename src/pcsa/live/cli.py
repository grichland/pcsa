"""`pcsa run` and `pcsa nav`: the live-play commands, registered by `pcsa.cli`."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from pcsa.players.factory import LIVE_KINDS

COMMANDS = ("run", "nav")


def add_parsers(commands, config_args: Callable[[argparse.ArgumentParser], None]) -> None:
    nav = commands.add_parser("nav", help="navigate to a screen (default: the lobby)")
    config_args(nav)
    nav.add_argument("--to", default="lobby")
    nav.add_argument("--seconds", type=float, default=180.0)

    run = commands.add_parser("run", help="live evaluation: play Ranked Singles battles on the phone")
    config_args(run)
    run.add_argument(
        "--player", choices=LIVE_KINDS, default=None, help="player kind (default: agent.yaml policy.kind)"
    )
    run.add_argument("--battles", type=int, default=None, help="decided battles before the run finishes")
    run.add_argument("--max-total", type=int, default=None, help="bound on battles started, forfeits included")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--name", default="run")
    run.add_argument("--resume", default=None, help="an existing run directory to continue")


def dispatch(args: argparse.Namespace, config) -> int:
    if args.command == "nav":
        from pcsa.live.phone import nav_command

        return nav_command(config, args.to, args.seconds)
    from pcsa.live.play import run_command

    return run_command(
        config,
        player_kind=args.player,
        battles=args.battles,
        max_total=args.max_total,
        seed=args.seed,
        name=args.name,
        resume=Path(args.resume) if args.resume else None,
    )
