"""`pcsa run`, `pcsa check` and `pcsa nav`: the live-play commands, registered by `pcsa.cli`.

Building the parsers imports no phone code; each command imports its module when it runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pcsa.decide.load import PLAYERS

if TYPE_CHECKING:
    from pcsa.common.config import Config

COMMANDS = ("run", "check", "nav")


def add_parsers(commands, config_args: Callable[[argparse.ArgumentParser], None]) -> None:
    run = commands.add_parser("run", help="play Ranked Singles battles on the phone")
    config_args(run)
    run.add_argument("--player", choices=PLAYERS, default=None, help="Player (default: agent.yaml decide.player)")
    run.add_argument("--battles", type=int, default=None, help="decided battles before the run finishes")
    run.add_argument("--max-total", type=int, default=None, help="bound on battles started, forfeits included")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--name", default="run")
    run.add_argument("--resume", default=None, help="an existing run directory to continue")

    check = commands.add_parser("check", help="connect to the device and classify the current screen (no taps)")
    config_args(check)

    nav = commands.add_parser("nav", help="navigate to a screen (default: the lobby)")
    config_args(nav)
    nav.add_argument("--to", default="lobby")
    nav.add_argument("--seconds", type=float, default=180.0)


def dispatch(args: argparse.Namespace, config: Config) -> int:
    if args.command == "check":
        from pcsa.live.play import check_command

        return check_command(config)
    if args.command == "nav":
        from pcsa.live.phone import nav_command

        return nav_command(config, args.to, args.seconds)
    from pcsa.live.play import play_command

    return play_command(
        config,
        player_name=args.player,
        battles=args.battles,
        max_total=args.max_total,
        seed=args.seed,
        name=args.name,
        resume=Path(args.resume) if args.resume else None,
    )
