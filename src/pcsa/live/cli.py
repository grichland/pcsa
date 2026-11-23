"""`pcsa run`, `pcsa check` and `pcsa nav`: the live-play commands, registered by `pcsa.cli`.

Every setting comes from agent.yaml (`--set section.key=value` for a single run; `run.battles`, `run.name`,
`run.seed` and `decide.player` included). Building the parsers imports no phone code.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pcsa.common.config import Config

COMMANDS = ("run", "check", "nav")


def add_parsers(commands, config_args: Callable[[argparse.ArgumentParser], None]) -> None:
    run = commands.add_parser("run", help="play Ranked Singles battles on the phone")
    config_args(run)
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

    return play_command(config, Path(args.resume) if args.resume else None)
