"""`pcsa run` plays Ranked Singles on the phone; `pcsa check` connects to the
device and classifies the current screen without a tap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pcsa.common import paths
from pcsa.common.config import load_config
from pcsa.common.errors import ConfigError
from pcsa.decide.load import PLAYERS

COMMANDS = ("run", "check")


def config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(paths.config_dir() / "agent.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="config override")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pcsa", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    play = commands.add_parser("run", help="play Ranked Singles battles on the phone")
    config_args(play)
    play.add_argument("--player", choices=PLAYERS, default=None, help="Player (default: agent.yaml decide.player)")
    play.add_argument("--battles", type=int, default=None, help="decided battles before the run finishes")
    play.add_argument("--max-total", type=int, default=None, help="bound on battles started, forfeits included")
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--name", default="run")
    play.add_argument("--resume", default=None, help="an existing run directory to continue")
    check = commands.add_parser("check", help="connect to the device and classify the current screen (no taps)")
    config_args(check)
    return root


def dispatch(args: argparse.Namespace) -> int:
    config = load_config(args.config, overrides=args.set)
    if args.command == "check":
        from pcsa.live.play import check_command

        return check_command(config)
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


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return dispatch(args)
    except ConfigError as error:
        sys.stderr.write(f"pcsa: {error}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
