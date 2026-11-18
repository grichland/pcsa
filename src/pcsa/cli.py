"""The `pcsa` command.

    pcsa setup [--with torch] [--check] | doctor          install or check the tools, output folder and phone
    pcsa run / check / nav                                 live play on the phone
    pcsa observe / calibrate / scout                       the phone without playing
    pcsa watch                                             read-only dashboard over a run
    pcsa train sprites                                     the sprite network trainer (torch extra)

Each command imports its own code when it runs: `pcsa watch` and `pcsa setup` load no phone, CG or torch code.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from typing import TYPE_CHECKING

from pcsa import __version__
from pcsa.common.errors import ConfigError
from pcsa.live import cli as live

if TYPE_CHECKING:
    from pcsa.common.config import Config

SETUP_GROUPS = ("torch",)
WATCH_PORT = 8765


def _config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="agent.yaml (default: config/agent.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="config override")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcsa", description="Pokémon Champions singles from a phone screen")
    parser.add_argument("--version", action="version", version=f"pcsa {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="install pinned tools, create the output folder, check the phone")
    setup.add_argument("--with", dest="groups", action="append", choices=SETUP_GROUPS, default=[])
    setup.add_argument("--check", action="store_true", help="report only; install nothing")
    commands.add_parser("doctor", help="check tools, Champions Gym, output folder and phone without installing")

    live.add_parsers(commands, _config_args)

    observe = commands.add_parser("observe", help="capture and classify the screen in a loop (no taps)")
    _config_args(observe)
    observe.add_argument("--seconds", type=float, default=60.0)
    observe.add_argument("--interval", type=float, default=1.0)

    calibrate = commands.add_parser("calibrate", help="measure this phone with the game on the lobby")
    _config_args(calibrate)

    scout = commands.add_parser("scout", help="record a manual walk through the game and report changes (no taps)")
    _config_args(scout)
    scout.add_argument("--interval", type=float, default=1.0)

    watch = commands.add_parser("watch", help="read-only page over a run directory (default: the latest run)")
    watch.add_argument("--run", default=None, help="run folder (default: the newest under output/runs)")
    watch.add_argument("--port", type=int, default=WATCH_PORT)
    watch.add_argument("--no-browser", action="store_true")

    _train_parser(commands)
    return parser


def _train_parser(commands) -> None:
    from pcsa.sprite_training import sprite_cli  # argparse only; the trainer imports torch when it runs

    train = commands.add_parser("train", help="the sprite network trainer (needs the torch extra)")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    sprite_cli.add_parser(train_commands)


def _load(args: argparse.Namespace) -> Config:
    from pcsa.common import paths
    from pcsa.common.config import load_config

    return load_config(args.config or paths.config_dir() / "agent.yaml", args.set)


def dispatch(args: argparse.Namespace) -> int:
    if args.command in ("setup", "doctor"):
        from pcsa.common import setup

        groups = set(args.groups) if args.command == "setup" else set()
        return setup.run(groups, check_only=args.command == "doctor" or args.check)
    if args.command in ("observe", "calibrate", "scout"):
        from pcsa.phone import commands, scout

        config = _load(args)
        if args.command == "observe":
            return commands.observe_command(config, args.seconds, args.interval)
        if args.command == "calibrate":
            return commands.calibrate_command(config)
        return scout.scout_command(config, args.interval)
    if args.command in live.COMMANDS:
        return live.dispatch(args, _load(args))
    if args.command == "watch":
        return _watch(args)
    return _train(args)


def _watch(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pcsa.run.watch import latest_run, serve

    run_dir = Path(args.run) if args.run else latest_run()
    if run_dir is None:
        print("no run found under output/runs", file=sys.stderr)
        return 1
    print(f"watching {run_dir} at http://127.0.0.1:{args.port}/ (read-only; Ctrl+C stops it)", flush=True)
    try:
        return serve(run_dir, "127.0.0.1", args.port, not args.no_browser)
    except OSError as error:
        print(f"error: cannot serve on port {args.port}: {error}", file=sys.stderr)
        return 1


def _train(args: argparse.Namespace) -> int:
    if args.train_command != "sprites":
        raise ValueError(f"unknown train command {args.train_command!r}")
    if importlib.util.find_spec("torch") is None:
        raise ConfigError("`pcsa train sprites` needs the torch extra: uv sync --extra torch")
    from pcsa.sprite_training import sprite_train

    return sprite_train.run(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return dispatch(args)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
