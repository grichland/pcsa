"""The `pcsa` command.

pcsa setup [--with sim] [--with train] [--check]   install or check tools, output folder, device
pcsa doctor                                         same checks, no installs
pcsa observe / calibrate / scout / sim play / train ...
pcsa run / nav                                      live play on the phone
"""

from __future__ import annotations

import argparse
import sys

from pcsa import __version__
from pcsa.common.errors import ConfigError
from pcsa.live import cli as live
from pcsa.players.factory import KINDS
from pcsa.train_cli import train, train_parser


def _config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="agent.yaml (default: config/agent.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="section.key=value")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcsa", description="Pokémon Champions singles from a phone screen")
    parser.add_argument("--version", action="version", version=f"pcsa {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="install pinned tools, create the output folder, check the phone")
    setup.add_argument("--with", dest="groups", action="append", choices=["sim", "train"], default=[])
    setup.add_argument("--check", action="store_true", help="report only; install nothing")
    commands.add_parser("doctor", help="check tools, output folder and phone without installing")

    observe = commands.add_parser("observe", help="capture and classify the screen in a loop (no taps)")
    _config_args(observe)
    observe.add_argument("--seconds", type=float, default=60.0)
    observe.add_argument("--interval", type=float, default=1.0)

    calibrate = commands.add_parser("calibrate", help="measure this phone with the game on the lobby")
    _config_args(calibrate)

    scout = commands.add_parser("scout", help="record a manual walk through the game and report changes (no taps)")
    _config_args(scout)
    scout.add_argument("--interval", type=float, default=1.0)

    sim = commands.add_parser("sim", help="simulator games between players (needs `pcsa setup --with sim`)")
    sim_commands = sim.add_subparsers(dest="sim_command", required=True)
    play = sim_commands.add_parser("play", help="paired games: A vs B, sides swapped within each pair")
    _config_args(play)
    play.add_argument("--a", default="custom_heuristic", choices=list(KINDS))
    play.add_argument("--b", default="max_damage", choices=list(KINDS))
    play.add_argument("--battles", type=int, default=20)
    play.add_argument("--seed", type=int, default=1)
    play.add_argument(
        "--noisy", action="store_true", help="teams use any catalog build, not only each species' top one"
    )
    play.add_argument("--port", type=int, default=None)

    train_parser(commands)
    live.add_parsers(commands, _config_args)
    return parser


def _load(args: argparse.Namespace):
    from pcsa.common import paths
    from pcsa.common.config import load_config

    return load_config(args.config or paths.config_dir() / "agent.yaml", args.overrides)


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
    return _sim(args) if args.command == "sim" else train(args)


def _sim(args: argparse.Namespace) -> int:
    from pcsa.run.sim_command import play_command

    config = _load(args)
    outcome = play_command(config, args.a, args.b, args.battles, args.seed, noisy=args.noisy, port=args.port)
    for line in outcome.lines:
        print(line)
    return outcome.code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return dispatch(args)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
