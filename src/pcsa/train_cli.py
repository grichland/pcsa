"""`pcsa train ...`: the training commands and their parser. Today only `sprites` (the sprite network)."""

from __future__ import annotations

import argparse

from pcsa.sprite_training import sprite_cli


def train_parser(commands) -> None:
    train = commands.add_parser("train", help="the sprite network trainer (needs the train extra)")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    sprite_cli.add_parser(train_commands)


def train(args: argparse.Namespace) -> int:
    if args.train_command == "sprites":
        from pcsa.sprite_training import sprite_train

        return sprite_train.run(args)
    raise ValueError(f"unknown train command {args.train_command!r}")
