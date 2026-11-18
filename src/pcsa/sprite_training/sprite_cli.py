"""`pcsa train sprites` arguments (argparse only: the trainer itself needs torch and is imported on use)."""

from __future__ import annotations


def add_parser(train_commands) -> None:
    sprites = train_commands.add_parser(
        "sprites", help="train the sprite network on reference views and recorded runs, measure it, export it"
    )
    sprites.add_argument("--source", default=None, help="local sprite source indexes (default: output/sprite_source)")
    sprites.add_argument("--runs", default=None, help="recorded runs with labelled crops (default: output/runs)")
    sprites.add_argument("--out", default=None, help="result folder (default: output/sprite_net/<stamp>)")
    sprites.add_argument("--steps", type=int, default=8000, help="training batches")
    sprites.add_argument("--seed", type=int, default=0)
    sprites.add_argument("--test-runs", type=int, default=2, help="newest runs held out for the measurement")
    sprites.add_argument(
        "--holdout-species",
        type=float,
        default=0.0,
        help="fraction of species whose real crops are kept out of training",
    )
    sprites.add_argument("--refresh-data", action="store_true", help="extract the crops again (cached otherwise)")
    sprites.add_argument("--install", action="store_true", help="copy the result into assets/sprites")
