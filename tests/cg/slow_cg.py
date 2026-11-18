"""CG's Session that answers one live request late, for a real controller-side timeout. In process: entry
"tests.cg.slow_cg:SlowSession" with slow_seq and slow_s among the args. As a process (CG's own arguments):
    python -m tests.cg.slow_cg --slow-seq N --slow-s S --player P [--player-args JSON] [--seed N] [--log-dir D]
"""

from __future__ import annotations

import argparse
import json
import sys
import time


class SlowSession:
    def __init__(self, slow_seq: int = -1, slow_s: float = 0.0, **session) -> None:
        from champions_gym.live import Session

        self.session = Session(**session)
        self.slow_seq, self.slow_s = slow_seq, slow_s

    def handle(self, message):
        reply = self.session.handle(message)
        live = isinstance(message, dict) and message.get("type") == "decision_request" and not message.get("replay")
        if live and message.get("seq") == self.slow_seq:
            time.sleep(self.slow_s)  # noqa: TID251 (runs in CG's Python, without the controller)
        return reply


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.cg.slow_cg")
    parser.add_argument("--slow-seq", type=int, default=-1)
    parser.add_argument("--slow-s", type=float, default=0.0)
    parser.add_argument("--player", default="tower")
    parser.add_argument("--player-args", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args(argv)
    session: dict = {"player": args.player, "seed": args.seed, "log_dir": args.log_dir}
    if args.player_args:
        session["player_args"] = json.loads(args.player_args)
    cg = SlowSession(args.slow_seq, args.slow_s, **session)
    for line in sys.stdin:
        if line.strip():
            sys.stdout.write(json.dumps(cg.handle(json.loads(line))) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
