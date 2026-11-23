"""A recorded battle for the tests: one battle played with the test checkpoint (`CHECKPOINT_NAME`), the messages
the controller sent to CG and CG's answers (`DATA`)."""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).with_name("data") / "neural-148-battle-1.jsonl"
CHECKPOINT_NAME = "league-20261118-g-iter-00148.pt"


def rows() -> list[dict]:
    return [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]


def sent() -> list[dict]:
    return [r["message"] for r in rows() if r["dir"] == "sent"]


def answers() -> dict:
    """seq -> CG's recorded answer (`picks` under the key "preview")."""
    out = {}
    for r in rows():
        m = r["message"]
        if r["dir"] == "received" and "picks" in m:
            out["preview"] = m
        elif r["dir"] == "received" and "order" in m:
            out[m["seq"]] = m
    return out


class Replayer:
    """A Player that gives the recorded picks and actions, keeping every Battle it is asked about."""

    name = "replayer"

    def __init__(self) -> None:
        self.recorded = answers()
        self.battles: list = []

    def teampreview(self, battle):
        self.battles.append(battle)
        return self.recorded["preview"]["picks"]

    def choose_move(self, battle):
        self.battles.append(battle)
        return self.recorded[battle.decision]["notes"]["action"]
