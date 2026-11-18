"""Requests by hand for the CG Player's unit tests: the turn rig's team, a six-species preview, LIVE events."""

from __future__ import annotations

import sys
from dataclasses import replace

from pcsa.cg.fake_cg import FakeCG
from pcsa.decide.api import Hints, MoveOffer, Offer, Request, SwitchOffer, TeamPreviewRequest
from pcsa.live.battle_events import set_of
from pcsa.live.decision_record import decision_request
from pcsa.live.team_read import own_team
from tests.live.turn_rig import TEAM

TAG = "b-test"
PREVIEW = ["glimmora", "lucario", "garchomp", "archaludon", "kingambit", "dragonite"]
START = {"v": "0.7", "battle": TAG, "type": "battle_start", "own_team": [set_of(m) for m in TEAM.members],
         "opp_preview": PREVIEW}  # fmt: skip
TURN1 = ({"k": "turn", "n": 1}, {"k": "enter", "side": "opp", "mon": "glimmora"},
         {"k": "enter", "side": "own", "mon": "gholdengo"})  # fmt: skip
TURN2 = ({"k": "move", "side": "own", "mon": "gholdengo", "move": "makeitrain"}, {"k": "turn", "n": 2})
GHOLDENGO_ROWS = ("makeitrain", "shadowball", "trick", "nastyplot")
FAKE_CG = [sys.executable, "-m", "pcsa.cg.fake_cg"]
ENTRY = "pcsa.cg.fake_cg:FakeCG"


def team(picks=(0, 1, 2)):
    return own_team(TEAM, TEAM.names, list(picks))


def preview(deadline: float = 1e9) -> TeamPreviewRequest:
    return TeamPreviewRequest(TAG, own_team(TEAM, TEAM.names), tuple(PREVIEW), deadline, (START,))


def action_offer() -> Offer:
    return Offer(tuple(MoveOffer(k, m) for k, m in enumerate(GHOLDENGO_ROWS)), False,
                 (SwitchOffer("Garchomp"), SwitchOffer("Golisopod")), None)  # fmt: skip


def request(seq: int, events: tuple, history: tuple = (), offer: Offer | None = None, deadline: float = 1e9,
            picks=(0, 1, 2), **live) -> Request:  # fmt: skip
    req = Request(TAG, seq, (), "full", team(picks), tuple(PREVIEW), offer or action_offer(), Hints(), deadline,
                  (START, *events), history=history)  # fmt: skip
    return replace(req, live=decision_request(req, "forced" if req.offer.forced else "action", **live))


def entry(req: Request, fallback: dict | None = None, played: dict | None = None) -> dict:
    """Request.history's entry for a decision that was made on `req`."""
    out = {"events": len(req.events), "decision_request": req.live}
    out = {**out, "fallback": fallback} if fallback else out
    return {**out, "played": played} if played else out


def fresh_fake() -> None:
    FakeCG.instances.clear()
    FakeCG.slow, FakeCG.exit_at, FakeCG.exits, FakeCG.player_status = {}, None, 0, None


def problems() -> list[str]:
    return [p for cg in FakeCG.instances for p in cg.problems]
