"""Requests rebuilt from the reference decisions (tests/reference): the Offer from the recorded Showdown request
(move rows from the active member, switches from the side), the team from tests/data/own_exports.json.gz and the
picks from the reference battles."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from pcsa.decide.api import Hints, MoveOffer, Offer, OwnTeam, Request, SwitchOffer

TESTS = Path(__file__).resolve().parent.parent
REFERENCE = TESTS / "reference"


@dataclass(frozen=True)
class Case:
    key: str  # run/battle/record
    request: Request
    raw: dict  # the reference entry
    active: str | None  # the request's active member


def _lines(name: str):
    with gzip.open(REFERENCE / name, "rt", encoding="utf-8") as f:
        yield from map(json.loads, f)


def member_name(member: dict) -> str:
    return member["ident"].split(": ", 1)[1]


def _hp(condition: str) -> tuple[int, int] | None:
    left, _, rest = condition.partition("/")
    return (int(left), int(rest.split()[0])) if rest and left.isdigit() else None


def offer_from_request(request: dict) -> Offer:
    side = request["side"]["pokemon"]
    active = next((p for p in side if p.get("active")), None)
    forced = "faint" if request.get("forceSwitch") else None
    slot = (request.get("active") or [{}])[0]
    gone = active is not None and active["condition"].endswith("fnt")
    moves = (
        ()
        if forced or gone
        else tuple(
            MoveOffer(k, m["id"], (m["pp"], m["maxpp"]) if "pp" in m else None, bool(m.get("disabled")))
            for k, m in enumerate(slot.get("moves") or [])
        )
    )
    switches = (
        ()
        if slot.get("trapped")
        else tuple(
            SwitchOffer(member_name(p), _hp(p["condition"]))
            for p in side
            if not p.get("active") and not p["condition"].endswith("fnt")
        )
    )
    can_mega = None if forced else bool(slot.get("canMegaEvo"))
    return Offer(moves, True, switches, can_mega, forced)


@cache
def teams() -> dict[str, tuple[str, tuple[str, ...]]]:
    with gzip.open(TESTS / "data" / "own_exports.json.gz", "rt", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    return {
        c["run"]: (c["own_team_txt"] or c["decision_exports"][0], tuple(m["name"] for m in c["team"]["members"]))
        for c in cases
    }


@cache
def cases() -> tuple[Case, ...]:
    picks = {(b["run"], b["battle"]): tuple(b["picks"]) for b in _lines("battles-regen-off.jsonl.gz")}
    out = []
    for d in _lines("decisions-regen-off.jsonl.gz"):
        export, names = teams()[d["run"]]
        team = OwnTeam(export, names, picks[(d["run"], d["battle"])])
        side = d["request"]["side"]["pokemon"]
        active = next((member_name(p) for p in side if p.get("active")), None)
        request = Request(
            f"{d['run']}/{d['battle']}",
            d["sequence"],
            (),
            "none",
            team,
            (None,) * 6,
            offer_from_request(d["request"]),
            Hints(),
            0.0,
        )
        out.append(Case(f"{d['run']}/{d['battle']}/{d['record']}", request, d, active))
    return tuple(out)
