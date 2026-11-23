"""A worked example from a recorded run (battle-10, decision 0002, turn 3) and small Offer builders."""

from __future__ import annotations

from dataclasses import replace

from pcsa.decide.api import Hints, MoveOffer, Offer, OwnTeam, Request, SwitchOffer
from tests.decide.reference import teams

RUN = "20261116-133224-run"
ROW_IDS = ("iceshard", "earthquake", "iciclespear", "endeavor")


def gold_team() -> OwnTeam:
    export, names = teams()[RUN]
    return OwnTeam(export, names, (4, 1, 2))


def unread_offer() -> Offer:
    return Offer(
        moves=tuple(MoveOffer(k, move) for k, move in enumerate(ROW_IDS)),
        moves_read=False,
        switches=(SwitchOffer("Garchomp"), SwitchOffer("Golisopod")),
        can_mega_evolve=None,
    )


def read_offer(*, disabled=(), pp_zero=(), mega: bool | None = False) -> Offer:
    moves = tuple(
        MoveOffer(k, move, (0 if k in pp_zero else 10, 10), k in disabled) for k, move in enumerate(ROW_IDS)
    )
    return replace(unread_offer(), moves=moves, moves_read=True, can_mega_evolve=mega)


def forced_offer(leaving: str | None = None, kind="faint") -> Offer:
    switches = (SwitchOffer("Mamoswine"), SwitchOffer("Garchomp"), SwitchOffer("Golisopod"))
    return Offer((), False, tuple(s for s in switches if s.name != "Mamoswine" or leaving), None, kind, leaving)


def request(offer: Offer | None = None, *, request_id: int = 2, deadline: float = 1040.0, hints=None) -> Request:
    return Request(
        battle_tag="20261116T143532-8a2d4236",
        request_id=request_id,
        rows=(),
        log_quality="full",
        team=gold_team(),
        opponent_preview=("Glimmora", "Dragalge", "Lucario", "Garchomp", "Mimikyu", None),
        offer=offer or unread_offer(),
        hints=hints
        or Hints(
            pp_counted={"iceshard": (19, 20), "earthquake": (11, 12), "iciclespear": (20, 20), "endeavor": (8, 8)}
        ),
        deadline=deadline,
    )
