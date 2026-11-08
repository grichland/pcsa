"""A p2 seat's protocol and request mirrored into the p1 view our Battle expects, and a request's legal actions."""

from __future__ import annotations

import json
import re

from poke_env.battle import Move

from pcsa.common.types import SemanticAction
from pcsa.players.base import LegalAction

_SIDE = re.compile(r"(?<![A-Za-z0-9])p([12])(?=[a-d]?:|\||$)")
_SIDE_JSON = re.compile(r'(?<=")p([12])(?=[a-d]?:|")')


def _swap(match: re.Match) -> str:
    return "p" + ("2" if match.group(1) == "1" else "1")


def mirror_protocol(lines: list[str]) -> list[str]:
    return [_SIDE.sub(_swap, line) for line in lines]


def mirror_request(request: dict) -> dict:
    return json.loads(_SIDE_JSON.sub(_swap, json.dumps(request)))


def _player_lines(lines: list[str]) -> list[str]:
    """`|player|` lines name us "me" (what the replayed Battle expects) and come in side order."""
    out = []
    for line in lines:
        if line.startswith("|player|p1|"):
            out.append("|player|p1|me||")
        elif line.startswith("|player|p2|"):
            out.append("|player|p2|opponent||")
        else:
            out.append(line)
    first = next((i for i, line in enumerate(out) if line.startswith("|player|")), None)
    if first is not None and out[first].startswith("|player|p2|"):
        second = next((i for i in range(first + 1, len(out)) if out[i].startswith("|player|p1|")), None)
        if second is not None:
            out[first], out[second] = out[second], out[first]
    return out


def as_p1(side: str | None, lines: list[str], request: dict) -> tuple[list[str], dict]:
    """A seat's protocol lines and request as p1 sees them (p2's are mirrored)."""
    if side == "p2":
        return _player_lines(mirror_protocol(lines)), mirror_request(request)
    return _player_lines(list(lines)), request


def legal_actions(battle) -> tuple[LegalAction, ...]:
    """poke-env's valid orders as LegalActions (the phone-side encoding carries no meaning here)."""
    actions = []
    for order in battle.valid_orders:
        if isinstance(order.order, Move):
            moves = list(battle.active_pokemon.moves) if battle.active_pokemon else []
            slot = moves.index(order.order.id) if order.order.id in moves else 0
            actions.append(LegalAction(SemanticAction("move", move=min(slot, 3), mega=order.mega), order))
        else:
            actions.append(LegalAction(SemanticAction("switch", slot=0), order))
    return tuple(actions)
