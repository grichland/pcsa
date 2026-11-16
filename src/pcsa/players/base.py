"""The player protocol and what a player is shown. The loop calls only these methods."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from poke_env.battle import Battle
from poke_env.player.battle_order import SingleBattleOrder

from pcsa.common.clock import Deadline
from pcsa.common.types import MoveInfo, SemanticAction


class Degrade(StrEnum):
    FULL = "full"
    TRACKED = "tracked"
    SCREEN = "screen"
    LEGAL_RANDOM = "legal_random"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class LegalAction:
    action: SemanticAction
    order: SingleBattleOrder | None

    @property
    def message(self) -> str:
        return self.order.message if self.order is not None else self.action.label()


@dataclass(frozen=True)
class ScreenFacts:
    moves: tuple[MoveInfo, ...] = ()
    mega_available: bool = False
    forced: bool = False
    active_hp: tuple[int, int] | None = None


@dataclass(frozen=True)
class DecisionView:
    legal: tuple[LegalAction, ...]
    screen: ScreenFacts
    degrade: Degrade
    rng: random.Random
    battle: Battle | None = None


@dataclass(frozen=True)
class TeamPreviewView:
    battle: Battle | None
    roster: tuple[str | None, ...]
    rng: random.Random
    export: str | None = None


class Player(Protocol):
    name: str

    def params(self) -> dict: ...

    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        """Three roster rows, lead first."""
        ...

    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction: ...

    def notes(self) -> dict:
        """Anything worth recording about the last decision; drained by the caller."""
        ...


def random_team(rng: random.Random, size: int = 6, picks: int = 3) -> list[int]:
    rows = list(range(size))
    rng.shuffle(rows)
    return rows[:picks]
