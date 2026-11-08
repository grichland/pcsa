"""The mixed player: a pool of players with weights; one is drawn per game and plays the whole game.

The draw happens at team preview (the start of every game) and again whenever the turn counter goes
backwards, which is how a game boundary looks when no preview was shown. With `per_decision` on, a
member is drawn for every decision instead (an experiment setting: the result is an opponent nobody
plays like). Weights need not sum to one. `notes()` names the member that decided.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import yaml

from pcsa.common.clock import Deadline
from pcsa.common.errors import ConfigError
from pcsa.players.base import DecisionView, LegalAction, Player, TeamPreviewView


class MixedPolicyPlayer:
    name = "mixed"

    def __init__(
        self, members: list[tuple[Player, float]], rng: random.Random, *, per_decision: bool = False
    ) -> None:
        if not members:
            raise ConfigError("a mixed player needs at least one member")
        if any(weight < 0 for _, weight in members) or sum(weight for _, weight in members) <= 0:
            raise ConfigError("mixed player weights must be non-negative and not all zero")
        self.members = members
        self.rng = rng
        self.per_decision = per_decision
        self.current: Player = members[0][0]
        self._turn = -1

    @classmethod
    def build(cls, path: Path, build_member: Callable[[str], Player], seed: int | None = None) -> MixedPolicyPlayer:
        """From a yaml file: `members: [{kind, weight}, ...]`, `per_decision`, `seed`."""
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(data) - {"members", "per_decision", "seed"}
        if unknown:
            raise ConfigError(f"unknown mixed player settings in {path}: {sorted(unknown)}")
        members = []
        for entry in data.get("members") or []:
            kind, weight = str(entry.get("kind", "")), float(entry.get("weight", 1.0))
            if kind == cls.name:
                raise ConfigError(f"{path}: a mixed player cannot contain another mixed player")
            members.append((build_member(kind), weight))
        rng = random.Random(seed if seed is not None else data.get("seed"))
        return cls(members, rng, per_decision=bool(data.get("per_decision", False)))

    def params(self) -> dict:
        return {
            "per_decision": self.per_decision,
            "members": [{"kind": player.name, "weight": weight} for player, weight in self.members],
        }

    def notes(self) -> dict:
        return {"policy": self.current.name, **self.current.notes()}

    def draw(self) -> Player:
        """A member by weight; it becomes the current player."""
        players = [player for player, _ in self.members]
        weights = [weight for _, weight in self.members]
        self.current = self.rng.choices(players, weights=weights)[0]
        return self.current

    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        self._turn = -1
        return self.draw().choose_team(view, deadline)

    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction:
        turn = int(getattr(view.battle, "turn", 0) or 0) if view.battle is not None else self._turn
        if self.per_decision or turn < self._turn:
            self.draw()
        self._turn = turn
        return self.current.choose_action(view, deadline)
