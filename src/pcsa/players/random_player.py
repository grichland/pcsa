"""A legal move or switch at random. For harness tests and smoke runs."""

from __future__ import annotations

from pcsa.common.clock import Deadline
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView, random_team


class RandomPlayer:
    name = "random"

    def params(self) -> dict:
        return {}

    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        return random_team(view.rng, len(view.roster) or 6)

    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction:
        if not view.legal:
            raise ValueError("no legal action")
        return view.rng.choice(view.legal)

    def notes(self) -> dict:
        return {}
