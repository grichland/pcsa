"""pcsa players on a local Showdown server: a poke-env Player adapter and a paired match runner.

The adapter gives the wrapped player the same view the phone loop gives it: a Battle (ours as p1, mirrored
when the server seats us as p2) and the legal orders. Teams come from a list of exports consumed one battle
at a time. Each finished battle leaves a seat record in `finished` (pcsa.run.battle_log merges the two seats).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
from dataclasses import dataclass, field

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player as PokeEnvPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.teambuilder import Teambuilder

from pcsa.battle.state import replay
from pcsa.common.clock import Clock, Deadline, SystemClock
from pcsa.players import mirror
from pcsa.players.base import DecisionView, Degrade, Player, ScreenFacts, TeamPreviewView

_names = itertools.count(1)
ERROR_LEVEL = 25
NAME_LIMIT = 12


class ExportTeams(Teambuilder):
    """Team exports handed out in order, one per battle; `history` remembers them."""

    def __init__(self, exports: list[str]) -> None:
        self.exports = exports
        self.history: list[str] = []

    def yield_team(self) -> str:
        text = self.exports[len(self.history) % len(self.exports)]
        self.history.append(text)
        return self.join_team(self.parse_showdown_team(text))


class ArenaPlayer(PokeEnvPlayer):
    """A poke-env Player whose decisions come from a pcsa Player."""

    def __init__(
        self,
        inner: Player,
        teams: ExportTeams,
        name: str,
        server: ServerConfiguration,
        battle_format: str,
        rng: random.Random,
        clock: Clock | None = None,
        decision_s: float = 60.0,
    ) -> None:
        username = f"{name[:NAME_LIMIT]}{next(_names)}"
        super().__init__(
            account_configuration=AccountConfiguration(username, None),
            server_configuration=server,
            battle_format=battle_format,
            team=teams,
            start_listening=True,
            max_concurrent_battles=1,
        )
        self.inner = inner
        self.kind = name
        self.teams = teams
        self.rng = rng
        self.clock = clock or SystemClock()
        self.decision_s = decision_s
        self.picks: dict[str, list[str]] = {}
        self.notes: list[dict] = []
        self.team_of: dict[str, str] = {}  # battle tag -> the export this seat played
        self.finished: dict[str, dict] = {}  # battle tag -> seat record, see _battle_finished_callback

    def teampreview(self, battle) -> str:
        if self.teams.history:
            self.team_of[battle.battle_tag] = self.teams.history[-1]
        chosen = (battle.last_request or {}).get("maxChosenTeamSize")
        if chosen:
            battle._max_team_size = chosen
        roster = tuple(key.split(": ", 1)[1] for key in battle.team)
        preview = TeamPreviewView(battle, roster, self.rng)  # type: ignore[arg-type]
        picks = self.inner.choose_team(preview, Deadline.after(self.clock, 30.0))
        self.picks[battle.battle_tag] = [roster[i] for i in picks]
        notes = self.inner.notes()
        if notes:
            self.notes.append({"battle": battle.battle_tag, "turn": 0, "kind": "team", **notes})
        return "/team " + "".join(str(i + 1) for i in picks)

    def choose_move(self, battle) -> BattleOrder:
        protocol = ["|".join(p) for p in battle._replay_data]
        lines, request = mirror.as_p1(battle.player_role, protocol, battle.last_request)
        seen = battle if battle.player_role == "p1" else replay(lines, request, battle.battle_tag)
        legal = mirror.legal_actions(seen)
        if not legal:
            return self.choose_random_move(battle)
        view = DecisionView(legal, ScreenFacts(), Degrade.FULL, self.rng, seen)  # type: ignore[arg-type]
        chosen = self.inner.choose_action(view, Deadline.after(self.clock, self.decision_s))
        notes = self.inner.notes()
        if notes:
            self.notes.append({"battle": battle.battle_tag, "turn": battle.turn, **notes})
        assert chosen.order is not None
        return chosen.order

    def _battle_finished_callback(self, battle) -> None:
        """Keeps this seat's view of the finished battle; the replay data is referenced, not copied."""
        tag = battle.battle_tag
        team = self.team_of.get(tag) or (self.teams.history[-1] if self.teams.history else None)
        self.finished[tag] = {
            "role": battle.player_role,
            "team": team,
            "picks": self.picks.get(tag, []),
            "won": battle.won,
            "turns": battle.turn,
            "replay": battle._replay_data,
        }


@dataclass
class MatchResult:
    battles: list[dict]
    errors: list[str] = field(default_factory=list)

    @property
    def a_wins(self) -> int:
        return sum(b["winner"] == "a" for b in self.battles)

    @property
    def b_wins(self) -> int:
        return sum(b["winner"] == "b" for b in self.battles)


class _Collect(logging.Handler):
    def __init__(self, sink: list[str], who: str) -> None:
        super().__init__(level=ERROR_LEVEL)
        self.sink, self.who = sink, who

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(f"{self.who}: {record.getMessage()}")


async def _run(a: ArenaPlayer, b: ArenaPlayer, battles: int) -> MatchResult:
    errors: list[str] = []
    handlers = [(p.logger, _Collect(errors, who)) for p, who in ((a, "A"), (b, "B"))]
    for logger, h in handlers:
        logger.addHandler(h)
    try:
        for k in range(battles):  # the challenger sits p1: A in even battles, B in odd ones
            first, second = (a, b) if k % 2 == 0 else (b, a)
            await first.battle_against(second, n_battles=1)
    except Exception as error:  # partial results still count
        errors.append(f"battle_against: {type(error).__name__}: {error}")
    for logger, h in handlers:
        logger.removeHandler(h)
    rows = []
    for tag, battle in a.battles.items():
        rows.append(
            {
                "tag": tag,
                "winner": "a" if battle.won else "b" if battle.lost else None,
                "turns": battle.turn,
                "a_role": battle.player_role,
                "a_picks": a.picks.get(tag, []),
            }
        )
    return MatchResult(rows, errors)


def run_match(a: ArenaPlayer, b: ArenaPlayer, battles: int) -> MatchResult:
    """`battles` battles one at a time, the challenger (p1) alternating from A, so each pair swaps the seats."""
    return asyncio.run(_run(a, b, battles))


async def _close(*players: ArenaPlayer) -> None:
    for player in players:
        await player.ps_client.stop_listening()


def close(*players: ArenaPlayer) -> None:
    asyncio.run(_close(*players))
