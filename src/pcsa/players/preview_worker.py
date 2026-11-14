"""One playout worker of the playout team preview: a child process with its own Showdown server and two copies
of the live network, playing full games with forced lineups.

    python -m pcsa.players.preview_worker
      first line on stdin, the spec: {"checkpoint": path, "params": NeuralParams as a dict, "log": server log
      path, "timeout_s": per game}

Tasks then arrive as JSON lines on stdin ({"id", "ours", "our_pick", "theirs", "their_pick", "seed"}: team exports and
0-based lineups, the lead first), results leave as JSON lines on stdout ({"id", "won", "turns", "seconds"} or
{"id", "error"}); the first line out is {"ready": true}. Seat A (ours) always challenges. The process ends when
stdin closes (the parent is gone or asked for it) and stops its server. Nothing else writes to stdout.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import sys
from pathlib import Path

from pcsa.common.clock import Deadline, SystemClock, precise_now
from pcsa.players.base import DecisionView, LegalAction, TeamPreviewView

POLL_S = 0.05
DEFAULT_TIMEOUT_S = 60.0


class BattleFailed(RuntimeError):
    pass


class ForcedPlayer:
    """The neural player with the lineup given: `forced` is returned at team preview (after the player's own
    new-battle resets), every action is the player's."""

    name = "playout"

    def __init__(self, inner) -> None:
        self.inner = inner
        self.forced: list[int] = [0, 1, 2]

    def params(self) -> dict:
        return {}

    def notes(self) -> dict:
        return self.inner.notes()

    def choose_team(self, view: TeamPreviewView, deadline: Deadline) -> list[int]:
        n = self.inner
        n.fallback.choose_team(view, deadline)  # resets the fallback's per-game state
        n.fallback.notes()
        if n._threat is not None:
            n._threat[1].reset()
        if n.policy is not None and hasattr(n.policy, "reset"):
            n.policy.reset()
        return list(self.forced)

    def choose_action(self, view: DecisionView, deadline: Deadline) -> LegalAction:
        return self.inner.choose_action(view, deadline)


class _Popups(logging.Handler):
    """poke-env popups (a rejected team or challenge): the battle will never start."""

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "popup" in message.lower():
            self.sink.append(message[:300])


async def drive(first, second, timeout: float) -> None:
    """`first` challenges `second` for one battle; fails fast on a popup, gives up after `timeout`."""
    sink: list[str] = []
    watched = [(p.logger, _Popups(sink)) for p in (first, second)]
    for logger, handler in watched:
        logger.addHandler(handler)
    task = asyncio.ensure_future(first.battle_against(second, n_battles=1))
    deadline = Deadline.after(SystemClock(), timeout)
    try:
        while not task.done():
            if sink:
                raise BattleFailed(sink[0])
            if deadline.expired():
                raise BattleFailed(f"no result within {timeout:.0f}s")
            await asyncio.wait({task}, timeout=POLL_S)
        task.result()
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        for logger, handler in watched:
            logger.removeHandler(handler)


def playout_params(data: dict):
    """The live settings as the playouts use them: sampled actions, the simulator's critical-hit lines, and no
    playouts of their own."""
    from pcsa.players.neural import NeuralParams

    return NeuralParams(**{**data, "sampling": "sample", "preview": "greedy", "belief_crit_lines": True})


class Worker:
    def __init__(self, spec: dict) -> None:
        from pcsa.common import paths
        from pcsa.common.config import load_config
        from pcsa.players.factory import build_player
        from pcsa.players.neural import NeuralPlayer
        from pcsa.sim.engine import Engine
        from pcsa.sim.server import ShowdownServer

        self.timeout = float(spec.get("timeout_s") or DEFAULT_TIMEOUT_S)
        self.config = load_config(paths.config_dir() / "agent.yaml")
        self.engine = Engine.locate(self.config.agent.sim.format)
        params = playout_params(spec["params"])
        checkpoint = Path(spec["checkpoint"])
        self.players = [
            ForcedPlayer(NeuralPlayer(params, build_player(self.config, params.fallback), checkpoint))
            for _ in range(2)
        ]
        errors = [p.inner.error for p in self.players if p.inner.error]
        if errors:
            raise RuntimeError(errors[0])
        log_path = Path(spec["log"]) if spec.get("log") else None
        self.server = ShowdownServer(self.engine, None, log_path=log_path).start()
        self.loop = asyncio.new_event_loop()  # one loop for every battle (a loop per battle leaks sockets on Windows)
        self.arenas: list = []

    def _arenas(self) -> list:
        if not self.arenas:
            from pcsa.players.arena import ArenaPlayer, ExportTeams

            asyncio.set_event_loop(self.loop)
            self.arenas = [
                ArenaPlayer(
                    p, ExportTeams([""]), name, self.server.configuration, self.engine.format, random.Random()
                )
                for p, name in zip(self.players, ("pla", "plb"), strict=True)
            ]
        return self.arenas

    def _drop_arenas(self) -> None:
        """After a failed battle the two clients are unusable: close them, the next task makes new ones."""
        arenas, self.arenas = self.arenas, []
        for arena in arenas:
            with contextlib.suppress(Exception):
                self.loop.run_until_complete(arena.ps_client.stop_listening())

    def play(self, task: dict) -> dict:
        started = precise_now()
        try:
            a, b = self._arenas()
            for arena, player, export, pick in (
                (a, self.players[0], task["ours"], task["our_pick"]),
                (b, self.players[1], task["theirs"], task["their_pick"]),
            ):
                arena.teams.exports, arena.teams.history = [export], []
                arena.rng.seed(int(task.get("seed", 0)))
                player.forced = [int(i) for i in pick]
            self.loop.run_until_complete(drive(a, b, self.timeout))
            battles = list(a.battles.values())
            if len(battles) != 1 or not battles[0].finished:
                raise BattleFailed(f"expected one finished battle, have {len(battles)}")
            won, turns = battles[0].won, int(battles[0].turn)
            for arena in (a, b):
                arena.reset_battles()
                arena.notes.clear()
                arena.picks.clear()
                arena.finished.clear()
                arena.team_of.clear()
            if won is None:
                raise BattleFailed("the battle ended without a winner")
            out = {"id": task["id"], "won": bool(won), "turns": turns}
        except Exception as error:  # one playout never stops the worker
            self._drop_arenas()
            out = {"id": task.get("id"), "error": f"{type(error).__name__}: {error}"[:300]}
        return {**out, "seconds": round(precise_now() - started, 2)}

    def close(self) -> None:
        self._drop_arenas()
        self.server.stop()
        with contextlib.suppress(Exception):
            self.loop.close()


def main() -> int:
    out = sys.stdout
    sys.stdout = sys.stderr  # nothing but results goes to the parent

    def emit(record: dict) -> None:
        out.write(json.dumps(record) + "\n")
        out.flush()

    try:
        worker = Worker(json.loads(sys.stdin.readline()))
    except Exception as error:
        emit({"ready": False, "error": f"{type(error).__name__}: {error}"[:300]})
        return 1
    try:
        emit({"ready": True, "pid": os.getpid()})
        for line in sys.stdin:
            task = json.loads(line)
            if task.get("quit"):
                break
            emit(worker.play(task))
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
