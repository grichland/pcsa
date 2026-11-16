"""Team preview by full playouts (`preview: playout` in config/players/neural.yaml).

The robust preview (pcsa.rl_training.preview_leads) picks the candidate lineups; sequential halving over `schedule`
plays full simulated games (pcsa.players.preview_pool) against sampled opposing lineups (preview_sampler). The robust
pick stays unless another lineup has `min_playouts` games and a win rate better by `margin`. Deadline-aware; any
failure keeps the robust pick (`playout_fallback`). Every playout is saved to <run folder>/preview-playouts.jsonl."""

from __future__ import annotations

import json
import logging
import queue
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pcsa.common import log, paths
from pcsa.common.clock import Deadline, SystemClock
from pcsa.common.errors import ConfigError
from pcsa.players import preview_sampler as sampler

if TYPE_CHECKING:
    from pcsa.battle.builds import Catalog
    from pcsa.players.preview_pool import Pool

FIRST_GUESS_S = 1.0  # a playout's length before any has finished (about 0.65 s)
WAIT_S = 0.2
ROBUST_S = 15.0  # the robust scores come first: max_preview_s must leave them this much
PAIR = 2  # a schedule round is [candidates, playouts each]; a round needs two candidates
THEIR_PICK = (0, 1, 2)  # their trio's export is the lineup itself, the lead first


@dataclass
class PlayoutParams:
    workers: int = 4  # child processes, one Showdown server each
    budget_s: float = 30.0  # playout time per preview, after the robust scores (the phone gives 90 s in all)
    reserve_s: float = 3.0  # of the caller's deadline kept back
    schedule: list = field(default_factory=lambda: [[16, 3], [8, 5], [4, 8], [2, 14]])  # [candidates, playouts each]
    min_playouts: int = 8  # a lineup replaces the robust pick only with at least this many games
    margin: float = 0.05  # ... and a win rate at least this much above the robust pick's
    noise: float = 0.05  # Gumbel noise scale on the robust scores when the candidates are chosen
    game_timeout_s: float = 60.0  # one playout

    @classmethod
    def load(cls, data: dict | None, where: str) -> PlayoutParams:
        data = data or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if not isinstance(data, dict) or unknown:
            raise ConfigError(f"{where}: unknown preview_playout settings {sorted(unknown)}")
        p = cls(**data)
        rounds = p.schedule
        ok = (
            isinstance(rounds, list)
            and rounds
            and all(
                isinstance(r, list | tuple) and len(r) == PAIR and all(type(x) is int and x >= 1 for x in r)
                for r in rounds
            )
        )
        if not ok or any(a[0] < b[0] for a, b in zip(rounds, rounds[1:], strict=False)):
            raise ConfigError(f"{where}: preview_playout.schedule must be [[candidates, playouts], ...], shrinking")
        if type(p.workers) is not int or p.workers < 1 or type(p.min_playouts) is not int or p.min_playouts < 1:
            raise ConfigError(f"{where}: preview_playout workers and min_playouts must be whole numbers, 1 or more")
        if min(p.budget_s, p.game_timeout_s) <= 0 or min(p.reserve_s, p.noise, p.margin) < 0:
            raise ConfigError(f"{where}: preview_playout times must be positive, noise and margin 0 or more")
        return p


@dataclass
class Stat:
    wins: int = 0
    n: int = 0

    def rate(self) -> float:
        return self.wins / self.n if self.n else -1.0


@dataclass
class Search:
    """One preview's search: what the pool is asked, what came back."""

    params: PlayoutParams
    picks: list
    scores: np.ndarray  # robust score per pick
    robust: int  # index of the robust pick
    export: str
    six: list
    tag: str
    stats: dict = field(default_factory=dict)  # pick index -> Stat
    rounds: list = field(default_factory=list)  # "16x2:30" = candidates x playouts each : results
    failed: int = 0
    no_build: int = 0
    stop: str = "done"
    records: list = field(default_factory=list)


class PlayoutPreview:
    """Kept on the neural player (`player.playout`): the workers, the lead prior and the catalog."""

    def __init__(self, params: PlayoutParams, spec: dict, **parts) -> None:
        """`parts` (tests): pool, clock, rng, prior, catalog in place of the real ones."""
        self.p, self.clock = params, parts.get("clock") or SystemClock()
        self.rng = parts.get("rng") or np.random.default_rng()
        self.prior = parts.get("prior") or sampler.LeadPrior.load()
        catalog = parts.get("catalog")
        if catalog is None:
            from pcsa.battle.builds import Catalog

            catalog = Catalog.load(seen=True)
        self.catalog: Catalog = catalog
        self.count = 0
        pool = parts.get("pool")
        if pool is None:
            from pcsa.players.preview_pool import Pool

            pool = Pool(params.workers, spec, paths.output_dir() / "preview-workers")
        self.pool: Pool = pool

    # the search ------------------------------------------------------------------------------------------------
    def candidates(self, scores: np.ndarray, rows: list[int], robust: int) -> list[int]:
        """The first round's lineups: the best by robust score + Gumbel noise, the robust pick always in."""
        n = self.p.schedule[0][0]
        noisy = scores[rows] + self.p.noise * self.rng.gumbel(size=len(rows))
        order = [rows[i] for i in np.argsort(-noisy, kind="stable")]
        return [robust] + [i for i in order if i != robust][: n - 1]

    def _ranked(self, s: Search, cands: list[int]) -> list[int]:
        return sorted(cands, key=lambda i: (-s.stats[i].rate(), -float(s.scores[i])))

    def _shrink(self, s: Search, keep: int, per: int, rate: float, end: Deadline) -> tuple[int, int]:
        """The round cut to what fits the time left: fewer playouts each first, then fewer candidates."""
        fits = int(rate * max(0.0, end.remaining()))
        per = min(per, max(1, fits // max(1, keep)))
        return (min(keep, fits), per) if fits < keep else (keep, per)

    def _tasks(self, s: Search, cands: list[int], per: int, number: int) -> list[dict]:
        tasks = []
        for k in range(per):
            lineup = sampler.their_lineup(s.six, self.prior, self.catalog, self.rng)
            if lineup is None:
                raise ValueError("fewer than three of their six have a known build")
            s.no_build = lineup.no_build
            seed = int(self.rng.integers(1 << 30))
            for i in cands:
                tasks.append({
                    "id": f"{s.tag}:{number}:{k}:{i}", "ours": s.export, "our_pick": list(s.picks[i]),
                    "theirs": lineup.export, "their_pick": list(THEIR_PICK), "seed": seed, "pick": i,
                    "their": list(lineup.species), "round": number,
                })  # fmt: skip
        return tasks

    def _collect(self, s: Search, tasks: list[dict], end: Deadline) -> int:
        """Play `tasks` on the idle workers until all are back or time is up; returns how many came back."""
        pending, flying, done = deque(tasks), {}, 0
        while pending or flying:
            if end.expired():
                s.stop = "deadline"
                break
            if not flying and not self.pool.ready():
                s.stop = "no workers"
                break
            for worker in self.pool.idle():
                if not pending:
                    break
                task = pending.popleft()
                if worker.send(task):
                    flying[task["id"]] = task
                else:
                    pending.appendleft(task)
            try:
                result = self.pool.results.get(timeout=max(0.01, min(WAIT_S, end.remaining())))
            except queue.Empty:
                continue
            task = flying.pop(result.get("id"), None)
            s.records.append({**(task or {}), **result, "late": task is None})
            if task is None:
                continue  # a game abandoned by an earlier preview
            done += 1
            if "error" in result:
                s.failed += 1
            else:
                stat = s.stats[task["pick"]]
                stat.n += 1
                stat.wins += bool(result["won"])
        return done

    def search(self, s: Search, rows: list[int], end: Deadline) -> int:
        """Sequential halving; returns the index of the pick to play."""
        cands = self.candidates(s.scores, rows, s.robust)
        s.stats = {i: Stat() for i in cands}
        started, total = self.clock.now(), 0
        for number, (keep, per) in enumerate(self.p.schedule):
            elapsed = self.clock.now() - started
            rate = total / elapsed if total and elapsed > 0 else max(1, self.pool.ready()) / FIRST_GUESS_S
            keep, per = self._shrink(s, min(keep, len(cands)), per, rate, end)
            if keep < PAIR:
                s.stop = "deadline"
                break
            ranked = self._ranked(s, cands)
            cands = ranked[:keep] if s.robust in ranked[:keep] else [*ranked[: keep - 1], s.robust]
            got = self._collect(s, self._tasks(s, cands, per, number), end)
            total += got
            s.rounds.append(f"{keep}x{per}:{got}")
            if s.stop != "done":
                break
        best = self._ranked(s, cands)[0]
        enough = s.stats[best].n >= self.p.min_playouts
        ahead = s.stats[best].rate() - max(0.0, s.stats[s.robust].rate())
        return best if enough and best != s.robust and ahead >= self.p.margin else s.robust

    # one preview -----------------------------------------------------------------------------------------------
    def refine(self, policy, view, deadline: Deadline, team: list[int], notes: dict) -> tuple[list[int], dict]:
        """(team, notes) after the playouts; the robust `team` and a `playout_fallback` note when they cannot run."""
        started = self.clock.now()
        try:
            s, rows = self._begin(policy, view, team)
            end = deadline.minus(self.p.reserve_s).earliest(Deadline.after(self.clock, self.p.budget_s))
            index = self.search(s, rows, end)
        except Exception as error:  # the robust pick stands
            why = f"{type(error).__name__}: {error}"[:200]
            log.warning("preview_playout_fallback", reason=why)
            return team, {**notes, "preview": "playout", "playout_fallback": why}
        self._save(s)
        names = view.roster
        label = lambda i: "/".join(str(names[j]) for j in s.picks[i])  # noqa: E731
        played = sorted((i for i in s.stats if s.stats[i].n), key=lambda i: -s.stats[i].n)
        finalists = self._ranked(s, played[:4])  # the lineups with the most games, the best win rate first
        out = {
            **notes,
            "preview": "playout",
            "playout_pick": label(index),
            "playout_robust": label(s.robust),
            "playout_changed": index != s.robust,
            "playout_finalists": [f"{label(i)} {s.stats[i].wins}/{s.stats[i].n}" for i in finalists],
            "playout_rounds": s.rounds,
            "playout_done": sum(st.n for st in s.stats.values()),
            "playout_failed": s.failed,
            "playout_stop": s.stop,
            "playout_workers": self.pool.ready(),
            "playout_no_build": s.no_build,
            "playout_s": round(self.clock.now() - started, 1),
        }
        return [int(i) for i in s.picks[index]], out

    def _begin(self, policy, view, team: list[int]) -> tuple[Search, list[int]]:
        from pcsa.rl_training import preview
        from pcsa.rl_training.preview_leads import their_species

        battle = view.battle
        scores, rows = policy.preview_scores
        picks = preview.all_picks(len(battle.team) or 6, battle.max_team_size or 3)
        if not getattr(view, "export", None):
            raise ValueError("no team export at preview")
        if not self.pool.ready():
            raise RuntimeError(f"no playout worker is ready {self.pool.errors()[:1]}")
        self.count += 1
        robust = picks.index(tuple(team))
        tag = f"{getattr(battle, 'battle_tag', 'preview')}#{self.count}"
        return Search(self.p, picks, np.asarray(scores, float), robust, view.export, their_species(battle), tag), rows

    def _save(self, s: Search) -> None:
        """Every playout of the preview, one JSON line each (the team exports once per preview)."""
        try:
            path = run_folder() / "preview-playouts.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                head = {"preview": s.tag, "six": s.six, "ours": s.export, "params": asdict(s.params)}
                fh.write(json.dumps(head) + "\n")
                for r in s.records:
                    fh.write(json.dumps({k: v for k, v in r.items() if k != "ours"}) + "\n")
        except OSError as error:
            log.warning("preview_playout_save_failed", reason=str(error))


def run_folder() -> Path:
    """The run folder the events go to, else output/preview-playouts."""
    for handler in logging.getLogger(log.LOGGER).handlers:
        if isinstance(handler, log.JsonlHandler):
            return handler.path.parent
    return paths.output_dir() / "preview-playouts"


def start(player) -> PlayoutPreview:
    """The player's playout preview: settings checked, workers started (they load in the background)."""
    params = PlayoutParams.load(player.p.preview_playout, "neural player")
    spec = {"checkpoint": str(player.checkpoint), "params": asdict(player.p), "timeout_s": params.game_timeout_s}
    return PlayoutPreview(params, spec)
