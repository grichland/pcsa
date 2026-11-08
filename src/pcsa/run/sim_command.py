"""`pcsa sim play`: paired simulator games between two players, with the win rate and its interval."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime

from pcsa.battle.builds import Catalog
from pcsa.common import log, paths
from pcsa.common.config import Config
from pcsa.common.interval import wilson
from pcsa.players.arena import ArenaPlayer, ExportTeams, close, run_match
from pcsa.players.factory import build_player
from pcsa.sim.engine import Engine
from pcsa.sim.server import ShowdownServer
from pcsa.sim.teams import sample_teams


@dataclass(frozen=True)
class PlayOutcome:
    code: int
    lines: list[str]


def paired_exports(teams: list[str], battles: int) -> tuple[list[str], list[str]]:
    """Battle k of pair k//2 gives A team k%2 of the pair and B the other, so each pair is played both ways."""
    a, b = [], []
    for k in range(battles):
        x, y = teams[2 * (k // 2)], teams[2 * (k // 2) + 1]
        a.append(x if k % 2 == 0 else y)
        b.append(y if k % 2 == 0 else x)
    return a, b


def play_command(
    config: Config, a_kind: str, b_kind: str, battles: int, seed: int, *, noisy: bool, port: int | None
) -> PlayOutcome:
    engine = Engine.locate(config.agent.sim.format)
    catalog = Catalog.load()
    rng = random.Random(seed)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{a_kind}-vs-{b_kind}-seed{seed}" + ("-noisy" if noisy else "")
    out_dir = paths.output_dir() / "sim" / stamp
    for suffix in ("", *(f"-{i}" for i in range(2, 100))):  # two runs started in the same second stay apart
        try:
            out_dir = out_dir.with_name(stamp + suffix)
            out_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    log.configure(out_dir, console=True)
    pairs = (battles + 1) // 2
    teams = sample_teams(catalog, engine, rng, 2 * pairs, noisy=noisy)
    a_teams, b_teams = paired_exports(teams, battles)
    inner_a, inner_b = build_player(config, a_kind), build_player(config, b_kind)
    log.event("sim_play_start", a=a_kind, b=b_kind, battles=battles, seed=seed, noisy=noisy, out=str(out_dir))
    with ShowdownServer(engine, port, log_path=out_dir / "server.log") as server:
        a = ArenaPlayer(inner_a, ExportTeams(a_teams), a_kind, server.configuration, engine.format, rng)
        b = ArenaPlayer(inner_b, ExportTeams(b_teams), b_kind, server.configuration, engine.format, rng)
        try:
            result = run_match(a, b, battles)
        finally:
            close(a, b)
    n = result.a_wins + result.b_wins
    low, high = wilson(result.a_wins, n)
    summary = {
        "a": a_kind,
        "b": b_kind,
        "battles": len(result.battles),
        "a_wins": result.a_wins,
        "b_wins": result.b_wins,
        "unfinished": len(result.battles) - n,
        "a_win_rate": result.a_wins / n if n else None,
        "wilson_95": [low, high],
        "avg_turns": sum(r["turns"] for r in result.battles) / max(1, len(result.battles)),
        "errors": result.errors,
        "seed": seed,
        "noisy": noisy,
    }
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "battles": result.battles, "teams": teams}, indent=1), encoding="utf-8"
    )
    (out_dir / "notes.jsonl").write_text("\n".join(json.dumps(n) for n in a.notes + b.notes) + "\n", encoding="utf-8")
    log.event("sim_play_end", **{k: v for k, v in summary.items() if k != "errors"}, error_count=len(result.errors))
    lines = [
        f"{a_kind} vs {b_kind}: {result.a_wins}-{result.b_wins} over {n} decided battles "
        f"({summary['a_win_rate']:.0%} for {a_kind}, 95% interval {low:.0%}-{high:.0%}); results in {out_dir}"
        if n
        else f"no battle finished; see {out_dir}"
    ]
    lines += [f"  error: {error}" for error in result.errors[:10]]
    return PlayOutcome(0 if not result.errors and n == battles else 1, lines)
