"""`pcsa sim play`: paired simulator games between two players, with the win rate and its interval.

Each pair is two battles with the seats (p1/p2) swapped. Ordinary players also swap teams within a pair; a
`style_*` player is always dealt a six that fits its style (config/players/styles.yaml), so teams then stay
with their players. Output: results.json, notes.jsonl, battles.jsonl.gz (pcsa.run.battle_log), events.jsonl and
server.log in output/sim/<stamp>/.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from functools import partial

from pcsa.battle.builds import Catalog
from pcsa.common import log, paths
from pcsa.common.config import Config
from pcsa.common.errors import ConfigError
from pcsa.common.interval import wilson
from pcsa.players import styles
from pcsa.players.arena import ArenaPlayer, ExportTeams, close, run_match
from pcsa.players.factory import build_player
from pcsa.run import battle_log
from pcsa.sim.engine import Engine
from pcsa.sim.server import ShowdownServer
from pcsa.sim.teams import sample_styled_team, sample_teams, validated_teams

STYLE_PREFIX = "style_"


@dataclass(frozen=True)
class PlayOutcome:
    code: int
    lines: list[str]


def style_of(kind: str) -> str | None:
    """The team style a `style_*` kind plays (its teams must fit it), else None."""
    return kind[len(STYLE_PREFIX) :] if kind.startswith(STYLE_PREFIX) else None


def paired_exports(
    a_pool: list[str], b_pool: list[str], battles: int, *, swap_teams: bool
) -> tuple[list[str], list[str]]:
    """Both battles of pair k//2 give A a_pool[k//2] and B b_pool[k//2] (the arena swaps the seats between them);
    with `swap_teams` the second battle also swaps the teams, so each team pair is played both ways."""
    a, b = [], []
    for k in range(battles):
        x, y = a_pool[k // 2], b_pool[k // 2]
        swapped = swap_teams and k % 2 == 1
        a.append(y if swapped else x)
        b.append(x if swapped else y)
    return a, b


def check_style(style: str | None) -> None:
    names = styles.load().names()
    if style is not None and style not in names:
        raise ConfigError(f"unknown team style {style!r}; known: {', '.join(names)}")


def team_styles(kinds: tuple[str, str], overrides: tuple[str | None, str | None]) -> tuple[str | None, str | None]:
    """Each side's team style: an explicit override, else the implicit style of a `style_*` kind, else None."""
    return (overrides[0] or style_of(kinds[0]), overrides[1] or style_of(kinds[1]))


def draw_pools(
    catalog: Catalog,
    engine: Engine,
    rng: random.Random,
    sides: tuple[str | None, str | None],
    pairs: int,
    noisy: bool,
) -> tuple[list[str], list[str], bool]:
    """A's and B's team per pair, and whether pairs swap them. With no team style both sides share one pool and
    swap teams within a pair; a styled side keeps its own fitting team, so then teams follow the players."""
    if not any(sides):
        teams = sample_teams(catalog, engine, rng, 2 * pairs, noisy=noisy)
        return teams[0::2], teams[1::2], True
    rules = styles.load()
    pools = []
    for style in sides:
        check_style(style)  # an unknown style fails before any drawing
        if style is None:
            pools.append(sample_teams(catalog, engine, rng, pairs, noisy=noisy))
            continue
        draw = partial(sample_styled_team, catalog, rng, style, rules, noisy=noisy)
        pools.append(validated_teams(engine, pairs, draw))
    return pools[0], pools[1], False


def play_command(
    config: Config,
    a_kind: str,
    b_kind: str,
    battles: int,
    seed: int,
    *,
    noisy: bool,
    port: int | None,
    a_team_style: str | None = None,
    b_team_style: str | None = None,
) -> PlayOutcome:
    sides = team_styles((a_kind, b_kind), (a_team_style, b_team_style))
    check_style(a_team_style)
    check_style(b_team_style)
    engine = Engine.locate(config.agent.sim.format)
    catalog = Catalog.load()
    rng = random.Random(seed)
    tag_a = f"{a_kind}@{a_team_style}" if a_team_style else a_kind
    tag_b = f"{b_kind}@{b_team_style}" if b_team_style else b_kind
    out_dir = _make_out_dir(
        f"{datetime.now():%Y%m%d-%H%M%S}-{tag_a}-vs-{tag_b}-seed{seed}" + ("-noisy" if noisy else "")
    )
    log.configure(out_dir, console=True)
    pairs = (battles + 1) // 2
    a_pool, b_pool, swap_teams = draw_pools(catalog, engine, rng, sides, pairs, noisy)
    a_teams, b_teams = paired_exports(a_pool, b_pool, battles, swap_teams=swap_teams)
    teams = [t for pair in zip(a_pool, b_pool, strict=True) for t in pair]  # pair p: teams[2p] A's, teams[2p+1] B's
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
    summary = _summary(result, (a_kind, b_kind), seed, noisy, sides)
    _write_results(out_dir, summary, result.battles, teams, (a, b))
    lines = [
        f"{a_kind} vs {b_kind}: {result.a_wins}-{result.b_wins} over {n} decided battles "
        f"({summary['a_win_rate']:.0%} for {a_kind}, 95% interval {low:.0%}-{high:.0%}); results in {out_dir}"
        if n
        else f"no battle finished; see {out_dir}"
    ]
    lines += [f"  error: {error}" for error in result.errors[:10]]
    return PlayOutcome(0 if not result.errors and n == battles else 1, lines)


def _summary(result, kinds: tuple[str, str], seed: int, noisy: bool, sides) -> dict:
    n = result.a_wins + result.b_wins
    return {
        "a": kinds[0],
        "b": kinds[1],
        "battles": len(result.battles),
        "a_wins": result.a_wins,
        "b_wins": result.b_wins,
        "unfinished": len(result.battles) - n,
        "a_win_rate": result.a_wins / n if n else None,
        "wilson_95": list(wilson(result.a_wins, n)),
        "avg_turns": sum(r["turns"] for r in result.battles) / max(1, len(result.battles)),
        "errors": result.errors,
        "seed": seed,
        "noisy": noisy,
        "team_styles": list(sides),
    }


def _make_out_dir(stamp: str):
    """A fresh results folder under output/sim; two runs started in the same second stay apart."""
    base = paths.output_dir() / "sim" / stamp
    for suffix in ("", *(f"-{i}" for i in range(2, 100))):
        out_dir = base.with_name(stamp + suffix)
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        except FileExistsError:
            continue
    raise ConfigError(f"could not create a results folder for {stamp}")


def _write_results(
    out_dir, summary: dict, battles: list, teams: list, players: tuple[ArenaPlayer, ArenaPlayer]
) -> None:
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "battles": battles, "teams": teams}, indent=1), encoding="utf-8"
    )
    notes = [n for player in players for n in player.notes]
    (out_dir / "notes.jsonl").write_text("\n".join(json.dumps(n) for n in notes) + "\n", encoding="utf-8")
    a, b = players
    logged = battle_log.records(a.battles, {"a": (a.kind, a.finished), "b": (b.kind, b.finished)})
    size = battle_log.write(out_dir / battle_log.FILE, logged)
    log.event(
        "sim_play_end", **{k: v for k, v in summary.items() if k != "errors"}, error_count=len(summary["errors"]),
        battle_log_records=len(logged), battle_log_bytes=size,
    )
