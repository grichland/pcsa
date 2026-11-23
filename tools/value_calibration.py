"""Fit cg.value_calibration {a, b} for a checkpoint on recorded ladder battles.

    uv run python tools/value_calibration.py [--runs DIR] [--checkpoint assets/models/<file>.pt]
        [--exclude FILE] [--out DIR] [--set section.key=value ...]

Every battle with saved CG messages and a result is replayed through pcsa.cg.serve with the checkpoint and the
neural player of config/agent.yaml (its rule masks and choice rule; search and calibration off), and each decision's raw value is
paired with the battle's result: a win, a forfeit and an unknown result count as wins, a loss as a loss. The fit is
Platt scaling, P(win) = sigmoid(a * v + b), so the value reported live is 2 * P(win) - 1. Battles listed in
--exclude (pcsa-benchmark's clean battles, `battle_id run/folder` per line) are left out. Held-out scores come from
five folds by battle; the a and b to use are fitted on every battle. Written to --out: <checkpoint>.json (the
report) and <checkpoint>.samples.jsonl (every sample).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]

import replay_run  # noqa: E402

from pcsa.cg import startup  # noqa: E402
from pcsa.common import paths  # noqa: E402
from pcsa.common.config import load_config  # noqa: E402

WON = {"win": 1, "forfeit": 1, "unknown": 1, "loss": 0}  # forfeit and unknown count as wins
FOLDS, BINS = 5, 10
BENCHMARK = REPO.parent / "pcsa-benchmark" / "v1" / "clean_battles.txt"


def result(battle: Path) -> int | None:
    """1 for a win (forfeit and unknown included), 0 for a loss, None without a battle_end."""
    for line in (battle / "cg-messages.jsonl").read_text(encoding="utf-8").splitlines():
        message = json.loads(line).get("message") if line.strip() else None
        if isinstance(message, dict) and message.get("type") == "battle_end":
            return WON.get(str(message.get("result")))
    return None


def excluded(path: Path | None) -> set[str]:
    """The `run/battle-N` folders listed in the exclude file."""
    if path is None or not path.is_file():
        return set()
    rows = [x.split() for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    return {r[1] for r in rows if len(r) > 1}


def battles(runs: Path, skip: set[str]) -> list[tuple[Path, Path, int]]:
    """(run, battle, won) for every replayable battle with a result, in time order."""
    out = []
    for run in sorted(p for p in runs.iterdir() if p.is_dir() and not p.name.endswith("-replay")):
        for battle in sorted(run.glob("battle-*"), key=lambda p: int(p.name.split("-")[1])):
            if f"{run.name}/{battle.name}" in skip or not (battle / "cg-messages.jsonl").is_file():
                continue
            won = result(battle)
            if won is not None:
                out.append((run, battle, won))
    return out


def player_args(cfg, checkpoint: Path) -> dict:
    """The config's neural player with `checkpoint`, no log folder, search and calibration off."""
    args = dict(startup.resolved(cfg).player_args or {})
    args.pop("log_dir", None)
    policy = {**args.get("policy_args", {}), "search": None, "value_calibration": None}
    return {**args, "checkpoint": str(checkpoint), "policy_args": policy}


def samples(found: list[tuple[Path, Path, int]], args: dict) -> list[dict]:
    """One row per replayed decision that carries a value."""
    rows = []
    for i, (run, battle, won) in enumerate(found):
        sent, _ = replay_run.battle_messages(battle, run)
        replies, _, _ = replay_run.replay_battle(sent, args)
        for key, reply in replies.items():
            net = (reply.get("notes") or {}).get("net") or {}
            if isinstance(net.get("value"), int | float):
                rows.append({"battle": i, "run": run.name, "folder": battle.name, "request": key,
                             "kind": net.get("kind"), "v": float(net["value"]), "won": won})  # fmt: skip
        sys.stderr.write(f"\r{i + 1}/{len(found)} battles, {len(rows)} decisions")
    sys.stderr.write("\n")
    return rows


def platt(v: np.ndarray, y: np.ndarray, steps: int = 50) -> tuple[float, float]:
    """a, b of P(y = 1) = sigmoid(a * v + b) by Newton's method on the log loss (a small ridge keeps it finite)."""
    w = np.zeros(2)
    x = np.column_stack([v, np.ones_like(v)])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        grad = x.T @ (p - y) + 1e-4 * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + 1e-4 * np.eye(2)
        w -= np.linalg.solve(hess, grad)
    return float(w[0]), float(w[1])


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    """Brier score, expected calibration error over BINS equal bins, AUC and the mean prediction."""
    bins = np.minimum((p * BINS).astype(int), BINS - 1)
    ece = sum(abs(p[bins == k].mean() - y[bins == k].mean()) * (bins == k).mean() for k in range(BINS) if (bins == k).any())
    pos, neg = p[y == 1], p[y == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else math.nan
    return {"brier": float(((p - y) ** 2).mean()), "ece": float(ece), "auc": auc, "mean_p": float(p.mean()), "n": int(len(y))}


def report(rows: list[dict]) -> dict:
    v = np.array([r["v"] for r in rows])
    y = np.array([r["won"] for r in rows], float)
    fold = np.array([r["battle"] % FOLDS for r in rows])
    held = np.zeros_like(v)
    for k in range(FOLDS):
        a, b = platt(v[fold != k], y[fold != k])
        held[fold == k] = 1.0 / (1.0 + np.exp(-(a * v[fold == k] + b)))
    a, b = platt(v, y)
    return {"a": round(a, 5), "b": round(b, 5), "battles": len({r["battle"] for r in rows}), "win_rate": float(y.mean()),
            "raw": scores(np.clip((v + 1) / 2, 0, 1), y), "held_out": scores(held, y)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--runs", type=Path, default=paths.runs_dir())
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: the config's checkpoint")
    ap.add_argument("--exclude", type=Path, default=BENCHMARK if BENCHMARK.is_file() else None)
    ap.add_argument("--out", type=Path, default=paths.output_dir() / "value-calibration")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value")
    a = ap.parse_args(argv)
    cfg = load_config(paths.config_dir() / "agent.yaml", overrides=a.set).agent.cg
    checkpoint = (a.checkpoint or Path(str(startup.checkpoint(startup.resolved(cfg))))).resolve()
    args = player_args(cfg, checkpoint)
    found = battles(a.runs.resolve(), excluded(a.exclude))
    rows = samples(found, args)
    out = {"checkpoint": checkpoint.name, "runs": str(a.runs), "excluded": str(a.exclude), "player_args": args,
           **report(rows)}  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{checkpoint.stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (a.out / f"{checkpoint.stem}.samples.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
