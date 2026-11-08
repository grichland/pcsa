"""Evaluate one player kind against a reference opponent (top and noisy builds) and record one row in results.md.

Side A's notes are split from side B's by order: notes.jsonl is A's records then B's, so the first battle tag that
reappears after a different tag marks the start of B. Concurrent runs are safe (free ports, locked file).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from pcsa.common import paths
from pcsa.common.clock import SystemClock
from pcsa.common.config import load_config
from pcsa.run.sim_command import play_command

DEFAULT_OUT = Path(r"C:\Users\geral\Code\research\players\results.md")
COLUMNS = (
    "kind | opponent | top: W-L (win%, Wilson) | noisy: W-L (win%, Wilson) | avg turns | vol. switches/battle "
    "| blocked/battle | top rule firings | errors | timestamp | results dirs"
)
HEADER = f"""# Player evaluation results

Protocol: each kind (side A) plays the reference opponent in the local simulator, 400 paired battles per run
(each team pair is played from both sides), seed 1, once with top builds and once with `--noisy` builds.
Teams follow the player: style_* kinds are dealt teams that fit their style. W-L counts decided battles; the
interval is the Wilson 95% interval on A's win rate. Avg turns, switches and blocked switches are averaged over both
runs; vol. switches are notes with a `switch` key not starting with "forced"; blocked are `switch_blocked` notes;
rule firings are counts of the `rule` note value up to the first ':' (top run, per battle). Errors sum both runs.
Rows are replaced per kind by `scripts/eval_players.py`.

| {COLUMNS} |
|{"---|" * 11}
"""


def split_side_a(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    last = None
    for i, r in enumerate(records):
        tag = r.get("battle")
        if tag in seen and tag != last:
            return records[:i]
        seen.add(tag)
        last = tag
    return records


def run_one(config, args: argparse.Namespace, *, noisy: bool) -> dict:
    outcome = play_command(
        config,
        args.kind,
        args.opponent,
        args.battles,
        args.seed,
        noisy=noisy,
        port=None,
        a_team_style=args.team_style,
    )
    for line in outcome.lines:
        print(line, flush=True)
    m = re.search(r"results in (.+)$", outcome.lines[0]) or re.search(r"see (.+)$", outcome.lines[0])
    out_dir = Path(m.group(1).strip())
    summary = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))["summary"]
    records = [json.loads(x) for x in (out_dir / "notes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    mine = split_side_a(records)
    keys: Counter = Counter()
    rules: Counter = Counter()
    vol = blocked = 0
    for r in mine:
        keys.update(k for k in r if k not in ("battle", "turn", "kind"))
        if r.get("kind") == "team":
            continue
        if "switch" in r and not str(r["switch"]).startswith("forced"):
            vol += 1
        blocked += "switch_blocked" in r
        if "rule" in r:
            rules[str(r["rule"]).split(":")[0].strip()] += 1
    return {"summary": summary, "dir": out_dir, "vol": vol, "blocked": blocked, "rules": rules, "keys": keys}


def cell(run: dict | None) -> str:
    if run is None:
        return "-"
    s = run["summary"]
    rate = s["a_win_rate"]
    low, high = s["wilson_95"]
    if rate is None:
        return f"{s['a_wins']}-{s['b_wins']} (n/a)"
    return f"{s['a_wins']}-{s['b_wins']} ({rate:.1%}, {low:.1%}-{high:.1%})"


def build_row(kind: str, opponent: str, top: dict | None, noisy: dict | None) -> str:
    runs = [r for r in (top, noisy) if r]
    n = sum(r["summary"]["battles"] for r in runs) or 1
    turns = sum(r["summary"]["avg_turns"] * r["summary"]["battles"] for r in runs) / n
    vol = sum(r["vol"] for r in runs) / n
    blocked = sum(r["blocked"] for r in runs) / n
    if top and top["rules"]:
        tb = top["summary"]["battles"] or 1
        rules = ", ".join(f"{k} {v / tb:.2f}" for k, v in top["rules"].most_common())
    else:
        rules = "-"
    errors = sum(len(r["summary"]["errors"]) for r in runs)
    dirs = "; ".join(f"{name}: {r['dir']}" for name, r in (("top", top), ("noisy", noisy)) if r)
    stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
    cells = [kind, opponent, cell(top), cell(noisy), f"{turns:.1f}", f"{vol:.2f}", f"{blocked:.2f}", rules]
    return "| " + " | ".join(cells + [str(errors), stamp, dirs]) + " |"


def update_file(out: Path, kind: str, row: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_suffix(".lock")
    clock = SystemClock()
    deadline = clock.now() + 120
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            if clock.now() > deadline:
                lock.unlink(missing_ok=True)  # stale lock
            clock.sleep(0.2)
    try:
        text = out.read_text(encoding="utf-8") if out.exists() else HEADER
        lines = text.splitlines()
        rows = [i for i, ln in enumerate(lines) if ln.startswith("| ") and not ln.startswith("| kind |")]
        mine = [i for i in rows if lines[i].split(" | ")[0] == f"| {kind}"]
        if mine:
            lines[mine[0]] = row
        else:
            lines.insert(rows[-1] + 1 if rows else len(lines), row)
        tmp = out.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, out)
    finally:
        lock.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--opponent", default="best_matchup_max_damage")
    ap.add_argument("--battles", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", choices=("top", "noisy"))
    ap.add_argument(
        "--team-style", default=None, help="deal side A teams fitting this style; the row becomes kind@style"
    )
    args = ap.parse_args()
    config = load_config(paths.config_dir() / "agent.yaml", [])
    top = noisy = None
    if args.only != "noisy":
        top = run_one(config, args, noisy=False)
    if args.only != "top":
        noisy = run_one(config, args, noisy=True)
    label = f"{args.kind}@{args.team_style}" if args.team_style else args.kind
    row = build_row(label, args.opponent, top, noisy)
    update_file(args.out, label, row)
    print(row)
    for name, r in (("top", top), ("noisy", noisy)):
        if r:
            print(f"{name} side-A note keys: {dict(r['keys'])}")
    runs = [r["summary"] for r in (top, noisy) if r]
    bad = any(s["errors"] or s["unfinished"] > 0.05 * s["battles"] for s in runs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
