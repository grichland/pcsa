"""Replay the recorded turn decisions through each choice rule (pcsa.network.choosing), offline.

    python tools/choice_replay.py [--runs output/runs] [--out output/choice-replay]

Every turn record with the network's matrix (notes.cg.net) is reduced again by each rule and k. Per setting:
how often the pick differs from the network's own (the mean over all replies), the expected score that costs by
the network's own measure, the cell difference against the reply they actually made, how often that reply was
among the kept ones, and the win rate of battles with and without a changed turn (forfeit and unknown are wins).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from pcsa.common.config_cg import ChoiceSettings
from pcsa.network.choosing import scores

WON = {"win", "opponent_forfeit", "unknown"}


def rule(name: str, ours: int, theirs: int, blend: float = 0.5) -> ChoiceSettings:
    return ChoiceSettings(rule=name, top_ours=ours, top_theirs=theirs, blend=blend)  # type: ignore[arg-type]


SETTINGS = [rule("mean", 0, k) for k in (2, 3, 4)] + [rule("min", 0, k) for k in (2, 3, 4, 6)] + [
    rule("mix", 0, 3), rule("mix", 0, 4)] + [rule("regret", 0, k) for k in (2, 3, 4, 6, 0)] + [
    rule("regret", 4, 3), rule("regret", 4, 4)]  # fmt: skip


@dataclass
class Turn:
    battle: str
    won: bool
    matrix: torch.Tensor  # [11,13]
    p_opp: torch.Tensor
    pi: torch.Tensor
    allowed: torch.Tensor  # bool [11]
    score: torch.Tensor  # the network's expected score
    actual: int | None  # their column this turn, when the log shows it


def reply_column(events: list, turn: int, cols: list[str]) -> int | None:
    """Their action in `turn` as a column of `cols` (4 moves, the same with Mega, 5 switches), or None."""
    seg, inside = [], False
    for e in events:
        if e.get("k") == "turn":
            if inside:
                break
            inside = e.get("n") == turn
        elif inside:
            seg.append(e)
    mega, fainted = False, False
    for e in seg:
        if e.get("side") != "opp":
            continue
        k = e.get("k")
        if k == "mega":
            mega = True
        elif k == "faint":
            fainted = True
        elif k == "move":
            hits = [i for i in range(8) if cols[i] == e.get("move")]
            pick = [i for i in hits if (i >= 4) == mega] or hits
            return pick[0] if pick else None
        elif k == "enter" and not fainted:
            mon = str(e.get("mon"))
            hits = [i for i in range(8, len(cols)) if cols[i] and (cols[i].startswith(mon) or mon.startswith(cols[i]))]
            return hits[0] if hits else None
    return None


def battle_turns(folder: Path) -> list[Turn]:
    try:
        outcome = json.loads((folder / "battle.json").read_text(encoding="utf-8")).get("outcome")
        events = json.loads((folder / "events.json").read_text(encoding="utf-8")).get("events") or []
    except (OSError, ValueError):
        return []
    if outcome is None:
        return []
    found = []
    for path in sorted((folder / "decisions").glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        net = ((d.get("notes") or {}).get("cg") or {}).get("net") or {}
        if net.get("kind") != "turn" or not net.get("matrix") or not net.get("p_opp"):
            continue
        allowed = torch.zeros(11, dtype=torch.bool)
        for a in set(net.get("legal") or []) - set(net.get("closed") or []):
            allowed[int(a)] = True
        if allowed.sum() < 2:
            continue
        cols = list(net.get("cols") or [])
        actual = reply_column(events, int(d.get("turn") or 0), cols) if len(cols) == 13 else None
        found.append(Turn(folder.as_posix(), outcome in WON, torch.tensor(net["matrix"]), torch.tensor(net["p_opp"]),
                          torch.tensor(net["pi"]), allowed, torch.tensor(net["score"]), actual))  # fmt: skip
    return found


def measure(turns: list[Turn], cfg: ChoiceSettings) -> dict:
    m = torch.stack([t.matrix for t in turns])
    allowed = torch.stack([t.allowed for t in turns])
    s = scores(m, torch.stack([t.p_opp for t in turns]), torch.stack([t.pi for t in turns]), allowed, cfg)
    full = torch.stack([t.score for t in turns])
    own = torch.where(allowed, full, torch.full_like(full, float("-inf"))).argmax(-1)
    pick = s.argmax(-1)
    changed = pick != own
    idx = torch.arange(len(turns))
    cost = full[idx, own] - full[idx, pick]
    known = [i for i, t in enumerate(turns) if t.actual is not None]
    vs_actual = [float(m[i, pick[i], turns[i].actual] - m[i, own[i], turns[i].actual]) for i in known if changed[i]]
    kept = scores_kept(turns, cfg, known)
    battles: dict[str, list] = {}
    for i, t in enumerate(turns):
        battles.setdefault(t.battle, [t.won, False])[1] |= bool(changed[i])
    with_change = [w for w, c in battles.values() if c]
    without = [w for w, c in battles.values() if not c]
    n_changed = int(changed.sum())
    return {"rule": cfg.rule, "top_ours": cfg.top_ours, "top_theirs": cfg.top_theirs, "blend": cfg.blend,
            "turns": len(turns), "changed": n_changed, "changed_pct": round(100 * n_changed / len(turns), 1),
            "cost_per_changed": round(float(cost[changed].mean()), 4) if n_changed else 0.0,
            "vs_actual_turns": len(vs_actual),
            "vs_actual_mean": round(sum(vs_actual) / len(vs_actual), 4) if vs_actual else None,
            "vs_actual_better_pct": round(100 * sum(x > 0 for x in vs_actual) / len(vs_actual), 1) if vs_actual else None,
            "reply_kept_pct": kept,
            "battles_changed": len(with_change), "win_pct_changed": _pct(with_change),
            "battles_unchanged": len(without), "win_pct_unchanged": _pct(without)}  # fmt: skip


def scores_kept(turns: list[Turn], cfg: ChoiceSettings, known: list[int]) -> float | None:
    """How often their actual reply was among the kept columns."""
    if not known:
        return None
    k = cfg.top_theirs
    hits = 0
    for i in known:
        p = turns[i].p_opp
        top = set(range(13)) if k <= 0 else set(p.topk(k).indices.tolist())
        hits += turns[i].actual in top and float(p[turns[i].actual]) > 0
    return round(100 * hits / len(known), 1)


def _pct(xs: list) -> float | None:
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def table(rows: list[dict]) -> str:
    head = ["rule", "ours", "theirs", "changed", "cost/changed", "vs actual", "better", "reply kept",
            "win% changed (n)", "win% unchanged (n)"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        name = r["rule"] + (f" {r['blend']}" if r["rule"] == "mix" else "")
        cells = [name, r["top_ours"] or "all", r["top_theirs"] or "all", f"{r['changed_pct']}% ({r['changed']})",
                 r["cost_per_changed"], r["vs_actual_mean"], f"{r['vs_actual_better_pct']}% of {r['vs_actual_turns']}",
                 f"{r['reply_kept_pct']}%", f"{r['win_pct_changed']} ({r['battles_changed']})",
                 f"{r['win_pct_unchanged']} ({r['battles_unchanged']})"]  # fmt: skip
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="output/runs")
    ap.add_argument("--out", default="output/choice-replay")
    args = ap.parse_args()
    turns = [t for folder in sorted(Path(args.runs).glob("*/battle-*")) for t in battle_turns(folder)]
    if not turns:
        raise SystemExit("no turn records with the network's matrix")
    battles = len({t.battle for t in turns})
    known = sum(t.actual is not None for t in turns)
    print(f"{len(turns)} turn decisions in {battles} battles; their reply read in {known}")
    rows = [measure(turns, cfg) for cfg in SETTINGS]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"turns": len(turns), "battles": battles, "known": known,
                                                  "rows": rows}, indent=1), encoding="utf-8")  # fmt: skip
    text = table(rows)
    (out / "results.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
