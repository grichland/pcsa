"""Search on recorded ladder turns: how long it takes, how often it changes the pick, and which pick the play-on
results favour.

    uv run python tools/search_recorded.py --checkpoint assets/models/<file>.pt \
        --battles ../pcsa-former/runs/live-battles.jsonl.gz --playon ../pcsa-former/runs/playon-20261119-c

Each recorded battle (pcsa-former's packed file: the messages the controller sent) is replayed through
pcsa.cg.serve with the neural player and search on. Per turn request: the search note (time, simulations, the
model's pick, the search's pick). Where the play-on folder holds that turn (game id = battle index * 10^6 + turn *
1000: every pair of our action and theirs played to the end), both picks are scored by the pair results, against
their actions weighted by the network's own p_opp and, where the log showed their real choice, against that one.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from pcsa.cg.serve import NEURAL, Server, make_player  # noqa: E402
from pcsa.common.clock import precise_now  # noqa: E402

POLICY = "pcsa.network.policy:Policy"
TURN = "turn"


def labels(folder: Path) -> dict[int, dict]:
    """game id -> the recorded turn's pair results (the sequence's last row), for ids of recorded turns."""
    out = {}
    for path in sorted(folder.glob("*.npz")):
        d = np.load(path)
        last = np.cumsum(d["lengths"]) - 1
        for game, row in zip(d["game"], last, strict=True):
            if int(game) % 1000 == 0:
                out[int(game)] = {"result": d["cf_result"][row], "ok": d["cf_ok"][row], "col": d["cf_opp_col"][row],
                                  "move": d["cf_opp_move"][row], "mask": d["mask"][row]}  # fmt: skip
    return out


def replay(rec: dict, args: dict) -> list[dict]:
    """The battle's turn requests in order, each new one once: {search, legal, p_opp, ms}."""
    server = Server(make_player(NEURAL, args))
    turns = []
    for m in rec["msgs"]:
        if "type" not in m:
            continue
        reply = server.handle(m)
        if not isinstance(reply, dict) or "order" not in reply:
            continue
        notes = reply.get("notes") or {}
        net, shell = notes.get("net") or {}, notes.get("model") or {}
        if net.get("kind") != TURN or shell.get("reused"):
            continue
        turns.append({"search": net.get("search"), "legal": net.get("legal"), "p_opp": net.get("p_opp"),
                      "ms": shell.get("ms")})  # fmt: skip
    return turns


def score(label: dict, action: int, weight: np.ndarray) -> float | None:
    ok = label["ok"][action]
    w = np.where(ok, weight, 0.0)
    return float((label["result"][action] * w).sum() / w.sum()) if w.sum() > 0 else None


def compare(label: dict, turn: dict, shown: list | None) -> dict | None:
    """The model's and the search's pick by the pair results; None when the turn does not line up."""
    note = turn["search"]
    legal = sorted(int(a) for a in np.flatnonzero(label["mask"]))
    if note is None or sorted(turn["legal"] or []) != legal:
        return None
    p_opp = np.array(turn["p_opp"] or [0.0] * 13)
    by_model = np.array([p_opp[c] if c >= 0 else 0.0 for c in label["col"]])
    even = np.ones(len(label["col"]))
    real = np.zeros(len(label["col"]))
    if shown and shown[0] == "move" and len(shown) > 3 and shown[3]:
        hit = np.flatnonzero(label["move"] == shown[3])
        lo = 4 if shown[2] else 0
        hit = [j for j in hit if lo <= j < lo + 4]
        if hit:
            real[hit[0]] = 1.0
    out = {"changed": bool(note["changed"]), "used": bool(note["used"])}
    for name, w in (("p_opp", by_model), ("even", even), ("real", real)):
        a, b = score(label, note["model_pick"], w), score(label, note["pick"], w)
        out[name] = None if a is None or b is None else (a, b)
    return out


def tally(rows: list[dict], key: str) -> dict:
    pairs = [r[key] for r in rows if r["changed"] and r[key] is not None]
    diff = [b - a for a, b in pairs]
    return {"turns": len(pairs), "search_better": sum(d > 1e-9 for d in diff), "model_better": sum(d < -1e-9 for d in diff),
            "equal": sum(abs(d) <= 1e-9 for d in diff), "mean_model": float(np.mean([a for a, _ in pairs])) if pairs else None,
            "mean_search": float(np.mean([b for _, b in pairs])) if pairs else None}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--battles", required=True)
    ap.add_argument("--playon", default=None)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--latency-turns", type=int, default=200)
    ap.add_argument("--policy-args", default="{}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    found = labels(Path(a.playon)) if a.playon else {}
    wanted = sorted({g // 10**6 for g in found})
    policy_args = {"threads": 4, "search": True, **json.loads(a.policy_args)}
    args = {"policy": POLICY, "checkpoint": str(Path(a.checkpoint).resolve()), "device": "cpu", "policy_args": policy_args,
            "fallback": "tower", "timeout_s": 30.0, "hard_timeout_s": 60.0}  # fmt: skip
    with gzip.open(a.battles, "rt", encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh]
    order = [i for i in wanted if i < len(recs)] + [i for i in range(len(recs)) if i not in set(wanted)]
    end = precise_now() + 60.0 * a.minutes
    notes, rows, battles = [], [], 0
    for index in order:
        if precise_now() > end:
            break
        turns = replay(recs[index], args)
        battles += 1
        for k, turn in enumerate(turns):
            if turn["search"] is not None:
                notes.append({**turn["search"], "shell_ms": turn["ms"], "battle": index, "turn": k})
            label = found.get(index * 10**6 + k * 1000)
            if label is not None:
                row = compare(label, turn, (recs[index].get("shown") or {}).get(str(k)))
                if row is not None:
                    rows.append(row)
        print(f"battle {index}: {len(turns)} turn requests, {len(notes)} searched so far, {len(rows)} with play-on results", flush=True)
    first = notes[: a.latency_turns]
    ms = [n["ms"] for n in first]
    shell = [n["shell_ms"] for n in first if n["shell_ms"] is not None]
    over = [n for n in first if n.get("budget_ms") is not None and n["ms"] > n["budget_ms"]]
    used = [n for n in notes if n["used"]]
    why: dict[str, int] = {}
    for n in notes:
        if not n["used"]:
            why[str(n["why"])[:70]] = why.get(str(n["why"])[:70], 0) + 1
    report = {
        "checkpoint": Path(a.checkpoint).name, "battles": battles, "turn_requests": len(notes),
        "latency": {"turns": len(first), "search_ms_median": float(np.median(ms)) if ms else None,
                    "search_ms_max": float(max(ms)) if ms else None,
                    "decision_ms_median": float(np.median(shell)) if shell else None,
                    "decision_ms_max": float(max(shell)) if shell else None,
                    "budget_ms_min": min((n["budget_ms"] for n in first if n.get("budget_ms") is not None), default=None),
                    "search_over_its_budget": len(over)},
        "search_used": len(used), "not_used": why,
        "sims_median": float(np.median([n["sims"] for n in used])) if used else None,
        "worlds_median": float(np.median([n["worlds"] for n in used])) if used else None,
        "worlds_dropped": int(sum(n.get("dropped", 0) for n in notes)),
        "changed": sum(bool(n["changed"]) for n in notes),
        "changed_share": round(sum(bool(n["changed"]) for n in notes) / len(notes), 4) if notes else None,
        "playon": {"turns_lined_up": len(rows), "changed": sum(r["changed"] for r in rows),
                   "by_p_opp": tally(rows, "p_opp"), "even": tally(rows, "even"), "their_real_choice": tally(rows, "real")},
    }  # fmt: skip
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"report": report, "notes": notes}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
