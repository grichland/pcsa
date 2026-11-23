"""The neural player with and without search in the simulator, same seeds: wins against a built-in player.

    uv run python tools/search_eval.py --checkpoint assets/models/<file>.pt --opponent tower --games 512

Each game is played twice from the same seed (same teams): once by the model alone, once with search. Both use the
policy settings of config/agent.yaml (`--set cg.search.ms=2500` and the like change them). Writes one JSON line per
finished game to --out and prints the totals; --minutes stops it early and reports what finished.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import champions_gym as cg  # noqa: E402
import numpy as np  # noqa: E402

from pcsa.cg import startup  # noqa: E402
from pcsa.common import paths  # noqa: E402
from pcsa.common.clock import precise_now  # noqa: E402
from pcsa.common.config import load_config  # noqa: E402
from pcsa.network.policy import Policy  # noqa: E402

TURN_ACTIONS = 11


def play(policy: Policy, opponent: str, seed: int, teams: str | None, weights: list | None) -> dict:
    """One game, the policy in seat 0: {win, turns, decisions, searched, changed, ms}."""
    env = cg.BatchEnv(teams, 1, seed, threads=1, agent_seat=0, team_weights=weights)
    policy.reset([0])
    stats = {"seed": seed, "decisions": 0, "searched": 0, "changed": 0, "ms": [], "why": {}}
    while True:
        t = env.observe_tensors()
        action = 0
        if int(t.seat[0]) == 0:
            out = policy.decide(t, [0], [env.battle(0)])
            legal = np.flatnonzero(np.asarray(t.action_mask)[0])
            scores = out["scores"][0].numpy()
            action = int(legal[np.argmax(scores[legal])])
            if int(t.kind[0]) == cg.KIND_TURN:
                policy.played(int(t.decision[0]), action)
                stats["decisions"] += 1
                note = out["extras"][0].get("search")
                if note is not None:
                    stats["ms"].append(note["ms"])
                    stats["searched"] += bool(note["used"])
                    stats["changed"] += bool(note["changed"])
                    if not note["used"]:
                        why = str(note["why"])[:60]
                        stats["why"][why] = stats["why"].get(why, 0) + 1
        env.act([action], opponent=opponent)
        r = env.results()
        if np.asarray(r.done)[0]:
            stats["win"] = int(np.asarray(r.winner)[0]) == 0
            stats["tie"] = int(np.asarray(r.winner)[0]) < 0
            stats["turns"] = int(np.asarray(r.turns)[0])
            return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="tower")
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teams", default=None, help="a CG teams file; default: CG's own pool")
    ap.add_argument("--own-last", action="store_true", help="we play the last team of --teams, they any other")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="config override")
    a = ap.parse_args()
    cfg = load_config(paths.config_dir() / "agent.yaml", ["cg.search.enabled=true", *a.set]).agent.cg
    kwargs = startup.policy_kwargs(cfg)
    players = {
        "model": Policy(a.checkpoint, **{**kwargs, "search": None}),
        "search": Policy(a.checkpoint, **kwargs),
    }
    weights = None
    if a.own_last:
        n = cg.check_teams(a.teams)[0]
        weights = [[0.0] * (n - 1) + [1.0], [1.0] * (n - 1) + [0.0]]
    end = precise_now() + 60.0 * a.minutes
    out = open(a.out, "a", encoding="utf-8") if a.out else None  # noqa: SIM115
    wins = {k: 0 for k in players}
    done, ms, changed, decisions, searched = 0, [], 0, 0, 0
    for g in range(a.games):
        if precise_now() > end:
            break
        row = {"game": g, "opponent": a.opponent}
        for name, policy in players.items():
            row[name] = play(policy, a.opponent, a.seed + g, a.teams, weights)
            wins[name] += row[name]["win"]
        s = row["search"]
        ms += s["ms"]
        changed, decisions, searched = changed + s["changed"], decisions + s["decisions"], searched + s["searched"]
        done += 1
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        print(f"game {g}: model {wins['model']}/{done}  search {wins['search']}/{done}", flush=True)
    total = {
        "opponent": a.opponent, "games": done, "asked": a.games, "model_wins": wins["model"],
        "search_wins": wins["search"], "turn_decisions": decisions, "searched": searched, "changed": changed,
        "search_ms_median": float(np.median(ms)) if ms else None, "search_ms_max": float(max(ms)) if ms else None,
        "settings": players["search"].info["search"], "checkpoint": Path(a.checkpoint).name,
    }  # fmt: skip
    print(json.dumps(total))
    if out:
        out.write(json.dumps({"total": total}) + "\n")
        out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
