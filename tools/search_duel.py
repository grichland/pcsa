"""Two settings of the neural player against each other in the simulator, both searching: A vs B on paired seeds.

    uv run python tools/search_duel.py --checkpoint assets/models/<file>.pt --teams <cg teams file> --own-last \
        --a cg.search.calibrate=false --b cg.search.calibrate=true --minutes 9.5 --out duel.jsonl

Each seed is played twice with the same teams: A on our team (seat 0) against B on theirs, then B on ours against
A on theirs, so each setting plays both sides of every matchup. Both use config/agent.yaml with search on
(--set for both, --a / --b for one). One JSON line per game pair; the totals at the end, and on --minutes the pairs
that finished.
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


def make(checkpoint: str, overrides: list[str]) -> Policy:
    cfg = load_config(paths.config_dir() / "agent.yaml", ["cg.search.enabled=true", *overrides]).agent.cg
    return Policy(checkpoint, **startup.policy_kwargs(cfg))


def game(seats: list[Policy], seed: int, teams: str | None, weights: list | None) -> dict:
    """One game, seats[0] in seat 0 and seats[1] in seat 1: {winner (0, 1, -1 tie), turns, searched per seat}."""
    env = cg.BatchEnv(teams, 1, seed, threads=1, team_weights=weights)
    for p in seats:
        p.reset([0])
    searched = [0, 0]
    while True:
        t = env.observe_tensors()
        seat = int(t.seat[0])
        p = seats[seat]
        out = p.decide(t, [0], [env.battle(0)])
        legal = np.flatnonzero(np.asarray(t.action_mask)[0])
        action = int(legal[np.argmax(out["scores"][0].numpy()[legal])])
        if int(t.kind[0]) == cg.KIND_TURN:
            p.played(int(t.decision[0]), action)
            note = out["extras"][0].get("search") or {}
            searched[seat] += bool(note.get("used"))
        env.act([action])
        r = env.results()
        if np.asarray(r.done)[0]:
            return {"winner": int(np.asarray(r.winner)[0]), "turns": int(np.asarray(r.turns)[0]), "searched": searched}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teams", default=None)
    ap.add_argument("--own-last", action="store_true", help="seat 0 plays the last team of --teams, seat 1 any other")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=9.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override for both")
    ap.add_argument("--a", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override for A")
    ap.add_argument("--b", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override for B")
    a = ap.parse_args()
    pa, pb = make(a.checkpoint, [*a.set, *a.a]), make(a.checkpoint, [*a.set, *a.b])
    weights = None
    if a.own_last:
        n = cg.check_teams(a.teams)[0]
        weights = [[0.0] * (n - 1) + [1.0], [1.0] * (n - 1) + [0.0]]
    end = precise_now() + 60.0 * a.minutes
    out = open(a.out, "a", encoding="utf-8") if a.out else None  # noqa: SIM115
    score = {"a": 0.0, "b": 0.0}
    pairs = 0
    for g in range(a.games):
        if precise_now() > end:
            break
        seed = a.seed + g
        first = game([pa, pb], seed, a.teams, weights)  # A on our team
        second = game([pb, pa], seed, a.teams, weights)  # B on our team
        for res, who in ((first, ("a", "b")), (second, ("b", "a"))):
            if res["winner"] < 0:
                score["a"] += 0.5
                score["b"] += 0.5
            else:
                score[who[res["winner"]]] += 1.0
        pairs += 1
        row = {"seed": seed, "a_on_ours": first, "b_on_ours": second}
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        print(f"pair {g}: A {score['a']:.1f}  B {score['b']:.1f}  of {2 * pairs}", flush=True)
    total = {"pairs": pairs, "games": 2 * pairs, "a": score["a"], "b": score["b"], "a_set": a.a, "b_set": a.b,
             "both_set": a.set, "checkpoint": Path(a.checkpoint).name}  # fmt: skip
    print(json.dumps(total))
    if out:
        out.write(json.dumps({"total": total}) + "\n")
        out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
