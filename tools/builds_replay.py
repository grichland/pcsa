"""Replay recorded live battles through the search's guesses of their sets, worlds_mode builds against legacy:
items and moves predicted before they are revealed (top-1, top-3), whether the guessed speed gets later turn
orders right, how often CG's team check refuses a drawn team, and the cost of the build weights per request.

    uv run python tools/builds_replay.py [--runs output/runs] [--max 400] [--out output/builds-replay.json]

Battles in pcsa-benchmark's lists (../archive/pcsa-benchmark/*/battles.json) are left out. The belief at each
decision is the one recorded with it (top 3 items and abilities, top 8 moves; the rest at the floor).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pcsa.common.clock import precise_now
from pcsa.network import builds, cg_adapter, evidence, worlds
from pcsa.network.preview_search import cg_set, checked

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parent / "archive" / "pcsa-benchmark"


def benchmark_battles() -> set[str]:
    out: set[str] = set()
    for f in BENCH.glob("*/battles.json"):
        out |= set(json.loads(f.read_text(encoding="utf-8")))
    return out


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def beliefs_by_turn(path: Path) -> dict[int, list[dict]]:
    """turn -> the recorded belief of the last decision at that turn."""
    out: dict[int, list[dict]] = {}
    for rec in read_jsonl(path) if path.is_file() else []:
        belief = (rec.get("extras") or {}).get("belief")
        if isinstance(belief, list) and isinstance(rec.get("turn"), int):
            out[rec["turn"]] = belief
    return out


def belief_arrays(g: builds.Guesser, recorded: list[dict], species: list[str]) -> worlds.Belief:
    t, n = g.tables, len(species)
    floor = builds.BELIEF_FLOOR
    moves, items = np.full((n, len(t.moves)), floor), np.full((n, len(t.items)), floor)
    abilities = np.full((n, len(t.abilities)), floor)
    by = {b.get("species"): b for b in recorded}
    for j, s in enumerate(species):
        b = by.get(s) or {}
        for key, arr, ix in (("moves", moves, t.move_id), ("item", items, g.item_ix), ("ability", abilities, g.ability_ix)):
            for name, p in b.get(key) or []:
                if name in ix and ix[name] < arr.shape[1]:
                    arr[j, ix[name]] = max(float(p), floor)
    return worlds.Belief(species, np.full(n, 0.5), moves, items, abilities)


class Battle:
    """What the events showed so far, per opposing Pokémon."""

    def __init__(self) -> None:
        self.moves: dict[str, set[str]] = defaultdict(set)
        self.ability: dict[str, str] = {}
        self.form: dict[str, str] = {}
        self.entered: list[str] = []

    def feed(self, events: list) -> None:
        for e in events:
            if not isinstance(e, dict) or e.get("side") != "opp":
                continue
            mon = builds.to_id(e.get("mon"))
            if e.get("k") == "enter" and mon not in self.entered:
                self.entered.append(mon)
            elif e.get("k") == "move" and not e.get("called_by"):
                self.moves[mon].add(builds.to_id(e.get("move")))
            elif e.get("k") == "ability":
                self.ability.setdefault(mon, builds.to_id(e.get("ability")))
            elif e.get("k") == "mega" and e.get("form"):
                self.form[mon] = builds.to_id(e.get("form"))

    def mon(self, mon: str, tracker: evidence.Tracker) -> SimpleNamespace:
        return SimpleNamespace(species=mon, form=self.form.get(mon), moves=[(m, 0) for m in self.moves[mon]],
                               item=tracker.had_item.get(mon), ability=self.ability.get(mon))  # fmt: skip


def ranked(scores: dict[str, float], drop: set[str]) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1]) if k and k not in drop]


def predictions(g: builds.Guesser, p: SimpleNamespace, name: str, belief: worlds.Belief, tracker: evidence.Tracker,
                common: dict) -> dict:  # fmt: skip
    """Item and move rankings and a speed guess (a build, or None for the common set) per mode."""
    row = belief.row([name])
    if row is None:
        return {}
    head_items = {g.tables.items[i]: float(belief.item[row][i]) for i in np.argsort(-belief.item[row])[:3]}
    head_moves = {g.tables.moves[i]: float(belief.moves[row][i]) for i in np.argsort(-belief.moves[row])[:8]}
    cands, w, _ = g.weigh(p, name, belief, tracker)
    items: dict[str, float] = defaultdict(float)
    moves: dict[str, float] = defaultdict(float)
    for c, x in zip(cands, w, strict=True):
        if c is None:
            for k, v in head_items.items():
                items[k] += x * v
            for k, v in head_moves.items():
                moves[k] += x * v
        else:
            items[c.item or "nothing"] += x
            for m in c.moves:
                moves[m] += x
    return {"builds": (ranked(items, set()), ranked(moves, set()), cands, w),
            "legacy": (ranked(head_items, set()), ranked(head_moves, set()), common.get(name), ranked(head_items, set()))}


def order_right(g: builds.Guesser, o: evidence.Order, species: str, build: builds.Build) -> bool | None:
    raw = g.dex.speed(o.form, build.nature, build.points) or g.dex.speed(species, build.nature, build.points)
    if raw is None or build.ability in evidence.ORDER_ABILITIES or build.item in evidence.ORDER_ITEMS:
        return None
    theirs = evidence.effective(raw, o.stage, build.item, o.par, o.tailwind)
    return None if theirs == o.ours else (theirs > o.ours) == o.first


def common_build(common: dict | None, item: str) -> builds.Build | None:
    if not common:
        return None
    return builds.Build(item, common.get("ability", ""), common.get("nature", ""), evidence.points_of(common.get("points")),
                        tuple(common.get("moves") or ()), 1.0, "common")  # fmt: skip


def three_check(start: dict, beliefs: dict, final: Battle, rates: worlds.Rates, stats: dict,
                rng: np.random.Generator) -> None:  # fmt: skip
    """At the first turn (their lead shown): how often the drawn three are the three they brought, by the belief
    head's `brought` (old) and by the live bring rates (new)."""
    preview = [builds.to_id(s) for s in start.get("opp_preview") or []]
    brought = set(final.entered)
    if len(preview) != 6 or len(brought) != 3 or 1 not in beliefs or not final.entered:  # noqa: PLR2004
        return
    lead = final.entered[0]
    team = [SimpleNamespace(species=s, form=s, revealed=s == lead, active=s == lead) for s in preview]
    battle = SimpleNamespace(opponent_preview=preview, opponent_team=team)
    head = {b.get("species"): float(b.get("brought") or 0.5) for b in beliefs[1]}
    for mode, w in (("head", [head.get(s, 0.5) for s in preview]), ("rates", [rates.brought(s) for s in preview])):
        belief = worlds.Belief(preview, np.array(w), np.zeros((6, 1)), np.zeros((6, 1)), np.zeros((6, 1)))
        for _ in range(50):
            three = {preview[i] for i in worlds.pick_three(battle, belief, rng, lead)}
            stats["three"][mode]["draws"] += 1
            stats["three"][mode]["all_right"] += three == brought
            stats["three"][mode]["members_right"] += len(three & brought) - 1  # of the two unseen


def replay(path: Path, g: builds.Guesser, common: dict, stats: dict, rng: np.random.Generator) -> None:
    messages = read_jsonl(path)
    start = next((m for m in messages if m.get("type") == "battle_start"), None)
    if start is None:
        return
    beliefs = beliefs_by_turn(path.with_name(path.stem + ".neural.jsonl"))
    final, end_tracker = Battle(), evidence.Tracker(g.dex, own=start.get("own_team") or [])
    turns: list[int] = []
    for m in messages:
        if m.get("type") == "events":
            final.feed(m["events"])
            for e in m["events"]:
                if isinstance(e, dict) and e.get("k") == "turn":
                    turns.append(len(end_tracker.orders))
            end_tracker.feed(m["events"])
    end_tracker.feed([{"k": "turn"}])
    if "rates" in stats:
        three_check(start, beliefs, final, stats["rates"], stats, rng)
    own = [cg_set({**s, "species": s["species"]}) for s in start.get("own_team") or []]
    now, tracker, turn = Battle(), evidence.Tracker(g.dex, own=start.get("own_team") or []), 0
    for m in messages:
        if m.get("type") == "events":
            now.feed(m["events"])
            tracker.feed(m["events"])
            turn += sum(1 for e in m["events"] if isinstance(e, dict) and e.get("k") == "turn")
            continue
        if m.get("type") != "decision_request" or m.get("kind") != "turn" or turn not in beliefs:
            continue
        species = [builds.to_id(s) for s in start.get("opp_preview") or []]
        belief = belief_arrays(g, beliefs[turn], species)
        for mon in now.entered:
            score(g, mon, now, final, tracker, end_tracker, belief, common, stats, len(tracker.orders))
        if len(species) == len(own):  # CG checks whole teams of six
            team = [(i, now.mon(s, tracker), s) for i, s in enumerate(species)]
            check_draws(g, team, belief, tracker, own, species, stats, rng)


def score(g, mon, now, final, tracker, end_tracker, belief, common, stats, seen_orders) -> None:  # noqa: PLR0913
    p = now.mon(mon, tracker)
    pred = predictions(g, p, mon, belief, tracker, common)
    final_item, shown = end_tracker.had_item.get(mon), now.moves[mon]
    later_moves = final.moves[mon] - shown
    for mode, (items, moves, *rest) in pred.items():
        s = stats[mode]
        if final_item and not tracker.had_item.get(mon):
            s["item_n"] += 1
            s["item_top1"] += items[:1] == [final_item]
            s["item_top3"] += final_item in items[:3]
        free = [m for m in moves if m not in shown]
        for mv in later_moves:
            s["move_n"] += 1
            s["move_top1"] += free[:1] == [mv]
            s["move_top3"] += mv in free[:3]
        future = [o for o in end_tracker.orders[seen_orders:] if evidence._same(o.mon, mon)]
        if not future:
            continue
        if mode == "builds":
            cands, w = rest
            for o in future:
                rights = [(x, order_right(g, o, mon, c)) for c, x in zip(cands, w, strict=True) if c is not None]
                rights = [(x, r) for x, r in rights if r is not None]
                if rights:
                    s["order_n"] += 1
                    s["order_right"] += sum(x for x, r in rights if r) / sum(x for x, _ in rights)
        else:
            base = common_build(rest[0], (rest[1] or [""])[0])
            for o in future:
                r = order_right(g, o, mon, base) if base else None
                if r is not None:
                    s["order_n"] += 1
                    s["order_right"] += float(r)


def check_draws(g, team, belief, tracker, own, species, stats, rng) -> None:  # noqa: PLR0913
    t0 = precise_now()
    drawn, _ = g.guess_team(team, belief, tracker, rng)
    stats["builds"]["ms"] += (precise_now() - t0) * 1000.0
    legacy = [worlds.guess_set(p, i, n, belief, g.tables, rng) for i, p, n in team]
    for mode, sets in (("builds", drawn), ("legacy", legacy)):
        cg = [cg_set({"species": n, **{k: v for k, v in s.items() if k != "slot"},
                      **_shown(p)}) for (_, p, n), s in zip(team, sets, strict=True)]  # fmt: skip
        stats[mode]["teams"] += 1
        stats[mode]["refused"] += bool(checked([own, cg]))


def _shown(p: SimpleNamespace) -> dict:
    """What the Battle shows wins over a guess, as Session.game does."""
    out: dict = {}
    if p.item:
        out["item"] = p.item
    if p.ability:
        out["ability"] = p.ability
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--runs", type=Path, default=ROOT / "output" / "runs")
    parser.add_argument("--max", type=int, default=400)
    parser.add_argument("--since", default="", help="only battles from this tag on (e.g. 20261123T2120)")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "builds-replay.json")
    args = parser.parse_args(argv)
    skip = benchmark_battles()
    g = builds.Guesser(worlds.Tables(cg_adapter.learnset().numpy(), {}))
    top = json.loads((ROOT / "assets" / "sets" / "top_builds.json").read_text(encoding="utf-8"))
    common = worlds.common_sets(top)
    keys = ("item_n", "item_top1", "item_top3", "move_n", "move_top1", "move_top3", "order_n", "order_right",
            "teams", "refused", "ms")  # fmt: skip
    stats: dict = {m: dict.fromkeys(keys, 0.0) for m in ("builds", "legacy")}
    stats["rates"] = worlds.Rates.load(str(ROOT / "assets" / "sets" / "opponent_preview_rates.json"))
    stats["three"] = {m: {"draws": 0, "all_right": 0, "members_right": 0} for m in ("head", "rates")}
    files = sorted(f for f in args.runs.glob("*/cg/*.jsonl") if not f.name.endswith(".neural.jsonl")
                   and f.stem not in skip and f.stem >= args.since)[: args.max]  # fmt: skip
    rng = np.random.default_rng(0)
    for k, f in enumerate(files):
        try:
            replay(f, g, common, stats, rng)
        except Exception as error:  # one bad record never stops the replay
            sys.stderr.write(f"{f.name}: {type(error).__name__}: {error}\n")
        if k % 25 == 0:
            sys.stdout.write(f"{k + 1}/{len(files)} battles\n")
            sys.stdout.flush()
    out = {"battles": len(files), "left_out_benchmark": len(skip)}
    for mode in ("builds", "legacy"):
        s = stats[mode]
        out[mode] = {
            "item_top1": round(s["item_top1"] / max(s["item_n"], 1), 3), "item_top3": round(s["item_top3"] / max(s["item_n"], 1), 3),
            "items": int(s["item_n"]),
            "move_top1": round(s["move_top1"] / max(s["move_n"], 1), 3), "move_top3": round(s["move_top3"] / max(s["move_n"], 1), 3),
            "moves": int(s["move_n"]),
            "order_right": round(s["order_right"] / max(s["order_n"], 1), 3), "orders": int(s["order_n"]),
            "refused": round(s["refused"] / max(s["teams"], 1), 3), "teams": int(s["teams"]),
        }  # fmt: skip
    out["their_three_at_turn_1"] = {
        m: {"all_three_right": round(t["all_right"] / max(t["draws"], 1), 3),
            "unseen_member_right": round(t["members_right"] / max(2 * t["draws"], 1), 3), "draws": t["draws"]}
        for m, t in stats["three"].items()}  # fmt: skip
    out["builds"]["weights_ms_per_request"] = round(stats["builds"]["ms"] / max(stats["builds"]["teams"], 1), 2)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
