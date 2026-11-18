"""Builds tests/fixtures/watch_run from a real run recorded before the CG wrapper existed.

It copies the manifest (rewritten to the flat neural form), events.jsonl, each battle's log, team_preview.json and a
few decision records, then adds what the wrapper adds (the pcsa-decision/4 contract): `notes.cg.view`, derived from the
record's own LIVE events, team and preview; and `notes.cg.net`, built from the run's `cg/<tag>.neural.jsonl` (the
real value, expected scores and their action probabilities) with a matrix that keeps every row's score, and a
plausible belief. Run: .venv/Scripts/python.exe tests/run/make_watch_fixture.py [source run]
"""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "output" / "runs" / "20261118-162900-neural-148"
TARGET = REPO / "tests" / "fixtures" / "watch_run"
PICKS = {1: (0, 1, 2, 6), 2: (0, 1, 5, 11), 3: (0, 2, 9, 10)}
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
COMMON_ITEMS = ("choicescarf", "focussash", "leftovers", "sitrusberry", "lifeorb", "choiceband", "assaultvest")
STACKING = ("spikes", "toxicspikes", "stealthrock", "stickyweb")


def table(kind: str) -> dict:
    return json.loads((REPO / "assets" / "game" / "cg" / f"{kind}.json").read_text(encoding="utf-8"))[kind]


SPECIES, MOVES, ITEMS = table("species"), table("moves"), table("items")
LEARNSETS = json.loads((REPO / "assets" / "game" / "learnsets.json").read_text(encoding="utf-8"))["learnsets"]


def to_id(text: str) -> str:
    return "".join(c for c in str(text).lower() if c.isalnum())


def types_of(species: str | None) -> list[str]:
    return [to_id(t) for t in SPECIES.get(species or "", {}).get("types", [])]


def mon(species: str | None, revealed: bool) -> dict:
    return {
        "species": species, "form": None, "hp": 1.0, "fainted": False, "status": "",
        "boosts": dict.fromkeys(BOOSTS, 0), "effects": [], "moves": [], "item": None, "ability": None,
        "types": types_of(species), "active": False, "revealed": revealed,
    }  # fmt: skip


class Tracker:
    """A small fold of LIVE events into the view's shape (enough for the dashboard fixture, not CG's tracker)."""

    def __init__(self, start: dict) -> None:
        self.turn = 0
        self.own, self.max_hp = [], {}
        for member in start["own_team"]:
            entry = mon(member["species"], True)
            entry.update(item=member["item"], ability=member["ability"])
            entry["moves"] = [[m, MOVES.get(m, {}).get("pp")] for m in member["moves"]]
            self.own.append(entry)
            self.max_hp[member["species"]] = member["stats"]["hp"]
        self.opp = [mon(s, False) for s in start["opp_preview"]]
        self.weather: dict = {}
        self.fields: dict = {}
        self.sides: dict = {"own": {}, "opp": {}}

    def find(self, side: str, species: str) -> dict:
        members = self.own if side == "own" else self.opp
        found = next((m for m in members if m["species"] == species), None)
        if found is None:
            found = mon(species, side == "own")
            members.append(found)
        return found

    def take(self, e: dict) -> None:  # noqa: C901, PLR0912
        k = e.get("k")
        if k == "turn":
            self.turn = e["n"]
        elif k == "weather":
            self.weather = {e["weather"]: self.turn} if e["what"] == "start" else {}
        elif k == "field" and e["what"] == "end":
            self.fields.pop(e["effect"], None)
        elif k == "field":
            self.fields.setdefault(e["effect"], self.turn)
        elif k == "side":
            self.side(e)
        elif "side" in e and "mon" in e:
            member = self.find(e["side"], e["mon"])
            if k == "enter":
                self.enter(e["side"], member)
            elif k == "withdraw":
                member.update(active=False, boosts=dict.fromkeys(BOOSTS, 0), effects=[])
            else:
                self.on_mon(str(k), e, member)

    def side(self, e: dict) -> None:
        conditions = self.sides[e["side"]]
        if e["what"] == "end":
            conditions.pop(e["effect"], None)
        elif e["what"] == "start" and e["effect"] in STACKING:
            conditions[e["effect"]] = conditions.get(e["effect"], 0) + 1
        elif e["what"] == "start":
            conditions.setdefault(e["effect"], self.turn)

    def enter(self, side: str, member: dict) -> None:
        for other in self.own if side == "own" else self.opp:
            if other["active"]:
                other.update(active=False, boosts=dict.fromkeys(BOOSTS, 0), effects=[])
        member.update(active=True, revealed=True)

    def on_mon(self, k: str, e: dict, m: dict) -> None:  # noqa: C901, PLR0912
        if k == "move":
            known = next((x for x in m["moves"] if x[0] == e["move"]), None)
            if known is None and e["side"] == "opp":
                m["moves"].append([e["move"], None])
            elif known is not None and isinstance(known[1], int):
                known[1] = max(0, known[1] - 1)
        elif k == "hp":
            whole = self.max_hp.get(m["species"])
            m["hp"] = e["after"] / whole if "after" in e and whole else e.get("after_pct", 100) / 100
        elif k == "faint":
            m.update(hp=0.0, fainted=True, status="")
        elif k == "status":
            m["status"] = e["status"]
        elif k == "cure":
            m["status"] = ""
        elif k == "boost":
            for stat in e["stats"]:
                m["boosts"][stat] = max(-6, min(6, m["boosts"].get(stat, 0) + e["by"]))
        elif k == "volatile" and e["effect"] not in m["effects"]:
            m["effects"].append(e["effect"])
        elif k == "volatile_end" and e["effect"] in m["effects"]:
            m["effects"].remove(e["effect"])
        elif k == "item":
            m["item"] = "" if e["what"] in ("used", "removed") else e["item"]
        elif k == "mega_stone":
            m["item"] = e["stone"]
        elif k == "ability":
            m["ability"] = e["ability"]
        elif k == "mega" and e.get("form"):
            m.update(form=e["form"], types=types_of(e["form"]))
        elif k == "types" and e.get("type"):
            m["types"] = [e["type"]]

    def view(self, kind: str, picks: list | None, preview: list) -> dict:
        def since(entries: dict) -> dict:
            return {key: max(0, self.turn - start) for key, start in entries.items()}

        sides = {
            s: {key: n if key in STACKING else max(0, self.turn - n) for key, n in c.items()}
            for s, c in self.sides.items()
        }
        return {
            "turn": self.turn, "kind": kind, "picks": list(picks) if picks else [None, None, None],
            "opp_preview": list(preview), "anomalies": 0, "own": self.own, "opp": self.opp,
            "weather": since(self.weather), "fields": since(self.fields), "side": sides["own"], "opp_side": sides["opp"],
        }  # fmt: skip


def norm(values: list[float]) -> list[float]:
    total = sum(values)
    return [v / total for v in values] if total > 0 else values


def belief(member: dict, rng: random.Random, brought: float, lead: float | None) -> dict:
    species = member["species"]
    name = SPECIES.get(species, {}).get("name", "")
    stones = [
        to_id(v["requiredItem"]) for v in SPECIES.values() if v.get("requiredItem") and v.get("baseSpecies") == name
    ]
    items = [i for i in dict.fromkeys([*stones, *rng.sample(COMMON_ITEMS, 4)]) if i in ITEMS][:3]
    abilities = [to_id(a) for a in SPECIES.get(species, {}).get("abilities", {}).values()][:3]
    revealed = [m[0] for m in member["moves"]]
    pool = [m for m in LEARNSETS.get(species, []) if m in MOVES and m not in revealed]
    others = rng.sample(pool, min(len(pool), 8 - len(revealed)))
    weights = sorted((rng.uniform(0.02, 0.2) for _ in others), reverse=True)
    moves = [[m, 0.24] for m in revealed] + [[m, round(w, 4)] for m, w in zip(others, weights, strict=True)]
    item_p, ability_p = norm([0.5, 0.25, 0.1][: len(items)]), norm([0.7, 0.2, 0.1][: len(abilities)])
    return {
        "species": species, "brought": round(brought, 4), "lead": lead,
        "item": [[i, round(p, 4)] for i, p in zip(items, item_p, strict=True)],
        "ability": [[a, round(p, 4)] for a, p in zip(abilities, ability_p, strict=True)],
        "moves": moves[:8],
    }  # fmt: skip


def beliefs(view: dict, rng: random.Random, leads: list[float] | None = None) -> list[dict]:
    opp = sorted(view["opp"], key=lambda m: not m["active"])  # their active first
    seen = sum(1 for m in opp if m["revealed"])
    left = max(0.0, (3 - seen) / max(1, 6 - seen))
    out = []
    for i, member in enumerate(opp):
        brought = 1.0 if member["revealed"] else min(1.0, left * rng.uniform(0.6, 1.4))
        out.append(belief(member, rng, brought, round(leads[i], 4) if leads else None))
    return out


def softmax(logits: list[float | None]) -> list[float]:
    top = max(x for x in logits if x is not None)
    raised = [math.exp(x - top) if x is not None else 0.0 for x in logits]
    return [round(x / sum(raised), 4) for x in raised]


def net_of(entry: dict, view: dict, rng: random.Random) -> dict:
    own_active = next((m for m in view["own"] if m["active"]), None)
    picks = [i for i in view["picks"] if i is not None]
    moves = [m[0] for m in (own_active or {}).get("moves", [])][:4] + [None] * 4
    legal = entry["legal"]
    rows = moves[:4] + (moves[:4] if any(4 <= i < 8 for i in legal) else [None] * 4)  # noqa: PLR2004
    rows += [view["own"][i]["species"] for i in picks] + [None] * (3 - len(picks))
    their = next((m for m in view["opp"] if m["active"]), None)
    known = [m[0] for m in (their or {}).get("moves", [])][:4]
    slots = (known + [""] + [None] * 4)[:4]
    others = [m for m in view["opp"] if m is not their][:5]
    cols = slots + slots + [m["species"] for m in others] + [None] * (5 - len(others))
    gone = [False] * 8 + [m["fainted"] for m in others] + [True] * (5 - len(others))  # no switch to a fainted one
    raw = zip(entry["extras"]["p_opp"], cols, gone, strict=True)
    p_opp = norm([p if c is not None and not out else 0.0 for p, c, out in raw])
    p_opp = [round(p, 4) for p in p_opp]
    target = [x if x is not None else 0.0 for x in entry["extras"]["score"]]
    matrix = []
    for a, goal in enumerate(target):
        noise = [rng.uniform(-0.5, 0.5) if rows[a] else 0.0 for _ in cols]
        mean = sum(p * d for p, d in zip(p_opp, noise, strict=True))
        matrix.append([round(goal + d - mean, 4) if rows[a] else 0.0 for d in noise])
    score = [round(sum(p * x for p, x in zip(p_opp, line, strict=True)), 4) for line in matrix]
    logits = entry["scores"]
    pairs = [(score[i] - score[j], logits[i] - logits[j]) for i in legal for j in legal if i < j]
    tau = sum(s * g for s, g in pairs) / max(1e-9, sum(g * g for _, g in pairs)) if pairs else 0.05
    return {
        "decision": entry["decision"], "kind": entry["kind"], "legal": legal, "value": entry["value"],
        "tau": round(max(tau, 0.01), 4), "pi": softmax([logits[i] if i in legal else None for i in range(11)]),
        "score": score, "rows": rows, "cols": cols, "p_opp": p_opp, "matrix": matrix, "belief": beliefs(view, rng),
    }  # fmt: skip


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def neural_log(source: Path, tag: str) -> dict[int, dict]:
    lines = (source / "cg" / f"{tag}.neural.jsonl").read_text(encoding="utf-8").splitlines()
    return {e["decision"]: e for e in map(json.loads, lines) if "decision" in e}


def make_battle(source: Path, index: int, picks: tuple[int, ...]) -> tuple[str, dict]:
    folder, out = source / f"battle-{index}", TARGET / f"battle-{index}"
    shutil.copy(folder / "battle_log.txt", out / "battle_log.txt")
    first = json.loads(next(iter(sorted((folder / "decisions").glob("*.json")))).read_text(encoding="utf-8"))
    tag = first["battle_id"]
    log = neural_log(source, tag)
    for path in sorted((folder / "decisions").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["sequence"] not in picks:
            continue
        tracker = Tracker(record["events"][0])
        for event in record["events"][1:]:
            tracker.take(event)
        seq = record["live"]["decision_request"]["seq"]
        rng = random.Random(f"{tag}:{seq}")
        view = tracker.view(record["notes"]["cg"]["kind"], record["team"]["picks"], record["opponent_preview"])
        record["schema"] = "pcsa-decision/4"
        record["notes"]["cg"].update(view=view, net=net_of(log[seq], view, rng))
        write(out / "decisions" / path.name, record)
    preview = json.loads((folder / "team_preview.json").read_text(encoding="utf-8"))
    start = Tracker(preview["request"]["events"][0])
    view = start.view("preview", None, preview["request"]["events"][0]["opp_preview"])
    rng = random.Random(f"{tag}:preview")
    leads = norm([rng.uniform(0.2, 1.0) for _ in range(6)])
    chosen = preview["decision"]["picks"]
    top = [[chosen, 0.41], [[chosen[1], chosen[0], chosen[2]], 0.17], [[0, 2, 3], 0.08]]
    net = {"decision": -1, "kind": "preview", "value": log[-1]["value"], "belief": beliefs(view, rng, leads)}
    preview["cg"] = {"view": view, "net": {**net, "picks_top": top}}
    write(out / "team_preview.json", preview)
    return tag, log[-1]


def manifest(source: Path) -> dict:
    data = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    cg = data["player_params"]
    return {
        **{k: v for k, v in data.items() if k not in ("player", "player_params", "decision_schema")},
        "decision_schema": "pcsa-decision/4", "player": "neural",
        "player_params": {
            "checkpoint": "assets\\models\\league-20261118-g-iter-00148.pt", "run": "league-20261118-g",
            "update": 148, "sampling": "argmax", "fallback": "tower", "cg": cg,
        },
    }  # fmt: skip


def main(source: Path) -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    for index in PICKS:
        (TARGET / f"battle-{index}").mkdir(parents=True)
    write(TARGET / "manifest.json", manifest(source))
    previews = {index: make_battle(source, index, picks)[1] for index, picks in PICKS.items()}
    lines = []
    for raw in (source / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(raw)
        if event.get("event") == "decision" and event.get("kind") == "team":
            event.update(v=previews[event["battle"]]["value"], ms=previews[event["battle"]]["ms"])
        lines.append(json.dumps(event, ensure_ascii=False))
    (TARGET / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE)
