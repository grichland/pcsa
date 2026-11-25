"""Replay a run's saved CG messages through pcsa.cg.serve and write a copy of the run with `view` and `net`.

    .venv/Scripts/python.exe tools/replay_run.py output/runs/<run> [--checkpoint assets/models/<file>.pt] [--out DIR]

Per battle the messages the controller sent (battle-N/cg-messages.jsonl, `dir: "sent"`; else cg/<tag>.jsonl) go,
one CG generation at a time, to a fresh pcsa.cg.serve Server with the neural player and the checkpoint the run used
(from the manifest, looked up in assets/models). The copy (<run>-replay) gets, per decision record, `notes.cg.view`
and `notes.cg.net` (schema pcsa-decision/4; `net.ms` is the replay's own decision time), `cg` in
team_preview.json, flat `v`/`ms` on the team `decision` events, and a manifest naming what produced the replay
(the original run's player under `replay.original`). Reported: whether every replayed answer equals the recorded one,
and how often the belief's top item, top ability and top-4 moves agree with what the battle revealed by its end.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcsa.cg import startup  # noqa: E402
from pcsa.cg.serve import NEURAL, Server, make_player  # noqa: E402
from pcsa.cg.view import view as view_of  # noqa: E402
from pcsa.common import paths  # noqa: E402
from pcsa.common.config import load_config  # noqa: E402
from pcsa.run.recorder import DECISION_SCHEMA, neural_player  # noqa: E402

LEGACY_SEARCH = {"search_ms": "ms", "search_sims": "sims", "search_seed": "seed"}  # flat-form policy argument names
LEGACY_RULES = {"max_same_move": 4, "max_switches": 3}  # flat-form defaults of the rule masks
# the rule masks added after the first two: off in a run that recorded none of them
LATER_RULES = {"max_boosts": 0, "max_heals": 0, "no_null_moves": False, "no_boost_when_ko": False,
               "no_failing_status": False, "no_redundant_field": False, "no_double_protect": False,
               "no_wasted_self": False, "status_free_mirrors": [], "always_mega": False,
               "one_mega_lineup": False}  # fmt: skip
LIVE_TYPES = {"battle_start", "preview_request", "events", "decision_request", "battle_end"}


def read_jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_checkpoint(run: Path) -> Path:
    """The checkpoint named in the run's CG command (or start-up status), found in assets/models."""
    params = _run_params(run)
    command = params.get("command") or []
    name = (params.get("player_args") or {}).get("checkpoint")
    if not name and "--player-args" in command:
        name = json.loads(command[command.index("--player-args") + 1]).get("checkpoint")
    status = ((params.get("start_status") or {}).get("status") or {}).get("player_status") or {}
    name = name or status.get("checkpoint")
    if not name:
        raise SystemExit("the run's manifest names no checkpoint; give --checkpoint")
    path = REPO / "assets" / "models" / Path(str(name).replace("\\", "/")).name
    if not path.is_file():
        raise SystemExit(f"{path} not found; give --checkpoint")
    return path


def _run_params(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    params = manifest.get("player_params") or {}
    return params.get("cg", params)


def legacy_policy_args(old: dict) -> dict:
    """Policy arguments in the flat form some runs recorded (rule_masks, max_same_move, ...) in the current form."""
    rules = None if old.get("rule_masks") is False else {k: old.get(k, v) for k, v in LEGACY_RULES.items()}
    out = {k: v for k, v in old.items() if k not in {"rule_masks", "search", *LEGACY_RULES, *LEGACY_SEARCH}}
    return {**out, "rules": rules}


def run_player_args(run: Path, checkpoint: Path) -> dict:
    """The neural player's arguments as the run used them (from its manifest; the shipped config when it has none),
    with `checkpoint`, no log folder and no search (the search is bound by time, so it does not replay)."""
    params = _run_params(run)
    command = params.get("command") or []
    if params.get("player_args"):
        args = dict(params["player_args"])
    elif "--player-args" in command:
        args = json.loads(command[command.index("--player-args") + 1])
        args["policy_args"] = legacy_policy_args(args.get("policy_args") or {})
    else:
        args = dict(startup.resolved(load_config(paths.config_dir() / "agent.yaml").agent.cg).player_args or {})
    args.pop("log_dir", None)
    policy = {**args.get("policy_args", {}), "search": None}
    if policy.get("rules") is not None:
        policy["rules"] = {**LATER_RULES, **policy["rules"]}
    return {**args, "checkpoint": str(checkpoint), "policy_args": policy}


def battle_messages(battle: Path, run: Path) -> tuple[list[tuple[int, dict]], list[dict]]:
    """([(CG generation, message sent)], [CG's recorded replies])."""
    saved = battle / "cg-messages.jsonl"
    if saved.is_file():
        rows = read_jsonl(saved)
        sent = [(r.get("cg", 1), r["message"]) for r in rows if r["dir"] == "sent"]
        return sent, [r["message"] for r in rows if r["dir"] == "received" and isinstance(r["message"], dict)]
    tag = json.loads((battle / "battle_id.json").read_text(encoding="utf-8")).get("battle_id")
    rows = read_jsonl(run / "cg" / f"{tag}.jsonl")
    sent = [(1, m) for m in rows if isinstance(m, dict) and m.get("type") in LIVE_TYPES]
    return sent, [m for m in rows if isinstance(m, dict) and ("picks" in m or "order" in m)]


def key(reply: dict):
    return "preview" if "picks" in reply else reply.get("seq")


def replay_battle(sent: list[tuple[int, dict]], args: dict) -> tuple[dict, object, Server | None]:
    """seq ("preview") -> the replayed reply, the Battle after the last events (before battle_end), the Server."""
    replies, final, server, generation = {}, None, None, None
    for gen, message in sent:
        if server is None or gen != generation:
            server, generation = Server(make_player(NEURAL, args)), gen
        if message.get("type") == "battle_end":
            final = server.session.battle
        reply = server.handle(message)
        if isinstance(reply, dict) and "error" not in reply and ("picks" in reply or "order" in reply):
            replies[key(reply)] = reply
    if final is None and server is not None:
        final = server.session.battle
    return replies, final, server


def compare(recorded: list[dict], replies: dict) -> dict:
    """The last recorded answer per request against the replayed one."""
    last = {key(r): r for r in recorded}
    differ = []
    for k, r in last.items():
        mine = replies.get(k)
        field = "picks" if k == "preview" else "order"
        if mine is None or mine.get(field) != r.get(field):
            differ.append({"request": k, "recorded": r.get(field), "replayed": None if mine is None else mine.get(field)})
    return {"answers": len(last), "equal": len(last) - len(differ), "differ": differ}


class Belief:
    """Agreement of the belief with what the battle revealed by its end, over every decision: `hidden` counts the
    decisions where that fact was not yet revealed, `known` those where it already was."""

    def __init__(self) -> None:
        whens = ("all", "hidden", "known")
        self.counts = {f"{what}_{when}": [0, 0] for what in ("item", "ability", "moves4") for when in whens}

    def add(self, what: str, hit: bool, hidden: bool) -> None:
        for when in ("all", "hidden" if hidden else "known"):
            c = self.counts[f"{what}_{when}"]
            c[0] += int(hit)
            c[1] += 1

    def decision(self, net: dict, now: dict, final: dict) -> None:
        """`now`: view.opp at the decision; `final`: view.opp at the end, both by species. Left out: an item shown
        as "" (used up or removed: which item is not said) and the ability of a Mega (the Battle shows the Mega's
        ability, the belief is about the set's)."""
        for entry in net.get("belief") or []:
            truth, seen = final.get(entry.get("species")), now.get(entry.get("species"))
            if truth is None or seen is None:
                continue
            if truth["item"]:
                self.add("item", bool(entry["item"]) and entry["item"][0][0] == truth["item"], seen["item"] is None)
            if truth["ability"] and truth["form"] in (None, truth["species"]):
                self.add("ability", bool(entry["ability"]) and entry["ability"][0][0] == truth["ability"],
                         seen["ability"] is None)  # fmt: skip
            top4 = {m for m, _ in entry["moves"][:4]}
            known = {m for m, _ in seen["moves"]}
            for move, _ in truth["moves"]:
                self.add("moves4", move in top4, move not in known)

    def report(self) -> dict:
        return {k: {"agree": a, "cases": n, "percent": round(100.0 * a / n, 1) if n else None}
                for k, (a, n) in self.counts.items()}  # fmt: skip


def additions(reply: dict) -> dict:
    """The replayed reply's `view` and `net`; `net.ms` is the replay's own decision time (the record's
    `notes.cg.model.ms` stays the live one)."""
    notes = reply.get("notes") or {}
    out = {k: notes[k] for k in ("view", "net") if k in notes}
    ms = (notes.get("model") or {}).get("ms")
    if "net" in out and ms is not None:
        out["net"] = {**out["net"], "ms": ms}
    return out


def rewrite_battle(out: Path, replies: dict, recorded: list[dict]) -> dict:
    """The battle's decision records and team_preview.json in the copy; returns counts."""
    done = {"records": 0, "with_view": 0, "with_net": 0}
    for path in sorted((out / "decisions").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["schema"] = DECISION_SCHEMA
        cg, reply = (rec.get("notes") or {}).get("cg"), replies.get(rec.get("request_id"))
        if isinstance(cg, dict) and reply is not None:
            cg.update(additions(reply))
        done["records"] += 1
        done["with_view"] += int(isinstance(cg, dict) and "view" in cg)
        done["with_net"] += int(isinstance(cg, dict) and "net" in cg)
        write_json(path, rec)
    preview, reply = out / "team_preview.json", replies.get("preview")
    if preview.is_file() and reply is not None:
        tp = json.loads(preview.read_text(encoding="utf-8"))
        original = next((r for r in recorded if "picks" in r), None)
        tp["cg"] = {**reply["notes"], **additions(reply), "model": (original or reply).get("notes", {}).get("model")}
        write_json(preview, tp)
    return done


def team_events(out: Path, models: dict) -> None:
    """Flat `v` and `ms` on each battle's team `decision` event (the recorded preview's notes.model)."""
    path = out / "events.jsonl"
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(raw) if raw.strip() else None
        model = models.get(e.get("battle")) if isinstance(e, dict) and e.get("event") == "decision" else None
        if isinstance(e, dict) and e.get("kind") == "team" and model:
            e.update({k: model[f] for k, f in (("v", "value"), ("ms", "ms")) if model.get(f) is not None})
            raw = json.dumps(e, ensure_ascii=False)
        lines.append(raw)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def manifest(out: Path, run: Path, args: dict, status: dict | None) -> None:
    """The copy's manifest describes what produced the replay (pcsa.cg.serve, this policy and checkpoint) in the
    flat neural form; the original run's player and player_params are kept under `replay.original`."""
    path = out / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    original = {"player": m.get("player"), "player_params": m.get("player_params"), "config": m.get("config")}
    params = {
        "transport": "inprocess",
        "entry": "pcsa.cg.serve:Server",
        "player": NEURAL,
        "player_args": args,
        "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "start_status": {"status": status, "problems": []},
    }
    m["player"], m["player_params"] = neural_player("cg", params)
    m["decision_schema"] = DECISION_SCHEMA
    m["config"] = None
    m["replay"] = {"of": run.name, "checkpoint": Path(args["checkpoint"]).name, "tool": "tools/replay_run.py",
                   "original": original}  # fmt: skip
    write_json(path, m)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    run = a.run.resolve()
    checkpoint = (a.checkpoint or run_checkpoint(run)).resolve()
    out = (a.out or run.with_name(run.name + "-replay")).resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(run, out)
    args = run_player_args(run, checkpoint)
    report: dict = {"run": run.name, "checkpoint": checkpoint.name, "battles": {}}
    belief, models, status = Belief(), {}, None
    for battle in sorted(run.glob("battle-*"), key=lambda p: int(p.name.split("-")[1])):
        sent, recorded = battle_messages(battle, run)
        replies, final, server = replay_battle(sent, args)
        n = int(battle.name.split("-")[1])
        models[n] = next((r.get("notes", {}).get("model") for r in recorded if "picks" in r), None) or {}
        written = rewrite_battle(out / battle.name, replies, recorded)
        report["battles"][battle.name] = {**compare(recorded, replies), **written}
        end = {m["species"]: m for m in view_of(final)["opp"]} if final is not None else {}
        for reply in replies.values():
            notes = reply["notes"]
            if "net" in notes and "view" in notes:
                belief.decision(notes["net"], {m["species"]: m for m in notes["view"]["opp"]}, end)
        if status is None and server is not None:
            status = server.session.status()
    team_events(out, models)
    manifest(out, run, args, status)
    report["belief"] = belief.report()
    write_json(out / "replay-report.json", report)
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    equal = all(b["differ"] == [] for b in report["battles"].values())
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
