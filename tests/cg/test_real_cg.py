"""T7 battles against CG's own entry point (champions_gym.live, player tower), in process and as a subprocess: every
request answered by CG, no error reply, the `end` event and battle_end in CG's own log; restarts of the controller
and of CG (the new CG gets the requests replayed with what was played and answers the same); our runner bringing
other picks than CG's (the preview fallback, and CG's switches are to what was brought). In process the Session is
watched (real_session.Watched): no anomalies and no corrections of the picks.

The phone is scripted from tower's own preview answer, so a change in tower's picks does not break the tests.
Skipped when champions_gym (the installed wheel) is not importable."""

from __future__ import annotations

import importlib
import json
import sys
from functools import cache
from pathlib import Path

import pytest

from pcsa.cg.adapter import CGPlayer
from pcsa.cg.transport import TransportFailed
from pcsa.common.clock import FakeClock
from pcsa.common.config import CGConfig, Config, load_config
from pcsa.common.errors import Recoverable
from pcsa.common.paths import repo_root
from pcsa.common.types import MoveInfo
from pcsa.live.turn import TurnPipeline
from tests.cg.helpers import START as BATTLE_START
from tests.live.t7_rig import action, end_of_battle, forced, lobby_to_team_select, peeked_action, run_battles
from tests.live.turn_rig import TEAM


def _importable() -> bool:
    try:
        importlib.import_module("champions_gym.live")
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _importable(), reason="champions_gym is not installed")

TRANSPORTS = ["inprocess", "subprocess"]
TIMEOUT_S = 6.0
WATCHED = "tests.cg.real_session:Watched"


@cache
def tower_picks() -> tuple[int, ...]:
    """Tower's preview answer for this team and preview (the controller sends the same battle_start)."""
    from champions_gym.live import Session

    session = Session(player="tower")
    session.handle(BATTLE_START)
    preview = {"v": BATTLE_START["v"], "battle": BATTLE_START["battle"], "type": "preview_request"}
    return tuple(session.handle(preview)["picks"])


def other_picks() -> tuple[int, ...]:
    """Three that tower did not pick."""
    return tuple(i for i in range(6) if i not in tower_picks())[:3]


class Script:
    """The phone's battle for these picks: the lead hits, then faints on turn 2; the second comes in and mega
    evolves when it can."""

    def __init__(self, picks: tuple[int, ...]) -> None:
        self.picks = picks
        self.lead, self.second, self.third = (TEAM.members[i] for i in picks)
        lead = self.lead.name
        start = ["Kai sent out Glimmora!", f"Go! {lead}!"]
        turn1 = [f"Move: {lead} used {self.lead.moves[0]}!", "HP Lost: Glimmora’s HP: 100% → 30%",
                 "Move: The opposing Glimmora used Earth Power!", f"HP Lost: {lead}’s HP: 180 → 70"]  # fmt: skip
        turn2 = ["Move: The opposing Glimmora used Earth Power!", f"HP Lost: {lead}’s HP: 70 → 0",
                 f"Fainted: {lead} fainted!", f"Go! {self.second.name}!"]  # fmt: skip
        self.visits = [{1: start}, {1: start + turn1}, {1: start + turn1, 2: turn2}]

    def steps(self) -> list:
        moves = tuple(MoveInfo(k, n, (16, 16)) for k, n in enumerate(self.second.moves))
        mega = (self.second.item or "").replace(" Z", "").endswith("ite")
        party = ((self.lead.name, True), (self.second.name, False), (self.third.name, False))
        return (lobby_to_team_select() + action(self.lead.name) + action(self.lead.name) + forced(*party)
                + peeked_action(self.second.name, moves, mega=mega) + end_of_battle("win"))  # fmt: skip

    def brought(self) -> set[str]:
        return {m.name.lower() for m in (self.lead, self.second, self.third)}


class Rig:
    def __init__(self, transport: str, tmp_path: Path, picks: tuple[int, ...] | None = None) -> None:
        from tests.cg.real_session import Watched

        Watched.records.clear()
        config = load_config(repo_root() / "config" / "agent.yaml", environ={})
        self.tmp, self.cg_logs, self.clock = tmp_path, tmp_path / "cg-live", FakeClock(1000.0)
        self.transport, self.script = transport, Script(picks or tower_picks())
        command = [sys.executable, "-m", "champions_gym.live", "--player", "tower", "--log-dir", str(self.cg_logs)]
        cg = CGConfig(transport=transport, entry=WATCHED,  # type: ignore[arg-type]
                      args={"player": "tower", "log_dir": str(self.cg_logs)} if transport == "inprocess" else {},
                      command=command, timeout_s=TIMEOUT_S)  # fmt: skip
        self.config: Config = config.model_copy(update={"agent": config.agent.model_copy(update={"cg": cg})})
        self.player = CGPlayer(self.config, clock=self.clock, run_dir=tmp_path / "run")

    def play(self, monkeypatch, **kw):
        try:
            return run_battles(self.config, self.script.steps(), self.tmp, monkeypatch, player=self.player,
                               clock=self.clock, visits=kw.pop("visits", self.script.visits), **kw)  # type: ignore[arg-type]  # fmt: skip
        finally:
            self.player.close()

    def cg_log(self, tag: str) -> list[dict]:
        """CG's own record: every message it got and every reply it gave."""
        path = self.cg_logs / f"{tag}.jsonl"
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]

    def sessions(self) -> list:
        from tests.cg.real_session import Watched

        return list(Watched.records) if self.transport == "inprocess" else []


def lines(played, direction: str, cg: int | None = None) -> list[dict]:
    path = played.recorder.dir / "battle-1" / "cg-messages.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return [r["message"] for r in rows if r["dir"] == direction and (cg is None or r["cg"] == cg)]


def assert_answered_by_cg(rig: Rig, played, decisions: int) -> None:
    assert played.ending == "completed" and played.state.outcomes == ["win"]
    errors = [r for r in lines(played, "received") if "error" in r]
    assert errors == [], f"CG error replies: {errors}"
    records = played.records()
    assert [(r["degrade"], r["reason"]) for r in records] == [("player", "player answer")] * decisions
    assert all(r["notes"]["cg"]["player"] in ("tower", "played") for r in records)
    assert played.json("battle-1/team_preview.json")["decision"]["source"] == "player"
    tag, _, events = played.ends[0]
    assert events[-1] == {"k": "end", "result": "win"}
    sent = lines(played, "sent")
    assert [m["type"] for m in sent[-2:]] == ["events", "battle_end"] and sent[-2]["events"][-1] == events[-1]
    got = rig.cg_log(tag)
    assert {"k": "end", "result": "win"} in [e for m in got if m.get("type") == "events" for e in m["events"]]
    assert got[-1] == sent[-1]  # battle_end, the last thing CG saw
    assert rig.sessions() or rig.transport != "inprocess"
    for session in rig.sessions():
        assert session["anomalies"] == [], f"CG anomalies: {session['anomalies']}"
        picks = [c for c in session["corrections"] if "picks" in c["what"]]  # rule 31 may put our choice right
        assert picks == [], f"CG corrections of the picks: {picks}"
        for seen in session["seen"]:  # every switch CG offered is to one brought
            assert {o["mon"] for _, o in seen["legal"] or [] if o["kind"] == "switch"} <= rig.script.brought()


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_real_cg_a_full_battle(transport, tmp_path: Path, monkeypatch) -> None:
    rig = Rig(transport, tmp_path)
    played = rig.play(monkeypatch)
    assert_answered_by_cg(rig, played, 4)
    assert tuple(lines(played, "received")[0]["picks"]) == rig.script.picks  # the phone was scripted for these
    others = {f"/choose switch {m.name}" for m in (rig.script.second, rig.script.third)}
    assert [r["order"] for r in played.records()][2] in others


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_real_cg_controller_restarts_with_cg_kept(transport, tmp_path: Path, monkeypatch) -> None:
    """The session restarts after CG answered turn 2 and before the replacement is asked: CG is asked nothing
    twice and keeps the battle."""
    real_save, saves = TurnPipeline._save, []

    def save(self, *args, **kw):
        saves.append(1)
        if len(saves) == 2:
            raise Recoverable("restart after the answer")
        return real_save(self, *args, **kw)

    monkeypatch.setattr(TurnPipeline, "_save", save)
    rig = Rig(transport, tmp_path)
    v = rig.script.visits
    played = rig.play(monkeypatch, visits=[v[0], v[1], v[1], v[2]], faults={("decide", 4)})
    assert played.state.restarts == 2
    assert_answered_by_cg(rig, played, 4)
    assert [m["type"] for m in lines(played, "sent")].count("battle_start") == 1


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_real_cg_restarted_mid_battle(transport, tmp_path: Path, monkeypatch) -> None:
    """CG is lost at the replacement: a new CG (a new Session) gets the whole battle again, the picks in
    battle_start and the earlier requests replayed with what was played, and answers them as the first did."""
    rig = Rig(transport, tmp_path)
    real, calls = rig.player.transport.exchange, []

    def exchange(messages, *args):
        calls.append(1)
        if any(m.get("seq") == 2 for m in messages) and not getattr(exchange, "failed", False):
            exchange.failed = True  # type: ignore[attr-defined]
            raise TransportFailed("lost")
        return real(messages, *args)

    rig.player.transport.exchange = exchange  # type: ignore[method-assign]
    played = rig.play(monkeypatch)
    assert rig.player.restarts == 1 and played.records()[2]["notes"]["attempt"] == 2
    assert_answered_by_cg(rig, played, 4)
    assert len(rig.sessions()) == (2 if transport == "inprocess" else 0)
    second = lines(played, "sent", cg=2)
    assert [m["type"] for m in second].count("battle_start") == 1
    assert second[0]["picks"] == list(rig.script.picks) and second[1]["replay"] is True
    first_answers = {r["seq"]: r["order"] for r in lines(played, "received", cg=1) if "seq" in r}
    replayed = [m for m in second if m.get("replay")][1:]
    assert [m["seq"] for m in replayed] == [0, 1]
    assert all(m["played"] == first_answers[m["seq"]] for m in replayed)
    answers = lines(played, "received", cg=2)
    assert answers[0] == {"picks": list(rig.script.picks), "notes": answers[0]["notes"]}
    assert answers[0]["notes"].get("given") is True
    again = {r["seq"]: r for r in answers if r.get("seq") in (0, 1)}
    assert {s: r["order"] for s, r in again.items()} == {s: first_answers[s] for s in (0, 1)}
    assert all(r["notes"].get("replayed") and r["notes"].get("player") == "played" for r in again.values())


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_real_cg_overruled_picks(transport, tmp_path: Path, monkeypatch) -> None:
    """Our runner brings three tower did not pick: CG gets the preview fallback on the first events message and
    offers only switches to the three brought."""
    brought = other_picks()
    rig = Rig(transport, tmp_path, picks=brought)
    real = rig.player.teampreview

    def teampreview(request):
        picks, notes = real(request)
        assert tuple(picks) == tower_picks()
        return brought, notes

    rig.player.teampreview = teampreview  # type: ignore[method-assign]
    played = rig.play(monkeypatch)
    assert_answered_by_cg(rig, played, 4)
    sent = lines(played, "sent")
    first_events = next(m for m in sent if m["type"] == "events")
    assert first_events["fallback"] == {"picks": list(brought), "reason": "overruled"}
    assert played.records()[0]["notes"]["picks_overruled"] == {"cg": list(tower_picks()), "brought": list(brought)}
    replacement = played.records()[2]["order"]
    assert replacement in {f"/choose switch {m.name}" for m in (rig.script.second, rig.script.third)}


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_real_cg_a_pokemon_outside_the_preview_enters(transport, tmp_path: Path, monkeypatch) -> None:
    """The preview misread one of their six: when it enters, battle_start is corrected and CG gets the battle
    again from the start in the same Session, so it holds the Pokémon on the field."""
    rig = Rig(transport, tmp_path)
    start, turn1, turn2 = rig.script.visits[2][1][:2], rig.script.visits[2][1][2:], rig.script.visits[2][2]
    switch = ["Kai withdrew Glimmora!", "Kai sent out Skeledirge!"]
    turn2 = [switch[0], switch[1], *(row.replace("Glimmora", "Skeledirge") for row in turn2)]
    played = rig.play(monkeypatch, visits=[{1: start}, {1: start + turn1}, {1: start + turn1, 2: turn2}])
    assert played.ending == "completed" and rig.player.restarts == 0
    starts = [m for m in lines(played, "sent") if m["type"] == "battle_start"]
    assert len(starts) == 2 and "skeledirge" not in starts[0]["opp_preview"]
    assert starts[1]["opp_preview"].count("skeledirge") == 1 and "glimmora" in starts[1]["opp_preview"]
    assert [r for r in lines(played, "received") if "error" in r] == []
    corrected = [r for r in played.records() if r["notes"].get("preview_corrected")]
    assert len(corrected) == 1 and corrected[0]["degrade"] == "player"
    for session in rig.sessions():
        assert not [a for a in session["anomalies"] if "skeledirge" in str(a)]
