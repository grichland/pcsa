"""The neural player: settings, checkpoint lookup, every fallback (no checkpoint, another network's checkpoint, an
error, a late answer, no time left), and (torch) synthetic checkpoints playing preview and decisions."""

from __future__ import annotations

import random
import threading
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config, load_config
from pcsa.common.errors import ConfigError
from pcsa.players import mirror
from pcsa.players.base import DecisionView, Degrade, ScreenFacts, TeamPreviewView
from pcsa.players.factory import LIVE_KINDS, build_player
from pcsa.players.max_damage import MaxDamagePlayer
from pcsa.players.neural import NETWORKS_ENV, NeuralParams, NeuralPlayer, resolve_checkpoint
from tests.conftest import REPO
from tests.players.test_disciplined import battle as decision_battle
from tests.rl_training.battles import preview_battle

REAL = Path(r"C:\Users\geral\Code\research\pcsa-rl\runs\v2-scratch\ckpt\latest.pt")


def view_of(b) -> DecisionView:
    return DecisionView(mirror.legal_actions(b), ScreenFacts(), Degrade.FULL, random.Random(0), b)


def preview_view() -> TeamPreviewView:
    b = preview_battle()
    return TeamPreviewView(b, tuple(m.name for m in b.team.values()), random.Random(0))


def soon(seconds: float = 30.0) -> Deadline:
    return Deadline.after(FakeClock(), seconds)


def test_neural_is_a_live_kind_and_reads_its_file(config: Config) -> None:
    assert "neural" in LIVE_KINDS
    params = NeuralParams.load(config.directory / config.agent.policy.neural)
    assert (params.sampling, params.preview, params.fallback, params.device) == (
        "greedy",
        "critic",
        "disciplined",
        "cpu",
    )


def test_bad_settings_are_rejected(tmp_path: Path) -> None:
    for text in ("sampling: argmax\n", "preview: random\n", "size: XL\n", "fallback: neural\n", "colour: red\n"):
        path = tmp_path / "neural.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            NeuralParams.load(path)


def test_relative_checkpoints_look_in_the_networks_folder_first(tmp_path: Path) -> None:
    (tmp_path / "s1.pt").write_bytes(b"x")
    assert resolve_checkpoint("s1.pt", {NETWORKS_ENV: str(tmp_path)}) == tmp_path / "s1.pt"
    assert resolve_checkpoint("other.pt", {NETWORKS_ENV: str(tmp_path)}).parent.name != tmp_path.name
    assert resolve_checkpoint(str(tmp_path / "s1.pt"), {}) == tmp_path / "s1.pt"


def test_missing_checkpoint_plays_the_fallback(tmp_path: Path) -> None:
    missing = (tmp_path / "nothing.pt").as_posix()
    config = load_config(REPO / "config" / "agent.yaml", [f"policy.neural_checkpoint={missing}"], environ={})
    player = build_player(config, "neural")
    assert isinstance(player, NeuralPlayer) and player.error and "no checkpoint" in player.error
    view = view_of(decision_battle("Azumarill"))
    chosen = player.choose_action(view, soon())
    notes = player.notes()
    assert chosen in view.legal
    assert (
        notes["rule"] == "neural:fallback" and "no checkpoint" in notes["why"] and notes["fallback"] == "disciplined"
    )
    team = player.choose_team(preview_view(), soon())
    assert len(set(team)) == 3 and player.notes()["rule"] == "neural:fallback"


class FakePolicy:
    """propose() waits for `release` (a slow network) or raises `error`; commit() returns the first legal action."""

    def __init__(self, error: Exception | None = None, wait: bool = False) -> None:
        self.error, self.release = error, threading.Event()
        if not wait:
            self.release.set()

    def propose(self, view: DecisionView):
        self.release.wait(5.0)
        if self.error is not None:
            raise self.error
        return view.legal[-1]

    def commit(self, proposal):
        return proposal

    def notes(self) -> dict:
        return {"rule": "rl"}


def fake_player(tmp_path: Path, policy: FakePolicy, **settings) -> NeuralPlayer:
    path = tmp_path / "fake.pt"
    path.write_bytes(b"x")
    params = replace(NeuralParams(), **settings)
    return NeuralPlayer(params, MaxDamagePlayer(), path, loader=lambda p, c: (policy, {"d": 128}))


def test_a_network_error_plays_the_fallback(tmp_path: Path) -> None:
    player = fake_player(tmp_path, FakePolicy(error=RuntimeError("boom")))
    view = view_of(decision_battle("Azumarill"))
    player.choose_action(view, soon())
    notes = player.notes()
    assert notes["rule"] == "neural:fallback" and "RuntimeError: boom" in notes["why"]
    assert notes["fallback"] == "max_damage"


def test_a_late_answer_is_dropped_and_the_fallback_decides(tmp_path: Path) -> None:
    policy = FakePolicy(wait=True)
    player = fake_player(tmp_path, policy, max_decision_s=0.05)
    view = view_of(decision_battle("Azumarill"))
    expected = MaxDamagePlayer().choose_action(view, soon())
    assert player.choose_action(view, soon()) == expected
    assert "no answer within" in player.notes()["why"]
    assert "still running" in (player.choose_action(view, soon()) and player.notes()["why"])
    policy.release.set()
    for _ in range(100):  # the late thread finishes and frees the network
        if player._lock.acquire(timeout=0.05):
            player._lock.release()
            break
    assert player.choose_action(view, soon()) == view.legal[-1] and player.notes()["rule"] == "neural"


def test_no_time_left_plays_the_fallback_without_asking_the_network(tmp_path: Path) -> None:
    policy = FakePolicy(error=AssertionError("must not be called"))
    player = fake_player(tmp_path, policy)
    player.choose_action(view_of(decision_battle("Azumarill")), soon(0.3))  # under reserve_s = 0.5
    assert player.notes()["why"].startswith("deadline")


def test_a_screen_level_decision_plays_the_fallback(tmp_path: Path) -> None:
    player = fake_player(tmp_path, FakePolicy())
    legal = view_of(decision_battle("Azumarill")).legal
    player.choose_action(DecisionView(legal, ScreenFacts(), Degrade.SCREEN, random.Random(0), None), soon())
    assert "no battle" in player.notes()["why"]


# torch ---------------------------------------------------------------------------------------------------
def synthetic_checkpoint(tmp_path: Path, d: int = 64, *, version: str | None = "v2") -> Path:
    import torch

    from pcsa.rl_training.former import Config, Net

    torch.manual_seed(0)
    cfg = Config(d=d, layers=2)
    state = {"net": Net(cfg).state_dict(), "update": 7, "config": {"name": f"test-{d}"}, "model": asdict(cfg)}
    if version:
        state["version"] = version
    path = tmp_path / f"{d}-{version}.pt"
    torch.save(state, path)
    return path


def neural(path: Path, **settings) -> NeuralPlayer:
    return NeuralPlayer(replace(NeuralParams(seed=0), **settings), MaxDamagePlayer(), path)


@pytest.mark.torch
@pytest.mark.parametrize("d", [64, 96])
def test_synthetic_checkpoints_load_and_play(tmp_path: Path, d: int) -> None:
    player = neural(synthetic_checkpoint(tmp_path, d))
    want = {"update": 7, "run": f"test-{d}", "d": d, "layers": 2, "columns": 9, "memory": 64}
    assert player.error is None and player.facts == want
    team = player.choose_team(preview_view(), soon())
    assert len(team) == 3 and len(set(team)) == 3 and all(0 <= i < 6 for i in team)
    assert player.notes()["rule"] == "neural"
    view = view_of(decision_battle("Azumarill"))
    assert player.choose_action(view, soon()) in view.legal
    notes = player.notes()
    assert notes["rule"] == "neural" and 0 < notes["p"] <= 0.99 and notes["ms"] >= 0
    assert isinstance(notes["v"], float) and -2.0 < notes["v"] < 2.0  # the critic's value, kept for the dashboard
    matrix = notes["matrix"]  # the whole pair matrix, for the run records and the dashboard
    assert len(matrix["cells"]) == 11 and len(matrix["cols"]) == 9 and matrix["legal"][matrix["chosen"]]


@pytest.mark.torch
def test_a_checkpoint_of_another_network_plays_the_fallback(tmp_path: Path) -> None:
    player = neural(synthetic_checkpoint(tmp_path, version=None))
    assert player.error and "poke-singles-former" in player.error
    player.choose_action(view_of(decision_battle("Azumarill")), soon())
    assert "poke-singles-former" in player.notes()["why"]


@pytest.mark.torch
def test_greedy_takes_the_most_likely_action_and_sample_draws(tmp_path: Path) -> None:
    import numpy as np

    path = synthetic_checkpoint(tmp_path)
    greedy, sampled = neural(path), neural(path, sampling="sample")
    b = decision_battle("Slowbro")
    view = view_of(b)
    proposal = greedy.policy.propose(view)
    probs = proposal.probs[0].numpy()
    assert proposal.index == int(np.argmax(probs)) and probs.max() <= 0.99 + 1e-6
    assert {greedy.choose_action(view_of(decision_battle("Slowbro")), soon()).message for _ in range(5)} == {
        proposal.chosen.message
    }
    draws = {sampled.policy.propose(view).index for _ in range(200)}
    assert len(draws) > 1 and draws <= set(np.flatnonzero(probs > 0))


@pytest.mark.torch
def test_greedy_preview_takes_the_best_value(tmp_path: Path) -> None:
    import numpy as np

    from pcsa.rl_training import preview

    player = neural(synthetic_checkpoint(tmp_path), preview="greedy")
    view = preview_view()
    values = player.policy.preview_values(view.battle, preview.all_picks())
    assert player.choose_team(view, soon()) == list(preview.all_picks()[int(np.argmax(values))])


@pytest.mark.torch
def test_a_same_turn_retry_replaces_the_last_decision(tmp_path: Path) -> None:
    player = neural(synthetic_checkpoint(tmp_path))
    player.choose_team(preview_view(), soon())
    first = player.choose_action(view_of(decision_battle("Azumarill")), soon())
    assert player.policy.game.cache["t"] == 1
    again = player.choose_action(view_of(decision_battle("Azumarill")), soon())
    assert again.message == first.message and player.policy.game.cache["t"] == 1 and player.notes().get("retry")
    later = [["|move|p1a: Azumarill|Aqua Jet|p2a: Dragonite"]]
    player.choose_action(view_of(decision_battle("Azumarill", later)), soon())
    assert player.policy.game.cache["t"] == 2


@pytest.mark.torch
def test_a_late_answer_leaves_no_trace_in_the_networks_memory(tmp_path: Path) -> None:
    player = neural(synthetic_checkpoint(tmp_path))
    player.choose_action(view_of(decision_battle("Azumarill")), soon())
    proposal = player.policy.propose(
        view_of(decision_battle("Azumarill", [["|move|p1a: Azumarill|Aqua Jet|p2a: Dragonite"]]))
    )
    assert proposal.cache["t"] == 2 and player.policy.game.cache["t"] == 1  # proposed, never committed


@pytest.mark.torch
def test_restore_rebuilds_the_memory_of_a_resumed_battle(tmp_path: Path) -> None:
    path = synthetic_checkpoint(tmp_path)
    later = [["|move|p1a: Azumarill|Aqua Jet|p2a: Dragonite"]]
    played = neural(path)
    played.choose_action(view_of(decision_battle("Azumarill")), soon())
    want = played.choose_action(view_of(decision_battle("Azumarill", later)), soon())
    resumed = neural(path)
    assert resumed.restore([view_of(decision_battle("Azumarill"))]) == 1
    got = resumed.choose_action(view_of(decision_battle("Azumarill", later)), soon())
    assert got.message == want.message and resumed.notes()["decisions"] == 2
    broken = NeuralPlayer(NeuralParams(), MaxDamagePlayer(), tmp_path / "absent.pt")
    assert broken.restore([view_of(decision_battle("Azumarill"))]) == 0


@pytest.mark.torch
def test_the_research_checkpoint_loads() -> None:
    if not REAL.is_file():
        pytest.skip(f"no research checkpoint at {REAL}")
    player = neural(REAL)
    assert player.error is None and player.facts["d"] == 128 and player.facts["columns"] == 9
    view = view_of(decision_battle("Charizard"))
    assert player.choose_action(view, soon()) in view.legal and player.notes()["rule"] == "neural"


@pytest.mark.torch
def test_override_rules_replace_a_full_hp_heal_with_the_networks_next_choice(tmp_path: Path, monkeypatch) -> None:
    path = synthetic_checkpoint(tmp_path)
    player = NeuralPlayer(replace(NeuralParams(seed=0), rules=["no_full_heal"]), MaxDamagePlayer(), path)
    b = decision_battle("Slowbro")  # full HP, Slack Off legal
    proposal = player.policy.propose(view_of(b))
    slackoff = next(a for a in view_of(b).legal if a.message == "/choose move slackoff")
    monkeypatch.setattr(player.policy, "propose", lambda view: replace(proposal, chosen=slackoff))
    chosen = player.choose_action(view_of(b), soon())
    notes = player.notes()
    assert chosen.message != "/choose move slackoff" and notes["override"] == "no_full_heal"
    assert notes["network"] == "/choose move slackoff" and notes["rule"] == "neural"
