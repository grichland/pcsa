"""The neural player's feature mode: taken from the checkpoint (missing = "base"), an override refused on a base
network, the belief kept per battle and reported in the notes, the live belief settings, the stage-3 hook (off),
and (when recorded live decisions are present) a dry run of both modes on them."""

from __future__ import annotations

import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from pcsa.common.errors import ConfigError
from pcsa.players.max_damage import MaxDamagePlayer
from pcsa.players.neural import NeuralParams, NeuralPlayer, feature_mode
from tests.conftest import REPO
from tests.rl_training.test_neural import decision_battle, preview_view, soon, view_of

GOLD = REPO / "assets" / "models" / "poke-transformer-gold940-20261112-1319.pt"
LIVE_RUNS = Path(os.environ.get("PCSA_LIVE_RUNS", r"C:\Users\geral\Code\pcsa\output\runs"))


def test_the_setting_follows_the_checkpoint_and_a_base_network_refuses_belief_columns() -> None:
    assert NeuralParams().features == "checkpoint" and NeuralParams().belief_crit_lines is False
    params = NeuralParams.load(REPO / "config" / "players" / "neural.yaml")
    assert params.features == "checkpoint" and params.belief_crit_lines is False
    assert feature_mode("checkpoint", "base") == "base"
    assert feature_mode("checkpoint", "belief+mega") == "belief+mega"
    assert feature_mode("base", "base") == "base"
    assert feature_mode("belief", "belief+mega") == "belief"  # between belief modes: allowed, logged
    for asked in ("belief", "belief+mega"):
        with pytest.raises(ConfigError, match="trained with feature mode base"):
            feature_mode(asked, "base")
    with pytest.raises(ConfigError, match="features must be one of"):
        replace(NeuralParams(), features="beleif").check("test")


def test_checkpoint_features() -> None:
    pytest.importorskip("torch")
    from pcsa.rl_training.former_player import checkpoint_features

    assert checkpoint_features({}) == "base"
    assert checkpoint_features({"config": {"name": "v2-gold"}}) == "base"
    assert checkpoint_features({"config": {"features": "belief+mega"}}) == "belief+mega"
    with pytest.raises(ValueError, match="unknown feature mode"):
        checkpoint_features({"config": {"features": "telepathy"}})


def checkpoint(tmp_path: Path, features: str | None) -> Path:
    import torch

    from pcsa.rl_training.former import Config, Net

    torch.manual_seed(0)
    cfg = Config(d=64, layers=2)
    config = {"name": "test"} | ({"features": features} if features else {})
    state = {"net": Net(cfg).state_dict(), "update": 3, "config": config, "model": asdict(cfg), "version": "v2"}
    path = tmp_path / f"{features or 'base'}.pt"
    torch.save(state, path)
    return path


@pytest.mark.torch
def test_a_base_checkpoint_plays_base_and_refuses_an_override(tmp_path: Path) -> None:
    path = checkpoint(tmp_path, None)
    player = NeuralPlayer(NeuralParams(seed=0), MaxDamagePlayer(), path)
    assert player.error is None and player.policy.features == "base" and player.params()["features_used"] == "base"
    player.choose_team(preview_view(), soon())
    player.choose_action(view_of(decision_battle("Azumarill")), soon())
    notes = player.notes()
    assert notes["rule"] == "neural" and "belief" not in notes and player.policy.game.belief is None
    with pytest.raises(ConfigError, match="trained with feature mode base"):
        NeuralPlayer(replace(NeuralParams(seed=0), features="belief+mega"), MaxDamagePlayer(), path)


@pytest.mark.torch
def test_a_belief_checkpoint_keeps_one_belief_per_battle(tmp_path: Path) -> None:
    player = NeuralPlayer(NeuralParams(seed=0), MaxDamagePlayer(), checkpoint(tmp_path, "belief+mega"))
    assert player.error is None and player.policy.features == "belief+mega"
    assert player.policy.belief_options == {"crit_reported": False, "empty_unknown": True}  # live settings
    player.choose_team(preview_view(), soon())
    assert player.notes()["rule"] == "neural"
    first = player.policy.game.belief
    assert first is not None and not first.tracker.crit_reported
    battle = decision_battle("Azumarill")
    view = view_of(battle)
    assert player.choose_action(view, soon()) in view.legal
    notes = player.notes()
    assert notes["rule"] == "neural" and notes["belief"]["updates"] == 1 and notes["belief"]["rebuilds"] == 0
    assert player.policy.game.belief is first  # the preview's belief goes on into the battle
    player.choose_action(view, soon())  # the same decision again (a re-read screen): the same belief, read again
    assert player.notes()["belief"]["updates"] == 2 and player.policy.game.belief is first
    player.choose_team(preview_view(), soon())  # a new battle, a new belief
    assert player.policy.game.belief is not first


@pytest.mark.torch
def test_the_stage3_hook_is_off_and_adds_to_the_logits(tmp_path: Path) -> None:
    import torch

    player = NeuralPlayer(NeuralParams(seed=0), MaxDamagePlayer(), checkpoint(tmp_path, None))
    policy = player.policy
    assert not policy.matrix_blend and policy.matrix_values is None
    view = view_of(decision_battle("Azumarill"))
    plain = policy.propose(view)
    seen = []

    def values(battle, legal, out, belief):
        seen.append(belief)
        return [1.0] * len(out["logits"]), {"cells": 0}

    policy.matrix_values = values
    assert torch.equal(policy.propose(view).out["logits"], plain.out["logits"])  # weight 0: off
    policy.matrix_blend = 0.5
    shifted = policy.propose(view).out
    # base mode: the hook gets a belief kept for the matrix only (research Player2._matrix_belief), live settings
    assert len(seen) == 1 and seen[0] is not None and not seen[0].tracker.crit_reported
    assert torch.allclose(shifted["logits"], plain.out["logits"] + 0.5)
    assert torch.equal(shifted["network_logits"], plain.out["logits"]) and shifted["matrix_info"] == {"cells": 0}

    def broken(battle, legal, out, belief):
        raise RuntimeError("no cells")

    policy.matrix_values = broken
    failed = policy.propose(view).out
    assert torch.equal(failed["logits"], plain.out["logits"]) and policy.matrix_errors == 1
    assert failed["matrix_error"] == "RuntimeError: no cells"


def _live_battles(count: int) -> list[Path]:
    runs = sorted(LIVE_RUNS.glob("*-run")) if LIVE_RUNS.is_dir() else []
    for run in reversed(runs):
        found = [b for b in sorted(run.glob("battle-*")) if any((b / "decisions").glob("[0-9]*-*.json"))]
        if found:
            return found[:count]
    return []


@pytest.mark.torch
@pytest.mark.parametrize("mode", ["base", "belief+mega"])
def test_dry_run_on_recorded_live_decisions(tmp_path: Path, mode: str, record_property) -> None:
    """The newest recorded live run's first battles, every decision the player was shown a Battle for, played again
    in order (gold940; for belief+mega the same weights under a belief+mega config: this checks the plumbing, not
    the play). No fallback, finite answers, the belief's notes in belief mode. research/pcsa-belief/stage4/
    dry_run_live.py runs every recorded battle with a trained belief network."""
    import torch

    from pcsa.live import restore

    battles = _live_battles(2)
    if not battles or not GOLD.is_file():
        pytest.skip("no recorded live decisions or no gold940 checkpoint")
    path = GOLD
    if mode != "base":
        state = torch.load(GOLD, map_location="cpu", weights_only=False)
        state["config"] = {**(state.get("config") or {}), "features": mode}
        path = tmp_path / "gold940-as-belief.pt"
        torch.save(state, path)
    params = replace(
        NeuralParams.load(REPO / "config" / "players" / "neural.yaml"), checkpoint=str(path), seed=0, matrix_blend=0.0
    )
    player = NeuralPlayer(params, MaxDamagePlayer(), path)
    assert player.error is None and player.policy.features == mode
    n = 0
    for bdir in battles:
        player.policy.reset()
        records, _ = restore.load_records(bdir / "decisions")
        for rec in records:
            if rec.get("degrade") not in restore.SHOWN_BATTLE:
                continue
            try:
                view = restore.view_of(rec, restore.battle_tag(rec.get("battle_id") or "b", rec["sequence"]))
            except ValueError:
                continue
            assert player.choose_action(view, soon()) in view.legal
            notes = player.notes()
            assert notes["rule"] == "neural", notes.get("why")
            assert ("belief" in notes) == (mode != "base")
            n += 1
    record_property("decisions", n)
    assert n > 5
