"""rule_mode: mask end to end on a synthetic checkpoint (pcsa.rl_training.rule_masks, 14 Nov 2026): the forbidden
actions leave the legality mask before the blend and the softmax, the network chooses among the rest, the notes
name the masked rows; all masked = no masks; a rule error = no masks, noted."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pcsa.players.max_damage import MaxDamagePlayer
from pcsa.players.neural import NeuralParams, NeuralPlayer
from pcsa.run.watch_matrix import masked_view, matrix_view
from tests.players.test_disciplined import FakeTracker
from tests.players.test_disciplined import battle as decision_battle
from tests.rl_training.test_neural import preview_view, soon, synthetic_checkpoint, view_of

SLACK_OFF = "/choose move slackoff"


def masked_player(path: Path, forbidden) -> NeuralPlayer:
    """Mask mode with the rules replaced by `forbidden(view, out) -> {message: (rule, reason)}`."""
    params = replace(NeuralParams(seed=0), rules=["no_full_heal"], rule_mode="mask")
    player = NeuralPlayer(params, MaxDamagePlayer(), path, threat=lambda: (None, FakeTracker()))
    player.policy.rule_mask = forbidden
    return player


@pytest.mark.torch
def test_the_masked_action_is_never_chosen_and_the_notes_say_why(tmp_path: Path) -> None:
    path = synthetic_checkpoint(tmp_path)
    b = decision_battle("Slowbro")
    free = masked_player(path, lambda view, out: {})
    free.choose_team(preview_view(), soon())
    first = free.choose_action(view_of(b), soon())
    player = masked_player(path, lambda view, out: {first.message: ("no_full_heal", "test reason")})
    player.choose_team(preview_view(), soon())
    chosen = player.choose_action(view_of(b), soon())
    notes = player.notes()
    assert chosen.message != first.message and "override" not in notes
    row = int(next(iter(notes["matrix"]["masked"])))
    assert notes["matrix"]["masked"][str(row)] == ["no_full_heal", "test reason"]
    assert notes["matrix"]["probs"][row] == 0.0 and notes["matrix"]["legal"][row]  # legal, but masked
    assert notes["mask_rules"] == ["no_full_heal"] and notes["network"] == first.message
    view = matrix_view(notes["matrix"], order=chosen.message, network=notes["network"])
    assert view is not None and view["masked"][row] == ["no_full_heal", "test reason"]
    assert view["chosen"] == row and view["played"] != row


@pytest.mark.torch
def test_the_real_no_full_heal_rule_masks_slack_off_at_full_hp(tmp_path: Path) -> None:
    params = replace(NeuralParams(seed=0), rules=["no_full_heal"], rule_mode="mask")
    player = NeuralPlayer(params, MaxDamagePlayer(), synthetic_checkpoint(tmp_path))
    player.choose_team(preview_view(), soon())
    assert player.choose_action(view_of(decision_battle("Slowbro")), soon()).message != SLACK_OFF
    notes = player.notes()
    masked = notes["matrix"]["masked"]
    assert [rule for rule, _ in masked.values()] == ["no_full_heal"] and "at full HP" in next(iter(masked.values()))[
        1
    ]


@pytest.mark.torch
def test_masks_that_leave_nothing_are_ignored_and_errors_never_stop_a_decision(tmp_path: Path) -> None:
    path = synthetic_checkpoint(tmp_path)
    b = decision_battle("Slowbro")
    everything = masked_player(path, lambda view, out: {a.message: ("no_full_heal", "all") for a in view.legal})
    everything.choose_team(preview_view(), soon())
    assert everything.choose_action(view_of(b), soon()) in view_of(b).legal
    notes = everything.notes()
    assert notes["mask_ignored"] and "masked" not in notes["matrix"] and notes["rule"] == "neural"

    def broken(view, out):
        raise RuntimeError("boom")

    failing = masked_player(path, broken)
    failing.choose_team(preview_view(), soon())
    assert failing.choose_action(view_of(b), soon()) in view_of(b).legal
    notes = failing.notes()
    assert "boom" in notes["mask_error"] and notes["rule"] == "neural"


def test_rule_mode_is_checked(tmp_path: Path) -> None:
    from pcsa.common.errors import ConfigError

    path = tmp_path / "neural.yaml"
    path.write_text("rule_mode: sometimes\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        NeuralParams.load(path)
    path.write_text("rule_mode: mask\n", encoding="utf-8")
    assert NeuralParams.load(path).rule_mode == "mask"


def test_the_dashboard_reads_masked_rows() -> None:
    assert masked_view(None) is None and masked_view({}) is None
    rows = masked_view({"3": ["no_effect", "immune"], "x": ["bad", "key"], "20": ["out", "of range"]})
    assert rows is not None and rows[3] == ["no_effect", "immune"] and sum(r is not None for r in rows) == 1
