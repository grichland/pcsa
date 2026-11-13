"""The robust preview: turn-1 Battles for (our pick, their lead), the worst-N score, and the player using it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pcsa.common.errors import ConfigError
from pcsa.players import mirror
from pcsa.players.max_damage import MaxDamagePlayer
from pcsa.players.neural import NeuralParams, NeuralPlayer
from pcsa.rl_training.preview_leads import lead_battle, robust_scores, their_species
from tests.rl_training.battles import preview_battle
from tests.rl_training.test_neural import preview_view, soon
from tests.rl_training.test_neural_features import checkpoint


def test_the_lead_battle_has_both_leads_out_at_turn_1() -> None:
    b = preview_battle()
    species = their_species(b)
    ours = list(b.team.values())
    lb = lead_battle(b, (2, 0, 4), species[3], species, "t")
    assert lb.turn == 1 and len(lb.team) == 3
    assert lb.active_pokemon is not None and lb.active_pokemon.species == ours[2].species
    assert lb.opponent_active_pokemon is not None and lb.opponent_active_pokemon.current_hp_fraction == 1.0
    assert lb.opponent_active_pokemon.species.startswith(species[3].lower().split("-")[0])
    assert len(lb.teampreview_opponent_team) == 6
    moves = [a for a in mirror.legal_actions(lb) if a.message.startswith("/choose move")]
    switches = [a for a in mirror.legal_actions(lb) if a.message.startswith("/choose switch")]
    assert moves and len(switches) == 2


def test_robust_scores_take_the_mean_of_the_worst() -> None:
    t = np.array([[0.9, 0.8, -0.6, 0.7], [0.3, 0.2, 0.25, 0.3]])
    assert np.allclose(robust_scores(t, 1), [-0.6, 0.2])
    assert np.allclose(robust_scores(t, 2), [0.05, 0.225])
    assert np.allclose(robust_scores(t, 9), t.mean(1))  # more than the leads: the plain mean
    assert int(np.argmax(robust_scores(t, 2))) == 1 and int(np.argmax(t.mean(1))) == 0


def test_the_setting_is_checked() -> None:
    NeuralParams(preview="robust", preview_robust_worst=2).check("x")
    for bad in (0, 7, 1.5, True):
        with pytest.raises(ConfigError, match="preview_robust_worst"):
            NeuralParams(preview="robust", preview_robust_worst=bad).check("x")  # type: ignore[arg-type]


@pytest.mark.torch
@pytest.mark.parametrize("mode", [None, "belief+mega"])
def test_the_player_picks_the_best_worst_case(tmp_path: Path, mode: str | None) -> None:
    from pcsa.rl_training import preview

    path = checkpoint(tmp_path, mode)
    player = NeuralPlayer(
        replace(NeuralParams(seed=0), preview="robust", preview_robust_worst=2), MaxDamagePlayer(), path
    )
    assert player.policy.preview_robust_worst == 2
    view = preview_view()
    team = player.choose_team(view, soon())
    notes = player.notes()
    assert (
        notes["rule"] == "neural"
        and notes["preview"] == "robust"
        and len(notes["their_leads"]) == 6
        and len(notes["values"]) == 6
    )
    picks = preview.all_picks(6, 3)
    table, _ = player.policy.lead_values(view.battle, picks)
    assert table.shape == (60, 6) and np.isfinite(table).all()
    best = picks[int(np.argmax(robust_scores(table, 2)))]
    assert tuple(team) == tuple(best)
    critic = NeuralPlayer(NeuralParams(seed=0), MaxDamagePlayer(), path)
    assert critic.policy.preview_robust_worst == 0
