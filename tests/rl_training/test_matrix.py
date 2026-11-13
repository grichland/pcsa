"""The matrix blend (stage 3 of research/pcsa-belief, pcsa.rl_training.matrix): the cell arithmetic, the setting
(off by default, the trial config), the player with the blend on in both feature modes, and (when research/pcsa-rl
is present) the same Q and cells as research's rl2/matrix.py on archived simulator decisions. The full measurement
is research/pcsa-belief/stage4/equiv_matrix.py."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from pcsa.common.config import load_config
from pcsa.common.errors import ConfigError
from pcsa.players.max_damage import MaxDamagePlayer
from pcsa.players.neural import NeuralParams, NeuralPlayer
from pcsa.rl_training.matrix_cells import NO_HIT, Hit, dealt_value, exchange, first_prob, q_values
from tests.conftest import REPO
from tests.rl_training.test_neural import decision_battle, preview_view, soon, view_of
from tests.rl_training.test_neural_features import checkpoint

RESEARCH = Path(os.environ.get("PCSA_RL_RESEARCH", r"C:\Users\geral\Code\research\pcsa-rl"))
BELIEF = RESEARCH.parent / "pcsa-belief"
STEM = RESEARCH / "runs" / "v2-teams-1111" / "archive" / "u00845"
TRIAL = REPO / "config" / "players" / "neural-belief.yaml"
BELIEF100 = REPO / "assets" / "models" / "poke-transformer-belief100-20261112-1905.pt"


def test_cell_arithmetic() -> None:
    hit = Hit(mean=0.4, ko=0.25)
    assert dealt_value(hit, 0.0) == 0.0
    assert math.isclose(dealt_value(hit, 0.5), 0.75 * 0.4 + 0.25 * 0.5 + 0.3 * 0.25)
    assert exchange(Hit(mean=0.5, protect=True), hit, 1.0, 1.0, 1.0) == 0.0  # Protect zeroes the cell
    sucker = Hit(mean=0.5, ko=0.0, sucker=True, damaging=True)
    assert exchange(sucker, NO_HIT, 1.0, 1.0, 1.0) == 0.0  # Sucker Punch fails on a non-damaging move
    ours, theirs = Hit(mean=0.3, ko=0.1, damaging=True), Hit(mean=0.2, ko=0.05, damaging=True)
    d, t = dealt_value(ours, 1.0), dealt_value(theirs, 1.0)
    want = 0.6 * (d - 0.9 * t) + 0.4 * (0.95 * d - t)
    assert math.isclose(exchange(ours, theirs, 0.6, 1.0, 1.0), want)
    assert first_prob(1, 0, 0.9, 0.0, False) == 1.0 and first_prob(0, 1, 0.0, 0.0, False) == 0.0
    assert math.isclose(first_prob(0, 0, 0.2, 0.1, False), 0.7 + 0.05)
    assert math.isclose(first_prob(0, 0, 0.2, 0.1, True), 0.2 + 0.05)  # Trick Room swaps the order
    assert q_values({0: {1: 1.0, 2: -1.0}}, {1: 0.3, 2: 0.1}) == {0: (0.3 - 0.1) / 0.4}
    assert q_values({0: {1: 1.0}}, {}) == {0: 0.0}


def test_the_blend_is_off_by_default_and_the_trial_config_turns_it_on() -> None:
    assert NeuralParams().matrix_blend == 0.0
    live = NeuralParams.load(REPO / "config" / "players" / "neural.yaml")
    assert live.matrix_blend == 0.0 and "gold940" in live.checkpoint
    trial = NeuralParams.load(TRIAL)
    assert (
        trial.matrix_blend == 4.0 and trial.checkpoint == "assets/models/poke-transformer-belief100-20261112-1905.pt"
    )
    differ = {k for k in vars(live) if getattr(live, k) != getattr(trial, k)}
    assert differ == {"checkpoint", "matrix_blend"}  # the trial is neural.yaml with these two changed
    config = load_config(REPO / "config" / "agent.yaml", overrides=["policy.neural=players/neural-belief.yaml"])
    assert NeuralParams.load(config.directory / config.agent.policy.neural) == trial
    for bad in (-1.0, float("nan"), True, "4"):
        with pytest.raises(ConfigError, match="matrix_blend"):
            replace(NeuralParams(), matrix_blend=bad).check("test")


@pytest.mark.torch
@pytest.mark.parametrize("mode", [None, "belief+mega"])
def test_the_player_blends_q_into_the_logits(tmp_path: Path, mode: str | None) -> None:
    import torch

    from pcsa.belief.fast import BattleBelief

    path = checkpoint(tmp_path, mode)
    plain = NeuralPlayer(NeuralParams(seed=0), MaxDamagePlayer(), path)
    player = NeuralPlayer(NeuralParams(seed=0, matrix_blend=4.0), MaxDamagePlayer(), path)
    assert plain.policy.matrix_values is None and player.policy.matrix_values is not None
    assert player.params()["matrix_blend"] == 4.0
    for p in (plain, player):
        p.choose_team(preview_view(), soon())
    view = view_of(decision_battle("Azumarill"))
    a, b = plain.policy.propose(view), player.policy.propose(view)
    assert "matrix_q" not in a.out and torch.equal(b.out["network_logits"], a.out["logits"])
    q = b.out["matrix_q"]
    assert torch.isfinite(q).all() and q.abs().max() > 0
    mask = torch.as_tensor(a.obs.mask)
    assert torch.allclose(b.out["logits"][mask], a.out["logits"][mask] + 4.0 * q[mask])
    assert (b.out["logits"][~mask] < -1e8).all()  # masked actions stay masked
    assert player.choose_action(view, soon()) in view.legal
    notes = player.notes()
    assert notes["rule"] == "neural" and notes["matrix"]["blend"]["blend"] == 4.0
    assert len(notes["matrix"]["blend"]["q"]) == len(a.obs.mask)
    belief = player.policy.game.belief
    assert isinstance(belief, BattleBelief) and not belief.tracker.crit_reported  # base: kept for the matrix only
    assert plain.choose_action(view, soon()) in view.legal and "blend" not in plain.notes()["matrix"]


@pytest.mark.torch
def test_a_matrix_error_plays_the_network(tmp_path: Path) -> None:
    import torch

    player = NeuralPlayer(NeuralParams(seed=0, matrix_blend=4.0), MaxDamagePlayer(), checkpoint(tmp_path, None))
    view = view_of(decision_battle("Azumarill"))

    def broken(battle, legal, out, belief):
        raise RuntimeError("no cells")

    player.policy.matrix_values = broken
    p = player.policy.propose(view)
    assert player.policy.matrix_errors == 1 and p.out["matrix_error"] == "RuntimeError: no cells"
    assert "network_logits" not in p.out and torch.isfinite(p.out["logits"][torch.as_tensor(p.obs.mask)]).all()
    assert player.choose_action(view, soon()) in view.legal
    assert player.notes()["matrix"]["blend"] == {"blend": 4.0, "error": "RuntimeError: no cells"}


@pytest.mark.torch
@pytest.mark.skipif(
    not ((RESEARCH / "rl2" / "matrix.py").is_file() and (BELIEF / "stage0" / "common.py").is_file()),
    reason=f"no research matrix code at {RESEARCH}",
)
@pytest.mark.skipif(not Path(str(STEM) + ".npz").is_file(), reason="no archived simulator update")
def test_q_equals_research_on_archived_decisions(record_property) -> None:
    import numpy as np
    import torch

    for p in (RESEARCH, BELIEF, BELIEF / "stage0"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import common  # research/pcsa-belief/stage0
    from rl2 import belief_features as rbf
    from rl2 import matrix as rm

    from pcsa.players import mirror
    from pcsa.rl_training import belief_features as pbf
    from pcsa.rl_training import former_features as pf
    from pcsa.rl_training import matrix as pm
    from pcsa.rl_training.former import Config, Net

    torch.manual_seed(0)
    net = Net(Config(d=64, layers=2)).eval()
    side, z = common.load_update(STEM)
    n = nonzero = 0
    for g in [i for i, e in enumerate(side) if e["seat"] == "p1"][:2]:
        e = side[g]
        s0, k = int(z["start"][g]), int(z["length"][g])
        lines = common.p1_lines(e)
        cuts = common.decision_cuts(lines, z["turn"][s0 : s0 + k], z["forced"][s0 : s0 + k], z["action"][s0 : s0 + k])
        ours = common.match_sets(common.parse_team(e["team"]), lines, "p1")
        pb, rb, cache = pbf.new_battle_belief(crit_reported=True, empty_unknown=False), rbf.new_battle_belief(), {}
        for cut in cuts or []:
            bp, br = (common.rebuild(lines[: cut.end], ours, f"e{g}", cut.forced) for _ in range(2))
            obs = pf.observe(bp, mirror.legal_actions(bp), belief=pb, feature_mode="belief+mega")
            rb.update_from(br)
            with torch.no_grad():
                out = net.step_cached(obs, cache)
            if obs.forced:
                continue
            qp, ip = pm.action_values(bp, mirror.legal_actions(bp), out, pb)
            qr, ir = rm.action_values(br, mirror.legal_actions(br), out, rb)
            assert np.asarray(qp).tobytes() == np.asarray(qr).tobytes()
            assert ip.get("cells") == ir.get("cells") and ip.get("p") == ir.get("p")
            n += 1
            nonzero += int(np.abs(qp).max() > 0)
    record_property("decisions", n)
    assert n > 5 and nonzero > 0
