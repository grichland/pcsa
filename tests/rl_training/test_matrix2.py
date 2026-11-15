"""The two-turn matrix search (pcsa.rl_training.matrix2, research/pcsa-search): the solver, stage changes, HP bins,
the patches, the turn-1 branches, the setting, and (when research/pcsa-rl is present) the same Q as research's
rl2/matrix2.py on archived simulator decisions."""

from __future__ import annotations

import sys
import types
from dataclasses import replace

import numpy as np
import pytest
from poke_env.battle import Pokemon

from pcsa.common.errors import ConfigError
from pcsa.players import opponent as opp
from pcsa.players.neural import NeuralParams
from pcsa.rl_training import matrix2_parts as P
from pcsa.rl_training.matrix_cells import Hit
from tests.conftest import REPO
from tests.rl_training.test_matrix import BELIEF, RESEARCH, STEM


def test_regret_matching_finds_the_equilibrium() -> None:
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)
    x, y = P.regret_matching(rps, 500)
    assert np.allclose(x, 1 / 3, atol=0.05) and np.allclose(y, 1 / 3, atol=0.05)
    x, y = P.regret_matching(np.array([[0.7, 0.4], [0.5, 0.3]]))  # row 0 dominates; column 1 is theirs
    assert x[0] > 0.95 and y[1] > 0.95


def test_stage_changes_from_the_move_data() -> None:
    sd, draco, dd = (opp.move_object(m) for m in ("swordsdance", "dracometeor", "dragondance"))
    assert P.after(sd, {}) == {"atk": 2} and P.stage_change(sd, {"atk": 5}, True) == {"atk": 6}
    assert P.after(draco, {}) == {"spa": -2} and P.after(dd, {}) == {"atk": 1, "spe": 1}
    assert P.after(opp.move_object("earthquake"), {"atk": 1}) == {"atk": 1} and P.after(None, {"x": 1}) == {"x": 1}


def test_hp_bins() -> None:
    assert P.hp_bin(0.0) == 0.0 and P.hp_bin(-0.2) == 0.0 and P.hp_bin(0.01) == P.HP_BIN / 2
    assert P.hp_bin(0.44) == 0.4 and P.hp_bin(0.96) == 1.0 and P.hp_bin(1.3) == 1.0


def test_the_patch_restores_stages_and_turn_counters() -> None:
    mon = Pokemon(gen=9, species="garchomp")
    before, turns = dict(mon._boosts), mon._active_turns
    with P.patched((mon, {"atk": 2}, True)):
        assert mon.boosts["atk"] == 2 and mon.first_turn
    assert mon._boosts == before and mon._active_turns == turns
    with pytest.raises(RuntimeError), P.patched((mon, {"spe": 1}, False)):
        raise RuntimeError
    assert mon._boosts == before


def test_branch_probabilities_sum_to_one() -> None:
    a, b = Pokemon(gen=9, species="garchomp"), Pokemon(gen=9, species="dragonite")
    x = P.Exchange(a, b, 1.0, 1.0, {}, {}, {"atk": 2}, {}, None, None)
    out = P.exchange_branches(x, Hit(mean=0.6, ko=0.3, damaging=True), Hit(mean=0.5, ko=0.2, damaging=True), 0.6)
    assert abs(sum(p for p, *_ in out) - 1.0) < 1e-9
    assert out[0][3] is b and out[0][1].our_boosts == (("atk", 2),)  # we moved first, set up and KO'd them
    prot = P.exchange_branches(replace(x, ours_after={}), Hit(protect=True), Hit(mean=0.5, damaging=True), 0.5)
    assert len(prot) == 1 and prot[0][0] == 1.0 and prot[0][1].our_hp == 1.0  # Protect: nothing happens


def test_the_depth_setting() -> None:
    assert NeuralParams().matrix_depth == 1
    assert NeuralParams.load(REPO / "config" / "players" / "neural.yaml").matrix_depth in (1, 2)
    for bad in (0, 3, True, "2"):
        with pytest.raises(ConfigError, match="matrix_depth"):
            replace(NeuralParams(), matrix_depth=bad).check("test")


def test_q_equals_research_on_archived_decisions(record_property) -> None:
    import torch

    if not STEM.with_suffix(".npz").is_file():
        pytest.skip("research archive not present")
    for p in (RESEARCH, BELIEF, BELIEF / "stage0"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import common  # research/pcsa-belief/stage0
    from rl2 import belief_features as rbf
    from rl2 import matrix2 as rm2

    from pcsa.players import mirror
    from pcsa.rl_training import belief_features as pbf
    from pcsa.rl_training import former_features as pf
    from pcsa.rl_training import matrix2 as pm2
    from pcsa.rl_training.former import Config, Net

    torch.manual_seed(0)
    net = Net(Config(d=64, layers=2)).eval()
    side, z = common.load_update(STEM)
    n = depth2 = 0
    for g in [i for i, e in enumerate(side) if e["seat"] == "p1"][:1]:
        e = side[g]
        s0, k = int(z["start"][g]), int(z["length"][g])
        lines = common.p1_lines(e)
        cuts = common.decision_cuts(lines, z["turn"][s0 : s0 + k], z["forced"][s0 : s0 + k], z["action"][s0 : s0 + k])
        ours = common.match_sets(common.parse_team(e["team"]), lines, "p1")
        pb, rb, cache = pbf.new_battle_belief(crit_reported=True, empty_unknown=False), rbf.new_battle_belief(), {}
        for cut in (cuts or [])[:6]:
            bp, br = (common.rebuild(lines[: cut.end], ours, f"e{g}", cut.forced) for _ in range(2))
            obs = pf.observe(bp, mirror.legal_actions(bp), belief=pb, feature_mode="belief+mega")
            rb.update_from(br)
            with torch.no_grad():
                out = net.step_cached(obs, cache)
            if obs.forced:
                continue
            qp, ip = pm2.TwoTurn(pb, time_cap_s=600).values(bp, mirror.legal_actions(bp), out)
            qr, ir = rm2.TwoTurn(rb, time_cap_s=600).values(br, mirror.legal_actions(br), out)
            assert np.allclose(np.asarray(qp), np.asarray(qr), atol=1e-6), (ip.get("cont"), ir.get("cont"))
            n += 1
            depth2 += int(ip.get("depth") == 2)
    record_property("decisions", n)
    assert n > 2 and depth2 == n


def test_the_notes_say_which_depth_ran() -> None:
    import torch

    from pcsa.rl_training.former_notes import blend_notes

    base = {"matrix_q": torch.zeros(11), "network_logits": torch.arange(11.0)}
    one = blend_notes({**base, "matrix_info": {"counts": {"hits": 3}}}, 2.0)
    assert one["depth_set"] == 1 and one["depth"] == 1 and "cont" not in one and one["network_index"] == 10
    two = blend_notes({**base, "matrix_info": {"depth": 2, "q1": [0.0] * 11, "cont": [0.1] * 11, "ms2": 12.5}}, 2.0)
    assert two["depth_set"] == 2 and two["depth"] == 2 and two["cont"] == [0.1] * 11 and two["ms2"] == 12.5
    cap = blend_notes({**base, "matrix_info": {"depth": 1, "depth2": "time cap", "ms2": 1500.0}}, 2.0)
    assert cap["depth_set"] == 2 and cap["depth"] == 1 and cap["depth2"] == "time cap"


def test_the_batched_solver_matches_one_game_at_a_time() -> None:
    from pcsa.rl_training.matrix2_fast import regret_matching_batch

    rng = np.random.default_rng(0)
    games = [rng.normal(size=(r, c)) for r, c in ((3, 9), (6, 9), (1, 4), (5, 2), (4, 4))]
    batched = regret_matching_batch(games)
    for V, y in zip(games, batched, strict=True):
        assert y.shape == (V.shape[1],) and np.allclose(y, P.regret_matching(V)[1], atol=1e-9)
    assert regret_matching_batch([]) == []


def test_the_memo_returns_the_first_result_until_the_stages_change(monkeypatch) -> None:
    from pcsa.rl_training.matrix import Lookahead
    from pcsa.rl_training.matrix2_fast import MemoLookahead

    calls: list = []
    monkeypatch.setattr(Lookahead, "p_first", lambda self, *args: calls.append(args) or 0.5)
    a, b = Pokemon(gen=9, species="garchomp"), Pokemon(gen=9, species="dragonite")
    la = MemoLookahead(types.SimpleNamespace(_matrix_across={}), 0.0, 0.0)
    assert la.p_first(None, a, b, None, 0, 0) == 0.5 and la.p_first(None, a, b, None, 0, 0) == 0.5
    assert len(calls) == 1
    with P.patched((a, {"spe": 1}, False)):
        la.p_first(None, a, b, None, 0, 0)
    assert len(calls) == 2  # a stage change is a new key
