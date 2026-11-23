"""The choice rules over the network's matrix: mean, min, mix and regret, with their top-k, and the policy step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pcsa.common.config_cg import ChoiceSettings
from pcsa.network import choosing
from pcsa.network import layout as L

INF = float("-inf")


def rule(name: str, top_ours: int = 0, top_theirs: int = 0, blend: float = 0.5) -> ChoiceSettings:
    return ChoiceSettings(rule=name, top_ours=top_ours, top_theirs=top_theirs, blend=blend)  # type: ignore[arg-type]


def _case(cells: list[list[float]], p: list[float], pi: list[float] | None = None):
    """Our actions 0..len(cells)-1 legal against their columns 0..len(p)-1; everything else empty."""
    m = torch.zeros(1, 11, 13)
    for a, row in enumerate(cells):
        m[0, a, : len(row)] = torch.tensor(row)
    p_opp = torch.zeros(1, 13)
    p_opp[0, : len(p)] = torch.tensor(p)
    legal = torch.zeros(1, 11, dtype=torch.bool)
    legal[0, : len(cells)] = True
    pol = torch.zeros(1, 11)
    pol[0, : len(cells)] = torch.tensor(pi or [1.0 / len(cells)] * len(cells))
    return m, p_opp, pol, legal


# our attack and switch against their two replies, "they KO us" and "they set up"
KO_OR_SETUP = [[-0.9, 0.5], [-0.8, -0.4]]


def test_min_and_regret_disagree_where_one_reply_beats_everything() -> None:
    m, p, pi, legal = _case(KO_OR_SETUP, [0.5, 0.5])
    worst = choosing.scores(m, p, pi, legal, rule("min"))
    regret = choosing.scores(m, p, pi, legal, rule("regret"))
    assert worst[0, :2].tolist() == pytest.approx([-0.9, -0.8]) and int(worst.argmax()) == 1
    assert regret[0, :2].tolist() == pytest.approx([-0.1, -0.9]) and int(regret.argmax()) == 0
    assert worst[0, 2:].eq(INF).all() and regret[0, 2:].eq(INF).all()  # actions not legal stay out


def test_mean_without_top_k_is_the_networks_expected_score_and_changes_nothing() -> None:
    m, p, pi, legal = _case(KO_OR_SETUP, [0.25, 0.75])
    mean = choosing.scores(m, p, pi, legal, rule("mean"))
    assert mean[0, :2].tolist() == pytest.approx([0.25 * -0.9 + 0.75 * 0.5, 0.25 * -0.8 + 0.75 * -0.4])
    assert choosing.config(rule("mean").model_dump()) is None and choosing.config(None) is None
    mix = choosing.scores(m, p, pi, legal, rule("mix", blend=0.5))
    assert mix[0, 0] == pytest.approx(0.5 * mean[0, 0] + 0.5 * -0.9)


def test_top_theirs_keeps_only_their_likeliest_replies() -> None:
    # reply 2 is unlikely and bad for action 0: with all replies min avoids 0, with the top 2 it does not
    m, p, pi, legal = _case([[0.5, 0.4, -1.0], [0.1, 0.2, 0.0]], [0.6, 0.35, 0.05])
    assert int(choosing.scores(m, p, pi, legal, rule("min")).argmax()) == 1
    top2 = choosing.scores(m, p, pi, legal, rule("min", top_theirs=2))
    assert top2[0, :2].tolist() == pytest.approx([0.4, 0.1]) and int(top2.argmax()) == 0
    weighted = choosing.scores(m, p, pi, legal, rule("mean", top_theirs=2))
    assert weighted[0, 0] == pytest.approx((0.6 * 0.5 + 0.35 * 0.4) / 0.95)


def test_top_ours_keeps_our_likeliest_actions_and_sets_regrets_best() -> None:
    m, p, pi, legal = _case([[0.0, 0.0], [0.3, -0.5], [-0.2, 0.6]], [0.5, 0.5], pi=[0.1, 0.6, 0.3])
    two = choosing.scores(m, p, pi, legal, rule("regret", top_ours=2))
    assert two[0, 0] == INF  # the least likely action is not kept
    assert two[0, 1:3].tolist() == pytest.approx([-1.1, -0.5])


def test_their_unknown_replies_count_equally() -> None:
    m, p, pi, legal = _case(KO_OR_SETUP, [0.0, 0.0])
    mean = choosing.scores(m, p, pi, legal, rule("mean", top_theirs=13))
    assert mean[0, 0] == pytest.approx((-0.9 + 0.5 + 11 * 0.0) / 13)


def test_apply_replaces_turn_scores_and_notes_the_pick() -> None:
    m, p, pi, legal = _case(KO_OR_SETUP, [0.5, 0.5], pi=[0.2, 0.8])
    out = {"matrix": m.repeat(2, 1, 1), "p_opp": p.repeat(2, 1), "pi": pi.repeat(2, 1), "legal": legal.repeat(2, 1)}
    scores = torch.full((2, 71), INF)
    scores[:, :2] = torch.tensor([0.2, 0.8]).log()
    kinds = np.array([L.KIND_TURN, L.KIND_REPLACE])
    notes = choosing.apply(rule("regret"), kinds, out, scores, 0.05)
    assert int(scores[0, :11].argmax()) == 0 and int(scores[1, :11].argmax()) == 1  # a replace keeps the network's
    assert notes[1] is None and notes[0] is not None
    assert notes[0]["rule"] == "regret" and notes[0]["model_pick"] == 1 and notes[0]["pick"] == 0
    assert notes[0]["changed"] and notes[0]["scores"][:2] == pytest.approx([-0.1, -0.9]) and notes[0]["scores"][2] is None
    assert choosing.apply(None, kinds, out, scores, 0.05) == [None, None]


def test_bad_settings_are_refused() -> None:
    with pytest.raises(ValueError):
        rule("maximax")
    with pytest.raises(ValueError):
        rule("mix", blend=1.5)
