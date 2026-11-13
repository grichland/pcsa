"""Their move options narrowed by their builds (pcsa.rl_training.opponent_moves, `opponent_moves_from_builds`)."""

from __future__ import annotations

from types import SimpleNamespace as N

import pytest

from pcsa.rl_training import opponent_moves as om

SEEN = [("Sludge Bomb", "Flamethrower", "Ice Beam", "Trick Room"), ("Chilly Reception", "Sludge Bomb", "Slack Off")]
LISTED = [("Psychic", "Toxic", "Ice Beam", "Slack Off")]


class FakeCatalog:
    def __init__(self, seen=SEEN, listed=LISTED) -> None:
        self.seen = N(ranked=lambda species: [N(build=N(moves=m)) for m in seen])
        self._listed = [N(moves=m) for m in [*seen, *listed]]

    def builds(self, species):
        return self._listed


def test_the_seen_builds_decide_when_one_fits() -> None:
    got = om.build_moves("slowkinggalar", frozenset({"flamethrower"}), FakeCatalog())
    assert got == {"sludgebomb", "flamethrower", "icebeam", "trickroom"}
    assert "icepunch" not in om.build_moves("slowkinggalar", frozenset(), FakeCatalog())


def test_a_move_no_seen_build_has_falls_back_to_the_listed_builds_then_to_no_mask() -> None:
    assert om.build_moves("slowkinggalar", frozenset({"toxic"}), FakeCatalog()) == {
        "psychic", "toxic", "icebeam", "slackoff"
    }  # fmt: skip
    assert om.build_moves("slowkinggalar", frozenset({"icepunch"}), FakeCatalog()) is None
    assert om.build_moves("slowkinggalar", frozenset(), FakeCatalog(seen=[], listed=[])) is None


def test_the_mask_holds_the_build_moves() -> None:
    from pcsa.rl_training.features_vocab import index

    mon = N(species="slowkinggalar", moves={"flamethrower": None})
    mask = om.allowed_mask(N(opponent_active_pokemon=mon), FakeCatalog())
    assert mask is not None and mask[index("move", "icebeam")] and not mask[index("move", "icepunch")]
    assert om.allowed_mask(N(opponent_active_pokemon=None), FakeCatalog()) is None


@pytest.mark.torch
def test_the_opponent_head_puts_no_weight_outside_the_mask_but_keeps_revealed_moves() -> None:
    import numpy as np
    import torch

    from pcsa.rl_training import former_features as fx
    from pcsa.rl_training.features_vocab import index
    from pcsa.rl_training.spec import VOCAB
    from tests.rl_training.test_former import DD, player, view_of

    p = player(3)
    view = view_of([[DD[0], DD[1]]])
    obs = fx.observe(view.battle, view.legal)
    keep = ["outrage", "extremespeed", "earthquake", "firepunch"]
    mask = np.zeros(VOCAB["move"], bool)
    mask[[index("move", m) for m in keep]] = True
    plain = p.net.step_cached(obs, {})["opp_logits"].softmax(-1)
    narrowed = p.net.step_cached(obs, {}, mask)["opp_logits"].softmax(-1)
    allowed = {index("move", m) for m in [*keep, "dragondance"]}
    outside = [i for i in range(1, VOCAB["move"]) if i not in allowed]
    assert float(plain[outside].sum()) > 0.1 and float(narrowed[outside].sum()) < 1e-6
    assert float(narrowed[index("move", "dragondance")]) > 0  # revealed: kept
    v = VOCAB["move"]  # switches keep their share relative to each other
    assert torch.allclose(
        plain[v:] / plain[v:].sum().clamp_min(1e-9), narrowed[v:] / narrowed[v:].sum().clamp_min(1e-9)
    )


def test_off_by_default_and_the_live_config_turns_it_on() -> None:
    from pcsa.players.neural import NeuralParams
    from tests.conftest import REPO

    assert NeuralParams().opponent_moves_from_builds is False
    assert NeuralParams.load(REPO / "config" / "players" / "neural.yaml").opponent_moves_from_builds is True
