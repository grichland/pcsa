"""The poke-singles-former player (torch): propose changes nothing, committed decisions equal the training pass,
a same-turn retry replaces, a restored memory decides the same, the matrix in the notes, checkpoints."""

from __future__ import annotations

import random
from dataclasses import asdict

import pytest

from pcsa.common.clock import Deadline, FakeClock
from pcsa.players import mirror
from pcsa.players.base import DecisionView, Degrade, ScreenFacts, TeamPreviewView
from tests.players.test_disciplined import battle
from tests.rl_training.battles import preview_battle

pytestmark = pytest.mark.torch

DD = ["|move|p2a: Dragonite|Dragon Dance|p2a: Dragonite", "|-boost|p2a: Dragonite|atk|1"]
OUTRAGE = ["|move|p2a: Dragonite|Outrage|p1a: Azumarill", "|-damage|p1a: Azumarill|50/175"]
ESPEED = ["|move|p2a: Dragonite|Extreme Speed|p1a: Azumarill", "|-damage|p1a: Azumarill|20/175"]
GAME = ([], [DD], [DD, OUTRAGE], [DD, OUTRAGE, ESPEED])


def view_of(turns) -> DecisionView:
    b = battle("Azumarill", list(turns))
    return DecisionView(mirror.legal_actions(b), ScreenFacts(), Degrade.FULL, random.Random(0), b)


def soon() -> Deadline:
    return Deadline.after(FakeClock(), 30.0)


def player(seed: int = 0, **kw):
    import torch

    from pcsa.rl_training.former import Config, Net
    from pcsa.rl_training.former_player import FormerPlayer

    torch.manual_seed(seed)
    return FormerPlayer(Net(Config()), Config(), temperature=0.0, rng=seed, **kw)


def test_committed_decisions_equal_the_training_pass_and_a_proposal_alone_changes_nothing() -> None:
    import torch

    from pcsa.rl_training.former import stack

    p = player()
    proposals = []
    for turns in GAME:
        first = p.propose(view_of(turns))
        again = p.propose(view_of(turns))  # nothing was kept from the first proposal
        assert torch.equal(first.out["logits"], again.out["logits"]) and not again.replaces
        proposals.append(again)
        p.commit(again)
    with torch.no_grad():
        full = p.net(stack([[q.obs for q in proposals]], "cpu"))
    for t, q in enumerate(proposals):
        assert torch.allclose(q.out["logits"], full["logits"][0, t], atol=1e-4)
        assert abs(q.value - float(full["value"][0, t])) < 1e-4
    assert p.game.cache["t"] == len(GAME) and all(len(layer) == len(GAME) for layer in p.game.cache["mem"])


def test_the_same_turn_asked_again_replaces_the_last_decision() -> None:
    import torch

    p = player()
    p.choose_action(view_of(GAME[0]), soon())
    first = p.propose(view_of(GAME[1]))
    p.commit(first)
    assert "retry" not in p.notes()
    retry = p.propose(view_of(GAME[1]))
    assert retry.replaces and torch.equal(retry.out["logits"], first.out["logits"])
    p.commit(retry)
    notes = p.notes()
    assert notes["retry"] is True and notes["decisions"] == 2
    assert p.game.cache["t"] == 2 and all(len(layer) == 2 for layer in p.game.cache["mem"])


def test_a_restored_memory_decides_as_the_player_that_never_stopped() -> None:
    import torch

    p = player()
    for turns in GAME[:3]:
        p.choose_action(view_of(turns), soon())
    want = p.propose(view_of(GAME[3]))
    fresh = player()
    views = [view_of(GAME[0]), view_of(GAME[1]), view_of(GAME[1]), view_of(GAME[2])]  # with a retry in the records
    assert fresh.restore(views) == 3
    got = fresh.propose(view_of(GAME[3]))
    assert torch.allclose(got.out["logits"], want.out["logits"], atol=1e-6) and got.index == want.index
    cold = player().propose(view_of(GAME[3]))
    assert not torch.allclose(cold.out["logits"], want.out["logits"], atol=1e-6)  # the memory matters


def test_a_view_that_cannot_be_replayed_keeps_what_was_rebuilt() -> None:
    broken = DecisionView(view_of(GAME[1]).legal, ScreenFacts(), Degrade.SCREEN, random.Random(0), None)
    assert player().restore([view_of(GAME[0]), broken, view_of(GAME[2])]) == 1


def test_the_notes_carry_the_matrix_with_every_revealed_move_as_a_column() -> None:
    from pcsa.rl_training.features_vocab import index, tables

    for seed in range(4):  # untrained networks: the three revealed moves are rarely the three most likely
        p = player(seed)
        chosen = p.choose_action(view_of(GAME[3]), soon())
        notes = p.notes()
        m = notes["matrix"]
        k = p.cfg.top_k
        assert notes["rule"] == "rl" and 0 < notes["p"] <= 0.99 and isinstance(notes["v"], float)
        assert len(m["rows"]) == len(m["legal"]) == len(m["probs"]) == len(m["cells"]) == 11
        assert len(m["cols"]) == len(m["p_cols"]) == len(m["shown"]) == k + 5 and all(
            len(r) == k + 5 for r in m["cells"]
        )
        assert m["col_kind"] == ["move"] * k + ["switch"] * 5
        shown = {name for name, yes in zip(m["cols"], m["shown"], strict=True) if yes}
        assert shown == {"dragondance", "outrage", "extremespeed"}
        assert abs(sum(m["p_cols"]) - 1.0) < 0.01 and abs(sum(m["probs"]) - 1.0) < 0.01
        assert m["legal"][m["chosen"]] and m["probs"][m["chosen"]] == max(m["probs"])
        assert m["rows"][:2] == ["playrough", "aquajet"] and m["rows"][4] == "playrough + mega"
        assert chosen.action in [a.action for a in view_of(GAME[3]).legal]
        assert all(index("move", name) for name in m["cols"][:k]) and "dragondance" in tables()["move"]


def test_preview_picks_three_of_six_with_the_value_head() -> None:
    from pcsa.rl_training import preview

    b = preview_battle()
    view = TeamPreviewView(b, tuple(m.name for m in b.team.values()), random.Random(0))
    p = player(greedy_preview=True)
    team = p.choose_team(view, soon())
    notes = p.notes()
    picks = preview.all_picks(6, 3)
    values = p.preview_values(b, picks)
    assert notes["rule"] == "rl" and len(set(team)) == 3 and team == list(picks[int(values.argmax())])


def test_only_a_former_checkpoint_loads(tmp_path) -> None:
    import torch

    from pcsa.rl_training.former import Config, Net
    from pcsa.rl_training.former_player import FormerPlayer, checkpoint_config

    cfg = Config(d=64, layers=2)
    good = {"net": Net(cfg).state_dict(), "update": 7, "version": "v2", "model": asdict(cfg)}
    torch.save(good, tmp_path / "new.pt")
    loaded = FormerPlayer.from_checkpoint(tmp_path / "new.pt", temperature=0.0)
    chosen = loaded.choose_action(view_of(GAME[0]), soon())
    assert loaded.params()["d"] == 64 and chosen.action in [a.action for a in view_of(GAME[0]).legal]
    assert loaded.notes()["rule"] == "rl"
    for old in ({"net": {}, "model": {"d": 64}}, {"net": {}, "config": {"size": "S"}}, {"net": {}, "version": "v2"}):
        with pytest.raises(ValueError, match="poke-singles-former"):
            checkpoint_config(old)
