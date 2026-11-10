"""Parity with research/pcsa-rl (skipped when that folder or the simulator is absent, as in CI).

1. Network: research's Net and ours, loaded from the same checkpoint, give bit-identical logits and values on the
   archived decisions of an update (runs/s1/archive/u01401.npz, played by 1400.pt), and our capped probabilities
   match the archived behaviour probabilities up to the archive's float16 rounding.
2. Observation and policy: simulator battles where, at every team preview and decision, research's
   PolicyPlayer and ours see the same Battle (the arena's p1 view: a live Battle as p1, a replayed one as p2)
   and must produce identical observations, preview values, probabilities, values and actions.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH = Path(os.environ.get("PCSA_RL_RESEARCH", r"C:\Users\geral\Code\research\pcsa-rl"))
ARCHIVE = RESEARCH / "runs" / "s1" / "archive" / "u01401.npz"
PLAYED_BY = RESEARCH / "runs" / "s1" / "ckpt" / "1400.pt"
LATEST = RESEARCH / "runs" / "s1" / "ckpt" / "latest.pt"

pytestmark = [
    pytest.mark.torch,
    pytest.mark.skipif(not (RESEARCH / "rl" / "model.py").is_file(), reason=f"no research code at {RESEARCH}"),
]


@pytest.fixture(scope="module")
def research():
    """Research's rl package (imported from its folder; it uses this branch's pcsa)."""
    if str(RESEARCH) not in sys.path:
        sys.path.insert(0, str(RESEARCH))
    import rl.features
    import rl.model
    import rl.player

    return rl


def _nets(research, path: Path):
    from pcsa.rl_training import model
    from pcsa.rl_training.player import checkpoint_config, load_checkpoint

    state = load_checkpoint(path)
    cfg, size = checkpoint_config(state)
    theirs = research.model.Net(research.model.SIZES[size]).eval()
    ours = model.Net(cfg).eval()
    theirs.load_state_dict(state["net"])
    ours.load_state_dict(state["net"])
    return theirs, ours


def _obs(z, i: int):
    from pcsa.rl_training.features import Obs

    return Obs(
        z["rows"][i].astype(np.float32), z["ids"][i].astype(np.int64), z["field"][i].astype(np.float32), z["mask"][i]
    )


def test_network_parity_on_archived_decisions(research, record_property) -> None:
    import torch

    from pcsa.rl_training import entropy

    if not (ARCHIVE.is_file() and PLAYED_BY.is_file()):
        pytest.skip("no archived update and its checkpoint")
    theirs, ours = _nets(research, PLAYED_BY)
    with np.load(ARCHIVE) as archive:  # every array read once (an npz re-reads on each access)
        z = {k: archive[k] for k in ("rows", "ids", "field", "mask", "probs", "pos", "start", "length")}
    starts, lengths = z["start"], z["length"]
    worst, n = 0.0, 0
    for e in range(0, len(starts), max(1, len(starts) // 40)):  # 40 seat-games spread over the update
        window: list = []
        for i in range(int(starts[e]), int(starts[e] + lengths[e])):
            window = [*window, _obs(z, i)][-16:]
            pos = int(z["pos"][i])
            a_logits, a_value = theirs.step(window, pos=pos)
            b_logits, b_value = ours.step(window, pos=pos)
            assert torch.equal(a_logits, b_logits) and torch.equal(a_value, b_value)
            mask = torch.as_tensor(z["mask"][i])[None]
            probs = entropy.capped_probs(b_logits[None], mask)[0].numpy()
            worst = max(worst, float(np.abs(probs - z["probs"][i].astype(np.float32)).max()))
            n += 1
    record_property("decisions", n)
    record_property("worst_prob_diff", worst)
    assert n > 200
    assert worst < 0.05, f"capped probabilities differ from the archived ones by up to {worst:.4f}"


def _engine():
    from pcsa.common.config import SimConfig
    from pcsa.common.errors import ConfigError
    from pcsa.sim.engine import Engine

    try:
        return Engine.locate(SimConfig().format)
    except ConfigError:
        return None


class Twin:
    """Plays research's PolicyPlayer and checks ours against it on the same views."""

    name = "twin"

    def __init__(self, research, theirs_net, ours_net, catalog) -> None:
        from pcsa.rl_training.model import SIZES
        from pcsa.rl_training.player import PolicyPlayer

        cfg = SIZES["S"]
        self.theirs = research.player.PolicyPlayer(theirs_net, cfg, catalog=catalog, rng=np.random.default_rng(5))
        self.ours = PolicyPlayer(ours_net, cfg, catalog=catalog, rng=np.random.default_rng(5))
        self.decisions = self.previews = 0
        self.failures: list[str] = []
        self.battles: set[str] = set()

    def params(self) -> dict:
        return {}

    def notes(self) -> dict:
        return {}

    def choose_team(self, view, deadline) -> list[int]:
        from pcsa.rl_training import preview

        team_a, team_b = self.theirs.choose_team(view, deadline), self.ours.choose_team(view, deadline)
        picks = preview.all_picks()
        values_a = self.theirs.preview_values(view.battle, picks)
        values_b = self.ours.preview_values(view.battle, picks)
        self.check(team_a == team_b and np.array_equal(values_a, values_b), "preview", view)
        self.previews += 1
        return team_a

    def choose_action(self, view, deadline):
        self.battles.add(view.battle.battle_tag)
        proposal = self.ours.propose(view)
        chosen = self.theirs.choose_action(view, deadline)
        g, o = self.theirs.game, proposal.obs
        same_obs = all(np.array_equal(getattr(g.obs[-1], k), getattr(o, k)) for k in ("rows", "ids", "field", "mask"))
        self.check(same_obs, "observation", view)
        same = np.array_equal(g.probs[-1], proposal.probs[0].float().numpy()) and g.value[-1] == proposal.value
        self.check(same and g.action[-1] == proposal.index, "probabilities, value or action", view)
        self.check(self.ours.commit(proposal).message == chosen.message, "chosen order", view)
        self.decisions += 1
        return chosen

    def check(self, ok: bool, what: str, view) -> None:
        if not ok:
            self.failures.append(f"{what} differs at turn {view.battle.turn} of {view.battle.battle_tag}")


def test_observations_and_policy_match_research_in_simulator_battles(research, config, record_property) -> None:
    from pcsa.battle.builds import Catalog
    from pcsa.players.arena import ArenaPlayer, ExportTeams, close, run_match
    from pcsa.players.factory import build_player
    from pcsa.sim.server import ShowdownServer
    from pcsa.sim.teams import sample_teams

    engine = _engine()
    if engine is None or not LATEST.is_file():
        pytest.skip("the simulator is not installed, or no research checkpoint")
    catalog = Catalog.load()
    theirs_net, ours_net = _nets(research, LATEST)
    twin = Twin(research, theirs_net, ours_net, catalog)
    rng = random.Random(3)
    teams = sample_teams(catalog, engine, rng, 4)
    with ShowdownServer(engine, None) as server:
        a = ArenaPlayer(twin, ExportTeams(teams[0::2]), "twin", server.configuration, engine.format, rng)
        b_inner = build_player(config, "disciplined")
        b = ArenaPlayer(b_inner, ExportTeams(teams[1::2]), "disc", server.configuration, engine.format, rng)
        try:
            result = run_match(a, b, 4)
        finally:
            close(a, b)
    assert not result.errors, result.errors
    record_property("decisions", twin.decisions)
    assert twin.failures == []
    assert twin.previews == 4 and len(twin.battles) == 4 and twin.decisions >= 20
