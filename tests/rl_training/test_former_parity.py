"""Parity of the poke-singles-former with research/pcsa-rl (skipped when that folder or the simulator is absent,
as in CI).

1. Network: research's Net and ours, loaded from the same checkpoint, give identical outputs (policy logits, value,
   opponent logits, Mega logit, pair matrix, kept columns and their probabilities) on archived decisions of the
   training run, and our step-by-step path (the live one) equals the whole-game pass.
2. Inputs and policy: simulator battles where, at every team preview and decision, research's observation code
   and ours see the same Battle (the arena's p1 view: a live Battle as p1, a replayed one as p2) and must produce
   identical observations, and research's network stepped on research's observations must give the outputs our
   player proposes.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH = Path(os.environ.get("PCSA_RL_RESEARCH", r"C:\Users\geral\Code\research\pcsa-rl"))
RUN = RESEARCH / "runs" / "v2-scratch"
OUTPUTS = ("logits", "value", "opp_logits", "opp_mega", "matrix", "kept_idx", "kept_p")
FIELDS = ("rows", "ids", "field", "last", "last_ids", "mask")

pytestmark = [
    pytest.mark.torch,
    pytest.mark.skipif(not (RESEARCH / "rl2" / "model.py").is_file(), reason=f"no research code at {RESEARCH}"),
]


@pytest.fixture(scope="module")
def research():
    """Research's rl2 package (imported from its folder; it uses this branch's pcsa)."""
    if str(RESEARCH) not in sys.path:
        sys.path.insert(0, str(RESEARCH))
    import rl2.features
    import rl2.model

    return rl2


def _checkpoint() -> Path | None:
    """A numbered checkpoint of the run (the trainer rewrites latest.pt while it runs)."""
    found = sorted((RUN / "ckpt").glob("[0-9]*.pt"), key=lambda p: int(p.stem))
    return found[-1] if found else None


def _nets(research, path: Path):
    from pcsa.rl_training import former
    from pcsa.rl_training.former_player import checkpoint_config, load_checkpoint

    state = load_checkpoint(path)
    cfg = checkpoint_config(state)
    theirs = research.model.Net(research.model.Config(**state["model"])).eval()
    ours = former.Net(cfg).eval()
    theirs.load_state_dict(state["net"])
    ours.load_state_dict(state["net"])
    return theirs, ours, cfg


def _games(z: dict, obs_type, count: int) -> list[list]:
    games = []
    for a, n in zip(z["start"].tolist(), z["length"].tolist(), strict=True):
        if 0 < n <= 64 and len(games) < count:
            games.append(
                [
                    obs_type(
                        rows=z["rows"][k].astype(np.float32),
                        ids=z["ids"][k].astype(np.int64),
                        field=z["field"][k].astype(np.float32),
                        last=z["last"][k].astype(np.float32),
                        last_ids=z["last_ids"][k].astype(np.int64),
                        mask=z["mask"][k],
                        turn=int(z["turn"][k]),
                        forced=bool(z["forced"][k]),
                    )
                    for k in range(a, a + n)
                ]
            )
    return games


def test_network_parity_on_archived_decisions(research, record_property) -> None:
    import torch

    from pcsa.rl_training import former, former_features

    archives = sorted((RUN / "archive").glob("u*.npz"))
    path = _checkpoint()
    if not archives or path is None:
        pytest.skip("no archived update or checkpoint of the training run")
    theirs, ours, _ = _nets(research, path)
    with np.load(archives[-2] if len(archives) > 1 else archives[-1]) as archive:  # the newest may be half-written
        z = {k: archive[k] for k in (*FIELDS, "turn", "forced", "start", "length")}
    a_games, b_games = _games(z, research.features.Obs, 40), _games(z, former_features.Obs, 40)
    with torch.no_grad():
        a, b = theirs(research.model.stack(a_games, "cpu")), ours(former.stack(b_games, "cpu"))
        for key in OUTPUTS:
            assert torch.equal(a[key], b[key]), key
        worst = 0.0
        for g, game in enumerate(b_games[:10]):
            cache: dict = {}
            for t, obs in enumerate(game):
                out = ours.step_cached(obs, cache)
                worst = max(worst, float((out["logits"] - b["logits"][g, t]).abs().max()))
                worst = max(worst, float((out["value"] - b["value"][g, t]).abs().max()))
    decisions = sum(len(g) for g in b_games)
    record_property("decisions", decisions)
    record_property("worst_step_difference", worst)
    assert decisions > 200 and worst < 1e-3


def _engine():
    from pcsa.common.config import SimConfig
    from pcsa.common.errors import ConfigError
    from pcsa.sim.engine import Engine

    try:
        return Engine.locate(SimConfig().format)
    except ConfigError:
        return None


class Twin:
    """Plays our FormerPlayer and checks research's observations and network against it on the same views."""

    name = "twin"

    def __init__(self, research, theirs_net, ours_net, cfg) -> None:
        from pcsa.rl_training.former_player import FormerPlayer

        self.research, self.theirs = research, theirs_net
        self.ours = FormerPlayer(ours_net, cfg, temperature=0.0, rng=np.random.default_rng(5), greedy_preview=True)
        self.cache: dict = {}
        self.tag = None
        self.decisions = self.previews = 0
        self.failures: list[str] = []
        self.battles: set[str] = set()

    def params(self) -> dict:
        return {}

    def notes(self) -> dict:
        return {}

    def same(self, a, b) -> bool:
        return all(np.array_equal(getattr(a, k), getattr(b, k)) for k in FIELDS)

    def choose_team(self, view, deadline) -> list[int]:
        import torch

        from pcsa.rl_training import former_features, preview

        picks = preview.all_picks()
        mine = former_features.preview_observations(view.battle, picks)
        want = self.research.features.preview_observations(view.battle, picks)
        self.check(
            len(mine) == len(want) and all(self.same(a, b) for a, b in zip(mine, want, strict=True)),
            "preview obs",
            view,
        )
        with torch.no_grad():
            values = self.theirs(self.research.model.stack([[o] for o in want], "cpu"))["value"][:, 0].numpy()
        self.check(np.array_equal(values, self.ours.preview_values(view.battle, picks)), "preview values", view)
        self.previews += 1
        return self.ours.choose_team(view, deadline)

    def choose_action(self, view, deadline):
        import torch

        tag = view.battle.battle_tag
        if tag != self.tag:
            self.tag, self.cache = tag, {}
        self.battles.add(tag)
        proposal = self.ours.propose(view)
        want = self.research.features.observe(view.battle, view.legal)
        self.check(self.same(want, proposal.obs), "observation", view)
        self.check((want.turn, want.forced) == (proposal.obs.turn, proposal.obs.forced), "turn or forced", view)
        out = self.theirs.step_cached(want, self.cache)
        same = all(torch.equal(out[k], proposal.out[k]) for k in OUTPUTS)
        self.check(same, "network outputs", view)
        self.decisions += 1
        return self.ours.commit(proposal)

    def check(self, ok: bool, what: str, view) -> None:
        if not ok:
            self.failures.append(f"{what} differs at turn {view.battle.turn} of {view.battle.battle_tag}")


def test_observations_and_policy_match_research_in_simulator_battles(research, config, record_property) -> None:
    from pcsa.battle.builds import Catalog
    from pcsa.players.arena import ArenaPlayer, ExportTeams, close, run_match
    from pcsa.players.factory import build_player
    from pcsa.sim.server import ShowdownServer
    from pcsa.sim.teams import sample_teams

    engine, path = _engine(), _checkpoint()
    if engine is None or path is None:
        pytest.skip("the simulator is not installed, or no research checkpoint")
    theirs_net, ours_net, cfg = _nets(research, path)
    twin = Twin(research, theirs_net, ours_net, cfg)
    rng = random.Random(3)
    teams = sample_teams(Catalog.load(), engine, rng, 4)
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
