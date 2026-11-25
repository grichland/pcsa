"""Preview search's race and choice (pcsa.network.preview_race), and parity with pcsa-former's reference search:
the same fake games give the same cell sums, cuts, final rows and choice."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pcsa.common.config_cg import PreviewSearch  # noqa: E402
from pcsa.common.paths import repo_root  # noqa: E402
from pcsa.network import layout as L  # noqa: E402
from pcsa.network import preview_race as race  # noqa: E402
from pcsa.network import preview_search as previewing  # noqa: E402

FORMER = Path(os.environ.get("PCSA_FORMER") or repo_root().parent / "pcsa-former")
P, N = race.P, race.P * race.P
GOOD, HEAD = 17, 5  # the clearly best row of the fake games; the head's pick


def test_the_cuts_give_the_configured_sizes_and_keep_the_head() -> None:
    r = race.Race(N, HEAD, [20, 6, 2])
    T = np.tile(np.linspace(1, 0, P)[:, None], (1, P))  # row 0 best, the head's row 5 below the final two
    for k, elapsed in enumerate((1.1, 2.1, 3.1)):
        assert r.cut_due(elapsed, 4.0) and not r.cut_due(elapsed - 0.5, 4.0)
        r.cut(T, elapsed)
        assert HEAD in r.rows and len(r.rows) == [20, 6, 3][k]
    assert list(r.rows) == [0, 1, HEAD] and not r.cut_due(10.0, 4.0)


def test_ended_slots_cycle_over_the_rows_in_the_race() -> None:
    r = race.Race(N, HEAD, [2])
    r.rows = np.array([3, HEAD])
    r.reassign(np.arange(4))
    assert list(r.cell[:4]) == [3 * P, 3 * P + 1, 3 * P + 2, 3 * P + 3]
    r.reassign(np.arange(4, 4 + 2 * P - 4))
    assert r.cell[4 + 2 * P - 5] == HEAD * P + P - 1


def test_a_race_from_open_picks_plays_only_those_and_the_head() -> None:
    r = race.Race(2 * P + 3, HEAD, [2], rows=[3, 9])
    assert list(r.rows) == [3, HEAD, 9] and set(r.cell // P) == {3, HEAD, 9}
    assert list(r.cell[:2]) == [3 * P, 3 * P + 1] and r.cell[P] == HEAD * P and r.cell[2 * P] == 9 * P and r.ptr == 0
    assert (race.Race(N, HEAD, [2]).cell == np.arange(N)).all()


def test_the_head_is_kept_unless_beaten_by_the_margin() -> None:
    T = np.full((P, P), 0.5)
    T[GOOD], T[HEAD] = 0.6, 0.58
    rows = np.array([HEAD, GOOD])
    assert race.choose(T, rows, HEAD, "mean", 0.03)[0] == HEAD
    T[GOOD] = 0.62
    pick, why = race.choose(T, rows, HEAD, "mean", 0.03)
    assert pick == GOOD and why["best"] == GOOD and why["head_value"] == 0.58
    T[GOOD, 0] = 0.0  # its worst cell
    assert race.choose(T, rows, HEAD, "maximin", 0.0)[0] == HEAD


class Games:
    """Fake games: our preview, theirs, our first turn, then the end. We win when our pick is GOOD, else one game
    in three. The clock moves only when the games do."""

    def __init__(self, n: int) -> None:
        self.n, self.phase, self.ours, self.k = n, np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        self.done, self.winner, self.now = np.zeros(n, bool), np.zeros(n, int), 0.0

    def kind_seat(self) -> tuple[np.ndarray, np.ndarray]:
        kind = np.where(self.phase < 2, L.KIND_PREVIEW, L.KIND_TURN)
        return kind, np.where(self.phase == 1, 1, 0)

    def act(self, action: np.ndarray) -> None:
        a = np.asarray(action)
        self.ours = np.where(self.phase == 0, a - L.PREVIEW_OFFSET, self.ours)
        self.done = self.phase == 2
        won = (self.ours == GOOD) | ((self.ours + self.k) % 3 == 0)
        self.winner = np.where(won, 0, 1)
        self.k += self.done
        self.phase = np.where(self.done, 0, self.phase + 1)
        self.now += 0.01


class RefEnv:
    """The fake games as pcsa-former's CGEnv shows them to the reference."""

    def __init__(self, games: Games) -> None:
        self.g = games

    def observe(self) -> dict:
        kind, seat = self.g.kind_seat()
        return {"kind": kind, "seat": seat}

    def act(self, action) -> None:
        self.g.act(action)

    def results(self) -> dict:
        e = np.flatnonzero(self.g.done)
        return {"env": e, "winner": self.g.winner[e]}


class OurEnv(RefEnv):
    """The same games as Champions Gym's BatchEnv shows them to pcsa."""

    def observe_tensors(self) -> SimpleNamespace:
        kind, seat = self.g.kind_seat()
        n = self.g.n
        zeros = {k: np.zeros((n, 1), np.float32) for k in ("rows", "row_ids", "field", "last", "last_ids", "opp_prior")}
        return SimpleNamespace(kind=kind, seat=seat, mask=np.ones((n, L.OWN_ACTIONS), np.uint8),
                               action_mask=np.ones((n, L.PREVIEW_OFFSET + P), np.uint8), **zeros)  # fmt: skip

    def results(self) -> SimpleNamespace:
        return SimpleNamespace(done=self.g.done, winner=self.g.winner)


def fake_out(n: int) -> dict:
    preview = torch.zeros(n, P)
    preview[:, HEAD] = 1.0
    return {"action": torch.zeros(n, dtype=torch.long), "score": torch.zeros(n, L.OWN_ACTIONS),
            "legal": torch.ones(n, L.OWN_ACTIONS, dtype=torch.bool), "preview_pi": preview}  # fmt: skip


@pytest.fixture
def reference(monkeypatch):
    """pcsa-former's tools/preview_search_test.py."""
    tools = FORMER / "tools" / "preview_search_test.py"
    if not tools.is_file():
        pytest.skip(f"no pcsa-former checkout at {FORMER}")
    monkeypatch.syspath_prepend(str(FORMER / "tools"))
    sys.modules.pop("preview_search_test", None)
    return importlib.import_module("preview_search_test")


def test_the_same_cuts_and_choice_as_the_reference(reference, monkeypatch) -> None:
    budget, margin = 2.0, 0.03
    ref_games, our_games = Games(N), Games(N)
    clock = {"ref": ref_games, "ours": our_games}
    players = SimpleNamespace(step=lambda obs: fake_out(N), reset=lambda e: None)
    monkeypatch.setattr(reference.cg_adapter, "CGEnv", lambda *a, **k: RefEnv(ref_games))
    monkeypatch.setattr(reference, "Players", lambda *a, **k: players)
    monkeypatch.setattr(reference, "time", SimpleNamespace(perf_counter=lambda: clock["ref"].now))
    args = SimpleNamespace(search_seconds=budget, threads=1, prior_games=2.0)
    ref = reference.search(None, "cpu", "unused.json", args, 0, guesses=3, halving=True)

    monkeypatch.setattr(previewing, "open_env", lambda *a, **k: OurEnv(our_games))
    monkeypatch.setattr(previewing, "precise_now", lambda: clock["ours"].now)
    monkeypatch.setattr(previewing, "model", SimpleNamespace(Memory=lambda *a: SimpleNamespace(reset=lambda s: None),
                                                             INPUTS=previewing.model.INPUTS))  # fmt: skip
    cfg = PreviewSearch(enabled=True, budget_s=budget, guesses=3, race=[20, 6, 2], prior_games=2.0, choose="mean",
                        keep_head_margin=margin, device="cpu", threads=1, seed=0, worlds_mode="legacy")  # fmt: skip
    p = previewing.Previewer.__new__(previewing.Previewer)
    p.policy, p.cfg = SimpleNamespace(rule_cfg=None, layout=None), cfg
    p.net = SimpleNamespace(act=lambda b, kind, **k: fake_out(N))
    ours = p.play([[], [], [], []], HEAD, budget, 0)

    for k in ("a_sum", "a_cnt", "wins", "games"):
        assert np.allclose(np.asarray(ours["cells"][k]), np.round(ref[k], 4)), k
    assert ours["rows"] == [int(r) for r in ref["rows"]] and ref["head"] == HEAD
    B = reference.table(ref["a_sum"], ref["a_cnt"], ref["wins"], ref["games"], 2.0)
    assert np.allclose(B, race.table(ref["a_sum"], ref["a_cnt"], ref["wins"], ref["games"], 2.0))
    fin = ref["rows"]
    best = int(fin[B[fin].mean(1).argmax()])  # the reference's H_guard, as main() makes it
    guard = best if B[best].mean() - B[HEAD].mean() >= margin else HEAD
    assert race.choose(B, np.asarray(ours["rows"]), HEAD, "mean", margin)[0] == guard == GOOD
