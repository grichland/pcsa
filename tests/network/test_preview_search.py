"""Preview search on the real network (pcsa.network.preview_search), on CPU with a small number of slots: off is the
head's pick, forced cells lead as numbered, deadlines and errors fall back to the head, and (marked gpu) the timing
on this machine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cg = pytest.importorskip("champions_gym")

from pcsa.common.clock import precise_now  # noqa: E402
from pcsa.common.config_cg import PreviewSearch  # noqa: E402
from pcsa.common.paths import repo_root  # noqa: E402
from pcsa.network import layout as L  # noqa: E402
from pcsa.network import model, vocab  # noqa: E402
from pcsa.network import preview_search as previewing  # noqa: E402
from tests.cg import neural  # noqa: E402
from tests.cg.neural import CHECKPOINT  # noqa: E402

SETTINGS = {"budget_s": 1.5, "guesses": 2, "race": [20, 6, 2], "prior_games": 2.0, "choose": "mean",
            "keep_head_margin": 0.03, "device": "cpu", "threads": 2, "seed": 0, "worlds_mode": "builds"}  # fmt: skip
pytestmark = pytest.mark.skipif(not CHECKPOINT.is_file(), reason=f"no test checkpoint at {CHECKPOINT}")


def teams() -> list[list[dict]]:
    return json.loads(Path(cg.TEAMS).read_text(encoding="utf-8"))[:2]


def own_team(team: list[dict]) -> list[dict]:
    """A teams-file team as battle_start gives our six (stat points as `points`)."""
    return [{**{k: v for k, v in m.items() if k != "evs"}, "points": m["evs"]} for m in team]


def preview_request() -> Any:
    """Seat 0's preview request of a game between the two teams."""
    ours, theirs = teams()
    env = previewing.open_env([ours, theirs], 1, 3, 1)
    t = env.observe_tensors()
    assert int(t.kind[0]) == L.KIND_PREVIEW and int(t.seat[0]) == 0
    return t


def decide(policy: Any, arrays: Any) -> tuple[int, dict]:
    out = policy.decide(arrays, [0])
    return int(out["scores"][0, L.PREVIEW_OFFSET :].argmax()), out["extras"][0]


@pytest.fixture(scope="module")
def searched() -> Any:
    policy = neural.policy(preview_search=SETTINGS)
    policy.previewer.own_team = own_team(teams()[0])
    return policy


def test_off_is_the_heads_pick_exactly(searched, monkeypatch) -> None:
    assert PreviewSearch.model_validate({"enabled": False, **SETTINGS}).args() is None
    policy, arrays = neural.policy(), preview_request()
    off = policy.decide(arrays, [0])
    assert "preview_search" not in off["extras"][0] and policy.info["preview_search"] is None
    monkeypatch.setattr(searched.previewer, "own_team", None)  # the search fails: the head's pick, unchanged
    fell_back = searched.decide(arrays, [0])
    assert torch.equal(off["scores"], fell_back["scores"])
    assert int(off["scores"][0, L.PREVIEW_OFFSET :].argmax()) == fell_back["extras"][0]["preview_search"]["head"]


def test_a_forced_cell_reaches_turn_one_with_the_right_leads(searched) -> None:
    ours, theirs = teams()
    p, n = searched.previewer, 2 * L.PREVIEW_ACTIONS
    env = previewing.open_env([ours, theirs], n, 5, 2)
    state, mem = p.start(n, 0), model.Memory(p.net, 2 * n, "cpu")
    state["race"].cell = np.arange(n) * 61 % (L.PREVIEW_ACTIONS**2)  # cells with many different picks
    combos, names, seen = L.preview_combos(), vocab.tables()["species"], set()
    species = lambda i: names[int(i)]  # noqa: E731
    lay = searched.layout
    for _ in range(12):
        t = env.observe_tensors()
        ids, rows = np.asarray(t.row_ids), np.asarray(t.rows)
        for g in np.flatnonzero((np.asarray(t.kind) == L.KIND_TURN) & (np.asarray(t.seat) == 0)):
            if g in seen:
                continue
            seen.add(g)
            cell = state["race"].cell[g]
            me = rows[g, : L.OWN_PICKS, lay.col(L.F_IS_ACTIVE)].argmax()
            assert species(ids[g, me, lay.col(L.F_SPECIES)]) == previewing.worlds.to_id(ours[combos[cell // 60][0]]["species"])
            assert species(ids[g, L.OPP_ACTIVE, lay.col(L.F_SPECIES)]) == previewing.worlds.to_id(
                theirs[combos[cell % 60][0]]["species"])  # fmt: skip
        p._step(env, mem, state)
    assert len(seen) == n


def test_the_search_answers_within_its_budget_and_its_pick_is_played(searched, monkeypatch) -> None:
    monkeypatch.setattr(previewing, "SLOTS", 60)  # few slots: fast steps, so games finish on a loaded CPU
    cfg = searched.previewer.cfg
    monkeypatch.setattr(searched.previewer, "cfg", cfg.model_copy(update={"budget_s": 6.0}))  # CPU: games finish
    t0 = precise_now()
    pick, extras = decide(searched, preview_request())
    note = extras["preview_search"]
    assert precise_now() - t0 < 6.0 + 3.0 and "error" not in note, note.get("error")
    assert pick == note["chosen"] and note["games"] > 0 and note["head"] in note["rows"]
    assert len(note["guesses"]) == SETTINGS["guesses"] and len(note["cells"]["wins"]) == L.PREVIEW_ACTIONS**2
    assert len(note["cuts"]) <= len(SETTINGS["race"])
    closed = [a - L.PREVIEW_OFFSET for a in extras["closed"]]  # the lineup rule's picks: never raced or played
    games = np.asarray(note["cells"]["games"]).reshape(L.PREVIEW_ACTIONS, -1)
    assert closed and not set(closed) & set(note["rows"]) and games[closed].sum() == 0


def test_a_deadline_or_an_error_plays_the_heads_pick_and_says_why(searched, monkeypatch) -> None:
    monkeypatch.setattr(previewing, "SLOTS", 240)
    arrays = preview_request()
    searched.deadline = precise_now() + 1.0  # less than the reserve: no time to play
    pick, extras = decide(searched, arrays)
    note = extras["preview_search"]
    assert pick == note["head"] == note["chosen"] and "no game finished" in note["error"] and note["seconds"] < 1.0
    searched.deadline = None
    monkeypatch.setattr(searched.previewer, "own_team", None)
    pick, extras = decide(searched, arrays)
    assert pick == extras["preview_search"]["head"] and "no own team" in extras["preview_search"]["error"]
    broken = neural.policy(preview_search={**SETTINGS, "device": "cuda:7"})
    assert broken.info["preview_search"]["start_error"]
    pick, extras = decide(broken, arrays)
    assert pick == extras["preview_search"]["head"] and extras["preview_search"]["error"].startswith("RuntimeError: start")


@pytest.mark.gpu
def test_the_full_budget_on_this_machines_gpu_reaches_ten_thousand_games(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA here")
    policy = neural.policy(preview_search={**SETTINGS, "budget_s": 30.0, "guesses": 3, "device": "cuda", "threads": 8})
    policy.previewer.own_team = own_team(teams()[0])
    note = decide(policy, preview_request())[1]["preview_search"]
    timing = {k: note.get(k) for k in ("games", "steps", "seconds", "play_seconds", "head", "chosen", "error")}
    Path(repo_root() / "output" / "preview-search").mkdir(parents=True, exist_ok=True)
    (repo_root() / "output" / "preview-search" / "timing-test.json").write_text(json.dumps(timing), encoding="utf-8")
    assert "error" not in note and note["games"] >= 10_000 and note["seconds"] < 31.0
