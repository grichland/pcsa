"""The playout team preview (pcsa.players.preview_playout, preview_sampler): their sampled lineups, the halving
search on a scripted pool, the deadline and error fallbacks, the settings."""

from __future__ import annotations

import json
import queue
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pcsa.battle.builds import Build
from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.errors import ConfigError
from pcsa.players import preview_playout as pp
from pcsa.players import preview_sampler as sampler
from pcsa.players.base import TeamPreviewView
from pcsa.players.neural import NeuralParams
from pcsa.rl_training import preview
from tests.rl_training.battles import OURS, THEIRS, preview_battle


class FakeCatalog:
    """Two builds per name, none for the names in `missing`; `forms` are the only entries of `by_id`."""

    seen = None

    def __init__(self, missing=(), forms=None) -> None:
        self.missing, self.by_id = set(missing), dict(forms or {})

    def builds(self, name: str) -> list[Build]:
        key = name.lower().replace("-", "")
        if name in self.missing or (self.by_id and key not in self.by_id and name != "Indeedee"):
            return []
        if key in self.by_id:
            return self.by_id[key]
        return [build(name, "Leftovers"), build(name, f"{name} Berry")]

    def species(self) -> list[str]:
        return ["Snorlax", "Blissey", "Tyranitar", "Dragonite"]


def build(name: str, item: str) -> Build:
    return Build(name, item, "Pressure", "Jolly", (2, 32, 0, 0, 0, 32), ("Tackle", "Protect"))


def test_the_lead_prior_is_a_smoothed_rate(tmp_path: Path) -> None:
    path = tmp_path / "lead_prior.json"
    path.write_text(json.dumps({"games": 40, "species": {"rotom": {"lead": 15, "six": 35}}}), encoding="utf-8")
    prior = sampler.LeadPrior.load(path)
    assert prior.games == 40 and prior.rate("Rotom-Heat") == pytest.approx((15 + 0.5) / 38)
    assert prior.rate("Garchomp") == pytest.approx(1 / 6)
    p = prior.probabilities(["Rotom", "Garchomp", "Milotic"])
    assert p.sum() == pytest.approx(1.0) and p[0] > p[1] == p[2]
    empty = sampler.LeadPrior.load(tmp_path / "missing.json")
    assert np.allclose(empty.probabilities(THEIRS), 1 / 6)


def test_gumbel_top_draws_without_replacement_in_proportion() -> None:
    rng = np.random.default_rng(0)
    scores = np.log([0.7, 0.2, 0.1])
    first = [sampler.gumbel_top(scores, 1, rng)[0] for _ in range(4000)]
    assert abs(first.count(0) / 4000 - 0.7) < 0.03 and abs(first.count(2) / 4000 - 0.1) < 0.02
    trio = sampler.gumbel_top(np.zeros(6), 3, rng)
    assert len(set(trio)) == 3 and all(0 <= i < 6 for i in trio)


def test_their_lineup_is_six_with_the_trio_first_and_one_of_each_item() -> None:
    rng = np.random.default_rng(1)
    lineup = sampler.their_lineup(THEIRS, sampler.LeadPrior({}), FakeCatalog(missing={"Volcarona"}), rng)
    assert lineup is not None and lineup.no_build == 1 and "Volcarona" not in lineup.species
    blocks = lineup.export.strip().split("\n\n")
    names = [b.split(" @ ")[0] for b in blocks]
    assert len(blocks) == 6 and tuple(names[:3]) == lineup.species and "Volcarona" not in names
    assert "Snorlax" in names
    items = [b.splitlines()[0].split(" @ ")[1] for b in blocks]
    assert len(set(items)) == 6
    assert sampler.their_lineup(THEIRS, sampler.LeadPrior({}), FakeCatalog(missing=set(THEIRS[:4])), rng) is None


def test_a_preview_name_without_builds_uses_its_formes() -> None:
    forms = {
        "rotomheat": [build("Rotom-Heat", "A"), build("Rotom-Heat", "B")],
        "rotomwash": [build("Rotom-Wash", "C")],
    }
    got = sampler.builds_of(FakeCatalog(forms=forms), "Rotom")
    assert [b.species for b in got] == ["Rotom-Heat", "Rotom-Wash", "Rotom-Heat"]
    assert sampler.builds_of(FakeCatalog(forms=forms), "Indeedee Female")[0].species == "Indeedee"
    assert sampler.builds_of(FakeCatalog(forms=forms), "Mudsdale") == []


# The search ---------------------------------------------------------------------------------------------------
class FakeWorker:
    def __init__(self, pool) -> None:
        self.pool = pool

    def send(self, task: dict) -> bool:
        self.pool.sent.append(task)
        self.pool.clock.advance(self.pool.seconds)
        self.pool.results.put({"id": task["id"], **self.pool.answer(task)})
        return True


class FakePool:
    """Answers every task at once: `answer(task)` -> {"won": bool} or {"error": text}; each costs `seconds`."""

    def __init__(self, clock: FakeClock, answer, seconds: float = 0.1, workers: int = 2) -> None:
        self.clock, self.answer, self.seconds, self.n = clock, answer, seconds, workers
        self.results: queue.Queue = queue.Queue()
        self.sent: list[dict] = []

    def idle(self) -> list:
        return [FakeWorker(self)] if self.n else []

    def ready(self) -> int:
        return self.n

    def errors(self) -> list[str]:
        return []


class Policy:
    def __init__(self, scores, rows) -> None:
        self.preview_scores = (np.asarray(scores, float), rows)


def setup(answer, *, params=None, seconds=0.1, workers=2, export="x"):
    clock = FakeClock()
    pool = FakePool(clock, answer, seconds, workers)
    parts = {"pool": pool, "clock": clock, "rng": np.random.default_rng(0), "catalog": FakeCatalog()}
    search = pp.PlayoutPreview(params or pp.PlayoutParams(noise=0.0), {}, prior=sampler.LeadPrior({}), **parts)
    battle = preview_battle()
    view = TeamPreviewView(battle, tuple(OURS), random.Random(0), export)
    picks = preview.all_picks(6, 3)
    scores = np.linspace(1.0, 0.0, len(picks))
    return search, pool, view, picks, Policy(scores, list(range(len(picks)))), clock


def test_halving_plays_the_schedule_and_takes_the_lineup_that_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pp, "run_folder", lambda: tmp_path)
    search, pool, view, picks, policy, clock = setup(lambda task: {"won": task["pick"] == 5})
    team, notes = search.refine(policy, view, Deadline.after(clock, 80.0), list(picks[0]), {"rule": "rl", "v": 0.4})
    assert team == list(picks[5]) and notes["playout_changed"] and notes["rule"] == "rl" and notes["v"] == 0.4
    assert notes["playout_rounds"] == ["16x3:48", "8x5:40", "4x8:32", "2x14:28"] and notes["playout_done"] == 148
    assert notes["playout_stop"] == "done" and notes["playout_failed"] == 0
    assert notes["playout_finalists"][0].endswith("30/30") and notes["playout_robust"].startswith("Azumarill/")
    assert all(not isinstance(v, dict) and (not isinstance(v, list) or len(v) <= 16) for v in notes.values())
    first = [t for t in pool.sent if t["round"] == 0]
    assert {t["pick"] for t in first} == set(range(16))
    assert len({(t["theirs"], t["seed"]) for t in first}) == 3
    assert all(t["their_pick"] == [0, 1, 2] and t["ours"] == "x" for t in pool.sent)
    saved = [json.loads(line) for line in (tmp_path / "preview-playouts.jsonl").read_text("utf-8").splitlines()]
    assert len(saved) == 149 and saved[0]["six"] and sum("won" in r for r in saved) == 148


def test_the_robust_pick_stays_without_a_clear_winner(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pp, "run_folder", lambda: tmp_path)
    search, _, view, picks, policy, clock = setup(lambda task: {"won": True})
    team, notes = search.refine(policy, view, Deadline.after(clock, 80.0), list(picks[0]), {"rule": "rl"})
    assert team == list(picks[0]) and not notes["playout_changed"]
    few = pp.PlayoutParams(noise=0.0, schedule=[[4, 2], [2, 3]])
    search, _, view, picks, policy, clock = setup(lambda task: {"won": task["pick"] == 1}, params=few)
    team, _ = search.refine(policy, view, Deadline.after(clock, 80.0), list(picks[0]), {"rule": "rl"})
    assert team == list(picks[0])


def test_the_search_stops_at_the_deadline_with_the_best_so_far(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pp, "run_folder", lambda: tmp_path)
    search, _, view, picks, policy, clock = setup(lambda task: {"won": task["pick"] == 2}, seconds=0.25)
    team, notes = search.refine(policy, view, Deadline.after(clock, 33.0), list(picks[0]), {"rule": "rl"})
    assert notes["playout_done"] < 148 and notes["playout_s"] <= 30.5
    assert notes["playout_stop"] == "deadline" and notes["playout_rounds"][:2] == ["16x3:48", "8x5:40"]
    assert team == list(picks[2])
    search, _, view, picks, policy, clock = setup(lambda task: {"won": task["pick"] == 2}, seconds=0.5)
    team, notes = search.refine(policy, view, Deadline.after(clock, 33.0), list(picks[0]), {"rule": "rl"})
    assert notes["playout_rounds"][1] == "8x1:8" and team == list(picks[0])
    search, _, view, picks, policy, clock = setup(lambda task: {"won": task["pick"] == 2}, seconds=2.0)
    team, notes = search.refine(policy, view, Deadline.after(clock, 9.0), list(picks[0]), {"rule": "rl"})
    assert team == list(picks[0]) and notes["playout_done"] <= 3


def test_failed_playouts_are_counted_and_do_not_score(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pp, "run_folder", lambda: tmp_path)
    search, _, view, picks, policy, clock = setup(lambda task: {"error": "Popup: team rejected"})
    team, notes = search.refine(policy, view, Deadline.after(clock, 80.0), list(picks[0]), {"rule": "rl"})
    assert team == list(picks[0]) and notes["playout_failed"] == 148 and notes["playout_done"] == 0


def test_anything_wrong_keeps_the_robust_pick_and_says_why(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pp, "run_folder", lambda: tmp_path)
    search, _, view, picks, policy, clock = setup(lambda task: {"won": True}, workers=0)
    team, notes = search.refine(policy, view, Deadline.after(clock, 80.0), list(picks[7]), {"rule": "rl"})
    assert team == list(picks[7]) and "no playout worker is ready" in notes["playout_fallback"]
    search, _, view, picks, policy, clock = setup(lambda task: {"won": True}, export=None)
    team, notes = search.refine(policy, view, Deadline.after(clock, 80.0), list(picks[7]), {"rule": "rl"})
    assert team == list(picks[7]) and "no team export" in notes["playout_fallback"] and notes["preview"] == "playout"


def test_settings() -> None:
    assert NeuralParams().preview_playout == {} and NeuralParams().preview == "critic"
    good = replace(NeuralParams(), preview="playout", max_preview_s=60.0)
    good.check("test")
    replace(good, preview_playout={"workers": 6, "schedule": [[8, 2], [2, 6]]}).check("test")
    with pytest.raises(ConfigError, match="max_preview_s"):
        replace(NeuralParams(), preview="playout", max_preview_s=20.0).check("test")
    for bad in ({"colour": 1}, {"schedule": [[2, 2], [8, 2]]}, {"schedule": []}, {"workers": 0}, {"budget_s": -1}):
        with pytest.raises(ConfigError, match="preview_playout"):
            replace(good, preview_playout=bad).check("test")
    assert pp.PlayoutParams.load(None, "test").schedule == [[16, 3], [8, 5], [4, 8], [2, 14]]
