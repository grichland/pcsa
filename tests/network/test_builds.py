"""Whole-build guesses (pcsa.network.builds) and their evidence (pcsa.network.evidence): the catalog, the filters
on shown facts and legal abilities, Item Clause, and turn order moving the weights."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pcsa.network import builds, evidence, vocab, worlds

pytest.importorskip("champions_gym")


@pytest.fixture(scope="module")
def guesser() -> builds.Guesser:
    from pcsa.network import cg_adapter

    return builds.Guesser(worlds.Tables(cg_adapter.learnset().numpy(), {}))


def mon(species: str, moves=(), item=None, ability=None, form=None) -> SimpleNamespace:
    return SimpleNamespace(species=species, form=form, moves=[(m, 8) for m in moves], item=item, ability=ability)


def flat_belief(g: builds.Guesser, species: list[str]) -> worlds.Belief:
    """A belief head that prefers nothing: every build is weighed by its prior alone."""
    n = len(species)
    return worlds.Belief(species, np.full(n, 0.5), np.full((n, len(g.tables.moves)), 1.0),
                         np.full((n, len(g.tables.items)), 1.0), np.full((n, len(g.tables.abilities)), 1.0))  # fmt: skip


def test_speed_is_cgs_stat_formula() -> None:
    jolly, neutral = ("spe", "spa"), (None, None)
    assert evidence.stat_from_points(102, 32, "spe", jolly) == 169  # Jolly Garchomp, 32 points
    assert evidence.stat_from_points(102, 0, "spe", neutral) == 122
    assert evidence.stat_from_points(108, 2, "hp", neutral) == 185
    assert evidence.effective(169, 1, "choicescarf", par=False, tailwind=False) == 379  # 253 at +1, x1.5 floored
    assert evidence.effective(169, 0, "", par=True, tailwind=True) == 169


def test_the_catalog_merges_both_sources_and_adds_sightings() -> None:
    seen = {"species": {"garchomp": [{"count": 3, "item": "Focus Sash", "ability": "Rough Skin", "nature": "Jolly",
                                      "points": {"spe": 32}, "moves": ["Earthquake", "Outrage"]}]}}
    top = {"species": {"Garchomp": {"builds": [
        {"item": "Focus Sash", "ability": "Rough Skin", "nature": "Jolly", "points": {"spe": 32},
         "moves": ["Outrage", "Earthquake"], "score": -1.0},
        {"item": "Choice Scarf", "ability": "Rough Skin", "nature": "Jolly", "points": {"spe": 32},
         "moves": ["Outrage"], "score": -1.0}]}}}  # fmt: skip
    cat = builds.load_catalog(seen, top)["garchomp"]
    assert len(cat) == 2 and cat[0].prior == pytest.approx(3.5) and cat[1].prior == pytest.approx(0.5)
    assert cat[0].points[5] == 32 and cat[0].moves == ("earthquake", "outrage")


def test_a_shown_fact_is_never_contradicted(guesser) -> None:
    p = mon("garchomp", moves=["earthquake"], item="garchompitez", form="garchompmegaz")
    belief = flat_belief(guesser, ["garchomp"])
    for seed in range(40):
        (s,), _ = guesser.guess_team([(0, p, "garchomp")], belief, None, np.random.default_rng(seed))
        assert "earthquake" in s["moves"] and s["item"] == "garchompitez"
    known = mon("garchomp", ability="roughskin")
    cands, w, n = guesser.weigh(known, "garchomp", belief, None)
    assert n > 0 and all(c is None or c.ability == "roughskin" for c in cands) and w.sum() == pytest.approx(1.0)


def test_an_item_an_event_named_counts_after_it_is_gone(guesser) -> None:
    tracker = evidence.Tracker(guesser.dex)
    tracker.feed([{"k": "item", "side": "opp", "mon": "garchomp", "item": "focussash", "what": "used"}])
    cands, _, n = guesser.weigh(mon("garchomp", item=""), "garchomp", flat_belief(guesser, ["garchomp"]), tracker)
    assert n > 0 and all(c is None or c.item == "focussash" for c in cands)


def test_item_clause_holds_and_a_teammates_shown_item_is_not_drawn(guesser) -> None:
    team = ["garchomp", "gholdengo", "rotomwash"]
    belief = flat_belief(guesser, team)
    sash = mon("gholdengo", item="focussash")
    for seed in range(60):
        sets, _ = guesser.guess_team([(0, mon("garchomp"), "garchomp"), (1, sash, "gholdengo"),
                                      (2, mon("rotomwash"), "rotomwash")], belief, None, np.random.default_rng(seed),
                                     shown_items=["focussash"])  # fmt: skip
        items = [s["item"] for s in sets if s.get("item")]
        assert len(items) == len(set(items)) and sets[1].get("item", "focussash") == "focussash"  # shown: the Game keeps it
        assert sets[0].get("item") != "focussash" and sets[2].get("item") != "focussash"


def test_only_legal_abilities_appear(guesser) -> None:
    belief = flat_belief(guesser, ["garchomp"])
    belief.ability[0][:] = 0.0
    belief.ability[0][guesser.ability_ix["levitate"]] = 1.0  # the head says an ability Garchomp cannot have
    legal = guesser.dex.abilities["garchomp"]
    for seed in range(40):
        (s,), _ = guesser.guess_team([(0, mon("garchomp"), "garchomp")], belief, None, np.random.default_rng(seed))
        assert s["ability"] in legal
    out = {"ability": "levitate"}
    guesser._legal_ability(out, mon("garchomp"), "garchomp", belief)
    assert out["ability"] in legal


def test_turn_order_moves_the_weight_to_faster_builds(guesser) -> None:
    fast = builds.Build("choicescarf", "roughskin", "jolly", (0, 0, 0, 0, 0, 32), ("outrage",), 1.0, "t")
    slow = builds.Build("leftovers", "roughskin", "brave", (0, 0, 0, 0, 0, 0), ("outrage",), 1.0, "t")
    own = [{"species": "Gholdengo", "nature": "Timid", "points": {"spe": 32}, "item": "Leftovers", "ability": ""}]
    tracker = evidence.Tracker(guesser.dex, own=own)
    tracker.feed([{"k": "turn", "n": 1}, {"k": "enter", "side": "own", "mon": "gholdengo"},
                  {"k": "enter", "side": "opp", "mon": "garchomp"}, {"k": "turn", "n": 2},
                  {"k": "move", "side": "opp", "mon": "garchomp", "move": "outrage"},
                  {"k": "move", "side": "own", "mon": "gholdengo", "move": "shadowball"}, {"k": "turn", "n": 3}])  # fmt: skip
    assert len(tracker.orders) == 1 and tracker.orders[0].first
    assert tracker.order_loglik("garchomp", fast) > tracker.order_loglik("garchomp", slow)
    assert tracker.order_loglik("garchomp", slow) == pytest.approx(np.log(evidence.ORDER_EPS))


def test_turns_that_say_nothing_about_speed_are_left_out(guesser) -> None:
    own = [{"species": "Gholdengo", "nature": "Timid", "points": {"spe": 32}, "item": "", "ability": ""}]
    tracker = evidence.Tracker(guesser.dex, own=own)
    tracker.feed([{"k": "turn", "n": 1}, {"k": "enter", "side": "own", "mon": "gholdengo"},
                  {"k": "move", "side": "opp", "mon": "kingambit", "move": "suckerpunch"},  # priority +1
                  {"k": "move", "side": "own", "mon": "gholdengo", "move": "shadowball"}, {"k": "turn", "n": 2},
                  {"k": "field", "effect": "trickroom", "what": "start"},
                  {"k": "move", "side": "opp", "mon": "kingambit", "move": "ironhead"},
                  {"k": "move", "side": "own", "mon": "gholdengo", "move": "shadowball"}, {"k": "turn", "n": 3}])  # fmt: skip
    assert tracker.orders == []


def test_mega_stones_are_in_the_vocabulary() -> None:
    assert "garchompitez" in vocab.mega_stones()


def test_serve_feeds_the_evidence_and_writes_the_battle_end_ranks(tmp_path) -> None:
    from pcsa.cg import serve

    calls: list = []
    searcher = SimpleNamespace(start_battle=lambda own: calls.append(("start", own)),
                               events=lambda ev: calls.append(("events", len(ev))),
                               end_battle=lambda: [{"species": "garchomp", "rank": 1}])  # fmt: skip
    server = SimpleNamespace(policy=SimpleNamespace(searcher=searcher), log_dir=str(tmp_path))
    serve.Server._evidence(server, '{"type": "battle_start", "own_team": [1]}')  # type: ignore[arg-type]
    serve.Server._evidence(server, {"type": "events", "events": [{"k": "turn"}]})  # type: ignore[arg-type]
    serve.Server._evidence(server, '{"type": "decision_request", "kind": "turn"}')  # type: ignore[arg-type]
    serve.Server._evidence(server, {"type": "battle_end", "battle": "b1"})  # type: ignore[arg-type]
    assert calls == [("start", [1]), ("events", 1)]
    assert '"rank": 1' in (tmp_path / "builds-b1.json").read_text(encoding="utf-8")
