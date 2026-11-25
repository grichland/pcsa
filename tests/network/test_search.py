"""Decision-time search (pcsa.network.search, worlds and searching; the policy's `search` setting): the tree
arithmetic on small nodes, the guess at a hidden set, and the policy with search on and off. The parts that build an
engine Game need Champions Gym 0.5.0 or later and are skipped without it."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cg = pytest.importorskip("champions_gym")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pcsa.network import layout as L  # noqa: E402
from pcsa.network import search, worlds  # noqa: E402
from pcsa.network.policy import Policy  # noqa: E402
from pcsa.network.searching import Follow, Searcher  # noqa: E402
from tests import configs  # noqa: E402
from tests.cg import neural  # noqa: E402
from tests.cg.neural import CHECKPOINT  # noqa: E402

HAS_GAMES = hasattr(cg.env, "game_from_battle")
needs_games = pytest.mark.skipif(not HAS_GAMES, reason="needs Champions Gym 0.5.0 (a Game from a Battle)")


def make(search: dict | None = None, **settings) -> Policy:
    """The test policy; `search` replaces some of the shipped search settings and turns the search on."""
    if not CHECKPOINT.is_file():
        pytest.skip(f"no test checkpoint at {CHECKPOINT}")
    return neural.policy(search=None if search is None else configs.search(**search), **settings)


@pytest.fixture(scope="module")
def position():
    """A fixed game at a turn request of seat 0 with two moves open: (arrays, Battle)."""
    env = cg.BatchEnv(None, 1, seed=3, threads=1)
    while True:
        t = env.observe_tensors()
        if int(t.kind[0]) == cg.KIND_TURN and int(t.seat[0]) == 0 and int(np.asarray(t.mask)[0, :4].sum()) >= 2:
            return t, env.battle(0), env
        env.act(env.builtin_actions("max_damage"))


def own(out: dict) -> list[float]:
    return out["scores"][0, :11].tolist()


# The arithmetic -----------------------------------------------------------------------------------------


def test_the_strategy_follows_regret_and_the_prior_fades_with_visits() -> None:
    prior = np.array([0.9, 0.1])
    fresh = search.strategy(np.zeros(2), prior, np.zeros(2), 0, 1.5)
    assert fresh == pytest.approx([0.9, 0.1])
    tried = search.strategy(np.zeros(2), prior, np.array([50.0, 0.0]), 50, 1.5)
    assert tried[1] > fresh[1]  # the untried action gains
    regret = search.strategy(np.array([0.0, 30.0]), prior, np.array([500.0, 500.0]), 1000, 1.5)
    assert regret[1] > 0.95
    assert search.strategy(np.zeros(3), np.zeros(3), np.zeros(3), 0, 1.5) == pytest.approx([1 / 3] * 3)


def test_a_backup_moves_the_cell_and_the_regrets_of_both_sides() -> None:
    node = SimpleNamespace(Q=np.array([[0.2, -0.4], [0.6, 0.0]]), n=np.ones((2, 2)), R_u=np.zeros(2), R_t=np.zeros(2))
    search.Search.backup([(node, 0, 1)], 0.8)
    assert node.n[0, 1] == 2 and node.Q[0, 1] == pytest.approx(0.2)  # the mean of the start (-0.4) and 0.8
    assert node.R_u == pytest.approx([0.0, 0.0])  # Q[:, 1] - u = [-0.6, -0.8], kept at 0
    assert node.R_t == pytest.approx([0.6, 0.6])  # u - Q[0, :] = [0.6, 0.6]: they did better elsewhere
    search.Search.backup([(node, 0, 1)], -1.0)
    assert node.Q[0, 1] == pytest.approx(-0.2) and node.R_u[1] == pytest.approx(1.0)  # our other row beats -1


def test_their_actions_map_to_the_network_columns() -> None:
    tables = SimpleNamespace(move_id={"surf": 5, "icebeam": 6, "protect": 7, "toxic": 8})
    world = worlds.World([{"moves": ["surf", "icebeam", "protect", "toxic"]}, {"moves": []}, {"moves": []}],
                         ["swampert", "garchomp", "rotomwash"], None)  # fmt: skip
    leaf = SimpleNamespace(named=np.array([6, 5, 0, 0]), bench=["rotomwash", None, "garchomp", None, None],
                           belief_slots=False)  # fmt: skip
    cols, share = search.their_columns(leaf, world, 0, [0, 1, 2, 3, 5, 9, 10, None], tables)  # type: ignore[arg-type]
    assert cols[:2] == [1, 0] and share[:2] == [1.0, 1.0]  # by move id, whatever their order
    assert cols[2:4] == [2, 2] and share[2:4] == [0.5, 0.5]  # two unseen moves share the one empty column
    assert cols[4] == 4  # icebeam with Mega: its column + 4
    assert cols[5:7] == [10, 8] and cols[7] == -1  # switches by species; no action, no column
    leaf.belief_slots, leaf.named = True, np.array([6, 5, 7, 0])
    cols, share = search.their_columns(leaf, world, 0, [2, 3], tables)  # type: ignore[arg-type]
    assert cols == [2, -1] and share == [1.0, 1.0]  # named columns: an unmatched move has none


def test_a_guessed_set_keeps_what_was_shown_and_stays_in_the_learnset() -> None:
    names = worlds.vocab.tables()
    species = next(i for i, n in enumerate(names["species"]) if n == "garchomp")
    moves = len(names["moves"])
    learnset = np.zeros((len(names["species"]), moves), bool)
    can = [i for i, n in enumerate(names["moves"]) if n in ("earthquake", "dragonclaw", "swordsdance", "stoneedge", "rockslide")]
    learnset[species, can] = True
    tables = worlds.Tables(learnset, {"garchomp": {"moves": ["surf"], "nature": "jolly", "points": {"atk": 32}}})
    prob = np.full((6, moves), 1.0 / moves)
    belief = worlds.Belief(["garchomp"] + [None] * 5, np.full(6, 0.5), prob, np.full((6, len(names["items"])), 0.01),
                           np.full((6, len(names["abilities"])), 0.01))  # fmt: skip
    mon = SimpleNamespace(species="garchomp", form=None, moves=[("earthquake", 10)], item="", ability=None)
    for seed in range(5):
        g = worlds.guess_set(mon, 2, "garchomp", belief, tables, np.random.default_rng(seed))
        assert g["slot"] == 2 and g["moves"][0] == "earthquake" and len(g["moves"]) == 4 == len(set(g["moves"]))
        assert all(tables.move_id[m] in can for m in g["moves"]) and "surf" not in g["moves"]
        assert "item" not in g and g["ability"] and g["nature"] == "jolly" and g["points"] == {"atk": 32}


def test_a_refused_part_is_taken_out_of_the_guess() -> None:
    guess = [{"slot": 0, "moves": ["surf", "hydropump", "toxic"], "item": "leftovers"}, {"slot": 1, "moves": ["tackle"]}]
    worlds.repair(guess, ["rotom", "goodra"], [("rotom", "moves", "rotom cannot learn hydropump"), ("rotom", "item", "x"),
                                               ("goodra", "brought", "it led: put first")])  # fmt: skip
    assert guess[0] == {"slot": 0, "moves": ["surf", "toxic"]} and guess[1]["moves"] == ["tackle"]
    shown = [{"slot": 3, "moves": ["thunderbolt", "hydropump", "voltswitch"]}]
    worlds.repair(
        shown, ["rotom"], [("rotom", "moves", "rotom cannot learn hydropump voltswitch")], {3: {"hydropump"}}
    )
    assert shown[0]["moves"] == ["thunderbolt", "hydropump"]  # a move the Battle showed stays: that guess is dropped


# The policy ---------------------------------------------------------------------------------------------


def test_search_off_adds_nothing(position) -> None:
    arrays, battle, _ = position
    plain = make()
    assert plain.info["search"] is None
    out = plain.decide(arrays, [0], [battle])
    assert "search" not in out["extras"][0]


def test_without_a_budget_the_models_answer_stands(position) -> None:
    arrays, battle, _ = position
    want = own(make().decide(arrays, [0], [battle]))
    for settings in ({"ms": 1.0}, {"sims": 32, "min_sims": 64}):
        if "sims" in settings and not HAS_GAMES:
            continue
        out = make(settings).decide(arrays, [0], [battle])
        note = out["extras"][0]["search"]
        assert own(out) == want and note["used"] is False and note["why"] and note["changed"] is False
        assert note["pick"] == note["model_pick"] == int(np.argmax(want))
        json.dumps(note, allow_nan=False)


def test_turns_before_from_turn_are_left_to_the_model(position) -> None:
    arrays, battle, _ = position
    want = own(make().decide(arrays, [0], [battle]))
    out = make({"sims": 64, "from_turn": int(battle.turn) + 1}).decide(arrays, [0], [battle])
    note = out["extras"][0]["search"]
    assert own(out) == want and note["used"] is False and "the search starts at turn" in note["why"]
    late = make({"sims": 64, "from_turn": int(battle.turn)}).decide(arrays, [0], [battle])
    assert "the search starts at turn" not in (late["extras"][0]["search"]["why"] or "")


def test_before_champions_gym_05_the_note_says_so(position, monkeypatch) -> None:
    arrays, battle, _ = position
    monkeypatch.delattr(cg.env, "game_from_battle", raising=False)
    want = own(make().decide(arrays, [0], [battle]))
    out = make({"sims": 64}).decide(arrays, [0], [battle])
    note = out["extras"][0]["search"]
    assert own(out) == want and note["used"] is False and "Champions Gym 0.5.0" in note["why"]


def test_a_failing_search_never_changes_the_answer(position, monkeypatch) -> None:
    arrays, battle, _ = position
    want = own(make().decide(arrays, [0], [battle]))
    policy = make({"sims": 64})

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(worlds, "build_worlds", boom)
    out = policy.decide(arrays, [0], [battle])
    assert own(out) == want and out["extras"][0]["search"]["why"] == "RuntimeError: boom"
    assert own(make({}).decide(arrays, [0], None)) == want  # no Battle: the model's answer


@needs_games
def test_the_search_answers_with_its_root_strategy(position) -> None:
    arrays, battle, _ = position
    policy = make({"sims": 160, "min_sims": 64, "worlds": 4})
    out = policy.decide(arrays, [0], [battle])
    note = out["extras"][0]["search"]
    json.dumps(out["extras"][0], allow_nan=False)
    assert note["used"] is True and note["why"] is None and note["sims"] == 160 and 1 <= note["worlds"] <= 4
    legal = [a for a in range(11) if np.asarray(arrays.mask)[0, a]]
    strategy, scores = note["strategy"], own(out)
    assert sum(strategy) == pytest.approx(1.0, abs=1e-6) and all(strategy[a] == 0 for a in range(11) if a not in legal)
    assert note["pick"] == int(np.argmax(strategy)) == int(np.argmax(scores)) and note["pick"] in legal
    assert all((scores[a] == float("-inf")) == (strategy[a] == 0) for a in range(11))
    assert note["changed"] == (note["pick"] != note["model_pick"]) and note["depth"] >= 1 and note["nodes"] > 0
    assert len(note["q"]) == 11 and all(len(line) == 13 for line in note["q"]) and len(note["p_them"]) == 13
    assert sum(note["p_them"]) + note["p_other"] == pytest.approx(1.0, abs=1e-6)
    assert all(-1.0 <= x <= 1.0 for line in note["q"] for x in line if x is not None)
    assert policy.info["search"]["worlds"] == 4 and out["extras"][0]["pi"]  # the model's own outputs stay


@needs_games
def test_the_search_is_the_same_for_a_seed_and_respects_the_rule_masks(position) -> None:
    arrays, battle, _ = position
    notes = []
    for seed in (7, 7, 8):
        note = make({"sims": 96, "seed": seed}).decide(arrays, [0], [battle])["extras"][0][
            "search"
        ]
        notes.append({k: note[k] for k in ("strategy", "q", "p_them", "visits", "sims", "nodes", "depth", "pick")})
    assert notes[0] == notes[1] and notes[0]["strategy"] != notes[2]["strategy"]
    policy = make({"sims": 96}, rules=configs.rules(max_same_move=1))
    move = int(np.flatnonzero(np.asarray(arrays.mask)[0, :4])[0])
    policy.decide(arrays._replace(decision=np.array([900])), [0], [battle])
    policy.played(900, move)
    out = policy.decide(arrays._replace(decision=np.array([901])), [0], [battle])
    note = out["extras"][0]["search"]
    assert move in out["extras"][0]["closed"] and note["used"] and note["strategy"][move] == 0 and note["pick"] != move


@needs_games
def test_the_search_stops_inside_its_time(position) -> None:
    arrays, battle, _ = position
    policy = make({"ms": 250.0, "min_sims": 1})
    policy.decide(arrays, [0], [battle])  # warm
    note = policy.decide(arrays._replace(decision=np.array([77])), [0], [battle])["extras"][0]["search"]
    assert note["used"] and note["ms"] <= 250.0 + 60.0 and note["budget_ms"] == 250.0
    from pcsa.common.clock import precise_now

    policy.deadline = precise_now() + 0.45  # the request's own deadline is nearer than the `ms` setting
    note = policy.decide(arrays._replace(decision=np.array([78])), [0], [battle])["extras"][0]["search"]
    assert note["budget_ms"] < 250.0 and note["ms"] <= note["budget_ms"] + 60.0 and policy.deadline is None


def test_the_likeliest_actions_are_kept_in_their_own_order() -> None:
    assert search._likeliest([0, 1, 2, 8], [0.1, 0.5, 0.0, 0.4], 2) == [1, 8]
    assert search._likeliest([0, 1, 2], [0.2, 0.2, 0.6], 2) == [0, 2]  # a tie: the earlier action
    assert search._likeliest([0, 1], [0.5, 0.5], 0) == [0, 1] and search._likeliest([3], [float("nan")], 1) == [3]


@needs_games
def test_top_ours_and_top_theirs_bound_what_the_search_looks_at(position) -> None:
    arrays, battle, _ = position
    wide = make({"sims": 128}).decide(arrays, [0], [battle])["extras"][0]
    out = make({"sims": 128, "top_ours": 2, "top_theirs": 2}).decide(arrays, [0], [battle])["extras"][0]
    note = out["search"]
    assert note["used"] and sum(x > 0 for x in note["strategy"]) <= 2 and sum(x > 0 for x in note["visits"]) <= 2
    top2 = sorted(range(11), key=lambda a: -out["pi"][a])[:2]
    assert all(note["strategy"][a] == 0 for a in range(11) if a not in top2)
    assert sum(x > 0 for x in note["p_them"]) <= 2 * 6  # two of their actions per guess of their team
    assert sum(x > 0 for x in wide["search"]["p_them"]) >= sum(x > 0 for x in note["p_them"])
    assert make({"top_theirs": 3}).info["search"]["top_theirs"] == 3


class _Root:
    def __init__(self, strategy: list, q: list) -> None:
        self.us, self.S_u, self.P_u = list(range(len(strategy))), np.array(strategy, float), np.array(strategy, float)
        self.S_t, self.P_t, self.Q = np.array([1.0]), np.array([1.0]), np.array(q, float).reshape(-1, 1)


class _World:
    def __init__(self, strategy: list, q: list, sims: int) -> None:
        self.root, self.sims = _Root(strategy, q), sims


def test_the_search_is_settled_when_one_action_leads_or_the_best_two_are_level() -> None:
    cfg = search.SearchSettings(**configs.search(settle_sims=100, settle_share=0.9, settle_gap=0.01))
    tree = search.Search(cfg, None, None, np.random.default_rng(0))  # type: ignore[arg-type]
    ahead = [_World([0.95, 0.05], [0.7, 0.4], 100), _World([0.93, 0.07], [0.6, 0.5], 100)]
    assert tree.settled(ahead, 100) is None  # it has led since simulation 100 only
    assert tree.settled(ahead, 200) == "clear"
    open_ = [_World([0.6, 0.4], [0.7, 0.4], 200)]
    assert search.Search(cfg, None, None, np.random.default_rng(0)).settled(open_, 200) is None  # type: ignore[arg-type]
    level = [_World([0.6, 0.4], [0.500, 0.495], 200)]
    assert search.Search(cfg, None, None, np.random.default_rng(0)).settled(level, 200) == "even"  # type: ignore[arg-type]


@needs_games
def test_a_settled_search_stops_before_its_simulations_run_out(position) -> None:
    arrays, battle, _ = position
    policy = make({"sims": 2000, "settle_sims": 64, "settle_share": 0.0, "settle_gap": 0.0})
    note = policy.decide(arrays, [0], [battle])["extras"][0]["search"]
    assert note["used"] and note["stopped"] == "clear" and note["sims"] < 2000
    assert policy.info["search"]["settle_sims"] == 64
    full = make({"sims": 128}).decide(arrays, [0], [battle])["extras"][0]["search"]
    assert full["stopped"] == "sims" and full["sims"] == 128


# Following a pivot move ------------------------------------------------------------------------------------


class _Game:
    def __init__(self, turn: int, kinds: tuple, fainted: list | None = None) -> None:
        self.turn, self.ended, self._kinds = turn, False, kinds
        self._ro = SimpleNamespace(active=np.array([0, 0]), fainted=np.array([fainted or [0, 0, 0], [0, 0, 0]]))

    def kinds(self) -> tuple:
        return self._kinds

    def readout(self) -> SimpleNamespace:
        return self._ro


def _node(game: _Game, us: list, s_u: list | None = None, n: int = 0) -> search.Node:
    node = search.Node(game)
    node.pending, node.us, node.N = False, us, n
    node.S_u = np.array(s_u or [0.0] * len(us))
    node.P_u = np.full(len(us), 1.0 / len(us))
    return node


def _pivot_world(children: dict) -> SimpleNamespace:
    """A root with our actions [0, 2] (2 the pivot move) and their one action; `children` per our action index."""
    root = _node(_Game(5, (1, 1)), [0, 2], [1.0, 1.0], 2)
    root.kids = {(x, 0): ([1.0] * len(kids), [None] * len(kids), dict(enumerate(kids))) for x, kids in children.items()}
    return SimpleNamespace(root=root, sims=10)


def test_the_followed_switch_in_is_the_one_the_trees_chose_after_the_pivot() -> None:
    pivot = _node(_Game(5, (L.KIND_REPLACE, -1)), [9, 10], [2.0, 6.0], 8)
    faint = _node(_Game(5, (L.KIND_REPLACE, -1), [1, 0, 0]), [9, 10], [7.0, 0.0], 7)  # our active fainted first
    later = _node(_Game(6, (L.KIND_TURN, L.KIND_TURN)), [0, 9], [5.0, 5.0], 10)  # the next turn
    other = _node(_Game(5, (L.KIND_REPLACE, -1)), [9, 10], [9.0, 0.0], 9)  # under our other action
    worlds = [_pivot_world({1: [pivot, faint, later], 0: [other]}),
              _pivot_world({1: [_node(_Game(5, (L.KIND_REPLACE, -1)), [9, 10], [1.0, 3.0], 4)]})]
    follow = search.pivot_follow(worlds, 2)  # type: ignore[arg-type]
    assert follow == {"target": 10, "share": pytest.approx(9 / 12), "pivot_visits": 12, "faint_visits": 7}
    assert search.pivot_follow(worlds, 0) is not None  # any pick the trees saw switch us out
    assert search.pivot_follow([_pivot_world({1: [later]})], 2) is None  # type: ignore[arg-type]
    assert search.pivot_follow(worlds, 4) is None  # type: ignore[arg-type]


def test_an_unvisited_switch_in_falls_back_to_the_network_prior() -> None:
    fresh = _node(_Game(5, (L.KIND_REPLACE, -1)), [8, 10])
    fresh.P_u = np.array([0.3, 0.7])
    assert search.pivot_follow([_pivot_world({1: [fresh]})], 2)["target"] == 10  # type: ignore[arg-type,index]


def _follower() -> SimpleNamespace:
    return SimpleNamespace(follows={})


def _replace(open_: list) -> tuple:
    mask = torch.zeros(1, 11)
    mask[0, open_] = 1
    scores = torch.full((1, 71), -5.0)
    scores[0, 8] = -0.1  # the network's own switch-in
    return np.array([L.KIND_REPLACE]), {"mask": mask}, scores


def test_the_replace_request_after_the_pivot_plays_the_followed_switch_in() -> None:
    kinds, b, scores = _replace([8, 10])
    p = _follower()
    p.follows = {0: Follow(3, 5, 2, 10)}
    note = Searcher.follow(p, [0], kinds, b, [SimpleNamespace(turn=5)], scores)[0]  # type: ignore[arg-type]
    assert note == {"used": True, "why": None, "target": 10, "model_pick": 8, "changed": True}
    assert int(scores[0, :11].argmax()) == 10 and not p.follows
    assert torch.isinf(scores[0, :11]).sum() == 10


def test_the_followed_switch_in_is_dropped_when_it_cannot_apply() -> None:
    for open_, turn, why in (([8, 9], 5, "not open"), ([8, 10], 6, "the pivot move was on turn 5")):
        kinds, b, scores = _replace(open_)
        p = _follower()
        p.follows = {0: Follow(3, 5, 2, 10)}
        note = Searcher.follow(p, [0], kinds, b, [SimpleNamespace(turn=turn)], scores)[0]  # type: ignore[arg-type]
        assert note["used"] is False and why in note["why"] and int(scores[0, :11].argmax()) == 8 and not p.follows


def test_a_pivot_move_that_did_not_go_out_drops_its_switch_in() -> None:
    p = _follower()
    p.follows = {0: Follow(3, 5, 2, 10), 1: Follow(4, 5, 2, 9)}
    Searcher.played(p, 3, 2)  # type: ignore[arg-type]
    assert set(p.follows) == {0, 1}
    Searcher.played(p, 4, 7)  # type: ignore[arg-type]
    assert set(p.follows) == {0}


@needs_games
def test_the_policy_plays_the_kept_switch_in_at_a_replace_request() -> None:
    env = cg.BatchEnv(None, 1, seed=5, threads=1)
    for _ in range(400):
        t = env.observe_tensors()
        if int(t.kind[0]) == L.KIND_REPLACE and int(t.seat[0]) == 0 and int(np.asarray(t.mask)[0].sum()) >= 2:
            break
        env.act(env.builtin_actions("max_damage"))
    else:
        pytest.skip("no replace request with two switch-ins in this game")
    battle = env.battle(0)
    policy = make({"sims": 64})
    plain = int(torch.as_tensor(policy.decide(t, [0], [battle])["scores"])[0, :11].argmax())
    target = next(a for a in np.flatnonzero(np.asarray(t.mask)[0]) if a != plain)
    policy.searcher.follows[0] = Follow(int(t.decision[0]) - 1, int(battle.turn), 2, int(target))  # type: ignore[union-attr]
    out = policy.decide(t, [0], [battle])
    assert int(out["scores"][0, :11].argmax()) == target
    assert out["extras"][0]["follow"] == {"used": True, "why": None, "target": target, "model_pick": plain,
                                          "changed": True}  # fmt: skip
    policy.reset([0])
    assert not policy.searcher.follows  # type: ignore[union-attr]
