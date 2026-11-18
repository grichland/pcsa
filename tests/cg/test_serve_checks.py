"""The content of `view` and `net` on recorded battles (battle 1 of neural-148 from tests/cg/data; battles 2 and 3
from output/runs/20261118-162900-neural-148 when it is here), replayed with their checkpoint:
- `rows[action]` is the move or species played, and `action >= 4` exactly when the order Mega Evolves
- `cols[0:4]` are the opposing active's revealed moves, then one "" while fewer than four are known, then None;
  `cols[8:13]` the other five opposing members
- every believed move is learnable by its species
- `view` agrees with the Battle, member by member
- probabilities are in 0..1; `lead` is set only at the preview, where it sums to 1."""

from __future__ import annotations

import json
from functools import cache

import pytest

pytest.importorskip("torch")
cg = pytest.importorskip("champions_gym")

from pcsa.cg.serve import NEURAL, Server, make_player  # noqa: E402
from pcsa.common.paths import repo_root  # noqa: E402
from pcsa.network import vocab  # noqa: E402
from tests.cg import neural, recorded  # noqa: E402

RUN = repo_root() / "output" / "runs" / "20261118-162900-neural-148"
BATTLES = ["fixture", "battle-2", "battle-3"]


@cache
def player():
    if neural.missing() or neural.CHECKPOINT.name != recorded.CHECKPOINT_NAME:
        pytest.skip(str(neural.missing() or f"the recording was made with {recorded.CHECKPOINT_NAME}"))
    return make_player(NEURAL, {**neural.player_args(), "policy_args": {"threads": 2}})


def messages(name: str) -> list[dict]:
    if name == "fixture":
        return recorded.sent()
    path = RUN / name / "cg-messages.jsonl"
    if not path.is_file():
        pytest.skip(f"{path} is not here")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r["message"] for r in rows if r["dir"] == "sent"]


@cache
def decided(name: str) -> list[tuple[dict, object]]:
    """(reply, the Battle it was decided on) for every request the network decided."""
    server = Server(player())
    out = []
    for message in messages(name):
        reply = server.handle(message)
        if reply is not None and "net" in (reply.get("notes") or {}):
            assert server.proxy is not None
            out.append((reply, server.proxy.battle))
    assert out
    return out


def turns(name: str) -> list[tuple[dict, object]]:
    return [(r, b) for r, b in decided(name) if "order" in r]


def names(p) -> set:
    return {p.species, p.form} - {None}


@pytest.mark.parametrize("name", BATTLES)
def test_rows_name_what_was_played(name: str) -> None:
    for reply, battle in turns(name):
        action, order, rows = reply["notes"]["action"], reply["order"], reply["notes"]["net"]["rows"]
        if order["kind"] == "move":
            assert rows[action] == order["move"] and (action >= 4) == bool(order["mega"]), (reply["seq"], rows)
        else:
            member = next(p for p in battle.team if p.species == order["mon"])
            assert action >= 8 and rows[action] in names(member), (reply["seq"], rows)


@pytest.mark.parametrize("name", BATTLES)
def test_cols_are_their_active_moves_then_their_bench(name: str) -> None:
    for reply, _ in turns(name):
        cols, opp = reply["notes"]["net"]["cols"], reply["notes"]["view"]["opp"]
        active = [m for m in opp if m["active"]]
        assert len(active) == 1
        known = [m for m, _ in active[0]["moves"]]
        want = known + ([""] if len(known) < 4 else []) + [None] * 4
        assert cols[0:4] == want[:4] and cols[4:8] == cols[0:4], (reply["seq"], cols, known)
        bench = [m for m in opp if not m["active"]]
        assert len(cols[8:13]) == 5 and all(c is not None for c in cols[8:13])
        assert sorted(cols[8:13]) == sorted(m["form"] or m["species"] for m in bench), (reply["seq"], cols)


@pytest.mark.parametrize("name", BATTLES)
def test_believed_moves_are_learnable(name: str) -> None:
    table = cg.learnset_table()
    species = {n: i for i, n in enumerate(vocab.tables()["species"]) if n}
    moves = {n: i for i, n in enumerate(vocab.tables()["moves"]) if n}
    checked = 0
    for reply, battle in decided(name):
        for e in reply["notes"]["net"]["belief"]:
            if e["species"] is None:
                continue
            forms = {f for p in battle.opponent_team if e["species"] in names(p) for f in names(p)} | {e["species"]}
            for move, _ in e["moves"]:
                assert any(table[species[f]][moves[move]] for f in forms if f in species), (e["species"], move)
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("name", BATTLES)
def test_view_agrees_with_the_battle(name: str) -> None:
    for reply, battle in decided(name):
        view = reply["notes"]["view"]
        for got, p in zip(view["own"] + view["opp"], list(battle.team) + list(battle.opponent_team), strict=True):
            assert got["species"] == p.species and got["form"] == p.form
            assert got["hp"] == pytest.approx(p.current_hp_fraction) and got["status"] == (p.status or "")
            assert got["fainted"] == p.fainted and got["active"] == p.active and got["item"] == p.item
            assert got["moves"] == [[m, pp] for m, pp in p.moves] and got["ability"] == p.ability
        assert view["turn"] == battle.turn and view["picks"] == [*battle.picks, None, None, None][:3]


@pytest.mark.parametrize("name", BATTLES)
def test_probabilities(name: str) -> None:
    for reply, _ in decided(name):
        net = reply["notes"]["net"]
        preview = "picks" in reply
        for e in net["belief"]:
            assert 0.0 <= e["brought"] <= 1.0
            assert all(0.0 <= p <= 1.0 for _, p in e["item"] + e["ability"] + e["moves"])
            assert (e["lead"] is not None) == preview and (e["lead"] is None or 0.0 <= e["lead"] <= 1.0)
        if preview:
            assert sum(e["lead"] for e in net["belief"]) == pytest.approx(1.0, abs=1e-3)
            assert all(0.0 <= p <= 1.0 for _, p in net["picks_top"])
        else:
            assert all(0.0 <= p <= 1.0 for p in net["pi"] + net["p_opp"])
            assert sum(net["pi"]) == pytest.approx(1.0, abs=1e-3) and sum(net["p_opp"]) == pytest.approx(1.0, abs=1e-3)
