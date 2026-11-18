"""pcsa.cg.serve with the neural player in this process (torch, CPU): the recorded battle 1 of neural-148 replayed
with its checkpoint gives the recorded picks and orders, `net` on every decided reply (and on the preview, with
the lead belief and picks_top), and no `net` on a repeated request."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("champions_gym")

from pcsa.cg.serve import NEURAL, Server, make_player  # noqa: E402
from tests.cg import neural, recorded  # noqa: E402

NET_KEYS = {"decision", "kind", "legal", "value", "tau", "pi", "score", "p_opp", "matrix", "rows", "cols", "belief"}


@pytest.fixture(scope="module")
def player():
    if neural.missing() or neural.CHECKPOINT.name != recorded.CHECKPOINT_NAME:
        pytest.skip(str(neural.missing() or f"the recording was made with {recorded.CHECKPOINT_NAME}"))
    return make_player(NEURAL, {**neural.player_args(), "policy_args": {"threads": 2}})


def test_recorded_battle_replays_the_same_with_net(player) -> None:
    server = Server(player)
    want = recorded.answers()
    got = {}
    for message in recorded.sent():
        reply = server.handle(message)
        json.dumps(reply, allow_nan=False)  # plain JSON
        if reply is not None and ("picks" in reply or "order" in reply):
            got["preview" if "picks" in reply else reply["seq"]] = reply
    assert set(got) == set(want)
    assert got["preview"]["picks"] == want["preview"]["picks"]
    assert {s: r["order"] for s, r in got.items() if s != "preview"} == {
        s: r["order"] for s, r in want.items() if s != "preview"
    }
    net = got["preview"]["notes"]["net"]
    assert net["decision"] == -1 and net["kind"] == "preview" and {"belief", "picks_top", "value"} <= set(net)
    assert all(e["lead"] is not None for e in net["belief"]) and net["picks_top"][0][0] == want["preview"]["picks"]
    opp = {m["species"] for m in got["preview"]["notes"]["view"]["opp"]}
    for seq, reply in got.items():
        if seq == "preview":
            continue
        net = reply["notes"]["net"]
        assert set(net) == NET_KEYS and net["decision"] == seq and net["legal"] == reply["notes"]["legal"]
        assert net["value"] == reply["notes"]["model"]["value"]
        assert {e["species"] for e in net["belief"]} <= opp  # named by species, as in view.opp (a Mega too)


def test_no_net_on_a_repeated_request(player) -> None:
    server = Server(player)
    messages = recorded.sent()
    first_turn = next(i for i, m in enumerate(messages) if m["type"] == "decision_request")
    for message in messages[: first_turn + 1]:
        reply = server.handle(message)
    assert reply is not None and "net" in reply["notes"]
    again = server.handle(messages[first_turn])
    assert again is not None and again["order"] == reply["order"]
    assert "net" not in again["notes"] and "view" not in again["notes"]
