"""Guard rules as action masks (neural_rules.masks, rule_mode: mask): every legal action is put to the rules; the ones
a rule would replace are masked, the rest stay for the network."""

from __future__ import annotations

import json

import pytest

from pcsa.battle.state import replay
from pcsa.players import mirror, neural_rules
from pcsa.players.neural_rules import Spot
from tests.conftest import REPO
from tests.players.test_disciplined import HITS, FakeThreat

MEGA = REPO / "tests" / "fixtures" / "mega_rule" / "case29-mega-into-flamethrower.json"
YAWN = REPO / "tests" / "fixtures" / "yawn_rule" / "garchomp-drowsy-vs-hippowdon.json"


@pytest.fixture
def threat(monkeypatch):
    def install(hits: dict | None = None, **kwargs) -> FakeThreat:
        estimates = FakeThreat({**HITS, **(hits or {})}, **kwargs)
        monkeypatch.setattr(neural_rules, "threat", estimates)
        return estimates

    return install


def position(path, with_their: bool = True):
    rec = json.loads(path.read_text(encoding="utf-8"))
    b = replay(rec["protocol"], rec["request"])
    legal = mirror.legal_actions(b)
    their = None
    if with_their:
        m = rec["notes"]["matrix"]
        their = tuple(
            ("switch", "", p) if k == "switch" else ("move", c, p)
            for c, p, k in zip(m["cols"], m["p_cols"], m["col_kind"], strict=True)
            if c
        )
    return b, legal, Spot(b, tuple(legal), legal[0], list(legal), None, None, their, 0.5)


def test_mega_weakness_masks_only_the_mega_variant(threat) -> None:
    threat()
    _, legal, spot = position(MEGA)
    masked = neural_rules.masks(["mega_weakness"], spot)
    assert "/choose move firstimpression mega" in masked
    assert masked["/choose move firstimpression mega"][0] == "mega_weakness"
    assert "/choose move firstimpression" not in masked
    assert all("mega" in m for m in masked)
    assert any(a.message not in masked for a in legal)


def test_yawn_switch_masks_the_moves_and_leaves_every_switch(threat) -> None:
    threat()
    _, legal, spot = position(YAWN, with_their=False)
    masked = neural_rules.masks(["yawn_switch"], spot)
    moves = {a.message for a in legal if "/choose move" in a.message}
    switches = {a.message for a in legal if "/choose switch" in a.message}
    assert moves and switches and set(masked) == moves
    assert {rule for rule, _ in masked.values()} == {"yawn_switch"}


def test_the_first_rule_that_forbids_an_action_names_it(threat) -> None:
    threat()
    _, _, spot = position(YAWN, with_their=False)
    both = neural_rules.masks(["yawn_switch", "no_full_heal"], spot)
    assert {rule for rule, _ in both.values()} == {"yawn_switch"}
    assert neural_rules.masks([], spot) == {}


def test_priority_ko_masks_the_others_but_never_its_own_move(threat) -> None:
    threat(priority={("golisopod", "slowkinggalar"): "firstimpression"})
    _, legal, spot = position(MEGA)
    masked = neural_rules.masks(["priority_ko"], spot)
    assert "/choose move firstimpression" not in masked and "/choose move firstimpression mega" not in masked
    others = [a.message for a in legal if "firstimpression" not in a.message]
    assert others and all(m in masked for m in others)
