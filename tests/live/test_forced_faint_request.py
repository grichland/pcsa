"""The request of a faint replacement (live/turn.py forced, turn_sync.fainted_active): the fainted active goes first,
active: true, "0 fnt", as Showdown lists a forced-switch request, so the network's members[0] is the fainted mon and
its bench rows are the live ones. Before 15 Nov 2026 the request kept pick order: when the fainted mon was not pick
0, a live mon was read as our active, two switch orders shared one action index and legal[0] was sent whatever the
network chose (152 of 455 live replacements, 12-15 Nov)."""

from __future__ import annotations

import json

from pcsa.battle import state as tracking
from pcsa.battle.teams import OwnMember, Team
from pcsa.battle.translate import Translation
from pcsa.live import orders
from pcsa.live.turn_sync import fainted_active, index_collisions
from pcsa.phone.log_merge import BattleLog
from pcsa.players.base import ScreenFacts
from pcsa.rl_training import features
from pcsa.run.battle_state import BattleRecord
from tests.conftest import REPO

EXAMPLE = REPO / "tests" / "fixtures" / "forced_faint" / "battle10-0003-forced.json"
FORCED = ScreenFacts((), False, True, None)
PRELUDE = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|teamsize|p1|3", "|teamsize|p2|3", "|start"]
STATS = {"hp": 160, "atk": 120, "def": 100, "spa": 100, "spd": 100, "spe": 100}
MOVES = {
    "Cinderace": ("Pyro Ball", "High Jump Kick", "Gunk Shot", "U-turn"),
    "Golisopod": ("Leech Life", "First Impression", "Sucker Punch", "Iron Head"),
    "Primarina": ("Sparkling Aria", "Aqua Jet", "Calm Mind", "Moonblast"),
}


def _team(names=tuple(MOVES)) -> Team:
    return Team(
        None, tuple(OwnMember(i, n, n, None, None, MOVES[n], dict(STATS), None, None) for i, n in enumerate(names))
    )


def team_from_request(request: dict) -> tuple[Team, list[int]]:
    """Our three from a saved forced request (pick order when nobody is marked active)."""
    members = []
    for i, p in enumerate(request["side"]["pokemon"]):
        name = p["ident"].split(": ", 1)[1]
        species = p["details"].split(",")[0]
        members.append(
            OwnMember(
                i,
                name,
                species,
                p.get("ability") or None,
                p.get("item") or None,
                tuple(p["moves"]),
                dict(p["stats"]),
                None,
                None,
            )
        )
    return Team(None, tuple(members)), list(range(len(members)))


def _record(names, visible_order, fainted) -> BattleRecord:
    b = BattleRecord(1, "id", tuple(names), BattleLog("id"), requested=[0, 1, 2])
    b.visible_order = list(visible_order)
    b.fainted = set(fainted)
    return b


def _forced(team, picks, protocol, active, fainted, names):
    """(Battle with the forced request applied, legal actions, request), the way forced() builds them."""
    battle = tracking.replay(protocol, None)
    request = tracking.battle_request(
        team, picks, battle, active=active, screen_hp={}, fainted=set(fainted), forced=True
    )
    out = Translation()
    tracking.apply_request(battle, out, request)
    assert not out.warnings
    legal, _ = orders.battle_actions(
        battle, FORCED, slot_moves=[], names=list(names), fainted=set(fainted), active=None, leaving=None
    )
    return battle, legal, request


def _rows_match(battle, legal) -> bool:
    """Every legal switch's action index is the bench row of the mon it sends."""
    bench = features._request_members(battle)[1:]
    for a in legal:
        index = features.action_index(a, battle) - features.SWITCH_AT
        if not (0 <= index < len(bench)) or bench[index] is not a.order.order:
            return False
    return True


def _uturn_then_faint() -> list[str]:
    # Cinderace leads, U-turns into Primarina; Primarina (pick 2) faints after the log was read
    return [
        *PRELUDE,
        "|switch|p1a: Cinderace|Cinderace, L50|160/160",
        "|switch|p2a: Gallade|Gallade, L50|100/100",
        "|turn|1",
        "|move|p1a: Cinderace|U-turn|p2a: Gallade",
        "|-damage|p2a: Gallade|50/100",
        "|switch|p1a: Primarina|Primarina, L50|160/160",
        "|turn|2",
    ]


def test_fainted_mon_not_pick_0_gives_distinct_indices_with_matching_rows() -> None:
    names, protocol = tuple(MOVES), _uturn_then_faint()
    b = _record(names, [2, 0, 1], fainted={2})
    first = fainted_active(b, tracking.replay(protocol, None), {})
    assert first == 2
    battle, legal, request = _forced(_team(), [0, 1, 2], protocol, first, {2}, names)
    side = request["side"]["pokemon"]
    assert side[0]["ident"] == "p1: Primarina" and side[0]["active"] and side[0]["condition"] == "0 fnt"
    assert [p["ident"] for p in side[1:]] == ["p1: Cinderace", "p1: Golisopod"]
    indices = [features.action_index(a, battle) for a in legal]
    assert sorted(a.message for a in legal) == ["/choose switch Cinderace", "/choose switch Golisopod"]
    assert len(set(indices)) == 2 and _rows_match(battle, legal)
    assert index_collisions(b, "forced", battle, legal) == {}
    # the old request (pick order, nobody active): both orders on one index
    battle, legal, _ = _forced(_team(), [0, 1, 2], protocol, None, {2}, names)
    assert len({features.action_index(a, battle) for a in legal}) == 1
    assert list(index_collisions(b, "forced", battle, legal).values()) == [[a.message for a in legal]]


def test_the_15_nov_example_battle_10_decision_3() -> None:
    # request [Cinderace, Golisopod, Primarina fnt], legal [Cinderace, Golisopod]: the network chose Golisopod
    # (p 1.0) and Cinderace was sent
    rec = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    team, picks = team_from_request(rec["request"])
    names = team.names
    assert names == ("Cinderace", "Golisopod", "Primarina") and rec["network_pick"] == "golisopod"
    before = tracking.replay(rec["protocol"], rec["request"])
    old = orders.battle_actions(before, FORCED, slot_moves=[], names=list(names), fainted={2}, active=None)[0]
    assert len({features.action_index(a, before) for a in old}) == 1  # the bug, as recorded
    b = _record(names, [2, 0, 1], fainted={2})  # the U-turn put Primarina first on the party screen
    first = fainted_active(b, tracking.replay(rec["protocol"], None), {})
    assert first == 2
    battle, legal, request = _forced(team, picks, rec["protocol"], first, {2}, names)
    assert [p["ident"] for p in request["side"]["pokemon"]] == ["p1: Primarina", "p1: Cinderace", "p1: Golisopod"]
    by = {a.message: features.action_index(a, battle) for a in legal}
    assert set(by) == set(rec["legal"]) and len(set(by.values())) == 2 and _rows_match(battle, legal)
    golisopod = features.legal_action(by["/choose switch Golisopod"], legal, battle)
    assert golisopod.message == "/choose switch Golisopod"  # the network's pick is now the order sent


def test_fainted_first_and_pivot_requests_are_unchanged() -> None:
    names = tuple(MOVES)
    lead_faints = [
        *PRELUDE,
        "|switch|p1a: Cinderace|Cinderace, L50|160/160",
        "|switch|p2a: Gallade|Gallade, L50|100/100",
        "|turn|1",
    ]
    b = _record(names, [0, 1, 2], fainted={0})
    first = fainted_active(b, tracking.replay(lead_faints, None), {})
    assert first == 0
    _, _, fixed = _forced(_team(), [0, 1, 2], lead_faints, first, {0}, names)
    _, _, old = _forced(_team(), [0, 1, 2], lead_faints, None, {0}, names)
    assert [p["ident"] for p in fixed["side"]["pokemon"]] == [p["ident"] for p in old["side"]["pokemon"]]
    assert fixed["side"]["pokemon"][0]["condition"] == "0 fnt"
    battle, legal, _ = _forced(_team(), [0, 1, 2], lead_faints, first, {0}, names)
    assert len({features.action_index(a, battle) for a in legal}) == 2 and _rows_match(battle, legal)
    # a pivot: forced() passes the leaving slot (leaving_slot), never fainted_active; the leaving mon stays first
    pivot = [*lead_faints, "|move|p1a: Cinderace|U-turn|p2a: Gallade"]
    battle, legal, request = _forced(_team(), [0, 1, 2], pivot, 0, set(), names)
    assert request["side"]["pokemon"][0]["ident"] == "p1: Cinderace" and request["side"]["pokemon"][0]["active"]
    assert sorted(a.message for a in legal) == ["/choose switch Golisopod", "/choose switch Primarina"]
    assert len({features.action_index(a, battle) for a in legal}) == 2 and _rows_match(battle, legal)


def test_which_fainted_mon_is_our_active() -> None:
    names, protocol = tuple(MOVES), _uturn_then_faint()
    tracked = tracking.replay(protocol, None)  # the log's active: Primarina, standing at its last read
    # the faint behind this screen is the tracked active's, even when visible_order lags (an unseen pivot)
    assert fainted_active(_record(names, [0, 1, 2], fainted={2}), tracked, {}) == 2
    # a pivot the log has not shown yet: the log still has Cinderace in; visible_order followed the switch
    before_uturn = protocol[:-3] + ["|turn|2"]
    assert fainted_active(_record(names, [2, 0, 1], fainted={2}), tracking.replay(before_uturn, None), {}) == 2
    # the replacement fainted before the next log read (the log shows the first faint only): visible_order wins
    gone = [*protocol[:-1], "|faint|p1a: Primarina", "|turn|2"]
    b = _record(names, [1, 0, 2], fainted={2, 1})
    assert fainted_active(b, tracking.replay(gone, None), {}) == 1
    # the party screen's 0 HP counts as fainted; no Battle: the bookkeeping alone
    assert fainted_active(_record(names, [1, 0, 2], fainted=set()), None, {1: (0, 160)}) == 1
    assert fainted_active(_record(names, [0, 1, 2], fainted={1}), None, {}) == 1  # the only fainted slot
    assert fainted_active(_record(names, [0, 1, 2], fainted=set()), None, {}) is None


def test_collisions_are_logged_loudly(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr("pcsa.live.turn_sync.log.warning", lambda event, **kw: seen.append((event, kw)))
    names, protocol = tuple(MOVES), _uturn_then_faint()
    battle, legal, _ = _forced(_team(), [0, 1, 2], protocol, None, {2}, names)
    index_collisions(_record(names, [2, 0, 1], {2}), "forced", battle, legal)
    assert seen and seen[0][0] == "ACTION_INDEX_COLLISION" and seen[0][1]["kind"] == "forced"
    assert index_collisions(_record(names, [2, 0, 1], {2}), "forced", None, legal) == {}
