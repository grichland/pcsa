from pcsa.common.types import IN_BATTLE, Flag, Observation, Screen, SemanticAction, jsonable


def test_semantic_action_label() -> None:
    assert SemanticAction("move", move=2, mega=True).label() == "mega+move2"
    assert SemanticAction("switch", slot=1).label() == "switch1"


def test_observation_record_is_json_friendly() -> None:
    obs = Observation(
        Screen.ACTION, (2340, 1080), frozenset({Flag.TIMER_WARN}), {"moves": (SemanticAction("move", 1),)}
    )
    record = obs.record()
    assert record["screen"] == "action"
    assert record["flags"] == ["timer_warn"]
    assert record["fields"]["moves"][0]["kind"] == "move"
    assert jsonable({"s": Screen.LOBBY}) == {"s": "lobby"}


def test_in_battle_set_excludes_menus() -> None:
    assert Screen.ACTION in IN_BATTLE
    assert Screen.LOBBY not in IN_BATTLE
    assert Screen.TITLE not in IN_BATTLE
