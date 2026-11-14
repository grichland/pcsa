import numpy as np

from pcsa.common.types import Flag, Screen
from pcsa.phone.ocr import TextBox
from pcsa.phone.screens import classify, matching_rules


def box(text, x, y, conf=0.95, w=220, h=40):
    return TextBox(text, conf, (x - w / 2, y - h / 2, x + w / 2, y + h / 2))


def frame(level: int = 90) -> np.ndarray:
    return np.full((1080, 2340, 3), level, np.uint8)


def test_lobby_with_rank() -> None:
    boxes = [
        box("Begin Matchmaking", 1600, 930),
        box("Master Ball Tier", 1600, 350),
        box("Rank 4", 1600, 400),
        box("Team 1", 1600, 545),
    ]
    obs = classify(boxes, frame(), 0.8)
    assert obs.screen == Screen.LOBBY and obs.reason == "Begin Matchmaking"
    assert obs.fields["own_rank"]["tier"] == "Master Ball" and obs.fields["own_rank"]["rank"] == 4
    assert obs.fields["team_label"] == "Team 1"


def test_action_screen_reads_the_hud() -> None:
    boxes = [box("Fight", 2094, 690), box("Pokemon", 2007, 901), box("View Log", 280, 378), box("Garchomp", 400, 990),
             box("150/180", 450, 1045), box("Primarina", 2025, 70), box("27", 2140, 275)]  # fmt: skip
    obs = classify(boxes, frame(), 0.8)
    assert obs.screen == Screen.ACTION
    assert obs.fields["active_name"] == "Garchomp" and obs.fields["active_hp"] == (150, 180)
    assert obs.fields["move_time"] == 27 and obs.fields["view_log"]


def test_options_with_split_labels() -> None:
    boxes = [box("Continue", 1650, 990, w=160), box("Battling", 1850, 990, w=160), box("Quit Battling", 600, 990)]
    assert classify(boxes, frame(), 0.8).screen == Screen.OPTIONS


def test_comm_error_and_date_change_are_interruptions_with_a_dismiss() -> None:
    boxes = [box("Communication was interrupted.", 1170, 500, w=900), box("OK", 1170, 800, w=120)]
    obs = classify(boxes, frame(), 0.8)
    assert obs.screen == Screen.COMM_ERROR and Flag.COMM_ERROR in obs.flags
    assert obs.fields["kind"] == "comm_error" and obs.fields["dismiss"]["name"] == "ok"
    obs = classify(
        [box("The date has changed. Returning to the title screen.", 1170, 500, w=1200), box("OK", 1170, 800, w=120)],
        frame(),
        0.8,
    )
    assert obs.screen == Screen.COMM_ERROR and obs.fields["kind"] == "date_changed"


def test_ranked_formats_and_home() -> None:
    boxes = [
        box("Single Battle", 1500, 500),
        box("Double Battle", 1500, 740),
        box("Master Ball Tier | Rank 4", 1500, 600, w=500),
        box("Poke Ball Tier | Rank 1", 1500, 800, w=500),
    ]
    obs = classify(boxes, frame(), 0.8)
    assert obs.screen == Screen.RANKED_FORMATS and obs.fields["target"]["name"] == "single_battle"
    boxes = [box("Battle", 1400, 300), box("Box", 1700, 300), box("Train", 1900, 300), box("Shop", 2000, 600)]
    assert classify(boxes, frame(), 0.8).screen == Screen.HOME


def test_black_frame_and_sparse_text_are_transitions() -> None:
    assert classify([], frame(5), 0.8).screen == Screen.TRANSITION
    assert classify([box("Garchomp", 400, 990)], frame(), 0.8).screen == Screen.TRANSITION


def test_dialog_needs_exactly_one_safe_control() -> None:
    obs = classify([box("Season M-8 has begun!", 1170, 300, w=800), box("Close", 1170, 800, w=160)], frame(), 0.8)
    assert obs.screen == Screen.DIALOG and obs.fields["dismiss"]["name"] == "close"
    obs = classify(
        [box("Rate this app?", 1170, 300, w=800), box("Submit", 1000, 800), box("Not now", 1400, 800)], frame(), 0.8
    )
    assert obs.screen == Screen.RATING_DIALOG and obs.fields["dismiss"]["name"] == "not now"
    assert obs.fields["unsafe"]


def test_synthetic_frames_match_one_rule_each() -> None:
    cases = [
        [box("Begin Matchmaking", 1600, 930)],
        [box("Fight", 2094, 690), box("Pokemon", 2007, 901), box("View Log", 280, 378)],
        [box("Continue Battling", 1700, 990), box("Quit Battling", 600, 990)],
        [box("Switch in", 600, 600), box("Cancel", 600, 900)],
    ]
    for boxes in cases:
        assert len(matching_rules(boxes, frame(), 0.8)) == 1, [b.text for b in boxes]


def plates(*extra: TextBox) -> list[TextBox]:
    """14 Nov frame 02247: the opposing plate, an item banner and our plate, no menus (boxes as OCR read them)."""
    boxes = [
        TextBox("Cinderace", 1.0, (1911, 51, 2047, 93)),
        TextBox("50 %", 0.895, (2058, 118, 2158, 173)),
        TextBox("Cinderace's", 1.0, (1920, 394, 2087, 438)),
        TextBox("Life Orb", 1.0, (1960, 438, 2088, 482)),
        TextBox("Golisopod", 1.0, (260, 969, 402, 1016)),
        TextBox("123", 1.0, (365, 1040, 445, 1079)),
        TextBox("/182", 0.994, (440, 1053, 507, 1079)),
    ]
    return boxes + list(extra)


def test_battle_scene_with_hp_plates_is_a_wait() -> None:
    obs = classify(plates(), frame(), 0.8)
    assert obs.screen == Screen.WAIT and obs.reason == "HP plates without menus"
    assert obs.fields["opponent_name"] == "Cinderace" and obs.fields["active_name"] == "Golisopod"
    ours_only = [b for b in plates() if b.center[1] > 900] + [box("Sitrus Berry", 2000, 460), box("x", 100, 100)]
    assert classify(ours_only, frame(), 0.8).screen == Screen.WAIT
    assert not matching_rules(plates(), frame(), 0.8)  # a fallback: never competes with a named screen


def test_hp_plates_never_hide_an_interruption_or_a_control() -> None:
    comm = plates(box("Communication was interrupted.", 1170, 450, w=700))
    assert classify(comm, frame(), 0.8).screen == Screen.COMM_ERROR
    assert classify(plates(box("OK", 1170, 800, w=120)), frame(), 0.8).screen == Screen.DIALOG
    assert classify(plates(box("Communicating...", 1170, 300, w=400)), frame(), 0.8).screen == Screen.WAIT
    many = plates(*(box(f"word{i}", 1170, 300 + 20 * i, w=60) for i in range(8)), box("Forfeit", 1170, 800))
    assert classify(many, frame(), 0.8).screen == Screen.FORFEIT_MSG
    quit_ = plates(*(box(f"word{i}", 600, 300 + 20 * i, w=60) for i in range(8)), box("Quit", 1170, 800))
    assert classify(quit_, frame(), 0.8).screen == Screen.UNKNOWN  # an unsafe control is never waited on


def test_trainer_cards_on_a_split_screen_are_a_transition() -> None:
    image = frame()
    image[:, 1155:1185] = 0
    boxes = [box("Gamer", 646, 947), box("Gamer", 1694, 947), box("pcsa", 675, 1010), box("Niji", 1665, 1008)]
    boxes += [box("888", 1946, 1011, w=90), box("L", 1595, 50, w=90)]
    obs = classify(boxes, image, 0.8)
    assert obs.screen == Screen.TRANSITION and obs.reason == "trainer cards split screen"
    assert classify(boxes, frame(), 0.8).screen == Screen.UNKNOWN  # no divider: not the cards
