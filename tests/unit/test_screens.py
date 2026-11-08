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
