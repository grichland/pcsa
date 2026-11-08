"""Real screenshots from past runs, classified from cached OCR boxes (no OCR engine needed).

The screenshots are game captures and stay out of git: `expected.json` and `ocr_cache.json` are tracked,
the JPGs are written locally by `scripts/curate_frames.py`, and every test that needs the pixels skips
in a checkout without them.

`expected.json` records the screen each frame is expected to classify as. Entries with
`agrees: false` are frames whose saved label and current classification disagree; they are kept
visible here as expected failures until a person settles them, never silently accepted.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from pcsa.common.types import Screen
from pcsa.phone.ocr import TextBox
from pcsa.phone.screens import classify, matching_rules, to_reference

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "frames"
FRAMES_CAP_MB = 20
TOTAL_FIXTURES_CAP_MB = 30
MIN_CONFIDENCE = 0.8


def corpus() -> dict[str, dict]:
    path = CORPUS / "expected.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def cached_boxes(name: str) -> list[TextBox]:
    cache = json.loads((CORPUS / "ocr_cache.json").read_text(encoding="utf-8"))
    return [TextBox(b["text"], b["confidence"], tuple(b["bounds"])) for b in cache[name]]


def screenshot(name: str):
    """The screenshot's pixels, or a skip when it is not on disk (screenshots are kept out of the repo)."""
    path = CORPUS / name
    if not path.is_file():
        pytest.skip(f"{name} is a local-only screenshot (scripts/curate_frames.py)")
    return cv2.imread(str(path))


CASES = sorted(corpus().items())
pytestmark = pytest.mark.skipif(not CASES, reason="corpus not curated yet (scripts/curate_frames.py)")


@pytest.mark.parametrize("name,expected", CASES, ids=[name for name, _ in CASES])
def test_frame_classifies_as_expected(name: str, expected: dict) -> None:
    if not expected["agrees"]:
        pytest.xfail(f"saved label {expected['old_label']} vs {expected['screen']}: needs review")
    image = screenshot(name)
    obs = classify(cached_boxes(name), to_reference(image), MIN_CONFIDENCE)
    assert str(obs.screen) == expected["screen"], obs.reason


@pytest.mark.parametrize("name,expected", CASES, ids=[name for name, _ in CASES])
def test_no_frame_matches_two_named_rules(name: str, expected: dict) -> None:
    image = screenshot(name)
    rules = matching_rules(cached_boxes(name), to_reference(image), MIN_CONFIDENCE)
    assert len({screen for _, screen in rules}) <= 1, rules


OTHER_SIZES = ((2400, 1080), (1920, 1080), (3200, 1440), (2340, 1080))


def letterbox(reference, width: int, height: int):
    """A frame of another size showing the game letterboxed at the reference aspect, plus its viewport."""
    import numpy as np

    from pcsa.phone.calibrate import letterboxed_viewport

    vp = letterboxed_viewport(width, height)
    frame = np.zeros((height, width, 3), np.uint8)
    frame[vp.y : vp.y + vp.h, vp.x : vp.x + vp.w] = cv2.resize(reference, (vp.w, vp.h), interpolation=cv2.INTER_AREA)
    return frame, vp


@pytest.mark.parametrize("name,expected", CASES[::6], ids=[name for name, _ in CASES[::6]])
@pytest.mark.parametrize("size", OTHER_SIZES, ids=[f"{w}x{h}" for w, h in OTHER_SIZES])
def test_pixel_rules_survive_other_frame_sizes(name: str, expected: dict, size: tuple[int, int]) -> None:
    """Cached boxes stand in for OCR; the pixel checks run on the frame mapped back through its viewport."""
    if not expected["agrees"]:
        pytest.skip("under review")
    reference = to_reference(screenshot(name))
    frame, viewport = letterbox(reference, *size)
    obs = classify(cached_boxes(name), to_reference(frame, viewport), MIN_CONFIDENCE)
    assert str(obs.screen) == expected["screen"], obs.reason


@pytest.mark.ocr
@pytest.mark.parametrize("name,expected", CASES[::6], ids=[name for name, _ in CASES[::6]])
def test_real_ocr_agrees_at_another_size(name: str, expected: dict, config) -> None:
    from pcsa.phone.ocr import RapidOcrReader
    from pcsa.phone.screens import Perception

    reader = RapidOcrReader(config.agent.ocr)
    reference = to_reference(screenshot(name))
    frame, viewport = letterbox(reference, 1920, 1080)
    obs = Perception(reader, config.screen, viewport).observe(frame)
    assert str(obs.screen) == expected["screen"], obs.reason


def test_every_expected_screen_is_a_known_screen() -> None:
    for expected in corpus().values():
        Screen(expected["screen"])


def test_fixture_sizes_stay_small() -> None:
    frames_mb = sum(p.stat().st_size for p in CORPUS.glob("*.jpg")) / 1e6
    total_mb = sum(p.stat().st_size for p in (CORPUS.parent).rglob("*") if p.is_file()) / 1e6
    assert frames_mb <= FRAMES_CAP_MB, f"frames corpus is {frames_mb:.1f} MB"
    assert total_mb <= TOTAL_FIXTURES_CAP_MB, f"fixtures are {total_mb:.1f} MB"
