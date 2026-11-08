"""Queries over OCR boxes and the few pixel checks the screen rules use. Reference-frame coordinates."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from pcsa.phone.ocr import REFERENCE, TextBox

Rect = tuple[int, int, int, int]
FULL: Rect = (0, 0, REFERENCE[0], REFERENCE[1])
LIME = ((22, 120, 170), (43, 255, 255))  # HSV focus highlight
LINE_LETTERS = re.compile(r"[A-Za-z]{3,}")


def inside(box: TextBox, region: Rect) -> bool:
    x, y = box.center
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


class Texts:
    """OCR boxes with region and pattern queries over normalized text."""

    def __init__(self, boxes: Sequence[TextBox], min_confidence: float = 0.80) -> None:
        self.all = list(boxes)
        self.boxes = [b for b in self.all if b.confidence >= min_confidence]

    def find(self, pattern: str, region: Rect = FULL) -> list[TextBox]:
        rx = re.compile(pattern)
        return [b for b in self.boxes if inside(b, region) and rx.fullmatch(b.norm)]

    def has(self, pattern: str, region: Rect = FULL) -> bool:
        return bool(self.find(pattern, region))

    def any_text(self, region: Rect) -> list[TextBox]:
        return [b for b in self.boxes if inside(b, region)]

    def joined(self, region: Rect) -> str:
        """Words inside a region joined left to right (OCR often splits one label into boxes)."""
        return " ".join(b.norm for b in sorted(self.any_text(region), key=lambda b: b.center[0]))

    def count(self) -> int:
        return len(self.boxes)


def fraction(texts: Texts, region: Rect) -> tuple[int, int] | None:
    """Read 'a/b', or a numerator above a separately recognized '/b'."""
    items = texts.any_text(region)
    whole = [re.fullmatch(r"(\d+)\s*/\s*(\d+)", b.norm.replace(" ", "")) for b in items]
    values = [(int(m[1]), int(m[2])) for m in whole if m]
    if not values:
        numerators = [b for b in items if re.fullmatch(r"\d+", b.norm)]
        denominators = [b for b in items if re.fullmatch(r"/\s*\d+", b.norm)]
        if len(numerators) == 1 and len(denominators) == 1:
            values = [(int(numerators[0].norm), int(denominators[0].norm.lstrip("/ ")))]
    if len(values) == 1 and 0 <= values[0][0] <= values[0][1] and values[0][1] > 0:
        return values[0]
    return None


def name_in(texts: Texts, region: Rect) -> str | None:
    names = [b for b in texts.any_text(region) if LINE_LETTERS.search(b.text)]
    return max(names, key=lambda b: b.confidence).text.strip() if names else None


def pp_from_crop(texts: Texts, image: np.ndarray, top: int, line: Callable | None) -> tuple[int, int] | None:
    """Re-read a PP fraction from a tight crop when full-frame OCR missed a thin numerator."""
    if line is None:
        return None
    result = line(image[top + 10 : top + 112, 2075:2228])
    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", result[0].replace(" ", "")) if result else None
    values = (int(match[1]), int(match[2])) if match else None
    if values is None:
        denominators = [
            b for b in texts.any_text((2060, top + 40, 2235, top + 115)) if re.fullmatch(r"/\s*\d+", b.norm)
        ]
        numerator = line(image[top + 12 : top + 78, 2080:2190])
        if len(denominators) == 1 and numerator and re.fullmatch(r"\d+", numerator[0].strip()):
            values = (int(numerator[0].strip()), int(denominators[0].norm.lstrip("/ ")))
    if values and 0 <= values[0] <= values[1] and values[1] > 0:
        return values
    return None


# Pixel checks -----------------------------------------------------------------------------------
def crop(image: np.ndarray, rect: Rect) -> np.ndarray:
    left, top, right, bottom = rect
    return image[top:bottom, left:right]


def mean_brightness(image: np.ndarray) -> float:
    return float(np.mean(image))


def lime_fraction(image: np.ndarray, rect: Rect) -> float:
    hsv = cv2.cvtColor(crop(image, rect), cv2.COLOR_BGR2HSV)
    return float(np.mean(cv2.inRange(hsv, np.array(LIME[0]), np.array(LIME[1])) > 0))


def yellow_fraction(image: np.ndarray, rect: Rect) -> float:
    """Gold WON!! banner pixels (the LOST... banner is silver)."""
    hsv = cv2.cvtColor(crop(image, rect), cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[..., 0] >= 20) & (hsv[..., 0] <= 38) & (hsv[..., 1] > 110) & (hsv[..., 2] > 170)))


def red_fraction(image: np.ndarray, rect: Rect) -> float:
    """Fainted badge on a party row: strong red, little green."""
    patch = crop(image, rect)
    if patch.size == 0:
        return 0.0
    return float(np.mean((patch[..., 2] > 170) & (patch[..., 1] < 90)))


def split_screen(image: np.ndarray) -> bool:
    """The result screen is split by a black vertical divider at x~1168; the sides stay lit."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    centre = gray[150:950, 1150:1190].mean(axis=0)
    sides = (gray[150:950, 1080:1130].mean() + gray[150:950, 1210:1260].mean()) / 2
    return float(centre.min()) <= 8 and float(sides) >= 15


def skip_button(image: np.ndarray) -> bool:
    """Cutscene fast-forward pill at the top right: light, unsaturated pill with a blue >> icon."""
    pill = cv2.cvtColor(crop(image, (1985, 12, 2095, 70)), cv2.COLOR_BGR2HSV)
    light = float(np.mean((pill[..., 1] < 40) & (pill[..., 2] > 200)))
    icon = cv2.cvtColor(crop(image, (2015, 25, 2065, 58)), cv2.COLOR_BGR2HSV)
    blue = float(
        np.mean(
            (icon[..., 0] >= 100)
            & (icon[..., 0] <= 130)
            & (icon[..., 1] > 120)
            & (icon[..., 2] > 60)
            & (icon[..., 2] < 200)
        )
    )
    return light >= 0.5 and blue >= 0.35
