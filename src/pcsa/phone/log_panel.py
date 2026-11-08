"""The in-game battle log panel: pixel checks on a frame and OCR of one settled view. Reference frame."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import cv2
import numpy as np

from pcsa.phone.ocr import REFERENCE, TextReader
from pcsa.phone.sprites import match_log_row

HEADER = (1080, 40, 1260, 105)
CONTENT = (550, 118, 1885, 979)
SEPARATORS = (390, 118, 1880, 979)
SCROLLBAR = (1933, 134, 1941, 959)
LABEL_RIGHT = 880
MESSAGE_LEFT = 900
WHITE_THRESHOLD = 170
LOG_ROW_MIN_CONFIDENCE = 0.85
TEXT_PADDING = 4
TIMER_BOX = (945, 612, 1395, 812)  # the centred MOVE TIME warning drawn over the log
TIMER_OVERLAY_RATIO = 0.9
CLIPPED_ROW_PX = 110  # a full row is 134 px; cut-off edge rows OCR as garbage


@dataclass
class Row:
    text: str
    bounds: tuple[int, int]
    segments: list[dict]
    sprite: dict | None = None


@dataclass
class View:
    source: str
    sha256: str
    turn: int
    scroll: float
    thumb: tuple[int, int]
    at_top: bool
    at_bottom: bool
    has_scrollbar: bool
    rows: list[Row]
    warnings: list[str]
    settled: bool = True  # False: the last capture of a budget-limited climb, never compared with a second one


# Pixel helpers ------------------------------------------------------------------------------------
def crop(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = rect
    return frame[top:bottom, left:right].copy()


def runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if not len(indices):
        return []
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def separator_lines(frame: np.ndarray) -> list[int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    left, top, right, bottom = SEPARATORS
    contrast = (
        gray[top:bottom, left:right]
        - (gray[top - 3 : bottom - 3, left:right] + gray[top + 3 : bottom + 3, left:right]) / 2
    )
    peaks = np.flatnonzero(np.median(contrast, axis=1) > 20) + top
    return [round((start + end) / 2) for start, end in runs(peaks)]


def same_layout(a: list[int], b: list[int], tolerance: int = 3) -> bool:
    return bool(a) and len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True))


def row_bounds(frame: np.ndarray) -> list[tuple[int, int]]:
    lines = separator_lines(frame)
    if not lines:
        raise ValueError("cannot identify battle-log separators")
    edges = [CONTENT[1], *lines, CONTENT[3]]
    return [(a + 5, b - 5) for a, b in zip(edges, edges[1:], strict=False) if b - a > 35]


def scrollbar(frame: np.ndarray) -> tuple[float, tuple[int, int], bool, bool, bool]:
    """(position, thumb, at_top, at_bottom, present). A short log has no scrollbar: top and bottom at once."""
    strip = crop(frame, SCROLLBAR)
    bright = np.median(np.min(strip, axis=2), axis=1) > 210
    candidates = [(a, b) for a, b in runs(np.flatnonzero(bright)) if b - a >= 10]
    if not candidates:
        return 0.0, (SCROLLBAR[1], SCROLLBAR[3] - 1), True, True, False
    if len(candidates) != 1:
        raise ValueError(f"expected at most one scrollbar thumb; found {len(candidates)}")
    start, end = candidates[0]
    travel = len(strip) - (end - start + 1)
    if travel <= 0:
        return 0.0, (start + SCROLLBAR[1], end + SCROLLBAR[1]), True, True, False
    return start / travel, (start + SCROLLBAR[1], end + SCROLLBAR[1]), start <= 5, travel - start <= 5, True


def text_mask(frame: np.ndarray) -> np.ndarray:
    content = crop(frame, CONTENT)
    small = cv2.resize(content, (content.shape[1] // 4, content.shape[0] // 4), interpolation=cv2.INTER_AREA)
    return np.min(small, axis=2) > WHITE_THRESHOLD


def mask_difference(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return 0.0 if union == 0 else np.count_nonzero(a ^ b) / union


def timer_overlay_ratio(frame: np.ndarray) -> float:
    """Median brightness inside the MOVE TIME box relative to the panel beside it."""
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    box = v[620:805, 960:1380]
    side = np.concatenate([v[620:805, 600:880].ravel(), v[620:805, 1450:1850].ravel()])
    return float(np.median(box) / max(1.0, float(np.median(side))))


def is_log_panel(frame: np.ndarray) -> bool:
    header = cv2.cvtColor(frame[42:94, 500:1850], cv2.COLOR_BGR2HSV)
    purple = (header[..., 0] > 105) & (header[..., 0] < 145) & (header[..., 1] > 65)
    return bool(np.mean(purple) >= 0.7)


def has_timer_overlay(frame: np.ndarray) -> bool:
    return is_log_panel(frame) and timer_overlay_ratio(frame) < TIMER_OVERLAY_RATIO


def same_viewport(previous: np.ndarray, current: np.ndarray, *, text_tolerance: float = 0.4) -> bool:
    if not is_log_panel(previous) or not is_log_panel(current):
        return False
    if not same_layout(separator_lines(previous), separator_lines(current)):
        return False
    try:
        before, after = scrollbar(previous), scrollbar(current)
    except ValueError:
        return False
    if before[4] != after[4] or any(abs(a - b) > 3 for a, b in zip(before[1], after[1], strict=True)):
        return False
    return mask_difference(text_mask(previous), text_mask(current)) <= text_tolerance


def text_pixels(frame: np.ndarray) -> np.ndarray:
    """White-text mask with the MOVE TIME warning's tinted strokes masked out of its box."""
    low, high = np.min(frame, axis=2), np.max(frame, axis=2)
    mask = low > WHITE_THRESHOLD
    left, top, right, bottom = TIMER_BOX
    spread = high[top:bottom, left:right].astype(int) - low[top:bottom, left:right]
    tinted = spread >= 60
    mask[top:bottom, left:right] &= ~tinted
    return mask


# OCR ------------------------------------------------------------------------------------------------
def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?<=\w)['‘](?=\w)", "’", text)


def recognize_lines(
    reader: TextReader,
    image: np.ndarray,
    left: int,
    top: int,
    role: str,
    warnings: list[str],
    mask: np.ndarray | None = None,
) -> list[dict]:
    """Locate white text bands, then recognize each tightly cropped line."""
    if mask is None:
        mask = np.min(image, axis=2) > WHITE_THRESHOLD
    segments = []
    for start, end in runs(np.flatnonzero(np.sum(mask, axis=1) > 3)):
        if end - start < 8:
            continue
        xs = np.flatnonzero(mask[start : end + 1].sum(axis=0) > 0)
        x1, x2 = max(0, int(xs[0]) - TEXT_PADDING), min(image.shape[1], int(xs[-1]) + 1 + TEXT_PADDING)
        y1, y2 = max(0, start - TEXT_PADDING), min(image.shape[0], end + 1 + TEXT_PADDING)
        result = reader.line(image[y1:y2, x1:x2])
        if result is None:
            warnings.append(f"unreadable {role} text at x={left + x1}, y={top + y1}")
            continue
        segments.append(
            {
                "role": role,
                "text": result[0],
                "confidence": result[1],
                "crop": [left + x1, top + y1, left + x2, top + y2],
                "possibly_clipped": start <= 1 or end >= image.shape[0] - 2,
            }
        )
    return segments


def read_turn(frame: np.ndarray, reader: TextReader) -> int | None:
    header = crop(frame, HEADER)
    for image in (header, cv2.copyMakeBorder(header, 12, 12, 40, 40, cv2.BORDER_REPLICATE)):
        result = reader.line(image)
        match = re.fullmatch(r"Turn\s*(\d+)", (result[0] if result else "").strip(), re.IGNORECASE)
        if match:
            return int(match[1])
    return None


def read_view(frame: np.ndarray, reader: TextReader, source: str = "<array>", *, sprites: bool = True) -> View:
    """Read one settled reference-size log frame. Raises ValueError when it is not a readable panel."""
    if (frame.shape[1], frame.shape[0]) != REFERENCE:
        frame = cv2.resize(frame, REFERENCE, interpolation=cv2.INTER_AREA)
    sha = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
    turn = read_turn(frame, reader)
    if turn is None:
        raise ValueError(f"cannot read the turn header in {source}")
    scroll, thumb, at_top, at_bottom, present = scrollbar(frame)
    warnings: list[str] = []
    pixels = text_pixels(frame)
    rows = []
    for top, bottom in row_bounds(frame):
        labels = recognize_lines(
            reader,
            crop(frame, (CONTENT[0], top, LABEL_RIGHT, bottom)),
            CONTENT[0],
            top,
            "category",
            warnings,
            pixels[top:bottom, CONTENT[0] : LABEL_RIGHT],
        )
        messages = recognize_lines(
            reader,
            crop(frame, (MESSAGE_LEFT, top, CONTENT[2], bottom)),
            MESSAGE_LEFT,
            top,
            "message",
            warnings,
            pixels[top:bottom, MESSAGE_LEFT : CONTENT[2]],
        )
        if not labels and not messages:
            continue
        if not messages:
            warnings.append(f"row has a category but no message at y={top}:{bottom}")
            continue
        label = normalize(" ".join(d["text"] for d in labels))
        message = normalize(" ".join(d["text"] for d in messages))
        text = f"{label}: {message}" if label else message
        segments = labels + messages
        sprite = match_log_row(frame, (top, bottom)) if sprites else None
        rows.append(Row(text, (top, bottom), segments, sprite))
        if any(d["possibly_clipped"] for d in segments):
            warnings.append(f"possibly clipped text at y={top}: {text}")
        lowest = min(d["confidence"] for d in segments)
        if lowest < LOG_ROW_MIN_CONFIDENCE:
            warnings.append(f"low OCR confidence {lowest:.3f} at y={top}: {text}")
    if not rows:
        raise ValueError(f"no readable log rows: {source}")
    return View(source, sha, turn, scroll, thumb, at_top, at_bottom, present, rows, warnings)
