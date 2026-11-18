"""Reading one battle-log row from a frame: text bands, the category pill and priority badge, the strict wording
check, and re-reads of a row that fails it (larger scale, binarised text). Reference-frame coordinates."""

from __future__ import annotations

import re

import cv2
import numpy as np

from pcsa.phone.log_wording import check
from pcsa.phone.ocr import TextReader

Rect = tuple[int, int, int, int]
WHITE_THRESHOLD = 170
TEXT_PADDING = 4
MIN_BAND = 8
EDGE_GAP = 12  # short text this close to a cut-off edge row's edge is cut too ("Ivove" for "Move")
FULL_TEXT_PX = 28  # a whole line of text is 30-40 px tall
TIMER_BOX: Rect = (945, 612, 1395, 812)  # the centred MOVE TIME warning
BATTLE_TIMER_BOX: Rect = (884, 312, 1456, 516)  # the "Until Battle Ends" countdown
OVERLAY_BOXES = (TIMER_BOX, BATTLE_TIMER_BOX)
OVERLAY_RATIO = 0.85
DIM_TEXT = (18, 34)  # row text seen through an overlay box: brighter than the box, greyer (spread) than it
PRIORITY = re.compile(r"^([+\-—–−])\s*(\d)\s*Priority$")
CUT_TEXT = "text cut off at the panel edge"
RE_READS = (("x2", 2, False), ("x3", 3, False), ("bin2", 2, True))


def overlay_ratio(frame: np.ndarray, box: Rect) -> float:
    """Median brightness inside an overlay box relative to the panel beside it (below ~0.85: box shown)."""
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    left, top, right, bottom = box
    inner = v[top + 8 : bottom - 8, left + 15 : right - 15]
    side = np.concatenate(
        [v[top + 8 : bottom - 8, 600 : left - 60].ravel(), v[top + 8 : bottom - 8, right + 60 : 1850].ravel()]
    )
    return float(np.median(inner) / max(1.0, float(np.median(side))))


def overlay_boxes(frame: np.ndarray) -> list[Rect]:
    return [box for box in OVERLAY_BOXES if overlay_ratio(frame, box) < OVERLAY_RATIO]


def text_pixels(frame: np.ndarray, boxes: list[Rect]) -> np.ndarray:
    """White-text mask; inside a shown overlay box, the dimmed row text and none of the box's own text."""
    low, high = np.min(frame, axis=2).astype(int), np.max(frame, axis=2).astype(int)
    mask = low > WHITE_THRESHOLD
    left, top, right, bottom = TIMER_BOX
    mask[top:bottom, left:right] &= (high - low)[top:bottom, left:right] < 60  # the warning's tinted strokes
    for left, top, right, bottom in boxes:
        lo, spread = low[top:bottom, left:right], (high - low)[top:bottom, left:right]
        own = cv2.dilate(((lo >= WHITE_THRESHOLD) | (spread >= 60)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        dim = (lo > np.median(lo) + DIM_TEXT[0]) & (spread < DIM_TEXT[1])
        mask[top:bottom, left:right] = dim & ~own  # not the box's own caption, digits and their edges
    return mask


def bands(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(x1, y1, x2, y2) of each text line in a mask, padded."""
    out = []
    rows = np.flatnonzero(np.sum(mask, axis=1) > 3)
    if not len(rows):
        return out
    for group in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1):
        start, end = int(group[0]), int(group[-1])
        if end - start < MIN_BAND:
            continue
        xs = np.flatnonzero(mask[start : end + 1].sum(axis=0) > 0)
        x1, x2 = max(0, int(xs[0]) - TEXT_PADDING), min(mask.shape[1], int(xs[-1]) + 1 + TEXT_PADDING)
        out.append((x1, max(0, start - TEXT_PADDING), x2, min(mask.shape[0], end + 1 + TEXT_PADDING)))
    return out


def line_image(frame: np.ndarray, mask: np.ndarray, rect: Rect, scale: int, binary: bool) -> np.ndarray:
    left, top, right, bottom = rect
    image = frame[top:bottom, left:right]
    if binary:
        plain = np.where(mask[top:bottom, left:right], 255, 40).astype(np.uint8)
        image = cv2.merge([cv2.GaussianBlur(plain, (3, 3), 0)] * 3)
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(image)


def segments(
    frame: np.ndarray, mask: np.ndarray, reader: TextReader, area: Rect, role: str, how: tuple[str, int, bool]
) -> tuple[list[dict], list[str]]:
    """Recognise each text line inside one area of the frame."""
    left, top, right, bottom = area
    found, problems = [], []
    for x1, y1, x2, y2 in bands(mask[top:bottom, left:right]):
        rect = (left + x1, top + y1, left + x2, top + y2)
        result = reader.line(line_image(frame, mask, rect, how[1], how[2]))
        if result is None:
            problems.append(f"unreadable {role} text at x={rect[0]}, y={rect[1]}")
            continue
        clipped = y1 <= 1 or y2 >= bottom - top - 1
        found.append(
            {
                "role": role,
                "text": result[0],
                "confidence": result[1],
                "crop": list(rect),
                "possibly_clipped": clipped,
            }
        )
    return found, problems


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?<=\w)['‘](?=\w)", "’", text)


def compose(labels: list[dict], messages: list[dict]) -> tuple[str, str | None, str, str | None]:
    """(category, priority, message, problem) from the pill text, the badge under it and the message lines."""
    category = normalize(labels[0]["text"]) if labels else ""
    priority, problem = None, None
    if len(labels) > 1:
        badge = normalize(" ".join(s["text"] for s in labels[1:]))
        m = PRIORITY.match(badge)
        if m:
            priority = ("+" if m[1] == "+" else "-") + m[2]
        else:
            problem = f"unreadable priority badge {badge!r}"
    return category, priority, normalize(" ".join(s["text"] for s in messages)), problem


def evaluate(parts: tuple[str, str | None, str, str | None], how: str, flags: dict) -> dict:
    """One reading of the row: its raw text, and the strict check's verdict and spelling."""
    category, priority, message, problem = parts
    label = f"{category} {priority[0]}{priority[1:]} Priority" if priority else category
    raw = f"{label}: {message}" if label else message
    reading = {"text": raw, "how": how, "valid": False, "canonical": None, "wording": None, "reason": problem}
    if flags.get("clipped"):
        reading["reason"] = CUT_TEXT
    elif problem is None:
        if priority and category != "Move":
            reading["reason"] = f"priority badge on a {category!r} row"
        else:
            result = check(category, message)
            canonical = result.text.replace(f"{category}: ", f"{label}: ", 1) if priority else result.text
            reading.update(valid=result.valid, canonical=canonical if result.valid else None, wording=result.wording)
            reading["reason"] = result.reason
    reading["covered"] = bool(flags.get("covered"))
    return reading


def edge_clipped(found: list[dict], bounds: tuple[int, int], edges: tuple[bool, bool]) -> bool:
    """Text against the top of a cut-off first row or the bottom of a cut-off last row."""
    top, bottom = bounds
    return any(
        s["crop"][3] - s["crop"][1] < FULL_TEXT_PX
        and ((edges[0] and s["crop"][1] - top < EDGE_GAP) or (edges[1] and bottom - s["crop"][3] < EDGE_GAP))
        for s in found
    )


def read_row(
    frame: np.ndarray,
    mask: np.ndarray,
    reader: TextReader,
    areas: tuple[Rect, Rect],
    edges: tuple[bool, bool],
    boxes: list[Rect],
) -> tuple[list[dict], list[dict], list[str]]:
    """(segments of the first reading, every reading, problems). A reading that fails the check is re-read."""
    bounds = (areas[0][1], areas[0][3])
    covered = any(b[1] < bounds[1] and b[3] > bounds[0] for b in boxes)
    readings, first, problems = [], [], []
    plans = (("bin2", 2, True), ("bin3", 3, True)) if covered else (("x1", 1, False), *RE_READS)
    for how in plans:
        labels, p1 = segments(frame, mask, reader, areas[0], "category", how)
        messages, p2 = segments(frame, mask, reader, areas[1], "message", how)
        if not first:
            first, problems = labels + messages, p1 + p2
        if not messages and not covered:
            if not labels:
                break
            continue
        flags = {"covered": covered, "clipped": edge_clipped(labels + messages, bounds, edges)}
        readings.append(evaluate(compose(labels, messages), how[0], flags))
        if readings[-1]["valid"] or flags["clipped"]:
            break
    return first, readings, problems
