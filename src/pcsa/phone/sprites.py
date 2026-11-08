"""Sprites -> Pokémon names, by the sprite network's vectors (sprite_net.py).

A sprite is segmented from its local background, cropped to its silhouette and normalized to a fixed
square (`feature`); the network turns that into a unit vector, scored by cosine against one reference
vector per known sprite view. Two uses: the six opponent cards on team selection ("preview") and the
sprite at the left of each battle-log row ("log"). Reference-frame coordinates.

The matching flow takes any `Scorer`, so training can run the retired thumbnail matcher
(sprite_training/sprite_baseline.py) through exactly the same steps when it measures the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import cv2
import numpy as np

from pcsa.common import paths
from pcsa.phone import sprite_net

OPPONENT_BOXES = tuple((1815, 158 + 126 * i, 1958, 268 + 126 * i) for i in range(6))
LOG_SPRITE_COLUMN = (390, 545)
SIZE = sprite_net.SIZE
BG_DISTANCE = 60
STREAK_HUE, STREAK_SAT = 8, 120
DIM_LEVEL, CARD_LEVEL = 70, 135
MIN_PIXELS = 30
MIN_SPECK = 12


@dataclass(frozen=True)
class Feature:
    mask: np.ndarray  # SIZE x SIZE float32 silhouette in [0, 1]
    lab: np.ndarray  # SIZE x SIZE x 3 float32 Lab
    aspect: float


@dataclass(frozen=True)
class SlotMatch:
    slot: int
    box: tuple[int, int, int, int]
    top: tuple[tuple[str, float], ...]  # index names incl. form, best first
    species: tuple[tuple[str, float], ...]  # base names, best first

    @property
    def name(self) -> str:
        return self.top[0][0] if self.top else ""


class Scorer(Protocol):
    names: list[str]
    base_names: dict[str, str]

    def score(self, features: list[Feature]) -> np.ndarray:
        """(len(features), len(names)) similarity of each feature to each reference view."""
        ...


# Features ----------------------------------------------------------------------------------------
def border_colour(crop: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [crop[:3].reshape(-1, 3), crop[-3:].reshape(-1, 3), crop[:, :3].reshape(-1, 3), crop[:, -3:].reshape(-1, 3)]
    )
    return np.median(border, axis=0)


def segment(crop: np.ndarray, background=None, light_streaks: bool = False) -> np.ndarray:
    """Boolean silhouette: pixels away from the background colour, specks dropped, holes filled."""
    colour = border_colour(crop) if background is None else np.asarray(background)
    mask = np.abs(crop.astype(np.int16) - colour.astype(np.int16)).sum(axis=2) > BG_DISTANCE
    if light_streaks:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.int16)
        pixel = np.asarray(colour, dtype=np.uint8).reshape(1, 1, 3)
        bg_hue = int(cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0, 0])
        hue_gap = np.abs(hsv[..., 0] - bg_hue)
        hue_gap = np.minimum(hue_gap, 180 - hue_gap)
        mask &= ~((hue_gap <= STREAK_HUE) & (hsv[..., 1] >= STREAK_SAT))
    mask = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros(mask.shape, bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    x, y, w, h = (stats[1:, i] for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    on_border = (x == 0) | (y == 0) | (x + w == mask.shape[1]) | (y + h == mask.shape[0])
    keep = (
        np.flatnonzero((areas >= max(MIN_SPECK, 0.03 * areas.max())) & ~(on_border & (areas < 0.2 * areas.max()))) + 1
    )
    mask = np.isin(labels, keep).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    flood = np.pad(mask, 1).copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return (mask > 0) | holes


def feature(crop: np.ndarray, background=None, light_streaks: bool = False) -> Feature | None:
    mask = segment(crop, background, light_streaks)
    ys, xs = np.nonzero(mask)
    if len(ys) < MIN_PIXELS:
        return None
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = bottom - top, right - left
    scale = (SIZE - 2) / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    y0, x0 = (SIZE - nh) // 2, (SIZE - nw) // 2
    sub = crop[top:bottom, left:right].copy()
    sub_mask = mask[top:bottom, left:right].astype(np.float32)
    sub[sub_mask == 0] = 0
    small_mask = cv2.resize(sub_mask, (nw, nh), interpolation=cv2.INTER_AREA)
    bgr = cv2.resize(sub.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    bgr = np.clip(bgr / np.maximum(small_mask, 1e-3)[..., None], 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    out_mask = np.zeros((SIZE, SIZE), np.float32)
    out_lab = np.zeros((SIZE, SIZE, 3), np.float32)
    out_mask[y0 : y0 + nh, x0 : x0 + nw] = small_mask
    out_lab[y0 : y0 + nh, x0 : x0 + nw] = lab
    return Feature(out_mask, out_lab, w / h)


def undim(crop: np.ndarray) -> np.ndarray:
    level = float(border_colour(crop).max())
    if level >= DIM_LEVEL or level <= 0:
        return crop
    return np.clip(crop.astype(np.float32) * (CARD_LEVEL / level), 0, 255).astype(np.uint8)


def top(scorer: Scorer, scores: np.ndarray, k: int, species: bool = False) -> list[tuple[str, float]]:
    out, seen = [], set()
    for i in np.argsort(-scores, kind="stable"):
        label = scorer.base_names[scorer.names[i]] if species else scorer.names[i]
        if label not in seen:
            seen.add(label)
            out.append((label, round(float(scores[i]), 4)))
            if len(out) == k:
                break
    return out


# The network --------------------------------------------------------------------------------------
class NetScorer:
    def __init__(self, net: sprite_net.SpriteNet, domain: str) -> None:
        self.net = net
        refs = net.references[domain]
        self.names, self.base_names, self.vectors = refs.names, refs.base_names, refs.vectors

    def score(self, features: list[Feature]) -> np.ndarray:
        if not features:
            return np.zeros((0, len(self.names)), np.float32)
        vectors = self.net.embed(
            np.stack([f.mask for f in features]),
            np.stack([f.lab for f in features]),
            np.array([f.aspect for f in features]),
        )
        return vectors @ self.vectors.T


@lru_cache(maxsize=1)
def network() -> sprite_net.SpriteNet:
    return sprite_net.load(paths.assets_dir() / "sprites" / "sprite_net.json")


@lru_cache(maxsize=2)
def scorer(domain: str) -> NetScorer:
    """ "preview" (team selection cards) or "log" (battle-log rows), loaded once."""
    return NetScorer(network(), domain)


# Team preview ------------------------------------------------------------------------------------
def preview_features(reference: np.ndarray) -> list[Feature | None]:
    """The six opponent cards of a reference-size team-selection frame (None: nothing segmented)."""
    return [feature(undim(reference[box[1] : box[3], box[0] : box[2]]), light_streaks=True) for box in OPPONENT_BOXES]


def preview_matches(features: list[Feature | None], lib: Scorer, k: int = 3) -> list[SlotMatch]:
    present = [f for f in features if f is not None]
    scores = iter(lib.score(present))
    out = []
    for i, (box, f) in enumerate(zip(OPPONENT_BOXES, features, strict=True)):
        if f is None:
            out.append(SlotMatch(i, box, (), ()))
        else:
            row = next(scores)
            out.append(SlotMatch(i, box, tuple(top(lib, row, k)), tuple(top(lib, row, k, species=True))))
    return out


def match_team_preview(reference: np.ndarray, k: int = 3, lib: Scorer | None = None) -> list[SlotMatch]:
    return preview_matches(preview_features(reference), lib or scorer("preview"), k)


# Battle log ---------------------------------------------------------------------------------------
LogVariant = Feature | str  # a feature, or "none" (no sprite) / "clipped" (cut by the panel edge)


def log_variant(image: np.ndarray) -> LogVariant:
    mask = segment(image)
    if np.count_nonzero(mask) < MIN_PIXELS:
        return "none"
    if np.any(mask[:2]) or np.any(mask[-2:]) or np.any(mask[:, :2]) or np.any(mask[:, -2:]):
        return "clipped"
    found = feature(image)
    return "none" if found is None else found


def without_panel(image: np.ndarray) -> np.ndarray:
    """Arena patterns show through the panel's blue: paint them with the panel colour."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    panel = (hsv[:, :, 0] >= 118) & (hsv[:, :, 0] <= 138) & (hsv[:, :, 1] >= 110)
    cleaned = image.copy()
    cleaned[panel] = border_colour(image)
    return cleaned


def log_decision(
    variant: LogVariant, scores: np.ndarray | None, lib: Scorer, box: list[int], min_score: float, min_margin: float
) -> dict:
    result: dict = {"status": "unknown", "name": None, "base_name": None, "box": box, "candidates": []}
    if variant == "none":
        return {**result, "status": "none"}
    if variant == "clipped":
        return {**result, "reason": "clipped_sprite"}
    assert scores is not None
    candidates = top(lib, scores, 3)
    name, score = candidates[0]
    margin = score - candidates[1][1] if len(candidates) > 1 else score
    result.update(candidates=candidates, score=score, margin=round(margin, 4))
    if score >= min_score and margin >= min_margin:
        result.update(status="matched", name=name, base_name=lib.base_names[name])
    return result


def needs_cleaning(raw: dict) -> bool:
    return raw["status"] not in ("matched", "none")


def pick(raw: dict, cleaned: dict) -> dict:
    return cleaned if cleaned["status"] != "unknown" or cleaned.get("score", 0) > raw.get("score", 0) else raw


def match_log_row(
    reference: np.ndarray,
    bounds: tuple[int, int],
    *,
    lib: Scorer | None = None,
    min_score: float | None = None,
    min_margin: float | None = None,
) -> dict:
    """The sprite at the left of one battle-log row: matched, none, or unknown."""
    lib = lib or scorer("log")
    min_score = network().log_min_score if min_score is None else min_score
    min_margin = network().log_min_margin if min_margin is None else min_margin
    top_y, bottom_y = bounds
    image = reference[top_y:bottom_y, LOG_SPRITE_COLUMN[0] : LOG_SPRITE_COLUMN[1]]
    box = [LOG_SPRITE_COLUMN[0], top_y, LOG_SPRITE_COLUMN[1], bottom_y]

    def decide(crop: np.ndarray) -> dict:
        variant = log_variant(crop)
        scores = lib.score([variant])[0] if isinstance(variant, Feature) else None
        return log_decision(variant, scores, lib, box, min_score, min_margin)

    raw = decide(image)
    return pick(raw, decide(without_panel(image))) if needs_cleaning(raw) else raw


def combine_observations(first: dict | None, second: dict | None) -> dict | None:
    """Keep clear observations across overlapping log views; conflicting matches become unknown."""
    if not first:
        return second
    if not second:
        return first
    if first.get("reason") == "conflicting_sprites" or second.get("reason") == "conflicting_sprites":
        return first if first.get("reason") == "conflicting_sprites" else second
    if first["status"] == second["status"] == "matched" and first["name"] != second["name"]:
        return {
            "status": "unknown",
            "name": None,
            "base_name": None,
            "reason": "conflicting_sprites",
            "observations": [first, second],
        }
    return first if first["status"] == "matched" else second
