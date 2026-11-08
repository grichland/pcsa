"""The retired thumbnail matcher, kept for training the sprite network and measuring it.

Its two indexes (`roster_index` for team preview, `log_index` for battle-log rows: 40x40 silhouettes and
Lab colour of every known sprite view) are game imagery, so they live outside git, in a local source
folder (default `output/sprite_source/`). They are the network's reference views, and this scorer is
the baseline the network must match or beat. Comparison is silhouette IoU times Lab colour agreement,
best over a few pixel shifts, with an aspect-ratio factor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcsa.common.errors import ConfigError
from pcsa.phone.sprites import SIZE, Feature

SOURCES = {"preview": "roster_index", "log": "log_index"}
SHIFTS = (-2, -1, 0, 1, 2)
COLOUR_SCALE = 18.0
SHAPE_FLOOR = {"preview": 0.15, "log": 0.65}
LOG_MIN_SCORE, LOG_MIN_MARGIN = 0.52, 0.07
PARAMETERS = {
    "algorithm": "silhouette-lab/1",
    "size": SIZE,
    "background_distance": 60,
    "shifts": list(SHIFTS),
    "colour_scale": COLOUR_SCALE,
    "streak_hue": 8,
    "streak_saturation": 120,
    "shape_floor": 0.15,
    "dim_level": 70,
    "card_level": 135,
}


@dataclass(frozen=True)
class SourceIndex:
    names: list[str]
    base_names: list[str]
    masks: np.ndarray  # (n, SIZE, SIZE)
    labs: np.ndarray  # (n, SIZE, SIZE, 3)
    aspects: np.ndarray  # (n,)


def load_source(folder: Path, domain: str) -> SourceIndex:
    path = folder / f"{SOURCES[domain]}.npz"
    if not path.is_file():
        raise ConfigError(f"{path} is missing: the sprite source indexes are local-only (see docs/sprites.md)")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "pcsa-sprite-index/1" or metadata.get("feature_parameters", PARAMETERS) != (
        PARAMETERS
    ):
        raise ConfigError(f"unsupported sprite index version or parameters: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("npz_sha256"):
        raise ConfigError(f"sprite index archive hash does not match its metadata: {path}")
    with np.load(path, allow_pickle=False) as archive:
        masks, labs, aspects = archive["masks"], archive["labs"], archive["aspects"]
        names, bases = archive["names"].tolist(), archive["base_names"].tolist()
    if masks.shape != (len(names), SIZE, SIZE) or labs.shape != (len(names), SIZE, SIZE, 3):
        raise ConfigError(f"invalid sprite index array dimensions: {path}")
    return SourceIndex(names, bases, masks.astype(np.float32), labs.astype(np.float32), aspects.astype(np.float32))


def compare(query: Feature, masks, labs, aspects, shifts=SHIFTS, shape_floor: float = 0.15) -> np.ndarray:
    best = np.zeros(len(masks), np.float32)
    for dy in shifts:
        for dx in shifts:
            qm = np.roll(query.mask, (dy, dx), axis=(0, 1))
            ql = np.roll(query.lab, (dy, dx), axis=(0, 1))
            inter = np.minimum(qm, masks)
            union = np.maximum(qm, masks).sum(axis=(1, 2))
            overlap = inter.sum(axis=(1, 2))
            iou = overlap / np.maximum(union, 1e-6)
            dist = (np.abs(ql - labs).sum(axis=3) * inter).sum(axis=(1, 2)) / np.maximum(overlap, 1e-6)
            colour = np.exp(-dist / COLOUR_SCALE)
            best = np.maximum(best, iou * (shape_floor + (1 - shape_floor) * colour))
    ratio = np.minimum(aspects, query.aspect) / np.maximum(aspects, query.aspect)
    return best * (0.5 + 0.5 * ratio)


class BaselineScorer:
    """The old matcher behind the `sprites.Scorer` interface: a coarse pass, then shifts on a shortlist."""

    def __init__(self, source: SourceIndex, domain: str, shortlist: int = 40) -> None:
        self.source = source
        self.names = source.names
        self.base_names = dict(zip(source.names, source.base_names, strict=True))
        self.shape_floor = SHAPE_FLOOR[domain]
        self.shortlist = shortlist

    def score(self, features: list[Feature]) -> np.ndarray:
        s = self.source
        out = np.zeros((len(features), len(self.names)), np.float32)
        for row, q in enumerate(features):
            coarse = compare(q, s.masks, s.labs, s.aspects, shifts=(0,), shape_floor=self.shape_floor)
            keep = np.argsort(-coarse)[: self.shortlist]
            fine = compare(q, s.masks[keep], s.labs[keep], s.aspects[keep], shape_floor=self.shape_floor)
            coarse[keep] = np.maximum(coarse[keep], fine)
            out[row] = coarse
        return out
