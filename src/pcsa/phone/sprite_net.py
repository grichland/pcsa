"""The sprite embedding network, run in numpy: silhouette features -> unit vectors compared by cosine.

A crop's silhouette feature (`sprites.feature`: a 40x40 mask and the Lab colour under it, plus the
silhouette's aspect ratio) becomes four channels; six 3x3 convolutions with ReLU, a 2x2 max-pool after
every second one, global average pooling, the log aspect ratio and two dense layers map it to a unit
vector. Two sprites match by the cosine of their vectors.

`assets/sprites/sprite_net.json` names the weights and the reference vectors (one per known sprite
view, for team preview and for the battle log) with their hashes, and the battle log's acceptance
thresholds. All of it is produced by `pcsa train sprites` (sprite_training/sprite_train.py, which also checks
this forward pass against torch's); no sprite pixels are shipped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pcsa.common.errors import ConfigError

SCHEMA = "pcsa-sprite-net/1"
SIZE = 40
CONVS = (("c1", 4, 32), ("c2", 32, 32), ("c3", 32, 64), ("c4", 64, 64), ("c5", 64, 128), ("c6", 128, 128))
POOL_AFTER = frozenset({"c2", "c4", "c6"})
HIDDEN, EMBED = 128, 64
LAB_CENTRE, LAB_SPREAD = 128.0, 64.0
DOMAINS = ("preview", "log")


def encode(masks: np.ndarray, labs: np.ndarray, aspects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 40, 40) masks, (n, 40, 40, 3) Lab and (n,) aspect ratios -> network inputs (n, 4, 40, 40), (n,)."""
    masks = masks.astype(np.float32)
    labs = labs.astype(np.float32)
    x = np.empty((len(masks), 4, SIZE, SIZE), np.float32)
    x[:, 0] = masks
    x[:, 1] = labs[..., 0] / 255.0 * masks
    x[:, 2] = (labs[..., 1] - LAB_CENTRE) / LAB_SPREAD * masks
    x[:, 3] = (labs[..., 2] - LAB_CENTRE) / LAB_SPREAD * masks
    return x, np.log(np.asarray(aspects, np.float32))


def weight_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for name, cin, cout in CONVS:
        shapes[f"{name}.w"], shapes[f"{name}.b"] = (cout, cin, 3, 3), (cout,)
    shapes["f1.w"], shapes["f1.b"] = (HIDDEN, CONVS[-1][2] + 1), (HIDDEN,)
    shapes["f2.w"], shapes["f2.b"] = (EMBED, HIDDEN), (EMBED,)
    return shapes


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, c, h, wd = x.shape
    windows = sliding_window_view(np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1))), (3, 3), axis=(2, 3))
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * wd, c * 9)
    return (cols @ w.reshape(len(w), -1).T + b).reshape(n, h, wd, -1).transpose(0, 3, 1, 2)


def _pool2(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def forward(weights: dict[str, np.ndarray], x: np.ndarray, log_aspect: np.ndarray) -> np.ndarray:
    """Network inputs -> (n, EMBED) unit vectors."""
    for name, _, _ in CONVS:
        x = np.maximum(_conv3x3(x, weights[f"{name}.w"], weights[f"{name}.b"]), 0.0)
        if name in POOL_AFTER:
            x = _pool2(x)
    v = np.concatenate([x.mean(axis=(2, 3)), log_aspect[:, None]], axis=1)
    v = np.maximum(v @ weights["f1.w"].T + weights["f1.b"], 0.0)
    v = v @ weights["f2.w"].T + weights["f2.b"]
    return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)).astype(np.float32)


@dataclass(frozen=True)
class References:
    names: list[str]  # index names incl. form, as the game's screens are matched to them
    base_names: dict[str, str]
    vectors: np.ndarray  # (n, EMBED) unit vectors, one per view (a name may have several views)


@dataclass(frozen=True)
class SpriteNet:
    weights: dict[str, np.ndarray]
    references: dict[str, References]  # by domain: "preview" (team selection cards) and "log" (battle-log rows)
    log_min_score: float
    log_min_margin: float

    def embed(self, masks: np.ndarray, labs: np.ndarray, aspects: np.ndarray) -> np.ndarray:
        return forward(self.weights, *encode(masks, labs, aspects))


def _checked(folder: Path, entry: dict) -> Path:
    path = folder / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ConfigError(f"{path} does not match the hash in sprite_net.json")
    return path


def load(metadata_path: Path) -> SpriteNet:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
        raise ConfigError(f"unsupported sprite network: {metadata_path}")
    folder = metadata_path.parent
    with np.load(_checked(folder, metadata["weights"]), allow_pickle=False) as archive:
        weights = {key: archive[key].astype(np.float32) for key in archive.files}
    expected = weight_shapes()
    if {k: v.shape for k, v in weights.items()} != expected:
        raise ConfigError("sprite network weights do not fit the architecture in sprite_net.py")
    references = {}
    for domain in DOMAINS:
        with np.load(_checked(folder, metadata["references"][domain]), allow_pickle=False) as archive:
            names, bases, vectors = archive["names"].tolist(), archive["base_names"].tolist(), archive["vectors"]
        if vectors.shape != (len(names), EMBED) or len(bases) != len(names):
            raise ConfigError(f"invalid {domain} reference vectors")
        vectors = vectors.astype(np.float32)
        vectors.setflags(write=False)
        references[domain] = References(names, dict(zip(names, bases, strict=True)), vectors)
    thresholds = metadata["log_thresholds"]
    return SpriteNet(weights, references, float(thresholds["min_score"]), float(thresholds["min_margin"]))
