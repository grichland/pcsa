"""`pcsa train sprites` end to end on CPU: made-up reference views and runs in both label formats, a few steps."""

from __future__ import annotations

import argparse
import hashlib
import json

import cv2
import numpy as np
import pytest

from pcsa.common import paths
from pcsa.phone import sprites
from pcsa.sprite_training import sprite_baseline as sb
from pcsa.sprite_training import sprite_data as sd

SPECIES = {"Tauros": (40, 90, 200), "Kommo-o": (60, 200, 230), "Farfetch’d": (200, 120, 60)}
BOX = [390, 20, 545, 154]


def blob(colour, shape: int) -> np.ndarray:
    """A sprite on the log panel's colour: an ellipse whose shape differs per species."""
    image = np.full((BOX[3] - BOX[1], BOX[2] - BOX[0], 3), (150, 60, 20), np.uint8)
    axes = ((30, 50), (50, 30), (40, 40))[shape]
    cv2.ellipse(image, (77, 67), axes, 0, 0, 360, colour, -1)
    return image


def write_source(folder) -> None:
    names = list(SPECIES)
    feats = [sprites.feature(blob(c, i)) for i, c in enumerate(SPECIES.values())]
    assert all(feats)
    for index in sb.SOURCES.values():
        path = folder / f"{index}.npz"
        np.savez(
            path,
            names=np.array(names),
            base_names=np.array(names),
            masks=np.stack([f.mask for f in feats if f]),
            labs=np.stack([f.lab for f in feats if f]),
            aspects=np.array([f.aspect for f in feats if f], np.float32),
        )
        meta = {
            "schema_version": "pcsa-sprite-index/1",
            "feature_parameters": sb.PARAMETERS,
            "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def write_battle(run, n: int, events: bool) -> None:
    folder = run / f"battle-{n}"
    (folder / "log-frames").mkdir(parents=True)
    rows = []
    for i, (name, colour) in enumerate(SPECIES.items()):
        frame = np.full((300, 2340, 3), (150, 60, 20), np.uint8)
        frame[BOX[1] : BOX[3], BOX[0] : BOX[2]] = blob(colour, i)
        source = f"battle-{n}/log-frames/visit01-{i}.png"
        cv2.imwrite(str(run / source), frame)
        sprite = {"status": "matched", "name": name, "base_name": name, "box": BOX, "source": source}
        rows.append({"text": f"Move: The opposing {name} used Tackle!", "sprite": sprite})
    (folder / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if events:
        body = {"events": [{"k": "enter", "side": "opp", "mon": "tauros"}]}
        (folder / "events.json").write_text(json.dumps(body), encoding="utf-8")
    else:
        (folder / "showdown.log").write_text("|switch|p2a: Tauros|Tauros, L50|100/100\n", encoding="utf-8")


@pytest.mark.torch
def test_the_trainer_extracts_trains_measures_and_exports(tmp_path, monkeypatch) -> None:
    import torch

    from pcsa.phone import sprite_net
    from pcsa.sprite_training import sprite_train

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(paths.OUTPUT_ENV, str(tmp_path / "out"))
    (tmp_path / "source").mkdir()
    write_source(tmp_path / "source")
    for r, events in enumerate((False, False, True)):
        run = tmp_path / "runs" / f"2026111{r}-000000-run"
        for n in range(1, 4):
            write_battle(run, n, events)
    args = argparse.Namespace(
        source=str(tmp_path / "source"), runs=str(tmp_path / "runs"), out=str(tmp_path / "result"), steps=3,
        seed=0, test_runs=1, holdout_species=0.0, refresh_data=True, install=False,
    )  # fmt: skip
    assert sprite_train.run(args, log=lambda _: None) == 0

    data = sd.load(tmp_path / "out" / "sprite_net" / "data.npz")
    assert len(data.species) == 27 and set(data.species) == {"tauros", "kommoo", "farfetchd"}
    meta = json.loads((tmp_path / "out" / "sprite_net" / "data.json").read_text(encoding="utf-8"))
    assert {b["labels"] for b in meta["battles"]} == {"showdown", "events"}
    assert all(b["opponent_sent"] == {"tauros": ["tauros"]} for b in meta["battles"])
    net = sprite_net.load(tmp_path / "result" / "sprite_net.json")
    assert set(net.references) == {"preview", "log"}
    report = json.loads((tmp_path / "result" / "report.json").read_text(encoding="utf-8"))
    assert report["test"]["log"]["network"]["rows"] == 9
