"""Contact sheets of sampled rows: the row as on screen above the text before and after the fix.

Run: python -m tools.ocr_eval.sheets  ->  docs/investigation/ocr-fix/sheets/{changed,unchanged}NN.png
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUNS = Path(__file__).resolve().parents[2] / "output" / "runs"
FIX = Path(__file__).resolve().parents[2] / "docs" / "investigation" / "ocr-fix"
PER_SHEET = 10
WIDTH = 1000


def font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 20)


def tile(number: int, pair: dict) -> np.ndarray:
    before, after = pair
    reading = next((r for r in after.get("readings") or [] if r.get("box")), None)
    source, box = (
        (reading["source"], reading["box"]) if reading else (before["sprite"]["source"], before["sprite"]["box"])
    )
    frame = cv2.imread(str(RUNS / after["run"] / source))
    crop = frame[box[1] : box[3], 390:1885] if frame is not None else np.zeros((134, 1495, 3), np.uint8)
    crop = cv2.resize(crop, (WIDTH, round(crop.shape[0] * WIDTH / crop.shape[1])))
    panel = Image.new("RGB", (WIDTH, 62), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), f"#{number} B: {before['text']}", fill="black", font=font())
    draw.text((6, 32), f"A[{after.get('read')}]: {after['text']}", fill="darkblue", font=font())
    text = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
    return np.vstack([crop, text, np.full((6, WIDTH, 3), 255, np.uint8)])


def main() -> None:
    samples = json.loads((FIX / "samples.json").read_text(encoding="utf-8"))
    (FIX / "sheets").mkdir(parents=True, exist_ok=True)
    index = []
    for kind, pairs in samples.items():
        for start in range(0, len(pairs), PER_SHEET):
            tiles = [tile(start + i, p) for i, p in enumerate(pairs[start : start + PER_SHEET])]
            name = f"{kind}{start // PER_SHEET:02d}.png"
            cv2.imwrite(str(FIX / "sheets" / name), np.vstack(tiles))
            index += [{"sheet": name, "n": start + i, "before": p[0]["text"], "after": p[1]["text"],
                       "read": p[1].get("read"), "run": p[1]["run"], "battle": p[1]["battle"],
                       "turn": p[1]["turn"], "index": p[1]["index"]} for i, p in enumerate(pairs[start : start + PER_SHEET])]  # fmt: skip
    (FIX / "sheets" / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
