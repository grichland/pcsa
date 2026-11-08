"""Build the perception corpus from saved run frames: a few real screenshots per screen, with cached OCR.

    uv run python scripts/curate_frames.py --source output/runs --per-screen 3

Frames whose saved name carries a screen label (<run>/frames/NNNNN-<tag>-<screen>.jpg) are candidates, and
battle-log panel captures (<run>/battle-N/log-frames/*.png) stand for the log screen. Each is
classified with the current rule table and real OCR; the OCR boxes are cached so the corpus test
runs without the OCR engine. Agreement with the saved label is recorded; disagreements are kept in
review.json for a human to settle rather than silently accepted.

The screenshots are git-ignored (they are game captures); only expected.json, ocr_cache.json and review.json
are tracked. The corpus tests that need pixels skip in a checkout without the screenshots.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcsa.common import paths  # noqa: E402
from pcsa.common.config import load_config  # noqa: E402
from pcsa.phone.ocr import RapidOcrReader  # noqa: E402
from pcsa.phone.screens import classify, to_reference  # noqa: E402

LABEL = re.compile(r"^\d{5}-[a-z-]+-([a-z_]+)\.jpg$")
WANTED = (
    "lobby", "action", "moves", "party", "forced", "log", "team_select", "result", "rewards", "options",
    "matchmaking", "prep", "transition", "wait", "rank_up", "forfeit_msg", "unknown",
)  # fmt: skip
JPEG_QUALITY = 80


def candidates(source: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(source.glob("*/frames/*.jpg")):
        match = LABEL.match(path.name)
        if match and match[1] in WANTED:
            found[match[1]].append(path)
    for path in sorted(source.glob("*/battle-*/log-frames/*.png"))[:400]:
        found["log"].append(path)
    return found


def spread(items: list[Path], count: int, rng: random.Random) -> list[Path]:
    """Pick from different runs where possible."""
    by_run: dict[str, list[Path]] = defaultdict(list)
    for path in items:
        by_run[path.parts[-3] if "log-frames" not in path.parts else path.parts[-4]].append(path)
    runs = list(by_run)
    rng.shuffle(runs)
    picked: list[Path] = []
    while len(picked) < count and runs:
        for run in list(runs):
            if len(picked) >= count:
                break
            pool = by_run[run]
            if not pool:
                runs.remove(run)
                continue
            picked.append(pool.pop(rng.randrange(len(pool))))
    return picked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default=str(ROOT / "tests/fixtures/frames"))
    parser.add_argument("--per-screen", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    config = load_config(paths.config_dir() / "agent.yaml")
    reader = RapidOcrReader(config.agent.ocr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    expected: dict[str, dict] = {}
    cache: dict[str, list[dict]] = {}
    review: list[dict] = []
    for label, paths_ in sorted(candidates(Path(args.source)).items()):
        for path in spread(paths_, args.per_screen, rng):
            image = cv2.imread(str(path))
            if image is None:
                continue
            reference = to_reference(image)
            boxes = reader.read(reference)
            obs = classify(boxes, reference, reader.min_confidence, reader.line)
            name = f"{label}-{len([k for k in expected if k.startswith(label + '-')]):02d}.jpg"
            cv2.imwrite(str(out / name), reference, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cache[name] = [b.record() for b in boxes]
            agrees = label == str(obs.screen)
            expected[name] = {
                "screen": str(obs.screen),
                "reason": obs.reason,
                "old_label": label,  # the label saved with the frame (the key tests/perception/test_corpus.py reads)
                "agrees": agrees,
                "source": str(path.relative_to(args.source)),
            }
            if not agrees:
                review.append({"frame": name, "saved": label, "classified": str(obs.screen), "reason": obs.reason})
            print(
                f"{name:28} saved={label:12} classified={obs.screen:14} {'' if agrees else '<- REVIEW'} {obs.reason}",
                flush=True,
            )
    (out / "expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")
    (out / "ocr_cache.json").write_text(json.dumps(cache), encoding="utf-8")
    (out / "review.json").write_text(json.dumps(review, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.glob("*.jpg")) / 1e6
    print(f"\n{len(expected)} frames, {size:.1f} MB of JPEG, {len(review)} to review -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
