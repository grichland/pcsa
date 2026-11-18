"""Text recognition: RapidOCR with the pinned PP-OCRv5 models, behind a small protocol so tests can cache."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pcsa.common import paths
from pcsa.common.config import OcrConfig
from pcsa.common.errors import ConfigError

REFERENCE = (2340, 1080)


@dataclass(frozen=True)
class TextBox:
    text: str
    confidence: float
    bounds: tuple[float, float, float, float]  # left, top, right, bottom (reference frame)

    @property
    def center(self) -> tuple[float, float]:
        return (self.bounds[0] + self.bounds[2]) / 2, (self.bounds[1] + self.bounds[3]) / 2

    @property
    def norm(self) -> str:
        return normalize_text(self.text)

    def record(self) -> dict:
        return {"text": self.text, "confidence": round(self.confidence, 3), "bounds": [round(v) for v in self.bounds]}


def normalize_text(text: str) -> str:
    """Casefold, strip accents, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


class TextReader(Protocol):
    min_confidence: float

    def read(self, frame: np.ndarray) -> list[TextBox]: ...

    def line(self, crop: np.ndarray) -> tuple[str, float] | None: ...


class RapidOcrReader:
    """One detection+recognition engine and one recognition-only engine over the pinned models."""

    def __init__(self, config: OcrConfig, model_dir: Path | None = None) -> None:
        from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        model_dir = model_dir or paths.tool("ocr")
        det, rec = model_dir / config.det_model, model_dir / config.rec_model
        for model in (det, rec):
            if not model.is_file():
                raise ConfigError(f"OCR model not found: {model} (run pcsa setup)")
        common = {
            "Global.model_root_dir": str(model_dir),
            "Global.use_cls": False,
            "Global.max_side_len": REFERENCE[0],
            "Global.text_score": 0.5,
            "EngineConfig.onnxruntime.intra_op_num_threads": config.threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Rec.model_path": str(rec),
            "Rec.lang_type": LangRec.EN,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
        }
        self.full = RapidOCR(
            params={
                **common,
                "Det.model_path": str(det),
                "Det.lang_type": LangDet.MULTI,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE,
                "Det.limit_side_len": config.det_limit_side_len,
            }
        )
        self.rec = RapidOCR(params={**common, "Global.use_det": False, "Det.model_path": str(det)})
        self.min_confidence = config.min_confidence

    def warm(self) -> None:
        """Run both engines once so the first live turn does not pay the session start-up."""
        import cv2

        blank = np.full((REFERENCE[1], REFERENCE[0], 3), 40, np.uint8)
        cv2.putText(blank, "Fight", (2000, 700), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        self.read(blank)
        self.line(blank[620:720, 1980:2300])

    def read(self, frame: np.ndarray) -> list[TextBox]:
        result: Any = self.full(np.ascontiguousarray(frame))
        txts = getattr(result, "txts", None)
        boxes = []
        if txts is not None and len(txts):
            for text, score, box in zip(txts, result.scores, result.boxes, strict=True):
                xs, ys = zip(*box, strict=True)
                boxes.append(TextBox(str(text), float(score), (min(xs), min(ys), max(xs), max(ys))))
        return boxes

    def line(self, crop: np.ndarray) -> tuple[str, float] | None:
        result: Any = self.rec(np.ascontiguousarray(crop), use_det=False)
        txts = getattr(result, "txts", None)
        if not txts or len(txts) != 1 or not str(txts[0]).strip():
            return None
        return str(txts[0]).strip(), float(result.scores[0])
