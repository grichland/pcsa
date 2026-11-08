"""`pcsa calibrate`: find where the game draws in this phone's frame and write its profile.

With the game on the lobby, capture a frame and look for the Begin Matchmaking label. If the frame
has the reference aspect ratio and the label sits where the reference layout puts it, the viewport
is the whole frame. If the frame is wider or taller, the game is assumed to letterbox at the
reference aspect ratio, centred; the label position confirms or refutes that guess.
"""

from __future__ import annotations

import numpy as np

from pcsa.common import log
from pcsa.common.config import Viewport
from pcsa.common.errors import ConfigError
from pcsa.phone.ocr import REFERENCE, TextReader
from pcsa.phone.profile import DeviceProfile
from pcsa.phone.screens import to_reference
from pcsa.phone.texts import Texts

REFERENCE_ASPECT = REFERENCE[0] / REFERENCE[1]
ASPECT_TOLERANCE = 0.01
LABEL_TOLERANCE = 0.03  # of the frame, after mapping through the proposed viewport
BEGIN_REGION = (1300, 905, 1900, 962)


def letterboxed_viewport(width: int, height: int) -> Viewport:
    """The largest reference-aspect rectangle centred in a frame."""
    if width / height > REFERENCE_ASPECT:
        w = round(height * REFERENCE_ASPECT)
        return Viewport(x=(width - w) // 2, y=0, w=w, h=height)
    h = round(width / REFERENCE_ASPECT)
    return Viewport(x=0, y=(height - h) // 2, w=width, h=h)


def propose_viewport(width: int, height: int) -> Viewport:
    if abs(width / height - REFERENCE_ASPECT) <= ASPECT_TOLERANCE:
        return Viewport.of_frame(width, height)
    return letterboxed_viewport(width, height)


def label_offset(reader: TextReader, image: np.ndarray, viewport: Viewport) -> tuple[float, float] | None:
    """How far Begin Matchmaking sits from its reference position, as a fraction of the reference frame."""
    reference = to_reference(image, viewport)
    t = Texts(reader.read(reference), reader.min_confidence)
    found = t.find(r"begin matchmaking")
    if not found:
        return None
    cx, cy = found[0].center
    ex, ey = (BEGIN_REGION[0] + BEGIN_REGION[2]) / 2, (BEGIN_REGION[1] + BEGIN_REGION[3]) / 2
    return (cx - ex) / REFERENCE[0], (cy - ey) / REFERENCE[1]


def calibrate(reader: TextReader, image: np.ndarray, model: str) -> DeviceProfile:
    height, width = image.shape[:2]
    viewport = propose_viewport(width, height)
    offset = label_offset(reader, image, viewport)
    if offset is None:
        raise ConfigError("Begin Matchmaking was not read; open the game on the Ranked Singles lobby and retry")
    if max(abs(offset[0]), abs(offset[1])) > LABEL_TOLERANCE:
        raise ConfigError(
            f"the lobby label is {offset[0]:+.3f},{offset[1]:+.3f} of the frame away from where the reference layout "
            "puts it; this phone's layout differs from the reference and needs a manual profile"
        )
    note = "full frame" if viewport == Viewport.of_frame(width, height) else "letterboxed to the reference aspect"
    log.event(
        "calibrated",
        model=model,
        size=[width, height],
        viewport=viewport.model_dump(),
        offset=[round(v, 4) for v in offset],
    )
    return DeviceProfile(
        model=model,
        frame_size=(width, height),
        viewport=viewport,
        calibrated_at=DeviceProfile.full_frame(model, width, height).calibrated_at,
        note=note,
    )
