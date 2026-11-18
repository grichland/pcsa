"""Per-phone profile: where the game draws inside the captured frame. Written by `pcsa calibrate`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from pcsa.common import paths
from pcsa.common.config import Strict, Viewport
from pcsa.common.errors import ConfigError


class DeviceProfile(Strict):
    model: str
    frame_size: tuple[int, int]
    viewport: Viewport
    calibrated_at: str = ""
    note: str = ""

    @classmethod
    def full_frame(cls, model: str, width: int, height: int, note: str = "") -> DeviceProfile:
        return cls(
            model=model,
            frame_size=(width, height),
            viewport=Viewport.of_frame(width, height),
            calibrated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            note=note,
        )


def profile_path(model: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)
    return paths.config_dir() / "devices" / f"{safe}.yaml"


def load_profile(model: str) -> DeviceProfile | None:
    path = profile_path(model)
    if not path.is_file():
        return None
    try:
        return DeviceProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as error:
        raise ConfigError(f"invalid device profile {path}: {error}") from None


def save_profile(profile: DeviceProfile) -> Path:
    path = profile_path(profile.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return path
