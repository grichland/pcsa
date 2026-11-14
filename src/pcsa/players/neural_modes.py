"""The neural player's feature mode (moved out of pcsa.players.neural for its length limit)."""

from __future__ import annotations

from pcsa.common import log
from pcsa.common.errors import ConfigError


def feature_mode(setting: str, trained: str, checkpoint: str = "the checkpoint") -> str:
    """The feature mode to play with: the checkpoint's (`trained`) unless `setting` names another. ConfigError when
    that would feed belief columns to a network trained without them."""
    if setting in ("checkpoint", trained):
        return trained
    if trained == "base":
        raise ConfigError(
            f"features: {setting} asked for, but {checkpoint} was trained with feature mode base: an old network "
            "cannot read belief columns (set features: checkpoint, or point policy.neural_checkpoint at a belief one)"
        )
    log.warning("neural_features_override", trained=trained, used=setting, checkpoint=checkpoint)
    return setting
