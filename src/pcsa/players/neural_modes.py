"""The neural player's feature mode and the guard rules' threat estimates."""

from __future__ import annotations

from typing import Any

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


def load_threat() -> tuple[Any, Any]:
    """(build catalog with the builds seen on the live ladder first, opponent tracker), for the guard rules."""
    from pcsa.battle.builds import Catalog
    from pcsa.players.opponent import Tracker

    catalog = Catalog.load(seen=True)
    return catalog, Tracker(catalog)
