"""`pcsa sim play` pairing: each team pair is played both ways, from one shared pool."""

from __future__ import annotations

import random

import pytest

from pcsa.battle.builds import Catalog
from pcsa.common.config import SimConfig
from pcsa.common.errors import ConfigError
from pcsa.run.sim_command import draw_pools, paired_exports
from pcsa.sim.engine import Engine


def _engine() -> Engine | None:
    try:
        return Engine.locate(SimConfig().format)
    except ConfigError:
        return None


ENGINE = _engine()


def test_pairs_swap_teams() -> None:
    a, b = paired_exports(["x0", "x1"], ["y0", "y1"], 4, swap_teams=True)
    assert (a, b) == (["x0", "y0", "x1", "y1"], ["y0", "x0", "y1", "x1"])
    a, b = paired_exports(["x0", "x1"], ["y0", "y1"], 3, swap_teams=False)
    assert (a, b) == (["x0", "x0", "x1"], ["y0", "y0", "y1"])


@pytest.mark.skipif(ENGINE is None, reason="the simulator is not installed; run `pcsa setup --with sim`")
def test_draw_pools_share_one_validated_pool() -> None:
    assert ENGINE is not None
    a_pool, b_pool = draw_pools(Catalog.load(), ENGINE, random.Random(4), 3, False)
    assert len(a_pool) == len(b_pool) == 3
    assert all(export.count("Ability:") == 6 for export in a_pool + b_pool)
