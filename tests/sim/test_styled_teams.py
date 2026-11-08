"""Style players' simulator teams: varied, legal sixes that all fit the style; and the pairing that keeps
them with their player."""

from __future__ import annotations

import random

import pytest
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.builds import Catalog
from pcsa.common.config import SimConfig
from pcsa.common.errors import ConfigError
from pcsa.players import styles
from pcsa.run.sim_command import draw_pools, paired_exports, style_of, team_styles
from pcsa.sim.engine import Engine
from pcsa.sim.teams import role_species, sample_styled_team, team_export, validated_teams
from pcsa.sim.validate import validate

STYLES = ("weather", "setup_sweeper", "hazard_lead")


def _engine() -> Engine | None:
    try:
        return Engine.locate(SimConfig().format)
    except ConfigError:
        return None


ENGINE = _engine()


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.load()


@pytest.fixture(scope="module")
def rules() -> styles.Styles:
    return styles.load()


@pytest.mark.parametrize("style", STYLES)
def test_styled_draws_fit_vary_and_respect_the_clauses(style: str, catalog: Catalog, rules: styles.Styles) -> None:
    rng = random.Random(5)
    species_sets = set()
    for i in range(30):
        six = sample_styled_team(catalog, rng, style, rules, noisy=i % 2 == 1)
        assert len(six) == 6
        assert rules.fit(style, six) is not None
        items = [to_id_str(m.item) for m in six if m.item]
        assert len(items) == len(set(items))
        assert len({game.base_name(to_id_str(m.species)) for m in six}) == 6
        species_sets.add(frozenset(m.species for m in six))
    assert len(species_sets) >= 10


def test_weather_draws_cover_every_weather(catalog: Catalog, rules: styles.Styles) -> None:
    rng = random.Random(9)
    seen = set()
    for _ in range(40):
        fit = rules.fit("weather", sample_styled_team(catalog, rng, "weather", rules))
        assert fit is not None
        seen.add(fit.weather)
    assert seen == {"sun", "rain", "sand", "snow"}


def test_role_species_lists_the_catalog_fillers(catalog: Catalog, rules: styles.Styles) -> None:
    roles = role_species(catalog, rules, "weather")
    assert "Pelipper" in roles["rain setter"] and "Torkoal" in roles["sun setter"]
    assert "Charizard" in roles["sun setter"]  # through Charizardite Y
    assert "Torkoal" in role_species(catalog, rules, "hazard_lead")["setter"]


def test_style_kinds() -> None:
    assert style_of("style_setup_sweeper") == "setup_sweeper"
    assert style_of("custom_heuristic") is None


def test_ordinary_pairs_swap_teams_and_style_pairs_keep_them() -> None:
    a, b = paired_exports(["x0", "x1"], ["y0", "y1"], 4, swap_teams=True)
    assert (a, b) == (["x0", "y0", "x1", "y1"], ["y0", "x0", "y1", "x1"])
    a, b = paired_exports(["x0", "x1"], ["y0", "y1"], 3, swap_teams=False)
    assert (a, b) == (["x0", "x0", "x1"], ["y0", "y0", "y1"])


@pytest.mark.skipif(ENGINE is None, reason="the simulator is not installed; run `pcsa setup --with sim`")
@pytest.mark.parametrize("style", STYLES)
def test_styled_teams_validate(style: str, catalog: Catalog, rules: styles.Styles) -> None:
    assert ENGINE is not None
    rng = random.Random(3)
    teams = validated_teams(ENGINE, 2, lambda: sample_styled_team(catalog, rng, style, rules))
    assert len(teams) == 2 and all(r["ok"] for r in validate(ENGINE, teams))
    assert team_export(sample_styled_team(catalog, rng, style, rules)).count("Ability:") == 6


@pytest.mark.skipif(ENGINE is None, reason="the simulator is not installed; run `pcsa setup --with sim`")
def test_a_team_style_fits_a_and_leaves_b_ordinary_without_swapping(catalog: Catalog, rules: styles.Styles) -> None:
    assert ENGINE is not None
    sides = team_styles(("best_matchup_max_damage", "best_matchup_max_damage"), ("weather", None))
    assert sides == ("weather", None)
    a_pool, b_pool, swap = draw_pools(catalog, ENGINE, random.Random(4), sides, 3, False)
    assert swap is False and len(a_pool) == len(b_pool) == 3
    assert all(export.count("Ability:") == 6 for export in a_pool + b_pool)
    assert paired_exports(a_pool, b_pool, 6, swap_teams=swap)[0] == [a_pool[k // 2] for k in range(6)]


def test_team_style_override_and_unknown_style() -> None:
    assert team_styles(("style_weather", "x"), (None, None)) == ("weather", None)
    assert team_styles(("style_weather", "x"), ("hazard_lead", None)) == ("hazard_lead", None)
    with pytest.raises(ConfigError):
        draw_pools(Catalog.load(), ENGINE, random.Random(1), ("nonsense", None), 1, False)  # type: ignore[arg-type]
