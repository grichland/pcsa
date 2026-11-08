from pathlib import Path

import pytest
import yaml

from pcsa.common.config import Config, Viewport, load_config, overrides_from_environ
from pcsa.common.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]


def test_repo_config_loads(config: Config) -> None:
    assert config.agent.timing.max_turn_s == 40.0
    assert config.agent.policy.kind == "custom_heuristic"


def test_every_screen_timeout_names_a_real_screen(config: Config) -> None:
    from pcsa.common.types import Screen

    unknown = set(config.agent.timing.screen_timeouts) - {str(s) for s in Screen}
    assert unknown == set()


def test_cli_override_and_env_override(tmp_path: Path) -> None:
    environ = {"PCSA__timing__max_turn_s": "30", "PCSA__run__battles": "3", "HOME": "x"}
    assert overrides_from_environ(environ) == ["run.battles=3", "timing.max_turn_s=30"]
    config = load_config(REPO / "config/agent.yaml", overrides=["timing.max_turn_s=25.5"], environ=environ)
    assert config.agent.timing.max_turn_s == 25.5  # explicit override wins over the environment
    assert config.agent.run.battles == 3


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    data = yaml.safe_load((REPO / "config/agent.yaml").read_text(encoding="utf-8"))
    data["timing"]["max_turn"] = 12
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    (tmp_path / "screen.yaml").write_text((REPO / "config/screen.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ConfigError, match="timing.max_turn"):
        load_config(tmp_path / "agent.yaml", environ={})


def test_bad_override_shape() -> None:
    with pytest.raises(ConfigError, match="section.key=value"):
        load_config(REPO / "config/agent.yaml", overrides=["timing"], environ={})


def test_regions_resolve_to_the_measured_pixels(config: Config) -> None:
    full = Viewport.of_frame(2340, 1080)
    fight = config.screen.resolve("fight", full)
    assert (fight.x1, fight.y1, fight.x2, fight.y2) == (2045, 645, 2145, 735)
    row = config.screen.resolve("move_rows", full, index=2)
    assert (row.x1, row.y1) == (1800, 800)


def test_regions_scale_and_offset_with_the_viewport(config: Config) -> None:
    letterboxed = Viewport(x=160, y=0, w=1600, h=738)  # a 16:9 phone drawing the game in a centred band
    fight = config.screen.resolve("fight", letterboxed)
    assert 160 < fight.x1 < fight.x2 <= 1760
    assert fight.center[0] == pytest.approx(160 + 0.8952 * 1600, abs=2)


def test_relative_region_needs_an_origin(config: Config) -> None:
    full = Viewport.of_frame(2340, 1080)
    with pytest.raises(ValueError):
        config.screen.resolve("switch_in", full)
    rect = config.screen.resolve_relative("switch_in", full, origin=(0.2, 0.3))
    assert rect.x1 == round((0.2 - 0.0385) * 2340)
