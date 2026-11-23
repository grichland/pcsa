"""config/agent.yaml is the only place a setting's value is written: the settings classes carry no defaults, and a
key missing from the file is refused."""

from __future__ import annotations

from typing import get_args

import pytest
import yaml
from pydantic import BaseModel

from pcsa.common.config import AgentConfig, load_config
from pcsa.common.errors import ConfigError
from tests import configs


def _models(model: type[BaseModel], seen: set[type[BaseModel]]) -> set[type[BaseModel]]:
    seen.add(model)
    for field in model.model_fields.values():
        for kind in (field.annotation, *get_args(field.annotation)):
            if isinstance(kind, type) and issubclass(kind, BaseModel) and kind not in seen:
                _models(kind, seen)
    return seen


def test_no_settings_class_has_a_default_value() -> None:
    defaults = [f"{model.__name__}.{name}" for model in _models(AgentConfig, set())
                for name, field in model.model_fields.items() if not field.is_required()]  # fmt: skip
    assert defaults == []


def test_a_key_missing_from_the_file_is_refused(tmp_path) -> None:
    data = yaml.safe_load(configs.SHIPPED.read_text(encoding="utf-8"))
    del data["run"]["battles"]
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    (tmp_path / "screen.yaml").write_text((configs.SHIPPED.parent / "screen.yaml").read_text(encoding="utf-8"),
                                          encoding="utf-8")  # fmt: skip
    with pytest.raises(ConfigError, match="run.battles"):
        load_config(tmp_path / "agent.yaml", environ={})


def test_run_counts_are_consistent() -> None:
    with pytest.raises(ConfigError, match="max_battles_total"):
        configs.shipped("run.battles=5", "run.max_battles_total=4")
    with pytest.raises(ConfigError, match="run.name"):
        configs.shipped("run.name='a b'")
