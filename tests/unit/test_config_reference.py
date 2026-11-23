"""docs/configuration.md is generated from config/agent.yaml (tools/config_reference.py) and must be current."""

from __future__ import annotations

import importlib.util

from pcsa.common import paths

ROOT = paths.repo_root()


def test_the_configuration_page_is_current() -> None:
    spec = importlib.util.spec_from_file_location("config_reference", ROOT / "tools" / "config_reference.py")
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    page = tool.render((ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
    current = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    assert current == page, "run: uv run python tools/config_reference.py"
