from pathlib import Path

from pcsa.common import paths


def test_repo_root_contains_the_package() -> None:
    assert (paths.repo_root() / "src" / "pcsa" / "cli.py").is_file()
    assert paths.config_dir() == paths.repo_root() / "config"
    assert paths.tools_lock().is_file()


def test_output_defaults_inside_the_repo_and_honours_the_override(tmp_path: Path) -> None:
    assert paths.output_dir(environ={}) == paths.repo_root() / "output"
    override = {paths.OUTPUT_ENV: str(tmp_path / "elsewhere")}
    assert paths.output_dir(override) == (tmp_path / "elsewhere").resolve()
    assert paths.runs_dir(override).name == "runs"


def test_ensure_output_tree_creates_every_subfolder(output_env: dict[str, str]) -> None:
    root = paths.ensure_output_tree(output_env)
    for name in paths.OUTPUT_SUBDIRS:
        assert (root / name).is_dir()
