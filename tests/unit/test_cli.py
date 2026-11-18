"""The `pcsa` command line: every command parses, and the light commands import no phone, CG or torch code."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from pcsa import __version__, cli
from pcsa.common.errors import ConfigError
from tests.conftest import REPO

COMMANDS = [
    ["setup"],
    ["setup", "--with", "torch", "--check"],
    ["doctor"],
    ["run", "--battles", "2", "--max-total", "3", "--seed", "4", "--name", "x", "--resume", "r"],
    ["check"],
    ["nav", "--to", "lobby", "--seconds", "5"],
    ["observe", "--seconds", "5", "--interval", "0.5"],
    ["calibrate"],
    ["scout", "--interval", "2"],
    ["watch"],
    ["train", "sprites"],
]


@pytest.mark.parametrize("argv", COMMANDS, ids=lambda argv: " ".join(argv[:2]))
def test_every_command_parses(argv: list[str]) -> None:
    assert cli.build_parser().parse_args(argv).command == argv[0]


def test_config_commands_take_config_and_overrides() -> None:
    for command in ("run", "check", "nav", "observe", "calibrate", "scout"):
        args = cli.build_parser().parse_args([command, "--config", "a.yaml", "--set", "run.battles=3"])
        assert (args.config, args.set) == ("a.yaml", ["run.battles=3"])


def test_setup_rejects_removed_groups() -> None:
    for group in ("sim", "train"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["setup", "--with", group])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["sim", "play"])


def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.build_parser().parse_args(["--version"])
    assert stop.value.code == 0 and capsys.readouterr().out.strip() == f"pcsa {__version__}"


def test_watch_flags() -> None:
    args = cli.build_parser().parse_args(["watch"])
    assert (args.run, args.port, args.no_browser) == (None, cli.WATCH_PORT, False)
    args = cli.build_parser().parse_args(["watch", "--run", "output/runs/x", "--port", "9000", "--no-browser"])
    assert (args.run, args.port, args.no_browser) == ("output/runs/x", 9000, True)


def test_config_error_exits_2(monkeypatch, capsys) -> None:
    def refuse(args):
        raise ConfigError("bad setting")

    monkeypatch.setattr(cli, "dispatch", refuse)
    assert cli.main(["doctor"]) == 2
    assert "bad setting" in capsys.readouterr().err


def test_train_without_torch_names_the_extra(monkeypatch) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    args = cli.build_parser().parse_args(["train", "sprites"])
    with pytest.raises(ConfigError, match="torch extra"):
        cli.dispatch(args)


HEAVY = ("cv2", "champions_gym", "torch", "pcsa.phone", "pcsa.cg", "pcsa.network")

PROBE = """
import json, sys
from pcsa import cli
parser = cli.build_parser()
for argv in (["watch", "--no-browser"], ["setup", "--check"], ["doctor"]):
    parser.parse_args(argv)
from pcsa.common import setup  # what `pcsa setup` imports before its steps run
print(json.dumps(sorted(sys.modules)))
"""


def test_parsing_imports_no_phone_cg_or_torch_code() -> None:
    out = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=REPO, capture_output=True, text=True, timeout=60, check=True
    )
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    heavy = [m for m in loaded if any(m == h or m.startswith(h + ".") for h in HEAVY)]
    live = [m for m in loaded if m.startswith("pcsa.live.") and m != "pcsa.live.cli"]
    assert heavy == [] and live == []
