"""The `pcsa` command line: every command parses, and the light commands import no phone, CG or torch code."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from pcsa import __version__, cli
from pcsa.common.errors import ConfigError
from tests.conftest import REPO

COMMANDS = [
    ["setup"],
    ["setup", "--with", "torch", "--check"],
    ["doctor"],
    ["run", "--resume", "r"],
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


CONFIG = object()  # what the patched cli._load returns


def _calls(monkeypatch) -> list[tuple]:
    """Patch config loading; returns the list the fakes append their (name, *arguments) to."""
    calls: list[tuple] = []

    def load(args):
        calls.append(("load", args.config, tuple(args.set)))
        return CONFIG

    monkeypatch.setattr(cli, "_load", load)
    return calls


def _fake(calls: list[tuple], name: str, result: object = 0):
    def fake(*args, **kwargs):
        calls.append((name, *args, *sorted(kwargs.items())))
        return result

    return fake


def test_dispatch_live_commands(monkeypatch, tmp_path) -> None:
    from pcsa.live import phone, play

    calls = _calls(monkeypatch)
    monkeypatch.setattr(play, "play_command", _fake(calls, "play"))
    monkeypatch.setattr(play, "check_command", _fake(calls, "check"))
    monkeypatch.setattr(phone, "nav_command", _fake(calls, "nav"))
    argv = ["run", "--config", "a.yaml", "--set", "run.battles=3", "--resume", str(tmp_path)]
    assert cli.main(argv) == 0
    assert cli.main(["check"]) == 0
    assert cli.main(["nav", "--to", "lobby", "--seconds", "5"]) == 0
    expected_run = ("play", CONFIG, tmp_path)
    assert calls == [
        ("load", "a.yaml", ("run.battles=3",)),
        expected_run,
        ("load", None, ()),
        ("check", CONFIG),
        ("load", None, ()),
        ("nav", CONFIG, "lobby", 5.0),
    ]


def test_dispatch_phone_commands(monkeypatch) -> None:
    from pcsa.phone import commands, scout

    calls = _calls(monkeypatch)
    monkeypatch.setattr(commands, "observe_command", _fake(calls, "observe"))
    monkeypatch.setattr(commands, "calibrate_command", _fake(calls, "calibrate"))
    monkeypatch.setattr(scout, "scout_command", _fake(calls, "scout"))
    assert cli.main(["observe", "--seconds", "5", "--interval", "0.5"]) == 0
    assert cli.main(["calibrate"]) == 0
    assert cli.main(["scout", "--interval", "2"]) == 0
    loaded = ("load", None, ())
    assert calls == [
        loaded,
        ("observe", CONFIG, 5.0, 0.5),
        loaded,
        ("calibrate", CONFIG),
        loaded,
        ("scout", CONFIG, 2.0),
    ]


def test_dispatch_setup_and_doctor(monkeypatch) -> None:
    from pcsa.common import setup

    calls = _calls(monkeypatch)
    monkeypatch.setattr(setup, "run", _fake(calls, "setup"))
    assert cli.main(["setup"]) == 0
    assert cli.main(["setup", "--with", "torch", "--check"]) == 0
    assert cli.main(["doctor"]) == 0
    assert calls == [
        ("setup", set(), ("check_only", False)),
        ("setup", {"torch"}, ("check_only", True)),
        ("setup", set(), ("check_only", True)),
    ]


def test_dispatch_watch(monkeypatch, tmp_path, capsys) -> None:
    from pcsa.run import watch

    calls: list[tuple] = []
    monkeypatch.setattr(watch, "latest_run", lambda: tmp_path)
    monkeypatch.setattr(watch, "serve", _fake(calls, "serve"))
    assert cli.main(["watch"]) == 0
    assert cli.main(["watch", "--run", "elsewhere", "--port", "9000", "--no-browser"]) == 0
    assert calls == [
        ("serve", tmp_path, "127.0.0.1", cli.WATCH_PORT, True),
        ("serve", Path("elsewhere"), "127.0.0.1", 9000, False),
    ]
    monkeypatch.setattr(watch, "latest_run", lambda: None)
    assert cli.main(["watch"]) == 1
    assert "no run found" in capsys.readouterr().err


def test_dispatch_train_sprites(monkeypatch) -> None:
    import pcsa.sprite_training as package

    calls: list[tuple] = []
    fake = types.SimpleNamespace(run=_fake(calls, "train"))
    monkeypatch.setitem(sys.modules, "pcsa.sprite_training.sprite_train", fake)
    monkeypatch.setattr(package, "sprite_train", fake, raising=False)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    assert cli.main(["train", "sprites", "--steps", "5"]) == 0
    assert len(calls) == 1 and calls[0][0] == "train" and calls[0][1].steps == 5
