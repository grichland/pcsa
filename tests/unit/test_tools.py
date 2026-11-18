import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from pcsa.common import paths, setup, tools
from pcsa.common.errors import ConfigError


def test_lock_file_pins_every_tool() -> None:
    lock = tools.load_lock()
    assert set(lock) >= {"adb", "scrcpy", "ocr", "champions_gym", "showdown"}
    assert "node" not in lock
    for key in ("adb", "scrcpy", "champions_gym"):
        assert len(lock[key]["sha256"]) == 64 and lock[key]["url"].startswith("https://")
    assert lock["scrcpy"]["url"].endswith(f"scrcpy-win64-v{lock['scrcpy']['version']}.zip")
    assert lock["scrcpy"]["extract_prefix"] == f"scrcpy-win64-v{lock['scrcpy']['version']}/"
    assert "scrcpy.exe" in lock["scrcpy"]["files"]
    assert all(len(f["sha256"]) == 64 for f in lock["ocr"]["files"])
    cg = lock["champions_gym"]
    assert cg["url"].endswith(f"champions_gym-{cg['version']}-cp312-cp312-win_amd64.whl")


def test_cg_pin_matches_pyproject() -> None:
    pyproject = (paths.repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    cg = tools.load_lock()["champions_gym"]
    assert f'"champions-gym=={cg["version"]}' in pyproject
    assert cg["url"] in pyproject


def test_verify_and_extract(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("platform-tools/adb.exe", b"adb")
        zf.writestr("platform-tools/extra.txt", b"x")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert tools.verify(archive, digest)
    assert not tools.verify(archive, "0" * 64)
    written = tools.extract(archive, tmp_path / "adb", "platform-tools/", ["adb.exe"])
    assert [p.name for p in written] == ["adb.exe"]
    assert (tmp_path / "adb" / "adb.exe").read_bytes() == b"adb"


def test_extract_refuses_escaping_members(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("p/../../evil.txt", b"x")
    with pytest.raises(ConfigError):
        tools.extract(archive, tmp_path / "out", "p/")


def test_download_rejects_a_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    payload = b"hello"

    class FakeResponse:
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n):
            sent = getattr(self, "_done", False)
            self._done = True
            return b"" if sent else payload

    monkeypatch.setattr(tools.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(ConfigError, match="checksum mismatch"):
        tools.download("https://example.invalid/x", tmp_path / "x.bin", "0" * 64)
    assert not (tmp_path / "x.bin").exists() and not (tmp_path / "x.bin.part").exists()
    good = hashlib.sha256(payload).hexdigest()
    assert tools.download("https://example.invalid/x", tmp_path / "x.bin", good).read_bytes() == payload


def test_manifest_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths, "_REPO", tmp_path)
    tools.record_install("adb", {"version": "r1"})
    manifest = json.loads((tmp_path / ".tools" / tools.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["installed"]["adb"]["version"] == "r1"
    assert tools.read_manifest()["installed"]["adb"]["at"]


def test_scrcpy_step_installs_the_flattened_release(monkeypatch, tmp_path: Path) -> None:
    lock = tools.load_lock()
    monkeypatch.setattr(paths, "_REPO", tmp_path)
    spec = lock["scrcpy"]
    fetched: list[str] = []

    def fake_download(url: str, target: Path, sha256: str, progress=None) -> Path:
        fetched.append(url)
        assert sha256 == spec["sha256"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w") as zf:
            for name in ("scrcpy.exe", "scrcpy-server", "SDL2.dll", "adb.exe"):
                zf.writestr(spec["extract_prefix"] + name, name.encode())
        return target

    monkeypatch.setattr(tools, "download", fake_download)
    step = setup.ScrcpyStep()
    assert not step.check(lock).ok
    step.install(lock)
    folder = tmp_path / ".tools" / "scrcpy"
    assert (folder / "scrcpy.exe").read_bytes() == b"scrcpy.exe"
    assert (folder / "scrcpy-server").is_file() and (folder / "SDL2.dll").is_file()
    assert not (folder / spec["extract_prefix"]).exists()
    status = step.check(lock)
    assert status.ok and spec["version"] in status.detail
    assert tools.read_manifest()["installed"]["scrcpy"]["version"] == spec["version"]
    assert fetched == [spec["url"]]


def _fake_cg(monkeypatch, version: str) -> None:
    modules = {
        "champions_gym": SimpleNamespace(__version__=version),
        "champions_gym._cg": SimpleNamespace(CONTRACT_VERSION="0.9"),
        "champions_gym.neural": SimpleNamespace(MODEL_CONTRACT="0.6"),
    }
    monkeypatch.setattr(setup, "_import", lambda name: modules[name])
    monkeypatch.setattr(sys, "platform", setup.CG_PLATFORM)


def test_cg_step_reports_version_and_contracts(monkeypatch) -> None:
    lock = tools.load_lock()
    _fake_cg(monkeypatch, lock["champions_gym"]["version"])
    status = setup.ChampionsGymStep().check(lock)
    assert status.ok
    assert lock["champions_gym"]["version"] in status.detail
    assert "LIVE contract 0.9" in status.detail and "MODEL contract 0.6" in status.detail


def test_cg_step_rejects_another_version(monkeypatch) -> None:
    lock = tools.load_lock()
    _fake_cg(monkeypatch, "0.0.1")
    status = setup.ChampionsGymStep().check(lock)
    assert not status.ok and "pinned" in status.detail and status.fix == setup.SYNC


def test_sync_hint_keeps_other_projects_packages() -> None:
    assert setup.SYNC == "uv sync --inexact --extra torch"
    assert setup.TorchStep.group == "torch"


def test_torch_step_installs_with_an_inexact_sync(monkeypatch) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(setup, "_run", lambda command, cwd=None: ran.append(command))
    setup.TorchStep().install({})
    assert ran == [["uv", "sync", "--inexact", "--extra", "torch"]]


def test_cg_step_reports_a_broken_wheel(monkeypatch) -> None:
    lock = tools.load_lock()

    def broken(name: str):
        raise OSError("DLL load failed while importing _cg")

    monkeypatch.setattr(setup, "_import", broken)
    monkeypatch.setattr(sys, "platform", setup.CG_PLATFORM)
    status = setup.ChampionsGymStep().check(lock)
    assert not status.ok and "OSError: DLL load failed" in status.detail and status.fix == setup.SYNC


def test_cg_step_when_missing_or_off_windows(monkeypatch) -> None:
    lock = tools.load_lock()

    def missing(name: str):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(setup, "_import", missing)
    monkeypatch.setattr(sys, "platform", setup.CG_PLATFORM)
    assert not setup.ChampionsGymStep().check(lock).ok
    monkeypatch.setattr(sys, "platform", "linux")
    status = setup.ChampionsGymStep().check(lock)
    assert not status.ok and "no Champions Gym wheel" in status.detail


def test_check_only_installs_nothing_and_fails_on_a_required_step(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class Fake(setup.Step):
        def __init__(self, name: str, ok: bool, group: str | None = None) -> None:
            self.name, self.ok, self.group = name, ok, group

        def check(self, lock):
            calls.append(f"check {self.name}")
            return setup.Status(self.ok, self.name)

        def install(self, lock):
            calls.append(f"install {self.name}")

    steps = (Fake("good", True), Fake("bad", False), Fake("extra", False, "torch"))
    monkeypatch.setattr(setup, "STEPS", steps)
    assert setup.run(set(), check_only=True) == 1
    assert calls == ["check good", "check bad"]
    assert "1 problem(s)" in capsys.readouterr().out
    monkeypatch.setattr(setup, "STEPS", steps[:1])
    assert setup.run(set(), check_only=True) == 0


def test_human_size() -> None:
    assert tools.human_size(None) == "?"
    assert tools.human_size(512) == "512 B"
    assert tools.human_size(8044989) == "7.7 MB"
