import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from pcsa.common import paths, setup, tools
from pcsa.common.errors import ConfigError


def test_lock_file_pins_every_tool() -> None:
    lock = tools.load_lock()
    assert set(lock) >= {"adb", "scrcpy", "ocr", "node", "showdown"}
    for key in ("adb", "scrcpy", "node"):
        assert len(lock[key]["sha256"]) == 64 and lock[key]["url"].startswith("https://")
    assert lock["scrcpy"]["url"].endswith(f"scrcpy-win64-v{lock['scrcpy']['version']}.zip")
    assert lock["scrcpy"]["extract_prefix"] == f"scrcpy-win64-v{lock['scrcpy']['version']}/"
    assert "scrcpy.exe" in lock["scrcpy"]["files"]
    assert all(len(f["sha256"]) == 64 for f in lock["ocr"]["files"])
    assert len(lock["showdown"]["commit"]) == 40


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
            data, self._rest = payload, b""
            payload_left = getattr(self, "_done", False)
            self._done = True
            return b"" if payload_left else data

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
    monkeypatch.setattr(paths, "_REPO", tmp_path)
    lock = tools.load_lock(Path(__file__).parents[2] / "tools.lock.json")
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


def test_human_size() -> None:
    assert tools.human_size(None) == "?"
    assert tools.human_size(512) == "512 B"
    assert tools.human_size(8044989) == "7.7 MB"
