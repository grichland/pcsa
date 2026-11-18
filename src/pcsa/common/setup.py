"""`pcsa setup` and `pcsa doctor`: install and check the pinned tools, Champions Gym, the output folder and the phone.

Each step has `check()` (doctor) and `install()` (setup, when check fails); the exit code is non-zero only when a
required step fails.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from pcsa.common import paths, tools
from pcsa.common.errors import ConfigError

SYNC = "uv sync --inexact --extra torch"  # inexact: keeps packages other projects on this venv installed
CG_PLATFORM = "win32"


@dataclass(frozen=True)
class Status:
    ok: bool
    detail: str
    fix: str | None = None


class Step:
    name = "step"
    group: str | None = None  # None: always required; "torch": only with --with torch

    def check(self, lock: dict[str, Any]) -> Status:
        raise NotImplementedError

    def install(self, lock: dict[str, Any]) -> None:
        raise NotImplementedError


class PythonStep(Step):
    name = "python"

    def check(self, lock: dict[str, Any]) -> Status:
        version = platform.python_version()
        ok = sys.version_info[:2] == (3, 12)
        return Status(ok, f"{version} at {sys.executable}", None if ok else "run through `uv run` (Python 3.12)")

    def install(self, lock: dict[str, Any]) -> None:
        raise ConfigError(f"the Python environment is managed by uv: run `{SYNC}`, then `pcsa setup`")


class AdbStep(Step):
    name = "adb"

    def check(self, lock: dict[str, Any]) -> Status:
        spec = lock["adb"]
        folder = paths.tool(spec["into"])
        missing = [f for f in spec["files"] if not (folder / f).is_file()]
        if missing:
            return Status(False, f"missing {', '.join(missing)}", "pcsa setup")
        return Status(True, f"{spec['version']} in {folder}")

    def install(self, lock: dict[str, Any]) -> None:
        spec = lock["adb"]
        archive = tools.download(
            spec["url"], paths.tools_dir() / "downloads" / "platform-tools.zip", spec["sha256"], _progress
        )
        tools.extract(archive, paths.tool(spec["into"]), spec["extract_prefix"], spec["files"])
        tools.record_install("adb", {"version": spec["version"], "sha256": spec["sha256"]})


class ScrcpyStep(Step):
    """The phone mirror window (scrcpy); the whole release unpacks so scrcpy finds its DLLs."""

    name = "scrcpy"

    def check(self, lock: dict[str, Any]) -> Status:
        spec = lock["scrcpy"]
        folder = paths.tool(spec["into"])
        missing = [f for f in spec["files"] if not (folder / f).is_file()]
        if missing:
            return Status(False, f"missing {', '.join(missing)}", "pcsa setup")
        return Status(True, f"v{spec['version']} in {folder}")

    def install(self, lock: dict[str, Any]) -> None:
        spec = lock["scrcpy"]
        archive = tools.download(
            spec["url"], paths.tools_dir() / "downloads" / "scrcpy.zip", spec["sha256"], _progress
        )
        tools.extract(archive, paths.tool(spec["into"]), spec["extract_prefix"])
        tools.record_install("scrcpy", {"version": spec["version"], "sha256": spec["sha256"]})


class OcrStep(Step):
    name = "ocr models"

    def check(self, lock: dict[str, Any]) -> Status:
        folder = paths.tool(lock["ocr"]["into"])
        bad = [f["name"] for f in lock["ocr"]["files"] if not tools.verify(folder / f["name"], f["sha256"])]
        if bad:
            return Status(False, f"missing or wrong checksum: {', '.join(bad)}", "pcsa setup")
        return Status(True, f"{lock['ocr']['version']} in {folder}")

    def install(self, lock: dict[str, Any]) -> None:
        folder = paths.tool(lock["ocr"]["into"])
        for spec in lock["ocr"]["files"]:
            if not tools.verify(folder / spec["name"], spec["sha256"]):
                tools.download(spec["url"], folder / spec["name"], spec["sha256"], _progress)
        tools.record_install("ocr", {"version": lock["ocr"]["version"]})


class OutputStep(Step):
    name = "output folder"

    def check(self, lock: dict[str, Any]) -> Status:
        root = paths.output_dir()
        if not root.is_dir():
            return Status(False, f"{root} does not exist", "pcsa setup")
        runs = sum(1 for p in (root / "runs").glob("*") if p.is_dir()) if (root / "runs").is_dir() else 0
        return Status(True, f"{root} ({runs} runs)")

    def install(self, lock: dict[str, Any]) -> None:
        paths.ensure_output_tree()


class DeviceStep(Step):
    name = "device"

    def check(self, lock: dict[str, Any]) -> Status:
        adb = paths.tool(lock["adb"]["into"]) / "adb.exe"
        if not adb.is_file():
            return Status(False, "adb not installed yet", "pcsa setup")
        try:
            out = subprocess.run([str(adb), "devices", "-l"], capture_output=True, text=True, timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            return Status(False, f"adb failed: {error}", "replug the phone and re-run")
        devices = [line.split() for line in out.stdout.splitlines()[1:] if line.strip()]
        ready = [d for d in devices if len(d) > 1 and d[1] == "device"]
        if not devices:
            return Status(False, "no device", "connect the phone over USB with USB debugging on")
        if not ready:
            return Status(False, f"device state {devices[0][1]}", "accept the USB debugging prompt on the phone")
        serial = ready[0][0]
        model = next((f.split(":", 1)[1] for f in ready[0] if f.startswith("model:")), "?")
        profile = paths.config_dir() / "devices" / f"{model}.yaml"
        if not profile.is_file():
            return Status(True, f"{serial} ({model}); no profile yet", "pcsa calibrate (with the game on the lobby)")
        return Status(True, f"{serial} ({model}); profile {profile.name}")

    def install(self, lock: dict[str, Any]) -> None:
        pass  # nothing to install; the check reports what the person has to do


class ChampionsGymStep(Step):
    """Champions Gym, the pinned wheel: the Battle, the legal choices and the neural player."""

    name = "champions gym"

    def check(self, lock: dict[str, Any]) -> Status:
        pinned = lock["champions_gym"]["version"]
        if sys.platform != CG_PLATFORM:
            return Status(False, f"no Champions Gym wheel for {sys.platform} (Windows x64 only)", None)
        try:
            cg = _import("champions_gym")
        except Exception as error:  # a broken wheel can raise OSError (a missing DLL) as well as ImportError
            return Status(False, f"not importable: {type(error).__name__}: {error}", SYNC)
        version = getattr(cg, "__version__", "?")
        if version != pinned:
            return Status(False, f"{version} installed, pinned {pinned}", SYNC)
        return Status(True, f"{version}; {_contracts()}")

    def install(self, lock: dict[str, Any]) -> None:
        raise ConfigError(f"Champions Gym is a project dependency: run `{SYNC}`")


class TorchStep(Step):
    name = "torch"
    group = "torch"

    def check(self, lock: dict[str, Any]) -> Status:
        try:
            torch = _import("torch")
        except ImportError:
            return Status(False, "not installed", SYNC)
        return Status(True, f"{torch.__version__}, cuda {torch.cuda.is_available()}")

    def install(self, lock: dict[str, Any]) -> None:
        _run([str(part) for part in SYNC.split()], cwd=paths.repo_root())


STEPS: tuple[Step, ...] = (
    PythonStep(),
    AdbStep(),
    ScrcpyStep(),
    OcrStep(),
    OutputStep(),
    DeviceStep(),
    ChampionsGymStep(),
    TorchStep(),
)


def run(groups: set[str], check_only: bool) -> int:
    """Run every required step plus the requested optional groups. Returns the exit code."""
    lock = tools.load_lock()
    selected = [s for s in STEPS if s.group is None or s.group in groups]
    problems = 0
    for number, step in enumerate(selected, 1):
        status = step.check(lock)
        if not status.ok and not check_only:
            try:
                print(f"[{number}/{len(selected)}] {step.name:<16} installing ...", flush=True)
                step.install(lock)
                status = step.check(lock)
            except (ConfigError, OSError, subprocess.CalledProcessError) as error:
                status = Status(False, str(error), status.fix)
        mark = "ok " if status.ok else "!! "
        print(f"[{number}/{len(selected)}] {step.name:<16} {mark} {status.detail}")
        if status.fix:
            print(f"{'':22} fix: {status.fix}")
        problems += not status.ok
    _print_footer(groups, problems)
    return 1 if problems else 0


def _print_footer(groups: set[str], problems: int) -> None:
    if problems:
        print(f"\n{problems} problem(s) to fix before using the phone")
        return
    ready = ["pcsa check", "pcsa run", "pcsa watch"]
    ready += ["pcsa train sprites"] if "torch" in groups else []
    print(f"\nready: {', '.join(ready)}")
    if "torch" not in groups:
        print("optional: pcsa setup --with torch   torch (the neural player's network, pcsa train sprites)")


def _contracts() -> str:
    """The LIVE and MODEL contract versions this CG build speaks, where the package exposes them."""
    found = []
    for label, module, attribute in (
        ("LIVE", "champions_gym._cg", "CONTRACT_VERSION"),
        ("MODEL", "champions_gym.neural", "MODEL_CONTRACT"),
    ):
        try:
            value = getattr(_import(module), attribute, None)
        except Exception:  # reported as unknown; the import check above already passed
            value = None
        found.append(f"{label} contract {value or '?'}")
    return ", ".join(found)


def _import(name: str) -> ModuleType:
    """Import by name: CG and torch stay out of pcsa.common's static imports (import-linter)."""
    return importlib.import_module(name)


def _progress(done: int, total: int | None) -> None:
    percent = f"{100 * done // total:3d}%" if total else "   "
    print(f"\r{'':22} downloading {tools.human_size(done):>9} {percent}", end="", flush=True)
    if total and done >= total:
        print()


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)
