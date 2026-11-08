"""`pcsa setup` and `pcsa doctor`: install and check the pinned tools, the output folder and the phone.

Each step has `check()` (what doctor runs) and `install()` (what setup runs when check fails).
Steps print one line each; the exit code is non-zero only when a required step is not satisfied.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcsa.common import paths, tools
from pcsa.common.errors import ConfigError

OPTIONAL_GROUPS = ("sim", "train")


@dataclass(frozen=True)
class Status:
    ok: bool
    detail: str
    fix: str | None = None


class Step:
    name = "step"
    group: str | None = None  # None: always required; "sim" / "train": only with --with

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
        raise ConfigError("the Python environment is managed by uv: run `uv sync` and then `uv run pcsa setup`")


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


class NodeStep(Step):
    name = "node"
    group = "sim"

    def check(self, lock: dict[str, Any]) -> Status:
        exe = paths.tool(lock["node"]["into"]) / "node.exe"
        if not exe.is_file():
            return Status(False, "not installed", "pcsa setup --with sim")
        return Status(True, f"v{lock['node']['version']} in {exe.parent}")

    def install(self, lock: dict[str, Any]) -> None:
        spec = lock["node"]
        archive = tools.download(spec["url"], paths.tools_dir() / "downloads" / "node.zip", spec["sha256"], _progress)
        tools.extract(archive, paths.tool(spec["into"]), spec["extract_prefix"])
        tools.record_install("node", {"version": spec["version"], "sha256": spec["sha256"]})


class ShowdownStep(Step):
    name = "showdown"
    group = "sim"

    def check(self, lock: dict[str, Any]) -> Status:
        spec = lock["showdown"]
        repo = paths.tool(spec["into"])
        marker = repo / ".pcsa-built"
        if not (repo / "dist" / "sim").is_dir() or not marker.is_file():
            return Status(False, "not built", "pcsa setup --with sim")
        built = marker.read_text(encoding="utf-8").strip()
        if built != spec["commit"]:
            return Status(False, f"built at {built[:10]}, pinned {spec['commit'][:10]}", "pcsa setup --with sim")
        return Status(True, f"{spec['commit'][:10]} built in {repo}")

    def install(self, lock: dict[str, Any]) -> None:
        spec = lock["showdown"]
        repo = paths.tool(spec["into"])
        node_dir = paths.tool(lock["node"]["into"])
        if not (repo / ".git").is_dir():
            _run(["git", "clone", "--filter=blob:none", "--no-checkout", spec["repository"], str(repo)])
        _run(["git", "-C", str(repo), "fetch", "--depth=1", "origin", spec["commit"]])
        _run(["git", "-C", str(repo), "checkout", "--detach", spec["commit"]])
        env = {**os.environ, "PATH": f"{node_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
        _run([str(node_dir / "npm.cmd"), "ci", "--omit=dev", "--ignore-scripts"], cwd=repo, env=env)
        _run([str(node_dir / "node.exe"), "build"], cwd=repo, env=env)
        (repo / ".pcsa-built").write_text(spec["commit"] + "\n", encoding="utf-8")
        tools.record_install("showdown", {"commit": spec["commit"]})


class TorchStep(Step):
    name = "torch"
    group = "train"

    def check(self, lock: dict[str, Any]) -> Status:
        try:
            import torch  # pyright: ignore[reportMissingImports]  (optional extra)
        except ImportError:
            return Status(False, "not installed", "uv sync --extra train")
        return Status(True, f"{torch.__version__}, cuda {torch.cuda.is_available()}")

    def install(self, lock: dict[str, Any]) -> None:
        _run(["uv", "sync", "--extra", "train"], cwd=paths.repo_root())


STEPS: tuple[Step, ...] = (
    PythonStep(),
    AdbStep(),
    ScrcpyStep(),
    OcrStep(),
    OutputStep(),
    DeviceStep(),
    NodeStep(),
    ShowdownStep(),
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
        print(f"\n{problems} problem(s) to fix before using the phone or the simulator")
        return
    print("\nready: pcsa observe, pcsa sim play")
    for group in OPTIONAL_GROUPS:
        if group not in groups:
            what = "Node + Showdown (for pcsa sim play)" if group == "sim" else "torch (for pcsa train)"
            print(f"optional: pcsa setup --with {group:<6} {what}")


def _progress(done: int, total: int | None) -> None:
    percent = f"{100 * done // total:3d}%" if total else "   "
    print(f"\r{'':22} downloading {tools.human_size(done):>9} {percent}", end="", flush=True)
    if total and done >= total:
        print()


def _run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)
