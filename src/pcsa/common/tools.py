"""Pinned external tools: the lock file, verified downloads, extraction and the install manifest.

Used by `pcsa setup` and `pcsa doctor`. Standard library only, so it works before dependencies do.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pcsa.common import paths
from pcsa.common.errors import ConfigError

CHUNK = 1 << 20
KIB = 1024
MANIFEST = "MANIFEST.json"


def load_lock(path: Path | None = None) -> dict[str, Any]:
    path = path or paths.tools_lock()
    if not path.is_file():
        raise ConfigError(f"tools lock file not found: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != "pcsa-tools-lock/1":
        raise ConfigError(f"unexpected tools lock schema in {path}")
    return lock


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, sha256: str) -> bool:
    return path.is_file() and sha256_of(path) == sha256


def download(url: str, target: Path, sha256: str, progress: Callable[[int, int | None], None] | None = None) -> Path:
    """Fetch `url` to `target`, verify the checksum; a mismatch deletes the file and raises."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "pcsa-setup"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0) or None
        done = 0
        for chunk in iter(lambda: response.read(CHUNK), b""):
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    actual = sha256_of(partial)
    if actual != sha256:
        partial.unlink(missing_ok=True)
        raise ConfigError(f"checksum mismatch for {url}: expected {sha256[:12]}..., got {actual[:12]}...")
    partial.replace(target)
    return target


def extract(archive: Path, into: Path, prefix: str = "", names: list[str] | None = None) -> list[Path]:
    """Extract members under `prefix` (dropping it) into `into`; `names` limits which members."""
    into.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            relative = member.filename[len(prefix) :]
            if names is not None and relative not in names:
                continue
            target = into / relative
            if not target.resolve().is_relative_to(into.resolve()):
                raise ConfigError(f"archive member escapes the target directory: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            written.append(target)
    return written


def read_manifest() -> dict[str, Any]:
    path = paths.tools_dir() / MANIFEST
    if not path.is_file():
        return {"schema": "pcsa-tools-manifest/1", "installed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def record_install(name: str, details: dict[str, Any]) -> None:
    manifest = read_manifest()
    manifest["installed"][name] = {**details, "at": datetime.now(UTC).isoformat(timespec="seconds")}
    path = paths.tools_dir() / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def human_size(size: int | None) -> str:
    if size is None:
        return "?"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < KIB or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= KIB
    return f"{value:.1f} GB"
