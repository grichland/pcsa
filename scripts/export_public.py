"""Export the public repository: every tracked and new (not ignored) file except the private live-play layer.

    uv run python scripts/export_public.py --to ..\\pcsa [--check]

The target is a folder that does not exist yet, an empty folder, or a git working copy (its `.git` is kept and
everything else is replaced). Private paths are left out; `.importlinter` loses the `live` layer and its
contract, and `pyproject.toml` the private files' lint exemptions. The export fails if any exported Python
file still imports `pcsa.live`. With `--check`, `uv sync` and the full `poe check` run inside the export.
Private (not exported itself). See docs/live.md.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = (
    "src/pcsa/live/",
    "tests/live/",
    "scripts/run_agent.ps1",
    "scripts/export_public.py",
    "docs/live.md",
)
LIVE_IMPORT = re.compile(r"^\s*(from\s+pcsa\.live\b|import\s+pcsa\.live\b)", re.MULTILINE)


def source_files() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode("utf-8")
    files = sorted({f for f in listing.split("\0") if f and (ROOT / f).is_file()})
    return [f for f in files if not f.startswith(PRIVATE)]


def public_importlinter(text: str) -> str:
    sections = re.split(r"(?m)^(?=\[)", text)
    kept = [s for s in sections if not s.startswith("[importlinter:contract:live-is-private]")]
    out = "".join(kept)
    out = re.sub(r"(?m)^    pcsa\.live\n", "", out)
    return out.replace(" <- live <- cli", " <- cli")


def public_pyproject(text: str) -> str:
    return re.sub(r'(?m)^"src/pcsa/live/[^\n]*\n', "", text)


TRANSFORMS = {".importlinter": public_importlinter, "pyproject.toml": public_pyproject}


def prepare_target(target: Path) -> None:
    if target == ROOT or ROOT in target.parents:
        sys.exit(f"refusing to export into the private repository itself: {target}")
    if target.exists():
        entries = [p for p in target.iterdir() if p.name != ".git"]
        if entries and not (target / ".git").exists():
            sys.exit(f"{target} is not empty and not a git working copy; refusing to replace its contents")
        for entry in entries:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    target.mkdir(parents=True, exist_ok=True)


def export(target: Path) -> list[str]:
    files = source_files()
    prepare_target(target)
    for name in files:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in TRANSFORMS:
            text = (ROOT / name).read_text(encoding="utf-8")
            destination.write_text(TRANSFORMS[name](text), encoding="utf-8", newline="\n")
        else:
            shutil.copy2(ROOT / name, destination)
    leaks = [
        str(p.relative_to(target))
        for p in target.rglob("*.py")
        if ".git" not in p.parts and LIVE_IMPORT.search(p.read_text(encoding="utf-8"))
    ]
    if leaks:
        sys.exit(f"exported files still import pcsa.live: {', '.join(leaks)}")
    return files


def check(target: Path) -> int:
    for command in (["uv", "sync", "--group", "dev", "--extra", "train"], ["uv", "run", "poe", "check"]):
        print("$", " ".join(command), flush=True)
        if subprocess.run(command, cwd=target).returncode:
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--to", required=True, help="the public working copy (created if missing)")
    parser.add_argument("--check", action="store_true", help="run uv sync and poe check inside the export")
    args = parser.parse_args()
    target = Path(args.to).resolve()
    files = export(target)
    print(f"exported {len(files)} files to {target} (private paths left out: {', '.join(PRIVATE)})")
    return check(target) if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
