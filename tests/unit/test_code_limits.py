"""Rules that keep the code readable, enforced on every source file.

- no file over MAX_FILE_LINES, no function over MAX_FUNCTION_LINES
- `Path(...).parents` only in common/paths.py (every other module asks paths.py)
- `time.*` only in common/clock.py (so tests control time)
- `print` only in cli.py and common/setup.py (their job is console output)
"""

from __future__ import annotations

import ast
from pathlib import Path

from pcsa.common import paths

MAX_FILE_LINES = 400
MAX_FUNCTION_LINES = 60
SRC = paths.repo_root() / "src" / "pcsa"
PRINT_ALLOWED = {"cli.py", "setup.py"}


def source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_file_and_function_lengths() -> None:
    problems = []
    for path in source_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_FILE_LINES:
            problems.append(f"{path.relative_to(SRC)}: {len(lines)} lines")
        tree = ast.parse("\n".join(lines))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                length = (node.end_lineno or node.lineno) - node.lineno + 1
                if length > MAX_FUNCTION_LINES:
                    problems.append(f"{path.relative_to(SRC)}:{node.lineno} {node.name} is {length} lines")
    assert problems == []


def test_only_paths_module_locates_the_repo() -> None:
    offenders = []
    for path in source_files():
        if path.name == "paths.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "parents":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == []


def test_only_clock_reads_time_and_only_console_modules_print() -> None:
    offenders = []
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            reads_time = (
                isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "time"
            )
            if reads_time and path.name != "clock.py":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno} time.{node.attr}")
            prints = isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
            if prints and path.name not in PRINT_ALLOWED:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno} print")
    assert offenders == []
