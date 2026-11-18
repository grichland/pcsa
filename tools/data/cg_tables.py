"""Read CG's generated Rust tables (read-only): the legal id sets and the values used for cross-checks.

The tables come from a Champions Gym checkout: the environment variable CG_SOURCE (or a tool's --source), else
../cg-publish next to this repository (the public clone). The file is engine/src/sim/data/generated.rs in it.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

DEFAULT_CHECKOUT = Path(__file__).resolve().parents[3] / "cg-publish"
GENERATED = Path("engine") / "src" / "sim" / "data" / "generated.rs"


def generated_rs(checkout: Path | str | None = None) -> Path:
    """generated.rs of the checkout given, else of CG_SOURCE, else of ../cg-publish."""
    return Path(checkout or os.environ.get("CG_SOURCE") or DEFAULT_CHECKOUT) / GENERATED


TABLES = {"species": "SPECIES", "moves": "MOVES", "items": "ITEMS", "abilities": "ABILITIES", "natures": "NATURES"}
TYPE_ORDER = (
    "bug dark dragon electric fairy fighting fire flying ghost grass ground ice normal poison psychic rock steel "
    "water"
).split()


def say(*parts: object) -> None:
    sys.stdout.write(" ".join(str(p) for p in parts) + "\n")


def source(path: Path | None = None) -> str:
    return (path or generated_rs()).read_text(encoding="utf-8")


def _block(src: str, name: str) -> tuple[int, str]:
    m = re.search(rf"pub static {name}: \[\w+; (\d+)\] = \[\n(.*?)\n\];", src, re.S)
    if m is None:
        raise SystemExit(f"table {name} not found in generated.rs")
    return int(m.group(1)), m.group(2)


def rows(src: str, name: str) -> tuple[int, list[str]]:
    """(declared length, one text line per entry); the empty id-0 placeholder is dropped."""
    declared, body = _block(src, name)
    lines = [line.strip() for line in body.splitlines() if re.match(r'\s*\w+Data \{ id: "', line)]
    return declared, [line for line in lines if field(line, "id")]


def field(line: str, key: str) -> str:
    m = re.search(rf"\b{key}: ((?:\"[^\"]*\")|[^,}}]+(?:\([^)]*\))?)", line)
    return m.group(1).strip().strip('"') if m else ""


def legal(src: str) -> dict[str, dict[str, str]]:
    """Kind -> {id: CG display name}, in generated.rs order."""
    out = {}
    for kind, table in TABLES.items():
        _, lines = rows(src, table)
        out[kind] = {field(line, "id"): field(line, "name") for line in lines}
    return out


def declared_counts(src: str) -> dict[str, int]:
    return {kind: _block(src, table)[0] for kind, table in TABLES.items()}


def move_values(src: str) -> dict[str, dict]:
    """CG's own move values, for the cross-check only."""
    out = {}
    for line in rows(src, "MOVES")[1]:
        acc = field(line, "accuracy")
        out[field(line, "id")] = {
            "type": field(line, "ty").removeprefix("Type::").lower(),
            "category": field(line, "category").removeprefix("Category::").lower(),
            "basePower": int(field(line, "base_power")),
            "accuracy": 101 if acc == "ACC_ALWAYS" else int(acc),
            "pp": int(field(line, "max_pp")),
            "priority": int(field(line, "priority")),
        }
    return out


def type_chart(src: str) -> dict[str, dict[str, float]]:
    """CG's chart as attacker -> defender -> multiplier (rows are attackers, log2 values)."""
    m = re.search(r"pub static TYPE_CHART: \[\[i8; 18\]; 18\] = \[\n(.*?)\n\];", src, re.S)
    if m is None:
        raise SystemExit("TYPE_CHART not found")
    grid = [re.findall(r"IMMUNE|-?\d+", line) for line in m.group(1).splitlines() if line.strip()]
    mult = {"IMMUNE": 0.0, "-1": 0.5, "0": 1.0, "1": 2.0}
    return {
        a: {d: mult[v] for d, v in zip(TYPE_ORDER, row, strict=True)} for a, row in zip(TYPE_ORDER, grid, strict=True)
    }
