"""Write docs/configuration.md from config/agent.yaml: every key with its shipped value and its comment.

    uv run python tools/config_reference.py [--check]

The page is generated so that the settings and their descriptions have one source, the YAML file;
tests/unit/test_config_reference.py fails when the page is out of date. --check only compares.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config" / "agent.yaml"
PAGE = ROOT / "docs" / "configuration.md"
KEY = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][\w]*):(?P<value>[^#]*?)\s*(?:#\s?(?P<comment>.*))?$")
CONTINUED = re.compile(r"^(?P<indent> +)#\s?(?P<comment>.*)$")
HEADER = """# Configuration

Generated from `config/agent.yaml` by `tools/config_reference.py`; edit the YAML file, not this page.

Every value below is the shipped one. `--set section.key=value` on any `pcsa` command, or the environment
variable `PCSA__section__key`, changes one for a single run. Unknown and missing keys are refused at start-up.
"""


def _cell(text: str) -> str:
    return text.replace("|", "\\|").strip()


def entries(text: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    """(the file's opening comment lines, [(dotted key, value, comment)]) in file order."""
    intro: list[str] = []
    rows: list[list[str]] = []
    stack: list[tuple[int, str]] = []
    for line in text.splitlines():
        if not rows and line.startswith("#"):
            intro.append(line.lstrip("# ").rstrip())
            continue
        key = KEY.match(line)
        if key:
            indent = len(key["indent"])
            stack = [(i, k) for i, k in stack if i < indent] + [(indent, key["key"])]
            rows.append([".".join(k for _, k in stack), key["value"].strip(), (key["comment"] or "").strip()])
            continue
        more = CONTINUED.match(line)
        if more and rows:
            rows[-1][2] = f"{rows[-1][2]} {more['comment'].strip()}".strip()
    return intro, [(k, v, c) for k, v, c in rows]


def render(text: str) -> str:
    _, rows = entries(text)
    out = [HEADER]
    section = None
    for key, value, comment in rows:
        top = key.split(".")[0]
        if "." not in key and not value:
            section = top
            out += [f"\n## `{top}`\n", comment + "\n" if comment else "", "| Key | Value | Meaning |", "|---|---|---|"]
            continue
        if section != top:  # a top-level key with a value of its own
            section = None
            out += [f"\n## `{key}`\n", f"`{value}`" + (f": {comment}" if comment else ""), ""]
            continue
        shown = f"`{_cell(value)}`" if value else ""
        out.append(f"| `{key}` | {shown} | {_cell(comment)} |")
    return "\n".join(line for line in out if line is not None).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report whether the page is current; write nothing")
    a = ap.parse_args()
    page = render(SOURCE.read_text(encoding="utf-8"))
    if a.check:
        current = PAGE.is_file() and PAGE.read_text(encoding="utf-8") == page
        print("docs/configuration.md is current" if current else "docs/configuration.md is out of date")
        return 0 if current else 1
    PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
