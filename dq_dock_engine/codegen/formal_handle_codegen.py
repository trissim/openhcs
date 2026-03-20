"""Generate Python handle aliases from Lean HandleAliases.lean."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_LEAN_PATH = Path(
    "docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/HandleAliases.lean"
)
DEFAULT_OUTPUT_PATH = Path("dq_dock_engine/generated/formal_handle_aliases.py")

ABBREV_RE = re.compile(r"^abbrev\s+([A-Z0-9_]+)\s*:=")


def parse_alias_names(lean_text: str) -> list[str]:
    names: list[str] = []
    for line in lean_text.splitlines():
        match = ABBREV_RE.match(line.strip())
        if match:
            names.append(match.group(1))
    return names


def render_python_module(names: list[str], source_path: Path) -> str:
    lines = [
        '"""Generated handle aliases from Lean HandleAliases.lean."""',
        "",
        f"# Source: {source_path}",
        "",
    ]
    for name in names:
        lines.append(f'{name} = "{name}"')
    lines.append("")
    lines.append("__all__ = [")
    lines.extend(f'    "{name}",' for name in names)
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-path", type=Path, default=DEFAULT_LEAN_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    names = parse_alias_names(args.lean_path.read_text())
    output = render_python_module(names, args.lean_path)
    args.output_path.write_text(output)


if __name__ == "__main__":
    main()
