#!/usr/bin/env python3
"""Give an ephemeral CI checkout a PyPI-unique OpenHCS version."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from packaging.version import Version


def stage_ci_candidate_version(version_source: Path, run_id: str) -> Version:
    """Rewrite the declared version with a numeric CI development suffix."""

    if not run_id.isdigit():
        raise ValueError("CI run identifiers must contain only decimal digits.")

    source = version_source.read_text(encoding="utf-8")
    declarations = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__version__"
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if len(declarations) != 1:
        raise ValueError("OpenHCS must declare exactly one literal __version__.")

    declaration = declarations[0]
    if declaration.lineno != declaration.end_lineno:
        raise ValueError("The OpenHCS version declaration must occupy one line.")
    candidate = Version(f"{declaration.value.value}.dev{run_id}")
    lines = source.splitlines(keepends=True)
    line_index = declaration.lineno - 1
    line_ending = "\n" if lines[line_index].endswith("\n") else ""
    lines[line_index] = f'__version__ = "{candidate}"{line_ending}'
    version_source.write_text("".join(lines), encoding="utf-8")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--version-source",
        type=Path,
        default=Path("openhcs/__init__.py"),
    )
    arguments = parser.parse_args()
    print(stage_ci_candidate_version(arguments.version_source, arguments.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
