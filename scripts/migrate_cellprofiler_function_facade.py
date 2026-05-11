#!/usr/bin/env python3
"""Move a converted CellProfiler function module into OpenHCS and leave a facade.

This is intentionally narrow: it handles whole-file migrations where the
benchmark module should become a compatibility import surface and the moved
implementation remains byte-for-byte owned by an OpenHCS module.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def public_top_level_names(source: str) -> list[str]:
    tree = ast.parse(source)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id.isupper():
                names.append(target.id)
    return names


def facade_source(*, original_title: str, target_module: str, names: list[str]) -> str:
    if not names:
        raise ValueError("No public top-level names found for facade generation.")
    joined_imports = ",\n    ".join(names)
    joined_all = ",\n    ".join(f'"{name}"' for name in names)
    return (
        f'"""Converted from CellProfiler: {original_title}."""\n\n'
        f"from {target_module} import (\n"
        f"    {joined_imports},\n"
        ")\n\n"
        "__all__ = [\n"
        f"    {joined_all},\n"
        "]\n"
    )


def module_name_from_path(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--title",
        default=None,
        help="CellProfiler module title for the facade docstring.",
    )
    args = parser.parse_args()

    source_path = args.source
    target_path = args.target
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if target_path.exists():
        raise FileExistsError(target_path)

    source = source_path.read_text()
    names = public_top_level_names(source)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source)
    source_path.write_text(
        facade_source(
            original_title=args.title or source_path.stem,
            target_module=module_name_from_path(target_path),
            names=names,
        )
    )


if __name__ == "__main__":
    main()
