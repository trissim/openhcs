#!/usr/bin/env python3
"""Regenerate the Lean ArrayDSL export and Python/JAX wrappers."""

from pathlib import Path
import importlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    codegen = importlib.import_module("dq_dock_engine.codegen.arraydsl_codegen")
    for path in codegen.regenerate_bridge():
        print(path)


if __name__ == "__main__":
    main()
