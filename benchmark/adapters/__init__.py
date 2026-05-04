"""Tool adapters."""

from __future__ import annotations

import importlib
from typing import Any


_PUBLIC_EXPORTS: dict[str, tuple[str, str]] = {
    "CellProfilerAdapter": (
        "benchmark.adapters.cellprofiler",
        "CellProfilerAdapter",
    ),
    "OpenHCSAdapter": ("benchmark.adapters.openhcs", "OpenHCSAdapter"),
}


def __getattr__(name: str) -> Any:
    """Load adapter classes on demand."""
    if name not in _PUBLIC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _PUBLIC_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = tuple(_PUBLIC_EXPORTS)
