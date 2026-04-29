"""Typed settings lowering for CellProfiler UntangleWorms."""

from __future__ import annotations

from benchmark.cellprofiler_library.functions.untangleworms import (
    coerce_overlap_style,
)

from .parser import ModuleBlock


def untangle_worms_bound_kwargs(module: ModuleBlock) -> dict[str, str]:
    """Bind UntangleWorms settings that affect runtime output semantics."""
    overlap_style = coerce_overlap_style(
        module.get_setting("Overlap style", "Without overlap")
    )
    return {"overlap_style": overlap_style.value}
