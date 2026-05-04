"""Explicit registry for CellProfiler dialect compiler providers."""

from __future__ import annotations

from openhcs.interop.cellprofiler.pipeline_compiler import (
    CellProfilerDialectCompiler,
)


_REGISTERED_COMPILER: CellProfilerDialectCompiler | None = None


def register_cellprofiler_dialect_compiler(
    compiler: CellProfilerDialectCompiler,
) -> None:
    """Register the process-wide CellProfiler dialect compiler provider."""
    if not isinstance(compiler, CellProfilerDialectCompiler):
        raise TypeError(
            "compiler must implement CellProfilerDialectCompiler, got "
            f"{type(compiler).__name__}."
        )
    global _REGISTERED_COMPILER
    _REGISTERED_COMPILER = compiler


def clear_cellprofiler_dialect_compiler() -> None:
    """Clear the registered compiler provider."""
    global _REGISTERED_COMPILER
    _REGISTERED_COMPILER = None


def get_cellprofiler_dialect_compiler() -> CellProfilerDialectCompiler:
    """Return the registered compiler, failing loudly when unavailable."""
    if _REGISTERED_COMPILER is None:
        raise RuntimeError(
            "No CellProfiler dialect compiler is registered. Register an explicit "
            "CellProfilerDialectCompiler before importing .cppipe pipelines."
        )
    return _REGISTERED_COMPILER
