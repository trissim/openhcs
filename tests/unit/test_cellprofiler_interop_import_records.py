"""Deletion and signature gates for the pure CellProfiler import boundary."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints

from openhcs.constants import Backend
from openhcs.core.config import PipelineConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.pipeline_import import (
    import_cellprofiler_pipeline,
)


def test_import_cellprofiler_pipeline_has_the_exact_pure_boundary() -> None:
    signature = inspect.signature(import_cellprofiler_pipeline)
    parameters = signature.parameters
    hints = get_type_hints(import_cellprofiler_pipeline)

    assert tuple(parameters) == (
        "cppipe_path",
        "filemanager",
        "backend",
        "source_root",
    )
    assert parameters["cppipe_path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["filemanager"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["backend"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["source_root"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["filemanager"].default is None
    assert parameters["backend"].default is Backend.DISK
    assert parameters["source_root"].default is None
    assert hints == {
        "cppipe_path": str | Path,
        "filemanager": FileManagerLike | None,
        "backend": Backend,
        "source_root": str | Path | None,
        "return": tuple[list[FunctionStep], PipelineConfig],
    }
