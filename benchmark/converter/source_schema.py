"""Compatibility aliases for CellProfiler source-schema lowering."""

from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.interop.cellprofiler.source_schema import (
    SetupModuleCompiler,
    compile_image_schema,
)

__all__ = ("PipelineImageSchema", "SetupModuleCompiler", "compile_image_schema")
