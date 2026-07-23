"""CellProfiler output value and source-context policies."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, replace
from openhcs.core.source_metadata import SourceVoxelSpacing
from openhcs.core.runtime_image_values import (
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument


@dataclass(frozen=True, slots=True)
class CellProfilerObjectLabelOutputSourceContext:
    """Source provenance and CP parent-image context for one object-label output."""

    source_payload: RuntimeCallableArgument
    parent_image_payload: RuntimeCallableArgument | None

    @property
    def source_metadata(self):
        """Return metadata for the declared source payload."""
        return image_payload_metadata(self.source_payload)

    @property
    def parent_image_source_voxel_spacing(self) -> SourceVoxelSpacing:
        """Return spacing stamped from the CP parent image, or absence."""
        if self.parent_image_payload is None:
            return SourceVoxelSpacing()
        return image_payload_metadata(self.parent_image_payload).source_voxel_spacing


class InputObjectLabelWithoutParentImageOutputSourceContextPolicyMixin(
    ABC,
):
    """Use input object-label provenance without declaring a CP parent image."""

    @classmethod
    def source_context(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerObjectLabelOutputSourceContext:
        return replace(
            super().source_context(request),
            parent_image_payload=None,
        )
