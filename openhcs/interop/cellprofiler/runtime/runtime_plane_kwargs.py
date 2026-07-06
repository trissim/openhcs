"""Runtime-plane projection for CellProfiler invocation kwargs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.runtime_semantics import RuntimePlaneAxis, RuntimePlaneAxisProjector
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_values import (
    ObjectLabelData,
    ObjectLabelMeasurementPayloadStrategy,
    ObjectLabelSourcePlaneProjectionRequest,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelValue,
    SpatialGrid,
    image_payload_data,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.projection_requirements import (
    CellProfilerRuntimePlaneProjectionCapability,
    RuntimeArtifactImageInputProjectionCapability,
    RuntimePlaneProjectionRequirement,
    RuntimePlaneProjectionRequirementContext,
    RuntimeSliceKwargProjectionCapability,
    projection_capabilities_include,
)
from openhcs.interop.cellprofiler.runtime.runtime_special_values import (
    CellProfilerRuntimePlaneKwargValue,
)


@dataclass(frozen=True, slots=True)
class CurrentRuntimePlaneKwargValue:
    """Classify kwargs that carry a runtime-slice plane axis."""

    value: CellProfilerRuntimeValue

    def carries_runtime_slice_axis(self) -> bool:
        match self.value:
            case RuntimeSliceAlignedValueSet() as aligned_values:
                return self.aligned_values_carry_runtime_slice_axis(aligned_values)
            case SpatialGrid():
                return True
            case ObjectLabelValue() as label_value:
                return self.label_value_carries_runtime_slice_axis(label_value)
            case _:
                return False

    @classmethod
    def aligned_values_carry_runtime_slice_axis(
        cls,
        value: RuntimeSliceAlignedValueSet,
    ) -> bool:
        """Return whether any aligned slice value is plane-scoped."""
        return any(
            cls(value.value_for_slice(slice_index)).slice_value_carries_runtime_plane()
            for slice_index in range(value.slice_count)
        )

    def slice_value_carries_runtime_plane(self) -> bool:
        match self.value:
            case ObjectLabelValue() | SpatialGrid():
                return True
            case _:
                return False

    @staticmethod
    def label_value_carries_runtime_slice_axis(value: ObjectLabelValue) -> bool:
        if value.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return False
        return ObjectLabelRuntimeSliceStackContract.runtime_slice_count(value) is not None


@dataclass(frozen=True, slots=True)
class CurrentRuntimePlaneKwargProjectionContract:
    """Decide whether runtime-slice values should be projected to the current plane."""

    func: CellProfilerFunction
    default_execution_mode: ImagePayloadExecutionMode

    def projection_capabilities(
        self,
    ) -> frozenset[type[CellProfilerRuntimePlaneProjectionCapability]]:
        return RuntimePlaneProjectionRequirement.capabilities_for_context(
            RuntimePlaneProjectionRequirementContext(
                self.func,
                self.default_execution_mode,
            )
        )

    def requires_projection_capability(
        self,
        capability_type: type[CellProfilerRuntimePlaneProjectionCapability],
    ) -> bool:
        return projection_capabilities_include(
            self.projection_capabilities(),
            capability_type,
        )

    def projects_runtime_slice_kwargs(self) -> bool:
        return self.requires_projection_capability(
            RuntimeSliceKwargProjectionCapability
        )

    def projects_runtime_artifact_image_inputs(self) -> bool:
        return self.requires_projection_capability(
            RuntimeArtifactImageInputProjectionCapability
        )


@dataclass(frozen=True, slots=True)
class CurrentRuntimePlaneKwargProjection:
    """Project runtime-slice kwargs when execution is already scoped to one plane."""

    image: CellProfilerRuntimeValue
    kwargs: CellProfilerKwargs
    plane_projector: RuntimePlaneAxisProjector
    project_runtime_slice_kwargs: bool

    def kwargs_for_invocation(self) -> CellProfilerKwargs:
        if not self.project_runtime_slice_kwargs:
            return self.kwargs
        if not isinstance(self.plane_projector, RuntimePlaneAxisProjector):
            return self.kwargs
        plane_index = self.plane_projector.runtime_slice_plane_index()
        if plane_index is None:
            return self.kwargs
        plane_count = self.plane_projector.runtime_slice_axis_size()
        image_data = image_payload_data(self.image)
        if not isinstance(image_data, np.ndarray) or image_data.ndim != 2:
            return self.kwargs
        if not any(
            CurrentRuntimePlaneKwargValue(value).carries_runtime_slice_axis()
            for value in self.kwargs.values()
        ):
            return self.kwargs
        return {
            key: self.project_value(value, plane_index, plane_count)
            for key, value in self.kwargs.items()
        }

    @staticmethod
    def project_value(
        value: CellProfilerRuntimePlaneKwargValue,
        plane_index: int,
        plane_count: int | None,
    ) -> CellProfilerRuntimePlaneKwargValue:
        if isinstance(value, RuntimeSliceAlignedValueSet):
            if plane_count is not None:
                return value.value_for_aligned_slice(plane_index, plane_count)
            if plane_index < 0 or plane_index >= value.slice_count:
                raise ValueError(
                    "Runtime-slice kwarg projection index is outside aligned value "
                    f"count: plane {plane_index} for {value.slice_count} slices."
                )
            return value.value_for_slice(plane_index)
        return CurrentPlaneObjectLabelProjection(
            value=value,
            plane_index=plane_index,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).projected_value()


@dataclass(frozen=True, slots=True)
class CurrentPlaneObjectLabelProjection:
    """Project object labels only when the current invocation selects their plane axis."""

    value: ObjectLabelValue | ObjectLabelData
    plane_index: int
    plane_axis: RuntimePlaneAxis

    def projected_value(self) -> ObjectLabelValue | ObjectLabelData:
        if not isinstance(self.value, ObjectLabelValue):
            return self.value
        if self.value.plane_axis is not self.plane_axis:
            return self.value
        labels = object_label_dense_array(self.value)
        if not isinstance(labels, np.ndarray) or labels.ndim < 3:
            return self.value
        projected_index = self.projected_plane_index(labels)
        if projected_index is None:
            return self.value
        return ObjectLabelMeasurementPayloadStrategy.for_source(
            self.value
        ).materialize(
            self.value,
            ObjectLabelSourcePlaneProjectionRequest(
                labels[projected_index],
                projected_index,
            ),
        )

    def projected_plane_index(self, labels: np.ndarray) -> int | None:
        """Return a label-stack plane index when the grouped plane is in domain."""
        if labels.shape[0] == 1:
            return 0
        if self.plane_index < 0 or self.plane_index >= labels.shape[0]:
            return None
        return self.plane_index
