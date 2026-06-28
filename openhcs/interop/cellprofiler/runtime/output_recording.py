"""CellProfiler output recording."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from functools import lru_cache
import time
from types import MappingProxyType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection
from openhcs.core.runtime_semantics import ParentChildRelationshipPayload
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_values import (
    SpatialGrid,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerInvocationRequest,
)
from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
    CellProfilerMeasurementMaterializer,
)
from openhcs.interop.cellprofiler.runtime.output_contexts import (
    CellProfilerImageOutputContextStrategy,
    CellProfilerObjectLabelOutputContextStrategy,
)
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.interop.cellprofiler.runtime.output_value_resolution import (
    CellProfilerResolvedOutputValues,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    CellProfilerRuntimeValueSequence,
)
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    cellprofiler_profile_payload_fields,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    measurement_record_for_module,
)
from openhcs.interop.cellprofiler.runtime.relationship_endpoints import (
    RelationshipEndpointResolver,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)


@dataclass(frozen=True, slots=True)
class CellProfilerOutputRecordingPlan:
    """Prepared output order and recorders for one module contract."""

    ordered_outputs: tuple[ArtifactSpec, ...]
    recorders: Mapping[ArtifactKind, CellProfilerOutputRecorder]

    @classmethod
    def from_outputs(
        cls,
        outputs: tuple[ArtifactSpec, ...],
    ) -> "CellProfilerOutputRecordingPlan":
        ordered_outputs = _output_recording_order(outputs)
        return cls(
            ordered_outputs=ordered_outputs,
            recorders=MappingProxyType(
                {
                    kind: CellProfilerOutputRecorder.for_kind(kind)
                    for kind in {spec.kind for spec in ordered_outputs}
                }
            ),
        )


class CellProfilerOutputRecorder(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal output writer selected by artifact kind."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    stable_key_axis: ClassVar[str] = "kind"
    kind: ClassVar[ArtifactKind | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_kind(cls, kind: ArtifactKind) -> "CellProfilerOutputRecorder":
        recorder_type = cls.__registry__.get(kind)
        if recorder_type is None:
            raise TypeError(f"Unsupported CellProfiler output kind {kind.value}.")
        return recorder_type()

    @classmethod
    def recording_dependency_depth(cls) -> int:
        """Return dependency order from the recorder inheritance chain."""
        return cls.mro().index(CellProfilerOutputRecorder)

    @classmethod
    def record_module_outputs(
        cls,
        *,
        runtime_plan: "CellProfilerModuleRuntimePlan",
        adapter: CellProfilerRuntimeAdapter,
        main_output: CellProfilerRuntimeValue,
        artifact_values: CellProfilerRuntimeValues,
        invocation: CellProfilerInvocationRequest,
        image_request: CellProfilerImageRequest,
        current_image: CellProfilerRuntimeValue,
    ) -> None:
        """Record one module invocation's returned artifacts."""
        contract = runtime_plan.contract
        func = runtime_plan.func
        function_name = runtime_plan.function_name
        if not contract.outputs:
            return

        profile_enabled = CellProfilerRuntimeProfileLogger.enabled()
        if profile_enabled:
            values_started_at = time.perf_counter()
        resolved_values = CellProfilerResolvedOutputValues.from_returned_outputs(
            recorded_specs=contract.outputs,
            context_specs=contract.declared_outputs or contract.outputs,
            main_output=main_output,
            artifact_values=artifact_values,
            func=func,
            declared_output_specs=contract.declared_outputs,
        )
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_output_values_by_kind",
                time.perf_counter() - values_started_at,
                module=contract.module_name,
                function=function_name,
                outputs=len(contract.outputs),
            )

        output_source = replace(
            image_request,
            payload=invocation.image,
            source_image_name=invocation.source_image_name,
            image_count=invocation.image_count,
            execution_mode=invocation.execution_mode,
        )
        for spec in runtime_plan.output_recording_plan.ordered_outputs:
            if profile_enabled:
                record_started_at = time.perf_counter()
            output_value = resolved_values.recorded_value(spec)
            runtime_plan.output_recording_plan.recorders[spec.kind].record(
                CellProfilerOutputRecordRequest(
                    runtime_plan=runtime_plan,
                    adapter=adapter,
                    spec=spec,
                    output_value=output_value,
                    output_values=resolved_values.context_values,
                    source=output_source,
                    func=func,
                    function_name=function_name,
                    call_kwargs=invocation.kwargs,
                    current_image=current_image,
                )
            )
            if profile_enabled:
                CellProfilerRuntimeProfileLogger.log_module_profile_deferred(
                    "cp_output_record_one",
                    time.perf_counter() - record_started_at,
                    lambda: {
                        "module": contract.module_name,
                        "function": function_name,
                        "artifact": spec.name,
                        "kind": spec.kind.value,
                        **cellprofiler_profile_payload_fields("value", output_value),
                    },
                )

    @abstractmethod
    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        """Record one output artifact through the runtime adapter."""


class ImmediateOutputRecorder(CellProfilerOutputRecorder):
    """Recorder family for artifacts that create no later recording dependency."""


class RelationshipDependentOutputRecorder(ImmediateOutputRecorder):
    """Recorder family for artifacts that require object endpoints to exist."""


class MeasurementDependentOutputRecorder(RelationshipDependentOutputRecorder):
    """Recorder family for artifacts that may require prior relationships."""


class ImageOutputRecorder(ImmediateOutputRecorder):
    """Record image outputs."""

    kind = ArtifactKind.IMAGE

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        output_value = request.runtime_plan.image_output_value_policy.output_value(
            request
        )
        source_payload = (
            request.runtime_plan.image_output_source_payload_policy.source_payload(
                request
            )
        )
        value = CellProfilerImageOutputContextStrategy.for_value(
            output_value
        ).runtime_image_value(
            output_value,
            source_payload,
        )
        request.adapter.add_image(
            request.spec.name,
            value,
            source_image_name=request.source.source_image_name,
        )


class ObjectLabelsOutputRecorder(ImmediateOutputRecorder):
    """Record object-label outputs."""

    kind = ArtifactKind.OBJECT_LABELS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        source_payload = request.object_label_output_source_payload()
        source_metadata = image_payload_metadata(source_payload)
        if isinstance(request.output_value, np.ndarray):
            request.adapter.add_source_image_objects(
                request.spec.name,
                request.output_value,
                source_image_name=request.source.source_image_name,
                source_image_names=(
                    request.source.source_aliases
                    or source_metadata.source_image_names
                ),
                source_image_payload=source_payload,
                domain_scope=request.object_label_output_domain_scope(),
            )
            return
        value = CellProfilerObjectLabelOutputContextStrategy.for_value(
            request.output_value
        ).runtime_object_label_value(
            request.output_value,
            source_payload,
            request.object_label_output_domain_scope(),
        )
        request.adapter.add_objects(
            request.spec.name,
            value,
            source_image_name=request.source.source_image_name,
            source_image_names=(
                request.source.source_aliases or source_metadata.source_image_names
            ),
            source_image_payload=source_payload,
        )


class MeasurementsOutputRecorder(MeasurementDependentOutputRecorder):
    """Record measurement outputs with inferred image/object ownership."""

    kind = ArtifactKind.MEASUREMENTS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        function_name = request.function_name
        profile_enabled = CellProfilerRuntimeProfileLogger.enabled()
        if profile_enabled:
            build_started_at = time.perf_counter()
        measurement_record = measurement_record_for_module(request)
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_measurement_record_build",
                time.perf_counter() - build_started_at,
                module=request.module_name,
                function=function_name,
                artifact=request.spec.name,
                rows=len(measurement_record.rows),
            )
        if self._records_runtime_artifact(request):
            if profile_enabled:
                materialize_started_at = time.perf_counter()
            CellProfilerMeasurementMaterializer.record(
                measurement_record.materialization_request(
                    adapter=request.adapter,
                    name=request.spec.name,
                    output_values=request.output_values,
                    current_image=request.current_image,
                )
            )
            if profile_enabled:
                CellProfilerRuntimeProfileLogger.log_module_profile(
                    "cp_measurement_record_materialize",
                    time.perf_counter() - materialize_started_at,
                    module=request.module_name,
                    function=function_name,
                    artifact=request.spec.name,
                    rows=len(measurement_record.rows),
                )
            return

        row_policy = request.runtime_plan.object_measurement_row_policy
        if profile_enabled:
            partitions_started_at = time.perf_counter()
        partitions = row_policy.record_partitions(measurement_record)
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_measurement_record_partitions",
                time.perf_counter() - partitions_started_at,
                module=request.module_name,
                function=function_name,
                artifact=request.spec.name,
                rows=len(measurement_record.rows),
                partitions=len(partitions),
            )
        for partition in partitions:
            if profile_enabled:
                materialize_started_at = time.perf_counter()
            CellProfilerMeasurementMaterializer.record(
                partition.materialization_request(
                    adapter=request.adapter,
                    name=request.spec.name,
                    output_values=request.output_values,
                    current_image=request.current_image,
                )
            )
            if profile_enabled:
                CellProfilerRuntimeProfileLogger.log_module_profile(
                    "cp_measurement_record_materialize",
                    time.perf_counter() - materialize_started_at,
                    module=request.module_name,
                    function=function_name,
                    artifact=request.spec.name,
                    rows=len(partition.rows),
                )

    @staticmethod
    def _records_runtime_artifact(request: CellProfilerOutputRecordRequest) -> bool:
        """Return whether the adapter has one compiled output slot for this table."""
        return (
            isinstance(request.adapter, CellProfilerRuntimeAdapter)
            and request.spec.name in request.adapter.artifact_outputs
        )


class RelationshipsOutputRecorder(RelationshipDependentOutputRecorder):
    """Record parent-child relationship artifacts."""

    kind = ArtifactKind.RELATIONSHIPS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        if not isinstance(request.output_value, ParentChildRelationshipPayload):
            raise TypeError(
                f"{request.module_name} relationship output "
                f"'{request.spec.name}' must be ParentChildRelationshipPayload, "
                f"got {type(request.output_value).__name__}."
            )
        parent_spec, child_spec = RelationshipEndpointResolver.for_request(
            request
        ).endpoint_specs(request.spec)
        source_metadata = image_payload_metadata(request.source.payload)
        request.adapter.add_relationship(
            request.spec.name,
            parent_object_name=parent_spec.name,
            child_object_name=child_spec.name,
            parent_ids=request.output_value.parent_ids,
            child_ids=request.output_value.child_ids,
            slice_indices=request.output_value.explicit_slice_indices(),
            slice_count=request.output_value.slice_count,
            source_path=source_metadata.source_path,
            source_component_metadata=source_metadata.source_component_metadata,
            source_image_provenance_planes=(
                source_metadata.source_image_provenance_planes
            ),
        )


class SpatialGridOutputRecorder(ImmediateOutputRecorder):
    """Record spatial-grid outputs."""

    kind = ArtifactKind.SPATIAL_GRID

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_spatial_grid(
            request.spec.name,
            _coerce_spatial_grid(request.output_value, request.spec.name),
        )


def _output_recording_order(
    output_specs: tuple[ArtifactSpec, ...],
) -> tuple[ArtifactSpec, ...]:
    return tuple(
        sorted(
            output_specs,
            key=lambda spec: type(
                CellProfilerOutputRecorder.for_kind(spec.kind)
            ).recording_dependency_depth(),
        )
    )


def _coerce_spatial_grid(
    value: CellProfilerRuntimeValue,
    name: str,
) -> SpatialGrid | CellProfilerKwargs | RuntimeSliceAlignedValues[CellProfilerRuntimeValue]:
    match value:
        case RuntimeSliceAlignedValues(slices=slices):
            return RuntimeSliceAlignedValues(
                slices=tuple(_coerce_spatial_grid(item, name) for item in slices)
            )
        case SpatialGrid() as grid:
            return grid.with_name(name)
        case Mapping() as mapping:
            return mapping
        case _ if is_dataclass(value):
            return asdict(value)
        case _:
            raise TypeError(
                f"Spatial grid output '{name}' must be SpatialGrid or mapping-backed, "
                f"got {type(value).__name__}."
            )
