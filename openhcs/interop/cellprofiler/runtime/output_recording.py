"""CellProfiler output recording."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from graphlib import TopologicalSorter
from types import MappingProxyType
from typing import ClassVar

from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ArtifactType,
    ArtifactTypeStrategyMatchMixin,
    ArtifactTypeValue,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import InvocationArtifactInputEdgePlan
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.core.runtime_adapters import RuntimeFunctionInvocationRequest
from openhcs.core.runtime_image_values import image_payload_metadata
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.runtime_output_matching import RuntimeMatchedOutput
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    ObjectRelationship,
    ObjectRelationshipDeclaration,
)
from openhcs.core.steps.function_runtime import FunctionOutputContextStrategy
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    measurement_table_for_module,
)
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    cellprofiler_profile_payload_fields,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)


class CellProfilerOutputRecorder(
    ArtifactTypeStrategyMatchMixin,
    MostDerivedContextStrategyMixin[type[ArtifactType]],
    ABC,
):
    """Nominal output writer selected by artifact type."""

    artifact_type: ClassVar[type[ArtifactType] | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_artifact_type(
        cls,
        artifact_type: ArtifactTypeValue,
    ) -> "CellProfilerOutputRecorder":
        return cls.for_context(
            ArtifactType.coerce(artifact_type),
            error_subject="CellProfiler output recorder",
        )

    @classmethod
    def transient_output_values(
        cls,
        *,
        callable_contract: CallableContract,
        active_output_plans: tuple[ArtifactOutputPlan, ...],
        returned_values: Mapping[ArtifactSpecRef, RuntimeCallableArgument],
    ) -> Mapping[ArtifactSpecRef, RuntimeCallableArgument]:
        """Return callable outputs not recorded by this active invocation."""

        runtime_adapter = callable_contract.runtime_adapter
        recorded_refs = (
            frozenset(plan.ref() for plan in active_output_plans)
            if runtime_adapter is not None
            and runtime_adapter.manages_artifact_outputs
            else frozenset()
        )
        return MappingProxyType(
            {
                spec.ref(): returned_values[spec.ref()]
                for spec in callable_contract.artifact_outputs
                if spec.ref() not in recorded_refs
            }
        )

    @classmethod
    def record_module_outputs(
        cls,
        *,
        callable_contract: CallableContract,
        active_input_edges: tuple[InvocationArtifactInputEdgePlan, ...],
        adapter: CellProfilerRuntimeAdapter,
        returned_values: Mapping[ArtifactSpecRef, RuntimeCallableArgument],
        matched_outputs: tuple[RuntimeMatchedOutput, ...],
        invocation: RuntimeFunctionInvocationRequest,
        image_request: CellProfilerImageRequest,
        current_image: RuntimeCallableArgument,
    ) -> Mapping[ArtifactSpecRef, RuntimeCallableArgument]:
        """Record one module invocation's returned artifacts."""
        function_name = callable_contract.function_name
        active_output_plans = tuple(plan for plan, _spec, _value in matched_outputs)
        active_output_refs = frozenset(plan.ref() for plan in active_output_plans)
        output_pairs = {
            output_plan.ref(): (output_plan, spec, output_value)
            for output_plan, spec, output_value in matched_outputs
        }
        output_dependencies = {
            output_plan.ref(): tuple(
                dependency_ref
                for relation in output_plan.relations
                for dependency_ref in relation.dependency_refs()
                if dependency_ref in active_output_refs
            )
            for output_plan, _spec, _output_value in matched_outputs
        }
        recording_order = tuple(
            output_pairs[ref]
            for ref in TopologicalSorter(output_dependencies).static_order()
        )

        profile_enabled = CellProfilerRuntimeProfileLogger.enabled()
        declared_only_outputs = cls.transient_output_values(
            callable_contract=callable_contract,
            active_output_plans=active_output_plans,
            returned_values=returned_values,
        )
        runtime_adapter = callable_contract.runtime_adapter
        if (
            runtime_adapter is None
            or not runtime_adapter.manages_artifact_outputs
            or not active_output_plans
        ):
            return declared_only_outputs

        output_source = replace(
            image_request,
            payload=invocation.image,
            source_image_name=invocation.source_image_name,
            image_count=invocation.image_count,
            execution_mode=invocation.execution_mode,
            plane_projection=invocation.plane_projection,
        )
        for output_plan, spec, output_value in recording_order:
            if profile_enabled:
                record_started_at = time.perf_counter()
            CellProfilerOutputRecorder.for_artifact_type(spec.artifact_type).record(
                CellProfilerOutputRecordRequest(
                    callable_contract=callable_contract,
                    active_input_edges=active_input_edges,
                    adapter=adapter,
                    spec=spec,
                    output_plan=output_plan,
                    output_value=output_value,
                    source=output_source,
                    call_kwargs=invocation.kwargs,
                    current_image=current_image,
                    declared_only_outputs=declared_only_outputs,
                )
            )
            if profile_enabled:
                CellProfilerRuntimeProfileLogger.log_module_profile_deferred(
                    "cp_output_record_one",
                    time.perf_counter() - record_started_at,
                    lambda: {
                        "function": function_name,
                        "artifact": spec.name,
                        "artifact_type": spec.artifact_type.value,
                        **cellprofiler_profile_payload_fields("value", output_value),
                    },
                )
        return declared_only_outputs

    @abstractmethod
    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        """Record one output artifact through the runtime adapter."""


class ImageOutputRecorder(CellProfilerOutputRecorder):
    """Record image outputs."""

    artifact_type = ImageArtifactType

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        module_type = CellProfilerModule.for_function_name(
            request.callable_contract.function_name
        )
        if module_type is None:
            raise KeyError(
                "No CellProfiler module declaration owns callable "
                f"{request.callable_contract.function_name!r}."
            )
        output_value = module_type.output_value(request)
        source_payload = module_type.source_payload(request)
        value = FunctionOutputContextStrategy.for_output_plan(
            request.output_plan,
        ).contextualize(
            source_payload,
            output_value,
            request.output_plan,
            request.source.plane_projection,
        )
        request.adapter.add_image(
            request.spec.name,
            value,
            materialization_source_metadata=(request.materialization_source_metadata()),
        )


class ObjectLabelsOutputRecorder(CellProfilerOutputRecorder):
    """Record object-label outputs."""

    artifact_type = ObjectLabelsArtifactType

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        module_type = CellProfilerModule.for_function_name(
            request.callable_contract.function_name
        )
        if module_type is None:
            raise KeyError(
                "No CellProfiler module declaration owns callable "
                f"{request.callable_contract.function_name!r}."
            )
        source_context = module_type.source_context(request)
        if not isinstance(request.output_value, ObjectLabelValue):
            raise TypeError(
                f"CellProfiler object-label output {request.spec.name!r} must be "
                "an ObjectLabelValue."
            )
        value = request.output_value.with_source_image_context(
            source_context.source_payload
        )
        if source_context.parent_image_payload is not None:
            value = value.with_parent_image_context(source_context.parent_image_payload)
        request.adapter.add_objects(
            request.spec.name,
            value,
            source_image_name=request.source.source_image_name,
            source_image_names=(
                request.source.source_aliases
                or source_context.source_metadata.source_image_names
            ),
            source_image_payload=source_context.source_payload,
        )


class MeasurementsOutputRecorder(CellProfilerOutputRecorder):
    """Record measurement outputs with inferred image/object ownership."""

    artifact_type = MeasurementsArtifactType

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        module_type = CellProfilerModule.for_function_name(
            request.callable_contract.function_name
        )
        if module_type is None:
            raise KeyError(
                "No CellProfiler module declaration owns callable "
                f"{request.callable_contract.function_name!r}."
            )
        module_name = module_type.require_module_name()
        function_name = request.callable_contract.function_name
        profile_enabled = CellProfilerRuntimeProfileLogger.enabled()
        if profile_enabled:
            build_started_at = time.perf_counter()
        measurement_table = measurement_table_for_module(request)
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_measurement_record_build",
                time.perf_counter() - build_started_at,
                module=module_name,
                function=function_name,
                artifact=request.spec.name,
                rows=len(measurement_table.rows),
            )
        row_policy = module_type.runtime_object_measurement_row_policy()
        row_policy.validate_table_ownership(measurement_table)
        if profile_enabled:
            materialize_started_at = time.perf_counter()
        request.adapter.add_measurements(measurement_table)
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_measurement_record_materialize",
                time.perf_counter() - materialize_started_at,
                module=module_name,
                function=function_name,
                artifact=request.spec.name,
                rows=len(measurement_table.rows),
            )


class RelationshipsOutputRecorder(CellProfilerOutputRecorder):
    """Record contract-bound directed object-lineage artifacts."""

    artifact_type = ObjectLineageArtifactType

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        if not isinstance(request.output_value, DirectedObjectRelationshipPayload):
            raise TypeError(
                f"CellProfiler callable {request.callable_contract.function_name!r} "
                "relationship output "
                f"'{request.spec.name}' must be a directed relationship payload, "
                f"got {type(request.output_value).__name__}."
            )
        relations = ArtifactSpecCollection((request.spec,)).relation_refs(
            ObjectRelationshipDeclaration
        )
        if len(relations) != 1:
            raise ValueError(
                f"Relationship output {request.spec.ref()!r} requires exactly one "
                f"ObjectRelationshipDeclaration, got {len(relations)}."
            )
        _relationship_spec, declaration = relations[0]
        source_metadata = image_payload_metadata(request.source.payload)
        request.adapter.add_relationship(
            ObjectRelationship.from_payload(
                name=request.spec.name,
                declaration=declaration,
                payload=request.output_value,
                source_provenance=source_metadata.source_provenance,
            ),
            artifact_type=request.spec.artifact_type,
        )


class SpatialGridOutputRecorder(CellProfilerOutputRecorder):
    """Record spatial-grid outputs."""

    artifact_type = SpatialGridArtifactType

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_spatial_grid(
            request.spec.name,
            request.output_value,
        )
