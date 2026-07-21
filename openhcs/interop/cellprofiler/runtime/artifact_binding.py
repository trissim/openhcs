"""Runtime artifact binding authorities for CellProfiler modules."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import inspect
from typing import cast
from typing import get_origin

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ArtifactType,
    ArtifactTypeStrategyMatchMixin,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
    MeasurementsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    ImageOutputBundle,
)
from openhcs.core.callable_contract import ImagePayloadConsumption
from openhcs.core.function_patterns import InvocationArtifactInputEdgePlan
from openhcs.core.pipeline.function_contracts import (
    object_label_input_execution_mode_from_callable,
)
from openhcs.core.runtime_artifact_queries import MeasurementTableUnion
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelSet,
    ObjectLabelValue,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_slice_alignment import (
    RuntimeSliceAlignedValues,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_stores import RuntimeArtifactInput
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.source_plane_alignment import (
    SourcePayloadPlaneIdentitySequence,
    SourcePlaneIdentitySequenceAlignment,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisStrategy,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.variable_component_stack_requirement import (
    VariableComponentStackRequirementRequest,
)
from openhcs.interop.cellprofiler.image_normalization import (
    normalize_cellprofiler_image_payload,
)
from openhcs.interop.cellprofiler.runtime.adapter import (
    CellProfilerRuntimeAdapter,
)
from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
    single_source_name,
)
from openhcs.interop.cellprofiler.runtime.main_flow import (
    cellprofiler_main_flow_output,
)
from openhcs.core.steps.function_runtime import (
    RuntimeCallableArgument,
    RuntimeCallableKwargs,
    RuntimeFunctionOutput,
)
from openhcs.core.registry_strategies import (
    MostDerivedContextStrategyMixin,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeArtifactInputRequest:
    """One resolved artifact value dispatched through its nominal type strategy."""

    spec: ArtifactSpec
    value: RuntimeCallableArgument
    image_payload_consumption: ImagePayloadConsumption = ImagePayloadConsumption.NATURAL

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ArtifactSpec):
            raise TypeError(
                "RuntimeArtifactInputRequest.spec must be ArtifactSpec, got "
                f"{type(self.spec).__name__}."
            )
        if not isinstance(self.image_payload_consumption, ImagePayloadConsumption):
            raise TypeError(
                "RuntimeArtifactInputRequest.image_payload_consumption must be "
                "ImagePayloadConsumption, got "
                f"{type(self.image_payload_consumption).__name__}."
            )

class RuntimeArtifactTypeStrategy(
    ArtifactTypeStrategyMatchMixin,
    MostDerivedContextStrategyMixin[type[ArtifactType]],
    ABC,
):
    """Nominal strategy family for ArtifactType-specific runtime semantics."""

    @classmethod
    @lru_cache(maxsize=None)
    def for_artifact_type(
        cls,
        artifact_type: ArtifactType,
    ) -> "RuntimeArtifactTypeStrategy":
        return cls.for_context(
            ArtifactType.coerce(artifact_type),
            error_subject="CellProfiler runtime artifact type strategy",
        )

    @classmethod
    def for_main_flow_outputs(
        cls,
        outputs: tuple[tuple[ArtifactSpec, RuntimeCallableArgument], ...],
    ) -> "RuntimeArtifactTypeStrategy":
        """Select one nominal strategy from the complete exact output set."""

        artifact_types = frozenset(spec.artifact_type for spec, _value in outputs)
        if not artifact_types:
            raise ValueError("CellProfiler main-flow publication requires an output.")
        if len(artifact_types) != 1:
            raise TypeError(
                "CellProfiler main-flow outputs require one exact artifact type; "
                f"got {tuple(sorted(kind.require_value() for kind in artifact_types))!r}."
            )
        (artifact_type,) = artifact_types
        return cls.for_artifact_type(artifact_type)

    def runtime_input_value(
        self, request: RuntimeArtifactInputRequest
    ) -> RuntimeCallableArgument:
        """Return the runtime payload bound into absorbed function kwargs."""

        return request.value

    def raw_runtime_input_value(
        self, request: RuntimeArtifactInputRequest
    ) -> RuntimeCallableArgument:
        """Return the runtime payload before CellProfiler intensity coercion."""
        return self.runtime_input_value(request)

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        """Return the transitive source image name for one artifact input."""
        del request
        return None

    def source_image_payload(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> RuntimeCallableArgument | None:
        """Return an image payload that carries this artifact's source paths."""
        return None

    def published_main_flow_output(
        self,
        input_value: RuntimeCallableArgument,
        outputs: tuple[tuple[ArtifactSpec, RuntimeCallableArgument], ...],
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> RuntimeCallableArgument:
        """Publish one recorded artifact through the canonical OpenHCS flow."""

        del input_value, plane_projection
        self.validate_main_flow_outputs(outputs)
        if len(outputs) != 1:
            raise ValueError(
                f"{type(self).__name__} requires exactly one main-flow output, "
                f"got {len(outputs)}."
            )
        return outputs[0][1]

    def validate_main_flow_outputs(
        self,
        outputs: tuple[tuple[ArtifactSpec, RuntimeCallableArgument], ...],
    ) -> None:
        """Require every published output to belong to this nominal strategy."""

        artifact_type = type(self).artifact_type
        mismatched = tuple(
            spec.ref()
            for spec, _value in outputs
            if spec.artifact_type is not artifact_type
        )
        if mismatched:
            raise TypeError(
                f"{type(self).__name__} cannot publish outputs {mismatched!r}."
            )


class ImageArtifactTypeStrategy(RuntimeArtifactTypeStrategy):
    """Resolve image artifact payloads and source-image lineage."""

    artifact_type = ImageArtifactType

    def raw_runtime_input_value(
        self, request: RuntimeArtifactInputRequest
    ) -> RuntimeCallableArgument:
        payload = request.value
        metadata = image_payload_metadata(payload)
        metadata = metadata.with_source_provenance(
            metadata.source_provenance.with_derived_source_image_names(
                (request.spec.name,)
            )
        )
        return metadata.payload_with(
            image_payload_data(payload),
            mask=image_payload_mask(payload),
        )

    def runtime_input_value(
        self, request: RuntimeArtifactInputRequest
    ) -> RuntimeCallableArgument:
        return normalize_cellprofiler_image_payload(
            self.raw_runtime_input_value(request)
        )

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        return single_source_name(
            image_payload_metadata(
                self.raw_runtime_input_value(request)
            ).source_provenance.represented_source_image_names
        )

    def source_image_payload(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> RuntimeCallableArgument | None:
        return self.raw_runtime_input_value(request)

    def published_main_flow_output(
        self,
        input_value: RuntimeCallableArgument,
        outputs: tuple[tuple[ArtifactSpec, RuntimeCallableArgument], ...],
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> RuntimeCallableArgument:
        """Publish one or more named image outputs with exact plane context."""

        self.validate_main_flow_outputs(outputs)
        if not outputs:
            raise ValueError("Image main-flow publication requires an output.")
        return ImageOutputBundle(
            tuple(
                cellprofiler_main_flow_output(
                    input_value,
                    output_value,
                    plane_projection,
                )
                for _spec, output_value in outputs
            ),
            tuple(
                AlignedImageSliceContext.main_flow(
                    output_key=spec.name,
                    artifact_kind=spec.artifact_type.value,
                )
                for spec, _value in outputs
            ),
        )


class ObjectLabelsArtifactTypeStrategy(RuntimeArtifactTypeStrategy):
    """Resolve object-label payloads and lineage."""

    artifact_type = ObjectLabelsArtifactType

    def object_labels(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> ObjectLabelSet:
        """Return the native object value carrying its source-image provenance."""

        value = request.value
        if isinstance(value, ObjectLabelSet):
            return value
        return SourceImageObjectLabelBuildRequest(
            image=value,
            labels=image_payload_data(value),
        ).label_set(
            name=request.spec.name,
            source_image_name=request.spec.name,
        )

    def runtime_input_value(
        self, request: RuntimeArtifactInputRequest
    ) -> RuntimeCallableArgument:
        return self.object_labels(request)

    def raw_runtime_input_value(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> RuntimeCallableArgument:
        """Return the nominal label set in the invocation's component scope."""

        return self.object_labels(request)

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        return self.object_labels(request).source_image_name

    def source_image_payload(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> RuntimeCallableArgument | None:
        return self.object_labels(request)

class MeasurementsArtifactTypeStrategy(RuntimeArtifactTypeStrategy):
    """Resolve measurement payloads and lineage."""

    artifact_type = MeasurementsArtifactType

    def runtime_input_value(
        self, request: RuntimeArtifactInputRequest
    ) -> RuntimeCallableArgument:
        value = request.value
        if not isinstance(value, MeasurementTable):
            raise TypeError(
                f"Measurement artifact {request.spec.name!r} requires a "
                f"MeasurementTable, got {type(value).__name__}."
            )
        return value.rows

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        value = request.value
        if not isinstance(value, MeasurementTable):
            raise TypeError(
                f"Measurement artifact {request.spec.name!r} requires a "
                f"MeasurementTable, got {type(value).__name__}."
            )
        return value.source_image_name


class RelationshipsArtifactTypeStrategy(RuntimeArtifactTypeStrategy):
    """Resolve relationship payloads."""

    artifact_type = ObjectLineageArtifactType


class SpatialGridArtifactTypeStrategy(RuntimeArtifactTypeStrategy):
    """Resolve spatial-grid payloads."""

    artifact_type = SpatialGridArtifactType


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeInputBindingRequest:
    """Exact compiled artifact context for one callable invocation."""

    adapter: CellProfilerRuntimeAdapter
    kwargs: RuntimeCallableKwargs
    current_image: RuntimeArrayData
    selected_object_inputs: tuple[ArtifactSpec, ...] | None = None

    def __post_init__(self) -> None:
        declared_inputs = self.adapter.request.selected_artifact_input_specs()
        if self.selected_object_inputs is None:
            return
        declared_refs = declared_inputs.ref_set()
        selected_refs = ArtifactSpecCollection(self.selected_object_inputs).ref_set()
        if not selected_refs.issubset(declared_refs):
            raise ValueError(
                "Selected object inputs must belong to the callable input "
                f"set, got undeclared refs {selected_refs - declared_refs!r}."
            )

    def require_string_kwarg(self, name: str) -> str:
        """Return one required non-empty string from this exact invocation."""

        value = self.kwargs.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{self.module_name} requires non-empty kwarg {name!r}."
            )
        return value

    @property
    def module_name(self) -> str:
        """Return the nominal module name carried by the callable contract."""

        module_name = self.adapter.request.require_callable_contract().module_name
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("CellProfiler callable contract requires a module name.")
        return module_name

    @property
    def declared_inputs(self) -> ArtifactSpecCollection:
        """Return active compiled input declarations in callable ABI order."""

        return self.adapter.request.selected_artifact_input_specs()

    @property
    def input_edges(self) -> tuple[InvocationArtifactInputEdgePlan, ...]:
        """Return exact compiled input occurrences in declaration order."""

        return tuple(self.adapter.request.artifact_inputs.values())

    def input_edge_for_spec(
        self,
        spec: ArtifactSpec,
    ) -> InvocationArtifactInputEdgePlan:
        """Return one value-equivalent compiled occurrence for a declaration."""

        edge = self.adapter.request.require_artifact_input_edge(spec.ref())
        if edge.spec != spec:
            raise ValueError(
                f"{self.module_name} compiled input occurrence drifts from its "
                f"declared spec {spec!r}: {edge.spec!r}."
            )
        return edge

    @property
    def parameter_input_edges(self) -> tuple[InvocationArtifactInputEdgePlan, ...]:
        """Return compiled input edges that bind exact callable parameters."""

        return tuple(
            edge
            for edge in self.input_edges
            if edge.spec.parameter_name is not None
        )

    @property
    def object_inputs(self) -> tuple[ArtifactSpec, ...]:
        """Return the exact object inputs selected for this binding pass."""

        inputs = (
            self.declared_inputs.specs
            if self.selected_object_inputs is None
            else self.selected_object_inputs
        )
        return ArtifactSpecCollection(inputs).of_artifact_type(ObjectLabelsArtifactType)

    @property
    def unbound_object_inputs(self) -> tuple[ArtifactSpec, ...]:
        """Return object inputs not already owned by the special-input ABI."""

        parameter_refs = frozenset(edge.spec.ref() for edge in self.parameter_input_edges)
        return tuple(
            spec for spec in self.object_inputs if spec.ref() not in parameter_refs
        )

    @property
    def image_inputs(self) -> tuple[ArtifactSpec, ...]:
        """Return image artifacts assigned to special-input parameters."""

        return tuple(
            edge.spec
            for edge in self.parameter_input_edges
            if edge.spec.artifact_type is ImageArtifactType
        )

    @property
    def primary_image_inputs(self) -> tuple[ArtifactSpec, ...]:
        """Return image declarations carried by the ordinary main image argument."""

        parameter_refs = frozenset(edge.spec.ref() for edge in self.parameter_input_edges)
        return tuple(
            spec
            for spec in self.declared_inputs.of_artifact_type(ImageArtifactType)
            if spec.ref() not in parameter_refs
        )

    def with_object_inputs(
        self,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> "RuntimeInputBindingRequest":
        """Restrict object binding to an exact compiled input subset."""

        return replace(self, selected_object_inputs=object_inputs)

    @property
    def func(self) -> Callable[..., RuntimeFunctionOutput]:
        """Return the canonical callable owned by the compiled contract."""
        return cast(
            Callable[..., RuntimeFunctionOutput],
            self.adapter.request.require_callable_contract().resolve_canonical_raw_callable(),
        )

    @property
    def declared_measurement_specs(self) -> tuple[ArtifactSpec, ...]:
        """Return measurement artifacts explicitly declared for this runtime binding."""
        return tuple(
            spec
            for spec in self.declared_inputs
            if spec.artifact_type is MeasurementsArtifactType
        )

    def declared_measurement_tables(self) -> tuple[MeasurementTable, ...]:
        """Return measurement tables from artifacts explicitly declared as inputs."""
        measurement_specs = self.declared_measurement_specs
        if not measurement_specs:
            return ()
        tables: list[MeasurementTable] = []
        slice_axis = MeasurementRowAxisField.SLICE_INDEX
        for spec in measurement_specs:
            spec_tables = tuple(
                cast(MeasurementTable, record.value.data)
                for record in self.adapter.artifact_input_records(
                    spec.name,
                    MeasurementsArtifactType,
                )
            )
            MeasurementTableUnion(spec.name, spec_tables).row_axis_domain(slice_axis)
            tables.extend(spec_tables)
        return tuple(tables)

    def main_flow_value(self, spec: ArtifactSpec) -> RuntimeCallableArgument:
        """Project one exact callable input from the current OpenHCS image flow."""

        current_image = self.current_image
        if isinstance(current_image, AlignedImageStack) and current_image.slice_contexts:
            payload = current_image.output_payload(spec.ref())
            if payload is None:
                raise ValueError(
                    "Aligned main-flow payload does not carry callable input "
                    f"{spec.ref()!r}; carried contexts are "
                    f"{current_image.slice_contexts!r}."
                )
            return payload
        metadata = image_payload_metadata(current_image)
        if (
            len(self.primary_image_inputs) == 1
            and self.primary_image_inputs[0] == spec
            and metadata.source_provenance.source_plane_count == 0
        ):
            return current_image
        return metadata.project_declared_source_image(current_image, spec.name)

    def stack_broadcast_source_value(
        self,
        source_ref: ArtifactSpecRef,
    ) -> RuntimeCallableArgument:
        """Resolve the exact image payload owning a stack-broadcast input."""

        if source_ref.artifact_type is not ImageArtifactType:
            raise TypeError(
                f"{self.module_name} stack-broadcast source {source_ref!r} must "
                "be an image artifact."
            )
        if source_ref in self.declared_inputs.ref_set():
            source_spec = self.declared_inputs.by_ref(source_ref)
            source_edge = self.input_edge_for_spec(source_spec)
            return RuntimeArtifactTypeStrategy.for_artifact_type(
                ImageArtifactType
            ).runtime_input_value(self.artifact_request(source_edge))

        current_image = self.current_image
        if isinstance(current_image, AlignedImageStack) and current_image.slice_contexts:
            payload = current_image.output_payload(source_ref)
            if payload is None:
                raise ValueError(
                    f"{self.module_name} current main flow does not carry declared "
                    f"stack-broadcast source {source_ref!r}."
                )
            return payload
        return image_payload_metadata(current_image).project_declared_source_image(
            current_image,
            source_ref.name,
        )

    def artifact_request(
        self,
        edge: InvocationArtifactInputEdgePlan,
    ) -> RuntimeArtifactInputRequest:
        """Resolve one declaration from exactly one compiled runtime authority."""

        spec = edge.spec
        declared = self.declared_inputs.by_ref(spec.ref())
        if declared != spec:
            raise ValueError(
                f"{self.module_name} does not declare artifact input {spec.ref()!r}."
            )
        source_plan = self.adapter.request.source_binding_plan
        source_binding = source_plan.binding_for_artifact_ref(spec.ref())
        consumes_main_flow = edge.consumes_main_flow
        source_artifact_binding = source_binding if not consumes_main_flow else None
        runtime_edge = (
            edge
            if edge.storage_plan is not None and not consumes_main_flow
            else None
        )
        authority_count = sum(
            (
                source_artifact_binding is not None,
                runtime_edge is not None,
                consumes_main_flow,
            )
        )
        if authority_count != 1:
            raise ValueError(
                f"{self.module_name} input {spec.ref()!r} must resolve through "
                "exactly one source binding, main-flow payload, or compiled input "
                f"edge; found {authority_count}."
            )
        if source_artifact_binding is not None:
            value = cast(
                RuntimeCallableArgument,
                self.adapter.request.source_artifact_payload(spec.ref()),
            )
        elif runtime_edge is not None:
            runtime_input = RuntimeArtifactInput(
                edge_plan=runtime_edge,
                axis_scope=self.adapter.request.axis_scope,
                backend=self.adapter.backend,
            )
            value = cast(
                RuntimeCallableArgument,
                runtime_input.composed_value(
                    runtime_input.records(
                        self.adapter.request.context.runtime_value_store
                    )
                ),
            )
        else:
            value = self.main_flow_value(spec)
        return RuntimeArtifactInputRequest(
            spec=spec,
            value=value,
            image_payload_consumption=(
                self.adapter.request.require_callable_contract().image_payload_consumption
            ),
        )

    def artifact_request_for_spec(
        self,
        spec: ArtifactSpec,
    ) -> RuntimeArtifactInputRequest:
        """Resolve a declaration whose occurrences share one runtime authority."""

        return self.artifact_request(self.input_edge_for_spec(spec))

    def label_payload_for(self, spec: ArtifactSpec) -> ObjectLabelValue:
        return ObjectLabelsArtifactTypeStrategy().object_labels(
            self.artifact_request_for_spec(spec)
        )

    def current_plane_label_payload(
        self,
        payload: ObjectLabelValue,
    ) -> ObjectLabelValue:
        """Project loaded object labels to the invocation's current plane."""
        if payload.plane_axis is None:
            return payload
        plane_count = payload.declared_plane_count()
        if plane_count is None:
            raise ValueError(
                "Slice-aligned object labels require a declared plane count."
            )
        plane_index = RuntimePlaneAxisStrategy.for_enum_member(
            payload.plane_axis
        ).plane_index(
            self.adapter,
            payload.source_aliases,
        )
        if plane_index is None:
            if plane_count != 1:
                return payload
            plane_index = 0
        projection = RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=payload.plane_axis,
            source_aliases=payload.source_aliases,
            plane_index=plane_index,
            axis_size=plane_count,
        )
        return cast(
            ObjectLabelValue,
            RuntimeSliceProjection.value_for_slice(payload, projection),
        )

    def label_argument_for(
        self,
        spec: ArtifactSpec,
        parameter_name: str,
    ) -> ObjectLabelValue:
        """Bind the nominal object-label artifact without erasing its semantics."""

        del parameter_name
        return self.project_label_argument(self.label_payload_for(spec))

    def project_label_argument(
        self,
        payload: ObjectLabelValue,
    ) -> ObjectLabelValue:
        """Project one loaded object-label payload into callable invocation scope."""

        execution_mode = object_label_input_execution_mode_from_callable(self.func)
        stack_requirement = (
            self.adapter.request.require_callable_contract()
            .variable_component_stack_requirement
        )
        image_stack_required = (
            stack_requirement is not None
            and stack_requirement.is_required(
                VariableComponentStackRequirementRequest(
                    func=self.func,
                    kwargs=self.kwargs,
                )
            )
        )
        if execution_mode.preserves_full_stack(
            image_stack_required=image_stack_required
        ):
            return payload

        return self.current_plane_label_payload(payload)

    def current_plane_relationship_for(
        self, spec: ArtifactSpec
    ) -> RuntimeCallableArgument:
        """Return a relationship payload projected to the invocation plane."""
        relationship = self.adapter.get_relationship(
            spec.name,
        )
        plane_index = self.relationship_runtime_slice_index()
        if plane_index is None:
            return relationship
        if isinstance(relationship, RuntimeSliceAlignedValues):
            plane_count = self.adapter.runtime_slice_axis_size()
            if plane_count is None:
                plane_count = relationship.payload.slice_count
            return relationship.value_for_aligned_slice(plane_index, plane_count)
        return relationship.project_runtime_slice(plane_index)

    def relationship_runtime_slice_index(self) -> int | None:
        """Return the relationship row slice index for true runtime-slice groups."""
        plane_index = self.adapter.runtime_slice_plane_index()
        if plane_index is None:
            return None
        if self.object_inputs_projected_by_source_binding_axis():
            return None
        return plane_index

    def object_inputs_projected_by_source_binding_axis(self) -> bool:
        """Return whether object inputs are already scoped by source binding."""
        for spec in self.object_inputs:
            labels = self.label_payload_for(spec)
            if labels.plane_axis is not RuntimePlaneAxis.SOURCE_BINDING:
                continue
            aliases = labels.source_aliases
            if not aliases:
                raise ValueError(
                    "Source-binding object labels require explicit source aliases."
                )
            declared_projection = labels.declared_plane_projection()
            if declared_projection is None:
                raise ValueError(
                    "Source-binding object labels require a declared plane projection."
                )
            source_plane_index = labels.source_alias_plane_index(
                aliases,
                declared_projection.axis_size,
            )
            if source_plane_index is not None:
                return True
        return False

    def runtime_value(
        self,
        edge: InvocationArtifactInputEdgePlan,
        parameter_name: str | None = None,
    ) -> RuntimeCallableArgument:
        """Load one declared runtime artifact for callable binding."""

        spec = edge.spec
        if spec.artifact_type is ObjectLabelsArtifactType:
            payload = ObjectLabelsArtifactTypeStrategy().object_labels(
                self.artifact_request(edge)
            )
            if parameter_name is not None:
                return self.project_label_argument(payload)
            return payload
        value = RuntimeArtifactTypeStrategy.for_artifact_type(
            spec.artifact_type
        ).runtime_input_value(self.artifact_request(edge))
        if spec.artifact_type is not ImageArtifactType:
            return value
        if not spec.stack_broadcast_sources():
            return value
        (broadcast_source_ref,) = spec.stack_broadcast_sources()
        broadcast_source = self.stack_broadcast_source_value(broadcast_source_ref)
        slice_count = RuntimeSliceProjection.slice_count_from_values((value,))
        if slice_count is None:
            return value
        runtime_projection = RuntimePlaneAxisValueProjection.from_projector(
            self.adapter,
            RuntimePlaneAxis.RUNTIME_SLICE,
            (),
        )
        if (
            runtime_projection is not None
            and runtime_projection.plane_index is not None
            and runtime_projection.axis_size == slice_count
        ):
            return cast(
                RuntimeCallableArgument,
                RuntimeSliceProjection.value_for_slice(value, runtime_projection),
            )
        source_slice_count = RuntimeSliceProjection.slice_count_from_values(
            (broadcast_source,)
        )
        if source_slice_count == slice_count:
            return value
        if source_slice_count is None and slice_count == 1:
            return cast(
                RuntimeCallableArgument,
                RuntimeSliceProjection.value_for_singleton_slice(
                    value,
                    source_description=(
                        f"{self.module_name} stack-broadcast input {spec.ref()!r}"
                    ),
                ),
            )
        identity_policy = SourceImageSetIdentityPolicy.from_source_bindings(
            self.adapter.request.source_binding_plan
        )
        image_axis = SourcePayloadPlaneIdentitySequence(
            broadcast_source,
            identity_policy,
        ).runtime_axis_identities()
        value_axis = SourcePayloadPlaneIdentitySequence(
            value,
            identity_policy,
        ).runtime_axis_identities()
        selected_indices = SourcePlaneIdentitySequenceAlignment(
            image_axis,
            value_axis,
        ).target_indexes_for_image_planes()
        if selected_indices is None or len(selected_indices) != 1:
            raise ValueError(
                f"{self.module_name} stack-broadcast input {spec.ref()!r} cannot "
                f"align to its declared source {broadcast_source_ref!r}."
            )
        return cast(
            RuntimeCallableArgument,
            RuntimeSliceProjection.value_for_slice(
                value,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE,
                    plane_index=selected_indices[0],
                    axis_size=slice_count,
                ),
            ),
        )

    def runtime_value_for_spec(
        self,
        spec: ArtifactSpec,
    ) -> RuntimeCallableArgument:
        """Load a declaration whose occurrences share one runtime authority."""

        return self.runtime_value(self.input_edge_for_spec(spec))

    def bind_parameters(self) -> dict[str, RuntimeCallableArgument]:
        """Bind compiled input edges to exact callable parameters."""

        signature = inspect.signature(self.func, eval_str=True)
        edges_by_parameter: dict[str, list[InvocationArtifactInputEdgePlan]] = {}
        for edge in self.parameter_input_edges:
            parameter_name = edge.spec.parameter_name
            if parameter_name is None:
                raise RuntimeError("Parameter input edge lost its compiled name.")
            if parameter_name not in signature.parameters:
                raise ValueError(
                    f"{self.module_name} compiled input edge targets absent callable "
                    f"parameter {parameter_name!r}."
                )
            edges_by_parameter.setdefault(parameter_name, []).append(edge)
        bound: dict[str, RuntimeCallableArgument] = {}
        for parameter_name, edges in edges_by_parameter.items():
            parameter = signature.parameters[parameter_name]
            values = tuple(
                self.runtime_value(edge, parameter_name=parameter_name)
                for edge in edges
            )
            accepts_sequence = get_origin(parameter.annotation) in (
                Sequence,
                tuple,
                list,
            )
            if accepts_sequence:
                bound[parameter_name] = values
            elif len(values) == 1:
                bound[parameter_name] = values[0]
            elif len(values) > 1:
                if any(
                    edge.spec.artifact_type is not ImageArtifactType for edge in edges
                ):
                    raise TypeError(
                        f"{self.module_name} scalar runtime input "
                        f"{parameter_name!r} can align repeated inputs "
                        "only for image artifacts."
                    )
                bound[parameter_name] = AlignedImageStack(values)
            else:
                raise ValueError(
                    f"{self.module_name} compiled parameter {parameter_name!r} "
                    "has no artifact input edges."
                )
        return bound
