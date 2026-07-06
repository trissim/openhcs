"""Output-record request authority for CellProfiler runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ImageArtifactType,
    GroupLineageSourceRelation,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.runtime_semantics import FieldSpec, ObjectLabelDomainScope
from openhcs.core.runtime_values import (
    SourceImageObjectLabelDomainRequest,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.runtime.current_image_context import (
    CellProfilerOptionalCurrentImageContext,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerMeasurementImage,
)
from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
    CellProfilerMeasurementFieldSchema,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
    MeasurementRowsInput,
)
from openhcs.interop.cellprofiler.runtime.relationship_endpoints import (
    RelationshipEndpointResolver,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
    from openhcs.interop.cellprofiler.runtime.module_execution import (
        CellProfilerModuleRuntimePlan,
    )


@dataclass(frozen=True, slots=True)
class CellProfilerOutputRecordContext(
    CellProfilerOptionalCurrentImageContext,
):
    """Shared invocation context for one declared CellProfiler artifact output."""

    runtime_plan: CellProfilerModuleRuntimePlan
    adapter: CellProfilerRuntimeAdapter
    spec: ArtifactSpec
    output_value: CellProfilerRuntimeValue
    output_values: CellProfilerKwargs
    source: CellProfilerImageRequest | CellProfilerMeasurementImage
    func: CellProfilerFunction
    call_kwargs: CellProfilerKwargs
    function_name: str = ""

    @property
    def contract(self) -> ModuleArtifactContract:
        return self.runtime_plan.contract

    @property
    def module_name(self) -> str:
        return self.contract.module_name

    @property
    def declared_input_specs(self) -> tuple[ArtifactSpec, ...]:
        return self.contract.declared_input_specs()

    @property
    def outputs(self) -> tuple[ArtifactSpec, ...]:
        return self.contract.outputs

    @property
    def declared_outputs(self) -> tuple[ArtifactSpec, ...]:
        return self.contract.declared_outputs

    @property
    def runtime_image_names(self) -> tuple[str, ...]:
        return self.contract.runtime_input_names(ImageArtifactType)

    @property
    def runtime_image_name_set(self) -> frozenset[str]:
        return self.contract.runtime_input_name_set(ImageArtifactType)

    @property
    def external_source_object_names(self) -> tuple[str, ...]:
        return self.contract.external_input_names(ObjectLabelsArtifactType)

    def fields_for_rows(self, rows: MeasurementRowsInput) -> tuple[FieldSpec, ...]:
        """Return measurement field schema for rows emitted by this context."""
        return CellProfilerMeasurementFieldSchema.for_record(
            self.spec,
            rows,
            self.func,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerOutputRecordRequest(CellProfilerOutputRecordContext):
    """Inputs and semantic authorities for recording one CellProfiler output."""

    def input_image_source_payload(
        self,
        spec: ArtifactSpec,
    ) -> CellProfilerRuntimeValue | None:
        """Resolve one declared image input into a metadata-bearing payload."""
        if spec.name in self.runtime_image_name_set:
            return self.runtime_input_image_payload(spec)
        return self.external_input_image_payload(spec)

    def primary_image_output_source_payload(
        self,
    ) -> CellProfilerRuntimeValue | None:
        """Resolve the declared primary-image source for this image output."""
        relation_payload = self.group_lineage_source_payload()
        if relation_payload is not None:
            return relation_payload
        ordinal_payload = self.ordinal_primary_image_output_source_payload()
        if ordinal_payload is not None:
            return ordinal_payload
        unique_payload = self.unique_primary_image_source_payload()
        if unique_payload is not None:
            return unique_payload
        return self.source.payload

    def group_lineage_source_payload(self) -> CellProfilerRuntimeValue | None:
        """Resolve source payload from this output's declared group-lineage source."""
        source_specs = tuple(
            self.contract.declared_input_collection().by_ref(relation.source)
            for relation in self.spec.relations
            if (
                isinstance(relation, GroupLineageSourceRelation)
                and relation.source.plan_type is ArtifactInputPlan
                and relation.source.artifact_type is ImageArtifactType
            )
        )
        resolved_specs = tuple(spec for spec in source_specs if spec is not None)
        if len(resolved_specs) != 1:
            return None
        source_spec = resolved_specs[0]
        invocation_payload = self.invocation_primary_image_source_payload(source_spec)
        if invocation_payload is not None:
            return invocation_payload
        return self.input_image_source_payload(source_spec)

    def ordinal_primary_image_output_source_payload(
        self,
    ) -> CellProfilerRuntimeValue | None:
        """Map image output ordinal to primary-image input ordinal when declared."""
        output_index = self.image_output_index()
        if output_index is None:
            return None
        if len(self.runtime_plan.primary_image_inputs) != len(self.image_outputs()):
            return None
        primary_input = self.runtime_plan.primary_image_inputs[output_index]
        invocation_payload = self.invocation_primary_image_source_payload(primary_input)
        if invocation_payload is not None:
            return invocation_payload
        return self.input_image_source_payload(primary_input)

    def unique_primary_image_source_payload(
        self,
    ) -> CellProfilerRuntimeValue | None:
        """Resolve the unique declared primary image input, when one exists."""
        if len(self.runtime_plan.primary_image_inputs) != 1:
            return None
        primary_input = self.runtime_plan.primary_image_inputs[0]
        invocation_payload = self.invocation_primary_image_source_payload(primary_input)
        if invocation_payload is not None:
            return invocation_payload
        input_payload = self.input_image_source_payload(primary_input)
        if input_payload is None:
            return None
        return input_payload

    def invocation_primary_image_source_payload(
        self,
        spec: ArtifactSpec,
    ) -> CellProfilerRuntimeValue | None:
        """Return invocation payload when it already owns the primary input source."""
        payload = self.source.source_payload_for_name(spec.name)
        if payload is None:
            return None
        if image_payload_metadata(payload).has_values:
            return payload
        return None

    def runtime_input_image_payload(
        self,
        spec: ArtifactSpec,
    ) -> CellProfilerRuntimeValue | None:
        """Return runtime image input data with the correct current-image scope."""
        runtime_current_image = (
            self.runtime_plan.primary_image_input_policy.runtime_image_current_image(
                self.module_name,
                spec,
                self.current_image,
            )
        )
        return self.adapter.get_image(
            spec.name,
            current_image=runtime_current_image,
        ).data

    def external_input_image_payload(
        self,
        spec: ArtifactSpec,
    ) -> CellProfilerRuntimeValue | None:
        """Return source image data for a declared external image input."""
        if self.current_image is None:
            return self.source.payload
        return self.adapter.resolve_source_image(
            spec.name,
            self.current_image,
        )

    def image_outputs(self) -> tuple[ArtifactSpec, ...]:
        """Return declared image outputs in contract order."""
        return self.contract.output_collection().of_artifact_type(ImageArtifactType)

    def image_output_index(self) -> int | None:
        """Return this output's declared image-output ordinal, when unique."""
        matches = tuple(
            index
            for index, spec in enumerate(self.image_outputs())
            if spec.name == self.spec.name
        )
        if len(matches) != 1:
            return None
        return matches[0]

    def metadata_bearing_primary_image_source_payload(
        self,
    ) -> CellProfilerRuntimeValue | None:
        """Resolve the unique primary image input when it carries metadata."""
        if len(self.runtime_plan.primary_image_inputs) != 1:
            return None
        primary_input = self.runtime_plan.primary_image_inputs[0]
        input_payload = self.invocation_primary_image_source_payload(primary_input)
        if input_payload is None:
            input_payload = self.input_image_source_payload(primary_input)
        if input_payload is None:
            return None
        if image_payload_metadata(input_payload).has_values:
            return input_payload
        return None

    def default_object_label_output_source_payload(self) -> CellProfilerRuntimeValue:
        """Resolve default source context for object-label outputs."""
        object_inputs = self.contract.declared_input_collection().of_artifact_type(
            ObjectLabelsArtifactType
        )
        if object_inputs and not self.runtime_plan.primary_image_inputs:
            relationship_source_spec = (
                self.relationship_derived_object_label_source_spec(object_inputs)
            )
            if relationship_source_spec is None:
                source_spec = object_inputs[0]
            else:
                source_spec = relationship_source_spec
            return self.object_label_source_payload_for_spec(source_spec)
        primary_image_source = self.unique_primary_image_source_payload()
        if primary_image_source is not None:
            return primary_image_source
        return self.object_label_source_payload_for_current_invocation()

    def input_object_label_output_source_payload(self) -> CellProfilerRuntimeValue:
        """Resolve source context from the declared object-label input."""
        object_inputs = self.contract.declared_input_collection().of_artifact_type(
            ObjectLabelsArtifactType
        )
        if not object_inputs:
            raise ValueError(
                f"{self.module_name} requested input-object source context for "
                f"object-label output {self.spec.name!r}, but declares no object "
                "inputs."
            )
        relationship_source_spec = self.relationship_derived_object_label_source_spec(
            object_inputs
        )
        if relationship_source_spec is None:
            source_spec = object_inputs[0]
        else:
            source_spec = relationship_source_spec
        return self.object_label_source_payload_for_spec(source_spec)

    def object_label_output_domain_scope(self) -> ObjectLabelDomainScope | None:
        """Return the declared object-label output domain for this invocation."""
        if self.source.execution_mode is ImagePayloadExecutionMode.FULL_STACK:
            return ObjectLabelDomainScope.PAYLOAD
        return None

    def relationship_derived_object_label_source_spec(
        self,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> ArtifactSpec | None:
        """Return the object input owning this relationship-derived output."""
        parent_specs = self.relationship_parent_specs_for_output_child()
        match parent_specs:
            case ():
                return None
            case (source_spec,):
                return source_spec
            case _:
                return self.multi_parent_relationship_output_source_spec(
                    object_inputs,
                    parent_specs,
                )

    def relationship_parent_specs_for_output_child(self) -> tuple[ArtifactSpec, ...]:
        """Return parent object specs from relationships that target this output."""
        endpoint_resolver = RelationshipEndpointResolver.for_request(self)
        relationship_outputs = self.contract.output_collection().of_artifact_type(
            RelationshipsArtifactType
        )
        return tuple(
            parent_spec
            for relationship_spec in relationship_outputs
            for parent_spec, child_spec in (
                endpoint_resolver.endpoint_specs(relationship_spec),
            )
            if child_spec.name == self.spec.name
        )

    def multi_parent_relationship_output_source_spec(
        self,
        object_inputs: tuple[ArtifactSpec, ...],
        parent_specs: tuple[ArtifactSpec, ...],
    ) -> ArtifactSpec:
        """Resolve multi-parent child outputs through declared primary input order."""
        primary_input = object_inputs[0]
        if primary_input.name in {spec.name for spec in parent_specs}:
            return primary_input
        raise ValueError(
            f"{self.module_name} object-label output '{self.spec.name}' has "
            f"multiple relationship parent candidates "
            f"{[spec.name for spec in parent_specs]}, but declared primary "
            f"object input '{primary_input.name}' is not one of them."
        )

    def object_label_source_payload_for_spec(
        self,
        spec: ArtifactSpec,
    ) -> CellProfilerRuntimeValue:
        """Resolve an object input spec to the payload that carries its source."""
        if spec.name in self.external_source_object_names:
            current_image = self.required_current_image(
                f"External object input {spec.name!r} source-binding resolution"
            )
            return self.adapter.resolve_source_objects(
                spec.name,
                current_image,
            )
        return self.adapter.get_objects(
            spec.name,
            current_image=self.current_image,
        )

    def object_label_source_payload_for_current_invocation(
        self,
    ) -> CellProfilerRuntimeValue:
        """Return metadata-bearing source context for array-lowered invocations."""
        source_payload = self.source.payload
        current_image = self.current_image
        match current_image:
            case None:
                current_explains_planes = False
                current_has_metadata = False
            case _:
                current_explains_planes = self.payload_explains_output_label_planes(
                    current_image
                )
                current_has_metadata = image_payload_metadata(current_image).has_values
        match (
            self.payload_explains_output_label_planes(source_payload),
            current_explains_planes,
            current_image is None,
            image_payload_metadata(source_payload).has_values,
            current_has_metadata,
        ):
            case (True, _, _, _, _):
                return source_payload
            case (_, True, _, _, _):
                return current_image
            case (_, _, True, _, _):
                return source_payload
            case (_, _, _, True, _):
                return source_payload
            case (_, _, _, _, True):
                return current_image
            case _:
                return source_payload

    def payload_explains_output_label_planes(
        self,
        payload: CellProfilerRuntimeValue,
    ) -> bool:
        """Return whether a source payload defines the output label plane domain."""
        return (
            SourceImageObjectLabelDomainRequest(
                image=payload,
                labels=self.output_value,
            ).plane_semantics()
            is not None
        )

    def single_output_object_name(self) -> str:
        """Return the unique object-label output owned by this record request."""
        object_outputs = self.contract.output_collection().of_artifact_type(
            ObjectLabelsArtifactType
        )
        if len(object_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} threshold measurement semantics "
                f"require exactly one object-label output, got "
                f"{[spec.name for spec in object_outputs]}."
            )
        return object_outputs[0].name
