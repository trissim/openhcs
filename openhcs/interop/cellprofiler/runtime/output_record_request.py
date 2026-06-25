"""Output-record request authority for CellProfiler runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
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
        CellProfilerPrimaryImageInputPolicy,
    )


@dataclass(frozen=True, slots=True)
class CorrectIlluminationOriginalImageName:
    """Original-image naming rule used by CorrectIlluminationApply outputs."""

    name: str

    prefix: ClassVar[str] = "Orig"

    def is_original_source(self) -> bool:
        return self.name[: len(type(self).prefix)] == type(self).prefix

    def output_candidate_names(self) -> tuple[str, ...]:
        if self.is_original_source():
            return (self.name,)
        return (f"{type(self).prefix}{self.name}", self.name)


@dataclass(frozen=True, slots=True)
class CellProfilerOutputRecordContext(
    CellProfilerOptionalCurrentImageContext,
):
    """Shared invocation context for one declared CellProfiler artifact output."""

    contract: ModuleArtifactContract
    primary_image_input_policy: CellProfilerPrimaryImageInputPolicy
    adapter: CellProfilerRuntimeAdapter
    spec: ArtifactSpec
    output_value: CellProfilerRuntimeValue
    output_values: CellProfilerKwargs
    source: CellProfilerImageRequest | CellProfilerMeasurementImage
    func: CellProfilerFunction
    call_kwargs: CellProfilerKwargs
    function_name: str = ""

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
        return self.contract.runtime_input_names(ArtifactKind.IMAGE)

    @property
    def runtime_image_name_set(self) -> frozenset[str]:
        return self.contract.runtime_input_name_set(ArtifactKind.IMAGE)

    @property
    def external_source_object_names(self) -> tuple[str, ...]:
        return self.contract.external_input_names(ArtifactKind.OBJECT_LABELS)

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
        ordinal_payload = self.ordinal_primary_image_output_source_payload()
        if ordinal_payload is not None:
            return ordinal_payload
        unique_payload = self.unique_primary_image_source_payload()
        if unique_payload is not None:
            return unique_payload
        return self.source.payload

    def ordinal_primary_image_output_source_payload(
        self,
    ) -> CellProfilerRuntimeValue | None:
        """Map image output ordinal to primary-image input ordinal when declared."""
        output_index = self._image_output_index()
        if output_index is None:
            return None
        primary_image_inputs = self.primary_image_input_policy.primary_image_inputs(
            self.module_name,
            self.func,
            self.declared_input_specs,
        )
        if len(primary_image_inputs) != len(self._image_outputs()):
            return None
        return self.input_image_source_payload(primary_image_inputs[output_index])

    def unique_primary_image_source_payload(
        self,
    ) -> CellProfilerRuntimeValue | None:
        """Resolve the unique declared primary image input, when one exists."""
        primary_image_inputs = self.primary_image_input_policy.primary_image_inputs(
            self.module_name,
            self.func,
            self.declared_input_specs,
        )
        if len(primary_image_inputs) != 1:
            return None
        primary_input = primary_image_inputs[0]
        input_payload = self.input_image_source_payload(primary_input)
        invocation_payload = self.invocation_primary_image_source_payload(primary_input)
        if input_payload is None:
            return invocation_payload
        if invocation_payload is None:
            return input_payload
        return self.preferred_unique_primary_source_payload(
            input_payload=input_payload,
            invocation_payload=invocation_payload,
        )

    def preferred_unique_primary_source_payload(
        self,
        *,
        input_payload: CellProfilerRuntimeValue,
        invocation_payload: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        """Select source context for one declared primary-image input."""
        if (
            self.payload_explains_output_label_planes(invocation_payload)
            and not self.payload_explains_output_label_planes(input_payload)
        ):
            return invocation_payload
        return input_payload

    def invocation_primary_image_source_payload(
        self,
        spec: ArtifactSpec,
    ) -> CellProfilerRuntimeValue | None:
        """Return invocation payload when it already owns the primary input source."""
        return self.source.source_payload_for_name(spec.name)

    def runtime_input_image_payload(
        self,
        spec: ArtifactSpec,
    ) -> CellProfilerRuntimeValue | None:
        """Return runtime image input data with the correct current-image scope."""
        runtime_current_image = (
            self.primary_image_input_policy.runtime_image_current_image(
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

    def correct_illumination_apply_source_spec(self) -> ArtifactSpec | None:
        """Return the original source image input for a corrected image output."""
        original_inputs = self.original_image_inputs()
        input_specs = {spec.name: spec for spec in original_inputs}
        candidate_names = CorrectIlluminationOriginalImageName(
            self.spec.name
        ).output_candidate_names()
        for candidate_name in candidate_names:
            if candidate_name in input_specs:
                return input_specs[candidate_name]
        output_index = self._image_output_index()
        image_outputs = self._image_outputs()
        if (
            output_index is not None
            and len(original_inputs) == len(image_outputs)
        ):
            return original_inputs[output_index]
        return None

    def original_image_inputs(self) -> tuple[ArtifactSpec, ...]:
        """Return declared primary image inputs with CellProfiler Orig* semantics."""
        return tuple(
            spec
            for spec in self.primary_image_input_policy.primary_image_inputs(
                self.module_name,
                self.func,
                self.declared_input_specs,
            )
            if CorrectIlluminationOriginalImageName(spec.name).is_original_source()
        )

    def _image_outputs(self) -> tuple[ArtifactSpec, ...]:
        """Return declared image outputs in contract order."""
        return self.contract.output_collection().of_kind(ArtifactKind.IMAGE)

    def _image_output_index(self) -> int | None:
        """Return this output's declared image-output ordinal, when unique."""
        matches = tuple(
            index
            for index, spec in enumerate(self._image_outputs())
            if spec.name == self.spec.name
        )
        if len(matches) != 1:
            return None
        return matches[0]

    def object_label_output_source_payload(self) -> CellProfilerRuntimeValue:
        """Resolve source context for object-label outputs from semantic inputs."""
        object_inputs = self.contract.declared_input_collection().of_kind(
            ArtifactKind.OBJECT_LABELS
        )
        image_inputs = self.primary_image_input_policy.primary_image_inputs(
            self.module_name,
            self.func,
            self.declared_input_specs,
        )
        if object_inputs and not image_inputs:
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
        endpoint_resolver = RelationshipEndpointResolver(self)
        relationship_outputs = self.contract.output_collection().of_kind(
            ArtifactKind.RELATIONSHIPS
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
        object_outputs = self.contract.output_collection().of_kind(
            ArtifactKind.OBJECT_LABELS
        )
        if len(object_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} threshold measurement semantics "
                f"require exactly one object-label output, got "
                f"{[spec.name for spec in object_outputs]}."
            )
        return object_outputs[0].name
