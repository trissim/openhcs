"""Output-record request authority for CellProfiler runtime artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecRef,
    ObjectLabelsArtifactType,
)
from openhcs.core.runtime_object_label_domains import ObjectLabelDomainScope
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.source_plane_alignment import (
    SourcePayloadPlaneIdentitySequence,
    SourcePlaneIdentitySequenceAlignment,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import ImagePayloadMetadata, image_payload_metadata
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import InvocationArtifactInputEdgePlan
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerMeasurementImage,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeArtifactInputRequest,
    RuntimeArtifactTypeStrategy,
    RuntimeInputBindingRequest,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument, RuntimeCallableKwargs

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter


@dataclass(frozen=True, slots=True)
class CellProfilerOutputRecordRequest:
    """Inputs and semantic authorities for recording one CellProfiler output."""

    callable_contract: CallableContract
    active_input_edges: tuple[InvocationArtifactInputEdgePlan, ...]
    adapter: CellProfilerRuntimeAdapter
    spec: ArtifactSpec
    output_plan: ArtifactOutputPlan
    output_value: RuntimeCallableArgument
    source: CellProfilerImageRequest | CellProfilerMeasurementImage
    call_kwargs: RuntimeCallableKwargs
    current_image: RuntimeArrayData
    declared_only_outputs: Mapping[
        ArtifactSpecRef,
        RuntimeCallableArgument,
    ] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.output_plan.ref() != self.spec.ref():
            raise ValueError(
                f"Output plan {self.output_plan.ref()!r} does not match active "
                f"output {self.spec.ref()!r}."
            )
        declared_inputs = self.callable_contract.artifact_inputs
        previous_input_index = -1
        for edge in self.active_input_edges:
            input_index = edge.key.input_index
            if input_index >= len(declared_inputs):
                raise ValueError(
                    f"Callable {self.callable_contract.function_name!r} input edge "
                    f"{edge.key!r} is outside its declared occurrence range "
                    f"[0, {len(declared_inputs)})."
                )
            if input_index <= previous_input_index:
                raise ValueError(
                    f"Callable {self.callable_contract.function_name!r} input edge "
                    f"{edge.key!r} does not preserve strictly increasing declared "
                    f"occurrence order after index {previous_input_index}."
                )
            declared = declared_inputs[input_index]
            if (
                edge.spec != declared
                or edge.spec.parameter_name != declared.parameter_name
            ):
                raise ValueError(
                    f"Callable {self.callable_contract.function_name!r} input edge "
                    f"{edge.key!r} does not match its exact declared occurrence "
                    f"{declared.ref()!r}."
                )
            previous_input_index = input_index

    def artifact_output_value(
        self,
        spec: ArtifactSpec,
    ) -> RuntimeCallableArgument:
        """Return one output from its single declared runtime authority."""

        ref = spec.ref()
        output_plan = self.adapter.request.artifact_output_plan(ref)
        runtime_adapter = self.callable_contract.runtime_adapter
        recorded = bool(
            output_plan is not None
            and runtime_adapter is not None
            and runtime_adapter.manages_artifact_outputs
        )
        transient = ref in self.declared_only_outputs
        match recorded, transient:
            case True, False:
                return self.adapter.artifact_output_value(output_plan)
            case False, True:
                return self.declared_only_outputs[ref]
            case True, True:
                raise RuntimeError(
                    f"Callable {self.callable_contract.function_name!r} output "
                    f"{ref!r} has overlapping runtime "
                    "authorities."
                )
            case False, False:
                raise RuntimeError(
                    f"Callable {self.callable_contract.function_name!r} output "
                    f"{ref!r} is neither adapter-recorded "
                    "nor present in the current declared-only return."
                )

    def artifact_value(
        self,
        spec: ArtifactSpec,
    ) -> RuntimeCallableArgument:
        """Return one artifact from its exact declared input/output role."""

        if spec.require_plan_type() is ArtifactInputPlan:
            return self.artifact_input_value(self.exact_input_edge(spec))
        declared_output = self.callable_contract.artifact_outputs.by_ref(spec.ref())
        if declared_output is not None:
            if declared_output != spec:
                raise ValueError(
                    f"Callable {self.callable_contract.function_name!r} declared "
                    f"output {spec.ref()!r} differs from its requested declaration."
                )
            return self.artifact_output_value(declared_output)
        raise ValueError(
            f"Callable {self.callable_contract.function_name!r} artifact "
            f"{spec.ref()!r} is not declared by this compiled invocation."
        )

    def exact_input_edge(
        self,
        spec: ArtifactSpec,
    ) -> InvocationArtifactInputEdgePlan:
        """Return the compiled edge at this declaration's exact occurrence."""

        resolved: InvocationArtifactInputEdgePlan | None = None
        for edge in self.active_input_edges:
            if self.callable_contract.artifact_inputs[edge.key.input_index] is spec:
                if resolved is not None:
                    raise RuntimeError(
                        f"Callable {self.callable_contract.function_name!r} declaration "
                        f"{spec.ref()!r} has multiple exact compiled input edges."
                    )
                resolved = edge
        if resolved is not None:
            return resolved
        raise RuntimeError(
            f"Callable {self.callable_contract.function_name!r} declaration "
            f"{spec.ref()!r} has no exact compiled input edge."
        )

    def artifact_input_value(
        self,
        edge: InvocationArtifactInputEdgePlan,
    ) -> RuntimeCallableArgument:
        """Return one exact compiled input occurrence in invocation scope."""

        return RuntimeInputBindingRequest(
            adapter=self.adapter,
            kwargs=self.call_kwargs,
            current_image=self.current_image,
        ).runtime_value(
            edge,
            parameter_name=edge.spec.parameter_name,
        )

    def measurement_source_metadata(
        self,
        specs: tuple[ArtifactSpec, ...],
    ) -> ImagePayloadMetadata:
        """Return the contract-ordered image-set axis of exact artifacts."""

        if not specs:
            raise ValueError(
                "Measurement source context requires declared artifacts."
            )
        artifact_values = tuple(self.artifact_value(spec) for spec in specs)
        metadata = tuple(image_payload_metadata(value) for value in artifact_values)
        source_group_component = self.output_plan.group_component
        identity_policy = SourceImageSetIdentityPolicy(
            frozenset(
                () if source_group_component is None else (source_group_component,)
            )
        )
        image_set_axes = tuple(
            SourcePayloadPlaneIdentitySequence(
                value,
                identity_policy,
            ).runtime_axis_identities()
            for value in artifact_values
        )
        unaligned_indexes = SourcePlaneIdentitySequenceAlignment.unaligned_axis_indexes(
            image_set_axes
        )
        unaligned_specs = tuple(specs[index].ref() for index in unaligned_indexes)
        if unaligned_specs:
            raise ValueError(
                f"Callable {self.callable_contract.function_name!r} measurement "
                "artifacts do not share one "
                "source image-set axis: "
                f"reference={specs[0].ref()!r}; unaligned={unaligned_specs!r}."
            )
        return metadata[0]

    def artifact_source_payload(
        self,
        edge: InvocationArtifactInputEdgePlan,
    ) -> RuntimeCallableArgument:
        """Resolve one declared input in the callable invocation's exact scope."""

        spec = edge.spec
        binding_request = RuntimeInputBindingRequest(
            adapter=self.adapter,
            kwargs=self.call_kwargs,
            current_image=self.current_image,
        )
        payload = RuntimeArtifactTypeStrategy.for_artifact_type(
            spec.artifact_type
        ).source_image_payload(
            RuntimeArtifactInputRequest(
                spec=spec,
                value=binding_request.runtime_value(
                    edge,
                    parameter_name=spec.parameter_name,
                ),
                image_payload_consumption=(
                    self.callable_contract.image_payload_consumption
                ),
            )
        )
        if payload is None:
            raise TypeError(
                f"Callable {self.callable_contract.function_name!r} input "
                f"{spec.ref()!r} does not carry source "
                "image context."
            )
        return payload

    def declared_source_payload(self) -> RuntimeCallableArgument:
        """Resolve this output's exact compiled runtime-context source."""

        source_ref = self.output_plan.source_context_source()
        if source_ref is None:
            raise RuntimeError(
                f"Callable {self.callable_contract.function_name!r} output "
                f"{self.spec.ref()!r} has no declared "
                "runtime-context source."
            )
        return self.artifact_source_payload(
            self.adapter.request.require_artifact_input_edge(source_ref)
        )

    def materialization_source_metadata(self) -> ImagePayloadMetadata | None:
        """Return independent filename-source metadata declared by this output."""

        source_ref = self.output_plan.materialization_source()
        if source_ref is None or source_ref == self.output_plan.source_context_source():
            return None
        return image_payload_metadata(
            self.artifact_source_payload(
                self.adapter.request.require_artifact_input_edge(source_ref)
            )
        )

    def object_label_output_domain_scope(self) -> ObjectLabelDomainScope | None:
        """Return the declared object-label output domain for this invocation."""
        if (
            self.spec.source_context_sources()
            and not self.spec.preserves_source_stack_scope()
        ):
            return ObjectLabelDomainScope.PAYLOAD
        return None

    def single_output_object_name(self) -> str:
        """Return the unique object-label output owned by this record request."""
        object_outputs = self.callable_contract.artifact_outputs.of_artifact_type(
            ObjectLabelsArtifactType
        )
        if len(object_outputs) != 1:
            raise NotImplementedError(
                f"Callable {self.callable_contract.function_name!r} threshold "
                "measurement semantics "
                f"require exactly one object-label output, got "
                f"{[spec.name for spec in object_outputs]}."
            )
        return object_outputs[0].name
