"""Typed runtime adapter injection contracts for callable execution."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from python_introspect import add_parameter_exclusions

from openhcs.constants.constants import Backend, VariableComponents
from openhcs.core.aligned_image_payload import (
    ImagePayloadExecutionMode,
    stack_image_payloads,
)
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpecCollection,
    ArtifactSpecRef,
)
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.function_patterns import (
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
)
from openhcs.core.runtime_image_loading import ImagePayloadSourceMetadataContext
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadataCompositionMode,
    image_payload_metadata,
)
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingRuntimeContext,
)
from openhcs.core.source_binding_selection import (
    SourceBindingMatchedImageSet,
    SourcePatternResolutionContext,
)
from openhcs.core.source_image_provenance import (
    SourceImageIdentity,
    SourceImageProvenance,
)
from openhcs.core.source_image_semantics import apply_source_binding_payload
from openhcs.core.source_load_plan import SourceLoadPlan
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
    RuntimePlaneProjection,
)
from openhcs.core.source_workspace_projection import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjectionAuthority,
)

if TYPE_CHECKING:
    from openhcs.core.callable_contract import CallableContract
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.steps.function_runtime import FunctionRuntimeScope


class RuntimeAdapterValue(Protocol):
    """Nominal protocol for callable-owned runtime adapter instances."""


_F = TypeVar("_F", bound=Callable[..., Any])
PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeImageExecutionContext:
    """Source provenance and execution mode for an image-like invocation."""

    source_image_name: str | None
    execution_mode: ImagePayloadExecutionMode = ImagePayloadExecutionMode.NATURAL
    plane_projection: RuntimePlaneAxisValueProjection | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeImageRequest(RuntimeImageExecutionContext, Generic[PayloadT]):
    """Resolved image payload and source metadata for one runtime invocation."""

    image_count: int
    payload: PayloadT


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeFunctionInvocationRequest(
    RuntimeImageExecutionContext,
    Generic[PayloadT],
):
    """Resolved callable inputs for one runtime function invocation."""

    image_count: int
    image: PayloadT
    kwargs: Mapping[str, object]


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeAdapterRequest:
    """Runtime data needed to build an invocation-scoped adapter."""

    context: "ProcessingContext"
    callable_contract: "CallableContract | None" = None
    source_payload: object | None = None
    artifact_inputs: Mapping[
        "InvocationArtifactInputProjectionKey",
        "InvocationArtifactInputEdgePlan",
    ] = field(
        default_factory=dict
    )
    artifact_outputs: Mapping[ArtifactSpecRef, ArtifactOutputPlan] = field(
        default_factory=dict
    )
    source_binding_plan: CompiledSourceBindingPlan = CompiledSourceBindingPlan.empty()
    source_binding_context: SourceBindingRuntimeContext = field(
        default_factory=SourceBindingRuntimeContext.empty
    )
    group_key: str | None = None
    axis_scope: RuntimeExecutionAxisScope
    plane_projection: RuntimePlaneProjection = field(
        default_factory=RuntimePlaneProjection.stack
    )
    variable_components: tuple[VariableComponents, ...] = ()
    source_load_plan: SourceLoadPlan = field(default_factory=SourceLoadPlan)

    def __post_init__(self) -> None:
        for key, edge in self.artifact_inputs.items():
            if not isinstance(key, InvocationArtifactInputProjectionKey):
                raise TypeError(
                    "Runtime adapter input maps require "
                    "InvocationArtifactInputProjectionKey keys, got "
                    f"{type(key).__name__}."
                )
            if not isinstance(edge, InvocationArtifactInputEdgePlan):
                raise TypeError(
                    "Runtime adapter input maps require "
                    "InvocationArtifactInputEdgePlan values, got "
                    f"{type(edge).__name__} for {key!r}."
                )
            if key != edge.key:
                raise ValueError(
                    f"Runtime adapter input key {key!r} conflicts with compiled "
                    f"edge key {edge.key!r}."
                )
        ArtifactOutputPlan.require_exact_map(
            self.artifact_outputs,
            boundary="Runtime adapter output",
        )

    def source_binding_for_artifact_ref(
        self,
        ref: ArtifactSpecRef,
    ) -> NamedSourceBinding:
        """Return the exact source binding compiled for one artifact input."""

        binding = self.source_binding_plan.binding_for_artifact_ref(ref)
        if binding is None:
            raise ValueError(
                "Runtime adapter source artifact has no exact compiled source "
                f"binding for {ref!r}."
            )
        return binding

    def require_callable_contract(self) -> "CallableContract":
        """Return the exact compiled callable contract for this invocation."""

        from openhcs.core.callable_contract import CallableContract

        contract = self.callable_contract
        if not isinstance(contract, CallableContract):
            raise TypeError(
                "RuntimeAdapterRequest requires the compiled CallableContract "
                "for artifact role resolution."
            )
        return contract

    def artifact_output_plan(
        self,
        ref: ArtifactSpecRef,
    ) -> ArtifactOutputPlan | None:
        """Return the selected output plan for one exact artifact identity."""

        if not isinstance(ref, ArtifactSpecRef):
            raise TypeError(
                "RuntimeAdapterRequest.artifact_output_plan requires an "
                f"ArtifactSpecRef, got {type(ref).__name__}."
            )
        if ref.plan_type is not ArtifactOutputPlan:
            raise TypeError(
                "Runtime adapter output-plan lookup requires an output artifact "
                f"ref, got {ref!r}."
            )
        plan = self.artifact_outputs.get(ref)
        if plan is None:
            return None
        return plan

    def require_artifact_output_plan(
        self,
        ref: ArtifactSpecRef,
    ) -> ArtifactOutputPlan:
        """Return one exact selected output plan or fail loudly."""

        plan = self.artifact_output_plan(ref)
        if plan is None:
            raise RuntimeError(
                f"Runtime adapter has no selected artifact output plan for {ref!r}."
            )
        return plan

    def selected_artifact_input_specs(self) -> ArtifactSpecCollection:
        """Return exact active input declarations in callable ABI order."""

        declared = self.require_callable_contract().artifact_inputs
        selected_occurrences = declared.select_declared_occurrences(
            edge.spec for edge in self.artifact_inputs.values()
        )
        selected_refs = selected_occurrences.ref_set()
        return ArtifactSpecCollection(
            spec for spec in declared if spec.ref() in selected_refs
        )

    def require_artifact_input_edge(
        self,
        ref: ArtifactSpecRef,
    ) -> "InvocationArtifactInputEdgePlan":
        """Return one exact runtime authority for a compiled input identity."""

        if not isinstance(ref, ArtifactSpecRef):
            raise TypeError(
                "RuntimeAdapterRequest.require_artifact_input_edge requires an "
                f"ArtifactSpecRef, got {type(ref).__name__}."
            )
        if ref.plan_type is not ArtifactInputPlan:
            raise TypeError(
                "Runtime adapter input-edge lookup requires an input artifact ref, "
                f"got {ref!r}."
            )
        matches = tuple(
            edge
            for edge in self.artifact_inputs.values()
            if edge.spec.ref() == ref
        )
        if not matches:
            raise RuntimeError(
                f"Runtime adapter has no compiled artifact input occurrence for "
                f"{ref!r}."
            )
        first = matches[0]
        if any(
            edge.spec != first.spec
            or (edge.storage_plan, edge.projection, edge.consumes_main_flow)
            != (first.storage_plan, first.projection, first.consumes_main_flow)
            for edge in matches[1:]
        ):
            raise ValueError(
                f"Compiled artifact input occurrences for {ref!r} resolve to "
                "different runtime authorities."
            )
        return first

    def source_artifact_payload(self, ref: ArtifactSpecRef) -> object:
        """Resolve one source-bound artifact through workspace matching and VFS."""

        binding = self.source_binding_for_artifact_ref(ref)
        source_payload = self.source_payload
        source_provenance = (
            None
            if source_payload is None
            else image_payload_metadata(source_payload).source_provenance
        )
        if source_provenance is not None and not source_provenance.has_values:
            raise ValueError(
                f"Source-bound artifact {ref!r} requires main-flow source provenance."
            )

        cache = self.context.runtime_source_workspace_projection_cache
        projection = VirtualWorkspaceSourceProjectionAuthority.from_context(
            self.context,
            cache=cache,
        ).projection_if_available()
        if projection is None:
            raise ValueError(
                f"Source-bound artifact {ref!r} requires a virtual-workspace "
                "source projection."
            )
        projection = cache.filtered_by_axis(
            projection,
            axis_id=self.axis_scope.axis_id,
        )
        source_context = SourcePatternResolutionContext.from_projection(
            parser=self.context.microscope_handler.parser,
            projection=projection,
            metadata_rules=self.source_binding_plan.metadata_rules,
        )
        matched_set = SourceBindingMatchedImageSet.from_plan(
            bindings=self.source_binding_plan.binding_declarations,
            match_plan=self.source_binding_plan.match_plan,
            source_context=source_context,
            identity_policy=self.context.source_image_set_identity_policy,
        )
        source_universe = tuple(
            dict.fromkeys(
                source_path
                for declared_binding in self.source_binding_plan.binding_declarations
                for source_path in projection.files_for_projection_role(
                    declared_binding.projection_role,
                    axis_id=self.axis_scope.axis_id,
                )
            )
        )
        members = matched_set.members_for_binding(
            binding,
            anchor_provenance=(
                source_provenance
                if source_provenance is not None
                else SourceImageProvenance()
            ),
            source_universe=source_universe,
        )
        if not members:
            raise ValueError(
                f"Source-bound artifact {ref!r} resolved no workspace members."
            )

        payloads = self.context.filemanager.load_batch(
            list(members),
            Backend.VIRTUAL_WORKSPACE.value,
        )
        if len(payloads) != len(members):
            raise ValueError(
                f"Source-bound artifact {ref!r} loaded {len(payloads)} payloads "
                f"for {len(members)} workspace members."
            )
        projected_payloads = []
        for member, payload in zip(members, payloads, strict=True):
            lookup = VirtualWorkspacePathLookup.from_paths(member, member)
            source_projection = projection.require_source_projection_for(lookup)
            if not source_projection.matches_binding(binding):
                raise ValueError(
                    f"Workspace projection for {member!r} does not match compiled "
                    f"source artifact {ref!r}."
                )
            projected = projection.project_payload(lookup, payload)
            projected_payloads.append(
                apply_source_binding_payload(
                    projected,
                    binding,
                    ImagePayloadSourceMetadataContext(
                        SourceImageIdentity(
                            member,
                            projection.source_metadata_for(lookup),
                        ),
                        source_projection.ref.backend,
                        self.context.filemanager,
                        source_projection.ref.backend_address,
                    ),
                )
            )
        return stack_image_payloads(
            projected_payloads,
            metadata_mode=ImagePayloadMetadataCompositionMode.for_plane_axis(
                RuntimePlaneAxis.RUNTIME_SLICE
            ),
        )

    @classmethod
    def from_source_context(
        cls,
        *,
        context: "ProcessingContext",
        source_payload: object | None,
        artifact_inputs: Mapping[
            "InvocationArtifactInputProjectionKey",
            "InvocationArtifactInputEdgePlan",
        ],
        artifact_outputs: Mapping[ArtifactSpecRef, ArtifactOutputPlan],
        source_binding_plan: CompiledSourceBindingPlan,
        source_binding_context: SourceBindingRuntimeContext,
        group_key: str | None = None,
        axis_scope: RuntimeExecutionAxisScope | None = None,
        plane_projection: RuntimePlaneProjection | None = None,
        variable_components: tuple[VariableComponents, ...] | None = None,
        source_load_plan: SourceLoadPlan | None = None,
        callable_contract: "CallableContract | None" = None,
    ) -> "RuntimeAdapterRequest":
        """Project a source-binding runtime context into an adapter request."""
        return cls(
            context=context,
            callable_contract=callable_contract,
            source_payload=source_payload,
            artifact_inputs=artifact_inputs,
            artifact_outputs=artifact_outputs,
            source_binding_plan=source_binding_plan,
            source_binding_context=source_binding_context,
            group_key=group_key,
            axis_scope=(
                axis_scope
                if axis_scope is not None
                else RuntimeExecutionAxisScope.from_context(context)
            ),
            plane_projection=(
                plane_projection
                if plane_projection is not None
                else RuntimePlaneProjection.stack()
            ),
            variable_components=(
                tuple(variable_components) if variable_components is not None else ()
            ),
            source_load_plan=(
                source_load_plan if source_load_plan is not None else SourceLoadPlan()
            ),
        )

    @classmethod
    def from_runtime_scope(
        cls,
        *,
        runtime_scope: "FunctionRuntimeScope",
        artifact_inputs: Mapping[
            "InvocationArtifactInputProjectionKey",
            "InvocationArtifactInputEdgePlan",
        ],
        artifact_outputs: Mapping[ArtifactSpecRef, ArtifactOutputPlan],
        group_key: str | None,
        plane_projection: RuntimePlaneProjection,
        source_payload: object,
        callable_contract: "CallableContract | None" = None,
    ) -> "RuntimeAdapterRequest":
        """Project an invocation runtime scope into an adapter request."""
        return cls.from_source_context(
            context=runtime_scope.context,
            callable_contract=callable_contract,
            source_payload=source_payload,
            artifact_inputs=artifact_inputs,
            artifact_outputs=artifact_outputs,
            source_binding_plan=runtime_scope.source_binding_plan,
            source_binding_context=runtime_scope.source_binding_context,
            group_key=group_key,
            axis_scope=runtime_scope.axis_scope,
            plane_projection=plane_projection,
            variable_components=tuple(runtime_scope.execution_plan.variable_components),
            source_load_plan=runtime_scope.execution_plan.source_load_plan,
        )


RuntimeAdapterFactory = Callable[[RuntimeAdapterRequest], RuntimeAdapterValue]
RuntimeCallableFactory = Callable[
    [Callable[..., object], "CallableContract"],
    Callable[..., object],
]


@dataclass(frozen=True, slots=True)
class RuntimeAdapterSpec:
    """Callable-owned runtime adapter injection contract."""

    parameter_name: str
    factory: RuntimeAdapterFactory
    manages_artifact_inputs: bool = False
    manages_artifact_outputs: bool = False
    runtime_callable_factory: RuntimeCallableFactory | None = None

    def __post_init__(self) -> None:
        if not self.parameter_name:
            raise ValueError("RuntimeAdapterSpec.parameter_name cannot be empty.")
        if not callable(self.factory):
            raise TypeError("RuntimeAdapterSpec.factory must be callable.")
        if self.runtime_callable_factory is not None and not callable(
            self.runtime_callable_factory
        ):
            raise TypeError(
                "RuntimeAdapterSpec.runtime_callable_factory must be callable or None."
            )

    def require_parameter_name(self) -> str:
        """Return the callable ABI name for this adapter injection."""
        return self.parameter_name

    def validate_callable_signature(self, func: Callable[..., Any]) -> None:
        """Ensure the declared adapter parameter exists at declaration time."""
        parameter_name = self.require_parameter_name()
        if parameter_name in inspect.signature(func).parameters:
            return
        raise TypeError(
            "Runtime adapter declaration requires callable signature parameter "
            f"{parameter_name!r}."
        )

    def executable_callable(
        self,
        resolved_callable: Callable[..., object],
        contract: "CallableContract",
    ) -> Callable[..., object]:
        """Build the executable callable from one ordinarily resolved callable."""
        factory = self.runtime_callable_factory
        if factory is None:
            return resolved_callable
        return factory(resolved_callable, contract)


def runtime_adapter(
    parameter_name: str,
    factory: RuntimeAdapterFactory,
    *,
    manages_artifact_inputs: bool = False,
    manages_artifact_outputs: bool = False,
    runtime_callable_factory: RuntimeCallableFactory | None = None,
) -> Callable[[_F], _F]:
    """Declare that a callable needs an invocation-scoped runtime adapter."""
    spec = RuntimeAdapterSpec(
        parameter_name=parameter_name,
        factory=factory,
        manages_artifact_inputs=manages_artifact_inputs,
        manages_artifact_outputs=manages_artifact_outputs,
        runtime_callable_factory=runtime_callable_factory,
    )

    def decorator(func: _F) -> _F:
        spec.validate_callable_signature(func)
        namespace = vars(func)
        if not isinstance(namespace, MutableMapping):
            raise TypeError(f"{func!r} does not expose a mutable metadata namespace.")
        namespace[FunctionContractAttribute.runtime_adapter] = spec
        add_parameter_exclusions(func, parameter_name)
        return func

    return decorator


def runtime_adapter_spec_from_callable(func: Any) -> RuntimeAdapterSpec | None:
    """Return the callable's declared runtime adapter contract, if any."""
    if callable(func):
        spec = _callable_namespace_runtime_adapter(func)
        if spec is not None:
            return spec
    reference_spec = _function_reference_runtime_adapter(func)
    if reference_spec is None:
        return None
    return reference_spec


def _callable_namespace_runtime_adapter(
    func: Callable[..., Any],
) -> RuntimeAdapterSpec | None:
    """Return adapter metadata preserved on callable namespaces."""
    try:
        namespace = vars(func)
    except TypeError:
        return None
    value = namespace.get(FunctionContractAttribute.runtime_adapter)
    if value is None:
        return None
    if not isinstance(value, RuntimeAdapterSpec):
        raise TypeError(
            "Runtime adapter metadata must be a RuntimeAdapterSpec, "
            f"got {type(value).__name__}."
        )
    return value


def _function_reference_runtime_adapter(func: object) -> RuntimeAdapterSpec | None:
    from openhcs.core.function_reference import FunctionReference

    if not isinstance(func, FunctionReference):
        return None
    return func.metadata.runtime_adapter
