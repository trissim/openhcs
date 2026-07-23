"""Nominal helpers for OpenHCS function-pattern invocation identity.

FunctionStep accepts several pattern shapes: a callable, ``(callable, kwargs)``,
a list chain, or a dict keyed by component/group. The runtime already treats
each enabled callable position as the effective execution unit; this module
gives that unit a named identity for compile-time planning and runtime lookup.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from openhcs.core.pipeline.compilation_session import CompilationPathResolver

from openhcs.core.artifacts import (
    ArtifactInputProjectionPlan,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecAccumulator,
    ArtifactSpecRef,
)
from openhcs.core.callable_contract import (
    CallableContract,
    FunctionStepExecutionScope,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    CompositeInvocationContractProvider,
    InvocationContractProvider,
    InvocationArtifactDeclarationProviderLike,
    callable_contract_artifact_declarations,
)
from openhcs.core.function_reference import FunctionReference
from pyqt_reactive.pattern_metadata import PatternScopeToken
from python_introspect import Enableable


FunctionPatternCallable: TypeAlias = Callable | FunctionReference
FunctionPatternSyntax: TypeAlias = Callable | tuple | list | dict
FunctionGroupKey: TypeAlias = Hashable
RuntimeKwargMap: TypeAlias = Mapping
RuntimeKwargItems: TypeAlias = tuple[tuple, ...]
GroupedPatternMap: TypeAlias = dict[FunctionGroupKey, Sequence]
JsonScalar: TypeAlias = str | int | float | bool | None
RuntimeComponentValue: TypeAlias = JsonScalar
DEFAULT_GROUP_KEY = "default"


@dataclass(frozen=True, slots=True)
class RuntimeParameterBinding:
    """Compile-resolved runtime parameter value for one callable invocation."""

    parameter_name: str
    value: object

    def __post_init__(self) -> None:
        if not self.parameter_name:
            raise ValueError("Runtime parameter binding name cannot be empty.")


@dataclass(frozen=True, slots=True)
class RuntimeCallableKwargPolicy:
    """Classify FunctionStep item kwargs before runtime invocation."""

    enableable_type: type[Enableable] = Enableable
    scope_token_type: type[PatternScopeToken] = PatternScopeToken

    def item_is_disabled(self, kwargs: RuntimeKwargMap) -> bool:
        """Return whether an item should be removed from the function pattern."""
        return self.enableable_type.disabled_in(kwargs)

    def invocation_items(
        self,
        kwargs: RuntimeKwargMap,
    ) -> RuntimeKwargItems:
        """Return kwargs that are passed to the user callable."""
        return tuple(
            (key, value)
            for key, value in kwargs.items()
            if not self._is_control_key(key)
        )

    def _is_control_key(self, key: object) -> bool:
        return self.enableable_type.is_parameter_key(
            key
        ) or self.scope_token_type.is_key(key)


RUNTIME_CALLABLE_KWARG_POLICY = RuntimeCallableKwargPolicy()


class RuntimeInvocationDomain(str, Enum):
    """Compiled runtime domain for one callable invocation or group."""

    SOURCE_ANCHORED = "source_anchored"
    ARTIFACT_MANAGED = "artifact_managed"

    @classmethod
    def from_invocation(
        cls,
        invocation: "CompiledFunctionInvocation",
    ) -> "RuntimeInvocationDomain":
        stored_input_refs = frozenset(
            edge.spec.ref()
            for edge in invocation.artifact_input_edges
            if edge.storage_plan is not None
        )
        group_scope_refs = invocation.contract.group_scope_inputs.ref_set()
        return (
            cls.ARTIFACT_MANAGED
            if group_scope_refs and group_scope_refs <= stored_input_refs
            else cls.SOURCE_ANCHORED
        )

    @classmethod
    def from_invocations(
        cls,
        invocations: tuple["CompiledFunctionInvocation", ...],
    ) -> "RuntimeInvocationDomain":
        if invocations and all(
            invocation.runtime_domain is cls.ARTIFACT_MANAGED
            for invocation in invocations
        ):
            return cls.ARTIFACT_MANAGED
        return cls.SOURCE_ANCHORED

    def select_lifecycle_anchors(
        self,
        anchors: Sequence,
    ) -> Sequence:
        """Return lifecycle anchors needed to execute this runtime domain."""
        if self is RuntimeInvocationDomain.ARTIFACT_MANAGED:
            return anchors[:1]
        return anchors


class MainFlowInputProjection(str, Enum):
    """Compile-resolved projection of the current main-flow payload."""

    DECLARED_SOURCE_IMAGE = "declared_source_image"
    COMPLETE_PAYLOAD = "complete_payload"


@dataclass(frozen=True)
class FunctionInvocationKey:
    """Stable identity for one callable position inside a function pattern."""

    function_name: str
    group_key: str
    position: int

    def runtime_group_key(self, component_value: RuntimeComponentValue) -> str | None:
        """Return runtime artifact group identity for this invocation."""

        if self.group_key == DEFAULT_GROUP_KEY:
            if component_value is None:
                return None
            return str(component_value)
        return self.group_key

    @classmethod
    def from_callable(
        cls, func: Callable, group_key: FunctionGroupKey, position: int
    ) -> "FunctionInvocationKey":
        return cls.from_contract(
            CallableContract.from_callable(func),
            group_key,
            position,
        )

    @classmethod
    def from_contract(
        cls, contract: CallableContract, group_key: FunctionGroupKey, position: int
    ) -> "FunctionInvocationKey":
        return cls(
            function_name=contract.function_name,
            group_key=str(group_key),
            position=position,
        )


@dataclass(frozen=True, slots=True)
class InvocationArtifactInputProjectionKey:
    """Stable identity for one invocation-to-artifact input edge."""

    invocation_key: FunctionInvocationKey
    input_index: int

    def __post_init__(self) -> None:
        if type(self.input_index) is not int or self.input_index < 0:
            raise ValueError(
                "Invocation artifact input projection input_index must be a "
                f"non-negative integer, got {self.input_index!r}."
            )

    @classmethod
    def for_input_count(
        cls,
        invocation_key: FunctionInvocationKey,
        input_count: int,
    ) -> tuple["InvocationArtifactInputProjectionKey", ...]:
        """Return exact edge identities in compiled input order."""

        if type(input_count) is not int or input_count < 0:
            raise ValueError("Compiled input count must be a non-negative integer.")
        return tuple(
            cls(invocation_key=invocation_key, input_index=input_index)
            for input_index in range(input_count)
        )


@dataclass(frozen=True, slots=True)
class InvocationArtifactInputEdgePlan:
    """Exact invocation-owned artifact input occurrence."""

    key: InvocationArtifactInputProjectionKey
    spec: ArtifactSpec
    storage_plan: ArtifactInputPlan | None = field(compare=False)
    projection: ArtifactInputProjectionPlan | None
    consumes_main_flow: bool = False
    main_flow_projection: MainFlowInputProjection | None = None

    def __post_init__(self) -> None:
        if type(self.consumes_main_flow) is not bool:
            raise TypeError(
                "Invocation artifact input edge consumes_main_flow must be bool, "
                f"got {type(self.consumes_main_flow).__name__}."
            )
        if self.consumes_main_flow and self.main_flow_projection is None:
            object.__setattr__(
                self,
                "main_flow_projection",
                MainFlowInputProjection.DECLARED_SOURCE_IMAGE,
            )
        elif not self.consumes_main_flow and self.main_flow_projection is not None:
            raise ValueError(
                "Invocation artifact input edge cannot declare a main-flow "
                "projection when it does not consume main flow."
            )
        if self.main_flow_projection is not None and not isinstance(
            self.main_flow_projection,
            MainFlowInputProjection,
        ):
            raise TypeError(
                "Invocation artifact input edge main_flow_projection must be a "
                "MainFlowInputProjection or None, got "
                f"{type(self.main_flow_projection).__name__}."
            )
        if self.consumes_main_flow and self.storage_plan is not None:
            raise ValueError(
                "Invocation artifact input edge cannot consume main flow when an "
                "exact storage plan owns the input."
            )
        if (self.storage_plan is None) != (self.projection is None):
            raise ValueError(
                "Invocation artifact input edge storage and projection must be "
                "declared together."
            )
        if self.storage_plan is None:
            return
        storage_ref = self.storage_plan.ref()
        if self.spec.ref() != storage_ref:
            raise ValueError(
                f"Invocation input edge declaration {self.spec.ref()!r} "
                f"does not match storage plan {storage_ref!r}."
            )
        self.projection.validate_storage_plan(self.storage_plan)


@dataclass(frozen=True)
class FunctionInvocation:
    """One enabled callable extracted from a FunctionStep pattern."""

    contract: CallableContract
    key: FunctionInvocationKey

    @property
    def func(self) -> FunctionPatternCallable:
        """Callable reference used by the runtime invocation."""
        return self.contract.func


@dataclass(frozen=True, slots=True)
class NormalizedFunctionItem:
    """Compiler-normalized callable item with stable invocation identity."""

    key: FunctionInvocationKey
    contract: CallableContract
    kwargs: RuntimeKwargItems = ()

    @property
    def func(self) -> FunctionPatternCallable:
        """Callable reference used by the runtime invocation."""
        return self.contract.func

    @property
    def kwargs_dict(self) -> dict:
        """Return invocation kwargs as a runtime dict."""
        return dict(self.kwargs)


@dataclass(frozen=True, slots=True)
class NormalizedFunctionGroup:
    """Compiler-normalized callable chain for one pattern group."""

    source_group_key: FunctionGroupKey
    items: tuple[NormalizedFunctionItem, ...]

    @property
    def group_key(self) -> str:
        """Return the canonical runtime group identity."""

        return str(self.source_group_key)


@dataclass(frozen=True, slots=True)
class NormalizedFunctionPattern:
    """Raw FunctionStep.func syntax lowered into typed compiler input."""

    groups: tuple[NormalizedFunctionGroup, ...]
    is_grouped: bool

    def __post_init__(self) -> None:
        source_key_by_invocation: dict[
            FunctionInvocationKey,
            FunctionGroupKey,
        ] = {}
        for group in self.groups:
            for item in group.items:
                if item.key in source_key_by_invocation:
                    prior_source_key = source_key_by_invocation[item.key]
                    raise ValueError(
                        f"Original group keys {prior_source_key!r} and "
                        f"{group.source_group_key!r} normalize to duplicate "
                        f"{item.key!r}."
                    )
                source_key_by_invocation[item.key] = group.source_group_key

    def iter_items(self) -> Iterator[NormalizedFunctionItem]:
        """Yield normalized callable items in runtime order."""
        for group in self.groups:
            yield from group.items


@dataclass(frozen=True, slots=True)
class CompiledFunctionInvocation(NormalizedFunctionItem):
    """Executable compiler output for one callable in a function pattern."""

    artifact_output_plans: tuple[ArtifactOutputPlan, ...] = ()
    artifact_input_edges: tuple[InvocationArtifactInputEdgePlan, ...] = ()
    runtime_parameter_bindings: tuple[RuntimeParameterBinding, ...] = ()

    def __post_init__(self) -> None:
        expected_edge_keys = InvocationArtifactInputProjectionKey.for_input_count(
            self.key,
            len(self.artifact_input_edges),
        )
        actual_edge_keys = tuple(edge.key for edge in self.artifact_input_edges)
        if actual_edge_keys != expected_edge_keys:
            raise ValueError(
                f"Compiled invocation {self.key!r} artifact input edge order does "
                f"not match its compiled positions; expected {expected_edge_keys!r}, "
                f"got {actual_edge_keys!r}."
            )
        if any(
            not isinstance(plan, ArtifactOutputPlan)
            for plan in self.artifact_output_plans
        ):
            raise TypeError("Compiled artifact outputs must be ArtifactOutputPlan values.")

    @property
    def input_memory_type(self) -> str | None:
        """Declared input memory type from the callable contract."""
        return self.contract.input_memory_type

    @property
    def output_memory_type(self) -> str | None:
        """Declared output memory type from the callable contract."""
        return self.contract.output_memory_type

    @property
    def kwargs_dict(self) -> dict:
        """Return user-authored callable kwargs as a runtime dict."""
        return dict(self.kwargs)

    def for_runtime_outputs(
        self,
        *,
        output_plans: Sequence[ArtifactOutputPlan],
    ) -> "CompiledFunctionInvocation":
        """Select component-scoped storage plans without changing the callable ABI."""

        return replace(
            self,
            artifact_output_plans=tuple(output_plans),
        )

    def select_inputs(
        self,
        input_plans: Mapping[ArtifactSpecRef, ArtifactInputPlan],
    ) -> dict[InvocationArtifactInputProjectionKey, InvocationArtifactInputEdgePlan]:
        """Return exact compiled input occurrences after storage validation."""
        ArtifactInputPlan.require_exact_map(
            input_plans,
            boundary=f"Compiled invocation {self.key!r} received input plan",
        )
        for edge in self.artifact_input_edges:
            if edge.storage_plan is None:
                continue
            if input_plans.get(edge.storage_plan.ref()) != edge.storage_plan:
                raise ValueError(
                    f"Compiled invocation {self.key!r} input plan "
                    f"{edge.storage_plan.ref()!r} is unavailable in this step."
                )
        return {edge.key: edge for edge in self.artifact_input_edges}

    def with_artifact_input_edges(
        self,
        edges: Sequence[InvocationArtifactInputEdgePlan],
    ) -> "CompiledFunctionInvocation":
        """Return this invocation with its exact compiled input-edge projections."""

        return replace(self, artifact_input_edges=tuple(edges))

    def select_outputs(
        self,
        output_plans: Mapping[ArtifactSpecRef, ArtifactOutputPlan],
    ) -> dict[ArtifactSpecRef, ArtifactOutputPlan]:
        """Select exact runtime projections of this invocation's output plans."""
        ArtifactOutputPlan.require_exact_map(
            output_plans,
            boundary=f"Compiled invocation {self.key!r} received output plan",
        )
        selected: dict[ArtifactSpecRef, ArtifactOutputPlan] = {}
        for compiled_plan in self.artifact_output_plans:
            runtime_plan = output_plans.get(compiled_plan.ref())
            if runtime_plan is None:
                raise ValueError(
                    f"Compiled invocation {self.key!r} output plan "
                    f"{compiled_plan.ref()!r} "
                    "is unavailable in this step."
                )
            if runtime_plan != compiled_plan:
                expected_projection = compiled_plan.for_invocation_group(
                    runtime_plan.require_single_group_key()
                )
                if runtime_plan != expected_projection:
                    raise ValueError(
                        f"Compiled invocation {self.key!r} output plan "
                        f"{compiled_plan.ref()!r} has a runtime projection that "
                        "differs from its compiled owner."
                    )
            selected[runtime_plan.ref()] = runtime_plan
        return selected

    @property
    def runtime_domain(self) -> RuntimeInvocationDomain:
        """Return the compiled runtime invocation domain."""
        return RuntimeInvocationDomain.from_invocation(self)

    @property
    def adapter_manages_artifact_inputs(self) -> bool:
        """Return whether the callable adapter loads its declared artifact inputs."""
        runtime_adapter = self.contract.runtime_adapter
        return bool(
            runtime_adapter is not None
            and runtime_adapter.manages_artifact_inputs
            and self.artifact_input_edges
        )

    @property
    def adapter_records_artifact_outputs(self) -> bool:
        """Return whether this invocation's adapter records selected outputs."""
        return bool(
            self.artifact_output_plans
            and self.contract.runtime_adapter is not None
            and self.contract.runtime_adapter.manages_artifact_outputs
        )

    @property
    def main_flow_output_plans(self) -> tuple[ArtifactOutputPlan, ...]:
        """Return compiled outputs that publish canonical image flow."""

        canonical_refs = frozenset(
            spec.ref() for spec in self.contract.canonical_return_output_specs
        )
        return tuple(
            plan for plan in self.artifact_output_plans if plan.ref() in canonical_refs
        )


@dataclass(frozen=True, slots=True)
class CompiledFunctionGroup:
    """Compiled callable chain for one function-pattern group."""

    group_key: str
    invocations: tuple[CompiledFunctionInvocation, ...]

    @property
    def runtime_domain(self) -> RuntimeInvocationDomain:
        """Return the compiled runtime domain for this callable group."""
        return RuntimeInvocationDomain.from_invocations(self.invocations)

    @property
    def main_flow_input_refs(self) -> tuple[ArtifactSpecRef, ...] | None:
        """Return exact main-flow refs, or None for an implicit image argument."""

        main_flow_refs = tuple(
            dict.fromkeys(
                edge.spec.ref()
                for invocation in self.invocations
                for edge in invocation.artifact_input_edges
                if edge.consumes_main_flow
            )
        )
        if main_flow_refs:
            return main_flow_refs
        has_implicit_image_argument = any(
            invocation.contract.accepts_implicit_main_flow_input
            for invocation in self.invocations
        )
        return None if has_implicit_image_argument else ()

    def resulting_main_flow_output_plans(self) -> tuple[ArtifactOutputPlan, ...]:
        """Return exact named output plans carried after this callable chain."""

        plans: tuple[ArtifactOutputPlan, ...] = ()
        for invocation in self.invocations:
            if invocation.main_flow_output_plans:
                plans = invocation.main_flow_output_plans
            elif not invocation.contract.preserves_input_main_flow():
                plans = ()
        return plans

    def resulting_implicit_main_flow_invocation(
        self,
    ) -> CompiledFunctionInvocation | None:
        """Return the invocation owning the final unnamed main-flow value."""

        owner: CompiledFunctionInvocation | None = None
        for invocation in self.invocations:
            if invocation.main_flow_output_plans:
                owner = None
            elif not invocation.contract.preserves_input_main_flow():
                owner = invocation
        return owner

    def preserves_input_main_flow(self) -> bool:
        """Return whether every invocation leaves the group's input flow unchanged."""

        return bool(self.invocations) and all(
            invocation.contract.preserves_input_main_flow()
            for invocation in self.invocations
        )


@dataclass(frozen=True, slots=True)
class CompiledFunctionPattern:
    """Compiled function-pattern graph consumed by FunctionStep runtime."""

    groups: tuple[CompiledFunctionGroup, ...]
    is_grouped: bool

    @property
    def execution_scope(self) -> FunctionStepExecutionScope:
        """Return the uniform lifecycle scope derived from its invocations."""
        return FunctionStepExecutionScope.require_uniform(
            invocation.contract for invocation in self.iter_invocations()
        )

    @property
    def default_group(self) -> CompiledFunctionGroup:
        """Return the compiled default group."""
        return self.require_group(DEFAULT_GROUP_KEY)

    def require_group(self, group_key: str) -> CompiledFunctionGroup:
        """Return a compiled group or fail loudly when it is absent."""
        group = self.group_by_key(group_key)
        if group is None:
            raise ValueError(f"Compiled function pattern has no {group_key!r} group.")
        return group

    def group_by_key(self, group_key: str) -> CompiledFunctionGroup | None:
        """Return a compiled group by normalized group key."""
        for group in self.groups:
            if group.group_key == group_key:
                return group
        return None

    def iter_invocations(self) -> Iterator[CompiledFunctionInvocation]:
        """Yield all compiled invocations in runtime order."""
        for group in self.groups:
            yield from group.invocations

    def coalesced_artifact_output_specs(self) -> tuple[ArtifactSpec, ...]:
        """Return exact selected outputs under the canonical merge policy."""
        accumulator = ArtifactSpecAccumulator.empty("compiled invocation output")

        for invocation in self.iter_invocations():
            selected_refs = tuple(plan.ref() for plan in invocation.artifact_output_plans)
            for spec in invocation.contract.artifact_outputs:
                ref = spec.ref()
                if selected_refs.count(ref) != 1:
                    raise ValueError(
                        "Compiled invocation output declaration requires one exact "
                        "selected plan for "
                        f"{ref.plan_type.plan_role}:{ref.artifact_type.value}:"
                        f"{ref.name}."
                    )
                accumulator.add(spec)

        return tuple(accumulator.specs.values())

    def artifact_input_edges_by_key(
        self,
    ) -> dict[
        InvocationArtifactInputProjectionKey,
        InvocationArtifactInputEdgePlan,
    ]:
        """Return every exact invocation-input projection keyed by graph edge."""

        result: dict[
            InvocationArtifactInputProjectionKey,
            InvocationArtifactInputEdgePlan,
        ] = {}
        for invocation in self.iter_invocations():
            for edge in invocation.artifact_input_edges:
                if edge.key in result:
                    raise ValueError(
                        f"Duplicate invocation artifact input edge {edge.key!r}."
                    )
                result[edge.key] = edge
        return result

    def preserves_input_main_flow(self) -> bool:
        """Return whether every compiled invocation preserves input main flow."""

        return bool(self.groups) and all(
            group.preserves_input_main_flow() for group in self.groups
        )

    def group_for_component(
        self,
        component_value: FunctionGroupKey,
    ) -> CompiledFunctionGroup | None:
        """Return the compiled group selected for a discovered component value."""
        if not self.is_grouped:
            return self.default_group

        component_key = str(component_value)
        return self.group_by_key(component_key)

    def publishes_output_to_main_flow(
        self,
        output_plan: ArtifactOutputPlan,
        component_value: FunctionGroupKey,
    ) -> bool:
        """Return whether the selected group publishes this exact artifact output."""

        group = self.group_for_component(component_value)
        if group is None:
            raise ValueError(
                "Compiled function pattern has no group for artifact output "
                f"{output_plan.ref()!r} at component value {component_value!r}."
            )
        output_ref = output_plan.ref()
        return any(
            candidate.ref() == output_ref
            for candidate in group.resulting_main_flow_output_plans()
        )

    def prepare_grouped_patterns(
        self,
        patterns: FunctionPatternSyntax | GroupedPatternMap,
        default_component: FunctionGroupKey,
    ) -> GroupedPatternMap:
        """Filter detected pattern groups to those with compiled functions."""
        grouped_patterns = (
            patterns if isinstance(patterns, dict) else {default_component: patterns}
        )

        if not self.is_grouped:
            return grouped_patterns

        filtered = {
            component_value: pattern_list
            for component_value, pattern_list in grouped_patterns.items()
            if self.group_for_component(component_value) is not None
        }
        if not filtered:
            raise ValueError(
                "No components match between discovered data and compiled function pattern. "
                f"Discovered components: {list(grouped_patterns.keys())}. "
                f"Function pattern groups: {[group.group_key for group in self.groups]}."
            )
        return filtered


def iter_enabled_function_invocations(
    pattern: FunctionPatternSyntax,
) -> Iterator[FunctionInvocation]:
    """Yield enabled callable invocations from any supported function pattern.

    Positions are renumbered after disabled functions are filtered out, matching
    the current runtime behavior for list chains and dict-pattern branches.
    """
    for item in normalize_function_pattern(pattern).iter_items():
        yield FunctionInvocation(
            contract=item.contract,
            key=item.key,
        )


def get_core_callable(
    func_pattern: FunctionPatternSyntax,
) -> FunctionPatternCallable | None:
    """Extract the first effective callable reference from a function pattern."""
    if isinstance(func_pattern, FunctionReference):
        return func_pattern

    if callable(func_pattern) and not isinstance(func_pattern, type):
        return func_pattern

    if isinstance(func_pattern, tuple):
        _require_two_member_function_leaf(func_pattern)
        first_element = func_pattern[0]
        if isinstance(first_element, FunctionReference):
            return first_element
        if callable(first_element) and not isinstance(first_element, type):
            return first_element
        return None

    if isinstance(func_pattern, list) and func_pattern:
        return get_core_callable(func_pattern[0])

    if isinstance(func_pattern, dict) and func_pattern:
        for value in func_pattern.values():
            core_callable = get_core_callable(value)
            if core_callable is not None:
                return core_callable

    return None


def normalize_function_pattern(
    pattern: FunctionPatternSyntax | NormalizedFunctionPattern,
) -> NormalizedFunctionPattern:
    """Lower raw FunctionStep.func syntax into typed grouped callable items."""
    if isinstance(pattern, NormalizedFunctionPattern):
        return pattern
    normalizer = NormalizeFunctionGroupAuthority()
    if isinstance(pattern, dict):
        groups = tuple(
            normalizer.normalize(group_key=group_key, pattern=value)
            for group_key, value in pattern.items()
        )
        return NormalizedFunctionPattern(
            groups=groups,
            is_grouped=True,
        )

    return NormalizedFunctionPattern(
        groups=(
            normalizer.normalize(
                group_key=DEFAULT_GROUP_KEY,
                pattern=pattern,
            ),
        ),
        is_grouped=False,
    )


def resolve_function_pattern_execution_scope(
    pattern: FunctionPatternSyntax,
    invocation_contract_provider: InvocationContractProvider,
    step_context: ArtifactDeclarationStepContext,
) -> FunctionStepExecutionScope:
    """Resolve one uniform callable scope before path planning."""
    contracts = resolve_function_pattern_contracts(
        pattern,
        invocation_contract_provider,
        step_context,
    )
    return FunctionStepExecutionScope.require_uniform(contracts)


def resolve_function_pattern_contracts(
    pattern: FunctionPatternSyntax,
    invocation_contract_provider: InvocationContractProvider,
    step_context: ArtifactDeclarationStepContext,
) -> tuple[CallableContract, ...]:
    """Resolve compiler-visible contracts for every enabled pattern item."""

    contracts: list[CallableContract] = []
    for item in normalize_function_pattern(pattern).iter_items():
        contract_plan = invocation_contract_provider(item, step_context)
        contracts.append(
            item.contract if contract_plan is None else contract_plan.contract
        )
    return tuple(contracts)


def compile_function_pattern(
    pattern: FunctionPatternSyntax,
    input_plans: Mapping[ArtifactSpecRef, ArtifactInputPlan],
    output_plans: Mapping[ArtifactSpecRef, ArtifactOutputPlan],
    declaration_provider: InvocationArtifactDeclarationProviderLike = (
        callable_contract_artifact_declarations
    ),
    invocation_contract_provider: InvocationContractProvider = (
        CompositeInvocationContractProvider(())
    ),
    step_context: ArtifactDeclarationStepContext = (
        ArtifactDeclarationStepContext.empty()
    ),
    runtime_parameter_bindings: Sequence[RuntimeParameterBinding] = (),
    path_resolver: "CompilationPathResolver | None" = None,
) -> CompiledFunctionPattern:
    """Compile raw FunctionStep.func syntax into the runtime source of truth."""
    normalized = normalize_function_pattern(pattern)
    compiler = CompileFunctionGroupAuthority(
        input_plans=input_plans,
        output_plans=output_plans,
        declaration_provider=declaration_provider,
        invocation_contract_provider=invocation_contract_provider,
        step_context=step_context,
        runtime_parameter_bindings=tuple(runtime_parameter_bindings),
        path_resolver=path_resolver,
    )
    return CompiledFunctionPattern(
        groups=tuple(compiler.compile(group) for group in normalized.groups),
        is_grouped=normalized.is_grouped,
    )


def strip_disabled_functions(
    pattern: FunctionPatternSyntax,
) -> FunctionPatternSyntax | None:
    """Remove disabled function items from any supported function-pattern shape."""
    if isinstance(pattern, tuple):
        _func, kwargs = _split_function_item(pattern)
        if RUNTIME_CALLABLE_KWARG_POLICY.item_is_disabled(kwargs):
            return None
        return pattern

    if isinstance(pattern, list):
        stripped = [strip_disabled_functions(item) for item in pattern]
        return [item for item in stripped if item not in (None, [], {})]

    if isinstance(pattern, dict):
        stripped = {
            key: strip_disabled_functions(value) for key, value in pattern.items()
        }
        return {
            key: value for key, value in stripped.items() if value not in (None, [], {})
        }

    return pattern


def inject_kwargs_into_pattern(
    pattern: FunctionPatternSyntax,
    kwargs: RuntimeKwargMap,
) -> FunctionPatternSyntax:
    """Inject kwargs into every callable item in a function pattern."""
    if not kwargs:
        return pattern

    if _is_callable_pattern_item(pattern):
        return PatternItemKwargMerge(kwargs).merge(pattern)

    if isinstance(pattern, list):
        return [inject_kwargs_into_pattern(item, kwargs) for item in pattern]

    if isinstance(pattern, dict):
        return {
            key: inject_kwargs_into_pattern(value, kwargs)
            for key, value in pattern.items()
        }

    return pattern


def inject_artifact_input_values(
    pattern: FunctionPatternSyntax,
    values_by_key: RuntimeKwargMap,
) -> FunctionPatternSyntax:
    """Inject artifact input values only into callables that declare those inputs."""
    if not values_by_key:
        return pattern

    if _is_callable_pattern_item(pattern):
        core_callable = get_core_callable(pattern)
        contract = CallableContract.from_callable(core_callable)
        matched_values = {
            key: value
            for key, value in values_by_key.items()
            if key in contract.artifact_inputs.names()
        }
        if not matched_values:
            return pattern
        return PatternItemKwargMerge(matched_values).merge_replacing_existing(pattern)

    if isinstance(pattern, list):
        return [inject_artifact_input_values(item, values_by_key) for item in pattern]

    if isinstance(pattern, dict):
        return {
            key: inject_artifact_input_values(value, values_by_key)
            for key, value in pattern.items()
        }

    raise ValueError(
        f"Cannot inject artifact values into pattern type: {type(pattern)}"
    )


def _is_callable_pattern_item(pattern: FunctionPatternSyntax) -> bool:
    if get_core_callable(pattern) is None:
        return False
    return not isinstance(pattern, (list, dict))


@dataclass(frozen=True, slots=True)
class PatternItemKwargMerge:
    """Nominal merge authority for function-pattern item kwargs."""

    kwargs: RuntimeKwargMap

    def merge(self, pattern: FunctionPatternSyntax) -> FunctionPatternSyntax:
        """Return a callable pattern item with injected kwargs."""
        if isinstance(pattern, tuple):
            func, existing_kwargs = _split_function_item(pattern)
            return (func, {**self.kwargs, **existing_kwargs})

        return (pattern, dict(self.kwargs))

    def merge_replacing_existing(
        self,
        pattern: FunctionPatternSyntax,
    ) -> FunctionPatternSyntax:
        """Return a callable pattern item with injected kwargs taking precedence."""
        if isinstance(pattern, tuple):
            func, existing_kwargs = _split_function_item(pattern)
            return (func, {**existing_kwargs, **self.kwargs})

        return self.merge(pattern)


@dataclass(frozen=True, slots=True)
class NormalizeFunctionGroupAuthority:
    """Normalize one function-pattern group into callable invocation items."""

    def normalize(
        self,
        group_key: FunctionGroupKey,
        pattern: FunctionPatternSyntax,
    ) -> NormalizedFunctionGroup:
        items = pattern if isinstance(pattern, list) else [pattern]
        normalized_items: list[NormalizedFunctionItem] = []

        for item in items:
            func, kwargs = _split_function_item(item)
            if RUNTIME_CALLABLE_KWARG_POLICY.item_is_disabled(kwargs):
                continue
            contract = CallableContract.from_callable(func)
            position = len(normalized_items)
            normalized_items.append(
                NormalizedFunctionItem(
                    key=FunctionInvocationKey.from_contract(
                        contract,
                        group_key,
                        position,
                    ),
                    contract=contract,
                    kwargs=_freeze_runtime_kwargs(kwargs),
                )
            )

        return NormalizedFunctionGroup(
            source_group_key=group_key,
            items=tuple(normalized_items),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompileFunctionGroupAuthority:
    """Compile normalized function-pattern groups into invocation plans."""

    input_plans: Mapping[ArtifactSpecRef, ArtifactInputPlan]
    output_plans: Mapping[ArtifactSpecRef, ArtifactOutputPlan]
    declaration_provider: InvocationArtifactDeclarationProviderLike
    invocation_contract_provider: InvocationContractProvider
    step_context: ArtifactDeclarationStepContext
    runtime_parameter_bindings: tuple[RuntimeParameterBinding, ...] = ()
    path_resolver: "CompilationPathResolver | None" = None

    def compile(
        self, normalized_group: NormalizedFunctionGroup
    ) -> CompiledFunctionGroup:
        invocations = tuple(
            _compile_invocation(
                item=item,
                input_plans=self.input_plans,
                output_plans=self.output_plans,
                declaration_provider=self.declaration_provider,
                invocation_contract_provider=self.invocation_contract_provider,
                step_context=self.step_context,
                runtime_parameter_bindings=self.runtime_parameter_bindings,
                path_resolver=self.path_resolver,
            )
            for item in normalized_group.items
        )
        return CompiledFunctionGroup(
            group_key=normalized_group.group_key,
            invocations=invocations,
        )


def _compile_invocation(
    item: NormalizedFunctionItem,
    input_plans: Mapping[ArtifactSpecRef, ArtifactInputPlan],
    output_plans: Mapping[ArtifactSpecRef, ArtifactOutputPlan],
    declaration_provider: InvocationArtifactDeclarationProviderLike,
    invocation_contract_provider: InvocationContractProvider,
    step_context: ArtifactDeclarationStepContext,
    runtime_parameter_bindings: Sequence[RuntimeParameterBinding],
    path_resolver: "CompilationPathResolver | None",
) -> CompiledFunctionInvocation:
    contract_plan = invocation_contract_provider(item, step_context)
    if contract_plan is None:
        invocation_kwargs = item.kwargs
    else:
        invocation_kwargs = contract_plan.consume_authored_kwargs(
            item,
            step_context,
        )
        item = replace(item, contract=contract_plan.contract)
    artifact_selector = declaration_provider(item, step_context)
    artifact_input_plans = artifact_selector.select_plans(
        ArtifactInputPlan,
        input_plans,
    )
    user_kwargs, compiled_runtime_bindings = _compile_runtime_parameter_bindings(
        invocation_kwargs,
        runtime_parameter_bindings,
        (
            *item.contract.runtime_bound_parameters,
            *item.contract.config_bound_parameter_names,
        ),
    )
    public_kwargs = dict(user_kwargs)
    if path_resolver is not None:
        public_kwargs = item.contract.resolve_declared_paths(
            public_kwargs,
            path_resolver,
        )
    else:
        relative_parameters = tuple(
            parameter_name
            for parameter_name, (_declaration, value) in (
                item.contract.declared_path_values(public_kwargs).items()
            )
            if isinstance(value, (str, Path))
            and not Path(value).is_absolute()
        )
        if relative_parameters:
            raise ValueError(
                f"Callable {item.contract.function_name!r} has relative declared "
                f"paths {relative_parameters!r} but compilation supplied no "
                "CompilationPathResolver."
            )
    runtime_loaded_input_refs = frozenset(
        plan.ref() for plan in artifact_input_plans
    )
    validated_kwargs = item.contract.validate_public_kwargs(
        public_kwargs,
        runtime_loaded_artifact_parameter_names=(
            spec.parameter_name
            for spec in item.contract.artifact_inputs
            if spec.parameter_name is not None
            and spec.ref() in runtime_loaded_input_refs
        ),
    )
    return CompiledFunctionInvocation(
        key=item.key,
        contract=item.contract,
        kwargs=validated_kwargs,
        artifact_output_plans=artifact_selector.select_plans(
            ArtifactOutputPlan,
            output_plans,
        ),
        runtime_parameter_bindings=compiled_runtime_bindings,
    )


def _compile_runtime_parameter_bindings(
    kwargs: RuntimeKwargItems,
    runtime_parameter_bindings: Sequence[RuntimeParameterBinding],
    accepted_parameter_names: Sequence[str],
) -> tuple[RuntimeKwargItems, tuple[RuntimeParameterBinding, ...]]:
    """Move config-owned runtime parameters out of callable kwargs."""
    accepted_parameter_name_set = frozenset(accepted_parameter_names)
    if not runtime_parameter_bindings or not accepted_parameter_name_set:
        return kwargs, ()

    remaining_kwargs = dict(kwargs)
    compiled_bindings: list[RuntimeParameterBinding] = []
    for binding in runtime_parameter_bindings:
        if binding.parameter_name not in accepted_parameter_name_set:
            continue
        if binding.parameter_name in remaining_kwargs:
            value = remaining_kwargs.pop(binding.parameter_name)
        else:
            value = binding.value
        compiled_bindings.append(
            RuntimeParameterBinding(
                parameter_name=binding.parameter_name,
                value=value,
            )
        )
    return tuple(remaining_kwargs.items()), tuple(compiled_bindings)


def _split_function_item(
    func_item: FunctionPatternSyntax,
) -> tuple[FunctionPatternCallable, RuntimeKwargMap]:
    if isinstance(func_item, tuple):
        func, kwargs = _require_two_member_function_leaf(func_item)
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"Function kwargs must be a mapping, got {type(kwargs)}")
        return func, kwargs

    if get_core_callable(func_item) is not None:
        return func_item, {}

    raise TypeError(f"Invalid function-pattern item: {func_item}")


def _require_two_member_function_leaf(func_item: tuple) -> tuple[object, object]:
    if len(func_item) != 2:
        raise TypeError(
            "Function-pattern tuple leaves must contain exactly two members: "
            "(callable, kwargs)."
        )
    return func_item


def _freeze_runtime_kwargs(kwargs: RuntimeKwargMap) -> RuntimeKwargItems:
    return RUNTIME_CALLABLE_KWARG_POLICY.invocation_items(kwargs)
