"""Nominal helpers for OpenHCS function-pattern invocation identity.

FunctionStep accepts several pattern shapes: a callable, ``(callable, kwargs)``,
a list chain, or a dict keyed by component/group. The runtime already treats
each enabled callable position as the effective execution unit; this module
gives that unit a named identity for compile-time planning and runtime lookup.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan
from openhcs.core.callable_contract import CallableContract
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationArtifactDeclarationProviderLike,
    callable_contract_artifact_declarations,
)
from openhcs.core.function_reference import FunctionReference
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from pyqt_reactive.pattern_metadata import PatternScopeToken
from python_introspect import Enableable


DEFAULT_GROUP_KEY = "default"

FunctionPatternCallable: TypeAlias = Callable | FunctionReference
FunctionPatternSyntax: TypeAlias = Callable | tuple | list | dict
FunctionGroupKey: TypeAlias = Hashable
RuntimeKwargMap: TypeAlias = Mapping
RuntimeKwargItems: TypeAlias = tuple[tuple, ...]
GroupedPatternMap: TypeAlias = dict[FunctionGroupKey, Sequence]
JsonScalar: TypeAlias = str | int | float | bool | None
RuntimeComponentValue: TypeAlias = JsonScalar


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
        return (
            self.enableable_type.is_parameter_key(key)
            or self.scope_token_type.is_key(key)
        )


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
        runtime_adapter = invocation.contract.runtime_adapter
        if (
            runtime_adapter is not None
            and runtime_adapter.manages_artifact_inputs
            and invocation.artifact_input_keys
        ):
            return cls.ARTIFACT_MANAGED
        return cls.SOURCE_ANCHORED

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

    @property
    def adapter_manages_artifact_inputs(self) -> bool:
        """Return whether runtime artifact inputs are loaded by the adapter."""
        return self is RuntimeInvocationDomain.ARTIFACT_MANAGED

    def select_anchor_patterns(
        self,
        pattern_list: Sequence,
    ) -> Sequence:
        """Return source anchors needed to execute this runtime domain."""
        if self is RuntimeInvocationDomain.ARTIFACT_MANAGED:
            return pattern_list[:1]
        return pattern_list


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
    invocation_options: RuntimeInvocationOptions | None = None

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

    group_key: str
    items: tuple[NormalizedFunctionItem, ...]


@dataclass(frozen=True, slots=True)
class NormalizedFunctionPattern:
    """Raw FunctionStep.func syntax lowered into typed compiler input."""

    groups: tuple[NormalizedFunctionGroup, ...]
    is_grouped: bool

    def iter_items(self) -> Iterator[NormalizedFunctionItem]:
        """Yield normalized callable items in runtime order."""
        for group in self.groups:
            yield from group.items


@dataclass(frozen=True, slots=True)
class CompiledFunctionInvocation(NormalizedFunctionItem):
    """Executable compiler output for one callable in a function pattern."""

    artifact_input_keys: tuple[str, ...] = ()
    artifact_output_keys: tuple[str, ...] = ()

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
        """Return invocation kwargs as a runtime dict."""
        return dict(self.kwargs)

    def select_inputs(
        self,
        input_plans: Mapping[str, ArtifactInputPlan],
    ) -> dict[str, ArtifactInputPlan]:
        """Select artifact input plans consumed by this invocation."""
        return {
            key: input_plans[key]
            for key in self.artifact_input_keys
            if key in input_plans
        }

    def select_outputs(
        self,
        output_plans: Mapping[str, ArtifactOutputPlan],
    ) -> dict[str, ArtifactOutputPlan]:
        """Select artifact output plans produced by this invocation."""
        return {
            key: output_plans[key]
            for key in self.artifact_output_keys
            if key in output_plans
        }

    @property
    def runtime_domain(self) -> RuntimeInvocationDomain:
        """Return the compiled runtime invocation domain."""
        return RuntimeInvocationDomain.from_invocation(self)


@dataclass(frozen=True, slots=True)
class CompiledFunctionGroup:
    """Compiled callable chain for one function-pattern group."""

    group_key: str
    invocations: tuple[CompiledFunctionInvocation, ...]

    @property
    def runtime_domain(self) -> RuntimeInvocationDomain:
        """Return the compiled runtime domain for this callable group."""
        return RuntimeInvocationDomain.from_invocations(self.invocations)


@dataclass(frozen=True, slots=True)
class CompiledFunctionPattern:
    """Compiled function-pattern graph consumed by FunctionStep runtime."""

    groups: tuple[CompiledFunctionGroup, ...]
    is_grouped: bool

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

    def group_for_component(
        self,
        component_value: FunctionGroupKey,
    ) -> CompiledFunctionGroup | None:
        """Return the compiled group selected for a discovered component value."""
        if not self.is_grouped:
            return self.default_group

        component_key = str(component_value)
        return self.group_by_key(component_key)

    def prepare_grouped_patterns(
        self,
        patterns: FunctionPatternSyntax | GroupedPatternMap,
        default_component: FunctionGroupKey,
    ) -> GroupedPatternMap:
        """Filter detected pattern groups to those with compiled functions."""
        grouped_patterns = (
            patterns
            if isinstance(patterns, dict)
            else {default_component: patterns}
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


def get_core_callable(func_pattern: FunctionPatternSyntax) -> FunctionPatternCallable | None:
    """Extract the first effective callable reference from a function pattern."""
    if isinstance(func_pattern, FunctionReference):
        return func_pattern

    if callable(func_pattern) and not isinstance(func_pattern, type):
        return func_pattern

    if isinstance(func_pattern, tuple) and func_pattern:
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


def normalize_function_pattern(pattern: FunctionPatternSyntax) -> NormalizedFunctionPattern:
    """Lower raw FunctionStep.func syntax into typed grouped callable items."""
    normalizer = NormalizeFunctionGroupAuthority()
    if isinstance(pattern, dict):
        return NormalizedFunctionPattern(
            groups=tuple(
                normalizer.normalize(group_key=group_key, pattern=value)
                for group_key, value in pattern.items()
            ),
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


def compile_function_pattern(
    pattern: FunctionPatternSyntax,
    input_plans: Mapping[str, ArtifactInputPlan],
    output_plans: Mapping[str, ArtifactOutputPlan],
    declaration_provider: InvocationArtifactDeclarationProviderLike = (
        callable_contract_artifact_declarations
    ),
    step_context: ArtifactDeclarationStepContext = (
        ArtifactDeclarationStepContext.empty()
    ),
) -> CompiledFunctionPattern:
    """Compile raw FunctionStep.func syntax into the runtime source of truth."""
    normalized = normalize_function_pattern(pattern)
    compiler = CompileFunctionGroupAuthority.from_step_context(
        input_plans=input_plans,
        output_plans=output_plans,
        declaration_provider=declaration_provider,
        step_context=step_context,
    )
    return CompiledFunctionPattern(
        groups=tuple(
            compiler.compile(group)
            for group in normalized.groups
        ),
        is_grouped=normalized.is_grouped,
    )


def strip_disabled_functions(
    pattern: FunctionPatternSyntax,
) -> FunctionPatternSyntax | None:
    """Remove disabled function items from any supported function-pattern shape."""
    if (
        isinstance(pattern, tuple)
        and len(pattern) in {2, 3}
        and isinstance(pattern[1], dict)
    ):
        if RUNTIME_CALLABLE_KWARG_POLICY.item_is_disabled(pattern[1]):
            return None
        return pattern

    if isinstance(pattern, list):
        stripped = [strip_disabled_functions(item) for item in pattern]
        return [item for item in stripped if item not in (None, [], {})]

    if isinstance(pattern, dict):
        stripped = {
            key: strip_disabled_functions(value)
            for key, value in pattern.items()
        }
        return {
            key: value
            for key, value in stripped.items()
            if value not in (None, [], {})
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
            if key in contract.artifact_input_names
        }
        if not matched_values:
            return pattern
        return PatternItemKwargMerge(matched_values).merge(pattern)

    if isinstance(pattern, list):
        return [
            inject_artifact_input_values(item, values_by_key)
            for item in pattern
        ]

    if isinstance(pattern, dict):
        return {
            key: inject_artifact_input_values(value, values_by_key)
            for key, value in pattern.items()
        }

    raise ValueError(f"Cannot inject artifact values into pattern type: {type(pattern)}")


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
        if isinstance(pattern, tuple) and len(pattern) in {2, 3}:
            func, existing_kwargs, *invocation_options = pattern
            if not isinstance(existing_kwargs, Mapping):
                raise TypeError(
                    f"Function kwargs must be a mapping, got {type(existing_kwargs)}"
                )
            return (func, {**self.kwargs, **existing_kwargs}, *invocation_options)

        return (pattern, dict(self.kwargs))


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
            if _is_disabled_function_item(item):
                continue
            func, kwargs, invocation_options = _split_function_item(item)
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
                    invocation_options=invocation_options,
                )
            )

        return NormalizedFunctionGroup(
            group_key=str(group_key),
            items=tuple(normalized_items),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompileFunctionGroupAuthority(ArtifactDeclarationStepContext):
    """Compile normalized function-pattern groups into invocation plans."""

    input_plans: Mapping[str, ArtifactInputPlan]
    output_plans: Mapping[str, ArtifactOutputPlan]
    declaration_provider: InvocationArtifactDeclarationProviderLike

    @classmethod
    def from_step_context(
        cls,
        *,
        input_plans: Mapping[str, ArtifactInputPlan],
        output_plans: Mapping[str, ArtifactOutputPlan],
        declaration_provider: InvocationArtifactDeclarationProviderLike,
        step_context: ArtifactDeclarationStepContext,
    ) -> "CompileFunctionGroupAuthority":
        return cls(
            step_name=step_context.step_name,
            step_index=step_context.step_index,
            source_bindings=step_context.source_bindings,
            processing_config=step_context.processing_config,
            source_provenance=step_context.source_provenance,
            input_plans=input_plans,
            output_plans=output_plans,
            declaration_provider=declaration_provider,
        )

    def compile(self, normalized_group: NormalizedFunctionGroup) -> CompiledFunctionGroup:
        invocations = tuple(
            _compile_invocation(
                item=item,
                input_plans=self.input_plans,
                output_plans=self.output_plans,
                declaration_provider=self.declaration_provider,
                step_context=self,
            )
            for item in normalized_group.items
        )
        return CompiledFunctionGroup(
            group_key=normalized_group.group_key,
            invocations=invocations,
        )


def _compile_invocation(
    item: NormalizedFunctionItem,
    input_plans: Mapping[str, ArtifactInputPlan],
    output_plans: Mapping[str, ArtifactOutputPlan],
    declaration_provider: InvocationArtifactDeclarationProviderLike,
    step_context: ArtifactDeclarationStepContext,
) -> CompiledFunctionInvocation:
    declarations = declaration_provider(item, step_context)
    return CompiledFunctionInvocation(
        key=item.key,
        contract=item.contract,
        kwargs=item.kwargs,
        invocation_options=item.invocation_options,
        artifact_input_keys=declarations.select_input_plan_keys(input_plans),
        artifact_output_keys=declarations.select_output_plan_keys(output_plans),
    )


def _is_disabled_function_item(func_item: FunctionPatternSyntax) -> bool:
    return (
        isinstance(func_item, tuple)
        and len(func_item) in {2, 3}
        and isinstance(func_item[1], Mapping)
        and RUNTIME_CALLABLE_KWARG_POLICY.item_is_disabled(func_item[1])
    )


def _split_function_item(
    func_item: FunctionPatternSyntax,
) -> tuple[FunctionPatternCallable, RuntimeKwargMap, RuntimeInvocationOptions | None]:
    if isinstance(func_item, tuple) and len(func_item) == 3:
        func, kwargs, invocation_options = func_item
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"Function kwargs must be a mapping, got {type(kwargs)}")
        if not isinstance(invocation_options, RuntimeInvocationOptions):
            raise TypeError(
                "Function invocation options must inherit RuntimeInvocationOptions, "
                f"got {type(invocation_options).__name__}."
            )
        return func, kwargs, invocation_options

    if isinstance(func_item, tuple) and len(func_item) == 2:
        func, kwargs = func_item
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"Function kwargs must be a mapping, got {type(kwargs)}")
        return func, kwargs, None

    if get_core_callable(func_item) is not None:
        return func_item, {}, None

    raise TypeError(f"Invalid function-pattern item: {func_item}")


def _freeze_runtime_kwargs(kwargs: RuntimeKwargMap) -> RuntimeKwargItems:
    return RUNTIME_CALLABLE_KWARG_POLICY.invocation_items(kwargs)
