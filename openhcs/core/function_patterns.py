"""Nominal helpers for OpenHCS function-pattern invocation identity.

FunctionStep accepts several pattern shapes: a callable, ``(callable, kwargs)``,
a list chain, or a dict keyed by component/group. The runtime already treats
each enabled callable position as the effective execution unit; this module
gives that unit a named identity for compile-time planning and runtime lookup.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan
from openhcs.core.callable_contract import CallableContract
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.formats.func_arg_prep import get_core_callable


DEFAULT_GROUP_KEY = "default"


@dataclass(frozen=True)
class FunctionInvocationKey:
    """Stable identity for one callable position inside a function pattern."""

    function_name: str
    group_key: str
    position: int

    @classmethod
    def from_callable(
        cls, func: Callable, group_key: Any, position: int
    ) -> "FunctionInvocationKey":
        return cls.from_contract(
            CallableContract.from_callable(func),
            group_key,
            position,
        )

    @classmethod
    def from_contract(
        cls, contract: CallableContract, group_key: Any, position: int
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
    def func(self) -> Any:
        """Underlying callable or FunctionReference for compatibility."""
        return self.contract.func


@dataclass(frozen=True, slots=True)
class NormalizedFunctionItem:
    """Compiler-normalized callable item with stable invocation identity."""

    key: FunctionInvocationKey
    contract: CallableContract
    kwargs: tuple[tuple[str, Any], ...] = ()
    invocation_options: RuntimeInvocationOptions | None = None

    @property
    def func(self) -> Any:
        """Underlying callable or FunctionReference for compatibility."""
        return self.contract.func

    @property
    def kwargs_dict(self) -> dict[str, Any]:
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
class CompiledFunctionInvocation:
    """Executable compiler output for one callable in a function pattern."""

    key: FunctionInvocationKey
    contract: CallableContract
    kwargs: tuple[tuple[str, Any], ...] = ()
    invocation_options: RuntimeInvocationOptions | None = None
    artifact_input_keys: tuple[str, ...] = ()
    artifact_output_keys: tuple[str, ...] = ()

    @property
    def func(self) -> Any:
        """Underlying callable or FunctionReference resolved at runtime."""
        return self.contract.func

    @property
    def input_memory_type(self) -> str | None:
        """Declared input memory type from the callable contract."""
        return self.contract.input_memory_type

    @property
    def output_memory_type(self) -> str | None:
        """Declared output memory type from the callable contract."""
        return self.contract.output_memory_type

    @property
    def kwargs_dict(self) -> dict[str, Any]:
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


@dataclass(frozen=True, slots=True)
class CompiledFunctionGroup:
    """Compiled callable chain for one function-pattern group."""

    group_key: str
    invocations: tuple[CompiledFunctionInvocation, ...]


@dataclass(frozen=True, slots=True)
class CompiledFunctionPattern:
    """Compiled function-pattern graph consumed by FunctionStep runtime."""

    groups: tuple[CompiledFunctionGroup, ...]
    is_grouped: bool

    @property
    def default_group(self) -> CompiledFunctionGroup:
        for group in self.groups:
            if group.group_key == DEFAULT_GROUP_KEY:
                return group
        raise ValueError("Compiled function pattern has no default group.")

    def iter_invocations(self) -> Iterator[CompiledFunctionInvocation]:
        """Yield all compiled invocations in runtime order."""
        for group in self.groups:
            yield from group.invocations

    def group_for_component(self, component_value: Any) -> CompiledFunctionGroup | None:
        """Return the compiled group selected for a discovered component value."""
        if not self.is_grouped:
            return self.default_group

        component_key = str(component_value)
        for group in self.groups:
            if group.group_key == component_key:
                return group
        return None

    def prepare_grouped_patterns(
        self,
        patterns: Any,
        default_component: Any,
    ) -> dict[Any, Sequence[Any]]:
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


def iter_enabled_function_invocations(pattern: Any) -> Iterator[FunctionInvocation]:
    """Yield enabled callable invocations from any supported function pattern.

    Positions are renumbered after disabled functions are filtered out, matching
    the current runtime behavior for list chains and dict-pattern branches.
    """
    for item in normalize_function_pattern(pattern).iter_items():
        yield FunctionInvocation(
            contract=item.contract,
            key=item.key,
        )


def normalize_function_pattern(pattern: Any) -> NormalizedFunctionPattern:
    """Lower raw FunctionStep.func syntax into typed grouped callable items."""
    if isinstance(pattern, dict):
        return NormalizedFunctionPattern(
            groups=tuple(
                _normalize_function_group(group_key=group_key, pattern=value)
                for group_key, value in pattern.items()
            ),
            is_grouped=True,
        )

    return NormalizedFunctionPattern(
        groups=(
            _normalize_function_group(
                group_key=DEFAULT_GROUP_KEY,
                pattern=pattern,
            ),
        ),
        is_grouped=False,
    )


def compile_function_pattern(
    pattern: Any,
    input_plans: Mapping[str, ArtifactInputPlan],
    output_plans: Mapping[str, ArtifactOutputPlan],
) -> CompiledFunctionPattern:
    """Compile raw FunctionStep.func syntax into the runtime source of truth."""
    normalized = normalize_function_pattern(pattern)
    return CompiledFunctionPattern(
        groups=tuple(
            _compile_function_group(
                normalized_group=group,
                input_plans=input_plans,
                output_plans=output_plans,
            )
            for group in normalized.groups
        ),
        is_grouped=normalized.is_grouped,
    )


def strip_disabled_functions(pattern: Any) -> Any:
    """Remove disabled function items from any supported function-pattern shape."""
    if (
        isinstance(pattern, tuple)
        and len(pattern) in {2, 3}
        and isinstance(pattern[1], dict)
    ):
        if pattern[1].get("enabled", True) is False:
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


def inject_kwargs_into_pattern(pattern: Any, kwargs: Mapping[str, Any]) -> Any:
    """Inject kwargs into every callable item in a function pattern."""
    if not kwargs:
        return pattern

    if _is_callable_pattern_item(pattern):
        return _merge_pattern_item_kwargs(pattern, kwargs)

    if isinstance(pattern, list):
        return [inject_kwargs_into_pattern(item, kwargs) for item in pattern]

    if isinstance(pattern, dict):
        return {
            key: inject_kwargs_into_pattern(value, kwargs)
            for key, value in pattern.items()
        }

    return pattern


def inject_artifact_input_values(
    pattern: Any,
    values_by_key: Mapping[str, Any],
) -> Any:
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
        return _merge_pattern_item_kwargs(pattern, matched_values)

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


def _is_callable_pattern_item(pattern: Any) -> bool:
    if get_core_callable(pattern) is None:
        return False
    return not isinstance(pattern, (list, dict))


def _merge_pattern_item_kwargs(pattern: Any, kwargs: Mapping[str, Any]) -> Any:
    if isinstance(pattern, tuple) and len(pattern) in {2, 3}:
        func, existing_kwargs, *invocation_options = pattern
        if not isinstance(existing_kwargs, Mapping):
            raise TypeError(
                f"Function kwargs must be a mapping, got {type(existing_kwargs)}"
            )
        return (func, {**kwargs, **existing_kwargs}, *invocation_options)

    return (pattern, dict(kwargs))


def _normalize_function_group(
    group_key: Any,
    pattern: Any,
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


def _compile_function_group(
    normalized_group: NormalizedFunctionGroup,
    input_plans: Mapping[str, ArtifactInputPlan],
    output_plans: Mapping[str, ArtifactOutputPlan],
) -> CompiledFunctionGroup:
    invocations = tuple(
        _compile_invocation(
            item=item,
            input_plans=input_plans,
            output_plans=output_plans,
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
) -> CompiledFunctionInvocation:
    return CompiledFunctionInvocation(
        key=item.key,
        contract=item.contract,
        kwargs=item.kwargs,
        invocation_options=item.invocation_options,
        artifact_input_keys=item.contract.select_input_plan_keys(input_plans),
        artifact_output_keys=item.contract.select_output_plan_keys(output_plans),
    )


def _is_disabled_function_item(func_item: Any) -> bool:
    return (
        isinstance(func_item, tuple)
        and len(func_item) in {2, 3}
        and isinstance(func_item[1], Mapping)
        and func_item[1].get("enabled", True) is False
    )


def _split_function_item(
    func_item: Any,
) -> tuple[Any, Mapping[str, Any], RuntimeInvocationOptions | None]:
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


def _freeze_runtime_kwargs(kwargs: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (key, value)
        for key, value in kwargs.items()
        if key != "__pyqt_reactive_scope_token__"
    )
