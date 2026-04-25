"""Nominal helpers for OpenHCS function-pattern invocation identity.

FunctionStep accepts several pattern shapes: a callable, ``(callable, kwargs)``,
a list chain, or a dict keyed by component/group. The runtime already treats
each enabled callable position as the effective execution unit; this module
gives that unit a named identity for compile-time planning and runtime lookup.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

from openhcs.core.artifacts import ArtifactOutputPlan
from openhcs.formats.func_arg_prep import get_core_callable, iter_pattern_items


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
        return cls(
            function_name=getattr(func, "__name__", "unknown"),
            group_key=str(group_key),
            position=position,
        )


@dataclass(frozen=True)
class FunctionInvocation:
    """One enabled callable extracted from a FunctionStep pattern."""

    func: Callable
    key: FunctionInvocationKey


@dataclass(frozen=True)
class FunctionInvocationPlan:
    """Compile-time plan for one function-pattern callable invocation."""

    key: FunctionInvocationKey
    artifact_output_keys: tuple[str, ...] = ()

    def select_outputs(
        self,
        output_plans: Mapping[str, ArtifactOutputPlan],
    ) -> dict[str, ArtifactOutputPlan]:
        """Select the artifact output plan entries owned by this invocation."""
        return {
            key: output_plans[key]
            for key in self.artifact_output_keys
            if key in output_plans
        }


def iter_enabled_function_invocations(pattern: Any) -> Iterator[FunctionInvocation]:
    """Yield enabled callable invocations from any supported function pattern.

    Positions are renumbered after disabled functions are filtered out, matching
    the current runtime behavior for list chains and dict-pattern branches.
    """
    position_counters: dict[Any, int] = {}

    for func_item, group_key, _original_pos in iter_pattern_items(pattern):
        if (
            isinstance(func_item, tuple)
            and len(func_item) == 2
            and isinstance(func_item[1], dict)
            and func_item[1].get("enabled", True) is False
        ):
            continue

        core_callable = get_core_callable(func_item)
        if not core_callable:
            continue

        if group_key not in position_counters:
            position_counters[group_key] = 0

        position = position_counters[group_key]
        yield FunctionInvocation(
            func=core_callable,
            key=FunctionInvocationKey.from_callable(
                core_callable, group_key, position
            ),
        )
        position_counters[group_key] += 1


def function_invocation_key(
    func: Callable,
    group_key: Any,
    position: int,
) -> FunctionInvocationKey:
    """Build the nominal key for one function-pattern callable invocation."""
    return FunctionInvocationKey.from_callable(func, group_key, position)


def build_function_invocation_plans(
    pattern: Any,
    output_plans: Mapping[str, ArtifactOutputPlan],
) -> dict[FunctionInvocationKey, FunctionInvocationPlan]:
    """Build invocation plans for all enabled callables in a pattern."""
    plans: dict[FunctionInvocationKey, FunctionInvocationPlan] = {}

    for invocation in iter_enabled_function_invocations(pattern):
        declared_outputs = getattr(
            invocation.func,
            "__artifact_outputs__",
            {},
        )
        owned_outputs = tuple(
            key for key in output_plans if key in declared_outputs
        )
        plan = FunctionInvocationPlan(
            key=invocation.key,
            artifact_output_keys=owned_outputs,
        )
        plans[plan.key] = plan

    return plans
