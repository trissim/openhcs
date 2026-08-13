"""FunctionStep declaration for pattern-based processing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Callable, TypeAlias

from objectstate import mark_ui_special_fields

from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_execution import FunctionStepExecutor

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext


FunctionEntry: TypeAlias = Callable | tuple[Callable, dict]
FunctionSpec: TypeAlias = (
    FunctionEntry
    | list[FunctionEntry]
    | dict[str, list[FunctionEntry]]
)


@mark_ui_special_fields("func")
class FunctionStep(AbstractStep):
    """Pipeline step that delegates compiled pattern execution to FunctionStepExecutor."""

    def __init__(
        self,
        func: FunctionSpec | None = None,
        **kwargs,
    ):
        function_spec: FunctionSpec = [] if func is None else func
        if "name" not in kwargs or kwargs["name"] is None:
            kwargs["name"] = _function_step_name(function_spec)

        super().__init__(**kwargs)
        self.func = function_spec

    def process(self, context: "ProcessingContext", step_index: int) -> None:
        FunctionStepExecutor.execute(context, step_index)

    def function_spec(self) -> FunctionSpec | None:
        """Return the declaration function spec, or None after compile stripping."""
        return self.__dict__.get("func")

    def with_function_spec(self, func: FunctionSpec) -> "FunctionStep":
        """Return a shallow declaration copy with a replacement function spec."""
        from copy import copy

        step = copy(self)
        step.func = func
        return step

    def occurrence_authorities(self) -> tuple[object, ...]:
        """Prefer callable-pattern authority before nominal step authority."""

        return (
            _function_spec_authority(self.func),
            *super().occurrence_authorities(),
        )


def _first_callable(func: FunctionSpec | None) -> Callable | None:
    if isinstance(func, tuple):
        return func[0]
    if isinstance(func, list) and func:
        first_item = func[0]
        if isinstance(first_item, tuple):
            return first_item[0]
        if callable(first_item):
            return first_item
    if callable(func):
        return func
    return None


def _function_step_name(func: FunctionSpec | None) -> str:
    first_callable = _first_callable(func)
    if first_callable is None:
        return "FunctionStep"
    try:
        return first_callable.__name__
    except AttributeError:
        return first_callable.__class__.__name__


def _function_spec_authority(value: object) -> object:
    """Return recursive callable structure while excluding editable kwargs."""

    if _is_function_entry(value):
        return value[0] if isinstance(value, tuple) else value
    if isinstance(value, Mapping):
        return tuple(
            (key, _function_spec_authority(nested_value))
            for key, nested_value in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        )
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
    ):
        return tuple(
            _function_spec_authority(nested_value) for nested_value in value
        )
    return value


def _is_function_entry(value: object) -> bool:
    """Return whether a value is one callable pattern entry."""

    return callable(value) or (
        isinstance(value, tuple)
        and len(value) == 2
        and callable(value[0])
        and isinstance(value[1], Mapping)
    )
