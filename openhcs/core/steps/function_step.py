"""FunctionStep declaration for pattern-based processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from objectstate import mark_ui_special_fields

from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_execution import FunctionStepExecutor

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext


FunctionSpec = (
    Callable
    | tuple[Callable, dict]
    | list[Callable | tuple[Callable, dict]]
)


@mark_ui_special_fields("func")
class FunctionStep(AbstractStep):
    """Pipeline step that delegates compiled pattern execution to FunctionStepExecutor."""

    def __init__(
        self,
        func: FunctionSpec = [],
        **kwargs,
    ):
        if "name" not in kwargs or kwargs["name"] is None:
            kwargs["name"] = _function_step_name(func)

        super().__init__(**kwargs)
        self.func = func

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

def _first_callable(func: FunctionSpec) -> Callable | None:
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


def _function_step_name(func: FunctionSpec) -> str:
    first_callable = _first_callable(func)
    if first_callable is None:
        return "FunctionStep"
    try:
        return first_callable.__name__
    except AttributeError:
        return first_callable.__class__.__name__
