"""FunctionStep declaration for pattern-based processing."""

from __future__ import annotations

from typing import Callable

from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_execution import FunctionStepExecutor


FunctionSpec = (
    Callable
    | tuple[Callable, dict]
    | list[Callable | tuple[Callable, dict]]
)


class FunctionStep(AbstractStep):
    """Pipeline step that delegates compiled pattern execution to FunctionStepExecutor."""

    _ui_special_fields = ("func",)

    def __init__(
        self,
        func: FunctionSpec = [],
        **kwargs,
    ):
        if "name" not in kwargs or kwargs["name"] is None:
            kwargs["name"] = getattr(_first_callable(func), "__name__", "FunctionStep")

        super().__init__(**kwargs)
        self.func = func

    def process(self, context: "ProcessingContext", step_index: int) -> None:
        FunctionStepExecutor.execute(context, step_index)


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
