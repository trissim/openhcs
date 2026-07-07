"""FunctionStep declaration for pattern-based processing."""

from __future__ import annotations

from typing import Callable

from objectstate import mark_ui_special_fields

from openhcs.core.function_step_invocation_contracts import (
    EMPTY_FUNCTION_STEP_INVOCATION_CONTRACTS,
    FunctionStepInvocationContracts,
)
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_execution import FunctionStepExecutor


FunctionSpec = (
    Callable
    | tuple[Callable, dict]
    | tuple[Callable, dict, RuntimeInvocationOptions]
    | list[
        Callable
        | tuple[Callable, dict]
        | tuple[Callable, dict, RuntimeInvocationOptions]
    ]
)


@mark_ui_special_fields("func", "invocation_contracts")
class FunctionStep(AbstractStep):
    """Pipeline step that delegates compiled pattern execution to FunctionStepExecutor."""

    def __init__(
        self,
        func: FunctionSpec = [],
        invocation_contracts: FunctionStepInvocationContracts = (
            EMPTY_FUNCTION_STEP_INVOCATION_CONTRACTS
        ),
        **kwargs,
    ):
        if not isinstance(invocation_contracts, FunctionStepInvocationContracts):
            raise TypeError(
                "FunctionStep.invocation_contracts must be "
                "FunctionStepInvocationContracts, got "
                f"{type(invocation_contracts).__name__}."
            )
        if "name" not in kwargs or kwargs["name"] is None:
            kwargs["name"] = _function_step_name(func)

        super().__init__(**kwargs)
        self.func = func
        self.invocation_contracts = invocation_contracts

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

    def with_invocation_contracts(
        self,
        invocation_contracts: FunctionStepInvocationContracts,
    ) -> "FunctionStep":
        """Return a shallow declaration copy with replacement invocation contracts."""
        from copy import copy

        if not isinstance(invocation_contracts, FunctionStepInvocationContracts):
            raise TypeError(
                "FunctionStep.with_invocation_contracts requires "
                "FunctionStepInvocationContracts, got "
                f"{type(invocation_contracts).__name__}."
            )
        step = copy(self)
        step.invocation_contracts = invocation_contracts
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
