"""Nominal callable requirements for real variable-component stack inputs."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol


@dataclass(frozen=True, slots=True)
class VariableComponentStackRequirementRequest:
    """Invocation context used to evaluate stack-axis requirements."""

    func: Callable[..., object] | None
    kwargs: Mapping[str, object]


class RuntimeSemanticControlParameter(Protocol):
    """Contract for nominal runtime parameters that select semantic mode."""

    is_semantic_control: ClassVar[bool]

    @classmethod
    def require_parameter_name(cls) -> str:
        """Return the public invocation parameter name."""

    @classmethod
    def default_value(cls) -> object:
        """Return the parameter default used when invocation omits it."""


class VariableComponentStackRequirement(ABC):
    """Declaration owned by a callable or processing contract."""

    @abstractmethod
    def is_required(
        self,
        request: VariableComponentStackRequirementRequest,
    ) -> bool:
        """Return whether this invocation needs a non-empty variable axis."""


@dataclass(frozen=True, slots=True)
class AlwaysRequiresVariableComponentStack(VariableComponentStackRequirement):
    """Stack requirement for callables with full-stack-only semantics."""

    def is_required(
        self,
        request: VariableComponentStackRequirementRequest,
    ) -> bool:
        del request
        return True


@dataclass(frozen=True, slots=True)
class SemanticControlVariableComponentStackRequirement(
    VariableComponentStackRequirement
):
    """Stack requirement disabled by a declared semantic-control parameter."""

    parameter_types: tuple[type[RuntimeSemanticControlParameter], ...]

    def is_required(
        self,
        request: VariableComponentStackRequirementRequest,
    ) -> bool:
        return not any(
            bool(self._parameter_value(parameter_type, request))
            for parameter_type in self.parameter_types
            if parameter_type.is_semantic_control
        )

    @staticmethod
    def _parameter_value(
        parameter_type: type[RuntimeSemanticControlParameter],
        request: VariableComponentStackRequirementRequest,
    ) -> object:
        parameter_name = parameter_type.require_parameter_name()
        if parameter_name in request.kwargs:
            return request.kwargs[parameter_name]
        if request.func is not None:
            parameters = inspect.signature(request.func).parameters
            if parameter_name in parameters:
                default_value = parameters[parameter_name].default
                if default_value is not inspect.Parameter.empty:
                    return default_value
        return parameter_type.default_value()
