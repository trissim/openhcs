"""Nominal callable requirements for real variable-component stack inputs."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from python_introspect import RuntimeParameterDeclarationABC


@dataclass(frozen=True, slots=True)
class VariableComponentStackRequirementRequest:
    """Invocation context used to evaluate stack-axis requirements."""

    func: Callable[..., object] | None
    kwargs: Mapping[str, object]


class VariableComponentStackRequirement(ABC):
    """Declaration owned by a callable or processing contract."""

    @abstractmethod
    def is_required(
        self,
        request: VariableComponentStackRequirementRequest,
    ) -> bool:
        """Return whether this invocation needs a non-empty variable axis."""

    def bind_to_callable(
        self,
        func: Callable[..., object],
    ) -> "VariableComponentStackRequirement":
        """Bind callable-specific defaults when this requirement needs them."""

        del func
        return self


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

    parameter_types: tuple[type[RuntimeParameterDeclarationABC], ...]
    bound_defaults: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        parameter_types = RuntimeParameterDeclarationABC.require_declaration_types(
            self.parameter_types,
            boundary="SemanticControlVariableComponentStackRequirement.parameter_types",
        )
        non_controls = tuple(
            parameter_type.__name__
            for parameter_type in parameter_types
            if not parameter_type.is_semantic_control
        )
        if non_controls:
            raise TypeError(
                "Semantic-control stack requirements require declarations with "
                f"is_semantic_control=True, got {non_controls!r}."
            )
        missing_defaults = tuple(
            parameter_type.__name__
            for parameter_type in parameter_types
            if parameter_type.validated_parameter().default is inspect.Parameter.empty
        )
        if missing_defaults:
            raise TypeError(
                "Semantic-control stack requirements require declared defaults, "
                f"got {missing_defaults!r}."
            )
        object.__setattr__(self, "parameter_types", parameter_types)
        parameter_names = tuple(
            parameter_type.require_parameter_name()
            for parameter_type in parameter_types
        )
        bound_names = tuple(name for name, _ in self.bound_defaults)
        if len(bound_names) != len(set(bound_names)):
            raise ValueError(
                "Semantic-control stack requirements need unique bound defaults."
            )
        undeclared = tuple(name for name in bound_names if name not in parameter_names)
        if undeclared:
            raise ValueError(
                "Semantic-control bound defaults reference undeclared parameters: "
                f"{undeclared!r}."
            )

    def bind_to_callable(
        self,
        func: Callable[..., object],
    ) -> "SemanticControlVariableComponentStackRequirement":
        """Capture signature defaults before a callable becomes a reference."""

        parameters = inspect.signature(func).parameters
        bound_defaults = tuple(
            (
                parameter_name,
                (
                    parameters[parameter_name].default
                    if parameter_name in parameters
                    and parameters[parameter_name].default
                    is not inspect.Parameter.empty
                    else parameter_type.validated_parameter().default
                ),
            )
            for parameter_type in self.parameter_types
            for parameter_name in (parameter_type.require_parameter_name(),)
        )
        return replace(self, bound_defaults=bound_defaults)

    def is_required(
        self,
        request: VariableComponentStackRequirementRequest,
    ) -> bool:
        return not any(
            bool(self._parameter_value(parameter_type, request))
            for parameter_type in self.parameter_types
            if parameter_type.is_semantic_control
        )

    def _parameter_value(
        self,
        parameter_type: type[RuntimeParameterDeclarationABC],
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
        for bound_name, bound_default in self.bound_defaults:
            if bound_name == parameter_name:
                return bound_default
        return parameter_type.validated_parameter().default
