"""Typed runtime adapter injection contracts for callable execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar
from weakref import WeakKeyDictionary

from openhcs.core.artifacts import ArtifactOutputPlan


_F = TypeVar("_F", bound=Callable[..., Any])
_RUNTIME_ADAPTER_SPECS: WeakKeyDictionary[
    Callable[..., Any], "RuntimeAdapterSpec"
] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class RuntimeAdapterRequest:
    """Runtime data needed to build an invocation-scoped adapter."""

    context: Any
    artifact_outputs: Mapping[str, ArtifactOutputPlan]


@dataclass(frozen=True, slots=True)
class RuntimeAdapterSpec:
    """Callable-owned runtime adapter injection contract."""

    parameter_name: str
    factory: Callable[[RuntimeAdapterRequest], Any]

    def __post_init__(self) -> None:
        if not self.parameter_name:
            raise ValueError("RuntimeAdapterSpec.parameter_name cannot be empty.")
        if not callable(self.factory):
            raise TypeError("RuntimeAdapterSpec.factory must be callable.")


def runtime_adapter(
    parameter_name: str,
    factory: Callable[[RuntimeAdapterRequest], Any],
) -> Callable[[_F], _F]:
    """Declare that a callable needs an invocation-scoped runtime adapter."""
    spec = RuntimeAdapterSpec(parameter_name=parameter_name, factory=factory)

    def decorator(func: _F) -> _F:
        _RUNTIME_ADAPTER_SPECS[func] = spec
        return func

    return decorator


def runtime_adapter_spec_from_callable(func: Any) -> RuntimeAdapterSpec | None:
    """Return the callable's declared runtime adapter contract, if any."""
    if not callable(func):
        return None
    return _RUNTIME_ADAPTER_SPECS.get(func)
