"""Typed runtime adapter injection contracts for callable execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar
from weakref import WeakKeyDictionary

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    SourceBindingRuntimeContext,
)
from openhcs.core.runtime_semantics import RuntimePlaneProjection


_F = TypeVar("_F", bound=Callable[..., Any])
_RUNTIME_ADAPTER_SPECS: WeakKeyDictionary[
    Callable[..., Any], "RuntimeAdapterSpec"
] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class RuntimeAdapterRequest:
    """Runtime data needed to build an invocation-scoped adapter."""

    context: Any
    artifact_inputs: Mapping[str, ArtifactInputPlan] = field(default_factory=dict)
    artifact_outputs: Mapping[str, ArtifactOutputPlan] = field(default_factory=dict)
    source_binding_plan: CompiledSourceBindingPlan = CompiledSourceBindingPlan.empty()
    source_binding_context: SourceBindingRuntimeContext = (
        SourceBindingRuntimeContext.empty()
    )
    group_key: str | None = None
    plane_projection: RuntimePlaneProjection = field(
        default_factory=RuntimePlaneProjection.stack
    )


@dataclass(frozen=True, slots=True)
class RuntimeAdapterSpec:
    """Callable-owned runtime adapter injection contract."""

    parameter_name: str
    factory: Callable[[RuntimeAdapterRequest], Any]
    manages_artifact_inputs: bool = False

    def __post_init__(self) -> None:
        if not self.parameter_name:
            raise ValueError("RuntimeAdapterSpec.parameter_name cannot be empty.")
        if not callable(self.factory):
            raise TypeError("RuntimeAdapterSpec.factory must be callable.")


def runtime_adapter(
    parameter_name: str,
    factory: Callable[[RuntimeAdapterRequest], Any],
    *,
    manages_artifact_inputs: bool = False,
) -> Callable[[_F], _F]:
    """Declare that a callable needs an invocation-scoped runtime adapter."""
    spec = RuntimeAdapterSpec(
        parameter_name=parameter_name,
        factory=factory,
        manages_artifact_inputs=manages_artifact_inputs,
    )

    def decorator(func: _F) -> _F:
        _RUNTIME_ADAPTER_SPECS[func] = spec
        setattr(func, "__runtime_adapter__", spec)
        return func

    return decorator


def runtime_adapter_spec_from_callable(func: Any) -> RuntimeAdapterSpec | None:
    """Return the callable's declared runtime adapter contract, if any."""
    if callable(func):
        spec = _RUNTIME_ADAPTER_SPECS.get(func)
        if spec is not None:
            return spec
    fallback = _preserved_runtime_adapter_spec(func)
    if fallback is None:
        return None
    if isinstance(fallback, RuntimeAdapterSpec):
        return fallback
    raise TypeError(
        f"{type(func).__name__}.__runtime_adapter__ must be "
        f"RuntimeAdapterSpec, got {type(fallback).__name__}."
    )


def _preserved_runtime_adapter_spec(func: Any) -> Any:
    try:
        preserved_attrs = object.__getattribute__(func, "preserved_attrs")
    except AttributeError:
        return None
    if not isinstance(preserved_attrs, Mapping):
        raise TypeError(
            f"{type(func).__name__}.preserved_attrs must be Mapping, got "
            f"{type(preserved_attrs).__name__}."
        )
    return preserved_attrs.get("__runtime_adapter__")
