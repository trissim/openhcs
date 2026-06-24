"""Typed runtime adapter injection contracts for callable execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from weakref import WeakKeyDictionary

from python_introspect import set_parameter_exclusions

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan
from openhcs.core.component_set import ComponentSet
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    SourceBindingRuntimeContext,
)
from openhcs.core.runtime_semantics import RuntimePlaneProjection

RuntimeComponentValue = str | int | float | bool | None

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.steps.function_runtime import FunctionRuntimeScope


class RuntimeAdapterValue(Protocol):
    """Nominal protocol for callable-owned runtime adapter instances."""


_F = TypeVar("_F", bound=Callable[..., Any])
_RUNTIME_ADAPTER_SPECS: WeakKeyDictionary[
    Callable[..., Any], "RuntimeAdapterSpec"
] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class RuntimeExecutionAxisScope:
    """Typed runtime axis coordinate for source-axis projection."""

    axis_id: str
    component: AllComponents | None = None
    value: RuntimeComponentValue | None = None

    @classmethod
    def from_context(
        cls,
        context: "ProcessingContext",
        *,
        component: AllComponents | str | None = None,
        value: RuntimeComponentValue | None = None,
    ) -> "RuntimeExecutionAxisScope":
        axis_id = context.axis_id
        if not axis_id:
            raise RuntimeError(
                "ProcessingContext.axis_id is required for runtime execution."
            )
        return cls.from_raw(
            str(axis_id),
            component=component,
            value=value,
        )

    @classmethod
    def from_raw(
        cls,
        axis_id: str,
        *,
        component: AllComponents | Enum | str | None,
        value: RuntimeComponentValue | None,
    ) -> "RuntimeExecutionAxisScope":
        if not axis_id:
            raise ValueError("RuntimeExecutionAxisScope.axis_id cannot be empty.")
        if component is None and value is not None:
            raise ValueError(
                "RuntimeExecutionAxisScope component value requires a component."
            )
        if component is not None and value is None:
            raise ValueError(
                "RuntimeExecutionAxisScope component requires a component value."
            )
        return cls(
            axis_id=str(axis_id),
            component=ComponentSet.coerce_component(component),
            value=value,
        ) if component is not None else cls(axis_id=str(axis_id))

    def __post_init__(self) -> None:
        if not self.axis_id:
            raise ValueError("RuntimeExecutionAxisScope.axis_id cannot be empty.")
        if self.component is None and self.value is not None:
            raise ValueError(
                "RuntimeExecutionAxisScope component value requires a component."
            )
        if self.component is not None and self.value is None:
            raise ValueError(
                "RuntimeExecutionAxisScope component requires a component value."
            )

    @property
    def component_name(self) -> str | None:
        if self.component is None:
            return None
        return str(self.component.value)

    def require_component_name(self) -> str:
        component_name = self.component_name
        if component_name is None:
            raise ValueError("Runtime component-axis scope has no component.")
        return component_name

    @property
    def value_text(self) -> str | None:
        if self.value is None:
            return None
        return str(self.value)

    def require_value_text(self) -> str:
        value_text = self.value_text
        if value_text is None:
            raise ValueError("Runtime component-axis scope has no value.")
        return value_text

    @property
    def has_value(self) -> bool:
        return self.component is not None and self.value is not None

    @property
    def cache_key(self) -> tuple[str | None, str | None]:
        return (self.component_name, self.value_text)

    def source_axis_metadata_scope(self) -> "SourceAxisMetadataScope":
        """Return metadata constraints for this runtime axis."""
        from openhcs.constants.constants import get_multiprocessing_axis
        from openhcs.core.source_matching import SourceAxisMetadataScope

        component_values: list[tuple[str | None, str]] = [
            (str(get_multiprocessing_axis().value), self.axis_id),
        ]
        component_name = self.component_name
        if component_name is not None:
            component_values.append(
                (
                    component_name,
                    self.require_value_text(),
                )
            )
        return SourceAxisMetadataScope.from_component_values(tuple(component_values))


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeAdapterRequest(SourceBindingRuntimeContext):
    """Runtime data needed to build an invocation-scoped adapter."""

    context: "ProcessingContext"
    artifact_inputs: Mapping[str, ArtifactInputPlan] = field(default_factory=dict)
    artifact_outputs: Mapping[str, ArtifactOutputPlan] = field(default_factory=dict)
    source_binding_plan: CompiledSourceBindingPlan = CompiledSourceBindingPlan.empty()
    group_key: str | None = None
    axis_scope: RuntimeExecutionAxisScope
    plane_projection: RuntimePlaneProjection = field(
        default_factory=RuntimePlaneProjection.stack
    )
    source_identity_stack_axes: frozenset[str] = frozenset()

    @classmethod
    def from_source_context(
        cls,
        *,
        context: "ProcessingContext",
        artifact_inputs: Mapping[str, ArtifactInputPlan],
        artifact_outputs: Mapping[str, ArtifactOutputPlan],
        source_binding_plan: CompiledSourceBindingPlan,
        source_binding_context: SourceBindingRuntimeContext,
        group_key: str | None = None,
        axis_scope: RuntimeExecutionAxisScope | None = None,
        plane_projection: RuntimePlaneProjection | None = None,
        source_identity_stack_axes: frozenset[str] = frozenset(),
    ) -> "RuntimeAdapterRequest":
        """Project a source-binding runtime context into an adapter request."""
        return cls(
            context=context,
            artifact_inputs=artifact_inputs,
            artifact_outputs=artifact_outputs,
            source_binding_plan=source_binding_plan,
            source_binding_context=source_binding_context,
            group_key=group_key,
            axis_scope=(
                axis_scope
                if axis_scope is not None
                else RuntimeExecutionAxisScope.from_context(context)
            ),
            plane_projection=(
                plane_projection
                if plane_projection is not None
                else RuntimePlaneProjection.stack()
            ),
            source_identity_stack_axes=source_identity_stack_axes,
        )

    @classmethod
    def from_runtime_scope(
        cls,
        *,
        runtime_scope: "FunctionRuntimeScope",
        artifact_inputs: Mapping[str, ArtifactInputPlan],
        artifact_outputs: Mapping[str, ArtifactOutputPlan],
        group_key: str | None,
        plane_projection: RuntimePlaneProjection,
    ) -> "RuntimeAdapterRequest":
        """Project an invocation runtime scope into an adapter request."""
        return cls.from_source_context(
            context=runtime_scope.context,
            artifact_inputs=artifact_inputs,
            artifact_outputs=artifact_outputs,
            source_binding_plan=runtime_scope.source_binding_plan,
            source_binding_context=runtime_scope,
            group_key=group_key,
            axis_scope=runtime_scope.axis_scope,
            plane_projection=plane_projection,
            source_identity_stack_axes=(
                runtime_scope.execution_plan.source_identity_stack_axes
            ),
        )

    @property
    def source_binding_context(self) -> SourceBindingRuntimeContext:
        """Return the source-binding runtime context owned by this request."""
        return self


RuntimeAdapterFactory = Callable[[RuntimeAdapterRequest], RuntimeAdapterValue]
RuntimeAdapterPrepare = Callable[[RuntimeAdapterRequest], None]


@dataclass(frozen=True, slots=True)
class RuntimeAdapterSpec:
    """Callable-owned runtime adapter injection contract."""

    parameter_name: str
    factory: RuntimeAdapterFactory
    manages_artifact_inputs: bool = False
    prepare: RuntimeAdapterPrepare | None = None

    def __post_init__(self) -> None:
        if not self.parameter_name:
            raise ValueError("RuntimeAdapterSpec.parameter_name cannot be empty.")
        if not callable(self.factory):
            raise TypeError("RuntimeAdapterSpec.factory must be callable.")
        if self.prepare is not None and not callable(self.prepare):
            raise TypeError("RuntimeAdapterSpec.prepare must be callable or None.")

    def prepare_request(self, request: RuntimeAdapterRequest) -> None:
        """Run the adapter's optional compile-time preparation hook."""
        if self.prepare is None:
            return
        self.prepare(request)


def runtime_adapter(
    parameter_name: str,
    factory: RuntimeAdapterFactory,
    *,
    manages_artifact_inputs: bool = False,
    prepare: RuntimeAdapterPrepare | None = None,
) -> Callable[[_F], _F]:
    """Declare that a callable needs an invocation-scoped runtime adapter."""
    spec = RuntimeAdapterSpec(
        parameter_name=parameter_name,
        factory=factory,
        manages_artifact_inputs=manages_artifact_inputs,
        prepare=prepare,
    )

    def decorator(func: _F) -> _F:
        _RUNTIME_ADAPTER_SPECS[func] = spec
        namespace = vars(func)
        if not isinstance(namespace, MutableMapping):
            raise TypeError(f"{func!r} does not expose a mutable metadata namespace.")
        namespace["__runtime_adapter__"] = spec
        set_parameter_exclusions(func, (parameter_name,))
        return func

    return decorator


def runtime_adapter_spec_from_callable(func: Any) -> RuntimeAdapterSpec | None:
    """Return the callable's declared runtime adapter contract, if any."""
    if callable(func):
        spec = _RUNTIME_ADAPTER_SPECS.get(func)
        if spec is not None:
            return spec
    reference_spec = _function_reference_runtime_adapter(func)
    if reference_spec is None:
        return None
    return reference_spec


def _function_reference_runtime_adapter(func: object) -> RuntimeAdapterSpec | None:
    from openhcs.core.function_reference import FunctionReference

    if not isinstance(func, FunctionReference):
        return None
    return func.metadata.runtime_adapter
