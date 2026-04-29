"""Typed callable contracts used by compiler phases.

This module centralizes metadata extraction from processing callables so the
compiler has one source of truth for memory and artifact declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.runtime_adapters import (
    RuntimeAdapterSpec,
    runtime_adapter_spec_from_callable,
)


ArtifactSpecItems = tuple[tuple[str, ArtifactSpec], ...]
CallableNamespace = Mapping[str, Any]
PROCESSING_CONTRACT_ATTR = "__processing_contract__"
DECLARED_PROCESSING_CONTRACT_ATTR = "__openhcs_declared_processing_contract__"
RAW_PROCESSING_FUNCTION_ATTR = "__openhcs_raw_processing_function__"


@dataclass(frozen=True, slots=True)
class CallableContract:
    """Compiler-visible contract declared by one processing callable."""

    func: Any
    function_name: str
    module_name: str | None
    input_memory_type: str | None
    output_memory_type: str | None
    artifact_inputs: ArtifactSpecItems = ()
    artifact_outputs: ArtifactSpecItems = ()
    runtime_adapter: RuntimeAdapterSpec | None = None
    processing_contract: Any | None = None
    declared_processing_contract: str | None = None
    raw_processing_function: Any | None = None

    @classmethod
    def from_callable(cls, func: Any) -> "CallableContract":
        """Build a contract from callable attributes once at compiler boundary."""
        namespace = _callable_namespace(func)
        function_name = _callable_name(func)
        return cls(
            func=func,
            function_name=function_name,
            module_name=_callable_module(func),
            input_memory_type=_optional_memory_type(
                namespace,
                function_name,
                "input_memory_type",
            ),
            output_memory_type=_optional_memory_type(
                namespace,
                function_name,
                "output_memory_type",
            ),
            artifact_inputs=_artifact_spec_items(
                namespace,
                function_name,
                "__artifact_inputs__",
            ),
            artifact_outputs=_artifact_spec_items(
                namespace,
                function_name,
                "__artifact_outputs__",
            ),
            runtime_adapter=runtime_adapter_spec_from_callable(func),
            processing_contract=namespace.get(PROCESSING_CONTRACT_ATTR),
            declared_processing_contract=_optional_string(
                namespace,
                function_name,
                DECLARED_PROCESSING_CONTRACT_ATTR,
            ),
            raw_processing_function=namespace.get(RAW_PROCESSING_FUNCTION_ATTR),
        )

    @property
    def artifact_input_names(self) -> tuple[str, ...]:
        """Declared artifact input names in declaration order."""
        return tuple(name for name, _ in self.artifact_inputs)

    @property
    def artifact_output_names(self) -> tuple[str, ...]:
        """Declared artifact output names in declaration order."""
        return tuple(name for name, _ in self.artifact_outputs)

    @property
    def artifact_inputs_dict(self) -> dict[str, ArtifactSpec]:
        """Return declared artifact inputs as a runtime mapping."""
        return dict(self.artifact_inputs)

    @property
    def artifact_outputs_dict(self) -> dict[str, ArtifactSpec]:
        """Return declared artifact outputs as a runtime mapping."""
        return dict(self.artifact_outputs)

    def select_input_plan_keys(
        self,
        input_plans: Mapping[str, ArtifactInputPlan],
    ) -> tuple[str, ...]:
        """Select compiled artifact inputs consumed by this callable."""
        declared = set(self.artifact_input_names)
        return tuple(key for key in input_plans if key in declared)

    def select_output_plan_keys(
        self,
        output_plans: Mapping[str, ArtifactOutputPlan],
    ) -> tuple[str, ...]:
        """Select compiled artifact outputs produced by this callable."""
        declared = set(self.artifact_output_names)
        return tuple(key for key in output_plans if key in declared)


def _callable_namespace(func: Any) -> CallableNamespace:
    """Return user-declared callable metadata."""
    if _is_function_reference(func):
        return func.preserved_attrs
    return func.__dict__


def attach_callable_contract_metadata(
    func: Any,
    *,
    declared_processing_contract: str | None = None,
    raw_processing_function: Any | None = None,
) -> None:
    """Attach OpenHCS callable metadata used by compiler/runtime phases."""
    if declared_processing_contract is not None:
        if (
            not isinstance(declared_processing_contract, str)
            or not declared_processing_contract.strip()
        ):
            raise ValueError(
                "declared_processing_contract must be a non-empty string."
            )
        setattr(
            func,
            DECLARED_PROCESSING_CONTRACT_ATTR,
            declared_processing_contract,
        )
    if raw_processing_function is not None:
        if not callable(raw_processing_function):
            raise TypeError(
                "raw_processing_function must be callable, "
                f"got {type(raw_processing_function).__name__}."
            )
        setattr(func, RAW_PROCESSING_FUNCTION_ATTR, raw_processing_function)


def _callable_name(func: Any) -> str:
    """Return the callable's nominal function name."""
    name = func.function_name if _is_function_reference(func) else func.__name__
    if not isinstance(name, str):
        raise TypeError(f"Callable name must be a string, got {type(name).__name__}.")
    return name


def _callable_module(func: Any) -> str | None:
    """Return the callable's declaring module when available."""
    module_name = (
        func.original_module
        if _is_function_reference(func)
        else func.__module__
    )
    if module_name is None or isinstance(module_name, str):
        return module_name
    raise TypeError(
        f"{_callable_name(func)!r}.__module__ must be a string or None, "
        f"got {type(module_name).__name__}."
    )


def _is_function_reference(func: Any) -> bool:
    """Return whether func is the compiler's nominal picklable reference."""
    from openhcs.core.pipeline.compiler import FunctionReference

    return isinstance(func, FunctionReference)


def _optional_memory_type(
    namespace: CallableNamespace,
    function_name: str,
    field_name: str,
) -> str | None:
    memory_type = namespace.get(field_name)
    if memory_type is None:
        return None
    if not isinstance(memory_type, str):
        raise TypeError(
            f"{function_name!r}.{field_name} must be a string, "
            f"got {type(memory_type).__name__}."
        )
    return memory_type


def _optional_string(
    namespace: CallableNamespace,
    function_name: str,
    field_name: str,
) -> str | None:
    value = namespace.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"{function_name!r}.{field_name} must be a string, "
            f"got {type(value).__name__}."
        )
    return value


def _artifact_spec_items(
    namespace: CallableNamespace,
    function_name: str,
    attr_name: str,
) -> ArtifactSpecItems:
    raw_specs = namespace.get(attr_name)
    if not raw_specs:
        return ()
    if not isinstance(raw_specs, Mapping):
        raise TypeError(
            f"{function_name!r}.{attr_name} must be a mapping, "
            f"got {type(raw_specs).__name__}."
        )

    items: list[tuple[str, ArtifactSpec]] = []
    for name, spec in raw_specs.items():
        if not isinstance(name, str):
            raise TypeError(
                f"{function_name!r}.{attr_name} contains a non-string "
                f"artifact name: {name!r}."
            )
        if not isinstance(spec, ArtifactSpec):
            raise TypeError(
                f"{function_name!r}.{attr_name}['{name}'] "
                f"must be ArtifactSpec, got {type(spec).__name__}."
            )
        if spec.name != name:
            raise ValueError(
                f"{function_name!r}.{attr_name} key '{name}' "
                f"does not match ArtifactSpec.name '{spec.name}'."
            )
        items.append((name, spec))
    return tuple(items)
