"""Typed callable contracts used by compiler phases.

This module centralizes metadata extraction from processing callables so the
compiler has one source of truth for memory and artifact declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan, ArtifactSpec


ArtifactSpecItems = tuple[tuple[str, ArtifactSpec], ...]


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

    @classmethod
    def from_callable(cls, func: Any) -> "CallableContract":
        """Build a contract from callable attributes once at compiler boundary."""
        return cls(
            func=func,
            function_name=getattr(func, "__name__", "unknown"),
            module_name=getattr(func, "__module__", None),
            input_memory_type=getattr(func, "input_memory_type", None),
            output_memory_type=getattr(func, "output_memory_type", None),
            artifact_inputs=_artifact_spec_items(func, "__artifact_inputs__"),
            artifact_outputs=_artifact_spec_items(func, "__artifact_outputs__"),
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


def _artifact_spec_items(func: Any, attr_name: str) -> ArtifactSpecItems:
    raw_specs = getattr(func, attr_name, None)
    if not raw_specs:
        return ()
    if not isinstance(raw_specs, Mapping):
        raise TypeError(
            f"{getattr(func, '__name__', func)!r}.{attr_name} must be a mapping, "
            f"got {type(raw_specs).__name__}."
        )

    items: list[tuple[str, ArtifactSpec]] = []
    for name, spec in raw_specs.items():
        if not isinstance(name, str):
            raise TypeError(
                f"{getattr(func, '__name__', func)!r}.{attr_name} contains "
                f"a non-string artifact name: {name!r}."
            )
        if not isinstance(spec, ArtifactSpec):
            raise TypeError(
                f"{getattr(func, '__name__', func)!r}.{attr_name}['{name}'] "
                f"must be ArtifactSpec, got {type(spec).__name__}."
            )
        if spec.name != name:
            raise ValueError(
                f"{getattr(func, '__name__', func)!r}.{attr_name} key '{name}' "
                f"does not match ArtifactSpec.name '{spec.name}'."
            )
        items.append((name, spec))
    return tuple(items)
