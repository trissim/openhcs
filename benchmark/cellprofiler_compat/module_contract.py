"""Typed runtime contract for generated CellProfiler module execution."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.artifacts import ArtifactSpec


@dataclass(frozen=True, slots=True)
class CellProfilerModuleContract:
    """Typed execution contract consumed by CellProfilerModuleExecutor."""

    module_name: str
    inputs: tuple[ArtifactSpec, ...] = ()
    runtime_artifact_inputs: tuple[ArtifactSpec, ...] = ()
    outputs: tuple[ArtifactSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.module_name:
            raise ValueError(
                "CellProfilerModuleContract.module_name cannot be empty."
            )
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(
            self,
            "runtime_artifact_inputs",
            tuple(self.runtime_artifact_inputs),
        )
        object.__setattr__(self, "outputs", tuple(self.outputs))
        for field_name in ("inputs", "runtime_artifact_inputs", "outputs"):
            for spec in getattr(self, field_name):
                if not isinstance(spec, ArtifactSpec):
                    raise TypeError(
                        f"CellProfilerModuleContract.{field_name} must contain "
                        f"ArtifactSpec values, got {type(spec).__name__}."
                    )
