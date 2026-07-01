"""Pipeline contracts for benchmark platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from benchmark.contracts.values import BenchmarkParameterMap


def immutable_benchmark_parameters(
    values: BenchmarkParameterMap | None = None,
) -> BenchmarkParameterMap:
    """Return an immutable benchmark parameter mapping."""
    return MappingProxyType(dict(values or ()))


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """Immutable benchmark pipeline specification."""

    name: str
    description: str
    parameters: BenchmarkParameterMap = field(
        default_factory=immutable_benchmark_parameters
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            immutable_benchmark_parameters(self.parameters),
        )
