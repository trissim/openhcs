"""Shared benchmark value contracts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

BenchmarkScalarValue: TypeAlias = str | int | float | bool | None
BenchmarkParameterValue: TypeAlias = (
    BenchmarkScalarValue
    | Path
    | tuple["BenchmarkParameterValue", ...]
    | Mapping[str, "BenchmarkParameterValue"]
)
BenchmarkParameterMap: TypeAlias = Mapping[str, BenchmarkParameterValue]
BenchmarkMetricValue: TypeAlias = BenchmarkScalarValue
BenchmarkMetricMap: TypeAlias = Mapping[str, BenchmarkMetricValue]
BenchmarkProvenanceValue: TypeAlias = BenchmarkParameterValue
BenchmarkProvenanceMap: TypeAlias = Mapping[str, BenchmarkProvenanceValue]
