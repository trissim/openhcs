"""CellProfiler dialect support for OpenHCS.

This package owns product-facing CellProfiler semantics. Benchmark modules may
use these APIs to run parity checks, but benchmark code should not be the
canonical owner of `.cppipe` import or CellProfiler measurement semantics.
"""

from openhcs.interop.cellprofiler.measurement_dialect import (
    BENCHMARK_CACHE_DOMAINS,
    CELLPROFILER_FEATURE_NUMERIC_TOLERANCES,
    CELLPROFILER_MEASUREMENT_DIALECT,
    cellprofiler_runtime_equivalence_policy,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
    CellProfilerMeasurementTargetScope,
    coerce_cellprofiler_measurement_target_scope,
)

__all__ = (
    "BENCHMARK_CACHE_DOMAINS",
    "CELLPROFILER_FEATURE_NUMERIC_TOLERANCES",
    "CELLPROFILER_MEASUREMENT_DIALECT",
    "CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG",
    "CellProfilerMeasurementTargetScope",
    "cellprofiler_runtime_equivalence_policy",
    "coerce_cellprofiler_measurement_target_scope",
)

