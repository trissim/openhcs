"""CellProfiler interop namespace ownership tests."""

from benchmark.cellprofiler_compat import measurement_dialect as benchmark_dialect
from benchmark.cellprofiler_compat import measurement_scope as benchmark_scope
from openhcs.interop import cellprofiler
from openhcs.interop.cellprofiler import measurement_dialect
from openhcs.interop.cellprofiler import measurement_scope


def test_cellprofiler_measurement_dialect_has_product_namespace_owner():
    """The product namespace owns CellProfiler measurement equivalence semantics."""
    assert (
        cellprofiler.CELLPROFILER_MEASUREMENT_DIALECT
        is measurement_dialect.CELLPROFILER_MEASUREMENT_DIALECT
    )
    assert (
        benchmark_dialect.CELLPROFILER_MEASUREMENT_DIALECT
        is measurement_dialect.CELLPROFILER_MEASUREMENT_DIALECT
    )
    assert (
        benchmark_dialect.cellprofiler_runtime_equivalence_policy
        is measurement_dialect.cellprofiler_runtime_equivalence_policy
    )


def test_cellprofiler_measurement_scope_has_product_namespace_owner():
    """Benchmark scope imports remain compatibility aliases."""
    assert (
        cellprofiler.CellProfilerMeasurementTargetScope
        is measurement_scope.CellProfilerMeasurementTargetScope
    )
    assert (
        benchmark_scope.CellProfilerMeasurementTargetScope
        is measurement_scope.CellProfilerMeasurementTargetScope
    )
    assert (
        benchmark_scope.CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG
        == measurement_scope.CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG
    )

