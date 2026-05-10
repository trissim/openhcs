"""Compatibility exports for CellProfiler measurement lookup semantics."""

from openhcs.interop.cellprofiler.measurement_lookup import (
    CellProfilerChildCountFeatureParser,
    CellProfilerMeasurementFeature,
    CellProfilerMeasurementFeatureKind,
    CellProfilerMeasurementFeatureParser,
    CellProfilerObjectCountFeatureParser,
    child_count_feature_child_name,
    count_feature_object_name,
    measurement_scalar_value_for_feature,
    measurement_values_for_feature,
    measurement_values_for_label_slices,
)

__all__ = (
    "CellProfilerChildCountFeatureParser",
    "CellProfilerMeasurementFeature",
    "CellProfilerMeasurementFeatureKind",
    "CellProfilerMeasurementFeatureParser",
    "CellProfilerObjectCountFeatureParser",
    "child_count_feature_child_name",
    "count_feature_object_name",
    "measurement_scalar_value_for_feature",
    "measurement_values_for_feature",
    "measurement_values_for_label_slices",
)
