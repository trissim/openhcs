"""Thin CellProfiler measurement-name compatibility over core runtime queries."""

from __future__ import annotations

from openhcs.core.runtime_artifact_queries import (
    measurement_scalar_value_for_feature,
    measurement_values_for_feature,
    measurement_values_for_label_slices,
)

CELLPROFILER_OBJECT_COUNT_FEATURE_PREFIX = "Count_"

__all__ = (
    "CELLPROFILER_OBJECT_COUNT_FEATURE_PREFIX",
    "count_feature_object_name",
    "measurement_scalar_value_for_feature",
    "measurement_values_for_feature",
    "measurement_values_for_label_slices",
)


def count_feature_object_name(feature_name: str | None) -> str | None:
    """Return the object-set name encoded by a CellProfiler Count_* feature."""
    if feature_name is None:
        return None
    if not feature_name.startswith(CELLPROFILER_OBJECT_COUNT_FEATURE_PREFIX):
        return None
    object_name = feature_name[len(CELLPROFILER_OBJECT_COUNT_FEATURE_PREFIX):].strip()
    return object_name or None
