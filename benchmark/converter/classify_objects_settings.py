"""Compatibility aliases for CellProfiler ClassifyObjects settings."""

from openhcs.interop.cellprofiler.classify_objects_settings import (
    CLASSIFICATION_DECISION_COUNT_SETTING,
    FIRST_MEASUREMENT_FEATURE_SETTING,
    SECOND_MEASUREMENT_FEATURE_SETTING,
    SINGLE_MEASUREMENT_FEATURE_SETTING,
    ClassifyObjectsVariant,
    IndexedClassifySetting,
    TypedClassifySetting,
    classify_objects_bound_kwargs,
)

__all__ = (
    "CLASSIFICATION_DECISION_COUNT_SETTING",
    "FIRST_MEASUREMENT_FEATURE_SETTING",
    "SECOND_MEASUREMENT_FEATURE_SETTING",
    "SINGLE_MEASUREMENT_FEATURE_SETTING",
    "ClassifyObjectsVariant",
    "IndexedClassifySetting",
    "TypedClassifySetting",
    "classify_objects_bound_kwargs",
)
