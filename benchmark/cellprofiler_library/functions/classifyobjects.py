"""Converted from CellProfiler: ClassifyObjects."""

from openhcs.processing.backends.cellprofiler.classification import (
    CLASSIFICATION_RESULT_FIELDS,
    ClassificationBinChoice as BinChoice,
    ClassificationMethod,
    ClassificationResult,
    ClassificationThresholdMethod as ThresholdMethod,
    IntensityBinsClassificationRequest,
    ObjectClassificationBackendStrategy,
    SingleMeasurementClassificationRequest,
    TwoMeasurementClassificationRequest,
    classify_objects_by_intensity_bins,
    classify_objects_single_measurement,
    classify_objects_two_measurements,
    object_classification_backend,
)

__all__ = [
    "BinChoice",
    "CLASSIFICATION_RESULT_FIELDS",
    "ClassificationMethod",
    "ClassificationResult",
    "IntensityBinsClassificationRequest",
    "ObjectClassificationBackendStrategy",
    "SingleMeasurementClassificationRequest",
    "ThresholdMethod",
    "TwoMeasurementClassificationRequest",
    "classify_objects_by_intensity_bins",
    "classify_objects_single_measurement",
    "classify_objects_two_measurements",
    "object_classification_backend",
]
