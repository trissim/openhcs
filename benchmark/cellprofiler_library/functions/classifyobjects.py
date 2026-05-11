"""
Converted from CellProfiler: ClassifyObjects
Original: ClassifyObjects module
"""

from typing import Any, Optional, Tuple

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.classification import (
    ClassificationBinChoice as BinChoice,
    ClassificationMethod,
    ClassificationResult,
    ClassificationThresholdMethod as ThresholdMethod,
    IntensityBinsClassificationRequest,
    ObjectClassificationBackendStrategy,
    SingleMeasurementClassificationRequest,
    TwoMeasurementClassificationRequest,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


CLASSIFICATION_RESULT_FIELDS = [
    "slice_index",
    "total_objects",
    "bin_counts",
    "bin_percentages",
    "object_classes",
]


def _classification_backend(
    classification_backend_provider: BackendProviderInput,
) -> ObjectClassificationBackendStrategy:
    return ObjectClassificationBackendStrategy.for_memory_type(
        backend_provider=classification_backend_provider,
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "classification_results",
        csv_materializer(
            fields=CLASSIFICATION_RESULT_FIELDS,
            analysis_type="classification",
        ),
    )
)
def classify_objects_single_measurement(
    image: np.ndarray,
    labels: np.ndarray,
    measurement_values: Optional[np.ndarray] = None,
    measurement_values_by_rule: tuple[np.ndarray, ...] = (),
    classification_rules: tuple[dict[str, Any], ...] = (),
    bin_choice: BinChoice = BinChoice.EVEN,
    bin_count: int = 3,
    low_threshold: float = 0.0,
    high_threshold: float = 1.0,
    wants_low_bin: bool = False,
    wants_high_bin: bool = False,
    custom_thresholds: str = "0,1",
    bin_names: Optional[str] = None,
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[np.ndarray, ClassificationResult]:
    """Classify objects based on a single measurement into bins."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    backend = _classification_backend(classification_backend_provider)
    if classification_rules:
        results: list[ClassificationResult] = []
        classified_labels = labels
        for rule_index, rule in enumerate(classification_rules):
            rule_values = (
                measurement_values_by_rule[rule_index]
                if rule_index < len(measurement_values_by_rule)
                else None
            )
            classified_labels, result = SingleMeasurementClassificationRequest(
                measurement_values=rule_values,
                bin_choice=rule.get("bin_choice", BinChoice.EVEN),
                bin_count=int(rule.get("bin_count", 3)),
                low_threshold=float(rule.get("low_threshold", 0.0)),
                high_threshold=float(rule.get("high_threshold", 1.0)),
                wants_low_bin=bool(rule.get("wants_low_bin", False)),
                wants_high_bin=bool(rule.get("wants_high_bin", False)),
                custom_thresholds=str(rule.get("custom_thresholds", "0,1")),
                bin_names=rule.get("bin_names"),
            ).classify(image, labels, backend)
            results.append(result)
        return classified_labels, tuple(results)

    return SingleMeasurementClassificationRequest(
        measurement_values=measurement_values,
        bin_choice=bin_choice,
        bin_count=bin_count,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        wants_low_bin=wants_low_bin,
        wants_high_bin=wants_high_bin,
        custom_thresholds=custom_thresholds,
        bin_names=bin_names,
    ).classify(image, labels, backend)


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "classification_results",
        csv_materializer(
            fields=CLASSIFICATION_RESULT_FIELDS,
            analysis_type="classification",
        ),
    )
)
def classify_objects_two_measurements(
    image: np.ndarray,
    labels: np.ndarray,
    measurement1_values: Optional[np.ndarray] = None,
    measurement2_values: Optional[np.ndarray] = None,
    threshold1_method: ThresholdMethod = ThresholdMethod.MEAN,
    threshold1_value: float = 0.5,
    threshold2_method: ThresholdMethod = ThresholdMethod.MEAN,
    threshold2_value: float = 0.5,
    low_low_name: str = "low_low",
    low_high_name: str = "low_high",
    high_low_name: str = "high_low",
    high_high_name: str = "high_high",
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[np.ndarray, ClassificationResult]:
    """Classify objects based on two measurements into four quadrants."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    return TwoMeasurementClassificationRequest(
        measurement1_values=measurement1_values,
        measurement2_values=measurement2_values,
        threshold1_method=threshold1_method,
        threshold1_value=threshold1_value,
        threshold2_method=threshold2_method,
        threshold2_value=threshold2_value,
        low_low_name=low_low_name,
        low_high_name=low_high_name,
        high_low_name=high_low_name,
        high_high_name=high_high_name,
    ).classify(image, labels, _classification_backend(classification_backend_provider))


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "classification_results",
        csv_materializer(
            fields=CLASSIFICATION_RESULT_FIELDS,
            analysis_type="classification",
        ),
    )
)
def classify_objects_by_intensity_bins(
    image: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 3,
    use_percentiles: bool = True,
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[np.ndarray, ClassificationResult]:
    """Classify objects by mean intensity into evenly distributed bins."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    return IntensityBinsClassificationRequest(
        num_bins=num_bins,
        use_percentiles=use_percentiles,
    ).classify(image, labels, _classification_backend(classification_backend_provider))
