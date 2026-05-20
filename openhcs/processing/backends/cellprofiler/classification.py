"""Classification backends for CellProfiler-compatible object measurements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_semantics import ObjectLabelMeasurementValues
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer

CLASSIFICATION_RESULT_FIELDS = [
    "slice_index",
    "total_objects",
    "bin_counts",
    "bin_percentages",
    "object_classes",
]


class ClassificationMethod(Enum):
    """CellProfiler ClassifyObjects measurement-count mode."""

    SINGLE_MEASUREMENT = "single_measurement"
    TWO_MEASUREMENTS = "two_measurements"


class ClassificationThresholdMethod(Enum):
    """CellProfiler ClassifyObjects threshold selection mode."""

    MEAN = "mean"
    MEDIAN = "median"
    CUSTOM = "custom"


class ClassificationBinChoice(Enum):
    """CellProfiler ClassifyObjects bin spacing mode."""

    EVEN = "even"
    CUSTOM = "custom"


class ClassificationThresholdStrategy(ABC, metaclass=AutoRegisterMeta):
    """Threshold calculation strategy for ClassifyObjects measurement vectors."""

    __registry_key__ = "method"
    __skip_if_no_key__ = True
    method: ClassVar[ClassificationThresholdMethod | None] = None

    @classmethod
    def for_method(
        cls,
        method: ClassificationThresholdMethod,
    ) -> "ClassificationThresholdStrategy":
        return cls.__registry__[method]()

    def threshold(self, values: np.ndarray, custom_value: float) -> float:
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return custom_value
        return self._threshold(valid_values, custom_value)

    @abstractmethod
    def _threshold(self, valid_values: np.ndarray, custom_value: float) -> float:
        """Return a threshold for finite measurement values."""


class MeanClassificationThresholdStrategy(ClassificationThresholdStrategy):
    """Mean-based ClassifyObjects threshold."""

    method = ClassificationThresholdMethod.MEAN

    def _threshold(self, valid_values: np.ndarray, custom_value: float) -> float:
        del custom_value
        return float(np.mean(valid_values))


class MedianClassificationThresholdStrategy(ClassificationThresholdStrategy):
    """Median-based ClassifyObjects threshold."""

    method = ClassificationThresholdMethod.MEDIAN

    def _threshold(self, valid_values: np.ndarray, custom_value: float) -> float:
        del custom_value
        return float(np.median(valid_values))


class CustomClassificationThresholdStrategy(ClassificationThresholdStrategy):
    """User-specified ClassifyObjects threshold."""

    method = ClassificationThresholdMethod.CUSTOM

    def _threshold(self, valid_values: np.ndarray, custom_value: float) -> float:
        del valid_values
        return custom_value


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Results from object classification."""

    slice_index: int
    total_objects: int
    bin_counts: str
    bin_percentages: str
    object_classes: str = "{}"

    @classmethod
    def empty(cls, *, total_objects: int = 0) -> "ClassificationResult":
        """Return an empty classification result row."""
        return cls(
            slice_index=0,
            total_objects=total_objects,
            bin_counts=json.dumps({}),
            bin_percentages=json.dumps({}),
        )


@dataclass(frozen=True, slots=True)
class ClassificationMeasurementVector:
    """Measurement vector normalized to the current object-label domain."""

    values: np.ndarray

    @classmethod
    def from_value(
        cls,
        values: np.ndarray,
    ) -> "ClassificationMeasurementVector":
        return cls(np.asarray(values, dtype=np.float64).reshape(-1))

    def aligned_to_labels(self, label_ids: np.ndarray) -> np.ndarray:
        """Return values ordered like the materially present object labels."""
        if label_ids.size == 0:
            return np.zeros(0, dtype=np.float64)
        if self.values.size == label_ids.size:
            return self.values.copy()

        max_label = int(label_ids[-1])
        if self.values.size >= max_label and max_label > label_ids.size:
            return ObjectLabelMeasurementValues.from_label_indexed_values(
                tuple(int(label_id) for label_id in label_ids),
                self.values,
            ).values

        aligned = np.full(label_ids.size, np.nan, dtype=np.float64)
        copied = min(self.values.size, aligned.size)
        if copied:
            aligned[:copied] = self.values[:copied]
        return aligned


@dataclass(frozen=True, slots=True)
class SingleMeasurementClassificationRequest:
    """Semantic request for single-measurement object classification."""

    measurement_values: np.ndarray | None = None
    bin_choice: ClassificationBinChoice | str = ClassificationBinChoice.EVEN
    bin_count: int = 3
    low_threshold: float = 0.0
    high_threshold: float = 1.0
    wants_low_bin: bool = False
    wants_high_bin: bool = False
    custom_thresholds: str = "0,1"
    bin_names: str | None = None

    def classify(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        backend: ObjectClassificationBackendStrategy,
    ) -> tuple[np.ndarray, ClassificationResult]:
        bin_choice = coerce_cellprofiler_enum(ClassificationBinChoice, self.bin_choice)
        unique_labels = backend.positive_label_ids(labels)
        num_objects = len(unique_labels)
        if num_objects == 0:
            return labels, ClassificationResult.empty()

        if self.measurement_values is None:
            values = backend.mean_intensity_values(labels, image, unique_labels)
        else:
            values = ClassificationMeasurementVector.from_value(
                self.measurement_values
            ).aligned_to_labels(unique_labels)

        if bin_choice == ClassificationBinChoice.EVEN:
            low_threshold = self.low_threshold
            high_threshold = self.high_threshold
            if low_threshold >= high_threshold:
                low_threshold, high_threshold = high_threshold, low_threshold
            thresholds = np.linspace(low_threshold, high_threshold, self.bin_count + 1)
        else:
            thresholds = np.array(
                [float(x.strip()) for x in self.custom_thresholds.split(",")]
            )

        threshold_list = []
        if self.wants_low_bin:
            threshold_list.append(-np.inf)
        threshold_list.extend(thresholds.tolist())
        if self.wants_high_bin:
            threshold_list.append(np.inf)
        thresholds = np.array(threshold_list)

        num_bins = len(thresholds) - 1
        if self.bin_names is not None:
            names = [name.strip() for name in self.bin_names.split(",")]
        else:
            names = [f"Bin_{index + 1}" for index in range(num_bins)]

        while len(names) < num_bins:
            names.append(f"Bin_{len(names) + 1}")

        object_bins = np.zeros(num_objects, dtype=np.int32)
        for index, value in enumerate(values):
            if np.isnan(value):
                object_bins[index] = 0
            else:
                for bin_index in range(num_bins):
                    if thresholds[bin_index] < value <= thresholds[bin_index + 1]:
                        object_bins[index] = bin_index + 1
                        break

        return labels, classification_result_from_bins(
            unique_labels,
            object_bins,
            names,
        )


@dataclass(frozen=True, slots=True)
class TwoMeasurementClassificationRequest:
    """Semantic request for two-measurement object classification."""

    measurement1_values: np.ndarray | None = None
    measurement2_values: np.ndarray | None = None
    threshold1_method: ClassificationThresholdMethod | str = (
        ClassificationThresholdMethod.MEAN
    )
    threshold1_value: float = 0.5
    threshold2_method: ClassificationThresholdMethod | str = (
        ClassificationThresholdMethod.MEAN
    )
    threshold2_value: float = 0.5
    low_low_name: str = "low_low"
    low_high_name: str = "low_high"
    high_low_name: str = "high_low"
    high_high_name: str = "high_high"

    def classify(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        backend: ObjectClassificationBackendStrategy,
    ) -> tuple[np.ndarray, ClassificationResult]:
        threshold1_method = coerce_cellprofiler_enum(
            ClassificationThresholdMethod,
            self.threshold1_method,
        )
        threshold2_method = coerce_cellprofiler_enum(
            ClassificationThresholdMethod,
            self.threshold2_method,
        )
        unique_labels = backend.positive_label_ids(labels)
        num_objects = len(unique_labels)
        if num_objects == 0:
            return labels, ClassificationResult.empty()

        if self.measurement1_values is None:
            values1 = backend.mean_intensity_values(labels, image, unique_labels)
        else:
            values1 = ClassificationMeasurementVector.from_value(
                self.measurement1_values
            ).aligned_to_labels(unique_labels)

        if self.measurement2_values is None:
            values2 = np.bincount(
                labels.astype(np.intp, copy=False).ravel(),
                minlength=(int(unique_labels[-1]) + 1 if num_objects else 1),
            )[unique_labels].astype(float)
        else:
            values2 = ClassificationMeasurementVector.from_value(
                self.measurement2_values
            ).aligned_to_labels(unique_labels)

        t1 = classification_threshold(
            values1,
            threshold1_method,
            self.threshold1_value,
        )
        t2 = classification_threshold(
            values2,
            threshold2_method,
            self.threshold2_value,
        )

        high1 = values1 >= t1
        high2 = values2 >= t2
        has_nan = np.isnan(values1) | np.isnan(values2)

        object_class = np.zeros(num_objects, dtype=np.int32)
        object_class[(~high1) & (~high2) & (~has_nan)] = 1
        object_class[(high1) & (~high2) & (~has_nan)] = 2
        object_class[(~high1) & (high2) & (~has_nan)] = 3
        object_class[(high1) & (high2) & (~has_nan)] = 4

        names = [
            self.low_low_name,
            self.high_low_name,
            self.low_high_name,
            self.high_high_name,
        ]
        return labels, classification_result_from_bins(unique_labels, object_class, names)


@dataclass(frozen=True, slots=True)
class IntensityBinsClassificationRequest:
    """Semantic request for intensity-bin object classification."""

    num_bins: int = 3
    use_percentiles: bool = True

    def classify(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        backend: ObjectClassificationBackendStrategy,
    ) -> tuple[np.ndarray, ClassificationResult]:
        unique_labels = backend.positive_label_ids(labels)
        num_objects = len(unique_labels)
        if num_objects == 0:
            return labels, ClassificationResult.empty()

        values = backend.mean_intensity_values(labels, image, unique_labels)
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        if len(valid_values) == 0:
            return labels, ClassificationResult.empty(total_objects=num_objects)

        if self.use_percentiles:
            percentiles = np.linspace(0, 100, self.num_bins + 1)
            thresholds = np.percentile(valid_values, percentiles)
        else:
            thresholds = np.linspace(
                np.min(valid_values),
                np.max(valid_values),
                self.num_bins + 1,
            )

        object_bins = np.zeros(num_objects, dtype=np.int32)
        for index, value in enumerate(values):
            if np.isnan(value):
                continue
            for bin_index in range(self.num_bins):
                if bin_index == self.num_bins - 1:
                    if thresholds[bin_index] <= value <= thresholds[bin_index + 1]:
                        object_bins[index] = bin_index + 1
                else:
                    if thresholds[bin_index] <= value < thresholds[bin_index + 1]:
                        object_bins[index] = bin_index + 1
                        break

        bin_names = [f"Intensity_Bin_{index + 1}" for index in range(self.num_bins)]
        return labels, classification_result_from_bins(
            unique_labels,
            object_bins,
            bin_names,
        )


def classification_threshold(
    values: np.ndarray,
    method: ClassificationThresholdMethod,
    custom_value: float,
) -> float:
    """Return the threshold for one ClassifyObjects measurement vector."""
    method = coerce_cellprofiler_enum(ClassificationThresholdMethod, method)
    return ClassificationThresholdStrategy.for_method(method).threshold(
        values,
        custom_value,
    )


def classification_result_from_bins(
    unique_labels: np.ndarray,
    object_bins: np.ndarray,
    names: list[str],
) -> ClassificationResult:
    """Return serialized ClassifyObjects measurement rows from bin ids."""
    num_objects = len(unique_labels)
    bin_counts: dict[str, int] = {}
    bin_percentages: dict[str, float] = {}
    for bin_index, name in enumerate(names):
        count = np.sum(object_bins == (bin_index + 1))
        bin_counts[name] = int(count)
        bin_percentages[name] = (
            float(count / num_objects * 100) if num_objects > 0 else 0.0
        )

    object_classes: dict[int, str] = {}
    for index, label_value in enumerate(unique_labels):
        if object_bins[index] > 0:
            object_classes[int(label_value)] = names[object_bins[index] - 1]

    return ClassificationResult(
        slice_index=0,
        total_objects=num_objects,
        bin_counts=json.dumps(bin_counts),
        bin_percentages=json.dumps(bin_percentages),
        object_classes=json.dumps(object_classes),
    )


class ObjectClassificationBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Object classification primitives keyed by memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def positive_label_ids(self, labels: np.ndarray) -> np.ndarray:
        """Return positive label ids present in ``labels``."""

    @abstractmethod
    def mean_intensity_values(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        """Return mean intensity for ``label_ids``."""

    @abstractmethod
    def apply_object_bins(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
        object_bins: np.ndarray,
    ) -> np.ndarray:
        """Map source labels to classification bin ids in one image pass."""


class NumbaNumpyObjectClassificationBackendStrategy(
    ObjectClassificationBackendStrategy
):
    """Numba-backed NumPy object classification primitives."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
        image = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        label_ids = np.array([1, 2], dtype=np.int32)
        object_bins = np.array([1, 2], dtype=np.int32)
        self.mean_intensity_values(labels, image, label_ids)
        self.apply_object_bins(labels, label_ids, object_bins)

    def positive_label_ids(self, labels: np.ndarray) -> np.ndarray:
        labels_array = np.asarray(labels, dtype=np.int32)
        if labels_array.size == 0:
            return np.zeros(0, dtype=np.int32)
        max_label = int(labels_array.max())
        if max_label <= 0:
            return np.zeros(0, dtype=np.int32)
        present = np.bincount(labels_array.ravel(), minlength=max_label + 1) > 0
        return np.flatnonzero(present[1:]).astype(np.int32) + 1

    def mean_intensity_values(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        return _mean_intensity_values_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(image, dtype=np.float64),
            np.asarray(label_ids, dtype=np.int32),
        )

    def apply_object_bins(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
        object_bins: np.ndarray,
    ) -> np.ndarray:
        return _apply_object_bins_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
            np.asarray(object_bins, dtype=np.int32),
        )


def object_classification_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> ObjectClassificationBackendStrategy:
    """Return the selected object-classification backend."""
    return ObjectClassificationBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
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
    measurement_values: np.ndarray | None = None,
    measurement_values_by_rule: tuple[np.ndarray, ...] = (),
    classification_rules: tuple[dict[str, object], ...] = (),
    bin_choice: ClassificationBinChoice = ClassificationBinChoice.EVEN,
    bin_count: int = 3,
    low_threshold: float = 0.0,
    high_threshold: float = 1.0,
    wants_low_bin: bool = False,
    wants_high_bin: bool = False,
    custom_thresholds: str = "0,1",
    bin_names: str | None = None,
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, ClassificationResult | tuple[ClassificationResult, ...]]:
    """Classify objects based on one measurement or declared rule rows."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    backend = object_classification_backend(
        backend_provider=classification_backend_provider
    )
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
                bin_choice=rule.get("bin_choice", ClassificationBinChoice.EVEN),
                bin_count=int(rule.get("bin_count", 3)),
                low_threshold=float(rule.get("low_threshold", 0.0)),
                high_threshold=float(rule.get("high_threshold", 1.0)),
                wants_low_bin=bool(rule.get("wants_low_bin", False)),
                wants_high_bin=bool(rule.get("wants_high_bin", False)),
                custom_thresholds=str(rule.get("custom_thresholds", "0,1")),
                bin_names=rule.get("bin_names"),  # type: ignore[arg-type]
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
    measurement1_values: np.ndarray | None = None,
    measurement2_values: np.ndarray | None = None,
    threshold1_method: ClassificationThresholdMethod = ClassificationThresholdMethod.MEAN,
    threshold1_value: float = 0.5,
    threshold2_method: ClassificationThresholdMethod = ClassificationThresholdMethod.MEAN,
    threshold2_value: float = 0.5,
    low_low_name: str = "low_low",
    low_high_name: str = "low_high",
    high_low_name: str = "high_low",
    high_high_name: str = "high_high",
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, ClassificationResult]:
    """Classify objects from two measurements into four quadrants."""
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
    ).classify(
        image,
        labels,
        object_classification_backend(backend_provider=classification_backend_provider),
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
def classify_objects_by_intensity_bins(
    image: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 3,
    use_percentiles: bool = True,
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, ClassificationResult]:
    """Classify objects by mean intensity into evenly distributed bins."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    return IntensityBinsClassificationRequest(
        num_bins=num_bins,
        use_percentiles=use_percentiles,
    ).classify(
        image,
        labels,
        object_classification_backend(backend_provider=classification_backend_provider),
    )


@njit(cache=True)
def _mean_intensity_values_numba(
    labels: np.ndarray,
    image: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    max_label = 0
    for i in range(label_ids.size):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id
    sums = np.zeros(max_label + 1, dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    rows, cols = labels.shape
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                sums[label] += image[row, col]
                counts[label] += 1
    values = np.empty(label_ids.size, dtype=np.float64)
    for i in range(label_ids.size):
        label = int(label_ids[i])
        if label <= 0 or label > max_label or counts[label] == 0:
            values[i] = np.nan
        else:
            values[i] = sums[label] / counts[label]
    return values


@njit(cache=True)
def _apply_object_bins_numba(
    labels: np.ndarray,
    label_ids: np.ndarray,
    object_bins: np.ndarray,
) -> np.ndarray:
    max_label = 0
    for i in range(label_ids.size):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id
    bin_by_label = np.zeros(max_label + 1, dtype=np.int32)
    count = label_ids.size
    if object_bins.size < count:
        count = object_bins.size
    for i in range(count):
        label = int(label_ids[i])
        if label > 0 and label <= max_label:
            bin_by_label[label] = int(object_bins[i])

    output = np.zeros(labels.shape, dtype=np.int32)
    rows, cols = labels.shape
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                output[row, col] = bin_by_label[label]
    return output


__all__ = public_names_from_objects(
    ClassificationBinChoice,
    ClassificationMethod,
    ClassificationResult,
    ClassificationThresholdMethod,
    "CLASSIFICATION_RESULT_FIELDS",
    IntensityBinsClassificationRequest,
    NumbaNumpyObjectClassificationBackendStrategy,
    ObjectClassificationBackendStrategy,
    SingleMeasurementClassificationRequest,
    TwoMeasurementClassificationRequest,
    classify_objects_by_intensity_bins,
    classify_objects_single_measurement,
    classify_objects_two_measurements,
    classification_result_from_bins,
    classification_threshold,
    object_classification_backend,
)
