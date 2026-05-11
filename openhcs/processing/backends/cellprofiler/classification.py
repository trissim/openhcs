"""Classification backends for CellProfiler-compatible object measurements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


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


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Results from object classification."""

    slice_index: int
    total_objects: int
    bin_counts: str
    bin_percentages: str
    object_classes: str = "{}"


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
            return labels, ClassificationResult(
                slice_index=0,
                total_objects=0,
                bin_counts=json.dumps({}),
                bin_percentages=json.dumps({}),
            )

        if self.measurement_values is None:
            values = backend.mean_intensity_values(labels, image, unique_labels)
        else:
            values = self.measurement_values.copy()

        if len(values) < num_objects:
            values = np.concatenate([values, np.full(num_objects - len(values), np.nan)])

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
            return labels, ClassificationResult(
                slice_index=0,
                total_objects=0,
                bin_counts=json.dumps({}),
                bin_percentages=json.dumps({}),
            )

        if self.measurement1_values is None:
            values1 = backend.mean_intensity_values(labels, image, unique_labels)
        else:
            values1 = self.measurement1_values.copy()

        if self.measurement2_values is None:
            values2 = np.bincount(
                labels.astype(np.intp, copy=False).ravel(),
                minlength=(int(unique_labels[-1]) + 1 if num_objects else 1),
            )[unique_labels].astype(float)
        else:
            values2 = self.measurement2_values.copy()

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
            return labels, ClassificationResult(
                slice_index=0,
                total_objects=0,
                bin_counts=json.dumps({}),
                bin_percentages=json.dumps({}),
            )

        values = backend.mean_intensity_values(labels, image, unique_labels)
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        if len(valid_values) == 0:
            return labels, ClassificationResult(
                slice_index=0,
                total_objects=num_objects,
                bin_counts=json.dumps({}),
                bin_percentages=json.dumps({}),
            )

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
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return custom_value
    if method == ClassificationThresholdMethod.MEAN:
        return float(np.mean(valid_values))
    if method == ClassificationThresholdMethod.MEDIAN:
        return float(np.median(valid_values))
    if method == ClassificationThresholdMethod.CUSTOM:
        return custom_value
    raise ValueError(f"Unsupported classification threshold method: {method!r}")


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

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

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


__all__ = [
    "ClassificationBinChoice",
    "ClassificationMethod",
    "ClassificationResult",
    "ClassificationThresholdMethod",
    "IntensityBinsClassificationRequest",
    "NumbaNumpyObjectClassificationBackendStrategy",
    "ObjectClassificationBackendStrategy",
    "SingleMeasurementClassificationRequest",
    "TwoMeasurementClassificationRequest",
    "classification_result_from_bins",
    "classification_threshold",
    "object_classification_backend",
]
