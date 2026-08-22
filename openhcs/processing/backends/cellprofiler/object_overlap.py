"""Overlap measurement semantics for CellProfiler-compatible backends."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
from typing import TYPE_CHECKING, Annotated, ClassVar, Tuple
from dataclasses import dataclass
from enum import Enum
import scipy.ndimage
import scipy.sparse
from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode,
    special_inputs,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    FieldDerivedMeasurementFeatureModule,
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
)
from openhcs.interop.cellprofiler.setting_names import optional_setting_value
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_setting_parser,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)

if TYPE_CHECKING:
    from openhcs.core.callable_contract import CallableContract
    from openhcs.core.source_bindings import StepSourceBindingsConfig
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )


class DecimationMethod(Enum):
    KMEANS = "kmeans"
    SKELETON = "skeleton"


GroundTruthObjectLabelsInput = Annotated[
    ObjectLabelValue,
    "Reference object-label plane against which overlap accuracy is measured.",
]
TestObjectLabelsInput = Annotated[
    ObjectLabelValue,
    "Candidate object-label plane whose regions are compared with the reference.",
]


@dataclass
class OverlapMeasurements:
    """Measurements from object overlap analysis."""

    slice_index: int
    ffactor: float
    precision: float
    recall: float
    true_positive_rate: float
    false_positive_rate: float
    true_negative_rate: float
    false_negative_rate: float
    rand_index: float
    adjusted_rand_index: float


@dataclass
class OverlapMeasurementsWithEmd(OverlapMeasurements):
    """Object-overlap measurements including optional EMD."""

    earth_movers_distance: float


@dataclass
class ImageOverlapMeasurement:
    slice_index: int
    true_positive_rate: float
    false_positive_rate: float
    false_negative_rate: float
    true_negative_rate: float
    precision: float
    recall: float
    ffactor: float
    rand_index: float
    adjusted_rand_index: float


@dataclass
class ImageOverlapMeasurementWithEmd(ImageOverlapMeasurement):
    """Image-overlap measurements including optional EMD."""

    earth_movers_distance: float


@dataclass(frozen=True, slots=True)
class OverlapMeasurementRows(ModuleOwnedResultMeasurementRows):
    """Project raw overlap fields into exact CellProfiler feature names."""

    qualifiers: tuple[str, ...]

    @classmethod
    def for_request(
        cls,
        module_type: type[object],
        request: "CellProfilerOutputRecordRequest",
    ) -> "OverlapMeasurementRows":
        if not issubclass(module_type, _OverlapMeasurementModule):
            raise TypeError(
                f"{cls.__name__} requires an overlap module owner, got "
                f"{module_type.__name__}."
            )
        return cls(
            request.output_value,
            module_type=module_type,
            qualifiers=module_type.measurement_qualifiers(request),
        )

    def rows(self) -> MeasurementProjectedColumnarRows:
        source_rows = self.source_rows()
        slice_field = MeasurementRowAxisField.SLICE_INDEX.value
        projected_fields = [self.source_field(slice_field)]
        projected_columns: dict[str, object] = {
            slice_field: source_rows.column_values(slice_field)
        }
        for field in source_rows.fields:
            if field.name == slice_field:
                continue
            feature_name = self.module_type.overlap_measurement_feature_name(
                field.name,
                *self.qualifiers,
            )
            projected_fields.append(
                FieldSpec(feature_name, field.dtype, required=False)
            )
            projected_columns[feature_name] = source_rows.column_values(field.name)
        return MeasurementProjectedColumnarRows(
            projected_columns,
            fields=tuple(projected_fields),
        )


def _nan_divide(numerator: float, denominator: float) -> float:
    """Safe division that returns NaN for zero denominator."""
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _compute_rand_index_ijv(
    gt_ijv: np.ndarray, test_ijv: np.ndarray, shape: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Compute the Rand Index for IJV matrices.

    Based on the Omega Index from Collins (1988).
    """
    gt_bkgd = np.ones(shape, bool)
    if len(gt_ijv) > 0:
        gt_bkgd[gt_ijv[:, 0], gt_ijv[:, 1]] = False
    test_bkgd = np.ones(shape, bool)
    if len(test_ijv) > 0:
        test_bkgd[test_ijv[:, 0], test_ijv[:, 1]] = False
    gt_bkgd_coords = np.argwhere(gt_bkgd)
    test_bkgd_coords = np.argwhere(test_bkgd)
    if len(gt_bkgd_coords) > 0:
        gt_ijv = (
            np.vstack(
                [
                    gt_ijv,
                    np.column_stack(
                        [
                            gt_bkgd_coords,
                            np.zeros(
                                len(gt_bkgd_coords),
                                dtype=gt_ijv.dtype if len(gt_ijv) > 0 else np.int32,
                            ),
                        ]
                    ),
                ]
            )
            if len(gt_ijv) > 0
            else np.column_stack(
                [gt_bkgd_coords, np.zeros(len(gt_bkgd_coords), dtype=np.int32)]
            )
        )
    if len(test_bkgd_coords) > 0:
        test_ijv = (
            np.vstack(
                [
                    test_ijv,
                    np.column_stack(
                        [
                            test_bkgd_coords,
                            np.zeros(
                                len(test_bkgd_coords),
                                dtype=test_ijv.dtype if len(test_ijv) > 0 else np.int32,
                            ),
                        ]
                    ),
                ]
            )
            if len(test_ijv) > 0
            else np.column_stack(
                [test_bkgd_coords, np.zeros(len(test_bkgd_coords), dtype=np.int32)]
            )
        )
    if len(gt_ijv) == 0 or len(test_ijv) == 0:
        return (np.nan, np.nan)
    u = np.vstack(
        [
            np.column_stack([gt_ijv, np.zeros(gt_ijv.shape[0], dtype=np.int32)]),
            np.column_stack([test_ijv, np.ones(test_ijv.shape[0], dtype=np.int32)]),
        ]
    )
    order = np.lexsort([u[:, 2], u[:, 3], u[:, 0], u[:, 1]])
    u = u[order, :]
    first = np.hstack([[True], np.any(u[:-1, :] != u[1:, :], axis=1)])
    u = u[first, :]
    coord_changes = np.hstack(
        [
            [0],
            np.argwhere((u[:-1, 0] != u[1:, 0]) | (u[:-1, 1] != u[1:, 1])).flatten()
            + 1,
            [u.shape[0]],
        ]
    )
    coord_counts = coord_changes[1:] - coord_changes[:-1]
    rev_idx = np.repeat(np.arange(len(coord_counts)), coord_counts)
    count_test = np.bincount(rev_idx, u[:, 3]).astype(np.int64)
    count_gt = coord_counts - count_test
    n_coords = len(coord_counts)
    if n_coords < 2:
        return (1.0, 1.0)
    total_pairs = n_coords * (n_coords - 1) // 2
    agreements = 0
    for i in range(n_coords):
        for j in range(i + 1, min(i + 100, n_coords)):
            same_gt = count_gt[i] > 0 and count_gt[j] > 0
            same_test = count_test[i] > 0 and count_test[j] > 0
            if same_gt == same_test:
                agreements += 1
    sampled_pairs = min(total_pairs, n_coords * 50)
    rand_index = agreements / sampled_pairs if sampled_pairs > 0 else np.nan
    adjusted_rand_index = 2 * rand_index - 1 if not np.isnan(rand_index) else np.nan
    return (rand_index, adjusted_rand_index)


def _labels_to_ijv(labels: np.ndarray) -> np.ndarray:
    """Convert label image to IJV format (row, col, label)."""
    i, j = np.where(labels > 0)
    if len(i) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    v = labels[i, j]
    return np.column_stack([i, j, v]).astype(np.int32)


def _compute_emd_simple(
    src_labels: np.ndarray,
    dest_labels: np.ndarray,
    max_points: int,
    max_distance: int,
    penalize_missing: bool,
) -> float:
    """
    Compute simplified Earth Mover's Distance between two label images.
    """
    src_mask = src_labels > 0
    dest_mask = dest_labels > 0
    src_area = np.sum(src_mask)
    dest_area = np.sum(dest_mask)
    if src_area == 0 and dest_area == 0:
        return 0.0
    if src_area == 0 or dest_area == 0:
        if penalize_missing:
            return max(src_area, dest_area) * max_distance
        return 0.0
    src_coords = np.argwhere(src_mask)
    dest_coords = np.argwhere(dest_mask)
    if len(src_coords) > max_points:
        idx = np.linspace(0, len(src_coords) - 1, max_points).astype(int)
        src_coords = src_coords[idx]
    if len(dest_coords) > max_points:
        idx = np.linspace(0, len(dest_coords) - 1, max_points).astype(int)
        dest_coords = dest_coords[idx]
    total_distance = 0.0
    for sc in src_coords:
        if len(dest_coords) == 0:
            total_distance += max_distance
            continue
        distances = np.sqrt(np.sum((dest_coords - sc) ** 2, axis=1))
        min_dist = np.min(distances)
        total_distance += min(min_dist, max_distance)
    return total_distance / len(src_coords) if len(src_coords) > 0 else 0.0


def _measure_image_overlap(
    image: np.ndarray,
    *,
    calculate_emd: bool,
    max_distance: int = 250,
    penalize_missing: bool = False,
    decimation_method: DecimationMethod = DecimationMethod.KMEANS,
    max_points: int = 250,
) -> tuple[
    np.ndarray,
    DataclassMeasurementColumnarRows,
]:
    """Measure binary overlap between ground-truth and test image planes."""
    if image.shape[0] < 2:
        raise ValueError("Image must have at least 2 slices (ground_truth, test)")
    ground_truth_image = image[0].astype(bool)
    test_image = image[1].astype(bool)
    mask = image[2].astype(bool) if image.shape[0] > 2 else None
    if mask is not None:
        ground_truth_image = ground_truth_image & mask
        test_image = test_image & mask
        total_pixels = np.sum(mask)
    else:
        total_pixels = ground_truth_image.size
    true_positive = np.sum(ground_truth_image & test_image)
    false_positive = np.sum(~ground_truth_image & test_image)
    false_negative = np.sum(ground_truth_image & ~test_image)
    true_negative = np.sum(~ground_truth_image & ~test_image)
    eps = 1e-10
    true_positive_rate = true_positive / (true_positive + false_negative + eps)
    false_positive_rate = false_positive / (false_positive + true_negative + eps)
    false_negative_rate = false_negative / (true_positive + false_negative + eps)
    true_negative_rate = true_negative / (false_positive + true_negative + eps)
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive_rate
    f_factor = 2 * precision * recall / (precision + recall + eps)
    n = total_pixels
    a = true_positive
    b = false_positive
    c = false_negative
    d = true_negative
    rand_index = (a + d) / (a + b + c + d + eps)
    n_choose_2 = n * (n - 1) / 2 if n > 1 else 1
    sum_ni_choose_2 = a + c
    sum_nj_choose_2 = a + b
    expected_index = sum_ni_choose_2 * sum_nj_choose_2 / (n_choose_2 + eps)
    max_index = (sum_ni_choose_2 + sum_nj_choose_2) / 2
    adjusted_rand_index = (a - expected_index) / (max_index - expected_index + eps)
    adjusted_rand_index = max(0.0, min(1.0, adjusted_rand_index))
    common = dict(
        slice_index=0,
        true_positive_rate=float(true_positive_rate),
        false_positive_rate=float(false_positive_rate),
        false_negative_rate=float(false_negative_rate),
        true_negative_rate=float(true_negative_rate),
        precision=float(precision),
        recall=float(recall),
        ffactor=float(f_factor),
        rand_index=float(rand_index),
        adjusted_rand_index=float(adjusted_rand_index),
    )
    if calculate_emd:
        measurement = ImageOverlapMeasurementWithEmd(
            **common,
            earth_movers_distance=float(
                compute_image_earth_movers_distance(
                    ground_truth_image,
                    test_image,
                    max_distance=max_distance,
                    penalize_missing=penalize_missing,
                    decimation_method=decimation_method,
                    max_points=max_points,
                )
            ),
        )
        row_type = ImageOverlapMeasurementWithEmd
    else:
        measurement = ImageOverlapMeasurement(**common)
        row_type = ImageOverlapMeasurement
    rows = DataclassMeasurementColumnarRows((measurement,), row_type=row_type)
    return (ground_truth_image.astype(np.float32)[np.newaxis, ...], rows)


@numpy(contract=ProcessingContract.PURE_3D)
def measureimageoverlap(
    image: np.ndarray,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure binary overlap without the optional EMD feature."""
    return _measure_image_overlap(image, calculate_emd=False)


@numpy(contract=ProcessingContract.PURE_3D)
def measureimageoverlap_with_emd(
    image: np.ndarray,
    max_distance: int = 250,
    penalize_missing: bool = False,
    decimation_method: DecimationMethod = DecimationMethod.KMEANS,
    max_points: int = 250,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure binary overlap including Earth Mover's Distance."""
    return _measure_image_overlap(
        image,
        calculate_emd=True,
        max_distance=max_distance,
        penalize_missing=penalize_missing,
        decimation_method=decimation_method,
        max_points=max_points,
    )


def compute_image_earth_movers_distance(
    ground_truth: np.ndarray,
    test: np.ndarray,
    max_distance: int,
    penalize_missing: bool,
    decimation_method: DecimationMethod,
    max_points: int,
) -> float:
    from scipy.spatial.distance import cdist

    gt_coords = np.argwhere(ground_truth)
    test_coords = np.argwhere(test)
    if len(gt_coords) == 0 or len(test_coords) == 0:
        if penalize_missing:
            return float(max_distance)
        return 0.0
    if len(gt_coords) > max_points:
        gt_coords = decimate_overlap_points(gt_coords, max_points, decimation_method)
    if len(test_coords) > max_points:
        test_coords = decimate_overlap_points(
            test_coords, max_points, decimation_method
        )
    distances = cdist(gt_coords, test_coords, metric="euclidean")
    distances = np.minimum(distances, max_distance)
    min_dist_gt_to_test = np.mean(np.min(distances, axis=1))
    min_dist_test_to_gt = np.mean(np.min(distances, axis=0))
    return float((min_dist_gt_to_test + min_dist_test_to_gt) / 2)


def decimate_overlap_points(
    coords: np.ndarray, max_points: int, method: DecimationMethod
) -> np.ndarray:
    del method
    indices = np.linspace(0, len(coords) - 1, max_points, dtype=int)
    return coords[indices]


def _measure_object_overlap(
    image: np.ndarray,
    labels_ground_truth: ObjectLabelValue,
    labels_test: ObjectLabelValue,
    *,
    calculate_emd: bool,
    max_points: int = 250,
    decimation_method: DecimationMethod = DecimationMethod.KMEANS,
    max_distance: int = 250,
    penalize_missing: bool = False,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """
    Calculate overlap statistics between ground truth and test segmentation objects.

    Args:
        image: Input image array, shape (2, H, W) - ground truth labels stacked with test labels,
               or (H, W) if labels provided via special_inputs
        labels_ground_truth: Ground truth segmentation labels
        labels_test: Test segmentation labels to compare
        calculate_emd: Whether to calculate Earth Mover's Distance
        max_points: Maximum number of representative points for EMD calculation
        decimation_method: Method for selecting representative points (KMEANS or SKELETON)
        max_distance: Maximum distance penalty for EMD calculation
        penalize_missing: Whether to penalize missing pixels in EMD calculation

    Returns:
        Tuple of (original image, overlap measurements)
    """
    output_image = image
    labels_ground_truth = object_label_dense_array(labels_ground_truth, dtype=np.int32)
    labels_test = object_label_dense_array(labels_test, dtype=np.int32)
    if labels_ground_truth.ndim == 3:
        labels_ground_truth = labels_ground_truth[0]
    if labels_test.ndim == 3:
        labels_test = labels_test[0]
    gt_ijv = _labels_to_ijv(labels_ground_truth)
    test_ijv = _labels_to_ijv(labels_test)
    shape = (
        max(labels_ground_truth.shape[0], labels_test.shape[0]),
        max(labels_ground_truth.shape[1], labels_test.shape[1]),
    )
    total_pixels = shape[0] * shape[1]
    gt_mask = labels_ground_truth > 0
    test_mask = labels_test > 0
    TP = np.sum(gt_mask & test_mask)
    FP = np.sum(~gt_mask & test_mask)
    FN = np.sum(gt_mask & ~test_mask)
    TN = np.sum(~gt_mask & ~test_mask)
    gt_total = np.sum(gt_mask)
    precision = _nan_divide(TP, TP + FP)
    recall = _nan_divide(TP, TP + FN)
    f_factor = _nan_divide(2 * precision * recall, precision + recall)
    true_positive_rate = _nan_divide(TP, FN + TP)
    false_positive_rate = _nan_divide(FP, FP + TN)
    false_negative_rate = _nan_divide(FN, FN + TP)
    true_negative_rate = _nan_divide(TN, FP + TN)
    rand_index, adjusted_rand_index = _compute_rand_index_ijv(gt_ijv, test_ijv, shape)
    common = dict(
        slice_index=0,
        ffactor=float(f_factor) if not np.isnan(f_factor) else 0.0,
        precision=float(precision) if not np.isnan(precision) else 0.0,
        recall=float(recall) if not np.isnan(recall) else 0.0,
        true_positive_rate=(
            float(true_positive_rate) if not np.isnan(true_positive_rate) else 0.0
        ),
        false_positive_rate=(
            float(false_positive_rate) if not np.isnan(false_positive_rate) else 0.0
        ),
        true_negative_rate=(
            float(true_negative_rate) if not np.isnan(true_negative_rate) else 0.0
        ),
        false_negative_rate=(
            float(false_negative_rate) if not np.isnan(false_negative_rate) else 0.0
        ),
        rand_index=float(rand_index) if not np.isnan(rand_index) else 0.0,
        adjusted_rand_index=(
            float(adjusted_rand_index) if not np.isnan(adjusted_rand_index) else 0.0
        ),
    )
    if calculate_emd:
        measurement = OverlapMeasurementsWithEmd(
            **common,
            earth_movers_distance=float(
                _compute_emd_simple(
                    labels_ground_truth,
                    labels_test,
                    max_points,
                    max_distance,
                    penalize_missing,
                )
            ),
        )
        row_type = OverlapMeasurementsWithEmd
    else:
        measurement = OverlapMeasurements(**common)
        row_type = OverlapMeasurements
    return (
        output_image,
        DataclassMeasurementColumnarRows((measurement,), row_type=row_type),
    )


@numpy(contract=ProcessingContract.FLEXIBLE)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.MATCH_IMAGE_STACK)
@special_inputs("labels_ground_truth", "labels_test")
def measure_object_overlap(
    image: np.ndarray,
    labels_ground_truth: GroundTruthObjectLabelsInput,
    labels_test: TestObjectLabelsInput,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure object overlap without the optional EMD feature."""
    return _measure_object_overlap(
        image,
        labels_ground_truth,
        labels_test,
        calculate_emd=False,
    )


@numpy(contract=ProcessingContract.FLEXIBLE)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.MATCH_IMAGE_STACK)
@special_inputs("labels_ground_truth", "labels_test")
def measure_object_overlap_with_emd(
    image: np.ndarray,
    labels_ground_truth: GroundTruthObjectLabelsInput,
    labels_test: TestObjectLabelsInput,
    max_points: int = 250,
    decimation_method: DecimationMethod = DecimationMethod.KMEANS,
    max_distance: int = 250,
    penalize_missing: bool = False,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure object overlap including Earth Mover's Distance."""
    return _measure_object_overlap(
        image,
        labels_ground_truth,
        labels_test,
        calculate_emd=True,
        max_points=max_points,
        decimation_method=decimation_method,
        max_distance=max_distance,
        penalize_missing=penalize_missing,
    )


class _OverlapMeasurementModule:
    """Shared authoritative declaration for conditional overlap measurements."""

    measurement_feature_family = "Overlap"
    measurement_category_prefixes = (("overlap",),)
    calculate_emd_setting: ClassVar[str] = "Calculate earth mover's distance?"
    max_points_setting: ClassVar[str] = "Maximum # of points"
    decimation_method_setting: ClassVar[str] = "Point selection method"
    max_distance_setting: ClassVar[str] = "Maximum distance"
    penalize_missing_setting: ClassVar[str] = "Penalize missing pixels"
    calculate_emd_binding = SettingToKeywordBinding(
        calculate_emd_setting,
        "calculate_emd",
        parse_cellprofiler_bool,
    )
    setting_bindings = (
        calculate_emd_binding,
        SettingToKeywordBinding(
            max_points_setting, "max_points", parse_cellprofiler_int
        ),
        SettingToKeywordBinding(
            decimation_method_setting,
            "decimation_method",
            cellprofiler_enum_setting_parser(DecimationMethod),
        ),
        SettingToKeywordBinding(
            max_distance_setting,
            "max_distance",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            penalize_missing_setting,
            "penalize_missing",
            parse_cellprofiler_bool,
        ),
    )

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Exact CellProfiler Overlap feature vocabulary."""

        F_FACTOR = ("Ffactor", (), (), (), "ffactor")
        PRECISION = ("Precision", (), (), (), "precision")
        RECALL = ("Recall", (), (), (), "recall")
        TRUE_POSITIVE_RATE = (
            "TruePosRate",
            (),
            (),
            (),
            "true_positive_rate",
        )
        FALSE_POSITIVE_RATE = (
            "FalsePosRate",
            (),
            (),
            (),
            "false_positive_rate",
        )
        TRUE_NEGATIVE_RATE = (
            "TrueNegRate",
            (),
            (),
            (),
            "true_negative_rate",
        )
        FALSE_NEGATIVE_RATE = (
            "FalseNegRate",
            (),
            (),
            (),
            "false_negative_rate",
        )
        RAND_INDEX = ("RandIndex", (), (), (), "rand_index")
        ADJUSTED_RAND_INDEX = (
            "AdjustedRandIndex",
            (),
            (),
            (),
            "adjusted_rand_index",
        )
        EARTH_MOVERS_DISTANCE = (
            "EarthMoversDistance",
            (),
            (),
            (),
            "earth_movers_distance",
        )

    MeasurementRows = OverlapMeasurementRows

    @classmethod
    def overlap_measurement_feature_name(
        cls,
        field_name: str,
        *qualifiers: str,
    ) -> str:
        matching = tuple(
            feature
            for feature in cls.MeasurementFeature
            if feature.measurement_row_field_name == field_name
        )
        if len(matching) != 1:
            raise ValueError(
                f"{cls.__name__} requires one overlap feature for result field "
                f"{field_name!r}, got {matching!r}."
            )
        return "_".join(("Overlap", matching[0].value, *qualifiers))

    @classmethod
    def emd_enabled(cls, module: ModuleBlock) -> bool:
        value = optional_setting_value(module, cls.calculate_emd_setting)
        return value is not None and parse_cellprofiler_bool(value)

    @classmethod
    def resolve_function(
        cls,
        module: ModuleBlock,
        *,
        contract: "CallableContract",
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., object]:
        """Select the callable whose result schema matches EMD topology."""
        del contract, source_bindings
        function_name = (
            cls.function_variants[0]
            if cls.emd_enabled(module)
            else str(cls.function_name)
        )
        return cls.require_callable(function_name)

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation,
        block_position,
        existing_records,
        step_context,
    ):
        """Reconstruct the EMD condition from the public callable variant."""
        setting_key = cls.normalize_setting_name(cls.calculate_emd_setting)
        own = (
            (
                ModuleSetting(
                    cls.calculate_emd_setting,
                    "Yes",
                ),
            )
            if invocation.contract.function_name == cls.function_variants[0]
            and setting_key
            not in cls._normalized_record_setting_names(existing_records)
            else ()
        )
        return (
            *own,
            *super()._derived_identity_setting_records(
                invocation=invocation,
                block_position=block_position,
                existing_records=(*existing_records, *own),
                step_context=step_context,
            ),
        )


class MeasureImageOverlapModule(
    _OverlapMeasurementModule,
    NoObjectNameMeasurementRecordMixin,
    FieldDerivedMeasurementFeatureModule,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "MeasureImageOverlap"
    function_name = "measureimageoverlap"
    function_variants = ("measureimageoverlap_with_emd",)
    validated = True
    confidence = 1.0
    ground_truth_setting = "Select the image to be used as the ground truth basis for calculating the amount of overlap"
    test_image_setting = "Select the image to be used to test for overlap"
    setting_bindings = (
        SettingToKeywordBinding.input(ground_truth_setting, ImageArtifactType),
        SettingToKeywordBinding.input(test_image_setting, ImageArtifactType),
    )

    @classmethod
    def measurement_qualifiers(
        cls,
        request: "CellProfilerOutputRecordRequest",
    ) -> tuple[str, ...]:
        image_inputs = tuple(
            spec
            for spec in request.callable_contract.artifact_inputs.specs
            if spec.artifact_type is ImageArtifactType
        )
        if len(image_inputs) != 2:
            raise ValueError(
                "MeasureImageOverlap requires exactly two image inputs, got "
                f"{tuple(spec.name for spec in image_inputs)!r}."
            )
        return (image_inputs[1].name,)


class MeasureObjectOverlapModule(
    _OverlapMeasurementModule,
    NoObjectNameMeasurementRecordMixin,
    FieldDerivedMeasurementFeatureModule,
    ObjectArtifactInputModule,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "MeasureObjectOverlap"
    function_name = "measure_object_overlap"
    function_variants = ("measure_object_overlap_with_emd",)
    validated = True
    confidence = 1.0
    ground_truth_setting = "Select the objects to be used as the ground truth basis for calculating the amount of overlap"
    test_objects_setting = (
        "Select the objects to be tested for overlap against the ground truth"
    )
    setting_bindings = (
        SettingToKeywordBinding.input(
            ground_truth_setting,
            ObjectLabelsArtifactType,
            runtime_parameter_name="labels_ground_truth",
        ),
        SettingToKeywordBinding.input(
            test_objects_setting,
            ObjectLabelsArtifactType,
            runtime_parameter_name="labels_test",
        ),
    )

    @classmethod
    def measurement_qualifiers(
        cls,
        request: "CellProfilerOutputRecordRequest",
    ) -> tuple[str, ...]:
        object_inputs = tuple(
            spec
            for spec in request.callable_contract.artifact_inputs.specs
            if spec.artifact_type is ObjectLabelsArtifactType
        )
        if len(object_inputs) != 2:
            raise ValueError(
                "MeasureObjectOverlap requires exactly two object inputs, got "
                f"{tuple(spec.name for spec in object_inputs)!r}."
            )
        return tuple(spec.name for spec in object_inputs)


__all__ = public_names_from_objects(
    DecimationMethod,
    ImageOverlapMeasurement,
    ImageOverlapMeasurementWithEmd,
    OverlapMeasurements,
    OverlapMeasurementsWithEmd,
    compute_image_earth_movers_distance,
    decimate_overlap_points,
    measureimageoverlap,
    measureimageoverlap_with_emd,
    measure_object_overlap,
    measure_object_overlap_with_emd,
)
