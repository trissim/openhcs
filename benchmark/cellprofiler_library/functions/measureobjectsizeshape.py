"""
Converted from CellProfiler: MeasureObjectSizeShape
Original: measureobjectsizeshape
"""

import logging
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import Any
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution,
    special_outputs,
)

from openhcs.core.runtime_semantics import (
    dense_object_label_id_domain,
    MeasurementRowAxisField,
    ObjectLabelRepresentation,
    ShapeObjectFeatureValueTable,
    object_shape_measurement_all_field_names,
    object_shape_measurement_field_names,
)
from openhcs.core.runtime_values import (
    ObjectLabelSet,
    ObjectLabelRuntimeSliceStackContract,
    SparseIJVLabelRows,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler.shape import (
    measure_object_size_shape_feature_arrays,
)
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.materialization import csv_materializer


_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@numpy
@object_label_measurement_execution(ObjectLabelMeasurementExecution.FULL_STACK)
@special_outputs(
    (
        "measurements",
        csv_materializer(fields=list(object_shape_measurement_all_field_names())),
    )
)
def measure_object_size_shape(
    image: np.ndarray,
    labels: np.ndarray | ObjectLabelSet,
    calculate_advanced: bool = True,
    calculate_zernikes: bool = True,
    shape_backend_provider: CellProfilerBackendProvider | None = None,
    zernike_backend_provider: CellProfilerBackendProvider | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Measure size and shape features of labeled objects.
    
    Args:
        image: Input intensity image (H, W)
        labels: Label image where each object has unique integer label (H, W)
        calculate_advanced: Whether to calculate advanced features like moments
        calculate_zernikes: Whether to calculate Zernike moments
    
    Returns:
        Tuple of (original image, list of measurement rows per object)
    """
    total_started_at = time.perf_counter()
    dense_slice_count = (
        ObjectLabelRuntimeSliceStackContract.runtime_slice_count(labels)
        if isinstance(labels, ObjectLabelSet)
        else None
    )
    if (
        isinstance(labels, ObjectLabelSet)
        and labels.representation is ObjectLabelRepresentation.DENSE_LABELS
        and dense_slice_count is not None
        and dense_slice_count > 1
    ):
        rows = DenseRuntimeSliceObjectSizeShapeMeasurement(
            labels=labels,
            slice_count=dense_slice_count,
            calculate_advanced=calculate_advanced,
            calculate_zernikes=calculate_zernikes,
            shape_backend_provider=shape_backend_provider,
            zernike_backend_provider=zernike_backend_provider,
        ).rows()
        _log_profile(
            "moss_total",
            time.perf_counter() - total_started_at,
            function="measure_object_size_shape",
            objects=len(rows),
            representation=labels.representation.value,
        )
        return image, rows

    if (
        isinstance(labels, ObjectLabelSet)
        and labels.representation is ObjectLabelRepresentation.SPARSE_IJV
    ):
        rows = _object_size_shape_sparse_ijv_rows(
            labels,
            calculate_advanced=calculate_advanced,
            calculate_zernikes=calculate_zernikes,
            shape_backend_provider=shape_backend_provider,
            zernike_backend_provider=zernike_backend_provider,
        )
        _log_profile(
            "moss_total",
            time.perf_counter() - total_started_at,
            function="measure_object_size_shape",
            objects=len(rows),
            representation=labels.representation.value,
        )
        return image, rows

    label_array = object_label_dense_array(labels, dtype=np.int32)
    if not np.any(label_array > 0):
        return image, []

    phase_started_at = time.perf_counter()
    feature_values, measured_labels = measure_object_size_shape_feature_arrays(
        label_array,
        calculate_advanced=calculate_advanced,
        calculate_zernikes=calculate_zernikes,
        shape_backend_provider=shape_backend_provider,
        zernike_backend_provider=zernike_backend_provider,
    )
    _log_profile(
        "moss_features_total",
        time.perf_counter() - phase_started_at,
        function="measure_object_size_shape",
        objects=len(measured_labels),
    )
    phase_started_at = time.perf_counter()
    rows = ShapeObjectFeatureValueTable.from_feature_arrays(
        feature_values,
        measured_labels,
        object_domain=dense_object_label_id_domain(labels),
    ).rows()
    _log_profile(
        "moss_rows",
        time.perf_counter() - phase_started_at,
        function="measure_object_size_shape",
        rows=len(rows),
    )
    _log_profile(
        "moss_total",
        time.perf_counter() - total_started_at,
        function="measure_object_size_shape",
        objects=len(measured_labels),
    )
    return image, rows


@dataclass(frozen=True, slots=True)
class DenseRuntimeSliceObjectSizeShapeMeasurement:
    """Per-plane 2D size/shape measurement for runtime-slice object domains."""

    labels: ObjectLabelSet
    slice_count: int
    calculate_advanced: bool
    calculate_zernikes: bool
    shape_backend_provider: CellProfilerBackendProvider | None
    zernike_backend_provider: CellProfilerBackendProvider | None

    def rows(self) -> list[dict[str, Any]]:
        label_stack = object_label_dense_array(self.labels, dtype=np.int32)
        if label_stack.ndim != 3 or label_stack.shape[0] != self.slice_count:
            raise ValueError(
                "Dense runtime-slice object labels must have shape "
                f"(slice, y, x), got {label_stack.shape!r} for "
                f"{self.slice_count} runtime slices."
            )
        rows: list[dict[str, Any]] = []
        for slice_index in range(self.slice_count):
            rows.extend(self.slice_rows(label_stack[slice_index], slice_index))
        return rows

    def slice_rows(
        self,
        labels_2d: np.ndarray,
        slice_index: int,
    ) -> list[dict[str, Any]]:
        feature_values, measured_labels = measure_object_size_shape_feature_arrays(
            labels_2d,
            calculate_advanced=self.calculate_advanced,
            calculate_zernikes=self.calculate_zernikes,
            shape_backend_provider=self.shape_backend_provider,
            zernike_backend_provider=self.zernike_backend_provider,
        )
        slice_domain = self.labels.object_label_domain().project_slice(
            slice_index,
            self.slice_count,
        )
        rows = ShapeObjectFeatureValueTable.from_feature_arrays(
            feature_values,
            measured_labels,
            object_domain=dense_object_label_id_domain(
                labels_2d,
                declared_object_count=slice_domain.declared_object_count,
                declared_object_ids=slice_domain.declared_object_ids,
            ),
        ).rows()
        for row in rows:
            row[MeasurementRowAxisField.SLICE_INDEX.value] = int(slice_index)
        return rows


def _object_size_shape_sparse_ijv_rows(
    labels: ObjectLabelSet,
    *,
    calculate_advanced: bool,
    calculate_zernikes: bool,
    shape_backend_provider: CellProfilerBackendProvider | None,
    zernike_backend_provider: CellProfilerBackendProvider | None,
) -> list[dict[str, Any]]:
    raw_labels = labels.labels
    sparse_rows = (
        raw_labels
        if isinstance(raw_labels, SparseIJVLabelRows)
        else SparseIJVLabelRows.from_yx_label(raw_labels)
    )
    if sparse_rows.as_array().size == 0:
        return []
    if sparse_rows.has_slice_index:
        rows: list[dict[str, Any]] = []
        for slice_index in sparse_rows.slice_indices():
            slice_ijv = np.asarray(sparse_rows.slice(slice_index).as_array(), dtype=np.int32)
            for row in _object_size_shape_sparse_ijv_2d_rows(
                labels,
                slice_ijv,
                calculate_advanced=calculate_advanced,
                calculate_zernikes=calculate_zernikes,
                shape_backend_provider=shape_backend_provider,
                zernike_backend_provider=zernike_backend_provider,
            ):
                row["slice_index"] = int(slice_index)
                rows.append(row)
        return rows
    ijv = np.asarray(sparse_rows.as_yx_label_array(), dtype=np.int32)
    return _object_size_shape_sparse_ijv_2d_rows(
        labels,
        ijv,
        calculate_advanced=calculate_advanced,
        calculate_zernikes=calculate_zernikes,
        shape_backend_provider=shape_backend_provider,
        zernike_backend_provider=zernike_backend_provider,
    )


def _object_size_shape_sparse_ijv_2d_rows(
    labels: ObjectLabelSet,
    ijv: np.ndarray,
    *,
    calculate_advanced: bool,
    calculate_zernikes: bool,
    shape_backend_provider: CellProfilerBackendProvider | None,
    zernike_backend_provider: CellProfilerBackendProvider | None,
) -> list[dict[str, Any]]:
    object_ids = _sparse_ijv_object_ids(labels, ijv)
    rows: list[dict[str, Any]] = []
    for object_id in object_ids:
        object_pixels = ijv[ijv[:, 2] == int(object_id)]
        if object_pixels.size == 0:
            rows.append(_empty_sparse_ijv_shape_row(int(object_id)))
            continue
        pixel_y = object_pixels[:, 0]
        pixel_x = object_pixels[:, 1]
        min_y = int(pixel_y.min())
        min_x = int(pixel_x.min())
        max_y = int(pixel_y.max()) + 1
        max_x = int(pixel_x.max()) + 1
        local = np.zeros((max_y - min_y, max_x - min_x), dtype=np.int32)
        local[pixel_y - min_y, pixel_x - min_x] = int(object_id)
        feature_values, measured_labels = measure_object_size_shape_feature_arrays(
            local,
            calculate_advanced=calculate_advanced,
            calculate_zernikes=calculate_zernikes,
            shape_backend_provider=shape_backend_provider,
            zernike_backend_provider=zernike_backend_provider,
        )
        if len(measured_labels) == 0:
            rows.append(_empty_sparse_ijv_shape_row(int(object_id)))
            continue
        _offset_sparse_ijv_shape_features(feature_values, offset_y=min_y, offset_x=min_x)
        row = ShapeObjectFeatureValueTable.from_feature_arrays(
            feature_values,
            np.asarray([int(object_id)], dtype=np.int32),
            object_domain=(int(object_id),),
        ).rows()[0]
        rows.append(row)
    return rows


def _sparse_ijv_object_ids(labels: ObjectLabelSet, ijv: np.ndarray) -> np.ndarray:
    if labels.declared_object_ids:
        return np.asarray(labels.declared_object_ids, dtype=np.int32)
    if labels.declared_object_count is not None:
        return np.arange(1, labels.declared_object_count + 1, dtype=np.int32)
    return np.unique(ijv[:, 2]).astype(np.int32, copy=False)


def _empty_sparse_ijv_shape_row(object_id: int) -> dict[str, Any]:
    axis_fields = {
        MeasurementRowAxisField.SLICE_INDEX.value,
        MeasurementRowAxisField.OBJECT_LABEL.value,
    }
    return ShapeObjectFeatureValueTable.from_feature_arrays(
        {
            field: np.asarray([], dtype=float)
            for field in object_shape_measurement_field_names()
            if field not in axis_fields
        },
        (),
        object_domain=(object_id,),
    ).rows()[0]


def _offset_sparse_ijv_shape_features(
    feature_values: dict[str, np.ndarray],
    *,
    offset_y: int,
    offset_x: int,
) -> None:
    x_fields = (
        _shape_feature(ObjectShapeMeasurementFeature.CENTER_X),
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_X),
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_X),
    )
    y_fields = (
        _shape_feature(ObjectShapeMeasurementFeature.CENTER_Y),
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Y),
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Y),
    )
    for field in x_fields:
        if field in feature_values:
            feature_values[field] = np.asarray(feature_values[field], dtype=float) + offset_x
    for field in y_fields:
        if field in feature_values:
            feature_values[field] = np.asarray(feature_values[field], dtype=float) + offset_y


def _prepare_measure_object_size_shape() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 1
    measure_object_size_shape.__wrapped__(image, labels)


measure_object_size_shape.__openhcs_prepare__ = _prepare_measure_object_size_shape
