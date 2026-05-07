"""
Converted from CellProfiler: MeasureObjectSizeShape
Original: measureobjectsizeshape
"""

import logging
import numpy as np
import os
import skimage.measure
import time
from typing import Any
from numba import njit
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_outputs

from openhcs.core.runtime_semantics import (
    ObjectLabelRepresentation,
    ObjectShapeMeasurementFeature,
    indexed_measurement_feature_name,
    object_shape_measurement_field_names,
)
from openhcs.core.runtime_values import (
    ObjectLabelSet,
    SparseIJVLabelRows,
    object_label_dense_array,
)
from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler.shape import ShapeMeasurementBackendStrategy
from openhcs.processing.backends.cellprofiler.zernike import shape_zernike_moments
from openhcs.processing.backends.cellprofiler.morphology import MorphologyBackendStrategy
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.materialization import csv_materializer


_ZERNIKE_MAX_ORDER = 9
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
@special_outputs(
    (
        "measurements",
        csv_materializer(fields=list(object_shape_measurement_field_names())),
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
    feature_values, measured_labels = _measure_object_size_shape_features(
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
    rows = _object_size_shape_rows(feature_values, measured_labels)
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


def _measure_object_size_shape_features(
    labels: np.ndarray,
    *,
    calculate_advanced: bool,
    calculate_zernikes: bool,
    shape_backend_provider: CellProfilerBackendProvider | None,
    zernike_backend_provider: CellProfilerBackendProvider | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if labels.ndim == 2:
        return _measure_object_size_shape_features_2d(
            labels,
            calculate_advanced=calculate_advanced,
            calculate_zernikes=calculate_zernikes,
            shape_backend_provider=shape_backend_provider,
            zernike_backend_provider=zernike_backend_provider,
        )
    if labels.ndim == 3:
        return _measure_object_size_shape_features_3d(
            labels,
            calculate_advanced=calculate_advanced,
        )
    raise ValueError(f"Object labels must be 2D or 3D, got {labels.ndim}D.")


def _measure_object_size_shape_features_2d(
    labels: np.ndarray,
    *,
    calculate_advanced: bool,
    calculate_zernikes: bool,
    shape_backend_provider: CellProfilerBackendProvider | None,
    zernike_backend_provider: CellProfilerBackendProvider | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    shape_backend = ShapeMeasurementBackendStrategy.for_memory_type(
        backend_provider=shape_backend_provider,
    )
    _log_profile(
        "moss_backend_resolution",
        time.perf_counter() - phase_started_at,
        function="measure_object_size_shape",
    )
    phase_started_at = time.perf_counter()
    fast_region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        labels
    )
    _log_profile(
        "moss_region_properties",
        time.perf_counter() - phase_started_at,
        function="measure_object_size_shape",
        objects=int(fast_region_props.label.size),
    )
    phase_started_at = time.perf_counter()
    props = fast_region_props.as_regionprops_table_subset()
    _log_profile(
        "moss_regionprops_table_subset",
        time.perf_counter() - phase_started_at,
        function="measure_object_size_shape",
        fields=len(props),
    )
    phase_started_at = time.perf_counter()
    convex_area, solidity = _convex_area_and_solidity_from_labels(
        labels,
        fast_region_props,
    )
    _log_profile(
        "moss_convex_area_solidity",
        time.perf_counter() - phase_started_at,
        function="measure_object_size_shape",
        objects=int(fast_region_props.label.size),
    )
    props["convex_area"] = convex_area
    props["solidity"] = solidity
    measured_labels = np.asarray(props["label"])
    nobjects = len(measured_labels)
    if nobjects == 0:
        return {}, measured_labels

    perimeter = np.asarray(props["perimeter"], dtype=float)
    area = np.asarray(props["area"], dtype=float)
    phase_started_at = time.perf_counter()
    max_radius, mean_radius, median_radius = shape_backend.radius_features_from_labels(
        labels,
        measured_labels,
    )
    _log_profile(
        "moss_radius_features",
        time.perf_counter() - phase_started_at,
        function="measure_object_size_shape",
        objects=nobjects,
    )
    dense_labels = np.arange(1, int(np.max(labels)) + 1, dtype=measured_labels.dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        form_factor = 4.0 * np.pi * area / perimeter**2
    with np.errstate(divide="ignore", invalid="ignore"):
        compactness = 1.0 / form_factor
    phase_started_at = time.perf_counter()
    min_feret_diameter, max_feret_diameter = shape_backend.feret_diameters(
        labels,
        dense_labels,
    )
    _log_profile(
        "moss_feret_diameters",
        time.perf_counter() - phase_started_at,
        function="measure_object_size_shape",
        objects=int(dense_labels.size),
    )

    phase_started_at = time.perf_counter()
    dense_center_y, dense_center_x = _dense_label_centers_2d(labels)
    _log_profile(
        "moss_dense_centers",
        time.perf_counter() - phase_started_at,
        function="measure_object_size_shape",
        objects=int(dense_labels.size),
    )
    center_x = _compact_values_with_dense_tail(
        np.asarray(props["centroid-1"], dtype=float),
        dense_center_x,
        measured_labels=measured_labels,
    )
    center_y = _compact_values_with_dense_tail(
        np.asarray(props["centroid-0"], dtype=float),
        dense_center_y,
        measured_labels=measured_labels,
    )

    features = {
        _shape_feature(ObjectShapeMeasurementFeature.AREA): area,
        _shape_feature(ObjectShapeMeasurementFeature.PERIMETER): perimeter,
        _shape_feature(ObjectShapeMeasurementFeature.MAJOR_AXIS_LENGTH): props[
            "major_axis_length"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.MINOR_AXIS_LENGTH): props[
            "minor_axis_length"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.ECCENTRICITY): props[
            "eccentricity"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.ORIENTATION): (
            _cellprofiler_orientation_degrees(props)
        ),
        _shape_feature(ObjectShapeMeasurementFeature.CENTER_X): center_x,
        _shape_feature(ObjectShapeMeasurementFeature.CENTER_Y): center_y,
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_AREA): props[
            "bbox_area"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_X): props[
            "bbox-1"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_X): props[
            "bbox-3"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Y): props[
            "bbox-0"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Y): props[
            "bbox-2"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.FORM_FACTOR): form_factor,
        _shape_feature(ObjectShapeMeasurementFeature.EXTENT): props["extent"],
        _shape_feature(ObjectShapeMeasurementFeature.SOLIDITY): props["solidity"],
        _shape_feature(ObjectShapeMeasurementFeature.COMPACTNESS): compactness,
        _shape_feature(ObjectShapeMeasurementFeature.EULER_NUMBER): props[
            "euler_number"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.MAXIMUM_RADIUS): max_radius,
        _shape_feature(ObjectShapeMeasurementFeature.MEAN_RADIUS): mean_radius,
        _shape_feature(ObjectShapeMeasurementFeature.MEDIAN_RADIUS): median_radius,
        _shape_feature(ObjectShapeMeasurementFeature.CONVEX_AREA): props["convex_area"],
        _shape_feature(ObjectShapeMeasurementFeature.MIN_FERET_DIAMETER): (
            min_feret_diameter
        ),
        _shape_feature(ObjectShapeMeasurementFeature.MAX_FERET_DIAMETER): (
            max_feret_diameter
        ),
        _shape_feature(ObjectShapeMeasurementFeature.EQUIVALENT_DIAMETER): props[
            "equivalent_diameter"
        ],
    }
    if calculate_advanced:
        phase_started_at = time.perf_counter()
        features.update(_advanced_2d_features(props))
        _log_profile(
            "moss_advanced_features",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
        )
    if calculate_zernikes:
        phase_started_at = time.perf_counter()
        features.update(
            _zernike_features(
                labels,
                measured_labels,
                backend_provider=zernike_backend_provider,
            )
        )
        _log_profile(
            "moss_zernike_features",
            time.perf_counter() - phase_started_at,
            function="measure_object_size_shape",
            objects=nobjects,
        )
    _log_profile(
        "moss_features_2d_total",
        time.perf_counter() - total_started_at,
        function="measure_object_size_shape",
        objects=nobjects,
    )
    return features, measured_labels


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
        feature_values, measured_labels = _measure_object_size_shape_features_2d(
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
        row = _object_size_shape_rows(
            feature_values,
            np.asarray([int(object_id)], dtype=np.int32),
        )[0]
        row["object_label"] = int(object_id)
        rows.append(row)
    return rows


def _sparse_ijv_object_ids(labels: ObjectLabelSet, ijv: np.ndarray) -> np.ndarray:
    if labels.declared_object_ids:
        return np.asarray(labels.declared_object_ids, dtype=np.int32)
    if labels.declared_object_count is not None:
        return np.arange(1, labels.declared_object_count + 1, dtype=np.int32)
    return np.unique(ijv[:, 2]).astype(np.int32, copy=False)


def _empty_sparse_ijv_shape_row(object_id: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        field: _missing_shape_feature_value(field)
        for field in object_shape_measurement_field_names()
    }
    row["slice_index"] = 0
    row["object_label"] = object_id
    row["Center_Z"] = 0.0
    return row


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


def _measure_object_size_shape_features_3d(
    labels: np.ndarray,
    *,
    calculate_advanced: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    props = skimage.measure.regionprops_table(
        labels,
        properties=_desired_region_properties(3, calculate_advanced),
    )
    measured_labels = np.asarray(props["label"])
    surface_areas = np.zeros(len(measured_labels), dtype=float)
    for index, label in enumerate(measured_labels):
        volume = labels[
            max(props["bbox-0"][index] - 1, 0) : min(
                props["bbox-3"][index] + 1,
                labels.shape[0],
            ),
            max(props["bbox-1"][index] - 1, 0) : min(
                props["bbox-4"][index] + 1,
                labels.shape[1],
            ),
            max(props["bbox-2"][index] - 1, 0) : min(
                props["bbox-5"][index] + 1,
                labels.shape[2],
            ),
        ]
        surface_areas[index] = _surface_area(volume == label)

    features = {
        _shape_feature(ObjectShapeMeasurementFeature.VOLUME): props["area"],
        _shape_feature(ObjectShapeMeasurementFeature.SURFACE_AREA): surface_areas,
        _shape_feature(ObjectShapeMeasurementFeature.MAJOR_AXIS_LENGTH): props[
            "major_axis_length"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.MINOR_AXIS_LENGTH): props[
            "minor_axis_length"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.CENTER_X): props["centroid-2"],
        _shape_feature(ObjectShapeMeasurementFeature.CENTER_Y): props["centroid-1"],
        _shape_feature(ObjectShapeMeasurementFeature.CENTER_Z): props["centroid-0"],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_VOLUME): props[
            "bbox_area"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_X): props[
            "bbox-2"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_X): props[
            "bbox-5"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Y): props[
            "bbox-1"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Y): props[
            "bbox-4"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Z): props[
            "bbox-0"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Z): props[
            "bbox-3"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.EXTENT): props["extent"],
        _shape_feature(ObjectShapeMeasurementFeature.EULER_NUMBER): props[
            "euler_number"
        ],
        _shape_feature(ObjectShapeMeasurementFeature.EQUIVALENT_DIAMETER): props[
            "equivalent_diameter"
        ],
    }
    if calculate_advanced:
        features[_shape_feature(ObjectShapeMeasurementFeature.SOLIDITY)] = props[
            "solidity"
        ]
    return features, measured_labels


def _convex_area_and_solidity_from_labels(
    labels: np.ndarray,
    region_props: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact skimage-compatible convex area and solidity per label."""
    morphology_backend = MorphologyBackendStrategy.for_memory_type(MemoryType.NUMPY)
    object_count = int(region_props.label.size)
    convex_area = np.zeros(object_count, dtype=float)
    solidity = np.ones(object_count, dtype=float)
    for index, label_id in enumerate(region_props.label):
        min_y = int(region_props.bbox_min_y[index])
        min_x = int(region_props.bbox_min_x[index])
        max_y = int(region_props.bbox_max_y[index])
        max_x = int(region_props.bbox_max_x[index])
        crop = labels[min_y:max_y, min_x:max_x] == int(label_id)
        hull = morphology_backend.convex_hull_image(crop)
        hull_area = float(np.count_nonzero(hull))
        convex_area[index] = hull_area
        solidity[index] = (
            float(region_props.area[index]) / hull_area if hull_area > 0.0 else np.nan
        )
    return convex_area, solidity


def _desired_region_properties(
    dimensions: int,
    calculate_advanced: bool,
) -> list[str]:
    if dimensions == 2:
        properties = [
            "label",
            "image",
            "area",
            "perimeter",
            "bbox",
            "bbox_area",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "centroid",
            "equivalent_diameter",
            "extent",
            "eccentricity",
            "convex_area",
            "solidity",
            "euler_number",
        ]
        if calculate_advanced:
            properties.extend(
                [
                    "inertia_tensor",
                    "inertia_tensor_eigvals",
                    "moments",
                    "moments_central",
                    "moments_hu",
                    "moments_normalized",
                ]
            )
        return properties

    properties = [
        "label",
        "image",
        "area",
        "centroid",
        "bbox",
        "bbox_area",
        "major_axis_length",
        "minor_axis_length",
        "extent",
        "equivalent_diameter",
        "euler_number",
    ]
    if calculate_advanced:
        properties.append("solidity")
    return properties


def _object_size_shape_rows(
    feature_values: dict[str, np.ndarray],
    measured_labels: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    measured_label_ids = tuple(int(label) for label in np.asarray(measured_labels))
    measured_label_index = {
        object_id: index for index, object_id in enumerate(measured_label_ids)
    }
    feature_items = tuple(
        (
            feature_name,
            np.asarray(values),
            _python_feature_values(values),
            _missing_shape_feature_value(feature_name),
        )
        for feature_name, values in feature_values.items()
    )
    domain_count = max(
        max(measured_label_ids, default=0),
        *(
            values.shape[0]
            for _feature_name, values, _python_values, _missing_value in feature_items
            if values.ndim > 0
        ),
        0,
    )
    for index in range(domain_count):
        object_label = index + 1
        row: dict[str, float | int] = {
            "slice_index": 0,
            "object_label": object_label,
            "Center_Z": 0.0,
        }
        for feature_name, values, python_values, missing_value in feature_items:
            if values.ndim > 0:
                value_index = _feature_value_index(
                    object_label,
                    values=values,
                    domain_count=domain_count,
                    measured_label_index=measured_label_index,
                )
                if value_index is None:
                    value = missing_value
                else:
                    value = python_values[value_index]
            else:
                value = python_values
            row[feature_name] = value
        rows.append(row)
    return rows


def _feature_value_index(
    object_label: int,
    *,
    values: np.ndarray,
    domain_count: int,
    measured_label_index: dict[int, int],
) -> int | None:
    if values.shape[0] == domain_count:
        value_index = object_label - 1
        return value_index if value_index < values.shape[0] else None
    value_index = measured_label_index.get(object_label)
    if value_index is None or value_index >= values.shape[0]:
        return None
    return value_index


def _python_feature_values(values: np.ndarray) -> object:
    array = np.asarray(values)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _shape_feature(feature: ObjectShapeMeasurementFeature) -> str:
    return feature.value


def _indexed_shape_feature(
    feature: ObjectShapeMeasurementFeature,
    *indices: int,
) -> str:
    return indexed_measurement_feature_name(feature, *indices)


def _dense_label_centers_2d(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _dense_label_centers_2d_numba(np.ascontiguousarray(labels, dtype=np.int32))


@njit(cache=True)
def _dense_label_centers_2d_numba(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_label = int(labels.max()) if labels.size else 0
    if max_label <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty

    counts = np.zeros(max_label + 1, dtype=np.float64)
    y_sums = np.zeros(max_label + 1, dtype=np.float64)
    x_sums = np.zeros(max_label + 1, dtype=np.float64)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1.0
            y_sums[label] += float(y)
            x_sums[label] += float(x)

    center_y = np.empty(max_label, dtype=np.float64)
    center_x = np.empty(max_label, dtype=np.float64)
    for label in range(1, max_label + 1):
        count = counts[label]
        if count > 0.0:
            center_y[label - 1] = y_sums[label] / count
            center_x[label - 1] = x_sums[label] / count
        else:
            center_y[label - 1] = np.nan
            center_x[label - 1] = np.nan
    return center_y, center_x


def _compact_values_with_dense_tail(
    compact_values: np.ndarray,
    dense_values: np.ndarray,
    *,
    measured_labels: np.ndarray,
) -> np.ndarray:
    compact = np.asarray(compact_values, dtype=float)
    dense = np.asarray(dense_values, dtype=float)
    if dense.shape[0] <= compact.shape[0]:
        return compact
    values = dense.copy()
    for index, object_label in enumerate(np.asarray(measured_labels, dtype=np.int32)):
        value_index = int(object_label) - 1
        if 0 <= value_index < values.shape[0] and index < compact.shape[0]:
            values[value_index] = compact[index]
    return values


def _advanced_2d_features(props: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for row in range(3):
        for column in range(4):
            features[
                _indexed_shape_feature(
                    ObjectShapeMeasurementFeature.SPATIAL_MOMENT,
                    row,
                    column,
                )
            ] = props[f"moments-{row}-{column}"]
            features[
                _indexed_shape_feature(
                    ObjectShapeMeasurementFeature.CENTRAL_MOMENT,
                    row,
                    column,
                )
            ] = props[f"moments_central-{row}-{column}"]

    for row in range(4):
        for column in range(4):
            features[
                _indexed_shape_feature(
                    ObjectShapeMeasurementFeature.NORMALIZED_MOMENT,
                    row,
                    column,
                )
            ] = props[f"moments_normalized-{row}-{column}"]

    for index in range(7):
        features[
            _indexed_shape_feature(ObjectShapeMeasurementFeature.HU_MOMENT, index)
        ] = props[f"moments_hu-{index}"]

    for row in range(2):
        for column in range(2):
            features[
                _indexed_shape_feature(
                    ObjectShapeMeasurementFeature.INERTIA_TENSOR,
                    row,
                    column,
                )
            ] = props[f"inertia_tensor-{row}-{column}"]

    for index in range(2):
        features[
            _indexed_shape_feature(
                ObjectShapeMeasurementFeature.INERTIA_TENSOR_EIGENVALUES,
                index,
            )
        ] = props[f"inertia_tensor_eigvals-{index}"]
    return features


def _cellprofiler_orientation_degrees(props: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(props["orientation"], dtype=float) * (180 / np.pi)


def _zernike_features(
    labels: np.ndarray,
    measured_labels: np.ndarray,
    *,
    backend_provider: CellProfilerBackendProvider | None,
) -> dict[str, np.ndarray]:
    zernike_numbers, zernike_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=_ZERNIKE_MAX_ORDER,
        backend_provider=backend_provider,
    )
    return {
        _indexed_shape_feature(ObjectShapeMeasurementFeature.ZERNIKE, int(n), int(m)): (
            values
        )
        for (n, m), values in zip(zernike_numbers, zernike_values.transpose())
    }


def _surface_area(volume: np.ndarray) -> float:
    if not np.any(volume):
        return 0.0
    try:
        verts, faces, _normals, _values = skimage.measure.marching_cubes(
            volume,
            method="lewiner",
            spacing=(1.0,) * volume.ndim,
            level=0,
        )
    except ValueError:
        return 0.0
    return float(skimage.measure.mesh_surface_area(verts, faces))


def _missing_shape_feature_value(feature_name: str) -> float:
    if feature_name in {
        _shape_feature(ObjectShapeMeasurementFeature.MAXIMUM_RADIUS),
        _shape_feature(ObjectShapeMeasurementFeature.MEAN_RADIUS),
        _shape_feature(ObjectShapeMeasurementFeature.MEDIAN_RADIUS),
    }:
        return 0.0
    return np.nan


def _prepare_measure_object_size_shape() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 1
    measure_object_size_shape.__wrapped__(image, labels)


measure_object_size_shape.__openhcs_prepare__ = _prepare_measure_object_size_shape
