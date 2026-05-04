"""
Converted from CellProfiler: MeasureObjectSizeShape
Original: measureobjectsizeshape
"""

import numpy as np
import skimage.measure
import skimage.morphology
from typing import Any
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_outputs

from openhcs.core.runtime_semantics import (
    ObjectShapeMeasurementFeature,
    indexed_measurement_feature_name,
    object_shape_measurement_field_names,
)
from openhcs.processing.backends.cellprofiler.shape import ShapeMeasurementBackendStrategy
from openhcs.processing.backends.cellprofiler.zernike import shape_zernike_moments
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.materialization import csv_materializer


_ZERNIKE_MAX_ORDER = 9


@numpy
@special_outputs(
    (
        "measurements",
        csv_materializer(fields=list(object_shape_measurement_field_names())),
    )
)
def measure_object_size_shape(
    image: np.ndarray,
    labels: np.ndarray,
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
    label_array = labels.astype(np.int32, copy=False)
    if not np.any(label_array > 0):
        return image, []

    feature_values, measured_labels = _measure_object_size_shape_features(
        label_array,
        calculate_advanced=calculate_advanced,
        calculate_zernikes=calculate_zernikes,
        shape_backend_provider=shape_backend_provider,
        zernike_backend_provider=zernike_backend_provider,
    )
    rows = _object_size_shape_rows(feature_values, measured_labels)
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
    shape_backend = ShapeMeasurementBackendStrategy.for_memory_type(
        backend_provider=shape_backend_provider,
    )
    fast_region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        labels
    )
    props = fast_region_props.as_regionprops_table_subset()
    convex_area, solidity = _convex_area_and_solidity_from_labels(
        labels,
        fast_region_props,
    )
    props["convex_area"] = convex_area
    props["solidity"] = solidity
    measured_labels = np.asarray(props["label"])
    nobjects = len(measured_labels)
    if nobjects == 0:
        return {}, measured_labels

    perimeter = np.asarray(props["perimeter"], dtype=float)
    area = np.asarray(props["area"], dtype=float)
    max_radius, mean_radius, median_radius = shape_backend.radius_features_from_labels(
        labels,
        measured_labels,
    )
    dense_labels = np.arange(1, int(np.max(labels)) + 1, dtype=measured_labels.dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        form_factor = 4.0 * np.pi * area / perimeter**2
    with np.errstate(divide="ignore", invalid="ignore"):
        compactness = 1.0 / form_factor
    min_feret_diameter, max_feret_diameter = shape_backend.feret_diameters(
        labels,
        dense_labels,
    )

    center_x = _compact_values_with_dense_tail(
        np.asarray(props["centroid-1"], dtype=float),
        _dense_label_centers_2d(labels, axis=1),
    )
    center_y = _compact_values_with_dense_tail(
        np.asarray(props["centroid-0"], dtype=float),
        _dense_label_centers_2d(labels, axis=0),
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
        features.update(_advanced_2d_features(props))
    if calculate_zernikes:
        features.update(
            _zernike_features(
                labels,
                measured_labels,
                backend_provider=zernike_backend_provider,
            )
        )
    return features, measured_labels


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
    object_count = int(region_props.label.size)
    convex_area = np.zeros(object_count, dtype=float)
    solidity = np.ones(object_count, dtype=float)
    for index, label_id in enumerate(region_props.label):
        min_y = int(region_props.bbox_min_y[index])
        min_x = int(region_props.bbox_min_x[index])
        max_y = int(region_props.bbox_max_y[index])
        max_x = int(region_props.bbox_max_x[index])
        crop = labels[min_y:max_y, min_x:max_x] == int(label_id)
        hull = skimage.morphology.convex_hull_image(crop)
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
    feature_items = tuple(
        (
            feature_name,
            np.asarray(values),
            _missing_shape_feature_value(feature_name),
        )
        for feature_name, values in feature_values.items()
    )
    row_count = max(
        len(np.asarray(measured_labels)),
        *(
            values.shape[0]
            for _feature_name, values, _missing_value in feature_items
            if values.ndim > 0
        ),
        0,
    )
    for index in range(row_count):
        row: dict[str, float | int] = {
            "slice_index": 0,
            "object_label": index + 1,
            "Center_Z": 0.0,
        }
        for feature_name, values, missing_value in feature_items:
            if values.ndim > 0:
                if index >= values.shape[0]:
                    value = missing_value
                else:
                    value = values[index]
            else:
                value = values.item()
            row[feature_name] = value.item() if isinstance(value, np.generic) else value
        rows.append(row)
    return rows


def _shape_feature(feature: ObjectShapeMeasurementFeature) -> str:
    return feature.value


def _indexed_shape_feature(
    feature: ObjectShapeMeasurementFeature,
    *indices: int,
) -> str:
    return indexed_measurement_feature_name(feature, *indices)


def _dense_label_centers_2d(labels: np.ndarray, *, axis: int) -> np.ndarray:
    labels_int = labels.astype(np.intp, copy=False)
    max_label = int(labels_int.max()) if labels_int.size else 0
    if max_label <= 0:
        return np.zeros(0, dtype=float)

    valid = labels_int > 0
    valid_labels = labels_int[valid]
    counts = np.bincount(valid_labels, minlength=max_label + 1).astype(float)
    coordinates = np.indices(labels_int.shape, sparse=False)[axis]
    sums = np.bincount(
        valid_labels,
        weights=coordinates[valid],
        minlength=max_label + 1,
    )
    centers = np.full(max_label + 1, np.nan, dtype=float)
    np.divide(sums, counts, out=centers, where=counts > 0)
    return centers[1:]


def _compact_values_with_dense_tail(
    compact_values: np.ndarray,
    dense_values: np.ndarray,
) -> np.ndarray:
    compact = np.asarray(compact_values, dtype=float)
    dense = np.asarray(dense_values, dtype=float)
    if dense.shape[0] <= compact.shape[0]:
        return compact
    values = dense.copy()
    values[: compact.shape[0]] = compact
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
