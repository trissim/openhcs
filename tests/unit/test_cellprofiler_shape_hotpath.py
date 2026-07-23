"""Exactness checks for the CellProfiler object-size/shape hot path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
import skimage.measure

from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    ObjectFeatureValueTable,
)
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelRepresentation,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.core.runtime_tabular_values import MeasurementObjectRowIdentity
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
    ShapeObjectFeatureValueTable,
    measure_object_size_shape,
    measure_object_size_shape_feature_arrays,
)


def _assert_rows_strict(
    actual_rows: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object]],
    field_names: tuple[str, ...],
) -> None:
    assert len(actual_rows) == len(expected_rows)
    axis_fields = frozenset(MeasurementRowAxisField.field_names())
    for field_name in field_names:
        actual = np.asarray([row[field_name] for row in actual_rows])
        expected = np.asarray([row[field_name] for row in expected_rows])
        if field_name in axis_fields:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=0.0,
                atol=1e-6,
                equal_nan=True,
            )


def _generic_shape_rows(
    labels: np.ndarray,
    *,
    calculate_advanced: bool,
    calculate_zernikes: bool,
) -> list[dict[str, float | int]]:
    feature_values, measured_labels = measure_object_size_shape_feature_arrays(
        labels,
        calculate_advanced=calculate_advanced,
        calculate_zernikes=calculate_zernikes,
    )
    table = ShapeObjectFeatureValueTable.from_feature_arrays(
        feature_values,
        measured_labels,
        object_domain=range(1, int(labels.max(initial=0)) + 1),
    )
    return ObjectFeatureValueTable.rows(table)


def test_all_enabled_2d_shape_vectors_match_generic_row_projection() -> None:
    labels = np.zeros((32, 36), dtype=np.int32)
    labels[3:12, 4:15] = 1
    labels[18:29, 20:33] = 3
    labels[18:22, 20:24] = 0
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1, 3)),
    )

    _image, actual_rows = measure_object_size_shape(
        np.zeros(labels.shape, dtype=np.float32),
        payload,
        calculate_advanced=True,
        calculate_zernikes=True,
    )
    expected_rows = _generic_shape_rows(
        labels,
        calculate_advanced=True,
        calculate_zernikes=True,
    )
    field_names = MeasureObjectSizeShapeModule.measurement_field_names(
        dimensions=2,
        calculate_advanced=True,
        calculate_zernikes=True,
    )

    assert tuple(field.name for field in actual_rows.fields) == field_names
    assert actual_rows.object_row_identity is MeasurementObjectRowIdentity.ROW_SEQUENCE
    assert tuple(field.dtype for field in actual_rows.fields[:2]) == (int, int)
    assert all(field.dtype is float for field in actual_rows.fields[2:])
    _assert_rows_strict(actual_rows, expected_rows, field_names)


@pytest.mark.parametrize(
    ("mask", "expected_orientation"),
    (
        (np.eye(5, dtype=np.int32), -45.0),
        (np.fliplr(np.eye(5, dtype=np.int32)), 45.0),
    ),
)
def test_shape_orientation_ties_remain_exact(
    mask: np.ndarray,
    expected_orientation: float,
) -> None:
    labels = np.pad(mask, 2)

    feature_values, measured_labels = measure_object_size_shape_feature_arrays(
        labels,
        calculate_advanced=True,
        calculate_zernikes=False,
    )

    np.testing.assert_array_equal(measured_labels, np.asarray([1]))
    assert feature_values["Orientation"][0] == expected_orientation


def test_nonempty_stacked_shape_vectors_and_surface_areas_match_oracles() -> None:
    labels = np.zeros((7, 18, 20), dtype=np.int32)
    labels[1:5, 2:9, 3:11] = 1
    labels[2:6, 10:16, 12:18] = 2
    labels[2:4, 12:14, 12:15] = 0
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )

    _image, actual_rows = measure_object_size_shape(
        np.zeros(labels.shape, dtype=np.float32),
        payload,
        calculate_advanced=True,
        calculate_zernikes=False,
    )
    expected_rows = _generic_shape_rows(
        labels,
        calculate_advanced=True,
        calculate_zernikes=False,
    )
    field_names = MeasureObjectSizeShapeModule.measurement_field_names(
        dimensions=3,
        calculate_advanced=True,
        calculate_zernikes=False,
    )

    assert tuple(field.name for field in actual_rows.fields) == field_names
    _assert_rows_strict(actual_rows, expected_rows, field_names)

    props = skimage.measure.regionprops_table(labels, properties=("label", "bbox"))
    expected_surface_areas = []
    for object_index, label_id in enumerate(props["label"]):
        minimum = tuple(int(props[f"bbox-{axis}"][object_index]) for axis in range(3))
        maximum = tuple(
            int(props[f"bbox-{axis}"][object_index]) for axis in range(3, 6)
        )
        bounds = tuple(
            slice(
                max(minimum[axis] - 1, 0),
                min(maximum[axis] + 1, labels.shape[axis]),
            )
            for axis in range(3)
        )
        volume = labels[bounds] == int(label_id)
        vertices, faces, _normals, _values = skimage.measure.marching_cubes(
            volume,
            method="lewiner",
            spacing=(1.0, 1.0, 1.0),
            level=0,
        )
        expected_surface_areas.append(
            skimage.measure.mesh_surface_area(vertices, faces)
        )

    np.testing.assert_allclose(
        [row["SurfaceArea"] for row in actual_rows],
        expected_surface_areas,
        rtol=0.0,
        atol=1e-6,
    )


def test_empty_stacked_shape_preserves_declared_metadata_and_dtypes() -> None:
    labels = np.zeros((4, 9, 11), dtype=np.int32)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=()),
    )

    _image, rows = measure_object_size_shape(
        np.zeros(labels.shape, dtype=np.float32),
        payload,
        calculate_advanced=False,
        calculate_zernikes=False,
    )
    field_names = MeasureObjectSizeShapeModule.measurement_field_names(
        dimensions=3,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert len(rows) == 0
    assert rows.object_row_identity is MeasurementObjectRowIdentity.ROW_SEQUENCE
    assert tuple(field.name for field in rows.fields) == field_names
    assert tuple(field.dtype for field in rows.fields[:2]) == (int, int)
    assert all(field.dtype is float for field in rows.fields[2:])
    assert tuple(rows.columns) == field_names
    assert all(values == () for values in rows.columns.values())


def test_sparse_high_id_shape_preserves_label_domain_and_row_identity() -> None:
    sparse_labels = SparseIJVLabelRows(
        np.asarray(
            (
                (2, 3, 892),
                (2, 4, 892),
                (3, 3, 892),
                (3, 4, 892),
            ),
            dtype=np.int32,
        )
    )
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=sparse_labels),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        domain=ObjectLabelDomain(declared_object_ids=(892,)),
    )

    _image, rows = measure_object_size_shape(
        np.zeros((6, 8), dtype=np.float32),
        payload,
        calculate_advanced=False,
        calculate_zernikes=False,
    )

    assert len(rows) == 1
    assert rows.object_row_identity is MeasurementObjectRowIdentity.ROW_SEQUENCE
    assert rows[0][MeasurementRowAxisField.OBJECT_LABEL.value] == 892
    assert rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.AREA.value] == 4.0
    assert (
        rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_X.value] == 3.5
    )
    assert (
        rows[0][MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Y.value] == 2.5
    )
