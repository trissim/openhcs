"""Exact regressions for the CellProfiler 3-D object-intensity leaf."""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    ObjectIntensityMeasurementRows,
    ObjectIntensityPreparedLabels,
    object_intensity_backend,
)


def test_object_intensity_3d_batch_preserves_all_features_and_row_axes(
    monkeypatch,
) -> None:
    z_indices, y_indices, x_indices = np.indices((60, 48, 48))
    image_a = np.ascontiguousarray(
        ((z_indices % 5) * 7 + y_indices % 11 + x_indices % 13).astype(np.float32)
        / np.float32(100.0)
    )
    image_b = np.ascontiguousarray(
        (
            ((59 - z_indices) % 7) * 3 + (y_indices // 2) % 9 + (x_indices // 3) % 6
        ).astype(np.float32)
        / np.float32(50.0)
    )
    image_a[3, 4, 5] = image_a[4, 5, 6] = np.float32(2.0)
    image_b[20, 22, 9] = image_b[21, 23, 10] = np.float32(2.0)
    labels = np.zeros(image_a.shape, dtype=np.int32)
    labels[2:40, 3:21, 4:22] = 2
    labels[15:57, 27:44, 25:44] = 7
    labels[5:30, 21:26, 8:18] = 100_003
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(2, 7, 42, 100_003)),
    )
    prepared = ObjectIntensityPreparedLabels.from_source(label_payload, labels)
    foreground_index = prepared.foreground_index
    assert foreground_index is not None
    assert prepared.object_count == 3
    assert prepared.object_domain == (2, 7, 42, 100_003)
    assert prepared.measurement_row_identity is not None
    assert prepared.dense_labels.dtype == np.int32

    backend = object_intensity_backend()
    images = (image_a, image_b)
    image_batch = np.ascontiguousarray(np.stack(images, axis=0))
    measured_without_locations = backend._measure_sparse_prepared_batch(
        image_batch,
        prepared,
        foreground_index,
    )
    expected = tuple(
        measurements.with_max_intensity_positions(
            backend.maximum_intensity_positions(image, prepared)
        )
        for image, measurements in zip(
            images,
            measured_without_locations,
            strict=True,
        )
    )

    def reject_repeated_position_scan(*_args, **_kwargs):
        raise AssertionError("3-D batches must not repeat full-volume position scans")

    monkeypatch.setattr(
        type(backend),
        "maximum_intensity_positions",
        reject_repeated_position_scan,
    )
    actual = backend.measure_prepared_batch(images, prepared)

    axis_fields = {
        MeasurementRowAxisField.SLICE_INDEX.value,
        MeasurementRowAxisField.OBJECT_LABEL.value,
    }
    expected_field_names = tuple(
        field.name for field in ObjectIntensityMeasurementRows.fields
    )
    assert len(expected_field_names) * len(images) == 46
    for image_index, (expected_arrays, actual_arrays) in enumerate(
        zip(expected, actual, strict=True)
    ):
        slice_index = 37 + image_index
        expected_rows = ObjectIntensityMeasurementRows.from_arrays(
            expected_arrays,
            slice_index=slice_index,
            object_domain=prepared.object_domain,
            object_row_identity=prepared.measurement_row_identity,
        )
        actual_rows = ObjectIntensityMeasurementRows.from_arrays(
            actual_arrays,
            slice_index=slice_index,
            object_domain=prepared.object_domain,
            object_row_identity=prepared.measurement_row_identity,
        )
        assert tuple(actual_rows.columns) == expected_field_names
        assert len(actual_rows) == 4
        assert actual_rows.object_row_identity is prepared.measurement_row_identity
        assert (
            actual_rows.columns[MeasurementRowAxisField.SLICE_INDEX.value].dtype
            == np.int64
        )
        assert (
            actual_rows.columns[MeasurementRowAxisField.OBJECT_LABEL.value].dtype
            == np.int32
        )
        for field_name in expected_field_names:
            if field_name in axis_fields:
                np.testing.assert_array_equal(
                    actual_rows.columns[field_name],
                    expected_rows.columns[field_name],
                )
            else:
                np.testing.assert_allclose(
                    actual_rows.columns[field_name],
                    expected_rows.columns[field_name],
                    rtol=0.0,
                    atol=1e-6,
                    equal_nan=True,
                )
                assert actual_rows.columns[field_name].dtype == np.float64


def test_prepared_labels_validate_the_existing_present_id_projection() -> None:
    source = textwrap.dedent(
        inspect.getsource(ObjectIntensityPreparedLabels.from_source)
    )
    tree = ast.parse(source)
    validation_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dense_object_label_measurement_row_domain"
    ]

    assert len(validation_calls) == 1
    validation_call = validation_calls[0]
    assert len(validation_call.args) == 2
    assert ast.dump(validation_call.args[1], include_attributes=False) == ast.dump(
        ast.Attribute(
            value=ast.Name(id="projection", ctx=ast.Load()),
            attr="positive_label_ids",
            ctx=ast.Load(),
        ),
        include_attributes=False,
    )


def test_prepared_labels_reject_undeclared_sparse_high_id() -> None:
    labels = np.zeros((60, 8, 8), dtype=np.int32)
    labels[3:8, 1:4, 2:5] = 7
    labels[40:44, 4:7, 3:6] = 100_003
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(7, 42)),
    )

    with pytest.raises(
        ValueError,
        match=r"pixels contain IDs outside the declared measurement-row domain: \(100003,\)",
    ):
        ObjectIntensityPreparedLabels.from_source(label_payload, labels)
