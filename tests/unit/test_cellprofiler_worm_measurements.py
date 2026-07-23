from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.measurement_row_materialization import MeasurementSparseColumnarRows
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.interop.cellprofiler.worm_measurements import (
    WormControlPointAxis,
    WormControlPointMeasurementSchema,
)


def _columnar_rows(rows: list[dict[str, object]]) -> MeasurementSparseColumnarRows:
    field_names = tuple(dict.fromkeys(field_name for row in rows for field_name in row))
    return MeasurementSparseColumnarRows.from_rows(
        rows,
        fields=tuple(
            FieldSpec(
                field_name,
                type(next(row[field_name] for row in rows if field_name in row)),
            )
            for field_name in field_names
        ),
    )


def test_control_points_select_descriptor_rows_from_mixed_measurements() -> None:
    schema = WormControlPointMeasurementSchema(num_control_points=2)
    first_control_points = np.array(((1.0, 11.0), (2.0, 12.0)))
    second_control_points = np.array(((3.0, 13.0), (4.0, 14.0)))
    object_name_field = MeasurementRowAxisField.OBJECT_NAME.value
    object_number_field = MeasurementRowAxisField.OBJECT_NUMBER.value
    object_label_field = MeasurementRowAxisField.OBJECT_LABEL.value
    rows = [
        {
            object_number_field: 2,
            object_name_field: "Worms",
            **schema.row_fields(second_control_points),
        },
        {
            object_label_field: 1,
            object_name_field: "Worms",
        },
        {
            object_number_field: 1,
            object_name_field: "OtherObjects",
            **schema.row_fields(second_control_points),
        },
        {
            object_number_field: 1,
            object_name_field: "Worms",
            **schema.row_fields(first_control_points),
        },
    ]

    control_points = schema.control_points_from_rows(
        _columnar_rows(rows),
        object_name="Worms",
    )

    np.testing.assert_array_equal(
        control_points,
        np.stack((first_control_points.T, second_control_points.T)),
    )


def test_control_points_reject_malformed_descriptor_row() -> None:
    schema = WormControlPointMeasurementSchema(num_control_points=2)
    missing_field = schema.field(WormControlPointAxis.COLUMN, 2)
    descriptor_row = {
        MeasurementRowAxisField.OBJECT_NUMBER.value: 1,
        MeasurementRowAxisField.OBJECT_NAME.value: "Worms",
        **schema.row_fields(np.array(((1.0, 11.0), (2.0, 12.0)))),
    }
    descriptor_row.pop(missing_field.name)

    with pytest.raises(ValueError) as exc_info:
        schema.control_points_from_rows(
            _columnar_rows([descriptor_row]),
            object_name="Worms",
        )

    assert repr(missing_field.name) in str(exc_info.value)
