from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

import pytest

from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_tabular_values import ColumnarRows, is_table_payload


@dataclass(frozen=True, slots=True)
class _TestColumnarRows(ColumnarRows):
    columns: dict[str, tuple[object, ...]]
    fields: tuple[FieldSpec, ...] = ()

    def __post_init__(self) -> None:
        self.validate_fields()


@dataclass(frozen=True, slots=True)
class _TestMeasurementRow:
    feature_name: str
    value: float


def test_table_payload_accepts_only_nominal_columnar_rows() -> None:
    assert is_table_payload(
        _TestColumnarRows(
            {"value": (1.0,)}, fields=(FieldSpec("value", float),)
        )
    )


@pytest.mark.parametrize(
    "payload",
    (
        {"feature_name": "Area", "value": 1.0},
        _TestMeasurementRow(feature_name="Area", value=1.0),
        [],
        (),
    ),
)
def test_table_payload_rejects_structural_rows_and_sequences(payload: object) -> None:
    assert not is_table_payload(payload)


def test_measurement_table_rejects_arbitrary_sequence_before_row_layout() -> None:
    rows = range(2)

    assert isinstance(rows, Sequence)
    assert not is_table_payload(rows)
    with pytest.raises(TypeError, match="requires schema-bearing ColumnarRows"):
        MeasurementTable(
            name="InvalidMeasurements",
            rows=rows,
            subject=MeasurementSubject(
                MeasurementScope.ARTIFACT,
                "InvalidMeasurements",
            ),
        )


def test_columnar_rows_rejects_schema_less_runtime_value_inference() -> None:
    with pytest.raises(ValueError, match="field/column names and order"):
        _TestColumnarRows({"value": (1.0,)})


def test_columnar_rows_rejects_field_column_order_mismatch_immediately() -> None:
    with pytest.raises(ValueError, match="field/column names and order"):
        _TestColumnarRows(
            {"first": (), "second": ()},
            fields=(FieldSpec("second", int), FieldSpec("first", str)),
        )


def test_math_result_columnar_rows_owns_exact_zero_row_schema() -> None:
    from openhcs.processing.backends.cellprofiler.measurement_math import (
        MathResultColumnarRows,
    )

    rows = MathResultColumnarRows(
        MappingProxyType(
            {
                "slice_index": (),
                "object_name": (),
                "object_label": (),
                "output_name": (),
                "feature_name": (),
                "result_value": (),
            }
        )
    )

    assert rows.fields == (
        FieldSpec("slice_index", int),
        FieldSpec("object_name", str, required=False),
        FieldSpec("object_label", int, required=False),
        FieldSpec("output_name", str),
        FieldSpec("feature_name", str),
        FieldSpec("result_value", float),
    )
