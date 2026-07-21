from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import openhcs.core.measurement_row_materialization as row_materialization
from openhcs.core.measurement_row_materialization import (
    ColumnarMeasurementRowsAxisProjection,
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
    MeasurementRowQualifier,
    MeasurementRowsAxisProjection,
    MeasurementSparseColumnarRows,
    QualifiedMeasurementColumnarRows,
    is_structural_missing_measurement_cell,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)


@dataclass(frozen=True, slots=True)
class _AnnotatedRow:
    object_label: int
    value: float | None


def test_zero_row_projected_and_sparse_carriers_preserve_exact_fields() -> None:
    fields = (FieldSpec("object_label", int), FieldSpec("value", float))

    projected = MeasurementProjectedColumnarRows(
        {"object_label": (), "value": ()},
        fields=fields,
    )
    sparse = MeasurementSparseColumnarRows.from_rows((), fields=fields)

    assert projected.fields == fields
    assert sparse.fields == fields
    assert tuple(projected.columns) == ("object_label", "value")
    assert tuple(sparse.columns) == ("object_label", "value")
    assert projected.row_count() == sparse.row_count() == 0


def test_projected_mapping_rejects_runtime_value_schema_inference() -> None:
    with pytest.raises(ValueError, match="field/column names and order"):
        MeasurementProjectedColumnarRows({"value": (1.0,)})


def test_projected_rows_select_columns_and_preserve_sparse_metadata() -> None:
    fields = (FieldSpec("object_label", int), FieldSpec("value", float))
    source = MeasurementSparseColumnarRows.from_rows(
        (
            {"object_label": 1, "value": 2.5},
            {"object_label": 3},
        ),
        fields=fields,
        declared_object_measurement_domain_covered=True,
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )

    reused = MeasurementProjectedColumnarRows.from_columnar_rows(
        source,
        declared_object_measurement_domain_covered=True,
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )
    selected = MeasurementProjectedColumnarRows.from_columnar_rows(
        source,
        row_indices=(1, 0),
        declared_object_measurement_domain_covered=True,
        object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
    )

    assert reused.columns["object_label"] is source.columns["object_label"]
    assert reused.columns["value"] is source.columns["value"]
    assert selected.fields == fields
    assert selected.covers_declared_object_measurement_domain
    assert selected.object_row_identity is MeasurementObjectRowIdentity.LABEL_ID
    assert selected.row_mappings() == (
        {"object_label": 3},
        {"object_label": 1, "value": 2.5},
    )
    assert is_structural_missing_measurement_cell(selected.columns["value"][0])


def test_projected_rows_reject_indices_outside_columnar_domain() -> None:
    rows = MeasurementProjectedColumnarRows(
        {"value": (1.0,)},
        fields=(FieldSpec("value", float),),
    )

    with pytest.raises(IndexError, match="outside the row domain"):
        MeasurementProjectedColumnarRows.from_columnar_rows(
            rows,
            row_indices=(1,),
            declared_object_measurement_domain_covered=False,
            object_row_identity=None,
        )


def test_zero_row_dataclass_carrier_uses_nominal_annotations() -> None:
    rows = DataclassMeasurementColumnarRows((), row_type=_AnnotatedRow)

    assert rows.fields == (
        FieldSpec("object_label", int),
        FieldSpec("value", float, required=False),
    )
    assert tuple(rows.columns) == ("object_label", "value")
    assert rows.columns == {"object_label": (), "value": ()}


def test_zero_row_dataclass_carrier_requires_explicit_row_type() -> None:
    with pytest.raises(TypeError, match="requires row_type for zero rows"):
        DataclassMeasurementColumnarRows(())


def test_zero_row_qualified_carrier_appends_declared_string_field() -> None:
    source = MeasurementProjectedColumnarRows(
        {"value": ()},
        fields=(FieldSpec("value", float),),
    )

    qualified = QualifiedMeasurementColumnarRows(
        source,
        (MeasurementRowQualifier("object_name", "Cells"),),
    )

    assert qualified.fields == (
        FieldSpec("value", float),
        FieldSpec("object_name", str),
    )
    assert tuple(qualified.columns) == ("value", "object_name")
    assert qualified.columns["object_name"] == ()


def test_runtime_slice_projection_preserves_fields_and_appends_exact_axis() -> None:
    source = MeasurementProjectedColumnarRows(
        {"value": ()},
        fields=(FieldSpec("value", float),),
    )

    projected = ColumnarMeasurementRowsAxisProjection(
        source
    ).project_runtime_slice_index(3)

    assert projected.fields == (
        FieldSpec("value", float),
        FieldSpec("slice_index", int),
    )
    assert tuple(projected.columns) == ("value", "slice_index")
    assert projected.columns["slice_index"] == ()


def test_runtime_slice_projection_preserves_concatenated_batch_schemas() -> None:
    image_rows = MeasurementProjectedColumnarRows(
        {"slice_index": (0,), "slope": (0.25,)},
        fields=(FieldSpec("slice_index", int), FieldSpec("slope", float)),
    )
    object_rows = MeasurementProjectedColumnarRows(
        {"slice_index": (0,), "object_label": (1,), "correlation": (0.5,)},
        fields=(
            FieldSpec("slice_index", int),
            FieldSpec("object_label", int),
            FieldSpec("correlation", float),
        ),
    )

    projected = MeasurementRowsAxisProjection.from_rows(
        ConcatenatedColumnarRows((image_rows, object_rows))
    ).project_runtime_slice_index(3)

    assert isinstance(projected, ConcatenatedColumnarRows)
    projected_image_rows, projected_object_rows = projected.row_batches
    assert projected_image_rows.fields == image_rows.fields
    assert projected_object_rows.fields == object_rows.fields
    assert projected_image_rows.columns["slice_index"] == (3,)
    assert projected_object_rows.columns["slice_index"] == (3,)


def test_concatenation_exact_merges_fields_in_first_declared_order() -> None:
    first = MeasurementProjectedColumnarRows(
        {"object_label": (1,), "area": (2.5,)},
        fields=(FieldSpec("object_label", int), FieldSpec("area", float)),
    )
    second = MeasurementProjectedColumnarRows(
        {"object_label": (2,), "class_name": ("large",)},
        fields=(FieldSpec("object_label", int), FieldSpec("class_name", str)),
    )

    rows = ConcatenatedColumnarRows((first, second))

    assert rows.fields == (
        FieldSpec("object_label", int),
        FieldSpec("area", float),
        FieldSpec("class_name", str),
    )
    assert tuple(rows.columns) == ("object_label", "area", "class_name")
    assert rows.columns["object_label"].tolist() == [1, 2]
    area_values = rows.columns["area"].tolist()
    class_values = rows.columns["class_name"].tolist()
    assert area_values[0] == 2.5
    assert is_structural_missing_measurement_cell(area_values[1])
    assert is_structural_missing_measurement_cell(class_values[0])
    assert class_values[1] == "large"


def test_dense_concatenation_preserves_numpy_dtype_and_column_cache() -> None:
    fields = (FieldSpec("value", float),)
    rows = ConcatenatedColumnarRows(
        (
            MeasurementProjectedColumnarRows(
                {"value": np.asarray((1.25, 2.5), dtype=np.float32)},
                fields=fields,
            ),
            MeasurementProjectedColumnarRows(
                {"value": np.asarray((3.75,), dtype=np.float32)},
                fields=fields,
            ),
        )
    )

    values = rows.columns["value"]

    assert values.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        values,
        np.asarray((1.25, 2.5, 3.75), dtype=np.float32),
    )
    assert rows.columns["value"] is values


def test_sparse_concatenation_preserves_gaps_order_and_marker_identity() -> None:
    value_field = (FieldSpec("value", float),)
    other_field = (FieldSpec("other", int),)
    rows = ConcatenatedColumnarRows(
        (
            MeasurementProjectedColumnarRows(
                {"value": np.asarray((1.25, 2.5), dtype=np.float32)},
                fields=value_field,
            ),
            MeasurementProjectedColumnarRows(
                {"other": np.asarray((10,), dtype=np.int32)},
                fields=other_field,
            ),
            MeasurementProjectedColumnarRows(
                {"value": np.asarray((3.75,), dtype=np.float32)},
                fields=value_field,
            ),
            MeasurementProjectedColumnarRows(
                {"other": np.asarray((20, 30), dtype=np.int32)},
                fields=other_field,
            ),
        )
    )

    values = rows.columns["value"]
    other_values = rows.columns["other"]

    assert values.dtype == other_values.dtype == np.dtype(object)
    assert values[:2].tolist() == [1.25, 2.5]
    assert values[3] == 3.75
    assert other_values[2] == 10
    assert other_values[4:].tolist() == [20, 30]
    assert all(
        is_structural_missing_measurement_cell(values[index])
        for index in (2, 4, 5)
    )
    assert all(
        is_structural_missing_measurement_cell(other_values[index])
        for index in (0, 1, 3)
    )
    missing_cell = values[2]
    assert values[4] is missing_cell and values[5] is missing_cell
    assert rows.columns["value"] is values


def test_sparse_concatenation_ast_uses_one_fill_instead_of_gap_tuples() -> None:
    source_path = Path(row_materialization.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ConcatenatedColumnarRowColumns"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getitem__"
    )

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fill"
        for node in ast.walk(method)
    )
    assert not any(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and any(
            isinstance(child, ast.Name)
            and child.id == "MEASUREMENT_SPARSE_CELL"
            for child in ast.walk(node)
        )
        for node in ast.walk(method)
    )


def test_zero_row_concatenation_preserves_named_columns() -> None:
    first = MeasurementProjectedColumnarRows(
        {"first": ()},
        fields=(FieldSpec("first", int),),
    )
    second = MeasurementProjectedColumnarRows(
        {"second": ()},
        fields=(FieldSpec("second", str),),
    )

    rows = ConcatenatedColumnarRows((first, second))

    assert rows.fields == (FieldSpec("first", int), FieldSpec("second", str))
    assert tuple(rows.columns) == ("first", "second")
    assert rows.columns["first"].size == rows.columns["second"].size == 0


@pytest.mark.parametrize(
    "conflicting_field",
    (
        FieldSpec("value", float),
        FieldSpec("value", int, required=False),
    ),
)
def test_concatenation_rejects_same_name_field_conflicts(
    conflicting_field: FieldSpec,
) -> None:
    first = MeasurementProjectedColumnarRows(
        {"value": (1,)},
        fields=(FieldSpec("value", int),),
    )
    second = MeasurementProjectedColumnarRows(
        {"value": (2,)},
        fields=(conflicting_field,),
    )

    with pytest.raises(ValueError, match="Conflicting concatenated columnar"):
        ConcatenatedColumnarRows((first, second))


def test_sparse_rows_require_declared_fields_instead_of_value_inference() -> None:
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        MeasurementSparseColumnarRows.from_rows(({"value": 1.0},))


def test_sparse_rows_reject_columns_absent_from_declared_fields() -> None:
    with pytest.raises(ValueError, match="absent from their declared fields"):
        MeasurementSparseColumnarRows.from_rows(
            ({"value": 1.0},),
            fields=(FieldSpec("other", float),),
        )
