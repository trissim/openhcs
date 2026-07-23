from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ObjectLocationMeasurementRows,
)
from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
    RelationshipMeasurementRows,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureImageIntensityModule,
    measure_image_intensity,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    RelateObjectsRelationshipMeasurementRows,
)
from openhcs.processing.backends.cellprofiler.worms import (
    _worm_descriptor_rows,
    identify_dead_worms,
    untangle_worms,
)


@dataclass(frozen=True, slots=True)
class _DeclaredRelationshipRows(RelationshipMeasurementRows):
    def object_numbers_by_label_id(
        self,
        spec: object,
        *,
        slice_index: int | None,
        slice_count: int | None = None,
    ) -> dict[int, int]:
        del spec, slice_index, slice_count
        return {7: 1, 9: 2}


def test_object_location_rows_are_schema_bearing_columnar_rows() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1], [0, 1]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )

    rows = ObjectLocationMeasurementRows(payload, object_name="Cells").rows()

    assert isinstance(rows, ColumnarRows)
    assert rows.row_count() == 6
    assert tuple(field.name for field in rows.fields) == (
        MeasurementRowAxisField.OBJECT_NAME.value,
        MeasurementRowAxisField.OBJECT_LABEL.value,
        MeasurementRowAxisField.SLICE_INDEX.value,
        MeasurementRowAxisField.FEATURE_NAME.value,
        "result_value",
    )


def test_image_intensity_producer_and_projector_are_columnar() -> None:
    _image, source_rows = measure_image_intensity(
        np.asarray([[0.0, 1.0]], dtype=np.float32)
    )

    projected_rows = MeasureImageIntensityModule.MeasurementRows(
        source_rows,
        module_type=MeasureImageIntensityModule,
        measurement_name="DNA",
    ).rows()

    assert isinstance(source_rows, ColumnarRows)
    assert isinstance(projected_rows, ColumnarRows)
    assert projected_rows.row_count() == 11
    assert {
        row[MeasurementRowAxisField.FEATURE_NAME.value]
        for row in projected_rows.iter_row_mappings()
    } == {
        MeasureImageIntensityModule.measurement_feature_name(
            field_name,
            "DNA",
        )
        for field_name in (
            "total_intensity",
            "mean_intensity",
            "median_intensity",
            "std_intensity",
            "mad_intensity",
            "min_intensity",
            "max_intensity",
            "total_area",
            "percent_maximal",
            "lower_quartile_intensity",
            "upper_quartile_intensity",
        )
    }


def test_relationship_helpers_return_columnar_rows() -> None:
    rows = _DeclaredRelationshipRows(request=object())
    parent_spec = SimpleNamespace(name="Parents")
    child_spec = SimpleNamespace(name="Children")

    child_counts = rows.child_count_rows_for_ids(
        parent_spec=parent_spec,
        child_spec=child_spec,
        related_parent_ids=(7, 7, 9),
        parent_slice_index=3,
    )
    parent_rows = rows.parent_rows_for_pairs(
        parent_spec=parent_spec,
        child_spec=child_spec,
        pairs=((7, 9),),
        parent_slice_index=3,
        child_slice_index=4,
    )
    distance_rows = object.__new__(
        RelateObjectsRelationshipMeasurementRows
    ).parent_mean_distance_rows(
        parent_object_name="Parents",
        child_object_name="Children",
        centroid_child_feature_name="Distance_Centroid_Parents",
        minimum_child_feature_name="Distance_Minimum_Parents",
        pairs=((7, 1), (7, 2)),
        centroid_distances=np.asarray([2.0, 4.0]),
        minimum_distances=np.asarray([1.0, 3.0]),
        slice_index=3,
    )

    assert isinstance(child_counts, ColumnarRows)
    assert isinstance(parent_rows, ColumnarRows)
    assert isinstance(distance_rows, ColumnarRows)
    assert child_counts.row_count() == 2
    assert parent_rows.row_count() == 2
    assert distance_rows.row_count() == 1
    assert {
        row[MeasurementRowAxisField.SLICE_INDEX.value]
        for row in child_counts.iter_row_mappings()
    } == {3}
    assert {
        row[MeasurementRowAxisField.SLICE_INDEX.value]
        for row in parent_rows.iter_row_mappings()
    } == {4}


def test_worm_measurement_producers_are_columnar() -> None:
    descriptor_rows = _worm_descriptor_rows([], num_control_points=3)
    _image, dead_rows, _labels = identify_dead_worms(
        np.zeros((5, 5), dtype=np.float32),
        angle_count=1,
    )
    _image, untangle_rows, _nonoverlapping = untangle_worms(
        np.zeros((5, 5), dtype=np.float32),
        num_control_points=3,
    )

    assert isinstance(descriptor_rows, ColumnarRows)
    assert isinstance(dead_rows, ColumnarRows)
    assert isinstance(untangle_rows, ColumnarRows)
    assert descriptor_rows.row_count() == 0
    assert dead_rows.row_count() == 1
    assert untangle_rows.row_count() == 0


def test_deleted_projector_hierarchy_and_field_mixins_have_no_owned_definitions() -> (
    None
):
    repository_root = Path(__file__).resolve().parents[2]
    runtime_files = (
        "openhcs/interop/cellprofiler/runtime/measurement_rows.py",
        "openhcs/interop/cellprofiler/runtime/relationship_measurement_rows.py",
    )
    backend_files = (
        "openhcs/processing/backends/cellprofiler/alignment.py",
        "openhcs/processing/backends/cellprofiler/area_occupied.py",
        "openhcs/processing/backends/cellprofiler/classification.py",
        "openhcs/processing/backends/cellprofiler/image_quality.py",
        "openhcs/processing/backends/cellprofiler/intensity.py",
        "openhcs/processing/backends/cellprofiler/neighbors.py",
        "openhcs/processing/backends/cellprofiler/relationships.py",
        "openhcs/processing/backends/cellprofiler/secondary.py",
        "openhcs/processing/backends/cellprofiler/thresholding.py",
    )
    forbidden_classes = {
        "CellProfilerMeasurementRowProjection",
        "CellProfilerMeasurementRows",
        "GenericRelationshipMeasurementRows",
        "RelationshipMeasurementRow",
    }
    forbidden_functions = {"_measurement_rows_from_output"}
    forbidden_bases = {
        "NoFieldsMeasurementRecordMixin",
        "ColumnarFieldsMeasurementRecordMixin",
        "FieldsFromRowsMeasurementRecordMixin",
    }

    for relative_path in (*runtime_files, *backend_files):
        tree = ast.parse((repository_root / relative_path).read_text())
        class_definitions = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        function_definitions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        base_names = {
            base.id
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for base in node.bases
            if isinstance(base, ast.Name)
        }
        assert class_definitions.isdisjoint(forbidden_classes)
        assert function_definitions.isdisjoint(forbidden_functions)
        assert base_names.isdisjoint(forbidden_bases)

    for relative_path in backend_files:
        tree = ast.parse((repository_root / relative_path).read_text())
        assert not any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "registry_key"
                for target in node.targets
            )
            for node in ast.walk(tree)
        )
