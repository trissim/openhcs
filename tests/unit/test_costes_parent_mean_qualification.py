from __future__ import annotations

import math

from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureQuery,
    MeasurementFeatureValueIndex,
)
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
    MeasurementTable,
)
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.processing.backends.cellprofiler.relationships import RelateObjectsModule


def test_relateobjects_child_mean_qualification_retains_explicit_nan_only() -> None:
    feature_name = "Correlation_Costes_Hoechst_Mito"
    table = MeasurementTable(
        name="MeasureColocalization_measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "object_name": "Mitochondria",
                    "object_label": 1,
                    feature_name: 1.0,
                },
                {
                    "object_name": "Mitochondria",
                    "object_label": 2,
                    feature_name: float("nan"),
                },
                {"object_name": "Mitochondria", "object_label": 3},
                {
                    "object_name": "Mitochondria",
                    "object_label": 4,
                    feature_name: None,
                },
            ),
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec(feature_name, float, required=False),
            ),
        ),
        subject=MeasurementSubject(
            MeasurementScope.ARTIFACT,
            "MeasureColocalization_measurements",
        ),
    )
    query = MeasurementFeatureQuery(
        feature_name,
        object_name="Mitochondria",
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )

    default_indexes = MeasurementFeatureValueIndex.from_columnar_table_by_object(
        table,
        query,
        {"Mitochondria": "Mitochondria"},
    )
    qualified_indexes = MeasurementFeatureValueIndex.from_columnar_table_by_object(
        table,
        query,
        {"Mitochondria": "Mitochondria"},
        measurement_value_qualifier=(
            RelateObjectsModule.aggregate_child_measurement_value_is_qualified
        ),
    )

    assert default_indexes is not None
    assert qualified_indexes is not None
    assert default_indexes["Mitochondria"].values_by_label == {1: 1.0}
    qualified_values = qualified_indexes["Mitochondria"].values_by_label
    assert set(qualified_values) == {1, 2}
    assert qualified_values[1] == 1.0
    assert math.isnan(qualified_values[2])
