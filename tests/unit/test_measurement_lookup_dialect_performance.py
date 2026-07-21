from __future__ import annotations

from collections import Counter
import math

from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureQuery,
    MeasurementFeatureValueIndex,
)
from openhcs.core.measurement_lookup_dialect import RuntimeMeasurementLookupDialect
from openhcs.core.measurement_row_materialization import MeasurementSparseColumnarRows
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementSubject,
    MeasurementTable,
)
from openhcs.core.runtime_tabular_values import FieldSpec


def _lookup_dialect() -> RuntimeMeasurementLookupDialect:
    return RuntimeMeasurementLookupDialect(
        category_prefixes=(("intensity",), ("area", "shape")),
        alternative_feature_part_aliases={
            ("area",): (("volume",),),
        },
        source_qualified_feature_families=(("mean", "intensity"),),
    )


def test_source_family_scan_decomposes_immutable_lookup_once() -> None:
    provider_calls: Counter[str] = Counter()

    def category_prefixes() -> tuple[tuple[str, ...], ...]:
        provider_calls["category_prefixes"] += 1
        return (("intensity",),)

    def feature_part_aliases() -> dict[tuple[str, ...], tuple[str, ...]]:
        provider_calls["feature_part_aliases"] += 1
        return {}

    def source_families() -> tuple[tuple[str, ...], ...]:
        provider_calls["source_families"] += 1
        return (
            *((f"unrelated_{index}",) for index in range(64)),
            ("mean", "intensity"),
        )

    dialect = RuntimeMeasurementLookupDialect(
        category_prefixes_provider=category_prefixes,
        feature_part_aliases_provider=feature_part_aliases,
        source_qualified_feature_families_provider=source_families,
    )

    families = dialect.feature_lookup(
        "Intensity_MeanIntensity_DNA"
    ).source_qualified_feature_families

    assert families == (("mean", "intensity"),)
    assert provider_calls == {
        "category_prefixes": 1,
        "feature_part_aliases": 1,
        "source_families": 1,
    }


def test_lookup_preserves_source_qualification_aliases_and_object_domain() -> None:
    dialect = _lookup_dialect()
    source_lookup = dialect.feature_lookup("Intensity_MeanIntensity_DNA")
    alias_lookup = dialect.feature_lookup("AreaShape_Area")

    assert source_lookup.dialect_feature_parts == ("mean", "intensity", "dna")
    assert source_lookup.source_qualified_feature_families == (("mean", "intensity"),)
    assert source_lookup.source_aliases == ("dna",)
    assert source_lookup.source_qualified_field_names == (
        "mean_intensity",
        "meanintensity",
    )
    assert alias_lookup.field_aliases == (
        "area_shape_area",
        "areashapearea",
        "area",
        "volume",
    )
    assert source_lookup.query_object_name("Nuclei") == "Nuclei"


def test_columnar_lookup_distinguishes_source_object_absence_padding_and_nan() -> None:
    feature_field = "mean_intensity"
    table = MeasurementTable(
        name="Measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "source_image_name": "DNA",
                    feature_field: 1.5,
                },
                {
                    "object_name": "Nuclei",
                    "object_label": 2,
                    "source_image_name": "DNA",
                    feature_field: None,
                },
                {
                    "object_name": "Nuclei",
                    "object_label": 3,
                    "source_image_name": "DNA",
                },
                {
                    "object_name": "Nuclei",
                    "object_label": 4,
                    "source_image_name": "DNA",
                    feature_field: float("nan"),
                },
                {
                    "object_name": "Cells",
                    "object_label": 1,
                    "source_image_name": "DNA",
                    feature_field: 8.0,
                },
                {
                    "object_name": "Nuclei",
                    "object_label": 1,
                    "source_image_name": "RNA",
                    feature_field: 9.0,
                },
            ),
            fields=(
                FieldSpec("object_name", str),
                FieldSpec("object_label", int),
                FieldSpec("source_image_name", str),
                FieldSpec(feature_field, float, required=False),
            ),
        ),
        subject=MeasurementSubject(MeasurementScope.ARTIFACT, "Measurements"),
    )
    query = MeasurementFeatureQuery(
        "Intensity_MeanIntensity_DNA",
        object_name="Nuclei",
        dialect=_lookup_dialect(),
    )

    default_indexes = MeasurementFeatureValueIndex.from_columnar_table_by_object(
        table,
        query,
        {"Nuclei": "Nuclei"},
    )
    explicit_nan_indexes = MeasurementFeatureValueIndex.from_columnar_table_by_object(
        table,
        query,
        {"Nuclei": "Nuclei"},
        measurement_value_qualifier=(
            lambda value: isinstance(value, float) and math.isnan(value)
        ),
    )

    assert default_indexes is not None
    assert default_indexes["Nuclei"].values_by_label == {1: 1.5}
    assert explicit_nan_indexes is not None
    explicit_nan_values = explicit_nan_indexes["Nuclei"].values_by_label
    assert set(explicit_nan_values) == {4}
    assert math.isnan(explicit_nan_values[4])
