import pytest

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.special_outputs import (
    SpecialOutputKindClassifier,
    special_output_materialization,
    special_output_name,
)
from openhcs.processing.materialization import (
    CsvOptions,
    MaterializationSpec,
    ROIOptions,
    TiffStackOptions,
)


def test_special_output_kind_classifier_uses_materialization_options() -> None:
    assert (
        SpecialOutputKindClassifier.kind_for(
            ("measurements", MaterializationSpec(CsvOptions()))
        )
        is ArtifactKind.MEASUREMENTS
    )
    assert (
        SpecialOutputKindClassifier.kind_for(("objects", MaterializationSpec(ROIOptions())))
        is ArtifactKind.OBJECT_LABELS
    )
    assert (
        SpecialOutputKindClassifier.kind_for(("image", MaterializationSpec(TiffStackOptions())))
        is ArtifactKind.IMAGE
    )


def test_spatial_grid_name_owns_csv_materialized_grid_outputs() -> None:
    assert (
        SpecialOutputKindClassifier.kind_for(
            ("grid_info", MaterializationSpec(CsvOptions()))
        )
        is ArtifactKind.SPATIAL_GRID
    )


def test_special_output_kind_classifier_preserves_legacy_name_semantics() -> None:
    assert SpecialOutputKindClassifier.kind_for("grid_definition") is ArtifactKind.SPATIAL_GRID
    assert SpecialOutputKindClassifier.kind_for("parent_relationship_rows") is ArtifactKind.RELATIONSHIPS
    assert SpecialOutputKindClassifier.kind_for("object_labels") is ArtifactKind.OBJECT_LABELS
    assert SpecialOutputKindClassifier.kind_for("quality_rows") is ArtifactKind.MEASUREMENTS


def test_special_output_accessors_fail_loud_for_invalid_declarations() -> None:
    declaration = ("measurements", MaterializationSpec(CsvOptions()))

    assert special_output_name(declaration) == "measurements"
    assert isinstance(special_output_materialization(declaration), MaterializationSpec)

    with pytest.raises(ValueError, match="Invalid special output declaration"):
        special_output_name(("missing-name", "not-materialization", "extra"))

    with pytest.raises(TypeError, match="special_outputs materialization"):
        special_output_materialization(("measurements", object()))
