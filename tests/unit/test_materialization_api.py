from openhcs.processing.materialization import MaterializationSpec, csv_materializer


def test_csv_materializer_uses_analysis_type_suffix() -> None:
    spec = csv_materializer(
        fields=["slice_index", "value"],
        analysis_type="texture",
    )

    assert isinstance(spec, MaterializationSpec)
    assert spec.outputs[0].filename_suffix == "_texture.csv"
