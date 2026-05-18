from __future__ import annotations

import pandas as pd

from openhcs.formats.experimental_layout_rows import (
    ExperimentalAnalysisFeatureReaders,
    ExperimentalAnalysisPlateHandlers,
    ExperimentalAnalysisScope,
    ExperimentalLayoutRowRole,
)


def test_experimental_layout_row_role_classifies_replicate_count_rows() -> None:
    assert ExperimentalLayoutRowRole("N").is_replicate_count
    assert ExperimentalLayoutRowRole("replicates").is_replicate_count
    assert not ExperimentalLayoutRowRole("condition").is_replicate_count


def test_experimental_layout_row_role_classifies_well_rows() -> None:
    assert ExperimentalLayoutRowRole("well").is_well_all_replicates
    assert ExperimentalLayoutRowRole("wells").is_well_all_replicates
    assert not ExperimentalLayoutRowRole("well1").is_well_all_replicates
    assert ExperimentalLayoutRowRole("well1").is_well_specific_replicate
    assert not ExperimentalLayoutRowRole("well").is_well_specific_replicate


def test_experimental_analysis_scope_coerces_public_scope_values() -> None:
    assert ExperimentalAnalysisScope.coerce("EDDU_CX5") is ExperimentalAnalysisScope.CX5
    assert (
        ExperimentalAnalysisScope.coerce("EDDU_metaxpress")
        is ExperimentalAnalysisScope.METAXPRESS
    )


def test_experimental_analysis_scope_uses_scope_owned_sheet_metadata() -> None:
    class Workbook:
        sheet_names = ["first sheet"]

    assert ExperimentalAnalysisScope.CX5.sheet_name_for(Workbook()) == "Rawdata"
    assert (
        ExperimentalAnalysisScope.METAXPRESS.sheet_name_for(Workbook())
        == "first sheet"
    )


def test_experimental_analysis_scope_selects_feature_reader_without_case_dispatch() -> None:
    raw_df = pd.DataFrame({"value": [1]})
    readers = ExperimentalAnalysisFeatureReaders(
        cx5=lambda frame: ("cx5", len(frame)),
        metaxpress=lambda frame: ("metaxpress", len(frame)),
    )

    assert ExperimentalAnalysisScope.CX5.features(raw_df, readers) == ("cx5", 1)
    assert ExperimentalAnalysisScope.METAXPRESS.features(raw_df, readers) == (
        "metaxpress",
        1,
    )


def test_experimental_analysis_scope_selects_plate_handlers_without_case_dispatch() -> None:
    raw_df = pd.DataFrame({"value": [1]})
    handlers = ExperimentalAnalysisPlateHandlers(
        cx5_builder=lambda frame: ("cx5_builder", len(frame)),
        metaxpress_builder=lambda frame: ("metaxpress_builder", len(frame)),
        cx5_filler=lambda frame, plates, features: (
            "cx5_filler",
            len(frame),
            plates,
            features,
        ),
        metaxpress_filler=lambda frame, plates, features: (
            "metaxpress_filler",
            len(frame),
            plates,
            features,
        ),
    )

    assert ExperimentalAnalysisScope.CX5.create_plates_dict(raw_df, handlers) == (
        "cx5_builder",
        1,
    )
    assert ExperimentalAnalysisScope.METAXPRESS.fill_plates_dict(
        raw_df,
        {"plate": {}},
        ["feature"],
        handlers,
    ) == ("metaxpress_filler", 1, {"plate": {}}, ["feature"])
