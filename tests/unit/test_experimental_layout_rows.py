from __future__ import annotations

import sys
from types import ModuleType

import pandas as pd

sys.modules.setdefault("xlsxwriter", ModuleType("xlsxwriter"))

from openhcs.formats.experimental_analysis import (
    average_wells,
    individual_wells,
    normalize_experiment,
)
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


def test_experimental_analysis_well_value_projection_skips_missing_values() -> None:
    plates = {
        "plate_a": {
            "A01": {"feature": 2.0},
            "A02": {"feature": "bad"},
            "A03": {"feature": 4.0},
        }
    }
    plate_groups = {"N1": {"1": "plate_a", "2": "missing"}}
    locations = [("A01", 1), ("A02", 1), ("A03", 1), ("A04", 1), ("A01", 2)]

    assert average_wells(locations, "N1", "feature", plates, plate_groups) == {
        "averaged": 3.0
    }
    assert individual_wells(locations, "N1", "feature", plates, plate_groups) == {
        "A01_P1": 2.0,
        "A03_P1": 4.0,
    }


def test_normalize_experiment_handles_control_and_treatment_modes() -> None:
    experiment = {
        "DMSO_Control": {
            "N1": {"dose": {"feature": {"averaged": 2.0}}},
            "N2": {"dose": {"feature": {"averaged": 4.0}}},
        },
        "Drug": {
            "N1": {"dose": {"feature": {"well_a": 6.0, "well_b": None}}},
            "N2": {"dose": {"feature": {"well_c": 12.0}}},
        },
    }
    plates = {
        "plate_a": {
            "A01": {"feature": 2.0},
            "A02": {"feature": 4.0},
        },
        "plate_b": {
            "A01": {"feature": 3.0},
            "A02": {"feature": 6.0},
        },
    }
    plate_groups = {"N1": {"1": "plate_a"}, "N2": {"1": "plate_b"}}
    ctrl_positions = {
        "N1": [("A01", 1), ("A02", 1)],
        "N2": [("A01", 1), ("A02", 1)],
    }

    normalized = normalize_experiment(
        experiment,
        ctrl_positions,
        ["feature"],
        plates,
        plate_groups,
    )

    assert normalized["DMSO_Control"]["N1"]["dose"]["feature"] == {
        "averaged": 2.0 / 3.0
    }
    assert normalized["DMSO_Control"]["N2"]["dose"]["feature"] == {
        "averaged": 4.0 / 3.0
    }
    assert normalized["Drug"]["N1"]["dose"]["feature"] == {
        "well_a": 2.0,
        "well_b": None,
    }
    assert normalized["Drug"]["N2"]["dose"]["feature"] == {
        "well_c": 12.0 / 4.5
    }
