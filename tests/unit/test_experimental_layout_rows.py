from __future__ import annotations

import pandas as pd
import pytest

from openhcs.core.config import NormalizationMethod
from openhcs.formats.experimental_analysis import (
    average_wells,
    individual_wells,
    normalize_experiment,
    PlateLayoutBuilder,
    project_plates_without_excluded_positions,
    write_values_heat_map,
)
from openhcs.formats.experimental_layout_rows import (
    ExperimentalAnalysisScope,
    ExperimentalLayoutRowRole,
)
from openhcs.formats.experimental_result_formats import (
    CX5ExperimentalResultFormat,
    ExperimentalResultFormatStrategy,
    MetaXpressExperimentalResultFormat,
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
    assert ExperimentalLayoutRowRole("Wells 12").specific_replicate == 12
    assert ExperimentalLayoutRowRole("wells_12").specific_replicate == 12


def test_experimental_analysis_scope_coerces_public_scope_values() -> None:
    assert ExperimentalAnalysisScope.coerce("EDDU_CX5") is ExperimentalAnalysisScope.CX5
    assert (
        ExperimentalAnalysisScope.coerce("EDDU_metaxpress")
        is ExperimentalAnalysisScope.METAXPRESS
    )


def test_experimental_analysis_scope_selects_nominal_result_format() -> None:
    assert isinstance(
        ExperimentalResultFormatStrategy.for_enum_member(ExperimentalAnalysisScope.CX5),
        CX5ExperimentalResultFormat,
    )
    assert isinstance(
        ExperimentalResultFormatStrategy.for_enum_member(
            ExperimentalAnalysisScope.METAXPRESS
        ),
        MetaXpressExperimentalResultFormat,
    )


def test_cx5_result_format_owns_rawdata_sheet(monkeypatch) -> None:
    received: dict[str, object] = {}

    def read_excel(path, *, sheet_name):
        received.update(path=path, sheet_name=sheet_name)
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_excel", read_excel)

    CX5ExperimentalResultFormat().read_results("results.xlsx")

    assert received == {"path": "results.xlsx", "sheet_name": "Rawdata"}


def test_metaxpress_result_strategy_processes_consolidated_csv(tmp_path) -> None:
    results_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            ["Barcode", "barcode", None],
            ["Plate ID", "plate-a", None],
            ["Well", "Area", "Intensity"],
            ["A01", 2.0, 8.0],
        ]
    ).to_csv(results_path, header=False, index=False)

    processed = MetaXpressExperimentalResultFormat().process(results_path)

    assert processed["format_name"] == "EDDU_metaxpress"
    assert processed["features"] == ["Area", "Intensity"]
    assert processed["plates_dict"]["plate-a"]["A01"] == {
        "Area": 2.0,
        "Intensity": 8.0,
    }


def test_cx5_result_strategy_processes_rawdata_workbook(tmp_path) -> None:
    results_path = tmp_path / "results.xlsx"
    pd.DataFrame(
        [
            ["source", "plate-a", 1, 1, 1, 7.0, "tail"],
        ],
        columns=[
            "Source",
            "UniquePlateId",
            "Row",
            "Column",
            "Replicate",
            "Area",
            "Tail",
        ],
    ).to_excel(results_path, sheet_name="Rawdata", index=False)

    processed = CX5ExperimentalResultFormat().process(results_path)

    assert processed["format_name"] == "EDDU_CX5"
    assert processed["features"] == ["Area"]
    assert processed["plates_dict"]["plate-a"]["A01"] == {"Area": 7.0}


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


def test_normalize_experiment_uses_replicate_local_controls() -> None:
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
        "averaged": 4.0 / 4.5
    }
    assert normalized["Drug"]["N1"]["dose"]["feature"] == {
        "well_a": 2.0,
        "well_b": None,
    }
    assert normalized["Drug"]["N2"]["dose"]["feature"] == {"well_c": 12.0 / 4.5}


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (NormalizationMethod.FOLD_CHANGE, 2.0),
        (NormalizationMethod.Z_SCORE, 2.0),
        (NormalizationMethod.PERCENT_CONTROL, 200.0),
    ],
)
def test_normalize_experiment_executes_declared_method(method, expected) -> None:
    normalized = normalize_experiment(
        {"Drug": {"N1": {"dose": {"feature": 8.0}}}},
        {"N1": [("A01", 1), ("A02", 1)]},
        ["feature"],
        {
            "plate": {
                "A01": {"feature": 2.0},
                "A02": {"feature": 6.0},
            }
        },
        {"N1": {"1": "plate"}},
        method=method,
    )

    assert normalized["Drug"]["N1"]["dose"]["feature"] == expected


def test_normalize_experiment_returns_none_for_undefined_reference() -> None:
    normalized = normalize_experiment(
        {"Drug": {"N1": {"dose": {"feature": {"A01": 8.0}}}}},
        {"N1": [("A01", 1)]},
        ["feature"],
        {"plate": {"A01": {"feature": 4.0}}},
        {"N1": {"1": "plate"}},
        method=NormalizationMethod.Z_SCORE,
    )

    assert normalized["Drug"]["N1"]["dose"]["feature"] == {"A01": None}


def test_heatmap_plate_projection_applies_replicate_scoped_exclusions() -> None:
    plates = {
        "101": {"A01": {"feature": 1}, "A02": {"feature": 2}},
        "102": {"A01": {"feature": 3}},
    }

    projected = project_plates_without_excluded_positions(
        plates,
        {"N1": [("A01", 1)], "N2": [("A01", 2)]},
        {"N1": {"1": 101}, "N2": {"2": "102"}},
    )

    assert projected == {"101": {"A02": {"feature": 2}}, "102": {}}
    assert "A01" in plates["101"]


def test_heatmap_workbook_contains_plate_grid_and_color_scale(tmp_path) -> None:
    from openpyxl import load_workbook

    output_path = tmp_path / "heatmaps.xlsx"
    write_values_heat_map(
        {"plate-1": {"A01": {"feature": 1.0}, "H12": {"feature": 9.0}}},
        ["feature"],
        output_path,
    )

    workbook = load_workbook(output_path)
    worksheet = workbook["feature"]
    assert worksheet["A1"].value == "plate-1"
    assert worksheet["A2"].value == 1.0
    assert worksheet["L9"].value == 9.0
    assert len(worksheet.conditional_formatting) == 1


def test_metaxpress_csv_feature_discovery_uses_well_header_row() -> None:
    raw_df = pd.DataFrame(
        [
            ["Barcode", None, None],
            ["Plate ID", "plate_a", None],
            ["Well", "Area", "Intensity"],
            ["A01", 1, 2],
        ]
    )

    strategy = MetaXpressExperimentalResultFormat()
    assert strategy.well_header_row(raw_df) == 2
    assert strategy.features(raw_df) == ["Area", "Intensity"]


def test_metaxpress_excel_feature_discovery_uses_null_feature_row() -> None:
    raw_df = pd.DataFrame(
        [
            ["Plate Name", "plate_a", None, None],
            [None, None, "Area", "Intensity"],
            ["A01", "focus", 1, 2],
        ]
    )

    strategy = MetaXpressExperimentalResultFormat()
    assert strategy.well_header_row(raw_df) is None
    assert strategy.features(raw_df) == ["Area", "Intensity"]


def test_parse_plate_layout_frame_builds_controls_exclusions_and_assignments() -> None:
    layout_frame = pd.DataFrame(
        [
            [2, None, None],
            ["EDDU_CX5", None, None],
            [True, None, None],
            ["A01", "A02", None],
            [1, 1, None],
            [1, 2, None],
            ["B01", None, None],
            [2, None, None],
            [2, None, None],
            ["Drug", None, None],
            [0.1, 1.0, None],
            ["C01", "C02", None],
            [1, 2, None],
            ["D01", "D02", None],
            [2, 2, None],
        ],
        index=[
            "N",
            "scope",
            "per well datapoints",
            "control well",
            "plate group",
            "group n",
            "exclude wells",
            "plate group",
            "group n",
            "condition",
            "dose",
            "well",
            "plate group",
            "well2",
            "plate group",
        ],
    )

    scope, layout, conditions, ctrl_positions, excluded_positions, per_well = (
        PlateLayoutBuilder().parse(layout_frame.dropna(how="all"))
    )

    assert scope == "EDDU_CX5"
    assert conditions == ["Drug"]
    assert per_well is True
    assert ctrl_positions == {"N1": [("A01", 1)], "N2": [("A02", 1)]}
    assert excluded_positions == {"N1": [], "N2": [("B01", 2)]}
    assert layout["N1"]["Drug"] == {
        0.1: [("C01", 1)],
        1.0: [("C02", 2)],
    }
    assert layout["N2"]["Drug"] == {
        0.1: [("C01", 1), ("D01", 2)],
        1.0: [("C02", 2), ("D02", 2)],
    }


def test_plate_layout_supports_multi_digit_replicate_rows() -> None:
    rows = [[12, None], ["Drug", None], [0.1, None]]
    index = ["N", "condition", "dose"]
    for replicate in range(1, 13):
        rows.extend([[f"A{replicate:02d}", None], [1, None]])
        index.extend([f"wells{replicate}", "plate group"])

    _scope, layout, *_rest = PlateLayoutBuilder().parse(
        pd.DataFrame(rows, index=index).dropna(how="all")
    )

    assert layout["N12"]["Drug"] == {0.1: [("A12", 1)]}


def test_plate_layout_rejects_mismatched_assignment_columns() -> None:
    frame = pd.DataFrame(
        [
            [1, None, None],
            ["Drug", None, None],
            [0.1, 1.0, None],
            ["A01", None, None],
            [1, None, None],
        ],
        index=["N", "condition", "dose", "wells1", "plate group"],
    )

    with pytest.raises(ValueError, match="equal column counts"):
        PlateLayoutBuilder().parse(frame.dropna(how="all"))
