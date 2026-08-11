"""Tests for the declaration-configured experimental-analysis workflow."""

import inspect
from pathlib import Path

import pandas as pd

from openhcs.core.config import ExperimentalAnalysisConfig, NormalizationMethod
from openhcs.processing.backends.experimental_analysis import (
    ExperimentalAnalysisEngine,
)


def test_directory_workflow_projects_declaration_owned_paths(tmp_path: Path) -> None:
    config = ExperimentalAnalysisConfig(
        config_file_name="design.xlsx",
        results_file_name="measurements.csv",
        compiled_results_file_name="analysis.xlsx",
        raw_results_file_name="analysis-raw.xlsx",
        heatmap_file_name="maps.xlsx",
    )
    engine = ExperimentalAnalysisEngine(config)
    received: dict[str, object] = {}

    def capture(**kwargs):
        received.update(kwargs)
        return {"complete": True}

    engine.run_analysis = capture

    assert engine.run_directory(tmp_path) == {"complete": True}
    assert received == {
        "results_path": str(tmp_path / "measurements.csv"),
        "config_file": str(tmp_path / "design.xlsx"),
        "compiled_results_path": str(tmp_path / "analysis.xlsx"),
        "heatmap_path": str(tmp_path / "maps.xlsx"),
        "raw_results_path": str(tmp_path / "analysis-raw.xlsx"),
    }


def test_directory_workflow_omits_disabled_heatmap_output(tmp_path: Path) -> None:
    engine = ExperimentalAnalysisEngine(
        ExperimentalAnalysisConfig(export_heatmaps=False)
    )
    received: dict[str, object] = {}

    def capture(**kwargs):
        received.update(kwargs)
        return {}

    engine.run_analysis = capture

    engine.run_directory(tmp_path)

    assert received["heatmap_path"] is None
    assert received["raw_results_path"] == str(
        tmp_path / ExperimentalAnalysisConfig().raw_results_file_name
    )


def test_desktop_workflow_does_not_copy_config_filenames_or_use_legacy_runner() -> None:
    from openhcs.pyqt_gui.main import OpenHCSMainWindow

    source = inspect.getsource(OpenHCSMainWindow._on_run_experimental_analysis)
    defaults = ExperimentalAnalysisConfig()

    assert defaults.config_file_name not in source
    assert defaults.results_file_name not in source
    assert defaults.compiled_results_file_name not in source
    assert defaults.raw_results_file_name not in source
    assert defaults.heatmap_file_name not in source
    assert "formats.experimental_analysis import" not in source


def test_engine_projects_declaration_owned_sheet_names(monkeypatch) -> None:
    from openhcs.formats import experimental_analysis

    config = ExperimentalAnalysisConfig(
        design_sheet_name="design",
        plate_groups_sheet_name="groups",
    )
    engine = ExperimentalAnalysisEngine(config)
    received: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        experimental_analysis,
        "read_plate_layout",
        lambda path, *, sheet_name: received.append(("design", path, sheet_name)),
    )
    monkeypatch.setattr(
        experimental_analysis,
        "load_plate_groups",
        lambda path, *, sheet_name: received.append(("groups", path, sheet_name)),
    )

    engine._read_plate_layout("layout.xlsx")
    engine._load_plate_groups("layout.xlsx")

    assert received == [
        ("design", "layout.xlsx", "design"),
        ("groups", "layout.xlsx", "groups"),
    ]


def test_engine_delegates_to_declared_normalization_method(monkeypatch) -> None:
    from openhcs.formats import experimental_analysis

    engine = ExperimentalAnalysisEngine(
        ExperimentalAnalysisConfig(normalization_method=NormalizationMethod.Z_SCORE)
    )
    received: dict[str, object] = {}

    def capture(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return {"normalized": True}

    monkeypatch.setattr(experimental_analysis, "normalize_experiment", capture)

    assert engine._normalize_experiment({}, {}, [], {}, {}) == {"normalized": True}
    assert received["kwargs"] == {"method": NormalizationMethod.Z_SCORE}


def test_engine_applies_exclusions_to_locations_and_controls() -> None:
    engine = ExperimentalAnalysisEngine(ExperimentalAnalysisConfig())
    locations = {"Drug": {"N1": {"dose": [("A01", 1), ("A02", 1)]}}}
    controls = {"N1": [("A01", 1), ("A02", 1)]}

    engine._apply_exclusions(locations, controls, {"N1": [("A01", 1)]})

    assert locations == {"Drug": {"N1": {"dose": [("A02", 1)]}}}
    assert controls == {"N1": [("A02", 1)]}


def test_legacy_experimental_format_mirrors_are_removed() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    legacy_paths = (
        "openhcs/formats/metaxpress.py",
        "openhcs/processing/backends/analysis/cx5_format.py",
        "openhcs/processing/backends/experimental_analysis/format_registry.py",
        "openhcs/processing/backends/experimental_analysis/format_registry_service.py",
        "openhcs/processing/backends/experimental_analysis/cx5_registry.py",
        "openhcs/processing/backends/experimental_analysis/metaxpress_registry.py",
    )

    assert all(not (repository_root / path).exists() for path in legacy_paths)


def test_directory_workflow_runs_declared_metaxpress_analysis_end_to_end(
    tmp_path: Path,
) -> None:
    layout = pd.DataFrame(
        [
            ["N", 1, None],
            ["Scope", "EDDU_metaxpress", None],
            ["Controls", "A01", "A02"],
            ["Plate Group", 1, 1],
            ["Group N", 1, 1],
            ["Condition", "Drug", None],
            ["Dose", 1, None],
            ["Wells1", "A03", None],
            ["Plate Group", 1, None],
        ]
    )
    plate_groups = pd.DataFrame([[None, 1], ["N1", "plate-a"]])
    with pd.ExcelWriter(tmp_path / "config.xlsx") as writer:
        layout.to_excel(
            writer,
            sheet_name="drug_curve_map",
            index=False,
            header=False,
        )
        plate_groups.to_excel(
            writer,
            sheet_name="plate_groups",
            index=False,
            header=False,
        )
    pd.DataFrame(
        [
            ["Barcode", "barcode", None],
            ["Plate ID", "plate-a", None],
            ["Well", "Area", None],
            ["A01", 2, None],
            ["A02", 4, None],
            ["A03", 6, None],
        ]
    ).to_csv(
        tmp_path / "metaxpress_style_summary.csv",
        index=False,
        header=False,
    )

    result = ExperimentalAnalysisEngine(ExperimentalAnalysisConfig()).run_directory(
        tmp_path
    )

    assert result["format_name"] == "EDDU_metaxpress"
    assert result["feature_tables"]["Area"].iloc[0, 0] == 2.0
    assert {
        "compiled_results_normalized.xlsx",
        "compiled_results_raw.xlsx",
        "heatmaps.xlsx",
    }.issubset(path.name for path in tmp_path.iterdir())
