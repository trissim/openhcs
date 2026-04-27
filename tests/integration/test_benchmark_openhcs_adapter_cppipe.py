from __future__ import annotations

from pathlib import Path

from benchmark.adapters.openhcs import OpenHCSAdapter
from benchmark.metrics.time import TimeMetric
from openhcs.tests.generators.generate_synthetic_data import (
    SyntheticMicroscopyGenerator,
)


def test_openhcs_adapter_runs_converted_cppipe_pipeline(tmp_path: Path) -> None:
    plate_path = _generate_plate(tmp_path / "plate")
    cppipe_path = _write_cppipe(tmp_path / "identify_primary_objects.cppipe")

    result = OpenHCSAdapter().run(
        dataset_path=plate_path,
        pipeline_name="converted_cppipe_smoke",
        pipeline_params={
            "dataset_id": "synthetic_cppipe_smoke",
            "microscope_type": "imagexpress",
            "cppipe_path": str(cppipe_path),
        },
        metrics=[TimeMetric()],
        output_dir=tmp_path / "benchmark_outputs",
    )

    assert result.success is True
    assert result.metrics["execution_time_seconds"] >= 0.0
    assert result.provenance["pipeline_source"] == "converted_cppipe"
    assert result.provenance["axis_count"] == 1


def _generate_plate(plate_path: Path) -> Path:
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_path),
        grid_size=(1, 1),
        tile_size=(128, 128),
        wavelengths=1,
        z_stack_levels=1,
        num_cells=12,
        cell_size_range=(8, 12),
        cell_intensity_range=(28000, 42000),
        background_intensity=200,
        noise_level=10,
        wells=["A01"],
        format="ImageXpress",
        random_seed=7,
    )
    generator.generate_dataset()
    return plate_path


def _write_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:3",
                "HasImagePlaneDetails:False",
                (
                    "LoadData:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Input data file location:Elsewhere...",
                (
                    "IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:OrigBlue",
                "    Name the primary objects to be identified:Nuclei",
                (
                    "ExportToSpreadsheet:[module_num:3|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select measurements to export:No",
                "",
            )
        )
    )
    return cppipe_path
