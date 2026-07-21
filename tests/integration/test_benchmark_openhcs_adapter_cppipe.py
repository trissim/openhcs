from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from benchmark.adapters.openhcs import OpenHCSAdapter
from benchmark.contracts.dataset import AcquiredDataset
from benchmark.contracts.tool_adapter import (
    BenchmarkResult,
    ToolAdapter,
    ToolExecutionError,
)
from benchmark.datasets.registry import BBBC021_SINGLE_PLATE
from benchmark.metrics.time import TimeMetric
from benchmark.pipelines.registry import NUCLEI_SEGMENTATION
from benchmark.runner import run_benchmark
from openhcs.tests.generators.generate_synthetic_data import (
    SyntheticMicroscopyGenerator,
)


def test_openhcs_adapter_runs_converted_cppipe_pipeline(tmp_path: Path) -> None:
    plate_path = _generate_plate(tmp_path / "plate")
    cppipe_path = _write_cppipe(tmp_path / "identify_primary_objects.cppipe")

    result = _run_openhcs_adapter(
        OpenHCSAdapterRunCase.local_cppipe(
            plate_path,
            "converted_cppipe_smoke",
            "synthetic_cppipe_smoke",
            cppipe_path,
            tmp_path / "benchmark_outputs",
        )
    )

    assert result.success is True
    assert result.metrics["execution_time_seconds"] >= 0.0
    assert result.provenance["pipeline_source"] == "converted_cppipe"
    assert result.provenance["axis_count"] == 1

    parity_result = _run_openhcs_adapter(
        OpenHCSAdapterRunCase.local_cppipe(
            plate_path,
            "converted_cppipe_parity",
            "synthetic_cppipe_smoke",
            cppipe_path,
            tmp_path / "benchmark_outputs",
            equivalence_reference_output_dir=result.output_path,
        )
    )

    assert parity_result.success is True
    assert parity_result.provenance["equivalence_reference_output_dir"] == str(
        result.output_path
    )
    assert parity_result.provenance["equivalence_difference_count"] == 0


def test_openhcs_adapter_resolves_dataset_reference_cppipe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plate_path = _generate_plate(tmp_path / "plate")
    cppipe_path = _write_cppipe(tmp_path / "identify_primary_objects.cppipe")

    def _materialize_reference(self, reference_url: str, target_dir: Path) -> Path:
        assert reference_url == BBBC021_SINGLE_PLATE.reference_cppipe_urls[0]
        assert target_dir == (tmp_path / "benchmark_outputs" / "cppipe_references")
        return cppipe_path

    monkeypatch.setattr(
        OpenHCSAdapter,
        "_materialize_cppipe_reference",
        _materialize_reference,
    )

    result = _run_openhcs_adapter(
        OpenHCSAdapterRunCase(
            dataset_path=plate_path,
            pipeline_name="converted_cppipe_reference",
            dataset_id=BBBC021_SINGLE_PLATE.id,
            cppipe_reference_index=0,
            output_dir=tmp_path / "benchmark_outputs",
        )
    )

    assert result.success is True
    assert result.provenance["pipeline_source"] == "converted_cppipe"
    assert result.provenance["cppipe_path"] == str(cppipe_path)
    assert (
        result.provenance["cppipe_reference_url"]
        == BBBC021_SINGLE_PLATE.reference_cppipe_urls[0]
    )


def test_openhcs_adapter_rejects_reference_output_mismatch(tmp_path: Path) -> None:
    plate_path = _generate_plate(tmp_path / "plate")
    cppipe_path = _write_cppipe(tmp_path / "identify_primary_objects.cppipe")
    reference_output = tmp_path / "native_reference"
    reference_output.mkdir()
    (reference_output / "wrong.csv").write_text(
        "not_a_generated_schema\n1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ToolExecutionError,
        match="Converted CellProfiler output did not match semantic reference output",
    ):
        _run_openhcs_adapter(
            OpenHCSAdapterRunCase.local_cppipe(
                plate_path,
                "converted_cppipe_mismatch",
                "synthetic_cppipe_smoke",
                cppipe_path,
                tmp_path / "benchmark_outputs",
                equivalence_reference_output_dir=reference_output,
            )
        )


def test_default_benchmark_pipeline_uses_dataset_cppipe_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = _CapturingAdapter()
    acquired = AcquiredDataset(
        id=BBBC021_SINGLE_PLATE.id,
        path=tmp_path / "plate",
        microscope_type=BBBC021_SINGLE_PLATE.microscope_type,
        image_count=0,
        metadata={},
    )
    acquired.path.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("benchmark.runner.acquire_dataset", lambda spec: acquired)

    run_benchmark(
        BBBC021_SINGLE_PLATE,
        [adapter],
        NUCLEI_SEGMENTATION.name,
        metrics=[],
    )

    assert adapter.pipeline_params["cppipe_reference_index"] == 0
    assert adapter.pipeline_params["dataset_id"] == BBBC021_SINGLE_PLATE.id
    assert (
        adapter.pipeline_params["microscope_type"]
        == BBBC021_SINGLE_PLATE.microscope_type
    )
    assert "threshold_method" not in adapter.pipeline_params


def test_openhcs_adapter_requires_converted_cppipe_source(
    tmp_path: Path,
) -> None:
    plate_path = _generate_plate(tmp_path / "plate")

    with pytest.raises(
        ToolExecutionError,
        match=(
            "CellProfiler pipeline execution requires cppipe_path, cppipe_file, "
            "cppipe_reference_url, or cppipe_reference_index\\."
        ),
    ):
        _run_openhcs_adapter(
            OpenHCSAdapterRunCase(
                dataset_path=plate_path,
                pipeline_name="no_cppipe",
                dataset_id="synthetic_without_cppipe",
                output_dir=tmp_path / "benchmark_outputs",
            )
        )


def test_openhcs_adapter_rejects_legacy_examplefly_load_data_cppipe(
    tmp_path: Path,
) -> None:
    plate_path = _generate_two_channel_plate(tmp_path / "examplefly_plate")
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / "ExampleFly.cppipe"
    )

    with pytest.raises(
        KeyError,
        match="No CellProfiler module declaration.*LoadData",
    ):
        _run_openhcs_adapter(
            OpenHCSAdapterRunCase.local_cppipe(
                plate_path,
                "examplefly",
                "examplefly_cppipe",
                cppipe_path,
                tmp_path / "benchmark_outputs",
            )
        )


def test_openhcs_adapter_reports_missing_source_schema_images(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    plate_path.mkdir()
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / "ExampleHuman.cppipe"
    )

    with pytest.raises(
        ToolExecutionError,
        match=(
            "Failed to materialize CellProfiler source schema for "
            "ExampleHuman\\.cppipe: Source schema image alias 'DNA' matched "
            "no image files\\."
        ),
    ):
        _run_openhcs_adapter(
            OpenHCSAdapterRunCase.local_cppipe(
                plate_path,
                "examplehuman",
                "examplehuman_cppipe",
                cppipe_path,
                tmp_path / "benchmark_outputs",
            )
        )


@dataclass(frozen=True, slots=True)
class OpenHCSAdapterRunCase:
    dataset_path: Path
    pipeline_name: str
    dataset_id: str
    output_dir: Path
    microscope_type: str = "imagexpress"
    cppipe_path: Path | None = None
    cppipe_reference_index: int | None = None
    equivalence_reference_output_dir: Path | None = None

    @classmethod
    def local_cppipe(
        cls,
        dataset_path: Path,
        pipeline_name: str,
        dataset_id: str,
        cppipe_path: Path,
        output_dir: Path,
        equivalence_reference_output_dir: Path | None = None,
    ) -> OpenHCSAdapterRunCase:
        return cls(
            dataset_path=dataset_path,
            pipeline_name=pipeline_name,
            dataset_id=dataset_id,
            cppipe_path=cppipe_path,
            output_dir=output_dir,
            equivalence_reference_output_dir=equivalence_reference_output_dir,
        )

    @property
    def pipeline_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "microscope_type": self.microscope_type,
        }
        if self.cppipe_path is not None:
            params["cppipe_path"] = str(self.cppipe_path)
        if self.cppipe_reference_index is not None:
            params["cppipe_reference_index"] = self.cppipe_reference_index
        if self.equivalence_reference_output_dir is not None:
            params["equivalence_reference_output_dir"] = str(
                self.equivalence_reference_output_dir
            )
        return params


def _run_openhcs_adapter(run_case: OpenHCSAdapterRunCase) -> BenchmarkResult:
    return OpenHCSAdapter().run(
        dataset_path=run_case.dataset_path,
        pipeline_name=run_case.pipeline_name,
        pipeline_params=run_case.pipeline_params,
        metrics=[TimeMetric()],
        output_dir=run_case.output_dir,
    )


class _CapturingAdapter(ToolAdapter):
    name = "capture"
    version = "test"

    def __init__(self) -> None:
        self.pipeline_params: dict[str, Any] = {}

    def validate_installation(self) -> None:
        return None

    def run(
        self,
        dataset_path: Path,
        pipeline_name: str,
        pipeline_params: dict[str, Any],
        metrics: list[Any],
        output_dir: Path,
    ) -> BenchmarkResult:
        self.pipeline_params = dict(pipeline_params)
        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=str(pipeline_params["dataset_id"]),
            pipeline_name=pipeline_name,
            metrics={},
            output_path=output_dir,
            success=True,
        )


def _generate_plate(plate_path: Path) -> Path:
    return _generate_imagexpress_plate(
        plate_path,
        wavelengths=1,
        random_seed=7,
    )


def _generate_two_channel_plate(plate_path: Path) -> Path:
    return _generate_imagexpress_plate(
        plate_path,
        wavelengths=2,
        random_seed=11,
    )


def _generate_imagexpress_plate(
    plate_path: Path,
    *,
    wavelengths: int,
    random_seed: int,
) -> Path:
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_path),
        grid_size=(1, 1),
        tile_size=(128, 128),
        wavelengths=wavelengths,
        z_stack_levels=1,
        num_cells=12,
        cell_size_range=(8, 12),
        cell_intensity_range=(28000, 42000),
        background_intensity=200,
        noise_level=10,
        wells=["A01"],
        format="ImageXpress",
        random_seed=random_seed,
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
