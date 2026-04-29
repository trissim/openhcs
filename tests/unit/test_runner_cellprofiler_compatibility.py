from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.contracts.dataset import AcquiredDataset
from benchmark.contracts.tool_adapter import BenchmarkResult, ToolAdapter
from benchmark.datasets.registry import BBBC021_SINGLE_PLATE
from benchmark.pipelines.registry import NUCLEI_SEGMENTATION
from benchmark.runner import run_cellprofiler_compatibility_benchmark


def test_cellprofiler_compatibility_runner_feeds_native_output_to_openhcs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    native_adapter = _NativeReferenceAdapter()
    openhcs_adapter = _OpenHCSParityAdapter()
    acquired = AcquiredDataset(
        id=BBBC021_SINGLE_PLATE.id,
        path=tmp_path / "plate",
        microscope_type=BBBC021_SINGLE_PLATE.microscope_type,
        image_count=0,
        metadata={},
    )
    acquired.path.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("benchmark.runner.acquire_dataset", lambda _spec: acquired)

    result = run_cellprofiler_compatibility_benchmark(
        BBBC021_SINGLE_PLATE,
        NUCLEI_SEGMENTATION.name,
        metrics=[],
        cellprofiler_adapter=native_adapter,
        openhcs_adapter=openhcs_adapter,
    )

    assert result.is_equivalent
    assert native_adapter.validated is True
    assert openhcs_adapter.validated is True
    assert native_adapter.pipeline_params["cppipe_reference_index"] == 0
    assert openhcs_adapter.pipeline_params["cppipe_reference_index"] == 0
    assert (
        openhcs_adapter.pipeline_params["equivalence_reference_output_dir"]
        == str(native_adapter.output_path)
    )


class _NativeReferenceAdapter(ToolAdapter):
    name = "CellProfiler"
    version = "test"

    def __init__(self) -> None:
        self.validated = False
        self.pipeline_params: dict[str, Any] = {}
        self.output_path = Path()

    def validate_installation(self) -> None:
        self.validated = True

    def run(
        self,
        dataset_path: Path,
        pipeline_name: str,
        pipeline_params: dict[str, Any],
        metrics: list[Any],
        output_dir: Path,
    ) -> BenchmarkResult:
        self.pipeline_params = dict(pipeline_params)
        self.output_path = output_dir / "native_reference"
        self.output_path.mkdir(parents=True)
        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=str(pipeline_params["dataset_id"]),
            pipeline_name=pipeline_name,
            metrics={},
            output_path=self.output_path,
            success=True,
            provenance={"pipeline_source": "native_cppipe"},
        )


class _OpenHCSParityAdapter(ToolAdapter):
    name = "OpenHCS"
    version = "test"

    def __init__(self) -> None:
        self.validated = False
        self.pipeline_params: dict[str, Any] = {}

    def validate_installation(self) -> None:
        self.validated = True

    def run(
        self,
        dataset_path: Path,
        pipeline_name: str,
        pipeline_params: dict[str, Any],
        metrics: list[Any],
        output_dir: Path,
    ) -> BenchmarkResult:
        self.pipeline_params = dict(pipeline_params)
        output_dir.mkdir(parents=True)
        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=str(pipeline_params["dataset_id"]),
            pipeline_name=pipeline_name,
            metrics={},
            output_path=output_dir,
            success=True,
            provenance={
                "pipeline_source": "converted_cppipe",
                "equivalence_difference_count": 0,
            },
        )
