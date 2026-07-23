from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmark.contracts.dataset import AcquiredDataset
from benchmark.contracts.tool_adapter import BenchmarkResult, ToolAdapter
from benchmark.datasets.registry import BBBC021_SINGLE_PLATE
from benchmark.pipelines.registry import NUCLEI_SEGMENTATION
from benchmark.adapters.cellprofiler import (
    CELLPROFILER_FIRST_IMAGE_SET_PARAM,
    CELLPROFILER_LAST_IMAGE_SET_PARAM,
    NativeCellProfilerInputDomainStrategyKey,
    NativeCellProfilerProvenanceField,
    _write_native_reference_success_marker,
    native_cellprofiler_well_filter_scope_slug,
)
from benchmark.cellprofiler_comparison import (
    CellProfilerComparisonCase,
    NativeCellProfilerReferenceScope,
    _native_reference_location,
)
from benchmark.datasets.visible_source import resolve_visible_source_path
from benchmark.runner import (
    run_cellprofiler_compatibility_benchmark,
    run_cellprofiler_cppipe_parity,
)
from openhcs.core.config import GlobalPipelineConfig, WellFilterConfig


SOURCE_ONLY_CPIPE = "\n".join(
    (
        "CellProfiler Pipeline: http://www.cellprofiler.org",
        "Images:[module_num:1|enabled:True]",
        "    Filter images?:Images only",
        "    Select the rule criteria:and (extension does isimage)",
        "NamesAndTypes:[module_num:2|enabled:True]",
        "    Assign a name to:Images matching rules",
        "    Select the image type:Grayscale image",
        "    Name to assign these images:DNA",
        "    Match metadata:[]",
        "    Image set matching method:Order",
        "    Assignments count:1",
        "    Single images count:0",
        "    Maximum intensity:255.0",
        "    Process as 3D?:No",
        "    Relative pixel spacing in X:1.0",
        "    Relative pixel spacing in Y:1.0",
        "    Relative pixel spacing in Z:1.0",
        "    Select the rule criteria:and (file does contain \"\")",
        "    Name to assign these images:DNA",
        "    Select the image type:Grayscale image",
    )
) + "\n"


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


def test_cellprofiler_cppipe_parity_runner_accepts_local_cppipe(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "Example Fly Images"
    dataset_path.mkdir()
    cppipe_path = tmp_path / "Example Fly.cppipe"
    cppipe_path.write_text(SOURCE_ONLY_CPIPE, encoding="utf-8")
    native_adapter = _NativeReferenceAdapter()
    openhcs_adapter = _OpenHCSParityAdapter()

    result = run_cellprofiler_cppipe_parity(
        dataset_path,
        cppipe_path,
        metrics=[],
        dataset_id="examplefly_official",
        microscope_type="imagexpress",
        output_root=tmp_path / "outputs",
        cellprofiler_adapter=native_adapter,
        openhcs_adapter=openhcs_adapter,
    )

    assert result.is_equivalent
    assert native_adapter.pipeline_params["dataset_id"] == "examplefly_official"
    assert native_adapter.pipeline_params["cppipe_path"] == str(cppipe_path)
    assert openhcs_adapter.pipeline_params["cppipe_path"] == str(cppipe_path)
    assert openhcs_adapter.pipeline_params["microscope_type"] == "imagexpress"
    assert (
        openhcs_adapter.pipeline_params["equivalence_reference_output_dir"]
        == str(native_adapter.output_path)
    )
    assert native_adapter.output_path.parent.name == "CellProfiler_examplefly_official_Example_Fly"


def test_native_reference_lookup_uses_visible_source_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    visible_root = tmp_path / "visible_sources"
    monkeypatch.setenv("OPENHCS_BENCHMARK_VISIBLE_SOURCE_ROOT", str(visible_root))
    dataset_path = tmp_path / ".cache" / "datasets" / "images"
    dataset_path.mkdir(parents=True)
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(SOURCE_ONLY_CPIPE, encoding="utf-8")
    case = CellProfilerComparisonCase(
        name="Example",
        dataset_path=dataset_path,
        cppipe_path=cppipe_path,
        dataset_id="HiddenDataset",
    )
    visible_dataset_path = resolve_visible_source_path(dataset_path)
    native_reference_root = tmp_path / "native_refs"
    reference_dir = NativeCellProfilerReferenceScope(
        case=case,
        native_reference_root=native_reference_root,
        pipeline_params=case.pipeline_params,
    ).expected_reference
    reference_dir.mkdir(parents=True)
    _write_native_reference_success_marker(
        reference_dir,
        {
            NativeCellProfilerProvenanceField.INPUT_DOMAIN_STRATEGY: (
                NativeCellProfilerInputDomainStrategyKey.DATASET_FOLDER
            )
        },
    )

    location = _native_reference_location(case, native_reference_root)

    assert location.reference_output_dir == reference_dir


def test_native_reference_lookup_rejects_unproven_semantic_snapshot(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ExampleIlluminationCorrection" / "images"
    dataset_path.mkdir(parents=True)
    cppipe_path = tmp_path / "ExampleIlluminationCorrection" / "pipeline.cppipe"
    cppipe_path.write_text(SOURCE_ONLY_CPIPE, encoding="utf-8")
    case = CellProfilerComparisonCase(
        name="Example1_AllMethod",
        dataset_path=dataset_path,
        cppipe_path=cppipe_path,
        dataset_id="ExampleIlluminationCorrection",
    )
    native_reference_root = tmp_path / "native_refs"
    reference_dir = NativeCellProfilerReferenceScope(
        case=case,
        native_reference_root=native_reference_root,
        pipeline_params=case.pipeline_params,
    ).expected_reference
    reference_dir.mkdir(parents=True)
    np.save(reference_dir / "Illum.npy", np.ones((2, 2), dtype=np.float32))

    location = _native_reference_location(case, native_reference_root)

    assert location.reference_output_dir is None


def test_native_reference_lookup_separates_bounded_image_set_scope(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(SOURCE_ONLY_CPIPE, encoding="utf-8")
    case = CellProfilerComparisonCase(
        name="ExampleBounded",
        dataset_path=tmp_path / "images",
        cppipe_path=cppipe_path,
        dataset_id="example",
        pipeline_params={
            CELLPROFILER_FIRST_IMAGE_SET_PARAM: 1,
            CELLPROFILER_LAST_IMAGE_SET_PARAM: 1,
        },
    )
    native_reference_root = tmp_path / "native_refs"

    location = _native_reference_location(case, native_reference_root)

    assert location.output_dir == (
        native_reference_root
        / "example_ExampleBounded_image_sets_first1_last1"
    )


def test_native_reference_lookup_separates_public_well_filter_scope(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "    Select the rule criteria:and (extension does isimage)",
                "Metadata:[module_num:2|enabled:True]",
                "    Extract metadata?:Yes",
                "    Metadata source:File name",
                "    Regular expression to extract from file name:^(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9]+)_w(?P<Channel>[0-9]+)",
                "NamesAndTypes:[module_num:3|enabled:True]",
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                "    Match metadata:[]",
                "    Image set matching method:Order",
                "    Assignments count:1",
                "    Single images count:0",
                "    Maximum intensity:255.0",
                "    Process as 3D?:No",
                "    Relative pixel spacing in X:1.0",
                "    Relative pixel spacing in Y:1.0",
                "    Relative pixel spacing in Z:1.0",
                "    Select the rule criteria:and (file does contain \"\")",
                "    Name to assign these images:DNA",
                "    Select the image type:Grayscale image",
            ]
        ),
        encoding="utf-8",
    )
    case = CellProfilerComparisonCase(
        name="ExampleOneWell",
        dataset_path=tmp_path / "images",
        cppipe_path=cppipe_path,
        dataset_id="example",
    )
    native_reference_root = tmp_path / "native_refs"
    global_config = GlobalPipelineConfig(
        well_filter_config=WellFilterConfig(well_filter=1),
    )

    location = _native_reference_location(
        case,
        native_reference_root,
        global_config=global_config,
    )

    assert (
        native_cellprofiler_well_filter_scope_slug(global_config.well_filter_config)
        == "wells_include_first1"
    )
    assert location.output_dir == (
        native_reference_root
        / "example_ExampleOneWell_wells_include_first1"
    )


def test_cellprofiler_cppipe_parity_runner_always_executes_current_openhcs(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "Example Fly Images"
    dataset_path.mkdir()
    cppipe_path = tmp_path / "Example Fly.cppipe"
    cppipe_path.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    reference_output = tmp_path / "native_reference"
    reference_output.mkdir()
    (reference_output / "Image.csv").write_text("ImageNumber\n1\n")
    output_root = tmp_path / "outputs"

    first_openhcs_adapter = _OpenHCSParityAdapter()
    first_result = run_cellprofiler_cppipe_parity(
        dataset_path,
        cppipe_path,
        metrics=[],
        dataset_id="examplefly_official",
        output_root=output_root,
        equivalence_reference_output_dir=reference_output,
        openhcs_adapter=first_openhcs_adapter,
    )
    second_openhcs_adapter = _OpenHCSParityAdapter()
    second_result = run_cellprofiler_cppipe_parity(
        dataset_path,
        cppipe_path,
        metrics=[],
        dataset_id="examplefly_official",
        output_root=output_root,
        equivalence_reference_output_dir=reference_output,
        openhcs_adapter=second_openhcs_adapter,
    )

    assert first_result.is_equivalent
    assert second_result.is_equivalent
    assert first_openhcs_adapter.run_count == 1
    assert second_openhcs_adapter.run_count == 1
    assert second_openhcs_adapter.validated is True


class _NativeReferenceAdapter(ToolAdapter):
    name = "CellProfiler"
    version = "test"

    def __init__(self) -> None:
        self.validated = False
        self.pipeline_params: dict[str, Any] = {}
        self.run_count = 0
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
        self.run_count += 1
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
        self.run_count = 0

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
        self.run_count += 1
        self.pipeline_params = dict(pipeline_params)
        output_dir.mkdir(parents=True, exist_ok=True)
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
