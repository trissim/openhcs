from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from benchmark.contracts.dataset import AcquiredDataset
from benchmark.contracts.tool_adapter import BenchmarkResult, ToolAdapter
from benchmark.datasets.registry import BBBC021_SINGLE_PLATE
from benchmark.pipelines.registry import NUCLEI_SEGMENTATION
from benchmark.adapters.openhcs import (
    _candidate_image_snapshots_for_equivalence,
    _candidate_measurement_snapshot_cache_key,
    _reference_snapshot_for_equivalence_fallback,
    _reference_measurement_snapshot_cache_key,
    _runtime_execution_cache_key_for_snapshot,
    _runtime_execution_cache_key_matches,
)
from benchmark.adapters.cellprofiler import NATIVE_CELLPROFILER_SUCCESS_MARKER
from benchmark.cellprofiler_comparison import (
    CellProfilerComparisonCase,
    _native_reference_location,
    _benchmark_path_slug,
)
from benchmark.datasets.visible_source import resolve_visible_source_path
from benchmark.runner import (
    _LEGACY_SOURCE_TREE_CACHE_KEY,
    _source_file_cache_domains,
    _source_file_has_excluded_cache_domain,
    _source_file_is_path_excluded,
    run_cellprofiler_compatibility_benchmark,
    run_cellprofiler_cppipe_parity,
)
from openhcs.core.runtime_equivalence import (
    RuntimeEquivalencePolicy,
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSubjectKey,
    runtime_measurement_projection_cache_identity,
)
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    RuntimeExportObservation,
    RuntimeImageExportSpec,
)
from openhcs.core.runtime_semantics import MeasurementScope


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
    cppipe_path.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
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
    cppipe_path.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    case = CellProfilerComparisonCase(
        name="Example",
        dataset_path=dataset_path,
        cppipe_path=cppipe_path,
        dataset_id="HiddenDataset",
    )
    visible_dataset_path = resolve_visible_source_path(dataset_path)
    native_reference_root = tmp_path / "native_refs"
    reference_dir = (
        native_reference_root
        / _benchmark_path_slug(f"{case.resolved_dataset_id}_{case.name}")
        / f"{visible_dataset_path.name}_{case.name}_native_cellprofiler"
    )
    reference_dir.mkdir(parents=True)
    (reference_dir / NATIVE_CELLPROFILER_SUCCESS_MARKER).write_text("{}")

    location = _native_reference_location(case, native_reference_root)

    assert location.reference_output_dir == reference_dir


def test_cellprofiler_cppipe_parity_runner_reuses_cached_openhcs_output(
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
    assert second_openhcs_adapter.run_count == 0
    assert second_openhcs_adapter.validated is False
    assert (
        second_result.openhcs_converted.provenance or {}
    )["reused_cached_output"] is True


def test_cellprofiler_cppipe_parity_runner_invalidates_openhcs_cache_on_cppipe_change(
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
    run_cellprofiler_cppipe_parity(
        dataset_path,
        cppipe_path,
        metrics=[],
        dataset_id="examplefly_official",
        output_root=output_root,
        equivalence_reference_output_dir=reference_output,
        openhcs_adapter=first_openhcs_adapter,
    )
    cppipe_path.write_text(
        "CellProfiler Pipeline: http://www.cellprofiler.org\n"
        "ModuleCount: 1\n"
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

    assert second_result.is_equivalent
    assert first_openhcs_adapter.run_count == 1
    assert second_openhcs_adapter.run_count == 1
    assert not (second_result.openhcs_converted.provenance or {}).get(
        "reused_cached_output"
    )


def test_cellprofiler_cppipe_parity_runner_keeps_execution_cache_key_for_reference_change(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "Example Fly Images"
    dataset_path.mkdir()
    cppipe_path = tmp_path / "Example Fly.cppipe"
    cppipe_path.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    first_reference_output = tmp_path / "native_reference_1"
    second_reference_output = tmp_path / "native_reference_2"
    first_reference_output.mkdir()
    second_reference_output.mkdir()
    (first_reference_output / "Image.csv").write_text("ImageNumber\n1\n")
    (second_reference_output / "Image.csv").write_text("ImageNumber\n2\n")
    output_root = tmp_path / "outputs"

    first_openhcs_adapter = _OpenHCSParityAdapter()
    run_cellprofiler_cppipe_parity(
        dataset_path,
        cppipe_path,
        metrics=[],
        dataset_id="examplefly_official",
        output_root=output_root,
        equivalence_reference_output_dir=first_reference_output,
        openhcs_adapter=first_openhcs_adapter,
    )

    second_openhcs_adapter = _OpenHCSParityAdapter()
    run_cellprofiler_cppipe_parity(
        dataset_path,
        cppipe_path,
        metrics=[],
        dataset_id="examplefly_official",
        output_root=output_root,
        equivalence_reference_output_dir=second_reference_output,
        openhcs_adapter=second_openhcs_adapter,
    )

    assert first_openhcs_adapter.run_count == 1
    assert second_openhcs_adapter.run_count == 1
    assert (
        first_openhcs_adapter.pipeline_params["runtime_execution_cache_key"]
        == second_openhcs_adapter.pipeline_params["runtime_execution_cache_key"]
    )


def test_openhcs_execution_cache_rejects_stale_legacy_source_tree() -> None:
    cached_key = _execution_cache_key(source_field="source_tree", digest="old")
    expected_key = _execution_cache_key(
        source_field="execution_source_tree",
        digest="new",
        legacy_digest="current",
    )

    assert not _runtime_execution_cache_key_matches(cached_key, expected_key)


def test_openhcs_execution_cache_accepts_matching_legacy_source_tree() -> None:
    cached_key = _execution_cache_key(source_field="source_tree", digest="current")
    expected_key = _execution_cache_key(
        source_field="execution_source_tree",
        digest="new",
        legacy_digest="current",
    )

    assert _runtime_execution_cache_key_matches(cached_key, expected_key)


def test_openhcs_execution_cache_ignores_helper_source_tree_for_current_keys() -> None:
    cached_key = _execution_cache_key(
        source_field="execution_source_tree",
        digest="execution",
        legacy_digest="old-full",
    )
    expected_key = _execution_cache_key(
        source_field="execution_source_tree",
        digest="execution",
        legacy_digest="new-full",
    )

    assert _runtime_execution_cache_key_matches(cached_key, expected_key)


def test_measurement_snapshot_key_omits_cache_helper_source_tree() -> None:
    cache_key = _execution_cache_key(
        source_field="execution_source_tree",
        digest="execution",
        legacy_digest="full-source",
    )

    snapshot_key = _runtime_execution_cache_key_for_snapshot(cache_key)

    assert _LEGACY_SOURCE_TREE_CACHE_KEY not in snapshot_key
    assert snapshot_key["execution_source_tree"] == {"digest": "execution"}


def test_measurement_snapshot_keys_include_semantic_projection_fingerprint(
    tmp_path: Path,
) -> None:
    cache_key = _execution_cache_key(
        source_field="execution_source_tree",
        digest="execution",
        legacy_digest="full-source",
    )
    policy = RuntimeEquivalencePolicy()
    projection_identity = runtime_measurement_projection_cache_identity()
    required_key = RuntimeMeasurementFeatureKey(
        RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, "Cells"),
        "object_number",
    )

    reference_key = _reference_measurement_snapshot_cache_key(
        tmp_path,
        policy=policy,
        known_source_names=(),
    )
    candidate_key = _candidate_measurement_snapshot_cache_key(
        SimpleNamespace(runtime_execution_cache_key=cache_key),
        policy=policy,
        known_source_names=(),
        required_measurement_keys=frozenset({required_key}),
        candidate_observation_fingerprint="observation",
    )

    assert reference_key["semantic_measurement_projection"] == projection_identity
    assert candidate_key["semantic_measurement_projection"] == projection_identity


def test_saveimages_export_specs_use_runtime_artifacts_not_incidental_files(
    tmp_path: Path,
) -> None:
    validation = SimpleNamespace(
        expectation=SimpleNamespace(
            exports=RuntimeExportExpectation.from_flags(
                table_exports=False,
                image_exports=True,
                image_export_specs=(RuntimeImageExportSpec("SelectedImage"),),
            )
        ),
        observation=SimpleNamespace(
            exports=RuntimeExportObservation(
                table_outputs=(),
                image_outputs=(tmp_path / "incidental_final_step.npy",),
                table_headers_by_path={},
                table_row_counts_by_path={},
            )
        ),
    )

    assert _candidate_image_snapshots_for_equivalence(validation) is None


def test_exported_image_files_are_used_without_declared_image_artifacts(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "candidate.npy"
    np.save(image_path, np.ones((2, 3), dtype=np.float32))
    validation = SimpleNamespace(
        expectation=SimpleNamespace(
            exports=RuntimeExportExpectation.from_flags(
                table_exports=False,
                image_exports=True,
            )
        ),
        observation=SimpleNamespace(
            exports=RuntimeExportObservation(
                table_outputs=(),
                image_outputs=(image_path,),
                table_headers_by_path={},
                table_row_counts_by_path={},
            )
        ),
    )

    snapshots = _candidate_image_snapshots_for_equivalence(validation)

    assert snapshots is not None
    assert len(snapshots) == 1
    assert snapshots[0].shape == (2, 3)


def test_value_only_fallback_strips_reference_images(tmp_path: Path) -> None:
    (tmp_path / "Image.csv").write_text("ImageNumber\n1\n")
    np.save(tmp_path / "SavedImage.npy", np.ones((2, 3), dtype=np.float32))

    snapshot = _reference_snapshot_for_equivalence_fallback(
        tmp_path,
        compare_image_outputs=False,
    )

    assert len(snapshot.tables) == 1
    assert snapshot.images == ()


def test_image_fallback_keeps_reference_images(tmp_path: Path) -> None:
    np.save(tmp_path / "SavedImage.npy", np.ones((2, 3), dtype=np.float32))

    snapshot = _reference_snapshot_for_equivalence_fallback(
        tmp_path,
        compare_image_outputs=True,
    )

    assert len(snapshot.images) == 1


def test_source_cache_domain_parser_handles_bom_python(tmp_path: Path) -> None:
    marked_file = tmp_path / "marked.py"
    marked_file.write_bytes(
        b"\xef\xbb\xbf"
        b"BENCHMARK_CACHE_DOMAINS = frozenset({'parity', 'harness'})\n"
    )
    stat = marked_file.stat()

    assert _source_file_cache_domains(
        str(marked_file),
        stat.st_size,
        stat.st_mtime_ns,
    ) == frozenset({"parity", "harness"})
    assert _source_file_has_excluded_cache_domain(
        marked_file,
        excluded_cache_domains=frozenset({"parity"}),
    )


def test_source_cache_domain_parser_includes_unparseable_files(tmp_path: Path) -> None:
    broken_file = tmp_path / "broken.py"
    broken_file.write_text("BENCHMARK_CACHE_DOMAINS = frozenset({'parity'})\nif")
    stat = broken_file.stat()

    assert _source_file_cache_domains(
        str(broken_file),
        stat.st_size,
        stat.st_mtime_ns,
    ) == frozenset()
    assert not _source_file_has_excluded_cache_domain(
        broken_file,
        excluded_cache_domains=frozenset({"parity"}),
    )


def test_source_cache_excludes_local_cellprofiler_source_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert _source_file_is_path_excluded(
        repo_root / "benchmark/cellprofiler_source/modules/identifyprimaryobjects.py",
        repo_root=repo_root,
    )
    assert not _source_file_is_path_excluded(
        repo_root / "benchmark/cellprofiler_library/functions/identifyprimaryobjects.py",
        repo_root=repo_root,
    )


def _execution_cache_key(
    *,
    source_field: str,
    digest: str,
    legacy_digest: str | None = None,
) -> dict[str, Any]:
    key = {
        "schema_version": 1,
        "tool_name": "OpenHCS",
        "tool_version": "test",
        "pipeline_name": "pipeline",
        "pipeline_params": {"dataset_id": "dataset"},
        "dataset_tree": {"digest": "dataset"},
        "cppipe_file": {"digest": "cppipe"},
        source_field: {"digest": digest},
    }
    if legacy_digest is not None:
        key[_LEGACY_SOURCE_TREE_CACHE_KEY] = {"digest": legacy_digest}
    return key


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
