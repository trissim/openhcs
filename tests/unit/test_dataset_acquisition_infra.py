from pathlib import Path
import zipfile

from benchmark.contracts.dataset import (
    AcquiredDataset,
    ArchiveFormat,
    BenchmarkCategory,
    CellProfilerBenchmarkCaseSpec,
    DatasetSourceKind,
    DatasetSourceSpec,
    DatasetSpec,
    DatasetValidationRule,
)
from benchmark.datasets.acquire import (
    DatasetAcquisitionContext,
    DatasetSourceHandler,
    DatasetValidationContext,
    DatasetValidationStrategy,
    GitSparseSourceHandler,
    acquire_dataset,
    _materialize_nested_archives,
)
from benchmark.datasets.cache import (
    BenchmarkPathRootKind,
    default_benchmark_dataset_cache_root,
    default_cellprofiler_examples_root,
    resolve_benchmark_path_root,
)
from benchmark.datasets.manifest import comparison_manifest_payload
from benchmark.datasets.cppipe_case_catalog import official_cp3_case_category
from benchmark.datasets.registry import (
    CELL_ORIENTATION_WOUND_HEALING,
    CELLPROFILER4_BENCHMARK_SUPPLEMENT,
    CELLPROFILER_TUTORIALS,
    CHROMTRANS_3D_FISH,
)


def test_validation_rules_are_registered_by_enum() -> None:
    assert (
        DatasetValidationStrategy.for_rule(
            DatasetValidationRule.NON_EMPTY
        ).validation_rule
        is DatasetValidationRule.NON_EMPTY
    )


def test_benchmark_dataset_paths_default_to_persistent_user_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert default_benchmark_dataset_cache_root() == (
        tmp_path / ".cache" / "openhcs" / "benchmark_datasets"
    )
    assert default_cellprofiler_examples_root() == (
        tmp_path / ".cache" / "openhcs" / "cellprofiler_examples"
    )


def test_benchmark_path_root_env_override_is_explicit(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom-cache"
    monkeypatch.setenv("OPENHCS_BENCHMARK_DATASET_CACHE_ROOT", str(override))

    assert resolve_benchmark_path_root(
        BenchmarkPathRootKind.DATASET_CACHE,
        env_name="OPENHCS_BENCHMARK_DATASET_CACHE_ROOT",
    ) == override


def test_source_handlers_are_registered_by_enum() -> None:
    source = DatasetSourceSpec(kind=DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES)

    assert (
        DatasetSourceHandler.for_source(source).source_kind
        is DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES
    )


def test_git_sparse_source_fetches_immutable_revision_after_clone(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "dataset" / "data"
    source = DatasetSourceSpec(
        kind=DatasetSourceKind.GIT_SPARSE,
        git_url="https://example.invalid/dataset.git",
        git_ref="a" * 40,
        sparse_paths=("examples",),
    )
    context = DatasetAcquisitionContext(
        spec=DatasetSpec(
            id="immutable_revision",
            urls=[],
            size_bytes=1,
            archive_format=ArchiveFormat.ZIP,
            microscope_type="example",
            validation_rule=DatasetValidationRule.NON_EMPTY,
            source=source,
        ),
        cache_root=data_dir.parent,
        archive_dir=data_dir.parent / "archives",
        data_dir=data_dir,
    )
    git_commands: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_run_git(args: list[str], cwd: Path | None) -> None:
        git_commands.append((tuple(args), cwd))
        if args[0] == "clone":
            (data_dir / ".git").mkdir(parents=True)

    monkeypatch.setattr(
        GitSparseSourceHandler,
        "_run_git",
        staticmethod(fake_run_git),
    )

    assert GitSparseSourceHandler().acquire(context, source) is False
    assert git_commands == [
        (
            (
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                source.git_url,
                str(data_dir),
            ),
            None,
        ),
        (("fetch", "--depth", "1", "origin", source.git_ref), data_dir),
        (("sparse-checkout", "set", "examples"), data_dir),
        (("checkout", "FETCH_HEAD"), data_dir),
    ]


def test_url_file_source_acquires_plain_files(monkeypatch, tmp_path: Path) -> None:
    def fake_download(url: str, destination: Path, *, tls_verify: bool = True) -> None:
        destination.write_bytes(b"image")

    monkeypatch.setattr(
        "benchmark.datasets.acquire.DEFAULT_DATASET_FILE_DOWNLOADER.download",
        fake_download,
    )
    spec = DatasetSpec(
        id="plain_files",
        urls=[],
        size_bytes=5,
        archive_format=ArchiveFormat.ZIP,
        microscope_type="bioformats",
        validation_rule=DatasetValidationRule.IMAGE_COUNT,
        expected_count=1,
        source=DatasetSourceSpec(
            kind=DatasetSourceKind.URL_FILES,
            urls=("https://example.test/data/Well%20A01.DIB",),
        ),
    )

    acquired = acquire_dataset(spec, cache_base=tmp_path)

    assert acquired.image_count == 1
    assert acquired.metadata["source_kind"] == DatasetSourceKind.URL_FILES.value
    assert acquired.metadata["source_urls"] == spec.acquisition_source().urls
    assert (acquired.path / "Well A01.DIB").read_bytes() == b"image"


def test_non_empty_validation_counts_registered_image_extensions(tmp_path: Path) -> None:
    (tmp_path / "image.tif").write_bytes(b"not really a tiff")
    context = DatasetValidationContext(
        spec=DatasetSpec(
            id="example",
            urls=[],
            size_bytes=1,
            archive_format=ArchiveFormat.ZIP,
            microscope_type="example",
            validation_rule=DatasetValidationRule.NON_EMPTY,
        ),
        data_dir=tmp_path,
    )

    assert DatasetValidationStrategy.for_rule(
        DatasetValidationRule.NON_EMPTY
    ).validate(context) == 1


def test_dataset_acquisition_source_normalizes_legacy_urls() -> None:
    spec = DatasetSpec(
        id="legacy",
        urls=["https://example.test/data.zip"],
        size_bytes=1,
        archive_format=ArchiveFormat.ZIP,
        microscope_type="example",
        validation_rule=DatasetValidationRule.NON_EMPTY,
    )

    assert spec.acquisition_source() == DatasetSourceSpec(
        kind=DatasetSourceKind.ARCHIVE_URLS,
        urls=("https://example.test/data.zip",),
    )


def test_nested_dataset_archives_materialize_missing_payloads_without_clobbering(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True)
    existing = image_dir / "existing.png"
    existing.write_bytes(b"checkout")
    archive_path = data_dir / "Archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("images/existing.png", b"archive")
        archive.writestr("images/missing.tiff", b"payload")

    _materialize_nested_archives(data_dir)

    assert existing.read_bytes() == b"checkout"
    assert (image_dir / "missing.tiff").read_bytes() == b"payload"


def test_comparison_manifest_materializes_dataset_relative_cases(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True)
    cppipe_path = data_dir / "pipeline.cppipe"
    cppipe_path.write_text("CellProfiler Pipeline", encoding="utf-8")
    spec = DatasetSpec(
        id="case_dataset",
        urls=[],
        size_bytes=1,
        archive_format=ArchiveFormat.ZIP,
        microscope_type="example_scope",
        validation_rule=DatasetValidationRule.NON_EMPTY,
        benchmark_cases=(
            CellProfilerBenchmarkCaseSpec(
                name="case",
                cppipe_path=Path("pipeline.cppipe"),
                dataset_path=Path("images"),
                category=BenchmarkCategory(
                    assay="Example assay",
                    module="Example module",
                ),
            ),
        ),
    )
    acquired = AcquiredDataset(
        id=spec.id,
        path=data_dir,
        microscope_type=spec.microscope_type,
        image_count=0,
        metadata={},
    )

    payload = comparison_manifest_payload([(spec, acquired)])

    assert payload["cases"] == [
        {
            "name": "case",
            "dataset_path": str(image_dir),
            "cppipe_path": str(cppipe_path),
            "dataset_id": "case_dataset",
            "microscope_type": "example_scope",
            "assay_category": "Example assay",
            "module_category": "Example module",
            "value_only": False,
        }
    ]


def test_public_git_dataset_specs_expose_benchmark_cases() -> None:
    assert CELLPROFILER_TUTORIALS.acquisition_source().kind is DatasetSourceKind.GIT_SPARSE
    assert len(CELLPROFILER_TUTORIALS.benchmark_cases) >= 7
    assert (
        CELLPROFILER4_BENCHMARK_SUPPLEMENT.acquisition_source().kind
        is DatasetSourceKind.GIT_SPARSE
    )
    assert len(CELLPROFILER4_BENCHMARK_SUPPLEMENT.benchmark_cases) == 1


def test_public_git_archive_dataset_specs_expose_sources() -> None:
    for spec in (CELL_ORIENTATION_WOUND_HEALING, CHROMTRANS_3D_FISH):
        source = spec.acquisition_source()

        assert source.kind is DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES
        assert source.git_url
        assert source.urls
        assert "workflow" in source.sparse_paths
        assert source.tls_verify is False


def test_official_cp3_case_categories_are_declaration_backed() -> None:
    category = official_cp3_case_category("ExampleFly")

    assert category.assay == "Tissue/object morphology"
    assert category.module == "Segmentation + object measurement"
    assert official_cp3_case_category("UnknownPipeline").assay == "Uncategorized assay"
