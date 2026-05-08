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
    DatasetSourceHandler,
    DatasetValidationContext,
    DatasetValidationStrategy,
    _materialize_nested_archives,
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
    assert DatasetValidationStrategy.for_rule(
        DatasetValidationRule.NON_EMPTY
    ).validation_rule == DatasetValidationRule.NON_EMPTY.value


def test_source_handlers_are_registered_by_enum() -> None:
    source = DatasetSourceSpec(kind=DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES)

    assert (
        DatasetSourceHandler.for_source(source).source_kind
        == DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES.value
    )


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
