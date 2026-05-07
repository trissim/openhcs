"""Registry of benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark.contracts.dataset import (
    ArchiveFormat,
    BenchmarkCategory,
    CellProfilerBenchmarkCaseSpec,
    DatasetSourceKind,
    DatasetSourceSpec,
    DatasetSpec,
    DatasetValidationRule,
)

PUBLISHED_PIPELINE = "published_pipeline"
CELLPROFILER_TUTORIALS_REPO = "https://github.com/CellProfiler/tutorials.git"
CP4_BENCHMARK_SUPPLEMENT_REPO = (
    "https://github.com/carpenterlab/2021_Stirling_BMCBioInformatics.git"
)


@dataclass(frozen=True)
class DatasetCatalogRow:
    """Authoritative row for a benchmark dataset."""

    id: str
    microscope_type: str
    size_bytes: int
    validation_rule: DatasetValidationRule = DatasetValidationRule.NON_EMPTY
    urls: tuple[str, ...] = ()
    archive_format: ArchiveFormat = ArchiveFormat.ZIP
    expected_count: int | None = None
    manifest_path: Path | None = None
    reference_cppipe_urls: tuple[str, ...] = ()
    source: DatasetSourceSpec | None = None
    benchmark_cases: tuple[CellProfilerBenchmarkCaseSpec, ...] = ()

    def materialize(self) -> DatasetSpec:
        """Build the immutable DatasetSpec for this catalog row."""
        return DatasetSpec(
            id=self.id,
            urls=list(self.urls),
            size_bytes=self.size_bytes,
            archive_format=self.archive_format,
            microscope_type=self.microscope_type,
            validation_rule=self.validation_rule,
            reference_cppipe_urls=self.reference_cppipe_urls,
            expected_count=self.expected_count,
            manifest_path=self.manifest_path,
            source=self.source,
            benchmark_cases=self.benchmark_cases,
        )


def _case(
    name: str,
    cppipe_path: str,
    dataset_path: str,
    *,
    assay_category: str,
    module_category: str,
    dataset_id: str | None = None,
    value_only: bool = True,
    timeout_seconds: float | None = 900.0,
) -> CellProfilerBenchmarkCaseSpec:
    """Declare one dataset-relative CellProfiler benchmark case."""
    return CellProfilerBenchmarkCaseSpec(
        name=name,
        cppipe_path=Path(cppipe_path),
        dataset_path=Path(dataset_path),
        dataset_id=dataset_id,
        category=BenchmarkCategory(assay=assay_category, module=module_category),
        value_only=value_only,
        cellprofiler_timeout_seconds=timeout_seconds,
    )


def _git_sparse(
    git_url: str,
    *sparse_paths: str,
    git_ref: str = "HEAD",
) -> DatasetSourceSpec:
    """Declare a sparse git acquisition source."""
    return DatasetSourceSpec(
        kind=DatasetSourceKind.GIT_SPARSE,
        git_url=git_url,
        git_ref=git_ref,
        sparse_paths=tuple(sparse_paths),
    )


DATASET_CATALOG: tuple[DatasetCatalogRow, ...] = (
    DatasetCatalogRow(
        id="BBBC021_Week1_22123",
        urls=("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip",),
        size_bytes=839_000_000,
        microscope_type="bbbc021",
        validation_rule=DatasetValidationRule.IMAGE_COUNT,
        reference_cppipe_urls=(
            "https://data.broadinstitute.org/bbbc/BBBC021/analysis.cppipe",
            "https://data.broadinstitute.org/bbbc/BBBC021/illum.cppipe",
        ),
        expected_count=720,
    ),
    DatasetCatalogRow(
        id="BBBC022_20585_w1",
        urls=("http://www.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_20585w1.zip",),
        size_bytes=7_800_000_000,
        microscope_type="bbbc022",
        validation_rule=DatasetValidationRule.IMAGE_COUNT,
        expected_count=3_456,
    ),
    DatasetCatalogRow(
        id="BBBC010_worms",
        urls=(
            "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip",
            "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground.zip",
            "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip",
        ),
        size_bytes=72_222_003,
        microscope_type="bbbc010",
    ),
    DatasetCatalogRow(
        id="BBBC011_worms_metabolism",
        urls=("https://data.broadinstitute.org/bbbc/BBBC011/BBBC011_v1_images.zip",),
        size_bytes=39_876_190,
        microscope_type="bbbc011",
    ),
    DatasetCatalogRow(
        id="BBBC012_worms_infection_marker",
        urls=("https://data.broadinstitute.org/bbbc/BBBC012/BBBC012_v1_images.zip",),
        size_bytes=122_677_100,
        microscope_type="bbbc012",
    ),
    DatasetCatalogRow(
        id="BBBC013_u2os_translocation_bmp",
        urls=(
            "https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_v1_images_bmp.zip",
            "https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_reproduce_logan.zip",
        ),
        size_bytes=37_962_288,
        microscope_type="bbbc013",
        reference_cppipe_urls=(
            "https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_reproduce_logan.zip",
        ),
    ),
    DatasetCatalogRow(
        id="BBBC038_full",
        urls=(
            "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
            "https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip",
            "https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip",
        ),
        size_bytes=382_000_000,
        microscope_type="bbbc038",
        validation_rule=DatasetValidationRule.IMAGE_COUNT,
        expected_count=33_215,
    ),
    DatasetCatalogRow(
        id="BBBC039_nuclei_segmentation",
        urls=(
            "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
            "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
            "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
        ),
        size_bytes=80_687_375,
        microscope_type="bbbc039",
        validation_rule=DatasetValidationRule.IMAGE_COUNT,
        expected_count=800,
    ),
    DatasetCatalogRow(
        id="Singh_2014_illumination_correction",
        urls=(
            "https://cellprofiler-published-pipelines.s3.amazonaws.com/"
            "JMicroscopy_Singh_2014.zip",
        ),
        size_bytes=30_619_586,
        microscope_type=PUBLISHED_PIPELINE,
    ),
    DatasetCatalogRow(
        id="Sanz_2019_histology",
        urls=("https://cellprofiler-published-pipelines.s3.amazonaws.com/Sanz_JAP_2019.zip",),
        size_bytes=4_541_253,
        microscope_type=PUBLISHED_PIPELINE,
    ),
    DatasetCatalogRow(
        id="Tian_2019_neurons",
        urls=("https://cellprofiler-published-pipelines.s3.amazonaws.com/Tian_Neuron_2019.zip",),
        size_bytes=52_207,
        microscope_type=PUBLISHED_PIPELINE,
    ),
    DatasetCatalogRow(
        id="Sokolov_2023_neurons",
        urls=(
            "https://cellprofiler-published-pipelines.s3.amazonaws.com/"
            "AM+Sokolov+Cell+Morphology+pipeline.zip",
        ),
        size_bytes=3_403,
        microscope_type=PUBLISHED_PIPELINE,
    ),
    DatasetCatalogRow(
        id="CellProfiler_tutorials",
        size_bytes=650_000_000,
        microscope_type="cellprofiler_tutorials",
        source=_git_sparse(
            CELLPROFILER_TUTORIALS_REPO,
            "3DNoiseNuclei",
            "3d_monolayer",
            "AdvancedSegmentation",
            "BeginnerSegmentation",
            "PixelBasedClassification",
            "QualityControl",
            "Translocation",
        ),
        benchmark_cases=(
            _case(
                "cp_tutorial_3d_noise_nuclei",
                "3DNoiseNuclei/3DNucleiPipelineComputeConsumingFinal.cppipe",
                "3DNoiseNuclei/Input3DNuclei",
                assay_category="3D nuclei segmentation",
                module_category="3D segmentation",
            ),
            _case(
                "cp_tutorial_3d_monolayer",
                "3d_monolayer/3d_monolayer_final.cppipe",
                "3d_monolayer/images",
                assay_category="3D monolayer morphology",
                module_category="3D segmentation + measurement",
            ),
            _case(
                "cp_tutorial_advanced_segmentation_final",
                "AdvancedSegmentation/BBBC022_Analysis_Final.cppipe",
                "AdvancedSegmentation/BBBC022_20585_AE",
                assay_category="Cell Painting morphology",
                module_category="Advanced segmentation + measurement",
            ),
            _case(
                "cp_tutorial_quality_control",
                "QualityControl/BBBC022_QC.cppipe",
                "QualityControl/BBBC022_20585_AE",
                assay_category="Cell Painting quality control",
                module_category="Image quality measurement",
            ),
            _case(
                "cp_tutorial_beginner_segmentation_final",
                "BeginnerSegmentation/segmentation_final.cppipe",
                "BeginnerSegmentation/images_Illum-corrected",
                assay_category="Cell morphology",
                module_category="Segmentation + intensity measurement",
            ),
            _case(
                "cp_tutorial_pixel_based_classification",
                "PixelBasedClassification/pixel_based_classification_cho.cppipe",
                "PixelBasedClassification/images",
                assay_category="Pixel classification",
                module_category="Pixel classification",
            ),
            _case(
                "cp_tutorial_translocation_final",
                "Translocation/Translocation_final.cppipe",
                "Translocation/TranslocationData",
                assay_category="Translocation assay",
                module_category="Segmentation + classification",
            ),
        ),
    ),
    DatasetCatalogRow(
        id="CellProfiler4_benchmark_supplement",
        size_bytes=5_000_000,
        microscope_type="cellprofiler4_benchmark",
        source=_git_sparse(
            CP4_BENCHMARK_SUPPLEMENT_REPO,
            "CombineObjects",
        ),
        benchmark_cases=(
            _case(
                "cp4_supplement_combine_objects",
                "CombineObjects/CombineObjectsDemo.cppipe",
                "CombineObjects",
                assay_category="Object-combination benchmark",
                module_category="Object set algebra",
            ),
        ),
    ),
)

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    spec.id: spec for spec in (row.materialize() for row in DATASET_CATALOG)
}

BBBC021_SINGLE_PLATE = DATASET_REGISTRY["BBBC021_Week1_22123"]
BBBC022_SINGLE_PLATE_DNA = DATASET_REGISTRY["BBBC022_20585_w1"]
BBBC010_WORMS = DATASET_REGISTRY["BBBC010_worms"]
BBBC011_WORMS_METABOLISM = DATASET_REGISTRY["BBBC011_worms_metabolism"]
BBBC012_WORMS_INFECTION_MARKER = DATASET_REGISTRY["BBBC012_worms_infection_marker"]
BBBC013_U2OS_TRANSLOCATION = DATASET_REGISTRY["BBBC013_u2os_translocation_bmp"]
BBBC038_FULL = DATASET_REGISTRY["BBBC038_full"]
BBBC039_NUCLEI_SEGMENTATION = DATASET_REGISTRY["BBBC039_nuclei_segmentation"]
SINGH_2014_ILLUMINATION_CORRECTION = DATASET_REGISTRY[
    "Singh_2014_illumination_correction"
]
SANZ_2019_HISTOLOGY = DATASET_REGISTRY["Sanz_2019_histology"]
TIAN_2019_NEURONS = DATASET_REGISTRY["Tian_2019_neurons"]
SOKOLOV_2023_NEURONS = DATASET_REGISTRY["Sokolov_2023_neurons"]
CELLPROFILER_TUTORIALS = DATASET_REGISTRY["CellProfiler_tutorials"]
CELLPROFILER4_BENCHMARK_SUPPLEMENT = DATASET_REGISTRY[
    "CellProfiler4_benchmark_supplement"
]


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    """
    Retrieve a dataset specification by id.

    Raises:
        KeyError: if dataset id is unknown.
    """
    try:
        return DATASET_REGISTRY[dataset_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset id '{dataset_id}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        ) from exc
