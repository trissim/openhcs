"""Registry of benchmark datasets."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar
from pathlib import Path

from metaclass_registry import AutoRegisterMeta

from benchmark.contracts.dataset import (
    ArchiveFormat,
    BenchmarkCategory,
    BenchmarkDatasetTag,
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
CELL_ORIENTATION_REPO = "https://github.com/rgomez-AI/CellOrientation.git"
CHROMTRANS_REPO = "https://github.com/rgomez-AI/3DChromTrans.git"


class BenchmarkDatasetDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Registered declaration for one benchmark dataset."""

    __registry__: ClassVar[dict[str, type["BenchmarkDatasetDeclaration"]]] = {}
    __registry_key__ = "id"
    __skip_if_no_key__ = True

    id: ClassVar[str | None] = None
    public_alias: ClassVar[str | None] = None
    urls: ClassVar[tuple[str, ...]] = ()
    size_bytes: ClassVar[int]
    archive_format: ClassVar[ArchiveFormat] = ArchiveFormat.ZIP
    microscope_type: ClassVar[str]
    validation_rule: ClassVar[DatasetValidationRule] = DatasetValidationRule.NON_EMPTY
    reference_cppipe_urls: ClassVar[tuple[str, ...]] = ()
    expected_count: ClassVar[int | None] = None
    manifest_path: ClassVar[Path | None] = None
    source: ClassVar[DatasetSourceSpec | None] = None
    benchmark_cases: ClassVar[tuple[CellProfilerBenchmarkCaseSpec, ...]] = ()
    tags: ClassVar[frozenset[BenchmarkDatasetTag]] = frozenset()

    @classmethod
    def to_spec(cls) -> DatasetSpec:
        """Materialize this declaration as a public dataset spec."""
        if cls.id is None:
            raise ValueError(f"{cls.__name__} must declare a dataset id.")
        return DatasetSpec(
            id=cls.id,
            urls=list(cls.urls),
            size_bytes=cls.size_bytes,
            archive_format=cls.archive_format,
            microscope_type=cls.microscope_type,
            validation_rule=cls.validation_rule,
            reference_cppipe_urls=cls.reference_cppipe_urls,
            expected_count=cls.expected_count,
            manifest_path=cls.manifest_path,
            source=cls.source,
            benchmark_cases=cls.benchmark_cases,
            tags=cls.tags,
        )


class PublishedPipelineDatasetMixin:
    """Dataset declaration mixin for CellProfiler-published-pipeline examples."""

    microscope_type: ClassVar[str] = PUBLISHED_PIPELINE


class ImageCountValidatedDatasetMixin:
    """Dataset declaration mixin for image-count validated datasets."""

    validation_rule: ClassVar[DatasetValidationRule] = DatasetValidationRule.IMAGE_COUNT
    expected_count: ClassVar[int]


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


def _git_sparse_with_archives(
    git_url: str,
    urls: tuple[str, ...],
    *sparse_paths: str,
    git_ref: str = "HEAD",
    tls_verify: bool = True,
) -> DatasetSourceSpec:
    """Declare a sparse git acquisition source with companion data archives."""
    return DatasetSourceSpec(
        kind=DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES,
        urls=urls,
        git_url=git_url,
        git_ref=git_ref,
        sparse_paths=tuple(sparse_paths),
        tls_verify=tls_verify,
    )


class Bbbc021Week122123Dataset(
    ImageCountValidatedDatasetMixin, BenchmarkDatasetDeclaration
):
    """Dataset declaration for BBBC021_Week1_22123."""

    id = "BBBC021_Week1_22123"
    public_alias = "BBBC021_SINGLE_PLATE"
    urls = (
        "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip",
    )
    size_bytes = 839000000
    microscope_type = "bbbc021"
    reference_cppipe_urls = (
        "https://data.broadinstitute.org/bbbc/BBBC021/analysis.cppipe",
        "https://data.broadinstitute.org/bbbc/BBBC021/illum.cppipe",
    )
    expected_count = 720


class Bbbc02220585W1Dataset(
    ImageCountValidatedDatasetMixin, BenchmarkDatasetDeclaration
):
    """Dataset declaration for BBBC022_20585_w1."""

    id = "BBBC022_20585_w1"
    public_alias = "BBBC022_SINGLE_PLATE_DNA"
    urls = ("http://www.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_20585w1.zip",)
    size_bytes = 7800000000
    microscope_type = "bbbc022"
    expected_count = 3456


class Bbbc010WormsDataset(BenchmarkDatasetDeclaration):
    """Dataset declaration for BBBC010_worms."""

    id = "BBBC010_worms"
    public_alias = "BBBC010_WORMS"
    urls = (
        "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip",
        "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground.zip",
        "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip",
    )
    size_bytes = 72222003
    microscope_type = "bbbc010"


class Bbbc011WormsMetabolismDataset(BenchmarkDatasetDeclaration):
    """Dataset declaration for BBBC011_worms_metabolism."""

    id = "BBBC011_worms_metabolism"
    public_alias = "BBBC011_WORMS_METABOLISM"
    urls = ("https://data.broadinstitute.org/bbbc/BBBC011/BBBC011_v1_images.zip",)
    size_bytes = 39876190
    microscope_type = "bbbc011"


class Bbbc012WormsInfectionMarkerDataset(BenchmarkDatasetDeclaration):
    """Dataset declaration for BBBC012_worms_infection_marker."""

    id = "BBBC012_worms_infection_marker"
    public_alias = "BBBC012_WORMS_INFECTION_MARKER"
    urls = ("https://data.broadinstitute.org/bbbc/BBBC012/BBBC012_v1_images.zip",)
    size_bytes = 122677100
    microscope_type = "bbbc012"


class Bbbc013U2osTranslocationDataset(BenchmarkDatasetDeclaration):
    """Dataset declaration for BBBC013_u2os_translocation_bmp."""

    id = "BBBC013_u2os_translocation_bmp"
    public_alias = "BBBC013_U2OS_TRANSLOCATION"
    urls = (
        "https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_v1_images_bmp.zip",
        "https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_reproduce_logan.zip",
    )
    size_bytes = 37962288
    microscope_type = "bbbc013"
    reference_cppipe_urls = (
        "https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_reproduce_logan.zip",
    )


class Bbbc038FullDataset(ImageCountValidatedDatasetMixin, BenchmarkDatasetDeclaration):
    """Dataset declaration for BBBC038_full."""

    id = "BBBC038_full"
    public_alias = "BBBC038_FULL"
    urls = (
        "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
        "https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip",
        "https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip",
    )
    size_bytes = 382000000
    microscope_type = "bbbc038"
    expected_count = 33215


class Bbbc039NucleiSegmentationDataset(
    ImageCountValidatedDatasetMixin, BenchmarkDatasetDeclaration
):
    """Dataset declaration for BBBC039_nuclei_segmentation."""

    id = "BBBC039_nuclei_segmentation"
    public_alias = "BBBC039_NUCLEI_SEGMENTATION"
    urls = (
        "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
        "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
        "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
    )
    size_bytes = 80687375
    microscope_type = "bbbc039"
    expected_count = 800


class Singh2014IlluminationCorrectionDataset(
    PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration
):
    """Dataset declaration for Singh_2014_illumination_correction."""

    id = "Singh_2014_illumination_correction"
    public_alias = "SINGH_2014_ILLUMINATION_CORRECTION"
    urls = (
        "https://cellprofiler-published-pipelines.s3.amazonaws.com/JMicroscopy_Singh_2014.zip",
    )
    size_bytes = 30619586


class Sanz2019HistologyDataset(
    PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration
):
    """Dataset declaration for Sanz_2019_histology."""

    id = "Sanz_2019_histology"
    public_alias = "SANZ_2019_HISTOLOGY"
    urls = (
        "https://cellprofiler-published-pipelines.s3.amazonaws.com/Sanz_JAP_2019.zip",
    )
    size_bytes = 4541253


class Tian2019NeuronsDataset(
    PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration
):
    """Dataset declaration for Tian_2019_neurons."""

    id = "Tian_2019_neurons"
    public_alias = "TIAN_2019_NEURONS"
    urls = (
        "https://cellprofiler-published-pipelines.s3.amazonaws.com/Tian_Neuron_2019.zip",
    )
    size_bytes = 52207


class Sokolov2023NeuronsDataset(
    PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration
):
    """Dataset declaration for Sokolov_2023_neurons."""

    id = "Sokolov_2023_neurons"
    public_alias = "SOKOLOV_2023_NEURONS"
    urls = (
        "https://cellprofiler-published-pipelines.s3.amazonaws.com/AM+Sokolov+Cell+Morphology+pipeline.zip",
    )
    size_bytes = 3403


class CellOrientationWoundHealingDataset(
    PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration
):
    """Dataset declaration for CellOrientation_wound_healing."""

    id = "CellOrientation_wound_healing"
    public_alias = "CELL_ORIENTATION_WOUND_HEALING"
    size_bytes = 201274355
    source = _git_sparse_with_archives(
        CELL_ORIENTATION_REPO,
        ("https://public-docs.crg.es/almu/rgomez/Jennifer_Jungfleisch/Dataset.zip",),
        "workflow",
        tls_verify=False,
    )


class ChromTrans3dFishDataset(
    PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration
):
    """Dataset declaration for ChromTrans_3d_fish."""

    id = "ChromTrans_3d_fish"
    public_alias = "CHROMTRANS_3D_FISH"
    size_bytes = 98822670
    source = _git_sparse_with_archives(
        CHROMTRANS_REPO,
        ("https://public-docs.crg.es/almu/rgomez/Anna_Oncins/Dataset.zip",),
        "workflow",
        tls_verify=False,
    )


class CellProfilerTutorialsDataset(BenchmarkDatasetDeclaration):
    """Dataset declaration for CellProfiler_tutorials."""

    id = "CellProfiler_tutorials"
    public_alias = "CELLPROFILER_TUTORIALS"
    size_bytes = 650000000
    microscope_type = "cellprofiler_tutorials"
    source = _git_sparse(
        CELLPROFILER_TUTORIALS_REPO,
        "3DNoiseNuclei",
        "3d_monolayer",
        "AdvancedSegmentation",
        "BeginnerSegmentation",
        "PixelBasedClassification",
        "QualityControl",
        "Translocation",
    )
    benchmark_cases = (
        _case(
            "cp_tutorial_3d_noise_nuclei",
            "3DNoiseNuclei/3DNucleiPipelineComputeConsumingFinal.cppipe",
            "3DNoiseNuclei/Input3DNuclei",
            assay_category="3D nuclei segmentation",
            module_category="3D segmentation",
            timeout_seconds=None,
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
    )


class CellProfiler4BenchmarkSupplementDataset(BenchmarkDatasetDeclaration):
    """Dataset declaration for CellProfiler4_benchmark_supplement."""

    id = "CellProfiler4_benchmark_supplement"
    public_alias = "CELLPROFILER4_BENCHMARK_SUPPLEMENT"
    size_bytes = 5000000
    microscope_type = "cellprofiler4_benchmark"
    source = _git_sparse(CP4_BENCHMARK_SUPPLEMENT_REPO, "CombineObjects")
    benchmark_cases = (
        _case(
            "cp4_supplement_combine_objects",
            "CombineObjects/CombineObjectsDemo.cppipe",
            "CombineObjects",
            assay_category="Object-combination benchmark",
            module_category="Object set algebra",
        ),
    )


from benchmark.datasets import bioformats_hcs as _bioformats_hcs_declarations  # noqa: E402,F401


def dataset_declarations() -> tuple[type[BenchmarkDatasetDeclaration], ...]:
    """Return registered benchmark dataset declarations."""
    return tuple(BenchmarkDatasetDeclaration.__registry__.values())


def dataset_specs() -> tuple[DatasetSpec, ...]:
    """Return materialized benchmark dataset specs."""
    return tuple(declaration.to_spec() for declaration in dataset_declarations())


DATASET_REGISTRY: dict[str, DatasetSpec] = {spec.id: spec for spec in dataset_specs()}


def _dataset_public_aliases() -> dict[str, DatasetSpec]:
    aliases: dict[str, DatasetSpec] = {}
    for declaration in dataset_declarations():
        dataset_id = declaration.id
        public_alias = declaration.public_alias
        if dataset_id is not None and public_alias is not None:
            aliases[public_alias] = DATASET_REGISTRY[dataset_id]
    return aliases


globals().update(_dataset_public_aliases())


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
