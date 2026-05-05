"""Registry of benchmark datasets."""

from benchmark.contracts.dataset import ArchiveFormat, DatasetSpec, DatasetValidationRule

# Core quick-start dataset (single BBBC021 plate)
BBBC021_SINGLE_PLATE = DatasetSpec(
    id="BBBC021_Week1_22123",
    urls=["https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip"],
    size_bytes=839_000_000,  # 839 MB
    archive_format=ArchiveFormat.ZIP,
    microscope_type="bbbc021",
    validation_rule=DatasetValidationRule.IMAGE_COUNT,
    reference_cppipe_urls=(
        "https://data.broadinstitute.org/bbbc/BBBC021/analysis.cppipe",
        "https://data.broadinstitute.org/bbbc/BBBC021/illum.cppipe",
    ),
    expected_count=720,  # ~96 wells × 2.5 FOVs × 3 channels
)

# Quick subset of BBBC022: single plate, DNA channel only (w1)
BBBC022_SINGLE_PLATE_DNA = DatasetSpec(
    id="BBBC022_20585_w1",
    urls=["http://www.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_20585w1.zip"],
    size_bytes=7_800_000_000,  # ~7.8 GB (approx)
    archive_format=ArchiveFormat.ZIP,
    microscope_type="bbbc022",
    validation_rule=DatasetValidationRule.IMAGE_COUNT,
    expected_count=3_456,  # 384 wells × 9 sites × 1 channel
)

BBBC010_WORMS = DatasetSpec(
    id="BBBC010_worms",
    urls=[
        "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip",
        "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground.zip",
        "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip",
    ],
    size_bytes=72_222_003,
    archive_format=ArchiveFormat.ZIP,
    microscope_type="bbbc010",
    validation_rule=DatasetValidationRule.NON_EMPTY,
)

BBBC011_WORMS_METABOLISM = DatasetSpec(
    id="BBBC011_worms_metabolism",
    urls=["https://data.broadinstitute.org/bbbc/BBBC011/BBBC011_v1_images.zip"],
    size_bytes=39_876_190,
    archive_format=ArchiveFormat.ZIP,
    microscope_type="bbbc011",
    validation_rule=DatasetValidationRule.NON_EMPTY,
)

BBBC012_WORMS_INFECTION_MARKER = DatasetSpec(
    id="BBBC012_worms_infection_marker",
    urls=["https://data.broadinstitute.org/bbbc/BBBC012/BBBC012_v1_images.zip"],
    size_bytes=122_677_100,
    archive_format=ArchiveFormat.ZIP,
    microscope_type="bbbc012",
    validation_rule=DatasetValidationRule.NON_EMPTY,
)

BBBC013_U2OS_TRANSLOCATION = DatasetSpec(
    id="BBBC013_u2os_translocation_bmp",
    urls=[
        "https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_v1_images_bmp.zip",
        "https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_reproduce_logan.zip",
    ],
    size_bytes=37_962_288,
    archive_format=ArchiveFormat.ZIP,
    microscope_type="bbbc013",
    validation_rule=DatasetValidationRule.NON_EMPTY,
    reference_cppipe_urls=(
        "https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_reproduce_logan.zip",
    ),
)

# Full BBBC038 dataset (all three archives)
BBBC038_FULL = DatasetSpec(
    id="BBBC038_full",
    urls=[
        "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
        "https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip",
        "https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip",
    ],
    size_bytes=382_000_000,  # ~382 MB total
    archive_format=ArchiveFormat.ZIP,
    microscope_type="bbbc038",
    validation_rule=DatasetValidationRule.IMAGE_COUNT,
    expected_count=33_215,  # actual discovered image count
)

BBBC039_NUCLEI_SEGMENTATION = DatasetSpec(
    id="BBBC039_nuclei_segmentation",
    urls=[
        "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
        "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
        "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
    ],
    size_bytes=80_687_375,
    archive_format=ArchiveFormat.ZIP,
    microscope_type="bbbc039",
    validation_rule=DatasetValidationRule.IMAGE_COUNT,
    expected_count=800,  # 400 TIFF source images + 400 PNG instance masks
)

SINGH_2014_ILLUMINATION_CORRECTION = DatasetSpec(
    id="Singh_2014_illumination_correction",
    urls=["https://cellprofiler-published-pipelines.s3.amazonaws.com/JMicroscopy_Singh_2014.zip"],
    size_bytes=30_619_586,
    archive_format=ArchiveFormat.ZIP,
    microscope_type="published_pipeline",
    validation_rule=DatasetValidationRule.NON_EMPTY,
)

SANZ_2019_HISTOLOGY = DatasetSpec(
    id="Sanz_2019_histology",
    urls=["https://cellprofiler-published-pipelines.s3.amazonaws.com/Sanz_JAP_2019.zip"],
    size_bytes=4_541_253,
    archive_format=ArchiveFormat.ZIP,
    microscope_type="published_pipeline",
    validation_rule=DatasetValidationRule.NON_EMPTY,
)

TIAN_2019_NEURONS = DatasetSpec(
    id="Tian_2019_neurons",
    urls=["https://cellprofiler-published-pipelines.s3.amazonaws.com/Tian_Neuron_2019.zip"],
    size_bytes=52_207,
    archive_format=ArchiveFormat.ZIP,
    microscope_type="published_pipeline",
    validation_rule=DatasetValidationRule.NON_EMPTY,
)

SOKOLOV_2023_NEURONS = DatasetSpec(
    id="Sokolov_2023_neurons",
    urls=[
        "https://cellprofiler-published-pipelines.s3.amazonaws.com/AM+Sokolov+Cell+Morphology+pipeline.zip"
    ],
    size_bytes=3_403,
    archive_format=ArchiveFormat.ZIP,
    microscope_type="published_pipeline",
    validation_rule=DatasetValidationRule.NON_EMPTY,
)

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    BBBC021_SINGLE_PLATE.id: BBBC021_SINGLE_PLATE,
    BBBC022_SINGLE_PLATE_DNA.id: BBBC022_SINGLE_PLATE_DNA,
    BBBC010_WORMS.id: BBBC010_WORMS,
    BBBC011_WORMS_METABOLISM.id: BBBC011_WORMS_METABOLISM,
    BBBC012_WORMS_INFECTION_MARKER.id: BBBC012_WORMS_INFECTION_MARKER,
    BBBC013_U2OS_TRANSLOCATION.id: BBBC013_U2OS_TRANSLOCATION,
    BBBC038_FULL.id: BBBC038_FULL,
    BBBC039_NUCLEI_SEGMENTATION.id: BBBC039_NUCLEI_SEGMENTATION,
    SINGH_2014_ILLUMINATION_CORRECTION.id: SINGH_2014_ILLUMINATION_CORRECTION,
    SANZ_2019_HISTOLOGY.id: SANZ_2019_HISTOLOGY,
    TIAN_2019_NEURONS.id: TIAN_2019_NEURONS,
    SOKOLOV_2023_NEURONS.id: SOKOLOV_2023_NEURONS,
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    """
    Retrieve a dataset specification by id.

    Raises:
        KeyError: if dataset id is unknown.
    """
    try:
        return DATASET_REGISTRY[dataset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset id '{dataset_id}'. "
                       f"Available: {list(DATASET_REGISTRY.keys())}") from exc
