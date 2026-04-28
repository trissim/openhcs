"""Registry of benchmark datasets."""

from benchmark.contracts.dataset import DatasetSpec

# Core quick-start dataset (single BBBC021 plate)
BBBC021_SINGLE_PLATE = DatasetSpec(
    id="BBBC021_Week1_22123",
    urls=["https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_images_Week1_22123.zip"],
    size_bytes=839_000_000,  # 839 MB
    archive_format="zip",
    microscope_type="bbbc021",
    validation_rule="count",
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
    archive_format="zip",
    microscope_type="bbbc022",
    validation_rule="count",
    expected_count=3_456,  # 384 wells × 9 sites × 1 channel
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
    archive_format="zip",
    microscope_type="bbbc038",
    validation_rule="count",
    expected_count=33_215,  # actual discovered image count
)

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    BBBC021_SINGLE_PLATE.id: BBBC021_SINGLE_PLATE,
    BBBC022_SINGLE_PLATE_DNA.id: BBBC022_SINGLE_PLATE_DNA,
    BBBC038_FULL.id: BBBC038_FULL,
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
