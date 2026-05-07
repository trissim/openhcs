"""Dataset contracts for benchmark platform."""

from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from nominal_refactor_advisor.record_algebra import product_record


class ArchiveFormat(Enum):
    """Supported dataset archive formats."""

    ZIP = "zip"


class DatasetValidationRule(Enum):
    """Dataset acquisition validation strategies."""

    IMAGE_COUNT = "image_count"
    MANIFEST = "manifest"
    NON_EMPTY = "non_empty"


class DatasetSourceKind(Enum):
    """Supported acquisition source families."""

    ARCHIVE_URLS = "archive_urls"
    GIT_SPARSE = "git_sparse"


DatasetSourceSpec = product_record(
    "DatasetSourceSpec",
    (
        "kind: DatasetSourceKind; urls: tuple[str, ...]; "
        "git_url: str | None; git_ref: str; sparse_paths: tuple[str, ...]"
    ),
    defaults={
        "urls": (),
        "git_url": None,
        "git_ref": "HEAD",
        "sparse_paths": (),
    },
    doc="Nominal description of where a benchmark dataset comes from.",
    module_name=__name__,
)
BenchmarkCategory = product_record(
    "BenchmarkCategory",
    "assay: str; module: str",
    doc="Semantic report category attached to a benchmark case declaration.",
    module_name=__name__,
)
CellProfilerBenchmarkCaseSpec = product_record(
    "CellProfilerBenchmarkCaseSpec",
    (
        "name: str; cppipe_path: Path; dataset_path: Path; "
        "dataset_id: str | None; microscope_type: str | None; "
        "category: BenchmarkCategory | None; value_only: bool; "
        "equivalence_reference_output_dir: Path | None; "
        "cellprofiler_timeout_seconds: float | None"
    ),
    defaults={
        "dataset_id": None,
        "microscope_type": None,
        "category": None,
        "value_only": False,
        "equivalence_reference_output_dir": None,
        "cellprofiler_timeout_seconds": None,
    },
    doc="Dataset-relative CellProfiler benchmark case.",
    module_name=__name__,
)


@dataclass(frozen=True)
class DatasetSpec:
    """
    Immutable dataset specification.

    This is the contract all benchmark datasets must satisfy.
    Adding a new dataset = defining a new DatasetSpec instance.
    """
    id: str
    """Unique identifier (e.g., 'BBBC021', 'BBBC038')"""

    urls: list[str]
    """Download URLs for dataset archives"""

    size_bytes: int
    """Total expected size after download"""

    archive_format: ArchiveFormat
    """Archive format."""

    microscope_type: str
    """Microscope handler type (e.g., 'bbbc021', 'bbbc038')"""

    validation_rule: DatasetValidationRule
    """How to validate extracted data."""

    reference_cppipe_urls: tuple[str, ...] = ()
    """Canonical CellProfiler pipelines associated with the dataset, if any."""

    expected_count: int | None = None
    """Expected number of image files (for 'count' validation)"""

    manifest_path: Path | None = None
    """Path to manifest CSV (for 'manifest' validation)"""

    source: DatasetSourceSpec | None = None
    """Acquisition source. If omitted, urls/archive_format are used."""

    benchmark_cases: tuple[CellProfilerBenchmarkCaseSpec, ...] = ()
    """Dataset-relative .cppipe benchmark cases materialized after acquisition."""

    def acquisition_source(self) -> DatasetSourceSpec:
        """Return the normalized acquisition source for this dataset."""
        if self.source is not None:
            return self.source
        return DatasetSourceSpec(
            kind=DatasetSourceKind.ARCHIVE_URLS,
            urls=tuple(self.urls),
        )


@dataclass
class AcquiredDataset:
    """
    Dataset returned by acquisition.

    This is what tool adapters receive.
    """
    id: str
    path: Path
    microscope_type: str
    image_count: int
    metadata: dict
