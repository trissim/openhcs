"""Dataset contracts for benchmark platform."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from enum import Enum


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
    URL_FILES = "url_files"
    GIT_SPARSE = "git_sparse"
    GIT_SPARSE_WITH_ARCHIVES = "git_sparse_with_archives"


@dataclass(frozen=True, slots=True)
class DatasetSourceSpec:
    """Nominal description of where a benchmark dataset comes from."""

    kind: DatasetSourceKind
    urls: tuple[str, ...] = ()
    git_url: str | None = None
    git_ref: str = "HEAD"
    sparse_paths: tuple[str, ...] = ()
    tls_verify: bool = True


@dataclass(frozen=True, slots=True)
class BenchmarkCategory:
    """Semantic report category attached to a benchmark case declaration."""

    assay: str
    module: str


@dataclass(frozen=True, slots=True)
class CellProfilerBenchmarkCaseSpec:
    """Dataset-relative CellProfiler benchmark case."""

    name: str
    cppipe_path: Path
    dataset_path: Path
    dataset_id: str | None = None
    microscope_type: str | None = None
    category: BenchmarkCategory | None = None
    value_only: bool = False
    equivalence_reference_output_dir: Path | None = None
    cellprofiler_timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
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
