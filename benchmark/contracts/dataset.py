"""Dataset contracts for benchmark platform."""

from pathlib import Path
from dataclasses import dataclass


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

    archive_format: str
    """Archive format: 'zip', 'tar.gz', etc."""

    microscope_type: str
    """Microscope handler type (e.g., 'bbbc021', 'bbbc038')"""

    validation_rule: str
    """How to validate: 'count' or 'manifest'"""

    reference_cppipe_urls: tuple[str, ...] = ()
    """Canonical CellProfiler pipelines associated with the dataset, if any."""

    expected_count: int | None = None
    """Expected number of image files (for 'count' validation)"""

    manifest_path: Path | None = None
    """Path to manifest CSV (for 'manifest' validation)"""


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
