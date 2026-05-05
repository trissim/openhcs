"""Dataset utilities and registry."""

from benchmark.contracts.dataset import ArchiveFormat, DatasetValidationRule
from benchmark.datasets.acquire import acquire_dataset, DatasetAcquisitionError
from benchmark.datasets.registry import BBBC021_SINGLE_PLATE, DATASET_REGISTRY, get_dataset_spec
from benchmark.datasets.visible_source import resolve_visible_source_path

__all__ = [
    "ArchiveFormat",
    "BBBC021_SINGLE_PLATE",
    "DATASET_REGISTRY",
    "DatasetValidationRule",
    "get_dataset_spec",
    "acquire_dataset",
    "DatasetAcquisitionError",
    "resolve_visible_source_path",
]
