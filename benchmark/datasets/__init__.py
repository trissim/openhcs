"""Dataset utilities and registry."""

from benchmark.contracts.dataset import (
    ArchiveFormat,
    CellProfilerBenchmarkCaseSpec,
    DatasetSourceKind,
    DatasetSourceSpec,
    DatasetValidationRule,
)
from benchmark.datasets.acquire import acquire_dataset, DatasetAcquisitionError
from benchmark.datasets.manifest import (
    comparison_manifest_cases,
    comparison_manifest_payload,
    write_comparison_manifest,
)
from benchmark.datasets.registry import (
    BBBC021_SINGLE_PLATE,
    CELLPROFILER4_BENCHMARK_SUPPLEMENT,
    CELLPROFILER_TUTORIALS,
    DATASET_REGISTRY,
    get_dataset_spec,
)
from benchmark.datasets.visible_source import resolve_visible_source_path

__all__ = sorted(name for name in globals() if not name.startswith("_"))
