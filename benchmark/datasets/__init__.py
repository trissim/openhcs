"""Dataset utilities and registry."""

from openhcs.core.public_api import public_names_from_objects

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
    BBBC021_SINGLE_PLATE as BBBC021_SINGLE_PLATE,
    CELLPROFILER4_BENCHMARK_SUPPLEMENT as CELLPROFILER4_BENCHMARK_SUPPLEMENT,
    CELLPROFILER_TUTORIALS as CELLPROFILER_TUTORIALS,
    DATASET_REGISTRY as DATASET_REGISTRY,
    get_dataset_spec,
)
from benchmark.datasets.visible_source import resolve_visible_source_path

__all__ = public_names_from_objects(
    ArchiveFormat,
    CellProfilerBenchmarkCaseSpec,
    DatasetSourceKind,
    DatasetSourceSpec,
    DatasetValidationRule,
    DatasetAcquisitionError,
    acquire_dataset,
    comparison_manifest_cases,
    comparison_manifest_payload,
    write_comparison_manifest,
    get_dataset_spec,
    resolve_visible_source_path,
    extra_names=(
        "BBBC021_SINGLE_PLATE",
        "CELLPROFILER4_BENCHMARK_SUPPLEMENT",
        "CELLPROFILER_TUTORIALS",
        "DATASET_REGISTRY",
    ),
)
