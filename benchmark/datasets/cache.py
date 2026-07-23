"""Benchmark dataset filesystem path authorities."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


OPENHCS_BENCHMARK_DATASET_CACHE_ROOT_ENV = "OPENHCS_BENCHMARK_DATASET_CACHE_ROOT"
CELLPROFILER_EXAMPLES_ROOT_ENV = "CELLPROFILER_EXAMPLES_ROOT"


class BenchmarkPathRootKind(str, Enum):
    """Named persistent roots used by benchmark manifests and dataset tooling."""

    DATASET_CACHE = ("benchmark_dataset_cache", "benchmark_datasets")
    CELLPROFILER_EXAMPLES = ("cellprofiler_examples", "cellprofiler_examples")

    def __new__(cls, value: str, cache_dir_name: str) -> "BenchmarkPathRootKind":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.cache_dir_name = cache_dir_name
        return obj

    def default_path(self) -> Path:
        """Return the persistent default location for this root kind."""
        return Path.home() / ".cache" / "openhcs" / self.cache_dir_name


def default_benchmark_dataset_cache_root() -> Path:
    """Return the default persistent dataset cache root."""
    return BenchmarkPathRootKind.DATASET_CACHE.default_path()


def default_cellprofiler_examples_root() -> Path:
    """Return the default persistent CellProfiler examples checkout root."""
    return BenchmarkPathRootKind.CELLPROFILER_EXAMPLES.default_path()


def resolve_benchmark_path_root(
    kind: BenchmarkPathRootKind,
    *,
    env_name: str | None = None,
) -> Path:
    """Resolve a benchmark root through an optional environment override."""
    if env_name is not None:
        env_value = os.environ.get(env_name)
        if env_value is not None:
            return Path(os.path.expandvars(env_value)).expanduser()
    return kind.default_path()
