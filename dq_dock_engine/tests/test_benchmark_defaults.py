from __future__ import annotations

import inspect

from dq_dock_engine.benchmark import benchmark_pdb


def test_benchmark_retry_defaults_are_one() -> None:
    assert benchmark_pdb.DEFAULT_BENCHMARK_MAX_RETRIES == 1
    assert (
        inspect.signature(benchmark_pdb.run_dq_dock).parameters["max_retries"].default
        == 1
    )
    assert (
        inspect.signature(benchmark_pdb.run_benchmark).parameters["max_retries"].default
        == 1
    )
