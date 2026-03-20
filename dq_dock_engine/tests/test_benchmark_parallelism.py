from __future__ import annotations

from dataclasses import dataclass

from dq_dock_engine.benchmark import benchmark_pdb


def test_benchmark_parallelism_rejects_non_positive_workers() -> None:
    try:
        benchmark_pdb.BenchmarkParallelism(max_workers=0)
    except ValueError as exc:
        assert "max_workers must be positive" in str(exc)
        return
    raise AssertionError("expected BenchmarkParallelism to reject max_workers=0")


def test_execute_benchmark_jobs_runs_sequentially_when_single_worker() -> None:
    seen: list[int] = []

    def run_job(value: int) -> int:
        seen.append(value)
        return value * 2

    results = list(
        benchmark_pdb.execute_benchmark_jobs(
            (1, 2, 3),
            run_job=run_job,
            parallelism=benchmark_pdb.BenchmarkParallelism(max_workers=1),
        )
    )

    assert results == [2, 4, 6]
    assert seen == [1, 2, 3]


def test_execute_benchmark_jobs_uses_process_executor_for_multi_worker(
    monkeypatch,
) -> None:
    submitted: list[int] = []

    @dataclass(frozen=True)
    class FakeFuture:
        value: int

        def result(self) -> int:
            return self.value * 10

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, run_job, job: int) -> FakeFuture:
            submitted.append(job)
            return FakeFuture(job)

    monkeypatch.setattr(benchmark_pdb, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(benchmark_pdb, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        benchmark_pdb.mp,
        "get_context",
        lambda method: f"ctx:{method}",
    )

    results = list(
        benchmark_pdb.execute_benchmark_jobs(
            (4, 5),
            run_job=lambda value: value,
            parallelism=benchmark_pdb.BenchmarkParallelism(max_workers=2),
        )
    )

    assert submitted == [4, 5]
    assert results == [40, 50]
