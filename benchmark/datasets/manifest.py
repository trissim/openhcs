"""Materialize acquired datasets into benchmark manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from benchmark.contracts.dataset import AcquiredDataset, DatasetSpec


def cached_acquired_dataset(
    spec: DatasetSpec,
    *,
    cache_base: Path | None = None,
) -> AcquiredDataset:
    """Resolve an already-acquired dataset without refreshing its source."""
    base_dir = cache_base or Path.home() / ".cache" / "openhcs" / "benchmark_datasets"
    data_dir = base_dir / spec.id / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Cached dataset {spec.id!r} not found at {data_dir}. "
            "Acquire it before building the merged benchmark manifest."
        )
    return AcquiredDataset(
        id=spec.id,
        path=data_dir,
        microscope_type=spec.microscope_type,
        image_count=0,
        metadata={"cached": True},
    )


def cached_case_bearing_datasets(
    specs: Iterable[DatasetSpec],
    *,
    cache_base: Path | None = None,
) -> list[tuple[DatasetSpec, AcquiredDataset]]:
    """Resolve cached acquired datasets for specs with declared benchmark cases."""
    return [
        (spec, cached_acquired_dataset(spec, cache_base=cache_base))
        for spec in specs
        if spec.benchmark_cases
    ]


def comparison_manifest_cases(
    spec: DatasetSpec,
    acquired: AcquiredDataset,
    *,
    case_names: set[str] | None = None,
) -> list[dict[str, object]]:
    """Build CellProfiler-vs-OpenHCS manifest cases for an acquired dataset."""
    cases: list[dict[str, object]] = []
    for case in spec.benchmark_cases:
        if case_names is not None and case.name not in case_names:
            continue
        dataset_path = acquired.path / case.dataset_path
        cppipe_path = acquired.path / case.cppipe_path
        _require_path(dataset_path, "dataset path", case.name)
        _require_path(cppipe_path, ".cppipe path", case.name)

        payload: dict[str, object] = {
            "name": case.name,
            "dataset_path": str(dataset_path),
            "cppipe_path": str(cppipe_path),
            "dataset_id": case.dataset_id or spec.id,
            "microscope_type": case.microscope_type or spec.microscope_type,
            "value_only": case.value_only,
        }
        if case.category is not None:
            payload["assay_category"] = case.category.assay
            payload["module_category"] = case.category.module
        if case.equivalence_reference_output_dir is not None:
            payload["equivalence_reference_output_dir"] = str(
                acquired.path / case.equivalence_reference_output_dir
            )
        if case.cellprofiler_timeout_seconds is not None:
            payload["cellprofiler_timeout_seconds"] = case.cellprofiler_timeout_seconds
        cases.append(payload)
    return cases


def comparison_manifest_payload(
    datasets: Iterable[tuple[DatasetSpec, AcquiredDataset]],
    *,
    case_names: set[str] | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable comparison manifest payload."""
    cases: list[dict[str, object]] = []
    for spec, acquired in datasets:
        cases.extend(comparison_manifest_cases(spec, acquired, case_names=case_names))
    return {"cases": cases}


def write_comparison_manifest(payload: dict[str, object], output_path: Path) -> None:
    """Write a comparison manifest JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _require_path(path: Path, label: str, case_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark case {case_name!r} references missing {label}: {path}"
        )
