"""Validation reports for public Bio-Formats HCS sample datasets."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from benchmark.contracts.dataset import (
    AcquiredDataset,
    BenchmarkDatasetTag,
    DatasetSpec,
)
from benchmark.datasets.acquire import acquire_dataset
from benchmark.datasets.registry import DATASET_REGISTRY
from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.bioformats import BioFormatsHandler


BIOFORMATS_HCS_VALIDATION_CSV = "bioformats_hcs_validation.csv"
BIOFORMATS_HCS_VALIDATION_JSON = "bioformats_hcs_validation.json"
BIOFORMATS_HCS_COMPONENT_FIELDS = (
    "wells",
    "sites",
    "channels",
    "z_indexes",
    "timepoints",
)


@dataclass(frozen=True, slots=True)
class BioFormatsHcsValidationResult:
    """Validation row for one public Bio-Formats HCS dataset."""

    dataset_id: str
    path: str
    status: str
    cached: bool
    size_bytes: int
    image_count: int | None
    virtual_file_count: int | None
    wells: tuple[str, ...]
    sites: tuple[str, ...]
    channels: tuple[str, ...]
    z_indexes: tuple[str, ...]
    timepoints: tuple[str, ...]
    well_count: int | None
    site_count: int | None
    channel_count: int | None
    z_count: int | None
    timepoint_count: int | None
    grid_dimensions: tuple[int, int] | None
    loaded_plane_count: int
    load_shapes: tuple[str, ...]
    load_dtypes: tuple[str, ...]
    initialize_seconds: float | None
    load_seconds: float | None
    error: str

    @classmethod
    def unavailable(
        cls,
        spec: DatasetSpec,
        *,
        status: str,
        error: str,
    ) -> "BioFormatsHcsValidationResult":
        """Build the report row for a dataset that was not validated."""
        return cls(
            dataset_id=spec.id,
            path="",
            status=status,
            cached=False,
            size_bytes=spec.size_bytes,
            image_count=None,
            virtual_file_count=None,
            wells=(),
            sites=(),
            channels=(),
            z_indexes=(),
            timepoints=(),
            well_count=None,
            site_count=None,
            channel_count=None,
            z_count=None,
            timepoint_count=None,
            grid_dimensions=None,
            loaded_plane_count=0,
            load_shapes=(),
            load_dtypes=(),
            initialize_seconds=None,
            load_seconds=None,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class BioFormatsHcsValidationOutputs:
    """Paths and result rows produced by a validation run."""

    results: tuple[BioFormatsHcsValidationResult, ...]
    summary_csv: Path
    summary_json: Path


def bioformats_hcs_validation_specs() -> tuple[DatasetSpec, ...]:
    """Return dataset-registry specs tagged for Bio-Formats HCS validation."""
    return tuple(
        spec
        for spec in DATASET_REGISTRY.values()
        if spec.microscope_type == "bioformats"
        and BenchmarkDatasetTag.BIOFORMATS_HCS_VALIDATION in spec.tags
    )


def validate_acquired_bioformats_hcs_dataset(
    spec: DatasetSpec,
    acquired: AcquiredDataset,
    *,
    load_sample_count: int = 1,
) -> BioFormatsHcsValidationResult:
    """Project and sample-load one already acquired Bio-Formats HCS dataset."""
    filemanager = _bioformats_filemanager()
    handler = create_microscope_handler(
        "auto",
        plate_folder=acquired.path,
        filemanager=filemanager,
    )
    if not isinstance(handler, BioFormatsHandler):
        raise ValueError(
            "Bio-Formats HCS validation expected auto-detection to select "
            f"BioFormatsHandler, got {type(handler).__name__}."
        )

    initialize_start = time.perf_counter()
    handler.initialize_workspace(acquired.path, filemanager)
    initialize_seconds = time.perf_counter() - initialize_start

    backend = handler.get_primary_backend(acquired.path, filemanager)
    virtual_files = tuple(
        filemanager.list_files(acquired.path, backend, extensions={".tif"})
    )
    sample_paths = virtual_files[: max(load_sample_count, 0)]

    load_shapes: list[str] = []
    load_dtypes: list[str] = []
    load_start = time.perf_counter()
    for virtual_path in sample_paths:
        plane = filemanager.load(virtual_path, backend)
        load_shapes.append(_shape_label(plane.shape))
        load_dtypes.append(str(plane.dtype))
    load_seconds = time.perf_counter() - load_start

    metadata_handler = handler.metadata_handler
    wells = _component_keys(metadata_handler.get_well_values(acquired.path))
    sites = _component_keys(metadata_handler.get_site_values(acquired.path))
    channels = _component_keys(metadata_handler.get_channel_values(acquired.path))
    z_indexes = _component_keys(metadata_handler.get_z_index_values(acquired.path))
    timepoints = _component_keys(metadata_handler.get_timepoint_values(acquired.path))
    grid_dimensions = metadata_handler.get_grid_dimensions(acquired.path)

    return BioFormatsHcsValidationResult(
        dataset_id=spec.id,
        path=str(acquired.path),
        status="passed",
        cached=bool(acquired.metadata.get("cached", False)),
        size_bytes=spec.size_bytes,
        image_count=acquired.image_count,
        virtual_file_count=len(virtual_files),
        wells=wells,
        sites=sites,
        channels=channels,
        z_indexes=z_indexes,
        timepoints=timepoints,
        well_count=len(wells),
        site_count=len(sites),
        channel_count=len(channels),
        z_count=len(z_indexes),
        timepoint_count=len(timepoints),
        grid_dimensions=grid_dimensions,
        loaded_plane_count=len(sample_paths),
        load_shapes=tuple(load_shapes),
        load_dtypes=tuple(load_dtypes),
        initialize_seconds=initialize_seconds,
        load_seconds=load_seconds,
        error="",
    )


def validate_bioformats_hcs_catalog(
    specs: Iterable[DatasetSpec] | None = None,
    *,
    cache_base: Path | None = None,
    output_dir: Path,
    load_sample_count: int = 1,
    max_size_bytes: int | None = None,
    continue_on_error: bool = False,
) -> BioFormatsHcsValidationOutputs:
    """Acquire public HCS datasets, validate projection/loadability, and write reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[BioFormatsHcsValidationResult] = []
    for spec in specs if specs is not None else bioformats_hcs_validation_specs():
        if max_size_bytes is not None and spec.size_bytes > max_size_bytes:
            result_rows.append(
                BioFormatsHcsValidationResult.unavailable(
                    spec,
                    status="skipped",
                    error=(
                        f"Dataset size {spec.size_bytes} exceeds max-size "
                        f"limit {max_size_bytes}."
                    ),
                )
            )
            continue
        try:
            acquired = acquire_dataset(spec, cache_base=cache_base)
            result_rows.append(
                validate_acquired_bioformats_hcs_dataset(
                    spec,
                    acquired,
                    load_sample_count=load_sample_count,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            result_rows.append(
                BioFormatsHcsValidationResult.unavailable(
                    spec,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    summary_csv = output_dir / BIOFORMATS_HCS_VALIDATION_CSV
    summary_json = output_dir / BIOFORMATS_HCS_VALIDATION_JSON
    write_bioformats_hcs_validation_outputs(result_rows, summary_csv, summary_json)
    return BioFormatsHcsValidationOutputs(
        results=tuple(result_rows),
        summary_csv=summary_csv,
        summary_json=summary_json,
    )


def write_bioformats_hcs_validation_outputs(
    results: Iterable[BioFormatsHcsValidationResult],
    summary_csv: Path,
    summary_json: Path,
) -> None:
    """Write CSV and JSON summaries with stable report columns."""
    result_rows = tuple(results)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=_result_fieldnames())
        writer.writeheader()
        for result in result_rows:
            writer.writerow(_csv_row(result))
    summary_json.write_text(
        json.dumps([_json_row(result) for result in result_rows], indent=2),
        encoding="utf-8",
    )


def _bioformats_filemanager():
    from polystore.base import ensure_storage_registry, storage_registry
    from polystore.filemanager import FileManager

    ensure_storage_registry()
    return FileManager(dict(storage_registry))


def _component_keys(values: Mapping[str, Any] | None) -> tuple[str, ...]:
    keys = tuple(str(value) for value in (values or {}).keys())
    if all(value.isdigit() for value in keys):
        return tuple(sorted(keys, key=int))
    return tuple(sorted(keys))


def _shape_label(shape: tuple[int, ...]) -> str:
    if not shape:
        return ""
    return "x".join(str(value) for value in shape)


def _result_fieldnames() -> tuple[str, ...]:
    return tuple(field.name for field in fields(BioFormatsHcsValidationResult))


def _csv_row(result: BioFormatsHcsValidationResult) -> dict[str, object]:
    row = asdict(result)
    for field_name in BIOFORMATS_HCS_COMPONENT_FIELDS:
        row[field_name] = ";".join(row[field_name])
    row["grid_dimensions"] = (
        "" if result.grid_dimensions is None else _shape_label(result.grid_dimensions)
    )
    row["load_shapes"] = ";".join(result.load_shapes)
    row["load_dtypes"] = ";".join(result.load_dtypes)
    return row


def _json_row(result: BioFormatsHcsValidationResult) -> dict[str, object]:
    row = asdict(result)
    for field_name in BIOFORMATS_HCS_COMPONENT_FIELDS:
        row[field_name] = list(row[field_name])
    row["grid_dimensions"] = (
        None if result.grid_dimensions is None else list(result.grid_dimensions)
    )
    row["load_shapes"] = list(result.load_shapes)
    row["load_dtypes"] = list(result.load_dtypes)
    return row
