"""Validation reports for public Bio-Formats HCS sample datasets."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

from benchmark.contracts.dataset import AcquiredDataset
from benchmark.datasets.acquire import acquire_dataset
from benchmark.datasets.bioformats_hcs import (
    BIOFORMATS_HCS_CATALOG,
    BioFormatsHcsAxisExpectation,
    BioFormatsHcsCatalogRow,
)
from openhcs.constants.constants import Backend
from openhcs.microscopes.bioformats import BioFormatsHandler


BIOFORMATS_HCS_VALIDATION_CSV = "bioformats_hcs_validation.csv"
BIOFORMATS_HCS_VALIDATION_JSON = "bioformats_hcs_validation.json"
BIOFORMATS_HCS_AXIS_FIELDS = (
    "wells",
    "sites",
    "channels",
    "z_indexes",
    "timepoints",
)


@dataclass(frozen=True)
class BioFormatsHcsAxisProjection:
    """Expected and observed OpenHCS axis keys for one Bio-Formats validation row."""

    expected: BioFormatsHcsAxisExpectation
    observed: BioFormatsHcsAxisExpectation

    @classmethod
    def unavailable(
        cls,
        expected: BioFormatsHcsAxisExpectation,
    ) -> "BioFormatsHcsAxisProjection":
        empty = BioFormatsHcsAxisExpectation(
            wells=(),
            sites=(),
            channels=(),
            z_indexes=(),
            timepoints=(),
        )
        return cls(expected=expected, observed=empty)

    @classmethod
    def from_metadata(
        cls,
        expected: BioFormatsHcsAxisExpectation,
        metadata: dict[str, Any],
    ) -> "BioFormatsHcsAxisProjection":
        return cls(expected=expected, observed=_observed_axes(metadata))

    def validate(self) -> None:
        mismatches = []
        for field_name in BIOFORMATS_HCS_AXIS_FIELDS:
            expected_value = getattr(self.expected, field_name)
            observed_value = getattr(self.observed, field_name)
            if observed_value != expected_value:
                mismatches.append(
                    f"{field_name}: expected {expected_value}, observed {observed_value}"
                )
        if mismatches:
            raise ValueError(
                "Bio-Formats HCS axis projection mismatch: " + "; ".join(mismatches)
            )

    def flat_csv_values(self) -> dict[str, str]:
        return {
            f"{role}_{field_name}": ";".join(getattr(axis_set, field_name))
            for role, axis_set in (
                ("expected", self.expected),
                ("observed", self.observed),
            )
            for field_name in BIOFORMATS_HCS_AXIS_FIELDS
        }

    def flat_json_values(self) -> dict[str, list[str]]:
        return {
            f"{role}_{field_name}": list(getattr(axis_set, field_name))
            for role, axis_set in (
                ("expected", self.expected),
                ("observed", self.observed),
            )
            for field_name in BIOFORMATS_HCS_AXIS_FIELDS
        }


@dataclass(frozen=True)
class BioFormatsHcsValidationResult:
    """Paper-facing validation row for one public Bio-Formats HCS dataset."""

    dataset_id: str
    display_name: str
    vendor: str
    format_name: str
    source_page: str
    path: str
    status: str
    cached: bool
    size_bytes: int
    image_count: int | None
    virtual_file_count: int | None
    well_count: int | None
    site_count: int | None
    channel_count: int | None
    z_count: int | None
    timepoint_count: int | None
    axis_projection: BioFormatsHcsAxisProjection
    loaded_plane_count: int
    load_shapes: tuple[str, ...]
    load_dtypes: tuple[str, ...]
    initialize_seconds: float | None
    load_seconds: float | None
    error: str
    notes: str

    @classmethod
    def unavailable(
        cls,
        row: BioFormatsHcsCatalogRow,
        *,
        status: str,
        error: str,
    ) -> "BioFormatsHcsValidationResult":
        """Build the report row for a dataset that was not validated."""
        return cls(
            dataset_id=row.spec.id,
            display_name=row.display_name,
            vendor=row.vendor,
            format_name=row.format_name,
            source_page=row.source_page,
            path="",
            status=status,
            cached=False,
            size_bytes=row.spec.size_bytes,
            image_count=None,
            virtual_file_count=None,
            well_count=None,
            site_count=None,
            channel_count=None,
            z_count=None,
            timepoint_count=None,
            axis_projection=BioFormatsHcsAxisProjection.unavailable(row.axes),
            loaded_plane_count=0,
            load_shapes=(),
            load_dtypes=(),
            initialize_seconds=None,
            load_seconds=None,
            error=error,
            notes=row.notes,
        )


@dataclass(frozen=True)
class BioFormatsHcsValidationOutputs:
    """Paths and result rows produced by a validation run."""

    results: tuple[BioFormatsHcsValidationResult, ...]
    summary_csv: Path
    summary_json: Path


def validate_acquired_bioformats_hcs_dataset(
    row: BioFormatsHcsCatalogRow,
    acquired: AcquiredDataset,
    *,
    load_sample_count: int = 1,
) -> BioFormatsHcsValidationResult:
    """Project and sample-load one already acquired Bio-Formats HCS dataset."""
    filemanager = _bioformats_filemanager()
    handler = BioFormatsHandler(filemanager)

    initialize_start = time.perf_counter()
    handler.initialize_workspace(acquired.path, filemanager)
    initialize_seconds = time.perf_counter() - initialize_start

    backend = filemanager.registry[Backend.BIOFORMATS.value]
    virtual_files = tuple(backend.list_files(acquired.path, extensions={".tif"}))
    sample_paths = virtual_files[: max(load_sample_count, 0)]

    load_shapes: list[str] = []
    load_dtypes: list[str] = []
    load_start = time.perf_counter()
    for virtual_path in sample_paths:
        plane = backend.load(virtual_path)
        load_shapes.append(_shape_label(getattr(plane, "shape", ())))
        load_dtypes.append(str(getattr(plane, "dtype", type(plane).__name__)))
    load_seconds = time.perf_counter() - load_start

    metadata = _workspace_subdirectory_metadata(acquired.path)
    axis_projection = BioFormatsHcsAxisProjection.from_metadata(row.axes, metadata)
    axis_projection.validate()
    return BioFormatsHcsValidationResult(
        dataset_id=row.spec.id,
        display_name=row.display_name,
        vendor=row.vendor,
        format_name=row.format_name,
        source_page=row.source_page,
        path=str(acquired.path),
        status="passed",
        cached=bool(acquired.metadata.get("cached", False)),
        size_bytes=row.spec.size_bytes,
        image_count=acquired.image_count,
        virtual_file_count=len(virtual_files),
        well_count=_component_count(metadata, "wells"),
        site_count=_component_count(metadata, "sites"),
        channel_count=_component_count(metadata, "channels"),
        z_count=_component_count(metadata, "z_indexes"),
        timepoint_count=_component_count(metadata, "timepoints"),
        axis_projection=axis_projection,
        loaded_plane_count=len(sample_paths),
        load_shapes=tuple(load_shapes),
        load_dtypes=tuple(load_dtypes),
        initialize_seconds=initialize_seconds,
        load_seconds=load_seconds,
        error="",
        notes=row.notes,
    )


def validate_bioformats_hcs_catalog(
    rows: Iterable[BioFormatsHcsCatalogRow] = BIOFORMATS_HCS_CATALOG,
    *,
    cache_base: Path | None = None,
    output_dir: Path,
    load_sample_count: int = 1,
    max_size_bytes: int | None = None,
    continue_on_error: bool = False,
) -> BioFormatsHcsValidationOutputs:
    """Acquire public HCS datasets, validate projection/loadability, and write reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[BioFormatsHcsValidationResult] = []
    for row in rows:
        if max_size_bytes is not None and row.spec.size_bytes > max_size_bytes:
            results.append(
                BioFormatsHcsValidationResult.unavailable(
                    row,
                    status="skipped",
                    error=(
                        f"Dataset size {row.spec.size_bytes} exceeds max-size "
                        f"limit {max_size_bytes}."
                    ),
                )
            )
            continue
        try:
            acquired = acquire_dataset(row.spec, cache_base=cache_base)
            results.append(
                validate_acquired_bioformats_hcs_dataset(
                    row,
                    acquired,
                    load_sample_count=load_sample_count,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            results.append(
                BioFormatsHcsValidationResult.unavailable(
                    row,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    summary_csv = output_dir / BIOFORMATS_HCS_VALIDATION_CSV
    summary_json = output_dir / BIOFORMATS_HCS_VALIDATION_JSON
    write_bioformats_hcs_validation_outputs(results, summary_csv, summary_json)
    return BioFormatsHcsValidationOutputs(
        results=tuple(results),
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


def _workspace_subdirectory_metadata(plate_path: Path) -> dict[str, Any]:
    payload = json.loads((plate_path / "openhcs_metadata.json").read_text(encoding="utf-8"))
    return payload["subdirectories"]["."]


def _component_count(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    return len(value or {})


def _observed_axes(metadata: dict[str, Any]) -> BioFormatsHcsAxisExpectation:
    return BioFormatsHcsAxisExpectation(
        wells=_component_keys(metadata, "wells"),
        sites=_component_keys(metadata, "sites"),
        channels=_component_keys(metadata, "channels"),
        z_indexes=_component_keys(metadata, "z_indexes"),
        timepoints=_component_keys(metadata, "timepoints"),
    )


def _component_keys(metadata: dict[str, Any], key: str) -> tuple[str, ...]:
    values = tuple(str(value) for value in (metadata.get(key) or {}).keys())
    if all(value.isdigit() for value in values):
        return tuple(sorted(values, key=int))
    return tuple(sorted(values))


def _shape_label(shape: Any) -> str:
    values = tuple(shape)
    if not values:
        return ""
    return "x".join(str(value) for value in values)


def _result_fieldnames() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(BioFormatsHcsValidationResult)
        if field.name != "axis_projection"
    ) + tuple(
        f"{role}_{field_name}"
        for role in ("expected", "observed")
        for field_name in BIOFORMATS_HCS_AXIS_FIELDS
    )


def _csv_row(result: BioFormatsHcsValidationResult) -> dict[str, object]:
    row = asdict(result)
    row["load_shapes"] = ";".join(result.load_shapes)
    row["load_dtypes"] = ";".join(result.load_dtypes)
    row.pop("axis_projection")
    row.update(result.axis_projection.flat_csv_values())
    return row


def _json_row(result: BioFormatsHcsValidationResult) -> dict[str, object]:
    row = asdict(result)
    row["load_shapes"] = list(result.load_shapes)
    row["load_dtypes"] = list(result.load_dtypes)
    row.pop("axis_projection")
    row.update(result.axis_projection.flat_json_values())
    return row
