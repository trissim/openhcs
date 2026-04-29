"""Semantic equivalence checks for runtime outputs."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionObservation,
)
from openhcs.core.runtime_exports import RuntimeExportObservation


class RuntimeEquivalenceDifferenceKind(str, Enum):
    """Closed families of semantic runtime-output differences."""

    RUNTIME_ARTIFACT_COUNTS = "runtime_artifact_counts"
    TABLE_SCHEMA = "table_schema"
    TABLE_COUNT = "table_count"
    TABLE_CONTENT = "table_content"
    IMAGE_COUNT = "image_count"
    IMAGE_CONTENT = "image_content"


class RuntimeCellValueKind(str, Enum):
    """Canonical scalar families used for exported table comparison."""

    EMPTY = "empty"
    NUMBER = "number"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class RuntimeEquivalencePolicy:
    """Policy controlling semantic output comparison strictness."""

    numeric_decimal_places: int = 10
    compare_table_values: bool = True
    compare_image_pixels: bool = True

    def __post_init__(self) -> None:
        if self.numeric_decimal_places < 0:
            raise ValueError("numeric_decimal_places cannot be negative.")


@dataclass(frozen=True, slots=True)
class RuntimeCellSignature:
    """Canonical scalar value for exported table comparison."""

    kind: RuntimeCellValueKind
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            (
                self.kind
                if isinstance(self.kind, RuntimeCellValueKind)
                else RuntimeCellValueKind(self.kind)
            ),
        )

    @property
    def sort_key(self) -> tuple[str, str]:
        """Return a stable ordering key for mixed scalar families."""
        return (self.kind.value, self.value)


@dataclass(frozen=True, slots=True)
class RuntimeTableSnapshot:
    """Semantic snapshot of one exported runtime table."""

    path: Path
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @classmethod
    def from_csv(cls, path: Path) -> "RuntimeTableSnapshot":
        """Read a CSV export into a semantic table snapshot."""
        with Path(path).open(newline="") as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader, ()))
            rows = tuple(tuple(row) for row in reader)
        return cls(path=Path(path), header=header, rows=rows)

    def __post_init__(self) -> None:
        path = Path(self.path)
        header = tuple(str(column).strip() for column in self.header)
        if not header:
            raise ValueError(f"Runtime table {path} has no header.")
        duplicate_headers = _duplicates(header)
        if duplicate_headers:
            raise ValueError(
                f"Runtime table {path} has duplicate headers "
                f"{duplicate_headers!r}."
            )
        rows = tuple(tuple(str(value).strip() for value in row) for row in self.rows)
        malformed_rows = tuple(
            index
            for index, row in enumerate(rows, start=1)
            if len(row) != len(header)
        )
        if malformed_rows:
            raise ValueError(
                f"Runtime table {path} rows do not match header width at "
                f"data rows {malformed_rows!r}."
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "header", header)
        object.__setattr__(self, "rows", rows)

    @property
    def schema_key(self) -> tuple[str, ...]:
        """File-order-independent schema identity for this table."""
        return tuple(sorted(self.header))

    def content_key(
        self,
        policy: RuntimeEquivalencePolicy,
    ) -> tuple[tuple[tuple[str, str], ...], ...]:
        """File-order-independent row identity for this table."""
        columns = self.schema_key
        indexes = {column: self.header.index(column) for column in self.header}
        return tuple(
            sorted(
                tuple(
                    _cell_signature(row[indexes[column]], policy).sort_key
                    for column in columns
                )
                for row in self.rows
            )
        )


@dataclass(frozen=True, slots=True)
class RuntimeImageSnapshot:
    """Semantic snapshot of one exported runtime image."""

    path: Path
    shape: tuple[int, ...]
    dtype: str
    pixel_digest: str

    @classmethod
    def from_image_file(cls, path: Path) -> "RuntimeImageSnapshot":
        """Read an image export into a decoded-pixel semantic snapshot."""
        array = np.asarray(imageio.imread(path))
        contiguous = np.ascontiguousarray(array)
        return cls(
            path=Path(path),
            shape=tuple(int(axis) for axis in contiguous.shape),
            dtype=str(contiguous.dtype),
            pixel_digest=hashlib.sha256(contiguous.tobytes()).hexdigest(),
        )

    def content_key(
        self,
        policy: RuntimeEquivalencePolicy,
    ) -> tuple[object, ...]:
        """Return image identity at the requested semantic strictness."""
        key: tuple[object, ...] = (self.shape, self.dtype)
        if policy.compare_image_pixels:
            key = (*key, self.pixel_digest)
        return key


@dataclass(frozen=True, slots=True)
class RuntimeOutputSnapshot:
    """Semantic snapshot of runtime file outputs."""

    tables: tuple[RuntimeTableSnapshot, ...] = ()
    images: tuple[RuntimeImageSnapshot, ...] = ()

    @classmethod
    def from_export_observation(
        cls,
        observation: RuntimeExportObservation,
    ) -> "RuntimeOutputSnapshot":
        """Build a semantic output snapshot from observed runtime exports."""
        return cls(
            tables=tuple(
                RuntimeTableSnapshot.from_csv(path)
                for path in observation.table_outputs
            ),
            images=tuple(
                RuntimeImageSnapshot.from_image_file(path)
                for path in observation.image_outputs
            ),
        )

    @classmethod
    def from_output_root(cls, output_root: Path) -> "RuntimeOutputSnapshot":
        """Build a semantic output snapshot from an output directory."""
        root = Path(output_root)
        if not root.exists():
            raise FileNotFoundError(f"Runtime output root does not exist: {root}")
        return cls(
            tables=tuple(
                RuntimeTableSnapshot.from_csv(path) for path in table_paths(root)
            ),
            images=tuple(
                RuntimeImageSnapshot.from_image_file(path)
                for path in image_paths(root)
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeEquivalenceDifference:
    """One semantic difference between two runtime outputs."""

    kind: RuntimeEquivalenceDifferenceKind
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            (
                self.kind
                if isinstance(self.kind, RuntimeEquivalenceDifferenceKind)
                else RuntimeEquivalenceDifferenceKind(self.kind)
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeEquivalenceReport:
    """Semantic equivalence result for two runtime outputs."""

    differences: tuple[RuntimeEquivalenceDifference, ...]

    @property
    def is_equivalent(self) -> bool:
        """Return whether the compared outputs are semantically equivalent."""
        return not self.differences

    def failure_messages(self) -> tuple[str, ...]:
        """Return stable human-readable failure messages."""
        return tuple(difference.message for difference in self.differences)


def runtime_output_equivalence(
    reference: RuntimeOutputSnapshot,
    candidate: RuntimeOutputSnapshot,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare two runtime output snapshots for semantic equivalence."""
    return RuntimeEquivalenceReport(
        differences=(
            *_table_differences(reference.tables, candidate.tables, policy),
            *_image_differences(reference.images, candidate.images, policy),
        )
    )


def runtime_output_root_equivalence(
    reference_output_root: Path,
    candidate_output_root: Path,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare two runtime output directories for semantic equivalence."""
    return runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_output_root),
        RuntimeOutputSnapshot.from_output_root(candidate_output_root),
        policy=policy,
    )


def runtime_artifact_execution_equivalence(
    reference: RuntimeArtifactExecutionObservation,
    candidate: RuntimeArtifactExecutionObservation,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare runtime artifact state and file outputs for semantic equivalence."""
    return RuntimeEquivalenceReport(
        differences=(
            *_runtime_artifact_count_differences(reference, candidate),
            *runtime_output_equivalence(
                RuntimeOutputSnapshot.from_export_observation(reference.exports),
                RuntimeOutputSnapshot.from_export_observation(candidate.exports),
                policy=policy,
            ).differences,
        )
    )


def table_paths(output_root: Path) -> tuple[Path, ...]:
    """Return non-empty CSV output paths under an output root."""
    root = Path(output_root)
    return tuple(
        path
        for path in sorted(root.rglob("*.csv"))
        if path.is_file() and path.stat().st_size > 0
    )


def image_paths(output_root: Path) -> tuple[Path, ...]:
    """Return image output paths under an output root."""
    root = Path(output_root)
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and _is_image_path(path)
    )


def _runtime_artifact_count_differences(
    reference: RuntimeArtifactExecutionObservation,
    candidate: RuntimeArtifactExecutionObservation,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    reference_counts = _total_record_counts(reference)
    candidate_counts = _total_record_counts(candidate)
    if reference_counts == candidate_counts:
        return ()
    return (
        RuntimeEquivalenceDifference(
            RuntimeEquivalenceDifferenceKind.RUNTIME_ARTIFACT_COUNTS,
            "runtime artifact counts differ: "
            f"reference={dict(reference_counts)!r}, "
            f"candidate={dict(candidate_counts)!r}",
        ),
    )


def _total_record_counts(
    observation: RuntimeArtifactExecutionObservation,
) -> Counter[ArtifactKind]:
    counts: Counter[ArtifactKind] = Counter()
    for axis_counts in observation.record_counts_by_axis.values():
        counts.update(axis_counts)
    return counts


def _table_differences(
    reference_tables: tuple[RuntimeTableSnapshot, ...],
    candidate_tables: tuple[RuntimeTableSnapshot, ...],
    policy: RuntimeEquivalencePolicy,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    differences: list[RuntimeEquivalenceDifference] = []
    reference_groups = _tables_by_schema(reference_tables)
    candidate_groups = _tables_by_schema(candidate_tables)
    reference_schemas = set(reference_groups)
    candidate_schemas = set(candidate_groups)
    for schema in sorted(reference_schemas - candidate_schemas):
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.TABLE_SCHEMA,
                f"candidate is missing table schema {schema!r}",
            )
        )
    for schema in sorted(candidate_schemas - reference_schemas):
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.TABLE_SCHEMA,
                f"candidate has extra table schema {schema!r}",
            )
        )
    for schema in sorted(reference_schemas & candidate_schemas):
        reference_group = reference_groups[schema]
        candidate_group = candidate_groups[schema]
        if len(reference_group) != len(candidate_group):
            differences.append(
                RuntimeEquivalenceDifference(
                    RuntimeEquivalenceDifferenceKind.TABLE_COUNT,
                    f"table schema {schema!r} count differs: "
                    f"reference={len(reference_group)}, "
                    f"candidate={len(candidate_group)}",
                )
            )
        differences.extend(
            _table_content_differences(
                schema,
                reference_group,
                candidate_group,
                policy,
            )
        )
    return tuple(differences)


def _table_content_differences(
    schema: tuple[str, ...],
    reference_group: tuple[RuntimeTableSnapshot, ...],
    candidate_group: tuple[RuntimeTableSnapshot, ...],
    policy: RuntimeEquivalencePolicy,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    reference_shapes = Counter(len(table.rows) for table in reference_group)
    candidate_shapes = Counter(len(table.rows) for table in candidate_group)
    differences: list[RuntimeEquivalenceDifference] = []
    if reference_shapes != candidate_shapes:
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.TABLE_CONTENT,
                f"table schema {schema!r} row counts differ: "
                f"reference={dict(reference_shapes)!r}, "
                f"candidate={dict(candidate_shapes)!r}",
            )
        )
    if not policy.compare_table_values:
        return tuple(differences)

    reference_content = Counter(
        table.content_key(policy) for table in reference_group
    )
    candidate_content = Counter(
        table.content_key(policy) for table in candidate_group
    )
    if reference_content != candidate_content:
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.TABLE_CONTENT,
                f"table schema {schema!r} values differ",
            )
        )
    return tuple(differences)


def _image_differences(
    reference_images: tuple[RuntimeImageSnapshot, ...],
    candidate_images: tuple[RuntimeImageSnapshot, ...],
    policy: RuntimeEquivalencePolicy,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    differences: list[RuntimeEquivalenceDifference] = []
    if len(reference_images) != len(candidate_images):
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.IMAGE_COUNT,
                f"image output count differs: reference={len(reference_images)}, "
                f"candidate={len(candidate_images)}",
            )
        )
    reference_content = Counter(
        image.content_key(policy) for image in reference_images
    )
    candidate_content = Counter(
        image.content_key(policy) for image in candidate_images
    )
    if reference_content != candidate_content:
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.IMAGE_CONTENT,
                "image output content differs",
            )
        )
    return tuple(differences)


def _tables_by_schema(
    tables: tuple[RuntimeTableSnapshot, ...],
) -> dict[tuple[str, ...], tuple[RuntimeTableSnapshot, ...]]:
    groups: dict[tuple[str, ...], list[RuntimeTableSnapshot]] = {}
    for table in tables:
        groups.setdefault(table.schema_key, []).append(table)
    return {schema: tuple(group) for schema, group in groups.items()}


def _cell_signature(
    value: str,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeCellSignature:
    text = value.strip()
    if not text:
        return RuntimeCellSignature(RuntimeCellValueKind.EMPTY, "")
    try:
        numeric = float(text)
    except ValueError:
        return RuntimeCellSignature(RuntimeCellValueKind.TEXT, text)
    if math.isnan(numeric):
        canonical = "nan"
    elif math.isinf(numeric):
        canonical = "inf" if numeric > 0 else "-inf"
    else:
        canonical = repr(round(numeric, policy.numeric_decimal_places))
    return RuntimeCellSignature(RuntimeCellValueKind.NUMBER, canonical)


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(value for value, count in counts.items() if count > 1)


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {
        ".bmp",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
    }
