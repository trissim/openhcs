"""Table and image comparison helpers for runtime equivalence."""

from __future__ import annotations

from collections import Counter

import numpy as np

from openhcs.core.equivalence.cells import (
    RuntimeCellSignature,
    runtime_cell_signature_counters_equivalent,
)
from openhcs.core.equivalence.images import RuntimeImageSnapshot
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    normalize_runtime_identifier,
)
from openhcs.core.equivalence.report import (
    RuntimeEquivalenceDifference,
    RuntimeEquivalenceDifferenceKind,
)
from openhcs.core.equivalence.tables import RuntimeTableSnapshot


def runtime_table_differences(
    reference_tables: tuple[RuntimeTableSnapshot, ...],
    candidate_tables: tuple[RuntimeTableSnapshot, ...],
    policy: RuntimeEquivalencePolicy,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    """Return semantic table differences between two output snapshots."""
    differences: list[RuntimeEquivalenceDifference] = []
    reference_tables = _comparable_table_snapshots(reference_tables)
    candidate_tables = _comparable_table_snapshots(candidate_tables)
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


def runtime_image_differences(
    reference_images: tuple[RuntimeImageSnapshot, ...],
    candidate_images: tuple[RuntimeImageSnapshot, ...],
    policy: RuntimeEquivalencePolicy,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    """Return semantic image differences between two output snapshots."""
    differences: list[RuntimeEquivalenceDifference] = []
    if len(reference_images) != len(candidate_images):
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.IMAGE_COUNT,
                f"image output count differs: reference={len(reference_images)}, "
                f"candidate={len(candidate_images)}",
            )
        )
    if not _image_snapshots_equivalent(reference_images, candidate_images, policy):
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.IMAGE_CONTENT,
                "image output content differs",
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

    if not _table_snapshots_equivalent(reference_group, candidate_group, policy):
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.TABLE_CONTENT,
                f"table schema {schema!r} values differ",
            )
        )
    return tuple(differences)


def _table_snapshots_equivalent(
    reference_tables: tuple[RuntimeTableSnapshot, ...],
    candidate_tables: tuple[RuntimeTableSnapshot, ...],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    unmatched = list(candidate_tables)
    for reference_table in reference_tables:
        match_index = next(
            (
                index
                for index, candidate_table in enumerate(unmatched)
                if reference_table.content_key(policy)
                == candidate_table.content_key(policy)
            ),
            None,
        )
        if match_index is None:
            match_index = next(
                (
                    index
                    for index, candidate_table in enumerate(unmatched)
                    if _table_rows_equivalent(
                        reference_table,
                        candidate_table,
                        policy,
                    )
                ),
                None,
            )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def _table_rows_equivalent(
    reference_table: RuntimeTableSnapshot,
    candidate_table: RuntimeTableSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    reference_rows = reference_table.row_signatures(policy)
    unmatched = list(candidate_table.row_signatures(policy))
    if len(reference_rows) != len(unmatched):
        return False
    for reference_row in reference_rows:
        match_index = next(
            (
                index
                for index, candidate_row in enumerate(unmatched)
                if _table_row_equivalent(reference_row, candidate_row, policy)
            ),
            None,
        )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def _table_row_equivalent(
    reference_row: tuple[RuntimeCellSignature, ...],
    candidate_row: tuple[RuntimeCellSignature, ...],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    return len(reference_row) == len(candidate_row) and all(
        runtime_cell_signature_counters_equivalent(
            Counter((reference_cell,)),
            Counter((candidate_cell,)),
            policy,
        )
        for reference_cell, candidate_cell in zip(
            reference_row,
            candidate_row,
            strict=True,
        )
    )


def _tables_by_schema(
    tables: tuple[RuntimeTableSnapshot, ...],
) -> dict[tuple[str, ...], tuple[RuntimeTableSnapshot, ...]]:
    groups: dict[tuple[str, ...], list[RuntimeTableSnapshot]] = {}
    for table in tables:
        groups.setdefault(table.schema_key, []).append(table)
    return {schema: tuple(group) for schema, group in groups.items()}


def _comparable_table_snapshots(
    tables: tuple[RuntimeTableSnapshot, ...],
) -> tuple[RuntimeTableSnapshot, ...]:
    return tuple(table for table in tables if not _is_metadata_table_snapshot(table))


def _is_metadata_table_snapshot(table: RuntimeTableSnapshot) -> bool:
    if normalize_runtime_identifier(table.path.stem) != "experiment":
        return False
    normalized_header = frozenset(
        normalize_runtime_identifier(column) for column in table.header
    )
    return normalized_header == frozenset(("key", "value"))


def _image_snapshots_equivalent(
    reference_images: tuple[RuntimeImageSnapshot, ...],
    candidate_images: tuple[RuntimeImageSnapshot, ...],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if len(reference_images) != len(candidate_images):
        return False

    unmatched = list(candidate_images)
    for reference_image in reference_images:
        match_index = _matching_image_index(reference_image, unmatched, policy)
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def _matching_image_index(
    reference_image: RuntimeImageSnapshot,
    candidate_images: list[RuntimeImageSnapshot],
    policy: RuntimeEquivalencePolicy,
) -> int | None:
    for index, candidate_image in enumerate(candidate_images):
        if reference_image.content_key(policy) == candidate_image.content_key(policy):
            return index
    for index, candidate_image in enumerate(candidate_images):
        if _image_pixels_equivalent(reference_image, candidate_image, policy):
            return index
    return None


def _image_pixels_equivalent(
    reference_image: RuntimeImageSnapshot,
    candidate_image: RuntimeImageSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if not policy.compare_image_pixels:
        return False
    comparable_pixels = _comparable_image_pixels(reference_image, candidate_image)
    if comparable_pixels is None:
        return False
    reference_pixels, candidate_pixels = comparable_pixels
    if (
        policy.image_abs_tolerance == 0
        and policy.image_rel_tolerance == 0
        and policy.image_max_different_fraction == 0
    ):
        return np.array_equal(reference_pixels, candidate_pixels, equal_nan=True)
    if reference_pixels.size == 0:
        return True
    if not (
        np.issubdtype(reference_pixels.dtype, np.number)
        and np.issubdtype(candidate_pixels.dtype, np.number)
    ):
        return False

    close_pixels = np.isclose(
        reference_pixels.astype(np.float64, copy=False),
        candidate_pixels.astype(np.float64, copy=False),
        rtol=policy.image_rel_tolerance,
        atol=policy.image_abs_tolerance,
        equal_nan=True,
    )
    different_fraction = 1.0 - (float(np.count_nonzero(close_pixels)) / close_pixels.size)
    return different_fraction <= policy.image_max_different_fraction


def _comparable_image_pixels(
    reference_image: RuntimeImageSnapshot,
    candidate_image: RuntimeImageSnapshot,
) -> tuple[np.ndarray, np.ndarray] | None:
    reference_pixels = np.asarray(reference_image.pixel_data)
    candidate_pixels = np.asarray(candidate_image.pixel_data)
    if reference_pixels.shape == candidate_pixels.shape:
        return reference_pixels, candidate_pixels
    reference_normalized = _grayscale_equivalent_pixels(reference_pixels)
    candidate_normalized = _grayscale_equivalent_pixels(candidate_pixels)
    if reference_normalized.shape != candidate_normalized.shape:
        return None
    return reference_normalized, candidate_normalized


def _grayscale_equivalent_pixels(pixels: np.ndarray) -> np.ndarray:
    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        color = pixels[..., :3]
        if np.all(color == color[..., :1]):
            return color[..., 0]
    return pixels
