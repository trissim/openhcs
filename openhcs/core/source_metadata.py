"""Typed source metadata roles shared across source matching and runtime contexts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

ORIGINAL_SOURCE_METADATA_FIELD = "OpenHCSOriginalSourceMetadata"
SOURCE_FILTER_PATHS_METADATA_FIELD = "OpenHCSSourceFilterPaths"
SOURCE_PLANE_INDEX_FIELD = "source_plane_index"
SOURCE_PLANE_COUNT_FIELD = "source_plane_count"
SOURCE_VOXEL_SPACING_FIELD = "OpenHCSSourceVoxelSpacingZYX"

SourceMetadataScalar: TypeAlias = str | int | float | bool | None
SourceMetadataValue: TypeAlias = SourceMetadataScalar | Mapping[str, SourceMetadataScalar]
SourceMetadataMapping: TypeAlias = Mapping[str, SourceMetadataValue]
SourceMetadataIdentityValue: TypeAlias = (
    SourceMetadataScalar | tuple[tuple[str, SourceMetadataScalar], ...]
)
SourceMetadataIdentityItems: TypeAlias = tuple[
    tuple[str, SourceMetadataIdentityValue],
    ...,
]


def source_metadata_scalar(value: SourceMetadataScalar) -> SourceMetadataScalar:
    """Return the canonical scalar representation stored in source metadata."""

    if value is None:
        return None
    return canonical_path_metadata_value(str(value))


def canonical_path_metadata_value(value: str) -> str:
    """Normalize absolute path values while leaving ordinary labels unchanged."""

    return _cached_canonical_path_metadata_value(value)


@lru_cache(maxsize=65536)
def _cached_canonical_path_metadata_value(value: str) -> str:
    """Return the canonical absolute path spelling for path-like metadata."""

    path = Path(value)
    if path.is_absolute():
        return str(path.resolve(strict=False))
    return value


def path_metadata_values_equivalent(left: str, right: str) -> bool:
    """Return whether two absolute path-like metadata values identify one path."""

    return _cached_path_metadata_values_equivalent(left, right)


@lru_cache(maxsize=65536)
def _cached_path_metadata_values_equivalent(left: str, right: str) -> bool:
    """Return cached absolute-path equivalence for source metadata values."""

    left_path = Path(left)
    right_path = Path(right)
    return (
        left_path.is_absolute()
        and right_path.is_absolute()
        and left_path.resolve(strict=False) == right_path.resolve(strict=False)
    )


@dataclass(frozen=True, slots=True)
class OriginalSourceMetadata:
    """Source-literal metadata preserved separately from canonical axis fields."""

    fields: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(
        cls,
        metadata: Mapping[str, SourceMetadataScalar],
    ) -> "OriginalSourceMetadata":
        return cls(
            tuple(
                (str(key), str(source_metadata_scalar(value)))
                for key, value in metadata.items()
            )
        )

    @classmethod
    def from_reserved_value(
        cls,
        value: SourceMetadataValue,
        *,
        path: str,
    ) -> "OriginalSourceMetadata":
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"{ORIGINAL_SOURCE_METADATA_FIELD} for {path!r} must be a mapping, "
                f"got {type(value).__name__}."
            )
        return cls.from_mapping(value)

    def as_dict(self) -> dict[str, str]:
        return dict(self.fields)

    def merge_into(
        self,
        target: dict[str, SourceMetadataValue],
        *,
        path: str,
    ) -> None:
        existing = target.get(ORIGINAL_SOURCE_METADATA_FIELD)
        merged = (
            {}
            if existing is None
            else OriginalSourceMetadata.from_reserved_value(
                existing,
                path=path,
            ).as_dict()
        )
        for key, value in self.fields:
            existing_value = merged.get(key)
            if (
                existing_value is not None
                and existing_value != value
                and not path_metadata_values_equivalent(existing_value, value)
            ):
                raise RuntimeError(
                    f"Conflicting original source metadata field {key!r} "
                    f"while parsing source candidate {path!r}: "
                    f"{existing_value!r} != {value!r}."
                )
            merged[key] = value
        target[ORIGINAL_SOURCE_METADATA_FIELD] = merged


@dataclass(frozen=True, slots=True)
class SourceFilterPathMetadata:
    """Source path identities that file selector clauses may target."""

    paths: tuple[str, ...]

    @classmethod
    def from_paths(
        cls,
        paths: tuple[str, ...],
    ) -> "SourceFilterPathMetadata":
        return cls(tuple(dict.fromkeys(str(path) for path in paths if str(path))))

    @classmethod
    def from_reserved_value(
        cls,
        value: SourceMetadataValue,
        *,
        path: str,
    ) -> "SourceFilterPathMetadata":
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"{SOURCE_FILTER_PATHS_METADATA_FIELD} for {path!r} must be a mapping, "
                f"got {type(value).__name__}."
            )
        return cls.from_paths(
            tuple(str(path_value) for _key, path_value in sorted(value.items()))
        )

    def as_dict(self) -> dict[str, str]:
        return {str(index): path for index, path in enumerate(self.paths)}

    def merge_into(
        self,
        target: dict[str, SourceMetadataValue],
        *,
        path: str,
    ) -> None:
        existing = target.get(SOURCE_FILTER_PATHS_METADATA_FIELD)
        merged = (
            ()
            if existing is None
            else SourceFilterPathMetadata.from_reserved_value(
                existing,
                path=path,
            ).paths
        )
        target[SOURCE_FILTER_PATHS_METADATA_FIELD] = SourceFilterPathMetadata.from_paths(
            (*merged, *self.paths)
        ).as_dict()


@dataclass(frozen=True, slots=True)
class SourceVoxelSpacing:
    """Relative physical spacing for source pixels, ordered like arrays."""

    values_zyx: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(float(value) for value in self.values_zyx)
        if any(value <= 0 for value in normalized):
            raise ValueError("SourceVoxelSpacing values must be positive.")
        if len(normalized) not in (0, 2, 3):
            raise ValueError(
                "SourceVoxelSpacing requires 2-D or 3-D spacing, got "
                f"{len(normalized)} values."
            )
        object.__setattr__(self, "values_zyx", normalized)

    @property
    def has_values(self) -> bool:
        return bool(self.values_zyx)

    @classmethod
    def from_cellprofiler_xyz(
        cls,
        *,
        x: float,
        y: float,
        z: float,
    ) -> "SourceVoxelSpacing":
        """Return CellProfiler Image.spacing semantics from NamesAndTypes values."""
        raw_y = float(y)
        if raw_y <= 0:
            raise ValueError("CellProfiler relative pixel spacing in Y must be positive.")
        return cls((float(z) / raw_y, 1.0, float(x) / raw_y))

    @classmethod
    def from_source_metadata(
        cls,
        metadata: SourceMetadataMapping | None,
    ) -> "SourceVoxelSpacing":
        if metadata is None:
            return cls()
        value = metadata.get(SOURCE_VOXEL_SPACING_FIELD)
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            values = tuple(
                float(value[axis])
                for axis in ("z", "y", "x")
                if axis in value and value[axis] is not None
            )
        else:
            values = tuple(
                float(part)
                for part in str(value).split(",")
                if part.strip()
            )
        return cls(values)

    def as_source_metadata_value(self) -> str:
        return ",".join(f"{value:.17g}" for value in self.values_zyx)

    def merge_into(
        self,
        target: dict[str, SourceMetadataValue],
        *,
        path: str,
    ) -> None:
        if not self.has_values:
            return
        existing = SourceVoxelSpacing.from_source_metadata(target)
        if existing.has_values and existing != self:
            raise RuntimeError(
                f"Conflicting source voxel spacing while parsing source candidate "
                f"{path!r}: {existing.values_zyx!r} != {self.values_zyx!r}."
            )
        target[SOURCE_VOXEL_SPACING_FIELD] = self.as_source_metadata_value()

    def with_missing_from(
        self,
        fallback: "SourceVoxelSpacing",
    ) -> "SourceVoxelSpacing":
        if self.has_values:
            return self
        return fallback

    def spacing_for_ndim(self, ndim: int) -> tuple[float, ...]:
        if ndim <= 0:
            raise ValueError("SourceVoxelSpacing ndim must be positive.")
        if not self.has_values:
            return (1.0,) * ndim
        if ndim > len(self.values_zyx):
            raise ValueError(
                f"Cannot project {len(self.values_zyx)}-D source voxel spacing "
                f"onto {ndim}-D data."
            )
        return self.values_zyx[-ndim:]


@dataclass(kw_only=True)
class SourceVoxelSpacingFields:
    """Source-image voxel spacing carried by runtime payload metadata."""

    source_voxel_spacing: SourceVoxelSpacing = field(
        default_factory=SourceVoxelSpacing
    )

    def normalize_source_voxel_spacing_fields(self) -> None:
        if not isinstance(self.source_voxel_spacing, SourceVoxelSpacing):
            self.source_voxel_spacing = SourceVoxelSpacing(
                tuple(self.source_voxel_spacing)
            )


@dataclass(frozen=True, slots=True)
class SourceMetadataRoleView:
    """Role-aware metadata view for literal selectors versus component axes."""

    metadata: SourceMetadataMapping

    def scalar_items(self) -> tuple[tuple[str, SourceMetadataScalar], ...]:
        items: list[tuple[str, SourceMetadataScalar]] = []
        for key, value in self.metadata.items():
            if key == ORIGINAL_SOURCE_METADATA_FIELD or isinstance(value, Mapping):
                continue
            items.append((str(key), value))
        return tuple(items)

    def scalar_values(self) -> tuple[SourceMetadataScalar, ...]:
        return tuple(value for _key, value in self.scalar_items())

    def original_items(self) -> tuple[tuple[str, str], ...]:
        original_metadata = self.metadata.get(ORIGINAL_SOURCE_METADATA_FIELD)
        if original_metadata is None:
            return ()
        return OriginalSourceMetadata.from_reserved_value(
            original_metadata,
            path=ORIGINAL_SOURCE_METADATA_FIELD,
        ).fields

    def source_filter_paths(self) -> tuple[str, ...]:
        source_filter_paths = self.metadata.get(SOURCE_FILTER_PATHS_METADATA_FIELD)
        if source_filter_paths is None:
            return ()
        return SourceFilterPathMetadata.from_reserved_value(
            source_filter_paths,
            path=SOURCE_FILTER_PATHS_METADATA_FIELD,
        ).paths


@dataclass(frozen=True, slots=True)
class SourceMetadataIdentityProjection:
    """Stable hash projection for source metadata records."""

    metadata: SourceMetadataMapping

    def items(self) -> SourceMetadataIdentityItems:
        projected: list[tuple[str, SourceMetadataIdentityValue]] = []
        for key, value in self.metadata.items():
            if isinstance(value, Mapping):
                projected.append(
                    (
                        str(key),
                        tuple(
                            sorted(
                                (str(nested_key), nested_value)
                                for nested_key, nested_value in value.items()
                            )
                        ),
                    )
                )
                continue
            projected.append((str(key), value))
        return tuple(sorted(projected))
