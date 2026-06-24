"""Typed source metadata roles shared across source matching and runtime contexts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

ORIGINAL_SOURCE_METADATA_FIELD = "OpenHCSOriginalSourceMetadata"

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

    path = Path(value)
    if path.is_absolute():
        return str(path.resolve(strict=False))
    return value


def path_metadata_values_equivalent(left: str, right: str) -> bool:
    """Return whether two absolute path-like metadata values identify one path."""

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
