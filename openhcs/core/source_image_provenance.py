"""Source-image provenance identity shared by runtime payload carriers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
    SourceMetadataMapping,
)
from openhcs.core.source_matching import (
    source_component_metadata_raw_value,
    source_component_metadata_value,
    source_metadata_value,
    with_source_component_metadata,
)


SourceComponentMetadata = SourceMetadataMapping
SourceImageProvenancePlanePathValues = tuple[str | None, ...]
SourceImageProvenancePlaneMetadataValues = tuple[SourceComponentMetadata | None, ...]
SourceProvenanceIdentity = tuple[str | None, tuple[tuple[str, str], ...] | None]
SourceImageProvenanceIdentity = tuple[
    SourceProvenanceIdentity,
    tuple[SourceProvenanceIdentity, ...],
    tuple[str, ...],
]
SourceProvenanceInitValues = tuple[
    str | None,
    SourceComponentMetadata | None,
    "SourceImageProvenancePlanes",
    tuple[str, ...],
]

SourceImageProvenanceAliasValueT = TypeVar("SourceImageProvenanceAliasValueT")
SourceImageProvenancePlaneValueT = TypeVar("SourceImageProvenancePlaneValueT")

if TYPE_CHECKING:
    from openhcs.microscopes.microscope_interfaces import FilenameParser


def normalize_source_path(source_path: str | None) -> str | None:
    """Normalize path-like source identities to their stored scalar value."""
    if source_path is None:
        return None
    return str(source_path)


def _normalize_component_metadata(
    metadata: SourceComponentMetadata | None,
) -> SourceComponentMetadata | None:
    if metadata is None:
        return None
    return MappingProxyType(dict(metadata))


def _component_metadata_identity(
    metadata: SourceComponentMetadata | None,
) -> tuple[tuple[str, str], ...] | None:
    if metadata is None:
        return None
    return tuple(sorted((str(key), repr(value)) for key, value in metadata.items()))


def _tuple_value(values: tuple[Any, ...], index: int) -> Any | None:
    if 0 <= index < len(values):
        return values[index]
    return None


@dataclass(slots=True)
class SourceImageIdentity:
    """Scalar source-image address from path and parsed component metadata."""

    path: str | None = None
    component_metadata: SourceComponentMetadata | None = None

    def __post_init__(self) -> None:
        self.path = normalize_source_path(self.path)
        self.component_metadata = _normalize_component_metadata(self.component_metadata)

    @property
    def addressable(self) -> bool:
        return self.path is not None or self.component_metadata is not None

    @property
    def identity(self) -> SourceProvenanceIdentity:
        return (
            self.path,
            _component_metadata_identity(self.component_metadata),
        )

    def with_missing_from(
        self,
        fallback: "SourceImageIdentity",
    ) -> "SourceImageIdentity":
        return type(self)(
            path=self.path if self.path is not None else fallback.path,
            component_metadata=self.component_metadata_with_missing_from(fallback),
        )

    def component_metadata_with_missing_from(
        self,
        fallback: "SourceImageIdentity",
    ) -> SourceComponentMetadata | None:
        """Fill missing OpenHCS component values from a fallback identity."""
        if self.component_metadata is None:
            return fallback.component_metadata
        if fallback.component_metadata is None:
            return self.component_metadata

        merged = dict(self.component_metadata)
        for component in AllComponents:
            if source_component_metadata_value(merged, component) is not None:
                continue
            fallback_value = source_component_metadata_raw_value(
                fallback.component_metadata,
                component,
            )
            if fallback_value is not None:
                merged[component.value] = fallback_value
        return MappingProxyType(merged)

    def with_parsed_path_components(
        self,
        parser: "FilenameParser",
    ) -> "SourceImageIdentity":
        """Fill missing component metadata from this identity's source path."""
        if self.path is None:
            return self
        parsed = parser.parse_filename(Path(self.path).name)
        if parsed is None:
            return self
        return self.with_missing_from(type(self)(component_metadata=parsed))


@dataclass(slots=True)
class SourceImageProvenancePlane:
    """One source-image provenance identity in a multi-plane payload."""

    source_identity: SourceImageIdentity = field(default_factory=SourceImageIdentity)

    @property
    def path(self) -> str | None:
        return self.source_identity.path

    @property
    def component_metadata(self) -> SourceComponentMetadata | None:
        return self.source_identity.component_metadata

    @property
    def addressable(self) -> bool:
        return self.source_identity.addressable

    @property
    def identity(self) -> SourceProvenanceIdentity:
        return self.source_identity.identity

    def with_missing_from(
        self,
        fallback: "SourceImageProvenancePlane",
    ) -> "SourceImageProvenancePlane":
        return type(self)(
            self.source_identity.with_missing_from(fallback.source_identity)
        )


@dataclass(slots=True)
class SourceImageProvenancePlaneRecord:
    """Serialized source-image provenance plane for runtime payload tables."""

    path: str | None = None
    component_metadata: SourceComponentMetadata | None = None

    def plane(self) -> SourceImageProvenancePlane:
        return SourceImageProvenancePlane(
            SourceImageIdentity(self.path, self.component_metadata)
        )


SourceImageProvenancePlaneRecords = Sequence[SourceImageProvenancePlaneRecord]


@dataclass(slots=True)
class SourceImageProvenancePlanes:
    """Nominal carrier for source-image provenance across payload planes."""

    planes: tuple[SourceImageProvenancePlane, ...] = ()

    def __post_init__(self) -> None:
        self.planes = tuple(
            plane
            if isinstance(plane, SourceImageProvenancePlane)
            else SourceImageProvenancePlane()
            for plane in self.planes
        )

    @classmethod
    def from_components(
        cls,
        *,
        paths: SourceImageProvenancePlanePathValues = (),
        component_metadata: SourceImageProvenancePlaneMetadataValues = (),
    ) -> "SourceImageProvenancePlanes":
        count = max(len(paths), len(component_metadata), 0)
        return cls(
            tuple(
                SourceImageProvenancePlane(
                    SourceImageIdentity(
                        path=_tuple_value(paths, index),
                        component_metadata=_tuple_value(component_metadata, index),
                    )
                )
                for index in range(count)
            )
        )

    @classmethod
    def from_records(
        cls,
        records: SourceImageProvenancePlaneRecords | "SourceImageProvenancePlanes" | None,
    ) -> "SourceImageProvenancePlanes":
        if records is None:
            return cls()
        if isinstance(records, cls):
            return records
        if isinstance(records, Sequence) and not isinstance(
            records,
            (str, bytes, bytearray),
        ):
            return cls(
                tuple(cls.plane_from_record(record) for record in records)
            )
        raise TypeError(
            "source_image_provenance_planes must be a sequence of mappings, "
            f"got {type(records).__name__}."
        )

    @staticmethod
    def plane_from_record(
        record: SourceImageProvenancePlaneRecord,
    ) -> SourceImageProvenancePlane:
        return record.plane()

    @property
    def records(self) -> tuple[SourceImageProvenancePlaneRecord, ...]:
        return tuple(
            SourceImageProvenancePlaneRecord(
                plane.path,
                plane.component_metadata,
            )
            for plane in self.planes
        )

    @property
    def paths(self) -> SourceImageProvenancePlanePathValues:
        return self._plane_values(lambda plane: plane.path)

    @property
    def component_metadata(self) -> SourceImageProvenancePlaneMetadataValues:
        return self._plane_values(lambda plane: plane.component_metadata)

    @property
    def count(self) -> int:
        return len(self.planes)

    @property
    def plane_count_sources(self) -> tuple[int, ...]:
        path_count = len(self.paths)
        component_metadata_count = len(self.component_metadata)
        return tuple(
            count
            for count in (path_count, component_metadata_count)
            if count > 0
        )

    @property
    def has_values(self) -> bool:
        return bool(self.planes)

    @property
    def identity(self) -> tuple[SourceProvenanceIdentity, ...]:
        return self._plane_values(lambda plane: plane.identity)

    def _plane_values(
        self,
        select: Callable[[SourceImageProvenancePlane], SourceImageProvenancePlaneValueT],
    ) -> tuple[SourceImageProvenancePlaneValueT, ...]:
        return tuple(select(plane) for plane in self.planes)

    def plane(self, index: int) -> SourceImageProvenancePlane:
        if 0 <= index < len(self.planes):
            return self.planes[index]
        return SourceImageProvenancePlane()

    def select(self, indices: Sequence[int]) -> "SourceImageProvenancePlanes":
        """Return provenance planes selected by source-plane index."""
        return type(self)(
            tuple(self.plane(int(index)) for index in indices)
        )

    def with_missing_from(
        self,
        fallback: "SourceImageProvenancePlanes",
    ) -> "SourceImageProvenancePlanes":
        if not self.planes:
            return fallback
        if not fallback.planes or len(self.planes) != len(fallback.planes):
            return self
        return type(self)(
            tuple(
                plane.with_missing_from(fallback_plane)
                for plane, fallback_plane in zip(self.planes, fallback.planes)
            )
        )


class SourceImageProvenance:
    """Source-image identity carried by image and object-label payloads."""

    __slots__ = (
        "source_identity",
        "source_image_provenance_planes",
        "source_image_names",
    )

    def __init__(
        self,
        source_path: str | None = None,
        source_component_metadata: SourceComponentMetadata | None = None,
        source_image_provenance_planes: SourceImageProvenancePlanes | None = None,
        source_image_names: tuple[str, ...] = (),
    ) -> None:
        self.source_identity = SourceImageIdentity(
            source_path,
            source_component_metadata
        )
        self.source_image_provenance_planes = (
            source_image_provenance_planes or SourceImageProvenancePlanes()
        )
        self.source_image_names = tuple(str(name) for name in source_image_names)

    @classmethod
    def from_init_values(
        cls,
        values: SourceProvenanceInitValues,
    ) -> "SourceImageProvenance":
        return cls(*values)

    @property
    def source_path(self) -> str | None:
        return self.source_identity.path

    @property
    def source_component_metadata(self) -> SourceComponentMetadata | None:
        return self.source_identity.component_metadata

    @property
    def has_values(self) -> bool:
        return any(
            (
                self.source_identity.addressable,
                self.source_image_provenance_planes.has_values,
                bool(self.source_image_names),
            )
        )

    @property
    def source_plane_count(self) -> int:
        return self.source_image_provenance_planes.count

    @property
    def plane_count_sources(self) -> tuple[int, ...]:
        return self.source_image_provenance_planes.plane_count_sources

    @property
    def addressable(self) -> bool:
        return self.source_identity.addressable

    @property
    def scalar_source_identity(self) -> SourceImageIdentity:
        """Return scalar source identity without collapsing multi-plane provenance."""
        if self.source_image_provenance_planes.count == 1:
            return self.source_identity.with_missing_from(
                self.source_image_provenance_planes.plane(0).source_identity
            )
        return self.source_identity

    def for_source_plane(self, plane_index: int) -> "SourceImageProvenance":
        provenance_plane = self.source_image_provenance_planes.plane(plane_index)
        source_identity = provenance_plane.source_identity.with_missing_from(
            self.source_identity
        )
        return type(self)(
            source_path=source_identity.path,
            source_component_metadata=source_identity.component_metadata,
            source_image_names=self.source_image_names_for_plane(plane_index),
        )

    def for_source_planes(self, plane_indices: Sequence[int]) -> "SourceImageProvenance":
        """Return provenance represented by a grouped set of source planes."""
        normalized_indices = tuple(int(index) for index in plane_indices)
        if len(normalized_indices) == 1:
            return self.for_source_plane(normalized_indices[0])
        return type(self)(
            source_path=self.source_identity.path,
            source_component_metadata=self.source_identity.component_metadata,
            source_image_provenance_planes=(
                self.source_image_provenance_planes.select(normalized_indices)
                .with_missing_from(
                    SourceImageProvenancePlanes.from_components(
                        paths=(self.source_identity.path,) * len(normalized_indices),
                        component_metadata=(
                            self.source_identity.component_metadata,
                        ) * len(normalized_indices),
                    )
                )
            ),
            source_image_names=self.source_image_names_for_planes(normalized_indices),
        )

    def source_image_names_for_plane(self, plane_index: int) -> tuple[str, ...]:
        """Return source-image aliases represented by a selected provenance plane."""
        if len(self.source_image_names) <= plane_index:
            return self.source_image_names
        return (self.source_image_names[plane_index],)

    def source_image_names_for_planes(
        self,
        plane_indices: Sequence[int],
    ) -> tuple[str, ...]:
        """Return aliases represented by selected provenance planes."""
        normalized_indices = tuple(int(index) for index in plane_indices)
        if not normalized_indices:
            return ()
        if len(self.source_image_names) <= max(normalized_indices):
            return self.source_image_names
        return tuple(self.source_image_names[index] for index in normalized_indices)

    def identity(self) -> SourceProvenanceIdentity:
        return self.source_identity.identity

    @property
    def equality_identity(self) -> SourceImageProvenanceIdentity:
        """Return every semantic field that defines this provenance value."""
        return (
            self.source_identity.identity,
            self.source_image_provenance_planes.identity,
            self.source_image_names,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SourceImageProvenance):
            return NotImplemented
        return self.equality_identity == other.equality_identity

    def with_missing_from(self, fallback: "SourceImageProvenance") -> Self:
        source_identity = self.source_identity.with_missing_from(
            fallback.source_identity
        )
        return type(self)(
            source_path=source_identity.path,
            source_component_metadata=source_identity.component_metadata,
            source_image_provenance_planes=(
                self.source_image_provenance_planes.with_missing_from(
                    fallback.source_image_provenance_planes
                )
            ),
            source_image_names=(
                self.source_image_names
                if self.source_image_names
                else fallback.source_image_names
            ),
        )

    def with_source_path(self, value: str | None) -> Self:
        """Return provenance with only the scalar source path replaced."""
        return type(self)(
            source_path=value,
            source_component_metadata=self.source_identity.component_metadata,
            source_image_provenance_planes=self.source_image_provenance_planes,
            source_image_names=self.source_image_names,
        )

    def with_source_component_metadata(
        self,
        value: SourceComponentMetadata | None,
    ) -> Self:
        """Return provenance with only the scalar component metadata replaced."""
        return type(self)(
            source_path=self.source_identity.path,
            source_component_metadata=value,
            source_image_provenance_planes=self.source_image_provenance_planes,
            source_image_names=self.source_image_names,
        )

    def with_source_image_provenance_planes(
        self,
        value: SourceImageProvenancePlanes,
    ) -> Self:
        """Return provenance with only the per-plane identities replaced."""
        return type(self)(
            source_path=self.source_identity.path,
            source_component_metadata=self.source_identity.component_metadata,
            source_image_provenance_planes=value,
            source_image_names=self.source_image_names,
        )

    def with_source_image_names(self, value: tuple[str, ...]) -> Self:
        """Return provenance with only source-image aliases replaced."""
        return type(self)(
            source_path=self.source_identity.path,
            source_component_metadata=self.source_identity.component_metadata,
            source_image_provenance_planes=self.source_image_provenance_planes,
            source_image_names=value,
        )

    def with_common_scalar_identity_from_planes(self) -> Self:
        """Return provenance whose scalar identity is common to all planes."""
        if not self.source_image_provenance_planes.has_values:
            return self
        return type(self)(
            source_path=common_source_path(
                self.source_image_provenance_planes.paths
            ),
            source_component_metadata=common_source_component_metadata(
                self.source_image_provenance_planes.component_metadata
            ),
            source_image_provenance_planes=self.source_image_provenance_planes,
            source_image_names=self.source_image_names,
        )

    @property
    def source_component_identity(self) -> tuple[tuple[str, str], ...] | None:
        return self.source_identity.identity[1]


@dataclass(frozen=True, slots=True)
class SourcePlaneIndexedMetadata:
    """Source metadata for one indexed plane inside a single source image."""

    scalar_metadata: SourceComponentMetadata
    scalar_plane_index: int
    source_plane_count: int

    @staticmethod
    def projected_component() -> AllComponents:
        """Return the component represented by indexed source-plane metadata."""
        return AllComponents.Z_INDEX

    @classmethod
    def from_metadata(
        cls,
        metadata: SourceComponentMetadata,
        *,
        expected_plane_count: int,
    ) -> "SourcePlaneIndexedMetadata | None":
        index_value = source_metadata_value(metadata, SOURCE_PLANE_INDEX_FIELD)
        count_value = source_metadata_value(metadata, SOURCE_PLANE_COUNT_FIELD)
        if index_value is None and count_value is None:
            return None
        if index_value is None or count_value is None:
            raise ValueError(
                "Source-plane metadata must carry both "
                f"{SOURCE_PLANE_INDEX_FIELD!r} and {SOURCE_PLANE_COUNT_FIELD!r}."
            )
        indexed_metadata = cls(
            scalar_metadata=metadata,
            scalar_plane_index=cls.parse_source_plane_value(
                index_value,
                field_name=SOURCE_PLANE_INDEX_FIELD,
            ),
            source_plane_count=cls.parse_source_plane_value(
                count_value,
                field_name=SOURCE_PLANE_COUNT_FIELD,
            ),
        )
        if indexed_metadata.source_plane_count != expected_plane_count:
            return None
        indexed_metadata.validate_scalar_plane_index()
        return indexed_metadata

    @classmethod
    def from_scalar_origin(
        cls,
        metadata: SourceComponentMetadata,
        *,
        source_plane_count: int,
    ) -> "SourcePlaneIndexedMetadata | None":
        index_value = source_metadata_value(metadata, SOURCE_PLANE_INDEX_FIELD)
        count_value = source_metadata_value(metadata, SOURCE_PLANE_COUNT_FIELD)
        if index_value is None and count_value is None:
            return None
        if (index_value is None) != (count_value is None):
            raise ValueError(
                "Source-plane metadata must carry both "
                f"{SOURCE_PLANE_INDEX_FIELD!r} and {SOURCE_PLANE_COUNT_FIELD!r}."
            )
        scalar_plane_index = (
            0
            if index_value is None
            else cls.parse_source_plane_value(
                index_value,
                field_name=SOURCE_PLANE_INDEX_FIELD,
            )
        )
        declared_count = (
            source_plane_count
            if count_value is None
            else cls.parse_source_plane_value(
                count_value,
                field_name=SOURCE_PLANE_COUNT_FIELD,
            )
        )
        if declared_count != source_plane_count:
            raise ValueError(
                "Volumetric source metadata plane count disagrees with image "
                f"shape: {declared_count} != {source_plane_count}."
            )
        indexed_metadata = cls(
            scalar_metadata=metadata,
            scalar_plane_index=scalar_plane_index,
            source_plane_count=source_plane_count,
        )
        indexed_metadata.validate_scalar_plane_index()
        return indexed_metadata

    @classmethod
    def from_declared_runtime_stack_origin(
        cls,
        metadata: SourceComponentMetadata,
        *,
        source_plane_count: int,
    ) -> "SourcePlaneIndexedMetadata | None":
        indexed_metadata = cls.from_scalar_origin(
            metadata,
            source_plane_count=source_plane_count,
        )
        if indexed_metadata is not None:
            return indexed_metadata
        scalar_axis_value = source_component_metadata_value(
            metadata,
            cls.projected_component(),
        )
        if scalar_axis_value is None:
            return None
        indexed_metadata = cls(
            scalar_metadata=metadata,
            scalar_plane_index=0,
            source_plane_count=source_plane_count,
        )
        indexed_metadata.validate_scalar_plane_index()
        return indexed_metadata

    @staticmethod
    def parse_source_plane_value(value: str, *, field_name: str) -> int:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Source-plane metadata field {field_name!r} must be numeric, "
                f"got {value!r}."
            ) from exc

    def validate_scalar_plane_index(self) -> None:
        if not 0 <= self.scalar_plane_index < self.source_plane_count:
            raise ValueError(
                "Source-plane metadata index is outside the declared plane "
                f"domain: {self.scalar_plane_index} for "
                f"{self.source_plane_count} planes."
            )

    def component_metadata(self) -> tuple[SourceComponentMetadata, ...]:
        return tuple(
            self.component_metadata_for_plane(plane_index)
            for plane_index in range(self.source_plane_count)
        )

    def component_metadata_for_plane(
        self,
        plane_index: int,
    ) -> SourceComponentMetadata:
        metadata = {
            **dict(self.scalar_metadata),
            SOURCE_PLANE_INDEX_FIELD: str(plane_index),
            SOURCE_PLANE_COUNT_FIELD: str(self.source_plane_count),
        }
        return MappingProxyType(
            with_source_component_metadata(
                metadata,
                AllComponents.Z_INDEX,
                self.z_index_for_plane(plane_index),
            )
        )

    def z_index_for_plane(self, plane_index: int) -> int:
        scalar_z_index = source_component_metadata_value(
            self.scalar_metadata,
            AllComponents.Z_INDEX,
        )
        if scalar_z_index is None:
            return plane_index + 1
        try:
            first_z_index = int(scalar_z_index) - self.scalar_plane_index
        except ValueError as exc:
            raise ValueError(
                "Source-plane metadata z_index must be numeric when expanding "
                f"indexed plane provenance, got {scalar_z_index!r}."
            ) from exc
        return first_z_index + plane_index

    def common_component_metadata(self) -> SourceComponentMetadata | None:
        return common_source_component_metadata(self.component_metadata())


@dataclass(frozen=True, slots=True)
class SourcePlaneIndexedProvenanceExpansion:
    """Expand scalar indexed source metadata into per-plane source provenance."""

    provenance: SourceImageProvenance
    expected_plane_count: int | None

    def expanded(self) -> SourceImageProvenance:
        if self.expected_plane_count is None:
            return self.provenance
        if self.provenance.source_image_provenance_planes.has_values:
            return self.provenance
        metadata = self.provenance.source_component_metadata
        if metadata is None:
            return self.provenance
        indexed_metadata = SourcePlaneIndexedMetadata.from_metadata(
            metadata,
            expected_plane_count=self.expected_plane_count,
        )
        if indexed_metadata is None:
            return self.provenance
        return SourceImageProvenance(
            source_path=self.provenance.source_path,
            source_component_metadata=indexed_metadata.common_component_metadata(),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(self.provenance.source_path,) * self.expected_plane_count,
                component_metadata=indexed_metadata.component_metadata(),
            ),
            source_image_names=self.provenance.source_image_names,
        )


def common_source_component_metadata(
    metadata_by_plane: Sequence[SourceComponentMetadata | None],
) -> SourceComponentMetadata | None:
    """Return source metadata values shared by every source plane."""
    metadata_values = tuple(
        dict(metadata) for metadata in metadata_by_plane if metadata is not None
    )
    if len(metadata_values) != len(metadata_by_plane):
        return None
    if not metadata_values:
        return None
    common_keys = set(metadata_values[0])
    for metadata in metadata_values[1:]:
        common_keys.intersection_update(metadata)
    common_metadata = {
        key: metadata_values[0][key]
        for key in common_keys
        if all(
            metadata[key] == metadata_values[0][key]
            for metadata in metadata_values
        )
    }
    if not common_metadata:
        return None
    return MappingProxyType(common_metadata)


def common_source_path(paths: Sequence[str | None]) -> str | None:
    """Return the source path shared by every plane, if one exists."""
    path_values = tuple(path for path in paths if path is not None)
    if len(path_values) != len(paths) or not path_values:
        return None
    first = path_values[0]
    if all(path == first for path in path_values):
        return first
    return None


@dataclass(frozen=True, slots=True)
class SourceImageProvenanceRequirement(ABC, metaclass=AutoRegisterMeta):
    """Nominal source-provenance invariant with policy-selected failure behavior."""

    __registry_key__ = "requirement_name"
    __skip_if_no_key__ = True
    requirement_name: ClassVar[str | None] = None

    provenance: SourceImageProvenance
    label_name: str

    @property
    @abstractmethod
    def satisfied(self) -> bool:
        """Return whether this source-provenance invariant is satisfied."""

    @abstractmethod
    def error(self) -> ValueError:
        """Return the fail-loud error for an unsatisfied invariant."""

    def validate(self) -> None:
        SourceImageProvenanceRequirementPolicy.for_requirement(self).validate(self)


class SourceImageProvenanceRequirementPolicy(ABC, metaclass=AutoRegisterMeta):
    """Registered satisfied/unsatisfied handling for provenance requirements."""

    __registry_key__ = "satisfied"
    __skip_if_no_key__ = True
    satisfied: ClassVar[bool | None] = None

    @classmethod
    def for_requirement(
        cls,
        requirement: SourceImageProvenanceRequirement,
    ) -> "SourceImageProvenanceRequirementPolicy":
        return cls.__registry__[requirement.satisfied]()

    def validate(self, requirement: SourceImageProvenanceRequirement) -> None:
        """Validate one source-provenance requirement."""
        return None


class SatisfiedSourceImageProvenanceRequirementPolicy(
    SourceImageProvenanceRequirementPolicy
):
    """Accept source-provenance requirements that are already satisfied."""

    satisfied = True


class UnsatisfiedSourceImageProvenanceRequirementPolicy(
    SourceImageProvenanceRequirementPolicy
):
    """Raise the requirement-specific error for failed provenance invariants."""

    satisfied = False

    def validate(self, requirement: SourceImageProvenanceRequirement) -> None:
        raise requirement.error()


@dataclass(frozen=True, slots=True)
class SourceImageProvenancePlaneCountRequirement(SourceImageProvenanceRequirement):
    """Require per-plane source identities to match source-aligned label planes."""

    requirement_name = "plane_count"
    expected_count: int

    @property
    def satisfied(self) -> bool:
        return self.provenance.source_plane_count == self.expected_count

    def error(self) -> ValueError:
        return ValueError(
            f"Object-label artifact {self.label_name!r} has {self.expected_count} "
            "source-aligned label planes but "
            f"{self.provenance.source_plane_count} per-plane source identities "
            "after source-image contextualization."
        )


@dataclass(frozen=True, slots=True)
class SourceImageProvenanceAddressRequirement(SourceImageProvenanceRequirement):
    """Require one plane source identity to be addressable."""

    requirement_name = "address"
    plane_index: int

    @property
    def satisfied(self) -> bool:
        return self.provenance.addressable

    def error(self) -> ValueError:
        return ValueError(
            f"Object-label artifact {self.label_name!r} plane {self.plane_index} "
            "has neither source_path nor source_component_metadata after "
            "source-image contextualization."
        )


@dataclass(frozen=True, slots=True)
class SourceImageProvenanceAlias(Generic[SourceImageProvenanceAliasValueT]):
    """Descriptor for scalar aliases backed by the source-provenance carrier."""

    getter: Callable[[SourceImageProvenance], SourceImageProvenanceAliasValueT]
    setter: Callable[
        [SourceImageProvenance, SourceImageProvenanceAliasValueT],
        SourceImageProvenance,
    ]

    def __get__(
        self,
        instance: "SourceImageProvenanceFields | None",
        _owner: type["SourceImageProvenanceFields"],
    ) -> SourceImageProvenanceAliasValueT | Self:
        if instance is None:
            return self
        provenance = instance.source_provenance
        return self.getter(provenance)

    def __set__(
        self,
        instance: "SourceImageProvenanceFields",
        value: SourceImageProvenanceAliasValueT,
    ) -> None:
        instance.source_provenance = self.setter(instance.source_provenance, value)


@dataclass(kw_only=True)
class SourceImageProvenanceFields:
    """Source-image provenance fields shared by image and object-label carriers."""

    source_provenance: SourceImageProvenance = field(
        default_factory=SourceImageProvenance
    )
    source_path: InitVar[str | None] = None
    source_component_metadata: InitVar[SourceComponentMetadata | None] = None
    source_image_provenance_planes: InitVar[SourceImageProvenancePlanes | None] = None
    source_image_names: InitVar[tuple[str, ...]] = ()

    def absorb_explicit_source_provenance(
        self,
        explicit: SourceImageProvenance,
    ) -> None:
        self.source_provenance = explicit.with_missing_from(self.source_provenance)

    def normalize_source_provenance_fields(self) -> None:
        self.source_provenance = SourceImageProvenance(
            self.source_provenance.source_path,
            self.source_provenance.source_component_metadata,
            self.source_provenance.source_image_provenance_planes,
            self.source_provenance.source_image_names,
        )


SourceImageProvenanceFields.source_path = SourceImageProvenanceAlias(
    lambda provenance: provenance.source_path,
    SourceImageProvenance.with_source_path,
)
SourceImageProvenanceFields.source_component_metadata = SourceImageProvenanceAlias(
    lambda provenance: provenance.source_component_metadata,
    SourceImageProvenance.with_source_component_metadata,
)
SourceImageProvenanceFields.source_image_provenance_planes = SourceImageProvenanceAlias(
    lambda provenance: provenance.source_image_provenance_planes,
    SourceImageProvenance.with_source_image_provenance_planes,
)
SourceImageProvenanceFields.source_image_names = SourceImageProvenanceAlias(
    lambda provenance: provenance.source_image_names,
    SourceImageProvenance.with_source_image_names,
)


@dataclass(frozen=True, slots=True)
class VariableComponentAxisProjection:
    """Project scalar source identity over variable component runtime planes."""

    axes: frozenset[str]

    @classmethod
    def empty(cls) -> "VariableComponentAxisProjection":
        return cls(frozenset())

    @classmethod
    def from_axes(
        cls,
        axes: Iterable[str],
    ) -> "VariableComponentAxisProjection":
        return cls(frozenset(axes))

    @property
    def is_empty(self) -> bool:
        return not self.axes

    def ordered_axes(self) -> tuple[str, ...]:
        component_order = AllComponents.ordered_names()
        ordered_known = tuple(axis for axis in component_order if axis in self.axes)
        ordered_unknown = tuple(sorted(self.axes.difference(ordered_known)))
        return ordered_known + ordered_unknown

    def project_component_metadata(
        self,
        metadata: SourceComponentMetadata,
        plane_indices: Sequence[int],
    ) -> SourceComponentMetadata:
        axes = self.ordered_axes()
        if len(axes) != len(plane_indices):
            raise ValueError(
                "Variable component axis projection cannot map "
                f"{len(plane_indices)} runtime coordinate(s) onto declared axes "
                f"{axes!r}."
            )
        projected_metadata = metadata
        for axis, plane_index in zip(axes, plane_indices, strict=True):
            component = self.component_for_axis(axis)
            projected_metadata = with_source_component_metadata(
                projected_metadata,
                component,
                self.projected_axis_value(
                    projected_metadata,
                    component,
                    plane_index,
                ),
            )
        return projected_metadata

    def provenance_planes(
        self,
        *,
        source_path: str | None,
        source_component_metadata: SourceComponentMetadata | None,
        plane_count: int,
    ) -> SourceImageProvenancePlanes:
        if source_component_metadata is None:
            return SourceImageProvenancePlanes()
        if plane_count <= 0:
            return SourceImageProvenancePlanes()
        if len(self.ordered_axes()) != 1:
            return SourceImageProvenancePlanes()
        return SourceImageProvenancePlanes.from_components(
            paths=(source_path,) * plane_count,
            component_metadata=tuple(
                self.project_component_metadata(
                    source_component_metadata,
                    (plane_index,),
                )
                for plane_index in range(plane_count)
            ),
        )

    @staticmethod
    def component_for_axis(axis: str) -> AllComponents:
        component = AllComponents.from_value(axis)
        if component is None:
            raise ValueError(
                "Variable component axis is not an OpenHCS component: "
                f"{axis!r}."
            )
        return component

    @staticmethod
    def projected_axis_value(
        metadata: SourceComponentMetadata,
        component: AllComponents,
        plane_index: int,
    ) -> int:
        current = source_component_metadata_raw_value(metadata, component)
        if current is None:
            raise ValueError(
                "Variable component axis projection requires scalar "
                f"{component.value!r} metadata."
            )
        if isinstance(current, bool):
            raise ValueError(
                "Variable component axis "
                f"{component.value!r} must be numeric, got bool."
            )
        try:
            base_value = int(current)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Variable component axis "
                f"{component.value!r} must be numeric, got {current!r}."
            ) from exc
        return base_value + plane_index
