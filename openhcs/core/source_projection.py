"""Typed source projection authority for OpenHCS workspaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, cast
from urllib.parse import quote

from polystore.virtual_workspace import SourcePixelRef

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactType, ImageArtifactType
from openhcs.core.source_bindings import (
    SOURCE_BINDING_ALIAS_METADATA_FIELD,
    NamedSourceBinding,
    SourceProjectionRole,
)
from openhcs.core.source_metadata import (
    SourceComponentProjectionStrategy,
    SourceMetadataIdentityProjection,
    SourceMetadataRoleView,
    SourceMetadataMapping,
    SourceMetadataValue,
    source_metadata_dict,
    source_metadata_scalar,
)
from openhcs.core.source_matching import (
    source_component_metadata_values,
    source_metadata_component,
    source_metadata_values_equal,
    with_original_source_metadata,
    with_source_component_metadata,
)


class SourceDatasetConflictError(ValueError):
    """Raised when store declarations cannot form one exact source dataset."""


class SourceDatasetDiagnostic(ABC):
    """Nominal root for typed store diagnostics retained with a source dataset."""

    @abstractmethod
    def metadata_payload(self) -> Mapping[str, Any]:
        """Return this diagnostic's leaf-owned canonical metadata payload."""


@dataclass(frozen=True, slots=True)
class OpenHCSPlaneAddress:
    """Canonical OpenHCS logical address for one image plane."""

    well: str
    site: str
    channel: str
    z_index: str
    timepoint: str

    def __post_init__(self) -> None:
        for component, value in self.component_values().items():
            if value is None or value == "":
                raise ValueError(
                    f"{component.value} cannot be empty in an OpenHCS plane address."
                )
            normalized = str(value)
            if component is not AllComponents.WELL and normalized.isdecimal():
                normalized = str(int(normalized))
            object.__setattr__(self, component.value, normalized)

    def component_values(self) -> dict[AllComponents, str]:
        """Return address values keyed by OpenHCS component enum."""

        return {
            AllComponents.WELL: self.well,
            AllComponents.SITE: self.site,
            AllComponents.CHANNEL: self.channel,
            AllComponents.Z_INDEX: self.z_index,
            AllComponents.TIMEPOINT: self.timepoint,
        }

    def as_component_metadata(self) -> dict[str, str]:
        """Return parser-compatible component metadata."""

        return {
            component.value: value
            for component, value in self.component_values().items()
        }

    @staticmethod
    def component_token(value: object) -> str:
        """Encode one exact identity as a filename-safe component token."""

        encoded = "".join(
            (
                chr(byte)
                if (
                    48 <= byte <= 57
                    or 65 <= byte <= 90
                    or 97 <= byte <= 122
                    or byte in (45, 46)
                )
                else f"%{byte:02X}"
            )
            for byte in str(value).encode("utf-8")
        )
        if not encoded:
            raise ValueError("Source component tokens cannot be empty.")
        return encoded

    @classmethod
    def from_parsed(cls, parsed: Mapping[str, Any]) -> "OpenHCSPlaneAddress":
        """Create an address from parser output."""

        missing = [
            component.value
            for component in AllComponents
            if parsed.get(component.value) is None
        ]
        if missing:
            raise ValueError(
                "Parsed OpenHCS virtual filename lacks required components: "
                + ", ".join(missing)
            )
        return cls(
            **{
                component.value: str(parsed[component.value])
                for component in AllComponents
            }
        )

    _filename_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<well>[^_]+)_s(?P<site>[^_]+)_w(?P<channel>[^_]+)"
        r"_z(?P<z_index>[^_]+)_t(?P<timepoint>[^_.]+)"
        r"(?:_[^.]*)?(?P<extension>(?:\.\w+)+)$"
    )

    @classmethod
    def construct_filename(
        cls,
        extension: str = ".tif",
        site_padding: int = 3,
        z_padding: int = 3,
        timepoint_padding: int = 3,
        **component_values: object,
    ) -> str:
        address = cls(
            **{
                component.value: str(component_values[component.value])
                for component in AllComponents
            }
        )
        return (
            f"{address.well}_s{_padded(address.site, site_padding)}"
            f"_w{address.channel}_z{_padded(address.z_index, z_padding)}"
            f"_t{_padded(address.timepoint, timepoint_padding)}{extension}"
        )

    @classmethod
    def parse_filename(cls, filename: str) -> Mapping[str, str | int] | None:
        match = cls._filename_pattern.match(Path(str(filename)).name)
        if match is None:
            return None
        values = match.groupdict()
        return {
            component.value: _coordinate_value(values[component.value])
            for component in AllComponents
        } | {"extension": values["extension"]}


@dataclass(frozen=True, slots=True)
class SourceDatasetIdentity:
    """Explicit metadata identity shared by every store in one source dataset."""

    value: str

    def __post_init__(self) -> None:
        value = str(self.value).strip()
        if not value:
            raise ValueError("SourceDatasetIdentity.value cannot be empty.")
        object.__setattr__(self, "value", value)

    @classmethod
    def for_root(cls, root: str | Path) -> "SourceDatasetIdentity":
        """Return the exact collection identity for one submitted source root."""

        return cls(Path(root).resolve(strict=False).as_uri())


@dataclass(frozen=True, slots=True)
class SourcePlaneStoreIdentity:
    """Exact store-local identities for one declared source plane."""

    container_paths: tuple[Path, ...]
    sample_group_id: str
    image_id: str
    series_id: str
    plane_id: str

    def __post_init__(self) -> None:
        container_paths = tuple(
            sorted(
                {Path(path).resolve(strict=False) for path in self.container_paths},
                key=str,
            )
        )
        if not container_paths:
            raise ValueError(
                "SourcePlaneStoreIdentity.container_paths cannot be empty."
            )
        object.__setattr__(self, "container_paths", container_paths)
        for field_name in (
            "sample_group_id",
            "image_id",
            "series_id",
            "plane_id",
        ):
            value = str(object.__getattribute__(self, field_name)).strip()
            if not value:
                raise ValueError(
                    f"SourcePlaneStoreIdentity.{field_name} cannot be empty."
                )
            object.__setattr__(self, field_name, value)

    @property
    def container_key(self) -> tuple[Path, ...]:
        return self.container_paths

    def is_strictly_subsumed_by(
        self,
        other: "SourcePlaneStoreIdentity",
    ) -> bool:
        """Return whether another store proves ownership of every container path."""

        return set(self.container_paths) < set(other.container_paths)

    @property
    def sample_group_key(self) -> tuple[tuple[Path, ...], str]:
        return (self.container_paths, self.sample_group_id)

    @property
    def image_key(self) -> tuple[tuple[Path, ...], str]:
        return (self.container_paths, self.image_id)

    @property
    def series_key(self) -> tuple[tuple[Path, ...], str]:
        return (self.container_paths, self.series_id)

    @property
    def plane_key(self) -> tuple[tuple[Path, ...], str]:
        return (self.container_paths, self.plane_id)


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """One store-emitted source plane and its exact execution identity."""

    source_ref: SourcePixelRef
    relative_path: str
    metadata: SourceMetadataMapping
    source_axis_shape: tuple[int, ...] = ()
    source_filter_paths: tuple[str, ...] = ()
    component_labels: Mapping[str, str | None] = field(default_factory=dict)
    declared_address: OpenHCSPlaneAddress | None = None
    dataset_identity: SourceDatasetIdentity | None = None
    store_identity: SourcePlaneStoreIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, SourcePixelRef):
            raise TypeError("SourceCandidate.source_ref must be SourcePixelRef.")
        relative_path = Path(str(self.relative_path)).as_posix()
        if not relative_path:
            raise ValueError("SourceCandidate.relative_path cannot be empty.")
        source_axis_shape = tuple(self.source_axis_shape)
        for size in source_axis_shape:
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise TypeError(
                    "SourceCandidate.source_axis_shape must contain positive integers."
                )
        if len(self.source_ref.source_axis_indices) != len(source_axis_shape):
            raise ValueError(
                "SourceCandidate source-axis indices and shape must have equal "
                "cardinality."
            )
        for index, size in zip(
            self.source_ref.source_axis_indices,
            source_axis_shape,
            strict=True,
        ):
            if index >= size:
                raise ValueError(
                    f"SourceCandidate source-axis index {index} exceeds size {size}."
                )
        source_filter_paths = tuple(
            dict.fromkeys(
                Path(str(path)).as_posix() for path in self.source_filter_paths
            )
        )
        if any(not path for path in source_filter_paths):
            raise ValueError(
                "SourceCandidate.source_filter_paths cannot contain empties."
            )
        if self.declared_address is not None and not isinstance(
            self.declared_address,
            OpenHCSPlaneAddress,
        ):
            raise TypeError(
                "SourceCandidate.declared_address must be OpenHCSPlaneAddress or None."
            )
        if self.dataset_identity is not None and not isinstance(
            self.dataset_identity,
            SourceDatasetIdentity,
        ):
            raise TypeError(
                "SourceCandidate.dataset_identity must be SourceDatasetIdentity or None."
            )
        if self.store_identity is not None and not isinstance(
            self.store_identity,
            SourcePlaneStoreIdentity,
        ):
            raise TypeError(
                "SourceCandidate.store_identity must be SourcePlaneStoreIdentity or None."
            )
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "source_axis_shape", source_axis_shape)
        object.__setattr__(self, "source_filter_paths", source_filter_paths)
        object.__setattr__(
            self,
            "component_labels",
            MappingProxyType(dict(self.component_labels)),
        )

    def source_filter_path_identities(self) -> tuple[str, ...]:
        """Return exact path spellings available to source filters."""

        return self.source_filter_paths or (self.relative_path,)

    def component_labels_for_address(
        self,
        address: OpenHCSPlaneAddress,
    ) -> Mapping[str, str | None]:
        """Retain store labels only for coordinates preserved by projection."""

        labels = dict(self.component_labels)
        if self.declared_address is None:
            return MappingProxyType(labels)
        projected_values = address.component_values()
        for component, source_value in self.declared_address.component_values().items():
            if source_value != projected_values[component]:
                labels[component.value] = None
        return MappingProxyType(labels)

    def identity_key(self) -> tuple[object, ...]:
        """Return the complete stable identity of this source candidate."""

        return (
            self.source_ref.backend,
            self.source_ref.backend_address,
            self.source_ref.source_axis_indices,
            self.source_axis_shape,
            self.relative_path,
            self.source_filter_paths,
            tuple(self.component_labels.items()),
            self.declared_address,
            self.dataset_identity,
            self.store_identity,
            SourceMetadataIdentityProjection(self.metadata).items(),
        )


@dataclass(frozen=True, slots=True)
class SourcePlaneDataset:
    """One explicitly identified dataset composed of exact store-emitted planes."""

    root: Path
    identity: SourceDatasetIdentity
    candidates: tuple[SourceCandidate, ...]
    pixel_size: float
    diagnostics: tuple[SourceDatasetDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        root = Path(self.root).resolve(strict=False)
        if not isinstance(self.identity, SourceDatasetIdentity):
            raise TypeError(
                "SourcePlaneDataset.identity must be SourceDatasetIdentity."
            )
        if not self.candidates:
            raise ValueError("SourcePlaneDataset requires at least one source plane.")
        if self.pixel_size <= 0:
            raise ValueError("SourcePlaneDataset.pixel_size must be positive.")
        diagnostics = tuple(self.diagnostics)
        if any(
            not isinstance(diagnostic, SourceDatasetDiagnostic)
            for diagnostic in diagnostics
        ):
            raise TypeError(
                "SourcePlaneDataset.diagnostics must contain "
                "SourceDatasetDiagnostic values."
            )
        for candidate in self.candidates:
            if candidate.dataset_identity != self.identity:
                raise SourceDatasetConflictError(
                    "Every source plane must declare the enclosing dataset identity."
                )
            if candidate.declared_address is None or candidate.store_identity is None:
                raise ValueError(
                    "Store-emitted source planes require exact address and store identity."
                )
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "candidates", self._validated_candidates())
        object.__setattr__(self, "diagnostics", diagnostics)

    @classmethod
    def aggregate(
        cls,
        datasets: tuple["SourcePlaneDataset", ...],
    ) -> "SourcePlaneDataset":
        """Aggregate stores that declare one identical dataset identity."""

        if not datasets:
            raise ValueError("SourcePlaneDataset.aggregate requires datasets.")
        roots = {dataset.root.resolve(strict=False) for dataset in datasets}
        if len(roots) != 1:
            raise SourceDatasetConflictError(
                "Source stores cannot aggregate across different collection roots."
            )
        root = roots.pop()
        collection_identity = SourceDatasetIdentity.for_root(root)
        declared_identities = {
            dataset.identity
            for dataset in datasets
            if dataset.identity != collection_identity
        }
        if len(declared_identities) > 1:
            raise SourceDatasetConflictError(
                "Source stores declare conflicting embedded dataset identities: "
                f"{sorted(identity.value for identity in declared_identities)!r}."
            )
        identity = (
            next(iter(declared_identities))
            if declared_identities
            else collection_identity
        )
        pixel_sizes = {dataset.pixel_size for dataset in datasets}
        if len(pixel_sizes) != 1:
            raise SourceDatasetConflictError(
                f"Source stores declare conflicting pixel sizes: {sorted(pixel_sizes)!r}."
            )
        return cls(
            root=root,
            identity=identity,
            candidates=tuple(
                (
                    candidate
                    if candidate.dataset_identity == identity
                    else replace(candidate, dataset_identity=identity)
                )
                for dataset in datasets
                for candidate in dataset.candidates
            ),
            pixel_size=pixel_sizes.pop(),
            diagnostics=tuple(
                diagnostic for dataset in datasets for diagnostic in dataset.diagnostics
            ),
        )

    def _validated_candidates(self) -> tuple[SourceCandidate, ...]:
        candidates: list[SourceCandidate] = []
        exact_planes: dict[tuple[tuple[Path, ...], str], SourceCandidate] = {}
        addresses: dict[OpenHCSPlaneAddress, SourceCandidate] = {}
        sample_groups: dict[object, object] = {}
        images: dict[object, object] = {}
        series: dict[object, object] = {}
        planes: dict[object, object] = {}
        for candidate in self.candidates:
            store_identity = cast(SourcePlaneStoreIdentity, candidate.store_identity)
            address = cast(OpenHCSPlaneAddress, candidate.declared_address)
            existing_plane = exact_planes.get(store_identity.plane_key)
            if existing_plane is not None:
                if existing_plane.identity_key() == candidate.identity_key():
                    continue
                raise SourceDatasetConflictError(
                    f"Conflicting source plane identity {store_identity.plane_id!r}."
                )
            existing_address = addresses.get(address)
            if existing_address is not None:
                raise SourceDatasetConflictError(
                    f"Duplicate source plane address {address!r}."
                )
            self._record_identity(
                sample_groups,
                store_identity.sample_group_key,
                (address.well, address.site),
                "sample group",
            )
            self._record_identity(
                images,
                store_identity.image_key,
                store_identity.sample_group_key,
                "image",
            )
            self._record_identity(
                series,
                store_identity.series_key,
                store_identity.image_key,
                "series",
            )
            self._record_identity(
                planes,
                store_identity.plane_key,
                (store_identity.series_key, address),
                "plane",
            )
            exact_planes[store_identity.plane_key] = candidate
            addresses[address] = candidate
            candidates.append(candidate)
        return tuple(candidates)

    @staticmethod
    def _record_identity(
        identities: dict[object, object],
        key: object,
        value: object,
        kind: str,
    ) -> None:
        existing = identities.get(key)
        if existing is not None and existing != value:
            raise SourceDatasetConflictError(
                f"Conflicting source {kind} identity {key!r}."
            )
        identities[key] = value


class SourceProjection:
    """Nominal root for typed source projections."""

    projection_role: ClassVar[SourceProjectionRole]
    address: OpenHCSPlaneAddress
    ref: SourcePixelRef
    source_alias: str | None
    artifact_kind: type[ArtifactType]
    source_metadata: SourceMetadataMapping
    component_labels: Mapping[str, str | None]

    @property
    def identity_key(self) -> tuple[object, ...]:
        """Return the projection identity enforced within one source set."""

        return (self.projection_role, self.address)

    @property
    def payload_composition_alias(self) -> str | None:
        """Return the alias that contributes a source-binding payload axis."""

        return None

    def virtual_workspace_path(
        self,
        canonical_path: str,
        *,
        execution_anchor: bool,
    ) -> str:
        """Return this projection's path within the virtual workspace."""

        del execution_anchor
        return canonical_path

    def extend_source_metadata(
        self,
        metadata: dict[str, SourceMetadataValue],
    ) -> None:
        """Add projection-specific fields to serialized source metadata."""

    def extend_serialized_payload(self, payload: dict[str, Any]) -> None:
        """Add projection-specific fields to the nominal wire payload."""

    def matches_binding(self, binding: NamedSourceBinding) -> bool:
        """Return whether this projection represents one exact source binding."""

        return (
            self.projection_role is binding.projection_role
            and self.source_alias == binding.alias
            and self.artifact_kind is binding.artifact_kind
        )


@dataclass(frozen=True, slots=True)
class SourcePlaneProjection(SourceProjection):
    """One canonical primary plane mapped to one source pixel reference."""

    projection_role: ClassVar[SourceProjectionRole] = SourceProjectionRole.PRIMARY_PLANE
    artifact_kind: ClassVar[type[ArtifactType]] = ImageArtifactType

    address: OpenHCSPlaneAddress
    ref: SourcePixelRef
    source_alias: str | None = None
    source_metadata: SourceMetadataMapping = field(
        default_factory=lambda: MappingProxyType({})
    )
    component_labels: Mapping[str, str | None] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _normalize_projection(self)

    @property
    def payload_composition_alias(self) -> str | None:
        """Return this plane's alias for source-binding bundle composition."""

        return self.source_alias


@dataclass(frozen=True, slots=True)
class SourceArtifactProjection(SourceProjection):
    """One typed source artifact attached to a source-set execution address."""

    projection_role: ClassVar[SourceProjectionRole] = (
        SourceProjectionRole.SOURCE_ARTIFACT
    )

    address: OpenHCSPlaneAddress
    ref: SourcePixelRef
    source_alias: str
    artifact_kind: type[ArtifactType]
    source_metadata: SourceMetadataMapping = field(
        default_factory=lambda: MappingProxyType({})
    )
    component_labels: Mapping[str, str | None] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        normalized_alias = str(self.source_alias).strip()
        if not normalized_alias:
            raise ValueError("SourceArtifactProjection.source_alias cannot be empty.")
        object.__setattr__(self, "source_alias", normalized_alias)
        object.__setattr__(
            self,
            "artifact_kind",
            ArtifactType.coerce(self.artifact_kind),
        )
        _normalize_projection(self)

    @property
    def identity_key(self) -> tuple[object, ...]:
        """Distinguish colocated source artifacts by their declared alias."""

        return (*super(SourceArtifactProjection, self).identity_key, self.source_alias)

    def virtual_workspace_path(
        self,
        canonical_path: str,
        *,
        execution_anchor: bool,
    ) -> str:
        """Namespace non-anchor artifacts beneath their source alias."""

        if execution_anchor:
            return canonical_path
        alias = quote(self.source_alias, safe="-_.")
        return str(Path("_source") / alias / canonical_path)

    def extend_source_metadata(
        self,
        metadata: dict[str, SourceMetadataValue],
    ) -> None:
        """Declare this source artifact's nominal kind in metadata."""

        metadata.setdefault("source_artifact_type", self.artifact_kind.require_value())

    def extend_serialized_payload(self, payload: dict[str, Any]) -> None:
        """Declare this source artifact's nominal kind in its wire payload."""

        payload["artifact_kind"] = self.artifact_kind.require_value()


@dataclass(frozen=True, slots=True)
class SourceProjectionSet:
    """Validated source projection set for one OpenHCS source workspace."""

    projections: tuple[SourceProjection, ...]
    diagnostics: tuple[SourceDatasetDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        projections = tuple(self.projections)
        if not projections:
            raise ValueError("SourceProjectionSet requires at least one projection.")
        if any(
            not isinstance(projection, SourceProjection) for projection in projections
        ):
            raise TypeError(
                "SourceProjectionSet.projections must contain SourceProjection values."
            )
        diagnostics = tuple(self.diagnostics)
        if any(
            not isinstance(diagnostic, SourceDatasetDiagnostic)
            for diagnostic in diagnostics
        ):
            raise TypeError(
                "SourceProjectionSet.diagnostics must contain "
                "SourceDatasetDiagnostic values."
            )
        seen: set[tuple[object, ...]] = set()
        for projection in projections:
            identity = projection.identity_key
            if identity in seen:
                raise ValueError(
                    f"Duplicate source projection address: {projection.address}"
                )
            seen.add(identity)
        object.__setattr__(self, "projections", projections)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def plane_projections(self) -> tuple[SourcePlaneProjection, ...]:
        """Return projections that define primary image-plane semantics."""

        return cast(
            tuple[SourcePlaneProjection, ...],
            tuple(
                projection
                for projection in self.projections
                if projection.projection_role is SourceProjectionRole.PRIMARY_PLANE
            ),
        )

    @property
    def artifact_projections(self) -> tuple[SourceArtifactProjection, ...]:
        """Return projections that define typed source artifacts."""

        return cast(
            tuple[SourceArtifactProjection, ...],
            tuple(
                projection
                for projection in self.projections
                if projection.projection_role is SourceProjectionRole.SOURCE_ARTIFACT
            ),
        )

    @property
    def execution_anchor_projections(self) -> tuple[SourceProjection, ...]:
        """Return primary planes, or typed artifacts for artifact-only source sets."""

        return self.plane_projections or self.artifact_projections

    def metadata_dict(
        self,
        *,
        parser: Any,
        microscope_handler_name: str,
        source_filename_parser_name: str,
        grid_dimensions: list[int],
        pixel_size: float,
        available_backends: Mapping[str, bool] | None = None,
        main: bool | None = None,
        results_dir: str | None = None,
        image_extension: str = ".tif",
    ) -> dict[str, Any]:
        """Serialize the projection set to one OpenHCS subdirectory metadata dict."""

        serializer = SourceProjectionMetadataSerializer(
            parser=parser,
            image_extension=image_extension,
        )
        return serializer.metadata_dict(
            self,
            microscope_handler_name=microscope_handler_name,
            source_filename_parser_name=source_filename_parser_name,
            grid_dimensions=grid_dimensions,
            pixel_size=pixel_size,
            available_backends=available_backends,
            main=main,
            results_dir=results_dir,
        )


@dataclass(frozen=True, slots=True)
class SourceProjectionMetadataSerializer:
    """Serialize projection identity into OpenHCS metadata-compatible fields."""

    parser: Any
    image_extension: str = ".tif"
    path_prefix: str | None = None

    def metadata_dict(
        self,
        projection_set: SourceProjectionSet,
        *,
        microscope_handler_name: str,
        source_filename_parser_name: str,
        grid_dimensions: list[int],
        pixel_size: float,
        available_backends: Mapping[str, bool] | None = None,
        main: bool | None = None,
        results_dir: str | None = None,
    ) -> dict[str, Any]:
        """Return an OpenHCS subdirectory metadata dictionary."""

        projection_paths = self.projection_paths(projection_set)
        execution_anchors = projection_set.execution_anchor_projections
        metadata: dict[str, Any] = {
            "microscope_handler_name": microscope_handler_name,
            "source_filename_parser_name": source_filename_parser_name,
            "grid_dimensions": list(grid_dimensions),
            "pixel_size": pixel_size,
            "image_files": [
                path
                for projection, path in projection_paths
                if projection in execution_anchors
            ],
            **{
                SourceComponentProjectionStrategy.for_enum_member(
                    component
                ).metadata_collection_field: self._component_values(
                    projection_set,
                    component,
                )
                for component in AllComponents
            },
            "available_backends": dict(
                available_backends
                if available_backends is not None
                else self._available_backends(projection_set)
            ),
            "workspace_mapping": {
                path: projection.ref.to_workspace_mapping()
                for projection, path in projection_paths
            },
            "source_metadata": {
                path: self._source_metadata(projection)
                for projection, path in projection_paths
            },
            "source_projection": [
                self._source_projection_payload(projection, path)
                for projection, path in projection_paths
            ],
        }
        if main is not None:
            metadata["main"] = main
        if results_dir is not None:
            metadata["results_dir"] = results_dir
        if projection_set.diagnostics:
            metadata["source_diagnostics"] = [
                dict(diagnostic.metadata_payload())
                for diagnostic in projection_set.diagnostics
            ]
        return metadata

    def projection_paths(
        self,
        projection_set: SourceProjectionSet,
    ) -> tuple[tuple[SourceProjection, str], ...]:
        """Return canonical workspace paths for one complete projection set."""

        execution_anchors = projection_set.execution_anchor_projections
        projection_paths = tuple(
            (
                projection,
                self.virtual_path(
                    projection,
                    execution_anchor=projection in execution_anchors,
                ),
            )
            for projection in projection_set.projections
        )
        paths = tuple(path for _projection, path in projection_paths)
        if len(paths) != len(set(paths)):
            raise ValueError(
                "Source projection addresses must produce unique virtual workspace "
                f"paths: {paths!r}."
            )
        return projection_paths

    def virtual_path(
        self,
        projection: SourceProjection,
        *,
        execution_anchor: bool,
    ) -> str:
        """Render and validate one canonical typed source virtual path."""

        address = projection.address
        path = self.parser.construct_filename(
            well=address.well,
            site=_parser_component(address.site),
            channel=_parser_component(address.channel),
            z_index=_parser_component(address.z_index),
            timepoint=_parser_component(address.timepoint),
            extension=self.image_extension,
        )
        parsed = self.parser.parse_filename(path)
        if parsed is None:
            raise ValueError(
                f"Generated virtual filename is not parser-readable: {path!r}"
            )
        parsed_address = OpenHCSPlaneAddress.from_parsed(parsed)
        if parsed_address != address:
            raise ValueError(
                "Generated virtual filename parsed to a different address: "
                f"{path!r} -> {parsed_address!r}, expected {address!r}."
            )
        virtual_path = projection.virtual_workspace_path(
            path,
            execution_anchor=execution_anchor,
        )
        if self.path_prefix is None:
            return virtual_path
        return str(PurePosixPath(self.path_prefix) / virtual_path)

    def _component_values(
        self,
        projection_set: SourceProjectionSet,
        component: AllComponents,
    ) -> dict[str, str | None]:
        values: dict[str, str | None] = {}
        for projection in projection_set.projections:
            key = projection.address.component_values()[component]
            label = projection.component_labels.get(component.value)
            previous = values.get(key)
            if previous is not None and label is not None and previous != label:
                raise ValueError(
                    f"Conflicting label for {component.value}={key!r}: "
                    f"{previous!r} vs {label!r}."
                )
            values[key] = label if label is not None else previous
        return values

    def _available_backends(
        self,
        projection_set: SourceProjectionSet,
    ) -> dict[str, bool]:
        return {
            projection.ref.backend: True for projection in projection_set.projections
        }

    def _source_metadata(
        self,
        projection: SourceProjection,
    ) -> dict[str, SourceMetadataValue]:
        metadata = source_metadata_dict(projection.source_metadata)
        source_component_fields = {
            field: value
            for field, value in SourceMetadataRoleView(metadata).scalar_items()
            if (
                (component := source_metadata_component(field)) is not None
                and field != component.value
            )
        }
        if source_component_fields:
            metadata = with_original_source_metadata(
                metadata,
                source_component_fields,
                path=projection.ref.backend_address,
            )
        original_metadata = dict(SourceMetadataRoleView(metadata).original_items())
        for component, value in projection.address.component_values().items():
            canonical_value = metadata.get(component.value)
            conflicts_with_address = (
                canonical_value is not None
                and not isinstance(canonical_value, Mapping)
                and not source_metadata_values_equal(
                    source_metadata_scalar(canonical_value),
                    value,
                )
            )
            provenance_values = source_component_metadata_values(
                original_metadata,
                component,
            )
            if conflicts_with_address and not any(
                source_metadata_values_equal(source_value, canonical_value)
                for source_value in provenance_values
            ):
                raise ValueError(
                    f"Source metadata {component.value}={canonical_value!r} conflicts "
                    f"with canonical {component.value}={value!r}."
                )
            metadata = with_source_component_metadata(metadata, component, value)
        if projection.source_alias is not None:
            metadata.setdefault(
                SOURCE_BINDING_ALIAS_METADATA_FIELD,
                str(projection.source_alias),
            )
        projection.extend_source_metadata(metadata)
        return metadata

    def _source_projection_payload(
        self,
        projection: SourceProjection,
        path: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "virtual_path": path,
            "address": projection.address.as_component_metadata(),
            "ref": projection.ref.to_workspace_mapping(),
            "projection_role": projection.projection_role.value,
        }
        if projection.source_alias is not None:
            payload["source_alias"] = projection.source_alias
        if projection.source_metadata:
            payload["source_metadata"] = source_metadata_dict(
                projection.source_metadata
            )
        if projection.component_labels:
            payload["component_labels"] = dict(projection.component_labels)
        projection.extend_serialized_payload(payload)
        return payload


def _parser_component(value: str) -> str | int:
    """Preserve semantic numeric axes while letting parsers own formatting."""

    return int(value) if value.isdecimal() else value


def _coordinate_value(value: str) -> str | int:
    return int(value) if value.isdecimal() else value


def _padded(value: str, width: int) -> str:
    return f"{int(value):0{width}d}" if value.isdecimal() else value


def _normalized_source_metadata_value(
    value: SourceMetadataValue,
) -> SourceMetadataValue:
    """Freeze one source-metadata value without erasing its nominal shape."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): source_metadata_scalar(nested_value)
                for key, nested_value in value.items()
            }
        )
    return source_metadata_scalar(value)


def _normalize_projection(projection: SourceProjection) -> None:
    """Normalize fields shared by every nominal source projection."""

    if not isinstance(projection.address, OpenHCSPlaneAddress):
        raise TypeError("Source projection address must be OpenHCSPlaneAddress.")
    if not isinstance(projection.ref, SourcePixelRef):
        raise TypeError("Source projection ref must be SourcePixelRef.")
    object.__setattr__(
        projection,
        "source_metadata",
        MappingProxyType(
            {
                str(key): _normalized_source_metadata_value(value)
                for key, value in projection.source_metadata.items()
            }
        ),
    )
    object.__setattr__(
        projection,
        "component_labels",
        MappingProxyType(
            {
                str(key): None if value is None else str(value)
                for key, value in projection.component_labels.items()
            }
        ),
    )
