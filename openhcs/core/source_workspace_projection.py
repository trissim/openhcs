"""Project OpenHCS virtual-workspace metadata for runtime source binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar

from openhcs.core.source_metadata import SourceMetadataMapping
from openhcs.core.source_matching import (
    source_component_metadata_values,
    source_metadata_values_equal,
)
from openhcs.core.source_schema_workspace import (
    source_schema_metadata_with_virtual_components,
)
from openhcs.core.virtual_workspace_metadata import (
    OpenHCSMetadataPayload,
    OpenHCSMetadataSubdirectories,
    OpenHCSSubdirectoryPayload,
    VirtualWorkspaceMapping,
    VirtualWorkspaceSourceMetadataEntries,
    WorkspaceSourceRef,
)
from openhcs.microscopes.openhcs import (
    workspace_mapping_source_path,
)

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.microscopes.microscope_interfaces import MetadataHandler


LookupValueT = TypeVar("LookupValueT")


@dataclass(frozen=True, slots=True)
class VirtualWorkspacePathLookup:
    """Virtual workspace path identity for source path and metadata lookup."""

    virtual_path: str
    full_virtual_path: str

    @classmethod
    def from_paths(
        cls,
        virtual_path: str,
        full_virtual_path: str,
    ) -> "VirtualWorkspacePathLookup":
        return cls(str(virtual_path), str(full_virtual_path))

    def candidates(self) -> tuple[str, str]:
        return (self.virtual_path, self.full_virtual_path)


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceSourceProjection:
    """Source-binding projection derived from OpenHCS virtual-workspace metadata."""

    source_paths_by_virtual_path: Mapping[str, str]
    source_metadata_by_path: Mapping[str, SourceMetadataMapping]
    workspace_root: str | None = None
    _pipeline_start_files_by_axis: dict[str | None, tuple[str, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def empty(cls, plate_path: Path | None = None) -> "VirtualWorkspaceSourceProjection":
        workspace_root = None
        if plate_path is not None:
            workspace_root = str(plate_path)
        return cls(
            source_paths_by_virtual_path=MappingProxyType({}),
            source_metadata_by_path=MappingProxyType({}),
            workspace_root=workspace_root,
        )

    @classmethod
    def from_openhcs_metadata(
        cls,
        plate_path: Path,
        metadata: OpenHCSMetadataPayload,
    ) -> "VirtualWorkspaceSourceProjection":
        builder = VirtualWorkspaceSourceProjectionBuilder(plate_path)
        for subdirectory in OpenHCSMetadataSubdirectories(metadata).values():
            builder.ingest_subdirectory(subdirectory)
        return builder.projection()

    @classmethod
    def from_openhcs_metadata_if_available(
        cls,
        plate_path: Path,
        metadata: OpenHCSMetadataPayload,
    ) -> "VirtualWorkspaceSourceProjection | None":
        subdirectories = OpenHCSMetadataSubdirectories(metadata)
        if not subdirectories.has_workspace_mapping():
            return None
        builder = VirtualWorkspaceSourceProjectionBuilder(plate_path)
        for subdirectory in subdirectories.values():
            builder.ingest_subdirectory(subdirectory)
        return builder.projection()

    @classmethod
    def openhcs_metadata_has_workspace_mapping(
        cls,
        metadata: OpenHCSMetadataPayload,
    ) -> bool:
        return OpenHCSMetadataSubdirectories(metadata).has_workspace_mapping()

    def first_virtual_path_value(
        self,
        mapping: Mapping[str, LookupValueT],
        lookup: VirtualWorkspacePathLookup,
    ) -> LookupValueT | None:
        """Return the first mapped value for a virtual/full path pair."""
        for key in lookup.candidates():
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    def source_path_for(
        self,
        lookup: VirtualWorkspacePathLookup,
    ) -> str:
        """Return the physical source path represented by a virtual workspace path."""
        source_path = self.first_virtual_path_value(
            self.source_paths_by_virtual_path,
            lookup,
        )
        if source_path is None:
            return lookup.full_virtual_path
        return str(source_path)

    def source_metadata_for(
        self,
        lookup: VirtualWorkspacePathLookup,
    ) -> SourceMetadataMapping | None:
        """Return source metadata represented by a virtual workspace path."""
        metadata = self.first_virtual_path_value(
            self.source_metadata_by_path,
            lookup,
        )
        if metadata is not None:
            return metadata
        source_path = self.source_path_for(lookup)
        metadata = self.source_metadata_by_path.get(source_path)
        if metadata is not None:
            return metadata
        for key in lookup.candidates():
            metadata = source_schema_filename_metadata(key)
            if metadata is not None:
                return metadata
        return None

    def pipeline_start_files(self, *, axis_id: str | None = None) -> tuple[str, ...]:
        """Return loadable virtual source paths for one runtime source universe."""
        cached = self._pipeline_start_files_by_axis.get(axis_id)
        if cached is not None:
            return cached

        relative_virtual_paths = tuple(
            virtual_path
            for virtual_path in self.source_paths_by_virtual_path
            if not Path(virtual_path).is_absolute()
        )
        if not relative_virtual_paths:
            relative_virtual_paths = tuple(self.source_paths_by_virtual_path)

        selected = tuple(
            virtual_path
            for virtual_path in relative_virtual_paths
            if self._path_belongs_to_axis(virtual_path, axis_id)
        )
        result = tuple(
            dict.fromkeys(
                self._loadable_virtual_path(virtual_path)
                for virtual_path in selected
            )
        )
        self._pipeline_start_files_by_axis[axis_id] = result
        return result

    def filtered_by_axis(
        self,
        *,
        axis_id: str | None,
    ) -> "VirtualWorkspaceSourceProjection":
        """Return a projection view restricted to one multiprocessing axis."""
        if axis_id is None:
            return self

        source_paths_by_virtual_path: dict[str, str] = {}
        source_metadata_by_path: dict[str, SourceMetadataMapping] = {}
        for virtual_path, source_path in self.source_paths_by_virtual_path.items():
            if not self._path_belongs_to_axis(virtual_path, axis_id):
                continue
            source_paths_by_virtual_path[virtual_path] = source_path
            for metadata_path in (
                virtual_path,
                self._loadable_virtual_path(virtual_path),
                source_path,
            ):
                metadata = self.source_metadata_by_path.get(metadata_path)
                if metadata is not None:
                    source_metadata_by_path[metadata_path] = metadata

        return VirtualWorkspaceSourceProjection(
            source_paths_by_virtual_path=MappingProxyType(source_paths_by_virtual_path),
            source_metadata_by_path=MappingProxyType(source_metadata_by_path),
            workspace_root=self.workspace_root,
        )

    def _path_belongs_to_axis(
        self,
        virtual_path: str,
        axis_id: str | None,
    ) -> bool:
        if axis_id is None:
            return True
        metadata = self.source_metadata_for(
            VirtualWorkspacePathLookup.from_paths(
                virtual_path,
                self._loadable_virtual_path(virtual_path),
            )
        )
        if metadata is None:
            return True
        from openhcs.constants import MULTIPROCESSING_AXIS

        values = source_component_metadata_values(metadata, MULTIPROCESSING_AXIS)
        if not values:
            return True
        return any(source_metadata_values_equal(value, axis_id) for value in values)

    def _loadable_virtual_path(self, virtual_path: str) -> str:
        if Path(virtual_path).is_absolute():
            return virtual_path
        if self.workspace_root is not None:
            return str(Path(self.workspace_root) / virtual_path)
        return virtual_path


@lru_cache(maxsize=8192)
def source_schema_filename_metadata(path: str) -> SourceMetadataMapping | None:
    """Return component metadata encoded in a normalized virtual source filename."""

    from openhcs.microscopes.source_schema import SourceSchemaFilenameParser

    parsed = SourceSchemaFilenameParser().parse_filename(path)
    if parsed is None:
        return None
    return dict(parsed)


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceSourceProjectionAxisCacheKey:
    """Cache key for an axis-filtered projection view."""

    projection_identity: int
    axis_id: str | None


@dataclass(slots=True)
class VirtualWorkspaceSourceProjectionCache:
    """Process-local cache for source-workspace projections keyed by plate path."""

    projections_by_plate_path: dict[str, VirtualWorkspaceSourceProjection] = field(
        default_factory=dict
    )
    axis_filtered_projections: dict[
        VirtualWorkspaceSourceProjectionAxisCacheKey,
        VirtualWorkspaceSourceProjection,
    ] = field(default_factory=dict)

    def projection_for(
        self,
        plate_path: Path,
        metadata: OpenHCSMetadataPayload,
    ) -> VirtualWorkspaceSourceProjection:
        plate_key = str(plate_path)
        projection = self.projections_by_plate_path.get(plate_key)
        if projection is None:
            projection = VirtualWorkspaceSourceProjection.from_openhcs_metadata(
                plate_path,
                metadata,
            )
            self.projections_by_plate_path[plate_key] = projection
        return projection

    def filtered_by_axis(
        self,
        projection: VirtualWorkspaceSourceProjection,
        *,
        axis_id: str | None,
    ) -> VirtualWorkspaceSourceProjection:
        """Return an axis-filtered projection owned by this cache."""
        if axis_id is None:
            return projection
        cache_key = VirtualWorkspaceSourceProjectionAxisCacheKey(
            projection_identity=id(projection),
            axis_id=axis_id,
        )
        filtered = self.axis_filtered_projections.get(cache_key)
        if filtered is None:
            filtered = projection.filtered_by_axis(axis_id=axis_id)
            self.axis_filtered_projections[cache_key] = filtered
        return filtered


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceSourceProjectionAuthority:
    """Projection authority for source-workspace metadata owned by a plate handler."""

    plate_path: Path
    metadata_handler: "MetadataHandler"
    cache: VirtualWorkspaceSourceProjectionCache | None = None

    @classmethod
    def from_context(
        cls,
        context: "ProcessingContext",
        *,
        cache: VirtualWorkspaceSourceProjectionCache | None = None,
    ) -> "VirtualWorkspaceSourceProjectionAuthority":
        return cls(
            plate_path=Path(context.plate_path),
            metadata_handler=context.microscope_handler.metadata_handler,
            cache=cache,
        )

    def metadata_document(self) -> OpenHCSMetadataPayload | None:
        metadata = self.metadata_handler.source_workspace_metadata_document(
            self.plate_path
        )
        if metadata is None:
            return None
        if not isinstance(metadata, Mapping):
            raise RuntimeError("Source workspace metadata document must be a mapping.")
        return metadata

    def projection_if_available(self) -> VirtualWorkspaceSourceProjection | None:
        metadata = self.metadata_document()
        if metadata is None:
            return None
        if not OpenHCSMetadataSubdirectories(metadata).has_workspace_mapping():
            return None
        if self.cache is None:
            return VirtualWorkspaceSourceProjection.from_openhcs_metadata(
                self.plate_path,
                metadata,
            )
        return self.cache.projection_for(self.plate_path, metadata)

    def projection_or_empty(self) -> VirtualWorkspaceSourceProjection:
        projection = self.projection_if_available()
        if projection is not None:
            return projection
        return VirtualWorkspaceSourceProjection.empty(self.plate_path)


@dataclass(slots=True)
class RealPathSourceMetadataIndex:
    """Track real-path source metadata while discarding ambiguous conflicts."""

    metadata_by_real_path: dict[str, SourceMetadataMapping] = field(default_factory=dict)
    conflicted_real_paths: set[str] = field(default_factory=set)

    def record(self, real_path: str, metadata_fields: SourceMetadataMapping) -> None:
        if real_path in self.conflicted_real_paths:
            return
        existing_metadata = self.metadata_by_real_path.get(real_path)
        if existing_metadata is None:
            self.metadata_by_real_path[real_path] = metadata_fields
            return
        if dict(existing_metadata) != dict(metadata_fields):
            del self.metadata_by_real_path[real_path]
            self.conflicted_real_paths.add(real_path)


@dataclass(slots=True)
class VirtualWorkspaceSourceProjectionBuilder:
    """Build source-binding projection data from OpenHCS virtual-workspace metadata."""

    plate_path: Path
    workspace_source_paths: dict[str, str] = field(default_factory=dict)
    source_metadata_by_path: dict[str, SourceMetadataMapping] = field(default_factory=dict)
    real_path_metadata: RealPathSourceMetadataIndex = field(
        default_factory=RealPathSourceMetadataIndex
    )

    def ingest_subdirectory(self, subdirectory: OpenHCSSubdirectoryPayload) -> None:
        workspace_mapping = VirtualWorkspaceMapping.from_subdirectory(subdirectory)
        self.ingest_workspace_mapping(workspace_mapping)
        self.ingest_source_metadata(
            VirtualWorkspaceSourceMetadataEntries.from_subdirectory(subdirectory),
            workspace_mapping,
        )

    def ingest_workspace_mapping(self, workspace_mapping: VirtualWorkspaceMapping) -> None:
        for virtual_path, source_ref in workspace_mapping.entries.items():
            self.record_workspace_source_path(virtual_path, source_ref)

    def record_workspace_source_path(
        self,
        virtual_path: str,
        source_ref: WorkspaceSourceRef,
    ) -> None:
        real_path = self.source_path(source_ref)
        self.workspace_source_paths[virtual_path] = real_path
        self.workspace_source_paths[str(self.plate_path / virtual_path)] = real_path

    def ingest_source_metadata(
        self,
        source_metadata: VirtualWorkspaceSourceMetadataEntries,
        workspace_mapping: VirtualWorkspaceMapping,
    ) -> None:
        for virtual_path, metadata_fields in source_metadata.entries.items():
            self.record_source_metadata(virtual_path, metadata_fields, workspace_mapping)

    def record_source_metadata(
        self,
        virtual_path: str,
        metadata_fields: SourceMetadataMapping,
        workspace_mapping: VirtualWorkspaceMapping,
    ) -> None:
        normalized_metadata = source_schema_metadata_with_virtual_components(
            virtual_path,
            metadata_fields,
        )
        self.source_metadata_by_path[virtual_path] = normalized_metadata
        self.source_metadata_by_path[str(self.plate_path / virtual_path)] = (
            normalized_metadata
        )
        source_ref = workspace_mapping.source_ref_for(virtual_path)
        if source_ref is not None:
            self.real_path_metadata.record(
                self.source_path(source_ref),
                normalized_metadata,
            )

    def projection(self) -> VirtualWorkspaceSourceProjection:
        for real_path, metadata_fields in self.real_path_metadata.metadata_by_real_path.items():
            self.source_metadata_by_path[real_path] = metadata_fields
        if not self.workspace_source_paths:
            raise RuntimeError(
                "virtual_workspace source binding resolution requires "
                "workspace_mapping entries in OpenHCS metadata."
            )
        return VirtualWorkspaceSourceProjection(
            source_paths_by_virtual_path=MappingProxyType(self.workspace_source_paths),
            source_metadata_by_path=MappingProxyType(self.source_metadata_by_path),
            workspace_root=str(self.plate_path),
        )

    def source_path(self, source_ref: WorkspaceSourceRef) -> str:
        return str(workspace_mapping_source_path(self.plate_path, source_ref))
