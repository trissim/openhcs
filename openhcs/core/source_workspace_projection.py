"""Project OpenHCS virtual-workspace metadata for runtime source binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar

from openhcs.constants import Backend
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadataCompositionMode,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.source_bindings import (
    SOURCE_BINDING_ALIAS_METADATA_FIELD,
    SourceProjectionRole,
)
from openhcs.core.source_metadata import SourceMetadataMapping
from openhcs.core.source_matching import (
    source_component_metadata_values,
    source_metadata_value,
    source_metadata_values_equal,
)
from openhcs.core.source_path_identity import source_path_identity_key
from openhcs.core.source_projection import SourceProjection
from openhcs.core.virtual_workspace_metadata import (
    OpenHCSMetadataPayload,
    OpenHCSMetadataSubdirectories,
    OpenHCSSubdirectoryPayload,
    VirtualWorkspaceMapping,
    VirtualWorkspaceSourceProjectionEntries,
    VirtualWorkspaceSourceMetadataEntries,
)
from polystore.virtual_workspace import SourcePixelRef

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.microscopes.microscope_interfaces import MetadataHandler
    from openhcs.core.vfs_protocol import FileManagerLike
    from polystore.filemanager import FileManager


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

    source_refs_by_virtual_path: Mapping[str, SourcePixelRef]
    source_metadata_by_path: Mapping[str, SourceMetadataMapping]
    workspace_root: str | None = None
    source_projections_by_virtual_path: Mapping[str, SourceProjection] = field(
        default_factory=lambda: MappingProxyType({})
    )
    _pipeline_start_files_by_axis: dict[str | None, tuple[str, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def empty(
        cls, plate_path: Path | None = None
    ) -> "VirtualWorkspaceSourceProjection":
        workspace_root = None
        if plate_path is not None:
            workspace_root = str(plate_path)
        return cls(
            source_refs_by_virtual_path=MappingProxyType({}),
            source_metadata_by_path=MappingProxyType({}),
            source_projections_by_virtual_path=MappingProxyType({}),
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
        """Return the opaque backend address represented by a virtual path."""
        source_ref = self.source_ref_for(lookup)
        if source_ref is None:
            return lookup.full_virtual_path
        return source_ref.backend_address

    def source_ref_for(
        self,
        lookup: VirtualWorkspacePathLookup,
    ) -> SourcePixelRef | None:
        """Return the complete backend-owned source reference for a virtual path."""

        return self.first_virtual_path_value(
            self.source_refs_by_virtual_path,
            lookup,
        )

    def resolved_source_path_for(
        self,
        lookup: VirtualWorkspacePathLookup,
        filemanager: "FileManagerLike",
    ) -> str:
        """Resolve one mapped source through its declared storage backend."""

        source_ref = self.source_ref_for(lookup)
        if source_ref is None:
            return lookup.full_virtual_path
        if self.workspace_root is None:
            raise ValueError(
                "Virtual workspace source resolution requires a workspace root."
            )
        return str(
            filemanager.source_path(
                source_ref.backend_address,
                source_ref.backend,
                base_path=self.workspace_root,
            )
        )

    def source_projection_for(
        self,
        lookup: VirtualWorkspacePathLookup,
    ) -> SourceProjection | None:
        """Return the nominal source projection represented by a virtual path."""
        return self.first_virtual_path_value(
            self.source_projections_by_virtual_path,
            lookup,
        )

    def require_source_projection_for(
        self,
        lookup: VirtualWorkspacePathLookup,
    ) -> SourceProjection:
        """Return the nominal projection or fail at the workspace boundary."""
        projection = self.source_projection_for(lookup)
        if projection is None:
            raise ValueError(
                "Virtual workspace source path has no declared source_projection: "
                f"{lookup.virtual_path!r}."
            )
        return projection

    def payload_composition_mode(
        self,
        lookups: tuple[VirtualWorkspacePathLookup, ...],
    ) -> ImagePayloadMetadataCompositionMode:
        """Return the leading-axis topology declared by workspace projections."""
        projections = tuple(
            self.require_source_projection_for(lookup) for lookup in lookups
        )
        source_aliases = tuple(
            projection.payload_composition_alias
            for projection in projections
            if projection.payload_composition_alias is not None
        )
        if (
            len(source_aliases) == len(projections)
            and len(source_aliases) > 1
            and len(set(source_aliases)) == len(source_aliases)
        ):
            return ImagePayloadMetadataCompositionMode.BUNDLE
        return ImagePayloadMetadataCompositionMode.STACK

    def project_payload(
        self,
        lookup: VirtualWorkspacePathLookup,
        payload: RuntimeArrayData,
    ) -> RuntimeArrayData:
        """Carry one nominal source projection into runtime payload provenance."""
        projection = self.require_source_projection_for(lookup)
        source_metadata = self.source_metadata_for(lookup)
        return self._project_payload_source_metadata(
            payload,
            source_metadata=source_metadata,
            source_alias=projection.source_alias,
        )

    def project_unbound_payload(
        self,
        lookup: VirtualWorkspacePathLookup,
        payload: RuntimeArrayData,
    ) -> RuntimeArrayData:
        """Carry workspace source metadata without requiring a step binding."""

        source_metadata = self.source_metadata_for(lookup)
        source_alias = (
            None
            if source_metadata is None
            else source_metadata_value(
                source_metadata,
                SOURCE_BINDING_ALIAS_METADATA_FIELD,
            )
        )
        return self._project_payload_source_metadata(
            payload,
            source_metadata=source_metadata,
            source_alias=source_alias,
        )

    @staticmethod
    def _project_payload_source_metadata(
        payload: RuntimeArrayData,
        *,
        source_metadata: SourceMetadataMapping | None,
        source_alias: str | None,
    ) -> RuntimeArrayData:
        """Apply component metadata and source-name provenance to one payload."""

        if source_metadata is not None:
            source_metadata = MappingProxyType(
                {
                    field: value
                    for field, value in source_metadata.items()
                    if field != SOURCE_BINDING_ALIAS_METADATA_FIELD
                }
            )
        metadata = image_payload_metadata(payload)
        if source_metadata is not None:
            metadata = metadata.with_source_component_metadata(source_metadata)
        if source_alias is not None:
            metadata = metadata.with_source_provenance(
                metadata.source_provenance.with_source_image_names((source_alias,))
            )
        return metadata.payload_with(
            image_payload_data(payload),
            image_payload_mask(payload),
        )

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
        for virtual_path in self.virtual_paths_for_source_path(lookup):
            metadata = self.source_metadata_by_path.get(virtual_path)
            if metadata is not None:
                return metadata
            metadata = self.source_metadata_by_path.get(
                self._loadable_virtual_path(virtual_path)
            )
            if metadata is not None:
                return metadata
            metadata = source_schema_filename_metadata(virtual_path)
            if metadata is not None:
                return metadata
        for key in lookup.candidates():
            metadata = source_schema_filename_metadata(key)
            if metadata is not None:
                return metadata
        return None

    def virtual_paths_for_source_path(
        self,
        lookup: VirtualWorkspacePathLookup,
    ) -> tuple[str, ...]:
        """Return virtual paths whose physical source path matches the lookup."""

        source_path_identities = frozenset(
            source_path_identity_key(candidate) for candidate in lookup.candidates()
        )
        return tuple(
            virtual_path
            for virtual_path, source_ref in self.source_refs_by_virtual_path.items()
            if not _cached_path_is_absolute(virtual_path)
            and source_path_identity_key(source_ref.backend_address)
            in source_path_identities
        )

    def pipeline_start_files(self, *, axis_id: str | None = None) -> tuple[str, ...]:
        """Return loadable virtual source paths for one runtime source universe."""
        cached = self._pipeline_start_files_by_axis.get(axis_id)
        if cached is not None:
            return cached

        relative_virtual_paths = self.relative_virtual_paths()
        selected = tuple(
            virtual_path
            for virtual_path in relative_virtual_paths
            if self._path_belongs_to_axis(virtual_path, axis_id)
        )
        result = tuple(
            dict.fromkeys(
                self._loadable_virtual_path(virtual_path) for virtual_path in selected
            )
        )
        self._pipeline_start_files_by_axis[axis_id] = result
        return result

    def files_for_projection_role(
        self,
        projection_role: SourceProjectionRole,
        *,
        axis_id: str | None = None,
    ) -> tuple[str, ...]:
        """Return source files belonging to one exact declared projection role."""

        role = (
            projection_role
            if isinstance(projection_role, SourceProjectionRole)
            else SourceProjectionRole(projection_role)
        )
        return tuple(
            self._loadable_virtual_path(virtual_path)
            for virtual_path in self.relative_virtual_paths()
            if self._path_belongs_to_axis(virtual_path, axis_id)
            and self.require_source_projection_for(
                VirtualWorkspacePathLookup.from_paths(
                    virtual_path,
                    self._loadable_virtual_path(virtual_path),
                )
            ).projection_role
            is role
        )

    def validate_runtime_metadata_projection(
        self,
        *,
        axis_id: str | None = None,
    ) -> None:
        """Fail if explicit source metadata cannot survive runtime path spelling."""

        failures: list[str] = []
        for virtual_path in self.relative_virtual_paths():
            if not self._path_belongs_to_axis(virtual_path, axis_id):
                continue
            loadable_path = self._loadable_virtual_path(virtual_path)
            expected_metadata = self.explicit_metadata_for_virtual_path(
                virtual_path,
                loadable_path,
            )
            if not expected_metadata:
                continue
            runtime_metadata = self.source_metadata_for(
                VirtualWorkspacePathLookup.from_paths(
                    loadable_path,
                    loadable_path,
                )
            )
            if runtime_metadata is None:
                failures.append(f"{loadable_path}: missing source metadata")
                continue
            mismatched = tuple(
                key
                for key, expected_value in expected_metadata.items()
                if runtime_metadata.get(key) != expected_value
            )
            if mismatched:
                failures.append(
                    f"{loadable_path}: metadata mismatch for {mismatched!r}"
                )

        if failures:
            preview = "; ".join(failures[:10])
            if len(failures) > 10:
                preview = f"{preview}; ... ({len(failures)} paths total)"
            raise ValueError(
                "Source workspace projection cannot preserve explicit source "
                f"metadata for runtime load paths: {preview}."
            )

    def explicit_metadata_for_virtual_path(
        self,
        virtual_path: str,
        loadable_path: str,
    ) -> SourceMetadataMapping | None:
        """Return metadata explicitly declared for one virtual path spelling."""

        metadata = self.source_metadata_by_path.get(virtual_path)
        if metadata is not None:
            return metadata
        return self.source_metadata_by_path.get(loadable_path)

    def relative_virtual_paths(self) -> tuple[str, ...]:
        """Return canonical relative virtual paths for source projection traversal."""

        relative_virtual_paths = tuple(
            virtual_path
            for virtual_path in self.source_refs_by_virtual_path
            if not _cached_path_is_absolute(virtual_path)
        )
        if relative_virtual_paths:
            return relative_virtual_paths
        return tuple(self.source_refs_by_virtual_path)

    def filtered_by_axis(
        self,
        *,
        axis_id: str | None,
    ) -> "VirtualWorkspaceSourceProjection":
        """Return a projection view restricted to one multiprocessing axis."""
        if axis_id is None:
            return self

        source_refs_by_virtual_path: dict[str, SourcePixelRef] = {}
        source_metadata_by_path: dict[str, SourceMetadataMapping] = {}
        source_projections_by_virtual_path: dict[str, SourceProjection] = {}
        for virtual_path, source_ref in self.source_refs_by_virtual_path.items():
            if not self._path_belongs_to_axis(virtual_path, axis_id):
                continue
            source_refs_by_virtual_path[virtual_path] = source_ref
            projection = self.source_projections_by_virtual_path.get(virtual_path)
            if projection is not None:
                source_projections_by_virtual_path[virtual_path] = projection
            for metadata_path in (
                virtual_path,
                self._loadable_virtual_path(virtual_path),
                source_ref.backend_address,
            ):
                metadata = self.source_metadata_by_path.get(metadata_path)
                if metadata is not None:
                    source_metadata_by_path[metadata_path] = metadata

        return VirtualWorkspaceSourceProjection(
            source_refs_by_virtual_path=MappingProxyType(source_refs_by_virtual_path),
            source_metadata_by_path=MappingProxyType(source_metadata_by_path),
            source_projections_by_virtual_path=MappingProxyType(
                source_projections_by_virtual_path
            ),
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
        if _cached_path_is_absolute(virtual_path):
            return virtual_path
        if self.workspace_root is not None:
            return _cached_join_workspace_path(str(self.workspace_root), virtual_path)
        return virtual_path


@lru_cache(maxsize=65536)
def _cached_path_is_absolute(path: str) -> bool:
    """Return whether a virtual/source path string is absolute."""

    return Path(path).is_absolute()


@lru_cache(maxsize=65536)
def _cached_join_workspace_path(workspace_root: str, virtual_path: str) -> str:
    """Return loadable path for a workspace-root/virtual-path pair."""

    return str(Path(workspace_root) / virtual_path)


@lru_cache(maxsize=8192)
def source_schema_filename_metadata(path: str) -> SourceMetadataMapping | None:
    """Return component metadata encoded in a normalized virtual source filename."""

    from openhcs.microscopes.source_schema import SourceSchemaFilenameParser

    parsed = SourceSchemaFilenameParser().parse_filename(path)
    if parsed is None:
        return None
    return parsed.wire_mapping()


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
    metadata_handlers: tuple["MetadataHandler", ...]
    cache: VirtualWorkspaceSourceProjectionCache | None = None

    @classmethod
    def from_context(
        cls,
        context: "ProcessingContext",
        *,
        cache: VirtualWorkspaceSourceProjectionCache | None = None,
    ) -> "VirtualWorkspaceSourceProjectionAuthority":
        return cls.from_plate_metadata(
            plate_path=Path(context.plate_path),
            metadata_handler=context.microscope_handler.metadata_handler,
            filemanager=context.filemanager,
            cache=cache,
        )

    @classmethod
    def from_plate_metadata(
        cls,
        *,
        plate_path: Path,
        metadata_handler: "MetadataHandler",
        filemanager: "FileManager",
        cache: VirtualWorkspaceSourceProjectionCache | None = None,
    ) -> "VirtualWorkspaceSourceProjectionAuthority":
        """Build projection authority from the plate-level metadata owners."""

        return cls(
            plate_path=plate_path,
            metadata_handlers=cls._plate_metadata_handlers(
                plate_path,
                metadata_handler,
                filemanager,
            ),
            cache=cache,
        )

    @staticmethod
    def _plate_metadata_handlers(
        plate_path: Path,
        metadata_handler: "MetadataHandler",
        filemanager: "FileManager",
    ) -> tuple["MetadataHandler", ...]:
        """Return metadata handlers that can own source-workspace metadata."""

        from openhcs.microscopes.openhcs import OpenHCSMetadataHandler

        handlers: list["MetadataHandler"] = [metadata_handler]
        metadata_path = plate_path / OpenHCSMetadataHandler.METADATA_FILENAME
        if not isinstance(handlers[0], OpenHCSMetadataHandler) and filemanager.exists(
            str(metadata_path), Backend.DISK.value
        ):
            handlers.append(OpenHCSMetadataHandler(filemanager))
        return tuple(handlers)

    def metadata_documents(self) -> tuple[OpenHCSMetadataPayload, ...]:
        documents: list[OpenHCSMetadataPayload] = []
        for metadata_handler in self.metadata_handlers:
            metadata = metadata_handler.source_workspace_metadata_document(
                self.plate_path
            )
            if metadata is None:
                continue
            if not isinstance(metadata, Mapping):
                raise RuntimeError(
                    "Source workspace metadata document must be a mapping."
                )
            documents.append(metadata)
        return tuple(documents)

    def projection_if_available(self) -> VirtualWorkspaceSourceProjection | None:
        for metadata in self.metadata_documents():
            if not OpenHCSMetadataSubdirectories(metadata).has_workspace_mapping():
                continue
            if self.cache is None:
                return VirtualWorkspaceSourceProjection.from_openhcs_metadata(
                    self.plate_path,
                    metadata,
                )
            return self.cache.projection_for(self.plate_path, metadata)
        return None

    def projection_or_empty(self) -> VirtualWorkspaceSourceProjection:
        projection = self.projection_if_available()
        if projection is not None:
            return projection
        return VirtualWorkspaceSourceProjection.empty(self.plate_path)


@dataclass(slots=True)
class VirtualWorkspaceSourceProjectionBuilder:
    """Build source-binding projection data from OpenHCS virtual-workspace metadata."""

    plate_path: Path
    workspace_source_refs: dict[str, SourcePixelRef] = field(default_factory=dict)
    source_metadata_by_path: dict[str, SourceMetadataMapping] = field(
        default_factory=dict
    )
    source_projections_by_virtual_path: dict[str, SourceProjection] = field(
        default_factory=dict
    )

    def ingest_subdirectory(self, subdirectory: OpenHCSSubdirectoryPayload) -> None:
        workspace_mapping = VirtualWorkspaceMapping.from_subdirectory(subdirectory)
        self.ingest_workspace_mapping(workspace_mapping)
        self.ingest_source_projections(
            VirtualWorkspaceSourceProjectionEntries.from_subdirectory(subdirectory)
        )
        self.ingest_source_metadata(
            VirtualWorkspaceSourceMetadataEntries.from_subdirectory(subdirectory),
        )

    def ingest_workspace_mapping(
        self, workspace_mapping: VirtualWorkspaceMapping
    ) -> None:
        for virtual_path, source_ref in workspace_mapping.entries.items():
            self.record_workspace_source_path(virtual_path, source_ref)

    def record_workspace_source_path(
        self,
        virtual_path: str,
        source_ref: SourcePixelRef,
    ) -> None:
        if not isinstance(source_ref, SourcePixelRef):
            raise TypeError(
                "Workspace source references must be SourcePixelRef values."
            )
        loadable_path = str(self.plate_path / virtual_path)
        self.workspace_source_refs[virtual_path] = source_ref
        self.workspace_source_refs[loadable_path] = source_ref

    def ingest_source_projections(
        self,
        source_projections: VirtualWorkspaceSourceProjectionEntries,
    ) -> None:
        for virtual_path, projection in source_projections.entries.items():
            mapped_ref = self.workspace_source_refs.get(virtual_path)
            if mapped_ref is None:
                raise RuntimeError(
                    "source_projection has no workspace_mapping entry for "
                    f"{virtual_path!r}."
                )
            if projection.ref != mapped_ref:
                raise RuntimeError(
                    "source_projection ref conflicts with workspace_mapping for "
                    f"{virtual_path!r}."
                )
            self.source_projections_by_virtual_path[virtual_path] = projection
            self.source_projections_by_virtual_path[
                str(self.plate_path / virtual_path)
            ] = projection

    def ingest_source_metadata(
        self,
        source_metadata: VirtualWorkspaceSourceMetadataEntries,
    ) -> None:
        for virtual_path, metadata_fields in source_metadata.entries.items():
            self.record_source_metadata(virtual_path, metadata_fields)

    def record_source_metadata(
        self,
        virtual_path: str,
        metadata_fields: SourceMetadataMapping,
    ) -> None:
        normalized_metadata = MappingProxyType(dict(metadata_fields))
        self.source_metadata_by_path[virtual_path] = normalized_metadata
        self.source_metadata_by_path[str(self.plate_path / virtual_path)] = (
            normalized_metadata
        )

    def projection(self) -> VirtualWorkspaceSourceProjection:
        if not self.workspace_source_refs:
            raise RuntimeError(
                "virtual_workspace source binding resolution requires "
                "workspace_mapping entries in OpenHCS metadata."
            )
        return VirtualWorkspaceSourceProjection(
            source_refs_by_virtual_path=MappingProxyType(self.workspace_source_refs),
            source_metadata_by_path=MappingProxyType(self.source_metadata_by_path),
            source_projections_by_virtual_path=MappingProxyType(
                self.source_projections_by_virtual_path
            ),
            workspace_root=str(self.plate_path),
        )
