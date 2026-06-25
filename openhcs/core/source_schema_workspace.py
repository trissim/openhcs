"""Project typed source schemas into native OpenHCS virtual workspaces."""

from __future__ import annotations

import csv
import json
import os
import shutil
import urllib.request
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias
from urllib.parse import unquote, urlparse

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents, Backend, FileFormat
from openhcs.core.pipeline_image_schema import (
    ImageAssignment,
    ImagePlaneSource,
    ImageTypeSourceRole,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    PipelineImageSchema,
    SOURCE_SCHEMA_ORDERED_IMAGE_SET_COMPONENTS,
    SOURCE_IMAGE_TYPE_METADATA_FIELD,
    SourceAssignmentBase,
    SourceArtifactAssignment,
)
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.core.source_bindings import (
    SourceBindingMatchMethod,
    SourceSelector,
)
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
    SourceMetadataIdentityItems,
    SourceMetadataIdentityProjection,
    SourceMetadataMapping,
    SourceMetadataRoleView,
    SourceMetadataScalar,
    SourceMetadataValue,
)
from openhcs.core.source_matching import (
    is_image_path,
    merge_source_metadata,
    metadata_from_rules,
    normalize_source_metadata_key,
    source_filters_match,
    source_component_metadata_values,
    source_metadata_component,
    source_metadata_values_equal,
    source_metadata_value,
    with_source_component_metadata,
    with_original_source_metadata,
)
from openhcs.core.source_projection import SourcePixelRef
from openhcs.core.virtual_workspace_metadata import (
    OpenHCSMetadataSubdirectories,
    VirtualWorkspaceChannelLabels,
    VirtualWorkspaceMapping,
    VirtualWorkspaceSourceMetadataEntries,
)
from openhcs.microscopes.openhcs import (
    FIELDS,
    OpenHCSMetadata,
    workspace_mapping_source_ref,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR = "_source"
SOURCE_SCHEMA_WORKSPACE_PIXEL_SIZE = 1.0
SOURCE_SCHEMA_WORKSPACE_GRID_DIMENSIONS = [1, 1]
SOURCE_SCHEMA_WORKSPACE_SINGLETON_AXIS_VALUE = "A01"
SOURCE_SCHEMA_IMAGE_TYPE_METADATA_FIELD = SOURCE_IMAGE_TYPE_METADATA_FIELD
_AUXILIARY_PAYLOAD_CACHE_LIMIT = 64
_AUXILIARY_PAYLOAD_CACHE: OrderedDict[str, object] = OrderedDict()
_Z_INDEX_METADATA_KEYS = (
    "zindex",
    "z",
    "zplane",
    "zslice",
    "plane",
    "slice",
)
_SOURCE_PLANE_GROUP_KEY = "source_plane_group_key"

WorkspaceComponentValues: TypeAlias = Mapping[
    AllComponents,
    Mapping[str, str | None],
]
WorkspaceSourceMetadata: TypeAlias = Mapping[str, SourceMetadataMapping]
SourceSchemaCandidatesByAlias: TypeAlias = Mapping[
    str,
    tuple["SourceSchemaCandidate", ...],
]
ImportedMetadataJoinIndex: TypeAlias = Mapping[
    tuple[str, ...],
    tuple[Mapping[str, str], ...],
]
ImageSetCandidateGroups: TypeAlias = Mapping[
    tuple[str, ...],
    Mapping[str, "SourceSchemaCandidate"],
]
ImageSetMetadataByGroup: TypeAlias = Mapping[
    tuple[str, ...],
    Mapping[str, str],
]
WorkspaceMappingResult: TypeAlias = tuple[
    Mapping[str, Any],
    WorkspaceSourceMetadata,
    WorkspaceComponentValues,
]


class SourceSchemaCandidateDiscoveryMode(Enum):
    """Provider selection semantics for source-schema candidate discovery."""

    AUTO = "auto"
    LOCAL_FILES = "local_files"


def cache_source_schema_auxiliary_payload(path: str | Path, payload: object) -> None:
    """Cache an immutable materialized auxiliary payload by workspace path."""
    for cache_key in _auxiliary_payload_cache_keys(path):
        _AUXILIARY_PAYLOAD_CACHE[cache_key] = payload
        _AUXILIARY_PAYLOAD_CACHE.move_to_end(cache_key)
        if len(_AUXILIARY_PAYLOAD_CACHE) > _AUXILIARY_PAYLOAD_CACHE_LIMIT:
            _AUXILIARY_PAYLOAD_CACHE.popitem(last=False)


def source_schema_auxiliary_payload(path: str | Path) -> object | None:
    """Return a cached materialized auxiliary payload for a workspace path."""
    for cache_key in _auxiliary_payload_cache_keys(path):
        payload = _AUXILIARY_PAYLOAD_CACHE.get(cache_key)
        if payload is not None:
            _AUXILIARY_PAYLOAD_CACHE.move_to_end(cache_key)
            return payload
    return None


def _auxiliary_payload_cache_keys(path: str | Path) -> tuple[str, ...]:
    raw_key = str(path)
    resolved_key = str(Path(path).resolve()) if Path(path).is_absolute() else raw_key
    if resolved_key == raw_key:
        return (raw_key,)
    return (raw_key, resolved_key)


class SourceSchemaAuxiliaryMaterializer(ABC, metaclass=AutoRegisterMeta):
    """Normalize auxiliary source files during source-schema workspace creation."""

    __registry_key__ = "materializer_key"
    __skip_if_no_key__ = True
    materializer_key: ClassVar[str | None] = None

    @classmethod
    def for_path(cls, path: Path) -> "SourceSchemaAuxiliaryMaterializer | None":
        for materializer_cls in cls.__registry__.values():
            materializer = materializer_cls()
            if materializer.accepts_path(path):
                return materializer
        return None

    @abstractmethod
    def accepts_path(self, path: Path) -> bool:
        """Return whether this materializer owns the source path format."""

    @abstractmethod
    def materialize(
        self,
        request: "SourceSchemaAuxiliaryMaterializationRequest",
    ) -> Path:
        """Return the workspace-local source path used by virtual mappings."""


@dataclass(frozen=True, slots=True)
class SourceSchemaAuxiliaryMaterializationRequest:
    """IO context for materializing one auxiliary source into a workspace."""

    source_path: Path
    workspace_root: Path
    filemanager: FileManagerLike | None
    workspace_backend: str
    alias: str
    index: int

    @property
    def target_dir(self) -> Path:
        return self.workspace_root / SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR / self.alias

    def ensure_target_dir(self) -> None:
        if self.filemanager is None:
            self.target_dir.mkdir(parents=True, exist_ok=True)
            return
        self.filemanager.ensure_directory(str(self.target_dir), self.workspace_backend)

    def target_path(self) -> Path:
        return SourceSchemaAuxiliaryTargetPathPolicy.default_policy().target_path(
            SourceSchemaAuxiliaryTargetPathRequest(
                target_dir=self.target_dir,
                source_path=self.source_path,
                index=self.index,
            )
        )

    def save_payload(self, target_path: Path, payload: object) -> None:
        if self.filemanager is None:
            import numpy as np

            np.save(target_path, payload)
            return
        self.filemanager.save(payload, str(target_path), self.workspace_backend)


@dataclass(frozen=True, slots=True)
class SourceSchemaAuxiliaryTargetPathRequest:
    """Inputs for assigning a materialized auxiliary source path."""

    target_dir: Path
    source_path: Path
    index: int


class SourceSchemaAuxiliaryTargetPathPolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal policy for preserving auxiliary source identity in workspaces."""

    __registry_key__ = "policy_key"
    __skip_if_no_key__ = True
    policy_key: ClassVar[str | None] = None

    @classmethod
    def default_policy(cls) -> "SourceSchemaAuxiliaryTargetPathPolicy":
        return cls.__registry__["source_basename"]()

    @abstractmethod
    def target_path(self, request: SourceSchemaAuxiliaryTargetPathRequest) -> Path:
        """Return the workspace path for one auxiliary source."""


class SourceBasenameAuxiliaryTargetPathPolicy(SourceSchemaAuxiliaryTargetPathPolicy):
    """Keep the source basename as the load-bearing filter identity."""

    policy_key = "source_basename"

    def target_path(self, request: SourceSchemaAuxiliaryTargetPathRequest) -> Path:
        return request.target_dir / request.source_path.name


class NumpyAuxiliaryMaterializer(SourceSchemaAuxiliaryMaterializer):
    """Rewrite NumPy auxiliary files into current-format workspace files."""

    materializer_key = "numpy"

    def accepts_path(self, path: Path) -> bool:
        return path.suffix.lower() in FileFormat.NUMPY.value

    def materialize(
        self,
        request: SourceSchemaAuxiliaryMaterializationRequest,
    ) -> Path:
        import numpy as np
        from openhcs.core.memory import (
            MEMORY_TYPE_NUMPY,
            convert_memory,
            detect_memory_type,
        )

        request.ensure_target_dir()
        target_path = request.target_path()

        payload = np.load(request.source_path)
        source_memory_type = detect_memory_type(payload)
        payload = convert_memory(
            payload,
            source_memory_type,
            MEMORY_TYPE_NUMPY,
            gpu_id=0,
        )
        request.save_payload(target_path, payload)
        cache_source_schema_auxiliary_payload(request.source_path, payload)
        cache_source_schema_auxiliary_payload(target_path, payload)
        return target_path


@dataclass(frozen=True, slots=True)
class SourceSchemaWorkspaceMaterialization:
    """Result of projecting a source schema into an OpenHCS workspace."""

    source_root: Path
    workspace_root: Path
    metadata_path: Path
    primary_mappings: Mapping[str, Any]
    auxiliary_mappings: Mapping[str, Any]
    source_metadata: Mapping[str, Mapping[str, str]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "workspace_root", Path(self.workspace_root))
        object.__setattr__(self, "metadata_path", Path(self.metadata_path))
        object.__setattr__(
            self,
            "primary_mappings",
            MappingProxyType(dict(self.primary_mappings)),
        )
        object.__setattr__(
            self,
            "auxiliary_mappings",
            MappingProxyType(dict(self.auxiliary_mappings)),
        )
        object.__setattr__(
            self,
            "source_metadata",
            MappingProxyType(
                {
                    str(path): MappingProxyType(
                        {str(key): str(value) for key, value in metadata.items()}
                    )
                    for path, metadata in self.source_metadata.items()
                }
            ),
        )

    def primary_wells(self) -> tuple[str, ...]:
        """Return sample identities represented by primary image mappings."""
        parser = SourceSchemaFilenameParser()
        wells: dict[str, None] = {}
        for virtual_path in self.primary_mappings:
            parsed = parser.parse_filename(virtual_path)
            if parsed is None:
                raise ValueError(
                    f"Cannot parse source-schema virtual filename {virtual_path!r}."
                )
            wells[str(parsed["well"])] = None
        return tuple(wells)

    def source_paths_for_primary_wells(
        self,
        well_ids: Iterable[str],
        *,
        imported_metadata_tables: Iterable[ImportedMetadataTable] = (),
    ) -> tuple[Path, ...]:
        """Return the complete source universe required for selected wells.

        Primary mappings define the selected sample identities. Auxiliary mappings are
        source artifacts, not sample-defining image sets. Imported metadata tables are
        also runtime inputs for native CellProfiler. Both remain available to native
        consumers without broadening the selected sample identity.
        """
        selected_wells = set(str(well_id) for well_id in well_ids)
        if not selected_wells:
            raise ValueError("well_ids must contain at least one well.")
        parser = SourceSchemaFilenameParser()
        paths: list[Path] = []
        for virtual_path, real_path in self.primary_mappings.items():
            parsed = parser.parse_filename(virtual_path)
            if parsed is None:
                raise ValueError(
                    f"Cannot parse source-schema virtual filename {virtual_path!r}."
                )
            if str(parsed["well"]) in selected_wells:
                paths.append(
                    (
                        self.workspace_root / workspace_mapping_source_ref(real_path)
                    ).resolve()
                )
        paths.extend(
            (
                self.workspace_root / workspace_mapping_source_ref(real_path)
            ).resolve()
            for real_path in self.auxiliary_mappings.values()
        )
        paths.extend(
            ImportedMetadataPathResolver(self.source_root).path(table)
            for table in imported_metadata_tables
        )
        return tuple(dict.fromkeys(paths))


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidate:
    """One source file plus metadata extracted from source-schema rules."""

    path: Path
    relative_path: str
    metadata: SourceMetadataMapping
    source_plane_index: int | None = None
    source_plane_count: int | None = None
    source_filter_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "relative_path", self.relative_path.replace(os.sep, "/"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(
            self,
            "source_filter_paths",
            tuple(
                dict.fromkeys(path.replace(os.sep, "/") for path in self.source_filter_paths)
            ),
        )
        if self.source_plane_index is not None:
            object.__setattr__(self, "source_plane_index", int(self.source_plane_index))
        if self.source_plane_count is not None:
            object.__setattr__(self, "source_plane_count", int(self.source_plane_count))

    def source_filter_path_identities(self) -> tuple[str, ...]:
        """Return path identities that source-filter clauses may target."""
        if self.source_filter_paths:
            return self.source_filter_paths
        return (self.relative_path,)


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateIdentity:
    """Stable identity for source-schema candidate deduplication."""

    path: str
    relative_path: str
    metadata_items: SourceMetadataIdentityItems
    source_plane_index: int | None
    source_filter_paths: tuple[str, ...] = ()

    @classmethod
    def from_candidate(
        cls,
        candidate: SourceSchemaCandidate,
    ) -> "SourceSchemaCandidateIdentity":
        return cls(
            path=str(candidate.path),
            relative_path=candidate.relative_path,
            metadata_items=SourceMetadataIdentityProjection(candidate.metadata).items(),
            source_plane_index=candidate.source_plane_index,
            source_filter_paths=candidate.source_filter_paths,
        )


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateDiscoveryRequest:
    """Inputs for discovering typed source-schema candidates from a source tree."""

    source_root: Path
    source_files: tuple[Path, ...]
    schema: PipelineImageSchema
    discovery_mode: SourceSchemaCandidateDiscoveryMode = (
        SourceSchemaCandidateDiscoveryMode.AUTO
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(
            self,
            "source_files",
            tuple(Path(path) for path in self.source_files),
        )
        object.__setattr__(
            self,
            "discovery_mode",
            SourceSchemaCandidateDiscoveryMode(self.discovery_mode),
        )
        if not isinstance(self.schema, PipelineImageSchema):
            raise TypeError(
                "SourceSchemaCandidateDiscoveryRequest.schema must be a "
                f"PipelineImageSchema, got {type(self.schema).__name__}."
            )


class SourceSchemaCandidateProvider(ABC, metaclass=AutoRegisterMeta):
    """Nominal authority for source-schema candidate discovery."""

    __registry_key__ = "provider_key"
    __skip_if_no_key__ = True
    provider_key: ClassVar[str | None] = None

    @classmethod
    def provider_types_by_mro(cls) -> tuple[type["SourceSchemaCandidateProvider"], ...]:
        registered = set(cls.__registry__.values())
        ordered: list[type[SourceSchemaCandidateProvider]] = []
        seen: set[type[SourceSchemaCandidateProvider]] = set()

        def visit(owner: type[SourceSchemaCandidateProvider]) -> None:
            for child in owner.__subclasses__():
                visit(child)
            if owner in registered and owner not in seen:
                ordered.append(owner)
                seen.add(owner)

        visit(cls)
        return tuple(ordered)

    @abstractmethod
    def available(self, request: SourceSchemaCandidateDiscoveryRequest) -> bool:
        """Return whether this provider owns candidate discovery for the request."""

    @abstractmethod
    def candidates(
        self,
        request: SourceSchemaCandidateDiscoveryRequest,
    ) -> tuple["SourceSchemaCandidate", ...]:
        """Return source-schema candidates for the request."""


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateMetadataRequest:
    """Inputs for deriving one source-schema candidate's metadata."""

    source_root: Path
    schema: PipelineImageSchema
    path: Path
    relative_path: str
    imported_metadata: tuple["ImportedMetadataRows", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "relative_path", self.relative_path.replace(os.sep, "/"))
        if not isinstance(self.schema, PipelineImageSchema):
            raise TypeError(
                "SourceSchemaCandidateMetadataRequest.schema must be a "
                f"PipelineImageSchema, got {type(self.schema).__name__}."
            )


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateMetadataResolver:
    """Derive source-schema candidate metadata from rules and imported tables."""

    request: SourceSchemaCandidateMetadataRequest

    def metadata(self) -> Mapping[str, str]:
        metadata = metadata_from_rules(
            str(self.request.path),
            self.request.schema.metadata_rules,
            filter_path=self.request.relative_path,
        )
        metadata = _metadata_with_imported_tables(
            metadata,
            self.request.imported_metadata,
            path=self.request.relative_path,
        )
        return MappingProxyType(metadata)


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateDiscovery:
    """Discover source files admitted by a typed source schema."""

    request: SourceSchemaCandidateDiscoveryRequest

    def candidates(self) -> tuple[SourceSchemaCandidate, ...]:
        if (
            self.request.discovery_mode
            is SourceSchemaCandidateDiscoveryMode.LOCAL_FILES
        ):
            return LocalFileSourceSchemaCandidateProvider().candidates(self.request)
        unusable_errors: list[str] = []
        for provider_type in SourceSchemaCandidateProvider.provider_types_by_mro():
            provider = provider_type()
            if provider.available(self.request):
                candidates = provider.candidates(self.request)
                image_set_probe = self._image_set_probe(provider, candidates)
                if image_set_probe.usable:
                    return candidates
                unusable_errors.append(
                    f"{provider.provider_key}: {image_set_probe.error_message}"
                )
        if unusable_errors:
            details = "; ".join(unusable_errors)
            raise RuntimeError(
                "No source-schema candidate provider could satisfy "
                f"{self.request.source_root}: {details}"
            )
        raise RuntimeError(
            f"No source-schema candidate provider available for {self.request.source_root}."
        )

    def _image_set_probe(
        self,
        provider: SourceSchemaCandidateProvider,
        candidates: tuple["SourceSchemaCandidate", ...],
    ) -> "SourceSchemaImageSetProbeResult":
        try:
            image_set_count = SourceSchemaCandidateImageSetViability(
                self.request.schema,
                candidates,
            ).image_set_count()
        except Exception as exc:
            return SourceSchemaImageSetProbeResult(
                source_root=self.request.source_root,
                image_set_count=0,
                error_message=(
                    f"{provider.provider_key} candidates do not satisfy schema: {exc}"
                ),
            )
        return SourceSchemaImageSetProbeResult(
            source_root=self.request.source_root,
            image_set_count=image_set_count,
            error_message=None,
        )


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateImageSetViability:
    """Evaluate whether discovered candidates can assemble the schema."""

    schema: PipelineImageSchema
    candidates: tuple["SourceSchemaCandidate", ...]

    def image_set_count(self) -> int:
        candidate_matches = SourceSchemaCandidateMatches(self.candidates, self.schema)
        stack_assignments = candidate_matches.stack_assignments()
        if stack_assignments:
            stack_candidates = candidate_matches.stack_candidates()
            return len(
                ImageSetAssembler.for_schema(self.schema).image_sets(
                    self.schema,
                    stack_candidates,
                )
            )
        auxiliary_assignments = candidate_matches.auxiliary_assignments()
        if auxiliary_assignments:
            return len(candidate_matches.auxiliary_candidates())
        return len(self.candidates)


@dataclass(frozen=True, slots=True)
class SourceSchemaImageSetProbeResult:
    """Non-mutating viability result for one source-schema source root."""

    source_root: Path
    image_set_count: int
    error_message: str | None = None

    @property
    def usable(self) -> bool:
        return self.error_message is None and self.image_set_count > 0


@dataclass(frozen=True, slots=True)
class SourceSchemaImageSetProbe:
    """Check whether source files can assemble schema image sets without writing."""

    source_root: Path
    source_files: tuple[Path, ...]
    schema: PipelineImageSchema
    discovery_mode: SourceSchemaCandidateDiscoveryMode = (
        SourceSchemaCandidateDiscoveryMode.AUTO
    )

    def result(self) -> SourceSchemaImageSetProbeResult:
        try:
            candidates = SourceSchemaCandidateDiscovery(
                SourceSchemaCandidateDiscoveryRequest(
                    self.source_root,
                    self.source_files,
                    self.schema,
                    self.discovery_mode,
                )
            ).candidates()
            candidate_matches = SourceSchemaCandidateMatches(candidates, self.schema)
            stack_candidates = candidate_matches.stack_candidates()
            image_sets = ImageSetAssembler.for_schema(self.schema).image_sets(
                self.schema,
                stack_candidates,
            )
        except Exception as exc:
            return SourceSchemaImageSetProbeResult(
                source_root=self.source_root,
                image_set_count=0,
                error_message=str(exc),
            )
        return SourceSchemaImageSetProbeResult(
            source_root=self.source_root,
            image_set_count=len(image_sets),
            error_message=None,
        )


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateMatches:
    """Match discovered source candidates against schema assignments."""

    candidates: tuple["SourceSchemaCandidate", ...]
    schema: PipelineImageSchema

    def stack_assignments(self) -> tuple[ImageAssignment, ...]:
        return tuple(
            assignment
            for assignment in self.schema.assignments_by_alias.values()
            if ImageTypeSourceRole.for_image_type(
                assignment.image_type
            ).participates_in_image_stack
        )

    def auxiliary_assignments(self) -> tuple[SourceAssignmentBase, ...]:
        auxiliary_assignments = tuple(
            assignment
            for assignment in self.schema.assignments_by_alias.values()
            if not ImageTypeSourceRole.for_image_type(
                assignment.image_type
            ).participates_in_image_stack
        )
        return auxiliary_assignments + tuple(
            self.schema.source_artifacts_by_alias.values()
        )

    def stack_candidates(self) -> SourceSchemaCandidatesByAlias:
        return self._candidates_by_alias(
            self.stack_assignments(),
            require_match=True,
        )

    def auxiliary_candidates(self) -> SourceSchemaCandidatesByAlias:
        return self._candidates_by_alias(
            self.auxiliary_assignments(),
            require_match=False,
        )

    def _candidates_by_alias(
        self,
        assignments: tuple[SourceAssignmentBase, ...],
        *,
        require_match: bool,
    ) -> SourceSchemaCandidatesByAlias:
        matched: dict[str, tuple[SourceSchemaCandidate, ...]] = {}
        for assignment in assignments:
            alias_candidates = tuple(
                candidate
                for candidate in self.candidates
                if _candidate_matches_selector(candidate, assignment.selector)
            )
            image_candidates = tuple(
                candidate
                for candidate in alias_candidates
                if is_image_path(str(candidate.path))
            )
            selected_candidates = image_candidates if require_match else alias_candidates
            if require_match and not selected_candidates:
                raise ValueError(
                    f"Source schema image alias {assignment.alias!r} matched no image files."
                )
            if selected_candidates:
                matched[assignment.alias] = selected_candidates
        return MappingProxyType(matched)


class OpenHCSWorkspaceSourceSchemaCandidateProvider(SourceSchemaCandidateProvider):
    """Discover candidates from existing OpenHCS virtual-workspace metadata."""

    provider_key = "openhcs_workspace"

    def available(self, request: SourceSchemaCandidateDiscoveryRequest) -> bool:
        return self.metadata_path(request).exists()

    def candidates(
        self,
        request: SourceSchemaCandidateDiscoveryRequest,
    ) -> tuple[SourceSchemaCandidate, ...]:
        payload = json.loads(self.metadata_path(request).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"OpenHCS metadata {self.metadata_path(request)} must be a mapping."
            )
        candidates: list[SourceSchemaCandidate] = []
        for _, subdirectory in OpenHCSMetadataSubdirectories(payload).items():
            workspace_mapping = VirtualWorkspaceMapping.from_subdirectory(subdirectory)
            source_metadata = VirtualWorkspaceSourceMetadataEntries.from_subdirectory(
                subdirectory
            )
            channels = VirtualWorkspaceChannelLabels.from_subdirectory(subdirectory)
            for virtual_path in workspace_mapping.entries:
                candidate = self.candidate_for_virtual_path(
                    request,
                    str(virtual_path),
                    workspace_mapping,
                    source_metadata,
                    channels,
                )
                if self.schema_admits_candidate(request.schema, candidate):
                    candidates.append(candidate)
        return tuple(candidates)

    @staticmethod
    def metadata_path(request: SourceSchemaCandidateDiscoveryRequest) -> Path:
        return request.source_root / "openhcs_metadata.json"

    @staticmethod
    def schema_admits_path(schema: PipelineImageSchema, relative_path: str) -> bool:
        images_rule = schema.images_rule
        if images_rule is None:
            return True
        return source_filters_match(relative_path, images_rule.filters)

    @staticmethod
    def schema_admits_candidate(
        schema: PipelineImageSchema,
        candidate: SourceSchemaCandidate,
    ) -> bool:
        images_rule = schema.images_rule
        if images_rule is None:
            return True
        return any(
            source_filters_match(path, images_rule.filters)
            for path in candidate.source_filter_path_identities()
        )

    def candidate_for_virtual_path(
        self,
        request: SourceSchemaCandidateDiscoveryRequest,
        virtual_path: str,
        workspace_mapping: VirtualWorkspaceMapping,
        source_metadata: VirtualWorkspaceSourceMetadataEntries,
        channels: VirtualWorkspaceChannelLabels,
    ) -> SourceSchemaCandidate:
        enriched = dict(
            source_schema_metadata_with_virtual_components(
                virtual_path,
                source_metadata.metadata_for(virtual_path),
            )
        )
        for channel_value in source_component_metadata_values(
            enriched,
            AllComponents.CHANNEL,
        ):
            channel_label = channels.label_for(channel_value)
            if channel_label is None:
                continue
            if "channel_name" not in enriched:
                enriched["channel_name"] = channel_label
            if "channel_label" not in enriched:
                enriched["channel_label"] = channel_label
        source_ref = workspace_mapping_source_ref(
            workspace_mapping.require_source_ref(virtual_path)
        )
        return SourceSchemaCandidate(
            path=request.source_root / source_ref,
            relative_path=virtual_path,
            metadata=enriched,
            source_filter_paths=(source_ref,),
        )


class LocalFileSourceSchemaCandidateProvider(SourceSchemaCandidateProvider):
    """Discover candidates from local/VFS-visible source files."""

    provider_key = "local_files"

    def available(self, request: SourceSchemaCandidateDiscoveryRequest) -> bool:
        del request
        return True

    def candidates(
        self,
        request: SourceSchemaCandidateDiscoveryRequest,
    ) -> tuple[SourceSchemaCandidate, ...]:
        imported_metadata = _imported_metadata_rows(
            request.source_root,
            request.schema,
        )
        candidates: list[SourceSchemaCandidate] = []
        for path in request.source_files:
            relative_path = path.relative_to(request.source_root).as_posix()
            if not self._schema_admits_path(request.schema, relative_path):
                continue
            candidate = SourceSchemaCandidate(
                path=path,
                relative_path=relative_path,
                metadata=SourceSchemaCandidateMetadataResolver(
                    SourceSchemaCandidateMetadataRequest(
                        request.source_root,
                        request.schema,
                        path,
                        relative_path,
                        imported_metadata,
                    )
                ).metadata(),
            )
            candidates.extend(
                SourceSchemaSourcePlaneInventory.for_path(path).candidates(candidate)
            )
        return tuple(candidates)

    @staticmethod
    def _schema_admits_path(schema: PipelineImageSchema, relative_path: str) -> bool:
        images_rule = schema.images_rule
        if images_rule is None:
            return True
        return source_filters_match(relative_path, images_rule.filters)


class SourceSchemaSourcePlaneInventory(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for exposing source files as OpenHCS-addressable planes."""

    __registry_key__ = "inventory_key"
    __skip_if_no_key__ = True
    inventory_key: ClassVar[str | None] = None

    @classmethod
    def for_path(cls, path: Path) -> "SourceSchemaSourcePlaneInventory":
        for inventory_type in cls.inventory_types_by_mro():
            inventory = inventory_type()
            if inventory.accepts_path(path):
                return inventory
        raise ValueError(f"No source-plane inventory accepted source path: {path}")

    @classmethod
    def inventory_types_by_mro(
        cls,
    ) -> tuple[type["SourceSchemaSourcePlaneInventory"], ...]:
        registered = set(cls.__registry__.values())
        ordered: list[type[SourceSchemaSourcePlaneInventory]] = []
        seen: set[type[SourceSchemaSourcePlaneInventory]] = set()

        def visit(owner: type[SourceSchemaSourcePlaneInventory]) -> None:
            for child in owner.__subclasses__():
                visit(child)
            if owner in registered and owner not in seen:
                ordered.append(owner)
                seen.add(owner)

        visit(cls)
        return tuple(ordered)

    @abstractmethod
    def accepts_path(self, path: Path) -> bool:
        """Return whether this inventory can inspect the source path."""

    @abstractmethod
    def candidates(
        self,
        candidate: SourceSchemaCandidate,
    ) -> tuple[SourceSchemaCandidate, ...]:
        """Return one or more plane-addressable candidates for a source file."""


class TiffPageSourcePlaneInventory(SourceSchemaSourcePlaneInventory):
    """Expose multi-page TIFF sources as individual source-plane candidates."""

    inventory_key = "tiff_pages"

    def accepts_path(self, path: Path) -> bool:
        name = path.name.lower()
        return path.suffix.lower() in {".tif", ".tiff"} or name.endswith(
            (".ome.tif", ".ome.tiff")
        )

    def candidates(
        self,
        candidate: SourceSchemaCandidate,
    ) -> tuple[SourceSchemaCandidate, ...]:
        inventory = TiffSourcePlaneInventory.from_path(candidate.path)
        if inventory.plane_count <= 1:
            return (candidate,)
        return tuple(
            SourceSchemaCandidate(
                path=candidate.path,
                relative_path=candidate.relative_path,
                metadata={
                    **dict(candidate.metadata),
                    AllComponents.Z_INDEX.value: str(plane_index + 1),
                    _SOURCE_PLANE_GROUP_KEY: candidate.relative_path,
                    SOURCE_PLANE_INDEX_FIELD: str(plane_index),
                    SOURCE_PLANE_COUNT_FIELD: str(inventory.plane_count),
                },
                source_plane_index=plane_index,
                source_plane_count=inventory.plane_count,
            )
            for plane_index in range(inventory.plane_count)
        )


class SinglePlaneSourcePlaneInventory(SourceSchemaSourcePlaneInventory):
    """Default inventory for source files that are already one logical plane."""

    inventory_key = "single_plane"

    def accepts_path(self, path: Path) -> bool:
        del path
        return True

    def candidates(
        self,
        candidate: SourceSchemaCandidate,
    ) -> tuple[SourceSchemaCandidate, ...]:
        return (candidate,)


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateCollection:
    """Deduplicated ordered source-schema candidates."""

    candidates: tuple[SourceSchemaCandidate, ...]

    def __post_init__(self) -> None:
        deduplicated: list[SourceSchemaCandidate] = []
        seen: set[SourceSchemaCandidateIdentity] = set()
        for candidate in self.candidates:
            if not isinstance(candidate, SourceSchemaCandidate):
                raise TypeError(
                    "SourceSchemaCandidateCollection values must be "
                    f"SourceSchemaCandidate, got {type(candidate).__name__}."
                )
            key = SourceSchemaCandidateIdentity.from_candidate(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(candidate)
        object.__setattr__(self, "candidates", tuple(deduplicated))

    @classmethod
    def merge(
        cls,
        *candidate_groups: Iterable[SourceSchemaCandidate],
    ) -> "SourceSchemaCandidateCollection":
        return cls(
            tuple(
                candidate
                for candidate_group in candidate_groups
                for candidate in candidate_group
            )
        )


class ImagePlaneSourceResolutionStage(IntEnum):
    """Ordered source-resolution stages for explicit image-plane sources."""

    LOCAL_EQUIVALENT = 10
    DECLARED_FILE = 20
    MATERIALIZED_REMOTE = 30


class ImagePlaneSourceResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for resolving explicit image-plane sources into candidates."""

    __registry_key__ = "resolver_key"
    __skip_if_no_key__ = True
    resolver_key: ClassVar[str | None] = None
    stage: ClassVar[ImagePlaneSourceResolutionStage]

    @classmethod
    def resolve(
        cls,
        source: ImagePlaneSource,
        *,
        source_index: int,
        workspace_root: Path,
        schema: PipelineImageSchema,
        local_candidates: tuple[SourceSchemaCandidate, ...],
    ) -> SourceSchemaCandidate:
        source_metadata = metadata_from_rules(source.uri, schema.metadata_rules)
        for resolver_type in sorted(
            cls.__registry__.values(),
            key=lambda registered_type: registered_type.stage,
        ):
            candidate = resolver_type().candidate(
                source,
                source_index=source_index,
                workspace_root=workspace_root,
                source_metadata=source_metadata,
                local_candidates=local_candidates,
            )
            if candidate is not None:
                return candidate
        raise ValueError(f"Could not resolve image-plane source URI {source.uri!r}.")

    @abstractmethod
    def candidate(
        self,
        source: ImagePlaneSource,
        *,
        source_index: int,
        workspace_root: Path,
        source_metadata: Mapping[str, str],
        local_candidates: tuple[SourceSchemaCandidate, ...],
    ) -> SourceSchemaCandidate | None:
        """Return the resolved source candidate or None if this resolver does not apply."""


class LocalEquivalentImagePlaneSourceResolver(ImagePlaneSourceResolver):
    """Resolve embedded URI sources to already-visible local files when equivalent."""

    resolver_key = "local_equivalent"
    stage = ImagePlaneSourceResolutionStage.LOCAL_EQUIVALENT

    def candidate(
        self,
        source: ImagePlaneSource,
        *,
        source_index: int,
        workspace_root: Path,
        source_metadata: Mapping[str, str],
        local_candidates: tuple[SourceSchemaCandidate, ...],
    ) -> SourceSchemaCandidate | None:
        del source_index, workspace_root
        source_name = _image_plane_source_basename(source)
        matches = tuple(
            candidate
            for candidate in local_candidates
            if candidate.path.name == source_name
            and _source_metadata_matches(candidate.metadata, source_metadata)
        )
        if len(matches) > 1:
            raise ValueError(
                f"Image-plane source {source.uri!r} matched multiple local "
                f"source files: {[match.relative_path for match in matches]}"
            )
        return matches[0] if matches else None


class DeclaredFileImagePlaneSourceResolver(ImagePlaneSourceResolver):
    """Resolve explicit local file image-plane sources."""

    resolver_key = "declared_file"
    stage = ImagePlaneSourceResolutionStage.DECLARED_FILE

    def candidate(
        self,
        source: ImagePlaneSource,
        *,
        source_index: int,
        workspace_root: Path,
        source_metadata: Mapping[str, str],
        local_candidates: tuple[SourceSchemaCandidate, ...],
    ) -> SourceSchemaCandidate | None:
        del source_index, workspace_root, local_candidates
        parsed = urlparse(source.uri)
        if parsed.scheme not in ("", "file"):
            return None
        path = Path(unquote(parsed.path if parsed.scheme == "file" else source.uri))
        return SourceSchemaCandidate(
            path=path,
            relative_path=source.uri,
            metadata=source_metadata,
        )


class RemoteMaterializedImagePlaneSourceResolver(ImagePlaneSourceResolver):
    """Materialize remote image-plane sources when no local equivalent exists."""

    resolver_key = "remote_materialized"
    stage = ImagePlaneSourceResolutionStage.MATERIALIZED_REMOTE

    def candidate(
        self,
        source: ImagePlaneSource,
        *,
        source_index: int,
        workspace_root: Path,
        source_metadata: Mapping[str, str],
        local_candidates: tuple[SourceSchemaCandidate, ...],
    ) -> SourceSchemaCandidate | None:
        del local_candidates
        parsed = urlparse(source.uri)
        if parsed.scheme not in ("http", "https"):
            return None
        target_dir = (
            workspace_root
            / SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR
            / "image_planes"
            / f"{source_index:03d}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / Path(unquote(parsed.path)).name
        with urllib.request.urlopen(source.uri) as response, target_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return SourceSchemaCandidate(
            path=target_path,
            relative_path=source.uri,
            metadata=source_metadata,
        )


@dataclass(frozen=True, slots=True)
class ImportedMetadataRows:
    """Rows loaded from one pipeline-level imported metadata table."""

    table: ImportedMetadataTable
    rows: tuple[Mapping[str, str], ...]
    rows_by_join_key: ImportedMetadataJoinIndex = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.table, ImportedMetadataTable):
            raise TypeError(
                "ImportedMetadataRows.table must be ImportedMetadataTable, "
                f"got {type(self.table).__name__}."
            )
        object.__setattr__(
            self,
            "rows",
            tuple(MappingProxyType(dict(row)) for row in self.rows),
        )
        if not self.rows:
            raise ValueError("Imported metadata tables must contain at least one row.")
        object.__setattr__(
            self,
            "rows_by_join_key",
            _indexed_imported_metadata_rows(self.rows, self.table.joins),
        )


@dataclass(frozen=True, slots=True)
class ImageSetRecord:
    """One projected OpenHCS image set keyed by source-schema match metadata."""

    index: int
    candidates_by_alias: Mapping[str, SourceSchemaCandidate]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidates_by_alias",
            MappingProxyType(dict(self.candidates_by_alias)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SourceSchemaVirtualFilename:
    """OpenHCS component projection for one source-schema virtual filename."""

    well: object
    site: object
    channel: object
    z_index: object = 1
    timepoint: object = 1
    extension: str = ""

    @classmethod
    def from_parsed(
        cls,
        parsed: Mapping[str, object],
        *,
        well: object | None = None,
        site: object | None = None,
        extension: object | None = None,
    ) -> "SourceSchemaVirtualFilename":
        return cls(
            parsed["well"] if well is None else well,
            parsed["site"] if site is None else site,
            parsed["channel"],
            parsed["z_index"],
            parsed["timepoint"],
            "" if extension is None else str(extension),
        )


@dataclass(frozen=True, slots=True)
class SourceSchemaFilenameProjection:
    """Authoritative formatter for source-schema virtual filenames."""

    parser: SourceSchemaFilenameParser = field(
        default_factory=SourceSchemaFilenameParser,
    )

    def filename(self, virtual_filename: SourceSchemaVirtualFilename) -> str:
        return self.parser.construct_filename(
            well=virtual_filename.well,
            site=virtual_filename.site,
            channel=virtual_filename.channel,
            z_index=virtual_filename.z_index,
            timepoint=virtual_filename.timepoint,
            extension=virtual_filename.extension,
        )

    def path_for_well_site(
        self,
        virtual_path: str,
        parsed: Mapping[str, object],
        well_id: str,
        site: object,
    ) -> str:
        path = PurePosixPath(virtual_path)
        original_name = path.name
        original_stem = self.filename(SourceSchemaVirtualFilename.from_parsed(parsed))
        expanded_stem = self.filename(
            SourceSchemaVirtualFilename.from_parsed(parsed, well=well_id, site=site)
        )
        expanded_name = (
            f"{expanded_stem}{original_name[len(original_stem):]}"
            if original_name.startswith(original_stem)
            else self.filename(
                SourceSchemaVirtualFilename.from_parsed(
                    parsed,
                    well=well_id,
                    site=site,
                    extension=parsed["extension"],
                )
            )
        )
        return str(path.with_name(expanded_name))


@dataclass(slots=True)
class WorkspaceMappingSink:
    """Conflict-aware accumulator for virtual workspace mappings."""

    mappings: dict[str, Any]

    def add(self, virtual_path: str, source_ref: Any) -> None:
        existing = self.mappings.get(virtual_path)
        if existing is not None and existing != source_ref:
            raise ValueError(
                f"Conflicting source workspace mapping for {virtual_path!r}: "
                f"{existing!r} != {source_ref!r}."
            )
        self.mappings[virtual_path] = source_ref


@dataclass(frozen=True, slots=True)
class SourceAssignmentImageTypeProjection:
    """Project source-assignment metadata into an optional image-type value."""

    assignment: SourceAssignmentBase | None

    def image_type(self) -> str | None:
        if self.assignment is None:
            return None
        if isinstance(self.assignment, ImageAssignment):
            return self.assignment.image_type
        if isinstance(self.assignment, SourceArtifactAssignment):
            return self.assignment.payload_type or None
        return None


@dataclass(frozen=True, slots=True)
class SourceVirtualPathMetadata:
    """Merge source metadata for one materialized virtual workspace path."""

    image_set_metadata: Mapping[str, str]
    candidate_metadata: Mapping[str, str]
    virtual_path: str | None = None
    assignment: SourceAssignmentBase | None = None

    def metadata(self) -> SourceMetadataMapping:
        metadata = dict(self.image_set_metadata)
        metadata.pop(_SOURCE_PLANE_GROUP_KEY, None)
        merge_source_metadata(metadata, self.candidate_metadata, path="source_metadata")
        metadata.pop(_SOURCE_PLANE_GROUP_KEY, None)
        if self.virtual_path is not None:
            metadata = dict(source_schema_metadata_with_virtual_components(
                self.virtual_path,
                metadata,
            ))
        image_type = SourceAssignmentImageTypeProjection(self.assignment).image_type()
        if image_type is not None:
            merge_source_metadata(
                metadata,
                {SOURCE_SCHEMA_IMAGE_TYPE_METADATA_FIELD: image_type},
                path="source_metadata",
            )
        return MappingProxyType(metadata)


@dataclass(frozen=True, slots=True)
class VirtualComponentOriginalMetadataProjection:
    """Project hidden source-literal component fields into the reserved role."""

    metadata: SourceMetadataMapping
    component: AllComponents
    canonical_value: str

    def fields(self) -> Mapping[str, SourceMetadataScalar]:
        canonical_key = normalize_source_metadata_key(self.component.value)
        original_fields: dict[str, SourceMetadataScalar] = {}
        for field_name, value in SourceMetadataRoleView(self.metadata).scalar_items():
            if source_metadata_component(field_name) is not self.component:
                continue
            field_is_canonical = normalize_source_metadata_key(field_name) == canonical_key
            if field_is_canonical and source_metadata_values_equal(
                str(value),
                self.canonical_value,
            ):
                continue
            original_fields[field_name] = value
        return MappingProxyType(original_fields)


@dataclass(frozen=True, slots=True)
class SourceMetadataJsonRecord:
    """JSON-ready representation of one source metadata record."""

    metadata: SourceMetadataMapping

    def as_dict(self) -> dict[str, SourceMetadataValue]:
        return {
            key: dict(value) if isinstance(value, Mapping) else value
            for key, value in self.metadata.items()
        }


@dataclass(slots=True)
class SourcePlaneGroupSiteAllocator:
    """Assign stable OpenHCS sites to source-plane stack groups."""

    sites_by_group: dict[str, int] = field(default_factory=dict)

    def site_component(
        self,
        metadata: Mapping[str, str],
        fallback_site: int | str,
    ) -> int | str:
        group_key = metadata.get(_SOURCE_PLANE_GROUP_KEY)
        if group_key is None:
            return fallback_site
        site_index = self.sites_by_group.setdefault(
            group_key,
            len(self.sites_by_group) + 1,
        )
        return site_index


def source_schema_metadata_with_virtual_components(
    virtual_path: str,
    metadata: SourceMetadataMapping,
) -> SourceMetadataMapping:
    """Return metadata enriched with canonical components from a virtual path."""
    parsed = SourceSchemaFilenameParser().parse_filename(virtual_path)
    if parsed is None:
        return MappingProxyType(dict(metadata))

    enriched = dict(metadata)
    for component in AllComponents:
        value = parsed.get(component.value)
        if value is not None:
            original_fields = VirtualComponentOriginalMetadataProjection(
                enriched,
                component,
                str(value),
            ).fields()
            if original_fields:
                enriched = with_original_source_metadata(
                    enriched,
                    original_fields,
                    path=virtual_path,
                )
            enriched = with_source_component_metadata(enriched, component, value)
    extension = parsed.get("extension")
    if extension is not None:
        enriched.setdefault("extension", str(extension))
    return MappingProxyType(enriched)


@dataclass(frozen=True, slots=True)
class ImageSetMetadataMerge:
    """Merge shared candidate metadata into one image-set metadata record."""

    group_metadata: Mapping[str, str]
    candidates: tuple[SourceSchemaCandidate, ...]

    @classmethod
    def from_candidates(
        cls,
        group_metadata: Mapping[str, str],
        candidates: Iterable[SourceSchemaCandidate],
    ) -> "ImageSetMetadataMerge":
        return cls(group_metadata, tuple(candidates))

    def metadata(self) -> SourceMetadataMapping:
        merged = dict(self.group_metadata)
        merge_source_metadata(
            merged,
            SharedCandidateMetadataProjection(self.candidates).metadata(),
            path="image_set",
        )
        merge_source_metadata(
            merged,
            _projected_candidate_components(merged, self.candidates),
            path="image_set",
        )
        return MappingProxyType(merged)


@dataclass(frozen=True, slots=True)
class SharedCandidateMetadataProjection:
    """Project scalar metadata shared by every candidate in one image set."""

    candidates: tuple[SourceSchemaCandidate, ...]

    def metadata(self) -> SourceMetadataMapping:
        value_sets_by_key: dict[str, set[SourceMetadataScalar]] = {}
        counts_by_key: dict[str, int] = {}
        for candidate in self.candidates:
            for key, value in SourceMetadataRoleView(candidate.metadata).scalar_items():
                value_sets_by_key.setdefault(key, set()).add(value)
                counts_by_key[key] = counts_by_key.get(key, 0) + 1
        candidate_count = len(self.candidates)
        return MappingProxyType(
            {
                key: next(iter(values))
                for key, values in value_sets_by_key.items()
                if counts_by_key[key] == candidate_count and len(values) == 1
            }
        )


@dataclass(frozen=True, slots=True)
class SourceSchemaImageSetSelection:
    """Nominal source-schema sample selection before workspace materialization."""

    well_filter: tuple[str, ...] = ()
    max_image_set_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "well_filter",
            tuple(dict.fromkeys(str(well) for well in self.well_filter)),
        )
        if self.max_image_set_count is not None and self.max_image_set_count <= 0:
            raise ValueError("max_image_set_count must be positive.")

    def apply(
        self,
        schema: PipelineImageSchema,
        image_sets: tuple[ImageSetRecord, ...],
    ) -> tuple[ImageSetRecord, ...]:
        """Return image sets selected by source-schema well identity."""
        selected = image_sets
        if self.well_filter:
            requested = set(self.well_filter)
            selected = tuple(
                image_set
                for image_set in selected
                if SourceSchemaImageSetIdentity(schema, image_set).well in requested
            )
            selected_wells = {
                SourceSchemaImageSetIdentity(schema, image_set).well
                for image_set in selected
            }
            missing = tuple(
                well for well in self.well_filter if well not in selected_wells
            )
            if missing:
                raise ValueError(
                    "Requested source-schema wells are not available: "
                    + ", ".join(missing)
                )
        if self.max_image_set_count is not None:
            selected = self._limit_by_sample_count(
                schema,
                selected,
                self.max_image_set_count,
            )
        if not selected:
            raise ValueError("Source-schema image-set selection resolved to no images.")
        return selected

    def _limit_by_sample_count(
        self,
        schema: PipelineImageSchema,
        image_sets: tuple[ImageSetRecord, ...],
        max_sample_count: int,
    ) -> tuple[ImageSetRecord, ...]:
        """Keep all image sets belonging to the first selected sample identities."""
        selected_wells: dict[str, None] = {}
        for image_set in image_sets:
            well = SourceSchemaImageSetIdentity(schema, image_set).well
            selected_wells.setdefault(well, None)
            if len(selected_wells) >= max_sample_count:
                break
        return tuple(
            image_set
            for image_set in image_sets
            if SourceSchemaImageSetIdentity(schema, image_set).well in selected_wells
        )


@dataclass(frozen=True, slots=True)
class SourceSchemaImageSetIdentity:
    """Projected OpenHCS component identity for one source-schema image set."""

    schema: PipelineImageSchema
    image_set: ImageSetRecord

    @property
    def well(self) -> str:
        grouping = self.schema.grouping
        if grouping is None or not grouping.metadata_fields:
            return ComponentProjection.resolve(
                AllComponents.WELL,
                self.image_set.metadata,
                self.image_set.index,
            )

        group_metadata = _grouping_metadata(
            self.image_set.metadata,
            grouping.metadata_fields,
        )
        if _grouping_fields_project_to_well(grouping.metadata_fields):
            projected = ComponentProjection.resolve_from_metadata(
                AllComponents.WELL,
                group_metadata,
            )
            if projected is not None:
                return projected
        return _grouping_axis_value(group_metadata, grouping.metadata_fields)


class ComponentProjection(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for projecting source metadata onto OpenHCS components."""

    __registry_key__ = "__name__"
    component: ClassVar[AllComponents | None] = None
    metadata_derived: ClassVar[bool] = True

    @classmethod
    def resolve(
        cls,
        component: AllComponents,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str:
        direct_value = cls.direct_metadata_value(component, metadata)
        if direct_value is not None:
            return direct_value
        metadata_value = cls.resolve_from_registered_projections(
            component,
            metadata,
            image_set_index,
            metadata_derived=True,
        )
        if metadata_value is not None:
            return metadata_value
        fallback_value = cls.resolve_from_registered_projections(
            component,
            metadata,
            image_set_index,
            metadata_derived=False,
        )
        if fallback_value is not None:
            return fallback_value
        raise ValueError(
            f"Could not project source metadata fields {sorted(metadata)} "
            f"onto OpenHCS component {component.value!r}."
        )

    @classmethod
    def resolve_from_metadata(
        cls,
        component: AllComponents,
        metadata: Mapping[str, str],
    ) -> str | None:
        direct_value = cls.direct_metadata_value(component, metadata)
        if direct_value is not None:
            return direct_value
        return cls.resolve_from_registered_projections(
            component,
            metadata,
            0,
            metadata_derived=True,
        )

    @classmethod
    def resolve_from_registered_projections(
        cls,
        component: AllComponents,
        metadata: Mapping[str, str],
        image_set_index: int,
        *,
        metadata_derived: bool,
    ) -> str | None:
        matches: list[tuple[type[ComponentProjection], str]] = []
        for projection_type in cls.projection_types_by_mro(
            component,
            metadata_derived=metadata_derived,
        ):
            value = projection_type().value(metadata, image_set_index)
            if value is not None:
                matches.append((projection_type, value))
        if not matches:
            return None
        value = matches[0][1]
        conflicting = tuple(
            projection_type.__name__
            for projection_type, candidate_value in matches
            if candidate_value != value
        )
        if conflicting:
            raise ValueError(
                f"Ambiguous source metadata projection for {component.value!r}: "
                f"{', '.join(conflicting)}."
            )
        return value

    @classmethod
    def projection_types_by_mro(
        cls,
        component: AllComponents,
        *,
        metadata_derived: bool,
    ) -> tuple[type["ComponentProjection"], ...]:
        registered = set(cls.__registry__.values())
        ordered: list[type[ComponentProjection]] = []
        seen: set[type[ComponentProjection]] = set()

        def visit(owner: type[ComponentProjection]) -> None:
            for child in owner.__subclasses__():
                visit(child)
            if (
                owner in registered
                and owner not in seen
                and owner.component is component
                and owner.metadata_derived is metadata_derived
            ):
                ordered.append(owner)
                seen.add(owner)

        visit(cls)
        return tuple(ordered)

    @staticmethod
    def direct_metadata_value(
        component: AllComponents,
        metadata: Mapping[str, str],
    ) -> str | None:
        return source_metadata_value(metadata, component.value)

    def first_metadata_value(
        self,
        metadata: Mapping[str, str],
        normalized_keys: tuple[str, ...],
    ) -> str | None:
        """Return the first normalized metadata value owned by this projection."""
        for key in normalized_keys:
            value = source_metadata_value(metadata, key)
            if value is not None:
                return value
        return None

    @abstractmethod
    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        """Return one OpenHCS component value or None if this projection does not apply."""


class WellRowColumnMetadataProjection(ComponentProjection):
    component = AllComponents.WELL

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        row = self.first_metadata_value(metadata, ("wellrow", "row"))
        column = self.first_metadata_value(
            metadata,
            ("wellcolumn", "wellcol", "column", "col"),
        )
        if row is None or column is None:
            return None
        return f"{row.strip().upper()}{int(column):02d}"


class SourceSchemaSingletonWellProjection(ComponentProjection):
    component = AllComponents.WELL
    metadata_derived = False

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        del metadata, image_set_index
        return SOURCE_SCHEMA_WORKSPACE_SINGLETON_AXIS_VALUE


class ImageNumberSiteProjection(ComponentProjection):
    component = AllComponents.SITE

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        return source_metadata_value(metadata, "imagenumber")


class FrameNumberSingletonSiteProjection(ComponentProjection):
    """Keep time-series frames on one site when no explicit site metadata exists."""

    component = AllComponents.SITE

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        del image_set_index
        if self.first_metadata_value(metadata, ("framenumber", "frame")) is None:
            return None
        return "1"


class ZIndexSingletonSiteProjection(ComponentProjection):
    """Keep Z-series planes on one site when no explicit site metadata exists."""

    component = AllComponents.SITE

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        del image_set_index
        if self.first_metadata_value(metadata, _Z_INDEX_METADATA_KEYS) is None:
            return None
        return "1"


class MetadataZIndexProjection(ComponentProjection):
    """Project explicit Z/slice/plane metadata onto the OpenHCS Z component."""

    component = AllComponents.Z_INDEX

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        del image_set_index
        return self.first_metadata_value(metadata, _Z_INDEX_METADATA_KEYS)


class FrameNumberTimepointProjection(ComponentProjection):
    """Project frame metadata onto the OpenHCS timepoint component."""

    component = AllComponents.TIMEPOINT

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        del image_set_index
        return self.first_metadata_value(
            metadata,
            (
                "timepoint",
                "time",
                "framenumber",
                "frame",
            ),
        )


class SourceSchemaSingletonTimepointProjection(ComponentProjection):
    component = AllComponents.TIMEPOINT
    metadata_derived = False

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        del metadata, image_set_index
        return "1"


class SourceSchemaSingletonZIndexProjection(ComponentProjection):
    component = AllComponents.Z_INDEX
    metadata_derived = False

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        del metadata, image_set_index
        return "1"


class OrdinalSiteProjection(ComponentProjection):
    """Project metadata-free ordered image sets onto the schema default axis."""

    component = AllComponents.SITE
    metadata_derived = False

    if SOURCE_SCHEMA_ORDERED_IMAGE_SET_COMPONENTS != (AllComponents.SITE,):
        raise TypeError(
            "OrdinalSiteProjection must match "
            "SOURCE_SCHEMA_ORDERED_IMAGE_SET_COMPONENTS."
        )

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        return str(image_set_index + 1)


class ImageSetAssembler(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for assembling source candidates into OpenHCS image sets."""

    __registry_key__ = "method_key"
    __skip_if_no_key__ = True
    method: ClassVar[SourceBindingMatchMethod | None] = None
    method_key: ClassVar[str | None] = None

    @classmethod
    def for_schema(
        cls,
        schema: PipelineImageSchema,
    ) -> "ImageSetAssembler":
        method = (
            SourceBindingMatchMethod.ORDER
            if schema.match_plan is None
            else schema.match_plan.method
        )
        return cls.__registry__[method.value]()

    @abstractmethod
    def image_sets(
        self,
        schema: PipelineImageSchema,
        candidates_by_alias: SourceSchemaCandidatesByAlias,
    ) -> tuple[ImageSetRecord, ...]:
        """Assemble candidate groups for projection into OpenHCS files."""


class MetadataImageSetAssembler(ImageSetAssembler):
    method = SourceBindingMatchMethod.METADATA
    method_key = SourceBindingMatchMethod.METADATA.value

    def image_sets(
        self,
        schema: PipelineImageSchema,
        candidates_by_alias: SourceSchemaCandidatesByAlias,
    ) -> tuple[ImageSetRecord, ...]:
        if schema.match_plan is None:
            raise ValueError("Metadata image-set assembly requires a match plan.")
        grouped: dict[tuple[str, ...], dict[str, SourceSchemaCandidate]] = {}
        metadata_by_key: dict[tuple[str, ...], dict[str, str]] = {}
        for alias, candidates in candidates_by_alias.items():
            for candidate in candidates:
                key_values: list[str] = []
                grouped_metadata: dict[str, str] = {}
                for dimension in schema.match_plan.dimensions:
                    field = dimension.field_for_alias(alias)
                    if field is None:
                        continue
                    value = _image_set_match_value(candidate.metadata, field)
                    if value is None:
                        raise ValueError(
                            f"Source candidate {candidate.relative_path!r} for alias "
                            f"{alias!r} lacks image-set match metadata field {field!r}."
                        )
                    key_values.append(str(value))
                    grouped_metadata[field] = str(value)
                key = tuple(key_values)
                if not key:
                    raise ValueError(
                        f"Source alias {alias!r} has no metadata dimensions in match plan."
                    )
                alias_group = grouped.setdefault(key, {})
                if alias in alias_group:
                    raise ValueError(
                        f"Multiple source files match alias {alias!r} for image-set "
                        f"key {key!r}."
                    )
                alias_group[alias] = candidate
                merge_source_metadata(
                    metadata_by_key.setdefault(key, {}),
                    grouped_metadata,
                    path=candidate.relative_path,
                )
        return _validated_image_sets(grouped, metadata_by_key, candidates_by_alias)


class OrderImageSetAssembler(ImageSetAssembler):
    method = SourceBindingMatchMethod.ORDER
    method_key = SourceBindingMatchMethod.ORDER.value

    def image_set_count(
        self,
        candidates_by_alias: SourceSchemaCandidatesByAlias,
    ) -> int:
        """Return the count of complete CellProfiler order-matched image sets."""
        return min(
            (len(candidates) for candidates in candidates_by_alias.values()),
            default=0,
        )

    def image_sets(
        self,
        schema: PipelineImageSchema,
        candidates_by_alias: SourceSchemaCandidatesByAlias,
    ) -> tuple[ImageSetRecord, ...]:
        aliases = tuple(candidates_by_alias)
        image_set_count = self.image_set_count(candidates_by_alias)
        image_sets: list[ImageSetRecord] = []
        for index in range(image_set_count):
            candidates = {
                alias: candidates_by_alias[alias][index]
                for alias in aliases
            }
            image_sets.append(
                ImageSetRecord(
                    index=index,
                    candidates_by_alias=candidates,
                    metadata=ImageSetMetadataMerge.from_candidates(
                        {},
                        candidates.values(),
                    ).metadata(),
                )
            )
        return tuple(image_sets)


def materialize_source_schema_workspace(
    source_root: Path,
    workspace_root: Path,
    schema: PipelineImageSchema,
    *,
    filemanager: FileManagerLike | None = None,
    source_backend: Backend | str = Backend.DISK,
    workspace_backend: Backend | str = Backend.DISK,
    source_files: Sequence[Path] | None = None,
    candidate_discovery_mode: SourceSchemaCandidateDiscoveryMode = (
        SourceSchemaCandidateDiscoveryMode.AUTO
    ),
    max_image_set_count: int | None = None,
    image_set_selection: SourceSchemaImageSetSelection | None = None,
) -> SourceSchemaWorkspaceMaterialization:
    """Create an OpenHCS virtual workspace from typed source-schema semantics."""

    source_root = Path(source_root)
    workspace_root = Path(workspace_root)
    if schema.is_empty:
        raise ValueError("Cannot materialize an empty source schema.")
    source_backend_name = _backend_name(source_backend)
    workspace_backend_name = _backend_name(workspace_backend)
    if max_image_set_count is not None and max_image_set_count <= 0:
        raise ValueError("max_image_set_count must be positive.")
    selection = image_set_selection or SourceSchemaImageSetSelection(
        max_image_set_count=max_image_set_count,
    )
    if not _vfs_is_dir(source_root, filemanager=filemanager, backend=source_backend_name):
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    _vfs_ensure_directory(
        workspace_root,
        filemanager=filemanager,
        backend=workspace_backend_name,
    )

    source_files = (
        tuple(Path(path) for path in source_files)
        if source_files is not None
        else _source_files(
            source_root,
            filemanager=filemanager,
            backend=source_backend_name,
        )
    )
    local_candidates = SourceSchemaCandidateDiscovery(
        SourceSchemaCandidateDiscoveryRequest(
            source_root,
            source_files,
            schema,
            candidate_discovery_mode,
        )
    ).candidates()
    candidates = SourceSchemaCandidateCollection.merge(
        local_candidates,
        _image_plane_source_candidates(workspace_root, schema, local_candidates),
    ).candidates
    candidate_matches = SourceSchemaCandidateMatches(candidates, schema)
    stack_assignments = candidate_matches.stack_assignments()
    auxiliary_assignments = candidate_matches.auxiliary_assignments()
    stack_candidates = candidate_matches.stack_candidates()
    auxiliary_candidates = candidate_matches.auxiliary_candidates()
    if stack_assignments:
        image_sets = ImageSetAssembler.for_schema(schema).image_sets(
            schema,
            stack_candidates,
        )
        image_sets = selection.apply(schema, image_sets)
        primary_mappings, primary_source_metadata, component_values = _primary_workspace_mappings(
            workspace_root,
            schema,
            image_sets,
            tuple(stack_assignments),
        )
    else:
        primary_mappings, primary_source_metadata, component_values = (
            _source_artifact_anchor_workspace_mappings(
                workspace_root,
                auxiliary_candidates,
                {assignment.alias: assignment for assignment in auxiliary_assignments},
            )
        )
    auxiliary_mappings, auxiliary_source_metadata = _auxiliary_workspace_mappings(
        workspace_root,
        auxiliary_candidates,
        {assignment.alias: assignment for assignment in auxiliary_assignments},
        filemanager=filemanager,
        workspace_backend=workspace_backend_name,
    )
    source_metadata = MappingProxyType(
        {
            **dict(primary_source_metadata),
            **dict(auxiliary_source_metadata),
        }
    )
    metadata_path = workspace_root / "openhcs_metadata.json"
    _write_workspace_metadata(
        metadata_path,
        primary_mappings,
        auxiliary_mappings,
        component_values,
        primary_source_metadata,
        auxiliary_source_metadata,
        filemanager=filemanager,
        backend=workspace_backend_name,
    )
    return SourceSchemaWorkspaceMaterialization(
        source_root=source_root,
        workspace_root=workspace_root,
        metadata_path=metadata_path,
        primary_mappings=primary_mappings,
        auxiliary_mappings=auxiliary_mappings,
        source_metadata=source_metadata,
    )


def expand_source_schema_workspace_wells(
    metadata_path: Path,
    well_ids: Iterable[str],
) -> tuple[str, ...]:
    """Duplicate source-schema image mappings across synthetic OpenHCS wells.

    This preserves the virtual-workspace abstraction: no source images are copied.
    Existing virtual filenames are parsed through ``SourceSchemaFilenameParser`` and
    re-emitted with each requested well identifier.
    """
    target_wells = tuple(dict.fromkeys(str(well_id) for well_id in well_ids))
    if not target_wells:
        raise ValueError("well_ids must contain at least one well.")

    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    subdirectories = metadata.get(FIELDS.SUBDIRECTORIES)
    if not isinstance(subdirectories, dict):
        raise ValueError(f"OpenHCS metadata lacks subdirectories: {metadata_path}")
    main_metadata = subdirectories.get(FIELDS.DEFAULT_SUBDIRECTORY)
    if not isinstance(main_metadata, dict):
        raise ValueError(
            f"OpenHCS metadata lacks main source-schema subdirectory: {metadata_path}"
        )
    workspace_mapping = main_metadata.get(FIELDS.WORKSPACE_MAPPING)
    if not isinstance(workspace_mapping, dict) or not workspace_mapping:
        raise ValueError(
            f"OpenHCS metadata lacks workspace mappings for well expansion: {metadata_path}"
        )

    parser = SourceSchemaFilenameParser()
    expanded_mapping: dict[str, Any] = {}
    expanded_source_metadata: dict[str, dict[str, str]] = {}
    original_source_metadata = main_metadata.get(FIELDS.SOURCE_METADATA) or {}
    if not isinstance(original_source_metadata, dict):
        raise ValueError(
            f"OpenHCS metadata source_metadata is not a mapping: {metadata_path}"
        )

    for virtual_path, real_path in workspace_mapping.items():
        parsed = parser.parse_filename(str(virtual_path))
        if parsed is None:
            raise ValueError(
                f"Cannot parse source-schema virtual filename {virtual_path!r}."
            )

    used_expanded_paths: set[str] = set()
    expanded_mapping_sink = WorkspaceMappingSink(expanded_mapping)
    for virtual_path, real_path in workspace_mapping.items():
        parsed = parser.parse_filename(str(virtual_path))
        if parsed is None:
            raise ValueError(
                f"Cannot parse source-schema virtual filename {virtual_path!r}."
            )
        path_source_metadata = original_source_metadata.get(str(virtual_path), {})
        if not isinstance(path_source_metadata, dict):
            raise ValueError(
                f"source_metadata for {virtual_path!r} is not a mapping."
            )
        for well_id in target_wells:
            expanded_path = _source_schema_virtual_path_for_well(
                parser,
                str(virtual_path),
                parsed,
                well_id,
                used_paths=used_expanded_paths,
            )
            expanded_mapping_sink.add(expanded_path, real_path)
            expanded_source_metadata[expanded_path] = dict(
                source_schema_metadata_with_virtual_components(
                    expanded_path,
                    with_source_component_metadata(
                        path_source_metadata,
                        AllComponents.WELL,
                        well_id,
                    ),
                )
            )

    main_metadata[FIELDS.IMAGE_FILES] = list(expanded_mapping)
    main_metadata[FIELDS.WORKSPACE_MAPPING] = expanded_mapping
    main_metadata[FIELDS.SOURCE_METADATA] = expanded_source_metadata
    main_metadata[FIELDS.WELLS] = {well_id: None for well_id in target_wells}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target_wells


def _source_schema_virtual_path_for_well(
    parser: SourceSchemaFilenameParser,
    virtual_path: str,
    parsed: Mapping[str, object],
    well_id: str,
    *,
    used_paths: set[str],
) -> str:
    projection = SourceSchemaFilenameProjection(parser)
    preferred_path = projection.path_for_well_site(
        virtual_path,
        parsed,
        well_id,
        parsed["site"],
    )
    if preferred_path not in used_paths:
        used_paths.add(preferred_path)
        return preferred_path

    ordinal_site = 1
    while True:
        expanded_path = projection.path_for_well_site(
            virtual_path,
            parsed,
            well_id,
            ordinal_site,
        )
        if expanded_path not in used_paths:
            used_paths.add(expanded_path)
            return expanded_path
        ordinal_site += 1


def _backend_name(backend: Backend | str) -> str:
    return backend.value if isinstance(backend, Backend) else str(backend)


def _vfs_is_dir(
    path: Path,
    *,
    filemanager: FileManagerLike | None,
    backend: str,
) -> bool:
    if filemanager is None:
        return path.is_dir()
    return bool(filemanager.is_dir(str(path), backend))


def _vfs_ensure_directory(
    path: Path,
    *,
    filemanager: FileManagerLike | None,
    backend: str,
) -> None:
    if filemanager is None:
        path.mkdir(parents=True, exist_ok=True)
        return
    filemanager.ensure_directory(str(path), backend)


def _vfs_save_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    filemanager: FileManagerLike | None,
    backend: str,
) -> None:
    if filemanager is None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    filemanager.ensure_directory(str(path.parent), backend)
    filemanager.save(dict(payload), str(path), backend)


def _source_files(
    source_root: Path,
    *,
    filemanager: FileManagerLike | None,
    backend: str,
) -> tuple[Path, ...]:
    if filemanager is not None:
        paths: list[Path] = []
        for raw_path in sorted(
            filemanager.list_files(str(source_root), backend, recursive=True)
        ):
            path = Path(raw_path)
            if not path.is_absolute():
                path = source_root / path
            if path.name != "openhcs_metadata.json":
                paths.append(path)
        return tuple(paths)
    return tuple(
        sorted(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.name != "openhcs_metadata.json"
        )
    )


@dataclass(frozen=True, slots=True)
class TiffSourcePlaneInventory:
    """TIFF source-plane inventory derived from explicit TIFF series axes."""

    axes: str
    shape: tuple[int, ...]
    plane_count: int

    @classmethod
    def from_path(cls, path: Path) -> "TiffSourcePlaneInventory":
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError(
                "TIFF source-plane inventory requires tifffile to preserve "
                f"axis semantics for {path}."
            ) from exc
        try:
            with tifffile.TiffFile(path) as tif:
                if not tif.series:
                    raise ValueError(f"TIFF source {path} has no image series.")
                series = tif.series[0]
                axes = str(series.axes)
                shape = tuple(int(value) for value in series.shape)
        except Exception as exc:
            raise ValueError(f"Could not inspect TIFF source-plane axes for {path}.") from exc
        return cls.from_axes(path, axes, shape)

    @classmethod
    def from_axes(
        cls,
        path: Path,
        axes: str,
        shape: tuple[int, ...],
    ) -> "TiffSourcePlaneInventory":
        if len(axes) != len(shape):
            raise ValueError(
                f"TIFF source {path} has inconsistent axes {axes!r} and shape {shape!r}."
            )
        source_plane_axes = tuple(axis for axis in axes if axis not in {"Y", "X", "S"})
        if not source_plane_axes:
            return cls(axes=axes, shape=shape, plane_count=1)
        if source_plane_axes == ("Z",) and axes[0] == "Z":
            return cls(
                axes=axes,
                shape=shape,
                plane_count=int(shape[axes.index("Z")]),
            )
        raise ValueError(
            f"TIFF source {path} has axes {axes!r}; source-schema TIFF "
            "projection currently supports only YX/YXS images and Z-first Z-stacks. "
            "Use an explicit Bio-Formats/source-projection path for C/T/series axes."
        )


def _tiff_page_count(path: Path) -> int:
    """Return the supported TIFF Z-plane count."""
    return TiffSourcePlaneInventory.from_path(path).plane_count


def _image_plane_source_candidates(
    workspace_root: Path,
    schema: PipelineImageSchema,
    local_candidates: tuple[SourceSchemaCandidate, ...],
) -> tuple[SourceSchemaCandidate, ...]:
    if not schema.image_plane_sources:
        return ()
    candidates: list[SourceSchemaCandidate] = []
    for index, source in enumerate(schema.image_plane_sources, start=1):
        candidates.append(
            ImagePlaneSourceResolver.resolve(
                source,
                source_index=index,
                workspace_root=workspace_root,
                schema=schema,
                local_candidates=local_candidates,
            )
        )
    return tuple(candidates)


def _image_plane_source_basename(source: ImagePlaneSource) -> str:
    parsed = urlparse(source.uri)
    path = unquote(parsed.path if parsed.scheme else source.uri)
    return Path(path).name


def _source_metadata_matches(
    candidate_metadata: Mapping[str, str],
    source_metadata: Mapping[str, str],
) -> bool:
    return all(
        source_metadata_values_equal(
            source_metadata_value(candidate_metadata, field),
            value,
        )
        for field, value in source_metadata.items()
    )


def _imported_metadata_rows(
    source_root: Path,
    schema: PipelineImageSchema,
) -> tuple[ImportedMetadataRows, ...]:
    return tuple(
        ImportedMetadataRows(
            table=table,
            rows=_read_imported_metadata_rows(source_root, table),
        )
        for table in schema.imported_metadata_tables
    )


def _read_imported_metadata_rows(
    source_root: Path,
    table: ImportedMetadataTable,
) -> tuple[Mapping[str, str], ...]:
    table_path = ImportedMetadataPathResolver(source_root).path(table)
    if not table_path.is_file():
        raise FileNotFoundError(f"Imported metadata table does not exist: {table_path}")
    with table_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                f"Imported metadata table {table_path} has no header row."
            )
        rows = tuple(
            MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in row.items()
                    if key is not None and value is not None
                }
            )
            for row in reader
        )
    if not rows:
        raise ValueError(f"Imported metadata table {table_path} has no data rows.")
    return rows


def _indexed_imported_metadata_rows(
    rows: tuple[Mapping[str, str], ...],
    joins: tuple[ImportedMetadataJoin, ...],
) -> ImportedMetadataJoinIndex:
    indexed: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for row in rows:
        key = _imported_metadata_row_join_key(row, joins)
        if key is None:
            continue
        indexed.setdefault(key, []).append(row)
    return MappingProxyType(
        {key: tuple(indexed_rows) for key, indexed_rows in indexed.items()}
    )


def _imported_metadata_row_join_key(
    row: Mapping[str, str],
    joins: tuple[ImportedMetadataJoin, ...],
) -> tuple[str, ...] | None:
    values: list[str] = []
    for join in joins:
        value = source_metadata_value(row, join.imported_metadata_field)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class ImportedMetadataPathResolver:
    """Resolve CellProfiler imported metadata paths against a source root."""

    source_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))

    def path(self, table: ImportedMetadataTable) -> Path:
        """Return the first existing path for an imported metadata table."""
        candidates = self.path_candidates(table)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    def path_candidates(self, table: ImportedMetadataTable) -> tuple[Path, ...]:
        """Return candidate local paths for a CellProfiler metadata declaration."""
        if table.location is None:
            raise ValueError("Imported metadata tables require a location.")
        location = Path(table.location)
        return tuple(dict.fromkeys(self._path_candidates_for_location(location)))

    def _path_candidates_for_location(self, location: Path) -> tuple[Path, ...]:
        resolved_source_root = self.source_root.resolve()
        if location.is_absolute():
            return (
                location,
                self.source_root / location.name,
                self.source_root.parent / location.name,
                resolved_source_root / location.name,
                resolved_source_root.parent / location.name,
            )
        return (
            self.source_root / location,
            self.source_root / location.name,
            self.source_root.parent / location,
            self.source_root.parent / location.name,
            resolved_source_root / location,
            resolved_source_root / location.name,
            resolved_source_root.parent / location,
            resolved_source_root.parent / location.name,
        )


def _metadata_with_imported_tables(
    metadata: Mapping[str, str],
    imported_metadata: tuple[ImportedMetadataRows, ...],
    *,
    path: str,
) -> Mapping[str, str]:
    if not imported_metadata:
        return MappingProxyType(dict(metadata))
    merged = dict(metadata)
    for table_rows in imported_metadata:
        row = _matched_imported_metadata_row(merged, table_rows, path=path)
        if row is None:
            continue
        _merge_imported_metadata_row(merged, row, path=path)
    return MappingProxyType(merged)


def _matched_imported_metadata_row(
    image_metadata: Mapping[str, str],
    imported_metadata: ImportedMetadataRows,
    *,
    path: str,
) -> Mapping[str, str] | None:
    joins = imported_metadata.table.joins
    if not joins:
        raise ValueError(
            "Imported metadata tables require explicit image-to-table joins."
        )
    join_values = {
        join.image_metadata_field: source_metadata_value(
            image_metadata,
            join.image_metadata_field,
        )
        for join in joins
    }
    present_join_values = {
        field: value
        for field, value in join_values.items()
        if value is not None
    }
    if not present_join_values:
        return None
    if len(present_join_values) != len(joins):
        return None
    join_key = tuple(
        present_join_values[join.image_metadata_field]
        for join in joins
    )
    matched_rows = imported_metadata.rows_by_join_key.get(join_key, ())
    if not matched_rows:
        raise ValueError(
            f"Source candidate {path!r} matched {len(matched_rows)} imported "
            f"metadata rows; expected exactly one."
        )
    if len(matched_rows) > 1:
        return _consensus_imported_metadata_row(matched_rows)
    return matched_rows[0]


def _consensus_imported_metadata_row(
    rows: tuple[Mapping[str, str], ...],
) -> Mapping[str, str]:
    """Return fields with identical values across repeated imported metadata rows."""
    common_fields = set(rows[0])
    for row in rows[1:]:
        common_fields.intersection_update(row)
    return MappingProxyType(
        {
            field: rows[0][field]
            for field in sorted(common_fields)
            if all(row[field] == rows[0][field] for row in rows[1:])
        }
    )


def _merge_imported_metadata_row(
    metadata: dict[str, str],
    row: Mapping[str, str],
    *,
    path: str,
) -> None:
    additions = {
        key: value
        for key, value in row.items()
        if source_metadata_value(metadata, key) in (None, str(value))
    }
    merge_source_metadata(metadata, additions, path=path)


def _candidate_matches_selector(
    candidate: SourceSchemaCandidate,
    selector: SourceSelector,
) -> bool:
    return (
        _candidate_matches_components(candidate, selector)
        and _candidate_matches_metadata(candidate, selector)
        and any(
            source_filters_match(path, selector.filters)
            for path in candidate.source_filter_path_identities()
        )
    )


def _candidate_matches_components(
    candidate: SourceSchemaCandidate,
    selector: SourceSelector,
) -> bool:
    return all(
        _component_selector_matches(candidate.metadata, component)
        for component in selector.components
    )


def _component_selector_matches(
    metadata: Mapping[str, str],
    selector: ComponentSelector,
) -> bool:
    return any(
        source_metadata_values_equal(value, selector.value)
        for value in source_component_metadata_values(metadata, selector.component)
    )


def _candidate_matches_metadata(
    candidate: SourceSchemaCandidate,
    selector: SourceSelector,
) -> bool:
    return all(
        (value := source_metadata_value(candidate.metadata, metadata.field)) is not None
        and source_metadata_values_equal(value, metadata.value)
        for metadata in selector.metadata
    )


def _validated_image_sets(
    grouped: ImageSetCandidateGroups,
    metadata_by_key: ImageSetMetadataByGroup,
    candidates_by_alias: SourceSchemaCandidatesByAlias,
) -> tuple[ImageSetRecord, ...]:
    aliases = tuple(candidates_by_alias)
    image_sets: list[ImageSetRecord] = []
    for index, key in enumerate(sorted(grouped)):
        candidates = grouped[key]
        missing_aliases = tuple(alias for alias in aliases if alias not in candidates)
        if missing_aliases:
            raise ValueError(
                f"Source image set {key!r} is missing aliases {missing_aliases!r}."
            )
        image_sets.append(
            ImageSetRecord(
                index=index,
                candidates_by_alias=dict(candidates),
                metadata=ImageSetMetadataMerge.from_candidates(
                    metadata_by_key[key],
                    candidates.values(),
                ).metadata(),
            )
        )
    return tuple(image_sets)


def _projected_candidate_components(
    group_metadata: Mapping[str, str],
    candidates: tuple[SourceSchemaCandidate, ...],
) -> Mapping[str, str]:
    projected: dict[str, str] = {}
    for component in AllComponents:
        values = {
            value
            for candidate in candidates
            if (
                value := ComponentProjection.resolve_from_metadata(
                    component,
                    candidate.metadata,
                )
            )
            is not None
        }
        if len(values) > 1:
            existing = source_metadata_value(group_metadata, component.value)
            if existing is not None:
                raise ValueError(
                    f"Source image set has conflicting {component.value!r} component "
                    f"values {existing!r} and {sorted(values)!r}."
                )
            continue
        if not values:
            continue
        value = next(iter(values))
        existing = source_metadata_value(group_metadata, component.value)
        if existing is not None:
            if existing != value:
                raise ValueError(
                    f"Source image set has conflicting {component.value!r} component "
                    f"values {existing!r} and {value!r}."
                )
            continue
        projected[component.value] = value
    return MappingProxyType(projected)


def _primary_workspace_mappings(
    workspace_root: Path,
    schema: PipelineImageSchema,
    image_sets: tuple[ImageSetRecord, ...],
    stack_assignments: tuple[ImageAssignment, ...],
) -> WorkspaceMappingResult:
    filename_projection = SourceSchemaFilenameProjection()
    channel_values = {
        str(index): assignment.alias
        for index, assignment in enumerate(stack_assignments, start=1)
    }
    wells: dict[str, None] = {}
    sites: dict[str, None] = {}
    z_indexes: dict[str, None] = {}
    timepoints: dict[str, None] = {}
    primary_mappings: dict[str, Any] = {}
    primary_mapping_sink = WorkspaceMappingSink(primary_mappings)
    source_metadata: dict[str, Mapping[str, str]] = {}
    site_indexes_by_well: dict[str, int] = {}
    source_plane_group_sites = SourcePlaneGroupSiteAllocator()
    used_paths_by_well_channel_z_timepoint: dict[
        tuple[str, int, int | str, int | str],
        set[str],
    ] = {}
    for image_set in image_sets:
        well = SourceSchemaImageSetIdentity(schema, image_set).well
        site_index = site_indexes_by_well.get(well, 0)
        site = ComponentProjection.resolve(
            AllComponents.SITE,
            image_set.metadata,
            site_index,
        )
        timepoint = ComponentProjection.resolve(
            AllComponents.TIMEPOINT,
            image_set.metadata,
            site_index,
        )
        z_index = ComponentProjection.resolve(
            AllComponents.Z_INDEX,
            image_set.metadata,
            site_index,
        )
        site_indexes_by_well[well] = site_index + 1
        preferred_site_component = source_plane_group_sites.site_component(
            image_set.metadata,
            _component_ordinal_or_label(site),
        )
        timepoint_component = _component_ordinal_or_label(timepoint)
        z_component = _component_ordinal_or_label(z_index)
        wells[well] = None
        sites[str(preferred_site_component)] = None
        z_indexes[str(z_component)] = None
        timepoints[str(timepoint_component)] = None
        for channel_index, assignment in enumerate(stack_assignments, start=1):
            candidate = image_set.candidates_by_alias[assignment.alias]
            site_component = _collision_free_site_component(
                filename_projection,
                well=well,
                preferred_site_component=preferred_site_component,
                ordinal_site_component=site_index + 1,
                channel_index=channel_index,
                z_index=z_component,
                timepoint=timepoint_component,
                extension=candidate.path.suffix,
                used_paths=used_paths_by_well_channel_z_timepoint.setdefault(
                    (well, channel_index, z_component, timepoint_component),
                    set(),
                ),
            )
            sites[str(site_component)] = None
            virtual_path = filename_projection.filename(
                SourceSchemaVirtualFilename(
                    well,
                    site_component,
                    channel_index,
                    z_component,
                    timepoint_component,
                    candidate.path.suffix,
                )
            )
            primary_mapping_sink.add(
                virtual_path,
                _source_schema_workspace_mapping_value(workspace_root, candidate),
            )
            source_metadata[virtual_path] = SourceVirtualPathMetadata(
                image_set.metadata,
                candidate.metadata,
                virtual_path=virtual_path,
                assignment=assignment,
            ).metadata()
    component_values: WorkspaceComponentValues = MappingProxyType(
        {
            AllComponents.CHANNEL: MappingProxyType(channel_values),
            AllComponents.WELL: MappingProxyType(wells),
            AllComponents.SITE: MappingProxyType(sites),
            AllComponents.Z_INDEX: MappingProxyType(z_indexes),
            AllComponents.TIMEPOINT: MappingProxyType(timepoints),
        }
    )
    return (
        MappingProxyType(primary_mappings),
        MappingProxyType(source_metadata),
        component_values,
    )


def _source_artifact_anchor_workspace_mappings(
    workspace_root: Path,
    auxiliary_candidates: SourceSchemaCandidatesByAlias,
    assignments_by_alias: Mapping[str, SourceAssignmentBase],
) -> WorkspaceMappingResult:
    anchor_alias, anchor_candidate = _source_artifact_anchor_candidate(
        auxiliary_candidates
    )
    virtual_path = SourceSchemaFilenameProjection().filename(
        SourceSchemaVirtualFilename(
            SOURCE_SCHEMA_WORKSPACE_SINGLETON_AXIS_VALUE,
            1,
            1,
            1,
            1,
            anchor_candidate.path.suffix,
        )
    )
    mappings = MappingProxyType(
        {
            virtual_path: _workspace_relative_path(
                workspace_root,
                anchor_candidate.path,
            )
        }
    )
    source_metadata = MappingProxyType(
        {
            virtual_path: SourceVirtualPathMetadata(
                {"source_alias": anchor_alias},
                anchor_candidate.metadata,
                assignment=assignments_by_alias.get(anchor_alias),
            ).metadata()
        }
    )
    component_values = MappingProxyType(
        {
            AllComponents.CHANNEL: MappingProxyType({"1": anchor_alias}),
            AllComponents.WELL: MappingProxyType(
                {SOURCE_SCHEMA_WORKSPACE_SINGLETON_AXIS_VALUE: None}
            ),
            AllComponents.SITE: MappingProxyType({"1": None}),
            AllComponents.Z_INDEX: MappingProxyType({"1": None}),
            AllComponents.TIMEPOINT: MappingProxyType({"1": None}),
        }
    )
    return mappings, source_metadata, component_values


def _source_artifact_anchor_candidate(
    auxiliary_candidates: SourceSchemaCandidatesByAlias,
) -> tuple[str, SourceSchemaCandidate]:
    for alias in sorted(auxiliary_candidates):
        for candidate in auxiliary_candidates[alias]:
            if is_image_path(str(candidate.path)):
                return alias, candidate
    raise ValueError(
        "Source schema declares no image-stack assignments and no loadable "
        "source artifact image that can anchor OpenHCS execution."
    )


def _collision_free_site_component(
    filename_projection: SourceSchemaFilenameProjection,
    *,
    well: str,
    preferred_site_component: int | str,
    ordinal_site_component: int,
    channel_index: int,
    z_index: int | str,
    timepoint: int | str,
    extension: str,
    used_paths: set[str],
) -> int | str:
    preferred_path = filename_projection.filename(
        SourceSchemaVirtualFilename(
            well,
            preferred_site_component,
            channel_index,
            z_index,
            timepoint,
            extension,
        )
    )
    if preferred_path not in used_paths:
        used_paths.add(preferred_path)
        return preferred_site_component

    ordinal_component = ordinal_site_component
    while True:
        ordinal_path = filename_projection.filename(
            SourceSchemaVirtualFilename(
                well,
                ordinal_component,
                channel_index,
                z_index,
                timepoint,
                extension,
            )
        )
        if ordinal_path not in used_paths:
            used_paths.add(ordinal_path)
            return ordinal_component
        ordinal_component += 1


def _grouping_metadata(
    metadata: Mapping[str, str],
    fields: tuple[str, ...],
) -> Mapping[str, str]:
    group_metadata: dict[str, str] = {}
    for field in fields:
        value = source_metadata_value(metadata, field)
        if value is None:
            raise ValueError(
                f"Grouped source image set lacks metadata field {field!r}; "
                f"available fields are {sorted(metadata)}."
            )
        group_metadata[field] = value
    return MappingProxyType(group_metadata)


def _grouping_fields_project_to_well(fields: tuple[str, ...]) -> bool:
    if len(fields) == 1:
        return source_metadata_component(fields[0]) is AllComponents.WELL
    normalized_fields = {
        normalize_source_metadata_key(field)
        for field in fields
    }
    return (
        len(fields) == 2
        and bool(normalized_fields & {"wellrow", "row"})
        and bool(normalized_fields & {"wellcolumn", "wellcol", "column", "col"})
    )


def _grouping_axis_value(
    group_metadata: Mapping[str, str],
    fields: tuple[str, ...],
) -> str:
    if len(fields) == 1:
        return _source_schema_component_token(
            _required_grouping_value(group_metadata, fields[0])
        )
    return "-".join(
        (
            _source_schema_component_token(field)
            + "-"
            + _source_schema_component_token(
                _required_grouping_value(group_metadata, field)
            )
        )
        for field in fields
    )


def _required_grouping_value(
    group_metadata: Mapping[str, str],
    field: str,
) -> str:
    value = group_metadata[field]
    if not value:
        raise ValueError(f"Grouping metadata field {field!r} cannot be empty.")
    return value


def _source_schema_component_token(value: str) -> str:
    """Percent-encode component labels so parser delimiters remain unambiguous."""

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
        raise ValueError("Source-schema component tokens cannot be empty.")
    return encoded


def _auxiliary_workspace_mappings(
    workspace_root: Path,
    auxiliary_candidates: SourceSchemaCandidatesByAlias,
    assignments_by_alias: Mapping[str, SourceAssignmentBase],
    *,
    filemanager: FileManagerLike | None,
    workspace_backend: str,
) -> tuple[Mapping[str, str], Mapping[str, Mapping[str, str]]]:
    mappings: dict[str, str] = {}
    mapping_sink = WorkspaceMappingSink(mappings)
    source_metadata: dict[str, Mapping[str, str]] = {}
    for alias, candidates in auxiliary_candidates.items():
        for index, candidate in enumerate(candidates, start=1):
            virtual_path = (
                f"{SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR}/"
                f"{alias}/{index:03d}_{candidate.path.name}"
            )
            source_path = _materialized_auxiliary_source_path(
                candidate.path,
                workspace_root=workspace_root,
                filemanager=filemanager,
                workspace_backend=workspace_backend,
                alias=alias,
                index=index,
            )
            mapping_sink.add(
                virtual_path,
                _workspace_relative_path(workspace_root, source_path),
            )
            cached_payload = source_schema_auxiliary_payload(source_path)
            if cached_payload is not None:
                cache_source_schema_auxiliary_payload(virtual_path, cached_payload)
                cache_source_schema_auxiliary_payload(
                    workspace_root / virtual_path,
                    cached_payload,
                )
            source_metadata[virtual_path] = SourceVirtualPathMetadata(
                {"source_alias": alias},
                candidate.metadata,
                virtual_path=virtual_path,
                assignment=assignments_by_alias.get(alias),
            ).metadata()
    return MappingProxyType(mappings), MappingProxyType(source_metadata)


def _materialized_auxiliary_source_path(
    source_path: Path,
    *,
    workspace_root: Path,
    filemanager: FileManagerLike | None,
    workspace_backend: str,
    alias: str,
    index: int,
) -> Path:
    """Return an auxiliary source path normalized for runtime loading."""
    materializer = SourceSchemaAuxiliaryMaterializer.for_path(source_path)
    if materializer is None:
        return source_path
    return materializer.materialize(
        SourceSchemaAuxiliaryMaterializationRequest(
            source_path=source_path,
            workspace_root=workspace_root,
            filemanager=filemanager,
            workspace_backend=workspace_backend,
            alias=alias,
            index=index,
        )
    )


def _write_workspace_metadata(
    metadata_path: Path,
    primary_mappings: Mapping[str, Any],
    auxiliary_mappings: Mapping[str, Any],
    component_values: WorkspaceComponentValues,
    primary_source_metadata: WorkspaceSourceMetadata,
    auxiliary_source_metadata: WorkspaceSourceMetadata,
    *,
    filemanager: FileManagerLike | None,
    backend: str,
) -> None:
    subdirectories = {
        FIELDS.DEFAULT_SUBDIRECTORY: _metadata_dict(
            image_files=tuple(primary_mappings),
            workspace_mapping=primary_mappings,
            component_values=component_values,
            source_metadata=primary_source_metadata,
            main=True,
        )
    }
    if auxiliary_mappings:
        subdirectories[SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR] = _metadata_dict(
            image_files=tuple(auxiliary_mappings),
            workspace_mapping=auxiliary_mappings,
            component_values=component_values,
            source_metadata=auxiliary_source_metadata,
            main=False,
        )
    _vfs_save_json(
        metadata_path,
        {FIELDS.SUBDIRECTORIES: subdirectories},
        filemanager=filemanager,
        backend=backend,
    )


def _metadata_dict(
    *,
    image_files: tuple[str, ...],
    workspace_mapping: Mapping[str, Any],
    component_values: WorkspaceComponentValues,
    source_metadata: WorkspaceSourceMetadata,
    main: bool,
) -> dict[str, object]:
    metadata = asdict(
        OpenHCSMetadata(
            microscope_handler_name=FIELDS.MICROSCOPE_TYPE,
            source_filename_parser_name="SourceSchemaFilenameParser",
            grid_dimensions=SOURCE_SCHEMA_WORKSPACE_GRID_DIMENSIONS,
            pixel_size=SOURCE_SCHEMA_WORKSPACE_PIXEL_SIZE,
            image_files=list(image_files),
            channels=dict(component_values[AllComponents.CHANNEL]),
            wells=dict(component_values[AllComponents.WELL]),
            sites=dict(component_values[AllComponents.SITE]),
            z_indexes=dict(component_values[AllComponents.Z_INDEX]),
            timepoints=dict(component_values[AllComponents.TIMEPOINT]),
            available_backends={
                Backend.DISK.value: True,
                Backend.VIRTUAL_WORKSPACE.value: True,
            },
            workspace_mapping=dict(workspace_mapping),
            source_metadata={
                path: SourceMetadataJsonRecord(metadata).as_dict()
                for path, metadata in source_metadata.items()
            },
            main=main,
        )
    )
    return {key: value for key, value in metadata.items() if value is not None}


def _workspace_relative_path(workspace_root: Path, path: Path) -> str:
    return os.path.relpath(path, workspace_root).replace(os.sep, "/")


def _source_schema_workspace_mapping_value(
    workspace_root: Path,
    candidate: SourceSchemaCandidate,
) -> str | dict[str, object]:
    source_path = _workspace_relative_path(workspace_root, candidate.path)
    if candidate.source_plane_index is None:
        return source_path
    source_z_index = None
    z_value = source_metadata_value(candidate.metadata, AllComponents.Z_INDEX.value)
    if z_value is not None and z_value.isdecimal():
        source_z_index = int(z_value)
    return SourcePixelRef(
        backend=Backend.DISK.value,
        source_path=source_path,
        plane_index=candidate.source_plane_index,
        source_z_index=source_z_index,
    ).to_legacy_workspace_mapping()


def _image_set_match_value(
    metadata: Mapping[str, str],
    field: str,
) -> str | None:
    value = source_metadata_value(metadata, field)
    if value is not None:
        return value
    component = source_metadata_component(field)
    if component is None:
        return None
    return ComponentProjection.resolve_from_metadata(component, metadata)


def _component_ordinal_or_label(value: str) -> int | str:
    return int(value) if value.isdecimal() else value
