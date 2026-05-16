"""Project typed source schemas into native OpenHCS virtual workspaces."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import urllib.request
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import ClassVar
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
    SOURCE_IMAGE_TYPE_METADATA_FIELD,
    SourceAssignmentBase,
)
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.core.source_bindings import (
    SourceBindingMatchMethod,
    SourceSelector,
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
)
from openhcs.microscopes.openhcs import FIELDS, OpenHCSMetadata
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR = "_source"
SOURCE_SCHEMA_WORKSPACE_PIXEL_SIZE = 1.0
SOURCE_SCHEMA_WORKSPACE_GRID_DIMENSIONS = [1, 1]
SOURCE_SCHEMA_WORKSPACE_SINGLETON_AXIS_VALUE = "A01"
SOURCE_SCHEMA_IMAGE_TYPE_METADATA_FIELD = SOURCE_IMAGE_TYPE_METADATA_FIELD
_AUXILIARY_PAYLOAD_CACHE_LIMIT = 64
_AUXILIARY_PAYLOAD_CACHE: OrderedDict[str, object] = OrderedDict()
_PLATE_WELL_TOKEN_PATTERN = re.compile(
    r"(?:^|[_-])(?P<well>[A-P][0-9]{2})(?=(?:f[0-9])|[_\-.]|$)",
    re.IGNORECASE,
)
_SITE_TOKEN_PATTERN = re.compile(r"(?:^|[_-])f(?P<site>[0-9]+)(?=[A-Za-z_\-.]|$)")


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
    primary_mappings: Mapping[str, str]
    auxiliary_mappings: Mapping[str, str]
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
    ) -> tuple[Path, ...]:
        """Return the complete source universe required for selected wells.

        Primary mappings define the selected sample identities. Auxiliary mappings are
        source artifacts, not sample-defining image sets, so they remain available to
        native consumers that need illumination images or object seeds.
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
                paths.append((self.workspace_root / real_path).resolve())
        paths.extend(
            (self.workspace_root / real_path).resolve()
            for real_path in self.auxiliary_mappings.values()
        )
        return tuple(dict.fromkeys(paths))


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidate:
    """One source file plus metadata extracted from source-schema rules."""

    path: Path
    relative_path: str
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "relative_path", self.relative_path.replace(os.sep, "/"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SourceSchemaCandidateCollection:
    """Deduplicated ordered source-schema candidates."""

    candidates: tuple[SourceSchemaCandidate, ...]

    def __post_init__(self) -> None:
        deduplicated: list[SourceSchemaCandidate] = []
        seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
        for candidate in self.candidates:
            if not isinstance(candidate, SourceSchemaCandidate):
                raise TypeError(
                    "SourceSchemaCandidateCollection values must be "
                    f"SourceSchemaCandidate, got {type(candidate).__name__}."
                )
            key = (
                str(candidate.path),
                candidate.relative_path,
                tuple(sorted(candidate.metadata.items())),
            )
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
    rows_by_join_key: Mapping[tuple[str, ...], tuple[Mapping[str, str], ...]] = field(
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
        row = _first_metadata_value(metadata, ("wellrow", "row"))
        column = _first_metadata_value(metadata, ("wellcolumn", "wellcol", "column", "col"))
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


class OrdinalSiteProjection(ComponentProjection):
    component = AllComponents.SITE
    metadata_derived = False

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
        candidates_by_alias: Mapping[str, tuple[SourceSchemaCandidate, ...]],
    ) -> tuple[ImageSetRecord, ...]:
        """Assemble candidate groups for projection into OpenHCS files."""


class MetadataImageSetAssembler(ImageSetAssembler):
    method = SourceBindingMatchMethod.METADATA
    method_key = SourceBindingMatchMethod.METADATA.value

    def image_sets(
        self,
        schema: PipelineImageSchema,
        candidates_by_alias: Mapping[str, tuple[SourceSchemaCandidate, ...]],
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
        candidates_by_alias: Mapping[str, tuple[SourceSchemaCandidate, ...]],
    ) -> int:
        """Return the count of complete CellProfiler order-matched image sets."""
        return min(
            (len(candidates) for candidates in candidates_by_alias.values()),
            default=0,
        )

    def image_sets(
        self,
        schema: PipelineImageSchema,
        candidates_by_alias: Mapping[str, tuple[SourceSchemaCandidate, ...]],
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
                    metadata=_merged_image_set_metadata({}, candidates.values()),
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

    source_files = _source_files(
        source_root,
        filemanager=filemanager,
        backend=source_backend_name,
    )
    local_candidates = _source_candidates(source_root, source_files, schema)
    candidates = SourceSchemaCandidateCollection.merge(
        local_candidates,
        _image_plane_source_candidates(workspace_root, schema, local_candidates),
    ).candidates
    stack_assignments, auxiliary_assignments = _partition_assignments(schema)
    stack_candidates = _matched_candidates_by_alias(
        candidates,
        stack_assignments,
        require_match=True,
    )
    auxiliary_candidates = _matched_candidates_by_alias(
        candidates,
        auxiliary_assignments,
        require_match=False,
    )
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
    expanded_mapping: dict[str, str] = {}
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
            _add_mapping(expanded_mapping, expanded_path, str(real_path))
            expanded_source_metadata[expanded_path] = with_source_component_metadata(
                path_source_metadata,
                AllComponents.WELL,
                well_id,
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
    preferred_path = _source_schema_virtual_path_for_well_site(
        parser,
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
        expanded_path = _source_schema_virtual_path_for_well_site(
            parser,
            virtual_path,
            parsed,
            well_id,
            ordinal_site,
        )
        if expanded_path not in used_paths:
            used_paths.add(expanded_path)
            return expanded_path
        ordinal_site += 1


def _source_schema_virtual_path_for_well_site(
    parser: SourceSchemaFilenameParser,
    virtual_path: str,
    parsed: Mapping[str, object],
    well_id: str,
    site: object,
) -> str:
    path = PurePosixPath(virtual_path)
    original_name = path.name
    original_stem = parser.construct_filename(
        well=parsed["well"],
        site=parsed["site"],
        channel=parsed["channel"],
        z_index=parsed["z_index"],
        timepoint=parsed["timepoint"],
        extension="",
    )
    expanded_stem = parser.construct_filename(
        well=well_id,
        site=site,
        channel=parsed["channel"],
        z_index=parsed["z_index"],
        timepoint=parsed["timepoint"],
        extension="",
    )
    expanded_name = (
        f"{expanded_stem}{original_name[len(original_stem):]}"
        if original_name.startswith(original_stem)
        else parser.construct_filename(
            well=well_id,
            site=site,
            channel=parsed["channel"],
            z_index=parsed["z_index"],
            timepoint=parsed["timepoint"],
            extension=str(parsed["extension"]),
        )
    )
    return str(path.with_name(expanded_name))


def _partition_assignments(
    schema: PipelineImageSchema,
) -> tuple[tuple[ImageAssignment, ...], tuple[SourceAssignmentBase, ...]]:
    stack_assignments: list[ImageAssignment] = []
    auxiliary_assignments: list[SourceAssignmentBase] = []
    for assignment in schema.assignments_by_alias.values():
        role = ImageTypeSourceRole.for_image_type(assignment.image_type)
        if role.participates_in_image_stack:
            stack_assignments.append(assignment)
        else:
            auxiliary_assignments.append(assignment)
    auxiliary_assignments.extend(schema.source_artifacts_by_alias.values())
    return tuple(stack_assignments), tuple(auxiliary_assignments)


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


def _source_candidates(
    source_root: Path,
    source_files: tuple[Path, ...],
    schema: PipelineImageSchema,
) -> tuple[SourceSchemaCandidate, ...]:
    imported_metadata = _imported_metadata_rows(source_root, schema)
    candidates: list[SourceSchemaCandidate] = []
    for path in source_files:
        relative_path = path.relative_to(source_root).as_posix()
        if schema.images_rule is not None and not source_filters_match(
            relative_path,
            schema.images_rule.filters,
        ):
            continue
        metadata = metadata_from_rules(
            str(path),
            schema.metadata_rules,
            filter_path=relative_path,
        )
        metadata = _metadata_with_imported_tables(
            metadata,
            imported_metadata,
            path=relative_path,
        )
        metadata = SourceSchemaFilenameComponentFallbackPolicy.for_schema(
            schema
        ).metadata(path.name, metadata)
        candidates.append(
            SourceSchemaCandidate(
                path=path,
                relative_path=relative_path,
                metadata=metadata,
            )
        )
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class SourceSchemaFilenameComponentFallbackPolicy:
    """Recover missing component tokens from common source filename conventions."""

    enabled: bool

    @classmethod
    def for_schema(
        cls,
        schema: PipelineImageSchema,
    ) -> "SourceSchemaFilenameComponentFallbackPolicy":
        del schema
        return cls(enabled=True)

    def metadata(
        self,
        filename: str,
        metadata: Mapping[str, str],
    ) -> Mapping[str, str]:
        if not self.enabled:
            return MappingProxyType(dict(metadata))
        return _metadata_with_filename_component_fallbacks(filename, metadata)


def _metadata_with_filename_component_fallbacks(
    filename: str,
    metadata: Mapping[str, str],
) -> Mapping[str, str]:
    """Recover common plate well/site tokens when a pipeline lacks usable metadata."""
    enriched = dict(metadata)
    if source_metadata_value(enriched, AllComponents.WELL.value) is None:
        well_match = _PLATE_WELL_TOKEN_PATTERN.search(filename)
        if well_match is not None:
            merge_source_metadata(
                enriched,
                {AllComponents.WELL.value: well_match.group("well").upper()},
                path=filename,
            )
    if source_metadata_value(enriched, AllComponents.SITE.value) is None:
        site_match = _SITE_TOKEN_PATTERN.search(filename)
        if site_match is not None:
            merge_source_metadata(
                enriched,
                {AllComponents.SITE.value: str(int(site_match.group("site")) + 1)},
                path=filename,
            )
    return MappingProxyType(enriched)


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
    table_path = _imported_metadata_path(source_root, table)
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
) -> Mapping[tuple[str, ...], tuple[Mapping[str, str], ...]]:
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


def _imported_metadata_path(
    source_root: Path,
    table: ImportedMetadataTable,
) -> Path:
    if table.location is None:
        raise ValueError("Imported metadata tables require a location.")
    location = Path(table.location)
    candidates = tuple(
        dict.fromkeys(_imported_metadata_path_candidates(source_root, location))
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _imported_metadata_path_candidates(
    source_root: Path,
    location: Path,
) -> tuple[Path, ...]:
    if location.is_absolute():
        return (
            location,
            source_root / location.name,
            source_root.parent / location.name,
        )
    return (
        source_root / location,
        source_root / location.name,
        source_root.parent / location,
        source_root.parent / location.name,
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


def _matched_candidates_by_alias(
    candidates: tuple[SourceSchemaCandidate, ...],
    assignments: tuple[SourceAssignmentBase, ...],
    *,
    require_match: bool,
) -> Mapping[str, tuple[SourceSchemaCandidate, ...]]:
    matched: dict[str, tuple[SourceSchemaCandidate, ...]] = {}
    for assignment in assignments:
        alias_candidates = tuple(
            candidate
            for candidate in candidates
            if _candidate_matches_selector(candidate, assignment.selector)
        )
        image_candidates = tuple(
            candidate for candidate in alias_candidates if is_image_path(str(candidate.path))
        )
        selected_candidates = image_candidates if require_match else alias_candidates
        if require_match and not selected_candidates:
            raise ValueError(
                f"Source schema image alias {assignment.alias!r} matched no image files."
            )
        if selected_candidates:
            matched[assignment.alias] = selected_candidates
    return MappingProxyType(matched)


def _candidate_matches_selector(
    candidate: SourceSchemaCandidate,
    selector: SourceSelector,
) -> bool:
    return (
        _candidate_matches_components(candidate, selector)
        and _candidate_matches_metadata(candidate, selector)
        and source_filters_match(candidate.relative_path, selector.filters)
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
    grouped: Mapping[tuple[str, ...], Mapping[str, SourceSchemaCandidate]],
    metadata_by_key: Mapping[tuple[str, ...], Mapping[str, str]],
    candidates_by_alias: Mapping[str, tuple[SourceSchemaCandidate, ...]],
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
                metadata=_merged_image_set_metadata(
                    metadata_by_key[key],
                    candidates.values(),
                ),
            )
        )
    return tuple(image_sets)


def _merged_image_set_metadata(
    group_metadata: Mapping[str, str],
    candidates: Iterable[SourceSchemaCandidate],
) -> Mapping[str, str]:
    candidate_tuple = tuple(candidates)
    merged = dict(group_metadata)
    merge_source_metadata(
        merged,
        _shared_candidate_metadata(candidate_tuple),
        path="image_set",
    )
    merge_source_metadata(
        merged,
        _projected_candidate_components(merged, candidate_tuple),
        path="image_set",
    )
    return MappingProxyType(merged)


def _shared_candidate_metadata(
    candidates: tuple[SourceSchemaCandidate, ...],
) -> Mapping[str, str]:
    value_sets_by_key: dict[str, set[str]] = {}
    counts_by_key: dict[str, int] = {}
    for candidate in candidates:
        for key, value in candidate.metadata.items():
            value_sets_by_key.setdefault(key, set()).add(str(value))
            counts_by_key[key] = counts_by_key.get(key, 0) + 1
    candidate_count = len(candidates)
    return MappingProxyType(
        {
            key: next(iter(values))
            for key, values in value_sets_by_key.items()
            if counts_by_key[key] == candidate_count and len(values) == 1
        }
    )


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
            raise ValueError(
                f"Source image set has conflicting {component.value!r} component "
                f"values {sorted(values)!r}."
            )
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
) -> tuple[
    Mapping[str, str],
    Mapping[str, Mapping[str, str]],
    Mapping[AllComponents, Mapping[str, str | None]],
]:
    parser = SourceSchemaFilenameParser()
    channel_values = {
        str(index): assignment.alias
        for index, assignment in enumerate(stack_assignments, start=1)
    }
    wells: dict[str, None] = {}
    sites: dict[str, None] = {}
    primary_mappings: dict[str, str] = {}
    source_metadata: dict[str, Mapping[str, str]] = {}
    site_indexes_by_well: dict[str, int] = {}
    used_paths_by_well_channel: dict[tuple[str, int], set[str]] = {}
    for image_set in image_sets:
        well = SourceSchemaImageSetIdentity(schema, image_set).well
        site_index = site_indexes_by_well.get(well, 0)
        site = ComponentProjection.resolve(
            AllComponents.SITE,
            image_set.metadata,
            site_index,
        )
        site_indexes_by_well[well] = site_index + 1
        preferred_site_component = _component_ordinal_or_label(site)
        wells[well] = None
        sites[str(preferred_site_component)] = None
        for channel_index, assignment in enumerate(stack_assignments, start=1):
            candidate = image_set.candidates_by_alias[assignment.alias]
            site_component = _collision_free_site_component(
                parser,
                well=well,
                preferred_site_component=preferred_site_component,
                ordinal_site_component=site_index + 1,
                channel_index=channel_index,
                extension=candidate.path.suffix,
                used_paths=used_paths_by_well_channel.setdefault(
                    (well, channel_index),
                    set(),
                ),
            )
            sites[str(site_component)] = None
            virtual_path = parser.construct_filename(
                well=well,
                site=site_component,
                channel=channel_index,
                z_index=1,
                timepoint=1,
                extension=candidate.path.suffix,
            )
            _add_mapping(
                primary_mappings,
                virtual_path,
                _workspace_relative_path(workspace_root, candidate.path),
            )
            source_metadata[virtual_path] = _source_metadata_for_virtual_path(
                image_set.metadata,
                candidate.metadata,
                assignment=assignment,
            )
    component_values: Mapping[AllComponents, Mapping[str, str | None]] = MappingProxyType(
        {
            AllComponents.CHANNEL: MappingProxyType(channel_values),
            AllComponents.WELL: MappingProxyType(wells),
            AllComponents.SITE: MappingProxyType(sites),
            AllComponents.Z_INDEX: MappingProxyType({"1": None}),
            AllComponents.TIMEPOINT: MappingProxyType({"1": None}),
        }
    )
    return (
        MappingProxyType(primary_mappings),
        MappingProxyType(source_metadata),
        component_values,
    )


def _empty_workspace_component_values(
) -> Mapping[AllComponents, Mapping[str, str | None]]:
    return MappingProxyType(
        {
            AllComponents.CHANNEL: MappingProxyType({}),
            AllComponents.WELL: MappingProxyType({}),
            AllComponents.SITE: MappingProxyType({}),
            AllComponents.Z_INDEX: MappingProxyType({}),
            AllComponents.TIMEPOINT: MappingProxyType({}),
        }
    )


def _source_artifact_anchor_workspace_mappings(
    workspace_root: Path,
    auxiliary_candidates: Mapping[str, tuple[SourceSchemaCandidate, ...]],
    assignments_by_alias: Mapping[str, SourceAssignmentBase],
) -> tuple[
    Mapping[str, str],
    Mapping[str, Mapping[str, str]],
    Mapping[AllComponents, Mapping[str, str | None]],
]:
    anchor_alias, anchor_candidate = _source_artifact_anchor_candidate(
        auxiliary_candidates
    )
    parser = SourceSchemaFilenameParser()
    virtual_path = parser.construct_filename(
        well=SOURCE_SCHEMA_WORKSPACE_SINGLETON_AXIS_VALUE,
        site=1,
        channel=1,
        z_index=1,
        timepoint=1,
        extension=anchor_candidate.path.suffix,
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
            virtual_path: _source_metadata_for_virtual_path(
                {"source_alias": anchor_alias},
                anchor_candidate.metadata,
                assignment=assignments_by_alias.get(anchor_alias),
            )
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
    auxiliary_candidates: Mapping[str, tuple[SourceSchemaCandidate, ...]],
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
    parser: SourceSchemaFilenameParser,
    *,
    well: str,
    preferred_site_component: int | str,
    ordinal_site_component: int,
    channel_index: int,
    extension: str,
    used_paths: set[str],
) -> int | str:
    preferred_path = parser.construct_filename(
        well=well,
        site=preferred_site_component,
        channel=channel_index,
        z_index=1,
        timepoint=1,
        extension=extension,
    )
    if preferred_path not in used_paths:
        used_paths.add(preferred_path)
        return preferred_site_component

    ordinal_component = ordinal_site_component
    while True:
        ordinal_path = parser.construct_filename(
            well=well,
            site=ordinal_component,
            channel=channel_index,
            z_index=1,
            timepoint=1,
            extension=extension,
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
    auxiliary_candidates: Mapping[str, tuple[SourceSchemaCandidate, ...]],
    assignments_by_alias: Mapping[str, SourceAssignmentBase],
    *,
    filemanager: FileManagerLike | None,
    workspace_backend: str,
) -> tuple[Mapping[str, str], Mapping[str, Mapping[str, str]]]:
    mappings: dict[str, str] = {}
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
            _add_mapping(
                mappings,
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
            source_metadata[virtual_path] = _source_metadata_for_virtual_path(
                {"source_alias": alias},
                candidate.metadata,
                assignment=assignments_by_alias.get(alias),
            )
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


def _source_metadata_for_virtual_path(
    image_set_metadata: Mapping[str, str],
    candidate_metadata: Mapping[str, str],
    *,
    assignment: SourceAssignmentBase | None = None,
) -> Mapping[str, str]:
    metadata = dict(image_set_metadata)
    merge_source_metadata(metadata, candidate_metadata, path="source_metadata")
    if assignment is not None:
        image_type = _source_assignment_image_type(assignment)
        if image_type is not None:
            merge_source_metadata(
                metadata,
                {SOURCE_SCHEMA_IMAGE_TYPE_METADATA_FIELD: image_type},
                path="source_metadata",
            )
    return MappingProxyType(metadata)


def _source_assignment_image_type(
    assignment: SourceAssignmentBase,
) -> str | None:
    if isinstance(assignment, ImageAssignment):
        return assignment.image_type
    payload_type = getattr(assignment, "payload_type", "")
    return payload_type or None


def _write_workspace_metadata(
    metadata_path: Path,
    primary_mappings: Mapping[str, str],
    auxiliary_mappings: Mapping[str, str],
    component_values: Mapping[AllComponents, Mapping[str, str | None]],
    primary_source_metadata: Mapping[str, Mapping[str, str]],
    auxiliary_source_metadata: Mapping[str, Mapping[str, str]],
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
    workspace_mapping: Mapping[str, str],
    component_values: Mapping[AllComponents, Mapping[str, str | None]],
    source_metadata: Mapping[str, Mapping[str, str]],
    main: bool,
) -> dict[str, object]:
    return asdict(
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
                path: dict(metadata)
                for path, metadata in source_metadata.items()
            },
            main=main,
        )
    )


def _add_mapping(
    mappings: dict[str, str],
    virtual_path: str,
    real_path: str,
) -> None:
    existing = mappings.get(virtual_path)
    if existing is not None and existing != real_path:
        raise ValueError(
            f"Conflicting source workspace mapping for {virtual_path!r}: "
            f"{existing!r} != {real_path!r}."
        )
    mappings[virtual_path] = real_path


def _workspace_relative_path(workspace_root: Path, path: Path) -> str:
    return os.path.relpath(path, workspace_root).replace(os.sep, "/")


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


def _first_metadata_value(
    metadata: Mapping[str, str],
    normalized_keys: tuple[str, ...],
) -> str | None:
    for key in normalized_keys:
        value = source_metadata_value(metadata, key)
        if value is not None:
            return value
    return None


def _component_ordinal_or_label(value: str) -> int | str:
    return int(value) if value.isdecimal() else value
