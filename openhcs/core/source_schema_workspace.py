"""Project typed source schemas into native OpenHCS virtual workspaces."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.pipeline_image_schema import (
    CellProfilerImageSchema,
    ImageAssignment,
    ImageTypeSourceRole,
    SourceAssignmentBase,
)
from openhcs.core.source_bindings import (
    SourceBindingMatchMethod,
    SourceSelector,
)
from openhcs.core.source_matching import (
    is_image_path,
    merge_source_metadata,
    metadata_from_rules,
    source_filters_match,
)
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
from openhcs.microscopes.openhcs import FIELDS, OpenHCSMetadata


SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR = "_source"
SOURCE_SCHEMA_WORKSPACE_PIXEL_SIZE = 1.0
SOURCE_SCHEMA_WORKSPACE_GRID_DIMENSIONS = [1, 1]


@dataclass(frozen=True, slots=True)
class SourceSchemaWorkspaceMaterialization:
    """Result of projecting a source schema into an OpenHCS workspace."""

    source_root: Path
    workspace_root: Path
    metadata_path: Path
    primary_mappings: Mapping[str, str]
    auxiliary_mappings: Mapping[str, str]

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


class ComponentProjection(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for projecting source metadata onto OpenHCS components."""

    __registry_key__ = "__name__"
    component: ClassVar[AllComponents | None] = None
    priority: ClassVar[int] = 100
    metadata_derived: ClassVar[bool] = True

    @classmethod
    def resolve(
        cls,
        component: AllComponents,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str:
        projection_types = sorted(
            (
                projection_type
                for projection_type in cls.__registry__.values()
                if projection_type.component is component
            ),
            key=lambda projection_type: projection_type.priority,
        )
        for projection_type in projection_types:
            projection = projection_type()
            value = projection.value(metadata, image_set_index)
            if value is not None:
                return value
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
        projection_types = sorted(
            (
                projection_type
                for projection_type in cls.__registry__.values()
                if (
                    projection_type.component is component
                    and projection_type.metadata_derived
                )
            ),
            key=lambda projection_type: projection_type.priority,
        )
        for projection_type in projection_types:
            value = projection_type().value(metadata, 0)
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


class WellMetadataProjection(ComponentProjection):
    component = AllComponents.WELL
    priority = 10

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        return _metadata_value(metadata, "well")


class WellRowColumnMetadataProjection(ComponentProjection):
    component = AllComponents.WELL
    priority = 20

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


class OrdinalWellProjection(ComponentProjection):
    component = AllComponents.WELL
    priority = 1000
    metadata_derived = False

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        return f"A{image_set_index + 1:02d}"


class SiteMetadataProjection(ComponentProjection):
    component = AllComponents.SITE
    priority = 10

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        return _metadata_value(metadata, "site")


class ImageNumberSiteProjection(ComponentProjection):
    component = AllComponents.SITE
    priority = 20

    def value(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str | None:
        return _metadata_value(metadata, "imagenumber")


class OrdinalSiteProjection(ComponentProjection):
    component = AllComponents.SITE
    priority = 1000
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
        schema: CellProfilerImageSchema,
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
        schema: CellProfilerImageSchema,
        candidates_by_alias: Mapping[str, tuple[SourceSchemaCandidate, ...]],
    ) -> tuple[ImageSetRecord, ...]:
        """Assemble candidate groups for projection into OpenHCS files."""


class MetadataImageSetAssembler(ImageSetAssembler):
    method = SourceBindingMatchMethod.METADATA
    method_key = SourceBindingMatchMethod.METADATA.value

    def image_sets(
        self,
        schema: CellProfilerImageSchema,
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

    def image_sets(
        self,
        schema: CellProfilerImageSchema,
        candidates_by_alias: Mapping[str, tuple[SourceSchemaCandidate, ...]],
    ) -> tuple[ImageSetRecord, ...]:
        aliases = tuple(candidates_by_alias)
        lengths = {alias: len(candidates_by_alias[alias]) for alias in aliases}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "Order-based source projection requires each image alias to match "
                f"the same number of files, got {lengths!r}."
            )
        image_sets: list[ImageSetRecord] = []
        for index in range(next(iter(lengths.values()), 0)):
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
    schema: CellProfilerImageSchema,
) -> SourceSchemaWorkspaceMaterialization:
    """Create an OpenHCS virtual workspace from typed source-schema semantics."""

    source_root = Path(source_root)
    workspace_root = Path(workspace_root)
    if schema.is_empty:
        raise ValueError("Cannot materialize an empty source schema.")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    workspace_root.mkdir(parents=True, exist_ok=True)

    source_files = _source_files(source_root)
    candidates = _source_candidates(source_root, source_files, schema)
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
    image_sets = ImageSetAssembler.for_schema(schema).image_sets(
        schema,
        stack_candidates,
    )
    primary_mappings, component_values = _primary_workspace_mappings(
        workspace_root,
        image_sets,
        tuple(stack_assignments),
    )
    auxiliary_mappings = _auxiliary_workspace_mappings(
        workspace_root,
        auxiliary_candidates,
    )
    metadata_path = workspace_root / "openhcs_metadata.json"
    _write_workspace_metadata(
        metadata_path,
        primary_mappings,
        auxiliary_mappings,
        component_values,
    )
    return SourceSchemaWorkspaceMaterialization(
        source_root=source_root,
        workspace_root=workspace_root,
        metadata_path=metadata_path,
        primary_mappings=primary_mappings,
        auxiliary_mappings=auxiliary_mappings,
    )


def _partition_assignments(
    schema: CellProfilerImageSchema,
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
    if not stack_assignments:
        raise ValueError("Source schema declares no image-stack assignments.")
    return tuple(stack_assignments), tuple(auxiliary_assignments)


def _source_files(source_root: Path) -> tuple[Path, ...]:
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
    schema: CellProfilerImageSchema,
) -> tuple[SourceSchemaCandidate, ...]:
    candidates: list[SourceSchemaCandidate] = []
    for path in source_files:
        relative_path = path.relative_to(source_root).as_posix()
        metadata = metadata_from_rules(str(path), schema.metadata_rules)
        candidates.append(
            SourceSchemaCandidate(
                path=path,
                relative_path=relative_path,
                metadata=metadata,
            )
        )
    return tuple(candidates)


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
        and source_filters_match(str(candidate.path), selector.filters)
    )


def _candidate_matches_components(
    candidate: SourceSchemaCandidate,
    selector: SourceSelector,
) -> bool:
    return all(
        _metadata_value(candidate.metadata, component.component.value) == component.value
        for component in selector.components
    )


def _candidate_matches_metadata(
    candidate: SourceSchemaCandidate,
    selector: SourceSelector,
) -> bool:
    return all(
        candidate.metadata.get(metadata.field) == metadata.value
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
        existing = _metadata_value(group_metadata, _normalized_metadata_key(component.value))
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
    image_sets: tuple[ImageSetRecord, ...],
    stack_assignments: tuple[ImageAssignment, ...],
) -> tuple[Mapping[str, str], Mapping[AllComponents, Mapping[str, str | None]]]:
    parser = ImageXpressFilenameParser()
    channel_values = {
        str(index): assignment.alias
        for index, assignment in enumerate(stack_assignments, start=1)
    }
    wells: dict[str, None] = {}
    sites: dict[str, None] = {}
    primary_mappings: dict[str, str] = {}
    for image_set in image_sets:
        well = ComponentProjection.resolve(
            AllComponents.WELL,
            image_set.metadata,
            image_set.index,
        )
        site = ComponentProjection.resolve(
            AllComponents.SITE,
            image_set.metadata,
            image_set.index,
        )
        site_component = _component_ordinal_or_label(site)
        wells[well] = None
        sites[str(site_component)] = None
        for channel_index, assignment in enumerate(stack_assignments, start=1):
            candidate = image_set.candidates_by_alias[assignment.alias]
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
    component_values: Mapping[AllComponents, Mapping[str, str | None]] = MappingProxyType(
        {
            AllComponents.CHANNEL: MappingProxyType(channel_values),
            AllComponents.WELL: MappingProxyType(wells),
            AllComponents.SITE: MappingProxyType(sites),
            AllComponents.Z_INDEX: MappingProxyType({"1": None}),
            AllComponents.TIMEPOINT: MappingProxyType({"1": None}),
        }
    )
    return MappingProxyType(primary_mappings), component_values


def _auxiliary_workspace_mappings(
    workspace_root: Path,
    auxiliary_candidates: Mapping[str, tuple[SourceSchemaCandidate, ...]],
) -> Mapping[str, str]:
    mappings: dict[str, str] = {}
    for alias, candidates in auxiliary_candidates.items():
        for index, candidate in enumerate(candidates, start=1):
            virtual_path = (
                f"{SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR}/"
                f"{alias}/{index:03d}_{candidate.path.name}"
            )
            _add_mapping(
                mappings,
                virtual_path,
                _workspace_relative_path(workspace_root, candidate.path),
            )
    return MappingProxyType(mappings)


def _write_workspace_metadata(
    metadata_path: Path,
    primary_mappings: Mapping[str, str],
    auxiliary_mappings: Mapping[str, str],
    component_values: Mapping[AllComponents, Mapping[str, str | None]],
) -> None:
    subdirectories = {
        FIELDS.DEFAULT_SUBDIRECTORY: _metadata_dict(
            image_files=tuple(primary_mappings),
            workspace_mapping=primary_mappings,
            component_values=component_values,
            main=True,
        )
    }
    if auxiliary_mappings:
        subdirectories[SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR] = _metadata_dict(
            image_files=tuple(auxiliary_mappings),
            workspace_mapping=auxiliary_mappings,
            component_values=component_values,
            main=False,
        )
    metadata_path.write_text(
        json.dumps({FIELDS.SUBDIRECTORIES: subdirectories}, indent=2),
        encoding="utf-8",
    )


def _metadata_dict(
    *,
    image_files: tuple[str, ...],
    workspace_mapping: Mapping[str, str],
    component_values: Mapping[AllComponents, Mapping[str, str | None]],
    main: bool,
) -> dict[str, object]:
    return asdict(
        OpenHCSMetadata(
            microscope_handler_name=FIELDS.MICROSCOPE_TYPE,
            source_filename_parser_name="ImageXpressFilenameParser",
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


def _metadata_value(metadata: Mapping[str, str], normalized_key: str) -> str | None:
    for key, value in metadata.items():
        if _normalized_metadata_key(key) == normalized_key:
            return value
    return None


def _image_set_match_value(
    metadata: Mapping[str, str],
    field: str,
) -> str | None:
    value = _metadata_value(metadata, _normalized_metadata_key(field))
    if value is not None:
        return value
    component = _component_for_match_field(field)
    if component is None:
        return None
    return ComponentProjection.resolve_from_metadata(component, metadata)


def _component_for_match_field(field: str) -> AllComponents | None:
    normalized = _normalized_metadata_key(field)
    for component in AllComponents:
        if _normalized_metadata_key(component.value) == normalized:
            return component
    return None


def _first_metadata_value(
    metadata: Mapping[str, str],
    normalized_keys: tuple[str, ...],
) -> str | None:
    for key in normalized_keys:
        value = _metadata_value(metadata, key)
        if value is not None:
            return value
    return None


def _normalized_metadata_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def _component_ordinal_or_label(value: str) -> int | str:
    return int(value) if value.isdecimal() else value
