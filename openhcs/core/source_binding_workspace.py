"""Project generic source-binding declarations into OpenHCS workspaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, cast

from metaclass_registry import AutoRegisterMeta
from polystore.virtual_workspace import SourcePixelRef

from openhcs.constants.constants import AllComponents, Backend, Microscope
from openhcs.core.image_shapes import ArrayShape
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_image_values import (
    image_payload_data,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    ImportedMetadataTable,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingsConfig,
    SourceProjectionRole,
    SourceSetRole,
)
from openhcs.core.source_matching import (
    merge_source_metadata,
    metadata_from_rules,
    semantic_source_metadata_value,
    source_component_metadata_values,
    source_filters_match,
    source_metadata_component,
    source_metadata_value,
    source_metadata_values_equal,
    with_source_component_metadata,
)
from openhcs.core.source_metadata import (
    ORIGINAL_SOURCE_METADATA_FIELD,
    OriginalSourceMetadata,
    SourceFilterPathMetadata,
    SourceMetadataMapping,
    SourceMetadataRoleView,
    SourceMetadataScalar,
    SourceComponentProjectionStrategy,
)
from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourceCandidate,
    SourceDatasetDiagnostic,
    SourceArtifactProjection,
    SourcePlaneProjection,
    SourceProjection,
    SourceProjectionMetadataSerializer,
    SourceProjectionSet,
)
from openhcs.core.virtual_workspace_metadata import (
    AtomicMetadataWriter,
    FIELDS,
    get_metadata_path,
)
from openhcs.core.vfs_protocol import FileManagerLike


@dataclass(frozen=True, slots=True)
class _SourceSet:
    index: int
    candidates_by_alias: Mapping[str, SourceCandidate]
    metadata: SourceMetadataMapping

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidates_by_alias",
            MappingProxyType(dict(self.candidates_by_alias)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def with_imported_metadata(
        self,
        imported_metadata: SourceMetadataMapping,
        *,
        path: str,
    ) -> "_SourceSet":
        """Apply one later declared imported-metadata stage to this source set."""

        metadata = dict(self.metadata)
        metadata.update(imported_metadata)
        OriginalSourceMetadata.from_mapping(imported_metadata).overlay_into(
            metadata,
            path=path,
        )
        return replace(self, metadata=metadata)


@dataclass(frozen=True, slots=True)
class _ImportedMetadataIndex:
    """Validated rows for one exact imported-metadata declaration."""

    table: ImportedMetadataTable
    rows_by_signature: Mapping[
        tuple[SourceMetadataScalar, ...],
        SourceMetadataMapping,
    ]

    @classmethod
    def from_payload(
        cls,
        table: ImportedMetadataTable,
        payload: object,
        source_bindings: SourceBindingsConfig,
    ) -> "_ImportedMetadataIndex":
        if table.location is None:
            raise ValueError("Imported metadata table requires a declared location.")
        if not table.joins:
            raise ValueError(
                f"Imported metadata table {table.location!r} requires at least one join."
            )
        if not isinstance(payload, Sequence) or isinstance(
            payload,
            (str, bytes, bytearray),
        ):
            raise TypeError(
                f"Imported metadata table {table.location!r} must load as rows, got "
                f"{type(payload).__name__}."
            )
        if not payload:
            raise ValueError(
                f"Imported metadata table {table.location!r} contains no data rows."
            )

        rows_by_signature: dict[
            tuple[SourceMetadataScalar, ...],
            SourceMetadataMapping,
        ] = {}
        headers: tuple[str, ...] | None = None
        for row_index, raw_row in enumerate(payload, start=2):
            if not isinstance(raw_row, Mapping):
                raise TypeError(
                    f"Imported metadata table {table.location!r} row {row_index} "
                    f"must be a mapping, got {type(raw_row).__name__}."
                )
            row_headers = tuple(raw_row)
            if any(not isinstance(header, str) or not header for header in row_headers):
                raise ValueError(
                    f"Imported metadata table {table.location!r} has an invalid header."
                )
            if headers is None:
                headers = row_headers
                missing_join_fields = tuple(
                    join.imported_metadata_field
                    for join in table.joins
                    if join.imported_metadata_field not in headers
                )
                if missing_join_fields:
                    raise ValueError(
                        f"Imported metadata table {table.location!r} lacks declared join "
                        f"fields {missing_join_fields!r}."
                    )
            elif row_headers != headers:
                raise ValueError(
                    f"Imported metadata table {table.location!r} row {row_index} has "
                    "inconsistent columns."
                )

            row: dict[str, str] = {}
            for field_name, value in raw_row.items():
                if not isinstance(field_name, str):
                    raise ValueError(
                        f"Imported metadata table {table.location!r} row {row_index} "
                        "has an invalid header."
                    )
                if value is None:
                    raise ValueError(
                        f"Imported metadata table {table.location!r} row {row_index} "
                        f"has no value for field {field_name!r}."
                    )
                row[field_name] = str(value)

            imported_metadata = MappingProxyType(
                source_bindings.coerce_metadata(row)
            )
            signature = tuple(
                semantic_source_metadata_value(
                    imported_metadata,
                    join.imported_metadata_field,
                )
                for join in table.joins
            )
            if any(value is None or value == "" for value in signature):
                raise ValueError(
                    f"Imported metadata table {table.location!r} row {row_index} "
                    "has an empty join value."
                )
            rows_by_signature.setdefault(signature, imported_metadata)

        return cls(
            table=table,
            rows_by_signature=MappingProxyType(rows_by_signature),
        )

    def metadata_for_source_set(
        self,
        source_set: _SourceSet,
    ) -> SourceMetadataMapping | None:
        signature: list[SourceMetadataScalar] = []
        for join in self.table.joins:
            values = tuple(
                value
                for candidate in source_set.candidates_by_alias.values()
                if (
                    value := semantic_source_metadata_value(
                        candidate.metadata,
                        join.image_metadata_field,
                    )
                )
                is not None
            )
            if not values:
                return None
            first = values[0]
            if any(
                not source_metadata_values_equal(first, value)
                for value in values[1:]
            ):
                raise ValueError(
                    f"Source set {source_set.index} has conflicting values for imported "
                    f"metadata join field {join.image_metadata_field!r}: {values!r}."
                )
            signature.append(first)

        matching_row = self.rows_by_signature.get(tuple(signature))
        if matching_row is None:
            raise ValueError(
                f"Source set {source_set.index} has complete imported metadata join "
                f"identity {tuple(signature)!r}, but table {self.table.location!r} has no "
                "matching row."
            )
        return matching_row


class SourceSetAssembler(
    EnumKeyedStrategyMixin[SourceBindingMatchMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Assemble matched source members through the declared match method."""

    strategy_key: ClassVar[SourceBindingMatchMethod | None] = None

    @classmethod
    def for_config(cls, config: SourceBindingsConfig) -> "SourceSetAssembler":
        method = (
            SourceBindingMatchMethod.ORDER
            if config.match_plan is None
            else config.match_plan.method
        )
        return cls.for_enum_member(method)

    def source_sets(
        self,
        match_plan: SourceBindingMatchPlan | None,
        candidates_by_alias: Mapping[str, tuple[SourceCandidate, ...]],
        imported_metadata_indexes: tuple[_ImportedMetadataIndex, ...],
    ) -> tuple[_SourceSet, ...]:
        """Return matched source sets enriched by declared imported metadata."""

        source_sets = self._matched_source_sets(match_plan, candidates_by_alias)
        for imported_metadata_index in imported_metadata_indexes:
            enriched_sets: list[_SourceSet] = []
            for source_set in source_sets:
                imported_metadata = imported_metadata_index.metadata_for_source_set(
                    source_set
                )
                if imported_metadata is None:
                    enriched_sets.append(source_set)
                    continue
                enriched_sets.append(
                    source_set.with_imported_metadata(
                        imported_metadata,
                        path=cast(str, imported_metadata_index.table.location),
                    )
                )
            source_sets = tuple(enriched_sets)
        return source_sets

    @abstractmethod
    def _matched_source_sets(
        self,
        match_plan: SourceBindingMatchPlan | None,
        candidates_by_alias: Mapping[str, tuple[SourceCandidate, ...]],
    ) -> tuple[_SourceSet, ...]:
        """Return source sets matched by the selected strategy leaf."""


class MetadataSourceSetAssembler(SourceSetAssembler):
    strategy_key = SourceBindingMatchMethod.METADATA

    def _matched_source_sets(
        self,
        match_plan: SourceBindingMatchPlan | None,
        candidates_by_alias: Mapping[str, tuple[SourceCandidate, ...]],
    ) -> tuple[_SourceSet, ...]:
        if match_plan is None:
            raise ValueError("Metadata source-set assembly requires a match plan.")
        aliases = tuple(candidates_by_alias)
        dimension_count = len(match_plan.dimensions)
        if not aliases:
            return ()
        if dimension_count == 0:
            raise ValueError("Metadata source-set assembly requires match dimensions.")

        alias_dimensions = {
            alias: tuple(
                index
                for index, dimension in enumerate(match_plan.dimensions)
                if dimension.field_for_alias(alias) is not None
            )
            for alias in aliases
        }
        join_order = tuple(
            sorted(
                aliases,
                key=lambda alias: (-len(alias_dimensions[alias]), aliases.index(alias)),
            )
        )
        states: dict[
            tuple[SourceMetadataScalar, ...],
            dict[str, SourceCandidate],
        ] = {(None,) * dimension_count: {}}

        for alias in join_order:
            dimensions = alias_dimensions[alias]
            candidates_by_signature: dict[
                tuple[SourceMetadataScalar, ...], SourceCandidate
            ] = {}
            for candidate in candidates_by_alias[alias]:
                signature: list[SourceMetadataScalar] = []
                for dimension_index in dimensions:
                    field = match_plan.dimensions[dimension_index].field_for_alias(alias)
                    if field is None:
                        raise RuntimeError(
                            "Metadata source-set dimension ownership changed during assembly."
                        )
                    value = _source_set_match_value(candidate.metadata, field)
                    if value is None:
                        raise ValueError(
                            f"Source candidate {candidate.relative_path!r} for alias "
                            f"{alias!r} lacks source-set match metadata field {field!r}."
                        )
                    signature.append(value)
                signature_key = tuple(signature)
                if signature_key in candidates_by_signature:
                    raise ValueError(
                        f"Multiple source files match alias {alias!r} for source-set "
                        f"key {signature_key!r}."
                    )
                candidates_by_signature[signature_key] = candidate

            next_states: dict[
                tuple[SourceMetadataScalar, ...],
                dict[str, SourceCandidate],
            ] = {}
            for state_key, state_candidates in states.items():
                resolved_signature = tuple(state_key[index] for index in dimensions)
                if all(value is not None for value in resolved_signature):
                    candidate_items = tuple(
                        item
                        for item in (
                            candidates_by_signature.get(resolved_signature),
                        )
                        if item is not None
                    )
                else:
                    candidate_items = tuple(candidates_by_signature.values())
                for candidate in candidate_items:
                    signature = tuple(
                        _source_set_match_value(
                            candidate.metadata,
                            match_plan.dimensions[index].field_for_alias(alias),
                        )
                        for index in dimensions
                    )
                    if any(value is None for value in signature):
                        raise RuntimeError(
                            "Validated source-set metadata disappeared during assembly."
                        )
                    merged_key = list(state_key)
                    compatible = True
                    for dimension_index, value in zip(
                        dimensions,
                        signature,
                        strict=True,
                    ):
                        existing = merged_key[dimension_index]
                        if existing is not None and existing != value:
                            compatible = False
                            break
                        merged_key[dimension_index] = value
                    if not compatible:
                        continue
                    normalized_key = tuple(merged_key)
                    if normalized_key in next_states:
                        raise ValueError(
                            "Metadata source-set assembly produced ambiguous member "
                            f"selection for key {normalized_key!r}."
                        )
                    next_states[normalized_key] = {
                        **state_candidates,
                        alias: candidate,
                    }
            states = next_states

        incomplete_key = next(
            (key for key in states if any(value is None for value in key)),
            None,
        )
        if incomplete_key is not None:
            raise ValueError(
                "Metadata source-set key remained incomplete after assembly: "
                f"{incomplete_key!r}."
            )

        source_sets: list[_SourceSet] = []
        for index, key in enumerate(sorted(states)):
            candidates = states[key]
            metadata: dict[str, object] = {}
            for alias, candidate in candidates.items():
                for dimension in match_plan.dimensions:
                    field = dimension.field_for_alias(alias)
                    if field is None:
                        continue
                    value = _source_set_match_value(candidate.metadata, field)
                    if value is not None:
                        merge_source_metadata(
                            metadata,
                            {field: value},
                            path=candidate.relative_path,
                        )
            source_sets.append(
                _source_set(
                    index,
                    {alias: candidates[alias] for alias in aliases},
                    metadata,
                )
            )
        return tuple(source_sets)


class OrderSourceSetAssembler(SourceSetAssembler):
    strategy_key = SourceBindingMatchMethod.ORDER

    def _matched_source_sets(
        self,
        match_plan: SourceBindingMatchPlan | None,
        candidates_by_alias: Mapping[str, tuple[SourceCandidate, ...]],
    ) -> tuple[_SourceSet, ...]:
        del match_plan
        ordered_candidates = {
            alias: tuple(candidates)
            for alias, candidates in candidates_by_alias.items()
        }
        if not ordered_candidates:
            return ()
        aliases = tuple(ordered_candidates)
        source_ref_universes = {
            alias: tuple(candidate.source_ref for candidate in ordered_candidates[alias])
            for alias in aliases
        }
        ambiguous_aliases = tuple(
            (left_alias, right_alias)
            for left_index, left_alias in enumerate(aliases)
            for right_alias in aliases[left_index + 1 :]
            if source_ref_universes[left_alias]
            == source_ref_universes[right_alias]
        )
        if ambiguous_aliases:
            raise ValueError(
                "ORDER source bindings must distinguish aliases through declared "
                "source metadata or selectors; aliases share the same exact source "
                f"references: {ambiguous_aliases!r}."
            )
        source_set_count = min(map(len, ordered_candidates.values()))
        return tuple(
            _source_set(
                index,
                {
                    alias: ordered_candidates[alias][index]
                    for alias in aliases
                },
            )
            for index in range(source_set_count)
        )

class SourceBindingProjectionStrategy(
    EnumKeyedStrategyMixin[SourceProjectionRole],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project one binding member through its declared projection role."""

    strategy_key: ClassVar[SourceProjectionRole | None] = None

    @abstractmethod
    def projection(
        self,
        binding: NamedSourceBinding,
        candidate: SourceCandidate,
        address: OpenHCSPlaneAddress,
    ) -> SourceProjection:
        """Return one typed source projection."""


class PrimaryPlaneBindingProjection(SourceBindingProjectionStrategy):
    strategy_key = SourceProjectionRole.PRIMARY_PLANE

    def projection(
        self,
        binding: NamedSourceBinding,
        candidate: SourceCandidate,
        address: OpenHCSPlaneAddress,
    ) -> SourceProjection:
        component_labels = dict(candidate.component_labels)
        declared_channels = binding.component_values(AllComponents.CHANNEL)
        if len(declared_channels) == 1 and source_metadata_values_equal(
            declared_channels[0],
            address.channel,
        ):
            component_labels[AllComponents.CHANNEL.value] = binding.alias
        return SourcePlaneProjection(
            address=address,
            ref=candidate.source_ref,
            source_alias=binding.alias,
            source_metadata=candidate.metadata,
            component_labels=component_labels,
        )


class SourceArtifactBindingProjection(SourceBindingProjectionStrategy):
    strategy_key = SourceProjectionRole.SOURCE_ARTIFACT

    def projection(
        self,
        binding: NamedSourceBinding,
        candidate: SourceCandidate,
        address: OpenHCSPlaneAddress,
    ) -> SourceProjection:
        return SourceArtifactProjection(
            address=address,
            ref=candidate.source_ref,
            source_alias=binding.alias,
            artifact_kind=binding.artifact_kind,
            source_metadata=candidate.metadata,
            component_labels=candidate.component_labels,
        )


def _validate_source_binding_projection_registry() -> None:
    strategy_keys = {
        strategy_type.strategy_key
        for strategy_type in SourceBindingProjectionStrategy.registered_strategy_types()
    }
    expected = set(SourceProjectionRole)
    if strategy_keys != expected:
        raise RuntimeError(
            "Source binding projection registry does not exactly cover "
            f"SourceProjectionRole: {strategy_keys!r} != {expected!r}."
        )


_validate_source_binding_projection_registry()


@dataclass(frozen=True, slots=True)
class SourceBindingWorkspaceMaterialization:
    """Result of projecting source bindings into one OpenHCS workspace."""

    source_root: Path
    workspace_root: Path
    metadata_path: Path
    plane_mappings: Mapping[str, Mapping[str, object]]
    artifact_mappings: Mapping[str, Mapping[str, object]]
    source_metadata: Mapping[str, SourceMetadataMapping] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "workspace_root", Path(self.workspace_root))
        object.__setattr__(self, "metadata_path", Path(self.metadata_path))
        object.__setattr__(
            self,
            "plane_mappings",
            _frozen_workspace_mappings(self.plane_mappings),
        )
        object.__setattr__(
            self,
            "artifact_mappings",
            _frozen_workspace_mappings(self.artifact_mappings),
        )
        object.__setattr__(
            self,
            "source_metadata",
            MappingProxyType(
                {
                    str(path): MappingProxyType(dict(metadata))
                    for path, metadata in self.source_metadata.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceBindingWorkspaceProjector:
    """Project one resolved source-binding config into a virtual workspace."""

    source_bindings: SourceBindingsConfig
    parser: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_bindings, SourceBindingsConfig):
            raise TypeError(
                "SourceBindingWorkspaceProjector.source_bindings must be "
                f"SourceBindingsConfig, got {type(self.source_bindings).__name__}."
            )

    def projection_set(
        self,
        plate_path: Path,
        image_files: Sequence[str | Path],
        *,
        filemanager: FileManagerLike,
        source_backend: Backend | str = Backend.DISK,
    ) -> SourceProjectionSet:
        """Return source projections for the exact submitted source universe."""
        plate_path = Path(plate_path)
        candidates = self.source_candidates(
            plate_path,
            image_files,
            source_backend=source_backend,
        )
        return self.projection_set_for_candidates(
            plate_path,
            candidates,
            filemanager=filemanager,
        )

    def projection_set_for_candidates(
        self,
        plate_path: Path,
        candidates: tuple[SourceCandidate, ...],
        *,
        filemanager: FileManagerLike,
        diagnostics: tuple[SourceDatasetDiagnostic, ...] = (),
    ) -> SourceProjectionSet:
        """Return projections for one already-resolved candidate universe."""

        if not self.source_bindings.binding_declarations:
            return SourceProjectionSet(
                self._ungrouped_projections(candidates),
                diagnostics=diagnostics,
            )
        candidates_by_alias = self._candidates_by_alias(
            plate_path,
            candidates,
            filemanager=filemanager,
        )
        matched_bindings = self.source_bindings.matched_source_bindings
        if not matched_bindings:
            raise ValueError(
                "Source-binding workspace requires at least one matched source-set "
                "binding."
            )
        matched_candidates = {
            binding.alias: candidates_by_alias[binding.alias]
            for binding in matched_bindings
            if candidates_by_alias[binding.alias]
        }
        source_sets = SourceSetAssembler.for_config(self.source_bindings).source_sets(
            self.source_bindings.match_plan,
            matched_candidates,
            self._imported_metadata_indexes(plate_path, filemanager),
        )
        source_sets = self._append_broadcast_members(
            source_sets,
            candidates_by_alias,
        )
        return SourceProjectionSet(
            self._source_set_projections(
                self.source_bindings.binding_declarations,
                source_sets,
            ),
            diagnostics=diagnostics,
        )

    def _imported_metadata_indexes(
        self,
        source_root: Path,
        filemanager: FileManagerLike,
    ) -> tuple[_ImportedMetadataIndex, ...]:
        resolved_tables = tuple(
            table.resolved(source_root)
            for table in self.source_bindings.imported_metadata_tables
        )
        indexes: list[_ImportedMetadataIndex] = []
        for table in resolved_tables:
            if table.location is None:
                raise ValueError("Imported metadata table requires a declared location.")
            indexes.append(
                _ImportedMetadataIndex.from_payload(
                    table,
                    filemanager.load(table.location, Backend.DISK.value),
                    self.source_bindings,
                )
            )
        return tuple(indexes)

    def materialize(
        self,
        source_root: Path,
        workspace_root: Path,
        *,
        filemanager: FileManagerLike,
        source_backend: Backend | str,
        workspace_backend: Backend | str,
        source_files: Sequence[str | Path],
    ) -> SourceBindingWorkspaceMaterialization:
        """Materialize the exact declared source universe into OpenHCS metadata."""
        if self.parser is None:
            raise TypeError("Source workspace materialization requires a filename parser.")
        source_root = Path(source_root)
        workspace_root = Path(workspace_root)
        source_backend_name = _backend_name(source_backend)
        workspace_backend_name = _backend_name(workspace_backend)
        if not filemanager.is_dir(source_root, source_backend_name):
            raise FileNotFoundError(f"Source root does not exist: {source_root}")
        filemanager.ensure_directory(workspace_root, workspace_backend_name)
        candidates = self._materialized_source_candidates(
            source_root,
            source_files,
            filemanager=filemanager,
            source_backend=source_backend_name,
        )
        projection_set = self.projection_set_for_candidates(
            source_root,
            candidates,
            filemanager=filemanager,
        )
        primary_metadata = projection_set.metadata_dict(
            parser=self.parser,
            microscope_handler_name=Microscope.SOURCE_BINDINGS.value,
            source_filename_parser_name=type(self.parser).__name__,
            grid_dimensions=[1, 1],
            pixel_size=1.0,
            main=True,
        )
        metadata_path = get_metadata_path(workspace_root)
        AtomicMetadataWriter().replace_subdirectory_metadata(
            metadata_path,
            FIELDS.DEFAULT_SUBDIRECTORY,
            primary_metadata,
        )
        serializer = SourceProjectionMetadataSerializer(parser=self.parser)
        projection_paths = serializer.projection_paths(projection_set)
        plane_mappings = {
            path: projection.ref.to_workspace_mapping()
            for projection, path in projection_paths
            if projection.projection_role is SourceProjectionRole.PRIMARY_PLANE
        }
        artifact_mappings = {
            path: projection.ref.to_workspace_mapping()
            for projection, path in projection_paths
            if projection.projection_role is SourceProjectionRole.SOURCE_ARTIFACT
        }
        return SourceBindingWorkspaceMaterialization(
            source_root=source_root,
            workspace_root=workspace_root,
            metadata_path=metadata_path,
            plane_mappings=plane_mappings,
            artifact_mappings=artifact_mappings,
            source_metadata=primary_metadata[FIELDS.SOURCE_METADATA],
        )

    def _materialized_source_candidates(
        self,
        source_root: Path,
        image_files: Sequence[str | Path],
        *,
        filemanager: FileManagerLike,
        source_backend: Backend | str,
    ) -> tuple[SourceCandidate, ...]:
        """Resolve backend-owned addresses before persisting workspace refs."""

        return tuple(
            replace(
                candidate,
                source_ref=replace(
                    candidate.source_ref,
                    backend_address=str(
                        filemanager.resolve_address(
                            candidate.source_ref.backend_address,
                            candidate.source_ref.backend,
                            base_path=source_root,
                        )
                    ),
                ),
            )
            for candidate in self.source_candidates(
                source_root,
                image_files,
                source_backend=source_backend,
            )
        )

    def source_candidates(
        self,
        source_root: Path,
        image_files: Sequence[str | Path],
        *,
        source_backend: Backend | str,
    ) -> tuple[SourceCandidate, ...]:
        """Return validated candidates for the exact declared source universe."""
        backend_name = _backend_name(source_backend)
        declared_paths = tuple(image_files) + tuple(
            Path(source.resolved(source_root).uri)
            for source in self.source_bindings.image_plane_sources
        )
        candidates: list[SourceCandidate] = []
        seen: set[tuple[object, ...]] = set()
        for path_value in declared_paths:
            path = Path(path_value)
            relative_path = _relative_source_path(source_root, path)
            filter_paths = tuple(dict.fromkeys((relative_path, _normalized_source_path(path))))
            if not any(
                source_filters_match(
                    filter_path,
                    self.source_bindings.source_filter_declarations,
                )
                for filter_path in filter_paths
            ):
                continue
            metadata = self.source_bindings.coerce_metadata(
                metadata_from_rules(
                    str(path),
                    self.source_bindings.metadata_rule_declarations,
                    filter_path=relative_path,
                )
            )
            self.source_bindings.source_voxel_spacing.merge_into(
                metadata,
                path=relative_path,
            )
            candidate = SourceCandidate(
                source_ref=SourcePixelRef(
                    backend=backend_name,
                    backend_address=relative_path,
                ),
                relative_path=relative_path,
                metadata=metadata,
                source_filter_paths=filter_paths,
            )
            identity = candidate.identity_key()
            if identity not in seen:
                candidates.append(candidate)
                seen.add(identity)
        return tuple(candidates)

    def _candidates_by_alias(
        self,
        source_root: Path,
        candidates: tuple[SourceCandidate, ...],
        *,
        filemanager: FileManagerLike,
    ) -> Mapping[str, tuple[SourceCandidate, ...]]:
        candidates_by_alias: dict[str, tuple[SourceCandidate, ...]] = {}
        for binding in self.source_bindings.binding_declarations:
            selected = tuple(
                projected
                for candidate in candidates
                if self.candidate_matches_binding(candidate, binding, source_root)
                for projected in self._expanded_binding_candidates(
                    source_root,
                    self._candidate_with_binding_components(
                        candidate,
                        binding,
                    ),
                    binding,
                    filemanager=filemanager,
                )
            )
            if binding.required and not selected:
                raise ValueError(
                    f"Required source alias {binding.alias!r} matched no source files."
                )
            candidates_by_alias[binding.alias] = selected
        return MappingProxyType(candidates_by_alias)

    def candidate_matches_binding(
        self,
        candidate: SourceCandidate,
        binding: NamedSourceBinding,
        source_root: Path,
    ) -> bool:
        """Return whether one candidate satisfies one typed binding selector."""
        if binding.explicit_source is not None:
            explicit_path = Path(
                binding.explicit_source.resolved(source_root).uri
            ).resolve()
            candidate_path = Path(candidate.relative_path)
            if not candidate_path.is_absolute():
                candidate_path = Path(source_root) / candidate_path
            if candidate_path.resolve() != explicit_path:
                return False
        selector = binding.selector
        return (
            any(
                source_filters_match(path, selector.filters)
                for path in candidate.source_filter_path_identities()
            )
            and all(
                (value := semantic_source_metadata_value(candidate.metadata, item.field))
                is not None
                and source_metadata_values_equal(value, item.value)
                for item in selector.metadata
            )
            and all(
                self._component_is_compatible(candidate.metadata, item)
                for item in selector.components
            )
        )

    @staticmethod
    def _component_is_compatible(
        metadata: SourceMetadataMapping,
        selector: ComponentSelector,
    ) -> bool:
        values = source_component_metadata_values(metadata, selector.component)
        return not values or any(
            source_metadata_values_equal(value, selector.value)
            for value in values
        )

    def _candidate_with_binding_components(
        self,
        candidate: SourceCandidate,
        binding: NamedSourceBinding,
    ) -> SourceCandidate:
        metadata = dict(candidate.metadata)
        assignments: dict[AllComponents, str] = {}
        for selector in (*binding.selector.components, *binding.component_identity):
            existing = assignments.get(selector.component)
            if existing is not None and not source_metadata_values_equal(
                existing,
                selector.value,
            ):
                raise ValueError(
                    f"Binding {binding.alias!r} assigns conflicting "
                    f"{selector.component.value!r} values."
                )
            assignments[selector.component] = selector.value
        for component, value in assignments.items():
            current = SourceComponentProjectionStrategy.metadata_component(
                component,
                metadata,
            )
            if current is not None and not source_metadata_values_equal(current, value):
                raise ValueError(
                    f"Source candidate {candidate.relative_path!r} has conflicting "
                    f"{component.value!r} values {current!r} and {value!r}."
                )
            metadata = with_source_component_metadata(metadata, component, value)
        return replace(candidate, metadata=metadata)

    def _expanded_binding_candidates(
        self,
        source_root: Path,
        candidate: SourceCandidate,
        binding: NamedSourceBinding,
        *,
        filemanager: FileManagerLike,
    ) -> tuple[SourceCandidate, ...]:
        if candidate.declared_address is not None:
            return (candidate,)
        if binding.source_set_role is SourceSetRole.BROADCAST:
            return (candidate,)
        components = self.source_bindings.source_stack_components
        if not components:
            return (candidate,)
        address = filemanager.resolve_address(
            candidate.source_ref.backend_address,
            candidate.source_ref.backend,
            base_path=source_root,
        )
        payload = filemanager.load(address, candidate.source_ref.backend)
        shape = ArrayShape.shape_for(image_payload_data(payload))
        if shape is None or len(shape) < len(components) + 2:
            raise ValueError(
                f"Source {candidate.relative_path!r} shape {shape!r} cannot expose "
                f"{len(components)} declared source stack axes."
            )
        binding.source_channel_axis_for_shape(
            shape,
            source_axis_count=len(components),
        )
        source_axis_shape = tuple(shape[: len(components)])
        expanded: list[SourceCandidate] = []
        for source_axis_indices in product(
            *(range(size) for size in source_axis_shape)
        ):
            metadata = dict(candidate.metadata)
            for component, index in zip(
                components,
                source_axis_indices,
                strict=True,
            ):
                metadata = with_source_component_metadata(
                    metadata,
                    component,
                    str(index + 1),
                )
            expanded.append(
                replace(
                    candidate,
                    source_ref=replace(
                        candidate.source_ref,
                        source_axis_indices=source_axis_indices,
                    ),
                    metadata=metadata,
                    source_axis_shape=source_axis_shape,
                )
            )
        return tuple(expanded)

    def _append_broadcast_members(
        self,
        source_sets: tuple[_SourceSet, ...],
        candidates_by_alias: Mapping[str, tuple[SourceCandidate, ...]],
    ) -> tuple[_SourceSet, ...]:
        broadcast_members: dict[str, SourceCandidate] = {}
        for binding in self.source_bindings.broadcast_source_bindings:
            candidates = candidates_by_alias[binding.alias]
            if not candidates and not binding.required:
                continue
            if len(candidates) != 1:
                raise ValueError(
                    f"Broadcast source alias {binding.alias!r} must resolve exactly "
                    f"one source, got {len(candidates)}."
                )
            broadcast_members[binding.alias] = candidates[0]
        if not broadcast_members:
            return source_sets
        return tuple(
            _source_set(
                source_set.index,
                {**source_set.candidates_by_alias, **broadcast_members},
                source_set.metadata,
            )
            for source_set in source_sets
        )

    def _source_set_projections(
        self,
        bindings: tuple[NamedSourceBinding, ...],
        source_sets: tuple[_SourceSet, ...],
    ) -> tuple[SourceProjection, ...]:
        projections: list[SourceProjection] = []
        for source_set in source_sets:
            well = self._source_set_well(source_set)
            projection_indexes = {
                role: 0 for role in SourceProjectionRole
            }
            for binding in bindings:
                candidate = source_set.candidates_by_alias.get(binding.alias)
                if candidate is None:
                    continue
                projection_index = projection_indexes[binding.projection_role]
                projection_indexes[binding.projection_role] += 1
                address = candidate.declared_address or OpenHCSPlaneAddress(
                    well=well,
                    site=SourceComponentProjectionStrategy.project_component(
                        AllComponents.SITE,
                        source_set.metadata,
                        source_set.index,
                    ),
                    channel=SourceComponentProjectionStrategy.project_component(
                        AllComponents.CHANNEL,
                        candidate.metadata,
                        projection_index,
                    ),
                    z_index=SourceComponentProjectionStrategy.project_component(
                        AllComponents.Z_INDEX,
                        source_set.metadata,
                        source_set.index,
                    ),
                    timepoint=SourceComponentProjectionStrategy.project_component(
                        AllComponents.TIMEPOINT,
                        source_set.metadata,
                        source_set.index,
                    ),
                )
                projections.append(
                    SourceBindingProjectionStrategy.for_enum_member(
                        binding.projection_role
                    ).projection(
                        binding,
                        replace(
                            candidate,
                            metadata=_workspace_source_metadata(
                                candidate,
                                source_set.metadata,
                            ),
                        ),
                        address,
                    )
                )
        return tuple(projections)

    def _source_set_well(self, source_set: _SourceSet) -> str:
        fields = self.source_bindings.grouping_metadata_fields
        if not fields:
            return SourceComponentProjectionStrategy.project_component(
                AllComponents.WELL,
                source_set.metadata,
                source_set.index,
            )

        group_metadata: dict[str, str] = {}
        for field_name in fields:
            value = semantic_source_metadata_value(source_set.metadata, field_name)
            if value is None or str(value) == "":
                raise ValueError(
                    "Grouped source set lacks non-empty metadata field "
                    f"{field_name!r}; available fields are {sorted(source_set.metadata)}."
                )
            group_metadata[field_name] = str(value)

        fields_are_well_identity = all(
            SourceComponentProjectionStrategy.component_for_metadata_field(field)
            is AllComponents.WELL
            for field in fields
        )
        if fields_are_well_identity:
            well = SourceComponentProjectionStrategy.metadata_component(
                AllComponents.WELL,
                group_metadata,
            )
            if well is None:
                raise ValueError(
                    "Source grouping fields declared well identity but did not "
                    f"resolve one from {group_metadata!r}."
                )
            return well

        encoded_fields: list[str] = []
        for field_name in fields:
            values = (
                (group_metadata[field_name],)
                if len(fields) == 1
                else (field_name, group_metadata[field_name])
            )
            encoded_values: list[str] = []
            for value in values:
                encoded = "".join(
                    chr(byte)
                    if (
                        48 <= byte <= 57
                        or 65 <= byte <= 90
                        or 97 <= byte <= 122
                        or byte in (45, 46)
                    )
                    else f"%{byte:02X}"
                    for byte in value.encode("utf-8")
                )
                if not encoded:
                    raise ValueError("Source component tokens cannot be empty.")
                encoded_values.append(encoded)
            encoded_fields.append("-".join(encoded_values))
        return "-".join(encoded_fields)

    def _ungrouped_projections(
        self,
        candidates: tuple[SourceCandidate, ...],
    ) -> tuple[SourcePlaneProjection, ...]:
        return tuple(
            SourcePlaneProjection(
                address=candidate.declared_address or OpenHCSPlaneAddress(
                    well=SourceComponentProjectionStrategy.project_component(
                        AllComponents.WELL,
                        candidate.metadata,
                        index,
                    ),
                    site=SourceComponentProjectionStrategy.project_component(
                        AllComponents.SITE,
                        candidate.metadata,
                        index,
                    ),
                    channel=SourceComponentProjectionStrategy.project_component(
                        AllComponents.CHANNEL,
                        candidate.metadata,
                        index,
                    ),
                    z_index=SourceComponentProjectionStrategy.project_component(
                        AllComponents.Z_INDEX,
                        candidate.metadata,
                        index,
                    ),
                    timepoint=SourceComponentProjectionStrategy.project_component(
                        AllComponents.TIMEPOINT,
                        candidate.metadata,
                        index,
                    ),
                ),
                ref=candidate.source_ref,
                source_metadata=_workspace_source_metadata(candidate),
                component_labels=candidate.component_labels,
            )
            for index, candidate in enumerate(candidates)
        )

def materialize_source_binding_workspace(
    source_root: Path,
    workspace_root: Path,
    source_bindings: SourceBindingsConfig,
    *,
    filemanager: FileManagerLike,
    source_backend: Backend | str,
    workspace_backend: Backend | str,
    source_files: Sequence[str | Path],
    parser: object,
) -> SourceBindingWorkspaceMaterialization:
    """Materialize one resolved generic source-binding configuration."""
    return SourceBindingWorkspaceProjector(source_bindings, parser=parser).materialize(
        source_root,
        workspace_root,
        filemanager=filemanager,
        source_backend=source_backend,
        workspace_backend=workspace_backend,
        source_files=source_files,
    )


def _source_set(
    index: int,
    candidates_by_alias: Mapping[str, SourceCandidate],
    group_metadata: SourceMetadataMapping | None = None,
) -> _SourceSet:
    candidates = tuple(candidates_by_alias.values())
    metadata = dict(group_metadata or {})
    merge_source_metadata(
        metadata,
        _shared_candidate_metadata(candidates),
        path="source_set",
    )
    merge_source_metadata(
        metadata,
        _projected_candidate_components(metadata, candidates),
        path="source_set",
    )
    return _SourceSet(index, candidates_by_alias, metadata)


def _workspace_source_metadata(
    candidate: SourceCandidate,
    source_set_metadata: SourceMetadataMapping | None = None,
) -> SourceMetadataMapping:
    """Project candidate and matched-set metadata into persisted source metadata."""

    metadata = dict(candidate.metadata)
    if source_set_metadata is not None:
        source_set_original_metadata = source_set_metadata.get(
            ORIGINAL_SOURCE_METADATA_FIELD
        )
        metadata.update(
            (field_name, value)
            for field_name, value in source_set_metadata.items()
            if field_name != ORIGINAL_SOURCE_METADATA_FIELD
        )
        if source_set_original_metadata is not None:
            OriginalSourceMetadata.from_reserved_value(
                source_set_original_metadata,
                path=candidate.relative_path,
            ).overlay_into(metadata, path=candidate.relative_path)
    SourceFilterPathMetadata.from_paths(
        candidate.source_filter_path_identities()
    ).merge_into(metadata, path=candidate.relative_path)
    return MappingProxyType(metadata)


def _shared_candidate_metadata(
    candidates: tuple[SourceCandidate, ...],
) -> SourceMetadataMapping:
    value_sets_by_key: dict[str, set[object]] = {}
    counts_by_key: dict[str, int] = {}
    for candidate in candidates:
        for key, value in SourceMetadataRoleView(candidate.metadata).scalar_items():
            value_sets_by_key.setdefault(key, set()).add(value)
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
    group_metadata: SourceMetadataMapping,
    candidates: tuple[SourceCandidate, ...],
) -> SourceMetadataMapping:
    projected: dict[str, str] = {}
    for component in AllComponents:
        values = {
            value
            for candidate in candidates
            if (
                value := SourceComponentProjectionStrategy.metadata_component(
                    component,
                    candidate.metadata,
                )
            )
            is not None
        }
        if len(values) > 1:
            continue
        if not values:
            continue
        value = next(iter(values))
        existing = source_metadata_value(group_metadata, component.value)
        if existing is not None and not source_metadata_values_equal(existing, value):
            raise ValueError(
                f"Source set has conflicting {component.value!r} values "
                f"{existing!r} and {value!r}."
            )
        if existing is None:
            projected[component.value] = value
    return MappingProxyType(projected)


def _source_set_match_value(
    metadata: SourceMetadataMapping,
    field: str,
) -> SourceMetadataScalar:
    value = semantic_source_metadata_value(metadata, field)
    if value is not None:
        return value
    component = source_metadata_component(field)
    if component is None:
        return None
    return SourceComponentProjectionStrategy.metadata_component(component, metadata)




def _relative_source_path(source_root: Path, source_path: Path) -> str:
    try:
        relative = source_path.relative_to(source_root)
    except ValueError:
        relative = source_path
    return _normalized_source_path(relative)


def _normalized_source_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def _backend_name(backend: Backend | str) -> str:
    return backend.value if isinstance(backend, Backend) else str(backend)


def _frozen_workspace_mappings(
    mappings: Mapping[str, Mapping[str, object]],
) -> Mapping[str, Mapping[str, object]]:
    return MappingProxyType(
        {
            str(path): MappingProxyType(dict(source_ref))
            for path, source_ref in mappings.items()
        }
    )
