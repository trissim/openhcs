"""Presentation-ready typed views over source binding declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.pipeline_image_schema import (
    GroupingPlan,
    ImageAssignment,
    ImagePlaneSource,
    ImportedMetadataTable,
    ImagesRule,
    PipelineImageSchema,
    SourceAssignmentBase,
    SourceArtifactAssignment,
)
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.core.source_bindings import (
    GroupedSourceBindings,
    MetadataExtractionRule,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchPlan,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.source_matching import (
    metadata_from_rules,
    semantic_source_metadata_value,
    source_component_metadata_values,
    source_filters_match,
    source_metadata_values_equal,
)
from openhcs.core.source_schema_workspace import (
    ImageSetAssembler,
    ImageSetRecord,
    SourceSchemaCandidate,
)


@dataclass(frozen=True, slots=True)
class SourceFilterView:
    """UI-neutral view of one source filter clause."""

    subject: str
    match_type: str
    value: str | None

    @classmethod
    def from_clause(cls, clause: SourceFilterClause) -> "SourceFilterView":
        return cls(
            subject=clause.subject.value,
            match_type=clause.match_type.value,
            value=clause.value,
        )


@dataclass(frozen=True, slots=True)
class SourceSelectorView:
    """UI-neutral view of component, metadata, and filter source selectors."""

    components: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    filters: tuple[SourceFilterView, ...] = ()
    inherit_current_scope: bool = True

    @classmethod
    def from_selector(cls, selector: SourceSelector) -> "SourceSelectorView":
        return cls(
            components=tuple(
                (cls.component_name(component_selector.component), component_selector.value)
                for component_selector in selector.components
            ),
            metadata=tuple(
                (metadata_selector.field, metadata_selector.value)
                for metadata_selector in selector.metadata
            ),
            filters=tuple(
                SourceFilterView.from_clause(clause) for clause in selector.filters
            ),
            inherit_current_scope=selector.inherit_current_scope,
        )

    @staticmethod
    def component_name(component: object) -> str:
        if isinstance(component, AllComponents):
            return component.value
        return str(component)

    @property
    def is_empty(self) -> bool:
        return (
            not self.components
            and not self.metadata
            and not self.filters
            and self.inherit_current_scope
        )


@dataclass(frozen=True, slots=True)
class SourceBindingView:
    """UI-neutral row for one named source binding or schema assignment."""

    alias: str
    artifact_kind: str
    origin: str
    required: bool
    selector: SourceSelectorView
    declaration_scope: str
    group_key: str | None = None
    payload_type: str | None = None

    @classmethod
    def from_named_binding(
        cls,
        binding: NamedSourceBinding,
        *,
        declaration_scope: str,
        group_key: str | None,
    ) -> "SourceBindingView":
        return cls(
            alias=binding.alias,
            artifact_kind=binding.artifact_kind.value,
            origin=binding.origin.value,
            required=binding.required,
            selector=SourceSelectorView.from_selector(binding.selector),
            declaration_scope=declaration_scope,
            group_key=group_key,
        )

    @classmethod
    def from_schema_assignment(
        cls,
        assignment: SourceAssignmentBase,
        *,
        declaration_scope: str,
    ) -> "SourceBindingView":
        payload_type = SourceBindingView.payload_type_for_assignment(assignment)
        return cls(
            alias=assignment.alias,
            artifact_kind=assignment.artifact_kind.value,
            origin=assignment.origin.value,
            required=True,
            selector=SourceSelectorView.from_selector(assignment.selector),
            declaration_scope=declaration_scope,
            payload_type=payload_type or None,
        )

    @staticmethod
    def payload_type_for_assignment(assignment: SourceAssignmentBase) -> str:
        if isinstance(assignment, ImageAssignment):
            return assignment.image_type
        if isinstance(assignment, SourceArtifactAssignment):
            return assignment.payload_type
        raise TypeError(
            "Unsupported source assignment view type "
            f"{type(assignment).__name__}."
        )


@dataclass(frozen=True, slots=True)
class SourceBindingGroupView:
    """UI-neutral group of step-local source bindings."""

    group_key: str | None
    bindings: tuple[SourceBindingView, ...]

    @classmethod
    def from_group(cls, group: GroupedSourceBindings) -> "SourceBindingGroupView":
        return cls(
            group_key=group.group_key,
            bindings=tuple(
                SourceBindingView.from_named_binding(
                    binding,
                    declaration_scope="step",
                    group_key=group.group_key,
                )
                for binding in group.bindings
            ),
        )


@dataclass(frozen=True, slots=True)
class MetadataRuleView:
    """UI-neutral row for one metadata extraction rule."""

    source: str
    pattern: str
    extracted_fields: tuple[str, ...]
    filters: tuple[SourceFilterView, ...]
    declaration_scope: str

    @classmethod
    def from_rule(
        cls,
        rule: MetadataExtractionRule,
        *,
        declaration_scope: str,
    ) -> "MetadataRuleView":
        return cls(
            source=rule.source.value,
            pattern=rule.pattern,
            extracted_fields=tuple(re.compile(rule.pattern).groupindex),
            filters=tuple(SourceFilterView.from_clause(clause) for clause in rule.filters),
            declaration_scope=declaration_scope,
        )


@dataclass(frozen=True, slots=True)
class MatchDimensionView:
    """UI-neutral row for one cross-alias image-set matching dimension."""

    fields: tuple[tuple[str, str], ...]

    @classmethod
    def from_dimension(
        cls,
        dimension: SourceBindingMatchDimension,
    ) -> "MatchDimensionView":
        return cls(
            fields=tuple(
                (field.alias, field.metadata_field) for field in dimension.fields
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceBindingMatchPlanView:
    """UI-neutral view of how source aliases are matched into image sets."""

    method: str
    dimensions: tuple[MatchDimensionView, ...]
    declaration_scope: str

    @classmethod
    def from_plan(
        cls,
        plan: SourceBindingMatchPlan,
        *,
        declaration_scope: str,
    ) -> "SourceBindingMatchPlanView":
        return cls(
            method=plan.method.value,
            dimensions=tuple(
                MatchDimensionView.from_dimension(dimension)
                for dimension in plan.dimensions
            ),
            declaration_scope=declaration_scope,
        )


@dataclass(frozen=True, slots=True)
class GroupingView:
    """UI-neutral view of pipeline-level grouping semantics."""

    metadata_fields: tuple[str, ...]

    @classmethod
    def from_grouping(cls, grouping: GroupingPlan) -> "GroupingView":
        return cls(metadata_fields=grouping.metadata_fields)


@dataclass(frozen=True, slots=True)
class ImportedMetadataTableView:
    """UI-neutral view of one imported metadata source."""

    location: str | None
    joins: tuple[tuple[str, str], ...]

    @classmethod
    def from_table(cls, table: ImportedMetadataTable) -> "ImportedMetadataTableView":
        return cls(
            location=table.location,
            joins=tuple(
                (join.image_metadata_field, join.imported_metadata_field)
                for join in table.joins
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineSourceUniverseView:
    """UI-neutral summary of pipeline-level source universe declarations."""

    filters: tuple[SourceFilterView, ...]
    image_plane_source_count: int
    imported_metadata_tables: tuple[ImportedMetadataTableView, ...]

    @classmethod
    def from_schema(cls, schema: PipelineImageSchema) -> "PipelineSourceUniverseView":
        return cls(
            filters=cls.filters_from_images_rule(schema.images_rule),
            image_plane_source_count=len(schema.image_plane_sources),
            imported_metadata_tables=tuple(
                ImportedMetadataTableView.from_table(table)
                for table in schema.imported_metadata_tables
            ),
        )

    @staticmethod
    def filters_from_images_rule(
        images_rule: ImagesRule | None,
    ) -> tuple[SourceFilterView, ...]:
        if images_rule is None:
            return ()
        return tuple(SourceFilterView.from_clause(clause) for clause in images_rule.filters)


@dataclass(frozen=True, slots=True)
class SourceBindingsViewModel:
    """Typed, GUI-neutral source-binding state for editor and preview surfaces."""

    pipeline_sources: PipelineSourceUniverseView
    pipeline_bindings: tuple[SourceBindingView, ...]
    step_binding_groups: tuple[SourceBindingGroupView, ...]
    metadata_rules: tuple[MetadataRuleView, ...]
    match_plans: tuple[SourceBindingMatchPlanView, ...]
    grouping: GroupingView | None

    @classmethod
    def from_schema_and_bindings(
        cls,
        *,
        schema: PipelineImageSchema,
        bindings: StepSourceBindingsConfig,
    ) -> "SourceBindingsViewModel":
        return cls(
            pipeline_sources=PipelineSourceUniverseView.from_schema(schema),
            pipeline_bindings=cls.pipeline_binding_views(schema),
            step_binding_groups=tuple(
                SourceBindingGroupView.from_group(group) for group in bindings.groups
            ),
            metadata_rules=cls.metadata_rule_views(schema, bindings),
            match_plans=cls.match_plan_views(schema, bindings),
            grouping=(
                None
                if schema.grouping is None
                else GroupingView.from_grouping(schema.grouping)
            ),
        )

    @staticmethod
    def pipeline_binding_views(
        schema: PipelineImageSchema,
    ) -> tuple[SourceBindingView, ...]:
        image_bindings = tuple(
            SourceBindingView.from_schema_assignment(
                assignment,
                declaration_scope="pipeline_image_schema",
            )
            for assignment in schema.assignments_by_alias.values()
        )
        artifact_bindings = tuple(
            SourceBindingView.from_schema_assignment(
                assignment,
                declaration_scope="pipeline_source_artifact_schema",
            )
            for assignment in schema.source_artifacts_by_alias.values()
        )
        return tuple(
            sorted(
                image_bindings + artifact_bindings,
                key=lambda row: (row.alias.lower(), row.artifact_kind),
            )
        )

    @staticmethod
    def metadata_rule_views(
        schema: PipelineImageSchema,
        bindings: StepSourceBindingsConfig,
    ) -> tuple[MetadataRuleView, ...]:
        return tuple(
            MetadataRuleView.from_rule(rule, declaration_scope="pipeline_image_schema")
            for rule in schema.metadata_rules
        ) + tuple(
            MetadataRuleView.from_rule(rule, declaration_scope="step")
            for rule in bindings.metadata_rules
        )

    @staticmethod
    def match_plan_views(
        schema: PipelineImageSchema,
        bindings: StepSourceBindingsConfig,
    ) -> tuple[SourceBindingMatchPlanView, ...]:
        plans: list[SourceBindingMatchPlanView] = []
        if schema.match_plan is not None:
            plans.append(
                SourceBindingMatchPlanView.from_plan(
                    schema.match_plan,
                    declaration_scope="pipeline_image_schema",
                )
            )
        if bindings.match_plan is not None:
            plans.append(
                SourceBindingMatchPlanView.from_plan(
                    bindings.match_plan,
                    declaration_scope="step",
                )
            )
        return tuple(plans)

    @property
    def all_bindings(self) -> tuple[SourceBindingView, ...]:
        return self.pipeline_bindings + tuple(
            binding
            for group in self.step_binding_groups
            for binding in group.bindings
        )

    @property
    def artifact_kinds(self) -> tuple[str, ...]:
        return tuple(
            sorted({binding.artifact_kind for binding in self.all_bindings})
        )


@dataclass(frozen=True, slots=True)
class SourceInventoryBuildRequest:
    """Authoritative build request for source-inventory candidate projection."""

    paths: tuple[str | Path, ...]
    schema: PipelineImageSchema
    bindings: StepSourceBindingsConfig = StepSourceBindingsConfig()
    source_root: str | Path | None = None

    def build(self) -> "SourceInventory":
        return SourceInventory.from_paths(
            self.paths,
            schema=self.schema,
            bindings=self.bindings,
            source_root=self.source_root,
        )


@dataclass(frozen=True, slots=True)
class SourceInventory:
    """Resolved source candidates available for source-binding previews."""

    candidates: tuple[SourceSchemaCandidate, ...]

    @classmethod
    def from_paths(
        cls,
        paths: tuple[str | Path, ...],
        *,
        schema: PipelineImageSchema,
        bindings: StepSourceBindingsConfig = StepSourceBindingsConfig(),
        source_root: str | Path | None = None,
    ) -> "SourceInventory":
        root = None if source_root is None else Path(source_root)
        metadata_rules = schema.metadata_rules + bindings.metadata_rules
        candidates = tuple(
            SourceInventory.candidate_from_path(
                path,
                metadata_rules=metadata_rules,
                source_root=root,
            )
            for path in paths
        )
        if schema.images_rule is not None:
            candidates = tuple(
                candidate
                for candidate in candidates
                if source_filters_match(
                    candidate.relative_path,
                    schema.images_rule.filters,
                )
            )
        return cls(candidates=candidates)

    @classmethod
    def from_directory(
        cls,
        source_root: str | Path,
        *,
        schema: PipelineImageSchema,
        bindings: StepSourceBindingsConfig = StepSourceBindingsConfig(),
    ) -> "SourceInventory":
        """Build preview candidates from a local source directory."""

        root = Path(source_root)
        return SourceInventoryBuildRequest(
            paths=tuple(path for path in sorted(root.rglob("*")) if path.is_file()),
            schema=schema,
            bindings=bindings,
            source_root=root,
        ).build()

    @classmethod
    def from_filemanager(
        cls,
        *,
        filemanager: FileManagerLike,
        source_root: str | Path,
        backend: str,
        schema: PipelineImageSchema,
        bindings: StepSourceBindingsConfig = StepSourceBindingsConfig(),
    ) -> "SourceInventory":
        """Build preview candidates from an OpenHCS VFS/FileManager backend."""

        paths = tuple(
            sorted(
                str(path)
                for path in filemanager.list_files(
                    source_root,
                    backend,
                    recursive=True,
                )
            )
        )
        return SourceInventoryBuildRequest(
            paths=paths,
            schema=schema,
            bindings=bindings,
            source_root=source_root,
        ).build()

    @classmethod
    def from_schema_context(
        cls,
        schema: PipelineImageSchema,
        *,
        bindings: StepSourceBindingsConfig = StepSourceBindingsConfig(),
        source_root: str | Path | None = None,
    ) -> "SourceInventory":
        """Build the best available inventory for a schema preview context."""

        if schema.image_plane_sources:
            return cls.from_schema_sources(
                schema,
                bindings=bindings,
                source_root=source_root,
            )
        if source_root is None:
            return cls(candidates=())
        return cls.from_directory(
            source_root,
            schema=schema,
            bindings=bindings,
        )

    @classmethod
    def from_schema_sources(
        cls,
        schema: PipelineImageSchema,
        *,
        bindings: StepSourceBindingsConfig = StepSourceBindingsConfig(),
        source_root: str | Path | None = None,
    ) -> "SourceInventory":
        """Build preview candidates from explicit pipeline image-plane sources."""

        return SourceInventoryBuildRequest(
            paths=tuple(
                SourceInventory.path_from_image_plane_source(source)
                for source in schema.image_plane_sources
            ),
            schema=schema,
            bindings=bindings,
            source_root=source_root,
        ).build()

    @staticmethod
    def path_from_image_plane_source(source: ImagePlaneSource) -> Path:
        """Return the local/URI path token used for preview matching."""

        parsed = urlparse(source.uri)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        return Path(source.uri)

    @staticmethod
    def candidate_from_path(
        path: str | Path,
        *,
        metadata_rules: tuple[MetadataExtractionRule, ...],
        source_root: Path | None,
    ) -> SourceSchemaCandidate:
        source_path = Path(path)
        relative_path = SourceInventory.relative_path(source_path, source_root)
        return SourceSchemaCandidate(
            path=source_path,
            relative_path=relative_path,
            metadata=metadata_from_rules(str(source_path), metadata_rules),
        )

    @staticmethod
    def relative_path(path: Path, source_root: Path | None) -> str:
        if source_root is None:
            return str(path)
        try:
            return str(path.relative_to(source_root))
        except ValueError:
            return str(path)


@dataclass(frozen=True, slots=True)
class SourceBindingPreviewRow:
    """Preview summary for one binding applied to a concrete source inventory."""

    alias: str
    declaration_scope: str
    group_key: str | None
    matched_source_count: int
    sample_paths: tuple[str, ...]

    @classmethod
    def from_binding(
        cls,
        binding: SourceBindingView,
        candidates: tuple[SourceSchemaCandidate, ...],
        *,
        sample_limit: int,
    ) -> "SourceBindingPreviewRow":
        return cls(
            alias=binding.alias,
            declaration_scope=binding.declaration_scope,
            group_key=binding.group_key,
            matched_source_count=len(candidates),
            sample_paths=tuple(
                candidate.relative_path for candidate in candidates[:sample_limit]
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceImageSetPreviewRow:
    """Preview summary for one matched source image set."""

    index: int
    paths_by_alias: tuple[tuple[str, str], ...]
    metadata: tuple[tuple[str, str], ...]

    @classmethod
    def from_record(cls, record: ImageSetRecord) -> "SourceImageSetPreviewRow":
        return cls(
            index=record.index,
            paths_by_alias=tuple(
                (alias, candidate.relative_path)
                for alias, candidate in sorted(record.candidates_by_alias.items())
            ),
            metadata=tuple(sorted(record.metadata.items())),
        )


@dataclass(frozen=True, slots=True)
class SourceBindingsPreview:
    """Concrete preview of source bindings against an inventory."""

    binding_rows: tuple[SourceBindingPreviewRow, ...]
    image_set_rows: tuple[SourceImageSetPreviewRow, ...]

    @classmethod
    def from_schema_and_bindings(
        cls,
        *,
        schema: PipelineImageSchema,
        bindings: StepSourceBindingsConfig,
        inventory: SourceInventory,
        sample_limit: int = 3,
    ) -> SourceBindingsPreview:
        view = SourceBindingsViewModel.from_schema_and_bindings(
            schema=schema,
            bindings=bindings,
        )
        matches_by_binding = tuple(
            (
                binding,
                cls.matching_candidates(binding.selector, inventory.candidates),
            )
            for binding in view.all_bindings
        )
        return cls(
            binding_rows=tuple(
                SourceBindingPreviewRow.from_binding(
                    binding,
                    candidates,
                    sample_limit=sample_limit,
                )
                for binding, candidates in matches_by_binding
            ),
            image_set_rows=SourceBindingsPreview.image_set_preview_rows(
                schema,
                matches_by_binding,
            ),
        )

    @staticmethod
    def image_set_preview_rows(
        schema: PipelineImageSchema,
        matches_by_binding: tuple[
            tuple[SourceBindingView, tuple[SourceSchemaCandidate, ...]], ...
        ],
    ) -> tuple[SourceImageSetPreviewRow, ...]:
        pipeline_aliases = set(schema.assignments_by_alias)
        candidates_by_alias = {
            binding.alias: candidates
            for binding, candidates in matches_by_binding
            if binding.alias in pipeline_aliases
        }
        if not candidates_by_alias:
            return ()
        image_sets = ImageSetAssembler.for_schema(schema).image_sets(
            schema,
            candidates_by_alias,
        )
        return tuple(
            SourceImageSetPreviewRow.from_record(record) for record in image_sets
        )

    @classmethod
    def matching_candidates(
        cls,
        selector: SourceSelectorView,
        candidates: tuple[SourceSchemaCandidate, ...],
    ) -> tuple[SourceSchemaCandidate, ...]:
        return tuple(
            candidate
            for candidate in candidates
            if cls.candidate_matches_selector(candidate, selector)
        )

    @classmethod
    def candidate_matches_selector(
        cls,
        candidate: SourceSchemaCandidate,
        selector: SourceSelectorView,
    ) -> bool:
        return (
            cls.candidate_matches_components(candidate.metadata, selector.components)
            and cls.candidate_matches_metadata(candidate.metadata, selector.metadata)
            and source_filters_match(
                candidate.relative_path,
                tuple(
                    SourceBindingsPreview.filter_clause_from_view(filter_view)
                    for filter_view in selector.filters
                ),
            )
        )

    @staticmethod
    def candidate_matches_components(
        metadata: Mapping[str, str],
        components: tuple[tuple[str, str], ...],
    ) -> bool:
        return all(
            any(
                source_metadata_values_equal(value, expected_value)
                for value in source_component_metadata_values(
                    metadata,
                    AllComponents(component_name),
                )
            )
            for component_name, expected_value in components
        )

    @staticmethod
    def candidate_matches_metadata(
        metadata: Mapping[str, str],
        selectors: tuple[tuple[str, str], ...],
    ) -> bool:
        return all(
            (value := semantic_source_metadata_value(metadata, field)) is not None
            and source_metadata_values_equal(value, expected_value)
            for field, expected_value in selectors
        )

    @staticmethod
    def filter_clause_from_view(filter_view: SourceFilterView) -> SourceFilterClause:
        return SourceFilterClause(
            subject=SourceFilterSubject(filter_view.subject),
            match_type=SourceFilterMatchType(filter_view.match_type),
            value=filter_view.value,
        )


def is_source_bindings_view_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and name.endswith("View")
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_source_bindings_view_export(name, value)
)
