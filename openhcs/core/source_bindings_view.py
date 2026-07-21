"""Presentation-ready typed views over source-binding declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from openhcs.constants.constants import AllComponents
from openhcs.core.source_binding_workspace import (
    SourceSetAssembler,
    SourceBindingWorkspaceProjector,
    SourceCandidate,
    _SourceSet,
)
from openhcs.core.source_metadata import SourceMetadataScalar, SourceMetadataValue
from openhcs.core.source_bindings import (
    EMPTY_SOURCE_BINDINGS,
    ImportedMetadataTable,
    MetadataExtractionRule,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchPlan,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.vfs_protocol import FileManagerLike


@dataclass(frozen=True, slots=True)
class SourceFilterView:
    """UI-neutral view of one source filter clause."""

    subject: str
    match_type: str
    value: str | None
    any_group: int | None = None

    @classmethod
    def from_clause(cls, clause: SourceFilterClause) -> "SourceFilterView":
        return cls(
            subject=clause.subject.value,
            match_type=clause.match_type.value,
            value=clause.value,
            any_group=clause.any_group,
        )


@dataclass(frozen=True, slots=True)
class SourceSelectorView:
    """UI-neutral view of component, metadata, and filter source selectors."""

    components: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, SourceMetadataScalar], ...] = ()
    filters: tuple[SourceFilterView, ...] = ()
    inherit_current_scope: bool = True

    @classmethod
    def from_selector(cls, selector: SourceSelector) -> "SourceSelectorView":
        return cls(
            components=tuple(
                (cls.component_name(item.component), item.value)
                for item in selector.components
            ),
            metadata=tuple((item.field, item.value) for item in selector.metadata),
            filters=tuple(SourceFilterView.from_clause(item) for item in selector.filters),
            inherit_current_scope=selector.inherit_current_scope,
        )

    @staticmethod
    def component_name(component: object) -> str:
        return component.value if isinstance(component, AllComponents) else str(component)

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
    """UI-neutral row for one named source binding."""

    alias: str
    artifact_kind: str
    origin: str
    required: bool
    source_set_role: str
    projection_role: str
    selector: SourceSelectorView
    declaration_scope: str

    @classmethod
    def from_named_binding(
        cls,
        binding: NamedSourceBinding,
        *,
        declaration_scope: str,
    ) -> "SourceBindingView":
        return cls(
            alias=binding.alias,
            artifact_kind=binding.artifact_kind.value,
            origin=binding.origin.value,
            required=binding.required,
            source_set_role=binding.source_set_role.value,
            projection_role=binding.projection_role.value,
            selector=SourceSelectorView.from_selector(binding.selector),
            declaration_scope=declaration_scope,
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
            extracted_fields=rule.capture_fields,
            filters=tuple(SourceFilterView.from_clause(item) for item in rule.filters),
            declaration_scope=declaration_scope,
        )


@dataclass(frozen=True, slots=True)
class MatchDimensionView:
    """UI-neutral row for one cross-alias source-set matching dimension."""

    fields: tuple[tuple[str, str], ...]

    @classmethod
    def from_dimension(
        cls,
        dimension: SourceBindingMatchDimension,
    ) -> "MatchDimensionView":
        return cls(
            fields=tuple(
                (field.alias, field.metadata_field) for field in dimension.fields
            )
        )


@dataclass(frozen=True, slots=True)
class SourceBindingMatchPlanView:
    """UI-neutral view of one declared source-set matching plan."""

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
                MatchDimensionView.from_dimension(item) for item in plan.dimensions
            ),
            declaration_scope=declaration_scope,
        )


@dataclass(frozen=True, slots=True)
class GroupingView:
    """UI-neutral view of pipeline-level grouping semantics."""

    metadata_fields: tuple[str, ...]

    @classmethod
    def from_config(cls, config: SourceBindingsConfig) -> "GroupingView":
        return cls(metadata_fields=config.grouping_metadata_fields)


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
    """UI-neutral summary of pipeline-level source declarations."""

    filters: tuple[SourceFilterView, ...]
    image_plane_source_count: int
    imported_metadata_tables: tuple[ImportedMetadataTableView, ...]

    @classmethod
    def from_config(cls, config: SourceBindingsConfig) -> "PipelineSourceUniverseView":
        return cls(
            filters=tuple(
                SourceFilterView.from_clause(item)
                for item in config.source_filter_declarations
            ),
            image_plane_source_count=len(config.image_plane_sources),
            imported_metadata_tables=tuple(
                ImportedMetadataTableView.from_table(item)
                for item in config.imported_metadata_tables
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceBindingsViewModel:
    """Typed, GUI-neutral source-binding state for editor and preview surfaces."""

    pipeline_sources: PipelineSourceUniverseView
    pipeline_bindings: tuple[SourceBindingView, ...]
    step_bindings: tuple[SourceBindingView, ...]
    metadata_rules: tuple[MetadataRuleView, ...]
    match_plans: tuple[SourceBindingMatchPlanView, ...]
    grouping: GroupingView | None

    @classmethod
    def from_config_and_step_bindings(
        cls,
        *,
        source_bindings: SourceBindingsConfig,
        step_bindings: StepSourceBindingsConfig,
    ) -> "SourceBindingsViewModel":
        return cls(
            pipeline_sources=PipelineSourceUniverseView.from_config(source_bindings),
            pipeline_bindings=cls.binding_views(
                source_bindings.binding_declarations,
                declaration_scope="pipeline",
            ),
            step_bindings=cls.binding_views(
                step_bindings.binding_declarations,
                declaration_scope="step",
            ),
            metadata_rules=(
                *(
                    MetadataRuleView.from_rule(item, declaration_scope="pipeline")
                    for item in source_bindings.metadata_rule_declarations
                ),
                *(
                    MetadataRuleView.from_rule(item, declaration_scope="step")
                    for item in step_bindings.metadata_rule_declarations
                ),
            ),
            match_plans=tuple(
                SourceBindingMatchPlanView.from_plan(plan, declaration_scope=scope)
                for scope, plan in (
                    ("pipeline", source_bindings.match_plan),
                    ("step", step_bindings.match_plan),
                )
                if plan is not None
            ),
            grouping=(
                None
                if not source_bindings.grouping_metadata_fields
                else GroupingView.from_config(source_bindings)
            ),
        )

    @staticmethod
    def binding_views(
        bindings: tuple[NamedSourceBinding, ...],
        *,
        declaration_scope: str,
    ) -> tuple[SourceBindingView, ...]:
        return tuple(
            SourceBindingView.from_named_binding(
                binding,
                declaration_scope=declaration_scope,
            )
            for binding in bindings
        )

    @property
    def all_bindings(self) -> tuple[SourceBindingView, ...]:
        return self.pipeline_bindings + self.step_bindings

    @property
    def artifact_kinds(self) -> tuple[str, ...]:
        return tuple(sorted({item.artifact_kind for item in self.all_bindings}))


def _active_source_bindings(
    source_bindings: SourceBindingsConfig,
    step_bindings: StepSourceBindingsConfig,
) -> SourceBindingsConfig:
    return step_bindings if step_bindings.enabled else source_bindings


@dataclass(frozen=True, slots=True)
class SourceInventory:
    """Resolved source candidates available for source-binding previews."""

    candidates: tuple[SourceCandidate, ...]
    source_root: Path = Path(".")

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if any(not isinstance(item, SourceCandidate) for item in candidates):
            raise TypeError("SourceInventory.candidates must contain SourceCandidate values.")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "source_root", Path(self.source_root))

    @classmethod
    def from_paths(
        cls,
        paths: tuple[str | Path, ...],
        *,
        source_root: str | Path,
        source_backend: str,
        source_bindings: SourceBindingsConfig,
        step_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS,
    ) -> "SourceInventory":
        config = _active_source_bindings(source_bindings, step_bindings)
        return cls(
            candidates=SourceBindingWorkspaceProjector(config).source_candidates(
                Path(source_root),
                paths,
                source_backend=source_backend,
            ),
            source_root=Path(source_root),
        )

    @classmethod
    def from_filemanager(
        cls,
        *,
        filemanager: FileManagerLike,
        source_root: str | Path,
        backend: str,
        source_bindings: SourceBindingsConfig,
        step_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS,
    ) -> "SourceInventory":
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
        return cls.from_paths(
            paths,
            source_root=source_root,
            source_backend=backend,
            source_bindings=source_bindings,
            step_bindings=step_bindings,
        )


@dataclass(frozen=True, slots=True)
class SourceBindingPreviewRow:
    """Preview summary for one binding applied to a concrete source inventory."""

    alias: str
    declaration_scope: str
    matched_source_count: int
    sample_paths: tuple[str, ...]

    @classmethod
    def from_binding(
        cls,
        binding: NamedSourceBinding,
        candidates: tuple[SourceCandidate, ...],
        *,
        declaration_scope: str,
        sample_limit: int,
    ) -> "SourceBindingPreviewRow":
        return cls(
            alias=binding.alias,
            declaration_scope=declaration_scope,
            matched_source_count=len(candidates),
            sample_paths=tuple(
                candidate.relative_path for candidate in candidates[:sample_limit]
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceSetPreviewRow:
    """Preview summary for one matched source set."""

    index: int
    paths_by_alias: tuple[tuple[str, str], ...]
    metadata: tuple[tuple[str, SourceMetadataValue], ...]

    @classmethod
    def from_record(cls, record: _SourceSet) -> "SourceSetPreviewRow":
        return cls(
            index=record.index,
            paths_by_alias=tuple(
                (alias, candidate.relative_path)
                for alias, candidate in sorted(record.candidates_by_alias.items())
            ),
            metadata=tuple(sorted(record.metadata.items())),
        )


class SourceBindingDiagnosticSeverity(str, Enum):
    """Closed severity values for source-binding preview diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class SourceBindingDiagnostic:
    """Pure diagnostic for unresolved source-binding state."""

    severity: SourceBindingDiagnosticSeverity
    code: str
    alias: str | None
    message: str
    candidate_count: int | None = None


BindingMatches = tuple[
    tuple[NamedSourceBinding, tuple[SourceCandidate, ...]],
    ...,
]


@dataclass(frozen=True, slots=True)
class SourceBindingsPreview:
    """Concrete preview of source bindings against an inventory."""

    binding_rows: tuple[SourceBindingPreviewRow, ...]
    source_set_rows: tuple[SourceSetPreviewRow, ...]
    diagnostics: tuple[SourceBindingDiagnostic, ...] = ()

    @classmethod
    def from_config_and_step_bindings(
        cls,
        *,
        source_bindings: SourceBindingsConfig,
        step_bindings: StepSourceBindingsConfig,
        inventory: SourceInventory,
        sample_limit: int = 3,
    ) -> "SourceBindingsPreview":
        config = _active_source_bindings(source_bindings, step_bindings)
        projector = SourceBindingWorkspaceProjector(config)
        matches: BindingMatches = tuple(
            (
                binding,
                tuple(
                    candidate
                    for candidate in inventory.candidates
                    if projector.candidate_matches_binding(
                        candidate,
                        binding,
                        inventory.source_root,
                    )
                ),
            )
            for binding in config.binding_declarations
        )
        binding_rows = tuple(
            SourceBindingPreviewRow.from_binding(
                binding,
                candidates,
                declaration_scope="step" if step_bindings.enabled else "pipeline",
                sample_limit=sample_limit,
            )
            for binding, candidates in matches
        )
        diagnostics = tuple(
            SourceBindingDiagnostic(
                severity=SourceBindingDiagnosticSeverity.ERROR,
                code="source_binding.no_match",
                alias=binding.alias,
                message=f"Required source alias {binding.alias!r} matched no candidates.",
                candidate_count=0,
            )
            for binding, candidates in matches
            if binding.required and not candidates
        )
        source_set_rows, assembly_diagnostic = cls._assemble_source_set_rows(
            config,
            matches,
        )
        return cls(
            binding_rows=binding_rows,
            source_set_rows=source_set_rows,
            diagnostics=diagnostics + assembly_diagnostic,
        )

    @staticmethod
    def _assemble_source_set_rows(
        config: SourceBindingsConfig,
        matches: BindingMatches,
    ) -> tuple[
        tuple[SourceSetPreviewRow, ...],
        tuple[SourceBindingDiagnostic, ...],
    ]:
        matched_members = {
            binding.alias: candidates
            for binding, candidates in matches
            if binding in config.matched_source_bindings
        }
        if not matched_members or any(
            not candidates for candidates in matched_members.values()
        ):
            return (), ()
        try:
            source_sets = SourceSetAssembler.for_config(config).source_sets(
                config.match_plan,
                matched_members,
                (),
            )
        except ValueError as exc:
            return (), (
                SourceBindingDiagnostic(
                    severity=SourceBindingDiagnosticSeverity.ERROR,
                    code="source_binding.source_set_assembly",
                    alias=None,
                    message=str(exc),
                ),
            )
        return tuple(SourceSetPreviewRow.from_record(item) for item in source_sets), ()


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
