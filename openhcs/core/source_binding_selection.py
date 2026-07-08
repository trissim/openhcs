"""Shared source-binding candidate selection for planning and runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence, TYPE_CHECKING

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import Backend
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    MostDerivedContextStrategyMixin,
)
from openhcs.core.path_pattern_matching import PathPatternTemplateMatcher
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    MetadataExtractionRule,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceBindingRuntimeContext,
)
from openhcs.core.source_metadata import (
    SourceMetadataMapping,
    SourceMetadataValue,
)
from openhcs.core.source_matching import (
    SourceImageSetIdentity,
    SourceImageSetIdentityCompatibility,
    SourceImageSetIdentityPolicy,
    merge_source_metadata,
    metadata_from_rules,
    source_component_metadata_values,
    source_filters_match,
    source_metadata_value,
    source_metadata_values_equal,
)
from openhcs.core.source_workspace_projection import VirtualWorkspaceSourceProjection

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
    from openhcs.microscopes.microscope_interfaces import FilenameParser


SourceCandidatePath = str


@lru_cache(maxsize=65536)
def _cached_source_candidate_pattern_keys(pattern_path: str) -> tuple[str, ...]:
    """Return candidate source path spellings used for selector matching."""

    path = Path(pattern_path)
    return tuple(dict.fromkeys((pattern_path, path.as_posix(), path.name)))


@lru_cache(maxsize=65536)
def _cached_path_is_absolute(path: str) -> bool:
    """Return whether a candidate virtual path is absolute."""

    return Path(path).is_absolute()


@lru_cache(maxsize=65536)
def _cached_path_name(path: str) -> str:
    """Return the filename component for candidate matching."""

    return Path(path).name


@dataclass(frozen=True, slots=True)
class SourceMetadataRecord(Mapping[str, SourceMetadataValue]):
    """Normalized source metadata carried across source-binding selection."""

    fields: tuple[tuple[str, SourceMetadataValue], ...]

    @classmethod
    def from_mapping(cls, metadata: SourceMetadataMapping) -> "SourceMetadataRecord":
        return cls(tuple((str(key), value) for key, value in metadata.items()))

    def __getitem__(self, key: str) -> SourceMetadataValue:
        for field_key, value in self.fields:
            if field_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self.fields)

    def __len__(self) -> int:
        return len(self.fields)


@dataclass(frozen=True, slots=True)
class SourceMetadataCandidates:
    """Candidate metadata records resolved for one source-binding pattern."""

    values: tuple[SourceMetadataRecord, ...]

    def __iter__(self):
        return iter(self.values)

    def __bool__(self) -> bool:
        return bool(self.values)

    def first_required(self, pattern: SourceCandidatePath) -> SourceMetadataRecord:
        if self.values:
            return self.values[0]
        raise ValueError(
            "Source binding metadata resolution found no parser-readable metadata "
            f"for candidate {pattern!s}."
        )


@dataclass(frozen=True, slots=True)
class SourceCandidatePathResolution:
    """Virtual and mapped-source path views for one source-binding candidate."""

    pattern_keys: tuple[str, ...]
    virtual_paths: tuple[str, ...]
    mapped_source_paths: tuple[str, ...]

    def metadata_paths(self) -> tuple[str, ...]:
        if self.virtual_paths:
            return tuple(
                dict.fromkeys(
                    (*self.virtual_paths, *self.mapped_source_paths, *self.pattern_keys)
                )
            )
        return tuple(dict.fromkeys((*self.pattern_keys, *self.mapped_source_paths)))

    def filter_paths(self) -> tuple[str, ...]:
        if self.mapped_source_paths:
            return tuple(dict.fromkeys(self.mapped_source_paths))
        if self.virtual_paths:
            return tuple(dict.fromkeys((*self.virtual_paths, *self.pattern_keys)))
        return self.pattern_keys


@dataclass(frozen=True, slots=True)
class SourcePatternResolutionContext:
    """Source paths and metadata available while filtering execution candidates."""

    parser: "FilenameParser"
    source_paths_by_virtual_path: Mapping[str, str]
    source_metadata_by_path: Mapping[str, SourceMetadataRecord] = field(
        default_factory=dict
    )
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()

    @classmethod
    def from_sources(
        cls,
        *,
        parser: "FilenameParser",
        source_paths_by_virtual_path: Mapping[str, str],
        source_metadata_by_path: Mapping[str, SourceMetadataMapping] | None = None,
        metadata_rules: tuple[MetadataExtractionRule, ...] = (),
    ) -> "SourcePatternResolutionContext":
        if source_metadata_by_path is None:
            metadata_by_path: Mapping[str, SourceMetadataRecord] = {}
        else:
            metadata_by_path = {
                str(path): (
                    metadata
                    if isinstance(metadata, SourceMetadataRecord)
                    else SourceMetadataRecord.from_mapping(metadata)
                )
                for path, metadata in source_metadata_by_path.items()
            }
        return cls(
            parser=parser,
            source_paths_by_virtual_path=source_paths_by_virtual_path,
            source_metadata_by_path=metadata_by_path,
            metadata_rules=metadata_rules,
        )

    @classmethod
    def from_projection(
        cls,
        *,
        parser: "FilenameParser",
        projection: VirtualWorkspaceSourceProjection,
        metadata_rules: tuple[MetadataExtractionRule, ...] = (),
    ) -> "SourcePatternResolutionContext":
        return cls.from_sources(
            parser=parser,
            source_paths_by_virtual_path=projection.source_paths_by_virtual_path,
            source_metadata_by_path=projection.source_metadata_by_path,
            metadata_rules=metadata_rules,
        )

    @property
    def has_virtual_source_workspace(self) -> bool:
        return bool(self.source_paths_by_virtual_path)

    def candidate_paths(self, pattern: SourceCandidatePath) -> tuple[str, ...]:
        return self._candidate_path_resolution(pattern).metadata_paths()

    def candidate_filter_paths(self, pattern: SourceCandidatePath) -> tuple[str, ...]:
        return self._candidate_path_resolution(pattern).filter_paths()

    def _candidate_path_resolution(
        self,
        pattern: SourceCandidatePath,
    ) -> SourceCandidatePathResolution:
        keys = _cached_source_candidate_pattern_keys(pattern)
        virtual_matches = tuple(
            virtual_path
            for key in keys
            for virtual_path in self._matching_virtual_paths(key)
        )
        mapped = tuple(
            self.source_paths_by_virtual_path[key]
            for key in (*keys, *virtual_matches)
            if key in self.source_paths_by_virtual_path
        )
        return SourceCandidatePathResolution(
            pattern_keys=keys,
            virtual_paths=virtual_matches,
            mapped_source_paths=mapped,
        )

    def _matching_virtual_paths(self, pattern_key: str) -> tuple[str, ...]:
        if pattern_key in self.source_paths_by_virtual_path:
            return ()
        matcher = PathPatternTemplateMatcher.from_pattern(pattern_key)
        if matcher is None:
            return ()
        return tuple(
            virtual_path
            for virtual_path in self.source_paths_by_virtual_path
            if not _cached_path_is_absolute(virtual_path)
            and matcher.matches(_cached_path_name(virtual_path))
        )

    def candidate_metadata(
        self,
        pattern: SourceCandidatePath,
    ) -> SourceMetadataCandidates:
        return self.metadata_for_paths(self.candidate_paths(pattern))

    def source_path_for(self, path: str) -> str:
        """Return the physical source path represented by a runtime path."""
        resolved_paths = self._candidate_path_resolution(path).mapped_source_paths
        if resolved_paths:
            return str(resolved_paths[0])
        return str(path)

    def metadata_for_paths(
        self,
        paths: tuple[str, ...],
    ) -> SourceMetadataCandidates:
        return SourceMetadataCandidates(
            tuple(
                metadata
                for path in paths
                for metadata in (self.metadata_for_path(path),)
                if metadata is not None
            )
        )

    def metadata_for_path(self, path: str) -> SourceMetadataRecord | None:
        metadata: dict[str, SourceMetadataValue] = {}
        declared_metadata = self.source_metadata_by_path.get(path)
        if declared_metadata is not None:
            merge_source_metadata(metadata, declared_metadata, path=path)
        parsed_metadata = self.parser.parse_filename(path)
        if parsed_metadata is not None:
            merge_source_metadata(metadata, parsed_metadata, path=path)
        rule_metadata = metadata_from_rules(path, self.metadata_rules)
        if rule_metadata:
            merge_missing_source_metadata(metadata, rule_metadata)
        if metadata:
            return SourceMetadataRecord.from_mapping(metadata)
        return None

    def merged_metadata_for_paths(
        self,
        paths: tuple[str, ...],
    ) -> SourceMetadataRecord | None:
        """Return one metadata record merged from all path identities."""
        metadata: dict[str, SourceMetadataValue] = {}
        for path in dict.fromkeys(paths):
            path_metadata = self.metadata_for_path(path)
            if path_metadata is not None:
                merge_source_metadata(metadata, path_metadata, path=path)
        if metadata:
            return SourceMetadataRecord.from_mapping(metadata)
        return None

    def source_metadata_by_paths(
        self,
        paths: Sequence[str],
    ) -> Mapping[str, SourceMetadataMapping]:
        """Return resolved metadata keyed by the paths visible in a source universe."""
        metadata_by_path: dict[str, SourceMetadataMapping] = {}
        for path in dict.fromkeys(str(path) for path in paths):
            metadata = self.metadata_for_path(path)
            if metadata is not None:
                metadata_by_path[path] = metadata
        return MappingProxyType(metadata_by_path)

    def has_metadata_field(
        self,
        patterns: Sequence[SourceCandidatePath],
        field: str,
    ) -> bool:
        return any(
            source_metadata_value(metadata, field) is not None
            for pattern in patterns
            for metadata in self.candidate_metadata(pattern)
        )


def merge_missing_source_metadata(
    target: dict[str, SourceMetadataValue],
    additions: Mapping[str, str],
) -> None:
    """Fill metadata fields not already provided by the source workspace."""
    for key, value in additions.items():
        if key not in target:
            target[key] = str(value)


class SourceBindingCandidateMatcher:
    """Match source-binding selectors against candidate source paths."""

    @staticmethod
    def selector_bindings(
        bindings: Sequence[NamedSourceBinding],
    ) -> tuple[NamedSourceBinding, ...]:
        return tuple(
            binding
            for binding in bindings
            if binding.required and binding.requires_selector_resolution
        )

    @classmethod
    def execution_anchor_bindings(
        cls,
        bindings: Sequence[NamedSourceBinding],
    ) -> tuple[NamedSourceBinding, ...]:
        """Return selector bindings that may choose execution anchor files."""
        return tuple(
            binding
            for binding in cls.selector_bindings(bindings)
            if binding.participates_in_execution_anchoring
        )

    @staticmethod
    def matches(
        candidate: SourceCandidatePath,
        *,
        binding: NamedSourceBinding,
        source_context: SourcePatternResolutionContext,
    ) -> bool:
        selector = binding.selector
        if not any(
            source_filters_match(path, selector.filters)
            for path in source_context.candidate_filter_paths(candidate)
        ):
            return False

        if not selector.components and not selector.metadata:
            return True

        metadata_candidates = source_context.candidate_metadata(candidate)
        for component_selector in selector.components:
            if not any(
                source_metadata_values_equal(value, str(component_selector.value))
                for metadata in metadata_candidates
                for value in source_component_metadata_values(
                    metadata,
                    component_selector.component,
                )
            ):
                return False

        for metadata_selector in selector.metadata:
            if not any(
                value is not None
                and source_metadata_values_equal(value, metadata_selector.value)
                for metadata in metadata_candidates
                for value in (
                    source_metadata_value(
                        metadata,
                        metadata_selector.field,
                    ),
                )
            ):
                return False

        return True

    @classmethod
    def compatible_candidates(
        cls,
        candidates: Sequence[SourceCandidatePath],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> tuple[SourceCandidatePath, ...]:
        selector_bindings = cls.selector_bindings(bindings)
        if not selector_bindings:
            return tuple(candidates)
        return tuple(
            candidate
            for candidate in candidates
            if any(
                cls.matches(
                    candidate,
                    binding=binding,
                    source_context=source_context,
                )
                for binding in selector_bindings
            )
        )


class SourceAnchorSelectionStatus(str, Enum):
    """Outcome of source-bound execution-anchor resolution."""

    SELECTED = "selected"
    DEFERRED_TO_RUNTIME = "deferred_to_runtime"


@dataclass(frozen=True, slots=True)
class SourceAnchorPatternSelection:
    """Resolved source-compatible anchors plus the authority that owns them."""

    patterns: tuple[SourceCandidatePath, ...]
    status: SourceAnchorSelectionStatus
    reason: str

    @classmethod
    def selected(
        cls,
        patterns: Sequence[SourceCandidatePath],
        *,
        reason: str = "source selectors resolved at anchor boundary",
    ) -> "SourceAnchorPatternSelection":
        return cls(
            patterns=tuple(patterns),
            status=SourceAnchorSelectionStatus.SELECTED,
            reason=reason,
        )

    @classmethod
    def deferred_to_runtime(
        cls,
        patterns: Sequence[SourceCandidatePath],
        *,
        reason: str,
    ) -> "SourceAnchorPatternSelection":
        return cls(
            patterns=tuple(patterns),
            status=SourceAnchorSelectionStatus.DEFERRED_TO_RUNTIME,
            reason=reason,
        )

    @property
    def owns_runtime_resolution(self) -> bool:
        return self.status is SourceAnchorSelectionStatus.DEFERRED_TO_RUNTIME


@dataclass(frozen=True, slots=True)
class SourceWorkspaceAnchorNarrowing:
    """Contract for partial anchors materialized from a virtual source workspace."""

    source_context: SourcePatternResolutionContext
    compatible_count: int
    alias_count: int

    def allows_runtime_completion(self) -> bool:
        return (
            self.source_context.has_virtual_source_workspace
            and 0 < self.compatible_count < self.alias_count
        )


class SourceBindingMatchResolutionStatus(str, Enum):
    """Outcome of matching one anchor pattern to source-binding aliases."""

    MATCHED = "matched"
    DEFERRED_TO_RUNTIME = "deferred_to_runtime"


@dataclass(frozen=True, slots=True)
class SourceBindingMatchResolution:
    """Alias match result for one source-bound anchor pattern."""

    status: SourceBindingMatchResolutionStatus
    binding: NamedSourceBinding | None
    reason: str

    @classmethod
    def matched(
        cls,
        binding: NamedSourceBinding,
    ) -> "SourceBindingMatchResolution":
        return cls(
            status=SourceBindingMatchResolutionStatus.MATCHED,
            binding=binding,
            reason="exactly one selector binding matched the anchor",
        )

    @classmethod
    def deferred_to_runtime(
        cls,
        *,
        reason: str,
    ) -> "SourceBindingMatchResolution":
        return cls(
            status=SourceBindingMatchResolutionStatus.DEFERRED_TO_RUNTIME,
            binding=None,
            reason=reason,
        )

    def require_binding(self) -> NamedSourceBinding:
        if self.binding is None:
            raise RuntimeError(
                "Source binding resolution was deferred to runtime: "
                f"{self.reason}."
            )
        return self.binding

    @property
    def owns_runtime_resolution(self) -> bool:
        return self.status is SourceBindingMatchResolutionStatus.DEFERRED_TO_RUNTIME


class SourceBoundAnchorPatternPolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal policy for choosing execution anchors from source-bound inputs."""

    __registry_key__ = "policy_key"
    __skip_if_no_key__ = True
    policy_key: ClassVar[str | None] = None

    def __init__(self, match_plan: SourceBindingMatchPlan | None = None) -> None:
        self._match_plan = match_plan

    @classmethod
    def for_plan(
        cls,
        plan: CompiledSourceBindingPlan,
    ) -> "SourceBoundAnchorPatternPolicy":
        if plan.match_plan is None:
            return DefaultSourceBoundAnchorPatternPolicy()
        policy_type = cls.__registry__.get(
            plan.match_plan.method.value,
            DefaultSourceBoundAnchorPatternPolicy,
        )
        return policy_type(plan.match_plan)

    @abstractmethod
    def select(
        self,
        pattern_list: Sequence[SourceCandidatePath],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[SourceCandidatePath]:
        """Return source-compatible anchor patterns for one execution group."""

    def _source_compatible_anchor_selection(
        self,
        pattern_list: Sequence[SourceCandidatePath],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> SourceAnchorPatternSelection:
        anchor_bindings = self._anchor_bindings(bindings)
        if not anchor_bindings:
            return SourceAnchorPatternSelection.selected(
                pattern_list,
                reason="no selector bindings participate in execution anchoring",
            )

        compatible = [
            pattern
            for pattern in pattern_list
            if any(
                self._pattern_matches_source_binding(
                    pattern,
                    binding=binding,
                    source_context=source_context,
                )
                for binding in anchor_bindings
            )
        ]
        if compatible:
            return SourceAnchorPatternSelection.selected(compatible)
        if self._metadata_selector_fields_are_unavailable(
            pattern_list,
            bindings=anchor_bindings,
            source_context=source_context,
        ):
            return SourceAnchorPatternSelection.deferred_to_runtime(
                pattern_list,
                reason=(
                    "selector metadata fields are unavailable at the execution "
                    "anchor boundary"
                ),
            )
        if self._file_selector_paths_are_unavailable(
            bindings=anchor_bindings,
            source_context=source_context,
        ):
            return SourceAnchorPatternSelection.deferred_to_runtime(
                pattern_list,
                reason=(
                    "selector file paths are unavailable at the execution "
                    "anchor boundary"
                ),
            )
        return SourceAnchorPatternSelection.selected(
            (),
            reason="source selectors resolved no compatible anchor patterns",
        )

    @staticmethod
    def _metadata_selector_fields_are_unavailable(
        pattern_list: Sequence[SourceCandidatePath],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> bool:
        metadata_fields = tuple(
            selector.field
            for binding in bindings
            for selector in binding.selector.metadata
        )
        return bool(metadata_fields) and not any(
            source_context.has_metadata_field(pattern_list, field)
            for field in metadata_fields
        )

    @staticmethod
    def _file_selector_paths_are_unavailable(
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> bool:
        return (
            not source_context.has_virtual_source_workspace
            and any(binding.selector.filters for binding in bindings)
        )

    @staticmethod
    def _selector_bindings(
        bindings: Sequence[NamedSourceBinding],
    ) -> tuple[NamedSourceBinding, ...]:
        return SourceBindingCandidateMatcher.selector_bindings(bindings)

    @staticmethod
    def _anchor_bindings(
        bindings: Sequence[NamedSourceBinding],
    ) -> tuple[NamedSourceBinding, ...]:
        return SourceBindingCandidateMatcher.execution_anchor_bindings(bindings)

    @staticmethod
    def _pattern_matches_source_binding(
        pattern: SourceCandidatePath,
        *,
        binding: NamedSourceBinding,
        source_context: SourcePatternResolutionContext,
    ) -> bool:
        return SourceBindingCandidateMatcher.matches(
            pattern,
            binding=binding,
            source_context=source_context,
        )


class DefaultSourceBoundAnchorPatternPolicy(SourceBoundAnchorPatternPolicy):
    """Keep every selector-compatible source anchor."""

    def select(
        self,
        pattern_list: Sequence[SourceCandidatePath],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[SourceCandidatePath]:
        selection = self._source_compatible_anchor_selection(
            pattern_list,
            bindings=bindings,
            source_context=source_context,
        )
        return list(selection.patterns)


class MatchedImageSetAnchorPatternPolicy(SourceBoundAnchorPatternPolicy):
    """Collapse multi-alias source anchors to one representative per image set."""

    policy_key = None

    def select(
        self,
        pattern_list: Sequence[SourceCandidatePath],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[SourceCandidatePath]:
        selection = self._source_compatible_anchor_selection(
            pattern_list,
            bindings=bindings,
            source_context=source_context,
        )
        compatible = list(selection.patterns)
        anchor_bindings = self._anchor_bindings(bindings)
        if len(anchor_bindings) < 2:
            return compatible

        return self._deduplicate_matched_image_sets(
            compatible,
            selector_bindings=anchor_bindings,
            source_context=source_context,
        )

    @abstractmethod
    def _deduplicate_matched_image_sets(
        self,
        compatible: Sequence[SourceCandidatePath],
        *,
        selector_bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[SourceCandidatePath]:
        """Return one execution anchor per matched image set."""


class OrderMatchedImageSetAnchorPatternPolicy(MatchedImageSetAnchorPatternPolicy):
    """Source aliases are paired by order within one logical image set."""

    policy_key = SourceBindingMatchMethod.ORDER.value

    def _deduplicate_matched_image_sets(
        self,
        compatible: Sequence[SourceCandidatePath],
        *,
        selector_bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[SourceCandidatePath]:
        alias_count = len(selector_bindings)
        if len(compatible) % alias_count:
            if SourceWorkspaceAnchorNarrowing(
                source_context=source_context,
                compatible_count=len(compatible),
                alias_count=alias_count,
            ).allows_runtime_completion():
                return list(compatible)
            raise ValueError(
                "ORDER source binding produced an incomplete image set: "
                f"{len(compatible)} source anchors for {alias_count} aliases."
            )
        return [
            pattern
            for index, pattern in enumerate(compatible)
            if index % alias_count == 0
        ]


class MetadataMatchedImageSetAnchorPatternPolicy(MatchedImageSetAnchorPatternPolicy):
    """Source aliases are paired by declared metadata dimensions."""

    policy_key = SourceBindingMatchMethod.METADATA.value

    def _deduplicate_matched_image_sets(
        self,
        compatible: Sequence[SourceCandidatePath],
        *,
        selector_bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[SourceCandidatePath]:
        if self._match_plan is None or not self._match_plan.dimensions:
            raise ValueError(
                "METADATA source binding requires explicit match dimensions "
                "to collapse source-bound execution anchors."
            )

        deduplicated: list[SourceCandidatePath] = []
        seen: set[tuple[str, ...]] = set()
        for pattern in compatible:
            metadata = source_context.candidate_metadata(pattern).first_required(
                pattern
            )
            binding = self._matching_binding(
                pattern,
                selector_bindings=selector_bindings,
                source_context=source_context,
            )
            if binding.owns_runtime_resolution:
                return list(compatible)
            key = self._metadata_image_set_key(
                metadata,
                binding=binding.require_binding(),
                allow_missing=source_context.has_virtual_source_workspace,
            )
            if key is None:
                return list(compatible)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(pattern)
        return deduplicated

    def _matching_binding(
        self,
        pattern: SourceCandidatePath,
        *,
        selector_bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> SourceBindingMatchResolution:
        matches = tuple(
            binding
            for binding in selector_bindings
            if self._pattern_matches_source_binding(
                pattern,
                binding=binding,
                source_context=source_context,
            )
        )
        if len(matches) > 1 and source_context.has_virtual_source_workspace:
            return SourceBindingMatchResolution.matched(matches[0])
        if len(matches) == 0 and source_context.has_virtual_source_workspace:
            return SourceBindingMatchResolution.deferred_to_runtime(
                reason=(
                    "virtual source workspace did not expose enough selector "
                    "metadata to bind this anchor to one alias"
                )
            )
        if len(matches) != 1:
            raise ValueError(
                "METADATA source binding expected exactly one alias match for "
                f"{pattern!s}, got {len(matches)}."
            )
        return SourceBindingMatchResolution.matched(matches[0])

    def _metadata_image_set_key(
        self,
        metadata: SourceMetadataMapping,
        *,
        binding: NamedSourceBinding,
        allow_missing: bool = False,
    ) -> tuple[str, ...] | None:
        if self._match_plan is None:
            raise ValueError("METADATA source binding policy has no match plan.")
        values: list[str] = []
        for dimension in self._match_plan.dimensions:
            field = dimension.field_for_alias(binding.alias)
            if field is None:
                raise ValueError(
                    "METADATA source binding dimension is missing alias "
                    f"{binding.alias!r}."
            )
            value = source_metadata_value(metadata, field)
            if value is None:
                if allow_missing:
                    return None
                raise ValueError(
                    "METADATA source binding could not read match field "
                    f"{field!r} for alias {binding.alias!r}."
                )
            values.append(value)
        return tuple(values)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceBindingMatchedImageSet(SourcePatternResolutionContext):
    """Resolve all declared source aliases for one matched image-set anchor."""

    bindings: tuple[NamedSourceBinding, ...]
    match_plan: SourceBindingMatchPlan | None
    identity_policy: SourceImageSetIdentityPolicy = field(
        default_factory=SourceImageSetIdentityPolicy
    )

    @classmethod
    def from_plan(
        cls,
        *,
        bindings: Sequence[NamedSourceBinding],
        match_plan: SourceBindingMatchPlan | None,
        source_context: SourcePatternResolutionContext,
        plane_member_fields: frozenset[str] = frozenset(),
    ) -> "SourceBindingMatchedImageSet":
        return cls(
            parser=source_context.parser,
            source_paths_by_virtual_path=source_context.source_paths_by_virtual_path,
            source_metadata_by_path=source_context.source_metadata_by_path,
            metadata_rules=source_context.metadata_rules,
            bindings=tuple(bindings),
            match_plan=match_plan,
            identity_policy=SourceImageSetIdentityPolicy.from_plane_member_fields(
                plane_member_fields
            ),
        )

    def expand(
        self,
        anchors: Sequence[SourceCandidatePath],
        *,
        source_universe: Sequence[SourceCandidatePath],
    ) -> tuple[SourceCandidatePath, ...]:
        """Return source files for every alias in each selected image-set anchor."""
        selector_bindings = tuple(
            binding
            for binding in SourceBindingCandidateMatcher.selector_bindings(
                self.bindings
            )
            if binding.participates_in_execution_anchoring
        )
        if (
            len(selector_bindings) == 1
        ):
            return self._expand_single_alias(
                anchors,
                binding=selector_bindings[0],
                source_universe=source_universe,
            )
        if (
            self.match_plan is None
            or self.match_plan.method is not SourceBindingMatchMethod.METADATA
            or not self.match_plan.dimensions
        ):
            return SourceBindingCandidateMatcher.compatible_candidates(
                anchors,
                bindings=selector_bindings,
                source_context=self,
            )

        selected_anchor_candidates = self._complete_alias_set(anchors, selector_bindings)
        if selected_anchor_candidates is not None:
            return selected_anchor_candidates

        expanded: list[SourceCandidatePath] = []
        for anchor in anchors:
            anchor_binding = self._matching_binding(
                anchor,
                selector_bindings=selector_bindings,
            )
            if anchor_binding is None:
                continue
            expanded.extend(
                self._expand_anchor(
                    anchor,
                    anchor_binding=anchor_binding,
                    selector_bindings=selector_bindings,
                    source_universe=source_universe,
                )
            )
        if not expanded:
            return SourceBindingCandidateMatcher.compatible_candidates(
                anchors,
                bindings=selector_bindings,
                source_context=self,
            )
        return tuple(dict.fromkeys(expanded))

    def _expand_single_alias(
        self,
        anchors: Sequence[SourceCandidatePath],
        *,
        binding: NamedSourceBinding,
        source_universe: Sequence[SourceCandidatePath],
    ) -> tuple[SourceCandidatePath, ...]:
        compatible_anchors = SourceBindingCandidateMatcher.compatible_candidates(
            anchors,
            bindings=(binding,),
            source_context=self,
        )
        if compatible_anchors:
            return compatible_anchors
        if not source_universe:
            return ()

        anchor_identities = frozenset(
            self._source_image_set_identity(anchor)
            for anchor in anchors
        )
        return tuple(
            candidate
            for candidate in SourceBindingCandidateMatcher.compatible_candidates(
                source_universe,
                bindings=(binding,),
                source_context=self,
            )
            if SourceImageSetIdentityCompatibility.any_match(
                frozenset((self._source_image_set_identity(candidate),)),
                anchor_identities,
            )
        )

    def _source_image_set_identity(
        self,
        candidate: SourceCandidatePath,
    ) -> SourceImageSetIdentity:
        metadata = self.candidate_metadata(candidate).first_required(candidate)
        return SourceImageSetIdentity.from_metadata(
            metadata,
            fallback_source_path=candidate,
            policy=self.identity_policy,
        )

    def _expand_anchor(
        self,
        anchor: SourceCandidatePath,
        *,
        anchor_binding: NamedSourceBinding,
        selector_bindings: tuple[NamedSourceBinding, ...],
        source_universe: Sequence[SourceCandidatePath],
    ) -> tuple[SourceCandidatePath, ...]:
        anchor_metadata = self.candidate_metadata(anchor).first_required(anchor)
        anchor_values = self._dimension_values(
            anchor_metadata,
            anchor_binding,
            allow_missing=self.has_virtual_source_workspace,
        )
        if anchor_values is None:
            return tuple(
                dict.fromkeys(
                    candidate
                    for binding in selector_bindings
                    for candidate in self._expand_single_alias(
                        (anchor,),
                        binding=binding,
                        source_universe=source_universe,
                    )
                )
            )
        if not anchor_values:
            raise ValueError(
                "Matched source image-set anchor lacks metadata declared by the "
                f"match plan: {anchor!s}."
            )

        selected: list[SourceCandidatePath] = []
        for binding in selector_bindings:
            matches = tuple(
                candidate
                for candidate in source_universe
                if self._candidate_matches_anchor_set(
                    candidate,
                    binding=binding,
                    anchor_values=anchor_values,
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    "Matched source image set expected exactly one candidate for "
                    f"alias {binding.alias!r} anchored by {anchor!s}, got "
                    f"{len(matches)}."
                )
            selected.append(matches[0])
        return tuple(selected)

    def _complete_alias_set(
        self,
        candidates: Sequence[SourceCandidatePath],
        selector_bindings: tuple[NamedSourceBinding, ...],
    ) -> tuple[SourceCandidatePath, ...] | None:
        selected: list[SourceCandidatePath] = []
        for binding in selector_bindings:
            matches = tuple(
                candidate
                for candidate in candidates
                if SourceBindingCandidateMatcher.matches(
                    candidate,
                    binding=binding,
                    source_context=self,
                )
            )
            if len(matches) != 1:
                return None
            selected.append(matches[0])
        return tuple(dict.fromkeys(selected))

    def _matching_binding(
        self,
        candidate: SourceCandidatePath,
        *,
        selector_bindings: tuple[NamedSourceBinding, ...],
    ) -> NamedSourceBinding | None:
        matches = tuple(
            binding
            for binding in selector_bindings
            if SourceBindingCandidateMatcher.matches(
                    candidate,
                    binding=binding,
                    source_context=self,
                )
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError(
                "Matched source image-set anchor must resolve to exactly one "
                f"source alias, got {len(matches)} for {candidate!s}."
            )
        return matches[0]

    def _candidate_matches_anchor_set(
        self,
        candidate: SourceCandidatePath,
        *,
        binding: NamedSourceBinding,
        anchor_values: Mapping[int, str],
    ) -> bool:
        if not SourceBindingCandidateMatcher.matches(
            candidate,
            binding=binding,
            source_context=self,
        ):
            return False
        metadata = self.candidate_metadata(candidate).first_required(candidate)
        candidate_values = self._dimension_values(
            metadata,
            binding,
            allow_missing=self.has_virtual_source_workspace,
        )
        if candidate_values is None:
            return False
        if not candidate_values:
            raise ValueError(
                f"Source alias {binding.alias!r} has no match-plan dimensions."
            )
        return all(
            anchor_values.get(dimension_index) == value
            for dimension_index, value in candidate_values.items()
        )

    def _dimension_values(
        self,
        metadata: SourceMetadataRecord,
        binding: NamedSourceBinding,
        *,
        allow_missing: bool = False,
    ) -> Mapping[int, str] | None:
        values: dict[int, str] = {}
        for dimension_index, dimension in enumerate(self.match_plan.dimensions):
            field = dimension.field_for_alias(binding.alias)
            if field is None:
                continue
            value = source_metadata_value(metadata, field)
            if value is None:
                if allow_missing:
                    return None
                raise ValueError(
                    "Source binding match plan could not read metadata field "
                    f"{field!r} for alias {binding.alias!r}."
                )
            values[dimension_index] = str(value)
        return MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class SourceFileUniverse:
    """Concrete file universe plus the backend that names those files."""

    files: tuple[str, ...]
    backend: Backend


@dataclass(frozen=True, slots=True)
class SourceUniverseRuntimeState:
    """Resolved source universes assembled from the registered request family."""

    step_input_universe: SourceFileUniverse | None = None
    pipeline_start_universe: SourceFileUniverse | None = None
    load_universe: SourceFileUniverse | None = None
    step_input_source_paths: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    source_metadata_by_path: Mapping[str, SourceMetadataMapping] = field(
        default_factory=lambda: MappingProxyType({})
    )
    pipeline_source_candidate_files: tuple[str, ...] = ()

    def with_source_metadata(
        self,
        source_metadata_by_path: Mapping[str, SourceMetadataMapping],
    ) -> "SourceUniverseRuntimeState":
        if self.source_metadata_by_path is source_metadata_by_path:
            return self
        if not self.source_metadata_by_path:
            return replace(
                self,
                source_metadata_by_path=source_metadata_by_path,
            )
        if not source_metadata_by_path:
            return self
        merged = dict(self.source_metadata_by_path)
        merged.update(source_metadata_by_path)
        return replace(self, source_metadata_by_path=MappingProxyType(merged))

    def require_step_input_universe(self) -> SourceFileUniverse:
        if self.step_input_universe is None:
            raise RuntimeError("Source universe runtime state has no step-input universe.")
        return self.step_input_universe

    def require_pipeline_start_universe(self) -> SourceFileUniverse:
        if self.pipeline_start_universe is None:
            raise RuntimeError(
                "Source universe runtime state has no pipeline-start universe."
            )
        return self.pipeline_start_universe

    def require_load_universe(self) -> SourceFileUniverse:
        if self.load_universe is None:
            raise RuntimeError("Source universe runtime state has no load universe.")
        return self.load_universe

    def runtime_context(
        self,
        request: "SourceBindingRuntimeContextRequest",
        source_metadata_by_path: Mapping[str, SourceMetadataMapping],
    ) -> SourceBindingRuntimeContext:
        """Build the runtime context from source-universe contributions."""
        step_input_universe = self.require_step_input_universe()
        pipeline_source_universe = self.require_pipeline_start_universe()
        return SourceBindingRuntimeContext(
            step_input_files=step_input_universe.files,
            current_step_input_files=request.current_step_input_files(
                step_input_universe
            ),
            current_image_files=request.matching_files,
            step_input_dir=str(request.plan.input_dir),
            step_input_source_backend=request.plan.read_backend,
            step_input_storage_backend=Backend.MEMORY.value,
            step_input_source_paths=self.step_input_source_paths,
            source_metadata_by_path=source_metadata_by_path,
            source_metadata_is_normalized=True,
            pipeline_input_files=pipeline_source_universe.files,
            pipeline_source_candidate_files=self.pipeline_source_candidate_files,
            pipeline_input_backend=pipeline_source_universe.backend.value,
        )


@dataclass(frozen=True, slots=True)
class SourceUniverseRequest(metaclass=AutoRegisterMeta):
    """Plan-artifact request for source-binding runtime universe resolution."""

    __registry_key__ = "universe_request_kind"
    __skip_if_no_key__ = True

    universe_request_kind: ClassVar[str | None] = None
    context: "ProcessingContext"
    plan: "FunctionStepExecutionPlan"
    matching_files: tuple[str, ...]
    source_backend: Backend
    source_projection: VirtualWorkspaceSourceProjection | None

    @classmethod
    def registered_request_types(cls) -> tuple[type["SourceUniverseRequest"], ...]:
        """Return registered concrete runtime plan request classes."""
        request_types: list[type[SourceUniverseRequest]] = []
        for request_type in cls.__registry__.values():
            if (
                issubclass(request_type, cls)
                and request_type not in request_types
            ):
                request_types.append(request_type)
        return tuple(request_types)

    @classmethod
    def from_runtime_context(
        cls,
        request: "SourceBindingRuntimeContextRequest",
    ) -> "SourceUniverseRequest":
        """Build this registered request type from the runtime context request."""
        return cls(
            context=request.context,
            plan=request.plan,
            matching_files=request.matching_files,
            source_backend=request.source_backend,
            source_projection=request.source_projection,
        )

    @classmethod
    def runtime_state(
        cls,
        request: "SourceBindingRuntimeContextRequest",
    ) -> SourceUniverseRuntimeState:
        """Resolve every registered source-universe request into runtime state."""
        state = SourceUniverseRuntimeState()
        for request_type in cls.registered_request_types():
            universe_request = request_type.from_runtime_context(request)
            universe = SourceUniverseStrategy.universe(universe_request)
            state = universe_request.contribute_runtime_state(state, universe)
        return state

    def contribute_runtime_state(
        self,
        state: SourceUniverseRuntimeState,
        universe: SourceFileUniverse,
    ) -> SourceUniverseRuntimeState:
        """Contribute resolved metadata to runtime state."""
        return state.with_source_metadata(self.source_metadata_by_universe(universe))

    @property
    def requires_step_input_selector_resolution(self) -> bool:
        return self.plan.source_universe_plan.requires_step_input_selector_resolution

    @property
    def uses_virtual_workspace_projection(self) -> bool:
        return (
            self.source_backend is Backend.VIRTUAL_WORKSPACE
            and self.source_projection is not None
        )

    @property
    def requires_full_pipeline_source_universe(self) -> bool:
        return self.plan.source_universe_plan.requires_full_pipeline_source_universe

    @property
    def uses_pipeline_start_binding_origin(self) -> bool:
        return self.plan.source_universe_plan.uses_pipeline_start_binding_origin

    @property
    def step_input_source_paths(self) -> Mapping[str, str]:
        projection = self.source_projection
        if projection is None:
            return MappingProxyType({})
        return projection.source_paths_by_virtual_path

    @property
    def source_metadata_by_path(self) -> Mapping[str, SourceMetadataMapping]:
        projection = self.source_projection
        if projection is None:
            return MappingProxyType({})
        return projection.source_metadata_by_path

    def source_context(self) -> SourcePatternResolutionContext:
        metadata_rules = self.plan.source_binding_plan.metadata_rules
        if self.source_projection is not None:
            return SourcePatternResolutionContext.from_projection(
                parser=self.context.microscope_handler.parser,
                projection=self.source_projection,
                metadata_rules=metadata_rules,
            )
        return SourcePatternResolutionContext.from_sources(
            parser=self.context.microscope_handler.parser,
            source_paths_by_virtual_path={},
            metadata_rules=metadata_rules,
        )

    def source_metadata_by_universe(
        self,
        universe: SourceFileUniverse,
    ) -> Mapping[str, SourceMetadataMapping]:
        if self.source_projection is not None:
            return self.source_projection.source_metadata_by_path
        return self.source_context().source_metadata_by_paths(universe.files)

    def require_source_projection(self) -> VirtualWorkspaceSourceProjection:
        projection = self.source_projection
        if projection is None:
            raise RuntimeError(
                "Virtual workspace source universe requires projection metadata."
            )
        return projection

    def axis_files(self) -> tuple[str, ...]:
        return tuple(
            self.plan.get_paths_for_axis(
                self.context.input_dir,
                self.source_backend.value,
            )
        )

    def physical_full_universe_backend(self) -> Backend:
        return PipelineStartListingBackendPolicy.backend_for(self.source_backend)


@dataclass(frozen=True, slots=True)
class StepInputSourceUniverseRequest(SourceUniverseRequest):
    """Request for the source universe represented by the current step input."""

    universe_request_kind = "step_input"

    def contribute_runtime_state(
        self,
        state: SourceUniverseRuntimeState,
        universe: SourceFileUniverse,
    ) -> SourceUniverseRuntimeState:
        state = replace(
            state,
            step_input_universe=universe,
            load_universe=state.load_universe or universe,
            step_input_source_paths=self.step_input_source_paths,
        )
        return SourceUniverseRequest.contribute_runtime_state(self, state, universe)


@dataclass(frozen=True, slots=True)
class PipelineStartSourceUniverseRequest(SourceUniverseRequest):
    """Request for the source universe represented by the pipeline start."""

    universe_request_kind = "pipeline_start"

    def contribute_runtime_state(
        self,
        state: SourceUniverseRuntimeState,
        universe: SourceFileUniverse,
    ) -> SourceUniverseRuntimeState:
        load_universe = self.load_universe()
        state = replace(
            state,
            pipeline_start_universe=universe,
            pipeline_source_candidate_files=self.pipeline_source_candidate_files(universe),
            load_universe=state.load_universe if load_universe is None else load_universe,
        )
        return SourceUniverseRequest.contribute_runtime_state(self, state, universe)

    def pipeline_source_candidate_files(
        self,
        universe: SourceFileUniverse,
    ) -> tuple[str, ...]:
        projection = self.source_projection
        if projection is None:
            return universe.files
        axis_id = None if self.requires_full_pipeline_source_universe else self.plan.axis_id
        return tuple(
            dict.fromkeys(
                (
                    *projection.pipeline_start_candidate_files(axis_id=axis_id),
                    *self.physical_pipeline_source_candidate_files(),
                )
            )
        )

    def physical_pipeline_source_candidate_files(self) -> tuple[str, ...]:
        universe_backend = self.physical_full_universe_backend()
        return tuple(
            str(path)
            for path in self.context.filemanager.list_files(
                str(self.context.input_dir),
                universe_backend.value,
                recursive=True,
            )
        )

    def load_universe(self) -> SourceFileUniverse | None:
        projection = self.source_projection
        if projection is None:
            return None
        if not self.uses_pipeline_start_binding_origin:
            return None
        return SourceFileUniverse(
            files=projection.pipeline_start_files(axis_id=self.plan.axis_id),
            backend=self.source_backend,
        )


class PipelineStartListingBackendPolicy(
    EnumKeyedStrategyMixin[Backend],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Backend policy for full pipeline-start file listing."""

    __registry_key__ = "backend_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "source_backend"
    __enum_label_attr__ = "backend_label"

    source_backend: ClassVar[Backend | None] = None
    backend_label: ClassVar[str | None] = None

    @classmethod
    def backend_for(cls, source_backend: Backend) -> Backend:
        strategy_type = cls.__registry__.get(source_backend.value)
        if strategy_type is None:
            return source_backend
        return strategy_type().listing_backend()

    @abstractmethod
    def listing_backend(self) -> Backend:
        """Return the backend used for recursive full-universe listing."""


class DiskPipelineStartListingBackendPolicy(PipelineStartListingBackendPolicy):
    """Pipeline-start fan-out policy that lists disk files."""

    def listing_backend(self) -> Backend:
        return Backend.DISK


class MemoryPipelineStartListingBackendPolicy(DiskPipelineStartListingBackendPolicy):
    """Memory-backed pipeline-start fan-out lists disk files."""

    source_backend = Backend.MEMORY


class VirtualWorkspacePipelineStartListingBackendPolicy(
    DiskPipelineStartListingBackendPolicy
):
    """Virtual-workspace pipeline-start fan-out lists disk files."""

    source_backend = Backend.VIRTUAL_WORKSPACE


class SourceUniverseStrategy(
    MostDerivedContextStrategyMixin[SourceUniverseRequest],
    ABC,
):
    """Registered source-universe selection for source-binding runtime scopes."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def universe(cls, request: SourceUniverseRequest) -> SourceFileUniverse:
        strategy = cls.for_context(
            request,
            error_subject="Source universe",
        )
        if strategy is None:
            raise ValueError("Source universe requires a strategy.")
        return strategy.source_universe(request)

    @abstractmethod
    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        """Return source files and backend for pipeline-start bindings."""


class StepInputSourceUniverseStrategy(SourceUniverseStrategy, ABC):
    """Source-universe strategy branch for current step input."""

    def matches(self, request: SourceUniverseRequest) -> bool:
        return isinstance(
            request,
            StepInputSourceUniverseRequest,
        ) and self.matches_step_input(request)

    @abstractmethod
    def matches_step_input(self, request: StepInputSourceUniverseRequest) -> bool:
        """Return whether this strategy owns one step-input request."""


class PipelineStartSourceUniverseStrategy(SourceUniverseStrategy, ABC):
    """Source-universe strategy branch for original pipeline input."""

    def matches(self, request: SourceUniverseRequest) -> bool:
        return isinstance(
            request,
            PipelineStartSourceUniverseRequest,
        ) and self.matches_pipeline_start(request)

    @abstractmethod
    def matches_pipeline_start(
        self,
        request: PipelineStartSourceUniverseRequest,
    ) -> bool:
        """Return whether this strategy owns one pipeline-start request."""


class AxisFilesSourceUniverseStrategy(SourceUniverseStrategy):
    """Source-universe strategy that uses current-axis files from the source backend."""

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        return SourceFileUniverse(
            files=request.axis_files(),
            backend=request.source_backend,
        )


class StepInputAxisFilesSourceUniverseStrategy(
    AxisFilesSourceUniverseStrategy,
    StepInputSourceUniverseStrategy,
    ABC,
):
    """Axis-file universe strategy branch for current step input."""


class PipelineStartAxisFilesSourceUniverseStrategy(
    AxisFilesSourceUniverseStrategy,
    PipelineStartSourceUniverseStrategy,
    ABC,
):
    """Axis-file universe strategy branch for original pipeline input."""


class CurrentPatternStepInputSourceUniverseStrategy(StepInputSourceUniverseStrategy):
    """Use the already-loaded pattern files when selectors do not need fan-out."""

    strategy_key = "step_input_current_pattern"

    def matches_step_input(self, request: StepInputSourceUniverseRequest) -> bool:
        return not request.requires_step_input_selector_resolution

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        return SourceFileUniverse(
            files=request.matching_files,
            backend=request.source_backend,
        )


class VirtualWorkspaceStepInputSourceUniverseStrategy(StepInputSourceUniverseStrategy):
    """Use source-schema virtual files when selector resolution must span sources."""

    strategy_key = "step_input_virtual_workspace_source_projection"

    def matches_step_input(self, request: StepInputSourceUniverseRequest) -> bool:
        return (
            request.requires_step_input_selector_resolution
            and request.uses_virtual_workspace_projection
        )

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        return SourceFileUniverse(
            files=request.require_source_projection().pipeline_start_files(
                axis_id=request.plan.axis_id
            ),
            backend=request.source_backend,
        )


class PhysicalAxisStepInputSourceUniverseStrategy(
    StepInputAxisFilesSourceUniverseStrategy,
):
    """Use physical axis files when source selectors need fan-out outside VWS."""

    strategy_key = "step_input_physical_axis"

    def matches_step_input(self, request: StepInputSourceUniverseRequest) -> bool:
        return (
            request.requires_step_input_selector_resolution
            and not request.uses_virtual_workspace_projection
        )


class AxisScopedPipelineStartSourceUniverseStrategy(
    PipelineStartAxisFilesSourceUniverseStrategy,
):
    """Use the current axis source files when full pipeline fan-out is unnecessary."""

    strategy_key = "axis_scoped"

    def matches_pipeline_start(
        self,
        request: PipelineStartSourceUniverseRequest,
    ) -> bool:
        return not request.requires_full_pipeline_source_universe

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        if request.uses_virtual_workspace_projection:
            return SourceFileUniverse(
                files=request.require_source_projection().pipeline_start_files(
                    axis_id=request.plan.axis_id
                ),
                backend=request.source_backend,
            )
        return SourceFileUniverse(
            files=request.axis_files(),
            backend=request.source_backend,
        )


class VirtualWorkspacePipelineStartSourceUniverseStrategy(
    PipelineStartSourceUniverseStrategy,
):
    """Use declared virtual-workspace source paths for pipeline-start fan-out."""

    strategy_key = "virtual_workspace_source_projection"

    def matches_pipeline_start(
        self,
        request: PipelineStartSourceUniverseRequest,
    ) -> bool:
        return (
            request.requires_full_pipeline_source_universe
            and request.source_projection is not None
        )

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        projection = request.require_source_projection()
        return SourceFileUniverse(
            files=tuple(
                dict.fromkeys(
                    str(path)
                    for path in projection.source_paths_by_virtual_path.values()
                )
            ),
            backend=Backend.DISK,
        )


class PhysicalPipelineStartSourceUniverseStrategy(PipelineStartSourceUniverseStrategy):
    """Use a file listing backend for full pipeline fan-out outside VWS."""

    strategy_key = "physical_full_universe"

    def matches_pipeline_start(
        self,
        request: PipelineStartSourceUniverseRequest,
    ) -> bool:
        return (
            request.requires_full_pipeline_source_universe
            and request.source_projection is None
        )

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        universe_backend = request.physical_full_universe_backend()
        return SourceFileUniverse(
            files=tuple(
                str(path)
                for path in request.context.filemanager.list_files(
                    str(request.context.input_dir),
                    universe_backend.value,
                    recursive=True,
                )
            ),
            backend=universe_backend,
        )


@dataclass(frozen=True, slots=True)
class SourceBindingRuntimeContextRequest:
    """Build the source-binding runtime context from one resolved source universe."""

    context: "ProcessingContext"
    plan: "FunctionStepExecutionPlan"
    matching_files: tuple[str, ...]
    source_backend: Backend
    source_projection: VirtualWorkspaceSourceProjection | None

    @classmethod
    def from_context(
        cls,
        *,
        context: "ProcessingContext",
        plan: "FunctionStepExecutionPlan",
        matching_files: Sequence[str],
        source_projection: VirtualWorkspaceSourceProjection | None,
    ) -> "SourceBindingRuntimeContextRequest":
        source_backend = Backend(
            context.microscope_handler.get_primary_backend(
                context.input_dir,
                context.filemanager,
            )
        )
        return cls(
            context=context,
            plan=plan,
            matching_files=tuple(matching_files),
            source_backend=source_backend,
            source_projection=source_projection,
        )

    def runtime_context(self) -> SourceBindingRuntimeContext:
        cache = self.context.runtime_source_binding_context_cache
        cached = cache.runtime_context(
            plan=self.plan,
            matching_files=self.matching_files,
            source_backend=self.source_backend,
            source_projection=self.source_projection,
        )
        if cached is not None:
            return cached
        universe_state = self.runtime_universe_state()
        source_metadata_by_path = (
            cache.normalized_source_metadata(
                universe_state.source_metadata_by_path
            )
        )
        return cache.store_runtime_context(
            universe_state.runtime_context(
                self,
                source_metadata_by_path,
            ),
            plan=self.plan,
            matching_files=self.matching_files,
            source_backend=self.source_backend,
            source_projection=self.source_projection,
        )

    def runtime_universe_state(self) -> SourceUniverseRuntimeState:
        """Return cached source-universe state for this request."""
        cache = self.context.runtime_source_binding_context_cache
        cached = cache.runtime_universe_state(
            plan=self.plan,
            matching_files=self.matching_files,
            source_backend=self.source_backend,
            source_projection=self.source_projection,
        )
        if cached is not None:
            return cached
        return cache.store_runtime_universe_state(
            SourceUniverseRequest.runtime_state(self),
            plan=self.plan,
            matching_files=self.matching_files,
            source_backend=self.source_backend,
            source_projection=self.source_projection,
        )

    def current_step_input_files(
        self,
        step_input_universe: SourceFileUniverse,
    ) -> tuple[str, ...]:
        del step_input_universe
        return self.matching_files
