"""Compiled-plan orchestration for FunctionStep."""

from __future__ import annotations

import logging
import os
import re
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from metaclass_registry import AutoRegisterMeta
import psutil

from openhcs.constants import MULTIPROCESSING_AXIS
from openhcs.constants.constants import (
    LOADABLE_IMAGE_EXTENSIONS,
    Backend,
)
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import CompiledFunctionGroup
from openhcs.core.progress import ProgressPhase, ProgressStatus, emit
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
)
from openhcs.core.source_matching import (
    source_component_metadata_values,
    source_filters_match,
    source_metadata_value,
    source_metadata_values_equal,
)
from openhcs.core.steps.function_io import (
    bulk_preload_step_images,
    generate_materialized_paths,
    save_materialized_data,
    update_metadata_for_zarr_conversion,
)
from openhcs.core.steps.function_outputs import finalize_function_step_outputs
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from openhcs.core.steps.function_runtime import (
    PatternGroupExecutionRequest,
    prepare_compiled_function_group,
    _process_single_pattern_group,
)


logger = logging.getLogger(__name__)
_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
_PROFILE_RUNTIME_PATH_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME_PATH"
PatternCollection = list[Any] | dict[Any, list[Any]]


@dataclass(frozen=True, slots=True)
class SourcePatternResolutionContext:
    """Source paths and metadata available while filtering execution anchors."""

    parser: Any
    source_paths_by_virtual_path: Mapping[str, str]
    source_metadata_by_path: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )

    @property
    def has_virtual_source_workspace(self) -> bool:
        return bool(self.source_paths_by_virtual_path)

    def candidate_paths(self, pattern: Any) -> tuple[str, ...]:
        pattern_path = str(pattern)
        path = Path(pattern_path)
        keys = tuple(dict.fromkeys((pattern_path, path.as_posix(), path.name)))
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
        return tuple(dict.fromkeys((*keys, *virtual_matches, *mapped)))

    def _matching_virtual_paths(self, pattern_key: str) -> tuple[str, ...]:
        if pattern_key in self.source_paths_by_virtual_path:
            return ()
        matcher = SourceAnchorPatternTemplateMatcher.from_pattern(pattern_key)
        if matcher is None:
            return ()
        return tuple(
            virtual_path
            for virtual_path in self.source_paths_by_virtual_path
            if not Path(virtual_path).is_absolute()
            and matcher.matches(Path(virtual_path).name)
        )

    def candidate_metadata(self, pattern: Any) -> tuple[Mapping[str, Any], ...]:
        paths = self.candidate_paths(pattern)
        declared = tuple(
            metadata
            for path in paths
            for metadata in (self.source_metadata_by_path.get(path),)
            if metadata
        )
        parsed = tuple(
            metadata
            for path in paths
            if path not in self.source_metadata_by_path
            for metadata in (self.parser.parse_filename(path) or {},)
            if metadata
        )
        return (*declared, *parsed)

    def has_metadata_field(
        self,
        patterns: Sequence[Any],
        field: str,
    ) -> bool:
        return any(
            source_metadata_value(metadata, field) is not None
            for pattern in patterns
            for metadata in self.candidate_metadata(pattern)
        )


@dataclass(frozen=True, slots=True)
class SourceAnchorPatternTemplateMatcher:
    """Matcher for OpenHCS anchor patterns with variable-component placeholders."""

    pattern: str
    regex: re.Pattern[str]

    @classmethod
    def from_pattern(
        cls,
        pattern: str,
    ) -> "SourceAnchorPatternTemplateMatcher | None":
        if "{" not in pattern or "}" not in pattern:
            return None
        regex_parts: list[str] = []
        cursor = 0
        for match in re.finditer(r"\{[^{}]+\}", pattern):
            regex_parts.append(re.escape(pattern[cursor : match.start()]))
            regex_parts.append(r"[^/]*")
            cursor = match.end()
        regex_parts.append(re.escape(pattern[cursor:]))
        return cls(
            pattern=pattern,
            regex=re.compile(rf"^{''.join(regex_parts)}$"),
        )

    def matches(self, virtual_path: str) -> bool:
        return bool(
            self.regex.match(virtual_path)
            or self.regex.match(Path(virtual_path).as_posix())
            or self.regex.match(Path(virtual_path).name)
        )


@dataclass(frozen=True, slots=True)
class SourceWorkspaceAnchorProjection:
    """Source workspace paths and metadata projected for anchor resolution."""

    paths_by_virtual_path: Mapping[str, str]
    metadata_by_path: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_openhcs_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> "SourceWorkspaceAnchorProjection":
        from openhcs.microscopes.openhcs import FIELDS, workspace_mapping_source_ref

        subdirectories = metadata.get(FIELDS.SUBDIRECTORIES)
        if not isinstance(subdirectories, Mapping):
            return cls(paths_by_virtual_path={}, metadata_by_path={})

        paths_by_virtual_path: dict[str, str] = {}
        metadata_by_path: dict[str, Mapping[str, Any]] = {}
        for subdirectory_metadata in subdirectories.values():
            if not isinstance(subdirectory_metadata, Mapping):
                continue

            workspace_mapping = subdirectory_metadata.get(FIELDS.WORKSPACE_MAPPING)
            if isinstance(workspace_mapping, Mapping):
                paths_by_virtual_path.update(
                    {
                        str(virtual_path): workspace_mapping_source_ref(source_ref)
                        for virtual_path, source_ref in workspace_mapping.items()
                    }
                )

            source_metadata = subdirectory_metadata.get(FIELDS.SOURCE_METADATA)
            if isinstance(source_metadata, Mapping):
                for virtual_path, values in source_metadata.items():
                    if isinstance(values, Mapping):
                        metadata_by_path[str(virtual_path)] = values

        return cls(
            paths_by_virtual_path=paths_by_virtual_path,
            metadata_by_path=metadata_by_path,
        )


class SourceAnchorSelectionStatus(str, Enum):
    """Outcome of source-bound execution-anchor resolution."""

    SELECTED = "selected"
    DEFERRED_TO_RUNTIME = "deferred_to_runtime"


@dataclass(frozen=True, slots=True)
class SourceAnchorPatternSelection:
    """Resolved source-compatible anchors plus the authority that owns them."""

    patterns: tuple[Any, ...]
    status: SourceAnchorSelectionStatus
    reason: str

    @classmethod
    def selected(
        cls,
        patterns: Sequence[Any],
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
        patterns: Sequence[Any],
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


def _runtime_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_step_profile(label: str, seconds: float, **fields: Any) -> None:
    if not _runtime_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)
    if profile_path := os.environ.get(_PROFILE_RUNTIME_PATH_ENV):
        with open(profile_path, "a", encoding="utf-8") as handle:
            handle.write(f"RUNTIME_PROFILE {label} {seconds:.6f}s {field_text}\n")


def _filter_patterns_by_component(
    patterns: PatternCollection,
    component: str,
    target_value: str,
    microscope_handler: Any,
) -> PatternCollection:
    """Filter pattern strings by a fixed parsed component value."""
    from openhcs.formats.pattern.pattern_discovery import PatternDiscoveryEngine

    def filter_pattern_list(pattern_list: list[Any]) -> list[Any]:
        filtered = []
        for pattern in pattern_list:
            metadata = microscope_handler.parser.parse_filename(str(pattern))
            if metadata and str(metadata.get(component)) == str(target_value):
                filtered.append(pattern)
        return filtered

    if isinstance(patterns, dict):
        filtered_by_group = {}
        for group_key, pattern_list in patterns.items():
            filtered_list = filter_pattern_list(pattern_list)
            if filtered_list:
                filtered_by_group[group_key] = filtered_list
        return filtered_by_group

    return filter_pattern_list(patterns)


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
        pattern_list: Sequence[Any],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[Any]:
        """Return source-compatible anchor patterns for one execution group."""

    def _source_compatible_anchor_selection(
        self,
        pattern_list: Sequence[Any],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> SourceAnchorPatternSelection:
        selector_bindings = self._selector_bindings(bindings)
        if not selector_bindings:
            return SourceAnchorPatternSelection.selected(
                pattern_list,
                reason="no selector bindings participate in anchor resolution",
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
                for binding in selector_bindings
            )
        ]
        if compatible:
            return SourceAnchorPatternSelection.selected(compatible)
        if self._metadata_selector_fields_are_unavailable(
            pattern_list,
            bindings=selector_bindings,
            source_context=source_context,
        ):
            return SourceAnchorPatternSelection.deferred_to_runtime(
                pattern_list,
                reason=(
                    "selector metadata fields are unavailable at the execution "
                    "anchor boundary"
                ),
            )
        return SourceAnchorPatternSelection.selected(
            (),
            reason="source selectors resolved no compatible anchor patterns",
        )

    @staticmethod
    def _metadata_selector_fields_are_unavailable(
        pattern_list: Sequence[Any],
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
    def _selector_bindings(
        bindings: Sequence[NamedSourceBinding],
    ) -> tuple[NamedSourceBinding, ...]:
        return tuple(
            binding
            for binding in bindings
            if binding.required and binding.requires_selector_resolution
        )

    @staticmethod
    def _pattern_matches_source_binding(
        pattern: Any,
        *,
        binding: NamedSourceBinding,
        source_context: SourcePatternResolutionContext,
    ) -> bool:
        selector = binding.selector
        if not any(
            source_filters_match(path, selector.filters)
            for path in source_context.candidate_paths(pattern)
        ):
            return False

        if not selector.components and not selector.metadata:
            return True

        metadata_candidates = source_context.candidate_metadata(pattern)
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
                    source_metadata_value(metadata, metadata_selector.field),
                )
            ):
                return False

        return True


class DefaultSourceBoundAnchorPatternPolicy(SourceBoundAnchorPatternPolicy):
    """Keep every selector-compatible source anchor."""

    def select(
        self,
        pattern_list: Sequence[Any],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[Any]:
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
        pattern_list: Sequence[Any],
        *,
        bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[Any]:
        selection = self._source_compatible_anchor_selection(
            pattern_list,
            bindings=bindings,
            source_context=source_context,
        )
        compatible = list(selection.patterns)
        selector_bindings = self._selector_bindings(bindings)
        anchor_bindings = tuple(
            binding
            for binding in selector_bindings
            if binding.participates_in_execution_anchoring
        )
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
        compatible: Sequence[Any],
        *,
        selector_bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[Any]:
        """Return one execution anchor per matched image set."""


class OrderMatchedImageSetAnchorPatternPolicy(MatchedImageSetAnchorPatternPolicy):
    """Source aliases are paired by order within one logical image set."""

    policy_key = SourceBindingMatchMethod.ORDER.value

    def _deduplicate_matched_image_sets(
        self,
        compatible: Sequence[Any],
        *,
        selector_bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[Any]:
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
        compatible: Sequence[Any],
        *,
        selector_bindings: Sequence[NamedSourceBinding],
        source_context: SourcePatternResolutionContext,
    ) -> list[Any]:
        if self._match_plan is None or not self._match_plan.dimensions:
            raise ValueError(
                "METADATA source binding requires explicit match dimensions "
                "to collapse source-bound execution anchors."
            )

        bindings_by_alias = {binding.alias: binding for binding in selector_bindings}
        deduplicated: list[Any] = []
        seen: set[tuple[str, ...]] = set()
        for pattern in compatible:
            metadata = next(iter(source_context.candidate_metadata(pattern)), {})
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
                bindings_by_alias=bindings_by_alias,
            )
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(pattern)
        return deduplicated

    def _matching_binding(
        self,
        pattern: Any,
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
        metadata: Mapping[str, Any],
        *,
        binding: NamedSourceBinding,
        bindings_by_alias: Mapping[str, NamedSourceBinding],
    ) -> tuple[str, ...]:
        assert self._match_plan is not None
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
                raise ValueError(
                    "METADATA source binding could not read match field "
                    f"{field!r} for alias {binding.alias!r}."
                )
            values.append(value)
        return tuple(values)



class FunctionStepExecutor:
    """Run one compiled FunctionStep plan for one multiprocessing axis."""

    def __init__(self, context: ProcessingContext, step_index: int) -> None:
        self.context = context
        self.plan = FunctionStepExecutionPlan.from_context(context, step_index)
        self._source_workspace_anchor_projection: (
            SourceWorkspaceAnchorProjection | None
        ) = None

    def _source_pattern_context(self) -> SourcePatternResolutionContext:
        """Return source-path context used to filter source-bound anchors."""

        projection = self._source_workspace_projection()
        return SourcePatternResolutionContext(
            parser=self.context.microscope_handler.parser,
            source_paths_by_virtual_path=projection.paths_by_virtual_path,
            source_metadata_by_path=projection.metadata_by_path,
        )

    def _source_workspace_projection(self) -> SourceWorkspaceAnchorProjection:
        """Return declared source workspace views used by anchor policies."""

        if self._source_workspace_anchor_projection is None:
            metadata_handler = self.context.microscope_handler.metadata_handler
            metadata = metadata_handler._load_metadata_dict(self.context.plate_path)
            self._source_workspace_anchor_projection = (
                SourceWorkspaceAnchorProjection.from_openhcs_metadata(metadata)
            )
        return self._source_workspace_anchor_projection

    @classmethod
    def execute(cls, context: ProcessingContext, step_index: int) -> None:
        step_plan = context.step_plans[step_index]
        step_name = step_plan.step_name or f"step_{step_index}"
        try:
            cls(context, step_index).run()
        except Exception as error:
            full_traceback = traceback.format_exc()
            logger.error(
                "Error in FunctionStep %s (%s): %s",
                step_index,
                step_name,
                error,
                exc_info=True,
            )
            logger.error(
                "Full traceback for FunctionStep %s (%s):\n%s",
                step_index,
                step_name,
                full_traceback,
            )
            raise

    def run(self) -> None:
        plan = self.plan
        step_started_at = time.perf_counter()
        self._log_execution_start()

        phase_started_at = time.perf_counter()
        patterns_by_axis = self._detect_patterns()
        _log_step_profile(
            "step_detect_patterns",
            time.perf_counter() - phase_started_at,
            step=plan.step_index,
            step_name=plan.step_name,
        )
        self._log_discovered_patterns(patterns_by_axis)
        phase_started_at = time.perf_counter()
        self._convert_input_if_needed()
        _log_step_profile(
            "step_convert_input",
            time.perf_counter() - phase_started_at,
            step=plan.step_index,
            step_name=plan.step_name,
        )
        self._require_patterns(patterns_by_axis)
        self._apply_sequential_filter(patterns_by_axis)

        phase_started_at = time.perf_counter()
        grouped_patterns = self._prepare_groups(patterns_by_axis)
        _log_step_profile(
            "step_prepare_groups",
            time.perf_counter() - phase_started_at,
            step=plan.step_index,
            step_name=plan.step_name,
        )
        total_groups = self._count_pattern_groups(grouped_patterns)
        phase_started_at = time.perf_counter()
        self._preload_inputs_if_needed(grouped_patterns)
        _log_step_profile(
            "step_preload_inputs",
            time.perf_counter() - phase_started_at,
            step=plan.step_index,
            step_name=plan.step_name,
        )
        phase_started_at = time.perf_counter()
        self._prepare_callables(grouped_patterns)
        _log_step_profile(
            "step_prepare_callables",
            time.perf_counter() - phase_started_at,
            step=plan.step_index,
            step_name=plan.step_name,
        )
        execution_started_at = time.perf_counter()
        self._execute_pattern_groups(
            grouped_patterns,
            total_groups,
        )
        execution_elapsed = time.perf_counter() - execution_started_at

        logger.info(
            "Completed step '%s' for axis %s in %.3fs.",
            plan.step_name,
            plan.axis_id,
            execution_elapsed,
        )
        finalization_started_at = time.perf_counter()
        finalize_function_step_outputs(self.context, plan)
        finalization_elapsed = time.perf_counter() - finalization_started_at
        logger.info(
            "FunctionStep %s (%s) completed for axis %s in %.3fs "
            "(execute=%.3fs, finalize=%.3fs).",
            plan.step_index,
            plan.step_name,
            plan.axis_id,
            time.perf_counter() - step_started_at,
            execution_elapsed,
            finalization_elapsed,
        )

    def _log_execution_start(self) -> None:
        plan = self.plan
        same_dir = str(plan.input_dir) == str(plan.output_dir)
        if plan.device_id is None:
            logger.debug(
                "Step %s is CPU-only, input_mem=%s, output_mem=%s",
                plan.step_index,
                plan.input_memory_type,
                plan.output_memory_type,
            )
        else:
            logger.debug(
                "Step %s uses gpu_id=%s, input_mem=%s, output_mem=%s",
                plan.step_index,
                plan.device_id,
                plan.input_memory_type,
                plan.output_memory_type,
            )
        logger.debug(
            "Step %s backends: read=%s, write=%s",
            plan.step_index,
            plan.read_backend,
            plan.write_backend,
        )
        logger.info(
            "Step %s (%s) I/O: read='%s', write='%s'.",
            plan.step_index,
            plan.step_name,
            plan.read_backend,
            plan.write_backend,
        )
        logger.info(
            "Step %s (%s) Paths: input_dir='%s', output_dir='%s', same_dir=%s",
            plan.step_index,
            plan.step_name,
            plan.input_dir,
            plan.output_dir,
            same_dir,
        )

    def _detect_patterns(self) -> dict[str, Any]:
        plan = self.plan
        axis_name = MULTIPROCESSING_AXIS.value
        return self.context.microscope_handler.auto_detect_patterns(
            str(plan.input_dir),
            self.context.filemanager,
            plan.read_backend,
            extensions=LOADABLE_IMAGE_EXTENSIONS,
            group_by=plan.group_by,
            variable_components=plan.variable_component_values,
            **{f"{axis_name}_filter": [plan.axis_id]},
        )

    def _log_discovered_patterns(self, patterns_by_axis: Mapping[str, Any]) -> None:
        plan = self.plan
        if plan.axis_id not in patterns_by_axis:
            logger.warning("No patterns found for axis %s.", plan.axis_id)
            return

        axis_patterns = patterns_by_axis[plan.axis_id]
        if isinstance(axis_patterns, dict):
            for component_value, pattern_list in axis_patterns.items():
                logger.debug(
                    "Component '%s' has %s patterns: %s",
                    component_value,
                    len(pattern_list),
                    pattern_list,
                )
            return

        logger.debug(
            "Found %s ungrouped patterns: %s",
            len(axis_patterns),
            axis_patterns,
        )

    def _convert_input_if_needed(self) -> None:
        plan = self.plan
        if not plan.has_input_conversion:
            return

        logger.info("Converting input data to zarr: %s", plan.input_conversion_dir)

        source_paths = plan.get_paths_for_axis(plan.input_dir, plan.read_backend)
        memory_data = self.context.filemanager.load_batch(source_paths, plan.read_backend)
        conversion_paths = generate_materialized_paths(
            source_paths,
            plan.input_dir,
            plan.input_conversion_dir,
        )

        save_materialized_data(
            self.context.filemanager,
            memory_data,
            conversion_paths,
            plan.input_conversion_backend,
            plan.zarr_config,
            self.context,
            plan.axis_id,
        )
        logger.info(
            "Converted %s input files to %s",
            len(conversion_paths),
            plan.input_conversion_dir,
        )

        conversion_dir = plan.input_conversion_dir
        zarr_subdir = (
            conversion_dir.name
            if plan.input_conversion_uses_virtual_workspace
            else None
        )
        update_metadata_for_zarr_conversion(
            conversion_dir.parent,
            plan.input_conversion_original_subdir,
            zarr_subdir,
            self.context,
        )

    def _require_patterns(self, patterns_by_axis: Mapping[str, Any]) -> None:
        plan = self.plan
        logger.info(
            "Starting step '%s' for axis %s (group_by=%s, variable_components=%s)",
            plan.step_name,
            plan.axis_id,
            plan.group_by_name,
            plan.variable_component_names,
        )
        if plan.axis_id not in patterns_by_axis:
            raise ValueError(
                f"No patterns detected for well '{plan.axis_id}' in step "
                f"'{plan.step_name}' (index: {plan.step_index}). "
                f"Check input directory: {plan.input_dir}"
            )
        if not tuple(plan.compiled_function_pattern.iter_invocations()):
            raise ValueError(
                f"Step plan missing compiled function invocations for step: {plan.step_name} "
                f"(index: {plan.step_index})"
            )

    def _apply_sequential_filter(self, patterns_by_axis: dict[str, Any]) -> None:
        if not self.context.current_sequential_combination:
            return

        seq_config = self.context.global_config.sequential_processing_config
        seq_component = seq_config.sequential_components[0].value
        target_value = self.context.current_sequential_combination[0]
        patterns_by_axis[self.plan.axis_id] = _filter_patterns_by_component(
            patterns_by_axis[self.plan.axis_id],
            seq_component,
            target_value,
            self.context.microscope_handler,
        )

    def _prepare_groups(
        self,
        patterns_by_axis: Mapping[str, Any],
    ) -> Mapping[Any, Sequence[Any]]:
        plan = self.plan
        grouped_patterns = (
            plan.compiled_function_pattern.prepare_grouped_patterns(
                patterns_by_axis[plan.axis_id],
                default_component=plan.group_by_value,
            )
        )
        grouped_patterns = self._filter_source_bound_anchor_patterns(grouped_patterns)
        grouped_patterns = self._collapse_artifact_driven_anchor_patterns(
            grouped_patterns,
        )
        if self._count_pattern_groups(grouped_patterns) == 0:
            raise ValueError(
                f"No pattern groups found for step {plan.step_index} "
                f"({plan.step_name}) in well {plan.axis_id}"
            )
        return grouped_patterns

    def _filter_source_bound_anchor_patterns(
        self,
        grouped_patterns: Mapping[Any, Sequence[Any]],
    ) -> Mapping[Any, Sequence[Any]]:
        """Restrict source-bound step anchors to compatible declared sources."""

        if not self.plan.source_binding_plan.has_primary_content:
            return grouped_patterns

        filtered: dict[Any, Sequence[Any]] = {}
        changed = False
        for component_value, pattern_list in grouped_patterns.items():
            compiled_group = self.plan.compiled_function_pattern.group_for_component(
                component_value
            )
            if compiled_group is None:
                filtered[component_value] = pattern_list
                continue
            bindings = self.plan.source_binding_plan.bindings_for_group(
                compiled_group.group_key
            )
            compatible = SourceBoundAnchorPatternPolicy.for_plan(
                self.plan.source_binding_plan
            ).select(
                pattern_list,
                bindings=bindings,
                source_context=self._source_pattern_context(),
            )
            filtered[component_value] = compatible
            changed = changed or len(compatible) != len(pattern_list)

        if changed:
            _log_step_profile(
                "step_filter_source_anchors",
                0.0,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                before=self._count_pattern_groups(grouped_patterns),
                after=self._count_pattern_groups(filtered),
            )
        return filtered

    def _collapse_artifact_driven_anchor_patterns(
        self,
        grouped_patterns: Mapping[Any, Sequence[Any]],
    ) -> Mapping[Any, Sequence[Any]]:
        """Drop duplicate main-image anchors for artifact-driven callables.

        A FunctionStep still needs a pattern item to bind source context and output
        naming. For adapter-managed artifact pipelines, however, the real inputs are
        declared runtime artifacts; the main image stack is not the semantic driver.
        If discovery finds multiple anchor patterns that differ only by components
        outside ``variable_components`` (for example channel), executing all of them
        repeats the same semantic module call.
        """
        collapsed: dict[Any, Sequence[Any]] = {}
        changed = False

        for component_value, pattern_list in grouped_patterns.items():
            compiled_group = self.plan.compiled_function_pattern.group_for_component(
                component_value
            )
            if compiled_group is None:
                collapsed[component_value] = pattern_list
                continue
            if not self._group_uses_artifact_driven_invocation(compiled_group):
                collapsed[component_value] = pattern_list
                continue

            deduplicated = self._deduplicate_anchor_patterns(pattern_list)
            collapsed[component_value] = deduplicated
            changed = changed or len(deduplicated) != len(pattern_list)

        if changed:
            _log_step_profile(
                "step_collapse_artifact_anchors",
                0.0,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                before=self._count_pattern_groups(grouped_patterns),
                after=self._count_pattern_groups(collapsed),
            )
        return collapsed

    def _group_uses_artifact_driven_invocation(
        self,
        compiled_group: CompiledFunctionGroup,
    ) -> bool:
        if not compiled_group.invocations:
            return False

        for invocation in compiled_group.invocations:
            runtime_adapter = invocation.contract.runtime_adapter
            if runtime_adapter is None or not runtime_adapter.manages_artifact_inputs:
                return False
            if not invocation.artifact_input_keys:
                return False
            if any(
                self.plan.artifact_outputs[key].kind is ArtifactKind.IMAGE
                for key in invocation.artifact_output_keys
                if key in self.plan.artifact_outputs
            ):
                return False
        return True

    def _deduplicate_anchor_patterns(
        self,
        pattern_list: Sequence[Any],
    ) -> list[Any]:
        seen: set[tuple[tuple[str, Any], ...]] = set()
        deduplicated: list[Any] = []
        identity_components = set(self.plan.variable_component_values)
        if self.plan.group_by_value is not None:
            identity_components.add(self.plan.group_by_value)
        parser = self.context.microscope_handler.parser

        for pattern in pattern_list:
            metadata = parser.parse_filename(str(pattern))
            if not metadata:
                deduplicated.append(pattern)
                continue
            key = tuple(
                sorted(
                    (component, value)
                    for component, value in metadata.items()
                    if component in identity_components
                )
            )
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(pattern)

        return deduplicated

    @staticmethod
    def _count_pattern_groups(grouped_patterns: Mapping[Any, Sequence[Any]]) -> int:
        return sum(len(pattern_list) for pattern_list in grouped_patterns.values())

    def _preload_inputs_if_needed(
        self,
        grouped_patterns: Mapping[Any, Sequence[str]],
    ) -> None:
        plan = self.plan
        if plan.read_backend == Backend.MEMORY.value:
            return

        process = psutil.Process(os.getpid())
        mem_before_mb = process.memory_info().rss / 1024 / 1024
        logger.debug("Memory before preload: %.1f MB RSS", mem_before_mb)

        if self.context.current_sequential_combination:
            patterns_to_preload = [
                pattern
                for pattern_list in grouped_patterns.values()
                for pattern in pattern_list
            ]
            logger.info(
                "Sequential mode: preloading %s filtered patterns",
                len(patterns_to_preload),
            )
            bulk_preload_step_images(
                plan.input_dir,
                plan.axis_id,
                plan.read_backend,
                self.context.filemanager,
                self.context.microscope_handler,
                plan.zarr_config,
                patterns_to_preload=patterns_to_preload,
                variable_components=plan.variable_component_values,
            )
        else:
            bulk_preload_step_images(
                plan.input_dir,
                plan.axis_id,
                plan.read_backend,
                self.context.filemanager,
                self.context.microscope_handler,
                plan.zarr_config,
            )

        mem_after_mb = process.memory_info().rss / 1024 / 1024
        logger.debug(
            "Memory after preload: %.1f MB RSS (+%.1f MB)",
            mem_after_mb,
            mem_after_mb - mem_before_mb,
        )

    def _prepare_callables(self, grouped_patterns: Mapping[Any, Sequence[Any]]) -> None:
        prepared_group_keys: set[str] = set()
        for component_value in grouped_patterns:
            compiled_group = self.plan.compiled_function_pattern.group_for_component(
                component_value
            )
            if compiled_group is None or compiled_group.group_key in prepared_group_keys:
                continue
            prepare_compiled_function_group(compiled_group)
            prepared_group_keys.add(compiled_group.group_key)

    def _execute_pattern_groups(
        self,
        grouped_patterns: Mapping[Any, Sequence[Any]],
        total_groups: int,
    ) -> None:
        completed_groups = 0
        for component_index, (component_value, current_pattern_list) in enumerate(
            grouped_patterns.items()
        ):
            compiled_group = self.plan.compiled_function_pattern.group_for_component(
                component_value
            )
            if compiled_group is None:
                raise ValueError(
                    f"No compiled function group for component {component_value!r}."
                )

            for pattern_item in current_pattern_list:
                _process_single_pattern_group(
                    PatternGroupExecutionRequest(
                        context=self.context,
                        execution_plan=self.plan,
                        pattern_group_info=pattern_item,
                        compiled_group=compiled_group,
                        component_value=component_value,
                        component_index=component_index,
                    )
                )
                completed_groups += 1
                self._emit_pattern_progress(
                    completed_groups,
                    total_groups,
                    component_value,
                    pattern_item,
                )

    def _emit_pattern_progress(
        self,
        completed_groups: int,
        total_groups: int,
        component_value: Any,
        pattern_item: Any,
    ) -> None:
        emit(
            execution_id=self.context.execution_id,
            plate_id=self.context.plate_id,
            axis_id=self.plan.axis_id,
            step_name=self.plan.step_name,
            phase=ProgressPhase.PATTERN_GROUP,
            status=ProgressStatus.RUNNING,
            completed=completed_groups,
            total=total_groups,
            percent=(completed_groups / total_groups) * 100.0,
            component=str(component_value),
            pattern=str(pattern_item),
            worker_slot=self.context.worker_slot,
            owned_wells=self.context.owned_wells,
        )
