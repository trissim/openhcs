"""Compiled-plan orchestration for FunctionStep."""

from __future__ import annotations

import logging
import os
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

import psutil

from openhcs.constants import MULTIPROCESSING_AXIS
from openhcs.constants.constants import (
    LOADABLE_IMAGE_EXTENSIONS,
    Backend,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import FunctionGroupKey
from openhcs.core.progress import ProgressPhase, ProgressStatus, emit
from openhcs.core.runtime_pattern_cache import RuntimePatternDiscoveryCacheKey
from openhcs.core.source_binding_selection import (
    SourceCandidatePath,
    SourceBoundAnchorPatternPolicy,
    SourcePatternResolutionContext,
)
from openhcs.core.source_workspace_projection import (
    VirtualWorkspaceSourceProjectionAuthority,
    VirtualWorkspaceSourceProjectionCache,
)
from openhcs.core.step_dependencies import StepInputDependencyKind
from openhcs.core.steps.function_io import (
    bulk_preload_step_images,
    generate_materialized_paths,
    save_materialized_data,
    update_metadata_for_zarr_conversion,
)
from openhcs.core.steps.function_outputs import finalize_function_step_outputs
from openhcs.core.steps.function_output_manifest import (
    NoStepOutputManifestMatch,
    StepOutputManifestStore,
    step_output_manifest,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from openhcs.core.steps.function_runtime import (
    PatternGroupExecutionRequest,
    _process_single_pattern_group,
)

if TYPE_CHECKING:
    from openhcs.microscopes.microscope_interfaces import FilenameParser


logger = logging.getLogger(__name__)
_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
_PROFILE_RUNTIME_PATH_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME_PATH"
RuntimeProfileFieldValue = str | int | float | bool | None
RuntimeProfileExtraFields = Mapping[str, RuntimeProfileFieldValue] | None
DiscoveredPatternCollection = (
    Sequence[SourceCandidatePath]
    | Mapping[FunctionGroupKey, Sequence[SourceCandidatePath]]
)
AnchorPatternSelector = Callable[
    [FunctionGroupKey, tuple[SourceCandidatePath, ...]],
    Sequence[SourceCandidatePath],
]


@dataclass(frozen=True, slots=True)
class RuntimeProfileSettings:
    """Environment-owned runtime profile output settings."""

    enabled: bool
    output_path: str | None

    @classmethod
    def from_environment(cls) -> "RuntimeProfileSettings":
        raw_enabled = os.environ.get(_PROFILE_RUNTIME_ENV)
        return cls(
            enabled=(
                raw_enabled is not None
                and raw_enabled.lower() in {"1", "true", "yes"}
            ),
            output_path=os.environ.get(_PROFILE_RUNTIME_PATH_ENV),
        )


@dataclass(frozen=True, slots=True)
class StepRuntimeProfileRecord:
    """One function-step runtime profile event."""

    label: str
    seconds: float
    fields: tuple[tuple[str, RuntimeProfileFieldValue], ...]

    @classmethod
    def from_step(
        cls,
        label: str,
        seconds: float,
        *,
        step_index: int,
        step_name: str | None,
        extra_fields: RuntimeProfileExtraFields = None,
    ) -> "StepRuntimeProfileRecord":
        fields: dict[str, RuntimeProfileFieldValue] = {
            "step": step_index,
            "step_name": step_name,
        }
        if extra_fields is not None:
            fields.update(extra_fields)
        return cls(
            label=label,
            seconds=seconds,
            fields=tuple(fields.items()),
        )

    def emit(self, settings: RuntimeProfileSettings | None = None) -> None:
        if settings is None:
            profile_settings = RuntimeProfileSettings.from_environment()
        else:
            profile_settings = settings
        if not profile_settings.enabled:
            return
        field_text = " ".join(
            f"{key}={value}" for key, value in self.fields
        )
        logger.info("RUNTIME_PROFILE %s %.6fs %s", self.label, self.seconds, field_text)
        if profile_settings.output_path is not None:
            with open(profile_settings.output_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"RUNTIME_PROFILE {self.label} {self.seconds:.6f}s {field_text}\n"
                )


def record_function_step_runtime_profile(
    plan: FunctionStepExecutionPlan,
    label: str,
    seconds: float,
    *,
    extra_fields: RuntimeProfileExtraFields = None,
) -> None:
    """Record one runtime profile event for a function-step plan."""
    StepRuntimeProfileRecord.from_step(
        label,
        seconds,
        step_index=plan.step_index,
        step_name=plan.step_name,
        extra_fields=extra_fields,
    ).emit()


@dataclass(frozen=True, slots=True)
class PatternGroups:
    """Execution anchors grouped by compiled function-pattern component."""

    groups: Mapping[FunctionGroupKey, tuple[SourceCandidatePath, ...]]

    @classmethod
    def from_prepared(
        cls,
        grouped_patterns: Mapping[FunctionGroupKey, Sequence[SourceCandidatePath]],
    ) -> "PatternGroups":
        return cls(
            {
                group_key: tuple(str(pattern) for pattern in pattern_list)
                for group_key, pattern_list in grouped_patterns.items()
            }
        )

    def items(
        self,
    ) -> Iterator[tuple[FunctionGroupKey, tuple[SourceCandidatePath, ...]]]:
        return iter(self.groups.items())

    def values(self) -> Iterator[tuple[SourceCandidatePath, ...]]:
        return iter(self.groups.values())

    def __len__(self) -> int:
        return len(self.groups)

    def total_count(self) -> int:
        return sum(len(pattern_list) for pattern_list in self.groups.values())

    def map_groups(
        self,
        selector: AnchorPatternSelector,
    ) -> "PatternGroups":
        return PatternGroups.from_prepared(
            {
                group_key: selector(group_key, pattern_list)
                for group_key, pattern_list in self.groups.items()
            }
        )


def _single_execution_group_patterns(
    patterns: DiscoveredPatternCollection,
) -> Sequence[SourceCandidatePath]:
    """Flatten discovered source groups when the planner chose no execution axis."""
    if not isinstance(patterns, dict):
        return patterns
    return tuple(
        pattern
        for pattern_list in patterns.values()
        for pattern in pattern_list
    )


@dataclass(frozen=True, slots=True)
class StepAnchorPatternFilter:
    """Apply source, producer, and artifact-domain filtering to anchor groups."""

    plan: FunctionStepExecutionPlan
    parser: "FilenameParser"
    output_manifest: StepOutputManifestStore
    source_workspace_authority: VirtualWorkspaceSourceProjectionAuthority
    source_workspace_projection_cache: VirtualWorkspaceSourceProjectionCache

    @classmethod
    def from_context(
        cls,
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> "StepAnchorPatternFilter":
        return cls(
            plan=plan,
            parser=context.microscope_handler.parser,
            output_manifest=step_output_manifest(context),
            source_workspace_authority=(
                VirtualWorkspaceSourceProjectionAuthority.from_context(
                    context,
                    cache=context.runtime_source_workspace_projection_cache,
                )
            ),
            source_workspace_projection_cache=(
                context.runtime_source_workspace_projection_cache
            ),
        )

    def filtered(self, grouped_patterns: PatternGroups) -> PatternGroups:
        grouped_patterns = self.source_bound_anchor_patterns(grouped_patterns)
        grouped_patterns = self.producer_anchor_patterns(grouped_patterns)
        return self.artifact_driven_anchor_patterns(grouped_patterns)

    def source_bound_anchor_patterns(
        self,
        grouped_patterns: PatternGroups,
    ) -> PatternGroups:
        """Restrict source-bound step anchors to compatible declared sources."""

        if self.plan.main_input_dependency.kind is StepInputDependencyKind.STEP_OUTPUT:
            return grouped_patterns
        if not self.plan.source_binding_plan.has_primary_content:
            return grouped_patterns

        policy = SourceBoundAnchorPatternPolicy.for_plan(
            self.plan.source_binding_plan
        )
        source_context = self.source_pattern_context()

        def select_compatible(
            component_value: FunctionGroupKey,
            pattern_list: tuple[SourceCandidatePath, ...],
        ) -> Sequence[SourceCandidatePath]:
            compiled_group = self.plan.compiled_function_pattern.group_for_component(
                component_value
            )
            if compiled_group is None:
                return pattern_list
            bindings = self.plan.source_binding_plan.bindings_for_component_group(
                self.plan.execution_group_component,
                component_value,
            )
            return policy.select(
                pattern_list,
                bindings=bindings,
                source_context=source_context,
            )

        return self.apply(
            "step_filter_source_anchors",
            grouped_patterns,
            select_compatible,
        )

    def producer_anchor_patterns(
        self,
        grouped_patterns: PatternGroups,
    ) -> PatternGroups:
        """Restrict previous-step anchors to the declared producer's files."""

        def select_producer_paths(
            component_value: FunctionGroupKey,
            pattern_list: tuple[SourceCandidatePath, ...],
        ) -> Sequence[SourceCandidatePath]:
            del component_value
            try:
                return self.output_manifest.filter_to_producer_paths(
                    self.plan,
                    tuple(pattern_list),
                    self.parser,
                )
            except NoStepOutputManifestMatch:
                return ()

        return self.apply(
            "step_filter_producer_anchors",
            grouped_patterns,
            select_producer_paths,
        )

    def artifact_driven_anchor_patterns(
        self,
        grouped_patterns: PatternGroups,
    ) -> PatternGroups:
        """Keep one source anchor per artifact-managed invocation group."""

        if (
            self.plan.main_input_dependency.kind
            is not StepInputDependencyKind.STEP_OUTPUT
            and self.plan.source_binding_plan.has_primary_content
        ):
            return grouped_patterns

        def collapse_if_artifact_driven(
            component_value: FunctionGroupKey,
            pattern_list: tuple[SourceCandidatePath, ...],
        ) -> Sequence[SourceCandidatePath]:
            compiled_group = self.plan.compiled_function_pattern.group_for_component(
                component_value
            )
            if compiled_group is None:
                return pattern_list
            return compiled_group.runtime_domain.select_anchor_patterns(pattern_list)

        return self.apply(
            "step_collapse_artifact_anchors",
            grouped_patterns,
            collapse_if_artifact_driven,
        )

    def apply(
        self,
        label: str,
        grouped_patterns: PatternGroups,
        selector: AnchorPatternSelector,
    ) -> PatternGroups:
        filtered = grouped_patterns.map_groups(selector)
        before_count = grouped_patterns.total_count()
        after_count = filtered.total_count()
        if before_count != after_count:
            record_function_step_runtime_profile(
                self.plan,
                label,
                0.0,
                extra_fields={
                    "before": before_count,
                    "after": after_count,
                },
            )
        return filtered

    def source_pattern_context(self) -> SourcePatternResolutionContext:
        """Return source-path context used to filter source-bound anchors."""

        projection = self.source_workspace_authority.projection_or_empty()
        return SourcePatternResolutionContext.from_projection(
            parser=self.parser,
            projection=self.source_workspace_projection_cache.filtered_by_axis(
                projection,
                axis_id=self.plan.axis_id,
            ),
            metadata_rules=self.plan.source_binding_plan.metadata_rules,
        )


def _filter_patterns_by_component(
    patterns: DiscoveredPatternCollection,
    component: str,
    target_value: str,
    parser: "FilenameParser",
) -> DiscoveredPatternCollection:
    """Filter pattern strings by a fixed parsed component value."""

    def filter_pattern_list(
        pattern_list: Sequence[SourceCandidatePath],
    ) -> list[SourceCandidatePath]:
        filtered: list[SourceCandidatePath] = []
        for pattern in pattern_list:
            metadata = parser.parse_filename(str(pattern))
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


class FunctionStepExecutor:
    """Run one compiled FunctionStep plan for one multiprocessing axis."""

    def __init__(self, context: ProcessingContext, step_index: int) -> None:
        self.context = context
        self.plan = FunctionStepExecutionPlan.from_context(context, step_index)

    def record_runtime_profile(
        self,
        label: str,
        seconds: float,
        *,
        extra_fields: RuntimeProfileExtraFields = None,
    ) -> None:
        record_function_step_runtime_profile(
            self.plan,
            label,
            seconds,
            extra_fields=extra_fields,
        )

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
        step_output_manifest(self.context).begin_step(plan)

        phase_started_at = time.perf_counter()
        patterns_by_axis = self._detect_patterns()
        self.record_runtime_profile(
            "step_detect_patterns",
            time.perf_counter() - phase_started_at,
        )
        self._log_discovered_patterns(patterns_by_axis)
        phase_started_at = time.perf_counter()
        self._convert_input_if_needed()
        self.record_runtime_profile(
            "step_convert_input",
            time.perf_counter() - phase_started_at,
        )
        self._require_patterns(patterns_by_axis)
        self._apply_sequential_filter(patterns_by_axis)

        phase_started_at = time.perf_counter()
        grouped_patterns = self._prepare_groups(patterns_by_axis)
        self.record_runtime_profile(
            "step_prepare_groups",
            time.perf_counter() - phase_started_at,
        )
        phase_started_at = time.perf_counter()
        self._preload_inputs_if_needed(grouped_patterns)
        self.record_runtime_profile(
            "step_preload_inputs",
            time.perf_counter() - phase_started_at,
        )
        total_groups = grouped_patterns.total_count()
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

    def _detect_patterns(self) -> dict[str, DiscoveredPatternCollection]:
        plan = self.plan
        axis_name = MULTIPROCESSING_AXIS.value
        axis_filter = {f"{axis_name}_filter": [plan.axis_id]}
        source_projection = (
            VirtualWorkspaceSourceProjectionAuthority.from_context(
                self.context,
                cache=self.context.runtime_source_workspace_projection_cache,
            )
            .projection_if_available()
        )
        if (
            plan.main_input_dependency.kind is StepInputDependencyKind.PIPELINE_START
            and source_projection is not None
        ):
            source_files = source_projection.pipeline_start_files(axis_id=plan.axis_id)
            if source_files:
                cache_key = RuntimePatternDiscoveryCacheKey.from_source_files(
                    axis_id=plan.axis_id,
                    source_files=source_files,
                    group_by=plan.group_by_value,
                    variable_components=plan.variable_component_values,
                )
                cached_patterns = self.context.runtime_pattern_discovery_cache.get(
                    cache_key
                )
                if cached_patterns is not None:
                    return cached_patterns
                from openhcs.formats.pattern.pattern_discovery import (
                    PatternDiscoveryEngine,
                )

                patterns_by_axis = PatternDiscoveryEngine(
                    self.context.microscope_handler.parser,
                    self.context.filemanager,
                ).auto_detect_patterns_from_axis_files(
                    list(source_files),
                    axis_id=plan.axis_id,
                    variable_components=plan.variable_component_values,
                    group_by=plan.group_by,
                )
                self.context.runtime_pattern_discovery_cache.store(
                    cache_key,
                    patterns_by_axis,
                )
                return patterns_by_axis
        return self.context.microscope_handler.auto_detect_patterns(
            str(plan.input_dir),
            self.context.filemanager,
            plan.read_backend,
            extensions=LOADABLE_IMAGE_EXTENSIONS,
            group_by=plan.group_by,
            variable_components=plan.variable_component_values,
            **axis_filter,
        )

    def _log_discovered_patterns(
        self,
        patterns_by_axis: Mapping[str, DiscoveredPatternCollection],
    ) -> None:
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
        zarr_subdir = None
        if plan.input_conversion_uses_virtual_workspace:
            zarr_subdir = conversion_dir.name
        update_metadata_for_zarr_conversion(
            conversion_dir.parent,
            plan.input_conversion_original_subdir,
            zarr_subdir,
            self.context,
        )

    def _require_patterns(
        self,
        patterns_by_axis: Mapping[str, DiscoveredPatternCollection],
    ) -> None:
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

    def _apply_sequential_filter(
        self,
        patterns_by_axis: dict[str, DiscoveredPatternCollection],
    ) -> None:
        if not self.plan.sequential_filter_plan.enabled:
            return

        filtered_patterns = patterns_by_axis[self.plan.axis_id]
        for sequential_filter in self.plan.sequential_filter_plan.filters:
            filtered_patterns = _filter_patterns_by_component(
                filtered_patterns,
                sequential_filter.component_name,
                sequential_filter.value,
                self.context.microscope_handler.parser,
            )
        patterns_by_axis[self.plan.axis_id] = filtered_patterns

    def _prepare_groups(
        self,
        patterns_by_axis: Mapping[str, DiscoveredPatternCollection],
    ) -> PatternGroups:
        plan = self.plan
        axis_patterns = patterns_by_axis[plan.axis_id]
        execution_group_value = plan.execution_group_value
        if (
            execution_group_value is None
            and plan.compiled_function_pattern.is_grouped
        ):
            raise ValueError(
                f"Step '{plan.step_name}' uses a dict function pattern without "
                "a concrete execution group component. Dict keys are dispatch "
                "groups and require group_by to resolve to a real component; "
                "GroupBy.NONE is only valid for non-dict function patterns."
            )
        if (
            execution_group_value is None
            and not plan.compiled_function_pattern.is_grouped
        ):
            axis_patterns = _single_execution_group_patterns(axis_patterns)
        grouped_patterns = (
            plan.compiled_function_pattern.prepare_grouped_patterns(
                axis_patterns,
                default_component=execution_group_value,
            )
        )
        grouped_patterns = PatternGroups.from_prepared(grouped_patterns)
        grouped_patterns = StepAnchorPatternFilter.from_context(
            context=self.context,
            plan=self.plan,
        ).filtered(
            grouped_patterns,
        )
        if grouped_patterns.total_count() == 0:
            raise ValueError(
                f"No pattern groups found for step {plan.step_index} "
                f"({plan.step_name}) in well {plan.axis_id}"
            )
        return grouped_patterns

    def _preload_inputs_if_needed(
        self,
        grouped_patterns: PatternGroups,
    ) -> None:
        plan = self.plan
        if plan.read_backend == Backend.MEMORY.value:
            return

        process = psutil.Process(os.getpid())
        mem_before_mb = process.memory_info().rss / 1024 / 1024
        logger.debug("Memory before preload: %.1f MB RSS", mem_before_mb)

        if plan.sequential_filter_plan.enabled:
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

    def _execute_pattern_groups(
        self,
        grouped_patterns: PatternGroups,
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
                        component_count=len(grouped_patterns),
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
        component_value: FunctionGroupKey,
        pattern_item: SourceCandidatePath,
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
