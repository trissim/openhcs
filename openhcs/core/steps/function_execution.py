"""Compiled-plan orchestration for FunctionStep."""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any, Mapping, Sequence

import psutil

from openhcs.constants import MULTIPROCESSING_AXIS
from openhcs.constants.constants import (
    LOADABLE_IMAGE_EXTENSIONS,
    Backend,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.progress import ProgressPhase, ProgressStatus, emit
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


def _runtime_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_step_profile(label: str, seconds: float, **fields: Any) -> None:
    if not _runtime_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


def _filter_patterns_by_component(
    patterns: list[Any] | dict[Any, list[Any]],
    component: str,
    target_value: str,
    microscope_handler: Any,
) -> list[Any] | dict[Any, list[Any]]:
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


class FunctionStepExecutor:
    """Run one compiled FunctionStep plan for one multiprocessing axis."""

    def __init__(self, context: ProcessingContext, step_index: int) -> None:
        self.context = context
        self.plan = FunctionStepExecutionPlan.from_context(context, step_index)

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
        if self._count_pattern_groups(grouped_patterns) == 0:
            raise ValueError(
                f"No pattern groups found for step {plan.step_index} "
                f"({plan.step_name}) in well {plan.axis_id}"
            )
        return grouped_patterns

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
        for component_value, current_pattern_list in grouped_patterns.items():
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
