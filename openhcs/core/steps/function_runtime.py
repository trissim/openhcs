"""Runtime execution helpers for FunctionStep.

This module owns callable invocation, artifact routing, and pattern-group stack
execution. FunctionStep remains responsible for step-level orchestration.
"""

import inspect
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan, StepResult
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
)
from openhcs.core.memory import convert_memory, stack_slices, unstack_slices
from openhcs.core.runtime_values import normalize_artifact_value
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan

logger = logging.getLogger(__name__)


ArtifactInputPlans = Mapping[str, ArtifactInputPlan]
ArtifactOutputPlans = Mapping[str, ArtifactOutputPlan]


@dataclass(frozen=True)
class FunctionExecutionRequest:
    """Nominal request for one callable invocation."""

    func_callable: Callable
    main_data_arg: Any
    base_kwargs: Mapping[str, Any]
    context: ProcessingContext
    artifact_inputs: ArtifactInputPlans
    artifact_outputs: ArtifactOutputPlans


@dataclass(frozen=True)
class FunctionChainExecutionRequest:
    """Nominal request for a chain of callables over one image stack."""

    initial_data_stack: Any
    invocations: Sequence[CompiledFunctionInvocation]
    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    artifact_inputs: ArtifactInputPlans
    artifact_outputs: ArtifactOutputPlans


@dataclass(frozen=True)
class ComponentArtifactPlans:
    """Artifact plans selected for one grouped component execution."""

    inputs: ArtifactInputPlans
    outputs: ArtifactOutputPlans


@dataclass(frozen=True)
class PatternGroupExecutionRequest:
    """All runtime data needed to process one pattern group."""

    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    pattern_group_info: Any
    compiled_group: CompiledFunctionGroup
    component_value: Any


@dataclass(frozen=True)
class PatternGroupData:
    """Loaded image data for one pattern group."""

    matching_files: list[str]
    main_data_stack: Any


def _save_artifact_value(
    context: ProcessingContext,
    output_plan: ArtifactOutputPlan,
    value: Any,
) -> None:
    """Validate and save one planned artifact value to the memory VFS."""
    vfs_path = output_plan.path
    axis_id = _require_axis_id(context)
    runtime_value = normalize_artifact_value(
        output_plan,
        value,
        axis_id=axis_id,
    )

    filemanager_memory_backend = context.filemanager._get_backend(
        Backend.MEMORY.value
    )
    filemanager_existing_keys = list(
        filemanager_memory_backend._memory_store.keys()
    )

    if vfs_path in filemanager_existing_keys:
        logger.warning(
            f"Artifact '{output_plan.name}' already exists in memory VFS at '{vfs_path}'."
        )

    parent_dir = str(Path(vfs_path).parent)
    context.filemanager.ensure_directory(parent_dir, Backend.MEMORY.value)
    context.filemanager.save(runtime_value.data, vfs_path, Backend.MEMORY.value)


def _require_axis_id(context: ProcessingContext) -> str:
    axis_id = getattr(context, "axis_id", None)
    if not axis_id:
        raise RuntimeError("ProcessingContext.axis_id is required for artifact values.")
    return str(axis_id)


def _select_artifact_plan_for_component(
    plan_by_group: Optional[Mapping[Any, ArtifactOutputPlans | ArtifactInputPlans]],
    component_key: Optional[str],
    default_plan: ArtifactOutputPlans | ArtifactInputPlans,
) -> ArtifactOutputPlans | ArtifactInputPlans:
    """Select precompiled artifact I/O plan for a component."""
    if not plan_by_group:
        return default_plan

    if component_key in plan_by_group:
        return plan_by_group[component_key]
    if None in plan_by_group:
        return plan_by_group[None]
    return default_plan


def _select_component_artifact_plans(
    plan: FunctionStepExecutionPlan,
    component_key: Optional[str],
    compiled_group: CompiledFunctionGroup,
) -> ComponentArtifactPlans:
    """Select artifact plans and invocation identity for one component."""
    return ComponentArtifactPlans(
        inputs=_select_artifact_plan_for_component(
            plan.artifact_inputs_by_group,
            component_key,
            plan.artifact_inputs,
        ),
        outputs=_select_artifact_plan_for_component(
            plan.artifact_outputs_by_group,
            component_key,
            plan.artifact_outputs,
        ),
    )


def _resolve_invocation_callable(invocation: CompiledFunctionInvocation) -> Callable:
    """Resolve one compiled invocation to the callable used in this worker."""
    from openhcs.core.pipeline.compiler import FunctionReference

    if isinstance(invocation.func, FunctionReference):
        return invocation.func.resolve()
    if callable(invocation.func):
        return invocation.func
    raise TypeError(f"Invalid compiled invocation function: {invocation.func}")


def _is_3d(array: Any) -> bool:
    """Check if an array is 3D."""
    return hasattr(array, "ndim") and array.ndim == 3


def _execute_function_core(request: FunctionExecutionRequest) -> Any:
    """Execute one callable and route declared artifact I/O."""
    func_callable = request.func_callable
    context = request.context
    artifact_outputs = request.artifact_outputs
    final_kwargs = dict(request.base_kwargs)

    if request.artifact_inputs:
        logger.info(
            f"Artifact inputs for {func_callable.__name__}: {request.artifact_inputs}"
        )
        for arg_name, input_plan in request.artifact_inputs.items():
            logger.info(
                f"Loading artifact input '{arg_name}' from path '{input_plan.path}' (memory backend)"
            )
            try:
                final_kwargs[arg_name] = context.filemanager.load(
                    input_plan.path, Backend.MEMORY.value
                )
            except Exception as e:
                logger.error(
                    f"Failed to load artifact input '{arg_name}' from '{input_plan.path}': {e}",
                    exc_info=True,
                )
                raise

    sig = inspect.signature(func_callable)
    if "context" in sig.parameters:
        final_kwargs["context"] = context

    logger.info(f"Executing function: {func_callable.__name__}")
    raw_function_output = func_callable(request.main_data_arg, **final_kwargs)

    if isinstance(raw_function_output, StepResult):
        main_output_data = raw_function_output.image
        if artifact_outputs:
            for output_key, output_plan in artifact_outputs.items():
                logger.info(
                    f"Saving artifact output '{output_key}' to VFS path '{output_plan.path}' (memory backend)"
                )
                if output_key not in raw_function_output.artifacts:
                    raise ValueError(
                        f"Function returned StepResult without planned artifact '{output_key}'."
                    )
                _save_artifact_value(
                    context,
                    output_plan,
                    raw_function_output.artifacts[output_key],
                )
    elif isinstance(raw_function_output, tuple):
        main_output_data = raw_function_output[0]
        returned_artifact_values_tuple = raw_function_output[1:]

        if artifact_outputs:
            for i, (output_key, output_plan) in enumerate(artifact_outputs.items()):
                logger.info(
                    f"Saving artifact output '{output_key}' to VFS path '{output_plan.path}' (memory backend)"
                )
                if i < len(returned_artifact_values_tuple):
                    _save_artifact_value(
                        context,
                        output_plan,
                        returned_artifact_values_tuple[i],
                    )
                else:
                    logger.error(
                        f"Artifact output plan wants to save '{output_key}', but function only returned {len(returned_artifact_values_tuple)} artifact values."
                    )
                    raise ValueError(
                        f"Function did not return enough values for all planned artifact outputs. Missing value for '{output_key}'."
                    )
    else:
        main_output_data = raw_function_output

    return main_output_data


def _execute_chain_core(request: FunctionChainExecutionRequest) -> Any:
    """Execute compiled invocations over one image stack."""
    plan = request.execution_plan
    current_stack = request.initial_data_stack
    current_memory_type = plan.input_memory_type

    for invocation in request.invocations:
        actual_callable = _resolve_invocation_callable(invocation)
        invocation_input_type = invocation.input_memory_type
        invocation_output_type = invocation.output_memory_type
        if invocation_input_type is None or invocation_output_type is None:
            raise ValueError(
                f"Compiled invocation {invocation.key} is missing memory types."
            )

        current_stack = convert_memory(
            data=current_stack,
            source_type=current_memory_type,
            target_type=invocation_input_type,
            gpu_id=plan.device_id,
        )

        current_stack = _execute_function_core(
            FunctionExecutionRequest(
                func_callable=actual_callable,
                main_data_arg=current_stack,
                base_kwargs=invocation.kwargs_dict,
                context=request.context,
                artifact_inputs=invocation.select_inputs(request.artifact_inputs),
                artifact_outputs=invocation.select_outputs(request.artifact_outputs),
            )
        )

        current_memory_type = invocation_output_type

    return current_stack


class PatternGroupRuntime:
    """Staged runtime for one pattern group."""

    def __init__(self, request: PatternGroupExecutionRequest) -> None:
        self.request = request
        self.pattern_repr = str(request.pattern_group_info)[:100]

    @property
    def context(self) -> ProcessingContext:
        return self.request.context

    @property
    def plan(self) -> FunctionStepExecutionPlan:
        return self.request.execution_plan

    def run(self) -> None:
        start_time = time.time()
        logger.debug(
            f"Processing pattern {self.pattern_repr} for axis {self.plan.axis_id}"
        )

        try:
            loaded = self._load_input_stack()
            processed_stack = self._execute_pattern(loaded.main_data_stack)
            output_slices = self._validate_and_unstack(processed_stack)
            self._save_outputs(output_slices, loaded.matching_files)
            self._cleanup_collapsed_inputs(output_slices, loaded.matching_files)
            logger.debug(
                f"Finished pattern group {self.pattern_repr} in {(time.time() - start_time):.2f}s."
            )
        except Exception as e:
            import traceback

            full_traceback = traceback.format_exc()
            logger.error(
                f"Error processing pattern group {self.pattern_repr}: {e}",
                exc_info=True,
            )
            logger.error(
                f"Full traceback for pattern group {self.pattern_repr}:\n{full_traceback}"
            )
            raise ValueError(
                f"Failed to process pattern group {self.pattern_repr}: {e}"
            ) from e

    def _load_input_stack(self) -> PatternGroupData:
        context = self.context
        request = self.request
        if not context.microscope_handler:
            raise RuntimeError("MicroscopeHandler not available in context.")

        matching_files = context.microscope_handler.path_list_from_pattern(
            str(self.plan.input_dir),
            request.pattern_group_info,
            context.filemanager,
            Backend.MEMORY.value,
            [vc.value for vc in self.plan.variable_components]
            if self.plan.variable_components
            else None,
        )

        if not matching_files:
            raise ValueError(
                f"No matching files found for pattern group {self.pattern_repr} in {self.plan.input_dir}. "
                f"This indicates either: (1) no image files exist in the directory, "
                f"(2) files don't match the pattern, or (3) pattern parsing failed. "
                f"Check that input files exist and match the expected naming convention."
            )

        logger.debug(
            f"Pattern {self.pattern_repr} matched {len(matching_files)} files: {[Path(f).name for f in matching_files]}"
        )

        matching_files.sort()
        logger.debug(
            f"Pattern {self.pattern_repr} sorted files: {[Path(f).name for f in matching_files]}"
        )

        full_file_paths = [str(self.plan.input_dir / f) for f in matching_files]
        raw_slices = context.filemanager.load_batch(
            full_file_paths,
            Backend.MEMORY.value,
        )

        if not raw_slices:
            raise ValueError(
                f"No valid images loaded for pattern group {self.pattern_repr} in {self.plan.input_dir}. "
                f"Found {len(matching_files)} matching files but failed to load any valid images. "
                f"This indicates corrupted image files, unsupported formats, or I/O errors. "
                f"Check file integrity and format compatibility."
            )

        main_data_stack = stack_slices(
            slices=raw_slices,
            memory_type=self.plan.input_memory_type,
            gpu_id=self.plan.device_id,
        )

        return PatternGroupData(
            matching_files=matching_files,
            main_data_stack=main_data_stack,
        )

    def _component_artifact_plans(self) -> ComponentArtifactPlans:
        request = self.request
        component_key = (
            None if request.component_value is None else str(request.component_value)
        )
        component_artifacts = _select_component_artifact_plans(
            self.plan,
            component_key,
            request.compiled_group,
        )

        logger.debug(
            "Selected artifact outputs for component %s: %s",
            component_key,
            component_artifacts.outputs,
        )

        return component_artifacts

    def _execute_pattern(self, main_data_stack: Any) -> Any:
        request = self.request
        component_artifacts = self._component_artifact_plans()

        if not request.compiled_group.invocations:
            raise ValueError(
                f"Compiled function group {request.compiled_group.group_key} has no invocations."
            )

        return _execute_chain_core(
            FunctionChainExecutionRequest(
                initial_data_stack=main_data_stack,
                invocations=request.compiled_group.invocations,
                context=self.context,
                execution_plan=self.plan,
                artifact_inputs=component_artifacts.inputs,
                artifact_outputs=component_artifacts.outputs,
            )
        )

    def _validate_and_unstack(self, processed_stack: Any) -> list[Any]:
        if not _is_3d(processed_stack):
            logger.error("Function output is not a 3D stack.")
            logger.error(f"Output type: {type(processed_stack)}")
            logger.error(
                f"Output shape: {getattr(processed_stack, 'shape', 'no shape attr')}"
            )
            logger.error(
                f"Output exposes ndim: {hasattr(processed_stack, 'ndim')}"
            )
            if hasattr(processed_stack, "ndim"):
                logger.error(f"Output ndim: {processed_stack.ndim}")
            raise ValueError(
                f"Main processing must result in a 3D array, got {getattr(processed_stack, 'shape', 'unknown')}"
            )

        return unstack_slices(
            array=processed_stack,
            memory_type=self.plan.output_memory_type,
            gpu_id=self.plan.device_id,
            validate_slices=True,
        )

    def _save_outputs(self, output_slices: list[Any], matching_files: list[str]) -> None:
        context = self.context
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs < num_inputs:
            logger.debug(
                f"Function returned {num_outputs} images from {num_inputs} inputs - likely flattening operation"
            )
        elif num_outputs > num_inputs:
            logger.warning(
                f"Function returned more images ({num_outputs}) than inputs ({num_inputs}) - unexpected"
            )

        output_data = []
        output_paths_batch = []

        for i, img_slice in enumerate(output_slices):
            if i >= len(matching_files):
                raise ValueError(
                    f"Function returned {num_outputs} output slices but only {num_inputs} input files available. "
                    f"Cannot generate filename for output slice {i}. This indicates a bug in the function or "
                    f"unstacking logic - functions should return same or fewer images than inputs."
                )

            input_filename = matching_files[i]
            output_filename = Path(input_filename).name
            output_path = self.plan.output_dir / output_filename

            if context.filemanager.exists(str(output_path), Backend.MEMORY.value):
                context.filemanager.delete(str(output_path), Backend.MEMORY.value)

            output_data.append(img_slice)
            output_paths_batch.append(str(output_path))

        context.filemanager.ensure_directory(
            str(self.plan.output_dir),
            Backend.MEMORY.value,
        )
        context.filemanager.save_batch(
            output_data,
            output_paths_batch,
            Backend.MEMORY.value,
        )

    def _cleanup_collapsed_inputs(
        self,
        output_slices: list[Any],
        matching_files: list[str],
    ) -> None:
        context = self.context
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs < num_inputs:
            for j in range(num_outputs, num_inputs):
                unused_input_filename = matching_files[j]
                unused_input_path = (
                    self.plan.input_dir / unused_input_filename
                )
                if context.filemanager.exists(
                    str(unused_input_path),
                    Backend.MEMORY.value,
                ):
                    context.filemanager.delete(
                        str(unused_input_path),
                        Backend.MEMORY.value,
                    )
                    logger.debug(
                        f"Deleted unused input file after collapsed output: {unused_input_filename}"
                    )


def _process_single_pattern_group(request: PatternGroupExecutionRequest) -> None:
    """Process one image pattern group through its assigned callable pattern."""
    PatternGroupRuntime(request).run()
