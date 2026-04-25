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
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    FunctionInvocationPlan,
    function_invocation_key,
)
from openhcs.core.memory import convert_memory, stack_slices, unstack_slices
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan

logger = logging.getLogger(__name__)


ArtifactInputPlans = Mapping[str, ArtifactInputPlan]
ArtifactOutputPlans = Mapping[str, ArtifactOutputPlan]
FunctionInvocationPlans = Mapping[FunctionInvocationKey, FunctionInvocationPlan]


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
    func_chain: Sequence[Any]
    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    artifact_inputs: ArtifactInputPlans
    artifact_outputs: ArtifactOutputPlans
    invocation_group_key: str = DEFAULT_GROUP_KEY


@dataclass(frozen=True)
class ComponentArtifactPlans:
    """Artifact plans selected for one grouped component execution."""

    invocation_group_key: str
    inputs: ArtifactInputPlans
    outputs: ArtifactOutputPlans


@dataclass(frozen=True)
class PatternGroupExecutionRequest:
    """All runtime data needed to process one pattern group."""

    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    pattern_group_info: Any
    executable_func_or_chain: Any
    base_func_args: Mapping[str, Any]
    component_value: Any


@dataclass(frozen=True)
class PatternGroupData:
    """Loaded image data for one pattern group."""

    matching_files: list[str]
    main_data_stack: Any


def _select_artifact_outputs_for_invocation(
    invocation_plans: FunctionInvocationPlans,
    func: Callable,
    group_key: str,
    position: int,
    artifact_outputs: ArtifactOutputPlans,
) -> dict[str, ArtifactOutputPlan]:
    """Select artifact outputs owned by one function-pattern invocation."""
    execution_key = function_invocation_key(func, group_key, position)

    if execution_key in invocation_plans:
        return invocation_plans[execution_key].select_outputs(artifact_outputs)

    return {}


def _save_artifact_value(
    context: ProcessingContext,
    output_plan: ArtifactOutputPlan,
    value: Any,
) -> None:
    """Save one planned artifact value to the memory VFS."""
    vfs_path = output_plan.path

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
    context.filemanager.save(value, vfs_path, Backend.MEMORY.value)


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
) -> ComponentArtifactPlans:
    """Select artifact plans and invocation identity for one component."""
    invocation_group_key = (
        component_key
        if isinstance(plan.func, dict) and component_key is not None
        else DEFAULT_GROUP_KEY
    )

    return ComponentArtifactPlans(
        invocation_group_key=invocation_group_key,
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


def _strip_runtime_metadata_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Remove UI-only metadata before invoking processing callables."""
    return {
        key: value
        for key, value in kwargs.items()
        if key != "__pyqt_reactive_scope_token__"
    }


def _resolve_callable_and_kwargs(func_item: Any) -> tuple[Callable, dict[str, Any]]:
    """Resolve one runtime function-pattern item to a callable and kwargs."""
    from openhcs.core.pipeline.compiler import FunctionReference

    if isinstance(func_item, FunctionReference):
        return func_item.resolve(), {}

    if isinstance(func_item, tuple) and len(func_item) == 2:
        func_or_ref, kwargs = func_item
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"Function kwargs must be a mapping, got {type(kwargs)}")

        if isinstance(func_or_ref, FunctionReference):
            actual_callable = func_or_ref.resolve()
        elif callable(func_or_ref):
            actual_callable = func_or_ref
        else:
            raise TypeError(f"Invalid function in tuple: {func_or_ref}")

        return actual_callable, _strip_runtime_metadata_kwargs(kwargs)

    if callable(func_item):
        return func_item, {}

    raise TypeError(f"Invalid function-pattern item: {func_item}")


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
    """Execute a list-chain function pattern over one image stack."""
    plan = request.execution_plan
    current_stack = request.initial_data_stack
    current_memory_type = plan.input_memory_type

    for i, func_item in enumerate(request.func_chain):
        actual_callable, base_kwargs_for_item = _resolve_callable_and_kwargs(func_item)

        current_stack = convert_memory(
            data=current_stack,
            source_type=current_memory_type,
            target_type=actual_callable.input_memory_type,
            gpu_id=plan.device_id,
        )

        outputs_plan_for_this_call = _select_artifact_outputs_for_invocation(
            plan.function_invocation_plans,
            actual_callable,
            request.invocation_group_key,
            i,
            request.artifact_outputs,
        )

        current_stack = _execute_function_core(
            FunctionExecutionRequest(
                func_callable=actual_callable,
                main_data_arg=current_stack,
                base_kwargs=base_kwargs_for_item,
                context=request.context,
                artifact_inputs=request.artifact_inputs,
                artifact_outputs=outputs_plan_for_this_call,
            )
        )

        current_memory_type = actual_callable.output_memory_type

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
        )

        logger.debug(
            "Selected artifact outputs for component %s: %s",
            component_key,
            component_artifacts.outputs,
        )

        return component_artifacts

    def _resolve_executable_pattern(self) -> Any:
        request = self.request
        from openhcs.core.pipeline.compiler import FunctionReference

        executable_func_or_chain = request.executable_func_or_chain
        if isinstance(executable_func_or_chain, FunctionReference):
            executable_func_or_chain = executable_func_or_chain.resolve()
        elif (
            isinstance(executable_func_or_chain, tuple)
            and len(executable_func_or_chain) == 2
        ):
            func_or_ref, kwargs = executable_func_or_chain
            if isinstance(func_or_ref, FunctionReference):
                executable_func_or_chain = (func_or_ref.resolve(), kwargs)

        return executable_func_or_chain

    def _execute_pattern(self, main_data_stack: Any) -> Any:
        request = self.request
        final_base_kwargs = dict(request.base_func_args)
        component_artifacts = self._component_artifact_plans()
        executable_func_or_chain = self._resolve_executable_pattern()

        if isinstance(executable_func_or_chain, list):
            return _execute_chain_core(
                FunctionChainExecutionRequest(
                    initial_data_stack=main_data_stack,
                    func_chain=executable_func_or_chain,
                    context=self.context,
                    execution_plan=self.plan,
                    artifact_inputs=component_artifacts.inputs,
                    artifact_outputs=component_artifacts.outputs,
                    invocation_group_key=component_artifacts.invocation_group_key,
                )
            )

        elif callable(executable_func_or_chain) or (
            isinstance(executable_func_or_chain, tuple)
            and len(executable_func_or_chain) == 2
        ):
            actual_func, call_kwargs = _resolve_callable_and_kwargs(
                executable_func_or_chain
            )
            final_base_kwargs.update(call_kwargs)

            filtered_artifact_outputs = _select_artifact_outputs_for_invocation(
                self.plan.function_invocation_plans,
                actual_func,
                component_artifacts.invocation_group_key,
                0,
                component_artifacts.outputs,
            )

            return _execute_function_core(
                FunctionExecutionRequest(
                    func_callable=actual_func,
                    main_data_arg=main_data_stack,
                    base_kwargs=final_base_kwargs,
                    context=self.context,
                    artifact_inputs=component_artifacts.inputs,
                    artifact_outputs=filtered_artifact_outputs,
                )
            )
        else:
            raise TypeError(
                f"Invalid executable_func_or_chain: {type(executable_func_or_chain)}"
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
