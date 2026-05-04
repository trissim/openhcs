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

import numpy as np

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan, StepResult
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
)
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import (
    convert_memory,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactLocation,
    RuntimeArtifactQuery,
    require_runtime_value_store,
    replace_runtime_artifact_payload,
)
from openhcs.core.runtime_adapters import RuntimeAdapterRequest, RuntimeAdapterSpec
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    SourceBindingOrigin,
    SourceBindingRuntimeContext,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
    image_payload_mask,
    image_payload_with_context,
    normalize_artifact_value,
    with_image_payload_data,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan

logger = logging.getLogger(__name__)

PROCESSING_CONTEXT_OWNER_NAME = ProcessingContext.__name__


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
    runtime_adapter: RuntimeAdapterSpec | None = None
    source_binding_plan: CompiledSourceBindingPlan = CompiledSourceBindingPlan.empty()
    source_binding_context: SourceBindingRuntimeContext = (
        SourceBindingRuntimeContext.empty()
    )
    group_key: str | None = None


@dataclass(frozen=True)
class FunctionChainExecutionRequest:
    """Nominal request for a chain of callables over one image stack."""

    initial_data_stack: Any
    invocations: Sequence[CompiledFunctionInvocation]
    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    artifact_inputs: ArtifactInputPlans
    artifact_outputs: ArtifactOutputPlans
    source_binding_context: SourceBindingRuntimeContext = (
        SourceBindingRuntimeContext.empty()
    )


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
    source_binding_context: SourceBindingRuntimeContext


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

    location = RuntimeArtifactLocation(
        path=vfs_path,
        backend=Backend.MEMORY.value,
    )
    runtime_value_store = require_runtime_value_store(
        context,
        owner_name=PROCESSING_CONTEXT_OWNER_NAME,
    )
    runtime_value_store.replace(
        runtime_value,
        path=location.path,
        backend=location.backend,
    )
    replace_runtime_artifact_payload(
        context.filemanager,
        runtime_value.data,
        location,
    )


def _require_axis_id(context: ProcessingContext) -> str:
    axis_id = getattr(context, "axis_id", None)
    if not axis_id:
        raise RuntimeError(
            f"{PROCESSING_CONTEXT_OWNER_NAME}.axis_id is required for artifact values."
        )
    return str(axis_id)


def _load_artifact_input_value(
    context: ProcessingContext,
    input_plan: ArtifactInputPlan,
) -> Any:
    """Load an artifact input from VFS through its typed runtime store record."""
    store = require_runtime_value_store(
        context,
        owner_name=PROCESSING_CONTEXT_OWNER_NAME,
    )
    axis_id = _require_axis_id(context)
    query = _artifact_input_query(
        input_plan=input_plan,
        axis_id=axis_id,
    )
    try:
        record = store.resolve(
            query,
            purpose="planned artifact input",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc} Refusing direct VFS fallback because this indicates a lost "
            "typed runtime contract or an artifact not produced through the runtime."
        ) from exc
    return context.filemanager.load(record.path, record.backend)


def _artifact_input_query(
    *,
    input_plan: ArtifactInputPlan,
    axis_id: str,
) -> RuntimeArtifactQuery:
    if input_plan.path != "self":
        return RuntimeArtifactQuery.by_location(
            name=input_plan.name,
            kind=input_plan.kind,
            axis_id=axis_id,
            location=RuntimeArtifactLocation(
                path=input_plan.path,
                backend=Backend.MEMORY.value,
            ),
        )

    return RuntimeArtifactQuery.by_group(
        name=input_plan.name,
        kind=input_plan.kind,
        axis_id=axis_id,
        group_key=_single_input_group_key(input_plan),
    )


def _single_input_group_key(input_plan: ArtifactInputPlan) -> str | None:
    group_keys = input_plan.group_keys or (None,)
    if len(group_keys) == 1:
        return group_keys[0]
    return None


def _select_artifact_plan_for_component(
    plan_by_group: Optional[Mapping[Any, ArtifactOutputPlans | ArtifactInputPlans]],
    component_key: Optional[str],
    default_plan: ArtifactOutputPlans | ArtifactInputPlans,
) -> ArtifactOutputPlans | ArtifactInputPlans:
    """Select precompiled artifact I/O plan for a component."""
    if not plan_by_group:
        return default_plan

    global_plan = plan_by_group.get(None, {})
    if component_key in plan_by_group:
        return {
            **global_plan,
            **plan_by_group[component_key],
        }
    if global_plan:
        return global_plan
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


def _execute_function_core(request: FunctionExecutionRequest) -> Any:
    """Execute one callable and route declared artifact I/O."""
    func_callable = request.func_callable
    context = request.context
    artifact_outputs = request.artifact_outputs
    final_kwargs = dict(request.base_kwargs)

    adapter_manages_artifact_inputs = (
        request.runtime_adapter is not None
        and request.runtime_adapter.manages_artifact_inputs
    )

    if request.artifact_inputs and not adapter_manages_artifact_inputs:
        logger.info(
            f"Artifact inputs for {func_callable.__name__}: {request.artifact_inputs}"
        )
        for arg_name, input_plan in request.artifact_inputs.items():
            logger.info(
                f"Loading artifact input '{arg_name}' from path '{input_plan.path}' (memory backend)"
            )
            try:
                final_kwargs[arg_name] = _load_artifact_input_value(
                    context,
                    input_plan,
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

    if request.runtime_adapter is not None:
        adapter_parameter = request.runtime_adapter.parameter_name
        if adapter_parameter not in sig.parameters:
            raise TypeError(
                f"{func_callable.__name__} declares runtime adapter parameter "
                f"'{adapter_parameter}', but its signature does not accept it."
            )
        final_kwargs[adapter_parameter] = request.runtime_adapter.factory(
            RuntimeAdapterRequest(
                context=context,
                artifact_outputs=artifact_outputs,
                source_binding_plan=request.source_binding_plan,
                source_binding_context=request.source_binding_context,
                group_key=request.group_key,
            )
        )

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

        current_stack = _convert_main_flow_memory(
            current_stack,
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
                runtime_adapter=invocation.contract.runtime_adapter,
                source_binding_plan=plan.source_binding_plan,
                source_binding_context=request.source_binding_context,
                group_key=invocation.key.group_key,
            )
        )

        current_memory_type = invocation_output_type

    return current_stack


def _convert_main_flow_memory(
    payload: Any,
    *,
    source_type: str,
    target_type: str,
    gpu_id: int,
) -> Any:
    """Convert main-flow image pixels while preserving image context."""
    data = image_payload_data(payload)
    converted = convert_memory(
        data=data,
        source_type=source_type,
        target_type=target_type,
        gpu_id=gpu_id,
    )
    return with_image_payload_data(payload, converted)


def _stack_payload_context(raw_slices: Sequence[Any], stack: Any) -> Any:
    """Attach per-slice image context to a freshly loaded stack."""
    metadata = _stack_payload_metadata(raw_slices)
    mask = _stack_payload_mask(raw_slices)
    return image_payload_with_context(stack, mask=mask, metadata=metadata)


def _stack_payload_metadata(raw_slices: Sequence[Any]) -> ImagePayloadMetadata:
    slice_metadata = tuple(image_payload_metadata(slice_data) for slice_data in raw_slices)
    if not any(metadata.has_values for metadata in slice_metadata):
        return ImagePayloadMetadata()
    return ImagePayloadMetadata(
        channel_intensity_scales=tuple(
            metadata.intensity_scale_for_channel(0)
            for metadata in slice_metadata
        ),
        channel_source_dtypes=tuple(
            metadata.source_dtype
            for metadata in slice_metadata
        ),
        channel_source_paths=tuple(
            metadata.source_path
            for metadata in slice_metadata
        ),
    )


def _stack_payload_mask(raw_slices: Sequence[Any]) -> Any | None:
    masks = tuple(image_payload_mask(slice_data) for slice_data in raw_slices)
    if not any(mask is not None for mask in masks):
        return None
    data_slices = tuple(image_payload_data(slice_data) for slice_data in raw_slices)
    resolved_masks = tuple(
        np.ones(np.asarray(data_slice).shape[:2], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
        for data_slice, mask in zip(data_slices, masks)
    )
    return np.stack(resolved_masks)


def _unstack_payload_context(payload: Any, slices: Sequence[Any]) -> list[Any]:
    """Attach per-slice image context after unstacking a runtime stack."""
    mask = image_payload_mask(payload)
    metadata = image_payload_metadata(payload)
    if mask is None and not metadata.has_values:
        return list(slices)
    return [
        image_payload_with_context(
            data=slice_data,
            mask=None if mask is None else _payload_mask_slice(mask, index),
            metadata=metadata.for_channel(index),
        )
        for index, slice_data in enumerate(slices)
    ]


def _payload_mask_slice(mask: Any, index: int) -> Any:
    if not hasattr(mask, "ndim"):
        return mask
    if mask.ndim == 3:
        return mask[index]
    return mask


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
            processed_stack = self._execute_pattern(loaded)
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

        raw_slice_data = tuple(image_payload_data(slice_data) for slice_data in raw_slices)
        main_data_stack = ImageStackLayout.for_slices(raw_slice_data).stack(
            slices=raw_slice_data,
            memory_type=self.plan.input_memory_type,
            gpu_id=self.plan.device_id,
        )
        main_data_stack = _stack_payload_context(raw_slices, main_data_stack)

        return PatternGroupData(
            matching_files=matching_files,
            main_data_stack=main_data_stack,
            source_binding_context=self._source_binding_context(matching_files),
        )

    def _source_binding_context(
        self,
        matching_files: list[str],
    ) -> SourceBindingRuntimeContext:
        if self.plan.source_binding_plan.is_empty:
            return SourceBindingRuntimeContext.empty()

        source_backend = self.context.microscope_handler.get_primary_backend(
            self.context.input_dir,
            self.context.filemanager,
        )
        step_input_source_paths = (
            self._virtual_workspace_source_paths_by_virtual_path()
            if source_backend == Backend.VIRTUAL_WORKSPACE.value
            else {}
        )
        source_metadata_by_path = (
            self._virtual_workspace_source_metadata_by_path()
            if source_backend == Backend.VIRTUAL_WORKSPACE.value
            else {}
        )
        pipeline_input_files, pipeline_input_backend = (
            self._pipeline_start_source_universe(
                source_backend,
                step_input_source_paths=step_input_source_paths,
            )
        )
        return SourceBindingRuntimeContext(
            step_input_files=tuple(matching_files),
            step_input_dir=str(self.plan.input_dir),
            step_input_source_paths=step_input_source_paths,
            source_metadata_by_path=source_metadata_by_path,
            pipeline_input_files=pipeline_input_files,
            pipeline_input_backend=pipeline_input_backend,
        )

    def _pipeline_start_source_universe(
        self,
        source_backend: str,
        *,
        step_input_source_paths: Mapping[str, str],
    ) -> tuple[tuple[str, ...], str]:
        if not self._requires_full_pipeline_source_universe():
            return (
                tuple(self.plan.get_paths_for_axis(self.context.input_dir, source_backend)),
                source_backend,
            )

        if source_backend == Backend.VIRTUAL_WORKSPACE.value:
            return (
                self._virtual_workspace_real_source_files(step_input_source_paths),
                Backend.DISK.value,
            )

        universe_backend = (
            Backend.DISK.value
            if source_backend == Backend.VIRTUAL_WORKSPACE.value
            else source_backend
        )
        return (
            tuple(
                str(path)
                for path in self.context.filemanager.list_files(
                    str(self.context.input_dir),
                    universe_backend,
                    recursive=True,
                )
            ),
            universe_backend,
        )

    def _requires_full_pipeline_source_universe(self) -> bool:
        plan = self.plan.source_binding_plan
        if plan.metadata_rules:
            return True
        return any(
            binding.origin is SourceBindingOrigin.PIPELINE_START
            for bindings in plan.bindings_by_group.values()
            for binding in bindings
        )

    def _virtual_workspace_source_paths_by_virtual_path(self) -> Mapping[str, str]:
        from openhcs.microscopes.openhcs import FIELDS, OpenHCSMetadataHandler

        metadata_handler = OpenHCSMetadataHandler(self.context.filemanager)
        metadata = metadata_handler._load_metadata_dict(self.context.plate_path)
        subdirectories = metadata.get(FIELDS.SUBDIRECTORIES, {})
        workspace_source_paths = {
            virtual_relative: str(Path(self.context.plate_path) / real_relative)
            for subdirectory in subdirectories.values()
            for virtual_relative, real_relative in subdirectory.get(
                "workspace_mapping",
                {},
            ).items()
        }
        if not workspace_source_paths:
            raise RuntimeError(
                "virtual_workspace source binding resolution requires "
                "workspace_mapping entries in OpenHCS metadata."
            )
        return workspace_source_paths

    def _virtual_workspace_source_metadata_by_path(
        self,
    ) -> Mapping[str, Mapping[str, str]]:
        from openhcs.microscopes.openhcs import FIELDS, OpenHCSMetadataHandler

        metadata_handler = OpenHCSMetadataHandler(self.context.filemanager)
        metadata = metadata_handler._load_metadata_dict(self.context.plate_path)
        source_metadata_by_path: dict[str, Mapping[str, str]] = {}
        for subdirectory in metadata.get(FIELDS.SUBDIRECTORIES, {}).values():
            workspace_mapping = subdirectory.get("workspace_mapping", {})
            source_metadata = subdirectory.get(FIELDS.SOURCE_METADATA, {})
            if not isinstance(source_metadata, Mapping):
                raise RuntimeError(
                    "virtual_workspace source metadata must be a path-keyed mapping."
                )
            for virtual_relative, metadata_fields in source_metadata.items():
                if not isinstance(metadata_fields, Mapping):
                    raise RuntimeError(
                        "virtual_workspace source metadata values must be mappings."
                    )
                normalized_metadata = {
                    str(key): str(value)
                    for key, value in metadata_fields.items()
                }
                virtual_path = str(virtual_relative)
                source_metadata_by_path[virtual_path] = normalized_metadata
                real_relative = workspace_mapping.get(virtual_path)
                if real_relative is not None:
                    real_path = str(Path(self.context.plate_path) / real_relative)
                    source_metadata_by_path[real_path] = normalized_metadata
        return source_metadata_by_path

    def _virtual_workspace_real_source_files(
        self,
        step_input_source_paths: Mapping[str, str],
    ) -> tuple[str, ...]:
        from openhcs.microscopes.openhcs import OpenHCSMetadataHandler

        workspace_source_files = tuple(step_input_source_paths.values())
        if not workspace_source_files:
            raise RuntimeError(
                "virtual_workspace pipeline-start source resolution requires "
                "workspace_mapping entries in OpenHCS metadata."
            )
        source_files = dict.fromkeys(
            (
                *workspace_source_files,
                *self._physical_plate_source_files(
                    excluded_names=(OpenHCSMetadataHandler.METADATA_FILENAME,)
                ),
            )
        )
        return tuple(source_files)

    def _physical_plate_source_files(
        self,
        *,
        excluded_names: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        return tuple(
            str(path)
            for path in self.context.filemanager.list_files(
                str(self.context.plate_path),
                Backend.DISK.value,
                recursive=True,
            )
            if Path(path).name not in excluded_names
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

    def _execute_pattern(
        self,
        loaded: PatternGroupData,
    ) -> Any:
        request = self.request
        component_artifacts = self._component_artifact_plans()

        if not request.compiled_group.invocations:
            raise ValueError(
                f"Compiled function group {request.compiled_group.group_key} has no invocations."
            )

        return _execute_chain_core(
            FunctionChainExecutionRequest(
                initial_data_stack=loaded.main_data_stack,
                invocations=request.compiled_group.invocations,
                context=self.context,
                execution_plan=self.plan,
                artifact_inputs=component_artifacts.inputs,
                artifact_outputs=component_artifacts.outputs,
                source_binding_context=loaded.source_binding_context,
            )
        )

    def _validate_and_unstack(self, processed_stack: Any) -> list[Any]:
        processed_data = image_payload_data(processed_stack)
        try:
            layout = ImageStackLayout.for_stack(processed_data)
        except ValueError as exc:
            logger.error("Function output is not an OpenHCS image stack.")
            logger.error(f"Output type: {type(processed_stack)}")
            logger.error(
                f"Output shape: {getattr(processed_data, 'shape', 'no shape attr')}"
            )
            logger.error(
                f"Output exposes ndim: {hasattr(processed_data, 'ndim')}"
            )
            if hasattr(processed_data, "ndim"):
                logger.error(f"Output ndim: {processed_data.ndim}")
            raise ValueError(
                "Main processing must result in an image stack shaped "
                f"(N, H, W) or (N, H, W, C), got "
                f"{getattr(processed_data, 'shape', 'unknown')}"
            ) from exc

        output_slices = layout.unstack(
            array=processed_data,
            memory_type=self.plan.output_memory_type,
            gpu_id=self.plan.device_id,
        )
        return _unstack_payload_context(processed_stack, output_slices)

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
