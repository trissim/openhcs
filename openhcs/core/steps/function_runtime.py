"""Runtime execution helpers for FunctionStep.

This module owns callable invocation, artifact routing, and pattern-group stack
execution. FunctionStep remains responsible for step-level orchestration.
"""

import inspect
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence
from weakref import WeakKeyDictionary

import numpy as np

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan, StepResult
from openhcs.core.callable_contract import prepare_processing_callable
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
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.source_image_semantics import apply_source_image_loading_semantics
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
from openhcs.core.runtime_semantics import RuntimePlaneProjection
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan

logger = logging.getLogger(__name__)
_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
PROCESSING_CONTEXT_OWNER_NAME = ProcessingContext.__name__
ArtifactInputPlans = Mapping[str, ArtifactInputPlan]
ArtifactOutputPlans = Mapping[str, ArtifactOutputPlan]
_CALLABLE_PARAMETER_NAMES: WeakKeyDictionary[Callable, frozenset[str]] = (
    WeakKeyDictionary()
)


def _runtime_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_runtime_profile(label: str, seconds: float, **fields: Any) -> None:
    if not _runtime_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


def _callable_parameter_names(func: Callable) -> frozenset[str]:
    """Return cached callable parameter names for runtime adapter injection."""
    names = _CALLABLE_PARAMETER_NAMES.get(func)
    if names is None:
        names = frozenset(inspect.signature(func).parameters)
        _CALLABLE_PARAMETER_NAMES[func] = names
    return names


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
    invocation_options: RuntimeInvocationOptions | None = None
    source_binding_plan: CompiledSourceBindingPlan = CompiledSourceBindingPlan.empty()
    source_binding_context: SourceBindingRuntimeContext = (
        SourceBindingRuntimeContext.empty()
    )
    group_key: str | None = None
    plane_projection: RuntimePlaneProjection = field(
        default_factory=RuntimePlaneProjection.stack
    )


@dataclass(frozen=True)
class FunctionChainExecutionRequest:
    """Nominal request for a chain of callables over one image stack."""

    initial_data_stack: Any
    invocations: Sequence[CompiledFunctionInvocation]
    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    artifact_inputs: ArtifactInputPlans
    artifact_outputs: ArtifactOutputPlans
    runtime_plane_index: int
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
    component_index: int


@dataclass(frozen=True)
class PatternGroupData:
    """Loaded image data for one pattern group."""

    matching_files: list[str]
    main_data_stack: Any
    source_slice_shapes: tuple[tuple[int, ...], ...]
    source_binding_context: SourceBindingRuntimeContext


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceSourceProjection:
    """Source-binding projection derived from OpenHCS virtual-workspace metadata."""

    source_paths_by_virtual_path: Mapping[str, str]
    source_metadata_by_path: Mapping[str, Mapping[str, str]]

    def source_path_for(
        self,
        *,
        virtual_path: str,
        full_virtual_path: str,
        fallback_path: str,
    ) -> str:
        """Return the physical source path represented by a virtual workspace path."""
        for key in (virtual_path, full_virtual_path):
            source_path = self.source_paths_by_virtual_path.get(str(key))
            if source_path is not None:
                return source_path
        return fallback_path

    def source_metadata_for(
        self,
        *,
        virtual_path: str,
        full_virtual_path: str,
    ) -> Mapping[str, str] | None:
        """Return source metadata represented by a virtual workspace path."""
        for key in (virtual_path, full_virtual_path):
            metadata = self.source_metadata_by_path.get(str(key))
            if metadata is not None:
                return metadata
        source_path = self.source_path_for(
            virtual_path=virtual_path,
            full_virtual_path=full_virtual_path,
            fallback_path=full_virtual_path,
        )
        return self.source_metadata_by_path.get(source_path)


@dataclass(slots=True)
class SourceBindingExecutionCache:
    """Process-local cache for source-binding metadata shared by step runtimes."""

    virtual_workspace_projections: dict[str, VirtualWorkspaceSourceProjection]
    physical_source_files: dict[tuple[str, tuple[str, ...]], tuple[str, ...]]

    @classmethod
    def empty(cls) -> "SourceBindingExecutionCache":
        return cls(
            virtual_workspace_projections={},
            physical_source_files={},
        )


_SOURCE_BINDING_EXECUTION_CACHES: WeakKeyDictionary[
    ProcessingContext,
    SourceBindingExecutionCache,
] = WeakKeyDictionary()


def _source_binding_execution_cache(
    context: ProcessingContext,
) -> SourceBindingExecutionCache:
    cache = _SOURCE_BINDING_EXECUTION_CACHES.get(context)
    if cache is None:
        cache = SourceBindingExecutionCache.empty()
        _SOURCE_BINDING_EXECUTION_CACHES[context] = cache
    return cache


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


def prepare_compiled_function_group(group: CompiledFunctionGroup) -> None:
    """Run optional preparation hooks for each callable in a compiled group."""
    for invocation in group.invocations:
        prepare_processing_callable(invocation.func)


def prepare_compiled_context_callables(compiled_contexts: Mapping[str, Any]) -> None:
    """Prepare every compiled callable visible in a set of execution contexts."""
    prepared_group_keys: set[tuple[str, int, str]] = set()
    prepared_invocation_count = 0
    for context_key, context in compiled_contexts.items():
        step_plans = getattr(context, "step_plans", None)
        if not step_plans:
            continue
        for step_plan in step_plans.values():
            compiled_pattern = getattr(step_plan, "compiled_function_pattern", None)
            if compiled_pattern is None:
                continue
            for group in compiled_pattern.groups:
                prepare_key = (
                    str(context_key),
                    int(step_plan.step_index),
                    group.group_key,
                )
                if prepare_key in prepared_group_keys:
                    continue
                prepare_compiled_function_group(group)
                prepared_invocation_count += len(group.invocations)
                prepared_group_keys.add(prepare_key)
    logger.info(
        "Prepared %d compiled callable invocations across %d groups.",
        prepared_invocation_count,
        len(prepared_group_keys),
    )


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
            load_started_at = time.perf_counter()
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
            _log_runtime_profile(
                "artifact_input_load",
                time.perf_counter() - load_started_at,
                function=func_callable.__name__,
                artifact=arg_name,
                kind=input_plan.kind.value,
            )

    parameter_names = _callable_parameter_names(func_callable)
    if "context" in parameter_names:
        final_kwargs["context"] = context
    if (
        request.invocation_options is not None
        and "runtime_invocation_options" in parameter_names
    ):
        final_kwargs["runtime_invocation_options"] = request.invocation_options

    if request.runtime_adapter is not None:
        adapter_parameter = request.runtime_adapter.parameter_name
        if adapter_parameter not in parameter_names:
            raise TypeError(
                f"{func_callable.__name__} declares runtime adapter parameter "
                f"'{adapter_parameter}', but its signature does not accept it."
            )
        adapter_started_at = time.perf_counter()
        final_kwargs[adapter_parameter] = request.runtime_adapter.factory(
            RuntimeAdapterRequest(
                context=context,
                artifact_inputs=request.artifact_inputs,
                artifact_outputs=artifact_outputs,
                source_binding_plan=request.source_binding_plan,
                source_binding_context=request.source_binding_context,
                group_key=request.group_key,
                plane_projection=request.plane_projection,
            )
        )
        _log_runtime_profile(
            "runtime_adapter_factory",
            time.perf_counter() - adapter_started_at,
            function=func_callable.__name__,
            adapter=adapter_parameter,
        )

    logger.info(f"Executing function: {func_callable.__name__}")
    call_started_at = time.perf_counter()
    raw_function_output = func_callable(request.main_data_arg, **final_kwargs)
    _log_runtime_profile(
        "function_call",
        time.perf_counter() - call_started_at,
        function=func_callable.__name__,
    )

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
                save_started_at = time.perf_counter()
                _save_artifact_value(
                    context,
                    output_plan,
                    raw_function_output.artifacts[output_key],
                )
                _log_runtime_profile(
                    "artifact_output_save",
                    time.perf_counter() - save_started_at,
                    function=func_callable.__name__,
                    artifact=output_key,
                    kind=output_plan.kind.value,
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
                    save_started_at = time.perf_counter()
                    _save_artifact_value(
                        context,
                        output_plan,
                        returned_artifact_values_tuple[i],
                    )
                    _log_runtime_profile(
                        "artifact_output_save",
                        time.perf_counter() - save_started_at,
                        function=func_callable.__name__,
                        artifact=output_key,
                        kind=output_plan.kind.value,
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

        invocation_started_at = time.perf_counter()
        current_stack = _execute_function_core(
            FunctionExecutionRequest(
                func_callable=actual_callable,
                main_data_arg=current_stack,
                base_kwargs=invocation.kwargs_dict,
                context=request.context,
                artifact_inputs=invocation.select_inputs(request.artifact_inputs),
                artifact_outputs=invocation.select_outputs(request.artifact_outputs),
                runtime_adapter=invocation.contract.runtime_adapter,
                invocation_options=invocation.invocation_options,
                source_binding_plan=plan.source_binding_plan,
                source_binding_context=request.source_binding_context,
                group_key=invocation.key.group_key,
                plane_projection=RuntimePlaneProjection.for_group_key(
                    invocation.key.group_key,
                    plane_index=(
                        request.runtime_plane_index
                        if invocation.key.group_key is not None
                        else None
                    ),
                ),
            )
        )
        _log_runtime_profile(
            "invocation_total",
            time.perf_counter() - invocation_started_at,
            function=getattr(actual_callable, "__name__", invocation.key.function_name),
            group=invocation.key.group_key,
            position=invocation.key.position,
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
        channel_unit_interval_intensity_scales=tuple(
            metadata.unit_interval_intensity_scale_for_channel(0)
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
            mask=(
                None
                if mask is None
                else _payload_mask_slice(mask, index, slice_count=len(slices))
            ),
            metadata=metadata.for_channel(index),
        )
        for index, slice_data in enumerate(slices)
    ]


def _payload_mask_slice(mask: Any, index: int, *, slice_count: int) -> Any:
    mask_array = np.asarray(mask)
    if mask_array.ndim >= 3 and mask_array.shape[0] == slice_count:
        return mask_array[index]
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
            load_started_at = time.perf_counter()
            loaded = self._load_input_stack()
            _log_runtime_profile(
                "pattern_load_stack",
                time.perf_counter() - load_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
            execute_started_at = time.perf_counter()
            processed_stack = self._execute_pattern(loaded)
            _log_runtime_profile(
                "pattern_execute_chain",
                time.perf_counter() - execute_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
            unstack_started_at = time.perf_counter()
            output_slices = self._validate_and_unstack(processed_stack, loaded)
            _log_runtime_profile(
                "pattern_validate_unstack",
                time.perf_counter() - unstack_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
            save_started_at = time.perf_counter()
            self._save_outputs(output_slices, loaded.matching_files)
            _log_runtime_profile(
                "pattern_save_outputs",
                time.perf_counter() - save_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
            cleanup_started_at = time.perf_counter()
            self._cleanup_collapsed_domains(output_slices, loaded.matching_files)
            _log_runtime_profile(
                "pattern_cleanup",
                time.perf_counter() - cleanup_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
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
        raw_slices = self._apply_source_image_loading_semantics(
            raw_slices,
            matching_files,
            full_file_paths,
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
            source_slice_shapes=tuple(
                tuple(slice_data.shape)
                for slice_data in raw_slice_data
            ),
            source_binding_context=self._source_binding_context(matching_files),
        )

    def _apply_source_image_loading_semantics(
        self,
        raw_slices: Sequence[Any],
        matching_files: Sequence[str],
        full_file_paths: Sequence[str],
    ) -> list[Any]:
        source_projection = self._source_schema_workspace_projection()
        if source_projection is None:
            return list(raw_slices)
        return [
            apply_source_image_loading_semantics(
                payload,
                source_metadata=source_projection.source_metadata_for(
                    virtual_path=virtual_path,
                    full_virtual_path=full_virtual_path,
                ),
                source_path=source_projection.source_path_for(
                    virtual_path=virtual_path,
                    full_virtual_path=full_virtual_path,
                    fallback_path=full_virtual_path,
                ),
                read_backend=Backend.DISK.value,
                filemanager=self.context.filemanager,
            )
            for payload, virtual_path, full_virtual_path in zip(
                raw_slices,
                matching_files,
                full_file_paths,
            )
        ]

    def _source_binding_context(
        self,
        matching_files: list[str],
    ) -> SourceBindingRuntimeContext:
        source_backend = self.context.microscope_handler.get_primary_backend(
            self.context.input_dir,
            self.context.filemanager,
        )
        source_projection = self._source_schema_workspace_projection()
        step_input_source_paths = (
            source_projection.source_paths_by_virtual_path
            if source_projection is not None
            else {}
        )
        source_metadata_by_path = (
            source_projection.source_metadata_by_path
            if source_projection is not None
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
        if any(
            invocation.contract.runtime_adapter is not None
            for invocation in self.plan.compiled_function_pattern.iter_invocations()
        ):
            return True
        plan = self.plan.source_binding_plan
        if plan.metadata_rules:
            return True
        return any(
            binding.origin is SourceBindingOrigin.PIPELINE_START
            for bindings in plan.bindings_by_group.values()
            for binding in bindings
        )

    def _virtual_workspace_source_paths_by_virtual_path(self) -> Mapping[str, str]:
        return self._virtual_workspace_source_projection().source_paths_by_virtual_path

    def _virtual_workspace_source_metadata_by_path(
        self,
    ) -> Mapping[str, Mapping[str, str]]:
        return self._virtual_workspace_source_projection().source_metadata_by_path

    def _source_schema_workspace_projection(
        self,
    ) -> VirtualWorkspaceSourceProjection | None:
        """Return source-schema metadata projection when OpenHCS metadata declares one."""

        metadata = self._openhcs_metadata_dict()
        if not self._declares_source_schema_workspace_projection(metadata):
            return None
        return self._virtual_workspace_source_projection_from_metadata(metadata)

    @staticmethod
    def _declares_source_schema_workspace_projection(
        metadata: Mapping[str, Any],
    ) -> bool:
        from openhcs.microscopes.openhcs import FIELDS

        return any(
            bool(subdirectory.get("workspace_mapping"))
            for subdirectory in metadata.get(FIELDS.SUBDIRECTORIES, {}).values()
            if isinstance(subdirectory, Mapping)
        )

    def _virtual_workspace_source_projection(self) -> VirtualWorkspaceSourceProjection:
        """Return cached virtual-workspace source-binding projection for this plate."""
        return self._virtual_workspace_source_projection_from_metadata(
            self._openhcs_metadata_dict()
        )

    def _virtual_workspace_source_projection_from_metadata(
        self,
        metadata: Mapping[str, Any],
    ) -> VirtualWorkspaceSourceProjection:
        """Return cached source-schema projection for this plate metadata."""
        plate_path = str(Path(self.context.plate_path))
        cache = _source_binding_execution_cache(self.context)
        projection = cache.virtual_workspace_projections.get(plate_path)
        if projection is not None:
            return projection

        from openhcs.microscopes.openhcs import FIELDS

        workspace_source_paths: dict[str, str] = {}
        source_metadata_by_path: dict[str, Mapping[str, str]] = {}
        source_metadata_by_real_path: dict[str, Mapping[str, str] | None] = {}
        for subdirectory in metadata.get(FIELDS.SUBDIRECTORIES, {}).values():
            workspace_mapping = subdirectory.get("workspace_mapping", {})
            for virtual_relative, real_relative in workspace_mapping.items():
                real_path = str(Path(self.context.plate_path) / real_relative)
                virtual_path = str(virtual_relative)
                workspace_source_paths[virtual_path] = real_path
                workspace_source_paths[
                    str(Path(self.context.plate_path) / virtual_path)
                ] = real_path

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
                normalized_metadata = MappingProxyType(
                    {
                        str(key): str(value)
                        for key, value in metadata_fields.items()
                    }
                )
                virtual_path = str(virtual_relative)
                source_metadata_by_path[virtual_path] = normalized_metadata
                real_relative = workspace_mapping.get(virtual_path)
                if real_relative is not None:
                    real_path = str(Path(self.context.plate_path) / real_relative)
                    existing_metadata = source_metadata_by_real_path.get(real_path)
                    if existing_metadata is None and real_path in source_metadata_by_real_path:
                        continue
                    if existing_metadata is None:
                        source_metadata_by_real_path[real_path] = normalized_metadata
                    elif dict(existing_metadata) != dict(normalized_metadata):
                        source_metadata_by_real_path[real_path] = None

        for real_path, metadata_fields in source_metadata_by_real_path.items():
            if metadata_fields is not None:
                source_metadata_by_path[real_path] = metadata_fields

        if not workspace_source_paths:
            raise RuntimeError(
                "virtual_workspace source binding resolution requires "
                "workspace_mapping entries in OpenHCS metadata."
            )

        projection = VirtualWorkspaceSourceProjection(
            source_paths_by_virtual_path=MappingProxyType(workspace_source_paths),
            source_metadata_by_path=MappingProxyType(source_metadata_by_path),
        )
        cache.virtual_workspace_projections[plate_path] = projection
        return projection

    def _openhcs_metadata_dict(self) -> Mapping[str, Any]:
        from openhcs.microscopes.openhcs import OpenHCSMetadataHandler

        metadata_handler = self.context.microscope_handler.metadata_handler
        if not isinstance(metadata_handler, OpenHCSMetadataHandler):
            metadata_handler = OpenHCSMetadataHandler(self.context.filemanager)
        return metadata_handler._load_metadata_dict(self.context.plate_path)

    def _virtual_workspace_real_source_files(
        self,
        step_input_source_paths: Mapping[str, str],
    ) -> tuple[str, ...]:
        from openhcs.microscopes.openhcs import OpenHCSMetadataHandler

        workspace_source_files = tuple(dict.fromkeys(step_input_source_paths.values()))
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
        plate_path = str(Path(self.context.plate_path))
        cache_key = (plate_path, tuple(sorted(str(name) for name in excluded_names)))
        cache = _source_binding_execution_cache(self.context)
        cached_files = cache.physical_source_files.get(cache_key)
        if cached_files is not None:
            return cached_files
        source_files = tuple(
            str(path)
            for path in self.context.filemanager.list_files(
                str(self.context.plate_path),
                Backend.DISK.value,
                recursive=True,
            )
            if Path(path).name not in excluded_names
        )
        cache.physical_source_files[cache_key] = source_files
        return source_files

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
                runtime_plane_index=request.component_index,
            )
        )

    def _validate_and_unstack(
        self,
        processed_stack: Any,
        loaded: PatternGroupData,
    ) -> list[Any]:
        processed_data = image_payload_data(processed_stack)
        try:
            output_slices = ImageStackLayout.unstack_result_for_source_slices(
                processed_data,
                source_slice_shapes=loaded.source_slice_shapes,
                memory_type=self.plan.output_memory_type,
                gpu_id=self.plan.device_id,
            )
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

    def _cleanup_collapsed_domains(
        self,
        output_slices: list[Any],
        matching_files: list[str],
    ) -> None:
        context = self.context
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs >= num_inputs:
            return

        for j in range(num_outputs, num_inputs):
            unused_filename = matching_files[j]
            for cleanup_dir in (self.plan.input_dir, self.plan.output_dir):
                unused_path = cleanup_dir / unused_filename
                if context.filemanager.exists(
                    str(unused_path),
                    Backend.MEMORY.value,
                ):
                    context.filemanager.delete(
                        str(unused_path),
                        Backend.MEMORY.value,
                    )
                    logger.debug(
                        "Deleted unused collapsed-domain file after reduced "
                        "output cardinality: %s",
                        unused_path,
                    )


def _process_single_pattern_group(request: PatternGroupExecutionRequest) -> None:
    """Process one image pattern group through its assigned callable pattern."""
    PatternGroupRuntime(request).run()
