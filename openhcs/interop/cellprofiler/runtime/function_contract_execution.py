"""CellProfiler runtime callable contract execution."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    ImagePayloadExecutionMode,
    aligned_image_stack_kwargs,
    pack_aligned_image_outputs,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.image_shapes import ArrayShape
from openhcs.core.measurement_lookup_dialect import runtime_measurement_lookup_dialect
from openhcs.core.memory import detect_memory_type, stack_slices
from openhcs.core.runtime_batch_contracts import (
    Pure2DSliceBatchExecutor,
    RuntimeBatchExecutionDomain,
    RuntimeBatchInvocationRequest,
    RuntimePure2DSliceBatchRequest,
    SliceIndexRuntimeParameter,
)
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_output_matching import split_runtime_output
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
    RuntimeSliceProjectionDeclarationError,
)
from openhcs.core.steps.function_runtime import (
    RuntimeCallableArgument,
    RuntimeCallableKwargs,
    RuntimeFunctionOutput,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    Pure2DAuxiliaryOutputAggregator,
    Pure2DInputSlicer,
    Pure2DSliceResultBatch,
    RuntimeCallablePolicy,
    RuntimeCallableView,
    RuntimeInvocationKwargPolicy,
)

_CELLPROFILER_RUNTIME_CALLABLE_POLICY = RuntimeCallablePolicy(
    callable_view=RuntimeCallableView.RAW,
    kwarg_policy=RuntimeInvocationKwargPolicy.SIGNATURE_FILTERED,
)


class CellProfilerFunctionContractExecutor:
    """Apply OpenHCS processing contracts after CellProfiler input resolution."""

    def __init__(
        self,
        plane_projection: RuntimePlaneAxisValueProjection | None = None,
    ) -> None:
        self.plane_projection = plane_projection

    def execute(
        self,
        callable_contract: CallableContract,
        func: Callable[..., RuntimeFunctionOutput],
        image: RuntimeCallableArgument,
        kwargs: RuntimeCallableKwargs,
        *,
        execution_mode: ImagePayloadExecutionMode,
        plane_projection: RuntimePlaneAxisValueProjection | None = None,
    ) -> RuntimeCallableArgument:
        if not isinstance(callable_contract, CallableContract):
            raise TypeError(
                "CellProfilerFunctionContractExecutor.execute requires a compiled "
                f"CallableContract, got {type(callable_contract).__name__}."
            )
        if not callable(func):
            raise TypeError(
                "CellProfilerFunctionContractExecutor.execute requires a resolved "
                f"raw callable, got {type(func).__name__}."
            )
        compiled_raw_func = callable_contract.resolve_canonical_raw_callable()
        if func is not compiled_raw_func:
            raise ValueError(
                "CellProfilerFunctionContractExecutor.execute received a raw "
                f"callable that does not match compiled contract "
                f"{callable_contract.module_name!r}/"
                f"{callable_contract.function_name!r}."
            )
        if not isinstance(execution_mode, ImagePayloadExecutionMode):
            raise TypeError(
                "CellProfilerFunctionContractExecutor.execute requires an exact "
                f"ImagePayloadExecutionMode, got {type(execution_mode).__name__}."
            )
        executor = type(self)(plane_projection=plane_projection)
        function_name = callable_contract.function_name
        processing_contract = callable_contract.require_processing_contract()
        mode = execution_mode
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_executor_mode_resolution",
            0.0,
            function=function_name,
            mode=mode.value,
        )
        execute_started_at = time.perf_counter()
        with runtime_measurement_lookup_dialect(
            CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT
        ):
            match mode, processing_contract:
                case (
                    ImagePayloadExecutionMode.NATURAL,
                    ProcessingContract.PURE_2D
                    | ProcessingContract.PURE_3D
                    | ProcessingContract.FLEXIBLE
                    | ProcessingContract.VOLUMETRIC_TO_SLICE,
                ):
                    result = processing_contract.execute(
                        executor,
                        func,
                        image,
                        callable_contract=callable_contract,
                        **dict(kwargs),
                    )
                case (
                    ImagePayloadExecutionMode.FULL_STACK,
                    ProcessingContract.PURE_2D
                    | ProcessingContract.PURE_3D
                    | ProcessingContract.FLEXIBLE
                    | ProcessingContract.VOLUMETRIC_TO_SLICE,
                ):
                    result = executor.execute_pure_3d(
                        func,
                        image,
                        callable_contract=callable_contract,
                        **dict(kwargs),
                    )
                case (
                    ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
                    ProcessingContract.PURE_2D
                    | ProcessingContract.PURE_3D
                    | ProcessingContract.FLEXIBLE
                    | ProcessingContract.VOLUMETRIC_TO_SLICE,
                ):
                    result = executor._execute_aligned_multi_image_stack(
                        callable_contract,
                        func,
                        image,
                        **dict(kwargs),
                    )
                case _:
                    raise ValueError(
                        f"CellProfiler module {callable_contract.module_name!r} "
                        f"callable {function_name!r} has unsupported execution "
                        f"combination {mode!r} and {processing_contract!r}."
                    )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_executor_execute",
            time.perf_counter() - execute_started_at,
            function=function_name,
            mode=mode.value,
        )
        return executor._contextualize_multi_canonical_output(
            callable_contract,
            result,
        )

    def _contextualize_multi_canonical_output(
        self,
        callable_contract: CallableContract,
        result: RuntimeCallableArgument,
    ) -> RuntimeCallableArgument:
        """Attach the compiled canonical ABI to one exact returned image axis."""

        canonical_specs = callable_contract.canonical_return_output_specs.specs
        if len(canonical_specs) <= 1:
            return result
        canonical_output, trailing_outputs = split_runtime_output(result)
        if isinstance(canonical_output, AlignedImageStack):
            if canonical_output.slice_contexts:
                return result
            output_values = canonical_output.slices
        else:
            projection = self.plane_projection
            function_name = callable_contract.function_name
            if projection is None:
                raise RuntimeSliceProjectionDeclarationError(
                    f"{function_name} declares {len(canonical_specs)} canonical "
                    "outputs but returned a non-aligned payload without a "
                    "compiled plane projection."
                )
            if projection.plane_index is not None:
                raise RuntimeSliceProjectionDeclarationError(
                    f"{function_name} declares {len(canonical_specs)} canonical "
                    "outputs after the compiled plane projection already selected "
                    f"plane {projection.plane_index}."
                )
            if projection.axis_size != len(canonical_specs):
                raise ValueError(
                    f"{function_name} declares {len(canonical_specs)} canonical "
                    "outputs but its compiled plane projection declares "
                    f"{projection.axis_size} value(s)."
                )
            output_axis = image_payload_metadata(canonical_output).plane_axis
            if output_axis is not projection.axis:
                raise RuntimeSliceProjectionDeclarationError(
                    f"{function_name} declares {len(canonical_specs)} canonical "
                    "outputs but its returned payload does not declare the "
                    f"compiled {projection.axis.value!r} plane axis; got "
                    f"{output_axis!r}."
                )
            output_values = tuple(
                RuntimeSliceProjection.value_for_slice(
                    canonical_output,
                    projection.selected_plane(output_index),
                )
                for output_index in range(projection.axis_size)
            )
        if len(output_values) != len(canonical_specs):
            raise ValueError(
                f"{callable_contract.function_name} returned {len(output_values)} "
                "canonical output value(s) for "
                f"{len(canonical_specs)} compiled output spec(s)."
            )
        contextualized_output = pack_aligned_image_outputs(
            output_values,
            slice_contexts=tuple(
                AlignedImageSliceContext.main_flow(
                    output_key=spec.name,
                    artifact_kind=spec.artifact_type.value,
                )
                for spec in canonical_specs
            ),
        )
        return (
            (contextualized_output, *trailing_outputs)
            if trailing_outputs
            else contextualized_output
        )

    def execute_pure_2d_slice_batch(
        self,
        callable_contract: CallableContract,
        func: Callable[..., RuntimeFunctionOutput],
        slices_2d: tuple[RuntimeCallableArgument, ...],
        kwargs: RuntimeCallableKwargs,
        execute_slice: Callable[
            [
                Callable[..., RuntimeFunctionOutput],
                RuntimeCallableArgument,
                RuntimeCallableKwargs,
                int,
                int,
            ],
            RuntimeCallableArgument,
        ],
    ) -> tuple[list[RuntimeCallableArgument], float]:
        """Execute a pure-2D slice batch through the callable-owned batch contract."""
        slice_request = RuntimePure2DSliceBatchRequest(
            func=func,
            slices_2d=slices_2d,
            kwargs=kwargs,
            execute_slice=execute_slice,
        )
        if slice_request.slice_count <= 0:
            return [], 0.0

        declared_batch_executor = callable_contract.runtime_batch_executor(
            RuntimeBatchExecutionDomain.PURE_2D_SLICES
        )
        batch_executor = (
            declared_batch_executor
            if callable(declared_batch_executor)
            else Pure2DSliceBatchExecutor.default_executor()
        )
        slice_started_at = time.perf_counter()
        if batch_executor is not None and slice_request.slice_count > 1:
            slice_results = list(batch_executor(slice_request))
        else:
            slice_results = [
                slice_request.execute_one(slice_index)
                for slice_index in range(slice_request.slice_count)
            ]
        return slice_results, time.perf_counter() - slice_started_at

    def execute_pure_3d(
        self,
        func: Callable[..., RuntimeFunctionOutput],
        image: RuntimeCallableArgument,
        *,
        callable_contract: CallableContract,
        **kwargs: RuntimeCallableArgument,
    ) -> RuntimeCallableArgument:
        function_name = callable_contract.function_name
        projection_started_at = time.perf_counter()
        projected_image = image
        projected_kwargs = dict(kwargs)
        if callable_contract.processing_contract is ProcessingContract.PURE_3D:
            _validate_pure_3d_kwargs_do_not_carry_runtime_slice_alignment(
                callable_contract,
                projected_kwargs,
            )
        label_value = projected_kwargs.get("labels")
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_full_stack_project_domains",
            time.perf_counter() - projection_started_at,
            function=function_name,
            image_shape=ArrayShape.shape_for(image_payload_data(projected_image)),
            labels_shape=(
                ArrayShape.shape_for(label_value) if label_value is not None else None
            ),
        )
        call_started_at = time.perf_counter()
        result = _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
            func,
            (projected_image,),
            projected_kwargs,
        ).call()
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_full_stack_raw_call",
            time.perf_counter() - call_started_at,
            function=function_name,
        )
        return result

    def _execute_aligned_multi_image_stack(
        self,
        callable_contract: CallableContract,
        func: Callable[..., RuntimeFunctionOutput],
        image: RuntimeCallableArgument,
        **kwargs: RuntimeCallableArgument,
    ) -> RuntimeCallableArgument:
        if not isinstance(image, AlignedImageStack):
            raise TypeError(
                f"CellProfiler module {callable_contract.module_name!r} callable "
                f"{callable_contract.function_name!r} requires AlignedImageStack "
                "for ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK, got "
                f"{type(image).__name__}."
            )
        projection = self.plane_projection
        if projection is None:
            raise RuntimeSliceProjectionDeclarationError(
                f"CellProfiler module {callable_contract.module_name!r} callable "
                f"{callable_contract.function_name!r} requires a compiled "
                "runtime-slice projection for ALIGNED_MULTI_IMAGE_STACK execution."
            )
        if projection.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            raise RuntimeSliceProjectionDeclarationError(
                f"CellProfiler module {callable_contract.module_name!r} callable "
                f"{callable_contract.function_name!r} aligned image execution "
                "requires RuntimePlaneAxis.RUNTIME_SLICE, got "
                f"{projection.axis!r}."
            )
        if projection.plane_index is not None:
            raise RuntimeSliceProjectionDeclarationError(
                f"CellProfiler module {callable_contract.module_name!r} callable "
                f"{callable_contract.function_name!r} received an AlignedImageStack "
                "after the compiled runtime-slice projection already selected "
                f"plane {projection.plane_index}."
            )
        if projection.axis_size != len(image.slices):
            raise RuntimeSliceProjectionDeclarationError(
                f"CellProfiler module {callable_contract.module_name!r} callable "
                f"{callable_contract.function_name!r} aligned image cardinality "
                "conflicts with its compiled runtime-slice projection: "
                f"{len(image.slices)} != {projection.axis_size}."
            )
        if callable_contract.processing_contract is ProcessingContract.PURE_3D:
            slice_plane_axes = tuple(
                image_payload_metadata(slice_payload).plane_axis
                for slice_payload in image.slices
            )
            if any(
                plane_axis is not RuntimePlaneAxis.SOURCE_BINDING
                for plane_axis in slice_plane_axes
            ):
                raise ValueError(
                    f"CellProfiler module {callable_contract.module_name!r} "
                    f"callable {callable_contract.function_name!r} with "
                    "ProcessingContract.PURE_3D requires every aligned image "
                    "slice to declare RuntimePlaneAxis.SOURCE_BINDING; got "
                    f"{slice_plane_axes!r}."
                )

        def execute_aligned_stack_slice(
            slice_func: Callable[..., RuntimeFunctionOutput],
            slice_payload: RuntimeCallableArgument,
            slice_kwargs: RuntimeCallableKwargs,
            slice_index: int,
            slice_count: int,
        ) -> RuntimeCallableArgument:
            return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
                slice_func,
                (slice_payload,),
                aligned_image_stack_kwargs(
                    slice_kwargs,
                    slice_index,
                    slice_count,
                    reference_payload=slice_payload,
                ),
            ).call()

        slice_results, _slice_execute_seconds = self.execute_pure_2d_slice_batch(
            callable_contract,
            func,
            tuple(
                RuntimeSliceProjection.value_for_slice(
                    image,
                    projection.selected_plane(slice_index),
                )
                for slice_index in range(projection.axis_size)
            ),
            kwargs,
            execute_aligned_stack_slice,
        )
        result_batch = Pure2DSliceResultBatch.from_results(slice_results)
        canonical_specs = callable_contract.canonical_return_output_specs.specs
        trailing_specs = callable_contract.trailing_return_output_specs.specs
        if len(result_batch.auxiliary_groups) != len(trailing_specs):
            raise ValueError(
                f"{callable_contract.function_name} returned "
                f"{len(result_batch.auxiliary_groups)} trailing output position(s); "
                f"the compiled CallableContract declares {len(trailing_specs)}."
            )
        memory_type = detect_memory_type(image_payload_data(image.slices[0]))
        if canonical_specs:
            if all(
                isinstance(output, AlignedImageStack)
                for output in result_batch.main_outputs
            ):
                aligned_main_output = Pure2DAuxiliaryOutputAggregator.aggregate(
                    result_batch.main_outputs,
                    memory_type,
                    plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                )
                if not isinstance(aligned_main_output, AlignedImageStack):
                    raise TypeError(
                        f"{callable_contract.function_name} must preserve aligned "
                        "canonical image outputs as AlignedImageStack, got "
                        f"{type(aligned_main_output).__name__}."
                    )
                if len(aligned_main_output.slices) != len(canonical_specs):
                    raise ValueError(
                        f"{callable_contract.function_name} produced "
                        f"{len(aligned_main_output.slices)} aligned main-flow "
                        f"value(s) for {len(canonical_specs)} declared output "
                        "spec(s)."
                    )
                stacked_main_output = (
                    aligned_main_output.slices[0]
                    if len(canonical_specs) == 1
                    else aligned_main_output
                )
            elif len(canonical_specs) == 1:
                stacked_main_output = Pure2DAuxiliaryOutputAggregator.aggregate(
                    result_batch.main_outputs,
                    memory_type,
                    plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                )
            elif len(canonical_specs) == len(result_batch.main_outputs):
                stacked_main_output = AlignedImageStack(
                    tuple(result_batch.main_outputs)
                )
            else:
                raise ValueError(
                    f"{callable_contract.function_name} produced "
                    f"{len(result_batch.main_outputs)} aligned main-flow value(s) "
                    f"for {len(canonical_specs)} declared output spec(s)."
                )
            if len(canonical_specs) > 1:
                if not isinstance(stacked_main_output, AlignedImageStack):
                    raise TypeError(
                        f"{callable_contract.function_name} must aggregate multiple "
                        "declared canonical image outputs into AlignedImageStack, got "
                        f"{type(stacked_main_output).__name__}."
                    )
        else:
            stacked_main_output = RuntimeSliceAlignedValues(
                slices=tuple(result_batch.main_outputs)
            )
        if not result_batch.auxiliary_groups:
            return stacked_main_output
        return (
            stacked_main_output,
            *(
                Pure2DAuxiliaryOutputAggregator.aggregate(
                    values,
                    memory_type,
                    plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                )
                for values in result_batch.auxiliary_groups
            ),
        )

    def execute_pure_2d(
        self,
        func: Callable[..., RuntimeFunctionOutput],
        image: RuntimeCallableArgument,
        *,
        callable_contract: CallableContract,
        **kwargs: RuntimeCallableArgument,
    ) -> RuntimeCallableArgument:
        function_name = callable_contract.function_name
        invocation_context = (
            f"CellProfiler module {callable_contract.module_name!r} callable "
            f"{function_name!r}"
        )
        image_data = image_payload_data(image)
        if not isinstance(image_data, np.ndarray):
            return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
                func,
                (image,),
                kwargs,
            ).call()

        prepare_started_at = time.perf_counter()
        memory_type = detect_memory_type(image_data)
        if self.plane_projection is None:
            declared_kwarg_slice_count = RuntimeSliceProjection.slice_count_from_values(
                kwargs.values()
            )
            if declared_kwarg_slice_count is not None:
                declared_kwarg_names = tuple(
                    name
                    for name, value in kwargs.items()
                    if RuntimeSliceProjection.slice_count_from_values((value,))
                    is not None
                )
                raise RuntimeSliceProjectionDeclarationError(
                    f"{invocation_context} with ProcessingContract.PURE_2D has "
                    "kwargs declaring a runtime-slice axis of size "
                    f"{declared_kwarg_slice_count} through {declared_kwarg_names!r}, "
                    "but the image invocation has no "
                    "declared plane projection. Kwargs cannot create image-axis "
                    "execution semantics."
                )
            return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
                func,
                (image,),
                kwargs,
            ).call()
        declared_plane_axis = image_payload_metadata(image).plane_axis
        if declared_plane_axis is not self.plane_projection.axis:
            raise RuntimeSliceProjectionDeclarationError(
                f"{invocation_context} with ProcessingContract.PURE_2D has an image "
                "payload plane axis that conflicts with the compiled "
                f"projection: {declared_plane_axis!r} != "
                f"{self.plane_projection.axis.value!r}."
            )
        slices_2d = Pure2DInputSlicer.strategy_for_value(image).slice_value(
            image,
            memory_type,
        )
        if len(slices_2d) != self.plane_projection.axis_size:
            raise RuntimeSliceProjectionDeclarationError(
                f"{invocation_context} with ProcessingContract.PURE_2D has an image "
                "payload slice count that conflicts with the compiled "
                f"projection: {len(slices_2d)} != "
                f"{self.plane_projection.axis_size}."
            )
        aggregation_plane_axis = self.plane_projection.axis
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_pure_2d_prepare_slices",
            time.perf_counter() - prepare_started_at,
            function=function_name,
            slices=len(slices_2d),
        )

        slice_count = len(slices_2d)
        slice_results, slice_execute_seconds = self.execute_pure_2d_slice_batch(
            callable_contract,
            func,
            tuple(slices_2d),
            kwargs,
            lambda slice_func, slice_payload, slice_kwargs, slice_index, slice_count: (
                self.execute_pure_2d_slice(
                    callable_contract,
                    slice_func,
                    slice_payload,
                    slice_kwargs,
                    slice_index,
                    slice_count,
                )
            ),
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_pure_2d_slice_execute",
            slice_execute_seconds,
            function=function_name,
            slices=slice_count,
        )
        aggregate_started_at = time.perf_counter()
        result_batch = Pure2DSliceResultBatch.from_results(slice_results)
        canonical_specs = callable_contract.canonical_return_output_specs.specs
        trailing_specs = callable_contract.trailing_return_output_specs.specs
        if len(result_batch.auxiliary_groups) != len(trailing_specs):
            raise ValueError(
                f"{callable_contract.function_name} returned "
                f"{len(result_batch.auxiliary_groups)} trailing output position(s); "
                f"the compiled CallableContract declares {len(trailing_specs)}."
            )
        stacked_main_output = (
            Pure2DAuxiliaryOutputAggregator.aggregate(
                result_batch.main_outputs,
                memory_type,
                plane_axis=aggregation_plane_axis,
            )
            if canonical_specs
            else RuntimeSliceAlignedValues(slices=tuple(result_batch.main_outputs))
        )
        if not result_batch.auxiliary_groups:
            result = stacked_main_output
        else:
            result = (
                stacked_main_output,
                *(
                    Pure2DAuxiliaryOutputAggregator.aggregate(
                        values,
                        memory_type,
                        plane_axis=aggregation_plane_axis,
                    )
                    for values in result_batch.auxiliary_groups
                ),
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_pure_2d_aggregate_outputs",
            time.perf_counter() - aggregate_started_at,
            function=function_name,
            auxiliary_groups=len(result_batch.auxiliary_groups),
        )
        return result

    def execute_volumetric_to_slice(
        self,
        func: Callable[..., RuntimeFunctionOutput],
        image: RuntimeCallableArgument,
        *,
        callable_contract: CallableContract,
        **kwargs: RuntimeCallableArgument,
    ) -> RuntimeCallableArgument:
        del callable_contract
        result_2d = _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
            func,
            (image,),
            kwargs,
        ).call()
        result_data = image_payload_data(result_2d)
        result_mask = image_payload_mask(result_2d)
        result_metadata = image_payload_metadata(result_2d)
        memory_type = detect_memory_type(result_data)
        stacked = stack_slices([result_data], memory_type, 0)
        return result_metadata.payload_with(stacked, mask=result_mask)

    def execute_pure_2d_slice(
        self,
        callable_contract: CallableContract,
        func: Callable[..., RuntimeFunctionOutput],
        slice_2d: RuntimeCallableArgument,
        kwargs: RuntimeCallableKwargs,
        slice_index: int,
        slice_count: int,
    ) -> RuntimeCallableArgument:
        """Execute one projected pure-2D slice."""
        sliced_kwargs = self.slice_pure_2d_kwargs(
            callable_contract,
            kwargs,
            slice_index,
            slice_count,
        )
        if _callable_declares_slice_index(callable_contract):
            sliced_kwargs = dict(sliced_kwargs)
            slice_index_name = SliceIndexRuntimeParameter.require_parameter_name()
            sliced_kwargs[slice_index_name] = slice_index
        return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
            func,
            (slice_2d,),
            sliced_kwargs,
        ).call()

    def slice_pure_2d_kwargs(
        self,
        callable_contract: CallableContract,
        kwargs: RuntimeCallableKwargs,
        slice_index: int,
        slice_count: int,
    ) -> dict[str, RuntimeCallableArgument]:
        """Project runtime kwargs to one PURE_2D slice invocation."""
        if self.plane_projection is None:
            raise AssertionError(
                f"CellProfiler module {callable_contract.module_name!r} callable "
                f"{callable_contract.function_name!r} PURE_2D stack execution has "
                "no plane projection."
            )
        if self.plane_projection.axis_size != slice_count:
            raise ValueError(
                f"CellProfiler module {callable_contract.module_name!r} callable "
                f"{callable_contract.function_name!r} PURE_2D slice batch "
                "cardinality conflicts with its declared "
                f"plane axis: {slice_count} != {self.plane_projection.axis_size}."
            )
        return RuntimeSliceProjection.kwargs_for_slice(
            kwargs,
            self.plane_projection.selected_plane(slice_index),
        )


def _callable_declares_slice_index(callable_contract: CallableContract) -> bool:
    """Return whether this callable declares runtime-supplied slice_index."""
    return SliceIndexRuntimeParameter in callable_contract.runtime_bound_parameter_types


def _execute_runtime_batch_invocation(
    callable_contract: CallableContract,
    func: Callable[..., RuntimeFunctionOutput],
    request: RuntimeBatchInvocationRequest,
) -> RuntimeCallableArgument:
    """Execute one invocation from a core runtime batch request."""
    return _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
        callable_contract,
        func,
        request.image,
        request.kwargs,
        execution_mode=request.execution_mode,
        plane_projection=request.plane_projection,
    )


def _validate_pure_3d_kwargs_do_not_carry_runtime_slice_alignment(
    callable_contract: CallableContract,
    kwargs: RuntimeCallableKwargs,
) -> None:
    aligned_names = tuple(
        name
        for name, value in kwargs.items()
        if isinstance(value, RuntimeSliceAlignedValues)
    )
    if not aligned_names:
        return
    raise ValueError(
        f"CellProfiler module {callable_contract.module_name!r} callable "
        f"{callable_contract.function_name!r} has ProcessingContract.PURE_3D but "
        "received "
        f"runtime-slice-aligned kwargs {list(aligned_names)}. PURE_3D callables "
        "consume whole-stack values; bind object-label special inputs as dense "
        "stack arrays or use a plane-local processing contract."
    )


_CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR = CellProfilerFunctionContractExecutor()
