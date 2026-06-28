"""CellProfiler runtime callable contract execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
import time

import numpy as np

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
    aligned_image_stack_kwargs,
    project_singleton_stack_image_domain,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.callable_contract import CallableContract
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.measurement_lookup_dialect import runtime_measurement_lookup_dialect
from openhcs.core.memory import detect_memory_type, stack_slices
from openhcs.core.runtime_output_matching import RuntimeOutputRole
from openhcs.core.pipeline.function_contracts import (
    Pure2DSliceBatchExecutor,
    RuntimeBatchExecutionDomain,
    RuntimePure2DSliceBatchRequest,
)
from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
from openhcs.core.runtime_semantics import RuntimePlaneAxis
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
    RuntimeProjectionAxis,
)
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    ObjectLabelValue,
    SourceImagePlaneAxisPolicy,
    SourceImagePlaneAxisRequest,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import _callable_parameters
from openhcs.interop.cellprofiler.runtime.image_execution_strategies import (
    CellProfilerImageExecutionStrategy,
)
from openhcs.interop.cellprofiler.runtime.image_payload_collapse import (
    SINGLETON_STACK_OUTPUT_COLLAPSE,
)
from openhcs.interop.cellprofiler.runtime.invocation import requested_image_execution_mode
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargDict,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.processing_contracts import (
    OBJECT_ROW_SEQUENCE_KWARGS,
    Pure2DSliceCountPolicy,
    RuntimeShapeInspection,
)
from openhcs.interop.cellprofiler.runtime.pure2d_output_aggregation import (
    CellProfilerPure2DOutputAggregator,
    _unstack_cellprofiler_image_slices,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    Pure2DSliceResultBatch,
    RuntimeCallablePolicy,
    RuntimeCallableView,
    RuntimeInvocationKwargPolicy,
)


_SLICE_INDEX_PARAMETER = "slice_index"
_CELLPROFILER_RUNTIME_CALLABLE_POLICY = RuntimeCallablePolicy(
    callable_view=RuntimeCallableView.RAW,
    kwarg_policy=RuntimeInvocationKwargPolicy.SIGNATURE_FILTERED,
)


@dataclass(frozen=True, slots=True)
class CellProfilerFunctionOutputAggregationContract:
    """Contract-derived aggregation behavior for one CellProfiler invocation."""

    main_output_replaces_runtime_flow: bool = True
    declared_output_specs: tuple[ArtifactSpec, ...] = ()

    @classmethod
    def from_main_flow_replacement(
        cls,
        replaces_main_flow: bool,
        declared_output_specs: tuple[ArtifactSpec, ...] = (),
    ) -> "CellProfilerFunctionOutputAggregationContract":
        return cls(
            main_output_replaces_runtime_flow=replaces_main_flow,
            declared_output_specs=declared_output_specs,
        )

    @property
    def main_output_spec(self) -> ArtifactSpec | None:
        if not self.declared_output_specs:
            return (
                ArtifactSpec("<main>", ArtifactKind.IMAGE)
                if self.main_output_replaces_runtime_flow
                else None
            )
        first_spec = self.declared_output_specs[0]
        if RuntimeOutputRole.for_spec(first_spec) is RuntimeOutputRole.MAIN_FLOW:
            return first_spec
        return None

    @property
    def auxiliary_output_specs(self) -> tuple[ArtifactSpec, ...]:
        if not self.declared_output_specs:
            return ()
        if self.main_output_spec == self.declared_output_specs[0]:
            return self.declared_output_specs[1:]
        return self.declared_output_specs

    def aggregate_main_outputs(
        self,
        slice_outputs: CellProfilerRuntimeValueSequence,
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> CellProfilerRuntimeValue:
        if self.main_output_spec is not None:
            return self.aggregate_declared_outputs(
                slice_outputs,
                memory_type,
                self.main_output_spec,
                plane_axis=plane_axis,
            )
        return RuntimeSliceAlignedValues(slices=tuple(slice_outputs))

    def aggregate_auxiliary_outputs(
        self,
        auxiliary_groups: tuple[list[CellProfilerRuntimeValue], ...],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> tuple[CellProfilerRuntimeValue, ...]:
        return tuple(
            self.aggregate_declared_outputs(
                values,
                memory_type,
                self.auxiliary_spec(index),
                plane_axis=plane_axis,
            )
            for index, values in enumerate(auxiliary_groups)
        )

    def auxiliary_spec(self, index: int) -> ArtifactSpec | None:
        specs = self.auxiliary_output_specs
        if index >= len(specs):
            return None
        return specs[index]

    def aggregate_declared_outputs(
        self,
        slice_outputs: CellProfilerRuntimeValueSequence,
        memory_type: str,
        spec: ArtifactSpec | None,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> CellProfilerRuntimeValue:
        return CellProfilerPure2DOutputAggregator.aggregate(
            slice_outputs,
            memory_type,
            plane_axis=plane_axis,
        )


DEFAULT_CELLPROFILER_OUTPUT_AGGREGATION_CONTRACT = (
    CellProfilerFunctionOutputAggregationContract()
)


class CellProfilerFunctionContractExecutor:
    """Apply OpenHCS processing contracts after CellProfiler input resolution."""

    def __init__(
        self,
        output_aggregation_contract: CellProfilerFunctionOutputAggregationContract = (
            DEFAULT_CELLPROFILER_OUTPUT_AGGREGATION_CONTRACT
        ),
    ) -> None:
        self.output_aggregation_contract = output_aggregation_contract

    def execute(
        self,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        *,
        force_full_stack: bool = False,
        execution_mode: ImagePayloadExecutionMode | None = None,
        output_aggregation_contract: CellProfilerFunctionOutputAggregationContract = (
            DEFAULT_CELLPROFILER_OUTPUT_AGGREGATION_CONTRACT
        ),
    ) -> CellProfilerRuntimeValue:
        executor = self.with_output_aggregation_contract(output_aggregation_contract)
        function_name = CallableContract.from_callable(func).function_name
        mode_started_at = time.perf_counter()
        mode = requested_image_execution_mode(
            force_full_stack=force_full_stack,
            execution_mode=execution_mode,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_executor_mode_resolution",
            time.perf_counter() - mode_started_at,
            function=function_name,
            mode=mode.value,
        )
        strategy_started_at = time.perf_counter()
        strategy = CellProfilerImageExecutionStrategy.for_mode(mode)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_executor_strategy_resolution",
            time.perf_counter() - strategy_started_at,
            function=function_name,
            mode=mode.value,
        )
        execute_started_at = time.perf_counter()
        with runtime_measurement_lookup_dialect(
            CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT
        ):
            result = strategy.execute(
                executor,
                func,
                image,
                kwargs,
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_executor_strategy_execute",
            time.perf_counter() - execute_started_at,
            function=function_name,
            mode=mode.value,
        )
        return result

    def with_output_aggregation_contract(
        self,
        output_aggregation_contract: CellProfilerFunctionOutputAggregationContract,
    ) -> "CellProfilerFunctionContractExecutor":
        if output_aggregation_contract == self.output_aggregation_contract:
            return self
        return type(self)(output_aggregation_contract)

    def execute_pure_2d_slice_batch(
        self,
        func: CellProfilerFunction,
        slices_2d: tuple[CellProfilerRuntimeValue, ...],
        kwargs: CellProfilerKwargs,
        execute_slice: Callable[
            [
                CellProfilerFunction,
                CellProfilerRuntimeValue,
                CellProfilerKwargs,
                int,
                int,
            ],
            CellProfilerRuntimeValue,
        ],
    ) -> tuple[list[CellProfilerRuntimeValue], float]:
        """Execute a pure-2D slice batch through the callable-owned batch contract."""
        slice_request = RuntimePure2DSliceBatchRequest(
            func=func,
            slices_2d=slices_2d,
            kwargs=kwargs,
            execute_slice=execute_slice,
        )
        if slice_request.slice_count <= 0:
            return [], 0.0

        declared_batch_executor = CallableContract.from_callable(
            func
        ).runtime_batch_executor(RuntimeBatchExecutionDomain.PURE_2D_SLICES)
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
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        function_name = CallableContract.from_callable(func).function_name
        projection_started_at = time.perf_counter()
        projected_image = project_singleton_stack_image_domain(image)
        projected_kwargs = {
            key: project_singleton_stack_image_domain(value)
            for key, value in kwargs.items()
        }
        label_value = projected_kwargs.get("labels")
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_full_stack_project_domains",
            time.perf_counter() - projection_started_at,
            function=function_name,
            image_shape=RuntimeShapeInspection(
                image_payload_data(projected_image)
            ).shape_tuple(),
            labels_shape=(
                RuntimeShapeInspection(label_value).shape_tuple()
                if label_value is not None
                else None
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

    def execute_full_stack(
        self,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        """Execute one full-stack invocation, preserving volumetric semantics."""
        return self.execute_pure_3d(func, image, **kwargs)

    def _execute_aligned_multi_image_stack(
        self,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        if not isinstance(image, AlignedImageStack):
            raise TypeError(
                "ALIGNED_MULTI_IMAGE_STACK execution requires "
                f"AlignedImageStack, got {type(image).__name__}."
            )
        def execute_aligned_stack_slice(
            slice_func: CellProfilerFunction,
            slice_payload: CellProfilerRuntimeValue,
            slice_kwargs: CellProfilerKwargs,
            slice_index: int,
            slice_count: int,
        ) -> CellProfilerRuntimeValue:
            return SINGLETON_STACK_OUTPUT_COLLAPSE.collapse(
                _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
                    slice_func,
                    (slice_payload,),
                    aligned_image_stack_kwargs(
                        slice_kwargs,
                        slice_index,
                        slice_count,
                        reference_payload=slice_payload,
                    ),
                ).call()
            )

        slice_results, _slice_execute_seconds = self.execute_pure_2d_slice_batch(
            func,
            tuple(image.slices),
            kwargs,
            execute_aligned_stack_slice,
        )
        result_batch = Pure2DSliceResultBatch.from_results(slice_results)
        memory_type = detect_memory_type(
            image_payload_data(result_batch.main_outputs[0])
        )
        stacked_main_output = self.output_aggregation_contract.aggregate_main_outputs(
            result_batch.main_outputs,
            memory_type,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )
        if not result_batch.auxiliary_groups:
            return stacked_main_output
        return (stacked_main_output, *self.output_aggregation_contract.aggregate_auxiliary_outputs(
            result_batch.auxiliary_groups,
            memory_type,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
        )

    def execute_pure_2d(
        self,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        function_name = CallableContract.from_callable(func).function_name
        image_data = image_payload_data(image)
        if not isinstance(image_data, np.ndarray):
            return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
                func,
                (image,),
                kwargs,
            ).call()

        prepare_started_at = time.perf_counter()
        memory_type = detect_memory_type(image_data)
        aggregation_plane_axis = (
            SourceImagePlaneAxisPolicy.for_request(
                SourceImagePlaneAxisRequest(image)
            ).axis()
            or RuntimePlaneAxis.RUNTIME_SLICE
        )
        if image_data.ndim == 2:
            slice_count = (
                RuntimeSliceProjection.first_axis_slice_count_from_values(
                    kwargs.values()
                )
                or Pure2DSliceCountPolicy.slice_count_from_kwargs(kwargs)
            )
            if slice_count is None:
                return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
                    func,
                    (image,),
                    kwargs,
                ).call()
            slices_2d = tuple(image for _ in range(slice_count))
        elif is_color_image_slice(image_data):
            slice_count = Pure2DSliceCountPolicy.slice_count_from_kwargs(kwargs)
            if slice_count is None:
                slice_count = 1
            slices_2d = tuple(image for _ in range(slice_count))
        else:
            slices_2d = _unstack_cellprofiler_image_slices(image, memory_type)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_pure_2d_prepare_slices",
            time.perf_counter() - prepare_started_at,
            function=function_name,
            slices=len(slices_2d),
        )

        slice_count = len(slices_2d)
        slice_results, slice_execute_seconds = self.execute_pure_2d_slice_batch(
            func,
            tuple(slices_2d),
            kwargs,
            _execute_pure_2d_slice,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_pure_2d_slice_execute",
            slice_execute_seconds,
            function=function_name,
            slices=slice_count,
        )
        aggregate_started_at = time.perf_counter()
        result_batch = Pure2DSliceResultBatch.from_results(slice_results)
        stacked_main_output = self.output_aggregation_contract.aggregate_main_outputs(
            result_batch.main_outputs,
            memory_type,
            plane_axis=aggregation_plane_axis,
        )
        if not result_batch.auxiliary_groups:
            result = stacked_main_output
        else:
            result = (
                stacked_main_output,
                *self.output_aggregation_contract.aggregate_auxiliary_outputs(
                    result_batch.auxiliary_groups,
                    memory_type,
                    plane_axis=aggregation_plane_axis,
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
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
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

    @staticmethod
    def slice_pure_2d_kwargs(
        kwargs: CellProfilerKwargs,
        slice_index: int,
        slice_count: int,
    ) -> CellProfilerKwargDict:
        """Project runtime kwargs to one PURE_2D slice invocation."""
        return RuntimeSliceProjection.kwargs_for_slice(
            kwargs,
            RuntimeProjectionAxis(
                slice_index=slice_index,
                extent=slice_count,
            ),
            sequence_kwargs=OBJECT_ROW_SEQUENCE_KWARGS,
        )

def _execute_pure_2d_slice(
    func: CellProfilerFunction,
    slice_2d: CellProfilerRuntimeValue,
    kwargs: CellProfilerKwargs,
    slice_index: int,
    slice_count: int,
) -> CellProfilerRuntimeValue:
    sliced_kwargs = CellProfilerFunctionContractExecutor.slice_pure_2d_kwargs(
        kwargs,
        slice_index,
        slice_count,
    )
    if _SLICE_INDEX_PARAMETER in _callable_parameters(func):
        sliced_kwargs = dict(sliced_kwargs)
        if _SLICE_INDEX_PARAMETER not in sliced_kwargs:
            sliced_kwargs[_SLICE_INDEX_PARAMETER] = slice_index
    trace_path = os.environ.get("OPENHCS_PURE2D_SLICE_TRACE_PATH")
    if trace_path:
        _trace_pure_2d_slice(
            trace_path,
            func,
            slice_2d,
            sliced_kwargs,
            slice_index,
            slice_count,
        )
    return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
        func,
        (slice_2d,),
        sliced_kwargs,
    ).call()


def _trace_pure_2d_slice(
    trace_path: str,
    func: CellProfilerFunction,
    image: CellProfilerRuntimeValue,
    kwargs: CellProfilerKwargs,
    slice_index: int,
    slice_count: int,
) -> None:
    function_name = CallableContract.from_callable(func).function_name
    if function_name not in {
        "filter_objects",
        "identify_objects_in_grid_with_guides",
    }:
        return
    record: CellProfilerKwargDict = {
        "function": function_name,
        "slice_index": slice_index,
        "slice_count": slice_count,
        "image": Pure2DTraceArrayStats.from_value(image).record(),
        "kwargs": {},
    }
    for name, value in kwargs.items():
        if isinstance(value, ObjectLabelValue):
            labels = object_label_dense_array(value)
            label_stats = Pure2DTraceLabelStats(labels)
            domain = value.domain
            record["kwargs"][name] = {
                "type": type(value).__name__,
                "shape": tuple(int(item) for item in labels.shape),
                "max_label": label_stats.max_label(),
                "positive_pixels": int(np.count_nonzero(labels)),
                "declared_object_count": domain.declared_object_count,
                "declared_object_ids": domain.declared_object_ids,
                "declared_object_id_domains": domain.declared_object_id_domains,
                "domain_scope": domain.scope.value,
                "plane_axis": value.plane_axis.value,
            }
        elif isinstance(value, tuple) and any(
            isinstance(item, ObjectLabelPayload) for item in value
        ):
            payload_items = []
            for item in value:
                if not isinstance(item, ObjectLabelPayload):
                    payload_items.append({"type": type(item).__name__})
                    continue
                labels = object_label_dense_array(item)
                label_stats = Pure2DTraceLabelStats(labels)
                domain = item.domain
                payload_items.append(
                    {
                        "type": type(item).__name__,
                        "shape": tuple(int(axis) for axis in labels.shape),
                        "max_label": label_stats.max_label(),
                        "positive_pixels": int(np.count_nonzero(labels)),
                        "declared_object_count": domain.declared_object_count,
                        "declared_object_ids": domain.declared_object_ids,
                        "declared_object_id_domains": domain.declared_object_id_domains,
                        "domain_scope": domain.scope.value,
                        "plane_axis": item.plane_axis.value,
                        }
                    )
            record["kwargs"][name] = payload_items
        elif isinstance(value, np.ndarray):
            record["kwargs"][name] = Pure2DTraceArrayStats.from_value(value).record()
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass(frozen=True, slots=True)
class Pure2DTraceLabelStats:
    """Diagnostic label statistics for pure-2D trace records."""

    labels: np.ndarray

    def max_label(self) -> int:
        if not self.labels.size:
            return 0
        return int(self.labels.max())


@dataclass(frozen=True, slots=True)
class Pure2DTraceArrayStats:
    """Diagnostic array statistics for pure-2D trace records."""

    array: np.ndarray
    value_type_name: str

    @classmethod
    def from_value(cls, value: CellProfilerRuntimeValue) -> "Pure2DTraceArrayStats":
        return cls(np.asarray(value), type(value).__name__)

    def min_value(self) -> float:
        if not self.array.size:
            return 0.0
        return float(self.array.min())

    def max_value(self) -> float:
        if not self.array.size:
            return 0.0
        return float(self.array.max())

    def positive_pixel_count(self) -> int:
        return int(np.count_nonzero(self.array > 0))

    def unique_positive_count(self) -> int:
        return int(np.unique(self.array[self.array > 0]).size)

    def record(self) -> CellProfilerKwargs:
        return {
            "type": self.value_type_name,
            "shape": tuple(int(axis) for axis in self.array.shape),
            "dtype": str(self.array.dtype),
            "min": self.min_value(),
            "max": self.max_value(),
            "positive_pixels": self.positive_pixel_count(),
            "unique_positive_count": self.unique_positive_count(),
        }


def _execute_runtime_batch_invocation(
    func: CellProfilerFunction,
    request: RuntimeBatchInvocationRequest,
) -> CellProfilerRuntimeValue:
    """Execute one invocation from a core runtime batch request."""
    return _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
        func,
        request.image,
        request.kwargs,
        execution_mode=request.execution_mode,
    )


_CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR = CellProfilerFunctionContractExecutor()
