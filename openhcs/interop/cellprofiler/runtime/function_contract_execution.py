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
from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.measurement_lookup_dialect import runtime_measurement_lookup_dialect
from openhcs.core.memory import detect_memory_type, stack_slices
from openhcs.core.runtime_output_matching import artifact_spec_participates_in_main_flow
from openhcs.core.pipeline.function_contracts import (
    Pure2DSliceBatchExecutor,
    RuntimeBatchExecutionDomain,
    RuntimePure2DSliceBatchRequest,
)
from openhcs.core.runtime_invocation import (
    RuntimeBatchInvocationRequest,
    RuntimeInvocationOptions,
    SliceIndexRuntimeParameter,
)
from openhcs.core.runtime_semantics import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    RuntimePlaneAxis,
)
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
    CellProfilerRuntimeValueSequence,
)
from openhcs.interop.cellprofiler.runtime.processing_contracts import (
    CellProfilerProcessingContractAuthority,
    Pure2DSliceCountPolicy,
    RuntimeShapeInspection,
)
from openhcs.interop.cellprofiler.runtime.pure2d_output_aggregation import (
    CellProfilerPure2DImagePlaneSemantics,
    CellProfilerPure2DOutputAggregator,
    _unstack_cellprofiler_image_slices,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    Pure2DSliceResultBatch,
    RuntimeCallablePolicy,
    RuntimeCallableView,
    RuntimeInvocationKwargPolicy,
)


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
                ArtifactSpec.output("<main>", ImageArtifactType)
                if self.main_output_replaces_runtime_flow
                else None
            )
        first_spec = self.declared_output_specs[0]
        if artifact_spec_participates_in_main_flow(first_spec):
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
        if spec is not None and spec.artifact_type is ObjectLabelsArtifactType:
            if all(isinstance(output, ObjectLabelValue) for output in slice_outputs):
                return CellProfilerPure2DOutputAggregator.aggregate(
                    slice_outputs,
                    memory_type,
                    plane_axis=plane_axis,
                )
            labels = stack_slices(
                [object_label_dense_array(output) for output in slice_outputs],
                memory_type,
                0,
            )
            return ObjectLabelPayload(
                labels=labels,
                plane_axis=plane_axis,
                domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
            )
        return CellProfilerPure2DOutputAggregator.aggregate(
            slice_outputs,
            memory_type,
            plane_axis=plane_axis,
        )

    def full_stack_output_spec(self, index: int) -> ArtifactSpec | None:
        """Return the declared output spec at a raw full-stack tuple position."""
        if index == 0:
            return self.main_output_spec or ArtifactSpec.output(
                "<main>",
                ImageArtifactType,
            )
        return self.auxiliary_spec(index - 1)

    def aggregate_full_stack_pure_2d_outputs(
        self,
        result: CellProfilerRuntimeValue,
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> CellProfilerRuntimeValue:
        """Restore declared PURE_2D output domains after one full-stack call."""
        if isinstance(result, tuple):
            return tuple(
                self.aggregate_full_stack_pure_2d_output_value(
                    value,
                    self.full_stack_output_spec(index),
                    memory_type,
                    plane_axis=plane_axis,
                )
                for index, value in enumerate(result)
            )
        return self.aggregate_full_stack_pure_2d_output_value(
            result,
            self.full_stack_output_spec(0),
            memory_type,
            plane_axis=plane_axis,
        )

    def aggregate_full_stack_pure_2d_output_value(
        self,
        value: CellProfilerRuntimeValue,
        spec: ArtifactSpec | None,
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> CellProfilerRuntimeValue:
        if spec is None or spec.artifact_type is not ObjectLabelsArtifactType:
            return value
        labels = object_label_dense_array(value)
        if isinstance(value, ObjectLabelValue):
            return value
        if labels.ndim == 2:
            labels = stack_slices([labels], memory_type, 0)
            return ObjectLabelPayload(
                labels=labels,
                plane_axis=plane_axis,
                domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
            )
        return ObjectLabelPayload(
            labels=labels,
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
        *,
        runtime_slice_sequence_parameter_names: frozenset[str] = frozenset(),
        measurement_table_parameter_names: frozenset[str] = frozenset(),
    ) -> None:
        self.output_aggregation_contract = output_aggregation_contract
        self.runtime_slice_sequence_parameter_names = (
            runtime_slice_sequence_parameter_names
        )
        self.measurement_table_parameter_names = measurement_table_parameter_names

    def execute(
        self,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        *,
        invocation_options: RuntimeInvocationOptions | None = None,
        force_full_stack: bool = False,
        execution_mode: ImagePayloadExecutionMode | None = None,
        output_aggregation_contract: CellProfilerFunctionOutputAggregationContract = (
            DEFAULT_CELLPROFILER_OUTPUT_AGGREGATION_CONTRACT
        ),
        runtime_slice_sequence_parameter_names: frozenset[str] = frozenset(),
        measurement_table_parameter_names: frozenset[str] = frozenset(),
    ) -> CellProfilerRuntimeValue:
        executor = self.with_output_aggregation_contract(
            output_aggregation_contract
        ).with_runtime_projection_parameters(
            runtime_slice_sequence_parameter_names=(
                runtime_slice_sequence_parameter_names
            ),
            measurement_table_parameter_names=measurement_table_parameter_names,
        )
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
                executor.raw_callable_kwargs(
                    func,
                    kwargs,
                    invocation_options,
                ),
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_executor_strategy_execute",
            time.perf_counter() - execute_started_at,
            function=function_name,
            mode=mode.value,
        )
        return result

    @staticmethod
    def raw_callable_kwargs(
        func: CellProfilerFunction,
        kwargs: CellProfilerKwargs,
        invocation_options: RuntimeInvocationOptions | None,
    ) -> CellProfilerKwargDict:
        """Return raw-call kwargs with typed invocation metadata when declared."""
        invocation_options_parameter = CallableContract.from_callable(
            func
        ).runtime_invocation_options_parameter
        if invocation_options_parameter is None or invocation_options is None:
            return dict(kwargs)
        return {
            **dict(kwargs),
            invocation_options_parameter: invocation_options,
        }

    def with_output_aggregation_contract(
        self,
        output_aggregation_contract: CellProfilerFunctionOutputAggregationContract,
    ) -> "CellProfilerFunctionContractExecutor":
        if output_aggregation_contract == self.output_aggregation_contract:
            return self
        return type(self)(
            output_aggregation_contract,
            runtime_slice_sequence_parameter_names=(
                self.runtime_slice_sequence_parameter_names
            ),
            measurement_table_parameter_names=self.measurement_table_parameter_names,
        )

    def with_runtime_projection_parameters(
        self,
        *,
        runtime_slice_sequence_parameter_names: frozenset[str],
        measurement_table_parameter_names: frozenset[str],
    ) -> "CellProfilerFunctionContractExecutor":
        if (
            runtime_slice_sequence_parameter_names
            == self.runtime_slice_sequence_parameter_names
            and measurement_table_parameter_names == self.measurement_table_parameter_names
        ):
            return self
        return type(self)(
            self.output_aggregation_contract,
            runtime_slice_sequence_parameter_names=(
                runtime_slice_sequence_parameter_names
            ),
            measurement_table_parameter_names=measurement_table_parameter_names,
        )

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
        contract = CallableContract.from_callable(func)
        function_name = contract.function_name
        projection_started_at = time.perf_counter()
        if contract.processing_contract is ProcessingContract.PURE_3D:
            projected_image = image
            projected_kwargs = dict(kwargs)
            _validate_pure_3d_kwargs_do_not_carry_runtime_slice_alignment(
                function_name,
                projected_kwargs,
            )
        else:
            projected_image = project_singleton_stack_image_domain(image)
            projected_kwargs = {
                key: (
                    value
                    if isinstance(value, ObjectLabelValue)
                    else project_singleton_stack_image_domain(value)
                )
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
        contract = CallableContract.from_callable(func)
        function_name = contract.function_name
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_full_stack_project_domains",
            0.0,
            function=function_name,
            image_shape=RuntimeShapeInspection(image_payload_data(image)).shape_tuple(),
            labels_shape=(
                RuntimeShapeInspection(kwargs["labels"]).shape_tuple()
                if "labels" in kwargs
                else None
            ),
        )
        call_started_at = time.perf_counter()
        result = _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
            func,
            (image,),
            kwargs,
        ).call()
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_full_stack_raw_call",
            time.perf_counter() - call_started_at,
            function=function_name,
        )
        if contract.processing_contract is not ProcessingContract.PURE_2D:
            return result
        plane_axis = (
            CellProfilerPure2DImagePlaneSemantics.from_image(image).plane_axis
            or RuntimePlaneAxis.RUNTIME_SLICE
        )
        return (
            self.output_aggregation_contract.aggregate_full_stack_pure_2d_outputs(
                result,
                detect_memory_type(image_payload_data(image)),
                plane_axis=plane_axis,
            )
        )

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
        processing_contract = CellProfilerProcessingContractAuthority.for_callable(func)
        if (
            processing_contract is ProcessingContract.PURE_3D
            and not _aligned_stack_slices_carry_source_binding_planes(image)
        ):
            function_name = CallableContract.from_callable(func).function_name
            raise ValueError(
                f"{function_name} has ProcessingContract.PURE_3D but received "
                "ALIGNED_MULTI_IMAGE_STACK input whose aligned slices do not "
                "carry source-binding plane semantics."
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
                or Pure2DSliceCountPolicy.slice_count_from_kwargs(
                    kwargs,
                    runtime_slice_sequence_parameter_names=(
                        self.runtime_slice_sequence_parameter_names
                    ),
                    measurement_table_parameter_names=(
                        self.measurement_table_parameter_names
                    ),
                )
            )
            if slice_count is None:
                return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.invocation(
                    func,
                    (image,),
                    kwargs,
                ).call()
            slices_2d = tuple(image for _ in range(slice_count))
        elif _is_cellprofiler_single_source_image_plane(image):
            slice_count = Pure2DSliceCountPolicy.slice_count_from_kwargs(
                kwargs,
                runtime_slice_sequence_parameter_names=(
                    self.runtime_slice_sequence_parameter_names
                ),
                measurement_table_parameter_names=self.measurement_table_parameter_names,
            )
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
            self.execute_pure_2d_slice,
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

    def execute_pure_2d_slice(
        self,
        func: CellProfilerFunction,
        slice_2d: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        slice_index: int,
        slice_count: int,
    ) -> CellProfilerRuntimeValue:
        """Execute one projected pure-2D slice."""
        sliced_kwargs = self.slice_pure_2d_kwargs(
            kwargs,
            slice_index,
            slice_count,
        )
        if _callable_declares_slice_index(func):
            sliced_kwargs = dict(sliced_kwargs)
            slice_index_name = SliceIndexRuntimeParameter.require_parameter_name()
            if slice_index_name not in sliced_kwargs:
                sliced_kwargs[slice_index_name] = slice_index
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

    def slice_pure_2d_kwargs(
        self,
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
            sequence_kwargs=self.runtime_slice_sequence_parameter_names,
        )


def _callable_declares_slice_index(func: CellProfilerFunction) -> bool:
    """Return whether this callable declares runtime-supplied slice_index."""
    return (
        SliceIndexRuntimeParameter
        in CallableContract.from_callable(func).runtime_bound_parameter_types
    )


def _is_cellprofiler_single_source_image_plane(image: CellProfilerRuntimeValue) -> bool:
    """Return whether PURE_2D should treat this payload as one image plane."""

    return CellProfilerPure2DImagePlaneSemantics.from_image(
        image
    ).is_single_source_plane()


def _trace_pure_2d_slice(
    trace_path: str,
    func: CellProfilerFunction,
    image: CellProfilerRuntimeValue,
    kwargs: CellProfilerKwargs,
    slice_index: int,
    slice_count: int,
) -> None:
    function_name = CallableContract.from_callable(func).function_name
    plane_semantics = CellProfilerPure2DImagePlaneSemantics.from_image(image)
    metadata = plane_semantics.metadata
    source_role = plane_semantics.source_role
    plane_axis = plane_semantics.plane_axis
    record: CellProfilerKwargDict = {
        "function": function_name,
        "slice_index": slice_index,
        "slice_count": slice_count,
        "image": Pure2DTraceArrayStats.from_value(image).record(),
        "source_role": (
            None if source_role is None else type(source_role).__name__
        ),
        "source_component_metadata": (
            None
            if metadata.source_component_metadata is None
            else dict(metadata.source_component_metadata)
        ),
        "source_provenance_plane_count": (
            metadata.source_image_provenance_planes.count
        ),
        "source_provenance_plane_metadata": (
            tuple(
                None if item is None else dict(item)
                for item in metadata.source_image_provenance_planes.component_metadata
            )
        ),
        "plane_axis": None if plane_axis is None else plane_axis.value,
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
        handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")


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


def _aligned_stack_slices_carry_source_binding_planes(
    image: AlignedImageStack,
) -> bool:
    """Return whether each aligned slice is itself a source-binding stack."""
    return all(
        SourceImagePlaneAxisPolicy.for_request(
            SourceImagePlaneAxisRequest(aligned_slice)
        ).axis()
        is RuntimePlaneAxis.SOURCE_BINDING
        for aligned_slice in image.slices
    )


def _validate_pure_3d_kwargs_do_not_carry_runtime_slice_alignment(
    function_name: str,
    kwargs: CellProfilerKwargs,
) -> None:
    aligned_names = tuple(
        name
        for name, value in kwargs.items()
        if isinstance(value, RuntimeSliceAlignedValues)
    )
    if not aligned_names:
        return
    raise ValueError(
        f"{function_name} has ProcessingContract.PURE_3D but received "
        f"runtime-slice-aligned kwargs {list(aligned_names)}. PURE_3D callables "
        "consume whole-stack values; bind object-label special inputs as dense "
        "stack arrays or use a plane-local processing contract."
    )


_CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR = CellProfilerFunctionContractExecutor()
