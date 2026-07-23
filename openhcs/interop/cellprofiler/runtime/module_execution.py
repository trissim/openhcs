"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import (
    dataclass,
    replace,
)
from typing import cast

from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    ImageOutputBundle,
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
)
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import (
    CallableContract,
    ImagePayloadConsumption,
)
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    MeasurementRowOwnership,
    MeasurementSparseColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    object_label_input_execution_mode_from_callable,
)
from openhcs.core.runtime_adapters import (
    RuntimeAdapterRequest,
    RuntimeFunctionInvocationRequest,
)
from openhcs.core.runtime_artifact_queries import MeasurementTableUnion
from openhcs.core.runtime_batch_contracts import RuntimeBatchExecutionDomain
from openhcs.core.runtime_image_values import (
    image_payload_metadata,
    preserved_image_plane_projection,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.runtime_output_matching import (
    RuntimeMatchedOutput,
    RuntimeReturnedOutputMatcher,
)
from openhcs.core.runtime_profile import RuntimeProfileTimer
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.interop.cellprofiler.image_normalization import (
    normalize_cellprofiler_image_payload,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
    RuntimeArtifactTypeStrategy,
)
from openhcs.interop.cellprofiler.runtime.function_contract_execution import (
    _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerMeasurementImage,
    CellProfilerMeasurementImageDomain,
)
from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
    CellProfilerRuntimeProfiler,
    ObjectMeasurementOutputRecorder,
    ObjectMeasurementOutputTimings,
    PreparedObjectMeasurementInvocation,
    PreparedObjectMeasurementInvocationBatch,
    object_measurement_batch_group_key,
    object_measurement_runtime_inputs,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    measurement_table_for_module,
)
from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
    measurement_source_name_for_specs,
    single_source_name,
)
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.interop.cellprofiler.runtime.output_recording import (
    CellProfilerOutputRecorder,
)
from openhcs.core.steps.function_runtime import (
    RuntimeCallableArgument,
    RuntimeCallableKwargs,
    RuntimeFunctionOutput,
)
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    cellprofiler_profile_payload_fields,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileEvent,
    CellProfilerRuntimeProfileLogger,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)

logger = logging.getLogger(__name__)


def cellprofiler_runtime_adapter_factory(
    request: RuntimeAdapterRequest,
) -> CellProfilerRuntimeAdapter:
    """Build a CellProfiler adapter for one FunctionStep invocation."""
    return CellProfilerRuntimeAdapter(request=request)


def cellprofiler_runtime_callable_factory(
    registered_func: Callable[..., RuntimeFunctionOutput],
    callable_contract: CallableContract,
) -> "CellProfilerModuleExecutor":
    """Build the runtime adapter callable from one compiled contract."""

    del registered_func
    return CellProfilerModuleExecutor(
        callable_contract.resolve_canonical_raw_callable(),
        callable_contract,
    )


@dataclass(slots=True)
class CellProfilerModuleExecutor:
    """Execute one generated CellProfiler module against a typed runtime adapter."""

    raw_func: Callable[..., RuntimeFunctionOutput]
    callable_contract: CallableContract

    def __post_init__(self) -> None:
        if not callable(self.raw_func):
            raise TypeError(
                "CellProfilerModuleExecutor.raw_func must be callable, got "
                f"{type(self.raw_func).__name__}."
            )
        if not isinstance(self.callable_contract, CallableContract):
            raise TypeError(
                "CellProfilerModuleExecutor.callable_contract must be "
                f"CallableContract, got {type(self.callable_contract).__name__}."
            )
        self.callable_contract.require_processing_contract()
        runtime_adapter = self.callable_contract.runtime_adapter
        if (
            runtime_adapter is None
            or not runtime_adapter.manages_artifact_inputs
            or not runtime_adapter.manages_artifact_outputs
        ):
            raise TypeError(
                "CellProfilerModuleExecutor requires a RuntimeAdapterSpec that "
                "manages artifact inputs and outputs."
            )
        if self.raw_func is not self.callable_contract.resolve_canonical_raw_callable():
            raise ValueError(
                "Compiled CellProfiler callable identity "
                f"{self.callable_contract.function_name!r} does not match resolved "
                f"raw callable {self.raw_func.__name__!r}."
            )
        module_type = self.module_type()
        if self.raw_func is not module_type.require_callable(
            self.callable_contract.function_name
        ):
            raise ValueError(
                f"CellProfiler callable {self.callable_contract.function_name!r} "
                f"is not the declaration-owned callable for module "
                f"{module_type.require_module_name()!r}."
            )

    def module_type(self) -> type[CellProfilerModule]:
        """Return the nominal module declaration that owns this callable."""

        module_type = CellProfilerModule.for_function_name(
            self.callable_contract.function_name
        )
        if module_type is None:
            raise KeyError(
                "No CellProfiler module declaration owns callable "
                f"{self.callable_contract.function_name!r}."
            )
        return module_type

    def active_input_specs(
        self,
        adapter: CellProfilerRuntimeAdapter,
    ) -> ArtifactSpecCollection:
        """Return exact callable inputs and validate compiled producer edges."""

        request_contract = adapter.request.require_callable_contract()
        if request_contract != self.callable_contract:
            raise ValueError(
                "CellProfiler runtime adapter request does not carry this "
                "executor's compiled CallableContract."
            )
        return adapter.request.selected_artifact_input_specs()

    def active_output_plans(
        self,
        adapter: CellProfilerRuntimeAdapter,
    ) -> tuple[ArtifactOutputPlan, ...]:
        """Return the exact runtime output plans selected by the compiler."""

        return tuple(adapter.request.artifact_outputs.values())

    def measurement_output_plan(
        self,
        active_output_plans: tuple[ArtifactOutputPlan, ...],
    ) -> ArtifactOutputPlan:
        """Return the one selected measurement plan for a measurement execution."""

        measurement_plans = tuple(
            plan
            for plan in active_output_plans
            if plan.artifact_type is MeasurementsArtifactType
        )
        if len(measurement_plans) != 1:
            raise ValueError(
                f"Callable {self.callable_contract.function_name!r} measurement "
                "execution requires exactly one selected Measurements output plan, "
                f"got {tuple(plan.ref() for plan in measurement_plans)!r}."
            )
        return measurement_plans[0]

    def prepare(self) -> None:
        """Resolve nominal policies used by this executor before timed execution."""
        for strategy_type in RuntimeArtifactTypeStrategy.registered_strategy_types():
            RuntimeArtifactTypeStrategy.for_artifact_type(strategy_type.artifact_type)
        for artifact_type in frozenset(
            output.artifact_type for output in self.callable_contract.artifact_outputs
        ):
            CellProfilerOutputRecorder.for_artifact_type(artifact_type)

    def __call__(
        self,
        image: RuntimeCallableArgument,
        *,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        **kwargs: RuntimeCallableArgument,
    ) -> RuntimeCallableArgument:
        """Call the absorbed function and record declared outputs through the adapter."""
        module_type = self.module_type()
        module_name = module_type.require_module_name()
        run_profile = RuntimeProfileTimer.start()
        phase_profile = RuntimeProfileTimer.start()
        active_inputs = self.active_input_specs(cellprofiler_runtime)
        object_inputs = active_inputs.of_artifact_type(ObjectLabelsArtifactType)
        active_outputs = self.active_output_plans(cellprofiler_runtime)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_runs_per_image_check",
            phase_profile.elapsed(),
            module=module_name,
            function=self.callable_contract.function_name,
        )
        phase_profile = RuntimeProfileTimer.start()
        if module_type.executes_per_image_measurements(
            self.raw_func,
            object_inputs,
            callable_contract=self.callable_contract,
        ):
            result = self._run_per_image_measurement(
                image=image,
                adapter=cellprofiler_runtime,
                kwargs=kwargs,
                module_type=module_type,
                active_input_specs=active_inputs.specs,
                active_output_plans=active_outputs,
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_run_per_image_measurement",
                phase_profile.elapsed(),
                module=module_name,
                function=self.callable_contract.function_name,
            )
        elif module_type.executes_per_object_measurements(object_inputs):
            primary_image_inputs = self._primary_image_inputs(
                image,
                cellprofiler_runtime,
                module_type=module_type,
            )
            image_request = None
            if primary_image_inputs:
                image_request = self._image_request(
                    image,
                    cellprofiler_runtime,
                    module_type=module_type,
                    active_input_specs=active_inputs.specs,
                )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_image_request",
                phase_profile.elapsed(),
                module=module_name,
                function=self.callable_contract.function_name,
            )
            phase_profile = RuntimeProfileTimer.start()
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_runs_per_object_check",
                phase_profile.elapsed(),
                module=module_name,
                function=self.callable_contract.function_name,
            )
            phase_profile = RuntimeProfileTimer.start()
            result = self._run_per_object_measurement(
                image=image,
                adapter=cellprofiler_runtime,
                kwargs=kwargs,
                image_request=image_request,
                module_type=module_type,
                active_input_specs=active_inputs.specs,
                active_output_plans=active_outputs,
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_run_per_object_measurement",
                phase_profile.elapsed(),
                module=module_name,
                function=self.callable_contract.function_name,
            )
        else:
            result = self._run_standard_image(
                image=image,
                adapter=cellprofiler_runtime,
                kwargs=kwargs,
                module_type=module_type,
                active_input_specs=active_inputs.specs,
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_module_run_total",
            run_profile.elapsed(),
            module=module_name,
            function=self.callable_contract.function_name,
        )
        return result

    def _run_standard_image(
        self,
        *,
        image: RuntimeCallableArgument,
        adapter: CellProfilerRuntimeAdapter,
        kwargs: RuntimeCallableKwargs,
        module_type: type[CellProfilerModule],
        active_input_specs: tuple[ArtifactSpec, ...],
    ) -> RuntimeCallableArgument:
        module_name = module_type.require_module_name()
        phase_profile = RuntimeProfileTimer.start()
        image_request = self._image_request(
            image,
            adapter,
            module_type=module_type,
            active_input_specs=active_input_specs,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_image_request",
            phase_profile.elapsed(),
            module=module_name,
            function=self.callable_contract.function_name,
        )
        phase_profile = RuntimeProfileTimer.start()
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_runs_per_object_check",
            phase_profile.elapsed(),
            module=module_name,
            function=self.callable_contract.function_name,
        )
        phase_profile = RuntimeProfileTimer.start()
        invocation = self._invocation_request(
            image_request=image_request,
            adapter=adapter,
            current_image=image,
            kwargs=kwargs,
            module_type=module_type,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_invocation_request",
            phase_profile.elapsed(),
            module=module_name,
            function=self.callable_contract.function_name,
        )
        phase_profile = RuntimeProfileTimer.start()
        raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
            self.callable_contract,
            self.raw_func,
            invocation.image,
            invocation.kwargs,
            execution_mode=invocation.execution_mode,
            plane_projection=invocation.plane_projection,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile_deferred(
            "cp_contract_execute",
            phase_profile.elapsed(),
            lambda: {
                "module": module_name,
                "function": self.callable_contract.function_name,
                **cellprofiler_profile_payload_fields("input", invocation.image),
                **cellprofiler_profile_payload_fields("output", raw_output),
            },
        )
        phase_profile = RuntimeProfileTimer.start()
        returned_values, matched_outputs = self._returned_output_values(
            raw_output,
            adapter,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_split_output",
            phase_profile.elapsed(),
            module=module_name,
            function=self.callable_contract.function_name,
        )
        phase_profile = RuntimeProfileTimer.start()
        declared_only_outputs = CellProfilerOutputRecorder.record_module_outputs(
            callable_contract=self.callable_contract,
            active_input_edges=tuple(adapter.request.artifact_inputs.values()),
            adapter=adapter,
            returned_values=returned_values,
            matched_outputs=matched_outputs,
            invocation=invocation,
            image_request=image_request,
            current_image=image,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_record_outputs",
            phase_profile.elapsed(),
            module=module_name,
            function=self.callable_contract.function_name,
        )
        phase_profile = RuntimeProfileTimer.start()
        result = self._published_active_main_flow_output(
            matched_outputs=matched_outputs,
            declared_only_outputs=declared_only_outputs,
            adapter=adapter,
            current_image=image,
            invocation_image=invocation.image,
            plane_projection=invocation.plane_projection,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_replace_main_flow_check",
            phase_profile.elapsed(),
            module=module_name,
            function=self.callable_contract.function_name,
        )
        return result

    def _published_active_main_flow_output(
        self,
        *,
        matched_outputs: tuple[RuntimeMatchedOutput, ...],
        declared_only_outputs: Mapping[
            ArtifactSpecRef,
            RuntimeCallableArgument,
        ],
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
        invocation_image: RuntimeCallableArgument,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> RuntimeCallableArgument:
        """Publish only outputs selected by the active compiled contract."""

        main_flow_refs = (
            self.callable_contract.canonical_return_output_specs.ref_set()
        )
        main_flow_matches = tuple(
            (plan, spec)
            for plan, spec, _value in matched_outputs
            if spec.ref() in main_flow_refs
        )
        if not main_flow_matches:
            return current_image
        return self._replacement_main_flow_output(
            outputs=main_flow_matches,
            declared_only_outputs=declared_only_outputs,
            adapter=adapter,
            current_image=current_image,
            invocation_image=invocation_image,
            plane_projection=plane_projection,
        )

    def _returned_output_values(
        self,
        raw_output: RuntimeFunctionOutput,
        adapter: CellProfilerRuntimeAdapter,
    ) -> tuple[
        Mapping[ArtifactSpecRef, RuntimeCallableArgument],
        tuple[RuntimeMatchedOutput, ...],
    ]:
        """Resolve the callable ABI against this invocation's selected plans."""

        return RuntimeReturnedOutputMatcher(
            callable_contract=self.callable_contract,
            returned_output=raw_output,
        ).resolve_plan_values(tuple(adapter.request.artifact_outputs.values()))

    def _replacement_main_flow_output(
        self,
        *,
        outputs: tuple[tuple[ArtifactOutputPlan, ArtifactSpec], ...],
        declared_only_outputs: Mapping[
            ArtifactSpecRef,
            RuntimeCallableArgument,
        ],
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
        invocation_image: RuntimeCallableArgument,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> RuntimeCallableArgument:
        if not outputs:
            raise ValueError("CellProfiler main-flow replacement requires an output.")
        output_values = tuple(
            (
                spec,
                (
                    declared_only_outputs[spec.ref()]
                    if spec.ref() in declared_only_outputs
                    else adapter.artifact_output_value(output_plan)
                ),
            )
            for output_plan, spec in outputs
        )
        replacement = RuntimeArtifactTypeStrategy.for_main_flow_outputs(
            output_values
        ).published_main_flow_output(
            invocation_image,
            output_values,
            plane_projection,
        )
        return self._merge_named_image_outputs(
            current_image,
            replacement,
            outputs,
        )

    @staticmethod
    def _merge_named_image_outputs(
        current_image: RuntimeCallableArgument,
        replacement: RuntimeCallableArgument,
        outputs: tuple[tuple[ArtifactOutputPlan, ArtifactSpec], ...],
    ) -> RuntimeCallableArgument:
        """Replace exact carried sources or append independent named outputs."""

        if (
            not isinstance(current_image, AlignedImageStack)
            or not current_image.slice_contexts
            or not isinstance(replacement, ImageOutputBundle)
        ):
            return replacement
        if len(replacement.slices) != len(outputs):
            raise ValueError(
                "Recorded image main-flow output count must match its active "
                f"compiled plans: {len(replacement.slices)} != {len(outputs)}."
            )

        slices = list(current_image.slices)
        contexts = list(current_image.slice_contexts)
        for (output_plan, _spec), output_slice, output_context in zip(
            outputs,
            replacement.slices,
            replacement.slice_contexts,
            strict=True,
        ):
            source_ref = output_plan.source_context_source()
            if source_ref is None:
                return replacement
            source_indices = tuple(
                index
                for index, context in enumerate(contexts)
                if context.matches_artifact_ref(source_ref)
            )
            if len(source_indices) > 1:
                raise ValueError(
                    "Named main-flow carrier has duplicate exact source context "
                    f"for {source_ref!r}."
                )
            if source_indices:
                index = source_indices[0]
                slices[index] = output_slice
                contexts[index] = output_context
            else:
                slices.append(output_slice)
                contexts.append(output_context)
        return ImageOutputBundle(tuple(slices), tuple(contexts))

    def _primary_image_inputs(
        self,
        current_image: RuntimeCallableArgument,
        adapter: CellProfilerRuntimeAdapter,
        *,
        module_type: type[CellProfilerModule],
    ) -> tuple[ArtifactSpec, ...]:
        """Apply nominal primary-image policy to exact non-parameter edges."""

        request = RuntimeInputBindingRequest(
            adapter=adapter,
            kwargs={},
            current_image=current_image,
        )
        return module_type.primary_image_inputs(
            self.raw_func,
            request.primary_image_inputs,
        )

    def _measurement_image_inputs(
        self,
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
        image_request: CellProfilerImageRequest | None,
        *,
        module_type: type[CellProfilerModule],
    ) -> tuple[CellProfilerMeasurementImage, ...]:
        image_inputs = self._primary_image_inputs(
            current_image,
            adapter,
            module_type=module_type,
        )
        if not image_inputs:
            return ()
        if (
            self.callable_contract.image_payload_consumption
            is ImagePayloadConsumption.COMPOSED
        ):
            if image_request is None:
                raise ValueError(
                    f"{module_type.require_module_name()} requires a composed "
                    "measurement "
                    "image request."
                )
            return (
                CellProfilerMeasurementImage(
                    source_image_name=measurement_source_name_for_specs(image_inputs),
                    source_aliases=ArtifactSpecCollection(image_inputs).names(),
                    payload=image_request.payload,
                    align_to_labels=False,
                    execution_mode=image_request.execution_mode,
                    plane_projection=image_request.plane_projection,
                ),
            )
        return self._resolved_measurement_images(
            image_inputs,
            adapter,
            current_image,
            reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
        )

    def _independent_measurement_image_inputs(
        self,
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
        *,
        module_type: type[CellProfilerModule],
    ) -> tuple[CellProfilerMeasurementImage, ...]:
        image_inputs = self._primary_image_inputs(
            current_image,
            adapter,
            module_type=module_type,
        )
        if image_inputs:
            return self._resolved_measurement_images(
                image_inputs,
                adapter,
                current_image,
            )
        return (
            CellProfilerMeasurementImage(
                source_image_name=None,
                source_aliases=(),
                payload=current_image,
                reference_domain=CellProfilerMeasurementImageDomain.SOURCE_IMAGE,
            ),
        )

    def _resolved_measurement_images(
        self,
        image_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
        *,
        reference_domain: CellProfilerMeasurementImageDomain = (
            CellProfilerMeasurementImageDomain.SOURCE_IMAGE
        ),
    ) -> tuple[CellProfilerMeasurementImage, ...]:
        source_aliases = ArtifactSpecCollection(image_inputs).names()
        return tuple(
            self._resolved_measurement_image(
                spec,
                adapter,
                current_image,
                source_aliases,
                reference_domain=reference_domain,
            )
            for spec in image_inputs
        )

    def _resolved_measurement_image(
        self,
        spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
        source_aliases: tuple[str, ...],
        *,
        reference_domain: CellProfilerMeasurementImageDomain,
    ) -> CellProfilerMeasurementImage:
        request = RuntimeInputBindingRequest(
            adapter=adapter,
            kwargs={},
            current_image=current_image,
        ).artifact_request_for_spec(spec)
        payload = normalize_cellprofiler_image_payload(
            RuntimeArtifactTypeStrategy.for_artifact_type(
                ImageArtifactType,
            ).raw_runtime_input_value(request)
        )
        metadata = image_payload_metadata(payload)
        plane_axis = metadata.plane_axis
        return CellProfilerMeasurementImage(
            source_image_name=spec.name,
            source_aliases=source_aliases,
            payload=payload,
            plane_projection=(
                replace(
                    preserved_image_plane_projection(
                        payload,
                        adapter,
                        source_aliases,
                    ),
                    source_aliases=source_aliases,
                )
                if plane_axis is not None
                else None
            ),
            reference_domain=reference_domain,
        )

    def _object_label_measurement_image(
        self,
        spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
    ) -> CellProfilerMeasurementImage:
        payload = RuntimeInputBindingRequest(
            adapter=adapter,
            kwargs={},
            current_image=current_image,
        ).label_payload_for(spec)
        if not isinstance(payload, ObjectLabelValue):
            raise TypeError(
                f"Object measurement input {spec.name!r} requires ObjectLabelValue, "
                f"got {type(payload).__name__}."
            )
        reference_image = payload.measurement_reference_image()
        return CellProfilerMeasurementImage(
            source_image_name=None,
            source_aliases=(),
            payload=reference_image,
            reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
            plane_projection=preserved_image_plane_projection(
                reference_image,
                adapter,
            ),
        )

    def _run_per_object_measurement(
        self,
        *,
        image: RuntimeCallableArgument,
        adapter: CellProfilerRuntimeAdapter,
        kwargs: RuntimeCallableKwargs,
        image_request: "CellProfilerImageRequest | None",
        module_type: type[CellProfilerModule],
        active_input_specs: tuple[ArtifactSpec, ...],
        active_output_plans: tuple[ArtifactOutputPlan, ...],
    ) -> RuntimeCallableArgument:
        func = self.raw_func
        current_image = image
        input_image = image
        cellprofiler_runtime = adapter
        source_image_name = (
            None if image_request is None else image_request.source_image_name
        )
        kwargs = dict(kwargs)
        function_name = self.callable_contract.function_name
        profiler = CellProfilerRuntimeProfiler(
            module_type.require_module_name(),
            function_name,
        )
        object_inputs = ArtifactSpecCollection(active_input_specs).of_artifact_type(
            ObjectLabelsArtifactType
        )
        measurement_output_plan = self.measurement_output_plan(active_output_plans)
        image_measurement_rows: list[ColumnarRows] = []
        profile_enabled = CellProfilerRuntimeProfileLogger.enabled()
        if profile_enabled:
            measurement_images_started_at = time.perf_counter()
        measurement_images = self._measurement_image_inputs(
            cellprofiler_runtime,
            current_image,
            image_request,
            module_type=module_type,
        )
        if not measurement_images:
            measurement_images = tuple(
                self._object_label_measurement_image(
                    object_spec,
                    cellprofiler_runtime,
                    current_image,
                )
                for object_spec in object_inputs
            )
            measurement_object_pairs = tuple(
                (
                    measurement_image,
                    object_spec,
                    True,
                )
                for measurement_image, object_spec in zip(
                    measurement_images,
                    object_inputs,
                    strict=True,
                )
            )
        else:
            measurement_object_pairs = tuple(
                (
                    measurement_image,
                    object_spec,
                    object_index == 0,
                )
                for measurement_image in measurement_images
                for object_index, object_spec in enumerate(object_inputs)
            )
        profile_events: list[CellProfilerRuntimeProfileEvent] = []
        if profile_enabled:
            profile_events.append(
                CellProfilerRuntimeProfileEvent(
                    "cp_per_object_measurement_images",
                    time.perf_counter() - measurement_images_started_at,
                    (
                        ("images", len(measurement_images)),
                        ("objects", len(object_inputs)),
                    ),
                )
            )
        measurement_row_policy = module_type.runtime_object_measurement_row_policy()
        label_payload_seconds = 0.0
        label_align_seconds = 0.0
        contract_execute_seconds = 0.0
        output_timings = ObjectMeasurementOutputTimings()
        columnar_rows: list[ColumnarRows] = []
        batch_executor = self.callable_contract.runtime_batch_executor(
            RuntimeBatchExecutionDomain.MEASUREMENT_IMAGES
        )
        processing_contract = self.callable_contract.require_processing_contract()
        output_recorder = ObjectMeasurementOutputRecorder(
            callable_contract=self.callable_contract,
            measurement_output_plan=measurement_output_plan,
            row_policy=measurement_row_policy,
            module_type=module_type,
            func=func,
            adapter=cellprofiler_runtime,
            measurement_images=measurement_images,
            object_inputs=object_inputs,
            image_measurement_rows=image_measurement_rows,
            columnar_rows=columnar_rows,
            timings=output_timings,
        )
        measurement_invocations = tuple(
            measurement_row_policy.invocations(measurement_image, kwargs)
            for (
                measurement_image,
                _object_spec,
                _include_image_measurements,
            ) in measurement_object_pairs
        )
        total_measurement_batch_count = sum(
            (len(invocations) for invocations in measurement_invocations)
        )
        prepared_invocations: list[PreparedObjectMeasurementInvocation] = []
        for (
            measurement_image,
            object_spec,
            include_image_measurements,
        ), invocations in zip(
            measurement_object_pairs, measurement_invocations, strict=True
        ):
            label_payload = RuntimeInputBindingRequest(
                adapter=cellprofiler_runtime,
                kwargs=kwargs,
                current_image=measurement_image.payload,
            ).label_payload_for(object_spec)
            (
                aligned_measurement_image,
                executable_labels,
                completion_label_payload,
                execution_mode,
                preparation_profile_events,
                label_payload_elapsed,
                label_align_elapsed,
            ) = object_measurement_runtime_inputs(
                object_label_execution=object_label_input_execution_mode_from_callable(
                    self.raw_func
                ),
                measurement_image=measurement_image,
                object_spec=object_spec,
                label_payload=label_payload,
                adapter=cellprofiler_runtime,
            )
            profile_events.extend(preparation_profile_events)
            label_payload_seconds += label_payload_elapsed
            label_align_seconds += label_align_elapsed
            for invocation in invocations:
                invocation_kwargs = module_type.object_measurement_invocation_kwargs(
                    invocation.lowered_kwargs(),
                    include_image_measurements=include_image_measurements,
                )
                prepared_invocations.append(
                    PreparedObjectMeasurementInvocation(
                        source_image_name=aligned_measurement_image.source_image_name,
                        execution_mode=execution_mode,
                        plane_projection=aligned_measurement_image.plane_projection,
                        func=func,
                        image=aligned_measurement_image.payload,
                        kwargs={
                            **invocation_kwargs,
                            **_execution_mode_semantic_control_kwargs(
                                processing_contract,
                                execution_mode,
                            ),
                            "labels": executable_labels,
                        },
                        batch_index=len(prepared_invocations),
                        batch_count=total_measurement_batch_count,
                        semantic_group_key=object_measurement_batch_group_key(
                            object_spec=object_spec, labels=completion_label_payload
                        ),
                        measurement_image=aligned_measurement_image,
                        object_spec=object_spec,
                        invocation=invocation,
                        completion_label_payload=completion_label_payload,
                    )
                )
        contract_execute_seconds = PreparedObjectMeasurementInvocationBatch(
            callable_contract=self.callable_contract,
            func=func,
            function_name=function_name,
            invocations=tuple(prepared_invocations),
            batch_executor=batch_executor,
        ).execute(output_recorder)
        if profile_enabled:
            profile_events.extend(
                (
                    CellProfilerRuntimeProfileEvent(
                        "cp_per_object_label_payload", label_payload_seconds
                    ),
                    CellProfilerRuntimeProfileEvent(
                        "cp_per_object_label_align", label_align_seconds
                    ),
                    CellProfilerRuntimeProfileEvent(
                        "cp_per_object_contract_execute", contract_execute_seconds
                    ),
                    CellProfilerRuntimeProfileEvent(
                        "cp_per_object_split_output", output_timings.split_seconds
                    ),
                    CellProfilerRuntimeProfileEvent(
                        "cp_per_object_complete_rows",
                        output_timings.complete_rows_seconds,
                    ),
                    CellProfilerRuntimeProfileEvent(
                        "cp_per_object_annotate_rows",
                        output_timings.annotate_seconds,
                        (
                            (
                                "rows",
                                sum(
                                    rows.row_count()
                                    for rows in (
                                        *image_measurement_rows,
                                        *columnar_rows,
                                    )
                                ),
                            ),
                        ),
                    ),
                )
            )
        combined_source_image_name = measurement_row_policy.table_source_image_name(
            measurement_images, source_image_name
        )
        combined_source_metadata = (
            CellProfilerMeasurementImage.composed_source_metadata(
                measurement_images,
                mode=measurement_row_policy.source_metadata_composition_mode(
                    measurement_images
                ),
            )
        )
        if combined_source_metadata is None:
            combined_source_metadata = image_payload_metadata(
                CellProfilerMeasurementImage.shared_source_payload(measurement_images)
            )
        if profile_enabled:
            record_started_at = time.perf_counter()
        table_groups = tuple(
            (row_batches, object_name)
            for row_batches, object_name in (
                (
                    tuple(image_measurement_rows),
                    measurement_row_policy.table_object_owner(
                        object_inputs,
                        contains_image_measurement_rows=True,
                    ),
                ),
                (
                    tuple(columnar_rows),
                    measurement_row_policy.table_object_owner(object_inputs),
                ),
            )
            if row_batches
        )
        if not table_groups:
            table_groups = (
                (
                    (MeasurementSparseColumnarRows.from_rows((), fields=()),),
                    measurement_row_policy.table_object_owner(object_inputs),
                ),
            )
        for row_batches, object_name in table_groups:
            rows = (
                row_batches[0]
                if len(row_batches) == 1
                else ConcatenatedColumnarRows(row_batches)
            )
            table = module_type.build_measurement_table(
                name=measurement_output_plan.name,
                rows=rows,
                object_name=object_name,
                source_image_name=combined_source_image_name,
                source_metadata=combined_source_metadata,
            )
            measurement_row_policy.validate_table_ownership(table)
            cellprofiler_runtime.add_measurements(table)
        if profile_enabled:
            profile_events.append(
                CellProfilerRuntimeProfileEvent(
                    "cp_per_object_record_measurements",
                    time.perf_counter() - record_started_at,
                    (
                        (
                            "rows",
                            sum(
                                rows.row_count()
                                for rows in (*image_measurement_rows, *columnar_rows)
                            ),
                        ),
                    ),
                )
            )
            profiler.record_events(tuple(profile_events))
        return input_image

    def _run_per_image_measurement(
        self,
        *,
        image: RuntimeCallableArgument,
        adapter: CellProfilerRuntimeAdapter,
        kwargs: RuntimeCallableKwargs,
        module_type: type[CellProfilerModule],
        active_input_specs: tuple[ArtifactSpec, ...],
        active_output_plans: tuple[ArtifactOutputPlan, ...],
    ) -> RuntimeCallableArgument:
        func = self.raw_func
        input_image = image
        current_image = image
        cellprofiler_runtime = adapter
        kwargs = dict(kwargs)
        function_name = self.callable_contract.function_name
        module_name = module_type.require_module_name()
        profiler = CellProfilerRuntimeProfiler(
            module_name,
            function_name,
        )
        measurement_output_plan = self.measurement_output_plan(active_output_plans)
        combined_rows: list[ColumnarRows] = []
        measurement_images_started_at = time.perf_counter()
        measurement_images = self._independent_measurement_image_inputs(
            cellprofiler_runtime,
            current_image,
            module_type=module_type,
        )
        profiler.record(
            "cp_per_image_measurement_images",
            time.perf_counter() - measurement_images_started_at,
            images=len(measurement_images),
        )
        kwargs_started_at = time.perf_counter()
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(
                cellprofiler_runtime,
                current_image,
                kwargs,
                module_type=module_type,
            ),
        }
        invocation_kwargs = runtime_kwargs
        profiler.record(
            "cp_per_image_prepare_kwargs",
            time.perf_counter() - kwargs_started_at,
        )
        contract_execute_seconds = 0.0
        split_rows_seconds = 0.0
        combined_tables: list[MeasurementTable] = []
        for measurement_image in measurement_images:
            contract_started_at = time.perf_counter()
            raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                self.callable_contract,
                func,
                measurement_image.payload,
                invocation_kwargs,
                execution_mode=measurement_image.execution_mode,
                plane_projection=measurement_image.plane_projection,
            )
            contract_execute_seconds += time.perf_counter() - contract_started_at
            split_rows_started_at = time.perf_counter()
            returned_values, matched_outputs = self._returned_output_values(
                raw_output,
                cellprofiler_runtime,
            )
            measurement_output = next(
                matched_output
                for matched_output in matched_outputs
                if matched_output[0] is measurement_output_plan
            )
            matched_plan, matched_spec, matched_value = measurement_output
            measurement_record_request = CellProfilerOutputRecordRequest(
                callable_contract=self.callable_contract,
                active_input_edges=tuple(
                    cellprofiler_runtime.request.artifact_inputs.values()
                ),
                adapter=cellprofiler_runtime,
                spec=matched_spec,
                output_plan=matched_plan,
                output_value=matched_value,
                source=measurement_image,
                call_kwargs=invocation_kwargs,
                current_image=measurement_image.payload,
                declared_only_outputs=CellProfilerOutputRecorder.transient_output_values(
                    callable_contract=self.callable_contract,
                    active_output_plans=active_output_plans,
                    returned_values=returned_values,
                ),
            )
            measurement_table = measurement_table_for_module(measurement_record_request)
            combined_tables.append(measurement_table)
            combined_rows.append(
                cast(
                    ColumnarRows,
                    MeasurementRowOwnership(
                        source_image_name=measurement_table.source_image_name
                    ).annotate_rows(measurement_table.rows),
                )
            )
            split_rows_seconds += time.perf_counter() - split_rows_started_at
        profiler.record("cp_per_image_contract_execute", contract_execute_seconds)
        profiler.record(
            "cp_per_image_split_rows",
            split_rows_seconds,
            rows=sum(rows.row_count() for rows in combined_rows),
        )
        if not combined_rows:
            raise ValueError(
                f"{module_name} per-image measurement execution produced "
                "no measurement tables."
            )
        rows = (
            combined_rows[0]
            if len(combined_rows) == 1
            else ConcatenatedColumnarRows(tuple(combined_rows))
        )
        object_name_field = MeasurementRowAxisField.OBJECT_NAME.value
        rows_only_declare_object_name = rows.row_count() > 0 and all(
            object_name_field in row for row in rows.iter_row_mappings()
        )
        table_source_names = tuple(
            dict.fromkeys(table.source_image_name for table in combined_tables)
        )
        image_measurement_source_name = (
            table_source_names[0] if len(table_source_names) == 1 else None
        )
        if image_measurement_source_name is None:
            image_measurement_source_name = (
                CellProfilerMeasurementImage.shared_source_image_name(
                    measurement_images
                )
            )
        if rows_only_declare_object_name:
            image_measurement_source_name = None
        source_metadata = MeasurementTableUnion(
            measurement_output_plan.name,
            tuple(combined_tables),
        ).source_metadata()
        record_started_at = time.perf_counter()
        cellprofiler_runtime.add_measurements(
            module_type.build_measurement_table(
                name=measurement_output_plan.name,
                rows=rows,
                object_name=None,
                source_image_name=image_measurement_source_name,
                source_metadata=source_metadata,
            )
        )
        profiler.record(
            "cp_per_image_record_measurements",
            time.perf_counter() - record_started_at,
            rows=rows.row_count(),
        )
        return input_image

    def _runtime_input_kwargs(
        self,
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
        kwargs: RuntimeCallableKwargs,
        *,
        module_type: type[CellProfilerModule],
        primary_image: RuntimeCallableArgument | None = None,
    ) -> dict[str, RuntimeCallableArgument]:
        if (
            not self.callable_contract.artifact_inputs
            and not module_type.binds_without_declared_inputs
        ):
            return {}
        return module_type.bind_runtime_inputs(
            RuntimeInputBindingRequest(
                adapter=adapter,
                kwargs=kwargs,
                current_image=module_type.binding_current_image(
                    current_image=current_image,
                    primary_image=primary_image,
                ),
            )
        )

    def _image_request(
        self,
        current_image: RuntimeCallableArgument,
        adapter: CellProfilerRuntimeAdapter,
        *,
        module_type: type[CellProfilerModule],
        active_input_specs: tuple[ArtifactSpec, ...],
    ) -> "CellProfilerImageRequest":
        input_binding = RuntimeInputBindingRequest(
            adapter=adapter,
            kwargs={},
            current_image=current_image,
        )
        image_inputs = module_type.primary_image_inputs(
            self.raw_func,
            input_binding.primary_image_inputs,
        )
        runtime_projection = RuntimePlaneAxisValueProjection.from_projector(
            adapter,
            RuntimePlaneAxis.RUNTIME_SLICE,
            (),
        )
        current_runtime_payload = current_image
        if (
            runtime_projection is not None
            and runtime_projection.plane_index is not None
        ):
            current_runtime_payload = cast(
                RuntimeCallableArgument,
                RuntimeSliceProjection.value_for_slice(
                    current_image,
                    runtime_projection,
                ),
            )
        if not image_inputs:
            current_image_payload = current_runtime_payload
            current_image_payload = normalize_cellprofiler_image_payload(
                current_image_payload
            )
            if (
                runtime_projection is not None
                and runtime_projection.plane_index is not None
                and image_payload_metadata(current_image_payload).plane_axis
                is runtime_projection.axis
            ):
                current_image_payload = cast(
                    RuntimeCallableArgument,
                    RuntimeSliceProjection.value_for_slice(
                        current_image_payload,
                        runtime_projection,
                    ),
                )
            return CellProfilerImageRequest(
                payload=current_image_payload,
                source_image_name=self._input_source_image_name(
                    adapter,
                    current_image_payload,
                    active_input_specs=active_input_specs,
                ),
                source_aliases=(),
                image_count=1,
                execution_mode=ImagePayloadExecutionMode.NATURAL,
                plane_projection=preserved_image_plane_projection(
                    current_image_payload,
                    adapter,
                ),
            )
        payloads = []
        source_names: list[str | None] = []
        image_strategy = RuntimeArtifactTypeStrategy.for_artifact_type(
            ImageArtifactType
        )
        input_binding = replace(input_binding, current_image=current_runtime_payload)
        for spec in image_inputs:
            request = input_binding.artifact_request_for_spec(spec)
            payloads.append(image_strategy.runtime_input_value(request))
            source_names.append(image_strategy.source_image_name(request))
        parameter_image_inputs = input_binding.image_inputs
        broadcast_sources = tuple(
            sources[0]
            for spec in parameter_image_inputs
            for sources in (spec.stack_broadcast_sources(),)
            if len(sources) == 1
        )
        align_primary_images = len(image_inputs) > 1 and broadcast_sources == tuple(
            spec.ref() for spec in image_inputs
        )
        composition = compose_aligned_image_payload(
            f"{module_type.require_module_name()} image inputs "
            f"{tuple(spec.name for spec in image_inputs)!r}",
            tuple(payloads),
            slice_contexts=(
                tuple(
                    AlignedImageSliceContext.main_flow(
                        output_key=spec.name,
                        artifact_kind=spec.artifact_type.value,
                    )
                    for spec in image_inputs
                )
                if align_primary_images
                else ()
            ),
            stack_broadcast_source_indices=self._stack_broadcast_source_indices(
                image_inputs
            ),
        )
        return CellProfilerImageRequest(
            payload=composition.payload,
            source_image_name=self._primary_image_source_name_from_sources(
                image_inputs, tuple(source_names)
            ),
            source_aliases=ArtifactSpecCollection(image_inputs).names(),
            image_count=len(payloads),
            execution_mode=composition.execution_mode,
            plane_projection=composition.preserved_plane_projection(
                adapter,
                source_aliases=ArtifactSpecCollection(image_inputs).names(),
            ),
        )

    def _input_source_image_name(
        self,
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
        *,
        active_input_specs: tuple[ArtifactSpec, ...],
    ) -> str | None:
        input_binding = RuntimeInputBindingRequest(
            adapter=adapter,
            kwargs={},
            current_image=current_image,
        )
        source_names: list[str] = []
        for spec in active_input_specs:
            if not spec.artifact_type.carries_source_image_context:
                continue
            source_name = RuntimeArtifactTypeStrategy.for_artifact_type(
                spec.artifact_type
            ).source_image_name(
                input_binding.artifact_request_for_spec(spec)
            )
            if source_name is not None:
                source_names.append(source_name)
        return single_source_name(tuple(source_names))

    @staticmethod
    def _primary_image_source_name_from_sources(
        image_inputs: tuple[ArtifactSpec, ...], source_names: tuple[str | None, ...]
    ) -> str | None:
        if len(source_names) > 1:
            return measurement_source_name_for_specs(image_inputs)
        if not source_names:
            return None
        return source_names[0]

    @staticmethod
    def _stack_broadcast_source_indices(
        input_specs: tuple[ArtifactSpec, ...],
    ) -> tuple[int | None, ...]:
        """Resolve exact stack-broadcast owners from declared input relations."""

        indices_by_ref: dict[ArtifactSpecRef, list[int]] = {}
        for input_index, spec in enumerate(input_specs):
            indices_by_ref.setdefault(spec.ref(), []).append(input_index)

        result: list[int | None] = []
        for input_index, spec in enumerate(input_specs):
            sources = spec.stack_broadcast_sources()
            if len(sources) > 1:
                raise ValueError(
                    f"Input {spec.ref()!r} declares multiple stack-broadcast "
                    f"owners: {sources!r}."
                )
            if not sources:
                result.append(None)
                continue
            source_indices = tuple(indices_by_ref.get(sources[0], ()))
            if len(source_indices) != 1:
                raise ValueError(
                    f"Input {spec.ref()!r} requires exactly one active occurrence "
                    f"of stack-broadcast owner {sources[0]!r}, got "
                    f"{source_indices!r}."
                )
            source_index = source_indices[0]
            if source_index == input_index:
                raise ValueError(
                    f"Input {spec.ref()!r} cannot broadcast from itself."
                )
            result.append(source_index)
        return tuple(result)

    def _invocation_request(
        self,
        *,
        image_request: "CellProfilerImageRequest",
        adapter: CellProfilerRuntimeAdapter,
        current_image: RuntimeCallableArgument,
        kwargs: RuntimeCallableKwargs,
        module_type: type[CellProfilerModule],
    ) -> "RuntimeFunctionInvocationRequest":
        profile_enabled = CellProfilerRuntimeProfileLogger.enabled()
        if profile_enabled:
            phase_started_at = time.perf_counter()
        else:
            phase_started_at = 0.0
        module_name = module_type.require_module_name()
        bound_runtime_kwargs = self._runtime_input_kwargs(
            adapter,
            current_image,
            kwargs,
            module_type=module_type,
            primary_image=image_request.payload,
        )
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_invocation_bind_runtime_inputs",
                time.perf_counter() - phase_started_at,
                module=module_name,
            )
        runtime_kwargs = {**kwargs, **bound_runtime_kwargs}
        if profile_enabled:
            phase_started_at = time.perf_counter()
        runtime_kwargs = module_type.invocation_runtime_kwargs(
            image_request=image_request,
            runtime_kwargs=runtime_kwargs,
        )
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_invocation_primary_policy_kwargs",
                time.perf_counter() - phase_started_at,
                module=module_name,
            )
        if profile_enabled:
            phase_started_at = time.perf_counter()
        runtime_projection = RuntimePlaneAxisValueProjection.from_projector(
            adapter,
            RuntimePlaneAxis.RUNTIME_SLICE,
            (),
        )
        if (
            runtime_projection is not None
            and runtime_projection.plane_index is not None
        ):
            runtime_kwargs = cast(
                dict[str, RuntimeCallableArgument],
                RuntimeSliceProjection.kwargs_for_slice(
                    runtime_kwargs,
                    runtime_projection,
                ),
            )
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_invocation_project_runtime_kwargs",
                time.perf_counter() - phase_started_at,
                module=module_name,
            )
            phase_started_at = time.perf_counter()
        image_request = module_type.project_invocation_image_request(
            image_request=image_request,
            runtime_kwargs=runtime_kwargs,
        )
        invocation_image = image_request.payload
        default_execution_mode = (
            self.callable_contract.runtime_image_execution_mode
            or image_request.execution_mode
        )
        execution_mode = module_type.execution_mode(
            default_execution_mode,
            image=invocation_image,
            kwargs=runtime_kwargs,
            variable_components=adapter.request.variable_components,
        )
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_invocation_execution_mode_policy",
                time.perf_counter() - phase_started_at,
                module=module_name,
            )
        invocation_kwargs = {
            **runtime_kwargs,
            **_execution_mode_semantic_control_kwargs(
                self.callable_contract.require_processing_contract(),
                execution_mode,
            ),
        }
        return RuntimeFunctionInvocationRequest(
            image=invocation_image,
            kwargs=invocation_kwargs,
            source_image_name=image_request.source_image_name,
            image_count=image_request.image_count,
            execution_mode=execution_mode,
            plane_projection=image_request.plane_projection,
        )


def _execution_mode_semantic_control_kwargs(
    processing_contract: ProcessingContract,
    execution_mode: ImagePayloadExecutionMode,
) -> dict[str, RuntimeCallableArgument]:
    """Return semantic controls required by a resolved image execution mode."""
    return {
        name: execution_mode is ImagePayloadExecutionMode.NATURAL
        for name in (
            processing_contract.declaration.injected_semantic_control_parameter_names()
        )
    }
