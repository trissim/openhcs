"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from enum import Enum
from inspect import Parameter, signature, unwrap
import re
from types import MappingProxyType
from typing import Any, ClassVar, get_args, get_origin, get_type_hints

from metaclass_registry import AutoRegisterMeta
import numpy as np
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
    aligned_image_stack_kwargs,
    compose_aligned_image_payload,
    payload_slice_count,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.callable_contract import CallableContract
from openhcs.core.config import DtypeConfig
from openhcs.core.memory import (
    MEMORY_TYPE_NUMPY,
    convert_memory,
    detect_memory_type,
    stack_slices,
    unstack_slices,
)
from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_image_stack,
)
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.core.runtime_semantics import FieldSpec
from openhcs.core.runtime_stores import require_runtime_value_store
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    _aggregate_pure_2d_auxiliary_output,
    _pure_2d_slice_results,
    _rewrite_slice_index,
)

from benchmark.cellprofiler_library import canonical_module_name
from benchmark.cellprofiler_compat.measurement_lookup import (
    annotate_measurement_row_object,
    count_feature_object_name,
    measurement_values_for_label_slices,
    measurement_values_for_feature,
)
from benchmark.cellprofiler_compat.relationship_payload import (
    CellProfilerRelationshipPayload,
)
from benchmark.cellprofiler_compat.runtime_adapter import CellProfilerRuntimeAdapter
from benchmark.converter.contract_inference import InferredContract, infer_contract

_MODULE_NAME_REGISTRY_KEY = "module_name"
_INVOCATION_CONTROL_KWARGS = frozenset(("dtype_config", "slice_by_slice"))


def _cellprofiler_image_payload(payload: Any) -> Any:
    """Return payload in CellProfiler's float image intensity domain."""
    array = np.asarray(payload)
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        max_value = np.iinfo(array.dtype).max
        if max_value <= 1:
            return array.astype(np.float32)
        return array.astype(np.float32) / float(max_value)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32, copy=False)
    return payload


def cellprofiler_runtime_adapter_factory(
    request: RuntimeAdapterRequest,
) -> CellProfilerRuntimeAdapter:
    """Build a CellProfiler adapter for one FunctionStep invocation."""
    axis_id = request.context.axis_id
    if not axis_id:
        raise RuntimeError(
            "ProcessingContext.axis_id is required for CellProfiler runtime."
        )
    return CellProfilerRuntimeAdapter(
        runtime_value_store=require_runtime_value_store(
            request.context,
            owner_name="ProcessingContext",
        ),
        axis_id=str(axis_id),
        artifact_outputs=request.artifact_outputs,
        source_binding_plan=request.source_binding_plan,
        source_binding_context=request.source_binding_context,
        group_key=request.group_key,
        processing_context=request.context,
        filemanager=request.context.filemanager,
    )


@dataclass(frozen=True, slots=True)
class CellProfilerModuleExecutor:
    """Execute one generated CellProfiler module against a typed runtime adapter."""

    contract: ModuleArtifactContract

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ModuleArtifactContract):
            raise TypeError(
                "CellProfilerModuleExecutor.contract must be "
                "ModuleArtifactContract, got "
                f"{type(self.contract).__name__}."
            )

    @property
    def module_name(self) -> str:
        return self.contract.module_name

    @property
    def inputs(self) -> tuple[ArtifactSpec, ...]:
        return self.contract.inputs

    @property
    def runtime_artifact_inputs(self) -> tuple[ArtifactSpec, ...]:
        return self.contract.runtime_artifact_inputs

    @property
    def outputs(self) -> tuple[ArtifactSpec, ...]:
        return self.contract.outputs

    def run(
        self,
        func: Callable[..., Any],
        image: Any,
        *,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        **kwargs: Any,
    ) -> Any:
        """Call the absorbed function and record declared outputs through the adapter."""
        if self._runs_per_image_measurement(func):
            return self._run_per_image_measurement(
                func,
                input_image=image,
                fallback_image=image,
                cellprofiler_runtime=cellprofiler_runtime,
                **kwargs,
            )

        image_request = self._image_request(
            func,
            image,
            cellprofiler_runtime,
        )
        if self._runs_per_object_measurement():
            return self._run_per_object_measurement(
                func,
                input_image=image,
                fallback_image=image,
                image_request=image_request,
                cellprofiler_runtime=cellprofiler_runtime,
                source_image_name=image_request.source_image_name,
                **kwargs,
            )

        invocation = self._invocation_request(
            func,
            image_request=image_request,
            adapter=cellprofiler_runtime,
            fallback_image=image,
            kwargs=kwargs,
        )
        raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
            func,
            invocation.image,
            invocation.kwargs,
            execution_mode=invocation.execution_mode,
        )
        main_output, artifact_values = _split_cellprofiler_output(raw_output)
        self._record_outputs(
            func,
            cellprofiler_runtime,
            main_output,
            artifact_values,
            source_image_name=invocation.source_image_name,
        )
        if not self._replaces_main_flow(
            input_image=image,
            output_image=main_output,
        ):
            return image
        return _openhcs_main_flow_output(image, main_output)

    def _runs_per_object_measurement(self) -> bool:
        return CellProfilerPerObjectMeasurementPolicy.matches(
            self.module_name,
            self._object_input_specs(),
        )

    def _runs_per_image_measurement(self, func: Callable[..., Any]) -> bool:
        return CellProfilerPerImageMeasurementPolicy.matches(
            CellProfilerPerImageMeasurementRequest(
                module_name=self.module_name,
                func=func,
                image_inputs=self._primary_image_inputs(func),
                object_inputs=self._object_input_specs(),
                outputs=self.outputs,
            )
        )

    def _produces_image_output(self) -> bool:
        return any(spec.kind is ArtifactKind.IMAGE for spec in self.outputs)

    def _replaces_main_flow(
        self,
        *,
        input_image: Any,
        output_image: Any,
    ) -> bool:
        if not self._produces_image_output():
            return False
        return payload_slice_count(output_image) == payload_slice_count(input_image)

    def _run_per_object_measurement(
        self,
        func: Callable[..., Any],
        *,
        input_image: Any,
        fallback_image: Any,
        image_request: "CellProfilerImageRequest",
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        source_image_name: str | None,
        **kwargs: Any,
    ) -> Any:
        object_inputs = self._object_input_specs()
        measurement_outputs = _specs_of_kind(self.outputs, ArtifactKind.MEASUREMENTS)
        if len(measurement_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} per-object execution requires exactly one "
                "measurement output."
            )

        combined_rows: list[Any] = []
        measurement_images = self._measurement_image_inputs(
            func,
            cellprofiler_runtime,
            fallback_image,
            image_request,
        )
        for measurement_image in measurement_images:
            for object_spec in object_inputs:
                raw_labels = self._object_labels(
                    object_spec,
                    cellprofiler_runtime,
                    input_image,
                )
                measurement_labels = _measurement_labels_for_image(
                    measurement_image.payload,
                    raw_labels,
                )
                aligned_image = (
                    _measurement_image_for_labels(
                        measurement_image.payload,
                        measurement_labels,
                        reference_domain=measurement_image.reference_domain,
                    )
                    if measurement_image.align_to_labels
                    else measurement_image.payload
                )
                raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                    func,
                    aligned_image,
                    {**kwargs, "labels": measurement_labels},
                    execution_mode=measurement_image.execution_mode,
                )
                _ignored_main_output, artifact_values = _split_cellprofiler_output(
                    raw_output
                )
                combined_rows.extend(
                    annotate_measurement_row_object(row, object_spec.name)
                    for row in _measurement_rows_from_output(artifact_values)
                )

        source_image_names = tuple(
            image.source_image_name
            for image in measurement_images
            if image.source_image_name is not None
        )
        combined_source_image_name = (
            source_image_name
            if not source_image_names
            else _single_source_name(source_image_names)
        )

        _record_measurements(
            cellprofiler_runtime,
            measurement_outputs[0].name,
            combined_rows,
            fields=_measurement_record_fields(combined_rows, func),
            object_name=object_inputs[0].name if len(object_inputs) == 1 else None,
            source_image_name=combined_source_image_name,
        )
        return input_image

    def _run_per_image_measurement(
        self,
        func: Callable[..., Any],
        *,
        input_image: Any,
        fallback_image: Any,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        **kwargs: Any,
    ) -> Any:
        measurement_outputs = _specs_of_kind(self.outputs, ArtifactKind.MEASUREMENTS)
        if len(measurement_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} per-image execution requires exactly one "
                "measurement output."
            )

        combined_rows: list[Any] = []
        measurement_images = self._independent_measurement_image_inputs(
            func,
            cellprofiler_runtime,
            fallback_image,
        )
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(
                func,
                cellprofiler_runtime,
                fallback_image,
                kwargs,
            ),
        }
        coerced_kwargs = _coerce_invocation_kwargs(func, runtime_kwargs)
        for measurement_image in measurement_images:
            raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                func,
                measurement_image.payload,
                coerced_kwargs,
                execution_mode=measurement_image.execution_mode,
            )
            _ignored_main_output, artifact_values = _split_cellprofiler_output(
                raw_output
            )
            combined_rows.extend(_measurement_rows_from_output(artifact_values))

        source_image_names = tuple(
            image.source_image_name
            for image in measurement_images
            if image.source_image_name is not None
        )
        _record_measurements(
            cellprofiler_runtime,
            measurement_outputs[0].name,
            combined_rows,
            fields=_measurement_record_fields(combined_rows, func),
            source_image_name=_single_source_name(source_image_names),
        )
        return input_image

    def _measurement_image_inputs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
        image_request: "CellProfilerImageRequest",
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        image_inputs = self._primary_image_inputs(func)
        if not image_inputs:
            return (
                self._measurement_carrier_image(
                    adapter,
                    fallback_image,
                    reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
                ),
            )

        if not CellProfilerPerObjectMeasurementPolicy.measures_images_independently(
            self.module_name
        ):
            return (
                self._composed_measurement_image(image_request),
            )

        return self._resolved_measurement_images(image_inputs, adapter, fallback_image)

    def _independent_measurement_image_inputs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        image_inputs = self._primary_image_inputs(func)
        if not image_inputs:
            return (
                self._measurement_carrier_image(
                    adapter,
                    fallback_image,
                    reference_domain=CellProfilerMeasurementImageDomain.SOURCE_IMAGE,
                ),
            )

        return self._resolved_measurement_images(image_inputs, adapter, fallback_image)

    def _measurement_carrier_image(
        self,
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
        *,
        reference_domain: "CellProfilerMeasurementImageDomain",
    ) -> "CellProfilerMeasurementImage":
        return CellProfilerMeasurementImage(
            source_image_name=self._input_source_image_name(adapter),
            payload=_object_only_reference_image(fallback_image),
            reference_domain=reference_domain,
        )

    def _composed_measurement_image(
        self,
        image_request: "CellProfilerImageRequest",
    ) -> "CellProfilerMeasurementImage":
        return CellProfilerMeasurementImage(
            source_image_name=image_request.source_image_name,
            payload=image_request.payload,
            align_to_labels=False,
            execution_mode=image_request.execution_mode,
        )

    def _resolved_measurement_images(
        self,
        image_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        runtime_image_names = frozenset(self._runtime_image_names())
        resolved_images: list[CellProfilerMeasurementImage] = []
        for spec in image_inputs:
            resolved_images.append(
                self._resolved_measurement_image(
                    spec,
                    adapter,
                    fallback_image,
                    runtime_image_names,
                )
            )
        return tuple(resolved_images)

    def _resolved_measurement_image(
        self,
        spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
        runtime_image_names: frozenset[str],
    ) -> "CellProfilerMeasurementImage":
        if spec.name in runtime_image_names:
            runtime_image = adapter.get_image(spec.name)
            return CellProfilerMeasurementImage(
                source_image_name=runtime_image.source_image_name or spec.name,
                payload=_cellprofiler_image_payload(runtime_image.data),
            )
        return CellProfilerMeasurementImage(
            source_image_name=spec.name,
            payload=_cellprofiler_image_payload(
                adapter.resolve_source_image(spec.name, fallback_image)
            ),
        )

    def _object_input_specs(self) -> tuple[ArtifactSpec, ...]:
        return _specs_of_kind(
            self._declared_input_specs(),
            ArtifactKind.OBJECT_LABELS,
        )

    def _object_labels(
        self,
        spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
    ) -> Any:
        if spec.name in self._external_source_object_names():
            return adapter.resolve_source_objects(spec.name, fallback_image).labels
        return adapter.get_objects(spec.name).labels

    def _runtime_input_kwargs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        runtime_inputs = self._special_runtime_inputs(func)
        if not runtime_inputs:
            return {}

        special_input_names = special_input_names_from_callable(func)
        if special_input_names:
            return CellProfilerSpecialInputPolicy.for_module(self.module_name).bind(
                CellProfilerSpecialInputBindingRequest(
                    module_name=self.module_name,
                    parameter_names=special_input_names,
                    runtime_inputs=runtime_inputs,
                    adapter=adapter,
                    kwargs=kwargs,
                    fallback_image=fallback_image,
                    external_image_names=frozenset(self._external_source_image_names()),
                    external_object_names=frozenset(
                        self._external_source_object_names()
                    ),
                    runtime_image_names=frozenset(self._runtime_image_names()),
                )
            )

        unsupported_non_object_inputs = tuple(
            spec
            for spec in runtime_inputs
            if spec.kind is not ArtifactKind.OBJECT_LABELS
        )
        if unsupported_non_object_inputs:
            raise NotImplementedError(
                f"{self.module_name} has runtime inputs "
                f"{[spec.name for spec in unsupported_non_object_inputs]} with "
                "no declared special_inputs binding."
            )

        object_inputs = _specs_of_kind(
            runtime_inputs,
            ArtifactKind.OBJECT_LABELS,
        )
        return CellProfilerObjectInputPolicy.for_module(self.module_name).bind(
            CellProfilerObjectInputBindingRequest(
                module_name=self.module_name,
                object_inputs=object_inputs,
                adapter=adapter,
                kwargs=kwargs,
                fallback_image=fallback_image,
                external_object_names=frozenset(self._external_source_object_names()),
            )
        )

    def _special_runtime_inputs(
        self,
        func: Callable[..., Any],
    ) -> tuple[ArtifactSpec, ...]:
        declared_inputs = self._declared_input_specs()
        non_image_inputs = tuple(
            spec
            for spec in declared_inputs
            if spec.kind is not ArtifactKind.IMAGE
        )
        special_image_inputs = CellProfilerSpecialInputPolicy.for_module(
            self.module_name
        ).special_image_inputs(
            self.module_name,
            func,
            declared_inputs,
        )
        return (
            *non_image_inputs,
            *special_image_inputs,
        )

    def _record_outputs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        main_output: Any,
        artifact_values: tuple[Any, ...],
        *,
        source_image_name: str | None,
    ) -> None:
        if not self.outputs:
            return

        output_values = _output_values_by_kind(
            self.outputs,
            main_output,
            artifact_values,
        )
        for spec in self.outputs:
            CellProfilerOutputRecorder.for_kind(spec.kind).record(
                CellProfilerOutputRecordRequest(
                    executor=self,
                    adapter=adapter,
                    spec=spec,
                    value=output_values[spec.name],
                    output_values=output_values,
                    source_image_name=source_image_name,
                    func=func,
                )
            )

    def _image_request(
        self,
        func: Callable[..., Any],
        fallback_image: Any,
        adapter: CellProfilerRuntimeAdapter,
    ) -> "CellProfilerImageRequest":
        image_inputs = self._primary_image_inputs(func)
        if not image_inputs:
            payload = (
                _object_only_reference_image(fallback_image)
                if self._object_input_specs()
                else _cellprofiler_image_payload(fallback_image)
            )
            return CellProfilerImageRequest(
                payload=payload,
                source_image_name=self._input_source_image_name(adapter),
                image_count=1,
                execution_mode=ImagePayloadExecutionMode.NATURAL,
            )

        runtime_image_names = {
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.IMAGE,
            )
        }
        external_image_names = tuple(
            spec.name
            for spec in image_inputs
            if spec.name not in runtime_image_names
        )
        adapter.require_resolvable_source_aliases(external_image_names)
        payloads = []
        for spec in image_inputs:
            if spec.name in runtime_image_names:
                payloads.append(
                    _cellprofiler_image_payload(adapter.get_image(spec.name).data)
                )
                continue
            payloads.append(
                _cellprofiler_image_payload(
                    adapter.resolve_source_image(spec.name, fallback_image)
                )
            )
        composition = compose_aligned_image_payload(self.module_name, tuple(payloads))
        return CellProfilerImageRequest(
            payload=composition.payload,
            source_image_name=self._input_source_image_name(adapter),
            image_count=len(payloads),
            execution_mode=composition.execution_mode,
        )

    def _primary_image_inputs(
        self,
        func: Callable[..., Any],
    ) -> tuple[ArtifactSpec, ...]:
        declared_inputs = self._declared_input_specs()
        image_inputs = _specs_of_kind(
            declared_inputs,
            ArtifactKind.IMAGE,
        )
        special_image_count = len(
            CellProfilerSpecialInputPolicy.for_module(
                self.module_name
            ).special_image_inputs(
                self.module_name,
                func,
                declared_inputs,
            )
        )
        if special_image_count == 0:
            return image_inputs
        return image_inputs[: len(image_inputs) - special_image_count]

    def _input_source_image_name(
        self,
        adapter: CellProfilerRuntimeAdapter,
    ) -> str | None:
        source_names: list[str] = []
        runtime_image_names = frozenset(
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.IMAGE,
            )
        )
        external_image_names = frozenset(self._external_source_image_names())
        for spec in self._declared_input_specs():
            source_name = _artifact_kind_strategy(spec.kind).source_image_name(
                CellProfilerArtifactKindRequest(
                    spec=spec,
                    adapter=adapter,
                    external_image_names=external_image_names,
                    external_object_names=frozenset(
                        self._external_source_object_names()
                    ),
                    runtime_image_names=runtime_image_names,
                )
            )
            if source_name is not None:
                source_names.append(source_name)

        return _single_source_name(tuple(source_names))

    def _invocation_request(
        self,
        func: Callable[..., Any],
        *,
        image_request: "CellProfilerImageRequest",
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
        kwargs: Mapping[str, Any],
    ) -> "CellProfilerInvocationRequest":
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(func, adapter, fallback_image, kwargs),
        }
        return CellProfilerInvocationRequest(
            image=image_request.payload,
            kwargs=_coerce_invocation_kwargs(func, runtime_kwargs),
            source_image_name=image_request.source_image_name,
            image_count=image_request.image_count,
            execution_mode=image_request.execution_mode,
        )

    def _external_source_image_names(self) -> tuple[str, ...]:
        runtime_image_names = frozenset(self._runtime_image_names())
        return tuple(
            spec.name
            for spec in _specs_of_kind(
                self._declared_input_specs(),
                ArtifactKind.IMAGE,
            )
            if spec.name not in runtime_image_names
        )

    def _external_source_object_names(self) -> tuple[str, ...]:
        runtime_object_names = frozenset(
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.OBJECT_LABELS,
            )
        )
        return tuple(
            spec.name
            for spec in _specs_of_kind(self.inputs, ArtifactKind.OBJECT_LABELS)
            if spec.name not in runtime_object_names
        )

    def _runtime_image_names(self) -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.IMAGE,
            )
        )

    def _declared_input_specs(self) -> tuple[ArtifactSpec, ...]:
        return _unique_specs((*self.inputs, *self.runtime_artifact_inputs))


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerImageExecutionContext(ABC):
    """Shared source provenance for CellProfiler image execution records."""

    source_image_name: str | None
    execution_mode: ImagePayloadExecutionMode = ImagePayloadExecutionMode.NATURAL


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerResolvedInputRequest(CellProfilerImageExecutionContext):
    """Shared source provenance for resolved CellProfiler invocation inputs."""

    image_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerImageRequest(CellProfilerResolvedInputRequest):
    """Resolved image payload and source metadata for one module invocation."""

    payload: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerInvocationRequest(CellProfilerResolvedInputRequest):
    """Resolved invocation inputs for one CellProfiler function call."""

    image: Any
    kwargs: Mapping[str, Any]


class CellProfilerImageExecutionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal executor mode family for CellProfiler image payload semantics."""

    __registry_key__ = "mode"
    __skip_if_no_key__ = True
    mode: ClassVar[ImagePayloadExecutionMode | None] = None

    @classmethod
    def for_mode(
        cls,
        mode: ImagePayloadExecutionMode,
    ) -> "CellProfilerImageExecutionStrategy":
        return cls.__registry__[mode]()

    @abstractmethod
    def execute(
        self,
        executor: "CellProfilerFunctionContractExecutor",
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Execute one resolved image payload according to its nominal mode."""


class NaturalImageExecutionStrategy(CellProfilerImageExecutionStrategy):
    """Delegate natural payloads through the callable processing contract."""

    mode = ImagePayloadExecutionMode.NATURAL

    def execute(
        self,
        executor: "CellProfilerFunctionContractExecutor",
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
    ) -> Any:
        return _processing_contract_for_callable(func).execute(
            executor,
            func,
            image,
            **dict(kwargs),
        )


class FullStackImageExecutionStrategy(CellProfilerImageExecutionStrategy):
    """Execute an already-volumetric payload without per-slice rewriting."""

    mode = ImagePayloadExecutionMode.FULL_STACK

    def execute(
        self,
        executor: "CellProfilerFunctionContractExecutor",
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
    ) -> Any:
        return executor._execute_pure_3d(func, image, **dict(kwargs))


class AlignedMultiImageStackExecutionStrategy(CellProfilerImageExecutionStrategy):
    """Execute aligned multi-image bundles slice-by-slice as a single payload."""

    mode = ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK

    def execute(
        self,
        executor: "CellProfilerFunctionContractExecutor",
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
    ) -> Any:
        return executor._execute_aligned_multi_image_stack(
            func,
            image,
            **dict(kwargs),
        )


class CellProfilerMeasurementImageDomain(Enum):
    """Semantic domain represented by a measurement image argument."""

    SOURCE_IMAGE = "source_image"
    OBJECT_LABELS = "object_labels"


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerMeasurementImage(CellProfilerImageExecutionContext):
    """One resolved image payload used by object measurement modules."""

    payload: Any
    align_to_labels: bool = True
    reference_domain: CellProfilerMeasurementImageDomain = (
        CellProfilerMeasurementImageDomain.SOURCE_IMAGE
    )

    def __post_init__(self) -> None:
        if not isinstance(self.reference_domain, CellProfilerMeasurementImageDomain):
            raise TypeError(
                "CellProfilerMeasurementImage.reference_domain must be "
                "CellProfilerMeasurementImageDomain, got "
                f"{type(self.reference_domain).__name__}."
            )

@dataclass(frozen=True, slots=True)
class CellProfilerSliceAlignedValues:
    """Non-image vector payload with one value array per object-label slice."""

    slices: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        slices = tuple(np.asarray(value) for value in self.slices)
        if not slices:
            raise ValueError("CellProfilerSliceAlignedValues.slices cannot be empty.")
        object.__setattr__(self, "slices", slices)

    @property
    def slice_count(self) -> int:
        return len(self.slices)

    def value_for_slice(self, slice_index: int) -> np.ndarray:
        return self.slices[slice_index]


def _coerce_invocation_kwargs(
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = signature(func).parameters
    coerced_kwargs = _accepted_invocation_kwargs(parameters, kwargs)
    annotations = _callable_type_hints(func)
    for name, value in tuple(coerced_kwargs.items()):
        enum_type = _enum_annotation_type(
            parameters.get(name),
            annotations.get(name),
        )
        if enum_type is None:
            continue
        coerced_kwargs[name] = _coerce_enum_argument(enum_type, value, name)
    return coerced_kwargs


def _accepted_invocation_kwargs(
    parameters: Mapping[str, Parameter],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {
        name: value
        for name, value in kwargs.items()
        if name in parameters or name in _INVOCATION_CONTROL_KWARGS
    }


def _callable_type_hints(func: Callable[..., Any]) -> Mapping[str, Any]:
    try:
        return get_type_hints(func)
    except (NameError, TypeError):
        return {}


def _enum_annotation_type(
    parameter: Any,
    resolved_annotation: Any = None,
) -> type[Enum] | None:
    if parameter is None:
        return None
    annotation = (
        resolved_annotation
        if resolved_annotation is not None
        else parameter.annotation
    )
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    return None


def _coerce_enum_argument(
    enum_type: type[Enum],
    value: Any,
    parameter_name: str,
) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError:
        pass
    if isinstance(value, str):
        normalized_value = re.sub(
            r"[^a-z0-9]+",
            "_",
            value.strip().lower(),
        ).strip("_")
        exact_matches = [
            member
            for member in enum_type
            if normalized_value in _normalized_member_literals(member)
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]

        prefix_matches = [
            member
            for member in enum_type
            if any(
                normalized_value.startswith(candidate)
                or candidate.startswith(normalized_value)
                for candidate in _normalized_member_literals(member)
            )
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

    raise ValueError(
        f"{parameter_name} must be coercible to {enum_type.__name__}; "
        f"got {value!r}."
    )


def _normalized_member_literals(member: Enum) -> tuple[str, ...]:
    return tuple(
        normalized
        for literal in _member_string_literals(member)
        if (
            normalized := re.sub(
                r"[^a-z0-9]+",
                "_",
                literal.strip().lower(),
            ).strip("_")
        )
    )


def _member_string_literals(member: Enum) -> tuple[str, ...]:
    literals = [member.name]
    if isinstance(member.value, str):
        literals.append(member.value)
    elif isinstance(member.value, tuple):
        literals.extend(
            item
            for item in member.value
            if isinstance(item, str)
        )
    return tuple(literals)


@dataclass(frozen=True, slots=True)
class CellProfilerArtifactKindRequest:
    """One artifact-spec request dispatched through a nominal kind strategy."""

    spec: ArtifactSpec
    adapter: CellProfilerRuntimeAdapter
    fallback_image: Any | None = None
    external_image_names: frozenset[str] = frozenset()
    external_object_names: frozenset[str] = frozenset()
    runtime_image_names: frozenset[str] = frozenset()


class CellProfilerArtifactKindStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy family for ArtifactKind-specific runtime semantics."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    kind: ClassVar[ArtifactKind | None] = None

    @classmethod
    def for_kind(cls, kind: ArtifactKind) -> "CellProfilerArtifactKindStrategy":
        return cls.__registry__[kind]()

    @abstractmethod
    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        """Return the runtime payload bound into absorbed function kwargs."""

    @abstractmethod
    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        """Return the transitive source image name for one artifact input."""


class ImageArtifactKindStrategy(CellProfilerArtifactKindStrategy):
    """Resolve image artifact payloads and source-image lineage."""

    kind = ArtifactKind.IMAGE

    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        if request.spec.name in request.runtime_image_names:
            return _cellprofiler_image_payload(
                request.adapter.get_image(request.spec.name).data
            )
        if request.spec.name in request.external_image_names:
            if request.fallback_image is None:
                raise RuntimeError(
                    f"External image input '{request.spec.name}' requires a "
                    "fallback image payload for source-binding resolution."
                )
            return _cellprofiler_image_payload(
                request.adapter.resolve_source_image(
                    request.spec.name,
                    request.fallback_image,
                )
            )
        return _cellprofiler_image_payload(
            request.adapter.get_image(request.spec.name).data
        )

    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        if request.spec.name in request.runtime_image_names:
            return request.adapter.get_image(request.spec.name).source_image_name
        if request.spec.name in request.external_image_names:
            return request.spec.name
        return None


class ObjectLabelsArtifactKindStrategy(CellProfilerArtifactKindStrategy):
    """Resolve object-label payloads and lineage."""

    kind = ArtifactKind.OBJECT_LABELS

    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        if request.spec.name in request.external_object_names:
            if request.fallback_image is None:
                raise RuntimeError(
                    f"External object input '{request.spec.name}' requires a "
                    "fallback image payload for source-binding resolution."
                )
            return _collapse_singleton_label_stack(
                request.adapter.resolve_source_objects(
                    request.spec.name,
                    request.fallback_image,
                ).labels
            )
        return _collapse_singleton_label_stack(
            request.adapter.get_objects(request.spec.name).labels
        )

    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        if request.spec.name in request.external_object_names:
            return request.spec.name
        return request.adapter.get_objects(request.spec.name).source_image_name


class MeasurementsArtifactKindStrategy(CellProfilerArtifactKindStrategy):
    """Resolve measurement payloads and lineage."""

    kind = ArtifactKind.MEASUREMENTS

    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        return request.adapter.get_measurements(request.spec.name).rows

    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        return request.adapter.get_measurements(request.spec.name).source_image_name


class RelationshipsArtifactKindStrategy(CellProfilerArtifactKindStrategy):
    """Resolve relationship payloads."""

    kind = ArtifactKind.RELATIONSHIPS

    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        raise NotImplementedError(
            f"Relationship runtime input '{request.spec.name}' needs an explicit "
            "binding contract before CellProfiler special_inputs can consume it."
        )

    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerInputBindingRequestBase(ABC):
    """Shared runtime context for CellProfiler runtime-input binding."""

    module_name: str
    adapter: CellProfilerRuntimeAdapter
    kwargs: Mapping[str, Any]
    fallback_image: Any
    external_object_names: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))
        object.__setattr__(
            self,
            "external_object_names",
            frozenset(self.external_object_names),
        )

    def labels_for(self, spec: ArtifactSpec) -> Any:
        return _object_input_labels(
            spec,
            self.adapter,
            fallback_image=self.fallback_image,
            external_object_names=self.external_object_names,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerObjectInputBindingRequest(CellProfilerInputBindingRequestBase):
    """Authoritative runtime context for binding CellProfiler object-label inputs."""

    object_inputs: tuple[ArtifactSpec, ...]

    def __post_init__(self) -> None:
        CellProfilerInputBindingRequestBase.__post_init__(self)
        object.__setattr__(self, "object_inputs", tuple(self.object_inputs))

    def with_object_inputs(
        self,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> "CellProfilerObjectInputBindingRequest":
        return type(self)(
            module_name=self.module_name,
            object_inputs=object_inputs,
            adapter=self.adapter,
            kwargs=self.kwargs,
            fallback_image=self.fallback_image,
            external_object_names=self.external_object_names,
        )

    def require_exact_object_count(self, expected_count: int) -> None:
        _require_exact_object_count(
            self.module_name,
            self.object_inputs,
            expected_count,
        )

    def labels_for_inputs(self) -> tuple[Any, ...]:
        return tuple(self.labels_for(spec) for spec in self.object_inputs)

    def measurement_tables_for_primary_object(self) -> tuple[Any, ...]:
        primary_object = self.object_inputs[0] if self.object_inputs else None
        if primary_object is None:
            return ()
        return self.adapter.measurement_tables_for_object(primary_object.name)


class CellProfilerObjectInputPolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal binding policy for CellProfiler object-label inputs."""

    __registry_key__ = _MODULE_NAME_REGISTRY_KEY
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "CellProfilerObjectInputPolicy":
        policy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            UnsupportedObjectInputPolicy,
        )
        return policy_type()

    @abstractmethod
    def bind(
        self,
        request: CellProfilerObjectInputBindingRequest,
    ) -> dict[str, Any]:
        """Return absorbed-function kwargs for object-label runtime inputs."""


class UnsupportedObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Reject undeclared object-input semantics instead of guessing."""

    def bind(
        self,
        request: CellProfilerObjectInputBindingRequest,
    ) -> dict[str, Any]:
        if not request.object_inputs:
            return {}
        raise NotImplementedError(
            f"{request.module_name} has object runtime inputs "
            f"{[spec.name for spec in request.object_inputs]}, but no nominal input "
            "binding policy has been declared for this CellProfiler module."
        )


class SingleObjectLabelInputPolicy(CellProfilerObjectInputPolicy):
    """Bind one object-label input into a module-specific parameter."""

    label_kwarg: ClassVar[str]

    def bind(
        self,
        request: CellProfilerObjectInputBindingRequest,
    ) -> dict[str, Any]:
        request.require_exact_object_count(1)
        return {
            self.label_kwarg: request.labels_for(request.object_inputs[0])
        }


@dataclass(frozen=True, slots=True)
class SingleObjectLabelInputPolicySpec:
    """Declarative leaf spec for one object-label binding policy."""

    module_name: str
    label_kwarg: str


class IdentifyTertiaryObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Bind smaller/larger labels to the absorbed tertiary-object signature."""

    module_name = "IdentifyTertiaryObjects"

    def bind(
        self,
        request: CellProfilerObjectInputBindingRequest,
    ) -> dict[str, Any]:
        request.require_exact_object_count(2)
        larger, smaller = request.object_inputs
        return {
            "primary_labels": request.labels_for(smaller),
            "secondary_labels": request.labels_for(larger),
        }


_MEASURE_OBJECT_SIZE_SHAPE_MODULE = "MeasureObjectSizeShape"
_MEASURE_OBJECT_INTENSITY_MODULE = "MeasureObjectIntensity"
_MEASURE_TEXTURE_MODULE = "MeasureTexture"
_MEASURE_COLOCALIZATION_MODULE = "MeasureColocalization"
_MEASURE_GRANULARITY_MODULE = "MeasureGranularity"
_MEASURE_OBJECT_NEIGHBORS_MODULE = "MeasureObjectNeighbors"


_SINGLE_OBJECT_LABEL_INPUT_POLICY_SPECS = (
    SingleObjectLabelInputPolicySpec("IdentifySecondaryObjects", "primary_labels"),
    SingleObjectLabelInputPolicySpec("Crop", "cropping_labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_OBJECT_SIZE_SHAPE_MODULE, "labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_OBJECT_INTENSITY_MODULE, "labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_TEXTURE_MODULE, "labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_COLOCALIZATION_MODULE, "labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_GRANULARITY_MODULE, "labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_OBJECT_NEIGHBORS_MODULE, "labels"),
)


class DeclaredSingleObjectLabelInputPolicy(SingleObjectLabelInputPolicy):
    """Generated base for modules with one declared label input."""


def _declare_single_object_label_input_policy(
    spec: SingleObjectLabelInputPolicySpec,
) -> None:
    type(
        f"{spec.module_name}InputPolicy",
        (DeclaredSingleObjectLabelInputPolicy,),
        {
            "__module__": __name__,
            "module_name": spec.module_name,
            "label_kwarg": spec.label_kwarg,
        },
    )


for _policy_spec in _SINGLE_OBJECT_LABEL_INPUT_POLICY_SPECS:
    _declare_single_object_label_input_policy(_policy_spec)


class OverlayOutlinesInputPolicy(CellProfilerObjectInputPolicy):
    """Bind ordered object outline rows for the generic overlay runner."""

    module_name = "OverlayOutlines"

    def bind(
        self,
        request: CellProfilerObjectInputBindingRequest,
    ) -> dict[str, Any]:
        return {"object_labels": request.labels_for_inputs()}


class ObjectRowsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind ordered object rows to object-label payloads."""

    def bind(
        self,
        request: CellProfilerObjectInputBindingRequest,
    ) -> dict[str, Any]:
        return {"object_labels": request.labels_for_inputs()}


class ObjectRowsWithMeasurementsInputPolicy(ObjectRowsInputPolicy):
    """Bind ordered object rows plus prior measurements for the primary object."""

    def bind(
        self,
        request: CellProfilerObjectInputBindingRequest,
    ) -> dict[str, Any]:
        bound = super().bind(request)
        bound["measurement_tables"] = request.measurement_tables_for_primary_object()
        return bound


@dataclass(frozen=True, slots=True)
class FilterObjectsRuntimeInputPlan:
    """Runtime object-label partition for one FilterObjects invocation."""

    object_specs: tuple[ArtifactSpec, ...]
    enclosing_spec: ArtifactSpec | None

    @classmethod
    def from_inputs(
        cls,
        object_inputs: tuple[ArtifactSpec, ...],
        kwargs: Mapping[str, Any],
    ) -> "FilterObjectsRuntimeInputPlan":
        object_count = int(kwargs.get("additional_object_count", 0)) + 1
        enclosing_name = kwargs.get("enclosing_object_name")
        object_specs = object_inputs[:object_count]
        enclosing_spec = None
        if enclosing_name is not None:
            enclosing_spec = _spec_by_name(object_inputs, str(enclosing_name))
            if enclosing_spec is None:
                raise RuntimeError(
                    "FilterObjects enclosing object input "
                    f"{enclosing_name!r} was not declared in the runtime contract."
                )
        return cls(object_specs=object_specs, enclosing_spec=enclosing_spec)


class MeasureImageAreaOccupiedInputPolicy(ObjectRowsInputPolicy):
    """Bind ordered object rows for the generic area-occupied runner."""

    module_name = "MeasureImageAreaOccupiedBinary"


class FilterObjectsInputPolicy(ObjectRowsWithMeasurementsInputPolicy):
    """Bind ordered primary/additional object rows for FilterObjects."""

    module_name = "FilterObjects"

    def bind(
        self,
        request: CellProfilerObjectInputBindingRequest,
    ) -> dict[str, Any]:
        plan = FilterObjectsRuntimeInputPlan.from_inputs(
            request.object_inputs,
            request.kwargs,
        )
        bound = super().bind(request.with_object_inputs(plan.object_specs))
        if plan.enclosing_spec is not None:
            bound["enclosing_object_labels"] = request.labels_for(plan.enclosing_spec)
        return bound


class CalculateMathInputPolicy(CellProfilerObjectInputPolicy):
    """Bind CalculateMath operands from runtime measurement/object state."""

    module_name = "CalculateMath"

    def bind(
        self,
        request: CellProfilerObjectInputBindingRequest,
    ) -> dict[str, Any]:
        return {
            "operand1_value": _calculate_math_operand_value(
                request.adapter,
                request.kwargs,
                feature_kwarg="operand1_feature",
                object_kwarg="operand1_object_name",
            ),
            "operand2_value": _calculate_math_operand_value(
                request.adapter,
                request.kwargs,
                feature_kwarg="operand2_feature",
                object_kwarg="operand2_object_name",
            ),
        }


class CellProfilerPerObjectMeasurementPolicy:
    """Predicate for modules that need one absorbed call per object set."""

    module_names: ClassVar[tuple[str, ...]] = (
        _MEASURE_OBJECT_SIZE_SHAPE_MODULE,
        _MEASURE_OBJECT_INTENSITY_MODULE,
        _MEASURE_TEXTURE_MODULE,
        _MEASURE_COLOCALIZATION_MODULE,
        _MEASURE_GRANULARITY_MODULE,
    )
    independent_image_modules: ClassVar[tuple[str, ...]] = (
        _MEASURE_OBJECT_INTENSITY_MODULE,
        _MEASURE_TEXTURE_MODULE,
        _MEASURE_GRANULARITY_MODULE,
    )

    @classmethod
    def matches(
        cls,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> bool:
        return canonical_module_name(module_name) in cls.module_names and bool(
            object_inputs
        )

    @classmethod
    def measures_images_independently(cls, module_name: str) -> bool:
        return canonical_module_name(module_name) in cls.independent_image_modules


@dataclass(frozen=True, slots=True)
class CellProfilerOutputRecordRequest:
    """Inputs needed to record one declared CellProfiler artifact output."""

    executor: CellProfilerModuleExecutor
    adapter: CellProfilerRuntimeAdapter
    spec: ArtifactSpec
    value: Any
    output_values: Mapping[str, Any]
    source_image_name: str | None
    func: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementRecord:
    """Rows and semantic owner for one CellProfiler measurement output."""

    rows: list[Any]
    object_name: str | None
    fields: tuple[FieldSpec, ...] = ()


class CellProfilerMeasurementRecordBuilder(ABC, metaclass=AutoRegisterMeta):
    """Nominal module-specific measurement-row enrichment."""

    __registry_key__ = _MODULE_NAME_REGISTRY_KEY
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(
        cls,
        module_name: str,
    ) -> "CellProfilerMeasurementRecordBuilder":
        builder_type = cls.__registry__.get(
            canonical_module_name(module_name),
            DefaultMeasurementRecordBuilder,
        )
        return builder_type()

    @abstractmethod
    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        """Return measurement rows plus the object set they describe."""


class DefaultMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Use the emitted rows and infer object ownership from declared inputs."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows = _measurement_table_rows(request.value)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=_measurement_object_name(
                request.executor._declared_input_specs()
            ),
            fields=_measurement_record_fields(rows, request.func),
        )


class RelateObjectsMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose CellProfiler parent-scoped relationship measurements."""

    module_name = "RelateObjects"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        parent_spec, child_spec = _relationship_object_inputs(request)
        return CellProfilerMeasurementRecord(
            rows=[
                *_measurement_table_rows(request.value),
                *_relationship_child_count_rows(
                    request,
                    parent_object_name=parent_spec.name,
                    child_object_name=child_spec.name,
                    payload=_relationship_payload(request),
                ),
            ],
            object_name=parent_spec.name,
        )


class CellProfilerOutputRecorder(ABC, metaclass=AutoRegisterMeta):
    """Nominal output writer selected by artifact kind."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    kind: ClassVar[ArtifactKind | None] = None

    @classmethod
    def for_kind(cls, kind: ArtifactKind) -> "CellProfilerOutputRecorder":
        recorder_type = cls.__registry__.get(kind)
        if recorder_type is None:
            raise TypeError(f"Unsupported CellProfiler output kind {kind.value}.")
        return recorder_type()

    @abstractmethod
    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        """Record one output artifact through the runtime adapter."""


class ImageOutputRecorder(CellProfilerOutputRecorder):
    """Record image outputs."""

    kind = ArtifactKind.IMAGE

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_image(
            request.spec.name,
            request.value,
            source_image_name=request.source_image_name,
        )


class ObjectLabelsOutputRecorder(CellProfilerOutputRecorder):
    """Record object-label outputs."""

    kind = ArtifactKind.OBJECT_LABELS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_objects(
            request.spec.name,
            request.value,
            source_image_name=request.source_image_name,
        )


class MeasurementsOutputRecorder(CellProfilerOutputRecorder):
    """Record measurement outputs with inferred image/object ownership."""

    kind = ArtifactKind.MEASUREMENTS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        measurement_record = CellProfilerMeasurementRecordBuilder.for_module(
            request.executor.module_name
        ).build(request)
        _record_measurements(
            request.adapter,
            request.spec.name,
            measurement_record.rows,
            fields=measurement_record.fields,
            object_name=measurement_record.object_name,
            source_image_name=request.source_image_name,
        )


class RelationshipsOutputRecorder(CellProfilerOutputRecorder):
    """Record parent-child relationship artifacts."""

    kind = ArtifactKind.RELATIONSHIPS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        if not isinstance(request.value, CellProfilerRelationshipPayload):
            raise TypeError(
                f"{request.executor.module_name} relationship output "
                f"'{request.spec.name}' must be CellProfilerRelationshipPayload, "
                f"got {type(request.value).__name__}."
            )
        parent_spec, child_spec = _relationship_object_inputs(request)
        request.adapter.add_relationship(
            request.spec.name,
            parent_object_name=parent_spec.name,
            child_object_name=child_spec.name,
            parent_ids=request.value.parent_ids,
            child_ids=request.value.child_ids,
        )


def _output_values_by_kind(
    output_specs: tuple[ArtifactSpec, ...],
    main_output: Any,
    artifact_values: tuple[Any, ...],
) -> dict[str, Any]:
    if len(output_specs) == 1:
        return {
            output_specs[0].name: _single_output_value(
                output_specs[0],
                main_output,
                artifact_values,
            )
        }

    if (
        output_specs
        and output_specs[0].kind is ArtifactKind.IMAGE
        and len(output_specs) == len(artifact_values) + 1
    ):
        return {
            output_specs[0].name: main_output,
            **{
                spec.name: value
                for spec, value in zip(
                    output_specs[1:],
                    artifact_values,
                    strict=True,
                )
            },
        }

    if len(output_specs) != len(artifact_values):
        raise ValueError(
            f"CellProfiler module declared {len(output_specs)} outputs but "
            f"returned {len(artifact_values)} artifact values."
        )
    return {
        spec.name: value
        for spec, value in zip(output_specs, artifact_values, strict=True)
    }


def _single_output_value(
    spec: ArtifactSpec,
    main_output: Any,
    artifact_values: tuple[Any, ...],
) -> Any:
    if spec.kind is ArtifactKind.IMAGE:
        return main_output
    if not artifact_values:
        raise ValueError(
            f"CellProfiler module did not return a value for output '{spec.name}'."
        )
    if spec.kind is ArtifactKind.OBJECT_LABELS:
        return artifact_values[-1]
    return artifact_values[0]


def _split_cellprofiler_output(raw_output: Any) -> tuple[Any, tuple[Any, ...]]:
    if isinstance(raw_output, tuple):
        return raw_output[0], tuple(raw_output[1:])
    return raw_output, ()


def _measurement_rows_from_output(artifact_values: tuple[Any, ...]) -> list[Any]:
    if not artifact_values:
        return []
    rows = artifact_values[0]
    return _measurement_table_rows(rows)


def _measurement_table_rows(rows: Any) -> list[Any]:
    if isinstance(rows, list):
        return rows
    if isinstance(rows, tuple):
        return list(rows)
    return [rows]


_MISSING_MEASUREMENT_OBJECT_NAME = object()


def _record_measurements(
    adapter: CellProfilerRuntimeAdapter,
    name: str,
    rows: Sequence[Any],
    *,
    fields: tuple[FieldSpec, ...] = (),
    object_name: str | None | object = _MISSING_MEASUREMENT_OBJECT_NAME,
    source_image_name: str | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "source_image_name": source_image_name,
    }
    if object_name is not _MISSING_MEASUREMENT_OBJECT_NAME:
        kwargs["object_name"] = object_name
    if fields:
        kwargs["fields"] = fields
    adapter.add_measurements(name, rows, **kwargs)


def _measurement_record_fields(
    rows: Sequence[Any],
    func: Callable[..., Any],
) -> tuple[FieldSpec, ...]:
    if _rows_have_inferable_fields(rows):
        return ()
    return _measurement_fields_from_callable(func)


def _rows_have_inferable_fields(rows: Sequence[Any]) -> bool:
    if not rows:
        return False
    row = rows[0]
    return bool(is_dataclass(row) or (isinstance(row, Mapping) and row))


def _measurement_fields_from_callable(
    func: Callable[..., Any],
) -> tuple[FieldSpec, ...]:
    return_type = _callable_type_hints(unwrap(func)).get("return")
    row_type = _measurement_row_type_from_annotation(return_type)
    if row_type is None:
        return ()
    return tuple(FieldSpec(field.name) for field in dataclass_fields(row_type))


def _measurement_row_type_from_annotation(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, tuple):
        return _measurement_row_type_from_sequence_args(args)
    return None


def _measurement_row_type_from_sequence_args(
    args: tuple[Any, ...],
) -> type[Any] | None:
    for arg in args:
        if arg is Ellipsis:
            continue
        row_type = _measurement_row_type_from_annotation(arg)
        if row_type is not None:
            return row_type
    return None


def _object_only_reference_image(image: Any) -> Any:
    """Use one plane to drive object-only CellProfiler modules once.

    Object-only modules consume runtime object artifacts; the image argument is a
    carrier required by the absorbed function signature, not the semantic domain
    to iterate over. Running them over every channel slice duplicates object
    artifacts and corrupts downstream measurement alignment.
    """
    if is_color_image_stack(image):
        return image[0, :, :, 0]
    if is_color_image_slice(image):
        return image[:, :, 0]
    if hasattr(image, "ndim") and image.ndim == 3 and image.shape[0] >= 1:
        return image[0]
    return image


def _measurement_image_for_labels(
    image: Any,
    labels: Any,
    *,
    reference_domain: CellProfilerMeasurementImageDomain = (
        CellProfilerMeasurementImageDomain.SOURCE_IMAGE
    ),
) -> Any:
    """Align a measurement reference image to one object-label payload.

    Many absorbed CellProfiler measurement functions expect a 2D intensity image
    paired with one 2D object-label set. When the OpenHCS main flow is carrying a
    higher-level stack for the whole image set, use a single reference slice
    instead of handing the raw multi-slice stack to functions that require shape
    parity with the labels.
    """
    if not isinstance(reference_domain, CellProfilerMeasurementImageDomain):
        raise TypeError(
            "_measurement_image_for_labels.reference_domain must be "
            "CellProfilerMeasurementImageDomain, got "
            f"{type(reference_domain).__name__}."
        )
    if not hasattr(image, "ndim") or not hasattr(labels, "ndim"):
        return image
    aligned_image = image
    if is_color_image_stack(image):
        if labels.ndim == 3:
            aligned_image = image[..., 0]
        elif labels.ndim == 2:
            aligned_image = image[0, :, :, 0]
    elif is_color_image_slice(image) and labels.ndim == 2:
        aligned_image = image[:, :, 0]
    elif image.ndim == labels.ndim:
        aligned_image = image
    elif image.ndim == labels.ndim + 1 and getattr(image, "shape", (0,))[0] >= 1:
        aligned_image = image[0]

    if (
        reference_domain is CellProfilerMeasurementImageDomain.OBJECT_LABELS
        and _measurement_image_shape_mismatches_labels(aligned_image, labels)
    ):
        return _object_label_domain_reference_image(aligned_image, labels)
    return aligned_image


def _measurement_image_shape_mismatches_labels(image: Any, labels: Any) -> bool:
    if not hasattr(image, "shape") or not hasattr(labels, "shape"):
        return False
    return tuple(image.shape) != tuple(labels.shape)


def _object_label_domain_reference_image(image: Any, labels: Any) -> Any:
    if not hasattr(labels, "shape"):
        return image
    return np.zeros(tuple(labels.shape), dtype=getattr(image, "dtype", np.float32))


def _measurement_labels(labels: Any) -> Any:
    """Normalize singleton stack labels for absorbed 2D measurement functions."""
    return _collapse_singleton_label_stack(labels)


def _measurement_labels_for_image(image: Any, labels: Any) -> Any:
    """Align object-label payload rank to the selected measurement image."""
    labels = _measurement_labels(labels)
    if not hasattr(image, "ndim") or not hasattr(labels, "ndim"):
        return labels
    if labels.ndim == 3 and image.ndim == 2:
        return labels[0]
    if (
        labels.ndim == 3
        and image.ndim == 3
        and getattr(image, "shape", (0,))[0] == 1
        and labels.shape[1:] == image.shape[1:]
    ):
        return labels[0]
    return labels


def _collapse_singleton_label_stack(labels: Any) -> Any:
    """Normalize singleton OpenHCS label stacks to one CellProfiler label plane."""
    if not hasattr(labels, "ndim"):
        return labels
    if labels.ndim == 3 and getattr(labels, "shape", (0,))[0] == 1:
        return labels[0]
    return labels


def _specs_of_kind(
    specs: Sequence[ArtifactSpec],
    kind: ArtifactKind,
) -> tuple[ArtifactSpec, ...]:
    return tuple(spec for spec in specs if spec.kind is kind)


def _spec_by_name(
    specs: Sequence[ArtifactSpec],
    name: str,
) -> ArtifactSpec | None:
    for spec in specs:
        if spec.name == name:
            return spec
    return None


def _unique_specs(specs: Sequence[ArtifactSpec]) -> tuple[ArtifactSpec, ...]:
    unique: dict[tuple[str, ArtifactKind], ArtifactSpec] = {}
    for spec in specs:
        key = (spec.name, spec.kind)
        existing = unique.get(key)
        if existing is not None and existing != spec:
            raise ValueError(
                f"Conflicting CellProfiler input spec declarations for "
                f"{spec.kind.value}:{spec.name}."
            )
        unique[key] = spec
    return tuple(unique.values())


def _require_exact_object_count(
    module_name: str,
    object_inputs: tuple[ArtifactSpec, ...],
    expected_count: int,
) -> None:
    if len(object_inputs) != expected_count:
        raise NotImplementedError(
            f"{module_name} requires {expected_count} object runtime input(s), "
            f"got {[spec.name for spec in object_inputs]}."
        )


def _object_input_labels(
    spec: ArtifactSpec,
    adapter: CellProfilerRuntimeAdapter,
    *,
    fallback_image: Any,
    external_object_names: frozenset[str],
) -> Any:
    if spec.name in external_object_names:
        return adapter.resolve_source_objects(spec.name, fallback_image).labels
    return adapter.get_objects(spec.name).labels


def _measurement_object_name(
    inputs: tuple[ArtifactSpec, ...],
) -> str | None:
    object_inputs = _specs_of_kind(inputs, ArtifactKind.OBJECT_LABELS)
    if len(object_inputs) == 1:
        return object_inputs[0].name
    return None


def _relationship_object_inputs(
    request: CellProfilerOutputRecordRequest,
) -> tuple[ArtifactSpec, ArtifactSpec]:
    object_inputs = _specs_of_kind(
        request.executor._declared_input_specs(),
        ArtifactKind.OBJECT_LABELS,
    )
    if len(object_inputs) != 2:
        raise NotImplementedError(
            f"{request.executor.module_name} relationship semantics require "
            f"exactly two object inputs, got {[spec.name for spec in object_inputs]}."
        )
    return object_inputs[0], object_inputs[1]


def _relationship_payload(
    request: CellProfilerOutputRecordRequest,
) -> CellProfilerRelationshipPayload:
    payloads = tuple(
        value
        for value in request.output_values.values()
        if isinstance(value, CellProfilerRelationshipPayload)
    )
    if len(payloads) != 1:
        raise ValueError(
            f"{request.executor.module_name} measurement enrichment expected one "
            f"relationship payload, got {len(payloads)}."
        )
    return payloads[0]


def _relationship_child_count_rows(
    request: CellProfilerOutputRecordRequest,
    *,
    parent_object_name: str,
    child_object_name: str,
    payload: CellProfilerRelationshipPayload,
) -> tuple[dict[str, int], ...]:
    related_parent_ids = tuple(int(parent_id) for parent_id in payload.parent_ids)
    parent_count = max(
        (
            _object_label_count(request.adapter, parent_object_name),
            *related_parent_ids,
        )
    )
    counts = {parent_id: 0 for parent_id in range(1, parent_count + 1)}
    for parent_id in related_parent_ids:
        if parent_id > 0:
            counts[parent_id] = counts.get(parent_id, 0) + 1
    feature_name = f"Children_{child_object_name}_Count"
    return tuple(
        {
            "object_label": parent_id,
            feature_name: count,
        }
        for parent_id, count in counts.items()
    )


def _object_label_count(
    adapter: CellProfilerRuntimeAdapter,
    object_name: str,
) -> int:
    return int(adapter.get_objects(object_name).labels.max())


def _slice_aligned_measurement_values(
    value_slices: tuple[np.ndarray, ...],
) -> np.ndarray | CellProfilerSliceAlignedValues:
    if len(value_slices) == 1:
        return value_slices[0]
    return CellProfilerSliceAlignedValues(value_slices)


@dataclass(frozen=True, slots=True, kw_only=True)
class CellProfilerSpecialInputBindingRequest(CellProfilerInputBindingRequestBase):
    """Authoritative runtime context for binding CellProfiler special_inputs."""

    parameter_names: tuple[str, ...]
    runtime_inputs: tuple[ArtifactSpec, ...]
    external_image_names: frozenset[str]
    runtime_image_names: frozenset[str]

    def __post_init__(self) -> None:
        CellProfilerInputBindingRequestBase.__post_init__(self)
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(self, "runtime_inputs", tuple(self.runtime_inputs))
        object.__setattr__(
            self,
            "external_image_names",
            frozenset(self.external_image_names),
        )
        object.__setattr__(
            self,
            "runtime_image_names",
            frozenset(self.runtime_image_names),
        )

    @property
    def object_inputs(self) -> tuple[ArtifactSpec, ...]:
        return _specs_of_kind(self.runtime_inputs, ArtifactKind.OBJECT_LABELS)


class CellProfilerSpecialInputPolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal module-specific binding for CellProfiler special_inputs."""

    __registry_key__ = _MODULE_NAME_REGISTRY_KEY
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(
        cls,
        module_name: str,
    ) -> "CellProfilerSpecialInputPolicy":
        policy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            PositionalSpecialInputPolicy,
        )
        return policy_type()

    def special_image_inputs(
        self,
        module_name: str,
        func: Callable[..., Any],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return trailing image specs consumed by special_inputs instead of primary image payload."""

        return _signature_special_image_inputs(module_name, func, declared_inputs)

    @abstractmethod
    def bind(
        self,
        request: CellProfilerSpecialInputBindingRequest,
    ) -> dict[str, Any]:
        """Return kwargs for a callable's declared special_inputs."""


class PositionalSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Bind special_inputs positionally to compiled runtime artifact specs."""

    def bind(
        self,
        request: CellProfilerSpecialInputBindingRequest,
    ) -> dict[str, Any]:
        return _bind_special_runtime_inputs(request)


class DisplayDataOnImageSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Resolve display annotations from object labels and measurement tables."""

    module_name = "DisplayDataOnImage"

    def special_image_inputs(
        self,
        module_name: str,
        func: Callable[..., Any],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func, declared_inputs
        return ()

    def bind(
        self,
        request: CellProfilerSpecialInputBindingRequest,
    ) -> dict[str, Any]:
        object_inputs = request.object_inputs
        _require_exact_object_count(request.module_name, object_inputs, 1)
        object_spec = object_inputs[0]
        labels = request.labels_for(object_spec)
        feature_name = _required_string_kwarg(
            request.kwargs,
            "measurement_feature",
            request.module_name,
        )
        return {
            "labels": labels,
            "measurements": _slice_aligned_measurement_values(
                measurement_values_for_label_slices(
                    request.adapter.measurement_tables_for_object(object_spec.name),
                    feature_name,
                    labels,
                    object_name=object_spec.name,
                )
            ),
        }


class ClassifyObjectsMeasurementInputPolicy(CellProfilerSpecialInputPolicy):
    """Resolve ClassifyObjects label and measurement-vector inputs."""

    measurement_kwarg_by_parameter: ClassVar[Mapping[str, str]] = {
        "measurement_values": "measurement_feature",
        "measurement1_values": "measurement1_feature",
        "measurement2_values": "measurement2_feature",
    }

    def special_image_inputs(
        self,
        module_name: str,
        func: Callable[..., Any],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func, declared_inputs
        return ()

    def bind(
        self,
        request: CellProfilerSpecialInputBindingRequest,
    ) -> dict[str, Any]:
        object_inputs = request.object_inputs
        _require_exact_object_count(request.module_name, object_inputs, 1)
        object_spec = object_inputs[0]
        labels = request.labels_for(object_spec)
        return {
            "labels": labels,
            **{
                parameter_name: _slice_aligned_measurement_values(
                    measurement_values_for_label_slices(
                        request.adapter.measurement_tables_for_object(
                            object_spec.name
                        ),
                        _required_string_kwarg(
                            request.kwargs,
                            kwarg_name,
                            request.module_name,
                        ),
                        labels,
                        object_name=object_spec.name,
                    )
                )
                for parameter_name, kwarg_name in (
                    type(self).measurement_kwarg_by_parameter.items()
                )
                if kwarg_name in request.kwargs
            },
        }


class ClassifyObjectsSingleMeasurementInputPolicy(ClassifyObjectsMeasurementInputPolicy):
    module_name = "ClassifyObjectsSingleMeasurement"


class ClassifyObjectsTwoMeasurementsInputPolicy(ClassifyObjectsMeasurementInputPolicy):
    module_name = "ClassifyObjectsTwoMeasurements"


def _signature_special_image_inputs(
    module_name: str,
    func: Callable[..., Any],
    declared_inputs: tuple[ArtifactSpec, ...],
) -> tuple[ArtifactSpec, ...]:
    image_inputs = _specs_of_kind(declared_inputs, ArtifactKind.IMAGE)
    special_input_count = len(special_input_names_from_callable(func))
    non_image_count = len(
        tuple(spec for spec in declared_inputs if spec.kind is not ArtifactKind.IMAGE)
    )
    special_image_count = max(0, special_input_count - non_image_count)
    if special_image_count == 0:
        return ()
    if special_image_count > len(image_inputs):
        raise NotImplementedError(
            f"{module_name} declares {special_image_count} image special "
            f"input(s), but only has image inputs {[spec.name for spec in image_inputs]}."
        )
    return image_inputs[-special_image_count:]


def _required_string_kwarg(
    kwargs: Mapping[str, Any],
    name: str,
    module_name: str,
) -> str:
    value = kwargs.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{module_name} requires non-empty kwarg {name!r}.")
    return value


def _optional_string_kwarg(
    kwargs: Mapping[str, Any],
    name: str,
) -> str | None:
    value = kwargs.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Expected string kwarg {name!r}, got {type(value).__name__}.")
    normalized = value.strip()
    return normalized or None


def _calculate_math_operand_value(
    adapter: CellProfilerRuntimeAdapter,
    kwargs: Mapping[str, Any],
    *,
    feature_kwarg: str,
    object_kwarg: str,
) -> Any:
    feature_name = _required_string_kwarg(kwargs, feature_kwarg, "CalculateMath")
    object_name = _optional_string_kwarg(kwargs, object_kwarg)
    count_object_name = count_feature_object_name(feature_name)
    if count_object_name is not None:
        return float(_object_label_count(adapter, count_object_name))
    if object_name is None:
        raise NotImplementedError(
            f"CalculateMath feature {feature_name!r} is not a Count_* object "
            "measurement and has no object subject."
        )
    values = measurement_values_for_feature(
        adapter.measurement_tables_for_object(object_name),
        feature_name,
        object_count=_object_label_count(adapter, object_name),
        object_name=object_name,
    )
    return float(values[0]) if len(values) == 1 else values


@dataclass(frozen=True, slots=True)
class CellProfilerPerImageMeasurementRequest:
    """Contract shape used to decide image-measurement invocation cardinality."""

    module_name: str
    func: Callable[..., Any]
    image_inputs: tuple[ArtifactSpec, ...]
    object_inputs: tuple[ArtifactSpec, ...]
    outputs: tuple[ArtifactSpec, ...]


class CellProfilerPerImageMeasurementPolicy:
    """Predicate for image measurements that execute once per named image."""

    @classmethod
    def matches(cls, request: CellProfilerPerImageMeasurementRequest) -> bool:
        if request.object_inputs or not request.image_inputs:
            return False
        measurement_outputs = _specs_of_kind(
            request.outputs,
            ArtifactKind.MEASUREMENTS,
        )
        if len(measurement_outputs) != 1:
            return False
        if len(request.outputs) != len(measurement_outputs):
            return False
        return not _callable_accepts_composed_image_payload(request.func)


_COMPOSED_IMAGE_PAYLOAD_PARAMETERS = frozenset(
    (
        "channel_1",
        "channel_2",
        "input_names",
        "operand_choices",
        "retained_image_names",
    )
)


def _callable_accepts_composed_image_payload(func: Callable[..., Any]) -> bool:
    """Return whether callable parameters describe a multi-image bundle contract."""
    parameters = signature(func).parameters
    return any(
        parameter_name in parameters
        for parameter_name in _COMPOSED_IMAGE_PAYLOAD_PARAMETERS
    )


def _bind_special_runtime_inputs(
    request: CellProfilerSpecialInputBindingRequest,
) -> dict[str, Any]:
    if len(request.parameter_names) != len(request.runtime_inputs):
        raise NotImplementedError(
            f"{request.module_name} declares special_inputs "
            f"{list(request.parameter_names)}, but compiled runtime inputs are "
            f"{[spec.name for spec in request.runtime_inputs]}."
        )
    return {
        parameter_name: _runtime_input_value(spec, request)
        for parameter_name, spec in zip(
            request.parameter_names,
            request.runtime_inputs,
            strict=True,
        )
    }


def _runtime_input_value(
    spec: ArtifactSpec,
    request: CellProfilerSpecialInputBindingRequest,
) -> Any:
    try:
        return _artifact_kind_strategy(spec.kind).runtime_input_value(
            CellProfilerArtifactKindRequest(
                spec=spec,
                adapter=request.adapter,
                fallback_image=request.fallback_image,
                external_image_names=request.external_image_names,
                external_object_names=request.external_object_names,
                runtime_image_names=request.runtime_image_names,
            )
        )
    except KeyError as exc:
        raise TypeError(
            f"Unsupported special runtime input kind {spec.kind.value} for "
            f"'{spec.name}'."
        ) from exc


def _artifact_kind_strategy(
    kind: ArtifactKind,
) -> CellProfilerArtifactKindStrategy:
    try:
        return CellProfilerArtifactKindStrategy.for_kind(kind)
    except KeyError as exc:
        raise TypeError(
            f"No CellProfiler artifact kind strategy registered for {kind.value}."
        ) from exc


def _collapse_singleton_stack_output(value: Any) -> Any:
    if hasattr(value, "ndim") and value.ndim == 3 and value.shape[0] == 1:
        return value[0]
    if isinstance(value, tuple):
        return tuple(_collapse_singleton_stack_output(item) for item in value)
    return value


def _openhcs_main_flow_output(
    input_image: Any,
    output_image: Any,
) -> Any:
    if not is_image_stack(input_image):
        return output_image
    if not _is_image_slice(output_image):
        return output_image
    memory_type = detect_memory_type(input_image)
    return ImageStackLayout.for_slices((output_image,)).stack(
        slices=(output_image,),
        memory_type=memory_type,
        gpu_id=0,
    )


def _is_image_slice(value: Any) -> bool:
    return (hasattr(value, "ndim") and value.ndim == 2) or is_color_image_slice(value)


def _single_source_name(source_names: tuple[str, ...]) -> str | None:
    unique_names = tuple(dict.fromkeys(source_names))
    if len(unique_names) == 1:
        return unique_names[0]
    return None


def _stack_cellprofiler_slice_outputs(
    slice_outputs: Sequence[Any],
    memory_type: str,
) -> Any:
    if all(_is_grayscale_slice_output(output) for output in slice_outputs):
        return stack_slices(list(slice_outputs), memory_type, 0)
    if all(is_color_image_slice(output) for output in slice_outputs):
        stacked = np.stack(
            tuple(
                _as_numpy_payload(output)
                for output in slice_outputs
            )
        )
        if memory_type == MEMORY_TYPE_NUMPY:
            return stacked
        return _convert_memory(stacked, MEMORY_TYPE_NUMPY, memory_type)
    raise ValueError(
        "CellProfiler slice outputs must be uniformly 2D grayscale or HWC "
        "color images; got shapes "
        f"{[getattr(output, 'shape', None) for output in slice_outputs]!r}."
    )


def _unstack_cellprofiler_image_slices(image: Any, memory_type: str) -> tuple[Any, ...]:
    if is_color_image_slice(image):
        return (image,)
    if is_color_image_stack(image):
        source_type = detect_memory_type(image)
        if source_type != memory_type:
            image = _convert_memory(image, source_type, memory_type)
        return tuple(image[index] for index in range(image.shape[0]))
    return tuple(unstack_slices(image, memory_type, 0))


def _is_grayscale_slice_output(output: Any) -> bool:
    return np.asarray(output).ndim == 2


def _as_numpy_payload(payload: Any) -> np.ndarray:
    source_type = detect_memory_type(payload)
    if source_type == MEMORY_TYPE_NUMPY:
        return payload
    return _convert_memory(payload, source_type, MEMORY_TYPE_NUMPY)


def _convert_memory(
    data: Any,
    source_type: str,
    target_type: str,
) -> Any:
    return convert_memory(
        data=data,
        source_type=source_type,
        target_type=target_type,
        gpu_id=0,
    )


def _requested_image_execution_mode(
    *,
    force_full_stack: bool,
    execution_mode: ImagePayloadExecutionMode | None,
) -> ImagePayloadExecutionMode:
    if execution_mode is not None:
        return execution_mode
    if force_full_stack:
        return ImagePayloadExecutionMode.FULL_STACK
    return ImagePayloadExecutionMode.NATURAL


class CellProfilerFunctionContractExecutor:
    """Apply OpenHCS processing contracts after CellProfiler input resolution."""

    def execute(
        self,
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
        *,
        force_full_stack: bool = False,
        execution_mode: ImagePayloadExecutionMode | None = None,
    ) -> Any:
        mode = _requested_image_execution_mode(
            force_full_stack=force_full_stack,
            execution_mode=execution_mode,
        )
        return CellProfilerImageExecutionStrategy.for_mode(mode).execute(
            self,
            func,
            image,
            kwargs,
        )

    def _execute_pure_3d(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        return func(image, **kwargs)

    def _execute_aligned_multi_image_stack(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(image, AlignedImageStack):
            raise TypeError(
                "ALIGNED_MULTI_IMAGE_STACK execution requires "
                f"AlignedImageStack, got {type(image).__name__}."
            )
        slice_results = tuple(
            _rewrite_slice_index(
                _collapse_singleton_stack_output(
                    func(
                        slice_payload,
                        **aligned_image_stack_kwargs(
                            kwargs,
                            slice_index,
                            len(image.slices),
                        ),
                    )
                ),
                slice_index,
            )
            for slice_index, slice_payload in enumerate(image.slices)
        )
        main_outputs, auxiliary_groups = _pure_2d_slice_results(slice_results)
        memory_type = detect_memory_type(main_outputs[0])
        stacked_main_output = _stack_cellprofiler_slice_outputs(
            main_outputs,
            memory_type,
        )
        if not auxiliary_groups:
            return stacked_main_output
        return (
            stacked_main_output,
            *(
                _aggregate_pure_2d_auxiliary_output(values, memory_type)
                for values in auxiliary_groups
            ),
        )

    def _execute_pure_2d(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        if not hasattr(image, "ndim"):
            return func(image, **kwargs)

        memory_type = detect_memory_type(image)
        if image.ndim == 2:
            slice_count = _slice_count_from_pure_2d_kwargs(kwargs)
            if slice_count is None:
                return func(image, **kwargs)
            slices_2d = tuple(image for _ in range(slice_count))
        elif is_color_image_slice(image):
            slice_count = _slice_count_from_pure_2d_kwargs(kwargs)
            slices_2d = tuple(image for _ in range(slice_count or 1))
        else:
            slices_2d = _unstack_cellprofiler_image_slices(image, memory_type)

        slice_count = len(slices_2d)
        slice_results = [
            _rewrite_slice_index(
                func(
                    slice_2d,
                    **_slice_pure_2d_kwargs(kwargs, slice_index, slice_count),
                ),
                slice_index,
            )
            for slice_index, slice_2d in enumerate(slices_2d)
        ]
        main_outputs, auxiliary_groups = _pure_2d_slice_results(slice_results)
        stacked_main_output = _stack_cellprofiler_slice_outputs(
            main_outputs,
            memory_type,
        )
        if not auxiliary_groups:
            return stacked_main_output
        return (
            stacked_main_output,
            *(
                _aggregate_pure_2d_auxiliary_output(values, memory_type)
                for values in auxiliary_groups
            ),
        )

    def _execute_flexible(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        slice_by_slice = bool(kwargs.pop("slice_by_slice", False))
        if slice_by_slice:
            return self._execute_pure_2d(func, image, **kwargs)
        return self._execute_pure_3d(func, image, **kwargs)

    def _execute_volumetric_to_slice(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        result_2d = func(image, **kwargs)
        memory_type = detect_memory_type(result_2d)
        return stack_slices([result_2d], memory_type, 0)


def _processing_contract_for_callable(func: Callable[..., Any]) -> ProcessingContract:
    contract = CallableContract.from_callable(func)
    if isinstance(contract.processing_contract, ProcessingContract):
        return contract.processing_contract
    if contract.declared_processing_contract == "unknown":
        inferred = _infer_processing_contract(func)
        if inferred is not None:
            return inferred
    if contract.declared_processing_contract is not None:
        declared = ProcessingContract.from_declared_name(
            contract.declared_processing_contract
        )
        if declared is not None:
            return declared
    return ProcessingContract.FLEXIBLE


def _infer_processing_contract(
    func: Callable[..., Any],
) -> ProcessingContract | None:
    inferred = infer_contract(func, dtype_config=DtypeConfig()).contract
    if inferred is InferredContract.UNKNOWN or inferred is InferredContract.ERROR:
        return None
    return ProcessingContract.from_declared_name(inferred.name)


def _slice_pure_2d_kwargs(
    kwargs: Mapping[str, Any],
    slice_index: int,
    slice_count: int,
) -> dict[str, Any]:
    return {
        name: _slice_pure_2d_value(value, slice_index, slice_count)
        for name, value in kwargs.items()
    }


def _slice_count_from_pure_2d_kwargs(
    kwargs: Mapping[str, Any],
) -> int | None:
    slice_counts = {
        stack.shape[0]
        for value in kwargs.values()
        if (stack := _slice_aligned_stack_view(value)) is not None
        and stack.shape[0] > 1
    }
    slice_counts.update(
        value.slice_count
        for value in kwargs.values()
        if isinstance(value, CellProfilerSliceAlignedValues) and value.slice_count > 1
    )
    if len(slice_counts) > 1:
        raise ValueError(
            "Cannot align PURE_2D invocation with conflicting kwarg slice "
            f"counts: {sorted(slice_counts)}."
        )
    if slice_counts:
        return next(iter(slice_counts))
    if any(
        (stack := _slice_aligned_stack_view(value)) is not None
        and stack.shape[0] == 1
        for value in kwargs.values()
    ):
        return 1
    return None


def _slice_pure_2d_value(value: Any, slice_index: int, slice_count: int) -> Any:
    if isinstance(value, CellProfilerSliceAlignedValues):
        return value.value_for_slice(slice_index)
    stack = _slice_aligned_stack_view(value)
    if stack is None:
        return value
    if stack.shape[0] == slice_count:
        return stack[slice_index]
    if stack.shape[0] == 1:
        return stack[0]
    return value


def _slice_aligned_stack_view(value: Any) -> Any | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    try:
        stack = np.asarray(value)
    except (TypeError, ValueError):
        return None
    return stack if stack.ndim == 3 else None


_CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR = CellProfilerFunctionContractExecutor()
