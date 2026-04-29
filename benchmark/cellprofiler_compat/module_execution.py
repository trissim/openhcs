"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from inspect import signature
import re
from typing import Any, ClassVar, get_type_hints

from metaclass_registry import AutoRegisterMeta
import numpy as np
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
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
)
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.core.runtime_stores import require_runtime_value_store
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    _aggregate_pure_2d_auxiliary_output,
    _pure_2d_slice_results,
    _rewrite_slice_index,
)

from benchmark.cellprofiler_library import canonical_module_name
from benchmark.cellprofiler_compat.module_contract import CellProfilerModuleContract
from benchmark.cellprofiler_compat.measurement_lookup import (
    count_feature_object_name,
    measurement_values_for_feature,
)
from benchmark.cellprofiler_compat.relationship_payload import (
    CellProfilerRelationshipPayload,
)
from benchmark.cellprofiler_compat.runtime_adapter import CellProfilerRuntimeAdapter
from benchmark.converter.contract_inference import InferredContract, infer_contract

_MODULE_NAME_REGISTRY_KEY = "module_name"


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

    contract: CellProfilerModuleContract

    def __post_init__(self) -> None:
        if not isinstance(self.contract, CellProfilerModuleContract):
            raise TypeError(
                "CellProfilerModuleExecutor.contract must be "
                "CellProfilerModuleContract, got "
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
        return main_output

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
        return _payload_slice_count(output_image) == _payload_slice_count(input_image)

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
                combined_rows.extend(_measurement_rows_from_output(artifact_values))

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

        cellprofiler_runtime.add_measurements(
            measurement_outputs[0].name,
            combined_rows,
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
        cellprofiler_runtime.add_measurements(
            measurement_outputs[0].name,
            combined_rows,
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
                CellProfilerMeasurementImage.natural(
                    source_image_name=self._input_source_image_name(adapter),
                    payload=_object_only_reference_image(fallback_image),
                ),
            )

        if not CellProfilerPerObjectMeasurementPolicy.measures_images_independently(
            self.module_name
        ):
            return (
                CellProfilerMeasurementImage.composed(image_request),
            )

        runtime_image_names = frozenset(
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.IMAGE,
            )
        )
        resolved_images: list[CellProfilerMeasurementImage] = []
        for spec in image_inputs:
            if spec.name in runtime_image_names:
                runtime_image = adapter.get_image(spec.name)
                resolved_images.append(
                    CellProfilerMeasurementImage.natural(
                        source_image_name=runtime_image.source_image_name or spec.name,
                        payload=runtime_image.data,
                    )
                )
                continue
            resolved_images.append(
                CellProfilerMeasurementImage.natural(
                    source_image_name=spec.name,
                    payload=adapter.resolve_source_image(spec.name, fallback_image),
                )
            )
        return tuple(resolved_images)

    def _independent_measurement_image_inputs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        image_inputs = self._primary_image_inputs(func)
        if not image_inputs:
            return (
                CellProfilerMeasurementImage.natural(
                    source_image_name=self._input_source_image_name(adapter),
                    payload=_object_only_reference_image(fallback_image),
                ),
            )

        runtime_image_names = frozenset(
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.IMAGE,
            )
        )
        resolved_images: list[CellProfilerMeasurementImage] = []
        for spec in image_inputs:
            if spec.name in runtime_image_names:
                runtime_image = adapter.get_image(spec.name)
                resolved_images.append(
                    CellProfilerMeasurementImage.natural(
                        source_image_name=runtime_image.source_image_name or spec.name,
                        payload=runtime_image.data,
                    )
                )
                continue
            resolved_images.append(
                CellProfilerMeasurementImage.natural(
                    source_image_name=spec.name,
                    payload=adapter.resolve_source_image(spec.name, fallback_image),
                )
            )
        return tuple(resolved_images)

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
                self.module_name,
                special_input_names,
                runtime_inputs,
                adapter,
                kwargs=kwargs,
                fallback_image=fallback_image,
                external_image_names=frozenset(self._external_source_image_names()),
                external_object_names=frozenset(
                    self._external_source_object_names()
                ),
                runtime_image_names=frozenset(self._runtime_image_names()),
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
            self.module_name,
            object_inputs,
            adapter,
            kwargs=kwargs,
            fallback_image=fallback_image,
            external_object_names=frozenset(self._external_source_object_names()),
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
                else fallback_image
            )
            return CellProfilerImageRequest(
                payload=payload,
                source_image_name=self._input_source_image_name(adapter),
                image_count=1,
                execution_mode=CellProfilerImageExecutionMode.NATURAL,
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
                payloads.append(adapter.get_image(spec.name).data)
                continue
            payloads.append(adapter.resolve_source_image(spec.name, fallback_image))
        composition = _compose_image_payload(self.module_name, tuple(payloads))
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


@dataclass(frozen=True, slots=True)
class CellProfilerResolvedInputRequest(ABC):
    """Shared source provenance for resolved CellProfiler invocation inputs."""

    source_image_name: str | None
    image_count: int
    execution_mode: "CellProfilerImageExecutionMode"


@dataclass(frozen=True, slots=True)
class CellProfilerImageRequest(CellProfilerResolvedInputRequest):
    """Resolved image payload and source metadata for one module invocation."""

    payload: Any


@dataclass(frozen=True, slots=True)
class CellProfilerInvocationRequest(CellProfilerResolvedInputRequest):
    """Resolved invocation inputs for one CellProfiler function call."""

    image: Any
    kwargs: Mapping[str, Any]


class CellProfilerImageExecutionMode(Enum):
    """How the CellProfiler executor should interpret the resolved image payload."""

    NATURAL = "natural"
    FULL_STACK = "full_stack"
    ALIGNED_MULTI_IMAGE_STACK = "aligned_multi_image_stack"


@dataclass(frozen=True, slots=True)
class CellProfilerImageComposition:
    """Resolved image payload plus its executor mode."""

    payload: Any
    execution_mode: CellProfilerImageExecutionMode


@dataclass(frozen=True, slots=True)
class CellProfilerAlignedImageStack:
    """Per-slice CellProfiler multi-image bundles aligned to one OpenHCS stack."""

    slices: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slices", tuple(self.slices))
        if not self.slices:
            raise ValueError("CellProfilerAlignedImageStack.slices cannot be empty.")


class CellProfilerImageExecutionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal executor mode family for CellProfiler image payload semantics."""

    __registry_key__ = "mode"
    __skip_if_no_key__ = True
    mode: ClassVar[CellProfilerImageExecutionMode | None] = None

    @classmethod
    def for_mode(
        cls,
        mode: CellProfilerImageExecutionMode,
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

    mode = CellProfilerImageExecutionMode.NATURAL

    def execute(
        self,
        executor: "CellProfilerFunctionContractExecutor",
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
    ) -> Any:
        return CellProfilerFunctionContractMetadata.from_callable(func).resolve(
            func
        ).execute(
            executor,
            func,
            image,
            **dict(kwargs),
        )


class FullStackImageExecutionStrategy(CellProfilerImageExecutionStrategy):
    """Execute an already-volumetric payload without per-slice rewriting."""

    mode = CellProfilerImageExecutionMode.FULL_STACK

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

    mode = CellProfilerImageExecutionMode.ALIGNED_MULTI_IMAGE_STACK

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


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementImage:
    """One resolved image payload used by object measurement modules."""

    source_image_name: str | None
    payload: Any
    align_to_labels: bool = True
    execution_mode: CellProfilerImageExecutionMode = (
        CellProfilerImageExecutionMode.NATURAL
    )

    @classmethod
    def natural(
        cls,
        *,
        source_image_name: str | None,
        payload: Any,
    ) -> "CellProfilerMeasurementImage":
        return cls(source_image_name=source_image_name, payload=payload)

    @classmethod
    def composed(
        cls,
        request: CellProfilerImageRequest,
    ) -> "CellProfilerMeasurementImage":
        return cls(
            source_image_name=request.source_image_name,
            payload=request.payload,
            align_to_labels=False,
            execution_mode=request.execution_mode,
        )


def _coerce_invocation_kwargs(
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    coerced_kwargs = dict(kwargs)
    parameters = signature(func).parameters
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
            return request.adapter.get_image(request.spec.name).data
        if request.spec.name in request.external_image_names:
            if request.fallback_image is None:
                raise RuntimeError(
                    f"External image input '{request.spec.name}' requires a "
                    "fallback image payload for source-binding resolution."
                )
            return request.adapter.resolve_source_image(
                request.spec.name,
                request.fallback_image,
            )
        return request.adapter.get_image(request.spec.name).data

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
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        """Return absorbed-function kwargs for object-label runtime inputs."""


class UnsupportedObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Reject undeclared object-input semantics instead of guessing."""

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        del kwargs, fallback_image, external_object_names
        if not object_inputs:
            return {}
        raise NotImplementedError(
            f"{module_name} has object runtime inputs "
            f"{[spec.name for spec in object_inputs]}, but no nominal input binding "
            "policy has been declared for this CellProfiler module."
        )


class SingleObjectLabelInputPolicy(CellProfilerObjectInputPolicy):
    """Bind one object-label input into a module-specific parameter."""

    label_kwarg: ClassVar[str]

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        del kwargs
        _require_exact_object_count(module_name, object_inputs, 1)
        return {
            self.label_kwarg: _object_input_labels(
                object_inputs[0],
                adapter,
                fallback_image=fallback_image,
                external_object_names=external_object_names,
            )
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
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        del kwargs
        _require_exact_object_count(module_name, object_inputs, 2)
        larger, smaller = object_inputs
        return {
            "primary_labels": _object_input_labels(
                smaller,
                adapter,
                fallback_image=fallback_image,
                external_object_names=external_object_names,
            ),
            "secondary_labels": _object_input_labels(
                larger,
                adapter,
                fallback_image=fallback_image,
                external_object_names=external_object_names,
            ),
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
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        del module_name, kwargs
        return {
            "object_labels": tuple(
                _object_input_labels(
                    spec,
                    adapter,
                    fallback_image=fallback_image,
                    external_object_names=external_object_names,
                )
                for spec in object_inputs
            )
        }


class ObjectRowsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind ordered object rows to object-label payloads."""

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        del module_name, kwargs
        return {
            "object_labels": tuple(
                _object_input_labels(
                    spec,
                    adapter,
                    fallback_image=fallback_image,
                    external_object_names=external_object_names,
                )
                for spec in object_inputs
            ),
        }


class ObjectRowsWithMeasurementsInputPolicy(ObjectRowsInputPolicy):
    """Bind ordered object rows plus prior measurements for the primary object."""

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        bound = super().bind(
            module_name,
            object_inputs,
            adapter,
            kwargs=kwargs,
            fallback_image=fallback_image,
            external_object_names=external_object_names,
        )
        primary_object = object_inputs[0] if object_inputs else None
        bound["measurement_tables"] = (
            adapter.measurement_tables_for_object(primary_object.name)
            if primary_object is not None
            else ()
        )
        return bound


class MeasureImageAreaOccupiedInputPolicy(ObjectRowsInputPolicy):
    """Bind ordered object rows for the generic area-occupied runner."""

    module_name = "MeasureImageAreaOccupiedBinary"


class FilterObjectsInputPolicy(ObjectRowsWithMeasurementsInputPolicy):
    """Bind ordered primary/additional object rows for FilterObjects."""

    module_name = "FilterObjects"


class CalculateMathInputPolicy(CellProfilerObjectInputPolicy):
    """Bind CalculateMath operands from runtime measurement/object state."""

    module_name = "CalculateMath"

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        del module_name, object_inputs, fallback_image, external_object_names
        return {
            "operand1_value": _calculate_math_operand_value(
                adapter,
                kwargs,
                feature_kwarg="operand1_feature",
                object_kwarg="operand1_object_name",
            ),
            "operand2_value": _calculate_math_operand_value(
                adapter,
                kwargs,
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


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementRecord:
    """Rows and semantic owner for one CellProfiler measurement output."""

    rows: list[Any]
    object_name: str | None


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
        return CellProfilerMeasurementRecord(
            rows=_measurement_table_rows(request.value),
            object_name=_measurement_object_name(
                request.executor._declared_input_specs()
            ),
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
        request.adapter.add_measurements(
            request.spec.name,
            measurement_record.rows,
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


def _measurement_image_for_labels(image: Any, labels: Any) -> Any:
    """Align a measurement reference image to one object-label payload.

    Many absorbed CellProfiler measurement functions expect a 2D intensity image
    paired with one 2D object-label set. When the OpenHCS main flow is carrying a
    higher-level stack for the whole image set, use a single reference slice
    instead of handing the raw multi-slice stack to functions that require shape
    parity with the labels.
    """
    if not hasattr(image, "ndim") or not hasattr(labels, "ndim"):
        return image
    if is_color_image_stack(image):
        if labels.ndim == 3:
            return image[..., 0]
        if labels.ndim == 2:
            return image[0, :, :, 0]
    if is_color_image_slice(image) and labels.ndim == 2:
        return image[:, :, 0]
    if image.ndim == labels.ndim:
        return image
    if image.ndim == labels.ndim + 1 and getattr(image, "shape", (0,))[0] >= 1:
        return image[0]
    return image


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
        module_name: str,
        parameter_names: tuple[str, ...],
        runtime_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_image_names: frozenset[str],
        external_object_names: frozenset[str],
        runtime_image_names: frozenset[str],
    ) -> dict[str, Any]:
        """Return kwargs for a callable's declared special_inputs."""


class PositionalSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Bind special_inputs positionally to compiled runtime artifact specs."""

    def bind(
        self,
        module_name: str,
        parameter_names: tuple[str, ...],
        runtime_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_image_names: frozenset[str],
        external_object_names: frozenset[str],
        runtime_image_names: frozenset[str],
    ) -> dict[str, Any]:
        del kwargs
        return _bind_special_runtime_inputs(
            module_name,
            parameter_names,
            runtime_inputs,
            adapter,
            fallback_image=fallback_image,
            external_image_names=external_image_names,
            external_object_names=external_object_names,
            runtime_image_names=runtime_image_names,
        )


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
        module_name: str,
        parameter_names: tuple[str, ...],
        runtime_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        kwargs: Mapping[str, Any],
        fallback_image: Any,
        external_image_names: frozenset[str],
        external_object_names: frozenset[str],
        runtime_image_names: frozenset[str],
    ) -> dict[str, Any]:
        del parameter_names, external_image_names, runtime_image_names
        object_inputs = _specs_of_kind(runtime_inputs, ArtifactKind.OBJECT_LABELS)
        _require_exact_object_count(module_name, object_inputs, 1)
        object_spec = object_inputs[0]
        labels = _object_input_labels(
            object_spec,
            adapter,
            fallback_image=fallback_image,
            external_object_names=external_object_names,
        )
        feature_name = _required_string_kwarg(
            kwargs,
            "measurement_feature",
            module_name,
        )
        return {
            "labels": labels,
            "measurements": measurement_values_for_feature(
                adapter.measurement_tables_for_object(object_spec.name),
                feature_name,
                object_count=int(labels.max()),
            ),
        }


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
) -> float:
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
    )
    if len(values) == 1:
        return float(values[0])
    raise NotImplementedError(
        f"CalculateMath scalar function cannot consume non-scalar feature "
        f"{feature_name!r} for object set {object_name!r}."
    )


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
    module_name: str,
    parameter_names: tuple[str, ...],
    runtime_inputs: tuple[ArtifactSpec, ...],
    adapter: CellProfilerRuntimeAdapter,
    *,
    fallback_image: Any,
    external_image_names: frozenset[str],
    external_object_names: frozenset[str],
    runtime_image_names: frozenset[str],
) -> dict[str, Any]:
    if len(parameter_names) != len(runtime_inputs):
        raise NotImplementedError(
            f"{module_name} declares special_inputs {list(parameter_names)}, but "
            f"compiled runtime inputs are {[spec.name for spec in runtime_inputs]}."
        )
    return {
        parameter_name: _runtime_input_value(
            spec,
            adapter,
            fallback_image=fallback_image,
            external_image_names=external_image_names,
            external_object_names=external_object_names,
            runtime_image_names=runtime_image_names,
        )
        for parameter_name, spec in zip(
            parameter_names,
            runtime_inputs,
            strict=True,
        )
    }


def _runtime_input_value(
    spec: ArtifactSpec,
    adapter: CellProfilerRuntimeAdapter,
    *,
    fallback_image: Any,
    external_image_names: frozenset[str],
    external_object_names: frozenset[str],
    runtime_image_names: frozenset[str],
) -> Any:
    try:
        return _artifact_kind_strategy(spec.kind).runtime_input_value(
            CellProfilerArtifactKindRequest(
                spec=spec,
                adapter=adapter,
                fallback_image=fallback_image,
                external_image_names=external_image_names,
                external_object_names=external_object_names,
                runtime_image_names=runtime_image_names,
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


def _compose_image_payload(
    module_name: str,
    image_payloads: tuple[Any, ...],
) -> CellProfilerImageComposition:
    if not image_payloads:
        raise ValueError(f"{module_name} cannot compose an empty image input set.")
    if len(image_payloads) == 1:
        return CellProfilerImageComposition(
            payload=image_payloads[0],
            execution_mode=CellProfilerImageExecutionMode.NATURAL,
        )

    payload_slices = tuple(_payload_slices_for_alignment(payload) for payload in image_payloads)
    slice_counts = tuple(len(slices) for slices in payload_slices)
    max_slice_count = max(slice_counts)
    invalid_counts = tuple(count for count in slice_counts if count not in {1, max_slice_count})
    if invalid_counts:
        raise ValueError(
            f"{module_name} cannot align multi-image inputs with incompatible "
            f"slice counts {slice_counts!r}."
        )

    if max_slice_count == 1:
        return CellProfilerImageComposition(
            payload=_compose_one_image_bundle(tuple(slices[0] for slices in payload_slices)),
            execution_mode=CellProfilerImageExecutionMode.FULL_STACK,
        )
    return CellProfilerImageComposition(
        payload=CellProfilerAlignedImageStack(
            slices=tuple(
                _compose_one_image_bundle(
                    tuple(_aligned_payload_slice(slices, slice_index) for slices in payload_slices)
                )
                for slice_index in range(max_slice_count)
            )
        ),
        execution_mode=CellProfilerImageExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )


def _payload_slices_for_alignment(payload: Any) -> tuple[Any, ...]:
    if hasattr(payload, "ndim") and payload.ndim == 2:
        return (payload,)
    if hasattr(payload, "ndim") and payload.ndim == 3:
        memory_type = detect_memory_type(payload)
        return tuple(unstack_slices(payload, memory_type, 0))
    return (payload,)


def _aligned_payload_slice(
    slices: tuple[Any, ...],
    slice_index: int,
) -> Any:
    if len(slices) == 1:
        return slices[0]
    return slices[slice_index]


def _aligned_multi_image_stack_kwargs(
    kwargs: Mapping[str, Any],
    slice_index: int,
    slice_count: int,
) -> dict[str, Any]:
    return {
        name: _aligned_multi_image_stack_kwarg(value, slice_index, slice_count)
        for name, value in kwargs.items()
    }


def _aligned_multi_image_stack_kwarg(
    value: Any,
    slice_index: int,
    slice_count: int,
) -> Any:
    if not hasattr(value, "ndim"):
        return value
    slices = _payload_slices_for_alignment(value)
    if len(slices) == slice_count:
        return slices[slice_index]
    if len(slices) == 1:
        return slices[0]
    return value


def _compose_one_image_bundle(
    image_payloads: tuple[Any, ...],
) -> Any:
    memory_type = detect_memory_type(image_payloads[0])
    return stack_slices(list(image_payloads), memory_type=memory_type, gpu_id=0)


def _collapse_singleton_stack_output(value: Any) -> Any:
    if hasattr(value, "ndim") and value.ndim == 3 and value.shape[0] == 1:
        return value[0]
    if isinstance(value, tuple):
        return tuple(_collapse_singleton_stack_output(item) for item in value)
    return value


def _single_source_name(source_names: tuple[str, ...]) -> str | None:
    unique_names = tuple(dict.fromkeys(source_names))
    if len(unique_names) == 1:
        return unique_names[0]
    return None


def _payload_slice_count(payload: Any) -> int:
    if hasattr(payload, "ndim") and payload.ndim == 2:
        return 1
    if hasattr(payload, "shape") and payload.shape:
        return int(payload.shape[0])
    return 1


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
    return hasattr(output, "ndim") and output.ndim == 2


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
    execution_mode: CellProfilerImageExecutionMode | None,
) -> CellProfilerImageExecutionMode:
    if execution_mode is not None:
        return execution_mode
    if force_full_stack:
        return CellProfilerImageExecutionMode.FULL_STACK
    return CellProfilerImageExecutionMode.NATURAL


class CellProfilerFunctionContractExecutor:
    """Apply OpenHCS processing contracts after CellProfiler input resolution."""

    def execute(
        self,
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
        *,
        force_full_stack: bool = False,
        execution_mode: CellProfilerImageExecutionMode | None = None,
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
        if not isinstance(image, CellProfilerAlignedImageStack):
            raise TypeError(
                "ALIGNED_MULTI_IMAGE_STACK execution requires "
                f"CellProfilerAlignedImageStack, got {type(image).__name__}."
            )
        slice_results = tuple(
            _rewrite_slice_index(
                _collapse_singleton_stack_output(
                    func(
                        slice_payload,
                        **_aligned_multi_image_stack_kwargs(
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


@dataclass(frozen=True, slots=True)
class CellProfilerFunctionContractMetadata:
    """Decorator-declared processing contract metadata for one absorbed callable."""

    explicit: ProcessingContract | None
    declared: str | None

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
    ) -> "CellProfilerFunctionContractMetadata":
        metadata = _callable_metadata(func)
        explicit = metadata.get("__processing_contract__")
        declared = metadata.get("__cellprofiler_declared_contract__")
        return cls(
            explicit=explicit if isinstance(explicit, ProcessingContract) else None,
            declared=declared if isinstance(declared, str) else None,
        )

    def resolve(self, func: Callable[..., Any]) -> ProcessingContract:
        if self.explicit is not None:
            return self.explicit
        if self.declared == "unknown":
            inferred = _infer_processing_contract(func)
            if inferred is not None:
                return inferred
        if self.declared is not None:
            declared = _declared_processing_contract(self.declared)
            if declared is not None:
                return declared
        return ProcessingContract.FLEXIBLE


def _callable_metadata(func: Callable[..., Any]) -> Mapping[str, Any]:
    try:
        return vars(func)
    except TypeError:
        return {}


def _infer_processing_contract(
    func: Callable[..., Any],
) -> ProcessingContract | None:
    inferred = infer_contract(func, dtype_config=DtypeConfig()).contract
    if inferred is InferredContract.UNKNOWN or inferred is InferredContract.ERROR:
        return None
    if inferred.name not in ProcessingContract.__members__:
        return None
    return ProcessingContract[inferred.name]


def _declared_processing_contract(
    contract_name: str,
) -> ProcessingContract | None:
    normalized = contract_name.upper()
    if normalized not in ProcessingContract.__members__:
        return None
    return ProcessingContract[normalized]


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
        value.shape[0]
        for value in kwargs.values()
        if hasattr(value, "ndim") and value.ndim == 3 and value.shape[0] > 1
    }
    if len(slice_counts) > 1:
        raise ValueError(
            "Cannot align PURE_2D invocation with conflicting kwarg slice "
            f"counts: {sorted(slice_counts)}."
        )
    if slice_counts:
        return next(iter(slice_counts))
    if any(
        hasattr(value, "ndim") and value.ndim == 3 and value.shape[0] == 1
        for value in kwargs.values()
    ):
        return 1
    return None


def _slice_pure_2d_value(value: Any, slice_index: int, slice_count: int) -> Any:
    if not hasattr(value, "ndim") or value.ndim != 3:
        return value
    if value.shape[0] == slice_count:
        return value[slice_index]
    if value.shape[0] == 1:
        return value[0]
    return value


_CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR = CellProfilerFunctionContractExecutor()
