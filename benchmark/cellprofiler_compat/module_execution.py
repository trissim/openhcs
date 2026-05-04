"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from enum import Enum
from inspect import Parameter, signature, unwrap
import json
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
from openhcs.core.runtime_artifact_queries import (
    MEASUREMENT_FEATURE_NAME_FIELD,
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_ID_FIELD,
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_RESULT_VALUE_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    MeasurementFeatureQuery,
    annotate_measurement_row_object,
    annotate_measurement_row_source_image,
    measurement_object_label,
    measurement_row_mapping,
    measurement_rows,
    measurement_scalar_value_for_feature,
    measurement_table_for_slice,
    measurement_values_for_label_slices,
    measurement_values_for_feature,
)
from openhcs.core.runtime_semantics import (
    MeasurementScope,
    FieldSpec,
    MeasurementObjectRowIdentity,
    ParentChildRelationshipPayload,
    dense_object_label_id_domain,
    measurement_row_axis_field_names,
    parent_child_relationship_artifact_name,
)
from openhcs.core.runtime_stores import require_runtime_value_store
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectLabelPayload,
    ObjectRelationship,
)
from openhcs.core.runtime_values import (
    ImageMetadataPayload,
    MaskedImagePayload,
    SpatialGrid,
    compose_image_payload_metadata,
    image_payload_data,
    image_payload_metadata,
    image_payload_mask,
    image_payload_with_context,
    normalize_image_payload_intensity,
    with_image_payload_data,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    _aggregate_pure_2d_auxiliary_output,
    _pure_2d_slice_results,
    _rewrite_slice_index,
)
from openhcs.processing.materialization import tabular_field_names_from_materialization

from benchmark.cellprofiler_library import canonical_module_name, require_function
from benchmark.cellprofiler_compat.measurement_scope import (
    CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
    CellProfilerMeasurementTargetScope,
    coerce_cellprofiler_measurement_target_scope,
)
from benchmark.cellprofiler_compat.measurement_lookup import (
    count_feature_object_name,
)
from benchmark.cellprofiler_compat.runtime_adapter import CellProfilerRuntimeAdapter
from benchmark.converter.contract_inference import InferredContract, infer_contract

_MODULE_NAME_REGISTRY_KEY = "module_name"
_INVOCATION_CONTROL_KWARGS = frozenset(
    (
        "dtype_config",
        "slice_by_slice",
        CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
    )
)


def _cellprofiler_image_payload(payload: Any) -> Any:
    """Return payload in CellProfiler's float image intensity domain."""
    return normalize_image_payload_intensity(payload, dtype=np.float32)


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
                current_image=image,
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
                current_image=image,
                image_request=image_request,
                cellprofiler_runtime=cellprofiler_runtime,
                source_image_name=image_request.source_image_name,
                **kwargs,
            )

        invocation = self._invocation_request(
            func,
            image_request=image_request,
            adapter=cellprofiler_runtime,
            current_image=image,
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
        current_image: Any,
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

        measurement_target_scope = _pop_measurement_target_scope(
            kwargs,
            default=CellProfilerMeasurementTargetScope.OBJECT,
        )
        combined_rows: list[Any] = []
        measurement_images = self._measurement_image_inputs(
            func,
            cellprofiler_runtime,
            current_image,
            image_request,
        )
        image_measurement_rows = self._dual_scope_image_measurement_rows(
            func,
            measurement_images,
            kwargs,
            measurement_target_scope,
        )
        combined_rows.extend(image_measurement_rows)
        row_source_names_required = _row_source_names_required(measurement_images)
        measurement_row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
            self.module_name
        )
        for measurement_image in measurement_images:
            for object_spec in object_inputs:
                raw_label_payload = self._object_label_payload(
                    object_spec,
                    cellprofiler_runtime,
                    input_image,
                )
                raw_labels = _label_payload_final(raw_label_payload)
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
                measurement_rows = _complete_object_measurement_rows(
                    _measurement_rows_from_output(artifact_values),
                    label_payload=raw_label_payload,
                    func=func,
                    object_identity=measurement_row_policy.object_identity(),
                    row_policy=measurement_row_policy,
                )
                for row in measurement_rows:
                    annotated_row = annotate_measurement_row_object(
                        row,
                        object_spec.name,
                    )
                    if (
                        row_source_names_required
                        and measurement_image.source_image_name is not None
                    ):
                        annotated_row = annotate_measurement_row_source_image(
                            annotated_row,
                            measurement_image.source_image_name,
                        )
                    combined_rows.append(annotated_row)

        combined_source_image_name = (
            source_image_name
            if not measurement_images
            else _single_measurement_image_source_name(measurement_images)
        )

        _record_measurements(
            cellprofiler_runtime,
            measurement_outputs[0].name,
            combined_rows,
            fields=_measurement_record_fields(measurement_outputs[0], combined_rows, func),
            object_name=(
                None
                if image_measurement_rows
                else object_inputs[0].name if len(object_inputs) == 1 else None
            ),
            source_image_name=combined_source_image_name,
        )
        return input_image

    def _dual_scope_image_measurement_rows(
        self,
        object_func: Callable[..., Any],
        measurement_images: tuple["CellProfilerMeasurementImage", ...],
        kwargs: Mapping[str, Any],
        target_scope: CellProfilerMeasurementTargetScope,
    ) -> list[Any]:
        if target_scope is not CellProfilerMeasurementTargetScope.BOTH:
            return []
        policy = CellProfilerDualScopeMeasurementPolicy.for_module(self.module_name)
        if policy is None:
            return []
        image_func = policy.image_function(object_func)
        rows: list[Any] = []
        row_source_names_required = _row_source_names_required(measurement_images)
        image_kwargs = _coerce_invocation_kwargs(image_func, kwargs)
        for measurement_image in measurement_images:
            raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                image_func,
                _image_scope_measurement_payload(measurement_image.payload),
                image_kwargs,
                execution_mode=measurement_image.execution_mode,
            )
            _ignored_main_output, artifact_values = _split_cellprofiler_output(
                raw_output
            )
            for row in _measurement_rows_from_output(artifact_values):
                if (
                    row_source_names_required
                    and measurement_image.source_image_name is not None
                ):
                    row = annotate_measurement_row_source_image(
                        row,
                        measurement_image.source_image_name,
                    )
                rows.append(row)
        return rows

    def _run_per_image_measurement(
        self,
        func: Callable[..., Any],
        *,
        input_image: Any,
        current_image: Any,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        **kwargs: Any,
    ) -> Any:
        measurement_outputs = _specs_of_kind(self.outputs, ArtifactKind.MEASUREMENTS)
        if len(measurement_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} per-image execution requires exactly one "
                "measurement output."
            )

        _pop_measurement_target_scope(
            kwargs,
            default=CellProfilerMeasurementTargetScope.IMAGE,
        )
        combined_rows: list[Any] = []
        measurement_images = self._independent_measurement_image_inputs(
            func,
            cellprofiler_runtime,
            current_image,
        )
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(
                func,
                cellprofiler_runtime,
                current_image,
                kwargs,
            ),
        }
        coerced_kwargs = _coerce_invocation_kwargs(func, runtime_kwargs)
        row_source_names_required = _row_source_names_required(measurement_images)
        for measurement_image in measurement_images:
            raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                func,
                _image_scope_measurement_payload(measurement_image.payload),
                coerced_kwargs,
                execution_mode=measurement_image.execution_mode,
            )
            _ignored_main_output, artifact_values = _split_cellprofiler_output(
                raw_output
            )
            for row in _measurement_rows_from_output(artifact_values):
                if (
                    row_source_names_required
                    and measurement_image.source_image_name is not None
                ):
                    row = annotate_measurement_row_source_image(
                        row,
                        measurement_image.source_image_name,
                    )
                combined_rows.append(row)

        _record_measurements(
            cellprofiler_runtime,
            measurement_outputs[0].name,
            combined_rows,
            fields=_measurement_record_fields(measurement_outputs[0], combined_rows, func),
            source_image_name=_single_measurement_image_source_name(
                measurement_images
            ),
        )
        return input_image

    def _measurement_image_inputs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        current_image: Any,
        image_request: "CellProfilerImageRequest",
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        image_inputs = self._primary_image_inputs(func)
        if not image_inputs:
            return (
                self._measurement_carrier_image(
                    adapter,
                    current_image,
                    reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
                ),
            )

        if not CellProfilerPerObjectMeasurementPolicy.measures_images_independently(
            self.module_name
        ):
            return (
                self._composed_measurement_image(image_request, image_inputs),
            )

        return self._resolved_measurement_images(
            image_inputs,
            adapter,
            current_image,
            reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
        )

    def _independent_measurement_image_inputs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        current_image: Any,
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        image_inputs = self._primary_image_inputs(func)
        if not image_inputs:
            return (
                self._measurement_carrier_image(
                    adapter,
                    current_image,
                    reference_domain=CellProfilerMeasurementImageDomain.SOURCE_IMAGE,
                ),
            )

        return self._resolved_measurement_images(
            image_inputs,
            adapter,
            current_image,
        )

    def _measurement_carrier_image(
        self,
        adapter: CellProfilerRuntimeAdapter,
        current_image: Any,
        *,
        reference_domain: "CellProfilerMeasurementImageDomain",
    ) -> "CellProfilerMeasurementImage":
        return CellProfilerMeasurementImage(
            source_image_name=None,
            payload=_object_only_reference_image(current_image),
            reference_domain=reference_domain,
        )

    def _composed_measurement_image(
        self,
        image_request: "CellProfilerImageRequest",
        image_inputs: tuple[ArtifactSpec, ...],
    ) -> "CellProfilerMeasurementImage":
        return CellProfilerMeasurementImage(
            source_image_name=_measurement_source_name_for_specs(image_inputs),
            payload=image_request.payload,
            align_to_labels=False,
            execution_mode=image_request.execution_mode,
        )

    def _resolved_measurement_images(
        self,
        image_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        current_image: Any,
        *,
        reference_domain: "CellProfilerMeasurementImageDomain | None" = None,
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        if reference_domain is None:
            reference_domain = CellProfilerMeasurementImageDomain.SOURCE_IMAGE
        runtime_image_names = frozenset(self._runtime_image_names())
        resolved_images: list[CellProfilerMeasurementImage] = []
        for spec in image_inputs:
            resolved_images.append(
                self._resolved_measurement_image(
                    spec,
                    adapter,
                    current_image,
                    runtime_image_names,
                    reference_domain=reference_domain,
                )
            )
        return tuple(resolved_images)

    def _resolved_measurement_image(
        self,
        spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        current_image: Any,
        runtime_image_names: frozenset[str],
        *,
        reference_domain: "CellProfilerMeasurementImageDomain",
    ) -> "CellProfilerMeasurementImage":
        if spec.name in runtime_image_names:
            runtime_image = adapter.get_image(spec.name)
            return CellProfilerMeasurementImage(
                source_image_name=spec.name,
                payload=_cellprofiler_image_payload(runtime_image.data),
                reference_domain=reference_domain,
            )
        return CellProfilerMeasurementImage(
            source_image_name=spec.name,
            payload=_cellprofiler_image_payload(
                adapter.resolve_source_image(spec.name, current_image)
            ),
            reference_domain=reference_domain,
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
        current_image: Any,
    ) -> Any:
        return _label_payload_final(
            self._object_label_payload(spec, adapter, current_image)
        )

    def _object_label_payload(
        self,
        spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        current_image: Any,
    ) -> Any:
        if spec.name in self._external_source_object_names():
            return adapter.resolve_source_objects(
                spec.name,
                current_image,
            ).runtime_payload()
        return adapter.get_objects(spec.name).runtime_payload()

    def _runtime_input_kwargs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        current_image: Any,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        runtime_inputs = self._special_runtime_inputs(func)
        object_input_policy = CellProfilerObjectInputPolicy.for_module(self.module_name)
        if not runtime_inputs:
            if object_input_policy.binds_without_declared_inputs:
                return object_input_policy.bind(
                    ObjectInputBindingRequest(
                        module_name=self.module_name,
                        object_inputs=(),
                        adapter=adapter,
                    kwargs=kwargs,
                    current_image=current_image,
                    external_object_names=frozenset(
                        self._external_source_object_names()
                    ),
                    runtime_inputs=runtime_inputs,
                )
            )
            return {}

        special_input_names = special_input_names_from_callable(func)
        if special_input_names:
            return CellProfilerSpecialInputPolicy.for_module(self.module_name).bind(
                SpecialInputBindingRequest(
                    module_name=self.module_name,
                    parameter_names=special_input_names,
                    runtime_inputs=runtime_inputs,
                    adapter=adapter,
                    kwargs=kwargs,
                    current_image=current_image,
                    external_image_names=frozenset(self._external_source_image_names()),
                    external_object_names=frozenset(
                        self._external_source_object_names()
                    ),
                    runtime_image_names=frozenset(self._runtime_image_names()),
                )
            )

        supported_non_object_kinds = object_input_policy.supported_non_object_input_kinds
        unsupported_non_object_inputs = tuple(
            spec
            for spec in runtime_inputs
            if spec.kind is not ArtifactKind.OBJECT_LABELS
            and spec.kind not in supported_non_object_kinds
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
        return object_input_policy.bind(
            ObjectInputBindingRequest(
                module_name=self.module_name,
                object_inputs=object_inputs,
                adapter=adapter,
                kwargs=kwargs,
                current_image=current_image,
                external_object_names=frozenset(self._external_source_object_names()),
                runtime_inputs=runtime_inputs,
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
        for spec in _output_recording_order(self.outputs):
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
        current_image: Any,
        adapter: CellProfilerRuntimeAdapter,
    ) -> "CellProfilerImageRequest":
        image_inputs = self._primary_image_inputs(func)
        if not image_inputs:
            payload = (
                _object_only_reference_image(current_image)
                if self._object_input_specs()
                or _specs_of_kind(
                    self._declared_input_specs(),
                    ArtifactKind.SPATIAL_GRID,
                )
                else _cellprofiler_image_payload(current_image)
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
                    adapter.resolve_source_image(spec.name, current_image)
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
                RuntimeArtifactInputRequest(
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
        current_image: Any,
        kwargs: Mapping[str, Any],
    ) -> "CellProfilerInvocationRequest":
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(func, adapter, current_image, kwargs),
        }
        runtime_kwargs.pop(CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG, None)
        if _should_slice_flexible_object_invocation(
            self._object_input_specs(),
            func,
            runtime_kwargs,
        ):
            runtime_kwargs.setdefault("slice_by_slice", True)
        execution_mode = CellProfilerInvocationExecutionModePolicy.for_module(
            self.module_name
        ).execution_mode(
            default=image_request.execution_mode,
            kwargs=runtime_kwargs,
        )
        return CellProfilerInvocationRequest(
            image=image_request.payload,
            kwargs=_coerce_invocation_kwargs(func, runtime_kwargs),
            source_image_name=image_request.source_image_name,
            image_count=image_request.image_count,
            execution_mode=execution_mode,
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
        declared = tuple(self.inputs)
        runtime_extras = tuple(
            spec for spec in self.runtime_artifact_inputs if spec not in declared
        )
        return (*declared, *runtime_extras)


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


class CellProfilerInvocationExecutionModePolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal policy for modules whose settings change stack execution mode."""

    __registry_key__ = _MODULE_NAME_REGISTRY_KEY
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "CellProfilerInvocationExecutionModePolicy":
        policy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            DefaultInvocationExecutionModePolicy,
        )
        return policy_type()

    def execution_mode(
        self,
        *,
        default: ImagePayloadExecutionMode,
        kwargs: Mapping[str, Any],
    ) -> ImagePayloadExecutionMode:
        return default


class DefaultInvocationExecutionModePolicy(CellProfilerInvocationExecutionModePolicy):
    """Use the execution mode implied by image payload composition."""


class CorrectIlluminationCalculateExecutionModePolicy(
    CellProfilerInvocationExecutionModePolicy
):
    """Run all-image illumination calculation once over the full image stack."""

    module_name = "CorrectIlluminationCalculate"

    def execution_mode(
        self,
        *,
        default: ImagePayloadExecutionMode,
        kwargs: Mapping[str, Any],
    ) -> ImagePayloadExecutionMode:
        if _illumination_scope_uses_all_images(kwargs.get("calculation_scope")):
            return ImagePayloadExecutionMode.FULL_STACK
        return default


class DefineGridManualExecutionModePolicy(CellProfilerInvocationExecutionModePolicy):
    """Manual grid definitions are image-independent and should be emitted once."""

    module_name = "DefineGridManual"

    def execution_mode(
        self,
        *,
        default: ImagePayloadExecutionMode,
        kwargs: Mapping[str, Any],
    ) -> ImagePayloadExecutionMode:
        del default, kwargs
        return ImagePayloadExecutionMode.FULL_STACK


class CellProfilerImageExecutionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal executor mode family for CellProfiler image payload semantics."""

    __registry_key__ = "mode_key"
    __skip_if_no_key__ = True
    mode: ClassVar[ImagePayloadExecutionMode | None] = None
    mode_key: ClassVar[str | None] = None

    @classmethod
    def for_mode(
        cls,
        mode: ImagePayloadExecutionMode,
    ) -> "CellProfilerImageExecutionStrategy":
        return cls.__registry__[mode.value]()

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
    mode_key = ImagePayloadExecutionMode.NATURAL.value

    def execute(
        self,
        executor: "CellProfilerFunctionContractExecutor",
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
    ) -> Any:
        contract = _processing_contract_for_callable(func)
        if (
            contract is ProcessingContract.PURE_2D
            and _slice_count_from_pure_2d_kwargs(kwargs) is not None
        ):
            return executor._execute_pure_2d(func, image, **dict(kwargs))
        return contract.execute(
            executor,
            func,
            image,
            **dict(kwargs),
        )


class FullStackImageExecutionStrategy(CellProfilerImageExecutionStrategy):
    """Execute an already-volumetric payload without per-slice rewriting."""

    mode = ImagePayloadExecutionMode.FULL_STACK
    mode_key = ImagePayloadExecutionMode.FULL_STACK.value

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
    mode_key = ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK.value

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
class RuntimeArtifactInputRequest:
    """One artifact-spec request dispatched through a nominal kind strategy."""

    spec: ArtifactSpec
    adapter: CellProfilerRuntimeAdapter
    current_image: Any | None = None
    external_image_names: frozenset[str] = frozenset()
    external_object_names: frozenset[str] = frozenset()
    runtime_image_names: frozenset[str] = frozenset()


class RuntimeArtifactKindStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy family for ArtifactKind-specific runtime semantics."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    kind: ClassVar[ArtifactKind | None] = None

    @classmethod
    def for_kind(cls, kind: ArtifactKind) -> "RuntimeArtifactKindStrategy":
        return cls.__registry__[kind]()

    @abstractmethod
    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        """Return the runtime payload bound into absorbed function kwargs."""

    @abstractmethod
    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        """Return the transitive source image name for one artifact input."""


class ImageArtifactKindStrategy(RuntimeArtifactKindStrategy):
    """Resolve image artifact payloads and source-image lineage."""

    kind = ArtifactKind.IMAGE

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        if request.spec.name in request.runtime_image_names:
            return _cellprofiler_image_payload(
                request.adapter.get_image(request.spec.name).data
            )
        if request.spec.name in request.external_image_names:
            if request.current_image is None:
                raise RuntimeError(
                    f"External image input '{request.spec.name}' requires a "
                    "current image payload for source-binding resolution."
                )
            return _cellprofiler_image_payload(
                request.adapter.resolve_source_image(
                    request.spec.name,
                    request.current_image,
                )
            )
        return _cellprofiler_image_payload(
            request.adapter.get_image(request.spec.name).data
        )

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        if request.spec.name in request.runtime_image_names:
            return request.adapter.get_image(request.spec.name).source_image_name
        if request.spec.name in request.external_image_names:
            return request.spec.name
        return None


class ObjectLabelsArtifactKindStrategy(RuntimeArtifactKindStrategy):
    """Resolve object-label payloads and lineage."""

    kind = ArtifactKind.OBJECT_LABELS

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        if request.spec.name in request.external_object_names:
            if request.current_image is None:
                raise RuntimeError(
                    f"External object input '{request.spec.name}' requires a "
                    "current image payload for source-binding resolution."
                )
            return _collapse_singleton_label_stack(
                request.adapter.resolve_source_objects(
                    request.spec.name,
                    request.current_image,
                ).labels
            )
        return _collapse_singleton_label_stack(
            request.adapter.get_objects(request.spec.name).labels
        )

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        if request.spec.name in request.external_object_names:
            return request.spec.name
        return request.adapter.get_objects(request.spec.name).source_image_name


class MeasurementsArtifactKindStrategy(RuntimeArtifactKindStrategy):
    """Resolve measurement payloads and lineage."""

    kind = ArtifactKind.MEASUREMENTS

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        return request.adapter.get_measurements(request.spec.name).rows

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        return request.adapter.get_measurements(request.spec.name).source_image_name


class RelationshipsArtifactKindStrategy(RuntimeArtifactKindStrategy):
    """Resolve relationship payloads."""

    kind = ArtifactKind.RELATIONSHIPS

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        return request.adapter.get_relationship(request.spec.name)

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        return None


class SpatialGridArtifactKindStrategy(RuntimeArtifactKindStrategy):
    """Resolve spatial-grid payloads."""

    kind = ArtifactKind.SPATIAL_GRID

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        return request.adapter.get_spatial_grid(request.spec.name)

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeInputBindingRequestBase(ABC):
    """Shared runtime context for artifact-backed runtime-input binding."""

    module_name: str
    adapter: CellProfilerRuntimeAdapter
    kwargs: Mapping[str, Any]
    current_image: Any
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
            current_image=self.current_image,
            external_object_names=self.external_object_names,
        )

    def label_payload_for(self, spec: ArtifactSpec) -> Any:
        if spec.name in self.external_object_names:
            return self.adapter.resolve_source_objects(
                spec.name,
                self.current_image,
            ).labels
        return self.adapter.get_objects(spec.name).runtime_payload()


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectInputBindingRequest(RuntimeInputBindingRequestBase):
    """Authoritative runtime context for binding object-label inputs."""

    object_inputs: tuple[ArtifactSpec, ...]
    runtime_inputs: tuple[ArtifactSpec, ...] = ()

    def __post_init__(self) -> None:
        RuntimeInputBindingRequestBase.__post_init__(self)
        object.__setattr__(self, "object_inputs", tuple(self.object_inputs))
        object.__setattr__(self, "runtime_inputs", tuple(self.runtime_inputs))

    def with_object_inputs(
        self,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> "ObjectInputBindingRequest":
        return type(self)(
            module_name=self.module_name,
            object_inputs=object_inputs,
            adapter=self.adapter,
            kwargs=self.kwargs,
            current_image=self.current_image,
            external_object_names=self.external_object_names,
            runtime_inputs=self.runtime_inputs,
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
    binds_without_declared_inputs: ClassVar[bool] = False
    supported_non_object_input_kinds: ClassVar[frozenset[ArtifactKind]] = frozenset()

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
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        """Return absorbed-function kwargs for object-label runtime inputs."""


class UnsupportedObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Reject undeclared object-input semantics instead of guessing."""

    def bind(
        self,
        request: ObjectInputBindingRequest,
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
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        request.require_exact_object_count(1)
        return {
            self.label_kwarg: request.labels_for(request.object_inputs[0])
        }


class IdentifySecondaryObjectsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind primary objects with generic label-variant context when available."""

    module_name = "IdentifySecondaryObjects"

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        request.require_exact_object_count(1)
        return {
            "primary_labels": request.label_payload_for(request.object_inputs[0])
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
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        request.require_exact_object_count(2)
        larger, smaller = request.object_inputs
        return {
            "primary_labels": request.labels_for(smaller),
            "secondary_labels": request.labels_for(larger),
        }


_MEASURE_OBJECT_SIZE_SHAPE_MODULE = "MeasureObjectSizeShape"
_MEASURE_OBJECT_INTENSITY_MODULE = "MeasureObjectIntensity"
_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE = "MeasureObjectIntensityDistribution"
_MEASURE_TEXTURE_MODULE = "MeasureTexture"
_MEASURE_COLOCALIZATION_MODULE = "MeasureColocalization"
_MEASURE_GRANULARITY_MODULE = "MeasureGranularity"
_MEASURE_OBJECT_NEIGHBORS_MODULE = "MeasureObjectNeighbors"
_OBJECT_ROW_SEQUENCE_KWARGS = frozenset({"object_labels"})
_MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS = (
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_OBJECT_ID_FIELD,
)


class MissingObjectMeasurementValuePolicy(str, Enum):
    """How missing per-object measurement result fields are materialized."""

    NAN = "nan"
    ZERO_WITHIN_POSITIVE_EXTENT = "zero_within_positive_extent"


class CellProfilerObjectMeasurementRowPolicy(metaclass=AutoRegisterMeta):
    """Nominal export-row policy for object-scoped measurement modules."""

    __registry_key__ = _MODULE_NAME_REGISTRY_KEY
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None
    row_identity: ClassVar[MeasurementObjectRowIdentity] = (
        MeasurementObjectRowIdentity.LABEL_ID
    )
    missing_value_policy: ClassVar[MissingObjectMeasurementValuePolicy] = (
        MissingObjectMeasurementValuePolicy.NAN
    )

    @classmethod
    def for_module(
        cls,
        module_name: str,
    ) -> "CellProfilerObjectMeasurementRowPolicy":
        policy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            DefaultObjectMeasurementRowPolicy,
        )
        return policy_type()

    def object_identity(self) -> MeasurementObjectRowIdentity:
        """Return the object identity projection for rows emitted by this module."""
        return MeasurementObjectRowIdentity(type(self).row_identity)

    def missing_measurement_value(
        self,
        *,
        object_id: int,
        label_payload: Any,
        field_name: str,
        positive_label_extent: int | None = None,
    ) -> float:
        """Return the value to use for a missing object measurement field."""
        policy = MissingObjectMeasurementValuePolicy(type(self).missing_value_policy)
        if policy is MissingObjectMeasurementValuePolicy.NAN:
            return np.nan
        if policy is MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT:
            extent = (
                _positive_object_label_extent(label_payload)
                if positive_label_extent is None
                else positive_label_extent
            )
            return (
                0.0
                if object_id <= extent
                else np.nan
            )
        raise ValueError(f"Unsupported missing measurement value policy: {policy}.")


class DefaultObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """Use runtime object-label IDs as measurement-row identities."""


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowPolicySpec:
    """Declarative leaf spec for one object measurement row policy."""

    module_name: str
    row_identity: MeasurementObjectRowIdentity = MeasurementObjectRowIdentity.LABEL_ID
    missing_value_policy: MissingObjectMeasurementValuePolicy = (
        MissingObjectMeasurementValuePolicy.NAN
    )


class DeclaredObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """Generated base for modules with declared measurement-row identity."""


def _declare_object_measurement_row_policy(
    spec: ObjectMeasurementRowPolicySpec,
) -> None:
    type(
        f"{spec.module_name}ObjectMeasurementRowPolicy",
        (DeclaredObjectMeasurementRowPolicy,),
        {
            "__module__": __name__,
            "module_name": spec.module_name,
            "row_identity": spec.row_identity,
            "missing_value_policy": spec.missing_value_policy,
        },
    )


for _row_policy_spec in (
    ObjectMeasurementRowPolicySpec(
        _MEASURE_OBJECT_SIZE_SHAPE_MODULE,
        MeasurementObjectRowIdentity.ROW_ORDINAL,
    ),
    ObjectMeasurementRowPolicySpec(
        _MEASURE_OBJECT_INTENSITY_MODULE,
        missing_value_policy=(
            MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
        ),
    ),
):
    _declare_object_measurement_row_policy(_row_policy_spec)


_SINGLE_OBJECT_LABEL_INPUT_POLICY_SPECS = (
    SingleObjectLabelInputPolicySpec("Crop", "cropping_labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_OBJECT_SIZE_SHAPE_MODULE, "labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_OBJECT_INTENSITY_MODULE, "labels"),
    SingleObjectLabelInputPolicySpec(
        _MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE,
        "labels",
    ),
    SingleObjectLabelInputPolicySpec(_MEASURE_TEXTURE_MODULE, "labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_COLOCALIZATION_MODULE, "labels"),
    SingleObjectLabelInputPolicySpec(_MEASURE_GRANULARITY_MODULE, "labels"),
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


class MeasureObjectNeighborsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind neighbor topology through generic object-label variants."""

    module_name = _MEASURE_OBJECT_NEIGHBORS_MODULE

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        if len(request.object_inputs) not in (1, 2):
            raise NotImplementedError(
                "MeasureObjectNeighbors requires one or two object runtime "
                f"inputs, got {[spec.name for spec in request.object_inputs]}."
            )

        measured = request.object_inputs[0]
        neighbor = request.object_inputs[-1]
        measured_payload = request.label_payload_for(measured)
        neighbor_payload = (
            measured_payload
            if measured == neighbor
            else request.label_payload_for(neighbor)
        )
        same_objects = measured == neighbor

        return {
            "labels": _label_payload_final(measured_payload),
            "small_removed_labels": _label_payload_small_removed(measured_payload),
            "neighbor_labels": (
                None if same_objects else _label_payload_final(neighbor_payload)
            ),
            "small_removed_neighbor_labels": (
                None
                if same_objects
                else _label_payload_small_removed(neighbor_payload)
            ),
            "neighbors_are_same_objects": same_objects,
        }


class OverlayOutlinesInputPolicy(CellProfilerObjectInputPolicy):
    """Bind ordered object outline rows for the generic overlay runner."""

    module_name = "OverlayOutlines"

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        return {"object_labels": request.labels_for_inputs()}


class ObjectRowsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind ordered object rows to object-label payloads."""

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        return {"object_labels": request.labels_for_inputs()}


class ObjectRowsWithMeasurementsInputPolicy(ObjectRowsInputPolicy):
    """Bind ordered object rows plus prior measurements for the primary object."""

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        bound = super().bind(request)
        bound["measurement_tables"] = request.measurement_tables_for_primary_object()
        return bound


_FILTER_OBJECTS_CHILD_COUNT_PREFIX = "Children_"
_FILTER_OBJECTS_CHILD_COUNT_SUFFIX = "_Count"


def _filter_objects_child_count_object_names(
    kwargs: Mapping[str, Any],
) -> tuple[str, ...]:
    feature_names = tuple(kwargs.get("measurement_features", ()))
    child_names = tuple(
        child_name
        for feature_name in feature_names
        for child_name in (_filter_objects_child_count_object_name(str(feature_name)),)
        if child_name is not None
    )
    return tuple(dict.fromkeys(child_names))


def _filter_objects_child_count_object_name(feature_name: str) -> str | None:
    if not feature_name.startswith(_FILTER_OBJECTS_CHILD_COUNT_PREFIX):
        return None
    if not feature_name.endswith(_FILTER_OBJECTS_CHILD_COUNT_SUFFIX):
        return None
    child_name = feature_name[
        len(_FILTER_OBJECTS_CHILD_COUNT_PREFIX) : -len(_FILTER_OBJECTS_CHILD_COUNT_SUFFIX)
    ]
    return child_name or None


@dataclass(frozen=True, slots=True)
class FilterObjectsRuntimeInputPlan:
    """Runtime object-label partition for one FilterObjects invocation."""

    object_specs: tuple[ArtifactSpec, ...]
    enclosing_spec: ArtifactSpec | None
    relationship_spec: ArtifactSpec | None = None
    measurement_relationship_specs: tuple[ArtifactSpec, ...] = ()

    @classmethod
    def from_inputs(
        cls,
        runtime_inputs: tuple[ArtifactSpec, ...],
        kwargs: Mapping[str, Any],
    ) -> "FilterObjectsRuntimeInputPlan":
        object_inputs = _specs_of_kind(runtime_inputs, ArtifactKind.OBJECT_LABELS)
        object_count = int(kwargs.get("additional_object_count", 0)) + 1
        enclosing_name = kwargs.get("enclosing_object_name")
        object_specs = object_inputs[:object_count]
        enclosing_spec = None
        relationship_spec = None
        measurement_relationship_specs: list[ArtifactSpec] = []
        if enclosing_name is not None:
            enclosing_spec = _spec_by_name(object_inputs, str(enclosing_name))
            if enclosing_spec is None:
                raise RuntimeError(
                    "FilterObjects enclosing object input "
                    f"{enclosing_name!r} was not declared in the runtime contract."
                )
            if object_specs:
                relationship_spec = _spec_by_name_and_kind(
                    runtime_inputs,
                    parent_child_relationship_artifact_name(
                        str(enclosing_name),
                        object_specs[0].name,
                    ),
                    ArtifactKind.RELATIONSHIPS,
                )
        if object_specs:
            for child_object_name in _filter_objects_child_count_object_names(kwargs):
                relationship = _spec_by_name_and_kind(
                    runtime_inputs,
                    parent_child_relationship_artifact_name(
                        object_specs[0].name,
                        child_object_name,
                    ),
                    ArtifactKind.RELATIONSHIPS,
                )
                if relationship is not None:
                    measurement_relationship_specs.append(relationship)
        return cls(
            object_specs=object_specs,
            enclosing_spec=enclosing_spec,
            relationship_spec=relationship_spec,
            measurement_relationship_specs=_unique_specs(measurement_relationship_specs),
        )


class MeasureImageAreaOccupiedInputPolicy(ObjectRowsInputPolicy):
    """Bind ordered object rows for the generic area-occupied runner."""

    module_name = "MeasureImageAreaOccupiedBinary"


class FilterObjectsInputPolicy(ObjectRowsWithMeasurementsInputPolicy):
    """Bind ordered primary/additional object rows for FilterObjects."""

    module_name = "FilterObjects"
    supported_non_object_input_kinds = frozenset({ArtifactKind.RELATIONSHIPS})

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        plan = FilterObjectsRuntimeInputPlan.from_inputs(
            request.runtime_inputs or request.object_inputs,
            request.kwargs,
        )
        bound = super().bind(request.with_object_inputs(plan.object_specs))
        if plan.enclosing_spec is not None:
            bound["enclosing_object_labels"] = request.labels_for(plan.enclosing_spec)
        if plan.relationship_spec is not None:
            bound["parent_child_relationship"] = request.adapter.get_relationship(
                plan.relationship_spec.name
            )
        if plan.measurement_relationship_specs:
            bound["parent_child_relationships"] = tuple(
                request.adapter.get_relationship(relationship_spec.name)
                for relationship_spec in plan.measurement_relationship_specs
            )
        return bound


class CalculateMathInputPolicy(CellProfilerObjectInputPolicy):
    """Bind CalculateMath operands from runtime measurement/object state."""

    module_name = "CalculateMath"
    binds_without_declared_inputs = True

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        return {
            "operand1_value": _calculate_math_operand_value(
                request.adapter,
                request.kwargs,
                feature_kwarg="operand1_feature",
                object_kwarg="operand1_object_name",
                object_inputs=request.object_inputs,
                labels_for=request.labels_for,
            ),
            "operand2_value": _calculate_math_operand_value(
                request.adapter,
                request.kwargs,
                feature_kwarg="operand2_feature",
                object_kwarg="operand2_object_name",
                object_inputs=request.object_inputs,
                labels_for=request.labels_for,
            ),
        }


class CellProfilerPerObjectMeasurementPolicy:
    """Predicate for modules that need one absorbed call per object set."""

    module_names: ClassVar[tuple[str, ...]] = (
        _MEASURE_OBJECT_SIZE_SHAPE_MODULE,
        _MEASURE_OBJECT_INTENSITY_MODULE,
        _MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE,
        _MEASURE_TEXTURE_MODULE,
        _MEASURE_COLOCALIZATION_MODULE,
        _MEASURE_GRANULARITY_MODULE,
    )
    # Per-object measurements usually measure each source image independently.
    # Channel-pair functions consume a composed image payload and declare that
    # exception here.
    composed_image_modules: ClassVar[tuple[str, ...]] = (
        _MEASURE_COLOCALIZATION_MODULE,
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
        return canonical_module_name(module_name) not in cls.composed_image_modules


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
    source_image_name: str | None
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
            source_image_name=(
                request.source_image_name
                or _measurement_source_name_for_specs(
                    request.executor._primary_image_inputs(request.func)
                )
            ),
            fields=_measurement_record_fields(request.spec, rows, request.func),
        )


class ObjectTopologyMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Object-only topology measurements are not qualified by image source."""

    module_name = _MEASURE_OBJECT_NEIGHBORS_MODULE

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
            source_image_name=None,
            fields=_measurement_record_fields(request.spec, rows, request.func),
        )


class CropMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Crop measurements describe the produced crop image artifact."""

    module_name = "Crop"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows = _measurement_table_rows(request.value)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            source_image_name=_primary_image_output_name(request.output_values),
            fields=_measurement_record_fields(request.spec, rows, request.func),
        )


class AlignMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose Align shifts as image-scoped measurements for each output image."""

    module_name = "Align"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        output_names = tuple(
            spec.name
            for spec in _specs_of_kind(request.executor.outputs, ArtifactKind.IMAGE)
        )
        return CellProfilerMeasurementRecord(
            rows=_align_measurement_rows(request.value, output_names),
            object_name=None,
            source_image_name=None,
        )


class RelateObjectsMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose CellProfiler parent-scoped relationship measurements."""

    module_name = "RelateObjects"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        return CellProfilerMeasurementRecord(
            rows=[
                *_measurement_table_rows(request.value),
                *_relationship_measurement_rows(request),
            ],
            object_name=None,
            source_image_name=request.source_image_name,
        )


class IdentifyObjectRelationshipsMeasurementRecordBuilder(
    CellProfilerMeasurementRecordBuilder
):
    """Expose object-creation relationships as generic measurement facts."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        return CellProfilerMeasurementRecord(
            rows=[
                *_threshold_measurement_rows(
                    request.value,
                    _single_output_object_name(request),
                ),
                *_relationship_measurement_rows(request),
            ],
            object_name=None,
            source_image_name=None,
        )


class ClassifyObjectsMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose classification bins as image and object measurement facts."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        object_name = _measurement_object_name(request.executor._declared_input_specs())
        return CellProfilerMeasurementRecord(
            rows=_classification_measurement_rows(request.value, object_name),
            object_name=None,
            source_image_name=None,
        )


class ClassifyObjectsSingleMeasurementRecordBuilder(
    ClassifyObjectsMeasurementRecordBuilder
):
    module_name = "ClassifyObjectsSingleMeasurement"


class ClassifyObjectsTwoMeasurementsRecordBuilder(
    ClassifyObjectsMeasurementRecordBuilder
):
    module_name = "ClassifyObjectsTwoMeasurements"


class CalculateMathMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose named math results without inherited image-source qualification."""

    module_name = "CalculateMath"

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
            source_image_name=None,
            fields=_measurement_record_fields(request.spec, rows, request.func),
        )


class IdentifyObjectsMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose segmentation threshold diagnostics as image-scope measurements."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        object_name = _single_output_object_name(request)
        return CellProfilerMeasurementRecord(
            rows=_threshold_measurement_rows(request.value, object_name),
            object_name=None,
            source_image_name=None,
        )


class IdentifyPrimaryObjectsMeasurementRecordBuilder(
    IdentifyObjectsMeasurementRecordBuilder
):
    module_name = "IdentifyPrimaryObjects"


class IdentifySecondaryObjectsMeasurementRecordBuilder(
    IdentifyObjectRelationshipsMeasurementRecordBuilder
):
    module_name = "IdentifySecondaryObjects"


class IdentifyTertiaryObjectsMeasurementRecordBuilder(
    CellProfilerMeasurementRecordBuilder
):
    """Expose tertiary parent-child relationships as generic measurement facts."""

    module_name = "IdentifyTertiaryObjects"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        return CellProfilerMeasurementRecord(
            rows=_relationship_measurement_rows(request),
            object_name=None,
            source_image_name=None,
        )


class TrackObjectsMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Expose TrackObjects long-form image and object measurements."""

    module_name = "TrackObjects"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows = [
            dict(measurement_row_mapping(row))
            for row in _measurement_table_rows(request.value)
        ]
        object_name = _measurement_object_name(request.executor._declared_input_specs())
        for row in rows:
            if _measurement_row_has_object_identity(row):
                if object_name is not None:
                    row.setdefault(MEASUREMENT_OBJECT_NAME_FIELD, object_name)
            else:
                row.setdefault(MEASUREMENT_SOURCE_IMAGE_NAME_FIELD, "Image")
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            source_image_name=None,
            fields=_field_specs_for_rows(rows),
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
            source_image_name=measurement_record.source_image_name,
        )


class RelationshipsOutputRecorder(CellProfilerOutputRecorder):
    """Record parent-child relationship artifacts."""

    kind = ArtifactKind.RELATIONSHIPS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        if not isinstance(request.value, ParentChildRelationshipPayload):
            raise TypeError(
                f"{request.executor.module_name} relationship output "
                f"'{request.spec.name}' must be ParentChildRelationshipPayload, "
                f"got {type(request.value).__name__}."
            )
        parent_spec, child_spec = _relationship_endpoint_specs(
            request,
            request.spec,
        )
        request.adapter.add_relationship(
            request.spec.name,
            parent_object_name=parent_spec.name,
            child_object_name=child_spec.name,
            parent_ids=request.value.parent_ids,
            child_ids=request.value.child_ids,
            slice_indices=request.value.slice_indices,
            slice_count=request.value.slice_count,
        )


class SpatialGridOutputRecorder(CellProfilerOutputRecorder):
    """Record spatial-grid outputs."""

    kind = ArtifactKind.SPATIAL_GRID

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_spatial_grid(
            request.spec.name,
            _coerce_spatial_grid(request.value, request.spec.name),
        )


_OUTPUT_RECORDING_PRIORITY = MappingProxyType(
    {
        ArtifactKind.IMAGE: 0,
        ArtifactKind.OBJECT_LABELS: 0,
        ArtifactKind.SPATIAL_GRID: 0,
        ArtifactKind.RELATIONSHIPS: 1,
        ArtifactKind.MEASUREMENTS: 2,
    }
)


def _output_recording_order(
    output_specs: tuple[ArtifactSpec, ...],
) -> tuple[ArtifactSpec, ...]:
    return tuple(
        sorted(
            output_specs,
            key=lambda spec: _OUTPUT_RECORDING_PRIORITY.get(spec.kind, 99),
        )
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


def _primary_image_output_name(output_values: Mapping[str, Any]) -> str | None:
    for name, value in output_values.items():
        try:
            array = np.asarray(value)
        except Exception:
            continue
        if array.ndim >= 2:
            return name
    return None


def _single_output_object_name(request: CellProfilerOutputRecordRequest) -> str:
    object_outputs = _specs_of_kind(request.executor.outputs, ArtifactKind.OBJECT_LABELS)
    if len(object_outputs) != 1:
        raise NotImplementedError(
            f"{request.executor.module_name} threshold measurement semantics "
            f"require exactly one object-label output, got "
            f"{[spec.name for spec in object_outputs]}."
        )
    return object_outputs[0].name


def _classification_measurement_rows(
    results: Any,
    object_name: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in _measurement_table_rows(results):
        bin_counts = _json_object_mapping(getattr(result, "bin_counts", {}))
        bin_percentages = _json_object_mapping(
            getattr(result, "bin_percentages", {})
        )
        object_classes = _json_object_mapping(getattr(result, "object_classes", {}))
        slice_index = int(getattr(result, "slice_index", 0))
        bin_names = tuple(str(name) for name in bin_counts)
        for bin_name, count in bin_counts.items():
            rows.append(
                {
                    "slice_index": slice_index,
                    MEASUREMENT_FEATURE_NAME_FIELD: (
                        f"Classify_{bin_name}_NumObjectsPerBin"
                    ),
                    MEASUREMENT_RESULT_VALUE_FIELD: count,
                }
            )
            rows.append(
                {
                    "slice_index": slice_index,
                    MEASUREMENT_FEATURE_NAME_FIELD: (
                        f"Classify_{bin_name}_PctObjectsPerBin"
                    ),
                    MEASUREMENT_RESULT_VALUE_FIELD: bin_percentages.get(bin_name, 0.0),
                }
            )
        if object_name is None:
            continue
        total_objects = int(getattr(result, "total_objects", 0))
        class_labels = tuple(sorted(int(label) for label in object_classes))
        dense_labels = tuple(range(1, total_objects + 1))
        object_labels = tuple(dict.fromkeys((*dense_labels, *class_labels)))
        for object_label in object_labels:
            class_name = object_classes.get(str(object_label))
            for bin_name in bin_names:
                rows.append(
                    {
                        MEASUREMENT_OBJECT_NAME_FIELD: object_name,
                        MEASUREMENT_OBJECT_LABEL_FIELD: object_label,
                        "slice_index": slice_index,
                        MEASUREMENT_FEATURE_NAME_FIELD: f"Classify_{bin_name}",
                        MEASUREMENT_RESULT_VALUE_FIELD: int(class_name == bin_name),
                    }
                )
    return rows


def _align_measurement_rows(
    results: Any,
    output_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in _measurement_table_rows(results):
        output_index = int(_measurement_row_value(result, "output_index", 0))
        if output_index < 0 or output_index >= len(output_names):
            raise ValueError(
                f"Align measurement output_index {output_index} does not match "
                f"declared image outputs {output_names!r}."
            )
        slice_index = int(_measurement_row_value(result, "slice_index", 0))
        source_image_name = output_names[output_index]
        rows.extend(
            (
                {
                    "slice_index": slice_index,
                    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD: source_image_name,
                    MEASUREMENT_FEATURE_NAME_FIELD: "Align_Xshift",
                    MEASUREMENT_RESULT_VALUE_FIELD: float(
                        _measurement_row_value(result, "x_shift", 0.0)
                    ),
                },
                {
                    "slice_index": slice_index,
                    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD: source_image_name,
                    MEASUREMENT_FEATURE_NAME_FIELD: "Align_Yshift",
                    MEASUREMENT_RESULT_VALUE_FIELD: float(
                        _measurement_row_value(result, "y_shift", 0.0)
                    ),
                },
            )
        )
    return rows


def _measurement_row_value(row: Any, name: str, default: Any) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _json_object_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value in (None, ""):
        return {}
    parsed = json.loads(str(value))
    if not isinstance(parsed, Mapping):
        raise TypeError(f"Expected JSON object mapping, got {type(parsed).__name__}.")
    return parsed


def _threshold_measurement_rows(stats: Any, object_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slice_stats in _measurement_table_rows(stats):
        slice_index = getattr(slice_stats, "slice_index", 0)
        final_threshold = getattr(
            slice_stats,
            "threshold_used",
            getattr(slice_stats, "threshold_value", 0.0),
        )
        values = {
            f"FinalThreshold_{object_name}": final_threshold,
            f"OrigThreshold_{object_name}": getattr(
                slice_stats,
                "original_threshold",
                final_threshold,
            ),
            f"WeightedVariance_{object_name}": getattr(
                slice_stats,
                "weighted_variance",
                0.0,
            ),
            f"SumOfEntropies_{object_name}": getattr(
                slice_stats,
                "sum_of_entropies",
                0.0,
            ),
        }
        rows.extend(
            {
                "slice_index": slice_index,
                MEASUREMENT_FEATURE_NAME_FIELD: feature_name,
                MEASUREMENT_RESULT_VALUE_FIELD: value,
            }
            for feature_name, value in values.items()
        )
    return rows


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


def _measurement_row_has_object_identity(row: Mapping[str, Any]) -> bool:
    return any(
        row.get(field_name) not in (None, "")
        for field_name in _MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS
    )


def _field_specs_for_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[FieldSpec, ...]:
    field_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field_name in row:
            if field_name in seen:
                continue
            seen.add(field_name)
            field_names.append(str(field_name))
    return tuple(FieldSpec(field_name) for field_name in field_names)


def _complete_object_measurement_rows(
    rows: Sequence[Any],
    *,
    label_payload: Any,
    func: Callable[..., Any],
    object_identity: MeasurementObjectRowIdentity | str = (
        MeasurementObjectRowIdentity.LABEL_ID
    ),
    row_policy: CellProfilerObjectMeasurementRowPolicy | None = None,
) -> list[Any]:
    """Pad per-object measurement rows across the dense object-label ID domain."""
    resolved_identity = MeasurementObjectRowIdentity(object_identity)
    resolved_row_policy = row_policy or DefaultObjectMeasurementRowPolicy()
    field_names = _measurement_row_field_names(rows, func)
    object_id_field = _measurement_object_id_field_for_fields(field_names)
    axis_fields = _measurement_axis_fields_for_fields(field_names)
    projected_rows = _project_object_measurement_row_identity(
        rows,
        object_identity=resolved_identity,
        object_id_field=object_id_field,
        axis_fields=axis_fields,
    )
    if projected_rows and not any(
        _measurement_row_has_object_identity(measurement_row_mapping(row))
        for row in projected_rows
    ):
        return list(projected_rows)
    axis_keys = _measurement_axis_keys(rows, axis_fields)
    if not axis_keys:
        axis_keys = ((),)
    object_ids_by_axis = {
        axis_key: _measurement_object_row_ids_for_axis(
            label_payload,
            object_identity=resolved_identity,
            axis_fields=axis_fields,
            axis_key=axis_key,
        )
        for axis_key in axis_keys
    }
    object_ids = tuple(
        sorted(
            {
                object_id
                for axis_object_ids in object_ids_by_axis.values()
                for object_id in axis_object_ids
            }
        )
    )
    if not object_ids:
        return list(projected_rows)

    present_row_keys = {
        (object_id, _measurement_axis_key(measurement_row_mapping(row), axis_fields))
        for row in projected_rows
        if (
            object_id := measurement_object_label(
                measurement_row_mapping(row),
                object_id_field=object_id_field,
            )
        )
        is not None
    }
    positive_label_extent_by_axis = {
        axis_key: _positive_label_extent_for_missing_measurements(
            _label_payload_for_measurement_axis(
                label_payload,
                axis_fields=axis_fields,
                axis_key=axis_key,
            ),
            row_policy=resolved_row_policy,
        )
        for axis_key in axis_keys
    }
    completed_rows = list(projected_rows)
    for axis_key in axis_keys:
        for object_id in object_ids_by_axis[axis_key]:
            if (object_id, axis_key) in present_row_keys:
                continue
            row = _missing_object_measurement_row(
                field_names,
                object_id_field=object_id_field,
                object_id=object_id,
                axis_fields=axis_fields,
                axis_key=axis_key,
                label_payload=label_payload,
                row_policy=resolved_row_policy,
                positive_label_extent=positive_label_extent_by_axis[axis_key],
            )
            completed_rows.append(row)
    return _order_object_measurement_rows(
        completed_rows,
        object_ids=object_ids,
        object_id_field=object_id_field,
        axis_fields=axis_fields,
        axis_keys=axis_keys,
    )


def _order_object_measurement_rows(
    rows: Sequence[Any],
    *,
    object_ids: Sequence[int],
    object_id_field: str,
    axis_fields: Sequence[str],
    axis_keys: Sequence[tuple[Any, ...]],
) -> list[Any]:
    """Return completed measurement rows in dense object-domain order."""
    object_order = {object_id: index for index, object_id in enumerate(object_ids)}
    axis_order = {axis_key: index for index, axis_key in enumerate(axis_keys)}
    indexed_rows = tuple(enumerate(rows))
    return [
        row
        for _index, row in sorted(
            indexed_rows,
            key=lambda item: _object_measurement_row_order_key(
                item[1],
                item[0],
                object_order=object_order,
                object_id_field=object_id_field,
                axis_fields=axis_fields,
                axis_order=axis_order,
            ),
        )
    ]


def _object_measurement_row_order_key(
    row: Any,
    fallback_index: int,
    *,
    object_order: Mapping[int, int],
    object_id_field: str,
    axis_fields: Sequence[str],
    axis_order: Mapping[tuple[Any, ...], int],
) -> tuple[int, int, int]:
    row_mapping = measurement_row_mapping(row)
    object_id = measurement_object_label(row_mapping, object_id_field=object_id_field)
    axis_key = _measurement_axis_key(row_mapping, axis_fields)
    return (
        axis_order.get(axis_key, len(axis_order)),
        object_order.get(object_id, len(object_order)) if object_id is not None else len(object_order),
        fallback_index,
    )


def _measurement_object_row_ids(
    label_payload: Any,
    *,
    object_identity: MeasurementObjectRowIdentity,
) -> tuple[int, ...]:
    label_ids = dense_object_label_id_domain(label_payload)
    if object_identity is MeasurementObjectRowIdentity.LABEL_ID:
        return label_ids
    if object_identity is MeasurementObjectRowIdentity.ROW_ORDINAL:
        return tuple(range(1, len(label_ids) + 1))
    raise ValueError(f"Unsupported measurement object row identity: {object_identity}.")


def _measurement_object_row_ids_for_axis(
    label_payload: Any,
    *,
    object_identity: MeasurementObjectRowIdentity,
    axis_fields: Sequence[str],
    axis_key: tuple[Any, ...],
) -> tuple[int, ...]:
    return _measurement_object_row_ids(
        _label_payload_for_measurement_axis(
            label_payload,
            axis_fields=axis_fields,
            axis_key=axis_key,
        ),
        object_identity=object_identity,
    )


def _label_payload_for_measurement_axis(
    label_payload: Any,
    *,
    axis_fields: Sequence[str],
    axis_key: tuple[Any, ...],
) -> Any:
    normalized_axis_fields = tuple(
        str(field_name).strip().lower() for field_name in axis_fields
    )
    if "slice_index" not in normalized_axis_fields:
        return label_payload
    slice_axis_position = normalized_axis_fields.index("slice_index")
    if slice_axis_position >= len(axis_key):
        return label_payload
    slice_index = int(axis_key[slice_axis_position])
    labels = np.asarray(_label_payload_final(label_payload))
    if labels.ndim < 3:
        return label_payload
    if slice_index < 0 or slice_index >= labels.shape[0]:
        raise ValueError(
            f"Measurement slice_index {slice_index} is outside label stack "
            f"with {labels.shape[0]} slices."
        )
    return _slice_pure_2d_value(label_payload, slice_index, labels.shape[0])


def _positive_object_label_extent(label_payload: Any) -> int:
    """Return the highest positive dense label ID present in the payload."""
    labels = np.asarray(_label_payload_final(label_payload))
    if labels.size == 0:
        return 0
    positive_labels = labels[labels > 0]
    if positive_labels.size == 0:
        return 0
    return int(np.max(positive_labels))


def _positive_label_extent_for_missing_measurements(
    label_payload: Any,
    *,
    row_policy: CellProfilerObjectMeasurementRowPolicy,
) -> int | None:
    policy = MissingObjectMeasurementValuePolicy(type(row_policy).missing_value_policy)
    if policy is not MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT:
        return None
    return _positive_object_label_extent(label_payload)


def _project_object_measurement_row_identity(
    rows: Sequence[Any],
    *,
    object_identity: MeasurementObjectRowIdentity,
    object_id_field: str,
    axis_fields: Sequence[str],
) -> list[Any]:
    if object_identity is MeasurementObjectRowIdentity.LABEL_ID:
        return list(rows)
    if object_identity is MeasurementObjectRowIdentity.ROW_ORDINAL:
        return _ordinal_object_measurement_rows(
            rows,
            object_id_field=object_id_field,
            axis_fields=axis_fields,
        )
    raise ValueError(f"Unsupported measurement object row identity: {object_identity}.")


def _ordinal_object_measurement_rows(
    rows: Sequence[Any],
    *,
    object_id_field: str,
    axis_fields: Sequence[str],
) -> list[Any]:
    ordinal_by_axis: dict[tuple[Any, ...], int] = {}
    ordinal_by_original_id: dict[tuple[tuple[Any, ...], int], int] = {}
    projected_rows: list[Any] = []
    for row in rows:
        row_mapping = measurement_row_mapping(row)
        axis_key = _measurement_axis_key(row_mapping, axis_fields)
        original_id = measurement_object_label(
            row_mapping,
            object_id_field=object_id_field,
        )
        ordinal_key = (axis_key, original_id) if original_id is not None else None
        ordinal = (
            ordinal_by_original_id.get(ordinal_key)
            if ordinal_key is not None
            else None
        )
        if ordinal is None:
            ordinal = ordinal_by_axis.get(axis_key, 0) + 1
            ordinal_by_axis[axis_key] = ordinal
            if ordinal_key is not None:
                ordinal_by_original_id[ordinal_key] = ordinal
        projected_rows.append(
            _measurement_row_with_object_id(
                row,
                object_id_field=object_id_field,
                object_id=ordinal,
            )
        )
    return projected_rows


def _measurement_row_with_object_id(
    row: Any,
    *,
    object_id_field: str,
    object_id: int,
) -> dict[str, Any]:
    projected_row = dict(measurement_row_mapping(row))
    projected_row[object_id_field] = object_id
    return projected_row


def _measurement_axis_fields_for_fields(field_names: Sequence[str]) -> tuple[str, ...]:
    axis_field_names = measurement_row_axis_field_names()
    return tuple(
        field_name
        for field_name in field_names
        if (
            field_name in axis_field_names
            and field_name not in _MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS
        )
    )


def _measurement_axis_keys(
    rows: Sequence[Any],
    axis_fields: Sequence[str],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        dict.fromkeys(
            _measurement_axis_key(measurement_row_mapping(row), axis_fields)
            for row in rows
        )
    )


def _measurement_axis_key(
    row: Mapping[str, Any],
    axis_fields: Sequence[str],
) -> tuple[Any, ...]:
    return tuple(row.get(field_name) for field_name in axis_fields)


def _missing_object_measurement_row(
    field_names: Sequence[str],
    *,
    object_id_field: str,
    object_id: int,
    axis_fields: Sequence[str],
    axis_key: Sequence[Any],
    label_payload: Any,
    row_policy: CellProfilerObjectMeasurementRowPolicy,
    positive_label_extent: int | None = None,
) -> dict[str, Any]:
    axis_values = dict(zip(axis_fields, axis_key, strict=True))
    row = {
        field_name: row_policy.missing_measurement_value(
            object_id=object_id,
            label_payload=label_payload,
            field_name=field_name,
            positive_label_extent=positive_label_extent,
        )
        for field_name in field_names
        if (
            field_name not in _MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS
            and field_name not in axis_values
        )
    }
    row.update(axis_values)
    row[object_id_field] = object_id
    return row


def _measurement_row_field_names(
    rows: Sequence[Any],
    func: Callable[..., Any],
) -> tuple[str, ...]:
    if rows:
        return tuple(str(key) for key in measurement_row_mapping(rows[0]).keys())
    return tuple(field.name for field in _measurement_fields_from_callable(func))


def _measurement_object_id_field_for_fields(field_names: Sequence[str]) -> str:
    for field_name in field_names:
        if field_name in _MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS:
            return field_name
    return MEASUREMENT_OBJECT_LABEL_FIELD


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


def _coerce_spatial_grid(value: Any, name: str) -> SpatialGrid | Mapping[str, Any]:
    if isinstance(value, SpatialGrid):
        return value.with_name(name)
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return {
            field.name: getattr(value, field.name)
            for field in dataclass_fields(value)
        }
    raise TypeError(
        f"Spatial grid output '{name}' must be SpatialGrid or mapping-backed, "
        f"got {type(value).__name__}."
    )


def _measurement_record_fields(
    spec: ArtifactSpec,
    rows: Sequence[Any],
    func: Callable[..., Any],
) -> tuple[FieldSpec, ...]:
    fields = _measurement_fields_from_materialization(spec)
    if fields:
        return fields
    fields = _measurement_fields_from_callable_materialization(func)
    if fields:
        return fields
    if _rows_have_inferable_fields(rows):
        return ()
    return _measurement_fields_from_callable(func)


def _measurement_fields_from_materialization(
    spec: ArtifactSpec,
) -> tuple[FieldSpec, ...]:
    field_names = tabular_field_names_from_materialization(spec.materialization)
    return tuple(FieldSpec(name) for name in field_names)


def _measurement_fields_from_callable_materialization(
    func: Callable[..., Any],
) -> tuple[FieldSpec, ...]:
    raw_outputs = vars(unwrap(func)).get("__special_outputs__", ())
    if not isinstance(raw_outputs, tuple):
        return ()
    field_sets = tuple(
        field_names
        for output_spec in raw_outputs
        if (
            isinstance(output_spec, tuple)
            and len(output_spec) == 2
            and (
                field_names := tabular_field_names_from_materialization(output_spec[1])
            )
        )
    )
    if len(field_sets) != 1:
        return ()
    return tuple(FieldSpec(name) for name in field_sets[0])


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
    image_data = image_payload_data(image)
    if isinstance(image_data, AlignedImageStack):
        return _object_only_reference_image(image_data.slices[0])
    if is_color_image_stack(image_data):
        return image_data[0, :, :, 0]
    if is_color_image_slice(image_data):
        return image_data[:, :, 0]
    if (
        hasattr(image_data, "ndim")
        and image_data.ndim == 3
        and image_data.shape[0] >= 1
    ):
        return image_data[0]
    return image_data


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
    if (
        reference_domain is CellProfilerMeasurementImageDomain.OBJECT_LABELS
        and is_image_stack(image)
        and labels.ndim == 2
    ):
        aligned_image = image[0]
        if _measurement_image_shape_mismatches_labels(aligned_image, labels):
            return _object_label_domain_reference_image(aligned_image, labels)
        return aligned_image
    if is_color_image_stack(image):
        if labels.ndim == 3:
            aligned_image = image[..., 0]
        elif labels.ndim == 2:
            aligned_image = image[0, :, :, 0]
    elif is_color_image_slice(image) and labels.ndim == 2:
        aligned_image = image[:, :, 0]
    elif image.ndim == labels.ndim:
        aligned_image = image
    elif (
        reference_domain is CellProfilerMeasurementImageDomain.OBJECT_LABELS
        and image.ndim == labels.ndim + 1
        and getattr(image, "shape", (0,))[0] >= 1
    ):
        aligned_image = image[0]

    if (
        reference_domain is CellProfilerMeasurementImageDomain.OBJECT_LABELS
        and _measurement_image_shape_mismatches_labels(aligned_image, labels)
    ):
        return _object_label_domain_reference_image(aligned_image, labels)
    return aligned_image


def _image_scope_measurement_payload(image: Any) -> Any:
    """Return one image plane for image-scoped measurement functions."""
    return _collapse_singleton_stack_output(image)


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
    return labels


def _collapse_singleton_label_stack(labels: Any) -> Any:
    """Normalize singleton OpenHCS label stacks to one CellProfiler label plane."""
    if not hasattr(labels, "ndim"):
        return labels
    if labels.ndim == 3 and getattr(labels, "shape", (0,))[0] == 1:
        return labels[0]
    return labels


def _label_payload_final(payload: Any) -> Any:
    """Return the final label plane from a runtime label payload."""
    if isinstance(payload, ObjectLabelPayload):
        payload = payload.labels
    return _collapse_singleton_label_stack(payload)


def _label_payload_small_removed(payload: Any) -> Any | None:
    """Return the small-removed label variant when the runtime provides it."""
    if not isinstance(payload, ObjectLabelPayload):
        return None
    if payload.small_removed_labels is None:
        return None
    return _collapse_singleton_label_stack(payload.small_removed_labels)


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


def _spec_by_name_and_kind(
    specs: Sequence[ArtifactSpec],
    name: str,
    kind: ArtifactKind,
) -> ArtifactSpec | None:
    for spec in specs:
        if spec.name == name and spec.kind is kind:
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
    current_image: Any,
    external_object_names: frozenset[str],
) -> Any:
    if spec.name in external_object_names:
        return adapter.resolve_source_objects(spec.name, current_image).labels
    return adapter.get_objects(spec.name).labels


def _measurement_object_name(
    inputs: tuple[ArtifactSpec, ...],
) -> str | None:
    object_inputs = _specs_of_kind(inputs, ArtifactKind.OBJECT_LABELS)
    if len(object_inputs) == 1:
        return object_inputs[0].name
    return None


def _relationship_endpoint_specs(
    request: CellProfilerOutputRecordRequest,
    relationship_spec: ArtifactSpec,
) -> tuple[ArtifactSpec, ArtifactSpec]:
    object_inputs = _specs_of_kind(
        request.executor._declared_input_specs(),
        ArtifactKind.OBJECT_LABELS,
    )
    object_outputs = _specs_of_kind(request.executor.outputs, ArtifactKind.OBJECT_LABELS)
    candidate_children = (*object_inputs, *object_outputs)
    matches = tuple(
        (parent_spec, child_spec)
        for parent_spec in object_inputs
        for child_spec in candidate_children
        if parent_spec.name != child_spec.name
        and relationship_spec.name
        == _relationship_artifact_name(parent_spec.name, child_spec.name)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"{request.executor.module_name} relationship output "
            f"'{relationship_spec.name}' matches multiple object endpoint pairs."
        )
    if len(object_inputs) == 2 and not object_outputs:
        return object_inputs[0], object_inputs[1]
    raise NotImplementedError(
        f"{request.executor.module_name} relationship output "
        f"'{relationship_spec.name}' cannot be mapped to object endpoints from "
        f"inputs={[spec.name for spec in object_inputs]} and "
        f"outputs={[spec.name for spec in object_outputs]}."
    )


def _relationship_artifact_name(parent_name: str, child_name: str) -> str:
    return parent_child_relationship_artifact_name(parent_name, child_name)


def _relationship_output_entries(
    request: CellProfilerOutputRecordRequest,
) -> tuple[tuple[ArtifactSpec, ParentChildRelationshipPayload], ...]:
    return tuple(
        (spec, value)
        for spec in request.executor.outputs
        if spec.kind is ArtifactKind.RELATIONSHIPS
        for value in (request.output_values.get(spec.name),)
        if isinstance(value, ParentChildRelationshipPayload)
    )


def _relationship_measurement_rows(
    request: CellProfilerOutputRecordRequest,
) -> tuple[dict[str, int | str], ...]:
    rows: list[dict[str, int | str]] = []
    for relationship_spec, payload in _relationship_output_entries(request):
        parent_spec, child_spec = _relationship_endpoint_specs(
            request,
            relationship_spec,
        )
        rows.extend(
            _relationship_child_count_rows(
                request,
                parent_object_name=parent_spec.name,
                child_object_name=child_spec.name,
                payload=payload,
            )
        )
        rows.extend(
            _relationship_parent_rows(
                request,
                parent_object_name=parent_spec.name,
                child_object_name=child_spec.name,
                payload=payload,
            )
        )
    return tuple(rows)


def _relationship_child_count_rows(
    request: CellProfilerOutputRecordRequest,
    *,
    parent_object_name: str,
    child_object_name: str,
    payload: ParentChildRelationshipPayload,
) -> tuple[dict[str, int | str], ...]:
    sliced_pairs = _relationship_payload_pairs_by_slice(payload)
    if sliced_pairs is not None:
        rows: list[dict[str, int | str]] = []
        for slice_index, pairs in sliced_pairs:
            related_parent_ids = tuple(parent_id for parent_id, _child_id in pairs)
            rows.extend(
                _relationship_child_count_rows_for_ids(
                    request,
                    parent_object_name=parent_object_name,
                    child_object_name=child_object_name,
                    related_parent_ids=related_parent_ids,
                    slice_index=slice_index,
                )
            )
        return tuple(rows)
    return _relationship_child_count_rows_for_ids(
        request,
        parent_object_name=parent_object_name,
        child_object_name=child_object_name,
        related_parent_ids=tuple(int(parent_id) for parent_id in payload.parent_ids),
        slice_index=None,
    )


def _relationship_child_count_rows_for_ids(
    request: CellProfilerOutputRecordRequest,
    *,
    parent_object_name: str,
    child_object_name: str,
    related_parent_ids: tuple[int, ...],
    slice_index: int | None,
) -> tuple[dict[str, int | str], ...]:
    related_parent_ids = tuple(int(parent_id) for parent_id in related_parent_ids)
    parent_count = max(
        (
            _object_label_count_for_request(
                request,
                parent_object_name,
                slice_index=slice_index,
            ),
            *related_parent_ids,
        )
    )
    counts = {parent_id: 0 for parent_id in range(1, parent_count + 1)}
    for parent_id in related_parent_ids:
        if parent_id > 0:
            counts[parent_id] = counts.get(parent_id, 0) + 1
    feature_name = f"Children_{child_object_name}_Count"
    rows: list[dict[str, int | str]] = []
    for parent_id, count in counts.items():
        row = {
            MEASUREMENT_OBJECT_NAME_FIELD: parent_object_name,
            MEASUREMENT_OBJECT_LABEL_FIELD: parent_id,
            feature_name: count,
        }
        if slice_index is not None:
            row["slice_index"] = slice_index
        rows.append(row)
    return tuple(rows)


def _relationship_parent_rows(
    request: CellProfilerOutputRecordRequest,
    *,
    parent_object_name: str,
    child_object_name: str,
    payload: ParentChildRelationshipPayload,
) -> tuple[dict[str, int | str], ...]:
    sliced_pairs = _relationship_payload_pairs_by_slice(payload)
    if sliced_pairs is not None:
        rows: list[dict[str, int | str]] = []
        for slice_index, pairs in sliced_pairs:
            rows.extend(
                _relationship_parent_rows_for_pairs(
                    request,
                    parent_object_name=parent_object_name,
                    child_object_name=child_object_name,
                    pairs=pairs,
                    slice_index=slice_index,
                )
            )
        return tuple(rows)
    return _relationship_parent_rows_for_pairs(
        request,
        parent_object_name=parent_object_name,
        child_object_name=child_object_name,
        pairs=tuple(
            (int(parent_id), int(child_id))
            for parent_id, child_id in zip(
                payload.parent_ids,
                payload.child_ids,
                strict=True,
            )
        ),
        slice_index=None,
    )


def _relationship_parent_rows_for_pairs(
    request: CellProfilerOutputRecordRequest,
    *,
    parent_object_name: str,
    child_object_name: str,
    pairs: tuple[tuple[int, int], ...],
    slice_index: int | None,
) -> tuple[dict[str, int | str], ...]:
    parent_by_child = {
        int(child_id): int(parent_id)
        for parent_id, child_id in pairs
    }
    child_count = max(
        (
            _object_label_count_for_request(
                request,
                child_object_name,
                slice_index=slice_index,
            ),
            *parent_by_child.keys(),
        )
    )
    feature_name = f"Parent_{parent_object_name}"
    rows: list[dict[str, int | str]] = []
    for child_id in range(1, child_count + 1):
        row = {
            MEASUREMENT_OBJECT_NAME_FIELD: child_object_name,
            MEASUREMENT_OBJECT_LABEL_FIELD: child_id,
            feature_name: parent_by_child.get(child_id, 0),
        }
        if slice_index is not None:
            row["slice_index"] = slice_index
        rows.append(row)
    return tuple(rows)


def _relationship_payload_pairs_by_slice(
    payload: ParentChildRelationshipPayload,
) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...] | None:
    if payload.slice_count is None and not payload.slice_indices:
        return None
    if payload.slice_count is None:
        slice_count = max(payload.slice_indices) + 1 if payload.slice_indices else 0
    else:
        slice_count = payload.slice_count
    pairs_by_slice: list[list[tuple[int, int]]] = [[] for _ in range(slice_count)]
    if payload.slice_indices:
        for slice_index, parent_id, child_id in zip(
            payload.slice_indices,
            payload.parent_ids,
            payload.child_ids,
            strict=True,
        ):
            pairs_by_slice[slice_index].append((parent_id, child_id))
    elif payload.parent_ids:
        if slice_count != 1:
            raise ValueError(
                "ParentChildRelationshipPayload with multiple slices must carry "
                "slice_indices for non-empty relationships."
            )
        pairs_by_slice[0].extend(
            zip(payload.parent_ids, payload.child_ids, strict=True)
        )
    return tuple(
        (slice_index, tuple(pairs))
        for slice_index, pairs in enumerate(pairs_by_slice)
    )


def _object_label_count_for_request(
    request: CellProfilerOutputRecordRequest,
    object_name: str,
    *,
    slice_index: int | None = None,
) -> int:
    if object_name in request.output_values:
        return _object_label_count_from_value(
            request.output_values[object_name],
            slice_index=slice_index,
        )
    return _object_label_count(request.adapter, object_name, slice_index=slice_index)


def _object_label_count(
    adapter: CellProfilerRuntimeAdapter,
    object_name: str,
    *,
    slice_index: int | None = None,
) -> int:
    return _object_label_count_from_value(
        adapter.get_objects(object_name).labels,
        slice_index=slice_index,
    )


def _object_label_count_from_value(
    value: Any,
    *,
    slice_index: int | None,
) -> int:
    labels = value.labels if isinstance(value, ObjectLabelPayload) else value
    label_array = np.asarray(labels)
    if slice_index is not None and label_array.ndim >= 3:
        if slice_index < label_array.shape[0]:
            label_array = label_array[slice_index]
        elif label_array.shape[0] == 1:
            label_array = label_array[0]
        else:
            raise ValueError(
                "Object label stack does not contain requested slice "
                f"{slice_index}; shape={label_array.shape!r}."
            )
    if label_array.size == 0:
        return 0
    return int(label_array.max())


def _slice_aligned_measurement_values(
    value_slices: tuple[np.ndarray, ...],
) -> np.ndarray | CellProfilerSliceAlignedValues:
    if len(value_slices) == 1:
        return value_slices[0]
    return CellProfilerSliceAlignedValues(value_slices)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecialInputBindingRequest(RuntimeInputBindingRequestBase):
    """Authoritative runtime context for binding declared special_inputs."""

    parameter_names: tuple[str, ...]
    runtime_inputs: tuple[ArtifactSpec, ...]
    external_image_names: frozenset[str]
    runtime_image_names: frozenset[str]

    def __post_init__(self) -> None:
        RuntimeInputBindingRequestBase.__post_init__(self)
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
        request: SpecialInputBindingRequest,
    ) -> dict[str, Any]:
        """Return kwargs for a callable's declared special_inputs."""


class PositionalSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Bind special_inputs positionally to compiled runtime artifact specs."""

    def bind(
        self,
        request: SpecialInputBindingRequest,
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
        request: SpecialInputBindingRequest,
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
        request: SpecialInputBindingRequest,
    ) -> dict[str, Any]:
        object_inputs = request.object_inputs
        _require_exact_object_count(request.module_name, object_inputs, 1)
        object_spec = object_inputs[0]
        labels = request.labels_for(object_spec)
        measurement_tables = request.adapter.measurement_tables_for_object(
            object_spec.name
        )
        if "classification_rules" in request.kwargs:
            rules = request.kwargs["classification_rules"]
            if not isinstance(rules, (tuple, list)):
                raise ValueError(
                    f"{request.module_name} classification_rules must be an "
                    "ordered tuple or list."
                )
            return {
                "labels": labels,
                "measurement_values_by_rule": tuple(
                    _slice_aligned_measurement_values(
                        measurement_values_for_label_slices(
                            measurement_tables,
                            _classification_rule_measurement_feature(
                                rule,
                                request.module_name,
                            ),
                            labels,
                            object_name=object_spec.name,
                        )
                    )
                    for rule in rules
                ),
            }
        return {
            "labels": labels,
            **{
                parameter_name: _slice_aligned_measurement_values(
                    measurement_values_for_label_slices(
                        measurement_tables,
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


def _classification_rule_measurement_feature(
    rule: Any,
    module_name: str,
) -> str:
    if not isinstance(rule, Mapping):
        raise ValueError(f"{module_name} classification rule must be a mapping.")
    value = rule.get("measurement_feature")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{module_name} classification rule requires non-empty "
            "'measurement_feature'."
        )
    return value


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
    object_inputs: tuple[ArtifactSpec, ...] = (),
    labels_for: Callable[[ArtifactSpec], Any] | None = None,
) -> Any:
    feature_name = _required_string_kwarg(kwargs, feature_kwarg, "CalculateMath")
    object_name = _optional_string_kwarg(kwargs, object_kwarg)
    count_object_name = count_feature_object_name(feature_name)
    if count_object_name is not None:
        return float(_object_label_count(adapter, count_object_name))
    if object_name is None:
        return _calculate_math_image_operand_value(adapter, feature_name)
    label_spec = _spec_by_name(object_inputs, object_name)
    if label_spec is not None and labels_for is not None:
        return _slice_aligned_measurement_values(
            measurement_values_for_label_slices(
                adapter.measurement_tables_for_object(object_name),
                feature_name,
                labels_for(label_spec),
                object_name=object_name,
            )
        )
    values = measurement_values_for_feature(
        adapter.measurement_tables_for_object(object_name),
        feature_name,
        object_count=_object_label_count(adapter, object_name),
        object_name=object_name,
    )
    return float(values[0]) if len(values) == 1 else values


def _calculate_math_image_operand_value(
    adapter: CellProfilerRuntimeAdapter,
    feature_name: str,
) -> Any:
    measurement_tables = adapter.measurement_tables(match_group=False)
    slice_values = _calculate_math_image_operand_values_by_slice(
        measurement_tables,
        feature_name,
    )
    if slice_values is None:
        return measurement_scalar_value_for_feature(measurement_tables, feature_name)
    return _slice_aligned_measurement_values(slice_values)


def _calculate_math_image_operand_values_by_slice(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
) -> tuple[np.ndarray, ...] | None:
    query = MeasurementFeatureQuery(feature_name)
    values_by_slice: dict[int, list[float]] = {}
    unindexed_values: list[float] = []
    for table in measurement_tables:
        for row in measurement_rows((table,)):
            row_mapping = measurement_row_mapping(row)
            if _measurement_row_has_object_identity(row_mapping):
                continue
            value = query.row_value(row)
            if value is None:
                continue
            if "slice_index" not in row_mapping:
                unindexed_values.append(float(value))
                continue
            values_by_slice.setdefault(int(row_mapping["slice_index"]), []).append(
                float(value)
            )

    if not values_by_slice:
        return None
    if unindexed_values:
        raise ValueError(
            f"Measurement feature {feature_name!r} mixes slice-indexed and "
            "unindexed image values."
        )
    expected_indices = set(range(max(values_by_slice) + 1))
    if set(values_by_slice) != expected_indices:
        raise ValueError(
            f"Measurement feature {feature_name!r} has non-contiguous "
            f"slice_index values {sorted(values_by_slice)}; expected "
            f"{sorted(expected_indices)}."
        )

    slice_values: list[np.ndarray] = []
    for slice_index in range(len(expected_indices)):
        values = values_by_slice[slice_index]
        if len(values) != 1:
            raise ValueError(
                f"Measurement feature {feature_name!r} resolved to "
                f"{len(values)} values on slice {slice_index}; expected exactly "
                "one scalar value."
            )
        slice_values.append(np.asarray(values[0], dtype=float))
    return tuple(slice_values)


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


class CellProfilerDualScopeMeasurementPolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal policy for modules whose `Both` scope emits image and object facts."""

    __registry_key__ = _MODULE_NAME_REGISTRY_KEY
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None
    image_function_name: ClassVar[str | None] = None

    @classmethod
    def for_module(
        cls,
        module_name: str,
    ) -> "CellProfilerDualScopeMeasurementPolicy | None":
        policy_type = cls.__registry__.get(canonical_module_name(module_name))
        if policy_type is None:
            return None
        return policy_type()

    def image_function(self, object_func: Callable[..., Any]) -> Callable[..., Any]:
        del object_func
        return require_function(
            _required_class_attr(type(self).module_name, "module_name"),
            function_name=_required_class_attr(
                type(self).image_function_name,
                "image_function_name",
            ),
        )


@dataclass(frozen=True, slots=True)
class DualScopeMeasurementPolicySpec:
    """Declarative leaf spec for one dual-scope measurement module."""

    module_name: str
    image_function_name: str


class DeclaredDualScopeMeasurementPolicy(CellProfilerDualScopeMeasurementPolicy):
    """Generated base for modules with image+object measurement scope."""


def _declare_dual_scope_measurement_policy(
    spec: DualScopeMeasurementPolicySpec,
) -> None:
    type(
        f"{spec.module_name}DualScopeMeasurementPolicy",
        (DeclaredDualScopeMeasurementPolicy,),
        {
            "__module__": __name__,
            "module_name": spec.module_name,
            "image_function_name": spec.image_function_name,
        },
    )


for _dual_scope_policy_spec in (
    DualScopeMeasurementPolicySpec("MeasureTexture", "measure_texture"),
    DualScopeMeasurementPolicySpec(
        "MeasureColocalization",
        "measure_colocalization",
    ),
):
    _declare_dual_scope_measurement_policy(_dual_scope_policy_spec)


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
    request: SpecialInputBindingRequest,
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
    request: SpecialInputBindingRequest,
) -> Any:
    try:
        return _artifact_kind_strategy(spec.kind).runtime_input_value(
            RuntimeArtifactInputRequest(
                spec=spec,
                adapter=request.adapter,
                current_image=request.current_image,
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
) -> RuntimeArtifactKindStrategy:
    try:
        return RuntimeArtifactKindStrategy.for_kind(kind)
    except KeyError as exc:
        raise TypeError(
            f"No CellProfiler artifact kind strategy registered for {kind.value}."
        ) from exc


def _collapse_singleton_stack_output(value: Any) -> Any:
    metadata = image_payload_metadata(value)
    mask = image_payload_mask(value)
    if mask is not None or metadata.has_values:
        data = image_payload_data(value)
        collapsed_data = _collapse_singleton_stack_output(data)
        collapsed = collapsed_data is not data
        return image_payload_with_context(
            data=collapsed_data,
            mask=None if mask is None else _collapse_singleton_mask(mask),
            metadata=metadata.for_channel(0) if collapsed else metadata,
        )
    if hasattr(value, "ndim") and value.ndim == 3 and value.shape[0] == 1:
        return value[0]
    if is_color_image_stack(value) and value.shape[0] == 1:
        return value[0]
    if isinstance(value, tuple):
        return tuple(_collapse_singleton_stack_output(item) for item in value)
    return value


def _collapse_singleton_mask(mask: Any) -> Any:
    if hasattr(mask, "ndim") and mask.ndim == 3 and mask.shape[0] == 1:
        return mask[0]
    if hasattr(mask, "ndim") and mask.ndim == 4 and mask.shape[0] == 1:
        return mask[0]
    return mask


def _openhcs_main_flow_output(
    input_image: Any,
    output_image: Any,
) -> Any:
    input_data = image_payload_data(input_image)
    output_data = image_payload_data(output_image)
    output_mask = image_payload_mask(output_image)
    output_metadata = image_payload_metadata(output_image)
    if not is_image_stack(input_data):
        return output_image
    if not _is_image_slice(output_data):
        return output_image
    memory_type = detect_memory_type(input_data)
    stacked = ImageStackLayout.for_slices((output_data,)).stack(
        slices=(output_data,),
        memory_type=memory_type,
        gpu_id=0,
    )
    return image_payload_with_context(
        stacked,
        mask=output_mask,
        metadata=output_metadata,
    )


def _is_image_slice(value: Any) -> bool:
    return (hasattr(value, "ndim") and value.ndim == 2) or is_color_image_slice(value)


def _single_source_name(source_names: tuple[str, ...]) -> str | None:
    unique_names = tuple(dict.fromkeys(source_names))
    if len(unique_names) == 1:
        return unique_names[0]
    return None


def _pop_measurement_target_scope(
    kwargs: dict[str, Any],
    *,
    default: CellProfilerMeasurementTargetScope,
) -> CellProfilerMeasurementTargetScope:
    return coerce_cellprofiler_measurement_target_scope(
        kwargs.pop(CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG, None),
        default=default,
    )


def _single_measurement_image_source_name(
    measurement_images: tuple["CellProfilerMeasurementImage", ...],
) -> str | None:
    unique_names = tuple(
        dict.fromkeys(image.source_image_name for image in measurement_images)
    )
    if len(unique_names) == 1:
        return unique_names[0]
    return None


def _measurement_source_name_for_specs(
    image_inputs: tuple[ArtifactSpec, ...],
) -> str | None:
    if not image_inputs:
        return None
    return "__".join(spec.name for spec in image_inputs)


def _row_source_names_required(
    measurement_images: tuple["CellProfilerMeasurementImage", ...],
) -> bool:
    unique_names = tuple(
        dict.fromkeys(image.source_image_name for image in measurement_images)
    )
    return len(unique_names) > 1


def _required_class_attr[T](value: T | None, name: str) -> T:
    if value is None:
        raise TypeError(f"CellProfiler policy must define {name}.")
    return value


def _stack_cellprofiler_slice_outputs(
    slice_outputs: Sequence[Any],
    memory_type: str,
) -> Any:
    normalized_outputs = tuple(
        _collapse_singleton_stack_output(output) for output in slice_outputs
    )
    output_masks = tuple(image_payload_mask(output) for output in normalized_outputs)
    output_data = tuple(image_payload_data(output) for output in normalized_outputs)
    if all(_is_grayscale_slice_output(output) for output in output_data):
        stacked = stack_slices(list(output_data), memory_type, 0)
        return _with_stacked_output_context(
            stacked,
            normalized_outputs,
            output_masks,
            memory_type,
        )
    if all(is_color_image_slice(output) for output in output_data):
        stacked = np.stack(
            tuple(
                _as_numpy_payload(output)
                for output in output_data
            )
        )
        if memory_type == MEMORY_TYPE_NUMPY:
            converted = stacked
        else:
            converted = _convert_memory(stacked, MEMORY_TYPE_NUMPY, memory_type)
        return _with_stacked_output_context(
            converted,
            normalized_outputs,
            output_masks,
            memory_type,
        )
    raise ValueError(
        "CellProfiler slice outputs must be uniformly 2D grayscale or HWC "
        "color images; got shapes "
        f"{[getattr(output, 'shape', None) for output in output_data]!r}."
    )


def _unstack_cellprofiler_image_slices(image: Any, memory_type: str) -> tuple[Any, ...]:
    image_data = image_payload_data(image)
    image_mask = image_payload_mask(image)
    image_metadata = image_payload_metadata(image)
    if is_color_image_slice(image_data):
        return (image,)
    if is_color_image_stack(image_data):
        source_type = detect_memory_type(image_data)
        if source_type != memory_type:
            image_data = _convert_memory(image_data, source_type, memory_type)
        return tuple(
            _image_payload_slice(image_data[index], image_mask, image_metadata, index)
            for index in range(image_data.shape[0])
        )
    return tuple(
        _image_payload_slice(slice_data, image_mask, image_metadata, index)
        for index, slice_data in enumerate(unstack_slices(image_data, memory_type, 0))
    )


def _image_payload_slice(data: Any, mask: Any | None, metadata: Any, index: int) -> Any:
    return image_payload_with_context(
        data=data,
        mask=None if mask is None else _slice_mask(mask, index),
        metadata=metadata.for_channel(index),
    )


def _slice_mask(mask: Any, index: int) -> Any:
    if hasattr(mask, "ndim") and mask.ndim == 3:
        return mask[index]
    return mask


def _with_stacked_output_context(
    stacked: Any,
    slice_outputs: Sequence[Any],
    masks: Sequence[Any | None],
    memory_type: str,
) -> Any:
    metadata = compose_image_payload_metadata(slice_outputs)
    present_masks = tuple(mask for mask in masks if mask is not None)
    if not present_masks:
        return image_payload_with_context(stacked, metadata=metadata)
    if len(present_masks) != len(masks):
        raise ValueError("Cannot stack a mix of masked and unmasked image outputs.")
    stacked_mask = (
        present_masks[0]
        if len(present_masks) == 1
        else stack_slices(list(present_masks), memory_type, 0)
    )
    return image_payload_with_context(stacked, mask=stacked_mask, metadata=metadata)


def _is_grayscale_slice_output(output: Any) -> bool:
    return np.asarray(output).ndim == 2


def _as_numpy_payload(payload: Any) -> np.ndarray:
    payload = image_payload_data(payload)
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


def _illumination_scope_uses_all_images(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Enum):
        value = value.value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized.startswith("all")


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
        memory_type = detect_memory_type(image_payload_data(main_outputs[0]))
        stacked_main_output = _stack_cellprofiler_slice_outputs(
            main_outputs,
            memory_type,
        )
        if not auxiliary_groups:
            return stacked_main_output
        return (
            stacked_main_output,
            *(
                _aggregate_cellprofiler_pure_2d_auxiliary_output(values, memory_type)
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

        image_data = image_payload_data(image)
        memory_type = detect_memory_type(image_data)
        if image_data.ndim == 2:
            slice_count = _slice_count_from_pure_2d_kwargs(kwargs)
            if slice_count is None:
                return func(image, **kwargs)
            slices_2d = tuple(image for _ in range(slice_count))
        elif is_color_image_slice(image_data):
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
                _aggregate_cellprofiler_pure_2d_auxiliary_output(values, memory_type)
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
        result_data = image_payload_data(result_2d)
        result_mask = image_payload_mask(result_2d)
        result_metadata = image_payload_metadata(result_2d)
        memory_type = detect_memory_type(result_data)
        stacked = stack_slices([result_data], memory_type, 0)
        return image_payload_with_context(
            stacked,
            mask=result_mask,
            metadata=result_metadata,
        )


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
        name: (
            tuple(
                _slice_pure_2d_value(item, slice_index, slice_count)
                for item in value
            )
            if name in _OBJECT_ROW_SEQUENCE_KWARGS and isinstance(value, tuple)
            else _slice_pure_2d_value(value, slice_index, slice_count)
        )
        for name, value in kwargs.items()
    }


def _aggregate_cellprofiler_pure_2d_auxiliary_output(
    values: list[Any],
    memory_type: str,
) -> Any:
    if values and all(
        isinstance(value, (MaskedImagePayload, ImageMetadataPayload))
        for value in values
    ):
        return _stack_cellprofiler_slice_outputs(values, memory_type)
    if values and all(
        isinstance(value, ParentChildRelationshipPayload)
        for value in values
    ):
        return ParentChildRelationshipPayload(
            parent_ids=tuple(
                parent_id
                for value in values
                for parent_id in value.parent_ids
            ),
            child_ids=tuple(
                child_id
                for value in values
                for child_id in value.child_ids
            ),
            slice_indices=tuple(
                slice_index
                for slice_index, value in enumerate(values)
                for _child_id in value.child_ids
            ),
            slice_count=len(values),
        )
    return _aggregate_pure_2d_auxiliary_output(values, memory_type)


def _slice_count_from_pure_2d_kwargs(
    kwargs: Mapping[str, Any],
) -> int | None:
    tensor_slice_counts = {
        stack.shape[0]
        for value in kwargs.values()
        for stack in _slice_aligned_stack_views(value)
        if stack.shape[0] > 1
    }
    tensor_slice_counts.update(
        value.slice_count
        for value in kwargs.values()
        if isinstance(value, CellProfilerSliceAlignedValues) and value.slice_count > 1
    )
    tensor_slice_counts.update(
        count
        for value in kwargs.values()
        if isinstance(value, (ParentChildRelationshipPayload, ObjectRelationship))
        for count in (_relationship_slice_count(value),)
        if count is not None and count > 1
    )
    tensor_slice_count = _single_slice_count(
        tensor_slice_counts,
        source_description="tensor/vector kwargs",
    )
    if tensor_slice_count is not None:
        return tensor_slice_count

    measurement_table_slice_counts = {
        count
        for value in kwargs.values()
        if (count := _measurement_table_slice_count(value)) is not None
        if count > 1
    }
    measurement_table_slice_count = _single_slice_count(
        measurement_table_slice_counts,
        source_description="measurement table kwargs",
    )
    if measurement_table_slice_count is not None:
        return measurement_table_slice_count

    if any(
        stack.shape[0] == 1
        for value in kwargs.values()
        for stack in _slice_aligned_stack_views(value)
    ):
        return 1
    return None


def _single_slice_count(
    slice_counts: set[int],
    *,
    source_description: str,
) -> int | None:
    if len(slice_counts) > 1:
        raise ValueError(
            "Cannot align PURE_2D invocation with conflicting "
            f"{source_description} slice counts: {sorted(slice_counts)}."
        )
    if slice_counts:
        return next(iter(slice_counts))
    return None


def _measurement_table_slice_count(value: Any) -> int | None:
    if isinstance(value, MeasurementTable):
        return _measurement_table_row_slice_count(value)
    if isinstance(value, tuple | list):
        slice_counts = {
            count
            for item in value
            if (count := _measurement_table_slice_count(item)) is not None
        }
        if len(slice_counts) > 1:
            raise ValueError(
                "Cannot align PURE_2D invocation with conflicting measurement "
                f"table slice counts: {sorted(slice_counts)}."
            )
        return next(iter(slice_counts)) if slice_counts else None
    return None


def _measurement_table_row_slice_count(table: MeasurementTable) -> int | None:
    if not _measurement_table_drives_slice_alignment(table):
        return None
    slice_indices = {
        int(row_mapping["slice_index"])
        for row in measurement_rows((table,))
        for row_mapping in (measurement_row_mapping(row),)
        if "slice_index" in row_mapping
    }
    if not slice_indices:
        return None
    expected_indices = set(range(max(slice_indices) + 1))
    if slice_indices != expected_indices:
        raise ValueError(
            f"Measurement table '{table.name}' has non-contiguous slice_index "
            f"values {sorted(slice_indices)}; expected "
            f"{sorted(expected_indices)}."
        )
    return len(slice_indices)


def _measurement_table_drives_slice_alignment(table: MeasurementTable) -> bool:
    """Only entity-scoped measurement tables can define a runtime slice axis."""
    return table.subject.scope in (MeasurementScope.IMAGE, MeasurementScope.OBJECT)


def _should_slice_flexible_object_invocation(
    object_inputs: tuple[ArtifactSpec, ...],
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> bool:
    if not object_inputs:
        return False
    if _processing_contract_for_callable(func) is not ProcessingContract.FLEXIBLE:
        return False
    slice_count = _slice_count_from_pure_2d_kwargs(kwargs)
    return slice_count is not None and slice_count > 1


def _slice_pure_2d_value(value: Any, slice_index: int, slice_count: int) -> Any:
    if isinstance(value, CellProfilerSliceAlignedValues):
        return value.value_for_slice(slice_index)
    if isinstance(value, MeasurementTable):
        return measurement_table_for_slice(value, slice_index)
    if isinstance(value, ParentChildRelationshipPayload):
        return _slice_parent_child_relationship_payload(value, slice_index)
    if isinstance(value, ObjectRelationship):
        return _slice_object_relationship(value, slice_index)
    if isinstance(value, ObjectLabelPayload):
        return ObjectLabelPayload(
            labels=_slice_pure_2d_value(value.labels, slice_index, slice_count),
            unedited_labels=(
                None
                if value.unedited_labels is None
                else _slice_pure_2d_value(
                    value.unedited_labels,
                    slice_index,
                    slice_count,
                )
            ),
            small_removed_labels=(
                None
                if value.small_removed_labels is None
                else _slice_pure_2d_value(
                    value.small_removed_labels,
                    slice_index,
                    slice_count,
                )
            ),
            declared_object_count=value.declared_object_count,
            declared_object_ids=value.declared_object_ids,
        )
    if isinstance(value, tuple):
        stack = _slice_aligned_stack_view(value) if len(value) > 1 else None
        if stack is not None:
            if stack.shape[0] == slice_count:
                return stack[slice_index]
            if stack.shape[0] == 1:
                return stack[0]
            return value
        return tuple(
            _slice_pure_2d_value(item, slice_index, slice_count)
            for item in value
        )
    if isinstance(value, list):
        stack = _slice_aligned_stack_view(value) if len(value) > 1 else None
        if stack is not None:
            if stack.shape[0] == slice_count:
                return stack[slice_index]
            if stack.shape[0] == 1:
                return stack[0]
            return value
        return [
            _slice_pure_2d_value(item, slice_index, slice_count)
            for item in value
        ]
    metadata = image_payload_metadata(value)
    mask = image_payload_mask(value)
    if mask is not None or metadata.has_values:
        data = _slice_pure_2d_value(
            image_payload_data(value),
            slice_index,
            slice_count,
        )
        return image_payload_with_context(
            data=data,
            mask=None if mask is None else _slice_mask(mask, slice_index),
            metadata=metadata.for_channel(slice_index),
        )
    stack = _slice_aligned_stack_view(value)
    if stack is None:
        return value
    if stack.shape[0] == slice_count:
        return stack[slice_index]
    if stack.shape[0] == 1:
        return stack[0]
    return value


def _slice_parent_child_relationship_payload(
    value: ParentChildRelationshipPayload,
    slice_index: int,
) -> ParentChildRelationshipPayload:
    if not value.slice_indices:
        if value.slice_count is not None and value.slice_count > 1 and value.parent_ids:
            raise ValueError(
                "Cannot slice multi-plane ParentChildRelationshipPayload without "
                "slice_indices."
            )
        return value

    parent_ids: list[int] = []
    child_ids: list[int] = []
    for parent_id, child_id, relationship_slice_index in zip(
        value.parent_ids,
        value.child_ids,
        value.slice_indices,
        strict=True,
    ):
        if relationship_slice_index != slice_index:
            continue
        parent_ids.append(parent_id)
        child_ids.append(child_id)
    return ParentChildRelationshipPayload(
        parent_ids=tuple(parent_ids),
        child_ids=tuple(child_ids),
        slice_count=1,
    )


def _slice_object_relationship(
    value: ObjectRelationship,
    slice_index: int,
) -> ObjectRelationship:
    source_ids_all = tuple(int(source_id) for source_id in value.source_ids)
    target_ids_all = tuple(int(target_id) for target_id in value.target_ids)
    if not value.slice_indices:
        if value.slice_count is not None and value.slice_count > 1 and source_ids_all:
            raise ValueError(
                "Cannot slice multi-plane ObjectRelationship without slice_indices."
            )
        return value

    source_ids: list[int] = []
    target_ids: list[int] = []
    for source_id, target_id, relationship_slice_index in zip(
        source_ids_all,
        target_ids_all,
        value.slice_indices,
        strict=True,
    ):
        if relationship_slice_index != slice_index:
            continue
        source_ids.append(source_id)
        target_ids.append(target_id)
    return ObjectRelationship(
        name=value.name,
        source=value.source,
        target=value.target,
        source_ids=tuple(source_ids),
        target_ids=tuple(target_ids),
        relationship_type=value.relationship_type,
        slice_count=1,
    )


def _relationship_slice_count(
    value: ParentChildRelationshipPayload | ObjectRelationship,
) -> int | None:
    if value.slice_count is not None:
        return value.slice_count
    if not value.slice_indices:
        return None
    return max(value.slice_indices) + 1


def _slice_aligned_stack_views(value: Any) -> tuple[Any, ...]:
    if isinstance(value, CellProfilerSliceAlignedValues):
        return ()
    if isinstance(value, ObjectLabelPayload):
        values = [
            value.labels,
            value.unedited_labels,
            value.small_removed_labels,
        ]
        return tuple(
            stack
            for item in values
            if item is not None
            if (stack := _slice_aligned_stack_view(item)) is not None
        )
    if isinstance(value, (tuple, list)):
        stack = _slice_aligned_stack_view(value) if len(value) > 1 else None
        if stack is not None:
            return (stack,)
        return tuple(
            stack
            for item in value
            for stack in _slice_aligned_stack_views(item)
        )
    stack = _slice_aligned_stack_view(value)
    return () if stack is None else (stack,)


def _slice_aligned_stack_view(value: Any) -> Any | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    try:
        stack = np.asarray(image_payload_data(value))
    except (TypeError, ValueError):
        return None
    return stack if stack.ndim == 3 else None


_CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR = CellProfilerFunctionContractExecutor()
