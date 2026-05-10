"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from enum import Enum
from functools import lru_cache
from inspect import Parameter, signature, unwrap
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any, ClassVar, get_args, get_origin, get_type_hints

from metaclass_registry import AutoRegisterMeta
from nominal_refactor_advisor.record_algebra import product_record
import numpy as np
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
    ImagePayloadSliceProjector,
    aligned_image_stack_kwargs,
    collapse_pairwise_slice_grid,
    compose_aligned_image_payload,
    is_pairwise_slice_grid,
    payload_slice_count,
    project_singleton_stack_image_domain,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection
from openhcs.core.callable_contract import (
    CallableContract,
)
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
    is_color_volume_slice,
    is_color_volume_stack,
    is_grayscale_volume_stack,
    is_image_stack,
)
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.measurement_image_alignment import (
    MeasurementImageLabelAlignmentRequest,
    MeasurementImageLabelAlignmentStrategy,
)
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.equivalence.keys import RuntimeMeasurementSourcePair
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    Pure2DSliceBatchExecutor,
    RuntimeBatchExecutionDomain,
    RuntimePure2DSliceBatchRequest,
    object_label_measurement_execution_from_callable,
)
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.core.runtime_slice_alignment import (
    RuntimeSliceAlignedValues,
    RuntimeSliceAlignedValueSet,
)
from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
from openhcs.core.runtime_output_matching import RuntimeReturnedOutputMatcher
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_artifact_queries import (
    MEASUREMENT_FEATURE_NAME_FIELD,
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_ID_FIELD,
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_RESULT_VALUE_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    MEASUREMENT_VALUE_FIELDS,
    MeasurementFeatureQuery,
    MeasurementRowOwnership,
    annotate_measurement_row_object,
    annotate_measurement_row_source_image,
    measurement_object_label,
    measurement_row_has_object_identity,
    measurement_row_object_name,
    measurement_row_mapping,
    measurement_row_source_image_name,
    measurement_rows,
    measurement_scalar_value_for_feature,
    measurement_table_for_slice,
    measurement_values_for_feature,
)
from openhcs.core.measurement_lookup_dialect import runtime_measurement_lookup_dialect
from openhcs.core.special_outputs import (
    SpecialOutputKindClassifier,
    special_output_name,
)
from openhcs.core.runtime_semantics import (
    MeasurementScope,
    FieldSpec,
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
    ObjectLocationMeasurementFeature,
    ObjectShapeMeasurementFeature,
    ObjectLabelDomain,
    ObjectLabelDomainMetadata,
    ObjectLabelDomainScope,
    ObjectLabelPlaneDomainStrategy,
    ObjectLabelRepresentation,
    ObjectLabelVariant,
    ParentChildRelationshipPayload,
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    aligned_dense_object_label_arrays,
    dense_object_label_id_domain,
    measurement_row_axis_field_names,
    parent_child_relationship_artifact_endpoints,
    parent_child_relationship_artifact_name,
    SourceSpatialDomainAdapter,
)
from openhcs.core.runtime_stores import require_runtime_value_store
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectLabelDenseDataStrategy,
    ObjectLabelPayload,
    ObjectLabelPure2DSliceAggregator,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelSet,
    ObjectRelationship,
    SparseIJVLabelRows,
    collapse_singleton_object_label_stack,
    object_label_dense_array,
    object_label_payload_with_measurement_labels,
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
    project_image_mask_to_data_domain,
    normalize_image_payload_intensity,
    with_derived_image_payload_data,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    GeneratedLeafClassSpec,
    NominalTypeKeyedStrategyMixin,
    RegisteredLeafClassSpec,
    RegisteredEnumMeta,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    Pure2DAuxiliaryOutputAggregator,
    Pure2DSliceIndexProjector,
    Pure2DSliceResultBatch,
    RuntimeCallablePolicy,
    RuntimeCallableView,
    RuntimeInvocationKwargPolicy,
    runtime_output_tuple,
)
from openhcs.processing.materialization import tabular_field_names_from_materialization
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.library import (
    canonical_module_name,
    coerce_registered_absorbed_processing_contract,
    require_function,
)
from openhcs.interop.cellprofiler.relationship_measurements import (
    RelationshipMeasurements,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    build_structuring_element,
)
from openhcs.interop.cellprofiler.image_module_settings import (
    WatershedInputKeyword,
    WatershedMethod,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.interop.cellprofiler.worm_measurements import (
    control_points_from_worm_measurement_rows,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
    CellProfilerMeasurementTargetScope,
    coerce_cellprofiler_measurement_target_scope,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CELLPROFILER_GRID_CYCLE_SCOPE_KWARG,
    CellProfilerGridCycleScope,
    CellProfilerImageExecutionContext,
    CellProfilerImageRequest,
    CellProfilerInvocationOptions,
    CellProfilerInvocationRequest,
    CellProfilerMeasurementImage,
    CellProfilerMeasurementImageDomain,
    CellProfilerResolvedInputRequest,
    CellProfilerSliceAlignedValues,
    CellProfilerSourceImagePair,
    CellProfilerSourcePairFeature,
    coerce_cellprofiler_grid_cycle_scope,
    illumination_scope_uses_all_images,
    requested_image_execution_mode,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    CellProfilerMeasurementFeature,
    CellProfilerMeasurementFeatureKind,
    count_feature_object_name,
)
from openhcs.interop.cellprofiler.runtime.adapter import (
    CellProfilerRuntimeAdapter,
    prepare_cellprofiler_runtime_adapter,
)

_MODULE_NAME_REGISTRY_KEY = "module_name"
_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
_CELLPROFILER_IMAGE_OVERRIDE_KWARG = "_cellprofiler_image_override"
_CELLPROFILER_EXECUTION_MODE_OVERRIDE_KWARG = "_cellprofiler_execution_mode_override"
logger = logging.getLogger(__name__)
_PROCESSING_CONTRACT_CACHE: dict[Callable[..., Any], ProcessingContract] = {}
_CELLPROFILER_RUNTIME_CALLABLE_POLICY = RuntimeCallablePolicy(
    callable_view=RuntimeCallableView.RAW,
    kwarg_policy=RuntimeInvocationKwargPolicy.SIGNATURE_FILTERED,
)
_INVOCATION_CONTROL_KWARGS = frozenset(
    (
        "dtype_config",
        "slice_by_slice",
        CELLPROFILER_GRID_CYCLE_SCOPE_KWARG,
        CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
    )
)


def _cellprofiler_module_policy_registry_key(name: str, cls: type) -> str | None:
    """Derive the canonical registry key from a policy leaf's module name."""
    del name
    module_name = cls.module_name  # type: ignore[attr-defined]
    if module_name is None:
        return None
    return canonical_module_name(str(module_name))


def _mro_declares_registry(cls: type) -> bool:
    """Return whether a class already belongs to an AutoRegisterMeta family."""
    return any("__registry__" in vars(mro_type) for mro_type in cls.__mro__)


@dataclass(frozen=True, slots=True)
class CellProfilerModulePolicyRegistryDefaults:
    """Registry defaults shared by CellProfiler module-name policy roots."""

    registry_key_attr: str = "registry_key"
    module_name_attr: str = _MODULE_NAME_REGISTRY_KEY

    def applies_to_root_bases(self, bases: tuple[type, ...]) -> bool:
        """Return whether a class declaration starts a new policy registry."""
        return not any(_mro_declares_registry(base) for base in bases)

    def apply_to(self, attrs: dict[str, Any]) -> None:
        """Install AutoRegisterMeta attributes for one policy root."""
        attrs.setdefault("__registry_key__", self.registry_key_attr)
        attrs.setdefault("__skip_if_no_key__", True)
        attrs.setdefault(
            "__key_extractor__",
            staticmethod(_cellprofiler_module_policy_registry_key),
        )
        attrs.setdefault(self.registry_key_attr, None)
        attrs.setdefault(self.module_name_attr, None)


CELLPROFILER_MODULE_POLICY_REGISTRY_DEFAULTS = (
    CellProfilerModulePolicyRegistryDefaults()
)


@dataclass(frozen=True, slots=True)
class CellProfilerModulePolicyRegistryConfigContext:
    """Metaclass registry-config context for CellProfiler module policies."""

    raw_registry_config: Any
    defaults: CellProfilerModulePolicyRegistryDefaults

    def apply_root_defaults(
        self, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> None:
        """Install implicit root defaults when this declaration starts a registry."""
        if self.defaults.applies_to_root_bases(bases):
            self.defaults.apply_to(attrs)


CELLPROFILER_MODULE_POLICY_IMPLICIT_REGISTRY_CONTEXT = (
    CellProfilerModulePolicyRegistryConfigContext(
        raw_registry_config=None,
        defaults=CELLPROFILER_MODULE_POLICY_REGISTRY_DEFAULTS,
    )
)


class CellProfilerModulePolicyMeta(AutoRegisterMeta):
    """AutoRegisterMeta variant for CellProfiler module-name policy families."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any],
        registry_config: CellProfilerModulePolicyRegistryConfigContext = (
            CELLPROFILER_MODULE_POLICY_IMPLICIT_REGISTRY_CONTEXT
        ),
    ):
        registry_config.apply_root_defaults(bases, attrs)
        return super().__new__(
            mcs,
            name,
            bases,
            attrs,
            registry_config.raw_registry_config,
        )


class CellProfilerSpecialInputPayloadSemantics(str, Enum):
    """Runtime value semantics for declared CellProfiler special inputs."""

    INTENSITY_IMAGE = "intensity_image"
    DENSE_LABEL_IMAGE = "dense_label_image"


def _runtime_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_module_profile(label: str, seconds: float, **fields: Any) -> None:
    if not _runtime_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


def _profile_payload_fields(prefix: str, value: Any) -> dict[str, Any]:
    """Return cheap payload shape/size fields for runtime profiling."""
    try:
        data = image_payload_data(value)
    except Exception:
        data = value
    return {
        f"{prefix}_type": type(data).__name__,
        f"{prefix}_shape": getattr(data, "shape", None),
        f"{prefix}_nbytes": getattr(data, "nbytes", None),
    }


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
        artifact_inputs=request.artifact_inputs,
        artifact_outputs=request.artifact_outputs,
        source_binding_plan=request.source_binding_plan,
        source_binding_context=request.source_binding_context,
        group_key=request.group_key,
        plane_projection=request.plane_projection,
        processing_context=request.context,
        filemanager=request.context.filemanager,
    )


@lru_cache(maxsize=None)
def _declared_input_specs_for_contract(
    contract: ModuleArtifactContract,
) -> tuple[ArtifactSpec, ...]:
    declared = tuple(contract.inputs)
    runtime_extras = tuple(
        spec for spec in contract.runtime_artifact_inputs if spec not in declared
    )
    return (*declared, *runtime_extras)


@dataclass(frozen=True, slots=True)
class CellProfilerModuleExecutor:
    """Execute one generated CellProfiler module against a typed runtime adapter."""

    contract: ModuleArtifactContract
    _canonical_module_name: str = field(init=False, repr=False, compare=False)
    _declared_inputs: tuple[ArtifactSpec, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _object_inputs: tuple[ArtifactSpec, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _spatial_grid_inputs: tuple[ArtifactSpec, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _runtime_image_names_cache: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _runtime_image_name_set: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _external_source_image_names_cache: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _external_source_object_names_cache: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _external_source_image_name_set: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _external_source_object_name_set: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _ordered_outputs: tuple[ArtifactSpec, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _output_recorders: Mapping[ArtifactKind, "CellProfilerOutputRecorder"] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ModuleArtifactContract):
            raise TypeError(
                "CellProfilerModuleExecutor.contract must be "
                "ModuleArtifactContract, got "
                f"{type(self.contract).__name__}."
            )
        declared_inputs = _declared_input_specs_for_contract(self.contract)
        runtime_image_names = tuple(
            spec.name
            for spec in ArtifactSpecCollection(
                self.contract.runtime_artifact_inputs
            ).of_kind(ArtifactKind.IMAGE)
        )
        runtime_image_name_set = frozenset(runtime_image_names)
        runtime_object_names = frozenset(
            spec.name
            for spec in ArtifactSpecCollection(
                self.contract.runtime_artifact_inputs
            ).of_kind(ArtifactKind.OBJECT_LABELS)
        )
        ordered_outputs = _output_recording_order(self.contract.outputs)

        object.__setattr__(
            self,
            "_canonical_module_name",
            canonical_module_name(self.contract.module_name),
        )
        object.__setattr__(self, "_declared_inputs", declared_inputs)
        object.__setattr__(
            self,
            "_object_inputs",
            ArtifactSpecCollection(declared_inputs).of_kind(ArtifactKind.OBJECT_LABELS),
        )
        object.__setattr__(
            self,
            "_spatial_grid_inputs",
            ArtifactSpecCollection(declared_inputs).of_kind(ArtifactKind.SPATIAL_GRID),
        )
        object.__setattr__(self, "_runtime_image_names_cache", runtime_image_names)
        object.__setattr__(self, "_runtime_image_name_set", runtime_image_name_set)
        object.__setattr__(
            self,
            "_external_source_image_names_cache",
            external_image_names := tuple(
                spec.name
                for spec in ArtifactSpecCollection(declared_inputs).of_kind(
                    ArtifactKind.IMAGE
                )
                if spec.name not in runtime_image_name_set
            ),
        )
        object.__setattr__(
            self,
            "_external_source_object_names_cache",
            external_object_names := tuple(
                spec.name
                for spec in ArtifactSpecCollection(self.contract.inputs).of_kind(
                    ArtifactKind.OBJECT_LABELS
                )
                if spec.name not in runtime_object_names
            ),
        )
        object.__setattr__(
            self,
            "_external_source_image_name_set",
            frozenset(external_image_names),
        )
        object.__setattr__(
            self,
            "_external_source_object_name_set",
            frozenset(external_object_names),
        )
        object.__setattr__(self, "_ordered_outputs", ordered_outputs)
        object.__setattr__(
            self,
            "_output_recorders",
            MappingProxyType(
                {
                    kind: CellProfilerOutputRecorder.for_kind(kind)
                    for kind in {spec.kind for spec in ordered_outputs}
                }
            ),
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

    @property
    def image_outputs(self) -> tuple[ArtifactSpec, ...]:
        return ArtifactSpecCollection(self.outputs).of_kind(ArtifactKind.IMAGE)

    def prepare(self, func: Callable[..., Any]) -> None:
        """Resolve nominal policies used by this executor before timed execution."""
        prepare_cellprofiler_runtime_adapter()
        for mode in ImagePayloadExecutionMode:
            CellProfilerImageExecutionStrategy.for_mode(mode)
        for kind in tuple(RuntimeArtifactKindStrategy.__registry__.keys()):
            RuntimeArtifactKindStrategy.for_kind(kind)
        self._declared_input_specs()
        self._primary_image_inputs(func)
        self._object_input_specs()
        CellProfilerObjectInputPolicy.for_module(self.module_name)
        CellProfilerSpecialInputPolicy.for_module(self.module_name)
        CellProfilerInvocationExecutionModePolicy.for_module(self.module_name)
        CellProfilerMainFlowReplacementPolicy.for_module(self.module_name)
        CellProfilerObjectMeasurementRowPolicy.for_module(self.module_name)
        CellProfilerDualScopeMeasurementPolicy.for_module(self.module_name)
        CellProfilerMeasurementRecordBuilder.for_module(self.module_name)
        self._runs_per_image_measurement(func)
        self._runs_per_object_measurement()
        for output in self.outputs:
            self._output_recorders[output.kind]

    def run(
        self,
        func: Callable[..., Any],
        image: Any,
        *,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        invocation_options: CellProfilerInvocationOptions | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call the absorbed function and record declared outputs through the adapter."""
        function_name = CallableContract.from_callable(func).function_name
        run_started_at = time.perf_counter()
        mode_started_at = time.perf_counter()
        if self._runs_per_image_measurement(func):
            _log_module_profile(
                "cp_runs_per_image_check",
                time.perf_counter() - mode_started_at,
                module=self.module_name,
                function=function_name,
            )
            per_image_started_at = time.perf_counter()
            result = self._run_per_image_measurement(
                func,
                input_image=image,
                current_image=image,
                cellprofiler_runtime=cellprofiler_runtime,
                **kwargs,
            )
            _log_module_profile(
                "cp_run_per_image_measurement",
                time.perf_counter() - per_image_started_at,
                module=self.module_name,
                function=function_name,
            )
            _log_module_profile(
                "cp_module_run_total",
                time.perf_counter() - run_started_at,
                module=self.module_name,
                function=function_name,
            )
            return result

        _log_module_profile(
            "cp_runs_per_image_check",
            time.perf_counter() - mode_started_at,
            module=self.module_name,
            function=function_name,
        )
        image_request_started_at = time.perf_counter()
        image_request = self._image_request(
            func,
            image,
            cellprofiler_runtime,
        )
        _log_module_profile(
            "cp_image_request",
            time.perf_counter() - image_request_started_at,
            module=self.module_name,
            function=function_name,
        )
        object_mode_started_at = time.perf_counter()
        if self._runs_per_object_measurement():
            _log_module_profile(
                "cp_runs_per_object_check",
                time.perf_counter() - object_mode_started_at,
                module=self.module_name,
                function=function_name,
            )
            per_object_started_at = time.perf_counter()
            result = self._run_per_object_measurement(
                func,
                input_image=image,
                current_image=image,
                image_request=image_request,
                cellprofiler_runtime=cellprofiler_runtime,
                source_image_name=image_request.source_image_name,
                **kwargs,
            )
            _log_module_profile(
                "cp_run_per_object_measurement",
                time.perf_counter() - per_object_started_at,
                module=self.module_name,
                function=function_name,
            )
            _log_module_profile(
                "cp_module_run_total",
                time.perf_counter() - run_started_at,
                module=self.module_name,
                function=function_name,
            )
            return result

        _log_module_profile(
            "cp_runs_per_object_check",
            time.perf_counter() - object_mode_started_at,
            module=self.module_name,
            function=function_name,
        )
        invocation_started_at = time.perf_counter()
        invocation = self._invocation_request(
            func,
            image_request=image_request,
            adapter=cellprofiler_runtime,
            current_image=image,
            kwargs=kwargs,
            invocation_options=invocation_options,
        )
        _log_module_profile(
            "cp_invocation_request",
            time.perf_counter() - invocation_started_at,
            module=self.module_name,
            function=function_name,
        )
        execute_started_at = time.perf_counter()
        raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
            func,
            invocation.image,
            invocation.kwargs,
            execution_mode=invocation.execution_mode,
        )
        _log_module_profile(
            "cp_contract_execute",
            time.perf_counter() - execute_started_at,
            module=self.module_name,
            function=function_name,
            **_profile_payload_fields("input", invocation.image),
            **_profile_payload_fields("output", raw_output),
        )
        split_started_at = time.perf_counter()
        main_output, artifact_values = _split_cellprofiler_output(raw_output)
        _log_module_profile(
            "cp_split_output",
            time.perf_counter() - split_started_at,
            module=self.module_name,
            function=function_name,
        )
        record_started_at = time.perf_counter()
        self._record_outputs(
            func,
            cellprofiler_runtime,
            main_output,
            artifact_values,
            source_image_payload=invocation.image,
            source_image_name=invocation.source_image_name,
        )
        _log_module_profile(
            "cp_record_outputs",
            time.perf_counter() - record_started_at,
            module=self.module_name,
            function=function_name,
        )
        replace_started_at = time.perf_counter()
        if not self._replaces_main_flow(
            input_image=image,
            output_image=main_output,
        ):
            _log_module_profile(
                "cp_replace_main_flow_check",
                time.perf_counter() - replace_started_at,
                module=self.module_name,
                function=function_name,
            )
            _log_module_profile(
                "cp_module_run_total",
                time.perf_counter() - run_started_at,
                module=self.module_name,
                function=function_name,
            )
            return image
        result = _openhcs_main_flow_output(image, main_output)
        _log_module_profile(
            "cp_replace_main_flow_check",
            time.perf_counter() - replace_started_at,
            module=self.module_name,
            function=function_name,
        )
        _log_module_profile(
            "cp_module_run_total",
            time.perf_counter() - run_started_at,
            module=self.module_name,
            function=function_name,
        )
        return result

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

    def _replaces_main_flow(
        self,
        *,
        input_image: Any,
        output_image: Any,
    ) -> bool:
        return CellProfilerMainFlowReplacementPolicy.for_module(
            self.module_name
        ).replaces_main_flow(
            CellProfilerMainFlowReplacementRequest(
                executor=self,
                input_image=input_image,
                output_image=output_image,
            )
        )

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
        function_name = CallableContract.from_callable(func).function_name
        object_inputs = self._object_input_specs()
        measurement_outputs = ArtifactSpecCollection(self.outputs).of_kind(
            ArtifactKind.MEASUREMENTS
        )
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
        measurement_images_started_at = time.perf_counter()
        measurement_images = self._measurement_image_inputs(
            func,
            cellprofiler_runtime,
            current_image,
            image_request,
        )
        _log_module_profile(
            "cp_per_object_measurement_images",
            time.perf_counter() - measurement_images_started_at,
            module=self.module_name,
            function=function_name,
            images=len(measurement_images),
            objects=len(object_inputs),
        )
        dual_scope_started_at = time.perf_counter()
        image_measurement_rows = self._dual_scope_image_measurement_rows(
            func,
            measurement_images,
            kwargs,
            measurement_target_scope,
        )
        _log_module_profile(
            "cp_per_object_dual_scope_rows",
            time.perf_counter() - dual_scope_started_at,
            module=self.module_name,
            function=function_name,
            rows=len(image_measurement_rows),
        )
        combined_rows.extend(image_measurement_rows)
        row_source_names_required = _row_source_names_required(measurement_images)
        row_object_names_required = (
            bool(image_measurement_rows)
            or len(object_inputs) != 1
            or row_source_names_required
        )
        measurement_row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
            self.module_name
        )
        requires_explicit_row_ownership = (
            measurement_row_policy.requires_explicit_row_ownership()
        )
        label_payload_seconds = 0.0
        label_align_seconds = 0.0
        contract_execute_seconds = 0.0
        split_seconds = 0.0
        complete_rows_seconds = 0.0
        annotate_seconds = 0.0
        batch_executor = CallableContract.from_callable(func).runtime_batch_executor(
            RuntimeBatchExecutionDomain.MEASUREMENT_IMAGES
        )

        def record_measurement_output(
            raw_output: Any,
            *,
            measurement_image: CellProfilerMeasurementImage,
            object_spec: ArtifactSpec,
            completion_label_payload: Any,
            invocation: ObjectMeasurementInvocation,
        ) -> None:
            nonlocal split_seconds, complete_rows_seconds, annotate_seconds

            split_started_at = time.perf_counter()
            _ignored_main_output, artifact_values = _split_cellprofiler_output(
                raw_output
            )
            split_seconds += time.perf_counter() - split_started_at
            raw_measurement_rows = measurement_row_policy.project_rows(
                _measurement_rows_from_output(artifact_values),
                invocation,
            )
            object_measurement_rows = [
                row
                for row in raw_measurement_rows
                if measurement_row_policy.row_is_object_scoped(row)
            ]
            non_object_measurement_rows = [
                row
                for row in raw_measurement_rows
                if not measurement_row_policy.row_is_object_scoped(row)
            ]
            combined_rows.extend(non_object_measurement_rows)
            complete_rows_started_at = time.perf_counter()
            measurement_rows = measurement_row_policy.complete_rows(
                object_measurement_rows,
                label_payload=completion_label_payload,
                func=func,
            )
            complete_rows_seconds += time.perf_counter() - complete_rows_started_at
            annotate_started_at = time.perf_counter()
            source_image_name = (
                measurement_image.source_image_name
                if row_source_names_required
                else None
            )
            combined_rows.extend(
                MeasurementRowOwnership(
                    object_name=(
                        object_spec.name
                        if row_object_names_required or requires_explicit_row_ownership
                        else None
                    ),
                    source_image_name=source_image_name,
                ).annotate_rows(measurement_rows)
            )
            annotate_seconds += time.perf_counter() - annotate_started_at

        def prepare_measurement_invocation(
            measurement_image: CellProfilerMeasurementImage,
            object_spec: ArtifactSpec,
        ) -> tuple[Any, Any, Any, ImagePayloadExecutionMode]:
            nonlocal label_payload_seconds, label_align_seconds

            label_payload_started_at = time.perf_counter()
            raw_label_payload = self._object_label_payload(
                object_spec,
                cellprofiler_runtime,
                input_image,
            )
            raw_labels = _measurement_labels_for_measurement_image(
                measurement_image,
                raw_label_payload,
                adapter=cellprofiler_runtime,
            )
            label_payload_seconds += time.perf_counter() - label_payload_started_at
            label_align_started_at = time.perf_counter()
            measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(
                measurement_image.payload,
                raw_labels,
                label_payload=raw_label_payload,
            )
            aligned_image = (
                _measurement_image_for_labels(
                    measurement_image.payload,
                    measurement_labels,
                    label_payload=raw_label_payload,
                    reference_domain=measurement_image.reference_domain,
                )
                if measurement_image.align_to_labels
                else measurement_image.payload
            )
            measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(
                aligned_image,
                measurement_labels,
                label_payload=raw_label_payload,
            )
            completion_label_payload = object_label_payload_with_measurement_labels(
                raw_label_payload,
                measurement_labels,
            )
            executable_labels = (
                CellProfilerObjectMeasurementLabelArgumentPolicy.for_enum_member(
                    object_label_measurement_execution_from_callable(func)
                ).label_argument(
                    CellProfilerObjectMeasurementLabelArgumentRequest(
                        dense_labels=measurement_labels,
                        label_payload=completion_label_payload,
                        measurement_image_payload=measurement_image.payload,
                    )
                )
            )
            label_align_seconds += time.perf_counter() - label_align_started_at
            execution_mode = (
                CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
                    self.module_name
                ).execution_mode(
                    func,
                    completion_label_payload,
                    measurement_image.execution_mode,
                    runtime_slice_count=payload_slice_count(current_image),
                )
            )
            return (
                aligned_image,
                executable_labels,
                completion_label_payload,
                execution_mode,
            )

        measurement_invocations = tuple(
            measurement_row_policy.invocations(
                measurement_image,
                kwargs,
            )
            for measurement_image in measurement_images
        )
        measurement_batch_count = sum(
            len(invocations) for invocations in measurement_invocations
        )
        total_measurement_batch_count = measurement_batch_count * len(object_inputs)
        use_measurement_image_batch = callable(batch_executor) and any(
            len(invocations) > 1 for invocations in measurement_invocations
        )
        if not use_measurement_image_batch:
            for measurement_image, invocations in zip(
                measurement_images,
                measurement_invocations,
                strict=True,
            ):
                for object_spec in object_inputs:
                    (
                        aligned_image,
                        executable_labels,
                        completion_label_payload,
                        execution_mode,
                    ) = prepare_measurement_invocation(measurement_image, object_spec)
                    for invocation in invocations:
                        contract_started_at = time.perf_counter()
                        raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                            func,
                            aligned_image,
                            {
                                **invocation.lowered_kwargs(),
                                "labels": executable_labels,
                            },
                            execution_mode=execution_mode,
                        )
                        contract_execute_seconds += (
                            time.perf_counter() - contract_started_at
                        )
                        record_measurement_output(
                            raw_output,
                            measurement_image=measurement_image,
                            object_spec=object_spec,
                            completion_label_payload=completion_label_payload,
                            invocation=invocation,
                        )
        else:
            ordered_batch_outputs: dict[
                int,
                tuple[
                    Any,
                    CellProfilerMeasurementImage,
                    ArtifactSpec,
                    Any,
                    ObjectMeasurementInvocation,
                ],
            ] = {}
            order_index = 0
            batch_requests: list[RuntimeBatchInvocationRequest] = []
            batch_contexts: list[
                tuple[
                    int,
                    CellProfilerMeasurementImage,
                    ArtifactSpec,
                    Any,
                    ObjectMeasurementInvocation,
                ]
            ] = []
            for object_spec in object_inputs:
                for measurement_image, invocations in zip(
                    measurement_images,
                    measurement_invocations,
                    strict=True,
                ):
                    (
                        aligned_image,
                        executable_labels,
                        completion_label_payload,
                        execution_mode,
                    ) = prepare_measurement_invocation(measurement_image, object_spec)
                    for invocation in invocations:
                        batch_requests.append(
                            RuntimeBatchInvocationRequest(
                                source_image_name=measurement_image.source_image_name,
                                execution_mode=execution_mode,
                                image=aligned_image,
                                kwargs={
                                    **invocation.lowered_kwargs(),
                                    "labels": executable_labels,
                                },
                                batch_index=len(batch_requests),
                                batch_count=total_measurement_batch_count,
                            )
                        )
                        batch_contexts.append(
                            (
                                order_index,
                                measurement_image,
                                object_spec,
                                completion_label_payload,
                                invocation,
                            )
                        )
                        order_index += 1
            contract_started_at = time.perf_counter()
            raw_outputs = batch_executor(
                func,
                tuple(batch_requests),
                _execute_runtime_batch_invocation,
            )
            contract_execute_seconds += time.perf_counter() - contract_started_at
            if len(raw_outputs) != len(batch_contexts):
                raise ValueError(
                    f"{function_name} measurement-image batch executor returned "
                    f"{len(raw_outputs)} outputs for {len(batch_contexts)} requests."
                )
            for raw_output, (
                order_index,
                measurement_image,
                object_spec,
                completion_label_payload,
                invocation,
            ) in zip(raw_outputs, batch_contexts, strict=True):
                ordered_batch_outputs[order_index] = (
                    raw_output,
                    measurement_image,
                    object_spec,
                    completion_label_payload,
                    invocation,
                )
            for order_index in range(len(ordered_batch_outputs)):
                (
                    raw_output,
                    measurement_image,
                    object_spec,
                    completion_label_payload,
                    invocation,
                ) = ordered_batch_outputs[order_index]
                record_measurement_output(
                    raw_output,
                    measurement_image=measurement_image,
                    object_spec=object_spec,
                    completion_label_payload=completion_label_payload,
                    invocation=invocation,
                )

        _log_module_profile(
            "cp_per_object_label_payload",
            label_payload_seconds,
            module=self.module_name,
            function=function_name,
        )
        _log_module_profile(
            "cp_per_object_label_align",
            label_align_seconds,
            module=self.module_name,
            function=function_name,
        )
        _log_module_profile(
            "cp_per_object_contract_execute",
            contract_execute_seconds,
            module=self.module_name,
            function=function_name,
        )
        _log_module_profile(
            "cp_per_object_split_output",
            split_seconds,
            module=self.module_name,
            function=function_name,
        )
        _log_module_profile(
            "cp_per_object_complete_rows",
            complete_rows_seconds,
            module=self.module_name,
            function=function_name,
        )
        _log_module_profile(
            "cp_per_object_annotate_rows",
            annotate_seconds,
            module=self.module_name,
            function=function_name,
            rows=len(combined_rows),
        )

        combined_source_image_name = measurement_row_policy.table_source_image_name(
            measurement_images,
            source_image_name,
        )

        record_started_at = time.perf_counter()
        _record_measurements(
            cellprofiler_runtime,
            measurement_outputs[0].name,
            combined_rows,
            fields=_measurement_record_fields(
                measurement_outputs[0], combined_rows, func
            ),
            object_name=(
                None
                if image_measurement_rows or requires_explicit_row_ownership
                else object_inputs[0].name if len(object_inputs) == 1 else None
            ),
            source_image_name=combined_source_image_name,
        )
        _log_module_profile(
            "cp_per_object_record_measurements",
            time.perf_counter() - record_started_at,
            module=self.module_name,
            function=function_name,
            rows=len(combined_rows),
        )
        return input_image

    def _dual_scope_image_measurement_rows(
        self,
        object_func: Callable[..., Any],
        measurement_images: tuple["CellProfilerMeasurementImage", ...],
        kwargs: Mapping[str, Any],
        target_scope: CellProfilerMeasurementTargetScope,
    ) -> list[Any]:
        function_name = CallableContract.from_callable(object_func).function_name
        if target_scope is not CellProfilerMeasurementTargetScope.BOTH:
            return []
        policy = CellProfilerDualScopeMeasurementPolicy.for_module(self.module_name)
        if policy is None:
            return []
        image_func = policy.image_function(object_func)
        rows: list[Any] = []
        row_source_names_required = _row_source_names_required(measurement_images)
        image_kwargs = _coerce_invocation_kwargs(image_func, kwargs)
        contract_execute_seconds = 0.0
        split_rows_seconds = 0.0
        for measurement_image in measurement_images:
            contract_started_at = time.perf_counter()
            raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                image_func,
                _image_scope_measurement_payload(measurement_image.payload),
                image_kwargs,
                execution_mode=measurement_image.execution_mode,
            )
            contract_execute_seconds += time.perf_counter() - contract_started_at
            split_rows_started_at = time.perf_counter()
            _ignored_main_output, artifact_values = _split_cellprofiler_output(
                raw_output
            )
            source_image_name = (
                measurement_image.source_image_name
                if row_source_names_required
                else None
            )
            rows.extend(
                MeasurementRowOwnership(
                    source_image_name=source_image_name,
                ).annotate_rows(_measurement_rows_from_output(artifact_values))
            )
            split_rows_seconds += time.perf_counter() - split_rows_started_at
        _log_module_profile(
            "cp_dual_scope_contract_execute",
            contract_execute_seconds,
            module=self.module_name,
            function=function_name,
        )
        _log_module_profile(
            "cp_dual_scope_split_rows",
            split_rows_seconds,
            module=self.module_name,
            function=function_name,
            rows=len(rows),
        )
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
        function_name = CallableContract.from_callable(func).function_name
        measurement_outputs = ArtifactSpecCollection(self.outputs).of_kind(
            ArtifactKind.MEASUREMENTS
        )
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
        measurement_images_started_at = time.perf_counter()
        measurement_images = self._independent_measurement_image_inputs(
            func,
            cellprofiler_runtime,
            current_image,
        )
        _log_module_profile(
            "cp_per_image_measurement_images",
            time.perf_counter() - measurement_images_started_at,
            module=self.module_name,
            function=function_name,
            images=len(measurement_images),
        )
        kwargs_started_at = time.perf_counter()
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
        _log_module_profile(
            "cp_per_image_prepare_kwargs",
            time.perf_counter() - kwargs_started_at,
            module=self.module_name,
            function=function_name,
        )
        row_source_names_required = _row_source_names_required(measurement_images)
        contract_execute_seconds = 0.0
        split_rows_seconds = 0.0
        combined_records: list[CellProfilerMeasurementRecord] = []
        for measurement_image in measurement_images:
            contract_started_at = time.perf_counter()
            raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                func,
                _image_scope_measurement_payload(measurement_image.payload),
                coerced_kwargs,
                execution_mode=measurement_image.execution_mode,
            )
            contract_execute_seconds += time.perf_counter() - contract_started_at
            split_rows_started_at = time.perf_counter()
            main_output, artifact_values = _split_cellprofiler_output(raw_output)
            source_image_name = (
                measurement_image.source_image_name
                if row_source_names_required
                else None
            )
            output_values = CellProfilerOutputValueResolution.from_returned_values(
                measurement_outputs,
                declared_specs=self.contract.declared_outputs,
                main_output=main_output,
                artifact_values=artifact_values,
                func=func,
            )
            measurement_record = CellProfilerMeasurementRecordBuilder.for_module(
                self.module_name
            ).build(
                CellProfilerOutputRecordRequest(
                    executor=self,
                    adapter=cellprofiler_runtime,
                    spec=measurement_outputs[0],
                    value=output_values.recorded_values[measurement_outputs[0].name],
                    output_values=output_values.context_values,
                    source_image_payload=measurement_image.payload,
                    source_image_name=source_image_name,
                    func=func,
                    source_image_names=measurement_image.source_image_names,
                )
            )
            combined_records.append(measurement_record)
            combined_rows.extend(
                MeasurementRowOwnership(
                    source_image_name=measurement_record.source_image_name,
                ).annotate_rows(measurement_record.rows)
            )
            split_rows_seconds += time.perf_counter() - split_rows_started_at

        _log_module_profile(
            "cp_per_image_contract_execute",
            contract_execute_seconds,
            module=self.module_name,
            function=function_name,
        )
        _log_module_profile(
            "cp_per_image_split_rows",
            split_rows_seconds,
            module=self.module_name,
            function=function_name,
            rows=len(combined_rows),
        )

        rows_declare_object_name = _measurement_rows_declare_object_name(combined_rows)
        record_started_at = time.perf_counter()
        _record_measurements(
            cellprofiler_runtime,
            measurement_outputs[0].name,
            combined_rows,
            fields=_measurement_record_fields(
                measurement_outputs[0], combined_rows, func
            ),
            object_name=(
                None if rows_declare_object_name else _MISSING_MEASUREMENT_OBJECT_NAME
            ),
            source_image_name=(
                None
                if rows_declare_object_name
                else (
                    CellProfilerMeasurementRecord.shared_source_image_name(
                        tuple(combined_records)
                    )
                    or CellProfilerMeasurementImage.shared_source_image_name(
                        measurement_images
                    )
                )
            ),
        )
        _log_module_profile(
            "cp_per_image_record_measurements",
            time.perf_counter() - record_started_at,
            module=self.module_name,
            function=function_name,
            rows=len(combined_rows),
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
            return (self._composed_measurement_image(image_request, image_inputs),)

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
            source_image_names=(),
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
            source_image_names=tuple(spec.name for spec in image_inputs),
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
        runtime_image_names = self._runtime_image_name_set
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
                source_image_names=(spec.name,),
                payload=_cellprofiler_image_payload(runtime_image.data),
                reference_domain=reference_domain,
            )
        return CellProfilerMeasurementImage(
            source_image_name=spec.name,
            source_image_names=(spec.name,),
            payload=_cellprofiler_image_payload(
                adapter.resolve_source_image(spec.name, current_image)
            ),
            reference_domain=reference_domain,
        )

    def _object_input_specs(self) -> tuple[ArtifactSpec, ...]:
        return self._object_inputs

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
            )
        return adapter.get_objects(spec.name)

    def _runtime_input_kwargs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        current_image: Any,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        runtime_inputs = self._special_runtime_inputs(func)
        object_input_policy = CellProfilerObjectInputPolicy.for_module(
            self._canonical_module_name
        )
        if not runtime_inputs:
            if object_input_policy.binds_without_declared_inputs:
                return object_input_policy.bind(
                    ObjectInputBindingRequest(
                        module_name=self.module_name,
                        object_inputs=(),
                        adapter=adapter,
                        kwargs=kwargs,
                        current_image=current_image,
                        external_object_names=self._external_source_object_name_set,
                        runtime_inputs=runtime_inputs,
                    )
                )
            return {}

        special_input_names = special_input_names_from_callable(func)
        if special_input_names:
            special_input_specs = runtime_inputs
            return CellProfilerSpecialInputPolicy.for_module(
                self._canonical_module_name
            ).bind(
                SpecialInputBindingRequest(
                    module_name=self.module_name,
                    parameter_names=special_input_names,
                    special_input_specs=special_input_specs,
                    runtime_inputs=runtime_inputs,
                    adapter=adapter,
                    kwargs=kwargs,
                    current_image=current_image,
                    external_image_names=self._external_source_image_name_set,
                    external_object_names=self._external_source_object_name_set,
                    runtime_image_names=self._runtime_image_name_set,
                )
            )

        supported_non_object_kinds = (
            object_input_policy.supported_non_object_input_kinds
        )
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

        object_inputs = ArtifactSpecCollection(runtime_inputs).of_kind(
            ArtifactKind.OBJECT_LABELS
        )
        return object_input_policy.bind(
            ObjectInputBindingRequest(
                module_name=self.module_name,
                object_inputs=object_inputs,
                adapter=adapter,
                kwargs=kwargs,
                current_image=current_image,
                external_object_names=self._external_source_object_name_set,
                runtime_inputs=runtime_inputs,
            )
        )

    def _special_runtime_inputs(
        self,
        func: Callable[..., Any],
    ) -> tuple[ArtifactSpec, ...]:
        declared_inputs = self._declared_input_specs()
        non_image_inputs = tuple(
            spec for spec in declared_inputs if spec.kind is not ArtifactKind.IMAGE
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
        source_image_payload: Any | None = None,
        source_image_name: str | None,
    ) -> None:
        if not self.outputs:
            return

        function_name = CallableContract.from_callable(func).function_name
        values_started_at = time.perf_counter()
        output_values = CellProfilerOutputValueResolution.from_returned_values(
            self.outputs,
            declared_specs=self.contract.declared_outputs,
            main_output=main_output,
            artifact_values=artifact_values,
            func=func,
        )
        _log_module_profile(
            "cp_output_values_by_kind",
            time.perf_counter() - values_started_at,
            module=self.module_name,
            function=function_name,
            outputs=len(self.outputs),
        )
        order_started_at = time.perf_counter()
        ordered_outputs = self._ordered_outputs
        _log_module_profile(
            "cp_output_recording_order",
            time.perf_counter() - order_started_at,
            module=self.module_name,
            function=function_name,
            outputs=len(self.outputs),
        )
        for spec in ordered_outputs:
            record_started_at = time.perf_counter()
            self._output_recorders[spec.kind].record(
                CellProfilerOutputRecordRequest(
                    executor=self,
                    adapter=adapter,
                    spec=spec,
                    value=output_values.recorded_values[spec.name],
                    output_values=output_values.context_values,
                    source_image_payload=source_image_payload,
                    source_image_name=source_image_name,
                    func=func,
                )
            )
            _log_module_profile(
                "cp_output_record_one",
                time.perf_counter() - record_started_at,
                module=self.module_name,
                function=function_name,
                artifact=spec.name,
                kind=spec.kind.value,
                **_profile_payload_fields(
                    "value",
                    output_values.recorded_values[spec.name],
                ),
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
                if self._object_input_specs() or self._spatial_grid_inputs
                else _cellprofiler_image_payload(current_image)
            )
            return CellProfilerImageRequest(
                payload=payload,
                source_image_name=self._input_source_image_name(adapter),
                image_count=1,
                execution_mode=ImagePayloadExecutionMode.NATURAL,
            )

        runtime_image_names = self._runtime_image_name_set
        external_image_names = tuple(
            spec.name for spec in image_inputs if spec.name not in runtime_image_names
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
            source_image_name=self._primary_image_source_name(
                adapter,
                image_inputs,
            ),
            image_count=len(payloads),
            execution_mode=composition.execution_mode,
        )

    def _primary_image_inputs(
        self,
        func: Callable[..., Any],
    ) -> tuple[ArtifactSpec, ...]:
        return CellProfilerPrimaryImageInputPolicy.for_module(
            self._canonical_module_name
        ).primary_image_inputs(
            self.module_name,
            func,
            self._declared_input_specs(),
        )

    def _input_source_image_name(
        self,
        adapter: CellProfilerRuntimeAdapter,
    ) -> str | None:
        source_names: list[str] = []
        runtime_image_names = self._runtime_image_name_set
        external_image_names = self._external_source_image_name_set
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

    def _primary_image_source_name(
        self,
        adapter: CellProfilerRuntimeAdapter,
        image_inputs: tuple[ArtifactSpec, ...],
    ) -> str | None:
        runtime_image_names = self._runtime_image_name_set
        source_names = tuple(
            (
                adapter.get_image(spec.name).source_image_name
                if spec.name in runtime_image_names
                else spec.name
            )
            for spec in image_inputs
        )
        if len(source_names) > 1:
            return _measurement_source_name_for_specs(image_inputs)
        return _single_source_name(source_names)

    def _invocation_request(
        self,
        func: Callable[..., Any],
        *,
        image_request: "CellProfilerImageRequest",
        adapter: CellProfilerRuntimeAdapter,
        current_image: Any,
        kwargs: Mapping[str, Any],
        invocation_options: CellProfilerInvocationOptions | None,
    ) -> "CellProfilerInvocationRequest":
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(func, adapter, current_image, kwargs),
        }
        image_override = runtime_kwargs.pop(_CELLPROFILER_IMAGE_OVERRIDE_KWARG, None)
        execution_mode_override = runtime_kwargs.pop(
            _CELLPROFILER_EXECUTION_MODE_OVERRIDE_KWARG,
            None,
        )
        if self._canonical_module_name == "TrackObjects":
            source_image_name = (
                image_request.source_image_name
                or self._object_input_source_image_name(adapter)
            )
            runtime_kwargs.setdefault(
                "image_number_start",
                _cellprofiler_image_number_start(
                    current_image,
                    adapter,
                    source_image_name=source_image_name,
                ),
            )
        runtime_kwargs.pop(CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG, None)
        if _should_slice_flexible_object_invocation(
            self._object_input_specs(),
            func,
            runtime_kwargs,
        ):
            runtime_kwargs.setdefault("slice_by_slice", True)
        execution_mode = CellProfilerInvocationExecutionModePolicy.for_module(
            self._canonical_module_name
        ).execution_mode(
            default=(
                CallableContract.from_callable(func).runtime_image_execution_mode
                or image_request.execution_mode
            ),
            image=(
                image_override if image_override is not None else image_request.payload
            ),
            kwargs=runtime_kwargs,
            invocation_options=invocation_options,
        )
        runtime_kwargs.pop(CELLPROFILER_GRID_CYCLE_SCOPE_KWARG, None)
        return CellProfilerInvocationRequest(
            image=(
                image_override if image_override is not None else image_request.payload
            ),
            kwargs=_coerce_invocation_kwargs(func, runtime_kwargs),
            source_image_name=image_request.source_image_name,
            image_count=image_request.image_count,
            execution_mode=execution_mode_override or execution_mode,
        )

    def _external_source_image_names(self) -> tuple[str, ...]:
        return self._external_source_image_names_cache

    def _object_input_source_image_name(
        self,
        adapter: CellProfilerRuntimeAdapter,
    ) -> str | None:
        source_names = tuple(
            adapter.get_objects(spec.name).source_image_name
            for spec in self._object_input_specs()
        )
        return _single_source_name(
            tuple(source_name for source_name in source_names if source_name)
        )

    def _external_source_object_names(self) -> tuple[str, ...]:
        return self._external_source_object_names_cache

    def _runtime_image_names(self) -> tuple[str, ...]:
        return self._runtime_image_names_cache

    def _declared_input_specs(self) -> tuple[ArtifactSpec, ...]:
        return self._declared_inputs


@dataclass(frozen=True, slots=True)
class CellProfilerMainFlowReplacementRequest:
    """Typed context for deciding whether a module output advances OpenHCS main flow."""

    executor: CellProfilerModuleExecutor
    input_image: Any
    output_image: Any


class CellProfilerMainFlowReplacementPolicy(
    ABC,
    metaclass=CellProfilerModulePolicyMeta,
):
    """Nominal policy for mapping declared CellProfiler image outputs to main flow."""

    @classmethod
    @lru_cache(maxsize=None)
    def for_module(cls, module_name: str) -> "CellProfilerMainFlowReplacementPolicy":
        policy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            ContractImageOutputMainFlowReplacementPolicy,
        )
        return policy_type()

    @abstractmethod
    def replaces_main_flow(
        self,
        request: CellProfilerMainFlowReplacementRequest,
    ) -> bool:
        """Return True when the declared module image output owns downstream flow."""


class ContractImageOutputMainFlowReplacementPolicy(
    CellProfilerMainFlowReplacementPolicy
):
    """Use the artifact contract, not runtime slice cardinality, as the authority."""

    def replaces_main_flow(
        self,
        request: CellProfilerMainFlowReplacementRequest,
    ) -> bool:
        return len(request.executor.image_outputs) == 1


class SideArtifactImageOutputMainFlowReplacementPolicy(
    CellProfilerMainFlowReplacementPolicy
):
    """Record declared image outputs without advancing OpenHCS main flow."""

    def replaces_main_flow(
        self,
        request: CellProfilerMainFlowReplacementRequest,
    ) -> bool:
        del request
        return False


class CorrectIlluminationCalculateMainFlowReplacementPolicy(
    SideArtifactImageOutputMainFlowReplacementPolicy
):
    """Illumination functions are side artifacts consumed by apply modules."""

    module_name = "CorrectIlluminationCalculate"


class CellProfilerInvocationExecutionModePolicy(
    ABC,
    metaclass=CellProfilerModulePolicyMeta,
):
    """Nominal policy for modules whose settings change stack execution mode."""

    @classmethod
    @lru_cache(maxsize=None)
    def for_module(
        cls, module_name: str
    ) -> "CellProfilerInvocationExecutionModePolicy":
        policy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            DefaultInvocationExecutionModePolicy,
        )
        return policy_type()

    def execution_mode(
        self,
        *,
        default: ImagePayloadExecutionMode,
        image: Any,
        kwargs: Mapping[str, Any],
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del image
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
        image: Any,
        kwargs: Mapping[str, Any],
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del image
        if illumination_scope_uses_all_images(kwargs.get("calculation_scope")):
            return ImagePayloadExecutionMode.FULL_STACK
        return default


class DefineGridManualExecutionModePolicy(CellProfilerInvocationExecutionModePolicy):
    """Honor CellProfiler's per-cycle versus once-only grid definition scope."""

    module_name = "DefineGridManual"

    def execution_mode(
        self,
        *,
        default: ImagePayloadExecutionMode,
        image: Any,
        kwargs: Mapping[str, Any],
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del image
        scope = (
            invocation_options.grid_cycle_scope
            if invocation_options is not None
            else coerce_cellprofiler_grid_cycle_scope(
                kwargs.get(CELLPROFILER_GRID_CYCLE_SCOPE_KWARG)
            )
        )
        if scope is CellProfilerGridCycleScope.ONCE:
            return ImagePayloadExecutionMode.FULL_STACK
        return default


class CellProfilerPayloadSpatialRankStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Resolve spatial rank from nominal runtime payload types."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def resolve_rank(cls, value: Any) -> int | None:
        strategy = cls.for_nominal_value(value)
        if strategy is None:
            return None
        return strategy.spatial_rank(value)

    @abstractmethod
    def spatial_rank(self, value: Any) -> int | None:
        """Return the spatial rank, excluding color channels, when known."""


class DenseArrayPayloadSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank for dense image arrays."""

    value_type = np.ndarray

    def spatial_rank(self, value: Any) -> int | None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Dense array rank strategy requires ndarray.")
        if is_color_image_slice(value) or is_color_image_stack(value):
            return 2
        if (
            is_grayscale_volume_stack(value)
            or is_color_volume_slice(value)
            or is_color_volume_stack(value)
        ):
            return 3
        return int(value.ndim)


class MaskedImagePayloadSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank through masked-image payload data."""

    value_type = MaskedImagePayload

    def spatial_rank(self, value: Any) -> int | None:
        if not isinstance(value, MaskedImagePayload):
            raise TypeError(
                "MaskedImagePayload rank strategy requires MaskedImagePayload."
            )
        return CellProfilerPayloadSpatialRankStrategy.resolve_rank(value.data)


class ImageMetadataPayloadSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank through image metadata payload data."""

    value_type = ImageMetadataPayload

    def spatial_rank(self, value: Any) -> int | None:
        if not isinstance(value, ImageMetadataPayload):
            raise TypeError(
                "ImageMetadataPayload rank strategy requires ImageMetadataPayload."
            )
        return CellProfilerPayloadSpatialRankStrategy.resolve_rank(value.data)


class ObjectLabelValueSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank for nominal object-label runtime values."""

    value_type = (ObjectLabelPayload, ObjectLabelSet)

    def spatial_rank(self, value: Any) -> int | None:
        if not isinstance(value, (ObjectLabelPayload, ObjectLabelSet)):
            raise TypeError(
                "Object-label rank strategy requires an object-label runtime value."
            )
        return ObjectLabelDenseDataStrategy.spatial_rank(value)


class VolumetricInputExecutionModePolicy(CellProfilerInvocationExecutionModePolicy):
    """Run full-stack when the nominal image payload contains a Z volume."""

    module_name = None

    def execution_mode(
        self,
        *,
        default: ImagePayloadExecutionMode,
        image: Any,
        kwargs: Mapping[str, Any],
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del kwargs, invocation_options
        if self.is_volumetric_payload(image):
            return ImagePayloadExecutionMode.FULL_STACK
        return default

    def is_volumetric_payload(self, image: Any) -> bool:
        spatial_rank = self.spatial_rank(image)
        return spatial_rank is not None and spatial_rank >= 3

    def spatial_rank(self, image: Any) -> int | None:
        data_rank = CellProfilerPayloadSpatialRankStrategy.resolve_rank(image)
        if data_rank is not None:
            return data_rank
        return CellProfilerPayloadSpatialRankStrategy.resolve_rank(
            image_payload_data(image)
        )

    def invocation_spatial_rank(
        self,
        *,
        image: Any,
        kwargs: Mapping[str, Any],
    ) -> int | None:
        ranks = [
            rank
            for rank in (
                self.spatial_rank(image),
                *(
                    CellProfilerPayloadSpatialRankStrategy.resolve_rank(value)
                    for value in kwargs.values()
                ),
            )
            if rank is not None
        ]
        return max(ranks) if ranks else None


class StructuringElementExecutionModePolicy(VolumetricInputExecutionModePolicy):
    """Match CellProfiler morphology dispatch from typed footprint rank."""

    module_name = None

    def execution_mode(
        self,
        *,
        default: ImagePayloadExecutionMode,
        image: Any,
        kwargs: Mapping[str, Any],
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del invocation_options
        spatial_rank = self.invocation_spatial_rank(image=image, kwargs=kwargs)
        if spatial_rank is None or spatial_rank < 3:
            return default
        footprint = build_structuring_element(
            kwargs["structuring_element"],
            int(kwargs["size"]),
        )
        if footprint.ndim == spatial_rank:
            return ImagePayloadExecutionMode.FULL_STACK
        return default


@dataclass(frozen=True, slots=True)
class InvocationExecutionModePolicySpec(GeneratedLeafClassSpec):
    """Declarative leaf spec for module-name execution-mode policies."""

    module_name: str

    def class_attributes(self) -> Mapping[str, object]:
        return {"module_name": self.module_name}


for _execution_mode_policy_spec in (
    InvocationExecutionModePolicySpec(
        class_name="ThresholdExecutionModePolicy",
        base_type=VolumetricInputExecutionModePolicy,
        module_name="Threshold",
    ),
    InvocationExecutionModePolicySpec(
        class_name="WatershedExecutionModePolicy",
        base_type=VolumetricInputExecutionModePolicy,
        module_name="Watershed",
    ),
    InvocationExecutionModePolicySpec(
        class_name="RemoveHolesExecutionModePolicy",
        base_type=VolumetricInputExecutionModePolicy,
        module_name="RemoveHoles",
    ),
    InvocationExecutionModePolicySpec(
        class_name="ClosingExecutionModePolicy",
        base_type=StructuringElementExecutionModePolicy,
        module_name="Closing",
    ),
    InvocationExecutionModePolicySpec(
        class_name="OpeningExecutionModePolicy",
        base_type=StructuringElementExecutionModePolicy,
        module_name="Opening",
    ),
    InvocationExecutionModePolicySpec(
        class_name="ErodeImageExecutionModePolicy",
        base_type=StructuringElementExecutionModePolicy,
        module_name="ErodeImage",
    ),
    InvocationExecutionModePolicySpec(
        class_name="ErodeObjectsExecutionModePolicy",
        base_type=StructuringElementExecutionModePolicy,
        module_name="ErodeObjects",
    ),
    InvocationExecutionModePolicySpec(
        class_name="DilateImageExecutionModePolicy",
        base_type=StructuringElementExecutionModePolicy,
        module_name="DilateImage",
    ),
):
    _execution_mode_policy_spec.declare_in(globals())


class CellProfilerImageExecutionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal executor mode family for CellProfiler image payload semantics."""

    __registry_key__ = "mode_key"
    __skip_if_no_key__ = True
    mode: ClassVar[ImagePayloadExecutionMode | None] = None
    mode_key: ClassVar[str | None] = None

    @classmethod
    @lru_cache(maxsize=None)
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
        function_name = CallableContract.from_callable(func).function_name
        contract_started_at = time.perf_counter()
        contract = _processing_contract_for_callable(func)
        _log_module_profile(
            "cp_natural_processing_contract",
            time.perf_counter() - contract_started_at,
            function=function_name,
            contract=contract.name,
        )
        execute_started_at = time.perf_counter()
        if (
            contract is ProcessingContract.PURE_2D
            and _slice_count_from_pure_2d_kwargs(kwargs) is not None
        ):
            result = executor._execute_pure_2d(func, image, **kwargs)
        else:
            result = contract.execute(
                executor,
                func,
                image,
                **kwargs,
            )
        _log_module_profile(
            "cp_natural_contract_execute",
            time.perf_counter() - execute_started_at,
            function=function_name,
            contract=contract.name,
        )
        return result


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
        return executor._execute_pure_3d(func, image, **kwargs)


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


def _coerce_invocation_kwargs(
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    accepts_var_keyword, accepted_names, enum_types = _callable_invocation_kwarg_spec(
        func
    )
    if accepts_var_keyword:
        coerced_kwargs = dict(kwargs)
    else:
        coerced_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in accepted_names or name in _INVOCATION_CONTROL_KWARGS
        }
    for name, enum_type in enum_types:
        if name not in coerced_kwargs:
            continue
        value = coerced_kwargs[name]
        coerced_kwargs[name] = _coerce_enum_argument(enum_type, value, name)
    return coerced_kwargs


@lru_cache(maxsize=256)
def _callable_invocation_kwarg_spec(
    func: Callable[..., Any],
) -> tuple[bool, frozenset[str], tuple[tuple[str, type[Enum]], ...]]:
    parameters = _callable_parameters(func)
    annotations = _callable_type_hints(func)
    accepts_var_keyword = any(
        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    enum_types = tuple(
        (name, enum_type)
        for name, parameter in parameters.items()
        if (
            enum_type := _enum_annotation_type(
                parameter,
                annotations.get(name),
            )
        )
        is not None
    )
    return accepts_var_keyword, frozenset(parameters), enum_types


def _accepted_invocation_kwargs(
    parameters: Mapping[str, Parameter],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if any(
        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return dict(kwargs)
    return {
        name: value
        for name, value in kwargs.items()
        if name in parameters or name in _INVOCATION_CONTROL_KWARGS
    }


@lru_cache(maxsize=256)
def _callable_parameters(func: Callable[..., Any]) -> Mapping[str, Parameter]:
    return signature(func).parameters


@lru_cache(maxsize=256)
def _callable_type_hints(func: Callable[..., Any]) -> Mapping[str, Any]:
    try:
        return get_type_hints(func)
    except (NameError, TypeError):
        return {}


def _cellprofiler_image_number_start(
    image_payload: Any,
    adapter: CellProfilerRuntimeAdapter,
    *,
    source_image_name: str | None = None,
) -> int:
    """Return CP's 1-based ImageNumber for the first plane in a stack."""
    source_paths = _payload_source_paths(image_payload)
    ordered_pipeline_paths = adapter.cellprofiler_ordered_pipeline_image_paths()
    if not ordered_pipeline_paths:
        return 1
    if not source_paths and source_image_name:
        source_paths = _runtime_image_source_paths(adapter, source_image_name)
    if not source_paths:
        return adapter.cellprofiler_axis_image_number_start()

    first_source_path = adapter.cellprofiler_source_order_path(source_paths[0])
    try:
        return ordered_pipeline_paths.index(first_source_path) + 1
    except ValueError:
        return adapter.cellprofiler_axis_image_number_start()


def _payload_source_paths(image_payload: Any) -> tuple[str, ...]:
    metadata = image_payload_metadata(image_payload)
    paths = tuple(
        str(path)
        for path in metadata.channel_source_paths
        if path is not None and str(path)
    )
    if paths:
        return paths
    if metadata.source_path:
        return (metadata.source_path,)
    return ()


def _runtime_image_source_paths(
    adapter: CellProfilerRuntimeAdapter,
    image_name: str,
) -> tuple[str, ...]:
    direct_records = adapter.runtime_value_store.find(
        name=image_name,
        kind=ArtifactKind.IMAGE,
        axis_id=adapter.axis_id,
    )
    lineage_records = adapter.runtime_value_store.find(
        kind=ArtifactKind.IMAGE,
        axis_id=adapter.axis_id,
    )
    for record in (*direct_records, *lineage_records):
        if (
            record.key.name != image_name
            and record.value.schema.source_image_name != image_name
        ):
            continue
        source_paths = _payload_source_paths(record.value.data)
        if source_paths:
            return source_paths
    return ()


def _enum_annotation_type(
    parameter: Any,
    resolved_annotation: Any = None,
) -> type[Enum] | None:
    if parameter is None:
        return None
    annotation = (
        resolved_annotation if resolved_annotation is not None else parameter.annotation
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
        literals.extend(item for item in member.value if isinstance(item, str))
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
    @lru_cache(maxsize=None)
    def for_kind(cls, kind: ArtifactKind) -> "RuntimeArtifactKindStrategy":
        return cls.__registry__[kind]()

    @abstractmethod
    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        """Return the runtime payload bound into absorbed function kwargs."""

    def raw_runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        """Return the runtime payload before CellProfiler intensity coercion."""
        return self.runtime_input_value(request)

    @abstractmethod
    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        """Return the transitive source image name for one artifact input."""


class ImageArtifactKindStrategy(RuntimeArtifactKindStrategy):
    """Resolve image artifact payloads and source-image lineage."""

    kind = ArtifactKind.IMAGE

    def raw_runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        if request.spec.name in request.runtime_image_names:
            return request.adapter.get_image(request.spec.name).data
        if request.spec.name in request.external_image_names:
            if request.current_image is None:
                raise RuntimeError(
                    f"External image input '{request.spec.name}' requires a "
                    "current image payload for source-binding resolution."
                )
            return request.adapter.resolve_source_image(
                request.spec.name,
                request.current_image,
            )
        return request.adapter.get_image(request.spec.name).data

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> Any:
        return _cellprofiler_image_payload(self.raw_runtime_input_value(request))

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
            return collapse_singleton_object_label_stack(
                _object_label_runtime_payload(
                    request.adapter.resolve_source_objects(
                        request.spec.name,
                        request.current_image,
                    )
                )
            )
        return collapse_singleton_object_label_stack(
            _object_label_runtime_payload(
                request.adapter.get_objects(request.spec.name)
            )
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


class CellProfilerSpecialInputValueStrategy(
    EnumKeyedStrategyMixin[CellProfilerSpecialInputPayloadSemantics],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Resolve special-input payloads by declared role semantics."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[CellProfilerSpecialInputPayloadSemantics | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def runtime_input_value(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> Any:
        """Return the callable value for one role-resolved special input."""


class IntensityImageSpecialInputValueStrategy(CellProfilerSpecialInputValueStrategy):
    """Bind image-like special inputs in CellProfiler's intensity domain."""

    strategy_key = CellProfilerSpecialInputPayloadSemantics.INTENSITY_IMAGE

    def runtime_input_value(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> Any:
        return _artifact_kind_strategy(request.spec.kind).runtime_input_value(request)


class DenseLabelImageSpecialInputValueStrategy(CellProfilerSpecialInputValueStrategy):
    """Bind image-carried label IDs without intensity normalization."""

    strategy_key = CellProfilerSpecialInputPayloadSemantics.DENSE_LABEL_IMAGE

    def runtime_input_value(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> np.ndarray:
        raw_value = _artifact_kind_strategy(request.spec.kind).raw_runtime_input_value(
            request
        )
        return object_label_dense_array(raw_value, dtype=np.int32)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeInputBindingRequestBase(ABC, metaclass=AutoRegisterMeta):
    """Shared runtime context for artifact-backed runtime-input binding."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None

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
        return collapse_singleton_object_label_stack(self.label_payload_for(spec))

    def label_payload_for(self, spec: ArtifactSpec) -> Any:
        if spec.name in self.external_object_names:
            return _object_label_runtime_payload(
                self.adapter.resolve_source_objects(
                    spec.name,
                    self.current_image,
                )
            )
        return _object_label_runtime_payload(self.adapter.get_objects(spec.name))


def _object_label_runtime_payload(objects: ObjectLabelSet) -> Any:
    if objects.representation is ObjectLabelRepresentation.SPARSE_IJV:
        return objects
    return objects.runtime_payload()


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectInputBindingRequest(RuntimeInputBindingRequestBase):
    """Authoritative runtime context for binding object-label inputs."""

    registry_key = "object_input"

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


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementVector:
    """CellProfiler-facing projection of one object/image measurement vector."""

    slices: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slices", tuple(self.slices))

    @property
    def slice_aligned_value(self) -> np.ndarray | CellProfilerSliceAlignedValues:
        if len(self.slices) == 1:
            return np.asarray(self.slices[0])
        return CellProfilerSliceAlignedValues(
            tuple(np.asarray(value) for value in self.slices)
        )

    @property
    def calculate_math_operand_value(self) -> Any:
        if len(self.slices) != 1:
            return self.slice_aligned_value
        values = np.asarray(self.slices[0])
        return float(values[0]) if values.size == 1 else values


@dataclass(frozen=True, slots=True)
class CellProfilerObjectMeasurementVectorBinding:
    """Nominal binding from object-label runtime inputs to measurement vectors."""

    request: RuntimeInputBindingRequestBase
    object_name: str
    feature_name: str
    object_spec: ArtifactSpec | None = None
    labels: Any | None = None

    @classmethod
    def for_object_input(
        cls,
        request: RuntimeInputBindingRequestBase,
        *,
        object_spec: ArtifactSpec,
        feature_name: str,
        labels: Any | None = None,
    ) -> "CellProfilerObjectMeasurementVectorBinding":
        return cls(
            request=request,
            object_name=object_spec.name,
            feature_name=feature_name,
            object_spec=object_spec,
            labels=labels,
        )

    @classmethod
    def for_object_name(
        cls,
        request: ObjectInputBindingRequest,
        *,
        object_name: str,
        feature_name: str,
    ) -> "CellProfilerObjectMeasurementVectorBinding":
        return cls(
            request=request,
            object_name=object_name,
            feature_name=feature_name,
            object_spec=ArtifactSpecCollection(request.object_inputs).by_name(
                object_name
            ),
        )

    def vector(self) -> CellProfilerMeasurementVector:
        if self.object_spec is None:
            return CellProfilerMeasurementVector(
                (
                    self.request.adapter.measurement_values_for_object_feature(
                        self.object_name,
                        self.feature_name,
                    ),
                )
            )
        return CellProfilerMeasurementVector(
            self.request.adapter.measurement_values_for_label_slices(
                self.object_name,
                self.feature_name,
                (
                    self.labels
                    if self.labels is not None
                    else self.request.labels_for(self.object_spec)
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class CellProfilerObjectMeasurementVectorBatchBinding:
    """Batch object-measurement vector bindings that share runtime semantics."""

    bindings: tuple[CellProfilerObjectMeasurementVectorBinding, ...]

    def vectors(self) -> tuple[CellProfilerMeasurementVector, ...]:
        if not self.bindings:
            return ()
        feature_names = tuple(
            dict.fromkeys(binding.feature_name for binding in self.bindings)
        )
        if len(feature_names) != 1 or any(
            binding.object_spec is None for binding in self.bindings
        ):
            return tuple(binding.vector() for binding in self.bindings)

        feature_name = feature_names[0]
        requests = {
            binding.object_name: (
                binding.feature_name,
                (
                    binding.labels
                    if binding.labels is not None
                    else binding.request.labels_for(binding.object_spec)
                ),
            )
            for binding in self.bindings
        }
        vectors = self.bindings[
            0
        ].request.adapter.measurement_values_for_label_slice_batch(
            requests,
            feature_name=feature_name,
        )
        return tuple(
            CellProfilerMeasurementVector(vectors[binding.object_name])
            for binding in self.bindings
        )


class CellProfilerObjectInputPolicy(
    ABC,
    metaclass=CellProfilerModulePolicyMeta,
):
    """Nominal binding policy for CellProfiler object-label inputs."""

    binds_without_declared_inputs: ClassVar[bool] = False
    supported_non_object_input_kinds: ClassVar[frozenset[ArtifactKind]] = frozenset()

    @classmethod
    @lru_cache(maxsize=None)
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


class CellProfilerPrimaryImageInputPolicy(
    ABC,
    metaclass=CellProfilerModulePolicyMeta,
):
    """Nominal policy for image artifacts that drive absorbed execution."""

    @classmethod
    @lru_cache(maxsize=None)
    def for_module(cls, module_name: str) -> "CellProfilerPrimaryImageInputPolicy":
        policy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            DefaultPrimaryImageInputPolicy,
        )
        return policy_type()

    @abstractmethod
    def primary_image_inputs(
        self,
        module_name: str,
        func: Callable[..., Any],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return image inputs that should drive function invocation slices."""


class DefaultPrimaryImageInputPolicy(CellProfilerPrimaryImageInputPolicy):
    """Use non-special image inputs as the algorithmic image domain."""

    def primary_image_inputs(
        self,
        module_name: str,
        func: Callable[..., Any],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        image_inputs = ArtifactSpecCollection(declared_inputs).of_kind(
            ArtifactKind.IMAGE
        )
        special_image_count = len(
            CellProfilerSpecialInputPolicy.for_module(
                canonical_module_name(module_name)
            ).special_image_inputs(
                module_name,
                func,
                declared_inputs,
            )
        )
        if special_image_count == 0:
            return image_inputs
        return image_inputs[: len(image_inputs) - special_image_count]


class ObjectLabelDrivenPrimaryImageInputPolicy(DefaultPrimaryImageInputPolicy):
    """Treat declared images as carriers; object labels define the domain."""

    module_names: ClassVar[frozenset[str]] = frozenset({"MaskObjects"})

    def primary_image_inputs(
        self,
        module_name: str,
        func: Callable[..., Any],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func, declared_inputs
        return ()


for (
    _object_label_driven_module_name
) in ObjectLabelDrivenPrimaryImageInputPolicy.module_names:
    _object_label_driven_policy_name = (
        f"{_object_label_driven_module_name}PrimaryImageInputPolicy"
    )
    globals()[_object_label_driven_policy_name] = type(
        _object_label_driven_policy_name,
        (ObjectLabelDrivenPrimaryImageInputPolicy,),
        {
            "__module__": __name__,
            "module_name": _object_label_driven_module_name,
        },
    )
del _object_label_driven_module_name, _object_label_driven_policy_name


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
        return {self.label_kwarg: request.labels_for(request.object_inputs[0])}


class IdentifySecondaryObjectsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind primary objects with generic label-variant context when available."""

    module_name = "IdentifySecondaryObjects"

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        request.require_exact_object_count(1)
        return {"primary_labels": request.label_payload_for(request.object_inputs[0])}


@dataclass(frozen=True, slots=True)
class SingleObjectLabelInputPolicySpec(GeneratedLeafClassSpec):
    """Declarative leaf spec for one object-label binding policy."""

    module_name: str
    label_kwarg: str

    def class_attributes(self) -> Mapping[str, object]:
        return {
            "module_name": self.module_name,
            "label_kwarg": self.label_kwarg,
        }


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
            "primary_labels": request.label_payload_for(smaller),
            "secondary_labels": request.label_payload_for(larger),
        }


_MEASURE_OBJECT_SIZE_SHAPE_MODULE = "MeasureObjectSizeShape"
_MEASURE_OBJECT_INTENSITY_MODULE = "MeasureObjectIntensity"
_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE = "MeasureObjectIntensityDistribution"
_MEASURE_TEXTURE_MODULE = "MeasureTexture"
_MEASURE_COLOCALIZATION_MODULE = "MeasureColocalization"
_MEASURE_GRANULARITY_MODULE = "MeasureGranularity"
_MEASURE_OBJECT_NEIGHBORS_MODULE = "MeasureObjectNeighbors"
_TRACK_OBJECTS_MODULE = "TrackObjects"
_OBJECT_ROW_SEQUENCE_KWARGS = frozenset({"object_labels"})
_MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS = (
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_OBJECT_ID_FIELD,
)
_MISSING_MEASUREMENT_ROW_VALUE = object()


class MissingObjectMeasurementValuePolicy(str, Enum):
    """How missing per-object measurement result fields are materialized."""

    NAN = "nan"
    ZERO_WITHIN_POSITIVE_EXTENT = "zero_within_positive_extent"


@dataclass(frozen=True, slots=True)
class ObjectMeasurementInvocation:
    """One semantic object-measurement function invocation."""

    kwargs: Mapping[str, Any]
    source_pair: CellProfilerSourceImagePair | None = None

    def lowered_kwargs(self) -> dict[str, Any]:
        """Return kwargs lowered to the CellProfiler function-call ABI."""
        return dict(self.kwargs)


@dataclass(frozen=True, slots=True)
class SourcePairObjectMeasurementInvocation(ObjectMeasurementInvocation):
    """Object measurement invocation over one ordered source-image pair."""

    first_channel_kwarg: str = "channel_1"
    second_channel_kwarg: str = "channel_2"

    def __post_init__(self) -> None:
        if self.source_pair is None:
            raise ValueError(
                "SourcePairObjectMeasurementInvocation requires a source_pair."
            )

    def lowered_kwargs(self) -> dict[str, Any]:
        assert self.source_pair is not None
        return {
            **self.kwargs,
            **self.source_pair.invocation_kwargs(
                first_channel_kwarg=self.first_channel_kwarg,
                second_channel_kwarg=self.second_channel_kwarg,
            ),
        }


class CellProfilerObjectMeasurementRowPolicy(
    ABC,
    metaclass=CellProfilerModulePolicyMeta,
):
    """Nominal export-row policy for object-scoped measurement modules."""

    row_identity: ClassVar[MeasurementObjectRowIdentity] = (
        MeasurementObjectRowIdentity.LABEL_ID
    )
    missing_value_policy: ClassVar[MissingObjectMeasurementValuePolicy] = (
        MissingObjectMeasurementValuePolicy.NAN
    )

    @classmethod
    @lru_cache(maxsize=None)
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

    def invocations(
        self,
        measurement_image: CellProfilerMeasurementImage,
        kwargs: Mapping[str, Any],
    ) -> tuple[ObjectMeasurementInvocation, ...]:
        """Return semantic function invocations for this measurement image."""
        del measurement_image
        return (ObjectMeasurementInvocation(kwargs=kwargs),)

    def project_rows(
        self,
        rows: Sequence[Any],
        invocation: ObjectMeasurementInvocation,
    ) -> list[Any]:
        """Return emitted rows projected into this module's feature namespace."""
        del invocation
        return list(rows)

    def table_source_image_name(
        self,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
        source_image_name: str | None,
    ) -> str | None:
        """Return table-level source ownership for rows emitted by this policy."""
        if not measurement_images:
            return source_image_name
        return CellProfilerMeasurementImage.shared_source_image_name(measurement_images)

    def requires_explicit_row_ownership(self) -> bool:
        """Return whether emitted rows carry mixed measurement ownership."""
        return False

    def record_partitions(
        self,
        record: CellProfilerMeasurementRecord,
    ) -> tuple[CellProfilerMeasurementRecordPartition, ...]:
        """Return single-owner measurement table partitions for a record."""
        if not self.requires_explicit_row_ownership():
            return (
                CellProfilerMeasurementRecordPartition(
                    rows=record.rows,
                    object_name=record.object_name,
                    source_image_name=record.source_image_name,
                    source_image_payload=record.source_image_payload,
                    fields=record.fields,
                ),
            )
        return self.explicit_owner_record_partitions(record)

    def explicit_owner_record_partitions(
        self,
        record: CellProfilerMeasurementRecord,
    ) -> tuple[CellProfilerMeasurementRecordPartition, ...]:
        """Return fail-loud table ownership for rows with explicit row owners."""
        object_rows: dict[str, list[Any]] = {}
        image_rows: dict[str, list[Any]] = {}
        for row in record.rows:
            row_mapping = measurement_row_mapping(row)
            object_name = measurement_row_object_name(row_mapping)
            source_image_name = measurement_row_source_image_name(row_mapping)
            if object_name is not None:
                object_rows.setdefault(object_name, []).append(row)
                continue
            if source_image_name is not None:
                image_rows.setdefault(source_image_name, []).append(row)
                continue
            raise ValueError(
                f"{type(self).__name__} requires every mixed-scope measurement row "
                "to declare object or source-image ownership."
            )
        if object_rows and image_rows:
            if len(object_rows) != 1:
                raise ValueError(
                    f"{type(self).__name__} requires one table-level object owner "
                    f"for mixed measurement rows, got {tuple(object_rows)}."
                )
            object_name = next(iter(object_rows))
            return (
                CellProfilerMeasurementRecordPartition(
                    rows=record.rows,
                    object_name=object_name,
                    source_image_name=None,
                    source_image_payload=record.source_image_payload,
                    fields=record.fields,
                ),
            )
        if image_rows:
            if len(image_rows) != 1:
                raise ValueError(
                    f"{type(self).__name__} requires one table-level source image "
                    f"for mixed measurement rows, got {tuple(image_rows)}."
                )
            source_image_name = next(iter(image_rows))
            return (
                CellProfilerMeasurementRecordPartition(
                    rows=record.rows,
                    object_name=None,
                    source_image_name=source_image_name,
                    source_image_payload=record.source_image_payload,
                    fields=record.fields,
                ),
            )
        return (
            *(
                CellProfilerMeasurementRecordPartition(
                    rows=rows,
                    object_name=object_name,
                    source_image_name=None,
                    source_image_payload=record.source_image_payload,
                    fields=record.fields,
                )
                for object_name, rows in object_rows.items()
            ),
            *(
                CellProfilerMeasurementRecordPartition(
                    rows=rows,
                    object_name=None,
                    source_image_name=source_image_name,
                    source_image_payload=record.source_image_payload,
                    fields=record.fields,
                )
                for source_image_name, rows in image_rows.items()
            ),
        )

    def annotate_record_rows(
        self,
        rows: Sequence[Any],
        *,
        object_name: str | None,
        source_image_name: str | None,
    ) -> list[Any]:
        """Return rows with row-level ownership declared when policy requires it."""
        if not self.requires_explicit_row_ownership():
            return list(rows)
        return [
            self.annotate_record_row(
                row,
                object_name=object_name,
                source_image_name=source_image_name,
            )
            for row in rows
        ]

    def annotate_record_row(
        self,
        row: Any,
        *,
        object_name: str | None,
        source_image_name: str | None,
    ) -> Any:
        """Return one row with the semantic owner required by this policy."""
        if self.row_is_object_scoped(row):
            if object_name is None:
                raise ValueError(
                    f"{type(self).__name__} requires an object name for object rows."
                )
            return annotate_measurement_row_object(row, object_name)
        resolved_source_image_name = self.image_row_source_image_name(source_image_name)
        if resolved_source_image_name is None:
            raise ValueError(
                f"{type(self).__name__} requires a source image name for image rows."
            )
        return annotate_measurement_row_source_image(row, resolved_source_image_name)

    def image_row_source_image_name(
        self,
        source_image_name: str | None,
    ) -> str | None:
        """Return the source owner for image-scoped rows emitted by this module."""
        return source_image_name

    def row_is_object_scoped(self, row: Any) -> bool:
        """Return whether a raw emitted row belongs to the object domain."""
        del row
        return True

    def row_has_measured_object(
        self,
        row_mapping: Mapping[str, object],
        *,
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> bool:
        """Return whether a source row should consume an object row identity."""
        return self.row_has_result_payload(
            row_mapping,
            object_id_field=object_id_field,
            axis_fields=axis_fields,
        )

    def row_has_result_payload(
        self,
        row_mapping: Mapping[str, object],
        *,
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> bool:
        """Return whether a row carries result values, not just identity padding."""
        metadata_fields = {
            object_id_field,
            *MEASUREMENT_OBJECT_ID_FIELDS,
            *axis_fields,
            MEASUREMENT_OBJECT_NAME_FIELD,
            MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
        }
        return any(
            field_name not in metadata_fields
            and self.measurement_value_is_present(value)
            for field_name, value in row_mapping.items()
        )

    def measurement_value_is_present(self, value: object) -> bool:
        """Return whether a measurement cell is an observed value, not padding."""
        if value is None or value == "":
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return True
        return not math.isnan(numeric)

    def retains_unmeasured_compact_row(
        self,
        row_mapping: Mapping[str, object],
        *,
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> bool:
        """Return whether compact row projection should keep an unmeasured row."""
        del row_mapping, object_id_field, axis_fields
        return True

    def required_object_ids_for_axis(
        self,
        *,
        label_payload: Any,
        projected_rows: Sequence[Any],
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_key: tuple[Any, ...],
    ) -> tuple[int, ...]:
        """Return object row IDs required by this policy for one measurement axis."""
        del projected_rows, object_id_field
        schema = ObjectMeasurementRowCompletionSchema(
            field_names=(),
            object_id_field=MEASUREMENT_OBJECT_LABEL_FIELD,
            axis_fields=tuple(axis_fields),
        )
        return schema.object_ids_for_axis(
            label_payload=label_payload,
            object_identity=object_identity,
            axis_key=axis_key,
        )

    def required_object_ids_by_axis(
        self,
        *,
        label_payload: Any,
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_keys: Sequence[tuple[Any, ...]],
    ) -> dict[tuple[Any, ...], tuple[int, ...]]:
        """Return required object row IDs for every measurement axis."""
        return {
            axis_key: self.required_object_ids_for_axis(
                label_payload=label_payload,
                projected_rows=projection.rows,
                object_identity=object_identity,
                object_id_field=object_id_field,
                axis_fields=axis_fields,
                axis_key=axis_key,
            )
            for axis_key in axis_keys
        }

    def complete_rows(
        self,
        rows: Sequence[Any],
        *,
        label_payload: Any,
        func: Callable[..., Any],
    ) -> list[Any]:
        """Pad per-object measurement rows across this policy's object domain."""
        schema = ObjectMeasurementRowCompletionSchema.from_rows(rows, func)
        object_identity = self.object_identity()
        projection_request = ObjectMeasurementRowIdentityProjectionRequest(
            rows=rows,
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
            row_policy=self,
        )
        projection = MeasurementObjectRowIdentityProjectionStrategy.project(
            object_identity,
            projection_request,
        )
        projected_rows = projection.rows
        if projected_rows and not any(
            object_id is not None for object_id, _axis_key in projection.row_keys
        ):
            return list(projected_rows)
        axis_keys = projection_request.axis_keys_for_label_payload(
            projection,
            label_payload=label_payload,
        )
        if not axis_keys:
            axis_keys = ((),)
        object_ids_by_axis = self.required_object_ids_by_axis(
            label_payload=label_payload,
            projection=projection,
            object_identity=object_identity,
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
            axis_keys=axis_keys,
        )
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

        required_row_keys = {
            (object_id, axis_key)
            for axis_key in axis_keys
            for object_id in object_ids_by_axis[axis_key]
        }
        present_row_keys: set[tuple[int, tuple[Any, ...]]] = set()
        for object_id, row_axis_key in projection.row_keys:
            if object_id is None:
                continue
            row_key = (object_id, row_axis_key)
            if row_key not in required_row_keys:
                continue
            present_row_keys.add(row_key)
            if len(present_row_keys) == len(required_row_keys):
                break
        completed_rows = list(projected_rows)
        appended_row_keys: list[tuple[int, tuple[Any, ...]]] = []
        missing_row_keys = [
            (object_id, axis_key)
            for axis_key in axis_keys
            for object_id in object_ids_by_axis[axis_key]
            if (object_id, axis_key) not in present_row_keys
        ]
        if not missing_row_keys:
            return list(projected_rows)

        unique_missing_axis_keys = tuple(
            dict.fromkeys(axis_key for _object_id, axis_key in missing_row_keys)
        )
        positive_label_extent_by_axis = {
            axis_key: schema.positive_extent_for_missing_measurements(
                label_payload=label_payload,
                axis_key=axis_key,
                row_policy=self,
            )
            for axis_key in unique_missing_axis_keys
        }
        for object_id, axis_key in missing_row_keys:
            row = schema.missing_row(
                object_id=object_id,
                axis_key=axis_key,
                label_payload=label_payload,
                row_policy=self,
                positive_label_extent=positive_label_extent_by_axis[axis_key],
            )
            completed_rows.append(row)
            appended_row_keys.append((object_id, axis_key))
        completed_row_keys = list(projection.row_keys)
        completed_row_keys.extend(appended_row_keys)
        return projection.ordered_rows(
            rows=completed_rows,
            row_keys=completed_row_keys,
            object_ids=object_ids,
            axis_keys=axis_keys,
        )

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
            return 0.0 if object_id <= extent else np.nan
        raise ValueError(f"Unsupported missing measurement value policy: {policy}.")


class DefaultObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """Use runtime object-label IDs as measurement-row identities."""


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowPolicySpec(GeneratedLeafClassSpec):
    """Declarative leaf spec for one object measurement row policy."""

    module_name: str
    row_identity: MeasurementObjectRowIdentity = MeasurementObjectRowIdentity.LABEL_ID
    missing_value_policy: MissingObjectMeasurementValuePolicy = (
        MissingObjectMeasurementValuePolicy.NAN
    )

    def class_attributes(self) -> Mapping[str, object]:
        return {
            "module_name": self.module_name,
            "row_identity": self.row_identity,
            "missing_value_policy": self.missing_value_policy,
        }


class DeclaredObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """Generated base for modules with declared measurement-row identity."""


class CompactMeasuredObjectMeasurementRowPolicy(DeclaredObjectMeasurementRowPolicy):
    """Use CP's compact row identity for emitted measurement rows."""

    row_identity = MeasurementObjectRowIdentity.ROW_ORDINAL

    def required_object_ids_for_axis(
        self,
        *,
        label_payload: Any,
        projected_rows: Sequence[Any],
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_key: tuple[Any, ...],
    ) -> tuple[int, ...]:
        """Compact CP rows are dense over emitted row ordinals."""
        del label_payload, object_identity
        projection_request = ObjectMeasurementRowIdentityProjectionRequest(
            rows=projected_rows,
            object_id_field=object_id_field,
            axis_fields=axis_fields,
            row_policy=self,
        )
        object_ids = tuple(
            object_id
            for row in projected_rows
            if projection_request.axis_key(row) == axis_key
            for object_id in (projection_request.object_label(row),)
            if object_id is not None
        )
        if not object_ids:
            return ()
        return tuple(range(1, max(object_ids) + 1))

    def required_object_ids_by_axis(
        self,
        *,
        label_payload: Any,
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        object_identity: MeasurementObjectRowIdentity,
        object_id_field: str,
        axis_fields: Sequence[str],
        axis_keys: Sequence[tuple[Any, ...]],
    ) -> dict[tuple[Any, ...], tuple[int, ...]]:
        """Compact CP rows are dense over emitted row ordinals per axis."""
        del label_payload, object_identity, object_id_field, axis_fields
        max_object_id_by_axis = {axis_key: 0 for axis_key in axis_keys}
        for object_id, axis_key in projection.row_keys:
            if object_id is None or axis_key not in max_object_id_by_axis:
                continue
            max_object_id_by_axis[axis_key] = max(
                max_object_id_by_axis[axis_key],
                object_id,
            )
        return {
            axis_key: tuple(range(1, max_object_id + 1))
            for axis_key, max_object_id in max_object_id_by_axis.items()
        }


class FeatureAnchoredCompactObjectMeasurementRowPolicy(
    CompactMeasuredObjectMeasurementRowPolicy
):
    """Compact rows whose measuredness is anchored by declared feature fields."""

    measured_object_features: ClassVar[tuple[ObjectShapeMeasurementFeature, ...]] = ()

    def row_has_measured_object(
        self,
        row_mapping: Mapping[str, object],
        *,
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> bool:
        del object_id_field, axis_fields
        return any(
            self.measurement_value_is_present(row_mapping.get(feature.value))
            for feature in type(self).measured_object_features
        )


class MeasureObjectSizeShapeObjectMeasurementRowPolicy(
    FeatureAnchoredCompactObjectMeasurementRowPolicy
):
    """Use CP's compact measured-object rows for object size/shape exports."""

    module_name = _MEASURE_OBJECT_SIZE_SHAPE_MODULE
    measured_object_features = (
        ObjectShapeMeasurementFeature.AREA,
        ObjectShapeMeasurementFeature.CENTER_X,
        ObjectShapeMeasurementFeature.CENTER_Y,
    )


class MeasureObjectIntensityDistributionObjectMeasurementRowPolicy(
    CompactMeasuredObjectMeasurementRowPolicy
):
    """Use CP's compact measured-object rows for intensity-distribution exports."""

    module_name = _MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE

    def complete_rows(
        self,
        rows: Sequence[Any],
        *,
        label_payload: Any,
        func: Callable[..., Any],
    ) -> list[Any]:
        """Intensity-distribution functions emit their full dense row domain."""
        del label_payload, func
        return list(rows)


class CellProfilerMeasurementRowIdentityField(str, Enum):
    """CellProfiler row fields that carry ownership/axis identity, not values."""

    SLICE_INDEX = MeasurementRowAxisField.SLICE_INDEX.value
    OBJECT_LABEL = MEASUREMENT_OBJECT_LABEL_FIELD
    OBJECT_NAME = MEASUREMENT_OBJECT_NAME_FIELD
    SOURCE_IMAGE_NAME = MEASUREMENT_SOURCE_IMAGE_NAME_FIELD


class MeasureColocalizationObjectMeasurementRowPolicy(
    CellProfilerObjectMeasurementRowPolicy
):
    """Expand composed source stacks into source-pair object measurements."""

    module_name = _MEASURE_COLOCALIZATION_MODULE
    identity_fields: ClassVar[frozenset[str]] = frozenset(
        field.value for field in CellProfilerMeasurementRowIdentityField
    )

    def invocations(
        self,
        measurement_image: CellProfilerMeasurementImage,
        kwargs: Mapping[str, Any],
    ) -> tuple[ObjectMeasurementInvocation, ...]:
        source_pairs = measurement_image.source_image_pairs()
        if not source_pairs:
            return super().invocations(measurement_image, kwargs)
        return tuple(
            SourcePairObjectMeasurementInvocation(
                kwargs={
                    **kwargs,
                    **source_pair.invocation_kwargs(
                        first_channel_kwarg="channel_1",
                        second_channel_kwarg="channel_2",
                    ),
                },
                source_pair=source_pair,
            )
            for source_pair in source_pairs
        )

    def project_rows(
        self,
        rows: Sequence[Any],
        invocation: ObjectMeasurementInvocation,
    ) -> list[Any]:
        if invocation.source_pair is None:
            return list(rows)
        source_pair_fields = CellProfilerSourcePairFeature.source_field_names()
        runtime_feature_names = (
            CellProfilerSourcePairFeature.runtime_feature_names_for_pair(
                invocation.source_pair
            )
        )
        identity_fields = type(self).identity_fields
        projected_rows: list[Any] = []
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            if not (source_pair_fields & row_mapping.keys()):
                projected_rows.append(dict(row_mapping))
                continue
            projected: dict[str, Any] = {
                field_name: value
                for field_name, value in row_mapping.items()
                if field_name in identity_fields
            }
            for (
                source_field_name,
                runtime_feature_name,
            ) in runtime_feature_names.items():
                if source_field_name not in row_mapping:
                    continue
                projected[runtime_feature_name] = row_mapping[source_field_name]
            projected_rows.append(projected)
        return projected_rows

    def project_row(
        self,
        row: Any,
        source_pair: CellProfilerSourceImagePair,
    ) -> dict[str, Any]:
        """Return one row with CellProfiler source-pair feature names."""
        row_mapping = measurement_row_mapping(row)
        has_pair_features = bool(
            CellProfilerSourcePairFeature.source_field_names() & row_mapping.keys()
        )
        if not has_pair_features:
            return dict(row_mapping)
        projected: dict[str, Any] = {
            field_name: value
            for field_name, value in row_mapping.items()
            if field_name in type(self).identity_fields
        }
        for (
            source_field_name,
            runtime_feature_name,
        ) in CellProfilerSourcePairFeature.runtime_feature_names_for_pair(
            source_pair
        ).items():
            if source_field_name not in row_mapping:
                continue
            projected[runtime_feature_name] = row_mapping[source_field_name]
        return projected

    def table_source_image_name(
        self,
        measurement_images: tuple[CellProfilerMeasurementImage, ...],
        source_image_name: str | None,
    ) -> str | None:
        del source_image_name
        source_pairs = tuple(
            source_pair
            for measurement_image in measurement_images
            for source_pair in measurement_image.source_image_pairs()
        )
        if len(source_pairs) == 1:
            return source_pairs[0].source_image_name
        return None


class TrackObjectsObjectMeasurementRowPolicy(CellProfilerObjectMeasurementRowPolicy):
    """TrackObjects emits object rows and image-level tracking counts together."""

    module_name = _TRACK_OBJECTS_MODULE

    def requires_explicit_row_ownership(self) -> bool:
        return True

    def row_is_object_scoped(self, row: Any) -> bool:
        row_mapping = measurement_row_mapping(row)
        return measurement_row_has_object_identity(row_mapping)

    def image_row_source_image_name(
        self,
        source_image_name: str | None,
    ) -> str | None:
        del source_image_name
        return MeasurementScope.IMAGE.value


for _row_policy_spec in (
    ObjectMeasurementRowPolicySpec(
        class_name=f"{_MEASURE_OBJECT_INTENSITY_MODULE}ObjectMeasurementRowPolicy",
        base_type=DeclaredObjectMeasurementRowPolicy,
        module_name=_MEASURE_OBJECT_INTENSITY_MODULE,
        missing_value_policy=(
            MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
        ),
    ),
):
    _row_policy_spec.declare_in(globals())


class DeclaredSingleObjectLabelInputPolicy(SingleObjectLabelInputPolicy):
    """Generated base for modules with one declared label input."""


_SINGLE_OBJECT_LABEL_INPUT_POLICY_SPECS = (
    SingleObjectLabelInputPolicySpec(
        class_name="CropInputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name="Crop",
        label_kwarg="cropping_labels",
    ),
    SingleObjectLabelInputPolicySpec(
        class_name=f"{_MEASURE_OBJECT_SIZE_SHAPE_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_OBJECT_SIZE_SHAPE_MODULE,
        label_kwarg="labels",
    ),
    SingleObjectLabelInputPolicySpec(
        class_name=f"{_MEASURE_OBJECT_INTENSITY_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_OBJECT_INTENSITY_MODULE,
        label_kwarg="labels",
    ),
    SingleObjectLabelInputPolicySpec(
        class_name=f"{_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE,
        label_kwarg="labels",
    ),
    SingleObjectLabelInputPolicySpec(
        class_name=f"{_MEASURE_TEXTURE_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_TEXTURE_MODULE,
        label_kwarg="labels",
    ),
    SingleObjectLabelInputPolicySpec(
        class_name=f"{_MEASURE_COLOCALIZATION_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_COLOCALIZATION_MODULE,
        label_kwarg="labels",
    ),
    SingleObjectLabelInputPolicySpec(
        class_name=f"{_MEASURE_GRANULARITY_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_GRANULARITY_MODULE,
        label_kwarg="labels",
    ),
)


for _policy_spec in _SINGLE_OBJECT_LABEL_INPUT_POLICY_SPECS:
    _policy_spec.declare_in(globals())


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
                None if same_objects else _label_payload_small_removed(neighbor_payload)
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


class CombineObjectsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind two object-label inputs as the CombineObjects label-pair payload."""

    module_name = "Combineobjects"

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> dict[str, Any]:
        request.require_exact_object_count(2)
        label_planes = tuple(
            np.asarray(request.labels_for(spec), dtype=np.int32)
            for spec in request.object_inputs
        )
        shapes = {tuple(labels.shape) for labels in label_planes}
        if len(shapes) != 1:
            raise ValueError(
                "CombineObjects requires object-label inputs with matching "
                f"shapes, got {sorted(shapes)!r}."
            )
        return {
            _CELLPROFILER_IMAGE_OVERRIDE_KWARG: np.stack(label_planes, axis=0),
            _CELLPROFILER_EXECUTION_MODE_OVERRIDE_KWARG: (
                ImagePayloadExecutionMode.FULL_STACK
            ),
        }


def _filter_objects_child_count_object_names(
    kwargs: Mapping[str, Any],
) -> tuple[str, ...]:
    feature_names = tuple(kwargs.get("measurement_features", ()))
    child_names = tuple(
        parsed.object_name
        for feature_name in feature_names
        for parsed in (CellProfilerMeasurementFeature.parse(str(feature_name)),)
        if (
            parsed is not None
            and parsed.kind is CellProfilerMeasurementFeatureKind.CHILD_COUNT
            and parsed.object_name is not None
        )
    )
    return tuple(dict.fromkeys(child_names))


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
        object_inputs = ArtifactSpecCollection(runtime_inputs).of_kind(
            ArtifactKind.OBJECT_LABELS
        )
        object_count = int(kwargs.get("additional_object_count", 0)) + 1
        enclosing_name = kwargs.get("enclosing_object_name")
        object_specs = object_inputs[:object_count]
        enclosing_spec = None
        relationship_spec = None
        measurement_relationship_specs: list[ArtifactSpec] = []
        if enclosing_name is not None:
            enclosing_spec = ArtifactSpecCollection(object_inputs).by_name(
                str(enclosing_name)
            )
            if enclosing_spec is None:
                raise RuntimeError(
                    "FilterObjects enclosing object input "
                    f"{enclosing_name!r} was not declared in the runtime contract."
                )
            if object_specs:
                relationship_spec = ArtifactSpecCollection(
                    runtime_inputs
                ).by_name_and_kind(
                    parent_child_relationship_artifact_name(
                        str(enclosing_name),
                        object_specs[0].name,
                    ),
                    ArtifactKind.RELATIONSHIPS,
                )
        if object_specs:
            for child_object_name in _filter_objects_child_count_object_names(kwargs):
                relationship = ArtifactSpecCollection(runtime_inputs).by_name_and_kind(
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
            measurement_relationship_specs=ArtifactSpecCollection(
                measurement_relationship_specs
            ).unique(conflict_context="CellProfiler input spec"),
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsMeasurementVectorPlan:
    """Single-feature measurement vector binding for FilterObjects selection."""

    object_spec: ArtifactSpec
    feature_name: str

    @classmethod
    def from_request(
        cls,
        request: ObjectInputBindingRequest,
        plan: FilterObjectsRuntimeInputPlan,
    ) -> "FilterObjectsMeasurementVectorPlan | None":
        if not plan.object_specs:
            return None
        feature_names = tuple(request.kwargs.get("measurement_features", ()))
        if len(feature_names) != 1:
            return None
        measurement_tables = request.measurement_tables_for_primary_object()
        if not measurement_tables:
            return None
        return cls(
            object_spec=plan.object_specs[0],
            feature_name=str(feature_names[0]),
        )

    def bind(self, request: ObjectInputBindingRequest) -> Any:
        labels = request.labels_for(self.object_spec)
        return (
            CellProfilerObjectMeasurementVectorBinding.for_object_input(
                request,
                object_spec=self.object_spec,
                feature_name=self.feature_name,
                labels=labels,
            )
            .vector()
            .slice_aligned_value
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
        measurement_vector_plan = FilterObjectsMeasurementVectorPlan.from_request(
            request.with_object_inputs(plan.object_specs),
            plan,
        )
        if measurement_vector_plan is not None:
            bound["measurement_values"] = measurement_vector_plan.bind(request)
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
        started_at = time.perf_counter()
        operand_bindings = self.object_operand_bindings(request)
        if operand_bindings is not None:
            vectors = CellProfilerObjectMeasurementVectorBatchBinding(
                operand_bindings
            ).vectors()
            _log_module_profile(
                "calculate_math_bind_total",
                time.perf_counter() - started_at,
            )
            return {
                "operand1_value": vectors[0].calculate_math_operand_value,
                "operand2_value": vectors[1].calculate_math_operand_value,
            }

        operand1_started_at = time.perf_counter()
        operand1_value = self.operand_value(
            request,
            feature_kwarg="operand1_feature",
            object_kwarg="operand1_object_name",
        )
        _log_module_profile(
            "calculate_math_operand_bind",
            time.perf_counter() - operand1_started_at,
            operand="1",
        )
        operand2_started_at = time.perf_counter()
        operand2_value = self.operand_value(
            request,
            feature_kwarg="operand2_feature",
            object_kwarg="operand2_object_name",
        )
        _log_module_profile(
            "calculate_math_operand_bind",
            time.perf_counter() - operand2_started_at,
            operand="2",
        )
        _log_module_profile(
            "calculate_math_bind_total",
            time.perf_counter() - started_at,
        )
        return {
            "operand1_value": operand1_value,
            "operand2_value": operand2_value,
        }

    def object_operand_bindings(
        self,
        request: ObjectInputBindingRequest,
    ) -> tuple[CellProfilerObjectMeasurementVectorBinding, ...] | None:
        bindings: list[CellProfilerObjectMeasurementVectorBinding] = []
        for feature_kwarg, object_kwarg in (
            ("operand1_feature", "operand1_object_name"),
            ("operand2_feature", "operand2_object_name"),
        ):
            feature_name = _required_string_kwarg(
                request.kwargs,
                feature_kwarg,
                "CalculateMath",
            )
            object_name = _optional_string_kwarg(request.kwargs, object_kwarg)
            if (
                object_name is None
                or count_feature_object_name(feature_name) is not None
            ):
                return None
            object_spec = ArtifactSpecCollection(request.object_inputs).by_name(
                object_name
            )
            if object_spec is None:
                return None
            bindings.append(
                CellProfilerObjectMeasurementVectorBinding.for_object_input(
                    request,
                    object_spec=object_spec,
                    feature_name=feature_name,
                )
            )
        return tuple(bindings)

    def operand_value(
        self,
        request: ObjectInputBindingRequest,
        *,
        feature_kwarg: str,
        object_kwarg: str,
    ) -> Any:
        feature_name = _required_string_kwarg(
            request.kwargs,
            feature_kwarg,
            "CalculateMath",
        )
        object_name = _optional_string_kwarg(request.kwargs, object_kwarg)
        count_object_name = count_feature_object_name(feature_name)
        if count_object_name is not None:
            return float(_object_label_count(request.adapter, count_object_name))
        if object_name is None:
            return self.image_operand_value(request.adapter, feature_name)

        return (
            CellProfilerObjectMeasurementVectorBinding.for_object_name(
                request,
                object_name=object_name,
                feature_name=feature_name,
            )
            .vector()
            .calculate_math_operand_value
        )

    def image_operand_value(
        self,
        adapter: CellProfilerRuntimeAdapter,
        feature_name: str,
    ) -> Any:
        tables_started_at = time.perf_counter()
        measurement_tables = adapter.measurement_tables(match_group=False)
        _log_module_profile(
            "calculate_math_measurement_tables",
            time.perf_counter() - tables_started_at,
            feature=feature_name,
            count=len(measurement_tables),
        )
        slice_started_at = time.perf_counter()
        slice_values = _calculate_math_image_operand_values_by_slice(
            measurement_tables,
            feature_name,
        )
        _log_module_profile(
            "calculate_math_image_operand_slices",
            time.perf_counter() - slice_started_at,
            feature=feature_name,
            sliced=slice_values is not None,
        )
        if slice_values is None:
            scalar_started_at = time.perf_counter()
            scalar_value = measurement_scalar_value_for_feature(
                measurement_tables,
                feature_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            )
            _log_module_profile(
                "calculate_math_image_operand_scalar",
                time.perf_counter() - scalar_started_at,
                feature=feature_name,
            )
            return scalar_value
        return CellProfilerMeasurementVector(slice_values).slice_aligned_value


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


class MeasurementLabelExecutionModeStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Choose object-measurement execution mode from the label domain shape."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @abstractmethod
    def execution_mode(
        self,
        func: Callable[..., Any],
        labels: object,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        """Return the execution mode required by the supplied labels."""

    @classmethod
    def resolve(
        cls,
        func: Callable[..., Any],
        labels: object,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        if (
            object_label_measurement_execution_from_callable(func)
            is not ObjectLabelMeasurementExecution.FULL_STACK
        ):
            return default
        strategy = cls.for_nominal_value(labels)
        if strategy is None:
            return default
        return strategy.execution_mode(
            func,
            labels,
            default,
            runtime_slice_count=runtime_slice_count,
        )


class DenseArrayMeasurementLabelExecutionModeStrategy(
    MeasurementLabelExecutionModeStrategy
):
    value_type = np.ndarray

    def execution_mode(
        self,
        func: Callable[..., Any],
        labels: object,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        del func, runtime_slice_count
        if not isinstance(labels, np.ndarray):
            raise TypeError("Dense label execution strategy requires ndarray labels.")
        if labels.ndim >= 3 and labels.shape[0] > 1:
            return ImagePayloadExecutionMode.FULL_STACK
        return default


class ObjectLabelValueMeasurementLabelExecutionModeStrategy(
    MeasurementLabelExecutionModeStrategy
):
    value_type = (ObjectLabelPayload, ObjectLabelSet)

    def execution_mode(
        self,
        func: Callable[..., Any],
        labels: object,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        if not isinstance(labels, (ObjectLabelPayload, ObjectLabelSet)):
            raise TypeError(
                "Object-label execution strategy requires an object-label runtime value."
            )
        if ObjectLabelRuntimeSliceStackContract.runtime_slice_count(labels) is not None:
            return ImagePayloadExecutionMode.NATURAL
        return MeasurementLabelExecutionModeStrategy.resolve(
            func,
            ObjectLabelDenseDataStrategy.dense_data(labels),
            default,
        )


class SparseIJVMeasurementLabelExecutionModeStrategy(
    MeasurementLabelExecutionModeStrategy
):
    value_type = SparseIJVLabelRows

    def execution_mode(
        self,
        func: Callable[..., Any],
        labels: object,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        del func, runtime_slice_count
        if not isinstance(labels, SparseIJVLabelRows):
            raise TypeError(
                "Sparse IJV execution strategy requires SparseIJVLabelRows."
            )
        if labels.has_slice_index:
            return ImagePayloadExecutionMode.FULL_STACK
        return default


class CellProfilerObjectMeasurementExecutionDomainPolicy(
    ABC,
    metaclass=CellProfilerModulePolicyMeta,
):
    """Choose CellProfiler object-measurement execution domain by module semantics."""

    @classmethod
    @lru_cache(maxsize=None)
    def for_module(
        cls,
        module_name: str,
    ) -> "CellProfilerObjectMeasurementExecutionDomainPolicy":
        policy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            DefaultObjectMeasurementExecutionDomainPolicy,
        )
        return policy_type()

    @abstractmethod
    def execution_mode(
        self,
        func: Callable[..., Any],
        labels: object,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        """Return the execution mode for one object-measurement invocation."""


class DefaultObjectMeasurementExecutionDomainPolicy(
    CellProfilerObjectMeasurementExecutionDomainPolicy
):
    """Apply the function/domain object-measurement contract uniformly."""

    def execution_mode(
        self,
        func: Callable[..., Any],
        labels: object,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        return MeasurementLabelExecutionModeStrategy.resolve(
            func,
            labels,
            default,
            runtime_slice_count=runtime_slice_count,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerObjectMeasurementLabelArgumentRequest:
    """Typed label-domain context for one object-measurement invocation."""

    dense_labels: Any
    label_payload: Any
    measurement_image_payload: Any


class SliceAlignedLabelArgumentStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Choose the executor-facing label payload for slice-aligned measurements."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def for_request(
        cls,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> "SliceAlignedLabelArgumentStrategy":
        strategy = cls.for_nominal_value(request.measurement_image_payload)
        return (
            strategy
            if strategy is not None
            else DenseSliceAlignedLabelArgumentStrategy()
        )

    @abstractmethod
    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> Any:
        """Return the label value visible to the execution strategy."""


class DenseSliceAlignedLabelArgumentStrategy(SliceAlignedLabelArgumentStrategy):
    """Default slice-aligned functions consume already-projected dense labels."""

    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> Any:
        return request.dense_labels


class AlignedStackSliceAlignedLabelArgumentStrategy(SliceAlignedLabelArgumentStrategy):
    """Defer object-label projection until each aligned image slice is selected."""

    value_type = AlignedImageStack

    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> Any:
        return request.label_payload


class CellProfilerObjectMeasurementLabelArgumentPolicy(
    EnumKeyedStrategyMixin[ObjectLabelMeasurementExecution],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Bind object-measurement labels from the declared callable domain contract."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "execution_mode"

    execution_mode: ClassVar[ObjectLabelMeasurementExecution]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> Any:
        """Return the labels object passed to the absorbed measurement function."""


class SliceAlignedObjectMeasurementLabelArgumentPolicy(
    CellProfilerObjectMeasurementLabelArgumentPolicy
):
    """Slice-aligned measurement functions consume the dense execution plane."""

    execution_mode = ObjectLabelMeasurementExecution.SLICE_ALIGNED

    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> Any:
        return SliceAlignedLabelArgumentStrategy.for_request(request).label_argument(
            request
        )


class FullStackObjectMeasurementLabelArgumentPolicy(
    CellProfilerObjectMeasurementLabelArgumentPolicy
):
    """Full-stack measurement functions consume labels with semantic domains."""

    execution_mode = ObjectLabelMeasurementExecution.FULL_STACK

    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> Any:
        return request.label_payload


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
    source_image_names: tuple[str, ...] = ()
    source_image_payload: Any | None = None

    def single_output_object_name(self) -> str:
        """Return the unique object-label output owned by this record request."""
        object_outputs = ArtifactSpecCollection(self.executor.outputs).of_kind(
            ArtifactKind.OBJECT_LABELS
        )
        if len(object_outputs) != 1:
            raise NotImplementedError(
                f"{self.executor.module_name} threshold measurement semantics "
                f"require exactly one object-label output, got "
                f"{[spec.name for spec in object_outputs]}."
            )
        return object_outputs[0].name


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementRecord:
    """Rows and semantic owner for one CellProfiler measurement output."""

    rows: list[Any]
    object_name: str | None
    source_image_name: str | None
    source_image_payload: Any | None = None
    fields: tuple[FieldSpec, ...] = ()
    owns_source_qualified_features: bool = False

    def __post_init__(self) -> None:
        if self.fields or not _rows_have_inferable_fields(self.rows):
            return
        object.__setattr__(
            self,
            "fields",
            _field_specs_for_rows(
                tuple(measurement_row_mapping(row) for row in self.rows)
            ),
        )

    @classmethod
    def shared_source_image_name(
        cls,
        records: tuple["CellProfilerMeasurementRecord", ...],
    ) -> str | None:
        """Return a table source only when every record declares the same one."""
        if (
            not records
            or any(record.owns_source_qualified_features for record in records)
            or any(record.source_image_name is None for record in records)
        ):
            return None
        unique_names = tuple(
            dict.fromkeys(record.source_image_name for record in records)
        )
        if len(unique_names) == 1:
            return unique_names[0]
        return None


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementRecordPartition:
    """One measurement table with coherent table and row-level ownership."""

    rows: list[Any]
    object_name: str | None
    source_image_name: str | None
    source_image_payload: Any | None = None
    fields: tuple[FieldSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.fields or not _rows_have_inferable_fields(self.rows):
            return
        object.__setattr__(
            self,
            "fields",
            _field_specs_for_rows(
                tuple(measurement_row_mapping(row) for row in self.rows)
            ),
        )


class CellProfilerImageMeasurementSource(ABC, metaclass=AutoRegisterMeta):
    """Nominal source for image-owned measurement row identity."""

    __registry_key__ = "__name__"

    @abstractmethod
    def source_image_name(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> str | None:
        """Return the image name used to qualify recorded measurement rows."""

    @abstractmethod
    def source_image_payload(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> Any | None:
        """Return the payload that anchors image-owned measurement rows."""

    def require_produced_artifact(self) -> "ProducedArtifactImageMeasurementSource":
        """Return this source as a produced artifact source, or fail loudly."""
        raise ValueError("Measurement ownership requires an image output.")


class ProducedArtifactImageMeasurementSourceBase(CellProfilerImageMeasurementSource):
    """Measurement source owned by a produced image artifact."""

    artifact_spec: ArtifactSpec

    def source_image_name(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> str | None:
        del request
        return self.artifact_spec.name

    def source_image_payload(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> Any | None:
        return request.output_values.get(
            self.artifact_spec.name,
            request.source_image_payload,
        )

    def require_produced_artifact(self) -> "ProducedArtifactImageMeasurementSource":
        return self


ProducedArtifactImageMeasurementSource = product_record(
    "ProducedArtifactImageMeasurementSource",
    "artifact_spec: ArtifactSpec",
    bases=(ProducedArtifactImageMeasurementSourceBase,),
    doc="Measurement source owned by a produced image artifact.",
    module_name=__name__,
)


class UnqualifiedRuntimeImageMeasurementSource(CellProfilerImageMeasurementSource):
    """Measurement source backed by the runtime input image without row naming."""

    def source_image_name(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> str | None:
        del request
        return None

    def source_image_payload(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> Any | None:
        return request.source_image_payload


class CellProfilerMeasurementImagePlaneCountStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal execution-plane count contract for measurement images."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True
    value_type: ClassVar[type[Any] | None] = None
    value_type_label: ClassVar[str | None] = None

    @classmethod
    def plane_count_for_payload(cls, payload: Any) -> int | None:
        """Return the semantic object-measurement plane count for ``payload``."""
        for value_type in type(payload).__mro__:
            for strategy_type in cls.registered_strategy_types():
                if strategy_type.value_type is value_type:
                    return strategy_type().plane_count(payload)
        return None

    @abstractmethod
    def plane_count(self, payload: Any) -> int | None:
        """Return the number of object-label planes this measurement image owns."""


class ArrayMeasurementImagePlaneCountStrategy(
    CellProfilerMeasurementImagePlaneCountStrategy
):
    """Dense image arrays expose planes through their CellProfiler image layout."""

    value_type = np.ndarray

    def plane_count(self, payload: Any) -> int | None:
        if not isinstance(payload, np.ndarray) or payload.ndim < 3:
            return None
        if is_color_image_slice(payload):
            return None
        if is_color_image_stack(payload):
            return int(payload.shape[0])
        return int(payload.reshape((-1, *payload.shape[-2:])).shape[0])


class ImageMetadataPayloadMeasurementImagePlaneCountStrategy(
    ArrayMeasurementImagePlaneCountStrategy
):
    """Image-metadata payloads derive measurement planes from their data."""

    value_type = ImageMetadataPayload

    def plane_count(self, payload: Any) -> int | None:
        return super().plane_count(image_payload_data(payload))


class MaskedImagePayloadMeasurementImagePlaneCountStrategy(
    ImageMetadataPayloadMeasurementImagePlaneCountStrategy
):
    """Masked image payloads derive measurement planes from their image data."""

    value_type = MaskedImagePayload


class AlignedImageStackMeasurementImagePlaneCountStrategy(
    CellProfilerMeasurementImagePlaneCountStrategy
):
    """Aligned multi-image stacks execute once per aligned slice."""

    value_type = AlignedImageStack

    def plane_count(self, payload: Any) -> int | None:
        if not isinstance(payload, AlignedImageStack):
            raise TypeError(
                "Aligned measurement-image plane counting requires "
                f"AlignedImageStack, got {type(payload).__name__}."
            )
        return len(payload.slices)


@dataclass(frozen=True, slots=True)
class ObjectLabelSourceBindingProjectionRequest:
    """Context for projecting object labels into a source-image-bound plane."""

    labels: Any
    label_payload: ObjectLabelSet
    measurement_image: CellProfilerMeasurementImage
    adapter: RuntimePlaneAxisProjector | None
    source_image_name: str | None
    source_image_names: tuple[str, ...]

    @property
    def label_plane_count(self) -> int | None:
        if not isinstance(self.labels, np.ndarray) or self.labels.ndim < 3:
            return None
        return int(self.labels.shape[0])

    @property
    def measurement_image_plane_count(self) -> int | None:
        return CellProfilerMeasurementImagePlaneCountStrategy.plane_count_for_payload(
            self.measurement_image.payload
        )

    @property
    def source_aliases(self) -> tuple[str, ...]:
        return self.source_image_names or (
            (self.source_image_name,) if self.source_image_name is not None else ()
        )

    def label_origin_plane_index(self) -> int | None:
        if self.label_payload.plane_axis is not RuntimePlaneAxis.SOURCE_BINDING:
            return None
        if self.source_image_name is None:
            return None
        if self.source_image_name not in self.source_image_names:
            return None
        if self.label_plane_count != len(self.source_image_names):
            return None
        return self.source_image_names.index(self.source_image_name)

    def current_source_binding_plane_index(self) -> int | None:
        if self.label_payload.plane_axis is not RuntimePlaneAxis.SOURCE_BINDING:
            return None
        return self.current_axis_plane_index(RuntimePlaneAxis.SOURCE_BINDING)

    def current_runtime_slice_plane_index(self) -> int | None:
        if self.label_payload.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return None
        return self.current_axis_plane_index(RuntimePlaneAxis.RUNTIME_SLICE)

    def runtime_slice_projection_plane_index(self) -> int | None:
        """Return the current runtime-slice plane only when projection is requested."""
        if self.label_payload.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return None
        if self.label_plane_count == self.measurement_image_plane_count:
            return None
        if (
            self.measurement_image.reference_domain
            is CellProfilerMeasurementImageDomain.OBJECT_LABELS
            and self.measurement_image.source_image_name is None
        ):
            return None
        return self.current_axis_plane_index(RuntimePlaneAxis.RUNTIME_SLICE)

    def current_axis_plane_index(self, axis: RuntimePlaneAxis) -> int | None:
        if self.adapter is None:
            return None
        if not isinstance(self.adapter, RuntimePlaneAxisProjector):
            raise TypeError(
                "Object-label plane projection requires a RuntimePlaneAxisProjector, "
                f"got {type(self.adapter).__name__}."
            )
        return self.adapter.runtime_plane_index(
            axis,
            source_aliases=self.source_aliases,
        )

    def project_plane(self, plane_index: int) -> Any:
        if not isinstance(self.labels, np.ndarray):
            return self.labels
        if self.labels.ndim < 3:
            return self.labels
        if plane_index < 0 or plane_index >= self.labels.shape[0]:
            raise RuntimeError(
                "Object-label source plane resolution produced an out-of-range "
                f"plane index {plane_index} for label shape {self.labels.shape!r}."
            )
        return self.labels[plane_index]


class ObjectLabelSourceBindingProjectionStrategy(
    EnumKeyedStrategyMixin[ObjectLabelDomainScope],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project object labels by source binding only for plane-scoped domains."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "scope"
    scope: ClassVar[ObjectLabelDomainScope]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def project(
        self,
        request: ObjectLabelSourceBindingProjectionRequest,
    ) -> Any:
        """Return labels projected into the requested source-image plane."""


class PayloadObjectLabelSourceBindingProjectionStrategy(
    ObjectLabelSourceBindingProjectionStrategy
):
    """Payload-scoped dense labels already describe the whole object domain."""

    scope = ObjectLabelDomainScope.PAYLOAD

    def project(
        self,
        request: ObjectLabelSourceBindingProjectionRequest,
    ) -> Any:
        return request.labels


class PlaneObjectLabelSourceBindingProjectionStrategy(
    ObjectLabelSourceBindingProjectionStrategy
):
    """Plane-scoped labels can be selected by source image binding."""

    scope = ObjectLabelDomainScope.PLANE

    def project(
        self,
        request: ObjectLabelSourceBindingProjectionRequest,
    ) -> Any:
        runtime_slice_plane_index = request.runtime_slice_projection_plane_index()
        if runtime_slice_plane_index is not None:
            return request.project_plane(runtime_slice_plane_index)
        origin_plane_index = request.label_origin_plane_index()
        if origin_plane_index is not None:
            return request.project_plane(origin_plane_index)
        current_plane_index = request.current_source_binding_plane_index()
        if current_plane_index is not None:
            return request.project_plane(current_plane_index)
        return request.labels


class CellProfilerMeasurementRecordBuilder(
    ABC,
    metaclass=CellProfilerModulePolicyMeta,
):
    """Nominal module-specific measurement-row enrichment."""

    @classmethod
    @lru_cache(maxsize=None)
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
        rows = [
            *_measurement_table_rows(request.value),
            *RelationshipMeasurementRows.for_request(request).rows(),
        ]
        rows_declare_object_name = _measurement_rows_declare_object_name(rows)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=(
                None
                if rows_declare_object_name
                else _measurement_object_name(request.executor._declared_input_specs())
            ),
            source_image_name=(
                None
                if rows_declare_object_name
                else (
                    request.source_image_name
                    or _measurement_source_name_for_specs(
                        request.executor._primary_image_inputs(request.func)
                    )
                )
            ),
            fields=_measurement_record_fields(request.spec, rows, request.func),
        )


class SourcePairMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Project source-pair result fields through the CellProfiler pair dialect."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        source_rows = _measurement_table_rows(request.value)
        owns_source_qualified_features = self.rows_have_source_pair_features(
            source_rows
        )
        rows = self.project_rows(source_rows, request)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            source_image_name=(
                None if owns_source_qualified_features else request.source_image_name
            ),
            fields=_measurement_record_fields(request.spec, rows, request.func),
            owns_source_qualified_features=owns_source_qualified_features,
        )

    def rows_have_source_pair_features(
        self,
        rows: Sequence[Any],
    ) -> bool:
        """Return whether rows carry fields owned by source-pair feature policies."""
        source_pair_fields = CellProfilerSourcePairFeature.source_field_names()
        return any(
            bool(source_pair_fields & measurement_row_mapping(row).keys())
            for row in rows
        )

    def project_rows(
        self,
        rows: Sequence[Any],
        request: CellProfilerOutputRecordRequest,
    ) -> list[Any]:
        source_pair = self.source_pair(request)
        if source_pair is None:
            return list(rows)
        return [self.project_row(row, source_pair) for row in rows]

    def source_pair(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerSourceImagePair | None:
        if len(request.source_image_names) == 2:
            first_name, second_name = request.source_image_names
            return CellProfilerSourceImagePair(
                first_index=0,
                second_index=1,
                runtime_pair=RuntimeMeasurementSourcePair(first_name, second_name),
                first_display_name=first_name,
                second_display_name=second_name,
            )
        return CellProfilerSourceImagePair.from_source_image_name(
            request.source_image_name
        )

    def project_row(
        self,
        row: Any,
        source_pair: CellProfilerSourceImagePair,
    ) -> dict[str, Any]:
        row_mapping = measurement_row_mapping(row)
        if not (
            CellProfilerSourcePairFeature.source_field_names() & row_mapping.keys()
        ):
            return dict(row_mapping)

        projected: dict[str, Any] = {}
        for field_name, value in row_mapping.items():
            if field_name not in CellProfilerSourcePairFeature.source_field_names():
                projected[field_name] = value

        for (
            source_field_name,
            runtime_feature_name,
        ) in CellProfilerSourcePairFeature.runtime_feature_names_for_pair(
            source_pair
        ).items():
            if source_field_name not in row_mapping:
                continue
            projected[runtime_feature_name] = row_mapping[source_field_name]
        return projected


class MeasureColocalizationMeasurementRecordBuilder(SourcePairMeasurementRecordBuilder):
    """CellProfiler MeasureColocalization emits source-pair measurement fields."""

    module_name = _MEASURE_COLOCALIZATION_MODULE


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


class ProducedImageMeasurementRecordBuilder(CellProfilerMeasurementRecordBuilder):
    """Base for diagnostics whose semantic owner is the produced image artifact."""

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        rows = _measurement_table_rows(request.value)
        source_image = self._primary_image_measurement_source(request)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            source_image_name=source_image.source_image_name(request),
            source_image_payload=source_image.source_image_payload(request),
            fields=_measurement_record_fields(request.spec, rows, request.func),
        )

    def _primary_image_measurement_source(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerImageMeasurementSource:
        image_specs = tuple(
            spec
            for spec in request.executor.contract.declared_outputs
            if spec.kind is ArtifactKind.IMAGE and spec.sidecar_role is None
        )
        if not image_specs:
            return UnqualifiedRuntimeImageMeasurementSource()
        if len(image_specs) == 1:
            return ProducedArtifactImageMeasurementSource(image_specs[0])
        retained_image_names = {
            name
            for name, value in request.output_values.items()
            if CellProfilerImagePayloadOutputTypes.owns(value)
        }
        retained_specs = tuple(
            spec for spec in image_specs if spec.name in retained_image_names
        )
        if len(retained_specs) == 1:
            return ProducedArtifactImageMeasurementSource(retained_specs[0])
        raise ValueError(
            "Produced-image measurement ownership requires exactly one primary image "
            f"output spec, got {[spec.name for spec in image_specs]!r}."
        )

    def _required_primary_image_measurement_source(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> ProducedArtifactImageMeasurementSource:
        return self._primary_image_measurement_source(
            request
        ).require_produced_artifact()


class CropMeasurementRecordBuilder(ProducedImageMeasurementRecordBuilder):
    """Crop measurements describe the produced crop image artifact."""

    module_name = "Crop"


class ThresholdMeasurementRecordBuilder(ProducedImageMeasurementRecordBuilder):
    """Threshold diagnostics describe the produced binary image artifact."""

    module_name = "Threshold"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        source_image = self._required_primary_image_measurement_source(request)
        return CellProfilerMeasurementRecord(
            rows=ThresholdMeasurementRows(
                request.value,
                object_name=source_image.artifact_spec.name,
            ).rows(),
            object_name=None,
            source_image_name=None,
            source_image_payload=source_image.source_image_payload(request),
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
            for spec in ArtifactSpecCollection(
                request.executor.contract.declared_outputs
            ).of_kind(ArtifactKind.IMAGE)
        )
        return CellProfilerMeasurementRecord(
            rows=AlignMeasurementRows(request.value, output_names=output_names).rows(),
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
        table_started_at = time.perf_counter()
        table_rows = _measurement_table_rows(request.value)
        _log_module_profile(
            "relate_measurement_table_rows",
            time.perf_counter() - table_started_at,
            module=self.module_name,
            rows=len(table_rows),
        )
        relationship_started_at = time.perf_counter()
        relationship_rows = RelationshipMeasurementRows.for_request(request).rows()
        _log_module_profile(
            "relate_relationship_rows",
            time.perf_counter() - relationship_started_at,
            module=self.module_name,
            rows=len(relationship_rows),
        )
        return CellProfilerMeasurementRecord(
            rows=[
                *table_rows,
                *relationship_rows,
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
                *ThresholdMeasurementRows(
                    request.value,
                    object_name=request.single_output_object_name(),
                ).rows(),
                *RelationshipMeasurementRows.for_request(request).rows(),
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
            rows=ClassifyObjectsMeasurementRows(
                request.value,
                object_name=object_name,
            ).rows(),
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
        object_name = request.single_output_object_name()
        return CellProfilerMeasurementRecord(
            rows=ThresholdMeasurementRows(
                request.value,
                object_name=object_name,
            ).rows(),
            object_name=None,
            source_image_name=None,
        )


class IdentifyObjectsInGridMeasurementRecordBuilder(
    CellProfilerMeasurementRecordBuilder
):
    """Expose grid object location facts emitted by CellProfiler object creation."""

    module_name = "IdentifyObjectsInGrid"

    def build(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerMeasurementRecord:
        object_name = request.single_output_object_name()
        rows = [
            *_measurement_table_rows(request.value),
            *ObjectLocationMeasurementRows(
                request.output_values[object_name],
                object_name=object_name,
            ).rows(),
        ]
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            source_image_name=None,
            fields=_measurement_record_fields(request.spec, rows, request.func),
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
            rows=RelationshipMeasurementRows.for_request(request).rows(),
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
        row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
            request.executor.module_name
        )
        rows = row_policy.annotate_record_rows(
            _measurement_table_rows(request.value),
            object_name=_measurement_object_name(
                request.executor._declared_input_specs()
            ),
            source_image_name=request.source_image_name or MeasurementScope.IMAGE.value,
        )
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=None,
            source_image_name=None,
            fields=_field_specs_for_rows(rows),
        )


class CellProfilerImageOutputContextStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Attach runtime image context to declared image outputs."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True
    value_type: ClassVar[type[Any] | None] = None

    @classmethod
    def for_value(cls, value: Any) -> "CellProfilerImageOutputContextStrategy":
        strategy = cls.for_nominal_value(value)
        if strategy is not None:
            return strategy
        raise TypeError(
            "CellProfiler image outputs must be image payloads or numpy arrays; "
            f"got {type(value).__name__}."
        )

    @abstractmethod
    def runtime_image_value(self, value: Any, source_image_payload: Any) -> Any:
        """Return the output in OpenHCS runtime image-payload form."""


class ContextualCellProfilerImageOutputStrategy(CellProfilerImageOutputContextStrategy):
    """Preserve outputs that already carry OpenHCS image context."""

    value_type = (ImageMetadataPayload, MaskedImagePayload)

    def runtime_image_value(self, value: Any, source_image_payload: Any) -> Any:
        del source_image_payload
        if not isinstance(value, (ImageMetadataPayload, MaskedImagePayload)):
            raise TypeError(
                "Contextual image output strategy requires an OpenHCS image payload."
            )
        return value


class NumpyCellProfilerImageOutputStrategy(CellProfilerImageOutputContextStrategy):
    """Attach source image context to raw CellProfiler numpy image outputs."""

    value_type = np.ndarray

    def runtime_image_value(self, value: Any, source_image_payload: Any) -> Any:
        if not isinstance(value, np.ndarray):
            raise TypeError("Numpy image output strategy requires numpy.ndarray.")
        return with_derived_image_payload_data(
            source_image_payload,
            _collapse_singleton_stack_output(value),
        )


class CellProfilerOutputRecorder(ABC, metaclass=AutoRegisterMeta):
    """Nominal output writer selected by artifact kind."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    kind: ClassVar[ArtifactKind | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_kind(cls, kind: ArtifactKind) -> "CellProfilerOutputRecorder":
        recorder_type = cls.__registry__.get(kind)
        if recorder_type is None:
            raise TypeError(f"Unsupported CellProfiler output kind {kind.value}.")
        return recorder_type()

    @classmethod
    def recording_dependency_depth(cls) -> int:
        """Return dependency order from the recorder inheritance chain."""
        return cls.mro().index(CellProfilerOutputRecorder)

    @abstractmethod
    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        """Record one output artifact through the runtime adapter."""


class ImmediateOutputRecorder(CellProfilerOutputRecorder):
    """Recorder family for artifacts that create no later recording dependency."""


class RelationshipDependentOutputRecorder(ImmediateOutputRecorder):
    """Recorder family for artifacts that require object endpoints to exist."""


class MeasurementDependentOutputRecorder(RelationshipDependentOutputRecorder):
    """Recorder family for artifacts that may require prior relationships."""


class ImageOutputRecorder(ImmediateOutputRecorder):
    """Record image outputs."""

    kind = ArtifactKind.IMAGE

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        value = CellProfilerImageOutputContextStrategy.for_value(
            request.value
        ).runtime_image_value(
            request.value,
            request.source_image_payload,
        )
        request.adapter.add_image(
            request.spec.name,
            value,
            source_image_name=request.source_image_name,
        )


class ObjectLabelsOutputRecorder(ImmediateOutputRecorder):
    """Record object-label outputs."""

    kind = ArtifactKind.OBJECT_LABELS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_objects(
            request.spec.name,
            request.value,
            source_image_name=request.source_image_name,
        )


class MeasurementsOutputRecorder(MeasurementDependentOutputRecorder):
    """Record measurement outputs with inferred image/object ownership."""

    kind = ArtifactKind.MEASUREMENTS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        measurement_record = CellProfilerMeasurementRecordBuilder.for_module(
            request.executor.module_name
        ).build(request)
        row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
            request.executor.module_name
        )
        for partition in row_policy.record_partitions(measurement_record):
            _record_measurements(
                request.adapter,
                request.spec.name,
                partition.rows,
                fields=partition.fields,
                object_name=partition.object_name,
                source_image_name=partition.source_image_name,
                source_image_payload=partition.source_image_payload,
            )


class RelationshipsOutputRecorder(RelationshipDependentOutputRecorder):
    """Record parent-child relationship artifacts."""

    kind = ArtifactKind.RELATIONSHIPS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        if not isinstance(request.value, ParentChildRelationshipPayload):
            raise TypeError(
                f"{request.executor.module_name} relationship output "
                f"'{request.spec.name}' must be ParentChildRelationshipPayload, "
                f"got {type(request.value).__name__}."
            )
        parent_spec, child_spec = RelationshipEndpointResolver(request).endpoint_specs(
            request.spec
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


class SpatialGridOutputRecorder(ImmediateOutputRecorder):
    """Record spatial-grid outputs."""

    kind = ArtifactKind.SPATIAL_GRID

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_spatial_grid(
            request.spec.name,
            _coerce_spatial_grid(request.value, request.spec.name),
        )


def _output_recording_order(
    output_specs: tuple[ArtifactSpec, ...],
) -> tuple[ArtifactSpec, ...]:
    return tuple(
        sorted(
            output_specs,
            key=lambda spec: type(
                CellProfilerOutputRecorder.for_kind(spec.kind)
            ).recording_dependency_depth(),
        )
    )


def _output_values_by_kind(
    output_specs: tuple[ArtifactSpec, ...],
    main_output: Any,
    artifact_values: tuple[Any, ...],
    *,
    func: Callable[..., Any] | None = None,
    declared_output_specs: tuple[ArtifactSpec, ...] = (),
) -> dict[str, Any]:
    resolved = RuntimeReturnedOutputMatcher(
        retained_specs=output_specs,
        declared_specs=declared_output_specs,
        main_output=main_output,
        artifact_values=artifact_values,
        returned_specs=CellProfilerCallableOutputSpecs(func).artifact_specs(),
    ).resolve()
    if resolved is not None:
        return resolved
    raise ValueError(
        f"CellProfiler module declared {len(output_specs)} outputs but "
        f"returned {len(artifact_values)} artifact values."
    )


@dataclass(frozen=True, slots=True)
class CellProfilerOutputValueResolution:
    """Returned values split into recorded outputs and full semantic context."""

    recorded_values: Mapping[str, Any]
    context_values: Mapping[str, Any]

    @classmethod
    def from_returned_values(
        cls,
        retained_specs: tuple[ArtifactSpec, ...],
        *,
        declared_specs: tuple[ArtifactSpec, ...],
        main_output: Any,
        artifact_values: tuple[Any, ...],
        func: Callable[..., Any] | None,
    ) -> "CellProfilerOutputValueResolution":
        recorded_values = _output_values_by_kind(
            retained_specs,
            main_output,
            artifact_values,
            func=func,
            declared_output_specs=declared_specs,
        )
        context_values = _output_values_by_kind(
            declared_specs or retained_specs,
            main_output,
            artifact_values,
            func=func,
            declared_output_specs=declared_specs,
        )
        return cls(recorded_values, context_values)


@dataclass(frozen=True, slots=True)
class CellProfilerCallableOutputSpecs:
    """CellProfiler special-output declarations lowered to artifact specs."""

    func: Callable[..., Any] | None = None

    def artifact_specs(self) -> tuple[ArtifactSpec, ...]:
        if self.func is None:
            return ()
        raw_outputs = self.callable_special_outputs(self.func)
        return tuple(
            ArtifactSpec(
                special_output_name(output_spec),
                SpecialOutputKindClassifier.kind_for(output_spec),
            )
            for output_spec in raw_outputs
        )

    @classmethod
    def callable_returned_output_specs(
        cls,
        func: Callable[..., Any] | None,
    ) -> tuple[ArtifactSpec, ...]:
        return cls(func).artifact_specs()

    @staticmethod
    def callable_special_outputs(func: Callable[..., Any]) -> tuple[object, ...]:
        contract = CallableContract.from_callable(func)
        candidates: list[Any] = [func]
        raw_func = contract.raw_processing_function
        if callable(raw_func):
            candidates.append(raw_func)
        unwrapped = unwrap(func)
        if unwrapped not in candidates:
            candidates.append(unwrapped)
        for candidate in candidates:
            raw_outputs = vars(candidate).get("__special_outputs__", ())
            if isinstance(raw_outputs, tuple) and raw_outputs:
                return raw_outputs
        return ()


class ClassifyObjectsMeasurementStatField(str, Enum):
    """Source fields emitted by absorbed ClassifyObjects functions."""

    BIN_COUNTS = "bin_counts"
    BIN_PERCENTAGES = "bin_percentages"
    OBJECT_CLASSES = "object_classes"
    TOTAL_OBJECTS = "total_objects"
    SLICE_INDEX = MeasurementRowAxisField.SLICE_INDEX.value


class FormattingMeasurementFeatureTemplate(str, Enum, metaclass=RegisteredEnumMeta):
    """Shared feature-name formatting contract for templated measurement names."""

    __registry_key__ = "__name__"

    def feature_name(self, **values: object) -> str:
        return self.value.format(**values)


class ClassifyObjectsMeasurementFeatureTemplate(FormattingMeasurementFeatureTemplate):
    """CellProfiler ClassifyObjects feature-name templates."""

    OBJECTS_PER_BIN = "Classify_{bin_name}_NumObjectsPerBin"
    PERCENT_PER_BIN = "Classify_{bin_name}_PctObjectsPerBin"
    OBJECT_CLASS = "Classify_{bin_name}"


class AlignMeasurementStatField(str, Enum):
    """Source fields emitted by absorbed Align functions."""

    OUTPUT_INDEX = "output_index"
    SLICE_INDEX = MeasurementRowAxisField.SLICE_INDEX.value
    X_SHIFT = "x_shift"
    Y_SHIFT = "y_shift"


class AlignMeasurementFeature(str, Enum):
    """CellProfiler Align feature names."""

    X_SHIFT = "Align_Xshift"
    Y_SHIFT = "Align_Yshift"

    @property
    def source_field(self) -> AlignMeasurementStatField:
        return AlignMeasurementStatField[self.name]


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementRows(ABC, metaclass=AutoRegisterMeta):
    """Base contract for emitted CellProfiler measurement fact rows."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None

    @abstractmethod
    def rows(self) -> list[dict[str, Any]]:
        """Return long/tall measurement rows."""


@dataclass(frozen=True, slots=True)
class CellProfilerResultMeasurementRows(CellProfilerMeasurementRows):
    """Measurement rows projected from absorbed function result records."""

    results: Any

    def source_rows(self) -> list[Any]:
        return _measurement_table_rows(self.results)

    @staticmethod
    def row_value(
        row: Any,
        field: Enum | str,
        default: Any,
    ) -> Any:
        field_name = field.value if isinstance(field, Enum) else field
        return measurement_row_mapping(row).get(str(field_name), default)

    @staticmethod
    def json_object_mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        if value in (None, ""):
            return {}
        parsed = json.loads(str(value))
        if not isinstance(parsed, Mapping):
            raise TypeError(
                f"Expected JSON object mapping, got {type(parsed).__name__}."
            )
        return parsed


@dataclass(frozen=True, slots=True)
class ClassifyObjectsMeasurementRows(CellProfilerResultMeasurementRows):
    """Project absorbed ClassifyObjects results into CP measurement rows."""

    registry_key = "classify_objects"

    object_name: str | None

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in self.source_rows():
            bin_counts = self.json_object_mapping(
                self.row_value(
                    result, ClassifyObjectsMeasurementStatField.BIN_COUNTS, {}
                )
            )
            bin_percentages = self.json_object_mapping(
                self.row_value(
                    result,
                    ClassifyObjectsMeasurementStatField.BIN_PERCENTAGES,
                    {},
                )
            )
            object_classes = self.json_object_mapping(
                self.row_value(
                    result,
                    ClassifyObjectsMeasurementStatField.OBJECT_CLASSES,
                    {},
                )
            )
            slice_index = int(
                self.row_value(
                    result, ClassifyObjectsMeasurementStatField.SLICE_INDEX, 0
                )
            )
            bin_names = tuple(str(name) for name in bin_counts)
            for bin_name, count in bin_counts.items():
                rows.append(
                    {
                        MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                        MEASUREMENT_FEATURE_NAME_FIELD: (
                            ClassifyObjectsMeasurementFeatureTemplate.OBJECTS_PER_BIN.feature_name(
                                bin_name=str(bin_name)
                            )
                        ),
                        MEASUREMENT_RESULT_VALUE_FIELD: count,
                    }
                )
                rows.append(
                    {
                        MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                        MEASUREMENT_FEATURE_NAME_FIELD: (
                            ClassifyObjectsMeasurementFeatureTemplate.PERCENT_PER_BIN.feature_name(
                                bin_name=str(bin_name)
                            )
                        ),
                        MEASUREMENT_RESULT_VALUE_FIELD: bin_percentages.get(
                            bin_name,
                            0.0,
                        ),
                    }
                )
            rows.extend(
                self.object_class_rows(
                    object_classes=object_classes,
                    bin_names=bin_names,
                    result=result,
                    slice_index=slice_index,
                )
            )
        return rows

    def object_class_rows(
        self,
        *,
        object_classes: Mapping[str, Any],
        bin_names: tuple[str, ...],
        result: Any,
        slice_index: int,
    ) -> list[dict[str, Any]]:
        if self.object_name is None:
            return []
        total_objects = int(
            self.row_value(
                result,
                ClassifyObjectsMeasurementStatField.TOTAL_OBJECTS,
                0,
            )
        )
        class_labels = tuple(sorted(int(label) for label in object_classes))
        dense_labels = tuple(range(1, total_objects + 1))
        object_labels = tuple(dict.fromkeys((*dense_labels, *class_labels)))
        return [
            {
                MEASUREMENT_OBJECT_NAME_FIELD: self.object_name,
                MEASUREMENT_OBJECT_LABEL_FIELD: object_label,
                MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                MEASUREMENT_FEATURE_NAME_FIELD: (
                    ClassifyObjectsMeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                        bin_name=bin_name
                    )
                ),
                MEASUREMENT_RESULT_VALUE_FIELD: int(
                    object_classes.get(str(object_label)) == bin_name
                ),
            }
            for object_label in object_labels
            for bin_name in bin_names
        ]


@dataclass(frozen=True, slots=True)
class AlignMeasurementRows(CellProfilerResultMeasurementRows):
    """Project absorbed Align results into CP image measurement rows."""

    registry_key = "align"

    output_names: tuple[str, ...]
    features: ClassVar[tuple[AlignMeasurementFeature, ...]] = (
        AlignMeasurementFeature.X_SHIFT,
        AlignMeasurementFeature.Y_SHIFT,
    )

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in self.source_rows():
            output_index = int(
                self.row_value(result, AlignMeasurementStatField.OUTPUT_INDEX, 0)
            )
            if output_index < 0 or output_index >= len(self.output_names):
                raise ValueError(
                    f"Align measurement output_index {output_index} does not match "
                    f"declared image outputs {self.output_names!r}."
                )
            slice_index = int(
                self.row_value(result, AlignMeasurementStatField.SLICE_INDEX, 0)
            )
            source_image_name = self.output_names[output_index]
            rows.extend(
                {
                    MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD: source_image_name,
                    MEASUREMENT_FEATURE_NAME_FIELD: feature.value,
                    MEASUREMENT_RESULT_VALUE_FIELD: float(
                        self.row_value(result, feature.source_field, 0.0)
                    ),
                }
                for feature in type(self).features
            )
        return rows


class ThresholdMeasurementStatField(str, Enum):
    """Known source fields for CellProfiler threshold measurement stats."""

    SLICE_INDEX = "slice_index"
    THRESHOLD_USED = "threshold_used"
    THRESHOLD_VALUE = "threshold_value"
    FINAL_THRESHOLD = "final_threshold"
    ORIGINAL_THRESHOLD = "original_threshold"
    WEIGHTED_VARIANCE = "weighted_variance"
    SUM_OF_ENTROPIES = "sum_of_entropies"


class ThresholdMeasurementFeatureTemplate(FormattingMeasurementFeatureTemplate):
    """CellProfiler image-measurement feature names parameterized by object."""

    FINAL_THRESHOLD = "FinalThreshold_{object_name}"
    ORIGINAL_THRESHOLD = "OrigThreshold_{object_name}"
    WEIGHTED_VARIANCE = "WeightedVariance_{object_name}"
    SUM_OF_ENTROPIES = "SumOfEntropies_{object_name}"


@dataclass(frozen=True, slots=True)
class ThresholdMeasurementStatSchema:
    """Nominal mapping from supported threshold stat rows to CP features."""

    final_threshold_fields: tuple[ThresholdMeasurementStatField, ...] = (
        ThresholdMeasurementStatField.THRESHOLD_USED,
        ThresholdMeasurementStatField.THRESHOLD_VALUE,
        ThresholdMeasurementStatField.FINAL_THRESHOLD,
    )

    def final_threshold(self, row: Mapping[str, Any]) -> Any:
        for field in self.final_threshold_fields:
            if field.value in row:
                return row[field.value]
        raise KeyError(
            "Threshold measurement row does not expose any known final-threshold "
            f"field: {tuple(field.value for field in self.final_threshold_fields)!r}."
        )

    def value_or_default(
        self,
        row: Mapping[str, Any],
        field: ThresholdMeasurementStatField,
        default: Any,
    ) -> Any:
        return row.get(field.value, default)


@dataclass(frozen=True, slots=True)
class ThresholdMeasurementRows(CellProfilerResultMeasurementRows):
    """Project absorbed threshold stats into CP image measurement rows."""

    registry_key = "threshold"

    object_name: str
    schema: ThresholdMeasurementStatSchema = ThresholdMeasurementStatSchema()

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for slice_stats in self.source_rows():
            stat_row = measurement_row_mapping(slice_stats)
            slice_index = self.schema.value_or_default(
                stat_row,
                ThresholdMeasurementStatField.SLICE_INDEX,
                0,
            )
            final_threshold = self.schema.final_threshold(stat_row)
            values = {
                ThresholdMeasurementFeatureTemplate.FINAL_THRESHOLD.feature_name(
                    object_name=self.object_name
                ): final_threshold,
                ThresholdMeasurementFeatureTemplate.ORIGINAL_THRESHOLD.feature_name(
                    object_name=self.object_name
                ): self.schema.value_or_default(
                    stat_row,
                    ThresholdMeasurementStatField.ORIGINAL_THRESHOLD,
                    final_threshold,
                ),
                ThresholdMeasurementFeatureTemplate.WEIGHTED_VARIANCE.feature_name(
                    object_name=self.object_name
                ): self.schema.value_or_default(
                    stat_row,
                    ThresholdMeasurementStatField.WEIGHTED_VARIANCE,
                    0.0,
                ),
                ThresholdMeasurementFeatureTemplate.SUM_OF_ENTROPIES.feature_name(
                    object_name=self.object_name
                ): self.schema.value_or_default(
                    stat_row,
                    ThresholdMeasurementStatField.SUM_OF_ENTROPIES,
                    0.0,
                ),
            }
            rows.extend(
                {
                    MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                    MEASUREMENT_FEATURE_NAME_FIELD: feature_name,
                    MEASUREMENT_RESULT_VALUE_FIELD: value,
                }
                for feature_name, value in values.items()
            )
        return rows


@dataclass(frozen=True, slots=True)
class ObjectLocationCenterValues:
    """XY center values for one object-label domain."""

    object_ids: tuple[int, ...]
    center_y: np.ndarray
    center_x: np.ndarray

    def feature_values(
        self,
        object_index: int,
    ) -> tuple[tuple[ObjectLocationMeasurementFeature, float], ...]:
        return (
            (
                ObjectLocationMeasurementFeature.CENTER_X,
                float(self.center_x[object_index]),
            ),
            (
                ObjectLocationMeasurementFeature.CENTER_Y,
                float(self.center_y[object_index]),
            ),
        )


@dataclass(frozen=True, slots=True)
class ObjectLocationMeasurementRows(CellProfilerMeasurementRows):
    """Emit CP object location rows from a declared object-label domain."""

    registry_key = "object_location"

    label_payload: Any
    object_name: str
    include_declared_empty: bool = True

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        label_planes = self.label_planes()
        slice_count = len(label_planes)
        for slice_index, label_plane in enumerate(label_planes):
            centers = self.centers_for_plane(
                label_plane,
                slice_index=slice_index,
                slice_count=slice_count,
            )
            rows.extend(
                self.rows_for_object(
                    object_label=object_label,
                    slice_index=slice_index,
                    feature_values=centers.feature_values(object_index),
                )
                for object_index, object_label in enumerate(centers.object_ids)
            )
        return [row for object_rows in rows for row in object_rows]

    def rows_for_object(
        self,
        *,
        object_label: int,
        slice_index: int,
        feature_values: tuple[tuple[ObjectLocationMeasurementFeature, float], ...],
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                MEASUREMENT_OBJECT_NAME_FIELD: self.object_name,
                MEASUREMENT_OBJECT_LABEL_FIELD: object_label,
                MEASUREMENT_FEATURE_NAME_FIELD: feature.value,
                MEASUREMENT_RESULT_VALUE_FIELD: value,
            }
            for feature, value in feature_values
        )

    def label_planes(self) -> tuple[np.ndarray, ...]:
        label_array = np.asarray(_label_payload_final(self.label_payload))
        if label_array.ndim <= 2:
            return (label_array,)
        return tuple(label_array[index] for index in range(label_array.shape[0]))

    def centers_for_plane(
        self,
        label_plane: Any,
        *,
        slice_index: int,
        slice_count: int,
    ) -> ObjectLocationCenterValues:
        domain = self.object_domain_for_plane(
            label_plane,
            slice_index=slice_index,
            slice_count=slice_count,
        )
        center_y, center_x = self.dense_label_centers_for_domain(label_plane, domain)
        return ObjectLocationCenterValues(
            object_ids=domain,
            center_y=center_y,
            center_x=center_x,
        )

    def object_domain_for_plane(
        self,
        label_plane: Any,
        *,
        slice_index: int,
        slice_count: int,
    ) -> tuple[int, ...]:
        if self.include_declared_empty:
            declared_domain = self.declared_domain_for_plane(
                slice_index=slice_index,
                slice_count=slice_count,
            )
            if declared_domain is not None:
                return declared_domain
        return self.present_domain_for_plane(
            label_plane,
            dense_extent=not self.include_declared_empty,
        )

    def declared_domain_for_plane(
        self,
        *,
        slice_index: int,
        slice_count: int,
    ) -> tuple[int, ...] | None:
        if not isinstance(self.label_payload, ObjectLabelDomainMetadata):
            return None
        return (
            self.label_payload.object_label_domain()
            .project_slice(slice_index, slice_count)
            .explicit_id_domain()
        )

    @staticmethod
    def present_domain_for_plane(
        label_plane: Any,
        *,
        dense_extent: bool,
    ) -> tuple[int, ...]:
        labels = np.asarray(label_plane)
        if labels.size == 0:
            return ()
        positive_labels = labels[labels > 0]
        if positive_labels.size == 0:
            return ()
        if dense_extent:
            return tuple(range(1, int(np.max(positive_labels)) + 1))
        return tuple(int(label_id) for label_id in np.unique(positive_labels))

    @staticmethod
    def dense_label_centers_for_domain(
        label_plane: Any,
        domain: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(label_plane, dtype=np.int64)
        center_y = np.full(len(domain), np.nan, dtype=np.float64)
        center_x = np.full(len(domain), np.nan, dtype=np.float64)
        if not domain or labels.size == 0:
            return center_y, center_x

        y_indices, x_indices = np.nonzero(labels > 0)
        if y_indices.size == 0:
            return center_y, center_x

        object_ids = labels[y_indices, x_indices]
        max_label = max(int(object_ids.max()), max(domain, default=0))
        counts = np.bincount(object_ids, minlength=max_label + 1)
        y_sums = np.bincount(object_ids, weights=y_indices, minlength=max_label + 1)
        x_sums = np.bincount(object_ids, weights=x_indices, minlength=max_label + 1)
        for index, object_label in enumerate(domain):
            if object_label <= 0 or object_label >= counts.shape[0]:
                continue
            count = counts[object_label]
            if count <= 0:
                continue
            center_y[index] = y_sums[object_label] / count
            center_x[index] = x_sums[object_label] / count
        return center_y, center_x


def _split_cellprofiler_output(raw_output: Any) -> tuple[Any, tuple[Any, ...]]:
    raw_output = runtime_output_tuple(raw_output)
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


def _measurement_rows_declare_object_name(rows: Sequence[Any]) -> bool:
    return any(
        measurement_row_mapping(row).get(MEASUREMENT_OBJECT_NAME_FIELD)
        not in (
            None,
            "",
        )
        for row in rows
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


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowIdentityProjectionRequest:
    """Typed context for projecting source rows into CellProfiler row identity."""

    rows: Sequence[Any]
    object_id_field: str
    axis_fields: Sequence[str]
    row_policy: CellProfilerObjectMeasurementRowPolicy

    def object_label(self, row: Any) -> int | None:
        """Return the object identity encoded by one source row."""
        return measurement_object_label(
            measurement_row_mapping(row),
            object_id_field=self.object_id_field,
        )

    def axis_key(self, row: Any) -> tuple[Any, ...]:
        """Return the measurement-axis key encoded by one source row."""
        return self.axis_key_from_mapping(measurement_row_mapping(row))

    def axis_key_from_mapping(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        """Return the measurement-axis key encoded by one row mapping."""
        if not self.axis_fields:
            return ()
        if len(self.axis_fields) == 1:
            return (row.get(self.axis_fields[0]),)
        return tuple(row.get(field_name) for field_name in self.axis_fields)

    def row_with_object_id(self, row: Any, object_id: int) -> dict[str, Any]:
        """Return a row projected to the requested object identity field."""
        projected_row = dict(measurement_row_mapping(row))
        projected_row[self.object_id_field] = object_id
        return projected_row

    def axis_keys_for_label_payload(
        self,
        projection: "ObjectMeasurementRowIdentityProjectionResult",
        *,
        label_payload: Any,
    ) -> tuple[tuple[Any, ...], ...]:
        """Return measurement axes valid for completing rows against labels."""
        if not self.axis_fields:
            return ((),)
        if not projection.rows:
            return ()
        labels = np.asarray(_label_payload_final(label_payload))
        if labels.ndim < 3 and tuple(self.axis_fields) == ("slice_index",):
            return (projection.row_keys[0][1],)
        return projection.axis_keys


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowCompletionSchema:
    """Nominal table schema for completing object-scoped measurement rows."""

    field_names: tuple[str, ...]
    object_id_field: str
    axis_fields: tuple[str, ...]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Any],
        func: Callable[..., Any],
    ) -> "ObjectMeasurementRowCompletionSchema":
        field_names = cls.field_names_from_rows(rows, func)
        return cls(
            field_names=field_names,
            object_id_field=cls.object_id_field_from_fields(field_names),
            axis_fields=cls.axis_fields_from_fields(field_names),
        )

    @staticmethod
    def field_names_from_rows(
        rows: Sequence[Any],
        func: Callable[..., Any],
    ) -> tuple[str, ...]:
        if rows:
            return tuple(str(key) for key in measurement_row_mapping(rows[0]).keys())
        return tuple(field.name for field in _measurement_fields_from_callable(func))

    @staticmethod
    def object_id_field_from_fields(field_names: Sequence[str]) -> str:
        for field_name in field_names:
            if field_name in _MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS:
                return field_name
        return MEASUREMENT_OBJECT_LABEL_FIELD

    @staticmethod
    def axis_fields_from_fields(field_names: Sequence[str]) -> tuple[str, ...]:
        axis_field_names = measurement_row_axis_field_names()
        return tuple(
            field_name
            for field_name in field_names
            if (
                field_name in axis_field_names
                and field_name not in _MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS
            )
        )

    def object_ids_for_axis(
        self,
        *,
        label_payload: Any,
        object_identity: MeasurementObjectRowIdentity,
        axis_key: tuple[Any, ...],
    ) -> tuple[int, ...]:
        axis_payload = self.label_payload_for_axis(label_payload, axis_key=axis_key)
        label_ids = dense_object_label_id_domain(axis_payload)
        if object_identity is MeasurementObjectRowIdentity.LABEL_ID:
            return label_ids
        if object_identity is MeasurementObjectRowIdentity.ROW_ORDINAL:
            return tuple(range(1, len(label_ids) + 1))
        raise ValueError(
            f"Unsupported measurement object row identity: {object_identity}."
        )

    def label_payload_for_axis(
        self,
        label_payload: Any,
        *,
        axis_key: tuple[Any, ...],
    ) -> Any:
        normalized_axis_fields = tuple(
            str(field_name).strip().lower() for field_name in self.axis_fields
        )
        slice_axis_name = MeasurementRowAxisField.SLICE_INDEX.value
        if slice_axis_name not in normalized_axis_fields:
            return label_payload
        slice_axis_position = normalized_axis_fields.index(slice_axis_name)
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
        return RuntimeSliceProjection.value_for_slice(
            label_payload,
            slice_index,
            labels.shape[0],
        )

    def positive_extent_for_missing_measurements(
        self,
        *,
        label_payload: Any,
        axis_key: tuple[Any, ...],
        row_policy: CellProfilerObjectMeasurementRowPolicy,
    ) -> int | None:
        policy = MissingObjectMeasurementValuePolicy(
            type(row_policy).missing_value_policy
        )
        if (
            policy
            is not MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
        ):
            return None
        axis_payload = self.label_payload_for_axis(label_payload, axis_key=axis_key)
        return self.positive_object_label_extent(axis_payload)

    @staticmethod
    def positive_object_label_extent(label_payload: Any) -> int:
        labels = np.asarray(_label_payload_final(label_payload))
        if labels.size == 0:
            return 0
        positive_labels = labels[labels > 0]
        if positive_labels.size == 0:
            return 0
        return int(np.max(positive_labels))

    def missing_row(
        self,
        *,
        object_id: int,
        axis_key: Sequence[Any],
        label_payload: Any,
        row_policy: CellProfilerObjectMeasurementRowPolicy,
        positive_label_extent: int | None = None,
    ) -> dict[str, Any]:
        axis_values = self.axis_values_for_key(axis_key)
        row = {
            field_name: row_policy.missing_measurement_value(
                object_id=object_id,
                label_payload=label_payload,
                field_name=field_name,
                positive_label_extent=positive_label_extent,
            )
            for field_name in self.field_names
            if (
                field_name not in _MEASUREMENT_COMPLETION_OBJECT_ID_FIELDS
                and field_name not in axis_values
            )
        }
        row.update(axis_values)
        row[self.object_id_field] = object_id
        return row

    def axis_values_for_key(self, axis_key: Sequence[Any]) -> dict[str, Any]:
        if len(axis_key) > len(self.axis_fields):
            raise ValueError(
                "Measurement axis key has more values than axis fields; got "
                f"{tuple(axis_key)!r} for fields {tuple(self.axis_fields)!r}."
            )
        return dict(zip(self.axis_fields, axis_key, strict=False))


@dataclass(frozen=True, slots=True)
class ObjectMeasurementRowIdentityProjectionResult:
    """Rows plus their nominal object/axis identity after projection."""

    rows: tuple[Any, ...]
    row_keys: tuple[tuple[int | None, tuple[Any, ...]], ...]
    axis_keys: tuple[tuple[Any, ...], ...]

    def ordered_rows(
        self,
        *,
        object_ids: Sequence[int],
        axis_keys: Sequence[tuple[Any, ...]],
        rows: Sequence[Any] | None = None,
        row_keys: Sequence[tuple[int | None, tuple[Any, ...]]] | None = None,
    ) -> list[Any]:
        """Return rows in dense object/axis order using projected identities."""
        ordered_rows = self.rows if rows is None else rows
        ordered_row_keys = self.row_keys if row_keys is None else row_keys
        object_order = {object_id: index for index, object_id in enumerate(object_ids)}
        axis_order = {axis_key: index for index, axis_key in enumerate(axis_keys)}
        indexed_rows = tuple(
            enumerate(zip(ordered_rows, ordered_row_keys, strict=True))
        )
        return [
            row
            for _index, (row, _row_key) in sorted(
                indexed_rows,
                key=lambda item: self.row_order_key(
                    item[1][1],
                    item[0],
                    object_order=object_order,
                    axis_order=axis_order,
                ),
            )
        ]

    @staticmethod
    def row_order_key(
        row_key: tuple[int | None, tuple[Any, ...]],
        fallback_index: int,
        *,
        object_order: Mapping[int, int],
        axis_order: Mapping[tuple[Any, ...], int],
    ) -> tuple[int, int, int]:
        """Return a stable ordering key for one projected measurement row."""
        object_id, axis_key = row_key
        return (
            axis_order.get(axis_key, len(axis_order)),
            (
                object_order.get(object_id, len(object_order))
                if object_id is not None
                else len(object_order)
            ),
            fallback_index,
        )


@dataclass(slots=True)
class ObjectMeasurementRowOrdinalProjectionState:
    """Mutable ordinal ownership state for one compact row projection pass."""

    ordinal_by_axis: dict[tuple[Any, ...], int] = field(default_factory=dict)
    ordinal_by_original_id: dict[tuple[tuple[Any, ...], int], int] = field(
        default_factory=dict
    )

    def register_measured_object(
        self,
        row_mapping: Mapping[str, object],
        *,
        axis_key: tuple[Any, ...],
        object_id_field: str,
    ) -> None:
        """Register a measured source object before compact row projection."""
        original_id = measurement_object_label(
            row_mapping,
            object_id_field=object_id_field,
        )
        ordinal_key = (axis_key, original_id) if original_id is not None else None
        ordinal = (
            self.ordinal_by_original_id.get(ordinal_key)
            if ordinal_key is not None
            else None
        )
        if ordinal is not None:
            return
        ordinal = self.ordinal_by_axis.get(axis_key, 0) + 1
        self.ordinal_by_axis[axis_key] = ordinal
        if ordinal_key is not None:
            self.ordinal_by_original_id[ordinal_key] = ordinal

    def ordinal_for_measured_object(
        self,
        row_mapping: Mapping[str, object],
        *,
        axis_key: tuple[Any, ...],
        object_id_field: str,
    ) -> int:
        """Return the compact row ordinal for a registered measured object."""
        original_id = measurement_object_label(
            row_mapping,
            object_id_field=object_id_field,
        )
        ordinal_key = (axis_key, original_id) if original_id is not None else None
        if ordinal_key is not None:
            return self.ordinal_by_original_id[ordinal_key]
        return self.next_unbound_ordinal(axis_key)

    def next_unbound_ordinal(self, axis_key: tuple[Any, ...]) -> int:
        """Allocate an ordinal for retained rows that have no measured object."""
        ordinal = self.ordinal_by_axis.get(axis_key, 0) + 1
        self.ordinal_by_axis[axis_key] = ordinal
        return ordinal


class MeasurementObjectRowIdentityProjectionStrategy(
    EnumKeyedStrategyMixin[MeasurementObjectRowIdentity],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered projection from source object IDs to exported row identity."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "object_identity"
    object_identity: ClassVar[MeasurementObjectRowIdentity]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def project(
        cls,
        object_identity: MeasurementObjectRowIdentity,
        request: ObjectMeasurementRowIdentityProjectionRequest,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        """Project rows using the strategy declared by ``object_identity``."""
        return cls.for_enum_member(object_identity).project_rows(request)

    @abstractmethod
    def project_rows(
        self,
        request: ObjectMeasurementRowIdentityProjectionRequest,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        """Return rows with the requested object-row identity."""


class LabelIdMeasurementObjectRowIdentityProjectionStrategy(
    MeasurementObjectRowIdentityProjectionStrategy
):
    """Preserve label IDs as exported object-row identities."""

    object_identity = MeasurementObjectRowIdentity.LABEL_ID

    def project_rows(
        self,
        request: ObjectMeasurementRowIdentityProjectionRequest,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        rows = tuple(request.rows)
        row_keys = tuple(
            (
                request.object_label(row),
                request.axis_key(row),
            )
            for row in rows
        )
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=rows,
            row_keys=row_keys,
            axis_keys=tuple(
                dict.fromkeys(axis_key for _object_id, axis_key in row_keys)
            ),
        )


class RowOrdinalMeasurementObjectRowIdentityProjectionStrategy(
    MeasurementObjectRowIdentityProjectionStrategy
):
    """Project measured objects into CP's compact row-ordinal identity."""

    object_identity = MeasurementObjectRowIdentity.ROW_ORDINAL

    def project_rows(
        self,
        request: ObjectMeasurementRowIdentityProjectionRequest,
    ) -> ObjectMeasurementRowIdentityProjectionResult:
        ordinal_state = ObjectMeasurementRowOrdinalProjectionState()
        row_entries: list[tuple[Any, Mapping[str, object], tuple[Any, ...], bool]] = []
        for row in request.rows:
            row_mapping = measurement_row_mapping(row)
            axis_key = request.axis_key_from_mapping(row_mapping)
            measured = request.row_policy.row_has_measured_object(
                row_mapping,
                object_id_field=request.object_id_field,
                axis_fields=request.axis_fields,
            )
            row_entries.append((row, row_mapping, axis_key, measured))
            if not measured:
                continue
            ordinal_state.register_measured_object(
                row_mapping,
                axis_key=axis_key,
                object_id_field=request.object_id_field,
            )

        projected_rows: list[Any] = []
        projected_row_keys: list[tuple[int, tuple[Any, ...]]] = []
        for row, row_mapping, axis_key, measured in row_entries:
            if measured:
                ordinal = ordinal_state.ordinal_for_measured_object(
                    row_mapping,
                    axis_key=axis_key,
                    object_id_field=request.object_id_field,
                )
            else:
                if not request.row_policy.retains_unmeasured_compact_row(
                    row_mapping,
                    object_id_field=request.object_id_field,
                    axis_fields=request.axis_fields,
                ):
                    continue
                ordinal = ordinal_state.next_unbound_ordinal(axis_key)
            projected_rows.append(request.row_with_object_id(row, ordinal))
            projected_row_keys.append((ordinal, axis_key))
        object_ids = tuple(
            sorted(dict.fromkeys(ordinal for ordinal, _axis_key in projected_row_keys))
        )
        axis_keys = tuple(
            dict.fromkeys(
                axis_key for _row, _mapping, axis_key, _measured in row_entries
            )
        )
        projection = ObjectMeasurementRowIdentityProjectionResult(
            rows=tuple(projected_rows),
            row_keys=tuple(projected_row_keys),
            axis_keys=axis_keys,
        )
        object_order = {object_id: index for index, object_id in enumerate(object_ids)}
        axis_order = {axis_key: index for index, axis_key in enumerate(axis_keys)}
        ordered_entries = tuple(
            sorted(
                enumerate(zip(projection.rows, projection.row_keys, strict=True)),
                key=lambda item: projection.row_order_key(
                    item[1][1],
                    item[0],
                    object_order=object_order,
                    axis_order=axis_order,
                ),
            )
        )
        return ObjectMeasurementRowIdentityProjectionResult(
            rows=tuple(row for _index, (row, _row_key) in ordered_entries),
            row_keys=tuple(row_key for _index, (_row, row_key) in ordered_entries),
            axis_keys=axis_keys,
        )


_MISSING_MEASUREMENT_OBJECT_NAME = object()


def _record_measurements(
    adapter: CellProfilerRuntimeAdapter,
    name: str,
    rows: Sequence[Any],
    *,
    fields: tuple[FieldSpec, ...] = (),
    object_name: str | None | object = _MISSING_MEASUREMENT_OBJECT_NAME,
    source_image_name: str | None = None,
    source_image_payload: Any | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "source_image_name": source_image_name,
    }
    if object_name is not _MISSING_MEASUREMENT_OBJECT_NAME:
        kwargs["object_name"] = object_name
    projection_started_at = time.perf_counter()
    projected_rows, projected_row_mappings = _cellprofiler_global_image_number_rows(
        adapter,
        rows,
        source_image_name=source_image_name,
        source_image_payload=source_image_payload,
        object_name=object_name,
        need_row_mappings=bool(fields),
    )
    _log_module_profile(
        "record_measurements_project_rows",
        time.perf_counter() - projection_started_at,
        rows=len(projected_rows),
        fields=bool(fields),
    )
    fields_started_at = time.perf_counter()
    fields = _measurement_fields_covering_mappings(fields, projected_row_mappings)
    _log_module_profile(
        "record_measurements_fields",
        time.perf_counter() - fields_started_at,
        rows=len(projected_row_mappings),
        fields=bool(fields),
    )
    if fields:
        kwargs["fields"] = fields
    add_started_at = time.perf_counter()
    adapter.add_measurements(
        name,
        projected_rows,
        **kwargs,
    )
    _log_module_profile(
        "record_measurements_add",
        time.perf_counter() - add_started_at,
        rows=len(projected_rows),
    )


def _measurement_fields_covering_mappings(
    fields: tuple[FieldSpec, ...],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[FieldSpec, ...]:
    """Preserve declared table order while retaining projected semantic fields."""
    if not fields:
        return fields
    declared_names = {field.name for field in fields}
    extra_names = tuple(
        dict.fromkeys(
            field_name
            for row in rows
            for field_name in row
            if field_name not in declared_names
        )
    )
    if not extra_names:
        return fields
    return (*fields, *(FieldSpec(field_name) for field_name in extra_names))


def _cellprofiler_global_image_number_rows(
    adapter: CellProfilerRuntimeAdapter,
    rows: Sequence[Any],
    *,
    source_image_name: str | None,
    source_image_payload: Any | None = None,
    object_name: str | None | object,
    need_row_mappings: bool = True,
) -> tuple[Sequence[Any], Sequence[Mapping[str, Any]]]:
    row_mappings: list[Mapping[str, Any]] = []
    has_image_number = False
    has_slice_index = False
    for row in rows:
        row_mapping = measurement_row_mapping(row)
        if need_row_mappings:
            row_mappings.append(row_mapping)
        has_image_number = has_image_number or "image_number" in row_mapping
        has_slice_index = has_slice_index or "slice_index" in row_mapping
        if (has_image_number or has_slice_index) and not need_row_mappings:
            row_mappings = [measurement_row_mapping(candidate) for candidate in rows]
            break

    if not rows:
        return rows, row_mappings
    if not has_image_number and not has_slice_index:
        return rows, row_mappings

    resolved_source_image_name = source_image_name
    if (
        resolved_source_image_name is None
        and object_name is not _MISSING_MEASUREMENT_OBJECT_NAME
        and object_name is not None
    ):
        resolved_source_image_name = adapter.get_objects(
            str(object_name)
        ).source_image_name

    start = _cellprofiler_image_number_start(
        source_image_payload if source_image_payload is not None else object(),
        adapter,
        source_image_name=resolved_source_image_name,
    )
    if start <= 1:
        if has_slice_index and not has_image_number:
            projected_rows = [dict(row) for row in row_mappings]
            for row in projected_rows:
                if _measurement_axis_value_is_present(row.get("slice_index")):
                    row["image_number"] = int(row["slice_index"]) + 1
            return projected_rows, projected_rows
        return rows, row_mappings

    if has_slice_index and not has_image_number:
        projected_rows = [dict(row) for row in row_mappings]
        for row in projected_rows:
            if _measurement_axis_value_is_present(row.get("slice_index")):
                row["image_number"] = int(row["slice_index"]) + start
        return projected_rows, projected_rows

    image_numbers = [
        int(row["image_number"])
        for row in row_mappings
        if _measurement_axis_value_is_present(row.get("image_number"))
    ]
    if not image_numbers or min(image_numbers) >= start:
        return rows, row_mappings

    offset = start - 1
    projected_rows = [dict(row) for row in row_mappings]
    for row in projected_rows:
        if _measurement_axis_value_is_present(row.get("image_number")):
            row["image_number"] = int(row["image_number"]) + offset
    return projected_rows, projected_rows


def _measurement_axis_value_is_present(value: Any) -> bool:
    """Return whether an axis value can participate in ImageNumber projection."""
    if value is None:
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return True
    return math.isfinite(numeric_value)


def _coerce_spatial_grid(
    value: Any,
    name: str,
) -> SpatialGrid | Mapping[str, Any] | RuntimeSliceAlignedValues[Any]:
    if isinstance(value, RuntimeSliceAlignedValues):
        return RuntimeSliceAlignedValues(
            slices=tuple(_coerce_spatial_grid(item, name) for item in value.slices)
        )
    if isinstance(value, SpatialGrid):
        return value.with_name(name)
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return {
            field.name: getattr(value, field.name) for field in dataclass_fields(value)
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
        return _field_specs_for_rows([measurement_row_mapping(row) for row in rows])
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
    """Use one 2D plane to carry object-only CellProfiler modules once.

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
    while isinstance(image_data, np.ndarray) and image_data.ndim > 2:
        if image_data.shape[0] < 1:
            break
        image_data = image_data[0]
        if is_color_image_slice(image_data):
            return image_data[:, :, 0]
    return image_data


def _measurement_image_for_labels(
    image: Any,
    labels: Any,
    *,
    label_payload: Any | None = None,
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
    return MeasurementImageLabelAlignmentStrategy.align(
        MeasurementImageLabelAlignmentRequest(
            image=image,
            image_data=image_payload_data(image),
            labels=labels,
            label_payload=label_payload,
            reference_domain=reference_domain,
        )
    )


def _image_scope_measurement_payload(image: Any) -> Any:
    """Return one image plane for image-scoped measurement functions."""
    return _collapse_singleton_stack_output(image)


def _measurement_labels(labels: Any) -> Any:
    """Normalize singleton stack labels for absorbed 2D measurement functions."""
    return collapse_singleton_object_label_stack(labels)


class MeasurementLabelSourceAlignmentStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Align measurement labels to the source domain of a measurement image."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def align(
        cls,
        image: Any,
        labels: Any,
        *,
        label_payload: Any | None = None,
    ) -> Any:
        strategy = cls.for_nominal_value(image)
        if strategy is None:
            strategy = DefaultMeasurementLabelSourceAlignmentStrategy()
        return strategy.labels_for_image(
            image,
            labels,
            label_payload=label_payload,
        )

    @abstractmethod
    def labels_for_image(
        self,
        image: Any,
        labels: Any,
        *,
        label_payload: Any | None = None,
    ) -> Any:
        """Return labels in the measurement image's execution domain."""


class DefaultMeasurementLabelSourceAlignmentStrategy(
    MeasurementLabelSourceAlignmentStrategy
):
    """Align labels directly to a non-aligned measurement image payload."""

    def labels_for_image(
        self,
        image: Any,
        labels: Any,
        *,
        label_payload: Any | None = None,
    ) -> Any:
        del label_payload
        return self.source_aligned_labels(image, labels)

    @staticmethod
    def source_aligned_labels(image: Any, labels: Any) -> Any:
        labels = _measurement_labels(labels)
        image_domain_adapter = _measurement_image_source_spatial_adapter(image)
        if image_domain_adapter is not None:
            labels = image_domain_adapter.extract_source_array(labels)
        image_array = np.asarray(image)
        label_array = np.asarray(labels)
        if image_array.ndim == 0 or label_array.ndim == 0:
            return labels
        return _collapse_repeated_label_stack_for_image(image, labels)


class AlignedStackMeasurementLabelSourceAlignmentStrategy(
    DefaultMeasurementLabelSourceAlignmentStrategy
):
    """Preserve runtime-slice labels until aligned image slices are selected."""

    value_type = AlignedImageStack

    def labels_for_image(
        self,
        image: Any,
        labels: Any,
        *,
        label_payload: Any | None = None,
    ) -> Any:
        if not isinstance(image, AlignedImageStack):
            raise TypeError(
                "AlignedStackMeasurementLabelSourceAlignmentStrategy requires "
                f"AlignedImageStack, got {type(image).__name__}."
            )
        if (
            label_payload is not None
            and ObjectLabelRuntimeSliceStackContract.preserves_runtime_slice_stack(
                label_payload,
                slice_count=len(image.slices),
            )
        ):
            return _measurement_labels(labels)
        return self.source_aligned_labels(image, labels)


def _measurement_image_source_spatial_adapter(
    image: Any,
) -> SourceSpatialDomainAdapter | None:
    """Return the nominal source-domain adapter for a measurement image."""
    if isinstance(image, AlignedImageStack):
        return image.first_slice_source_spatial_adapter()
    return SourceSpatialDomainAdapter.for_value(image)


def _measurement_labels_for_measurement_image(
    measurement_image: CellProfilerMeasurementImage,
    label_payload: Any,
    *,
    adapter: CellProfilerRuntimeAdapter | None = None,
) -> Any:
    """Resolve object labels into the semantic source plane for a measurement image."""
    labels = _label_payload_final(label_payload)
    if not isinstance(label_payload, ObjectLabelSet):
        return MeasurementLabelSourceAlignmentStrategy.align(
            measurement_image.payload,
            labels,
            label_payload=label_payload,
        )
    labels = ObjectLabelSourceBindingProjectionStrategy.for_enum_member(
        label_payload.domain_scope,
    ).project(
        ObjectLabelSourceBindingProjectionRequest(
            labels=labels,
            label_payload=label_payload,
            measurement_image=measurement_image,
            adapter=adapter,
            source_image_name=label_payload.source_image_name,
            source_image_names=measurement_image.source_image_names,
        )
    )
    return MeasurementLabelSourceAlignmentStrategy.align(
        measurement_image.payload,
        labels,
        label_payload=label_payload,
    )


def _collapse_repeated_label_stack_for_image(image: Any, labels: Any) -> Any:
    """Collapse channel-broadcast object labels before absorbed CP measurement calls."""
    if not isinstance(image, np.ndarray) or not isinstance(labels, np.ndarray):
        return labels
    if not is_image_stack(image) or labels.ndim != image.ndim:
        return labels
    if tuple(labels.shape[1:]) != tuple(image.shape[1:]):
        return labels
    if labels.shape[0] == 0:
        return labels
    first_plane = labels[0]
    if all(
        np.array_equal(first_plane, labels[index])
        for index in range(1, labels.shape[0])
    ):
        return first_plane
    return labels


def _label_payload_final(payload: Any) -> Any:
    """Return the final label plane from a runtime label payload."""
    if isinstance(payload, ObjectLabelSet):
        if payload.representation is ObjectLabelRepresentation.SPARSE_IJV:
            return payload
        payload = payload.runtime_payload()
    if isinstance(payload, ObjectLabelPayload):
        payload = payload.labels
    return collapse_singleton_object_label_stack(payload)


def _label_payload_small_removed(payload: Any) -> Any | None:
    """Return the small-removed label variant when the runtime provides it."""
    if not isinstance(payload, ObjectLabelPayload):
        return None
    if payload.small_removed_labels is None:
        return None
    return collapse_singleton_object_label_stack(payload.small_removed_labels)


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
        return _label_payload_final(
            adapter.resolve_source_objects(spec.name, current_image)
        )
    return _label_payload_final(adapter.get_objects(spec.name))


def _measurement_object_name(
    inputs: tuple[ArtifactSpec, ...],
) -> str | None:
    object_inputs = ArtifactSpecCollection(inputs).of_kind(ArtifactKind.OBJECT_LABELS)
    if len(object_inputs) == 1:
        return object_inputs[0].name
    return None


class CellProfilerRelationshipMeasurementFeature(FormattingMeasurementFeatureTemplate):
    """CellProfiler relationship measurement feature-name contract."""

    PARENT = "Parent_{parent_object_name}"
    DISTANCE_CENTROID = "Distance_Centroid_{parent_object_name}"
    DISTANCE_MINIMUM = "Distance_Minimum_{parent_object_name}"


# Backwards-compatible alias for existing local callers/tests.
RelationshipMeasurementFeatureTemplate = CellProfilerRelationshipMeasurementFeature


@dataclass(frozen=True, slots=True)
class RelationshipEndpointContract:
    """Nominal parent/child endpoint contract for one relationship artifact."""

    parent: ArtifactSpec
    child: ArtifactSpec


@dataclass(frozen=True, slots=True)
class RelationshipEndpointResolver:
    """Resolve declared relationship artifacts to parent/child object endpoints."""

    request: CellProfilerOutputRecordRequest

    @property
    def object_inputs(self) -> tuple[ArtifactSpec, ...]:
        return ArtifactSpecCollection(
            self.request.executor._declared_input_specs()
        ).of_kind(ArtifactKind.OBJECT_LABELS)

    @property
    def object_outputs(self) -> tuple[ArtifactSpec, ...]:
        return ArtifactSpecCollection(self.request.executor.outputs).of_kind(
            ArtifactKind.OBJECT_LABELS
        )

    def endpoint_specs(
        self,
        relationship_spec: ArtifactSpec,
    ) -> tuple[ArtifactSpec, ArtifactSpec]:
        contract = self.endpoint_contract(relationship_spec)
        return contract.parent, contract.child

    def endpoint_contract(
        self,
        relationship_spec: ArtifactSpec,
    ) -> RelationshipEndpointContract:
        matches = self.artifact_name_matches(relationship_spec)
        if len(matches) == 1:
            return RelationshipEndpointContract(*matches[0])
        if len(matches) > 1:
            raise ValueError(
                f"{self.request.executor.module_name} relationship output "
                f"'{relationship_spec.name}' matches multiple object endpoint pairs."
            )
        if len(self.object_inputs) == 2 and not self.object_outputs:
            return RelationshipEndpointContract(
                self.object_inputs[0], self.object_inputs[1]
            )
        endpoints = parent_child_relationship_artifact_endpoints(
            relationship_spec.name,
            parent_candidates=tuple(spec.name for spec in self.object_inputs),
        )
        if endpoints is not None:
            parent_name, child_name = endpoints
            parent_spec = ArtifactSpecCollection(self.object_inputs).by_name(
                parent_name
            )
            if parent_spec is not None:
                child_spec = ArtifactSpecCollection(
                    (*self.object_outputs, *self.object_inputs),
                ).by_name(child_name)
                return RelationshipEndpointContract(
                    parent_spec,
                    child_spec
                    or ArtifactSpec(
                        child_name,
                        ArtifactKind.OBJECT_LABELS,
                    ),
                )
        raise NotImplementedError(
            f"{self.request.executor.module_name} relationship output "
            f"'{relationship_spec.name}' cannot be mapped to object endpoints from "
            f"inputs={[spec.name for spec in self.object_inputs]} and "
            f"outputs={[spec.name for spec in self.object_outputs]}."
        )

    def artifact_name_matches(
        self,
        relationship_spec: ArtifactSpec,
    ) -> tuple[tuple[ArtifactSpec, ArtifactSpec], ...]:
        candidate_children = (*self.object_inputs, *self.object_outputs)
        return tuple(
            (parent_spec, child_spec)
            for parent_spec in self.object_inputs
            for child_spec in candidate_children
            if parent_spec.name != child_spec.name
            and relationship_spec.name
            == parent_child_relationship_artifact_name(
                parent_spec.name,
                child_spec.name,
            )
        )


@dataclass(frozen=True, slots=True)
class RelationshipMeasurementRows(CellProfilerMeasurementRows):
    """Project parent-child relationship payloads into CP object measurement rows."""

    __registry_key__ = _MODULE_NAME_REGISTRY_KEY
    __skip_if_no_key__ = True

    module_name: ClassVar[str | None] = None
    request: CellProfilerOutputRecordRequest

    @classmethod
    def for_request(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> "RelationshipMeasurementRows":
        row_type = cls.__registry__.get(
            canonical_module_name(request.executor.module_name)
        )
        if row_type is None:
            return GenericRelationshipMeasurementRows(request)
        return row_type(request)

    def rows(self) -> list[dict[str, int | str]]:
        rows: list[dict[str, int | str]] = []
        endpoint_resolver = RelationshipEndpointResolver(self.request)
        for relationship_spec, payload in self.output_entries():
            parent_spec, child_spec = endpoint_resolver.endpoint_specs(
                relationship_spec
            )
            rows.extend(
                self.child_count_rows(
                    parent_object_name=parent_spec.name,
                    child_object_name=child_spec.name,
                    payload=payload,
                )
            )
            rows.extend(
                self.parent_rows(
                    parent_object_name=parent_spec.name,
                    child_object_name=child_spec.name,
                    payload=payload,
                )
            )
        return rows

    def output_entries(
        self,
    ) -> tuple[tuple[ArtifactSpec, ParentChildRelationshipPayload], ...]:
        return tuple(
            (spec, value)
            for spec in self.request.executor.outputs
            if spec.kind is ArtifactKind.RELATIONSHIPS
            for value in (self.request.output_values.get(spec.name),)
            if isinstance(value, ParentChildRelationshipPayload)
        )

    def child_count_rows(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        payload: ParentChildRelationshipPayload,
    ) -> tuple[dict[str, int | str], ...]:
        sliced_pairs = self.payload_pairs_by_slice(payload)
        if sliced_pairs is not None:
            rows: list[dict[str, int | str]] = []
            for slice_index, pairs in sliced_pairs:
                related_parent_ids = tuple(parent_id for parent_id, _child_id in pairs)
                rows.extend(
                    self.child_count_rows_for_ids(
                        parent_object_name=parent_object_name,
                        child_object_name=child_object_name,
                        related_parent_ids=related_parent_ids,
                        slice_index=slice_index,
                    )
                )
            return tuple(rows)
        return self.child_count_rows_for_ids(
            parent_object_name=parent_object_name,
            child_object_name=child_object_name,
            related_parent_ids=tuple(
                int(parent_id) for parent_id in payload.parent_ids
            ),
            slice_index=None,
        )

    def child_count_rows_for_ids(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        related_parent_ids: tuple[int, ...],
        slice_index: int | None,
    ) -> tuple[dict[str, int | str], ...]:
        related_parent_ids = tuple(int(parent_id) for parent_id in related_parent_ids)
        parent_count = max(
            (
                self.object_label_count(parent_object_name, slice_index=slice_index),
                *related_parent_ids,
            )
        )
        counts = {parent_id: 0 for parent_id in range(1, parent_count + 1)}
        for parent_id in related_parent_ids:
            if parent_id > 0:
                counts[parent_id] = counts.get(parent_id, 0) + 1
        feature_name = CellProfilerMeasurementFeature.child_count(
            child_object_name
        ).name
        return tuple(
            self.axis_qualified_row(
                {
                    MEASUREMENT_OBJECT_NAME_FIELD: parent_object_name,
                    MEASUREMENT_OBJECT_LABEL_FIELD: parent_id,
                    feature_name: count,
                },
                slice_index=slice_index,
            )
            for parent_id, count in counts.items()
        )

    def parent_rows(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        payload: ParentChildRelationshipPayload,
    ) -> tuple[dict[str, int | str], ...]:
        sliced_pairs = self.payload_pairs_by_slice(payload)
        if sliced_pairs is not None:
            rows: list[dict[str, int | str]] = []
            for slice_index, pairs in sliced_pairs:
                rows.extend(
                    self.parent_rows_for_pairs(
                        parent_object_name=parent_object_name,
                        child_object_name=child_object_name,
                        pairs=pairs,
                        slice_index=slice_index,
                    )
                )
            return tuple(rows)
        return self.parent_rows_for_pairs(
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

    def parent_rows_for_pairs(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        pairs: tuple[tuple[int, int], ...],
        slice_index: int | None,
    ) -> tuple[dict[str, int | str], ...]:
        parent_by_child = {
            int(child_id): int(parent_id) for parent_id, child_id in pairs
        }
        child_count = max(
            (
                self.object_label_count(child_object_name, slice_index=slice_index),
                *parent_by_child.keys(),
            )
        )
        feature_name = CellProfilerRelationshipMeasurementFeature.PARENT.feature_name(
            parent_object_name=parent_object_name
        )
        return tuple(
            self.axis_qualified_row(
                {
                    MEASUREMENT_OBJECT_NAME_FIELD: child_object_name,
                    MEASUREMENT_OBJECT_LABEL_FIELD: child_id,
                    feature_name: parent_by_child.get(child_id, 0),
                },
                slice_index=slice_index,
            )
            for child_id in range(1, child_count + 1)
        )

    def object_label_count(
        self,
        object_name: str,
        *,
        slice_index: int | None = None,
    ) -> int:
        if object_name in self.request.output_values:
            return _object_label_count_from_value(
                self.request.output_values[object_name],
                slice_index=slice_index,
            )
        return _object_label_count(
            self.request.adapter,
            object_name,
            slice_index=slice_index,
        )

    @staticmethod
    def payload_pairs_by_slice(
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

    @staticmethod
    def axis_qualified_row(
        row: dict[str, int | str],
        *,
        slice_index: int | None,
    ) -> dict[str, int | str]:
        if slice_index is None:
            return row
        return {
            **row,
            MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
        }


class GenericRelationshipMeasurementRows(RelationshipMeasurementRows):
    """Default relationship rows: child counts plus parent ids."""


class RelateObjectsRelationshipMeasurementRows(RelationshipMeasurementRows):
    """RelateObjects additionally projects configured child-parent distances."""

    module_name = "RelateObjects"

    def rows(self) -> list[dict[str, int | str | float]]:
        rows: list[dict[str, int | str | float]] = list(super().rows())
        endpoint_resolver = RelationshipEndpointResolver(self.request)
        for relationship_spec, payload in self.output_entries():
            parent_spec, child_spec = endpoint_resolver.endpoint_specs(
                relationship_spec
            )
            if not self.distance_contract_applies(parent_spec, child_spec):
                continue
            rows.extend(
                self.distance_rows(
                    parent_object_name=parent_spec.name,
                    child_object_name=child_spec.name,
                    payload=payload,
                )
            )
        return rows

    def distance_contract_applies(
        self,
        parent_spec: ArtifactSpec,
        child_spec: ArtifactSpec,
    ) -> bool:
        object_inputs = RelationshipEndpointResolver(self.request).object_inputs
        if len(object_inputs) < 2:
            return False
        return parent_spec == object_inputs[0] and child_spec == object_inputs[1]

    def distance_rows(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        payload: ParentChildRelationshipPayload,
    ) -> tuple[dict[str, int | str | float], ...]:
        if not self.distance_measurements_declared():
            return ()
        sliced_pairs = self.payload_pairs_by_slice(payload)
        if sliced_pairs is not None:
            rows: list[dict[str, int | str | float]] = []
            for slice_index, pairs in sliced_pairs:
                rows.extend(
                    self.distance_rows_for_pairs(
                        parent_object_name=parent_object_name,
                        child_object_name=child_object_name,
                        pairs=pairs,
                        slice_index=slice_index,
                    )
                )
            return tuple(rows)
        return self.distance_rows_for_pairs(
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

    def distance_measurements_declared(self) -> bool:
        value = self.request.value
        if not isinstance(value, RelationshipMeasurements):
            return False
        return value.declares_distance_measurements

    def distance_rows_for_pairs(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        pairs: tuple[tuple[int, int], ...],
        slice_index: int | None,
    ) -> tuple[dict[str, int | str | float], ...]:
        if not pairs:
            return ()
        parent_labels = self.object_labels(parent_object_name, slice_index=slice_index)
        child_labels = self.object_labels(child_object_name, slice_index=slice_index)
        if parent_labels is None or child_labels is None:
            return ()
        parent_array = RuntimeSliceProjection.object_label_endpoint_dense_array(
            parent_labels,
            dtype=np.int32,
        )
        child_array = RuntimeSliceProjection.object_label_endpoint_dense_array(
            child_labels,
            dtype=np.int32,
        )
        parent_array, child_array = aligned_dense_object_label_arrays(
            parent_array,
            child_array,
        )
        parents_of = np.zeros(
            int(child_array.max()) if child_array.size else 0,
            dtype=np.int32,
        )
        for parent_id, child_id in pairs:
            if 0 < child_id <= len(parents_of):
                parents_of[child_id - 1] = parent_id
        backend = ObjectRelationshipBackendStrategy.for_memory_type()
        centroid_distances = backend.centroid_distances(
            parent_array,
            child_array,
            parents_of,
        )
        minimum_distances = backend.minimum_distances(
            parent_array,
            child_array,
            parents_of,
        )
        centroid_feature = (
            CellProfilerRelationshipMeasurementFeature.DISTANCE_CENTROID.feature_name(
                parent_object_name=parent_object_name
            )
        )
        minimum_feature = (
            CellProfilerRelationshipMeasurementFeature.DISTANCE_MINIMUM.feature_name(
                parent_object_name=parent_object_name
            )
        )
        return tuple(
            self.axis_qualified_row(
                {
                    MEASUREMENT_OBJECT_NAME_FIELD: child_object_name,
                    MEASUREMENT_OBJECT_LABEL_FIELD: child_id,
                    centroid_feature: float(centroid_distances[child_id - 1]),
                    minimum_feature: float(minimum_distances[child_id - 1]),
                },
                slice_index=slice_index,
            )
            for _parent_id, child_id in pairs
            if 0 < child_id <= len(parents_of)
        )

    def object_labels(
        self,
        object_name: str,
        *,
        slice_index: int | None = None,
    ) -> Any | None:
        value = self.request.output_values.get(object_name)
        if value is None:
            value = self.request.adapter.get_objects(object_name)
        labels = _label_payload_final(value)
        return RuntimeSliceProjection.object_label_endpoint(
            labels,
            slice_index=slice_index,
        )


def _object_label_count(
    adapter: CellProfilerRuntimeAdapter,
    object_name: str,
    *,
    slice_index: int | None = None,
) -> int:
    return _object_label_count_from_value(
        adapter.get_objects(object_name),
        slice_index=slice_index,
    )


def _object_label_count_from_value(
    value: Any,
    *,
    slice_index: int | None,
) -> int:
    if isinstance(value, ObjectLabelPayload | ObjectLabelSet):
        labels = value.labels
    else:
        labels = value
    label_array = (
        _sparse_ijv_array(labels)
        if isinstance(labels, SparseIJVLabelRows)
        else np.asarray(labels)
    )
    if (
        isinstance(value, ObjectLabelSet)
        and value.representation is ObjectLabelRepresentation.SPARSE_IJV
    ):
        if label_array.size == 0:
            return 0
        sparse_rows = (
            labels
            if isinstance(labels, SparseIJVLabelRows)
            else SparseIJVLabelRows(labels)
        )
        if slice_index is not None:
            sparse_rows = sparse_rows.slice(slice_index)
            label_array = sparse_rows.as_array()
            if label_array.size == 0:
                return 0
        return int(np.max(label_array[:, sparse_rows.label_column]))
    if (
        isinstance(value, ObjectLabelPayload | ObjectLabelSet)
        and value.declared_object_count is not None
        and (slice_index is None or label_array.ndim < 3 or label_array.shape[0] == 1)
    ):
        return int(value.declared_object_count)
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


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecialInputBindingRequest(RuntimeInputBindingRequestBase):
    """Authoritative runtime context for binding declared special_inputs."""

    registry_key = "special_input"

    parameter_names: tuple[str, ...]
    special_input_specs: tuple[ArtifactSpec, ...]
    runtime_inputs: tuple[ArtifactSpec, ...]
    external_image_names: frozenset[str]
    runtime_image_names: frozenset[str]

    def __post_init__(self) -> None:
        RuntimeInputBindingRequestBase.__post_init__(self)
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(
            self,
            "special_input_specs",
            tuple(self.special_input_specs),
        )
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
        return ArtifactSpecCollection(self.special_input_specs).of_kind(
            ArtifactKind.OBJECT_LABELS
        )

    @property
    def image_inputs(self) -> tuple[ArtifactSpec, ...]:
        return ArtifactSpecCollection(self.special_input_specs).of_kind(
            ArtifactKind.IMAGE
        )

    def runtime_value(
        self,
        spec: ArtifactSpec,
        semantics: CellProfilerSpecialInputPayloadSemantics,
    ) -> Any:
        return CellProfilerSpecialInputValueStrategy.for_enum_member(
            semantics
        ).runtime_input_value(
            RuntimeArtifactInputRequest(
                spec=spec,
                adapter=self.adapter,
                current_image=self.current_image,
                external_image_names=self.external_image_names,
                external_object_names=self.external_object_names,
                runtime_image_names=self.runtime_image_names,
            )
        )


class CellProfilerSpecialInputPolicy(
    ABC,
    metaclass=CellProfilerModulePolicyMeta,
):
    """Nominal module-specific binding for CellProfiler special_inputs."""

    @classmethod
    @lru_cache(maxsize=None)
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


class CropSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Bind Crop side inputs without making them primary image domains."""

    module_name = "Crop"

    def special_image_inputs(
        self,
        module_name: str,
        func: Callable[..., Any],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func
        image_inputs = ArtifactSpecCollection(declared_inputs).of_kind(
            ArtifactKind.IMAGE
        )
        return image_inputs[1:]

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> dict[str, Any]:
        image_inputs = request.image_inputs
        object_inputs = request.object_inputs
        if len(image_inputs) > 1:
            raise NotImplementedError(
                f"{request.module_name} supports at most one image mask input; "
                f"got {[spec.name for spec in image_inputs]}."
            )
        if len(object_inputs) > 1:
            raise NotImplementedError(
                f"{request.module_name} supports at most one object mask input; "
                f"got {[spec.name for spec in object_inputs]}."
            )
        bound: dict[str, Any] = {}
        if image_inputs:
            bound["mask_plane"] = _runtime_input_value(image_inputs[0], request)
        if object_inputs:
            bound["cropping_labels"] = request.label_payload_for(object_inputs[0])
        return bound


class WatershedSpecialInputBindingStrategy(
    EnumKeyedStrategyMixin[WatershedMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Bind Watershed special input roles for one nominal watershed method."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedMethod | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def bind(
        self,
        request: SpecialInputBindingRequest,
        image_inputs: tuple[ArtifactSpec, ...],
    ) -> dict[str, Any]:
        """Return callable kwargs for declared Watershed special image roles."""

    def _runtime_image_value(
        self,
        spec: ArtifactSpec,
        request: SpecialInputBindingRequest,
        semantics: CellProfilerSpecialInputPayloadSemantics = (
            CellProfilerSpecialInputPayloadSemantics.INTENSITY_IMAGE
        ),
    ) -> Any:
        return request.runtime_value(spec, semantics)


class MarkerWatershedSpecialInputBindingStrategy(WatershedSpecialInputBindingStrategy):
    """Marker mode consumes marker labels first and an optional mask second."""

    strategy_key = WatershedMethod.MARKERS

    def bind(
        self,
        request: SpecialInputBindingRequest,
        image_inputs: tuple[ArtifactSpec, ...],
    ) -> dict[str, Any]:
        if not image_inputs:
            return {}
        bound = {
            WatershedInputKeyword.MARKERS.value: self._runtime_image_value(
                image_inputs[0],
                request,
                CellProfilerSpecialInputPayloadSemantics.DENSE_LABEL_IMAGE,
            )
        }
        if len(image_inputs) > 1:
            bound[WatershedInputKeyword.MASK.value] = self._runtime_image_value(
                image_inputs[1],
                request,
            )
        return bound


class MaskedWatershedSpecialInputBindingStrategy(WatershedSpecialInputBindingStrategy):
    """Non-marker modes consume their special image as a mask."""

    strategy_key = WatershedMethod.DISTANCE

    def bind(
        self,
        request: SpecialInputBindingRequest,
        image_inputs: tuple[ArtifactSpec, ...],
    ) -> dict[str, Any]:
        if not image_inputs:
            return {}
        return {
            WatershedInputKeyword.MASK.value: self._runtime_image_value(
                image_inputs[0],
                request,
            )
        }


class IntensityWatershedSpecialInputBindingStrategy(
    MaskedWatershedSpecialInputBindingStrategy
):
    strategy_key = WatershedMethod.INTENSITY


class WatershedSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Bind optional Watershed marker/mask images without making them primary domains."""

    module_name = "Watershed"

    def special_image_inputs(
        self,
        module_name: str,
        func: Callable[..., Any],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func
        return ArtifactSpecCollection(declared_inputs).of_kind(ArtifactKind.IMAGE)[1:]

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> dict[str, Any]:
        image_inputs = request.image_inputs
        watershed_method = (
            WatershedMethod.DISTANCE
            if request.kwargs.get("watershed_method") is None
            else coerce_cellprofiler_enum(
                WatershedMethod,
                request.kwargs["watershed_method"],
            )
        )
        return WatershedSpecialInputBindingStrategy.for_enum_member(
            watershed_method
        ).bind(request, image_inputs)


class StraightenWormsSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Resolve worm labels plus producer-derived control points."""

    module_name = "StraightenWorms"

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> dict[str, Any]:
        object_inputs = request.object_inputs
        _require_exact_object_count(request.module_name, object_inputs, 1)
        measurement_inputs = ArtifactSpecCollection(request.runtime_inputs).of_kind(
            ArtifactKind.MEASUREMENTS
        )
        bound: dict[str, Any] = {
            "worm_labels": request.labels_for(object_inputs[0]),
        }
        if not measurement_inputs:
            return bound
        if len(measurement_inputs) > 1:
            raise NotImplementedError(
                f"{request.module_name} supports one producer measurement "
                f"input; got {[spec.name for spec in measurement_inputs]}."
            )
        num_control_points = int(request.kwargs.get("num_control_points", 21))
        control_points = control_points_from_worm_measurement_rows(
            _runtime_input_value(measurement_inputs[0], request),
            num_control_points=num_control_points,
            object_name=object_inputs[0].name,
        )
        if control_points is not None:
            bound["control_points"] = control_points
        return bound


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
            "measurements": (
                CellProfilerObjectMeasurementVectorBinding.for_object_input(
                    request,
                    object_spec=object_spec,
                    feature_name=feature_name,
                    labels=labels,
                )
                .vector()
                .slice_aligned_value
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
                    CellProfilerObjectMeasurementVectorBinding.for_object_input(
                        request,
                        object_spec=object_spec,
                        feature_name=_classification_rule_measurement_feature(
                            rule,
                            request.module_name,
                        ),
                        labels=labels,
                    )
                    .vector()
                    .slice_aligned_value
                    for rule in rules
                ),
            }
        return {
            "labels": labels,
            **{
                parameter_name: (
                    CellProfilerObjectMeasurementVectorBinding.for_object_input(
                        request,
                        object_spec=object_spec,
                        feature_name=_required_string_kwarg(
                            request.kwargs,
                            kwarg_name,
                            request.module_name,
                        ),
                        labels=labels,
                    )
                    .vector()
                    .slice_aligned_value
                )
                for parameter_name, kwarg_name in (
                    type(self).measurement_kwarg_by_parameter.items()
                )
                if kwarg_name in request.kwargs
            },
        }


class ClassifyObjectsSingleMeasurementInputPolicy(
    ClassifyObjectsMeasurementInputPolicy
):
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
    image_inputs = ArtifactSpecCollection(declared_inputs).of_kind(ArtifactKind.IMAGE)
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


def _calculate_math_image_operand_values_by_slice(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
) -> tuple[np.ndarray, ...] | None:
    query = MeasurementFeatureQuery(
        feature_name,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )
    values_by_slice: dict[int, list[float]] = {}
    unindexed_values: list[float] = []
    for table in measurement_tables:
        for row in measurement_rows((table,)):
            row_mapping = measurement_row_mapping(row)
            if measurement_row_has_object_identity(row_mapping):
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
        if special_input_names_from_callable(request.func):
            return False
        if any(
            spec.kind is not ArtifactKind.MEASUREMENTS
            for spec in CellProfilerCallableOutputSpecs.callable_returned_output_specs(
                request.func
            )
        ):
            return False
        measurement_outputs = ArtifactSpecCollection(request.outputs).of_kind(
            ArtifactKind.MEASUREMENTS
        )
        if len(measurement_outputs) != 1:
            return False
        if len(request.outputs) != len(measurement_outputs):
            return False
        return not _callable_accepts_composed_image_payload(request.func)


class CellProfilerDualScopeMeasurementPolicy(
    ABC,
    metaclass=CellProfilerModulePolicyMeta,
):
    """Nominal policy for modules whose `Both` scope emits image and object facts."""

    image_function_name: ClassVar[str | None] = None

    @classmethod
    @lru_cache(maxsize=None)
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
class DualScopeMeasurementPolicySpec(GeneratedLeafClassSpec):
    """Declarative leaf spec for one dual-scope measurement module."""

    module_name: str
    image_function_name: str

    def class_attributes(self) -> Mapping[str, object]:
        return {
            "module_name": self.module_name,
            "image_function_name": self.image_function_name,
        }


class DeclaredDualScopeMeasurementPolicy(CellProfilerDualScopeMeasurementPolicy):
    """Generated base for modules with image+object measurement scope."""


for _dual_scope_policy_spec in (
    DualScopeMeasurementPolicySpec(
        class_name="MeasureTextureDualScopeMeasurementPolicy",
        base_type=DeclaredDualScopeMeasurementPolicy,
        module_name="MeasureTexture",
        image_function_name="measure_texture",
    ),
    DualScopeMeasurementPolicySpec(
        class_name="MeasureColocalizationDualScopeMeasurementPolicy",
        base_type=DeclaredDualScopeMeasurementPolicy,
        module_name="MeasureColocalization",
        image_function_name="measure_colocalization",
    ),
):
    _dual_scope_policy_spec.declare_in(globals())


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
    parameters = _callable_parameters(func)
    return any(
        parameter_name in parameters
        for parameter_name in _COMPOSED_IMAGE_PAYLOAD_PARAMETERS
    )


def _bind_special_runtime_inputs(
    request: SpecialInputBindingRequest,
) -> dict[str, Any]:
    if len(request.parameter_names) != len(request.special_input_specs):
        raise NotImplementedError(
            f"{request.module_name} declares special_inputs "
            f"{list(request.parameter_names)}, but compiled runtime inputs are "
            f"{[spec.name for spec in request.special_input_specs]}."
        )
    return {
        parameter_name: _runtime_input_value(spec, request)
        for parameter_name, spec in zip(
            request.parameter_names,
            request.special_input_specs,
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
    if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[0] == 1:
        return value[0]
    if is_color_image_stack(value) and value.shape[0] == 1:
        return value[0]
    if isinstance(value, tuple):
        return tuple(_collapse_singleton_stack_output(item) for item in value)
    return value


def _collapse_singleton_mask(mask: Any) -> Any:
    if isinstance(mask, np.ndarray) and mask.ndim == 3 and mask.shape[0] == 1:
        return mask[0]
    if isinstance(mask, np.ndarray) and mask.ndim == 4 and mask.shape[0] == 1:
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
    memory_type = detect_memory_type(input_data)
    stacked = ImageStackLayout.stack_function_result_for_input_stack(
        output_data,
        input_stack=input_data,
        memory_type=memory_type,
        gpu_id=0,
    )
    stacked_mask = (
        ImageStackLayout.stack_function_result_for_input_stack(
            output_mask,
            input_stack=input_data,
            memory_type=memory_type,
            gpu_id=0,
        )
        if output_mask is not None
        else output_mask
    )
    stacked_mask = project_image_mask_to_data_domain(stacked_mask, stacked)
    return image_payload_with_context(
        stacked,
        mask=stacked_mask,
        metadata=output_metadata,
    )


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


class CellProfilerPure2DOutputAggregator(ABC, metaclass=AutoRegisterMeta):
    """Aggregate one per-slice CellProfiler output position."""

    __registry_key__ = "output_type"
    __registry__: ClassVar[dict[Any, type["CellProfilerPure2DOutputAggregator"]]] = {}
    output_type: ClassVar[type[Any] | None] = None

    @classmethod
    def aggregate(
        cls,
        slice_outputs: Sequence[Any],
        memory_type: str,
    ) -> Any:
        for aggregator_type in cls.registered_aggregator_families():
            if aggregator_type.supports(slice_outputs):
                return aggregator_type().aggregate_outputs(slice_outputs, memory_type)
        return Pure2DAuxiliaryOutputAggregator.aggregate(
            list(slice_outputs), memory_type
        )

    @classmethod
    def supports(cls, slice_outputs: Sequence[Any]) -> bool:
        """Return whether this aggregator owns the output payload type."""
        accepted_types = cls.accepted_output_types()
        return (
            bool(slice_outputs)
            and bool(accepted_types)
            and all(isinstance(output, accepted_types) for output in slice_outputs)
        )

    @classmethod
    def registered_aggregator_families(
        cls,
    ) -> tuple[type["CellProfilerPure2DOutputAggregator"], ...]:
        """Return registered aggregators plus typed family bases in MRO order."""
        family_types: list[type[CellProfilerPure2DOutputAggregator]] = []
        for aggregator_type in cls.__registry__.values():
            for candidate_type in aggregator_type.mro():
                if (
                    candidate_type is cls
                    or not isinstance(candidate_type, type)
                    or not issubclass(candidate_type, cls)
                    or candidate_type in family_types
                ):
                    continue
                family_types.append(candidate_type)
        return tuple(family_types)

    @classmethod
    def accepted_output_types(cls) -> tuple[type[Any], ...]:
        """Return nominal output types owned by this aggregator family."""
        return tuple(
            aggregator_type.output_type
            for aggregator_type in CellProfilerPure2DOutputAggregator.__registry__.values()
            if (
                aggregator_type.output_type is not None
                and issubclass(aggregator_type, cls)
            )
        )

    @abstractmethod
    def aggregate_outputs(
        self,
        slice_outputs: Sequence[Any],
        memory_type: str,
    ) -> Any:
        """Aggregate one output position across pure-2D slices."""


class ObjectLabelValuePure2DOutputAggregator(CellProfilerPure2DOutputAggregator):
    """Aggregate typed object-label outputs."""

    output_type = None

    def aggregate_outputs(
        self,
        slice_outputs: Sequence[Any],
        memory_type: str,
    ) -> Any:
        return ObjectLabelPure2DSliceAggregator.aggregate(slice_outputs, memory_type)


class ImagePayloadPure2DOutputAggregator(CellProfilerPure2DOutputAggregator):
    """Aggregate typed image payload outputs."""

    output_type = None

    def aggregate_outputs(
        self,
        slice_outputs: Sequence[Any],
        memory_type: str,
    ) -> Any:
        return _stack_cellprofiler_slice_outputs(slice_outputs, memory_type)


class CellProfilerImagePayloadOutputTypes:
    """Nominal image-payload ownership derived from registered output aggregators."""

    @classmethod
    def owns(cls, value: Any) -> bool:
        accepted_types = ImagePayloadPure2DOutputAggregator.accepted_output_types()
        return bool(accepted_types) and isinstance(value, accepted_types)


class ParentChildRelationshipPure2DOutputAggregator(CellProfilerPure2DOutputAggregator):
    """Aggregate typed parent-child relationship outputs."""

    output_type = ParentChildRelationshipPayload

    def aggregate_outputs(
        self,
        slice_outputs: Sequence[Any],
        memory_type: str,
    ) -> Any:
        del memory_type
        return ParentChildRelationshipPayload(
            parent_ids=tuple(
                parent_id for output in slice_outputs for parent_id in output.parent_ids
            ),
            child_ids=tuple(
                child_id for output in slice_outputs for child_id in output.child_ids
            ),
            slice_indices=tuple(
                slice_index
                for slice_index, output in enumerate(slice_outputs)
                for _child_id in output.child_ids
            ),
            slice_count=len(slice_outputs),
        )


@dataclass(frozen=True, slots=True)
class Pure2DOutputAggregatorSpec(GeneratedLeafClassSpec):
    """Declarative leaf spec for one pure-2D output aggregator."""

    output_type: type[object]

    def class_attributes(self) -> Mapping[str, object]:
        return {"output_type": self.output_type}


for _pure_2d_output_aggregator_spec in (
    Pure2DOutputAggregatorSpec(
        "ObjectLabelPayloadPure2DOutputAggregator",
        ObjectLabelValuePure2DOutputAggregator,
        ObjectLabelPayload,
    ),
    Pure2DOutputAggregatorSpec(
        "ObjectLabelSetPure2DOutputAggregator",
        ObjectLabelValuePure2DOutputAggregator,
        ObjectLabelSet,
    ),
    Pure2DOutputAggregatorSpec(
        "MaskedImagePayloadPure2DOutputAggregator",
        ImagePayloadPure2DOutputAggregator,
        MaskedImagePayload,
    ),
    Pure2DOutputAggregatorSpec(
        "ImageMetadataPayloadPure2DOutputAggregator",
        ImagePayloadPure2DOutputAggregator,
        ImageMetadataPayload,
    ),
    Pure2DOutputAggregatorSpec(
        "NumPyImagePure2DOutputAggregator",
        ImagePayloadPure2DOutputAggregator,
        np.ndarray,
    ),
):
    _pure_2d_output_aggregator_spec.declare_in(globals())


def _stack_cellprofiler_slice_outputs(
    slice_outputs: Sequence[Any],
    memory_type: str,
) -> Any:
    normalized_outputs = tuple(
        _collapse_singleton_stack_output(output) for output in slice_outputs
    )
    output_masks = tuple(image_payload_mask(output) for output in normalized_outputs)
    output_data = tuple(image_payload_data(output) for output in normalized_outputs)
    try:
        stacked = ImageStackLayout.stack_slices_or_single_stack(
            slices=output_data,
            memory_type=memory_type,
            gpu_id=0,
        )
    except ValueError as exc:
        raise ValueError(
            "CellProfiler slice outputs must share a registered OpenHCS image "
            "stack layout; got shapes "
            f"{[output.shape if isinstance(output, np.ndarray) else None for output in output_data]!r}."
        ) from exc
    return _with_stacked_output_context(
        stacked,
        normalized_outputs,
        output_masks,
        memory_type,
    )


def _unstack_cellprofiler_image_slices(image: Any, memory_type: str) -> tuple[Any, ...]:
    image_data = image_payload_data(image)
    image_mask = image_payload_mask(image)
    image_metadata = image_payload_metadata(image)
    if is_pairwise_slice_grid(image_data):
        pairwise_shape = image_data.shape
        image_data = collapse_pairwise_slice_grid(image_data)
        if (
            image_mask is not None
            and isinstance(image_mask, np.ndarray)
            and image_mask.shape[:2] == pairwise_shape[:2]
        ):
            image_mask = collapse_pairwise_slice_grid(image_mask)
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
    if (
        plane_stack := RuntimeSliceProjection.grayscale_plane_stack_view(
            image_data,
            flatten_high_rank=True,
        )
    ) is not None:
        return tuple(
            _image_payload_slice(plane_stack[index], image_mask, image_metadata, index)
            for index in range(plane_stack.shape[0])
        )
    return tuple(
        _image_payload_slice(slice_data, image_mask, image_metadata, index)
        for index, slice_data in enumerate(
            ImageStackLayout.for_stack(image_data).unstack(
                array=image_data,
                memory_type=memory_type,
                gpu_id=0,
            )
        )
    )


def _image_payload_slice(data: Any, mask: Any | None, metadata: Any, index: int) -> Any:
    return ImagePayloadSliceProjector(mask=mask, metadata=metadata).payload_for_slice(
        data,
        index,
    )


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
        else ImageStackLayout.for_slices(present_masks).stack(
            slices=present_masks,
            memory_type=memory_type,
            gpu_id=0,
        )
    )
    return image_payload_with_context(stacked, mask=stacked_mask, metadata=metadata)


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
        function_name = CallableContract.from_callable(func).function_name
        mode_started_at = time.perf_counter()
        mode = requested_image_execution_mode(
            force_full_stack=force_full_stack,
            execution_mode=execution_mode,
        )
        _log_module_profile(
            "cp_executor_mode_resolution",
            time.perf_counter() - mode_started_at,
            function=function_name,
            mode=mode.value,
        )
        strategy_started_at = time.perf_counter()
        strategy = CellProfilerImageExecutionStrategy.for_mode(mode)
        _log_module_profile(
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
                self,
                func,
                image,
                kwargs,
            )
        _log_module_profile(
            "cp_executor_strategy_execute",
            time.perf_counter() - execute_started_at,
            function=function_name,
            mode=mode.value,
        )
        return result

    def _execute_pure_3d(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        function_name = CallableContract.from_callable(func).function_name
        projection_started_at = time.perf_counter()
        projected_image = project_singleton_stack_image_domain(image)
        projected_kwargs = {
            key: project_singleton_stack_image_domain(value)
            for key, value in kwargs.items()
        }
        _log_module_profile(
            "cp_full_stack_project_domains",
            time.perf_counter() - projection_started_at,
            function=function_name,
        )
        call_started_at = time.perf_counter()
        result = _CELLPROFILER_RUNTIME_CALLABLE_POLICY.call(
            func,
            (projected_image,),
            projected_kwargs,
        )
        _log_module_profile(
            "cp_full_stack_raw_call",
            time.perf_counter() - call_started_at,
            function=function_name,
        )
        return result

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
            Pure2DSliceIndexProjector.project(
                _collapse_singleton_stack_output(
                    _CELLPROFILER_RUNTIME_CALLABLE_POLICY.call(
                        func,
                        (slice_payload,),
                        aligned_image_stack_kwargs(
                            kwargs,
                            slice_index,
                            len(image.slices),
                            reference_payload=slice_payload,
                        ),
                    )
                ),
                slice_index,
            )
            for slice_index, slice_payload in enumerate(image.slices)
        )
        result_batch = Pure2DSliceResultBatch.from_results(slice_results)
        memory_type = detect_memory_type(
            image_payload_data(result_batch.main_outputs[0])
        )
        stacked_main_output = CellProfilerPure2DOutputAggregator.aggregate(
            result_batch.main_outputs,
            memory_type,
        )
        if not result_batch.auxiliary_groups:
            return stacked_main_output
        return (
            stacked_main_output,
            *(
                CellProfilerPure2DOutputAggregator.aggregate(values, memory_type)
                for values in result_batch.auxiliary_groups
            ),
        )

    def _execute_pure_2d(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        function_name = CallableContract.from_callable(func).function_name
        image_data = image_payload_data(image)
        if not isinstance(image_data, np.ndarray):
            return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.call(
                func,
                (image,),
                kwargs,
            )

        prepare_started_at = time.perf_counter()
        memory_type = detect_memory_type(image_data)
        if image_data.ndim == 2:
            slice_count = _slice_count_from_pure_2d_kwargs(kwargs)
            if slice_count is None:
                return _CELLPROFILER_RUNTIME_CALLABLE_POLICY.call(
                    func,
                    (image,),
                    kwargs,
                )
            slices_2d = tuple(image for _ in range(slice_count))
        elif is_color_image_slice(image_data):
            slice_count = _slice_count_from_pure_2d_kwargs(kwargs)
            slices_2d = tuple(image for _ in range(slice_count or 1))
        else:
            slices_2d = _unstack_cellprofiler_image_slices(image, memory_type)
        _log_module_profile(
            "cp_pure_2d_prepare_slices",
            time.perf_counter() - prepare_started_at,
            function=function_name,
            slices=len(slices_2d),
        )

        slice_count = len(slices_2d)
        slice_execute_seconds = 0.0
        batch_executor = _pure_2d_batch_executor(func)
        if batch_executor is not None and slice_count > 1:
            slice_started_at = time.perf_counter()
            slice_results = batch_executor(
                RuntimePure2DSliceBatchRequest(
                    func=func,
                    slices_2d=tuple(slices_2d),
                    kwargs=kwargs,
                    execute_slice=_execute_pure_2d_slice,
                )
            )
            slice_execute_seconds = time.perf_counter() - slice_started_at
        else:
            slice_results = []
            for slice_index, slice_2d in enumerate(slices_2d):
                slice_started_at = time.perf_counter()
                slice_results.append(
                    _execute_pure_2d_slice(
                        func,
                        slice_2d,
                        kwargs,
                        slice_index,
                        slice_count,
                    )
                )
                slice_execute_seconds += time.perf_counter() - slice_started_at
        _log_module_profile(
            "cp_pure_2d_slice_execute",
            slice_execute_seconds,
            function=function_name,
            slices=slice_count,
        )
        aggregate_started_at = time.perf_counter()
        result_batch = Pure2DSliceResultBatch.from_results(slice_results)
        stacked_main_output = CellProfilerPure2DOutputAggregator.aggregate(
            result_batch.main_outputs,
            memory_type,
        )
        if not result_batch.auxiliary_groups:
            result = stacked_main_output
        else:
            result = (
                stacked_main_output,
                *(
                    CellProfilerPure2DOutputAggregator.aggregate(
                        values,
                        memory_type,
                    )
                    for values in result_batch.auxiliary_groups
                ),
            )
        _log_module_profile(
            "cp_pure_2d_aggregate_outputs",
            time.perf_counter() - aggregate_started_at,
            function=function_name,
            auxiliary_groups=len(result_batch.auxiliary_groups),
        )
        return result

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
        result_2d = _CELLPROFILER_RUNTIME_CALLABLE_POLICY.call(
            func,
            (image,),
            kwargs,
        )
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


def _pure_2d_batch_executor(
    func: Callable[..., Any],
) -> (
    Callable[
        [
            RuntimePure2DSliceBatchRequest,
        ],
        list[Any],
    ]
    | None
):
    executor = CallableContract.from_callable(func).runtime_batch_executor(
        RuntimeBatchExecutionDomain.PURE_2D_SLICES
    )
    return (
        executor if callable(executor) else Pure2DSliceBatchExecutor.default_executor()
    )


def _execute_pure_2d_slice(
    func: Callable[..., Any],
    slice_2d: Any,
    kwargs: Mapping[str, Any],
    slice_index: int,
    slice_count: int,
) -> Any:
    sliced_kwargs = _slice_pure_2d_kwargs(kwargs, slice_index, slice_count)
    return Pure2DSliceIndexProjector.project(
        _CELLPROFILER_RUNTIME_CALLABLE_POLICY.call(
            func,
            (slice_2d,),
            sliced_kwargs,
        ),
        slice_index,
    )


def _execute_runtime_batch_invocation(
    func: Callable[..., Any],
    request: RuntimeBatchInvocationRequest,
) -> Any:
    """Execute one invocation from a core runtime batch request."""
    return _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
        func,
        request.image,
        request.kwargs,
        execution_mode=request.execution_mode,
    )


def _processing_contract_for_callable(func: Callable[..., Any]) -> ProcessingContract:
    cached = _PROCESSING_CONTRACT_CACHE.get(func)
    if cached is not None:
        return cached
    contract = CallableContract.from_callable(func)
    if isinstance(contract.processing_contract, ProcessingContract):
        return _cache_processing_contract(func, contract.processing_contract)
    absorbed_contract = coerce_registered_absorbed_processing_contract(
        contract.function_name,
        func,
    )
    if absorbed_contract is not None:
        return _cache_processing_contract(func, absorbed_contract)
    raise TypeError(
        f"CellProfiler executable {contract.function_name!r} has no nominal "
        "__processing_contract__ metadata. Coerce the absorbed catalog contract "
        "before runtime execution."
    )


def _cache_processing_contract(
    func: Callable[..., Any],
    contract: ProcessingContract,
) -> ProcessingContract:
    _PROCESSING_CONTRACT_CACHE[func] = contract
    return contract


def _slice_pure_2d_kwargs(
    kwargs: Mapping[str, Any],
    slice_index: int,
    slice_count: int,
) -> dict[str, Any]:
    return RuntimeSliceProjection.kwargs_for_slice(
        kwargs,
        slice_index,
        slice_count,
        sequence_kwargs=_OBJECT_ROW_SEQUENCE_KWARGS,
    )


def _sparse_ijv_array(value: Any) -> np.ndarray:
    if not isinstance(value, SparseIJVLabelRows):
        return np.asarray(value, dtype=np.int32)
    return np.asarray(value.as_array(), dtype=np.int32)


def _slice_count_from_pure_2d_kwargs(
    kwargs: Mapping[str, Any],
) -> int | None:
    if _runtime_profile_enabled():
        _log_pure_2d_slice_count_candidates(kwargs)

    return RuntimeSliceProjection.slice_count_from_values(kwargs.values())


def _log_pure_2d_slice_count_candidates(kwargs: Mapping[str, Any]) -> None:
    """Log cheap provenance for PURE_2D slice-count arbitration."""
    for name, value in kwargs.items():
        stack_fields = []
        for stack in RuntimeSliceProjection.stack_views(value):
            stack_fields.append(
                f"{type(stack).__name__}:{tuple(getattr(stack, 'shape', ())) }"
            )
        runtime_count = (
            value.slice_count
            if isinstance(value, RuntimeSliceAlignedValueSet)
            else None
        )
        relationship_count = (
            RuntimeSliceProjection.relationship_slice_count(value)
            if isinstance(value, (ParentChildRelationshipPayload, ObjectRelationship))
            else None
        )
        measurement_count = RuntimeSliceProjection.measurement_table_slice_count(value)
        data = image_payload_data(value)
        _log_module_profile(
            "cp_pure_2d_slice_count_candidate",
            0.0,
            kwarg=name,
            value_type=type(value).__name__,
            data_type=type(data).__name__,
            data_shape=getattr(data, "shape", None),
            stacks="|".join(stack_fields) or None,
            runtime_count=runtime_count,
            relationship_count=relationship_count,
            measurement_count=measurement_count,
        )


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


_CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR = CellProfilerFunctionContractExecutor()
