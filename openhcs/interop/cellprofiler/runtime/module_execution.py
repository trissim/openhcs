"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import (
    asdict,
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from enum import Enum
from functools import lru_cache
from inspect import Parameter, get_annotations, signature, unwrap
import json
import logging
import os
import time
from types import MappingProxyType
from typing import ClassVar, get_args, get_origin, get_type_hints
import numpy as np
from python_introspect import (
    Enableable,
    mark_enableable,
    parameter_exclusions,
    set_parameter_exclusions,
    set_signature_analysis_target,
)
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    AlignedImageSliceContext,
    ImageArrayShapeSemantics,
    ImagePayloadExecutionMode,
    ImagePayloadSliceProjector,
    aligned_image_stack_kwargs,
    compose_aligned_image_payload,
    payload_slice_count,
    project_singleton_stack_image_domain,
)
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactType,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    NoMainFlowOutput,
    SpatialGridArtifactType,
)
from openhcs.core.callable_contract import (
    CallableContract,
    attach_callable_contract_metadata,
    prepare_processing_callable,
)
from openhcs.core.config import DtypeConfig
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.function_reference_rehydration import (
    FunctionReferenceRehydrationRequest,
    FunctionReferenceRehydrator,
)
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
from openhcs.core.image_stack_layout import (
    ImageStackLayout,
    ImageStackLayoutUnstackRequest,
)
from openhcs.core.measurement_image_alignment import (
    prepare_measurement_image_alignment_strategies,
)
from openhcs.core.module_artifact_contract import (
    ModuleArtifactContract,
    module_artifact_contract,
)
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
from openhcs.core.pipeline.function_contracts import (
    ImagePayloadConsumption,
    ObjectLabelMeasurementExecution,
    Pure2DSliceBatchExecutor,
    RuntimeBatchExecutionDomain,
    RuntimePure2DSliceBatchRequest,
    image_payload_consumption_from_callable,
    object_label_measurement_execution_from_callable,
    runtime_bound_parameter_names_from_callable,
)
from openhcs.core.runtime_adapters import RuntimeAdapterRequest, runtime_adapter
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.runtime_profile import RuntimeProfileTimer
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_output_matching import RuntimeReturnedOutputMatcher
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.measurement_row_materialization import (
    MeasurementRowOwnership,
    measurement_object_label,
    measurement_row_object_name,
    measurement_row_source_image_name,
)
from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureQuery,
    measurement_values_for_feature,
)
from openhcs.core.measurement_lookup_dialect import runtime_measurement_lookup_dialect
from openhcs.core.special_outputs import (
    SpecialOutputKindClassifier,
    special_output_name,
)
from openhcs.core.source_bindings import SourceBindingOrigin
from openhcs.core.runtime_semantics import (
    DenseObjectLabelPairAligner,
    FieldSpec,
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
    MeasurementRowAxisState,
    MeasurementScope,
    MeasurementScopeSelection,
    ObjectLabelDomain,
    ObjectLabelDomainMetadata,
    ObjectLabelDomainMetadataStrategy,
    ObjectLabelDomainScope,
    ObjectLabelIdDomainStrategy,
    ObjectLabelMeasurementValues,
    ObjectLabelPlaneDomainStrategy,
    ObjectLabelRepresentation,
    ObjectLabelVariant,
    ObjectMeasurementVectorDomain,
    ParentChildRelationshipPayload,
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    dense_object_label_id_domain,
    measurement_row_axis_field_names,
    measurement_row_mapping,
    parent_child_relationship_artifact_endpoints,
    parent_child_relationship_artifact_name,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    DenseObjectLabelPlaneDomainStack,
    DenseObjectLabelSliceStack,
    DenseObjectLabelSliceStackRequest,
    DerivedImagePayloadContext,
    ImagePayloadMetadataCarrier,
    ImagePayloadMetadataInput,
    MeasurementTable,
    ObjectLabelData,
    ObjectLabelDenseDataStrategy,
    ObjectLabelMeasurementPayloadStrategy,
    ObjectLabelPayload,
    ObjectLabelPure2DSliceAggregator,
    ObjectLabelReplacementRequest,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelSet,
    ObjectLabelSourcePlaneProjectionRequest,
    ObjectLabelValue,
    ObjectRelationship,
    SingletonObjectLabelStackCollapseStrategy,
    SourceImageObjectLabelBuildRequest,
    SourceImageObjectLabelDomainRequest,
    SourceImagePlaneAxisPolicy,
    SourceImagePlaneAxisRequest,
    SparseIJVLabelRows,
    object_label_dense_array,
)
from openhcs.core.runtime_values import (
    DerivedImagePayloadContext,
    ImageMetadataPayload,
    ImagePayloadMetadataCompositionMode,
    ImagePayloadMetadataCompositionRequest,
    MaskedImagePayload,
    RuntimeArrayPayload,
    SpatialGrid,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_slice_context,
    normalize_image_payload_intensity,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    NominalTypeKeyedStrategyMixin,
    StrategyLabelRegistryMixin,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    Pure2DAuxiliaryOutputAggregator,
    Pure2DSliceResultBatch,
)
from openhcs.processing.materialization import tabular_field_names_from_materialization
from openhcs.processing.backends.cellprofiler.texture import measure_texture_objects
from openhcs.processing.backends.cellprofiler.library import (
    canonical_module_name,
    require_function,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageExecutionContext,
    CellProfilerImageRequest,
    CellProfilerInvocationRequest,
    CellProfilerMeasurementImage,
    CellProfilerMeasurementImageDomain,
    CellProfilerResolvedInputRequest,
    CellProfilerSliceAlignedValues,
    CellProfilerSourceImagePair,
    CellProfilerSourcePairFeature,
    requested_image_execution_mode,
)
from openhcs.interop.cellprofiler.runtime.image_payload_collapse import (
    SINGLETON_STACK_OUTPUT_COLLAPSE,
)
from openhcs.interop.cellprofiler.runtime.pure2d_output_aggregation import (
    CellProfilerPure2DOutputAggregator,
    ImagePayloadPure2DOutputAggregator,
    _unstack_cellprofiler_image_slices,
)
from openhcs.interop.cellprofiler.runtime.image_execution_strategies import (
    AlignedMultiImageStackExecutionStrategy,
    CellProfilerImageExecutionStrategy,
    FullStackImageExecutionStrategy,
    NaturalImageExecutionStrategy,
)
from openhcs.interop.cellprofiler.runtime.function_contract_execution import (
    CellProfilerFunctionContractExecutor,
    CellProfilerFunctionOutputAggregationContract,
    _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR,
)
from openhcs.interop.cellprofiler.runtime.main_flow import (
    CELLPROFILER_MEASUREMENT_MAIN_FLOW,
    CELLPROFILER_SIDE_EFFECT_MAIN_FLOW,
    CellProfilerMainFlowReplacementPolicy,
    cellprofiler_recorded_image_main_flow_output,
)
from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
    measurement_source_name_for_specs,
    measurement_row_source_names_required,
    single_source_name,
)
from openhcs.interop.cellprofiler.runtime.measurement_image_sources import (
    CellProfilerImageMeasurementSource,
    ProducedArtifactImageMeasurementSource,
    UnqualifiedRuntimeImageMeasurementSource,
)
from openhcs.interop.cellprofiler.runtime.measurement_image_resolver import (
    CellProfilerMeasurementImageResolver,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    ObjectInputBindingRequest,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    CellProfilerObjectInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_execution import (
    CellProfilerObjectMeasurementExecutionDomainPolicy,
    CellProfilerObjectMeasurementLabelArgumentPolicy,
    CellProfilerObjectMeasurementLabelArgumentRequest,
    CellProfilerPerObjectMeasurementPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_only_reference_image import (
    OBJECT_ONLY_REFERENCE_IMAGE,
    ObjectOnlyReferenceImagePolicy,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
    CompactMeasuredObjectMeasurementRowPolicy,
    DefaultObjectMeasurementRowPolicy,
    ObjectMeasurementInvocation,
)
from openhcs.interop.cellprofiler.runtime.object_label_source_projection import (
    CurrentImageObjectLabelPlaneAlignment,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_completion import (
    ObjectMeasurementAxisKey,
    ObjectMeasurementAxisOrder,
    ObjectMeasurementConcreteRowKeys,
    ObjectMeasurementIdSet,
    ObjectMeasurementIdsByAxis,
    ObjectMeasurementIdsByAxisView,
    ObjectMeasurementRowsByName,
    ObjectMeasurementSliceRowKeys,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ConcatenatedMeasurementColumnarRows,
    ObjectLabelCountAuthority,
    ObjectLocationMeasurementRows,
    _measurement_rows_from_output,
    _measurement_object_name,
    _split_cellprofiler_output,
    measurement_table_rows,
)
from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
    CellProfilerRuntimeProfiler,
    ObjectMeasurementOutputRecorder,
    ObjectMeasurementOutputTimings,
    PerImageMeasurementProfile,
    PreparedObjectMeasurementInvocation,
    PreparedObjectMeasurementInvocationBatch,
    object_measurement_batch_group_key,
)
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    measurement_record_for_module,
)
from openhcs.interop.cellprofiler.runtime.output_recording import (
    CellProfilerOutputRecordingPlan,
    CellProfilerOutputRecorder,
)
from openhcs.interop.cellprofiler.runtime.output_value_resolution import (
    CellProfilerCallableOutputSpecs,
    CellProfilerResolvedOutputValues,
)
from openhcs.interop.cellprofiler.runtime.output_contexts import (
    CellProfilerImageOutputContextStrategy,
    CellProfilerImageOutputSourcePayloadPolicy,
    CellProfilerImageOutputValuePolicy,
    CellProfilerObjectLabelOutputContextStrategy,
    CellProfilerObjectLabelOutputSourceContextPolicy,
)
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    cellprofiler_profile_payload_fields,
)
from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
    RelationshipEndpointResolver,
    RelationshipMeasurementRows,
)
from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
    _MISSING_MEASUREMENT_OBJECT_NAME,
    CellProfilerMeasurementFieldSchema,
    CellProfilerMeasurementMaterializer,
    CellProfilerMeasurementOutputValue,
    CellProfilerMeasurementOutputValues,
    CellProfilerMeasurementRecord,
    CellProfilerMeasurementSourceContext,
    CellProfilerMeasurementSourcePayload,
    CellProfilerProjectedMeasurementRow,
    MeasurementRowColumnarMaterialization,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerClassAttributeDict,
    CellProfilerClassAttributes,
    CellProfilerFunction,
    CellProfilerKwargDict,
    CellProfilerKwargs,
    CellProfilerMutableClassNamespace,
    CellProfilerOptionalFunction,
    CellProfilerProfileFields,
    CellProfilerRuntimeType,
    CellProfilerRuntimeTypeOrNone,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    CellProfilerRuntimeValueSequence,
    MeasurementJsonRow,
    MeasurementJsonRows,
    MeasurementObjectName,
    MeasurementRowsInput,
    MissingObjectMeasurementCellValue,
)
from openhcs.interop.cellprofiler.runtime.processing_contracts import (
    CellProfilerProcessingContractAuthority,
    RuntimeShapeInspection,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileEvent,
    CellProfilerRuntimeProfileLogger,
)
from openhcs.interop.cellprofiler.runtime.runtime_plane_kwargs import (
    CurrentRuntimePlaneKwargProjection,
    CurrentRuntimePlaneKwargProjectionContract,
)
from openhcs.interop.cellprofiler.runtime.runtime_special_values import (
    CellProfilerSpecialInputKwargs,
    CellProfilerSpecialInputValue,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    CellProfilerMeasurementFeature,
    count_feature_object_name,
)
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import (
    object_measurement_tables_for_object,
)
from openhcs.interop.cellprofiler.runtime.source_candidates import (
    CellProfilerImageNumberResolver,
)
from openhcs.interop.cellprofiler.runtime.source_binding_runtime import (
    PipelineStartSourceFileLoader,
    SourceBindingResolver,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    CallableObjectLabelInputContract,
    ExternalImageArtifactInputOriginStrategy,
    ImageArtifactInputOriginStrategy,
    ImageArtifactTypeStrategy,
    MeasurementsArtifactTypeStrategy,
    NoSourceImageArtifactTypeStrategy,
    ObjectLabelsArtifactTypeStrategy,
    RelationshipsArtifactTypeStrategy,
    RuntimeArtifactBindingScope,
    RuntimeArtifactInputRequest,
    RuntimeArtifactTypeStrategy,
    RuntimeImageArtifactInputOriginStrategy,
    RuntimeInputBindingRequestBase,
    SpatialGridArtifactTypeStrategy,
    StoredImageArtifactInputOriginStrategy,
    _callable_parameters,
    _callable_type_hints,
    cellprofiler_image_payload,
)
from openhcs.interop.cellprofiler.runtime.binding_authorities import (
    CellProfilerInvocationOverrideKwarg,
)
from openhcs.interop.cellprofiler.runtime.execution_mode_policies import (
    CellProfilerInvocationExecutionModePolicy,
)
from openhcs.interop.cellprofiler.runtime.dual_scope_measurement_policies import (
    CellProfilerDualScopeMeasurementPolicy,
)
from openhcs.interop.cellprofiler.runtime.primary_image_input_policies import (
    CellProfilerPrimaryImageInputPolicy,
    DefaultPrimaryImageInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.special_input_policies import (
    CellProfilerSpecialInputPolicy,
    SpecialInputBindingRequest,
)

logger = logging.getLogger(__name__)


def _enum_annotation_type(
    parameter: Parameter, resolved_annotation: CellProfilerRuntimeValue = None
) -> type[Enum] | None:
    """Return the enum type accepted by one callable parameter, if any."""
    annotation = (
        resolved_annotation if resolved_annotation is not None else parameter.annotation
    )
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    return None


def cellprofiler_runtime_adapter_factory(
    request: RuntimeAdapterRequest,
) -> CellProfilerRuntimeAdapter:
    """Build a CellProfiler adapter for one FunctionStep invocation."""
    return CellProfilerRuntimeAdapter(
        runtime_value_store=request.context.runtime_value_store,
        artifact_inputs=request.artifact_inputs,
        artifact_outputs=request.artifact_outputs,
        source_binding_plan=request.source_binding_plan,
        source_binding_context=request.source_binding_context,
        group_key=request.group_key,
        axis_scope=request.axis_scope,
        plane_projection=request.plane_projection,
        variable_components=request.variable_components,
        source_load_plan=request.source_load_plan,
        processing_context=request.context,
        filename_parser=request.context.microscope_handler.parser,
        filemanager=request.context.filemanager,
        output_identity_cache=request.context.runtime_function_output_identity_cache,
    )


def prepare_cellprofiler_runtime_adapter(request: RuntimeAdapterRequest) -> None:
    """Prepare CellProfiler source resolution during compile preparation."""
    cellprofiler_runtime_adapter_factory(request).prepare_source_resolution()


@dataclass(frozen=True, slots=True)
class CellProfilerModuleContractResolution:
    """Normalize callable construction contract inputs to artifact contracts."""

    contract: ModuleArtifactContract

    def resolve(self) -> ModuleArtifactContract:
        if isinstance(self.contract, ModuleArtifactContract):
            return self.contract
        raise TypeError(
            f"cellprofiler_module_callable contract must be ModuleArtifactContract, got {type(self.contract).__name__}."
        )


def _attach_runtime_processing_contract(
    func: CellProfilerFunction, processing_contract: ProcessingContract
) -> None:
    """Attach the declaration-owned processing contract to a runtime callable."""
    namespace = vars(func)
    key = FunctionContractAttribute.processing_contract
    existing = namespace.get(key)
    if isinstance(existing, ProcessingContract) and existing is not processing_contract:
        raise ValueError(
            f"CellProfiler callable {func.__name__!r} declares {existing.name}, but runtime binding provided {processing_contract.name}."
        )
    namespace[key] = processing_contract


class CellProfilerRuntimeCallable:
    """Picklable callable wrapper for one artifact-managed CellProfiler module."""

    def __init__(
        self,
        raw_func: CellProfilerFunction,
        contract: ModuleArtifactContract,
        *,
        declared_processing_contract: str | None = None,
        processing_contract: ProcessingContract,
    ) -> None:
        from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog

        try:
            raw_func = CellProfilerFunctionCatalog.get_function(raw_func.__name__)
        except KeyError:
            pass
        _attach_runtime_processing_contract(raw_func, processing_contract)
        self.raw_func = raw_func
        self.contract = contract
        self.executor = CellProfilerModuleExecutor(contract)
        self.declared_processing_contract = declared_processing_contract
        self.processing_contract = processing_contract
        raw_contract = CallableContract.from_callable(raw_func)
        self.__name__ = raw_func.__name__
        self.__qualname__ = raw_func.__qualname__
        self.__module__ = raw_func.__module__
        self.__doc__ = raw_func.__doc__
        self.__signature__ = _cellprofiler_runtime_callable_signature(raw_func)
        self.__annotations__ = get_annotations(raw_func, eval_str=False)
        if raw_contract.input_memory_type is not None:
            self.input_memory_type = raw_contract.input_memory_type
        if raw_contract.output_memory_type is not None:
            self.output_memory_type = raw_contract.output_memory_type
        vars(self)[FunctionContractAttribute.processing_contract] = processing_contract
        module_artifact_contract(contract)(self)
        analysis_func = raw_contract.raw_processing_function or raw_func
        set_signature_analysis_target(self, analysis_func)
        mark_enableable(self)
        runtime_plan = self.executor.runtime_plan(raw_func)
        runtime_adapter(
            CellProfilerRuntimeAdapter.require_parameter_name(),
            cellprofiler_runtime_adapter_factory,
            manages_artifact_inputs=True,
            prepare=prepare_cellprofiler_runtime_adapter,
        )(self)
        set_parameter_exclusions(
            self,
            (
                *parameter_exclusions(raw_func),
                CellProfilerRuntimeAdapter.require_parameter_name(),
                RuntimeInvocationOptions.require_parameter_name(),
                *runtime_plan.bound_parameter_names,
            ),
        )
        attach_callable_contract_metadata(
            self,
            declared_processing_contract=declared_processing_contract,
            raw_processing_function=raw_func,
            prepare=self.prepare_runtime_callable,
            runtime_image_execution_mode=raw_contract.runtime_image_execution_mode,
        )

    def __call__(
        self,
        image: CellProfilerRuntimeValue,
        *,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        runtime_invocation_options: CellProfilerRuntimeValue | None = None,
        enabled: bool = True,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        if not enabled:
            return image
        return self.executor.run(
            self.raw_func,
            image,
            cellprofiler_runtime=cellprofiler_runtime,
            invocation_options=runtime_invocation_options,
            **kwargs,
        )

    def __reduce__(self) -> tuple[CellProfilerFunction, CellProfilerRuntimeValues]:
        return (
            rebuild_cellprofiler_runtime_callable,
            (
                self.raw_func,
                self.contract,
                self.declared_processing_contract,
                self.processing_contract,
            ),
        )

    def __eq__(self, other: CellProfilerRuntimeValue) -> bool:
        """Compare runtime callables by their nominal CellProfiler module binding."""
        if not isinstance(other, CellProfilerRuntimeCallable):
            return NotImplemented
        return (
            self.raw_func == other.raw_func
            and self.contract == other.contract
            and (
                self.declared_processing_contract == other.declared_processing_contract
            )
            and (self.processing_contract == other.processing_contract)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.raw_func,
                self.contract,
                self.declared_processing_contract,
                self.processing_contract,
            )
        )

    def prepare_runtime_callable(self) -> None:
        prepare_processing_callable(self.raw_func)
        self.executor.prepare(self.raw_func)


def _cellprofiler_runtime_callable_signature(raw_func: CellProfilerFunction):
    """Return raw callable signature plus OpenHCS runtime injection parameters."""
    runtime_parameters = (
        Parameter(
            CellProfilerRuntimeAdapter.require_parameter_name(),
            Parameter.KEYWORD_ONLY,
            annotation=CellProfilerRuntimeAdapter,
        ),
        Parameter(
            RuntimeInvocationOptions.require_parameter_name(),
            Parameter.KEYWORD_ONLY,
            annotation=CellProfilerRuntimeValue | None,
            default=None,
        ),
        Enableable.parameter(),
    )
    raw_signature = signature(raw_func)
    existing_names = frozenset(raw_signature.parameters)
    injected = [
        parameter
        for parameter in runtime_parameters
        if parameter.name not in existing_names
    ]
    if not injected:
        return raw_signature
    parameters = list(raw_signature.parameters.values())
    variadic_keyword_index = len(parameters)
    for index, parameter in enumerate(parameters):
        if parameter.kind is Parameter.VAR_KEYWORD:
            variadic_keyword_index = index
            break
    parameters[variadic_keyword_index:variadic_keyword_index] = injected
    return raw_signature.replace(parameters=parameters)


def rebuild_cellprofiler_runtime_callable(
    raw_func: CellProfilerFunction,
    contract: ModuleArtifactContract,
    declared_processing_contract: str | None,
    processing_contract: ProcessingContract,
) -> CellProfilerRuntimeCallable:
    """Rebuild a pickled CellProfiler runtime callable."""
    return CellProfilerRuntimeCallable(
        raw_func,
        contract,
        declared_processing_contract=declared_processing_contract,
        processing_contract=processing_contract,
    )


@dataclass(frozen=True, slots=True)
class CellProfilerRuntimeStepBinding:
    """Runtime wrapper binding for an already-declared generated FunctionStep."""

    raw_callable: CellProfilerFunction
    contract: ModuleArtifactContract
    processing_contract: ProcessingContract
    declared_processing_contract: str | None

    def load(self) -> CellProfilerFunction:
        """Return the artifact-managed runtime callable for a generated step."""
        return cellprofiler_module_callable(
            self.raw_callable,
            self.contract,
            declared_processing_contract=self.declared_processing_contract,
            processing_contract=self.processing_contract,
        )


def cellprofiler_module_callable(
    raw_func: CellProfilerFunction,
    contract: ModuleArtifactContract,
    *,
    declared_processing_contract: str | None = None,
    processing_contract: ProcessingContract,
) -> CellProfilerFunction:
    """Build the product-owned runtime callable for one CellProfiler module."""
    if not callable(raw_func):
        raise TypeError(
            f"cellprofiler_module_callable raw_func must be callable, got {type(raw_func).__name__}."
        )
    resolved_contract = CellProfilerModuleContractResolution(contract).resolve()
    return CellProfilerRuntimeCallable(
        raw_func,
        resolved_contract,
        declared_processing_contract=declared_processing_contract,
        processing_contract=processing_contract,
    )


class CellProfilerFunctionReferenceRehydrator(FunctionReferenceRehydrator):
    """Rebuild generated CellProfiler runtime callables from preserved contracts."""

    rehydrator_key = "cellprofiler"

    def supports(self, request: FunctionReferenceRehydrationRequest) -> bool:
        contract = request.contract
        return (
            contract.module_artifact_contract is not None
            and callable(contract.raw_processing_function)
            and (contract.runtime_adapter is not None)
            and (
                contract.runtime_adapter.require_parameter_name()
                == CellProfilerRuntimeAdapter.require_parameter_name()
            )
        )

    def rehydrate(
        self, request: FunctionReferenceRehydrationRequest
    ) -> CellProfilerFunction:
        contract = request.contract
        processing_contract = contract.processing_contract
        if not isinstance(processing_contract, ProcessingContract):
            raise TypeError(
                "CellProfiler function reference rehydration requires declared ProcessingContract metadata."
            )
        return cellprofiler_module_callable(
            contract.raw_processing_function,
            contract.module_artifact_contract,
            declared_processing_contract=contract.declared_processing_contract,
            processing_contract=processing_contract,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerModuleRuntimePlan:
    """Static runtime decisions for one CellProfiler module callable."""

    contract: ModuleArtifactContract
    module_type: type[CellProfilerModule] | None
    func: CellProfilerFunction
    function_name: str
    callable_contract: CallableContract
    kwarg_spec: "CallableInvocationKwargSpec"
    declared_input_specs: tuple[ArtifactSpec, ...]
    declared_input_collection: ArtifactSpecCollection
    primary_image_input_policy: "CellProfilerPrimaryImageInputPolicy"
    primary_image_inputs: tuple[ArtifactSpec, ...]
    primary_image_source_aliases: tuple[str, ...]
    runtime_image_name_set: frozenset[str]
    external_primary_image_names: tuple[str, ...]
    runtime_inputs: tuple[ArtifactSpec, ...]
    object_inputs: tuple[ArtifactSpec, ...]
    object_label_inputs: tuple[ArtifactSpec, ...]
    measurement_outputs: tuple[ArtifactSpec, ...]
    image_outputs: tuple[ArtifactSpec, ...]
    declared_output_specs: tuple[ArtifactSpec, ...]
    binding_scope: RuntimeArtifactBindingScope
    object_input_policy: CellProfilerObjectInputPolicy
    special_input_policy: "CellProfilerSpecialInputPolicy"
    invocation_execution_mode_policy: "CellProfilerInvocationExecutionModePolicy"
    main_flow_replacement_policy: CellProfilerMainFlowReplacementPolicy
    object_measurement_row_policy: CellProfilerObjectMeasurementRowPolicy
    dual_scope_measurement_policy: "CellProfilerDualScopeMeasurementPolicy | None"
    dual_scope_image_function: CellProfilerFunction | None
    dual_scope_image_kwarg_spec: "CallableInvocationKwargSpec | None"
    special_input_names: tuple[str, ...]
    supported_non_object_input_kinds: frozenset[ArtifactType]
    output_recording_plan: CellProfilerOutputRecordingPlan
    image_output_value_policy: CellProfilerImageOutputValuePolicy
    image_output_source_payload_policy: CellProfilerImageOutputSourcePayloadPolicy
    object_label_output_source_context_policy: (
        CellProfilerObjectLabelOutputSourceContextPolicy
    )
    runs_per_image_measurement: bool
    runs_per_object_measurement: bool
    replaces_main_flow: bool

    @property
    def bound_parameter_names(self) -> tuple[str, ...]:
        """Return callable parameters supplied by runtime artifact declarations."""
        return tuple(
            dict.fromkeys(
                (
                    *self.special_input_policy.bound_parameter_names(self),
                    *self.object_input_policy.bound_parameter_names(self),
                    *self.callable_contract.runtime_bound_parameters,
                    *runtime_bound_parameter_names_from_callable(self.func),
                )
            )
        )

    @property
    def runtime_slice_sequence_parameter_names(self) -> frozenset[str]:
        """Return bound tuple parameters projected item-wise during pure-2D slicing."""
        return frozenset(
            self.object_input_policy.runtime_slice_sequence_parameter_names(self)
        )

    @property
    def measurement_table_parameter_names(self) -> frozenset[str]:
        """Return bound parameters carrying measurement-table collections."""
        return frozenset(
            self.object_input_policy.measurement_table_parameter_names(self)
        )

    @property
    def records_only_measurements(self) -> bool:
        """Return whether every declared output is a measurement artifact."""
        return bool(self.measurement_outputs) and len(self.declared_output_specs) == len(
            self.measurement_outputs
        )

    @property
    def records_measurements_without_image_outputs(self) -> bool:
        """Return whether measurement artifact recording owns this module's output."""
        return self.records_only_measurements and not self.image_outputs

    @property
    def publishes_side_effect_main_flow(self) -> bool:
        """Return whether source-bound inputs may become the next main-flow image."""
        return not self.records_measurements_without_image_outputs

    @classmethod
    def build(
        cls,
        *,
        contract: ModuleArtifactContract,
        canonical_module_name: str,
        primary_image_input_policy: "CellProfilerPrimaryImageInputPolicy",
        func: CellProfilerFunction,
        processing_contract: ProcessingContract,
    ) -> "CellProfilerModuleRuntimePlan":
        declared_input_specs = contract.declared_input_specs()
        declared_input_collection = contract.declared_input_collection()
        callable_contract = CallableContract.from_callable(func)
        special_input_policy = CellProfilerSpecialInputPolicy.for_module(
            canonical_module_name
        )
        primary_image_inputs = primary_image_input_policy.primary_image_inputs(
            contract.module_name,
            func,
            declared_input_specs,
            special_input_policy=special_input_policy,
        )
        runtime_image_name_set = contract.runtime_input_name_set(ImageArtifactType)
        non_image_inputs = tuple(
            (
                spec
                for spec in declared_input_specs
                if spec.artifact_type is not ImageArtifactType
            )
        )
        special_image_inputs = special_input_policy.special_image_inputs(
            contract.module_name, func, declared_input_specs
        )
        runtime_inputs = (*non_image_inputs, *special_image_inputs)
        special_input_names = special_input_names_from_callable(func)
        object_input_policy = CellProfilerObjectInputPolicy.for_module(
            canonical_module_name
        )
        object_label_inputs = declared_input_collection.of_artifact_type(
            ObjectLabelsArtifactType
        )
        object_input_policy.validate_runtime_plan_object_inputs(
            module_name=contract.module_name,
            object_label_inputs=object_label_inputs,
            special_input_names=special_input_names,
        )
        output_collection = contract.output_collection()
        image_outputs = output_collection.of_artifact_type(ImageArtifactType)
        measurement_outputs = output_collection.of_artifact_type(
            MeasurementsArtifactType
        )
        module_type = CellProfilerModule.for_module(canonical_module_name)
        object_measurement_row_policy = (
            module_type.runtime_object_measurement_row_policy()
            if module_type is not None
            else DefaultObjectMeasurementRowPolicy()
        )
        dual_scope_policy = CellProfilerDualScopeMeasurementPolicy.for_module(
            canonical_module_name
        )
        dual_scope_image_function = (
            None
            if dual_scope_policy is None
            else dual_scope_policy.image_function(func)
        )
        dual_scope_image_kwarg_spec = (
            None
            if dual_scope_image_function is None
            else CallableInvocationKwargSpec.from_callable_contract(
                dual_scope_image_function, processing_contract
            )
        )
        return cls(
            contract=contract,
            module_type=module_type,
            func=func,
            function_name=callable_contract.function_name,
            callable_contract=callable_contract,
            kwarg_spec=CallableInvocationKwargSpec.from_callable_contract(
                func, processing_contract
            ),
            declared_input_specs=declared_input_specs,
            declared_input_collection=declared_input_collection,
            primary_image_input_policy=primary_image_input_policy,
            primary_image_inputs=primary_image_inputs,
            primary_image_source_aliases=ArtifactSpecCollection(
                primary_image_inputs
            ).names(),
            runtime_image_name_set=runtime_image_name_set,
            external_primary_image_names=tuple(
                (
                    spec.name
                    for spec in primary_image_inputs
                    if spec.name not in runtime_image_name_set
                )
            ),
            runtime_inputs=runtime_inputs,
            object_inputs=ArtifactSpecCollection(runtime_inputs).of_artifact_type(
                ObjectLabelsArtifactType
            ),
            object_label_inputs=object_label_inputs,
            measurement_outputs=measurement_outputs,
            image_outputs=image_outputs,
            declared_output_specs=contract.declared_outputs,
            binding_scope=RuntimeArtifactBindingScope.from_contract(contract),
            object_input_policy=object_input_policy,
            special_input_policy=special_input_policy,
            invocation_execution_mode_policy=CellProfilerInvocationExecutionModePolicy.for_module(
                canonical_module_name
            ),
            main_flow_replacement_policy=CellProfilerMainFlowReplacementPolicy.for_module(
                canonical_module_name
            ),
            object_measurement_row_policy=object_measurement_row_policy,
            dual_scope_measurement_policy=dual_scope_policy,
            dual_scope_image_function=dual_scope_image_function,
            dual_scope_image_kwarg_spec=dual_scope_image_kwarg_spec,
            special_input_names=special_input_names,
            supported_non_object_input_kinds=object_input_policy.supported_non_object_input_kinds,
            output_recording_plan=CellProfilerOutputRecordingPlan.from_outputs(
                contract.outputs
            ),
            image_output_value_policy=CellProfilerImageOutputValuePolicy.for_module(
                canonical_module_name
            ),
            image_output_source_payload_policy=CellProfilerImageOutputSourcePayloadPolicy.for_module(
                canonical_module_name
            ),
            object_label_output_source_context_policy=CellProfilerObjectLabelOutputSourceContextPolicy.for_module(
                canonical_module_name
            ),
            runs_per_image_measurement=CellProfilerPerImageMeasurementPolicy.matches(
                CellProfilerPerImageMeasurementRequest(
                    module_name=contract.module_name,
                    func=func,
                    image_inputs=primary_image_inputs,
                    object_inputs=object_label_inputs,
                    outputs=contract.outputs,
                )
            ),
            runs_per_object_measurement=CellProfilerPerObjectMeasurementPolicy.matches(
                contract.module_name, object_label_inputs
            ),
            replaces_main_flow=CellProfilerMainFlowReplacementPolicy.for_module(
                canonical_module_name
            ).replaces_main_flow(image_outputs),
        )

    @property
    def default_runtime_image_execution_mode(self) -> ImagePayloadExecutionMode | None:
        return self.callable_contract.runtime_image_execution_mode

    def runtime_batch_executor(
        self, domain: RuntimeBatchExecutionDomain
    ) -> Callable | None:
        return self.callable_contract.runtime_batch_executor(domain)

    def function_output_aggregation_contract(
        self,
    ) -> CellProfilerFunctionOutputAggregationContract:
        return CellProfilerFunctionOutputAggregationContract.from_main_flow_replacement(
            self.replaces_main_flow, declared_output_specs=self.declared_output_specs
        )


@dataclass(frozen=True, slots=True)
class CellProfilerModuleRunRequest:
    """Immutable execution-path request for one CellProfiler module call."""

    executor: "CellProfilerModuleExecutor"
    func: CellProfilerFunction
    plan: CellProfilerModuleRuntimePlan
    input_image: CellProfilerRuntimeValue
    current_image: CellProfilerRuntimeValue
    adapter: CellProfilerRuntimeAdapter
    invocation_options: RuntimeInvocationOptions | None
    kwargs: CellProfilerKwargDict

    @property
    def module_name(self) -> str:
        return self.executor.module_name

    @property
    def function_name(self) -> str:
        return self.plan.function_name


@dataclass(slots=True)
class CellProfilerModuleRunProfile:
    """Run-level profile checkpoints for one CellProfiler module execution."""

    module_name: str
    function_name: str
    enabled: bool
    run_started_at: float
    checkpoint_started_at: float

    @classmethod
    def start(
        cls, *, module_name: str, function_name: str
    ) -> "CellProfilerModuleRunProfile":
        timer = RuntimeProfileTimer.start()
        return cls(
            module_name=module_name,
            function_name=function_name,
            enabled=timer.enabled,
            run_started_at=timer.started_at,
            checkpoint_started_at=timer.started_at,
        )

    def checkpoint(self, event: str) -> None:
        """Record elapsed time since the previous checkpoint."""
        if not self.enabled:
            return
        now = time.perf_counter()
        CellProfilerRuntimeProfileLogger.log_module_profile(
            event,
            now - self.checkpoint_started_at,
            module=self.module_name,
            function=self.function_name,
        )
        self.checkpoint_started_at = now

    def checkpoint_deferred(
        self, event: str, fields: Callable[[], Mapping[str, CellProfilerRuntimeValue]]
    ) -> None:
        """Record a checkpoint whose fields are expensive to construct."""
        if not self.enabled:
            return
        now = time.perf_counter()
        CellProfilerRuntimeProfileLogger.log_module_profile_deferred(
            event, now - self.checkpoint_started_at, fields
        )
        self.checkpoint_started_at = now

    def total(self) -> None:
        """Record total module runtime."""
        if not self.enabled:
            return
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_module_run_total",
            time.perf_counter() - self.run_started_at,
            module=self.module_name,
            function=self.function_name,
        )


class CellProfilerModuleExecutionPath(ABC):
    """Nominal strategy for one CellProfiler module execution path."""

    @classmethod
    def for_plan(
        cls, plan: CellProfilerModuleRuntimePlan
    ) -> "CellProfilerModuleExecutionPath":
        for path_type in (
            PerImageMeasurementExecutionPath,
            PerObjectMeasurementExecutionPath,
            StandardImageExecutionPath,
        ):
            path = path_type()
            if path.matches(plan):
                return path
        raise RuntimeError(f"No CellProfiler execution path for {plan.function_name}.")

    @abstractmethod
    def matches(self, plan: CellProfilerModuleRuntimePlan) -> bool:
        """Return whether this path owns execution for the runtime plan."""

    @abstractmethod
    def execute(
        self,
        request: CellProfilerModuleRunRequest,
        profile: CellProfilerModuleRunProfile,
    ) -> CellProfilerRuntimeValue:
        """Execute the module through this path."""


class PerImageMeasurementExecutionPath(CellProfilerModuleExecutionPath):
    """Execute image-scoped measurement modules once per source image."""

    def matches(self, plan: CellProfilerModuleRuntimePlan) -> bool:
        return plan.runs_per_image_measurement

    def execute(
        self,
        request: CellProfilerModuleRunRequest,
        profile: CellProfilerModuleRunProfile,
    ) -> CellProfilerRuntimeValue:
        profile.checkpoint("cp_runs_per_image_check")
        result = request.executor._run_per_image_measurement(request)
        profile.checkpoint("cp_run_per_image_measurement")
        profile.total()
        return result


class PerObjectMeasurementExecutionPath(CellProfilerModuleExecutionPath):
    """Execute object-scoped measurement modules across object/image domains."""

    def matches(self, plan: CellProfilerModuleRuntimePlan) -> bool:
        return plan.runs_per_object_measurement

    def execute(
        self,
        request: CellProfilerModuleRunRequest,
        profile: CellProfilerModuleRunProfile,
    ) -> CellProfilerRuntimeValue:
        profile.checkpoint("cp_runs_per_image_check")
        image_request = request.executor._runtime_image_request(
            request.plan, request.current_image, request.adapter
        )
        profile.checkpoint("cp_image_request")
        profile.checkpoint("cp_runs_per_object_check")
        result = request.executor._run_per_object_measurement(
            request, image_request=image_request
        )
        profile.checkpoint("cp_run_per_object_measurement")
        profile.total()
        return result


class StandardImageExecutionPath(CellProfilerModuleExecutionPath):
    """Execute ordinary image-transforming CellProfiler modules."""

    def matches(self, plan: CellProfilerModuleRuntimePlan) -> bool:
        del plan
        return True

    def execute(
        self,
        request: CellProfilerModuleRunRequest,
        profile: CellProfilerModuleRunProfile,
    ) -> CellProfilerRuntimeValue:
        profile.checkpoint("cp_runs_per_image_check")
        image_request = request.executor._runtime_image_request(
            request.plan, request.current_image, request.adapter
        )
        profile.checkpoint("cp_image_request")
        profile.checkpoint("cp_runs_per_object_check")
        if image_request is None:
            raise RuntimeError(
                f"{request.module_name} image execution requires a resolved image request."
            )
        invocation = request.executor._invocation_request(
            request.plan,
            image_request=image_request,
            adapter=request.adapter,
            current_image=request.current_image,
            kwargs=request.kwargs,
            invocation_options=request.invocation_options,
        )
        profile.checkpoint("cp_invocation_request")
        raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
            request.func,
            invocation.image,
            invocation.kwargs,
            execution_mode=invocation.execution_mode,
            output_aggregation_contract=request.plan.function_output_aggregation_contract(),
            runtime_slice_sequence_parameter_names=request.plan.runtime_slice_sequence_parameter_names,
            measurement_table_parameter_names=request.plan.measurement_table_parameter_names,
        )
        profile.checkpoint_deferred(
            "cp_contract_execute",
            lambda: {
                "module": request.module_name,
                "function": request.function_name,
                **cellprofiler_profile_payload_fields("input", invocation.image),
                **cellprofiler_profile_payload_fields("output", raw_output),
            },
        )
        main_output, artifact_values = _split_cellprofiler_output(raw_output)
        profile.checkpoint("cp_split_output")
        CellProfilerOutputRecorder.record_module_outputs(
            runtime_plan=request.plan,
            adapter=request.adapter,
            main_output=main_output,
            artifact_values=artifact_values,
            invocation=invocation,
            image_request=image_request,
            current_image=request.current_image,
        )
        profile.checkpoint("cp_record_outputs")
        if not request.plan.replaces_main_flow:
            profile.checkpoint("cp_replace_main_flow_check")
            profile.total()
            if not request.plan.publishes_side_effect_main_flow:
                return NoMainFlowOutput()
            return CELLPROFILER_SIDE_EFFECT_MAIN_FLOW.output_image(
                current_image=request.current_image,
                image_request=image_request,
                variable_components=request.adapter.variable_components,
                parser=request.adapter.filename_parser,
                identity_cache=request.adapter.output_identity_cache,
            )
        result = request.executor._replacement_main_flow_output(
            request.plan,
            adapter=request.adapter,
            current_image=request.current_image,
            invocation_image=invocation.image,
            output_image=main_output,
        )
        profile.checkpoint("cp_replace_main_flow_check")
        profile.total()
        return result


@dataclass(slots=True)
class CellProfilerModuleExecutor:
    """Execute one generated CellProfiler module against a typed runtime adapter."""

    contract: ModuleArtifactContract
    _canonical_module_name: str = field(init=False, repr=False, compare=False)
    _primary_image_input_policy: "CellProfilerPrimaryImageInputPolicy" = field(
        init=False, repr=False, compare=False
    )
    _runtime_plans: dict[CellProfilerFunction, CellProfilerModuleRuntimePlan] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ModuleArtifactContract):
            raise TypeError(
                f"CellProfilerModuleExecutor.contract must be ModuleArtifactContract, got {type(self.contract).__name__}."
            )
        self._canonical_module_name = canonical_module_name(self.contract.module_name)
        self._primary_image_input_policy = (
            CellProfilerPrimaryImageInputPolicy.for_module(self._canonical_module_name)
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

    def prepare(self, func: CellProfilerFunction) -> None:
        """Resolve nominal policies used by this executor before timed execution."""
        for origin in SourceBindingOrigin:
            SourceBindingResolver.for_origin(origin)
        tuple(PipelineStartSourceFileLoader.__registry__.values())
        for mode in ImagePayloadExecutionMode:
            CellProfilerImageExecutionStrategy.for_mode(mode)
        for strategy_type in RuntimeArtifactTypeStrategy.registered_strategy_types():
            RuntimeArtifactTypeStrategy.for_artifact_type(strategy_type.artifact_type)
        prepare_measurement_image_alignment_strategies()
        plan = self.runtime_plan(func)
        for output in plan.output_recording_plan.ordered_outputs:
            plan.output_recording_plan.recorders[output.artifact_type]

    def runtime_plan(self, func: CellProfilerFunction) -> CellProfilerModuleRuntimePlan:
        """Return the prepared runtime plan for this callable and module contract."""
        plan = self._runtime_plans.get(func)
        if plan is not None:
            return plan
        plan = CellProfilerModuleRuntimePlan.build(
            contract=self.contract,
            canonical_module_name=self._canonical_module_name,
            primary_image_input_policy=self._primary_image_input_policy,
            func=func,
            processing_contract=CellProfilerProcessingContractAuthority.for_callable(
                func
            ),
        )
        self._runtime_plans[func] = plan
        return plan

    def run(
        self,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        *,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        invocation_options: RuntimeInvocationOptions | None = None,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        """Call the absorbed function and record declared outputs through the adapter."""
        plan = self.runtime_plan(func)
        request = CellProfilerModuleRunRequest(
            executor=self,
            func=func,
            plan=plan,
            input_image=image,
            current_image=image,
            adapter=cellprofiler_runtime,
            invocation_options=invocation_options,
            kwargs=dict(kwargs),
        )
        profile = CellProfilerModuleRunProfile.start(
            module_name=self.module_name, function_name=plan.function_name
        )
        return CellProfilerModuleExecutionPath.for_plan(plan).execute(request, profile)

    def _replacement_main_flow_output(
        self,
        plan: CellProfilerModuleRuntimePlan,
        *,
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerRuntimeValue,
        invocation_image: CellProfilerRuntimeValue,
        output_image: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        image_outputs = plan.image_outputs
        if len(image_outputs) > 1:
            composition = compose_aligned_image_payload(
                self.module_name,
                tuple(
                    (adapter.get_image(output.name).data for output in image_outputs)
                ),
                slice_contexts=tuple(
                    (
                        AlignedImageSliceContext.main_flow(
                            output_key=output.name,
                            artifact_kind=output.artifact_type.value,
                        )
                        for output in image_outputs
                    )
                ),
            )
            payload = composition.payload
            return payload
        return cellprofiler_recorded_image_main_flow_output(
            current_image=current_image,
            invocation_image=invocation_image,
            recorded_image=output_image,
        )

    def _run_per_object_measurement(
        self,
        request: CellProfilerModuleRunRequest,
        *,
        image_request: "CellProfilerImageRequest | None",
    ) -> CellProfilerRuntimeValue:
        func = request.func
        plan = request.plan
        current_image = request.current_image
        input_image = request.input_image
        cellprofiler_runtime = request.adapter
        source_image_name = self._source_image_name_for_measurement(image_request)
        kwargs = request.kwargs.copy()
        function_name = plan.function_name
        profiler = CellProfilerRuntimeProfiler(self.module_name, function_name)
        object_inputs = plan.object_label_inputs
        measurement_outputs = plan.measurement_outputs
        if len(measurement_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} per-object execution requires exactly one measurement output."
            )
        measurement_target_scope = self.pop_measurement_target_scope(
            kwargs, MeasurementScopeSelection.of(MeasurementScope.OBJECT)
        )
        combined_rows: list[CellProfilerRuntimeValue] = []
        profile_enabled = CellProfilerRuntimeProfileLogger.enabled()
        if profile_enabled:
            measurement_images_started_at = time.perf_counter()
        measurement_image_resolver = CellProfilerMeasurementImageResolver(self)
        measurement_images = measurement_image_resolver.measurement_image_inputs(
            func, cellprofiler_runtime, current_image, image_request
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
        if profile_enabled:
            dual_scope_started_at = time.perf_counter()
        image_measurement_rows = self._dual_scope_image_measurement_rows(
            func,
            plan,
            measurement_images,
            cellprofiler_runtime,
            kwargs,
            measurement_target_scope,
        )
        if profile_enabled:
            profile_events.append(
                CellProfilerRuntimeProfileEvent(
                    "cp_per_object_dual_scope_rows",
                    time.perf_counter() - dual_scope_started_at,
                    (("rows", len(image_measurement_rows)),),
                )
            )
        combined_rows.extend(image_measurement_rows)
        measurement_row_policy = plan.object_measurement_row_policy
        label_payload_seconds = 0.0
        label_align_seconds = 0.0
        contract_execute_seconds = 0.0
        output_timings = ObjectMeasurementOutputTimings()
        columnar_rows: list[ColumnarRows] = []
        batch_executor = plan.runtime_batch_executor(
            RuntimeBatchExecutionDomain.MEASUREMENT_IMAGES
        )
        processing_contract = CellProfilerProcessingContractAuthority.for_callable(func)
        output_recorder = ObjectMeasurementOutputRecorder(
            row_policy=measurement_row_policy,
            func=func,
            adapter=cellprofiler_runtime,
            measurement_images=measurement_images,
            object_inputs=object_inputs,
            contains_image_measurement_rows=bool(image_measurement_rows),
            combined_rows=combined_rows,
            columnar_rows=columnar_rows,
            timings=output_timings,
        )
        measurement_invocations = tuple(
            (
                measurement_row_policy.invocations(measurement_image, kwargs)
                for measurement_image in measurement_images
            )
        )
        total_measurement_batch_count = sum(
            (len(invocations) for invocations in measurement_invocations)
        ) * len(object_inputs)
        prepared_invocations: list[PreparedObjectMeasurementInvocation] = []
        for measurement_image, invocations in zip(
            measurement_images, measurement_invocations, strict=True
        ):
            for object_spec in object_inputs:
                (
                    aligned_image,
                    executable_labels,
                    completion_label_payload,
                    execution_mode,
                    preparation_profile_events,
                    label_payload_elapsed,
                    label_align_elapsed,
                ) = measurement_image_resolver.object_measurement_runtime_inputs(
                    func=func,
                    measurement_image=measurement_image,
                    object_spec=object_spec,
                    adapter=cellprofiler_runtime,
                )
                profile_events.extend(preparation_profile_events)
                label_payload_seconds += label_payload_elapsed
                label_align_seconds += label_align_elapsed
                for invocation in invocations:
                    prepared_invocations.append(
                        PreparedObjectMeasurementInvocation(
                            source_image_name=measurement_image.source_image_name,
                            execution_mode=execution_mode,
                            func=func,
                            image=aligned_image,
                            kwargs={
                                **invocation.lowered_kwargs(),
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
                            measurement_image=measurement_image,
                            object_spec=object_spec,
                            invocation=invocation,
                            completion_label_payload=completion_label_payload,
                        )
                    )
        contract_execute_seconds = PreparedObjectMeasurementInvocationBatch(
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
                        (("rows", len(combined_rows)),),
                    ),
                )
            )
        combined_source_image_name = measurement_row_policy.table_source_image_name(
            measurement_images, source_image_name
        )
        combined_source_image_payload = (
            CellProfilerMeasurementImage.shared_source_payload(measurement_images)
        )
        combined_source_metadata = (
            CellProfilerMeasurementImage.composed_source_metadata(
                measurement_images,
                mode=measurement_row_policy.source_metadata_composition_mode(
                    measurement_images
                ),
            )
        )
        if profile_enabled:
            record_started_at = time.perf_counter()
        CellProfilerMeasurementMaterializer.record_per_object(
            adapter=cellprofiler_runtime,
            spec=measurement_outputs[0],
            func=func,
            measurement_row_policy=measurement_row_policy,
            object_inputs=object_inputs,
            image_measurement_rows=image_measurement_rows,
            combined_rows=combined_rows,
            columnar_rows=columnar_rows,
            source_context=CellProfilerMeasurementSourceContext(
                source_image_name=combined_source_image_name,
                source_image_payload=combined_source_image_payload,
                source_metadata=combined_source_metadata,
            ),
        )
        if profile_enabled:
            profile_events.append(
                CellProfilerRuntimeProfileEvent(
                    "cp_per_object_record_measurements",
                    time.perf_counter() - record_started_at,
                    (
                        (
                            "rows",
                            sum((len(rows) for rows in columnar_rows))
                            + len(combined_rows),
                        ),
                    ),
                )
            )
            profiler.record_events(tuple(profile_events))
        if plan.records_measurements_without_image_outputs:
            return NoMainFlowOutput()
        return CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
            input_image=input_image,
            measurement_images=measurement_images,
            variable_components=cellprofiler_runtime.variable_components,
            parser=cellprofiler_runtime.filename_parser,
            identity_cache=cellprofiler_runtime.output_identity_cache,
        )

    def _dual_scope_image_measurement_rows(
        self,
        object_func: CellProfilerFunction,
        plan: CellProfilerModuleRuntimePlan,
        measurement_images: tuple["CellProfilerMeasurementImage", ...],
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        kwargs: CellProfilerKwargs,
        target_scope: MeasurementScopeSelection,
    ) -> list[CellProfilerRuntimeValue]:
        function_name = plan.function_name
        profiler = CellProfilerRuntimeProfiler(self.module_name, function_name)
        if not target_scope.includes_all(
            MeasurementScope.IMAGE, MeasurementScope.OBJECT
        ):
            return []
        policy = plan.dual_scope_measurement_policy
        if policy is None:
            return []
        image_func = plan.dual_scope_image_function
        image_kwarg_spec = plan.dual_scope_image_kwarg_spec
        if image_func is None or image_kwarg_spec is None:
            raise TypeError(
                f"{self.module_name} dual-scope policy did not prepare image function."
            )
        rows: list[CellProfilerRuntimeValue] = []
        row_source_names_required = measurement_row_source_names_required(
            measurement_images
        )
        image_kwargs = image_kwarg_spec.coerce_kwargs(kwargs)
        measurement_row_policy = plan.object_measurement_row_policy
        contract_execute_seconds = 0.0
        split_rows_seconds = 0.0
        for measurement_image in measurement_images:
            contract_started_at = time.perf_counter()
            raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                image_func,
                _image_scope_measurement_payload(measurement_image.payload),
                image_kwargs,
                execution_mode=measurement_image.execution_mode,
                output_aggregation_contract=plan.function_output_aggregation_contract(),
                runtime_slice_sequence_parameter_names=plan.runtime_slice_sequence_parameter_names,
                measurement_table_parameter_names=plan.measurement_table_parameter_names,
            )
            contract_execute_seconds += time.perf_counter() - contract_started_at
            split_rows_started_at = time.perf_counter()
            _ignored_main_output, artifact_values = _split_cellprofiler_output(
                raw_output
            )
            source_image_name = None
            if row_source_names_required:
                source_image_name = measurement_image.source_image_name
            owned_rows = MeasurementRowOwnership(
                source_image_name=source_image_name
            ).annotate_rows(_measurement_rows_from_output(artifact_values))
            source_pairs = measurement_image.source_image_pairs()
            if len(source_pairs) == 1:
                owned_rows = measurement_row_policy.project_rows(
                    owned_rows,
                    ObjectMeasurementInvocation(
                        kwargs={},
                        source_pair=source_pairs[0],
                    ),
                )
            elif source_pairs:
                raise ValueError(
                    f"{self.module_name} emitted image-scope source-pair "
                    "measurement rows for multiple source pairs; the dual-scope "
                    "image path must execute one source-pair invocation at a time."
                )
            projected_rows, _projected_row_mappings = (
                CellProfilerMeasurementRecord(
                    rows=owned_rows,
                    source_context=CellProfilerMeasurementSourceContext(
                        source_image_name=source_image_name,
                        source_image_payload=measurement_image.payload,
                    ),
                    object_name=None,
                )
                .projection_request(adapter=cellprofiler_runtime)
                .project_rows()
            )
            rows.extend(projected_rows)
            split_rows_seconds += time.perf_counter() - split_rows_started_at
        profiler.record("cp_dual_scope_contract_execute", contract_execute_seconds)
        profiler.record("cp_dual_scope_split_rows", split_rows_seconds, rows=len(rows))
        return rows

    def _run_per_image_measurement(
        self, request: CellProfilerModuleRunRequest
    ) -> CellProfilerRuntimeValue:
        func = request.func
        plan = request.plan
        input_image = request.input_image
        current_image = request.current_image
        cellprofiler_runtime = request.adapter
        kwargs = request.kwargs.copy()
        function_name = plan.function_name
        profiler = CellProfilerRuntimeProfiler(self.module_name, function_name)
        profile = PerImageMeasurementProfile(profiler)
        measurement_outputs = plan.measurement_outputs
        if len(measurement_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} per-image execution requires exactly one measurement output."
            )
        self.pop_measurement_target_scope(
            kwargs, MeasurementScopeSelection.of(MeasurementScope.IMAGE)
        )
        combined_rows: list[CellProfilerRuntimeValue] = []
        measurement_images_started_at = time.perf_counter()
        measurement_images = CellProfilerMeasurementImageResolver(
            self
        ).independent_measurement_image_inputs(
            func, cellprofiler_runtime, current_image
        )
        profile.measurement_images(
            time.perf_counter() - measurement_images_started_at, len(measurement_images)
        )
        kwargs_started_at = time.perf_counter()
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(
                plan, cellprofiler_runtime, current_image, kwargs
            ),
        }
        coerced_kwargs = plan.kwarg_spec.coerce_kwargs(runtime_kwargs)
        profile.prepare_kwargs(time.perf_counter() - kwargs_started_at)
        row_source_names_required = measurement_row_source_names_required(
            measurement_images
        )
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
                output_aggregation_contract=plan.function_output_aggregation_contract(),
                runtime_slice_sequence_parameter_names=plan.runtime_slice_sequence_parameter_names,
                measurement_table_parameter_names=plan.measurement_table_parameter_names,
            )
            contract_execute_seconds += time.perf_counter() - contract_started_at
            split_rows_started_at = time.perf_counter()
            main_output, artifact_values = _split_cellprofiler_output(raw_output)
            source_image_name = None
            if row_source_names_required:
                source_image_name = measurement_image.source_image_name
            resolved_values = CellProfilerResolvedOutputValues.from_returned_outputs(
                recorded_specs=measurement_outputs,
                context_specs=self.contract.declared_outputs or measurement_outputs,
                main_output=main_output,
                artifact_values=artifact_values,
                func=func,
                declared_output_specs=self.contract.declared_outputs,
            )
            measurement_record_request = CellProfilerOutputRecordRequest(
                runtime_plan=plan,
                adapter=cellprofiler_runtime,
                spec=measurement_outputs[0],
                output_value=resolved_values.recorded_value(measurement_outputs[0]),
                output_values=resolved_values.context_values,
                func=plan.func,
                function_name=plan.function_name,
                source=replace(measurement_image, source_image_name=source_image_name),
                call_kwargs=coerced_kwargs,
            )
            measurement_record = measurement_record_for_module(
                measurement_record_request
            )
            combined_records.append(measurement_record)
            projected_rows, _projected_row_mappings = (
                measurement_record.projection_request(
                    adapter=cellprofiler_runtime
                ).project_rows()
            )
            combined_rows.extend(
                MeasurementRowOwnership(
                    source_image_name=measurement_record.source_context.source_image_name
                ).annotate_rows(projected_rows)
            )
            split_rows_seconds += time.perf_counter() - split_rows_started_at
        profile.contract_execute(contract_execute_seconds)
        profile.split_rows(split_rows_seconds, len(combined_rows))
        rows_declare_object_name = (
            CellProfilerMeasurementFieldSchema.rows_declare_object_name(combined_rows)
        )
        image_measurement_object_name = _MISSING_MEASUREMENT_OBJECT_NAME
        image_measurement_source_name = (
            CellProfilerMeasurementRecord.shared_source_image_name(
                tuple(combined_records)
            )
        )
        if image_measurement_source_name is None:
            image_measurement_source_name = (
                CellProfilerMeasurementImage.shared_source_image_name(
                    measurement_images
                )
            )
        if rows_declare_object_name:
            image_measurement_object_name = None
            image_measurement_source_name = None
        record_started_at = time.perf_counter()
        CellProfilerMeasurementMaterializer.record(
            CellProfilerMeasurementRecord(
                rows=combined_rows,
                fields=CellProfilerMeasurementFieldSchema.for_record(
                    measurement_outputs[0], combined_rows, func
                ),
                object_name=image_measurement_object_name,
                source_context=CellProfilerMeasurementSourceContext(
                    source_image_name=image_measurement_source_name
                ),
            ).materialization_request(
                adapter=cellprofiler_runtime,
                name=measurement_outputs[0].name,
                axis_state=MeasurementRowAxisState.IMAGE_NUMBER,
            )
        )
        profile.record_measurements(
            time.perf_counter() - record_started_at, len(combined_rows)
        )
        if plan.records_measurements_without_image_outputs:
            return NoMainFlowOutput()
        return CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
            input_image=input_image,
            measurement_images=measurement_images,
            variable_components=cellprofiler_runtime.variable_components,
            parser=cellprofiler_runtime.filename_parser,
            identity_cache=cellprofiler_runtime.output_identity_cache,
        )

    def pop_measurement_target_scope(
        self, kwargs: CellProfilerKwargDict, default_scope: MeasurementScopeSelection
    ) -> MeasurementScopeSelection:
        """Consume the generated target-scope kwarg as OpenHCS measurement scopes."""
        from openhcs.interop.cellprofiler.measurement_scope import (
            cellprofiler_measurement_scope_selection,
        )

        return cellprofiler_measurement_scope_selection(
            kwargs.pop(
                CellProfilerInvocationOverrideKwarg.measurement_target_scope, None
            ),
            default_scope,
        )

    def _runtime_input_kwargs(
        self,
        plan: CellProfilerModuleRuntimePlan,
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        *,
        primary_image: CellProfilerRuntimeValue | None = None,
        project_object_labels_to_current_plane: bool = True,
    ) -> CellProfilerKwargDict:
        runtime_inputs = plan.runtime_inputs
        object_input_policy = plan.object_input_policy
        binding_scope = plan.binding_scope
        if not runtime_inputs:
            if object_input_policy.binds_without_declared_inputs:
                return object_input_policy.bind(
                    ObjectInputBindingRequest(
                        module_name=self.module_name,
                        func=plan.func,
                        object_inputs=(),
                        adapter=adapter,
                        kwargs=kwargs,
                        current_image=current_image,
                        binding_scope=binding_scope,
                        runtime_inputs=runtime_inputs,
                        project_object_labels_to_current_plane=project_object_labels_to_current_plane,
                    )
                )
            return {}
        special_input_names = plan.special_input_names
        if special_input_names:
            special_input_specs = runtime_inputs
            special_input_policy = plan.special_input_policy
            return special_input_policy.bind(
                SpecialInputBindingRequest(
                    module_name=self.module_name,
                    func=plan.func,
                    parameter_names=special_input_names,
                    special_input_specs=special_input_specs,
                    runtime_inputs=runtime_inputs,
                    adapter=adapter,
                    kwargs=kwargs,
                    current_image=special_input_policy.binding_current_image(
                        current_image=current_image, primary_image=primary_image
                    ),
                    binding_scope=binding_scope,
                    project_object_labels_to_current_plane=project_object_labels_to_current_plane,
                )
            )
        supported_non_object_kinds = plan.supported_non_object_input_kinds
        unsupported_non_object_inputs = tuple(
            (
                spec
                for spec in runtime_inputs
                if spec.artifact_type is not ObjectLabelsArtifactType
                and spec.artifact_type not in supported_non_object_kinds
            )
        )
        if unsupported_non_object_inputs:
            raise NotImplementedError(
                f"{self.module_name} has runtime inputs {[spec.name for spec in unsupported_non_object_inputs]} with no declared special_inputs binding."
            )
        return object_input_policy.bind(
            ObjectInputBindingRequest(
                module_name=self.module_name,
                func=plan.func,
                object_inputs=plan.object_inputs,
                adapter=adapter,
                kwargs=kwargs,
                current_image=current_image,
                binding_scope=binding_scope,
                runtime_inputs=runtime_inputs,
                project_object_labels_to_current_plane=project_object_labels_to_current_plane,
            )
        )

    def _image_request(
        self,
        plan: CellProfilerModuleRuntimePlan,
        current_image: CellProfilerRuntimeValue,
        adapter: CellProfilerRuntimeAdapter,
    ) -> "CellProfilerImageRequest":
        image_inputs = plan.primary_image_inputs
        if not image_inputs:
            current_image_payload = self._current_runtime_plane_image(
                plan, current_image, adapter
            )
            payload = (
                OBJECT_ONLY_REFERENCE_IMAGE.reference_image(current_image_payload)
                if plan.object_label_inputs
                or plan.declared_input_collection.of_artifact_type(
                    SpatialGridArtifactType
                )
                else cellprofiler_image_payload(current_image_payload)
            )
            return CellProfilerImageRequest(
                payload=payload,
                source_image_name=self._input_source_image_name(plan, adapter),
                source_aliases=(),
                image_count=1,
                execution_mode=ImagePayloadExecutionMode.NATURAL,
                projects_runtime_slice_kwargs=True,
                publishes_side_effect_main_flow=(
                    plan.publishes_side_effect_main_flow
                    and not (
                        plan.object_label_inputs
                        or plan.declared_input_collection.of_artifact_type(
                            SpatialGridArtifactType
                        )
                    )
                ),
            )
        adapter.require_resolvable_source_aliases(plan.external_primary_image_names)
        payloads = []
        source_names: list[str | None] = []
        for spec in image_inputs:
            if spec.name in plan.runtime_image_name_set:
                runtime_image = adapter.get_image(
                    spec.name,
                    current_image=self._runtime_image_current_image(
                        plan, adapter, spec, current_image
                    ),
                )
                payloads.append(cellprofiler_image_payload(runtime_image.data))
                source_names.append(runtime_image.source_image_name)
                continue
            source_names.append(spec.name)
            payloads.append(
                cellprofiler_image_payload(
                    adapter.resolve_source_image(spec.name, current_image)
                )
            )
        composition = compose_aligned_image_payload(
            self.module_name,
            tuple(payloads),
            metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
        )
        return CellProfilerImageRequest(
            payload=composition.payload,
            source_image_name=self._primary_image_source_name_from_sources(
                image_inputs, tuple(source_names)
            ),
            source_aliases=plan.primary_image_source_aliases,
            image_count=len(payloads),
            execution_mode=composition.execution_mode,
            publishes_side_effect_main_flow=plan.publishes_side_effect_main_flow,
        )

    def _requires_image_request(self, plan: CellProfilerModuleRuntimePlan) -> bool:
        if not plan.runs_per_object_measurement:
            return True
        return not CellProfilerPerObjectMeasurementPolicy.measures_images_independently(
            self.module_name
        )

    def _runtime_image_request(
        self,
        plan: CellProfilerModuleRuntimePlan,
        image: CellProfilerRuntimeValue,
        adapter: CellProfilerRuntimeAdapter,
    ) -> CellProfilerImageRequest | None:
        """Resolve the invocation image request when this execution path needs one."""
        if self._requires_image_request(plan):
            return self._image_request(plan, image, adapter)
        return None

    @staticmethod
    def _source_image_name_for_measurement(
        image_request: CellProfilerImageRequest | None,
    ) -> str | None:
        """Return the source image name carried by an optional image request."""
        if image_request is None:
            return None
        return image_request.source_image_name

    def _current_runtime_plane_image(
        self,
        plan: CellProfilerModuleRuntimePlan,
        current_image: CellProfilerRuntimeValue,
        adapter: CellProfilerRuntimeAdapter,
    ) -> CellProfilerRuntimeValue:
        default_execution_mode = (
            plan.default_runtime_image_execution_mode
            or ImagePayloadExecutionMode.NATURAL
        )
        if not CurrentRuntimePlaneKwargProjectionContract(
            plan.func, default_execution_mode
        ).projects_runtime_artifact_image_inputs():
            return current_image
        return adapter.image_payload_for_current_runtime_plane(
            current_image, current_image=current_image
        )

    def _input_source_image_name(
        self, plan: CellProfilerModuleRuntimePlan, adapter: CellProfilerRuntimeAdapter
    ) -> str | None:
        source_names: list[str] = []
        for spec in plan.declared_input_specs:
            source_name = RuntimeArtifactTypeStrategy.for_artifact_type(
                spec.artifact_type
            ).source_image_name(
                RuntimeArtifactInputRequest.from_spec(
                    spec, adapter=adapter, binding_scope=plan.binding_scope
                )
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

    def _runtime_image_current_image(
        self,
        plan: CellProfilerModuleRuntimePlan,
        adapter: CellProfilerRuntimeAdapter,
        spec: ArtifactSpec,
        current_image: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue | None:
        policy_current_image = (
            plan.primary_image_input_policy.runtime_image_current_image(
                self.module_name, spec, current_image
            )
        )
        if policy_current_image is None:
            return None
        default_execution_mode = (
            plan.default_runtime_image_execution_mode
            or ImagePayloadExecutionMode.NATURAL
        )
        if CurrentRuntimePlaneKwargProjectionContract(
            plan.func, default_execution_mode
        ).projects_runtime_artifact_image_inputs():
            return policy_current_image
        return None

    def _invocation_request(
        self,
        plan: CellProfilerModuleRuntimePlan,
        *,
        image_request: "CellProfilerImageRequest",
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: RuntimeInvocationOptions | None,
    ) -> "CellProfilerInvocationRequest":
        profile_enabled = CellProfilerRuntimeProfileLogger.enabled()
        default_execution_mode = (
            plan.default_runtime_image_execution_mode or image_request.execution_mode
        )
        projects_runtime_slice_kwargs = (
            image_request.projects_runtime_slice_kwargs
            and CurrentRuntimePlaneKwargProjectionContract(
                plan.func, default_execution_mode
            ).projects_runtime_slice_kwargs()
        )
        if profile_enabled:
            phase_started_at = time.perf_counter()
        else:
            phase_started_at = 0.0
        bound_runtime_kwargs = self._runtime_input_kwargs(
            plan,
            adapter,
            current_image,
            kwargs,
            primary_image=image_request.payload,
            project_object_labels_to_current_plane=projects_runtime_slice_kwargs,
        )
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_invocation_bind_runtime_inputs",
                time.perf_counter() - phase_started_at,
                module=self.module_name,
            )
        runtime_kwargs = {**kwargs, **bound_runtime_kwargs}
        image_override = runtime_kwargs.pop(
            CellProfilerInvocationOverrideKwarg.image, None
        )
        execution_mode_override = runtime_kwargs.pop(
            CellProfilerInvocationOverrideKwarg.execution_mode, None
        )
        if profile_enabled:
            phase_started_at = time.perf_counter()
        runtime_kwargs = plan.primary_image_input_policy.invocation_runtime_kwargs(
            module_name=self._canonical_module_name,
            plan=plan,
            image_request=image_request,
            adapter=adapter,
            current_image=current_image,
            runtime_kwargs=runtime_kwargs,
            object_input_source_image_name=lambda: self._object_input_source_image_name(
                plan, adapter
            ),
        )
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_invocation_primary_policy_kwargs",
                time.perf_counter() - phase_started_at,
                module=self.module_name,
            )
        invocation_image = (
            image_override if image_override is not None else image_request.payload
        )
        if profile_enabled:
            phase_started_at = time.perf_counter()
        runtime_kwargs = dict(
            CurrentRuntimePlaneKwargProjection(
                image=invocation_image,
                kwargs=runtime_kwargs,
                plane_projector=adapter,
                project_runtime_slice_kwargs=projects_runtime_slice_kwargs,
            ).kwargs_for_invocation()
        )
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_invocation_project_runtime_kwargs",
                time.perf_counter() - phase_started_at,
                module=self.module_name,
            )
        if profile_enabled:
            phase_started_at = time.perf_counter()
        execution_mode = plan.invocation_execution_mode_policy.execution_mode(
            default_execution_mode,
            image=invocation_image,
            kwargs=runtime_kwargs,
            invocation_options=invocation_options,
        )
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_invocation_execution_mode_policy",
                time.perf_counter() - phase_started_at,
                module=self.module_name,
            )
        if profile_enabled:
            phase_started_at = time.perf_counter()
        coerced_kwargs = plan.kwarg_spec.coerce_kwargs(runtime_kwargs)
        if profile_enabled:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_invocation_coerce_kwargs",
                time.perf_counter() - phase_started_at,
                module=self.module_name,
            )
        return CellProfilerInvocationRequest(
            image=invocation_image,
            kwargs=coerced_kwargs,
            source_image_name=image_request.source_image_name,
            image_count=image_request.image_count,
            execution_mode=execution_mode_override or execution_mode,
        )

    def _object_input_source_image_name(
        self, plan: CellProfilerModuleRuntimePlan, adapter: CellProfilerRuntimeAdapter
    ) -> str | None:
        source_names = tuple(
            (
                adapter.get_objects(spec.name).source_image_name
                for spec in plan.object_label_inputs
            )
        )
        return single_source_name(
            tuple((source_name for source_name in source_names if source_name))
        )


@dataclass(frozen=True, slots=True)
class CallableInvocationKwargSpec:
    """Cached callable kwarg contract used before CellProfiler invocation."""

    accepts_var_keyword: bool
    accepted_names: frozenset[str]
    contract_control_names: frozenset[str]
    callable_defaults: tuple[tuple[str, CellProfilerRuntimeValue], ...]
    enum_types: tuple[tuple[str, type[Enum]], ...]

    @classmethod
    @lru_cache(maxsize=256)
    def from_callable(cls, func: CellProfilerFunction) -> "CallableInvocationKwargSpec":
        return cls.from_callable_contract(
            func, CellProfilerProcessingContractAuthority.for_callable(func)
        )

    @classmethod
    @lru_cache(maxsize=256)
    def from_callable_contract(
        cls, func: CellProfilerFunction, processing_contract: ProcessingContract
    ) -> "CallableInvocationKwargSpec":
        parameters = _callable_parameters(func)
        annotations = _callable_type_hints(func)
        enum_types = tuple(
            (
                (name, enum_type)
                for name, parameter in parameters.items()
                if (
                    enum_type := _enum_annotation_type(parameter, annotations.get(name))
                )
                is not None
            )
        )
        contract = processing_contract
        return cls(
            accepts_var_keyword=any(
                (
                    parameter.kind is Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            ),
            accepted_names=frozenset(parameters),
            contract_control_names=contract.declaration.execution_parameter_names()
            | contract.declaration.injected_semantic_control_parameter_names(),
            callable_defaults=tuple(
                (
                    (name, parameter.default)
                    for name, parameter in parameters.items()
                    if parameter.default is not Parameter.empty
                )
            ),
            enum_types=enum_types,
        )

    def coerce_kwargs(self, kwargs: CellProfilerKwargs) -> CellProfilerKwargDict:
        """Filter unsupported kwargs and coerce enum-typed values."""
        if self.accepts_var_keyword:
            coerced_kwargs = dict(kwargs)
        else:
            accepted_names = self.accepted_names | self.contract_control_names
            coerced_kwargs = {
                name: value for name, value in kwargs.items() if name in accepted_names
            }
        for name, value in self.callable_defaults:
            coerced_kwargs.setdefault(name, value)
        for name, enum_type in self.enum_types:
            if name not in coerced_kwargs:
                continue
            try:
                coerced_kwargs[name] = coerce_cellprofiler_enum(
                    enum_type, coerced_kwargs[name]
                )
            except ValueError as exc:
                raise ValueError(
                    f"{name} must be coercible to {enum_type.__name__}; got {coerced_kwargs[name]!r}."
                ) from exc
        return coerced_kwargs


def _image_scope_measurement_payload(
    image: CellProfilerRuntimeValue,
) -> CellProfilerRuntimeValue:
    """Return one image plane for image-scoped measurement functions."""
    return SINGLETON_STACK_OUTPUT_COLLAPSE.collapse(image)


def _execution_mode_semantic_control_kwargs(
    processing_contract: ProcessingContract,
    execution_mode: ImagePayloadExecutionMode,
) -> CellProfilerKwargDict:
    """Return semantic controls required by a resolved image execution mode."""
    if execution_mode is not ImagePayloadExecutionMode.NATURAL:
        return {}
    return {
        name: True
        for name in (
            processing_contract.declaration.injected_semantic_control_parameter_names()
        )
    }


@dataclass(frozen=True, slots=True)
class CellProfilerPerImageMeasurementRequest:
    """Contract shape used to decide image-measurement invocation cardinality."""

    module_name: str
    func: CellProfilerFunction
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
            (
                spec.artifact_type is not MeasurementsArtifactType
                for spec in CellProfilerCallableOutputSpecs(
                    request.func
                ).artifact_specs()
            )
        ):
            return False
        measurement_outputs = ArtifactSpecCollection(request.outputs).of_artifact_type(
            MeasurementsArtifactType
        )
        if len(measurement_outputs) != 1:
            return False
        if len(request.outputs) != len(measurement_outputs):
            return False
        return (
            image_payload_consumption_from_callable(request.func)
            is not ImagePayloadConsumption.COMPOSED
        )
