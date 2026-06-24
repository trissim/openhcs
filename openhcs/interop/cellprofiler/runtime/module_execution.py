"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields as dataclass_fields, is_dataclass, replace
from enum import Enum
from functools import lru_cache
from inspect import Parameter, get_annotations, signature, unwrap
import json
import logging
import os
import time
from types import MappingProxyType
from typing import ClassVar, TypeVar, get_args, get_origin, get_type_hints

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
import numpy as np
from python_introspect import (
    mark_enableable,
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
    stack_image_payload_context,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection
from openhcs.core.callable_contract import (
    CallableContract,
    PROCESSING_CONTRACT_ATTR,
    attach_callable_contract_metadata,
    prepare_processing_callable,
)
from openhcs.core.config import DtypeConfig
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
from openhcs.core.image_stack_layout import ImageStackLayout, ImageStackLayoutUnstackRequest
from openhcs.core.measurement_image_alignment import (
    prepare_measurement_image_alignment_strategies,
)
from openhcs.core.module_artifact_contract import (
    ModuleArtifactContract,
    module_artifact_contract,
)
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    Pure2DSliceBatchExecutor,
    RuntimeBatchExecutionDomain,
    RuntimePure2DSliceBatchRequest,
    object_label_measurement_execution_from_callable,
)
from openhcs.core.runtime_adapters import RuntimeAdapterRequest, runtime_adapter
from openhcs.core.runtime_slice_alignment import (
    RuntimeSliceAlignedValueSet,
)
from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
from openhcs.core.runtime_output_matching import RuntimeReturnedOutputMatcher
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
)
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_ID_FIELD,
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
    MeasurementRowOwnership,
    measurement_object_label,
    measurement_row_object_name,
    measurement_row_source_image_name,
)
from openhcs.core.measurement_feature_queries import (
    MEASUREMENT_RESULT_VALUE_FIELD,
    MEASUREMENT_VALUE_FIELDS,
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
    MeasurementScope,
    MeasurementScopeSelection,
    FieldSpec,
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
    MeasurementRowAxisState,
    ObjectMeasurementVectorDomain,
    ObjectShapeMeasurementFeature,
    ObjectLabelDomain,
    ObjectLabelDomainMetadata,
    ObjectLabelDomainMetadataStrategy,
    measurement_row_mapping,
    ObjectLabelDomainScope,
    ObjectLabelIdDomainStrategy,
    ObjectLabelMeasurementValues,
    ObjectLabelPlaneDomainStrategy,
    ObjectLabelRepresentation,
    ObjectLabelVariant,
    ParentChildRelationshipPayload,
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    DenseObjectLabelPairAligner,
    dense_object_label_id_domain,
    measurement_row_axis_field_names,
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
    GeneratedLeafClassSpec,
    NominalTypeKeyedStrategyMixin,
    StrategyLabelRegistryMixin,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    Pure2DAuxiliaryOutputAggregator,
    Pure2DSliceResultBatch,
    RuntimeCallablePolicy,
    RuntimeCallableView,
    RuntimeInvocationKwargPolicy,
)
from openhcs.processing.materialization import tabular_field_names_from_materialization
from openhcs.processing.backends.cellprofiler.texture import (
    measure_texture_objects,
)
from openhcs.processing.backends.cellprofiler.library import (
    canonical_module_name,
    require_function,
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
    WormControlPointMeasurementSchema,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
    cellprofiler_measurement_scope_selection,
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
    _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR,
    _execute_pure_2d_slice,
    _execute_runtime_batch_invocation,
)
from openhcs.interop.cellprofiler.runtime.main_flow import (
    CELLPROFILER_MEASUREMENT_MAIN_FLOW,
    CELLPROFILER_SIDE_EFFECT_MAIN_FLOW,
    CellProfilerMainFlowReplacementPolicy,
    cellprofiler_recorded_image_main_flow_output,
)
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup
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
from openhcs.interop.cellprofiler.runtime.module_names import (
    CELLPROFILER_CONVERT_OBJECTS_TO_IMAGE_MODULE,
    CELLPROFILER_CORRECT_ILLUMINATION_CALCULATE_MODULE,
    CELLPROFILER_CROP_MODULE,
    CELLPROFILER_MEASURE_COLOCALIZATION_MODULE,
    CELLPROFILER_MEASURE_GRANULARITY_MODULE,
    CELLPROFILER_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE,
    CELLPROFILER_MEASURE_OBJECT_INTENSITY_MODULE,
    CELLPROFILER_MEASURE_OBJECT_NEIGHBORS_MODULE,
    CELLPROFILER_MEASURE_OBJECT_SIZE_SHAPE_MODULE,
    CELLPROFILER_MEASURE_TEXTURE_MODULE,
    CELLPROFILER_MASK_IMAGE_MODULE,
    CELLPROFILER_RELATE_OBJECTS_MODULE,
    CELLPROFILER_TRACK_OBJECTS_MODULE,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerObjectInputCountAuthority,
    CellProfilerMeasurementVector,
    CellProfilerObjectMeasurementVectorBatchBinding,
    CellProfilerObjectMeasurementVectorBinding,
    CellProfilerObjectMeasurementVectorSource,
    CellProfilerObjectMeasurementVectorSourceStrategy,
    CurrentObjectShapeFeatureVectorResult,
    CurrentObjectShapeFeatureVectorSourceStrategy,
    CurrentObjectShapeFeatureVectorStatus,
    MeasurementImageOperandVectorResolution,
    ObjectInputBindingRequest,
    RuntimeMeasurementsVectorSourceStrategy,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    CellProfilerObjectInputPolicy,
    IdentifySecondaryObjectsInputPolicy,
    IdentifyTertiaryObjectInputPolicy,
    SingleObjectLabelInputPolicy,
    UnsupportedObjectInputPolicy,
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
    AlignMeasurementFeature,
    AlignMeasurementRows,
    ClassifyObjectsMeasurementFeatureTemplate,
    ClassifyObjectsMeasurementRows,
    ConcatenatedMeasurementColumnarRows,
    LABEL_PAYLOAD_FINAL,
    ObjectLabelCountAuthority,
    ObjectLocationMeasurementRows,
    ThresholdMeasurementRows,
    _label_payload_small_removed,
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
    object_measurement_batch_group_key,
)
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
    CorrectIlluminationOriginalImageName,
)
from openhcs.interop.cellprofiler.runtime.output_recording import (
    CellProfilerMeasurementRecordBuilder,
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
)
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    CellProfilerSpecialInputPayloadSemantics,
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
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    MODULE_NAME_REGISTRY_KEY,
    ArtifactKindRegistryMixin,
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLeafSpec,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyMultiBaseLeafSpec,
    CellProfilerModulePolicyRegistryKey,
    EnumStrategyLabelRegistryMixin,
    NoSourceImageNameMixin,
)
from openhcs.interop.cellprofiler.runtime.processing_contracts import (
    MEASUREMENT_TABLES_BOUND_KEY,
    MEASUREMENT_VALUES_BOUND_KEY,
    OBJECT_ROW_SEQUENCE_KWARGS,
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
from openhcs.interop.cellprofiler.runtime.adapter import (
    CellProfilerRuntimeAdapter,
)
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
    ImageArtifactKindStrategy,
    MeasurementsArtifactKindStrategy,
    NoSourceImageArtifactKindStrategy,
    ObjectLabelsArtifactKindStrategy,
    RelationshipsArtifactKindStrategy,
    RuntimeArtifactBindingScope,
    RuntimeArtifactInputRequest,
    RuntimeArtifactKindStrategy,
    RuntimeImageArtifactInputOriginStrategy,
    RuntimeImageInputOrigin,
    RuntimeInputBindingRequestBase,
    SpatialGridArtifactKindStrategy,
    StoredImageArtifactInputOriginStrategy,
    _callable_parameters,
    _callable_type_hints,
    cellprofiler_image_payload,
)

_CELLPROFILER_IMAGE_OVERRIDE_KWARG = "_cellprofiler_image_override"
_CELLPROFILER_EXECUTION_MODE_OVERRIDE_KWARG = "_cellprofiler_execution_mode_override"


_SLICE_INDEX_PARAMETER = "slice_index"
_MASK_IMAGE_MODULE = CELLPROFILER_MASK_IMAGE_MODULE
_RELATE_OBJECTS_MODULE = CELLPROFILER_RELATE_OBJECTS_MODULE
_TRACK_OBJECTS_MODULE = CELLPROFILER_TRACK_OBJECTS_MODULE
logger = logging.getLogger(__name__)

RequiredAttrT = TypeVar("RequiredAttrT")


def _enum_annotation_type(
    parameter: Parameter,
    resolved_annotation: CellProfilerRuntimeValue = None,
) -> type[Enum] | None:
    """Return the enum type accepted by one callable parameter, if any."""
    annotation = (
        resolved_annotation if resolved_annotation is not None else parameter.annotation
    )
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    return None


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
        source_identity_stack_axes=request.source_identity_stack_axes,
        processing_context=request.context,
        filemanager=request.context.filemanager,
    )


def prepare_cellprofiler_runtime_adapter(request: RuntimeAdapterRequest) -> None:
    """Prepare CellProfiler source resolution during compile preparation."""
    cellprofiler_runtime_adapter_factory(request).prepare_source_resolution()


@dataclass(frozen=True, slots=True)
class CellProfilerModuleContractBinding:
    """Generated-module reference to a product-owned CellProfiler artifact contract."""

    generated_module_name: str
    module_num: int

    def __post_init__(self) -> None:
        if not self.generated_module_name:
            raise ValueError(
                "CellProfilerModuleContractBinding.generated_module_name cannot "
                "be empty."
            )
        if self.module_num < 1:
            raise ValueError(
                "CellProfilerModuleContractBinding.module_num must be one-based."
            )

    def resolve(self) -> ModuleArtifactContract:
        """Resolve this binding through the product-owned generated-module registry."""
        return CellProfilerModuleContractRegistry.contract_for(self)


class CellProfilerModuleContractRegistry:
    """Process-local contract registry for generated CellProfiler modules."""

    _contracts_by_generated_module: dict[str, dict[int, ModuleArtifactContract]] = {}

    @classmethod
    def register(
        cls,
        generated_module_name: str,
        contracts_by_module_num: Mapping[int, ModuleArtifactContract],
    ) -> None:
        """Register contracts compiled from a `.cppipe` for one generated module."""
        if not generated_module_name:
            raise ValueError("generated_module_name cannot be empty.")
        normalized: dict[int, ModuleArtifactContract] = {}
        for module_num, contract in contracts_by_module_num.items():
            if not isinstance(module_num, int) or module_num < 1:
                raise TypeError(
                    "CellProfiler module contract registry keys must be one-based "
                    f"ints, got {module_num!r}."
                )
            if not isinstance(contract, ModuleArtifactContract):
                raise TypeError(
                    "CellProfiler module contract registry values must be "
                    f"ModuleArtifactContract, got {type(contract).__name__}."
                )
            normalized[module_num] = contract
        cls._contracts_by_generated_module[generated_module_name] = normalized

    @classmethod
    @classmethod
    def contract_for(
        cls,
        binding: CellProfilerModuleContractBinding,
    ) -> ModuleArtifactContract:
        """Resolve a generated-module binding into a typed artifact contract."""
        try:
            contracts = cls._contracts_by_generated_module[
                binding.generated_module_name
            ]
        except KeyError as exc:
            raise KeyError(
                "No CellProfiler module contracts registered for generated module "
                f"{binding.generated_module_name!r}."
            ) from exc
        try:
            return contracts[binding.module_num]
        except KeyError as exc:
            raise KeyError(
                "No CellProfiler module contract registered for module "
                f"{binding.module_num} in generated module "
                f"{binding.generated_module_name!r}."
            ) from exc


CellProfilerModuleContractLike = (
    ModuleArtifactContract | CellProfilerModuleContractBinding
)


@dataclass(frozen=True, slots=True)
class CellProfilerModuleContractResolution:
    """Normalize callable construction contract inputs to artifact contracts."""

    contract: CellProfilerModuleContractLike

    def resolve(self) -> ModuleArtifactContract:
        match self.contract:
            case CellProfilerModuleContractBinding() as binding:
                return binding.resolve()
            case ModuleArtifactContract() as contract:
                return contract
            case _:
                raise TypeError(
                    "cellprofiler_module_callable contract must be "
                    "ModuleArtifactContract or CellProfilerModuleContractBinding, "
                    f"got {type(self.contract).__name__}."
                )


class CellProfilerRuntimeCallable:
    """Picklable callable wrapper for one artifact-managed CellProfiler module."""

    def __init__(
        self,
        raw_func: CellProfilerFunction,
        contract: ModuleArtifactContract,
        *,
        declared_processing_contract: str | None = None,
        processing_contract: CellProfilerRuntimeValue | None = None,
    ) -> None:
        from openhcs.processing.backends.cellprofiler import (
            CellProfilerFunctionCatalog,
        )

        try:
            raw_func = CellProfilerFunctionCatalog.get_function(raw_func.__name__)
        except KeyError:
            pass

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
        if processing_contract is not None:
            self.__processing_contract__ = processing_contract

        module_artifact_contract(contract)(self)
        analysis_func = raw_contract.raw_processing_function or raw_func
        set_signature_analysis_target(self, analysis_func)
        mark_enableable(self)
        runtime_adapter(
            "cellprofiler_runtime",
            cellprofiler_runtime_adapter_factory,
            manages_artifact_inputs=True,
            prepare=prepare_cellprofiler_runtime_adapter,
        )(self)
        set_parameter_exclusions(self, (
            "cellprofiler_runtime",
            "runtime_invocation_options",
        ))
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
            and self.declared_processing_contract == other.declared_processing_contract
            and self.processing_contract == other.processing_contract
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
            "cellprofiler_runtime",
            Parameter.KEYWORD_ONLY,
            annotation=CellProfilerRuntimeAdapter,
        ),
        Parameter(
            "runtime_invocation_options",
            Parameter.KEYWORD_ONLY,
            annotation=CellProfilerRuntimeValue | None,
            default=None,
        ),
        Parameter(
            "enabled",
            Parameter.KEYWORD_ONLY,
            annotation=bool,
            default=True,
        ),
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
    processing_contract: CellProfilerRuntimeValue | None,
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
    generated_module_name: str
    module_num: int
    declared_processing_contract: str | None

    def load(self) -> CellProfilerFunction:
        """Return the artifact-managed runtime callable for a generated step."""
        return cellprofiler_module_callable(
            self.raw_callable,
            CellProfilerModuleContractBinding(
                generated_module_name=self.generated_module_name,
                module_num=self.module_num,
            ),
            declared_processing_contract=self.declared_processing_contract,
            processing_contract=ProcessingContract.FLEXIBLE,
        )


def cellprofiler_module_callable(
    raw_func: CellProfilerFunction,
    contract: CellProfilerModuleContractLike,
    *,
    declared_processing_contract: str | None = None,
    processing_contract: CellProfilerRuntimeValue | None = None,
) -> CellProfilerFunction:
    """Build the product-owned runtime callable for one CellProfiler module."""
    if not callable(raw_func):
        raise TypeError(
            "cellprofiler_module_callable raw_func must be callable, "
            f"got {type(raw_func).__name__}."
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
            and contract.runtime_adapter is not None
            and contract.runtime_adapter.parameter_name == "cellprofiler_runtime"
        )

    def rehydrate(
        self,
        request: FunctionReferenceRehydrationRequest,
    ) -> CellProfilerFunction:
        contract = request.contract
        return cellprofiler_module_callable(
            contract.raw_processing_function,
            contract.module_artifact_contract,
            declared_processing_contract=contract.declared_processing_contract,
            processing_contract=contract.processing_contract,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerModuleRuntimePlan:
    """Static runtime decisions for one CellProfiler module callable."""

    func: CellProfilerFunction
    function_name: str
    callable_contract: CallableContract
    kwarg_spec: "CallableInvocationKwargSpec"
    declared_input_specs: tuple[ArtifactSpec, ...]
    declared_input_collection: ArtifactSpecCollection
    primary_image_inputs: tuple[ArtifactSpec, ...]
    primary_image_source_aliases: tuple[str, ...]
    runtime_image_name_set: frozenset[str]
    external_primary_image_names: tuple[str, ...]
    runtime_inputs: tuple[ArtifactSpec, ...]
    object_inputs: tuple[ArtifactSpec, ...]
    object_label_inputs: tuple[ArtifactSpec, ...]
    measurement_outputs: tuple[ArtifactSpec, ...]
    image_outputs: tuple[ArtifactSpec, ...]
    binding_scope: RuntimeArtifactBindingScope
    object_input_policy: CellProfilerObjectInputPolicy
    special_input_policy: "CellProfilerSpecialInputPolicy"
    invocation_execution_mode_policy: "CellProfilerInvocationExecutionModePolicy"
    main_flow_replacement_policy: CellProfilerMainFlowReplacementPolicy
    object_measurement_row_policy: CellProfilerObjectMeasurementRowPolicy
    measurement_record_builder: CellProfilerMeasurementRecordBuilder
    dual_scope_measurement_policy: "CellProfilerDualScopeMeasurementPolicy | None"
    dual_scope_image_function: CellProfilerFunction | None
    dual_scope_image_kwarg_spec: "CallableInvocationKwargSpec | None"
    special_input_names: tuple[str, ...]
    supported_non_object_input_kinds: frozenset[ArtifactKind]
    output_recording_plan: CellProfilerOutputRecordingPlan
    runs_per_image_measurement: bool
    runs_per_object_measurement: bool
    replaces_main_flow: bool

    @classmethod
    def build(
        cls,
        *,
        contract: ModuleArtifactContract,
        canonical_module_name: str,
        primary_image_input_policy: "CellProfilerPrimaryImageInputPolicy",
        func: CellProfilerFunction,
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
        )
        runtime_image_name_set = contract.runtime_input_name_set(ArtifactKind.IMAGE)
        non_image_inputs = tuple(
            spec for spec in declared_input_specs if spec.kind is not ArtifactKind.IMAGE
        )
        special_image_inputs = special_input_policy.special_image_inputs(
            contract.module_name,
            func,
            declared_input_specs,
        )
        runtime_inputs = (*non_image_inputs, *special_image_inputs)
        object_input_policy = CellProfilerObjectInputPolicy.for_module(
            canonical_module_name
        )
        object_label_inputs = declared_input_collection.of_kind(
            ArtifactKind.OBJECT_LABELS
        )
        output_collection = contract.output_collection()
        image_outputs = output_collection.of_kind(ArtifactKind.IMAGE)
        measurement_outputs = output_collection.of_kind(ArtifactKind.MEASUREMENTS)
        dual_scope_policy = CellProfilerDualScopeMeasurementPolicy.for_module(
            canonical_module_name
        )
        dual_scope_image_function = (
            None if dual_scope_policy is None else dual_scope_policy.image_function(func)
        )
        dual_scope_image_kwarg_spec = (
            None
            if dual_scope_image_function is None
            else CallableInvocationKwargSpec.from_callable(dual_scope_image_function)
        )
        return cls(
            func=func,
            function_name=callable_contract.function_name,
            callable_contract=callable_contract,
            kwarg_spec=CallableInvocationKwargSpec.from_callable(func),
            declared_input_specs=declared_input_specs,
            declared_input_collection=declared_input_collection,
            primary_image_inputs=primary_image_inputs,
            primary_image_source_aliases=ArtifactSpecCollection(
                primary_image_inputs
            ).names(),
            runtime_image_name_set=runtime_image_name_set,
            external_primary_image_names=tuple(
                spec.name
                for spec in primary_image_inputs
                if spec.name not in runtime_image_name_set
            ),
            runtime_inputs=runtime_inputs,
            object_inputs=ArtifactSpecCollection(runtime_inputs).of_kind(
                ArtifactKind.OBJECT_LABELS
            ),
            object_label_inputs=object_label_inputs,
            measurement_outputs=measurement_outputs,
            image_outputs=image_outputs,
            binding_scope=RuntimeArtifactBindingScope.from_contract(contract),
            object_input_policy=object_input_policy,
            special_input_policy=special_input_policy,
            invocation_execution_mode_policy=(
                CellProfilerInvocationExecutionModePolicy.for_module(
                    canonical_module_name
                )
            ),
            main_flow_replacement_policy=(
                CellProfilerMainFlowReplacementPolicy.for_module(
                    canonical_module_name
                )
            ),
            object_measurement_row_policy=(
                CellProfilerObjectMeasurementRowPolicy.for_module(
                    canonical_module_name
                )
            ),
            measurement_record_builder=CellProfilerMeasurementRecordBuilder.for_module(
                canonical_module_name
            ),
            dual_scope_measurement_policy=dual_scope_policy,
            dual_scope_image_function=dual_scope_image_function,
            dual_scope_image_kwarg_spec=dual_scope_image_kwarg_spec,
            special_input_names=special_input_names_from_callable(func),
            supported_non_object_input_kinds=(
                object_input_policy.supported_non_object_input_kinds
            ),
            output_recording_plan=CellProfilerOutputRecordingPlan.from_outputs(
                contract.outputs
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
                contract.module_name,
                object_label_inputs,
            ),
            replaces_main_flow=(
                CellProfilerMainFlowReplacementPolicy.for_module(
                    canonical_module_name
                ).replaces_main_flow(image_outputs)
            ),
        )

    @property
    def default_runtime_image_execution_mode(self) -> ImagePayloadExecutionMode | None:
        return self.callable_contract.runtime_image_execution_mode

    def runtime_batch_executor(
        self,
        domain: RuntimeBatchExecutionDomain,
    ) -> object | None:
        return self.callable_contract.runtime_batch_executor(domain)


@dataclass(slots=True)
class CellProfilerModuleExecutor:
    """Execute one generated CellProfiler module against a typed runtime adapter."""

    contract: ModuleArtifactContract

    _canonical_module_name: str = field(init=False, repr=False, compare=False)
    _primary_image_input_policy: "CellProfilerPrimaryImageInputPolicy" = field(
        init=False,
        repr=False,
        compare=False,
    )
    _runtime_plans: dict[CellProfilerFunction, CellProfilerModuleRuntimePlan] = field(
        init=False,
        default_factory=dict,
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
        self._canonical_module_name = canonical_module_name(self.contract.module_name)
        self._primary_image_input_policy = CellProfilerPrimaryImageInputPolicy.for_module(
            self._canonical_module_name
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
        for kind in tuple(RuntimeArtifactKindStrategy.__registry__.keys()):
            RuntimeArtifactKindStrategy.for_kind(kind)
        prepare_measurement_image_alignment_strategies()
        plan = self.runtime_plan(func)
        for output in plan.output_recording_plan.ordered_outputs:
            plan.output_recording_plan.recorders[output.kind]

    def runtime_plan(
        self,
        func: CellProfilerFunction,
    ) -> CellProfilerModuleRuntimePlan:
        """Return the prepared runtime plan for this callable and module contract."""
        plan = self._runtime_plans.get(func)
        if plan is not None:
            return plan
        plan = CellProfilerModuleRuntimePlan.build(
            contract=self.contract,
            canonical_module_name=self._canonical_module_name,
            primary_image_input_policy=self._primary_image_input_policy,
            func=func,
        )
        self._runtime_plans[func] = plan
        return plan

    def run(
        self,
        func: CellProfilerFunction,
        image: CellProfilerRuntimeValue,
        *,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        invocation_options: CellProfilerInvocationOptions | None = None,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        """Call the absorbed function and record declared outputs through the adapter."""
        plan = self.runtime_plan(func)
        function_name = plan.function_name
        run_started_at = time.perf_counter()
        mode_started_at = time.perf_counter()
        if plan.runs_per_image_measurement:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_runs_per_image_check",
                time.perf_counter() - mode_started_at,
                module=self.module_name,
                function=function_name,
            )
            per_image_started_at = time.perf_counter()
            result = self._run_per_image_measurement(
                func,
                plan,
                input_image=image,
                current_image=image,
                cellprofiler_runtime=cellprofiler_runtime,
                **kwargs,
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_run_per_image_measurement",
                time.perf_counter() - per_image_started_at,
                module=self.module_name,
                function=function_name,
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_module_run_total",
                time.perf_counter() - run_started_at,
                module=self.module_name,
                function=function_name,
            )
            return result

        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_runs_per_image_check",
            time.perf_counter() - mode_started_at,
            module=self.module_name,
            function=function_name,
        )
        image_request_started_at = time.perf_counter()
        image_request = self._image_request(
            plan,
            image,
            cellprofiler_runtime,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_image_request",
            time.perf_counter() - image_request_started_at,
            module=self.module_name,
            function=function_name,
        )
        object_mode_started_at = time.perf_counter()
        if plan.runs_per_object_measurement:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_runs_per_object_check",
                time.perf_counter() - object_mode_started_at,
                module=self.module_name,
                function=function_name,
            )
            per_object_started_at = time.perf_counter()
            result = self._run_per_object_measurement(
                func,
                plan,
                input_image=image,
                current_image=image,
                image_request=image_request,
                cellprofiler_runtime=cellprofiler_runtime,
                source_image_name=image_request.source_image_name,
                **kwargs,
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_run_per_object_measurement",
                time.perf_counter() - per_object_started_at,
                module=self.module_name,
                function=function_name,
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_module_run_total",
                time.perf_counter() - run_started_at,
                module=self.module_name,
                function=function_name,
            )
            return result

        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_runs_per_object_check",
            time.perf_counter() - object_mode_started_at,
            module=self.module_name,
            function=function_name,
        )
        invocation_started_at = time.perf_counter()
        invocation = self._invocation_request(
            plan,
            image_request=image_request,
            adapter=cellprofiler_runtime,
            current_image=image,
            kwargs=kwargs,
            invocation_options=invocation_options,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
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
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_contract_execute",
            time.perf_counter() - execute_started_at,
            module=self.module_name,
            function=function_name,
            **cellprofiler_profile_payload_fields("input", invocation.image),
            **cellprofiler_profile_payload_fields("output", raw_output),
        )
        split_started_at = time.perf_counter()
        main_output, artifact_values = _split_cellprofiler_output(raw_output)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_split_output",
            time.perf_counter() - split_started_at,
            module=self.module_name,
            function=function_name,
        )
        record_started_at = time.perf_counter()
        CellProfilerOutputRecorder.record_module_outputs(
            contract=self.contract,
            recording_plan=plan.output_recording_plan,
            primary_image_input_policy=self._primary_image_input_policy,
            adapter=cellprofiler_runtime,
            func=func,
            main_output=main_output,
            artifact_values=artifact_values,
            invocation=invocation,
            image_request=image_request,
            current_image=image,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_record_outputs",
            time.perf_counter() - record_started_at,
            module=self.module_name,
            function=function_name,
        )
        replace_started_at = time.perf_counter()
        if not plan.replaces_main_flow:
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_replace_main_flow_check",
                time.perf_counter() - replace_started_at,
                module=self.module_name,
                function=function_name,
            )
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_module_run_total",
                time.perf_counter() - run_started_at,
                module=self.module_name,
                function=function_name,
            )
            return CELLPROFILER_SIDE_EFFECT_MAIN_FLOW.output_image(
                current_image=image,
                image_request=image_request,
            )
        result = self._replacement_main_flow_output(
            plan,
            adapter=cellprofiler_runtime,
            current_image=image,
            invocation_image=invocation.image,
            output_image=main_output,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_replace_main_flow_check",
            time.perf_counter() - replace_started_at,
            module=self.module_name,
            function=function_name,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "cp_module_run_total",
            time.perf_counter() - run_started_at,
            module=self.module_name,
            function=function_name,
        )
        return result

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
                    adapter.get_image(output.name).data
                    for output in image_outputs
                ),
                slice_contexts=tuple(
                    AlignedImageSliceContext.main_flow(
                        output_key=output.name,
                        artifact_kind=output.kind.value,
                    )
                    for output in image_outputs
                ),
            )
            payload = composition.payload
            if isinstance(payload, AlignedImageStack):
                return stack_image_payload_context(
                    payload.slices,
                    np.stack(
                        tuple(image_payload_data(slice_payload) for slice_payload in payload.slices)
                    ),
                )
            return payload
        return cellprofiler_recorded_image_main_flow_output(
            current_image=current_image,
            invocation_image=invocation_image,
            recorded_image=output_image,
        )

    def _run_per_object_measurement(
        self,
        func: CellProfilerFunction,
        plan: CellProfilerModuleRuntimePlan,
        *,
        input_image: CellProfilerRuntimeValue,
        current_image: CellProfilerRuntimeValue,
        image_request: "CellProfilerImageRequest",
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        source_image_name: str | None,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        function_name = plan.function_name
        profiler = CellProfilerRuntimeProfiler(self.module_name, function_name)
        object_inputs = plan.object_label_inputs
        measurement_outputs = plan.measurement_outputs
        if len(measurement_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} per-object execution requires exactly one "
                "measurement output."
            )

        measurement_target_scope = self.pop_measurement_target_scope(
            kwargs,
            MeasurementScopeSelection.of(MeasurementScope.OBJECT),
        )
        combined_rows: list[CellProfilerRuntimeValue] = []
        measurement_images_started_at = time.perf_counter()
        measurement_image_resolver = CellProfilerMeasurementImageResolver(self)
        measurement_images = measurement_image_resolver.measurement_image_inputs(
            func,
            cellprofiler_runtime,
            current_image,
            image_request,
        )
        profile_events = [
            CellProfilerRuntimeProfileEvent(
                "cp_per_object_measurement_images",
                time.perf_counter() - measurement_images_started_at,
                (
                    ("images", len(measurement_images)),
                    ("objects", len(object_inputs)),
                ),
            )
        ]
        dual_scope_started_at = time.perf_counter()
        image_measurement_rows = self._dual_scope_image_measurement_rows(
            func,
            plan,
            measurement_images,
            cellprofiler_runtime,
            kwargs,
            measurement_target_scope,
        )
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
            measurement_row_policy.invocations(
                measurement_image,
                kwargs,
            )
            for measurement_image in measurement_images
        )
        total_measurement_batch_count = (
            sum(len(invocations) for invocations in measurement_invocations)
            * len(object_inputs)
        )
        prepared_invocations: list[
            tuple[
                RuntimeBatchInvocationRequest,
                CellProfilerMeasurementImage,
                ArtifactSpec,
                ObjectMeasurementInvocation,
                CellProfilerRuntimeValue,
            ]
        ] = []
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
                    preparation_profile_events,
                    label_payload_elapsed,
                    label_align_elapsed,
                ) = measurement_image_resolver.object_measurement_runtime_inputs(
                    func=func,
                    measurement_image=measurement_image,
                    object_spec=object_spec,
                    adapter=cellprofiler_runtime,
                    current_image=current_image,
                )
                profile_events.extend(preparation_profile_events)
                label_payload_seconds += label_payload_elapsed
                label_align_seconds += label_align_elapsed
                for invocation in invocations:
                    batch_request = RuntimeBatchInvocationRequest(
                        source_image_name=measurement_image.source_image_name,
                        execution_mode=execution_mode,
                        image=aligned_image,
                        kwargs={
                            **invocation.lowered_kwargs(),
                            "labels": executable_labels,
                        },
                        batch_index=len(prepared_invocations),
                        batch_count=total_measurement_batch_count,
                        semantic_group_key=object_measurement_batch_group_key(
                            object_spec=object_spec,
                            labels=executable_labels,
                        ),
                    )
                    prepared_invocations.append(
                        (
                            batch_request,
                            measurement_image,
                            object_spec,
                            invocation,
                            completion_label_payload,
                        )
                    )

        use_measurement_image_batch = (
            callable(batch_executor) and total_measurement_batch_count > 1
        )
        if not use_measurement_image_batch:
            for (
                batch_request,
                measurement_image,
                object_spec,
                invocation,
                completion_label_payload,
            ) in prepared_invocations:
                contract_started_at = time.perf_counter()
                raw_output = _execute_runtime_batch_invocation(func, batch_request)
                contract_execute_seconds += time.perf_counter() - contract_started_at
                output_recorder.record(
                    raw_output,
                    measurement_image=measurement_image,
                    object_spec=object_spec,
                    completion_label_payload=completion_label_payload,
                    invocation=invocation,
                )
        else:
            batch_requests = tuple(
                batch_request
                for (
                    batch_request,
                    _measurement_image,
                    _object_spec,
                    _invocation,
                    _completion_label_payload,
                ) in prepared_invocations
            )
            contract_started_at = time.perf_counter()
            raw_outputs = batch_executor(
                func,
                batch_requests,
                _execute_runtime_batch_invocation,
            )
            contract_execute_seconds += time.perf_counter() - contract_started_at
            if len(raw_outputs) != len(prepared_invocations):
                raise ValueError(
                    f"{function_name} measurement-image batch executor returned "
                    f"{len(raw_outputs)} outputs for {len(prepared_invocations)} "
                    "requests."
                )
            ordered_batch_outputs = {
                batch_request.batch_index: (
                    raw_output,
                    measurement_image,
                    object_spec,
                    invocation,
                    completion_label_payload,
                )
                for raw_output, (
                    batch_request,
                    measurement_image,
                    object_spec,
                    invocation,
                    completion_label_payload,
                ) in zip(
                    raw_outputs,
                    prepared_invocations,
                    strict=True,
                )
            }
            for order_index in range(len(ordered_batch_outputs)):
                (
                    raw_output,
                    measurement_image,
                    object_spec,
                    invocation,
                    completion_label_payload,
                ) = ordered_batch_outputs[order_index]
                output_recorder.record(
                    raw_output,
                    measurement_image=measurement_image,
                    object_spec=object_spec,
                    completion_label_payload=completion_label_payload,
                    invocation=invocation,
                )

        profile_events.extend(
            (
                CellProfilerRuntimeProfileEvent(
                    "cp_per_object_label_payload",
                    label_payload_seconds,
                ),
                CellProfilerRuntimeProfileEvent(
                    "cp_per_object_label_align",
                    label_align_seconds,
                ),
                CellProfilerRuntimeProfileEvent(
                    "cp_per_object_contract_execute",
                    contract_execute_seconds,
                ),
                CellProfilerRuntimeProfileEvent(
                    "cp_per_object_split_output",
                    output_timings.split_seconds,
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
            measurement_images,
            source_image_name,
        )
        combined_source_image_payload = CellProfilerMeasurementImage.shared_source_payload(
            measurement_images,
        )
        combined_source_metadata = CellProfilerMeasurementImage.composed_source_metadata(
            measurement_images,
        )

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
        profile_events.append(
            CellProfilerRuntimeProfileEvent(
                "cp_per_object_record_measurements",
                time.perf_counter() - record_started_at,
                (
                    (
                        "rows",
                        sum(len(rows) for rows in columnar_rows) + len(combined_rows),
                    ),
                ),
            )
        )
        profiler.record_events(tuple(profile_events))
        return CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
            input_image=input_image,
            measurement_images=measurement_images,
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
        if not target_scope.includes_all(MeasurementScope.IMAGE, MeasurementScope.OBJECT):
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
        row_source_names_required = measurement_row_source_names_required(measurement_images)
        image_kwargs = image_kwarg_spec.coerce_kwargs(kwargs)
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
            source_image_name = None
            if row_source_names_required:
                source_image_name = measurement_image.source_image_name
            owned_rows = (
                MeasurementRowOwnership(
                    source_image_name=source_image_name,
                ).annotate_rows(_measurement_rows_from_output(artifact_values))
            )
            projected_rows, _projected_row_mappings = (
                CellProfilerMeasurementRecord(
                    rows=owned_rows,
                    source_context=CellProfilerMeasurementSourceContext(
                        source_image_name=source_image_name,
                        source_image_payload=measurement_image.payload,
                    ),
                    object_name=None,
                ).projection_request(adapter=cellprofiler_runtime).project_rows()
            )
            rows.extend(projected_rows)
            split_rows_seconds += time.perf_counter() - split_rows_started_at
        profiler.record(
            "cp_dual_scope_contract_execute",
            contract_execute_seconds,
        )
        profiler.record(
            "cp_dual_scope_split_rows",
            split_rows_seconds,
            rows=len(rows),
        )
        return rows

    def _run_per_image_measurement(
        self,
        func: CellProfilerFunction,
        plan: CellProfilerModuleRuntimePlan,
        *,
        input_image: CellProfilerRuntimeValue,
        current_image: CellProfilerRuntimeValue,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        **kwargs: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        function_name = plan.function_name
        profiler = CellProfilerRuntimeProfiler(self.module_name, function_name)
        profile = PerImageMeasurementProfile(profiler)
        measurement_outputs = plan.measurement_outputs
        if len(measurement_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} per-image execution requires exactly one "
                "measurement output."
            )

        self.pop_measurement_target_scope(
            kwargs,
            MeasurementScopeSelection.of(MeasurementScope.IMAGE),
        )
        combined_rows: list[CellProfilerRuntimeValue] = []
        measurement_images_started_at = time.perf_counter()
        measurement_images = (
            CellProfilerMeasurementImageResolver(
                self
            ).independent_measurement_image_inputs(
                func,
                cellprofiler_runtime,
                current_image,
            )
        )
        profile.measurement_images(
            time.perf_counter() - measurement_images_started_at,
            len(measurement_images),
        )
        kwargs_started_at = time.perf_counter()
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(
                plan,
                cellprofiler_runtime,
                current_image,
                kwargs,
            ),
        }
        coerced_kwargs = plan.kwarg_spec.coerce_kwargs(runtime_kwargs)
        profile.prepare_kwargs(time.perf_counter() - kwargs_started_at)
        row_source_names_required = measurement_row_source_names_required(measurement_images)
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
            measurement_record = plan.measurement_record_builder.build(
                CellProfilerOutputRecordRequest(
                    contract=self.contract,
                    primary_image_input_policy=self._primary_image_input_policy,
                    adapter=cellprofiler_runtime,
                    spec=measurement_outputs[0],
                    output_value=resolved_values.recorded_value(
                        measurement_outputs[0]
                    ),
                    output_values=resolved_values.context_values,
                    func=func,
                    source=replace(
                        measurement_image,
                        source_image_name=source_image_name,
                    ),
                    call_kwargs=coerced_kwargs,
                )
            )
            combined_records.append(measurement_record)
            projected_rows, _projected_row_mappings = (
                measurement_record.projection_request(
                    adapter=cellprofiler_runtime
                ).project_rows()
            )
            combined_rows.extend(
                MeasurementRowOwnership(
                    source_image_name=(
                        measurement_record.source_context.source_image_name
                    ),
                ).annotate_rows(projected_rows)
            )
            split_rows_seconds += time.perf_counter() - split_rows_started_at

        profile.contract_execute(contract_execute_seconds)
        profile.split_rows(
            split_rows_seconds,
            len(combined_rows),
        )

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
                    measurement_outputs[0],
                    combined_rows,
                    func,
                ),
                object_name=image_measurement_object_name,
                source_context=CellProfilerMeasurementSourceContext(
                    source_image_name=image_measurement_source_name,
                ),
            ).materialization_request(
                adapter=cellprofiler_runtime,
                name=measurement_outputs[0].name,
                axis_state=MeasurementRowAxisState.IMAGE_NUMBER,
            )
        )
        profile.record_measurements(
            time.perf_counter() - record_started_at,
            len(combined_rows),
        )
        return CELLPROFILER_MEASUREMENT_MAIN_FLOW.output_image(
            input_image=input_image,
            measurement_images=measurement_images,
        )

    def pop_measurement_target_scope(
        self,
        kwargs: CellProfilerKwargDict,
        default_scope: MeasurementScopeSelection,
    ) -> MeasurementScopeSelection:
        """Consume the generated target-scope kwarg as OpenHCS measurement scopes."""
        return cellprofiler_measurement_scope_selection(
            kwargs.pop(CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG, None),
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
                        project_object_labels_to_current_plane=(
                            project_object_labels_to_current_plane
                        ),
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
                        current_image=current_image,
                        primary_image=primary_image,
                    ),
                    binding_scope=binding_scope,
                    project_object_labels_to_current_plane=(
                        project_object_labels_to_current_plane
                    ),
                )
            )

        supported_non_object_kinds = plan.supported_non_object_input_kinds
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
                project_object_labels_to_current_plane=(
                    project_object_labels_to_current_plane
                ),
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
            payload = (
                OBJECT_ONLY_REFERENCE_IMAGE.reference_image(current_image)
                if plan.object_label_inputs
                or plan.declared_input_collection.of_kind(ArtifactKind.SPATIAL_GRID)
                else cellprofiler_image_payload(current_image)
            )
            return CellProfilerImageRequest(
                payload=payload,
                source_image_name=self._input_source_image_name(plan, adapter),
                source_aliases=(),
                image_count=1,
                execution_mode=ImagePayloadExecutionMode.NATURAL,
                projects_runtime_slice_kwargs=not plan.object_label_inputs,
            )

        adapter.require_resolvable_source_aliases(plan.external_primary_image_names)
        payloads = []
        source_names: list[str | None] = []
        for spec in image_inputs:
            if spec.name in plan.runtime_image_name_set:
                runtime_image = adapter.get_image(
                    spec.name,
                    current_image=self._runtime_image_current_image(
                        plan,
                        adapter,
                        spec,
                        current_image,
                    ),
                )
                payloads.append(
                    cellprofiler_image_payload(runtime_image.data)
                )
                source_names.append(runtime_image.source_image_name)
                continue
            source_names.append(spec.name)
            payloads.append(
                cellprofiler_image_payload(
                    adapter.resolve_source_image(spec.name, current_image)
                )
            )
        composition = compose_aligned_image_payload(self.module_name, tuple(payloads))
        return CellProfilerImageRequest(
            payload=composition.payload,
            source_image_name=self._primary_image_source_name_from_sources(
                image_inputs,
                tuple(source_names),
            ),
            source_aliases=plan.primary_image_source_aliases,
            image_count=len(payloads),
            execution_mode=composition.execution_mode,
        )

    def _input_source_image_name(
        self,
        plan: CellProfilerModuleRuntimePlan,
        adapter: CellProfilerRuntimeAdapter,
    ) -> str | None:
        source_names: list[str] = []
        for spec in plan.declared_input_specs:
            source_name = RuntimeArtifactKindStrategy.for_kind(
                spec.kind
            ).source_image_name(
                RuntimeArtifactInputRequest.from_spec(
                    spec,
                    adapter=adapter,
                    binding_scope=plan.binding_scope,
                )
            )
            if source_name is not None:
                source_names.append(source_name)

        return single_source_name(tuple(source_names))

    @staticmethod
    def _primary_image_source_name_from_sources(
        image_inputs: tuple[ArtifactSpec, ...],
        source_names: tuple[str | None, ...],
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
            self._primary_image_input_policy.runtime_image_current_image(
                self.module_name,
                spec,
                current_image,
            )
        )
        if policy_current_image is None:
            return None
        default_execution_mode = (
            plan.default_runtime_image_execution_mode
            or ImagePayloadExecutionMode.NATURAL
        )
        if CurrentRuntimePlaneKwargProjectionContract(
            plan.func,
            default_execution_mode,
            source_identity_stack_axes=adapter.source_identity_stack_axes,
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
        invocation_options: CellProfilerInvocationOptions | None,
    ) -> "CellProfilerInvocationRequest":
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(
                plan,
                adapter,
                current_image,
                kwargs,
                primary_image=image_request.payload,
                project_object_labels_to_current_plane=(
                    image_request.projects_runtime_slice_kwargs
                ),
            ),
        }
        image_override = runtime_kwargs.pop(_CELLPROFILER_IMAGE_OVERRIDE_KWARG, None)
        execution_mode_override = runtime_kwargs.pop(
            _CELLPROFILER_EXECUTION_MODE_OVERRIDE_KWARG,
            None,
        )
        if self._canonical_module_name == _TRACK_OBJECTS_MODULE:
            source_image_name = (
                image_request.source_image_name
                or self._object_input_source_image_name(plan, adapter)
            )
            if "image_number_start" not in runtime_kwargs:
                source_paths = image_payload_metadata(current_image).source_image_paths
                if not source_paths:
                    source_paths = adapter.cellprofiler_source_paths_for_image_name(
                        source_image_name
                    )
                runtime_kwargs["image_number_start"] = (
                    adapter.cellprofiler_image_number_start_for_source_paths(source_paths)
                )
        runtime_kwargs.pop(CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG, None)
        invocation_image = (
            image_override if image_override is not None else image_request.payload
        )
        default_execution_mode = (
            plan.default_runtime_image_execution_mode
            or image_request.execution_mode
        )
        runtime_kwargs = dict(
            CurrentRuntimePlaneKwargProjection(
                image=invocation_image,
                kwargs=runtime_kwargs,
                plane_projector=adapter,
                project_runtime_slice_kwargs=(
                    image_request.projects_runtime_slice_kwargs
                    and CurrentRuntimePlaneKwargProjectionContract(
                        plan.func,
                        default_execution_mode,
                        source_identity_stack_axes=adapter.source_identity_stack_axes,
                    ).projects_runtime_slice_kwargs()
                ),
            ).kwargs_for_invocation()
        )
        execution_mode = plan.invocation_execution_mode_policy.execution_mode(
            default_execution_mode,
            image=invocation_image,
            kwargs=runtime_kwargs,
            invocation_options=invocation_options,
        )
        runtime_kwargs.pop(CELLPROFILER_GRID_CYCLE_SCOPE_KWARG, None)
        return CellProfilerInvocationRequest(
            image=invocation_image,
            kwargs=plan.kwarg_spec.coerce_kwargs(runtime_kwargs),
            source_image_name=image_request.source_image_name,
            image_count=image_request.image_count,
            execution_mode=execution_mode_override or execution_mode,
        )

    def _object_input_source_image_name(
        self,
        plan: CellProfilerModuleRuntimePlan,
        adapter: CellProfilerRuntimeAdapter,
    ) -> str | None:
        source_names = tuple(
            adapter.get_objects(spec.name).source_image_name
            for spec in plan.object_label_inputs
        )
        return single_source_name(
            tuple(source_name for source_name in source_names if source_name)
        )


class CellProfilerInvocationExecutionModePolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Nominal policy for modules whose settings change stack execution mode."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del image
        return default


class DefaultInvocationExecutionModePolicy(CellProfilerInvocationExecutionModePolicy):
    """Use the execution mode implied by image payload composition."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value


class CorrectIlluminationCalculateExecutionModePolicy(
    CellProfilerInvocationExecutionModePolicy
):
    """Run all-image illumination calculation once over the full image stack."""

    module_name = CELLPROFILER_CORRECT_ILLUMINATION_CALCULATE_MODULE

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del image
        if illumination_scope_uses_all_images(kwargs.get("calculation_scope")):
            return ImagePayloadExecutionMode.FULL_STACK
        return default


class ColorToGrayExecutionModePolicy(CellProfilerInvocationExecutionModePolicy):
    """ColorToGray consumes the channel composite, not independent channel slices."""

    module_name = "ColorToGray"

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del default, image, kwargs, invocation_options
        return ImagePayloadExecutionMode.FULL_STACK


class DefineGridManualExecutionModePolicy(CellProfilerInvocationExecutionModePolicy):
    """Honor CellProfiler's per-cycle versus once-only grid definition scope."""

    module_name = "DefineGridManual"

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
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

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def resolve_rank(cls, value: CellProfilerRuntimeValue) -> int | None:
        strategy = cls.for_nominal_value(value)
        if strategy is None:
            return None
        return strategy.spatial_rank(value)

    @abstractmethod
    def spatial_rank(self, value: CellProfilerRuntimeValue) -> int | None:
        """Return the spatial rank, excluding color channels, when known."""


class DenseArrayPayloadSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank for dense image arrays."""

    value_type = np.ndarray

    def spatial_rank(self, value: CellProfilerRuntimeValue) -> int | None:
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


class DataBackedPayloadSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank through payload objects that expose image data."""

    def spatial_rank(self, value: CellProfilerRuntimeValue) -> int | None:
        value_type = type(self).value_type
        if value_type is None:
            expected_type_name = "declared value_type"
        else:
            expected_type_name = value_type.__name__
        if value_type is None or not isinstance(value, value_type):
            raise TypeError(
                f"{type(self).__name__} requires {expected_type_name}."
            )
        return CellProfilerPayloadSpatialRankStrategy.resolve_rank(value.data)


class MaskedImagePayloadSpatialRankStrategy(DataBackedPayloadSpatialRankStrategy):
    """Resolve spatial rank through masked-image payload data."""

    value_type = MaskedImagePayload


class ImageMetadataPayloadSpatialRankStrategy(DataBackedPayloadSpatialRankStrategy):
    """Resolve spatial rank through image metadata payload data."""

    value_type = ImageMetadataPayload


class ObjectLabelValueSpatialRankStrategy(CellProfilerPayloadSpatialRankStrategy):
    """Resolve spatial rank for nominal object-label runtime values."""

    value_type = ObjectLabelValue

    def spatial_rank(self, value: CellProfilerRuntimeValue) -> int | None:
        if not isinstance(value, ObjectLabelValue):
            raise TypeError(
                "Object-label rank strategy requires an object-label runtime value."
            )
        return ObjectLabelDenseDataStrategy.spatial_rank(value)


@dataclass(frozen=True, slots=True)
class InvocationSpatialRankCandidates:
    """Spatial-rank observations available for one CellProfiler invocation."""

    ranks: tuple[int, ...]

    def max_rank_or_none(self) -> int | None:
        if not self.ranks:
            return None
        return max(self.ranks)


_STRUCTURING_ELEMENT_KWARG = "structuring_element"
_STRUCTURING_ELEMENT_SIZE_KWARG = "size"


@dataclass(frozen=True, slots=True)
class StructuringElementFootprintRequest:
    """Typed morphology footprint request from CellProfiler kwargs."""

    shape: CellProfilerRuntimeValue
    size: int

    @classmethod
    def from_kwargs(cls, kwargs: CellProfilerKwargs) -> "StructuringElementFootprintRequest":
        return cls(
            shape=kwargs[_STRUCTURING_ELEMENT_KWARG],
            size=int(kwargs[_STRUCTURING_ELEMENT_SIZE_KWARG]),
        )

    def footprint(self) -> np.ndarray:
        return build_structuring_element(self.shape, self.size)


class VolumetricInputExecutionModePolicy(CellProfilerInvocationExecutionModePolicy):
    """Run full-stack when the nominal image payload contains a Z volume."""

    module_name = None

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del kwargs, invocation_options
        if self.is_volumetric_payload(image):
            return ImagePayloadExecutionMode.FULL_STACK
        return default

    def is_volumetric_payload(self, image: CellProfilerRuntimeValue) -> bool:
        spatial_rank = self.spatial_rank(image)
        return spatial_rank is not None and spatial_rank >= 3

    def spatial_rank(self, image: CellProfilerRuntimeValue) -> int | None:
        data_rank = CellProfilerPayloadSpatialRankStrategy.resolve_rank(image)
        if data_rank is not None:
            return data_rank
        return CellProfilerPayloadSpatialRankStrategy.resolve_rank(
            image_payload_data(image)
        )

    def invocation_spatial_rank(
        self,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
    ) -> int | None:
        return InvocationSpatialRankCandidates(
            tuple(
                rank
                for rank in (
                    self.spatial_rank(image),
                    *(
                        CellProfilerPayloadSpatialRankStrategy.resolve_rank(value)
                        for value in kwargs.values()
                    ),
                )
                if rank is not None
            )
        ).max_rank_or_none()


class StructuringElementExecutionModePolicy(VolumetricInputExecutionModePolicy):
    """Match CellProfiler morphology dispatch from typed footprint rank."""

    module_name = None

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: CellProfilerInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del invocation_options
        spatial_rank = self.invocation_spatial_rank(image=image, kwargs=kwargs)
        if spatial_rank is None or spatial_rank < 3:
            return default
        footprint = StructuringElementFootprintRequest.from_kwargs(kwargs).footprint()
        if footprint.ndim == spatial_rank:
            return ImagePayloadExecutionMode.FULL_STACK
        return default


@dataclass(frozen=True, slots=True)
class InvocationExecutionModePolicySpec(GeneratedLeafClassSpec):
    """Declarative leaf spec for module-name execution-mode policies."""

    module_name: str

    def class_attributes(self) -> CellProfilerClassAttributes:
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


@dataclass(frozen=True, slots=True)
class CallableInvocationKwargSpec:
    """Cached callable kwarg contract used before CellProfiler invocation."""

    accepts_var_keyword: bool
    accepted_names: frozenset[str]
    invocation_control_defaults: tuple[tuple[str, CellProfilerRuntimeValue], ...]
    enum_types: tuple[tuple[str, type[Enum]], ...]

    @classmethod
    @lru_cache(maxsize=256)
    def from_callable(cls, func: CellProfilerFunction) -> "CallableInvocationKwargSpec":
        parameters = _callable_parameters(func)
        annotations = _callable_type_hints(func)
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
        return cls(
            accepts_var_keyword=any(
                parameter.kind is Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ),
            accepted_names=frozenset(parameters),
            invocation_control_defaults=tuple(
                (name, parameter.default)
                for name, parameter in parameters.items()
                if name in _INVOCATION_CONTROL_KWARGS
                and parameter.default is not Parameter.empty
            ),
            enum_types=enum_types,
        )

    def coerce_kwargs(self, kwargs: CellProfilerKwargs) -> CellProfilerKwargDict:
        """Filter unsupported kwargs and coerce enum-typed values."""
        if self.accepts_var_keyword:
            coerced_kwargs = dict(kwargs)
        else:
            coerced_kwargs = {
                name: value
                for name, value in kwargs.items()
                if name in self.accepted_names or name in _INVOCATION_CONTROL_KWARGS
            }
        for name, value in self.invocation_control_defaults:
            coerced_kwargs.setdefault(name, value)
        for name, enum_type in self.enum_types:
            if name not in coerced_kwargs:
                continue
            try:
                coerced_kwargs[name] = coerce_cellprofiler_enum(
                    enum_type,
                    coerced_kwargs[name],
                )
            except ValueError as exc:
                raise ValueError(
                    f"{name} must be coercible to {enum_type.__name__}; "
                    f"got {coerced_kwargs[name]!r}."
                ) from exc
        return coerced_kwargs


class CellProfilerPrimaryImageInputPolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Nominal policy for image artifacts that drive absorbed execution."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    @abstractmethod
    def primary_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return image inputs that should drive function invocation slices."""

    def runtime_image_current_image(
        self,
        module_name: str,
        spec: ArtifactSpec,
        current_image: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue | None:
        """Return source context used when resolving a primary runtime image."""
        del module_name, spec
        return current_image


class DefaultPrimaryImageInputPolicy(CellProfilerPrimaryImageInputPolicy):
    """Use non-special image inputs as the algorithmic image domain."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def primary_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
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

    registry_key = None

    def primary_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func, declared_inputs
        return ()


class MaskObjectsPrimaryImageInputPolicy(ObjectLabelDrivenPrimaryImageInputPolicy):
    """MaskObjects is driven by object labels; declared images are carriers."""

    module_name = "MaskObjects"


class TrackObjectsPrimaryImageInputPolicy(ObjectLabelDrivenPrimaryImageInputPolicy):
    """TrackObjects is driven by object labels across frame/site order."""

    module_name = "TrackObjects"


_MEASURE_OBJECT_SIZE_SHAPE_MODULE = CELLPROFILER_MEASURE_OBJECT_SIZE_SHAPE_MODULE
_MEASURE_OBJECT_INTENSITY_MODULE = CELLPROFILER_MEASURE_OBJECT_INTENSITY_MODULE
_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE = (
    CELLPROFILER_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE
)
_MEASURE_TEXTURE_MODULE = CELLPROFILER_MEASURE_TEXTURE_MODULE
_MEASURE_COLOCALIZATION_MODULE = CELLPROFILER_MEASURE_COLOCALIZATION_MODULE
_MEASURE_GRANULARITY_MODULE = CELLPROFILER_MEASURE_GRANULARITY_MODULE
_MEASURE_OBJECT_NEIGHBORS_MODULE = CELLPROFILER_MEASURE_OBJECT_NEIGHBORS_MODULE
_CROP_MODULE = CELLPROFILER_CROP_MODULE
_CONVERT_OBJECTS_TO_IMAGE_MODULE = CELLPROFILER_CONVERT_OBJECTS_TO_IMAGE_MODULE


@dataclass(frozen=True, slots=True)


class DeclaredSingleObjectLabelInputPolicy(SingleObjectLabelInputPolicy):
    """Generated base for modules with one declared label input."""


_SINGLE_OBJECT_LABEL_INPUT_POLICY_SPECS = (
    CellProfilerModulePolicyLeafSpec(
        class_name="CropInputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_CROP_MODULE,
        attributes={"label_kwarg": "cropping_labels"},
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name=f"{_MEASURE_OBJECT_SIZE_SHAPE_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_OBJECT_SIZE_SHAPE_MODULE,
        attributes={"label_kwarg": "labels"},
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name=f"{_MEASURE_OBJECT_INTENSITY_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_OBJECT_INTENSITY_MODULE,
        attributes={"label_kwarg": "labels"},
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name=f"{_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE,
        attributes={"label_kwarg": "labels"},
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name=f"{_MEASURE_TEXTURE_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_TEXTURE_MODULE,
        attributes={"label_kwarg": "labels"},
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name=f"{_MEASURE_COLOCALIZATION_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_COLOCALIZATION_MODULE,
        attributes={"label_kwarg": "labels"},
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name=f"{_MEASURE_GRANULARITY_MODULE}InputPolicy",
        base_type=DeclaredSingleObjectLabelInputPolicy,
        module_name=_MEASURE_GRANULARITY_MODULE,
        attributes={"label_kwarg": "labels"},
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
    ) -> CellProfilerKwargDict:
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
        neighbor_labels = None
        small_removed_neighbor_labels = None
        if not same_objects:
            neighbor_labels = LABEL_PAYLOAD_FINAL.value(neighbor_payload)
            small_removed_neighbor_labels = _label_payload_small_removed(
                neighbor_payload
            )

        return {
            "labels": LABEL_PAYLOAD_FINAL.value(measured_payload),
            "small_removed_labels": _label_payload_small_removed(measured_payload),
            "neighbor_labels": neighbor_labels,
            "small_removed_neighbor_labels": small_removed_neighbor_labels,
            "neighbors_are_same_objects": same_objects,
        }


class ObjectLabelsInputBindingMixin:
    """Bind object-label inputs under CellProfiler's object_labels kwarg."""

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        return {"object_labels": request.labels_for_inputs()}


class OverlayOutlinesInputPolicy(
    ObjectLabelsInputBindingMixin,
    CellProfilerObjectInputPolicy,
):
    """Bind ordered object outline rows for the generic overlay runner."""

    module_name = "OverlayOutlines"

class ObjectRowsInputPolicy(
    ObjectLabelsInputBindingMixin,
    CellProfilerObjectInputPolicy,
):
    """Bind ordered object rows to object-label payloads."""


class ObjectRowsWithMeasurementsInputPolicy(ObjectRowsInputPolicy):
    """Bind ordered object rows plus prior measurements for the primary object."""

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        bound = super().bind(request)
        bound[MEASUREMENT_TABLES_BOUND_KEY] = (
            request.measurement_tables_for_primary_object()
        )
        return bound


class CombineObjectsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind two object-label inputs as the CombineObjects label-pair payload."""

    module_name = "Combineobjects"

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        request.require_exact_object_count(2)
        label_planes = self.label_pair_payload(request)
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

    def label_pair_payload(
        self,
        request: ObjectInputBindingRequest,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the two CombineObjects inputs in a shared dense slice domain."""
        label_payloads = tuple(
            request.label_payload_for(spec)
            for spec in request.object_inputs
        )
        slice_counts = tuple(
            count
            for payload in label_payloads
            for count in (self.runtime_slice_count(payload),)
            if count is not None
        )
        if not slice_counts:
            return tuple(
                np.asarray(
                    SingletonObjectLabelStackCollapseStrategy.for_labels(payload).collapse(
                        payload
                    ),
                    dtype=np.int32,
                )
                for payload in label_payloads
            )
        slice_count_set = set(slice_counts)
        if len(slice_count_set) != 1:
            raise ValueError(
                "CombineObjects requires compatible object-label runtime slice "
                f"domains, got slice counts {sorted(slice_count_set)!r}."
            )
        slice_count = slice_count_set.pop()
        stacks = tuple(
            DenseObjectLabelSliceStackRequest(
                payload,
                slice_count,
                np.int32,
            ).stack()
            for payload in label_payloads
        )
        if any(stack is None for stack in stacks):
            shapes = [
                tuple(object_label_dense_array(payload, dtype=np.int32).shape)
                for payload in label_payloads
            ]
            raise ValueError(
                "CombineObjects requires object-label inputs compatible with "
                f"runtime slice count {slice_count}, got shapes {shapes!r}."
            )
        return tuple(stack.labels for stack in stacks if stack is not None)

    def runtime_slice_count(self, payload: CellProfilerRuntimeValue) -> int | None:
        """Return the declared or dense object-label slice count for CombineObjects."""
        declared_count = ObjectLabelRuntimeSliceStackContract.runtime_slice_count(payload)
        if declared_count is not None:
            return declared_count
        label_array = object_label_dense_array(payload, dtype=np.int32)
        if label_array.ndim != 3:
            return None
        return int(label_array.shape[0])


_FILTER_OBJECTS_ADDITIONAL_OBJECT_COUNT_KWARG = "additional_object_count"
_FILTER_OBJECTS_ENCLOSING_OBJECT_NAME_KWARG = "enclosing_object_name"
_FILTER_OBJECTS_MEASUREMENT_FEATURES_KWARG = "measurement_features"


@dataclass(frozen=True, slots=True)
class FilterObjectsKwargSettings:
    """Typed FilterObjects settings projected from CellProfiler kwargs."""

    additional_object_count: int
    enclosing_object_name: str | None
    measurement_features: tuple[str, ...]

    @classmethod
    def from_kwargs(cls, kwargs: CellProfilerKwargs) -> "FilterObjectsKwargSettings":
        raw_additional_count = kwargs.get(_FILTER_OBJECTS_ADDITIONAL_OBJECT_COUNT_KWARG)
        if raw_additional_count is None:
            additional_object_count = 0
        else:
            additional_object_count = int(raw_additional_count)
        raw_enclosing_name = kwargs.get(_FILTER_OBJECTS_ENCLOSING_OBJECT_NAME_KWARG)
        if raw_enclosing_name is None:
            enclosing_object_name = None
        else:
            enclosing_object_name = str(raw_enclosing_name)
        raw_measurement_features = kwargs.get(_FILTER_OBJECTS_MEASUREMENT_FEATURES_KWARG)
        if raw_measurement_features is None:
            measurement_features = ()
        else:
            measurement_features = tuple(str(value) for value in raw_measurement_features)
        return cls(
            additional_object_count=additional_object_count,
            enclosing_object_name=enclosing_object_name,
            measurement_features=measurement_features,
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsRuntimeInputPlan:
    """Runtime object-label partition for one FilterObjects invocation."""

    object_specs: tuple[ArtifactSpec, ...]
    enclosing_spec: ArtifactSpec | None
    settings: FilterObjectsKwargSettings
    relationship_spec: ArtifactSpec | None = None
    measurement_relationship_specs: tuple[ArtifactSpec, ...] = ()

    @classmethod
    def from_inputs(
        cls,
        runtime_inputs: tuple[ArtifactSpec, ...],
        kwargs: CellProfilerKwargs,
    ) -> "FilterObjectsRuntimeInputPlan":
        object_inputs = ArtifactSpecCollection(runtime_inputs).of_kind(
            ArtifactKind.OBJECT_LABELS
        )
        settings = FilterObjectsKwargSettings.from_kwargs(kwargs)
        object_count = settings.additional_object_count + 1
        enclosing_name = settings.enclosing_object_name
        object_specs = object_inputs[:object_count]
        enclosing_spec = None
        relationship_spec = None
        measurement_relationship_specs: list[ArtifactSpec] = []
        if enclosing_name is not None:
            enclosing_spec = ArtifactSpecCollection(object_inputs).by_name(
                enclosing_name
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
                        enclosing_name,
                        object_specs[0].name,
                    ),
                    ArtifactKind.RELATIONSHIPS,
                )
        if object_specs:
            for child_object_name in (
                CellProfilerMeasurementFeature.child_count_object_names(
                    settings.measurement_features
                )
            ):
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
            settings=settings,
            relationship_spec=relationship_spec,
            measurement_relationship_specs=ArtifactSpecCollection(
                measurement_relationship_specs
            ).unique(conflict_context="CellProfiler input spec"),
        )

    @property
    def primary_object_spec(self) -> ArtifactSpec | None:
        if not self.object_specs:
            return None
        return self.object_specs[0]

    def bind_measurement_inputs(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        """Return FilterObjects measurement bindings owned by this runtime plan."""
        scoped_request = request.with_object_inputs(self.object_specs)
        measurement_values = self.measurement_vector(scoped_request)
        if measurement_values is not None:
            return {MEASUREMENT_VALUES_BOUND_KEY: measurement_values}
        measurement_tables = self.measurement_tables(scoped_request)
        if measurement_tables is None:
            return {}
        return {MEASUREMENT_TABLES_BOUND_KEY: measurement_tables}

    def measurement_vector(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerRuntimeValue | None:
        object_spec = self.primary_object_spec
        if object_spec is None:
            return None
        feature_names = self.settings.measurement_features
        if len(feature_names) != 1:
            return None
        feature_name = str(feature_names[0])
        labels = request.labels_for(object_spec)
        return (
            CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=feature_name,
                labels=labels,
            )
            .vector()
            .slice_aligned_value
        )

    def measurement_tables(
        self,
        request: ObjectInputBindingRequest,
    ) -> tuple[MeasurementTable, ...] | None:
        object_spec = self.primary_object_spec
        if object_spec is None:
            return None
        feature_names = self.settings.measurement_features
        if not feature_names:
            return None
        labels = request.labels_for(object_spec)
        tables_by_identity: dict[int, MeasurementTable] = {}
        for feature_name in feature_names:
            binding = CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=str(feature_name),
                labels=labels,
            )
            tables = binding.measurement_tables(
                request.adapter,
                match_group=False,
            )
            for table in tables:
                table_identity = id(table)
                if table_identity not in tables_by_identity:
                    tables_by_identity[table_identity] = table
        if not tables_by_identity:
            return object_measurement_tables_for_object(request.adapter, object_spec.name)
        return tuple(tables_by_identity.values())


@dataclass(frozen=True, slots=True)
class FilterObjectsBoundMeasurementInputs:
    """Measurement binding profile for FilterObjects logging."""

    bound: CellProfilerKwargs

    @property
    def measurement_tables(self) -> tuple[MeasurementTable, ...]:
        value = self.bound.get(MEASUREMENT_TABLES_BOUND_KEY)
        if value is None:
            return ()
        return tuple(value)

    @property
    def measurement_values(self) -> CellProfilerRuntimeValue | None:
        return self.bound.get(MEASUREMENT_VALUES_BOUND_KEY)

    @property
    def measurement_values_type(self) -> str:
        measurement_values = self.measurement_values
        if measurement_values is None:
            return "none"
        return type(measurement_values).__name__


CellProfilerModulePolicyLeafSpec(
    class_name="MeasureImageAreaOccupiedInputPolicy",
    base_type=ObjectRowsInputPolicy,
    module_name="MeasureImageAreaOccupiedBinary",
).declare_in(globals())


CALCULATE_MATH_OPERAND_VALUE_KWARGS = ("operand1_value", "operand2_value")


def calculate_math_operand_value_kwargs(
    operand1: CellProfilerRuntimeValue,
    operand2: CellProfilerRuntimeValue,
) -> CellProfilerKwargDict:
    """Lower CalculateMath operand values to CellProfiler's kwarg ABI."""
    return dict(zip(CALCULATE_MATH_OPERAND_VALUE_KWARGS, (operand1, operand2)))


class FilterObjectsInputPolicy(ObjectRowsWithMeasurementsInputPolicy):
    """Bind ordered primary/additional object rows for FilterObjects."""

    module_name = "FilterObjects"
    supported_non_object_input_kinds = frozenset({ArtifactKind.RELATIONSHIPS})

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        runtime_inputs = request.runtime_inputs
        if not runtime_inputs:
            runtime_inputs = request.object_inputs
        plan = FilterObjectsRuntimeInputPlan.from_inputs(
            runtime_inputs,
            request.kwargs,
        )
        bound = super().bind(request.with_object_inputs(plan.object_specs))
        bound.update(plan.bind_measurement_inputs(request))
        bound_measurements = FilterObjectsBoundMeasurementInputs(bound)
        measurement_values = bound_measurements.measurement_values
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "filterobjects_bound_measurements",
            0.0,
            module=self.module_name,
            table_count=len(bound_measurements.measurement_tables),
            has_measurement_values=measurement_values is not None,
            measurement_values_type=bound_measurements.measurement_values_type,
            measurement_features=len(plan.settings.measurement_features),
        )
        if plan.enclosing_spec is not None:
            bound["enclosing_object_labels"] = request.labels_for(plan.enclosing_spec)
        if plan.relationship_spec is not None:
            bound["parent_child_relationship"] = request.current_plane_relationship_for(
                plan.relationship_spec,
            )
        if plan.measurement_relationship_specs:
            bound["parent_child_relationships"] = tuple(
                request.current_plane_relationship_for(relationship_spec)
                for relationship_spec in plan.measurement_relationship_specs
            )
        return bound


class CalculateMathInputPolicy(CellProfilerObjectInputPolicy):
    """Bind CalculateMath operands from runtime measurement/object state."""

    module_name = "CalculateMath"
    binds_without_declared_inputs = True
    supported_non_object_input_kinds = frozenset({ArtifactKind.MEASUREMENTS})

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        started_at = time.perf_counter()
        operand_bindings = self.object_operand_bindings(request)
        if operand_bindings is not None:
            vectors = CellProfilerObjectMeasurementVectorBatchBinding(
                operand_bindings
            ).vectors()
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "calculate_math_bind_total",
                time.perf_counter() - started_at,
            )
            return calculate_math_operand_value_kwargs(
                vectors[0].calculate_math_operand_value,
                vectors[1].calculate_math_operand_value,
            )

        operand1_started_at = time.perf_counter()
        operand1_value = self.operand_value(
            request,
            feature_kwarg="operand1_feature",
            object_kwarg="operand1_object_name",
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
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
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_operand_bind",
            time.perf_counter() - operand2_started_at,
            operand="2",
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_bind_total",
            time.perf_counter() - started_at,
        )
        return calculate_math_operand_value_kwargs(operand1_value, operand2_value)

    def object_operand_bindings(
        self,
        request: ObjectInputBindingRequest,
    ) -> tuple[CellProfilerObjectMeasurementVectorBinding, ...] | None:
        bindings: list[CellProfilerObjectMeasurementVectorBinding] = []
        for feature_kwarg, object_kwarg in (
            ("operand1_feature", "operand1_object_name"),
            ("operand2_feature", "operand2_object_name"),
        ):
            feature_name = CellProfilerStringKwargAuthority.required(
                request.kwargs,
                feature_kwarg,
                "CalculateMath",
            )
            object_name = CellProfilerStringKwargAuthority.optional(
                request.kwargs,
                object_kwarg,
            )
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
                CellProfilerObjectMeasurementVectorBinding.for_object(
                    request,
                    object_ref=object_spec,
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
    ) -> CellProfilerRuntimeValue:
        feature_name = CellProfilerStringKwargAuthority.required(
            request.kwargs,
            feature_kwarg,
            "CalculateMath",
        )
        object_name = CellProfilerStringKwargAuthority.optional(
            request.kwargs,
            object_kwarg,
        )
        count_object_name = count_feature_object_name(feature_name)
        if count_object_name is not None:
            return float(
                ObjectLabelCountAuthority.count_from_adapter(
                    request.adapter,
                    count_object_name,
                )
            )
        if object_name is None:
            return self.image_operand_value(request, feature_name)

        return (
            CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_name,
                feature_name=feature_name,
            )
            .vector()
            .calculate_math_operand_value
        )

    def image_operand_value(
        self,
        request: ObjectInputBindingRequest,
        feature_name: str,
    ) -> CellProfilerRuntimeValue:
        declared_measurement_tables = request.declared_measurement_tables()
        if declared_measurement_tables:
            declared_slice_values = MeasurementImageOperandVectorResolution(
                measurement_tables=declared_measurement_tables,
                feature_name=feature_name,
            ).resolve()
            if declared_slice_values is not None:
                return CellProfilerMeasurementVector(
                    declared_slice_values
                ).slice_aligned_value
            return MeasurementFeatureQuery(
                feature_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ).scalar_value(declared_measurement_tables)

        tables_started_at = time.perf_counter()
        measurement_resolution = MeasurementImageOperandVectorResolution.from_runtime_feature(
            request.adapter,
            feature_name,
            current_image=request.current_image,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_measurement_tables",
            time.perf_counter() - tables_started_at,
            feature=feature_name,
            count=len(measurement_resolution.measurement_tables),
        )
        slice_started_at = time.perf_counter()
        slice_values = measurement_resolution.resolve()
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_image_operand_slices",
            time.perf_counter() - slice_started_at,
            feature=feature_name,
            sliced=slice_values is not None,
        )
        if slice_values is None:
            scalar_started_at = time.perf_counter()
            scalar_value = MeasurementFeatureQuery(
                feature_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ).scalar_value(measurement_resolution.measurement_tables)
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "calculate_math_image_operand_scalar",
                time.perf_counter() - scalar_started_at,
                feature=feature_name,
            )
            return scalar_value
        return CellProfilerMeasurementVector(slice_values).slice_aligned_value


def _image_scope_measurement_payload(image: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
    """Return one image plane for image-scoped measurement functions."""
    return SINGLETON_STACK_OUTPUT_COLLAPSE.collapse(image)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecialInputBindingRequest(RuntimeInputBindingRequestBase):
    """Authoritative runtime context for binding declared special_inputs."""

    registry_key = "special_input"

    parameter_names: tuple[str, ...]
    special_input_specs: tuple[ArtifactSpec, ...]
    runtime_inputs: tuple[ArtifactSpec, ...]
    project_object_labels_to_current_plane: bool = False

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
        parameter_name: str | None = None,
        semantics: CellProfilerSpecialInputPayloadSemantics = (
            CellProfilerSpecialInputPayloadSemantics.INTENSITY_IMAGE
        ),
    ) -> CellProfilerRuntimeValue:
        if spec.kind is ArtifactKind.OBJECT_LABELS:
            if parameter_name is not None:
                return self.label_argument_for(spec, parameter_name)
            return self.object_label_runtime_value(spec, semantics)
        request = self.artifact_input_request(spec)
        artifact_strategy = RuntimeArtifactKindStrategy.for_kind(spec.kind)
        if semantics.dense_label_domain:
            return object_label_dense_array(
                artifact_strategy.raw_runtime_input_value(request),
                dtype=np.int32,
            )
        return artifact_strategy.runtime_input_value(request)

    def runtime_value_without_current_image_projection(
        self,
        spec: ArtifactSpec,
    ) -> CellProfilerSpecialInputValue:
        """Return a runtime artifact input without ambient source-image narrowing."""
        request = replace(self.artifact_input_request(spec), current_image=None)
        return RuntimeArtifactKindStrategy.for_kind(spec.kind).runtime_input_value(request)

    def object_label_runtime_value(
        self,
        spec: ArtifactSpec,
        semantics: CellProfilerSpecialInputPayloadSemantics,
    ) -> ObjectLabelData:
        """Return an object-label special input in the invocation's artifact domain."""
        payload = RuntimeArtifactKindStrategy.for_kind(spec.kind).runtime_input_value(
            self.artifact_input_request(spec)
        )
        del semantics
        return object_label_dense_array(payload, dtype=np.int32)

    def current_plane_object_label_runtime_value(
        self,
        spec: ArtifactSpec,
    ) -> ObjectLabelData:
        """Return object labels projected into the invocation's current plane."""
        payload = self.current_plane_label_payload_for(spec)
        return object_label_dense_array(payload, dtype=np.int32)

    def object_label_payload(
        self,
        spec: ArtifactSpec,
    ) -> ObjectLabelValue:
        """Return object labels with provenance preserved for special inputs."""
        payload = RuntimeArtifactKindStrategy.for_kind(spec.kind).runtime_input_value(
            self.artifact_input_request(spec)
        )
        if not isinstance(payload, ObjectLabelValue):
            raise TypeError(
                f"{self.module_name} special input {spec.name!r} resolved to "
                f"{type(payload).__name__}, expected ObjectLabelValue."
            )
        return payload

    def current_image_aligned_object_label_runtime_value(
        self,
        spec: ArtifactSpec,
        *,
        alignment_image: CellProfilerRuntimeValue | None = None,
    ) -> CellProfilerRuntimeValue:
        """Return object labels ordered to match the current image stack planes."""
        labels = self.adapter.get_objects(spec.name)
        aligned = CurrentImageObjectLabelPlaneAlignment(
            adapter=self.adapter,
            current_image=alignment_image if alignment_image is not None else self.current_image,
            labels=labels,
        ).aligned_dense_value()
        if aligned is not None:
            return aligned
        return self.current_plane_object_label_runtime_value(spec)

    def bind_positional_parameters(self) -> CellProfilerSpecialInputKwargs:
        """Bind declared special-input parameters to compiled runtime specs."""
        if len(self.parameter_names) != len(self.special_input_specs):
            raise NotImplementedError(
                f"{self.module_name} declares special_inputs "
                f"{list(self.parameter_names)}, but compiled runtime inputs are "
                f"{[spec.name for spec in self.special_input_specs]}."
            )
        return {
            parameter_name: self.runtime_value(spec, parameter_name=parameter_name)
            for parameter_name, spec in zip(
                self.parameter_names,
                self.special_input_specs,
                strict=True,
            )
        }


class CellProfilerSpecialInputPolicy(
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Nominal module-specific binding for CellProfiler special_inputs."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    def special_image_inputs(
        self,
        module_name: str,
        func: CellProfilerRuntimeCallable,
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return trailing image specs consumed by special_inputs instead of primary image payload."""

        return _signature_special_image_inputs(module_name, func, declared_inputs)

    def binding_current_image(
        self,
        *,
        current_image: ImagePayloadMetadataInput,
        primary_image: ImagePayloadMetadataInput | None,
    ) -> ImagePayloadMetadataInput:
        """Return the source image context used to bind special inputs."""
        del primary_image
        return current_image

    @abstractmethod
    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerSpecialInputKwargs:
        """Return kwargs for a callable's declared special_inputs."""


class PositionalSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Bind special_inputs positionally to compiled runtime artifact specs."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerSpecialInputKwargs:
        return request.bind_positional_parameters()


class MaskImageSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Bind mask object labels in the current runtime plane."""

    module_name = _MASK_IMAGE_MODULE

    def binding_current_image(
        self,
        *,
        current_image: CellProfilerRuntimeValue,
        primary_image: CellProfilerRuntimeValue | None,
    ) -> CellProfilerRuntimeValue:
        """Align mask labels to the image being masked."""
        return primary_image if primary_image is not None else current_image

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
        if len(request.parameter_names) != len(request.special_input_specs):
            raise NotImplementedError(
                f"{request.module_name} declares special_inputs "
                f"{list(request.parameter_names)}, but compiled runtime inputs are "
                f"{[spec.name for spec in request.special_input_specs]}."
            )
        bound: CellProfilerKwargDict = {}
        alignment_image: CellProfilerRuntimeValue | None = None
        deferred_object_specs: list[tuple[str, ArtifactSpec]] = []
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "mask_image_special_input_specs",
            0.0,
            special_specs=tuple(
                (spec.kind.value, spec.name) for spec in request.special_input_specs
            ),
            runtime_specs=tuple(
                (spec.kind.value, spec.name) for spec in request.runtime_inputs
            ),
        )
        for parameter_name, spec in zip(
            request.parameter_names,
            request.special_input_specs,
            strict=True,
        ):
            if spec.kind is ArtifactKind.OBJECT_LABELS:
                deferred_object_specs.append((parameter_name, spec))
                continue
            value = (
                request.runtime_value_without_current_image_projection(spec)
                if (
                    spec.kind is ArtifactKind.IMAGE
                    and request.binding_scope.image_origin(spec)
                    is RuntimeImageInputOrigin.RUNTIME
                )
                else request.runtime_value(spec)
            )
            bound[parameter_name] = value
            if spec.kind is ArtifactKind.IMAGE and alignment_image is None:
                alignment_image = value
        if alignment_image is None:
            for spec in request.runtime_inputs:
                if (
                    spec.kind is ArtifactKind.IMAGE
                    and request.binding_scope.image_origin(spec)
                    is RuntimeImageInputOrigin.RUNTIME
                ):
                    alignment_image = (
                        request.runtime_value_without_current_image_projection(spec)
                    )
                    break
        for parameter_name, spec in deferred_object_specs:
            bound[parameter_name] = request.current_image_aligned_object_label_runtime_value(
                spec,
                alignment_image=alignment_image,
            )
        return bound


class RelateObjectsSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Bind parent/child object labels in the current runtime plane."""

    module_name = _RELATE_OBJECTS_MODULE

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
        if len(request.parameter_names) != len(request.special_input_specs):
            raise NotImplementedError(
                f"{request.module_name} declares special_inputs "
                f"{list(request.parameter_names)}, but compiled runtime inputs are "
                f"{[spec.name for spec in request.special_input_specs]}."
            )
        bound = {
            parameter_name: (
                request.current_plane_object_label_runtime_value(spec)
                if spec.kind is ArtifactKind.OBJECT_LABELS
                else request.runtime_value(spec)
            )
            for parameter_name, spec in zip(
                request.parameter_names,
                request.special_input_specs,
                strict=True,
            )
        }
        plane_index = request.relationship_runtime_slice_index()
        if (
            plane_index is not None
            and request.func is not None
            and _SLICE_INDEX_PARAMETER in _callable_parameters(request.func)
        ):
            if _SLICE_INDEX_PARAMETER not in bound:
                bound[_SLICE_INDEX_PARAMETER] = plane_index
        return bound


class TrailingImageSpecialInputPolicy(CellProfilerSpecialInputPolicy):
    """Treat all image inputs after the primary image as special inputs."""

    def special_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func
        image_inputs = ArtifactSpecCollection(declared_inputs).of_kind(
            ArtifactKind.IMAGE
        )
        return image_inputs[1:]


class CropSpecialInputPolicy(TrailingImageSpecialInputPolicy):
    """Bind Crop side inputs without making them primary image domains."""

    module_name = _CROP_MODULE

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
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
        bound: CellProfilerKwargDict = {}
        if image_inputs:
            bound["mask_plane"] = request.runtime_value(image_inputs[0])
        if object_inputs:
            bound["cropping_labels"] = request.label_payload_for(object_inputs[0])
        return bound


class ImageMathSpecialInputPolicy(TrailingImageSpecialInputPolicy):
    """Bind trailing ImageMath image inputs as ordered operands."""

    module_name = "ImageMath"

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
        return {
            "image_operands": tuple(
                request.runtime_value(spec)
                for spec in request.image_inputs
            )
        }


class WatershedSpecialInputBindingStrategy(
    EnumStrategyLabelRegistryMixin,
    metaclass=AutoRegisterMeta,
):
    """Bind Watershed special input roles for one nominal watershed method."""

    strategy_key: ClassVar[WatershedMethod | None] = None

    @abstractmethod
    def bind(
        self,
        request: SpecialInputBindingRequest,
        image_inputs: tuple[ArtifactSpec, ...],
    ) -> CellProfilerKwargDict:
        """Return callable kwargs for declared Watershed special image roles."""

    def _runtime_image_value(
        self,
        spec: ArtifactSpec,
        request: SpecialInputBindingRequest,
        semantics: CellProfilerSpecialInputPayloadSemantics = (
            CellProfilerSpecialInputPayloadSemantics.INTENSITY_IMAGE
        ),
    ) -> CellProfilerRuntimeValue:
        return request.runtime_value(spec, semantics=semantics)


class MarkerWatershedSpecialInputBindingStrategy(WatershedSpecialInputBindingStrategy):
    """Marker mode consumes marker labels first and an optional mask second."""

    strategy_key = WatershedMethod.MARKERS

    def bind(
        self,
        request: SpecialInputBindingRequest,
        image_inputs: tuple[ArtifactSpec, ...],
    ) -> CellProfilerKwargDict:
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
    ) -> CellProfilerKwargDict:
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
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func
        return ArtifactSpecCollection(declared_inputs).of_kind(ArtifactKind.IMAGE)[1:]

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
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
    ) -> CellProfilerKwargDict:
        object_inputs = request.object_inputs
        CellProfilerObjectInputCountAuthority.require_exact(
            request.module_name,
            object_inputs,
            1,
        )
        measurement_inputs = ArtifactSpecCollection(request.runtime_inputs).of_kind(
            ArtifactKind.MEASUREMENTS
        )
        bound: CellProfilerKwargDict = {
            "worm_labels": request.labels_for(object_inputs[0]),
        }
        if not measurement_inputs:
            return bound
        if len(measurement_inputs) > 1:
            raise NotImplementedError(
                f"{request.module_name} supports one producer measurement "
                f"input; got {[spec.name for spec in measurement_inputs]}."
            )
        num_control_points = int(
            MappingValueLookup(request.kwargs, "num_control_points").value_or(21)
        )
        control_points = WormControlPointMeasurementSchema(
            num_control_points=num_control_points,
        ).control_points_from_rows(
            request.runtime_value(measurement_inputs[0]),
            object_name=object_inputs[0].name,
        )
        if control_points is not None:
            bound["control_points"] = control_points
        return bound


class NoSpecialImageInputsMixin:
    """Declare that a special-input policy consumes no image artifacts."""

    def special_image_inputs(
        self,
        module_name: str,
        func: CellProfilerFunction,
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del module_name, func, declared_inputs
        return ()


class ConvertObjectsToImageSpecialInputPolicy(
    NoSpecialImageInputsMixin,
    CellProfilerSpecialInputPolicy,
):
    """Bind object labels as payloads so rendered images inherit label provenance."""

    module_name = _CONVERT_OBJECTS_TO_IMAGE_MODULE

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
        object_inputs = request.object_inputs
        CellProfilerObjectInputCountAuthority.require_exact(
            request.module_name,
            object_inputs,
            1,
        )
        return {"labels": request.object_label_payload(object_inputs[0])}


class DisplayDataOnImageSpecialInputPolicy(
    NoSpecialImageInputsMixin,
    CellProfilerSpecialInputPolicy,
):
    """Resolve display annotations from object labels and measurement tables."""

    module_name = "DisplayDataOnImage"

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
        object_inputs = request.object_inputs
        CellProfilerObjectInputCountAuthority.require_exact(
            request.module_name,
            object_inputs,
            1,
        )
        object_spec = object_inputs[0]
        labels = request.labels_for(object_spec)
        feature_name = CellProfilerStringKwargAuthority.required(
            request.kwargs,
            "measurement_feature",
            request.module_name,
        )
        return {
            "labels": labels,
            "measurements": (
                CellProfilerObjectMeasurementVectorBinding.for_object(
                    request,
                    object_ref=object_spec,
                    feature_name=feature_name,
                    labels=labels,
                )
                .vector()
                .slice_aligned_value
            ),
        }


class ClassifyObjectsMeasurementInputPolicy(
    NoSpecialImageInputsMixin,
    CellProfilerSpecialInputPolicy,
):
    """Resolve ClassifyObjects label and measurement-vector inputs."""

    measurement_kwarg_by_parameter: ClassVar[Mapping[str, str]] = {
        "measurement_values": "measurement_feature",
        "measurement1_values": "measurement1_feature",
        "measurement2_values": "measurement2_feature",
    }

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
        object_inputs = request.object_inputs
        CellProfilerObjectInputCountAuthority.require_exact(
            request.module_name,
            object_inputs,
            1,
        )
        object_spec = object_inputs[0]
        labels = request.labels_for(object_spec)
        measurement_labels = request.label_payload_for(object_spec)
        image_number = RuntimeArtifactKindStrategy.for_kind(
            object_spec.kind
        ).cellprofiler_image_number(request.artifact_input_request(object_spec))
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
                    CellProfilerObjectMeasurementVectorBinding.for_object(
                        request,
                        object_ref=object_spec,
                        feature_name=_classification_rule_measurement_feature(
                            rule,
                            request.module_name,
                        ),
                        labels=measurement_labels,
                        image_number=image_number,
                    )
                    .vector()
                    .slice_aligned_value
                    for rule in rules
                ),
            }
        bound_values = {
            parameter_name: (
                CellProfilerObjectMeasurementVectorBinding.for_object(
                    request,
                    object_ref=object_spec,
                    feature_name=CellProfilerStringKwargAuthority.required(
                        request.kwargs,
                        kwarg_name,
                        request.module_name,
                    ),
                    labels=measurement_labels,
                    image_number=image_number,
                    source=(
                        CellProfilerObjectMeasurementVectorSource
                        .CURRENT_OBJECT_SHAPE_FEATURE
                    ),
                )
                .vector()
                .slice_aligned_value
            )
            for parameter_name, kwarg_name in (
                type(self).measurement_kwarg_by_parameter.items()
            )
            if kwarg_name in request.kwargs
        }
        return {
            "labels": labels,
            **bound_values,
        }


for _input_policy_spec in (
    CellProfilerModulePolicyLeafSpec(
        class_name="ClassifyObjectsSingleMeasurementInputPolicy",
        base_type=ClassifyObjectsMeasurementInputPolicy,
        module_name="ClassifyObjectsSingleMeasurement",
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name="ClassifyObjectsTwoMeasurementsInputPolicy",
        base_type=ClassifyObjectsMeasurementInputPolicy,
        module_name="ClassifyObjectsTwoMeasurements",
    ),
):
    _input_policy_spec.declare_in(globals())
del _input_policy_spec


def _classification_rule_measurement_feature(
    rule: CellProfilerRuntimeValue,
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
    func: CellProfilerFunction,
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


@dataclass(frozen=True, slots=True)
class CellProfilerOptionalNonemptyString:
    """Optional string text after CellProfiler kwarg type validation."""

    value: str

    def normalized_or_none(self) -> str | None:
        normalized = self.value.strip()
        if not normalized:
            return None
        return normalized


class CellProfilerStringKwargAuthority:
    """Typed string-kwarg validation shared by CellProfiler binding policies."""

    @staticmethod
    def required(
        kwargs: CellProfilerKwargs,
        name: str,
        module_name: str,
    ) -> str:
        value = kwargs.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{module_name} requires non-empty kwarg {name!r}.")
        return value

    @staticmethod
    def optional(
        kwargs: CellProfilerKwargs,
        name: str,
    ) -> str | None:
        raw_value = kwargs.get(name)
        if raw_value is None:
            return None
        if not isinstance(raw_value, str):
            raise TypeError(
                f"Expected string kwarg {name!r}, got {type(raw_value).__name__}."
            )
        return CellProfilerOptionalNonemptyString(raw_value).normalized_or_none()


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
            spec.kind is not ArtifactKind.MEASUREMENTS
            for spec in CellProfilerCallableOutputSpecs(request.func).artifact_specs()
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
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Nominal policy for modules whose `Both` scope emits image and object facts."""

    fallback_registry_key = None
    image_function_name: ClassVar[str | None] = None

    def image_function(self, object_func: CellProfilerFunction) -> CellProfilerFunction:
        del object_func
        return require_function(
            _required_class_attr(type(self).module_name, "module_name"),
            function_name=_required_class_attr(
                type(self).image_function_name,
                "image_function_name",
            ),
        )


class DeclaredDualScopeMeasurementPolicy(CellProfilerDualScopeMeasurementPolicy):
    """Generated base for modules with image+object measurement scope."""


for _dual_scope_policy_spec in (
    CellProfilerModulePolicyLeafSpec(
        class_name="MeasureTextureDualScopeMeasurementPolicy",
        base_type=DeclaredDualScopeMeasurementPolicy,
        module_name=_MEASURE_TEXTURE_MODULE,
        attributes={"image_function_name": "measure_texture"},
    ),
    CellProfilerModulePolicyLeafSpec(
        class_name="MeasureColocalizationDualScopeMeasurementPolicy",
        base_type=DeclaredDualScopeMeasurementPolicy,
        module_name=_MEASURE_COLOCALIZATION_MODULE,
        attributes={"image_function_name": "measure_colocalization"},
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


def _callable_accepts_composed_image_payload(func: CellProfilerFunction) -> bool:
    """Return whether callable parameters describe a multi-image bundle contract."""
    parameters = _callable_parameters(func)
    return any(
        parameter_name in parameters
        for parameter_name in _COMPOSED_IMAGE_PAYLOAD_PARAMETERS
    )


def _required_class_attr(value: RequiredAttrT | None, name: str) -> RequiredAttrT:
    if value is None:
        raise TypeError(f"CellProfiler policy must define {name}.")
    return value
