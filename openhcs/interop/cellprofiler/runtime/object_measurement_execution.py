"""Object-measurement execution domain and label argument policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
import numpy as np

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
)
from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution_from_callable,
)
from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.runtime_values import (
    ObjectLabelDenseDataStrategy,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelValue,
    SparseIJVLabelRows,
)
from openhcs.interop.cellprofiler.runtime.module_names import (
    CELLPROFILER_MEASURE_COLOCALIZATION_MODULE,
    CELLPROFILER_MEASURE_GRANULARITY_MODULE,
    CELLPROFILER_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE,
    CELLPROFILER_MEASURE_OBJECT_INTENSITY_MODULE,
    CELLPROFILER_MEASURE_OBJECT_SIZE_SHAPE_MODULE,
    CELLPROFILER_MEASURE_TEXTURE_MODULE,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
    CellProfilerModulePolicyRegistryKey,
    EnumStrategyLabelRegistryMixin,
)
from openhcs.processing.backends.cellprofiler.library import canonical_module_name

_MEASURE_OBJECT_SIZE_SHAPE_MODULE = CELLPROFILER_MEASURE_OBJECT_SIZE_SHAPE_MODULE
_MEASURE_OBJECT_INTENSITY_MODULE = CELLPROFILER_MEASURE_OBJECT_INTENSITY_MODULE
_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE = (
    CELLPROFILER_MEASURE_OBJECT_INTENSITY_DISTRIBUTION_MODULE
)
_MEASURE_TEXTURE_MODULE = CELLPROFILER_MEASURE_TEXTURE_MODULE
_MEASURE_COLOCALIZATION_MODULE = CELLPROFILER_MEASURE_COLOCALIZATION_MODULE
_MEASURE_GRANULARITY_MODULE = CELLPROFILER_MEASURE_GRANULARITY_MODULE


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

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @abstractmethod
    def execution_mode(
        self,
        func: CellProfilerFunction,
        labels: CellProfilerRuntimeValue,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        """Return the execution mode required by the supplied labels."""

    @classmethod
    def resolve(
        cls,
        func: CellProfilerFunction,
        labels: CellProfilerRuntimeValue,
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
        func: CellProfilerFunction,
        labels: CellProfilerRuntimeValue,
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
    value_type = ObjectLabelValue

    def execution_mode(
        self,
        func: CellProfilerFunction,
        labels: CellProfilerRuntimeValue,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        if not isinstance(labels, ObjectLabelValue):
            raise TypeError(
                "Object-label execution strategy requires an object-label runtime value."
            )
        if ObjectLabelRuntimeSliceStackContract.runtime_slice_count(labels) is not None:
            return ImagePayloadExecutionMode.NATURAL
        return MeasurementLabelExecutionModeStrategy.resolve(
            func,
            ObjectLabelDenseDataStrategy.for_payload(labels).data(labels),
            default,
        )


class SparseIJVMeasurementLabelExecutionModeStrategy(
    MeasurementLabelExecutionModeStrategy
):
    value_type = SparseIJVLabelRows

    def execution_mode(
        self,
        func: CellProfilerFunction,
        labels: CellProfilerRuntimeValue,
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
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Choose CellProfiler object-measurement execution domain by module semantics."""

    @abstractmethod
    def execution_mode(
        self,
        func: CellProfilerFunction,
        labels: CellProfilerRuntimeValue,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        """Return the execution mode for one object-measurement invocation."""


class DefaultObjectMeasurementExecutionDomainPolicy(
    CellProfilerObjectMeasurementExecutionDomainPolicy
):
    """Apply the function/domain object-measurement contract uniformly."""

    registry_key = CellProfilerModulePolicyRegistryKey.DEFAULT.value

    def execution_mode(
        self,
        func: CellProfilerFunction,
        labels: CellProfilerRuntimeValue,
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

    dense_labels: CellProfilerRuntimeValue
    label_payload: CellProfilerRuntimeValue
    measurement_image_payload: CellProfilerRuntimeValue


class SliceAlignedLabelArgumentStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Choose the executor-facing label payload for slice-aligned measurements."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @abstractmethod
    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> CellProfilerRuntimeValue:
        """Return the label value visible to the execution strategy."""


class DenseSliceAlignedLabelArgumentStrategy(SliceAlignedLabelArgumentStrategy):
    """Default slice-aligned functions consume already-projected dense labels."""

    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> CellProfilerRuntimeValue:
        return request.dense_labels


class SemanticLabelPayloadArgumentMixin:
    """Return the semantic label payload rather than the dense execution plane."""

    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> CellProfilerRuntimeValue:
        return request.label_payload


class AlignedStackSliceAlignedLabelArgumentStrategy(
    SemanticLabelPayloadArgumentMixin,
    SliceAlignedLabelArgumentStrategy,
):
    """Defer object-label projection until each aligned image slice is selected."""

    value_type = AlignedImageStack


class CellProfilerObjectMeasurementLabelArgumentPolicy(
    EnumStrategyLabelRegistryMixin,
    metaclass=AutoRegisterMeta,
):
    """Bind object-measurement labels from the declared callable domain contract."""

    __enum_member_attr__ = "execution_mode"

    execution_mode: ClassVar[ObjectLabelMeasurementExecution]

    @abstractmethod
    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> CellProfilerRuntimeValue:
        """Return the labels object passed to the absorbed measurement function."""


class SliceAlignedObjectMeasurementLabelArgumentPolicy(
    CellProfilerObjectMeasurementLabelArgumentPolicy
):
    """Slice-aligned measurement functions consume the dense execution plane."""

    execution_mode = ObjectLabelMeasurementExecution.SLICE_ALIGNED

    def label_argument(
        self,
        request: CellProfilerObjectMeasurementLabelArgumentRequest,
    ) -> CellProfilerRuntimeValue:
        strategy = SliceAlignedLabelArgumentStrategy.for_nominal_value(
            request.measurement_image_payload
        )
        if strategy is None:
            return request.dense_labels
        return strategy.label_argument(request)


class FullStackObjectMeasurementLabelArgumentPolicy(
    SemanticLabelPayloadArgumentMixin,
    CellProfilerObjectMeasurementLabelArgumentPolicy
):
    """Full-stack measurement functions consume labels with semantic domains."""

    execution_mode = ObjectLabelMeasurementExecution.FULL_STACK
