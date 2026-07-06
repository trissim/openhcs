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
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    EnumStrategyLabelRegistryMixin,
)


class CellProfilerPerObjectMeasurementPolicy:
    """Predicate for modules that need one absorbed call per object set."""

    @classmethod
    def matches(cls, module_name: str, object_inputs: tuple[ArtifactSpec, ...]) -> bool:
        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerModule,
            PerObjectMeasurementExecutionModule,
        )

        module_type = CellProfilerModule.for_module(module_name)
        return (
            bool(object_inputs)
            and module_type is not None
            and issubclass(module_type, PerObjectMeasurementExecutionModule)
        )

    @classmethod
    def measures_images_independently(cls, module_name: str) -> bool:
        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerModule,
            ComposedImageObjectMeasurementExecutionModule,
        )

        module_type = CellProfilerModule.for_module(module_name)
        return module_type is None or not issubclass(
            module_type, ComposedImageObjectMeasurementExecutionModule
        )


class MeasurementLabelExecutionModeStrategy(
    NominalTypeKeyedStrategyMixin, ABC, metaclass=AutoRegisterMeta
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
            func, labels, default, runtime_slice_count=runtime_slice_count
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
        if runtime_slice_count is not None:
            if ObjectLabelRuntimeSliceStackContract.preserves_runtime_slice_stack(
                labels,
                slice_count=runtime_slice_count,
            ):
                return ImagePayloadExecutionMode.NATURAL
        return MeasurementLabelExecutionModeStrategy.resolve(
            func,
            ObjectLabelDenseDataStrategy.for_payload(labels).data(labels),
            default,
            runtime_slice_count=runtime_slice_count,
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


class CellProfilerObjectMeasurementExecutionDomainPolicy(ABC):
    """Choose CellProfiler object-measurement execution domain by module semantics."""

    @classmethod
    def for_module(
        cls, module_name: str
    ) -> "CellProfilerObjectMeasurementExecutionDomainPolicy":
        del module_name
        return DefaultObjectMeasurementExecutionDomainPolicy()

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

    def execution_mode(
        self,
        func: CellProfilerFunction,
        labels: CellProfilerRuntimeValue,
        default: ImagePayloadExecutionMode,
        *,
        runtime_slice_count: int | None = None,
    ) -> ImagePayloadExecutionMode:
        return MeasurementLabelExecutionModeStrategy.resolve(
            func, labels, default, runtime_slice_count=runtime_slice_count
        )


@dataclass(frozen=True, slots=True)
class CellProfilerObjectMeasurementLabelArgumentRequest:
    """Typed label-domain context for one object-measurement invocation."""

    dense_labels: CellProfilerRuntimeValue
    label_payload: CellProfilerRuntimeValue
    measurement_image_payload: CellProfilerRuntimeValue


class SliceAlignedLabelArgumentStrategy(
    NominalTypeKeyedStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Choose the executor-facing label payload for slice-aligned measurements."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @abstractmethod
    def label_argument(
        self, request: CellProfilerObjectMeasurementLabelArgumentRequest
    ) -> CellProfilerRuntimeValue:
        """Return the label value visible to the execution strategy."""


class DenseSliceAlignedLabelArgumentStrategy(SliceAlignedLabelArgumentStrategy):
    """Default slice-aligned functions consume already-projected dense labels."""

    def label_argument(
        self, request: CellProfilerObjectMeasurementLabelArgumentRequest
    ) -> CellProfilerRuntimeValue:
        return request.dense_labels


class SemanticLabelPayloadArgumentMixin:
    """Return the semantic label payload rather than the dense execution plane."""

    def label_argument(
        self, request: CellProfilerObjectMeasurementLabelArgumentRequest
    ) -> CellProfilerRuntimeValue:
        return request.label_payload


class AlignedStackSliceAlignedLabelArgumentStrategy(
    SemanticLabelPayloadArgumentMixin, SliceAlignedLabelArgumentStrategy
):
    """Defer object-label projection until each aligned image slice is selected."""

    value_type = AlignedImageStack


class CellProfilerObjectMeasurementLabelArgumentPolicy(
    EnumStrategyLabelRegistryMixin, metaclass=AutoRegisterMeta
):
    """Bind object-measurement labels from the declared callable domain contract."""

    __enum_member_attr__ = "execution_mode"
    execution_mode: ClassVar[ObjectLabelMeasurementExecution]

    @abstractmethod
    def label_argument(
        self, request: CellProfilerObjectMeasurementLabelArgumentRequest
    ) -> CellProfilerRuntimeValue:
        """Return the labels object passed to the absorbed measurement function."""


class SliceAlignedObjectMeasurementLabelArgumentPolicy(
    CellProfilerObjectMeasurementLabelArgumentPolicy
):
    """Slice-aligned measurement functions consume the dense execution plane."""

    execution_mode = ObjectLabelMeasurementExecution.SLICE_ALIGNED

    def label_argument(
        self, request: CellProfilerObjectMeasurementLabelArgumentRequest
    ) -> CellProfilerRuntimeValue:
        strategy = SliceAlignedLabelArgumentStrategy.for_nominal_value(
            request.measurement_image_payload
        )
        if strategy is None:
            return request.dense_labels
        return strategy.label_argument(request)


class FullStackObjectMeasurementLabelArgumentPolicy(
    SemanticLabelPayloadArgumentMixin, CellProfilerObjectMeasurementLabelArgumentPolicy
):
    """Full-stack measurement functions consume labels with semantic domains."""

    execution_mode = ObjectLabelMeasurementExecution.FULL_STACK
