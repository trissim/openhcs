"""
Unified registry base class for external library function registration.

This module provides a common base class that eliminates ~70% of code duplication
across library registries (pyclesperanto, scikit-image, cupy, etc.) while enforcing
consistent behavior and making it impossible to skip dynamic testing or hardcode
function lists.

Key Benefits:
- Eliminates ~1000+ lines of duplicated code
- Enforces consistent testing and registration patterns
- Makes adding new libraries trivial (60-120 lines vs 350-400)
- Centralizes bug fixes and improvements
- Type-safe abstract interface prevents shortcuts

Architecture:
- LibraryRegistryBase: Abstract base class with common functionality
- ProcessingContract: Unified contract enum across all libraries
- Dimension error adapter factory for consistent error handling
- Integrated caching system using existing cache_utils.py patterns
"""

import importlib
import inspect
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    get_type_hints,
)


import numpy as np
from openhcs.core.aligned_image_payload import AlignedImageStack
from openhcs.core.xdg_paths import get_cache_file_path
from openhcs.core.memory import (
    stack_runtime_slices,
    unstack_runtime_slices,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.measurement_row_materialization import MeasurementRowsAxisProjection
from openhcs.core.measurement_row_materialization import ConcatenatedColumnarRows
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    ImagePayloadMetadataCompositionMode,
    MaskedImagePayload,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_slice_context,
    with_image_payload_data,
)
from openhcs.core.runtime_array_values import RuntimeArrayPayload
from openhcs.core.runtime_object_labels import ObjectLabelPayload, ObjectLabelSet
from openhcs.core.runtime_object_label_aggregation import (
    ObjectLabelPure2DSliceAggregator,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_spatial_grid import SpatialGrid
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_output_matching import (
    RuntimeOutputBundle,
    runtime_output_tuple,
)
from openhcs.core.runtime_batch_contracts import (
    Pure2DSliceBatchExecutor,
    RuntimeBatchExecutionDomain,
    RuntimePure2DSliceBatchRequest,
    runtime_batch_executors_from_callable,
)
from openhcs.core.callable_contract import KeywordRuntimeParameter
from openhcs.core.variable_component_stack_requirement import (
    AlwaysRequiresVariableComponentStack,
    SemanticControlVariableComponentStackRequirement,
    VariableComponentStackRequirement,
)
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from metaclass_registry import AutoRegisterMeta

logger = logging.getLogger(__name__)

PURE2D_VALUE_TYPE_REGISTRY_KEY = "value_type"


class RuntimeCallableView(Enum):
    """Nominal callable view used by contract execution."""

    DECORATED = auto()
    RAW = auto()


class RuntimeInvocationKwargPolicy(Enum):
    """Nominal kwarg policy used by runtime callable invocation."""

    PASS_THROUGH = auto()
    SIGNATURE_FILTERED = auto()


class RuntimeCallableViewStrategy(
    EnumKeyedStrategyMixin[RuntimeCallableView],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Resolve the callable object for a runtime invocation view."""

    __registry_key__ = "view_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "view"
    __enum_label_attr__ = "view_label"

    view: ClassVar[RuntimeCallableView | None] = None
    view_label: ClassVar[str | None] = None

    @classmethod
    def for_view(cls, view: RuntimeCallableView) -> "RuntimeCallableViewStrategy":
        return cls.for_enum_member(view)

    @abstractmethod
    def resolve(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Return the callable object selected by this view."""


class DecoratedRuntimeCallableViewStrategy(RuntimeCallableViewStrategy):
    view = RuntimeCallableView.DECORATED

    def resolve(self, func: Callable[..., Any]) -> Callable[..., Any]:
        return func


class RawRuntimeCallableViewStrategy(RuntimeCallableViewStrategy):
    view = RuntimeCallableView.RAW

    def resolve(self, func: Callable[..., Any]) -> Callable[..., Any]:
        from openhcs.core.callable_contract import CallableContract

        return CallableContract.from_callable(func).resolve_raw_runtime_callable()


class RuntimeInvocationKwargPolicyStrategy(
    EnumKeyedStrategyMixin[RuntimeInvocationKwargPolicy],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Filter invocation kwargs for a runtime kwarg policy."""

    __registry_key__ = "policy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "policy"
    __enum_label_attr__ = "policy_label"

    policy: ClassVar[RuntimeInvocationKwargPolicy | None] = None
    policy_label: ClassVar[str | None] = None

    @classmethod
    def for_policy(
        cls,
        policy: RuntimeInvocationKwargPolicy,
    ) -> "RuntimeInvocationKwargPolicyStrategy":
        return cls.for_enum_member(policy)

    @abstractmethod
    def accepted_kwargs(
        self,
        func: Callable[..., Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return kwargs accepted by ``func`` under this policy."""


class PassThroughRuntimeInvocationKwargPolicyStrategy(
    RuntimeInvocationKwargPolicyStrategy
):
    policy = RuntimeInvocationKwargPolicy.PASS_THROUGH

    def accepted_kwargs(
        self,
        func: Callable[..., Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        del func
        return dict(kwargs)


class SignatureFilteredRuntimeInvocationKwargPolicyStrategy(
    RuntimeInvocationKwargPolicyStrategy
):
    policy = RuntimeInvocationKwargPolicy.SIGNATURE_FILTERED

    def accepted_kwargs(
        self,
        func: Callable[..., Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        parameters = _runtime_callable_parameters(func)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return dict(kwargs)
        return {name: value for name, value in kwargs.items() if name in parameters}


@dataclass(frozen=True, slots=True)
class RuntimeCallableInvocation:
    """Typed runtime invocation boundary for processing-contract callables."""

    func: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    callable_view: RuntimeCallableView = RuntimeCallableView.DECORATED
    kwarg_policy: RuntimeInvocationKwargPolicy = (
        RuntimeInvocationKwargPolicy.PASS_THROUGH
    )

    def call(self) -> Any:
        target = RuntimeCallableViewStrategy.for_view(self.callable_view).resolve(
            self.func
        )
        return target(
            *self.args,
            **RuntimeInvocationKwargPolicyStrategy.for_policy(
                self.kwarg_policy
            ).accepted_kwargs(target, self.kwargs),
        )


@dataclass(frozen=True, slots=True)
class RuntimeCallablePolicy:
    """Reusable runtime callable invocation semantics."""

    callable_view: RuntimeCallableView = RuntimeCallableView.DECORATED
    kwarg_policy: RuntimeInvocationKwargPolicy = (
        RuntimeInvocationKwargPolicy.PASS_THROUGH
    )

    def invocation(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> RuntimeCallableInvocation:
        return RuntimeCallableInvocation(
            func=func,
            args=args,
            kwargs=kwargs,
            callable_view=self.callable_view,
            kwarg_policy=self.kwarg_policy,
        )


@lru_cache(maxsize=256)
def _runtime_callable_parameters(
    func: Callable[..., Any],
) -> Mapping[str, inspect.Parameter]:
    return inspect.signature(func).parameters


def _registry_runtime_parameter_exclusions(
    signature: inspect.Signature,
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return registry-owned injected parameter names present in a signature."""
    return tuple(
        parameter_name
        for parameter_name in parameter_names
        if parameter_name in signature.parameters
    )


def _set_registry_runtime_parameter_exclusions(
    target: object,
    signature: inspect.Signature,
    parameter_names: tuple[str, ...],
    *,
    source: object | None = None,
) -> None:
    """Merge registry-owned injected parameter names into analysis exclusions."""
    from python_introspect import add_parameter_exclusions, parameter_exclusions

    source_exclusions = () if source is None else parameter_exclusions(source)
    add_parameter_exclusions(
        target,
        (
            *source_exclusions,
            *_registry_runtime_parameter_exclusions(signature, parameter_names),
        ),
    )


@dataclass(frozen=True, slots=True)
class Pure2DSliceResultBatch:
    """Typed decomposition of per-slice PURE_2D outputs."""

    main_outputs: list[Any]
    auxiliary_groups: tuple[list[Any], ...] = ()

    @classmethod
    def from_results(cls, results: Iterable[Any]) -> "Pure2DSliceResultBatch":
        collected = [runtime_output_tuple(result) for result in results]
        if not collected:
            raise ValueError("PURE_2D execution cannot aggregate zero slice results.")

        first_result = collected[0]
        if not isinstance(first_result, tuple):
            return cls(main_outputs=collected)

        tuple_length = len(first_result)
        if tuple_length == 0:
            raise ValueError("PURE_2D slice result tuples cannot be empty.")

        main_outputs: list[Any] = []
        auxiliary_groups = [list() for _ in range(tuple_length - 1)]
        for result in collected:
            if not isinstance(result, tuple):
                raise TypeError(
                    "PURE_2D execution cannot mix tuple and non-tuple slice results."
                )
            if len(result) != tuple_length:
                raise ValueError(
                    "PURE_2D execution requires all tuple slice results to have the "
                    "same arity."
                )
            main_outputs.append(result[0])
            for index, value in enumerate(result[1:]):
                auxiliary_groups[index].append(value)

        return cls(main_outputs=main_outputs, auxiliary_groups=tuple(auxiliary_groups))


def contextualize_main_image_output(source_image: Any, result: Any) -> Any:
    """Preserve source image context when plain array callables return plain arrays."""
    if isinstance(result, RuntimeOutputBundle):
        return result
    if isinstance(result, tuple):
        if not result:
            return result
        return (
            contextualize_main_image_output(source_image, result[0]),
            *result[1:],
        )
    if isinstance(result, RuntimeArrayPayload):
        return result
    if not isinstance(result, np.ndarray):
        return result
    if (
        image_payload_mask(source_image) is None
        and not image_payload_metadata(source_image).has_values
    ):
        return result
    return with_image_payload_data(source_image, result)


class Pure2DRegisteredStrategyFamily(ABC):
    """Shared cached registry-family mechanics for PURE_2D strategy ABCs."""

    __registry__: ClassVar[Mapping[Any, type["Pure2DRegisteredStrategyFamily"]]]
    value_type: ClassVar[type[Any] | None] = None
    include_in_family: ClassVar[bool] = True

    @classmethod
    @abstractmethod
    def family_root(cls) -> type["Pure2DRegisteredStrategyFamily"]:
        """Return the concrete AutoRegisterMeta root for this strategy family."""

    @classmethod
    @lru_cache(maxsize=None)
    def registered_families(cls) -> tuple[type["Pure2DRegisteredStrategyFamily"], ...]:
        root_type = cls.family_root()
        family_types: list[type[Pure2DRegisteredStrategyFamily]] = []
        for strategy_type in root_type.__registry__.values():
            for candidate_type in strategy_type.mro():
                if (
                    candidate_type is root_type
                    or not isinstance(candidate_type, type)
                    or not issubclass(candidate_type, root_type)
                    or not candidate_type.include_in_family
                    or candidate_type in family_types
                ):
                    continue
                family_types.append(candidate_type)
        return tuple(family_types)

    @classmethod
    @lru_cache(maxsize=None)
    def registered_strategies(cls) -> tuple["Pure2DRegisteredStrategyFamily", ...]:
        """Return cached nominal strategy instances."""
        return tuple(strategy_type() for strategy_type in cls.registered_families())

    @classmethod
    def nearest_registered_strategy(
        cls,
        strategy_type: type[Any],
        *,
        supports: Callable[[Any], bool],
        distance: Callable[[Any], int],
    ) -> Any | None:
        """Return the nearest registered strategy satisfying ``supports``."""
        candidates = [
            strategy
            for strategy in cls.registered_strategies()
            if isinstance(strategy, strategy_type) and supports(strategy)
        ]
        if not candidates:
            return None
        return min(candidates, key=distance)

    @classmethod
    @lru_cache(maxsize=None)
    def accepted_value_types(cls) -> tuple[type[Any], ...]:
        """Return nominal value types owned by this registered family."""
        root_type = cls.family_root()
        return tuple(
            strategy_type.value_type
            for strategy_type in root_type.__registry__.values()
            if (strategy_type.value_type is not None and issubclass(strategy_type, cls))
        )

    def type_distance(self, value: Any) -> int:
        """Return nearest nominal MRO distance for this strategy family."""
        declared_types = self.accepted_value_types()
        if not declared_types:
            return len(object.__mro__)
        return min(
            type(value).mro().index(declared_type)
            for declared_type in declared_types
            if isinstance(value, declared_type)
        )


class Pure2DInputSlicer(Pure2DRegisteredStrategyFamily, metaclass=AutoRegisterMeta):
    """Unstack a PURE_2D main-flow input into nominal per-slice values."""

    __registry_key__ = PURE2D_VALUE_TYPE_REGISTRY_KEY
    __registry__: ClassVar[dict[Any, type["Pure2DInputSlicer"]]] = {}

    @classmethod
    def family_root(cls) -> type["Pure2DInputSlicer"]:
        return Pure2DInputSlicer

    @classmethod
    def strategy_for_value(cls, value: Any) -> "Pure2DInputSlicer":
        """Select the nearest registered slicer for a PURE_2D input value."""
        slicer = cls.nearest_registered_strategy(
            Pure2DInputSlicer,
            supports=lambda strategy: strategy.supports(value),
            distance=lambda strategy: strategy.type_distance(value),
        )
        if slicer is None:
            raise TypeError(
                "PURE_2D execution requires a registered input slicer for "
                f"{type(value).__name__}."
            )
        return slicer

    def supports(self, value: Any) -> bool:
        accepted_types = self.accepted_value_types()
        return bool(accepted_types) and isinstance(value, accepted_types)

    @abstractmethod
    def slice_value(self, value: Any, memory_type: str) -> tuple[Any, ...]:
        """Return nominal per-slice values for one PURE_2D input."""

    @abstractmethod
    def is_single_plane_value(self, value: Any) -> bool:
        """Return whether this value should bypass slice/restack execution."""


class NumPyPure2DInputSlicer(Pure2DInputSlicer):
    """Treat an unannotated ndarray as one image plane."""

    value_type = np.ndarray

    def is_single_plane_value(self, value: np.ndarray) -> bool:
        del value
        return True

    def slice_value(self, value: np.ndarray, memory_type: str) -> tuple[Any, ...]:
        del memory_type
        return (value,)


class ImagePayloadPure2DInputSlicer(Pure2DInputSlicer):
    """Slice image payloads while preserving per-slice image context."""

    value_type = None

    def is_single_plane_value(self, value: Any) -> bool:
        return image_payload_metadata(value).plane_axis is None

    def slice_value(self, value: Any, memory_type: str) -> tuple[Any, ...]:
        data = image_payload_data(value)
        if self.is_single_plane_value(value):
            return (value,)
        metadata = image_payload_metadata(value)
        slices = unstack_runtime_slices(
            data,
            memory_type,
            0,
            expected_count=metadata.source_provenance.source_plane_count or None,
        )
        return tuple(
            image_payload_slice_context(value, slice_data, slice_index)
            for slice_index, slice_data in enumerate(slices)
        )


class MaskedImagePayloadPure2DInputSlicer(ImagePayloadPure2DInputSlicer):
    """Register masked image payloads for PURE_2D input slicing."""

    value_type = MaskedImagePayload


class ImageMetadataPayloadPure2DInputSlicer(ImagePayloadPure2DInputSlicer):
    """Register image metadata payloads for PURE_2D input slicing."""

    value_type = ImageMetadataPayload


class Pure2DAuxiliaryOutputAggregator(
    Pure2DRegisteredStrategyFamily,
    metaclass=AutoRegisterMeta,
):
    """Aggregate one auxiliary PURE_2D output position across slices."""

    __registry_key__ = PURE2D_VALUE_TYPE_REGISTRY_KEY
    __registry__: ClassVar[dict[Any, type["Pure2DAuxiliaryOutputAggregator"]]] = {}

    @classmethod
    def family_root(cls) -> type["Pure2DAuxiliaryOutputAggregator"]:
        return Pure2DAuxiliaryOutputAggregator

    @classmethod
    def aggregate(
        cls,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis = RuntimePlaneAxis.RUNTIME_SLICE,
    ) -> Any:
        if not values:
            raise ValueError("PURE_2D auxiliary aggregation requires output values.")
        aggregator = cls.nearest_registered_strategy(
            Pure2DAuxiliaryOutputAggregator,
            supports=lambda strategy: strategy.supports(values),
            distance=lambda strategy: strategy.type_distance(values),
        )
        if aggregator is not None:
            return aggregator.aggregate_values(
                values,
                memory_type,
                plane_axis=plane_axis,
            )
        if len(values) == 1:
            return values[0]
        raise TypeError(
            "PURE_2D auxiliary outputs spanning multiple slices require a "
            "registered nominal aggregator, got "
            f"{type(values[0]).__name__}."
        )

    def supports(self, values: list[Any]) -> bool:
        accepted_types = self.accepted_value_types()
        return bool(accepted_types) and all(
            isinstance(value, accepted_types) for value in values
        )

    def owns_mixed_values(self, values: list[Any]) -> bool:
        return False

    def type_distance(self, values: list[Any]) -> int:
        declared_types = self.accepted_value_types()
        if not declared_types:
            return len(object.__mro__)
        return max(
            min(
                type(value).mro().index(declared_type)
                for declared_type in declared_types
                if isinstance(value, declared_type)
            )
            for value in values
        )

    @abstractmethod
    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> Any:
        """Aggregate compatible per-slice auxiliary values."""


class DirectedRelationshipPure2DOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Aggregate directed relationships over one declared PURE_2D plane axis."""

    value_type = DirectedObjectRelationshipPayload

    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> DirectedObjectRelationshipPayload:
        del memory_type, plane_axis
        return DirectedObjectRelationshipPayload(
            source_ids=tuple(
                source_id for value in values for source_id in value.source_ids
            ),
            target_ids=tuple(
                target_id for value in values for target_id in value.target_ids
            ),
            slice_indices=tuple(
                slice_index
                for slice_index, value in enumerate(values)
                for _target_id in value.target_ids
            ),
            slice_count=len(values),
        )


class SpatialGridPure2DOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Preserve one declared spatial grid for each PURE_2D runtime slice."""

    value_type = SpatialGrid

    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> RuntimeSliceAlignedValues[SpatialGrid]:
        del memory_type, plane_axis
        return RuntimeSliceAlignedValues(tuple(values))


class AlignedImageStackPure2DOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Transpose aligned image surfaces into declared output-plane stacks."""

    value_type = AlignedImageStack

    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> AlignedImageStack:
        aligned_values = tuple(values)
        first = aligned_values[0]
        output_count = len(first.slices)
        if any(len(value.slices) != output_count for value in aligned_values):
            raise ValueError(
                "Aligned main outputs must expose the same image surface count "
                "across every PURE_2D slice."
            )
        return first.with_slices(
            tuple(
                Pure2DAuxiliaryOutputAggregator.aggregate(
                    [value.slices[index] for value in aligned_values],
                    memory_type,
                    plane_axis=plane_axis,
                )
                for index in range(output_count)
            )
        )


class RuntimeArrayPure2DAuxiliaryOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Stack nominal runtime array payloads through their concrete array data."""

    value_type = None
    include_in_family = False

    def supports(self, values: list[Any]) -> bool:
        return super().supports(values) or self.owns_mixed_values(values)

    def owns_mixed_values(self, values: list[Any]) -> bool:
        accepted_types = self.accepted_value_types()
        return (
            bool(accepted_types)
            and any(isinstance(value, accepted_types) for value in values)
            and all(self._accepts_mixed_value(value) for value in values)
        )

    def type_distance(self, values: list[Any]) -> int:
        if self.owns_mixed_values(values):
            return 0
        return super().type_distance(values)

    def _accepts_mixed_value(self, value: Any) -> bool:
        return isinstance(value, np.ndarray)

    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> Any:
        del values, memory_type, plane_axis
        raise NotImplementedError(
            "Concrete runtime-array aggregators must implement aggregate_values."
        )

    def stack_array_slices(self, values: list[Any], memory_type: str) -> Any:
        """Stack an explicitly collected dense runtime-slice sequence."""

        return stack_runtime_slices(values, memory_type, 0)


class ImagePayloadPure2DAuxiliaryOutputAggregator(
    RuntimeArrayPure2DAuxiliaryOutputAggregator
):
    """Stack image payload slices and reattach composed runtime image context."""

    value_type = None
    include_in_family = True

    def _accepts_mixed_value(self, value: Any) -> bool:
        accepted_types = self.accepted_value_types()
        return (
            bool(accepted_types) and isinstance(value, accepted_types)
        ) or super()._accepts_mixed_value(value)

    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> Any:
        data_values = [image_payload_data(value) for value in values]
        data = stack_runtime_slices(data_values, memory_type, 0)
        masks = [image_payload_mask(value) for value in values]
        present_masks = [mask for mask in masks if mask is not None]
        if present_masks and len(present_masks) != len(masks):
            raise ValueError(
                "Cannot aggregate a mix of masked and unmasked image payloads."
            )
        mask = (
            None
            if not present_masks
            else stack_runtime_slices(present_masks, memory_type, 0)
        )
        return ImagePayloadMetadata.compose(
            tuple(values),
            mode=ImagePayloadMetadataCompositionMode.for_plane_axis(plane_axis),
        ).payload_with(data, mask)


class MaskedImagePayloadPure2DAuxiliaryOutputAggregator(
    ImagePayloadPure2DAuxiliaryOutputAggregator
):
    """Register masked image payloads for PURE_2D auxiliary aggregation."""

    value_type = MaskedImagePayload


class ImageMetadataPayloadPure2DAuxiliaryOutputAggregator(
    ImagePayloadPure2DAuxiliaryOutputAggregator
):
    """Register image metadata payloads for PURE_2D auxiliary aggregation."""

    value_type = ImageMetadataPayload


class ObjectLabelPayloadPure2DAuxiliaryOutputAggregator(
    RuntimeArrayPure2DAuxiliaryOutputAggregator
):
    """Delegate object-label slice aggregation to the runtime-value authority."""

    value_type = ObjectLabelPayload
    include_in_family = True

    def _accepts_mixed_value(self, value: Any) -> bool:
        return isinstance(value, self.value_type)

    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> Any:
        return ObjectLabelPure2DSliceAggregator.aggregate(
            values,
            memory_type,
            plane_axis=plane_axis,
        )


class ObjectLabelSetPure2DAuxiliaryOutputAggregator(
    ObjectLabelPayloadPure2DAuxiliaryOutputAggregator
):
    """Register native object-label values for PURE_2D auxiliary aggregation."""

    value_type = ObjectLabelSet


class NumPyPure2DAuxiliaryOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Stack ndarray outputs from an explicitly declared PURE_2D slice batch."""

    value_type = np.ndarray

    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> Any:
        return ImagePayloadMetadata(plane_axis=plane_axis).payload_with(
            stack_runtime_slices(values, memory_type, 0)
        )


class ColumnarRowsPure2DAuxiliaryOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Stamp nominal columnar measurement rows with their PURE_2D slice identity."""

    value_type = ColumnarRows

    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> ColumnarRows:
        del memory_type, plane_axis
        if len(values) == 1:
            return self.slice_aggregated_rows(values[0], 0)
        return ConcatenatedColumnarRows(
            tuple(
                self.slice_projected_rows(value, slice_index)
                for slice_index, value in enumerate(values)
            )
        )

    @staticmethod
    def slice_projected_rows(rows: ColumnarRows, slice_index: int) -> ColumnarRows:
        if not isinstance(rows, ColumnarRows):
            raise TypeError(
                "ColumnarRowsPure2DAuxiliaryOutputAggregator requires ColumnarRows, "
                f"got {type(rows).__name__}."
            )
        projected_rows = MeasurementRowsAxisProjection.from_rows(
            rows
        ).project_runtime_slice_index(slice_index)
        if not isinstance(projected_rows, ColumnarRows):
            raise TypeError(
                "ColumnarRows axis projection must return ColumnarRows, got "
                f"{type(projected_rows).__name__}."
            )
        return projected_rows

    @staticmethod
    def slice_aggregated_rows(rows: ColumnarRows, slice_index: int) -> ColumnarRows:
        if not isinstance(rows, ColumnarRows):
            raise TypeError(
                "ColumnarRowsPure2DAuxiliaryOutputAggregator requires ColumnarRows, "
                f"got {type(rows).__name__}."
            )
        projected_rows = MeasurementRowsAxisProjection.from_rows(
            rows
        ).project_runtime_slice_index(slice_index)
        if not isinstance(projected_rows, ColumnarRows):
            raise TypeError(
                "ColumnarRows axis projection must return ColumnarRows, got "
                f"{type(projected_rows).__name__}."
            )
        return projected_rows


class FlatSequencePure2DAuxiliaryOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Concatenate explicitly sequence-valued tuple/list auxiliary outputs."""

    value_type = None

    def supports(self, values: list[Any]) -> bool:
        return super().supports(values) and not any(
            isinstance(item, (np.ndarray, ColumnarRows))
            for value in values
            for item in value
        )

    def aggregate_values(
        self,
        values: list[Any],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> Any:
        del memory_type, plane_axis
        flattened: list[Any] = []
        for value in values:
            flattened.extend(value)
        return flattened


class ListPure2DAuxiliaryOutputAggregator(FlatSequencePure2DAuxiliaryOutputAggregator):
    """Register list auxiliary outputs for PURE_2D sequence aggregation."""

    value_type = list


class TuplePure2DAuxiliaryOutputAggregator(FlatSequencePure2DAuxiliaryOutputAggregator):
    """Register tuple auxiliary outputs for PURE_2D sequence aggregation."""

    value_type = tuple


# Enums for OpenHCS principle compliance (replace magic strings)
class ModuleFilterComponents(Enum):
    """Components to filter out when generating tags from module paths."""

    BACKENDS = "backends"
    PROCESSING = "processing"
    OPENHCS = "openhcs"

    @classmethod
    def should_skip(cls, component: str) -> bool:
        """Check if component should be skipped in tag generation."""
        return any(component == item.value for item in cls)


class ProcessingContractDeclaration(ABC):
    """Nominal execution contract owned by a ProcessingContract member."""

    def runtime_parameter_types(
        self,
    ) -> tuple[type["ContractRuntimeParameter"], ...]:
        """Return runtime control parameter declarations owned by this contract."""
        return ()

    def injected_runtime_parameter_types(
        self,
    ) -> tuple[type["ContractRuntimeParameter"], ...]:
        """Return contract controls that belong on the public wrapper signature."""
        return ()

    def execution_parameter_names(self) -> frozenset[str]:
        """Runtime controls that should remain present for contract execution."""
        return frozenset(
            parameter_type.require_parameter_name()
            for parameter_type in self.runtime_parameter_types()
            if parameter_type.preserve_for_execution
        )

    def semantic_control_parameter_names(self) -> frozenset[str]:
        """Runtime controls that select this contract's semantic execution mode."""
        return frozenset(
            parameter_type.require_parameter_name()
            for parameter_type in self.runtime_parameter_types()
            if parameter_type.is_semantic_control
        )

    def main_flow_output_source_payload(self, source_payload: Any) -> Any:
        """Return source context projected through this contract's output domain."""

        return source_payload

    def injected_semantic_control_parameter_names(self) -> frozenset[str]:
        """Semantic controls that this contract may inject into public callables."""
        return frozenset(
            parameter_type.require_parameter_name()
            for parameter_type in self.injected_runtime_parameter_types()
            if parameter_type.is_semantic_control
        )

    def consume_semantic_control(
        self,
        kwargs: MutableMapping[str, Any],
    ) -> bool:
        """Consume semantic selectors from kwargs and report whether enabled."""
        control_values = tuple(
            kwargs.pop(name)
            for name in tuple(self.semantic_control_parameter_names())
            if name in kwargs
        )
        return any(bool(value) for value in control_values)

    @property
    def variable_component_stack_requirement(
        self,
    ) -> VariableComponentStackRequirement | None:
        """Return the stack-axis requirement declared by this contract type."""
        return None

    @abstractmethod
    def execute(self, registry, func, image, *args, **kwargs):
        """Execute one callable according to this contract declaration."""


class VariableComponentStackProcessingContract(ProcessingContractDeclaration):
    """Marker parent for contracts that require a real variable-component stack."""

    @property
    def variable_component_stack_requirement(
        self,
    ) -> VariableComponentStackRequirement:
        return AlwaysRequiresVariableComponentStack()


class SemanticControlVariableComponentStackProcessingContract(
    VariableComponentStackProcessingContract
):
    """Stack contract selected off unless a semantic-control parameter is enabled."""

    @property
    def variable_component_stack_requirement(
        self,
    ) -> VariableComponentStackRequirement:
        return SemanticControlVariableComponentStackRequirement(
            self.runtime_parameter_types()
        )


class Pure3DProcessingContract(VariableComponentStackProcessingContract):
    """Execute a callable once with full image-domain semantics."""

    def execute(self, registry, func, image, *args, **kwargs):
        return registry.execute_pure_3d(func, image, *args, **kwargs)


class Pure2DProcessingContract(ProcessingContractDeclaration):
    """Execute a callable as independent 2D slices."""

    def execute(self, registry, func, image, *args, **kwargs):
        return registry.execute_pure_2d(func, image, *args, **kwargs)


class FlexibleProcessingContract(
    SemanticControlVariableComponentStackProcessingContract
):
    """Choose 2D or full-stack semantics using this contract's control hook."""

    def runtime_parameter_types(
        self,
    ) -> tuple[type["ContractRuntimeParameter"], ...]:
        return (SliceBySliceRuntimeParameter,)

    def injected_runtime_parameter_types(
        self,
    ) -> tuple[type["ContractRuntimeParameter"], ...]:
        return self.runtime_parameter_types()

    def execute(self, registry, func, image, *args, **kwargs):
        if self.consume_semantic_control(kwargs):
            return ProcessingContract.PURE_2D.execute(
                registry,
                func,
                image,
                *args,
                **kwargs,
            )
        return ProcessingContract.PURE_3D.execute(
            registry,
            func,
            image,
            *args,
            **kwargs,
        )


class VolumetricToSliceProcessingContract(VariableComponentStackProcessingContract):
    """Execute a volumetric-to-slice callable through its declared hook."""

    def main_flow_output_source_payload(self, source_payload: Any) -> Any:
        """Consume the declared leading plane axis while preserving provenance."""

        metadata = image_payload_metadata(source_payload)
        if not metadata.has_values:
            return source_payload
        if metadata.plane_axis is None:
            raise ValueError(
                "VOLUMETRIC_TO_SLICE output requires a declared input plane axis."
            )
        return metadata.collapse_leading_plane_axis().attach_to(source_payload)

    def execute(self, registry, func, image, *args, **kwargs):
        result = registry.execute_volumetric_to_slice(
            func,
            image,
            *args,
            **kwargs,
        )
        return contextualize_main_image_output(
            self.main_flow_output_source_payload(image),
            result,
        )


class ProcessingContract(Enum):
    """Unified contract classification with nominal declaration hooks."""

    PURE_3D = Pure3DProcessingContract
    PURE_2D = Pure2DProcessingContract
    FLEXIBLE = FlexibleProcessingContract
    VOLUMETRIC_TO_SLICE = VolumetricToSliceProcessingContract

    def __new__(
        cls,
        declaration_type: type[ProcessingContractDeclaration],
    ) -> "ProcessingContract":
        member = object.__new__(cls)
        member._value_ = declaration_type.__name__
        member._declaration_type = declaration_type
        return member

    @property
    def declaration(self) -> ProcessingContractDeclaration:
        """Return this member's nominal execution declaration."""
        return self._declaration_type()

    @property
    def declared_name(self) -> str:
        """Return the lowercase metadata name used in declarations."""
        return self.name.lower()

    @property
    def variable_component_stack_requirement(
        self,
    ) -> VariableComponentStackRequirement | None:
        """Return the stack-axis requirement declared by this contract type."""
        return self.declaration.variable_component_stack_requirement

    @classmethod
    def from_declared_name(cls, contract_name: str) -> "ProcessingContract | None":
        """Resolve a declared contract name to the canonical enum member."""
        normalized = contract_name.upper()
        if normalized not in cls.__members__:
            return None
        return cls[normalized]

    def execute(self, registry, func, image, *args, **kwargs):
        """Execute this contract through its declaration hook."""
        return self.declaration.execute(registry, func, image, *args, **kwargs)


@dataclass(frozen=True)
class FunctionMetadata:
    """Clean metadata with no library-specific leakage."""

    # Core fields only
    name: str
    func: Callable
    contract: ProcessingContract
    registry: "LibraryRegistryBase"  # Reference to the registry that registered this function - REQUIRED
    module: str = ""
    doc: str = ""
    tags: List[str] = field(default_factory=list)
    original_name: str = ""  # Original function name for cache reconstruction
    memory_type: str | None = None

    @property
    def display_name(self) -> str:
        """Human-readable function name for catalogs and selectors."""
        if self.original_name:
            return self.original_name
        return self.name

    def get_memory_type(self) -> str:
        """
        Get the actual memory type (backend) of this function.

        Returns the memory type recorded at metadata creation time, otherwise
        the registry-level memory type for older cache entries.

        Returns:
            Memory type string (e.g., "cupy", "numpy", "torch", "pyclesperanto")
        """
        if self.memory_type is not None:
            return self.memory_type
        return self.registry.get_memory_type()

    def get_registry_name(self) -> str:
        """
        Get the registry name that registered this function.

        Returns:
            Registry name string (e.g., "openhcs", "skimage", "cupy", "pyclesperanto")
        """
        return self.registry.library_name


class ContractRuntimeParameter(
    KeywordRuntimeParameter,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal declaration for parameters owned by ProcessingContract execution."""

    __registry_key__ = "parameter_name"
    __skip_if_no_key__ = True

    annotation_type: ClassVar[type]
    preserve_for_execution: ClassVar[bool] = False
    is_semantic_control: ClassVar[bool] = False

    @classmethod
    def registered_parameter_types(
        cls,
    ) -> tuple[type["ContractRuntimeParameter"], ...]:
        return tuple(cls.__registry__.values())


class SliceBySliceRuntimeParameter(ContractRuntimeParameter):
    """Flexible-contract semantic selector for plane-wise execution."""

    parameter_name = "slice_by_slice"
    annotation_type = bool
    parameter_default = False
    preserve_for_execution = True
    is_semantic_control = True


class LibraryRegistryBase(ABC, metaclass=AutoRegisterMeta):
    """
    Minimal ABC for all library registries.

    Provides only essential contracts that all registries must implement,
    regardless of whether they use runtime testing or explicit contracts.

    Registry auto-created and stored as LibraryRegistryBase.__registry__.
    Subclasses auto-register by setting _registry_name class attribute.
    """

    __registry_key__ = "_registry_name"

    _registry_name: Optional[str] = (
        None  # Override in subclasses (e.g., 'pyclesperanto', 'cupy')
    )

    # Common exclusions across all libraries
    COMMON_EXCLUSIONS = {
        "imread",
        "imsave",
        "load",
        "save",
        "read",
        "write",
        "show",
        "imshow",
        "plot",
        "display",
        "view",
        "visualize",
        "info",
        "help",
        "version",
        "test",
        "benchmark",
    }
    EXCLUSIONS = COMMON_EXCLUSIONS

    # Abstract class attributes - each implementation must define these
    MODULES_TO_SCAN: List[str]
    MEMORY_TYPE: (
        str  # Memory type string value (e.g., "pyclesperanto", "cupy", "numpy")
    )
    FLOAT_DTYPE: Any  # Library-specific float32 type (np.float32, cp.float32, etc.)

    def __init__(self, library_name: str):
        """
        Initialize registry for a specific library.

        Args:
            library_name: Name of the library (e.g., "pyclesperanto", "skimage")
        """
        self.library_name = library_name
        self._cache_path = get_cache_file_path(f"{library_name}_function_metadata.json")
        self._library_warmed = False
        self._function_metadata_cache: Optional[Dict[str, FunctionMetadata]] = None
        self._function_metadata_cache_modules: tuple[str, ...] | None = None

    # ===== ESSENTIAL ABC METHODS =====

    # ===== LIBRARY IDENTIFICATION =====
    @abstractmethod
    def get_library_version(self) -> str:
        """Get library version for cache validation."""
        pass

    @abstractmethod
    def is_library_available(self) -> bool:
        """Check if the library is available for import."""
        pass

    # ===== FUNCTION DISCOVERY =====
    @abstractmethod
    def discover_functions(self) -> Dict[str, FunctionMetadata]:
        """Discover and return function metadata. Must be implemented by subclasses."""
        pass

    # ===== CONTRACT HANDLING =====
    def apply_contract_wrapper(
        self, func: Callable, contract: ProcessingContract
    ) -> Callable:
        """Apply contract wrapper with nominal runtime parameter injection."""
        from functools import wraps
        import inspect
        from python_introspect import (
            Enableable,
            mark_enableable,
            set_signature_analysis_target,
        )
        from openhcs.core.callable_contract import (
            CallableContract,
            FunctionStepExecutionScope,
        )
        from openhcs.core.config import runtime_config_parameter

        callable_contract = CallableContract.from_callable(func)
        if callable_contract.execution_scope is FunctionStepExecutionScope.PLATE:
            original_sig = inspect.signature(func, eval_str=True)
            enabled_parameter = Enableable.parameter()
            parameters = list(original_sig.parameters.values())
            if enabled_parameter.name not in original_sig.parameters:
                insert_index = next(
                    (
                        index
                        for index, parameter in enumerate(parameters)
                        if parameter.kind is inspect.Parameter.VAR_KEYWORD
                    ),
                    len(parameters),
                )
                parameters.insert(insert_index, enabled_parameter)

            @wraps(func)
            def plate_wrapper(*args, **kwargs):
                return func(*args, **Enableable.without_parameter(kwargs))

            plate_wrapper.__signature__ = original_sig.replace(parameters=parameters)
            plate_wrapper.__annotations__ = inspect.get_annotations(
                func,
                eval_str=False,
            ).copy()
            plate_wrapper.__annotations__[enabled_parameter.name] = (
                enabled_parameter.annotation
            )
            set_signature_analysis_target(plate_wrapper, func)
            _set_registry_runtime_parameter_exclusions(
                plate_wrapper,
                plate_wrapper.__signature__,
                (),
                source=func,
            )
            from openhcs.core.callable_contract import attach_callable_contract_metadata

            attach_callable_contract_metadata(
                plate_wrapper,
                raw_processing_function=func,
            )
            mark_enableable(plate_wrapper, enabled_default=True)
            return plate_wrapper

        declaration = contract.declaration
        original_sig = inspect.signature(func, eval_str=True)
        allowed_semantic_control_names = (
            declaration.injected_semantic_control_parameter_names()
        )
        semantic_control_names = {
            parameter_type.require_parameter_name()
            for parameter_type in ContractRuntimeParameter.registered_parameter_types()
            if parameter_type.is_semantic_control
        }
        params_to_strip = semantic_control_names - allowed_semantic_control_names
        runtime_config_parameters: list[inspect.Parameter] = []
        public_original_parameters: list[inspect.Parameter] = []
        for parameter in original_sig.parameters.values():
            if parameter.name in params_to_strip:
                continue
            normalized_parameter = runtime_config_parameter(parameter)
            if normalized_parameter is None:
                public_original_parameters.append(parameter)
                continue
            runtime_config_parameters.append(normalized_parameter)
            public_original_parameters.append(normalized_parameter)
        public_original_parameters = tuple(public_original_parameters)
        public_sig = original_sig.replace(parameters=public_original_parameters)
        param_names = {p.name for p in public_sig.parameters.values()}

        runtime_parameter_types = declaration.injected_runtime_parameter_types()
        runtime_parameter_names = (
            *(parameter.name for parameter in runtime_config_parameters),
            *(
                parameter_type.require_parameter_name()
                for parameter_type in runtime_parameter_types
            ),
        )
        injected_signature_parameters = (
            Enableable.parameter(),
            *(
                parameter_type.parameter()
                for parameter_type in declaration.injected_runtime_parameter_types()
            ),
        )

        # Filter out already-existing parameters and declaration-name collisions.
        params_to_add: list[inspect.Parameter] = []
        seen_param_names = set(param_names)
        for parameter in injected_signature_parameters:
            if parameter.name in seen_param_names:
                continue
            params_to_add.append(parameter)
            seen_param_names.add(parameter.name)

        # If nothing to inject, return original function
        if not params_to_add and not params_to_strip and public_sig == original_sig:
            # Still brand the callable as Enableable metadata.
            from openhcs.core.callable_contract import attach_callable_contract_metadata

            mark_enableable(func, enabled_default=True)
            attach_callable_contract_metadata(
                func,
                runtime_bound_parameters=runtime_parameter_types,
            )
            _set_registry_runtime_parameter_exclusions(
                func,
                inspect.signature(func),
                runtime_parameter_names,
            )
            return func

        # Build new parameter list (insert before **kwargs)
        new_params = list(public_sig.parameters.values())
        insert_index = next(
            (
                i
                for i, parameter in enumerate(new_params)
                if parameter.kind == inspect.Parameter.VAR_KEYWORD
            ),
            len(new_params),
        )

        for parameter in params_to_add:
            new_params.insert(insert_index, parameter)
            insert_index += 1

        # Create wrapper
        @wraps(func)
        def wrapper(image, *args, **kwargs):
            if params_to_strip:
                kwargs = {
                    name: value
                    for name, value in kwargs.items()
                    if name not in params_to_strip
                }

            # Populate missing wrapper controls with their defaults from the signature
            # This is critical for internal calls between OpenHCS functions where
            # wrapper controls may not be explicitly passed (e.g., create_projection calling max_projection)
            from python_introspect import SignatureAnalyzer

            sig_params = SignatureAnalyzer.analyze(wrapper)
            signature_parameters = (
                *runtime_config_parameters,
                *injected_signature_parameters,
            )
            for parameter in signature_parameters:
                param_name = parameter.name
                if param_name not in kwargs and param_name in sig_params:
                    default_value = sig_params[param_name].default_value
                    if default_value is not inspect.Parameter.empty:
                        kwargs[param_name] = default_value

            # Keep only declared controls that participate in execution.
            execution_parameter_names = (
                frozenset(parameter.name for parameter in runtime_config_parameters)
                | declaration.execution_parameter_names()
            )
            params_to_filter = {
                parameter.name
                for parameter in signature_parameters
                if parameter.name not in execution_parameter_names
            }
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k not in params_to_filter
            }

            return contract.execute(self, func, image, *args, **filtered_kwargs)

        wrapper.__signature__ = public_sig.replace(parameters=new_params)
        wrapper.__annotations__ = inspect.get_annotations(func, eval_str=False).copy()
        for parameter in (
            *runtime_config_parameters,
            *injected_signature_parameters,
        ):
            wrapper.__annotations__[parameter.name] = parameter.annotation
        set_signature_analysis_target(wrapper, func)
        _set_registry_runtime_parameter_exclusions(
            wrapper,
            wrapper.__signature__,
            runtime_parameter_names,
            source=func,
        )

        # Explicitly copy nominal processing metadata when the wrapped callable owns it.
        from openhcs.core.callable_contract import attach_callable_contract_metadata
        from openhcs.core.function_contract_metadata import FunctionContractAttribute

        source_namespace = vars(func)
        processing_contract_key = FunctionContractAttribute.processing_contract
        if processing_contract_key in source_namespace:
            vars(wrapper)[processing_contract_key] = source_namespace[
                processing_contract_key
            ]
        attach_callable_contract_metadata(
            wrapper,
            raw_processing_function=func,
            runtime_bound_parameters=runtime_parameter_types,
        )

        # Nominal enable semantics: decorated callables are Enableable.
        # (Enableable is metadata only; enabled remains owned by python_introspect.)
        mark_enableable(wrapper, enabled_default=True)

        return wrapper

    def _get_function_by_name(self, module_path: str, func_name: str):
        """Get function object by module path and name."""
        module = importlib.import_module(module_path)
        try:
            return vars(module)[func_name]
        except KeyError as exc:
            raise AttributeError(func_name) from exc

    def create_library_adapter(
        self,
        original_func: Callable,
        contract: ProcessingContract,
    ) -> Callable:
        """Return the callable shape used before contract wrapping."""
        return original_func

    def reconstruct_cached_callable(
        self,
        func: Callable,
        contract: ProcessingContract,
    ) -> Callable:
        """Reconstruct one cached callable through this registry's runtime policy."""

        adapted_func = self.create_library_adapter(func, contract)
        return self.apply_contract_wrapper(adapted_func, contract)

    # ===== PROCESSING CONTRACT EXECUTION METHODS =====
    def execute_pure_3d(self, func, image, *args, **kwargs):
        """Execute a full-stack callable once and restore payload context."""
        result = (
            RuntimeCallablePolicy()
            .invocation(
                func,
                (image, *args),
                kwargs,
            )
            .call()
        )
        return contextualize_main_image_output(image, result)

    def execute_pure_2d(self, func, image, *args, **kwargs):
        """Execute 2D→2D function with unstack/restack wrapper."""
        # Get memory type from the decorated function
        memory_type = func.output_memory_type
        slicer = Pure2DInputSlicer.strategy_for_value(image)
        positional_kwargs = self._pure_2d_positional_kwargs(func, args)
        if self._pure_2d_full_stack_object_measurement(
            func,
            image,
            {**positional_kwargs, **kwargs},
        ):
            return (
                RuntimeCallablePolicy().invocation(func, (image, *args), kwargs).call()
            )
        if slicer.is_single_plane_value(image):
            return (
                RuntimeCallablePolicy().invocation(func, (image, *args), kwargs).call()
            )
        input_metadata = image_payload_metadata(image)
        plane_axis = input_metadata.plane_axis
        if plane_axis is None:
            raise ValueError(
                "PURE_2D multi-plane execution requires the input payload to "
                "declare its runtime plane axis."
            )
        source_aliases = input_metadata.source_image_names
        if args:
            signature = inspect.signature(func)
            parameters = tuple(signature.parameters.values())
            positional_parameters = tuple(
                parameter
                for parameter in parameters[1:]
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            )
            if len(args) > len(positional_parameters):
                raise TypeError(
                    f"{func.__name__} expected at "
                    f"most {len(positional_parameters)} positional argument(s) after "
                    f"image, got {len(args)}."
                )
            for parameter, value in zip(positional_parameters, args):
                kwargs.setdefault(parameter.name, value)
            args = args[len(positional_parameters) :]
        slices = slicer.slice_value(image, memory_type)
        declared_batch_executor = runtime_batch_executors_from_callable(func).get(
            RuntimeBatchExecutionDomain.PURE_2D_SLICES
        )
        batch_executor = (
            declared_batch_executor
            if callable(declared_batch_executor)
            else Pure2DSliceBatchExecutor.default_executor()
        )

        def execute_slice(
            slice_func: Callable[..., Any],
            slice_2d: Any,
            slice_kwargs: Mapping[str, Any],
            slice_index: int,
            slice_count: int,
        ) -> Any:
            projected_kwargs = RuntimeSliceProjection.kwargs_for_slice(
                slice_kwargs,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=plane_axis,
                    source_aliases=source_aliases,
                    plane_index=slice_index,
                    axis_size=slice_count,
                ),
            )
            return (
                RuntimeCallablePolicy()
                .invocation(
                    slice_func,
                    (slice_2d, *args),
                    projected_kwargs,
                )
                .call()
            )

        slice_results = batch_executor(
            RuntimePure2DSliceBatchRequest(
                func=func,
                slices_2d=tuple(slices),
                kwargs=kwargs,
                execute_slice=execute_slice,
            )
        )
        result_batch = Pure2DSliceResultBatch.from_results(slice_results)
        stacked_main_output = Pure2DAuxiliaryOutputAggregator.aggregate(
            result_batch.main_outputs,
            memory_type,
            plane_axis=plane_axis,
        )
        if not result_batch.auxiliary_groups:
            return stacked_main_output
        aggregated_auxiliary_outputs = tuple(
            Pure2DAuxiliaryOutputAggregator.aggregate(
                values,
                memory_type,
                plane_axis=plane_axis,
            )
            for values in result_batch.auxiliary_groups
        )
        return (stacked_main_output, *aggregated_auxiliary_outputs)

    @staticmethod
    def _pure_2d_positional_kwargs(
        func: Callable[..., Any],
        args: tuple[Any, ...],
    ) -> dict[str, Any]:
        """Project positional arguments after image into their callable names."""

        if not args:
            return {}
        signature = inspect.signature(func)
        parameters = tuple(signature.parameters.values())
        positional_parameters = tuple(
            parameter
            for parameter in parameters[1:]
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        )
        return {
            parameter.name: value
            for parameter, value in zip(positional_parameters, args)
        }

    @staticmethod
    def _pure_2d_full_stack_object_measurement(
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
    ) -> bool:
        """Return whether a PURE_2D wrapper must preserve a full object volume."""

        from openhcs.core.pipeline.function_contracts import (
            ObjectLabelInputExecutionMode,
            object_label_input_execution_mode_from_callable,
        )

        if (
            object_label_input_execution_mode_from_callable(func)
            is not ObjectLabelInputExecutionMode.FULL_STACK
        ):
            return False
        labels = kwargs.get("labels")
        if labels is None:
            return False
        del image
        return True

    def execute_volumetric_to_slice(self, func, image, *args, **kwargs):
        """Execute a 3D→2D function and return its scalar slice-domain result."""
        return (
            RuntimeCallablePolicy()
            .invocation(
                func,
                (image, *args),
                kwargs,
            )
            .call()
        )

    # ===== LIBRARY WARM-UP HOOK =====
    def _warmup_library(self) -> None:
        """
        Optional hook for registries that need to pre-initialize their library.

        Subclasses can override to run lightweight imports or self-tests that
        ensure required shared libraries are available before discovery begins.
        """
        return

    def _ensure_library_warmed(self) -> None:
        """Ensure library warm-up hook is invoked exactly once."""
        if self._library_warmed:
            return

        try:
            self._warmup_library()
        except Exception as exc:
            logger.warning(f"{self.library_name} warm-up failed: {exc}")
            raise

        self._library_warmed = True

    # ===== CACHING METHODS =====
    def load_or_discover_functions(self) -> Dict[str, FunctionMetadata]:
        """Load functions from cache or discover them if cache is invalid."""
        self._ensure_library_warmed()
        module_signature = tuple(self.MODULES_TO_SCAN)
        if (
            self._function_metadata_cache is not None
            and self._function_metadata_cache_modules == module_signature
        ):
            return self._function_metadata_cache
        logger.info(f"🔄 load_or_discover_functions called for {self.library_name}")

        cached_functions = self._load_from_cache()
        if cached_functions is not None:
            logger.info(
                f"✅ Loaded {len(cached_functions)} {self.library_name} functions from cache"
            )
            self._function_metadata_cache = cached_functions
            self._function_metadata_cache_modules = module_signature
            return cached_functions

        logger.info(
            f"🔍 Cache miss for {self.library_name} - performing full discovery"
        )
        functions = self.discover_functions()
        self._save_to_cache(functions)
        self._function_metadata_cache = functions
        self._function_metadata_cache_modules = module_signature
        return functions

    def _load_or_discover_functions(self) -> Dict[str, FunctionMetadata]:
        """Backward-compatible alias for older registry callers."""
        return self.load_or_discover_functions()

    def _load_from_cache(self) -> Optional[Dict[str, FunctionMetadata]]:
        """Load function metadata from cache with validation."""
        logger.debug(f"📂 LOAD FROM CACHE: Checking cache for {self.library_name}")

        if not self._cache_path.exists():
            logger.debug(
                f"📂 LOAD FROM CACHE: No cache file exists at {self._cache_path}"
            )
            return None

        try:
            with open(self._cache_path, "r") as f:
                cache_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Corrupt cache file {self._cache_path}, rebuilding")
            self._cache_path.unlink(missing_ok=True)
            return None

        if "functions" not in cache_data:
            return None

        cached_version = cache_data.get("library_version", "unknown")
        current_version = self.get_library_version()
        if cached_version != current_version:
            logger.info(
                f"{self.library_name} version changed ({cached_version} → {current_version}) - cache invalid"
            )
            return None

        cached_signature = cache_data.get("discovery_signature")
        current_signature = self.get_discovery_signature()
        if cached_signature != current_signature:
            logger.info(f"{self.library_name} discovery inputs changed - cache invalid")
            return None

        cache_timestamp = cache_data.get("timestamp", 0)
        cache_age_days = (time.time() - cache_timestamp) / (24 * 3600)
        if cache_age_days > 7:
            logger.debug(f"Cache is {cache_age_days:.1f} days old - rebuilding")
            return None

        logger.debug(
            f"📂 LOAD FROM CACHE: Loading {len(cache_data['functions'])} functions for {self.library_name}"
        )

        functions = {}
        for func_name, cached_data in cache_data["functions"].items():
            original_name = cached_data.get("original_name", func_name)
            try:
                func = self._get_function_by_name(
                    cached_data["module"],
                    original_name,
                )
            except (AttributeError, ImportError, ModuleNotFoundError) as exc:
                logger.warning(
                    "Registry cache entry %s is stale for %s; rebuilding %s cache: %s",
                    func_name,
                    self.library_name,
                    self.library_name,
                    exc,
                )
                self._discard_stale_cache()
                return None
            if not callable(func):
                logger.warning(
                    "Registry cache entry %s for %s resolved to non-callable %r; "
                    "rebuilding %s cache",
                    func_name,
                    self.library_name,
                    type(func).__name__,
                    self.library_name,
                )
                self._discard_stale_cache()
                return None
            contract = ProcessingContract[cached_data["contract"]]

            final_func = self.reconstruct_cached_callable(func, contract)

            metadata = FunctionMetadata(
                name=func_name,
                func=final_func,
                contract=contract,
                registry=self,
                module=cached_data.get("module", ""),
                doc=cached_data.get("doc", ""),
                tags=cached_data.get("tags", []),
                original_name=cached_data.get("original_name", func_name),
                memory_type=cached_data.get("memory_type", self.get_memory_type()),
            )
            functions[func_name] = metadata

        return functions

    def get_discovery_signature(self) -> str:
        """Return the existing JSON cache's discovery-input signature."""
        signature = {
            "registry_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "modules_to_scan": list(self.MODULES_TO_SCAN),
            "source_mtimes": self.cache_source_mtimes(),
        }
        return json.dumps(signature, sort_keys=True)

    def cache_source_mtimes(self) -> Dict[str, float]:
        """Return optional scanned source mtimes for the existing JSON cache."""
        return {}

    def _save_to_cache(self, functions: Dict[str, FunctionMetadata]) -> None:
        """Save function metadata to cache."""
        writable_parent = self._writable_cache_parent()
        if writable_parent is None:
            logger.warning(
                "Registry cache path %s is not writable; using discovered "
                "%s functions without refreshing the disk cache.",
                self._cache_path,
                self.library_name,
            )
            return

        cache_data = {
            "cache_version": "1.0",
            "library_version": self.get_library_version(),
            "discovery_signature": self.get_discovery_signature(),
            "timestamp": time.time(),
            "functions": {
                func_name: {
                    "name": metadata.name,
                    "original_name": metadata.original_name,
                    "module": metadata.module,
                    "memory_type": metadata.get_memory_type(),
                    "contract": metadata.contract.name,
                    "doc": metadata.doc,
                    "tags": metadata.tags,
                }
                for func_name, metadata in functions.items()
            },
        }

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"💾 Saved {len(functions)} {self.library_name} functions to cache")

    def _discard_stale_cache(self) -> None:
        """Best-effort stale-cache deletion without blocking fresh discovery."""
        try:
            self._cache_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Registry cache path %s could not be invalidated; using fresh "
                "%s discovery without refreshing the disk cache: %s",
                self._cache_path,
                self.library_name,
                exc,
            )

    def _writable_cache_parent(self) -> Optional[str]:
        """Return the nearest existing writable cache parent, or None."""
        parent = self._cache_path.parent
        while not parent.exists():
            if parent.parent == parent:
                return None
            parent = parent.parent
        if not parent.is_dir():
            return None
        if not os.access(parent, os.W_OK | os.X_OK):
            return None
        return str(parent)

    def get_memory_type(self) -> str:
        """Get the memory type string value for this library."""
        return self.MEMORY_TYPE

    def get_module_patterns(self) -> List[str]:
        """Get module patterns that identify this library (can be overridden by implementations)."""
        # Default: just the library name
        return [self.library_name.lower()]

    def get_display_name(self) -> str:
        """Get display name for this library (can be overridden by implementations)."""
        # Default: capitalize library name
        return self.library_name.title()

    # ===== FUNCTION DISCOVERY =====
    def get_modules_to_scan(self) -> List[Tuple[str, Any]]:
        """
        Get list of (module_name, module_object) tuples to scan for functions.
        Uses the MODULES_TO_SCAN class attribute and library object from get_library_object().

        Returns:
            List of (name, module) pairs where name is for identification
            and module is the actual module object to scan.
        """
        library = self.get_library_object()
        modules = []
        for module_name in self.MODULES_TO_SCAN:
            if module_name == "":
                # Empty string means scan the main library namespace
                module = library
                modules.append(("main", module))
            else:
                try:
                    module = vars(library)[module_name]
                except KeyError as exc:
                    raise AttributeError(module_name) from exc
                modules.append((module_name, module))
        return modules

    @abstractmethod
    def get_library_object(self):
        """Get the main library object to scan for modules. Library-specific implementation."""
        pass


class RuntimeTestingRegistryBase(LibraryRegistryBase):
    """
    Extended ABC for libraries that require runtime testing.

    Adds runtime testing methods for libraries that don't have explicit
    processing contracts and need behavioral classification through testing.
    """

    def create_test_arrays(self) -> Tuple[Any, Any]:
        """
        Create test arrays appropriate for this library.

        Returns:
            Tuple of (test_3d, test_2d) arrays for behavior testing
        """
        test_3d = self._create_array((3, 20, 20), self._get_float_dtype())
        test_2d = self._create_array((20, 20), self._get_float_dtype())
        return test_3d, test_2d

    @abstractmethod
    def _create_array(self, shape: Tuple[int, ...], dtype):
        """Create array with specified shape and dtype. Library-specific implementation."""
        pass

    def _get_float_dtype(self):
        """Get the appropriate float dtype for this library."""
        return self.FLOAT_DTYPE

    # ===== CORE BEHAVIOR CONTRACT =====
    def classify_function_behavior(
        self,
        func: Callable,
        declared_contract: Optional[ProcessingContract] = None,
    ) -> Tuple[ProcessingContract, bool]:
        """Classify function behavior by testing 3D and 2D inputs, or use declared contract if provided."""

        # Fast path: If explicit contract is declared, use it directly (skip runtime testing)
        if declared_contract is not None:
            return declared_contract, True
        test_3d, test_2d = self.create_test_arrays()

        def test_function(test_array):
            """Test function with array, return (success, result)."""
            try:
                result = func(test_array)
                return True, result
            except Exception:
                return False, None

        works_3d, result_3d = test_function(test_3d)
        works_2d, _ = test_function(test_2d)

        # Classification lookup table
        classification_map = {
            (True, True): self._classify_dual_support(result_3d),
            (True, False): ProcessingContract.PURE_3D,
            (False, True): ProcessingContract.PURE_2D,
            (False, False): None,  # Invalid function
        }

        contract = classification_map[(works_3d, works_2d)]
        is_valid = works_3d or works_2d

        return contract, is_valid

    def _classify_dual_support(self, result_3d):
        """Classify functions that work on both 3D and 2D inputs."""
        if result_3d is not None:
            # Handle tuple results (some functions return multiple arrays)
            if isinstance(result_3d, tuple):
                # Check the first element if it's a tuple
                first_result = result_3d[0] if len(result_3d) > 0 else None
                if (
                    isinstance(first_result, (RuntimeArrayPayload, np.ndarray))
                    and first_result.ndim == 2
                ):
                    return ProcessingContract.VOLUMETRIC_TO_SLICE
            # Handle single array results
            elif (
                isinstance(result_3d, (RuntimeArrayPayload, np.ndarray))
                and result_3d.ndim == 2
            ):
                return ProcessingContract.VOLUMETRIC_TO_SLICE
        return ProcessingContract.FLEXIBLE

    @abstractmethod
    def _stack_2d_results(self, func, test_3d):
        """Stack 2D results. Library-specific implementation required."""
        pass

    @abstractmethod
    def _arrays_close(self, arr1, arr2):
        """Compare arrays. Library-specific implementation required."""
        pass

    def create_library_adapter(
        self, original_func: Callable, contract: ProcessingContract
    ) -> Callable:
        """Create adapter with library-specific processing only."""
        import inspect

        func_name = original_func.__name__

        logger.debug(
            "CREATE LIBRARY ADAPTER: %s from %s",
            func_name,
            original_func.__module__,
        )

        # Get original signature to preserve it
        original_sig = inspect.signature(original_func)

        # Wrap external library functions with ArrayBridge decorator for dtype handling
        arraybridge_wrapped_func = original_func
        if self.MEMORY_TYPE is not None:
            from arraybridge.decorators import _create_dtype_wrapper
            from arraybridge.types import MemoryType as ABMemoryType

            # Map memory type string to ArrayBridge MemoryType enum
            mem_type = ABMemoryType(self.MEMORY_TYPE)
            arraybridge_wrapped_func = _create_dtype_wrapper(
                original_func, mem_type, func_name
            )

        def adapter(image, *args, **kwargs):
            processed_image = self._preprocess_input(image, func_name)
            result = arraybridge_wrapped_func(processed_image, *args, **kwargs)
            return self._postprocess_output(result, image, func_name)

        # Apply wraps and preserve signature
        wrapped_adapter = wraps(original_func)(adapter)
        wrapped_adapter.__signature__ = original_sig

        # Preserve and enhance annotations
        wrapped_adapter.__annotations__ = inspect.get_annotations(
            original_func,
            eval_str=False,
        ).copy()

        # Extract type hints from docstring if annotations are missing
        self._enhance_annotations_from_docstring(wrapped_adapter, original_func)

        # Set memory type attributes for contract execution compatibility
        # Only set if registry has a specific memory type (external libraries)
        if self.MEMORY_TYPE is not None:
            wrapped_adapter.input_memory_type = self.MEMORY_TYPE
            wrapped_adapter.output_memory_type = self.MEMORY_TYPE

        return wrapped_adapter

    def _enhance_annotations_from_docstring(
        self, wrapped_func: Callable, original_func: Callable
    ):
        """Extract type hints from docstring using mathematical simplification approach."""
        try:
            # Import from shared UI utilities (no circular dependency)
            from openhcs.introspection import SignatureAnalyzer
            import numpy as np

            logger.debug(
                f"🔍 ENHANCE ANNOTATIONS: {original_func.__name__} from {original_func.__module__}"
            )

            # Unified type extraction with compatibility validation (mathematical simplification)
            TYPE_PATTERNS = {
                "ndarray": np.ndarray,
                "array": np.ndarray,
                "array_like": np.ndarray,
                "int": int,
                "integer": int,
                "float": float,
                "scalar": float,
                "bool": bool,
                "boolean": bool,
                "str": str,
                "string": str,
                "tuple": tuple,
                "list": list,
                "dict": dict,
                "sequence": list,
            }

            COMPATIBLE_DEFAULTS = {
                float: (int, float, range),
                int: (int, float),
                list: (list, tuple, range),
                tuple: (list, tuple, range),
            }

            param_info = SignatureAnalyzer.analyze(
                original_func, skip_first_param=False
            )

            # Inline type extraction and validation (single-use function inlining rule)
            enhanced_count = 0
            for param_name, info in param_info.items():
                if param_name not in wrapped_func.__annotations__ and info.description:
                    # Extract first line of description (NumPy/SciPy convention: type is always on first line)
                    # This avoids false matches from type keywords appearing later in the description
                    first_line = info.description.split("\n")[0].strip().lower()
                    # Remove optional markers and split on 'or' for union types
                    first_line = (
                        first_line.replace(", optional", "")
                        .replace(" optional", "")
                        .split(" or ")[0]
                        .strip()
                    )

                    # Type extraction with priority patterns
                    python_type = (
                        str
                        if first_line.startswith("{") and "}" in first_line
                        else list
                        if any(
                            p in first_line
                            for p in ["sequence", "iterable", "array of", "list of"]
                        )
                        else next(
                            (
                                t
                                for pattern, t in TYPE_PATTERNS.items()
                                if pattern in first_line
                            ),
                            None,
                        )
                    )

                    # Inline compatibility check (single-use function inlining rule)
                    if python_type and (
                        info.default_value is None
                        or type(info.default_value)
                        in COMPATIBLE_DEFAULTS.get(python_type, (python_type,))
                    ):
                        logger.debug(
                            f"  ✓ Enhanced {param_name}: {python_type} (from first_line='{first_line[:50]}')"
                        )
                        wrapped_func.__annotations__[param_name] = python_type
                        enhanced_count += 1
                    elif info.description:
                        logger.debug(
                            f"  ✗ Could not enhance {param_name}: first_line='{first_line[:50]}', extracted_type={python_type}"
                        )

            if enhanced_count > 0:
                logger.debug(
                    f"  📝 Enhanced {enhanced_count} annotations for {original_func.__name__}"
                )
                logger.debug(f"  Final annotations: {wrapped_func.__annotations__}")
        except Exception as e:
            logger.error(
                f"  ❌ Error enhancing annotations for {original_func.__name__}: {e}",
                exc_info=True,
            )

    @abstractmethod
    def _preprocess_input(self, image, func_name: str):
        """Preprocess input image. Library-specific implementation."""
        pass

    @abstractmethod
    def _postprocess_output(self, result, original_image, func_name: str):
        """Postprocess output result. Library-specific implementation."""
        pass

    # ===== BASIC FILTERING =====
    def should_include_function(self, func: Callable, func_name: str) -> bool:
        """Single method for all filtering logic (blacklist, signature, etc.)"""
        # Skip private functions
        if func_name.startswith("_"):
            return False

        # Skip exclusions (check both common and library-specific)
        if func_name.lower() in self.EXCLUSIONS:
            return False

        # Skip classes and types
        if inspect.isclass(func) or isinstance(func, type):
            return False

        # Must be callable
        if not callable(func):
            return False

        # Pure functions must have at least one parameter
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params:
            return False

        # Validate that type hints can be resolved (skip functions with missing dependencies)
        if not self._validate_type_hints(func, func_name):
            return False

        # Library-specific signature validation
        return self._check_first_parameter(params[0], func_name)

    def _validate_type_hints(self, func: Callable, func_name: str) -> bool:
        """
        Validate that function type hints can be resolved.

        Returns False if type hints reference missing dependencies (e.g., torch when not installed).
        This prevents functions with unresolvable type hints from being registered.
        """
        try:
            # Try to resolve type hints - this will fail if dependencies are missing
            get_type_hints(func)
            return True
        except NameError as e:
            # Type hint references a missing dependency (e.g., 'torch' not defined)
            logger.warning(
                f"Skipping function '{func_name}' due to unresolvable type hints: {e}"
            )
            return False
        except Exception:
            # Other type hint resolution errors - be conservative and allow the function
            # (this handles edge cases where get_type_hints fails for other reasons)
            return True

    @abstractmethod
    def _check_first_parameter(self, first_param, func_name: str) -> bool:
        """Check if first parameter meets library-specific criteria. Library-specific implementation."""
        pass

    # ===== RUNTIME TESTING IMPLEMENTATION =====
    def discover_functions(self) -> Dict[str, FunctionMetadata]:
        """Discover and classify all library functions with runtime testing."""
        functions = {}
        modules = self.get_modules_to_scan()
        logger.info(f"🔍 Starting function discovery for {self.library_name}")
        logger.info(
            f"📦 Scanning {len(modules)} modules: {[name for name, _ in modules]}"
        )

        total_tested = 0
        total_accepted = 0

        for module_name, module in modules:
            logger.info(f"  📦 Analyzing {module_name} ({module})...")
            module_tested = 0
            module_accepted = 0

            for name in dir(module):
                if name.startswith("_"):
                    continue

                func = module.__dict__[name]
                full_path = self._get_full_function_path(module, name, module_name)

                if not self.should_include_function(func, name):
                    rejection_reason = self._get_rejection_reason(func, name)
                    if rejection_reason != "private":
                        logger.debug(f"    🚫 Skipping {full_path}: {rejection_reason}")
                    continue

                module_tested += 1
                total_tested += 1

                contract, is_valid = self.classify_function_behavior(func)
                logger.debug(f"    🧪 Testing {full_path}")
                logger.debug(
                    f"       Classification: {contract.name if contract else contract}"
                )

                if not is_valid:
                    logger.debug("       ❌ Rejected: Invalid classification")
                    continue

                doc = inspect.getdoc(func)
                doc_lines = doc.splitlines() if doc is not None else ()
                first_line_doc = doc_lines[0] if doc_lines else ""
                module_path = func.__module__
                if module_path is None:
                    module_path = ""
                func_name = self._generate_function_name(name, module_name)

                # Apply library adapter (preprocessing/postprocessing)
                adapted_func = self.create_library_adapter(func, contract)

                # Apply nominal contract wrapper.
                final_func = self.apply_contract_wrapper(adapted_func, contract)

                metadata = FunctionMetadata(
                    name=func_name,
                    func=final_func,
                    contract=contract,
                    registry=self,
                    module=module_path,
                    doc=first_line_doc,
                    tags=self._generate_tags(name),
                    original_name=name,
                    memory_type=self.get_memory_type(),
                )

                functions[func_name] = metadata
                module_accepted += 1
                total_accepted += 1
                logger.debug(f"       ✅ Accepted as '{func_name}'")

            logger.debug(
                f"  📊 Module {module_name}: {module_accepted}/{module_tested} functions accepted"
            )

        logger.info(
            f"✅ Discovery complete: {total_accepted}/{total_tested} functions accepted"
        )
        return functions

    def _get_full_function_path(self, module, func_name: str, module_name: str) -> str:
        """Generate full module path for logging."""
        if module_name == "main":
            return f"{self.library_name}.{func_name}"
        else:
            # Extract clean module path
            module_str = str(module)
            if "'" in module_str:
                clean_path = module_str.split("'")[1]
                return f"{clean_path}.{func_name}"
            else:
                return f"{module_name}.{func_name}"

    def _get_rejection_reason(self, func: Callable, func_name: str) -> str:
        """Get detailed reason why a function was rejected."""
        # Check each rejection criteria in order
        if func_name.startswith("_"):
            return "private"

        if func_name.lower() in self.EXCLUSIONS:
            return "blacklisted"

        if inspect.isclass(func) or isinstance(func, type):
            return "is class/type"

        if not callable(func):
            return "not callable"

        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params:
            return "no parameters (not pure function)"

        return "unknown"

    # ===== CUSTOMIZATION HOOKS =====
    def _generate_function_name(self, name: str, module_name: str) -> str:
        """Generate function name. Override in subclasses for custom naming."""
        return name

    def _generate_tags(self, func_name: str) -> List[str]:
        """Generate tags using library name."""
        return [self.library_name]


# ============================================================================
# Registry Export
# ============================================================================
# Auto-created registry from LibraryRegistryBase
LIBRARY_REGISTRIES = LibraryRegistryBase.__registry__
