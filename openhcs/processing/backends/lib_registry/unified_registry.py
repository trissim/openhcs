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
from dataclasses import dataclass, field, is_dataclass, replace
from enum import Enum
from functools import wraps
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Tuple, Type, get_type_hints


import numpy as np
from openhcs.core.xdg_paths import get_cache_file_path
from openhcs.core.memory import unstack_slices, stack_slices
from openhcs.core.runtime_semantics import ObjectLabelDomainScope
from openhcs.core.runtime_values import (
    ImageMetadataPayload,
    MaskedImagePayload,
    NativeRuntimeValue,
    ObjectLabelPayload,
    compose_image_payload_metadata,
    image_payload_data,
    image_payload_mask,
    image_payload_with_context,
)
from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from metaclass_registry import AutoRegisterMeta, LazyDiscoveryDict
from openhcs.core.config import LazyDtypeConfig , LazyWellFilterConfig, LazyStepWellFilterConfig

logger = logging.getLogger(__name__)


def _pure_2d_slice_results(
    results: Iterable[Any],
) -> tuple[list[Any], tuple[list[Any], ...]]:
    """Split per-slice PURE_2D results into main outputs and auxiliary groups."""
    collected = list(results)
    if not collected:
        raise ValueError("PURE_2D execution cannot aggregate zero slice results.")

    first_result = collected[0]
    if not isinstance(first_result, tuple):
        return collected, ()

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

    return main_outputs, tuple(auxiliary_groups)


def _aggregate_pure_2d_auxiliary_output(
    values: list[Any],
    memory_type: str,
) -> Any:
    """Aggregate one auxiliary PURE_2D output across slices."""
    return Pure2DAuxiliaryOutputAggregator.aggregate(values, memory_type)


class Pure2DAuxiliaryOutputAggregator(ABC, metaclass=AutoRegisterMeta):
    """Aggregate one auxiliary PURE_2D output position across slices."""

    __registry_key__ = "value_type"
    __registry__: ClassVar[dict[Any, type["Pure2DAuxiliaryOutputAggregator"]]] = {}
    value_type: ClassVar[type[Any] | None] = None
    can_aggregate_family: ClassVar[bool] = True

    @classmethod
    def aggregate(cls, values: list[Any], memory_type: str) -> Any:
        if not values:
            return []
        candidates: list[Pure2DAuxiliaryOutputAggregator] = []
        for aggregator_type in cls.registered_aggregator_families():
            aggregator = aggregator_type()
            if aggregator.supports(values):
                candidates.append(aggregator)
        if candidates:
            aggregator = min(
                candidates,
                key=lambda candidate: candidate.type_distance(values),
            )
            return aggregator.aggregate_values(values, memory_type)
        if len(values) == 1:
            return values[0]
        return list(values)

    @classmethod
    def registered_aggregator_families(
        cls,
    ) -> tuple[type["Pure2DAuxiliaryOutputAggregator"], ...]:
        """Return registered aggregator classes plus their typed family bases."""
        family_types: list[type[Pure2DAuxiliaryOutputAggregator]] = []
        for aggregator_type in cls.__registry__.values():
            for candidate_type in aggregator_type.mro():
                if (
                    candidate_type is cls
                    or not isinstance(candidate_type, type)
                    or not issubclass(candidate_type, cls)
                    or not candidate_type.can_aggregate_family
                    or candidate_type in family_types
                ):
                    continue
                family_types.append(candidate_type)
        return tuple(family_types)

    def supports(self, values: list[Any]) -> bool:
        accepted_types = self.accepted_value_types()
        return bool(accepted_types) and all(
            isinstance(value, accepted_types) for value in values
        )

    def owns_mixed_values(self, values: list[Any]) -> bool:
        return False

    @classmethod
    def accepted_value_types(cls) -> tuple[type[Any], ...]:
        """Return nominal value types owned by this registered aggregator family."""
        return tuple(
            aggregator_type.value_type
            for aggregator_type in Pure2DAuxiliaryOutputAggregator.__registry__.values()
            if (
                aggregator_type.value_type is not None
                and issubclass(aggregator_type, cls)
            )
        )

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
    def aggregate_values(self, values: list[Any], memory_type: str) -> Any:
        """Aggregate compatible per-slice auxiliary values."""


class RuntimeValuePure2DAuxiliaryOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Preserve non-array runtime values as slice-aligned payloads."""

    value_type = NativeRuntimeValue

    def aggregate_values(self, values: list[Any], memory_type: str) -> Any:
        del memory_type
        return RuntimeSliceAlignedValues(slices=tuple(values))


class RuntimeArrayPure2DAuxiliaryOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Stack nominal runtime array payloads through their concrete array data."""

    value_type = None
    can_aggregate_family = False

    def supports(self, values: list[Any]) -> bool:
        return super().supports(values) or self.owns_mixed_values(values)

    def owns_mixed_values(self, values: list[Any]) -> bool:
        accepted_types = self.accepted_value_types()
        return bool(accepted_types) and any(
            isinstance(value, accepted_types) for value in values
        ) and all(self._accepts_mixed_value(value) for value in values)

    def type_distance(self, values: list[Any]) -> int:
        if self.owns_mixed_values(values):
            return 0
        return super().type_distance(values)

    def _accepts_mixed_value(self, value: Any) -> bool:
        return isinstance(value, np.ndarray)

    def aggregate_values(self, values: list[Any], memory_type: str) -> Any:
        del values, memory_type
        raise NotImplementedError(
            "Concrete runtime-array aggregators must implement aggregate_values."
        )


class ImagePayloadPure2DAuxiliaryOutputAggregator(RuntimeArrayPure2DAuxiliaryOutputAggregator):
    """Stack image payload slices and reattach composed runtime image context."""

    value_type = None
    can_aggregate_family = True

    def _accepts_mixed_value(self, value: Any) -> bool:
        accepted_types = self.accepted_value_types()
        return (
            bool(accepted_types)
            and isinstance(value, accepted_types)
        ) or super()._accepts_mixed_value(value)

    def aggregate_values(self, values: list[Any], memory_type: str) -> Any:
        data = stack_slices([image_payload_data(value) for value in values], memory_type, 0)
        masks = [image_payload_mask(value) for value in values]
        present_masks = [mask for mask in masks if mask is not None]
        if present_masks and len(present_masks) != len(masks):
            raise ValueError("Cannot aggregate a mix of masked and unmasked image payloads.")
        mask = (
            None
            if not present_masks
            else present_masks[0]
            if len(present_masks) == 1
            else stack_slices(present_masks, memory_type, 0)
        )
        return image_payload_with_context(
            data,
            mask=mask,
            metadata=compose_image_payload_metadata(tuple(values)),
        )


class MaskedImagePayloadPure2DAuxiliaryOutputAggregator(ImagePayloadPure2DAuxiliaryOutputAggregator):
    """Register masked image payloads for PURE_2D auxiliary aggregation."""

    value_type = MaskedImagePayload


class ImageMetadataPayloadPure2DAuxiliaryOutputAggregator(ImagePayloadPure2DAuxiliaryOutputAggregator):
    """Register image metadata payloads for PURE_2D auxiliary aggregation."""

    value_type = ImageMetadataPayload


class ObjectLabelPayloadPure2DAuxiliaryOutputAggregator(RuntimeArrayPure2DAuxiliaryOutputAggregator):
    """Stack object-label payload slices and preserve object-domain metadata."""

    value_type = ObjectLabelPayload
    can_aggregate_family = True

    def _accepts_mixed_value(self, value: Any) -> bool:
        return isinstance(value, self.value_type)

    def aggregate_values(self, values: list[Any], memory_type: str) -> ObjectLabelPayload:
        return _aggregate_object_label_payload_slices(values, memory_type)


def _aggregate_object_label_payload_slices(
    values: list[ObjectLabelPayload],
    memory_type: str,
) -> ObjectLabelPayload:
    labels = _stack_runtime_array_slices(
        [value.labels for value in values],
        memory_type,
    )
    unedited_labels = (
        _stack_runtime_array_slices(
            [value.labels_for_variant("unedited") for value in values],
            memory_type,
        )
        if any(value.unedited_labels is not None for value in values)
        else None
    )
    small_removed_labels = (
        _stack_runtime_array_slices(
            [value.labels_for_variant("small_removed") for value in values],
            memory_type,
        )
        if any(value.small_removed_labels is not None for value in values)
        else None
    )
    declared_counts = {value.declared_object_count for value in values}
    declared_ids = {value.declared_object_ids for value in values}
    domain_scopes = {value.domain_scope for value in values}
    return ObjectLabelPayload(
        labels=labels,
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
        declared_object_count=(
            declared_counts.pop() if len(declared_counts) == 1 else None
        ),
        declared_object_ids=declared_ids.pop() if len(declared_ids) == 1 else (),
        domain_scope=ObjectLabelDomainScope.common(domain_scopes),
    )


class RegisteredImageLayoutPure2DAuxiliaryOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Stack arrays that are accepted by the registered OpenHCS image layout."""

    value_type = np.ndarray

    def aggregate_values(self, values: list[Any], memory_type: str) -> Any:
        stack_payload = _stack_registered_image_layout_slices(values, memory_type)
        if stack_payload is not None:
            return stack_payload
        return stack_slices(values, memory_type, 0)


class FlatSequencePure2DAuxiliaryOutputAggregator(Pure2DAuxiliaryOutputAggregator):
    """Concatenate flat tuple/list auxiliary outputs."""

    value_type = None

    def supports(self, values: list[Any]) -> bool:
        return super().supports(values) and not any(
            isinstance(item, np.ndarray)
            for value in values
            for item in value
        )

    def aggregate_values(self, values: list[Any], memory_type: str) -> Any:
        del memory_type
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


def _stack_runtime_array_slices(values: list[Any], memory_type: str) -> Any:
    stack_payload = _stack_registered_image_layout_slices(values, memory_type)
    if stack_payload is not None:
        return stack_payload
    return np.stack(tuple(np.asarray(value) for value in values), axis=0)


def _stack_registered_image_layout_slices(
    values: list[Any],
    memory_type: str,
) -> Any | None:
    try:
        from openhcs.core.image_stack_layout import ImageStackLayout

        return ImageStackLayout.for_slices(values).stack(
            slices=values,
            memory_type=memory_type,
            gpu_id=0,
        )
    except ValueError:
        return None


def _rewrite_slice_index(value: Any, slice_index: int) -> Any:
    """Project the real slice index into nested slice-local outputs."""
    if hasattr(value, "ndim"):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "slice_index"):
            return replace(value, slice_index=slice_index)
        return value
    if isinstance(value, dict):
        if "slice_index" in value:
            return {**value, "slice_index": slice_index}
        if _is_flat_mapping_row(value):
            return {**value, "slice_index": slice_index}
        return {
            key: _rewrite_slice_index(item, slice_index)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if _is_slice_index_free_flat_sequence(value):
            return value
        rewritten = None
        for index, item in enumerate(value):
            rewritten_item = _rewrite_slice_index(item, slice_index)
            if rewritten is not None:
                rewritten.append(rewritten_item)
            elif rewritten_item is not item:
                rewritten = [*value[:index], rewritten_item]
        return value if rewritten is None else rewritten
    if isinstance(value, tuple):
        if _is_slice_index_free_flat_sequence(value):
            return value
        rewritten_items = tuple(_rewrite_slice_index(item, slice_index) for item in value)
        if all(
            rewritten_item is item
            for rewritten_item, item in zip(rewritten_items, value, strict=True)
        ):
            return value
        return rewritten_items
    return value


def _is_slice_index_free_flat_sequence(value: list[Any] | tuple[Any, ...]) -> bool:
    """Return whether a sequence is a flat measurement-row collection."""
    if not value:
        return True
    first = value[0]
    if is_dataclass(first) and not isinstance(first, type):
        row_type = type(first)
        return (
            not hasattr(first, "slice_index")
            and all(type(item) is row_type for item in value)
        )
    return False


def _is_flat_mapping_row(value: Mapping[Any, Any]) -> bool:
    return all(not _value_may_contain_slice_index(item) for item in value.values())


def _value_may_contain_slice_index(value: Any) -> bool:
    return (
        hasattr(value, "ndim")
        or isinstance(value, (dict, list, tuple))
        or (is_dataclass(value) and not isinstance(value, type))
    )


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


class ProcessingContract(Enum):
    """
    Unified contract classification with direct method execution.
    """
    PURE_3D = "_execute_pure_3d"
    PURE_2D = "_execute_pure_2d"
    FLEXIBLE = "_execute_flexible"
    VOLUMETRIC_TO_SLICE = "_execute_volumetric_to_slice"

    @classmethod
    def from_declared_name(cls, contract_name: str) -> "ProcessingContract | None":
        """Resolve a declared contract name to the canonical enum member."""
        normalized = contract_name.upper()
        if normalized not in cls.__members__:
            return None
        return cls[normalized]

    def execute(self, registry, func, image, *args, **kwargs):
        """Execute the contract method on the registry."""
        method = getattr(registry, self.value)
        return method(func, image, *args, **kwargs)


@dataclass(frozen=True)
class FunctionMetadata:
    """Clean metadata with no library-specific leakage."""

    # Core fields only
    name: str
    func: Callable
    contract: ProcessingContract
    registry: 'LibraryRegistryBase'  # Reference to the registry that registered this function - REQUIRED
    module: str = ""
    doc: str = ""
    tags: List[str] = field(default_factory=list)
    original_name: str = ""  # Original function name for cache reconstruction

    def get_memory_type(self) -> str:
        """
        Get the actual memory type (backend) of this function.

        Returns the function's input_memory_type if available, otherwise falls back
        to the registry's memory type. This ensures UI shows the actual backend
        (cupy, numpy, etc.) instead of the registry name (openhcs).

        Returns:
            Memory type string (e.g., "cupy", "numpy", "torch", "pyclesperanto")
        """
        # First try to get memory type from function attributes
        if hasattr(self.func, 'input_memory_type'):
            return self.func.input_memory_type
        elif hasattr(self.func, 'output_memory_type'):
            return self.func.output_memory_type
        elif hasattr(self.func, 'backend'):
            return self.func.backend

        # Fallback to registry memory type
        return self.registry.get_memory_type()

    def get_registry_name(self) -> str:
        """
        Get the registry name that registered this function.

        Returns:
            Registry name string (e.g., "openhcs", "skimage", "cupy", "pyclesperanto")
        """
        return self.registry.library_name

    def get(self, key: str, default=None):
        """
        Dict-like get method for compatibility with code expecting dict-like access.

        Args:
            key: Attribute name to retrieve
            default: Default value if attribute doesn't exist

        Returns:
            Attribute value or default
        """
        return getattr(self, key, default)




class LibraryRegistryBase(ABC, metaclass=AutoRegisterMeta):
    """
    Minimal ABC for all library registries.

    Provides only essential contracts that all registries must implement,
    regardless of whether they use runtime testing or explicit contracts.

    Registry auto-created and stored as LibraryRegistryBase.__registry__.
    Subclasses auto-register by setting _registry_name class attribute.
    """
    __registry_key__ = '_registry_name'

    _registry_name: Optional[str] = None  # Override in subclasses (e.g., 'pyclesperanto', 'cupy')

    # Common exclusions across all libraries
    COMMON_EXCLUSIONS = {
        'imread', 'imsave', 'load', 'save', 'read', 'write',
        'show', 'imshow', 'plot', 'display', 'view', 'visualize',
        'info', 'help', 'version', 'test', 'benchmark'
    }

    # Abstract class attributes - each implementation must define these
    MODULES_TO_SCAN: List[str]
    MEMORY_TYPE: str  # Memory type string value (e.g., "pyclesperanto", "cupy", "numpy")
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

    # ===== DECLARATIVE INJECTABLE PARAMETERS =====
    # Parameters injected into all registered functions
    # Format: (param_name, default_value, type_annotation)
    # Use lambda for lazy instantiation to avoid circular imports
    INJECTABLE_PARAMS = [
        ('enabled', True, bool),
        ('dtype_config', lambda: LazyDtypeConfig(), 'LazyDtypeConfig'),
#        ('well_filter_config', lambda: LazyWellFilterConfig(), 'LazyWellFilterConfig'),
#        ('step_well_filter_config', lambda: LazyStepWellFilterConfig(), 'LazyStepWellFilterConfig'),
    ]

    # Parameters injected only into FLEXIBLE contract functions
    FLEXIBLE_ONLY_PARAMS = [
        ('slice_by_slice', False, bool),
    ]

    # ===== CONTRACT HANDLING =====
    def apply_contract_wrapper(self, func: Callable, contract: ProcessingContract) -> Callable:
        """Apply contract wrapper with parameter injection using declarative INJECTABLE_PARAMS."""
        from functools import wraps
        import inspect

        original_sig = inspect.signature(func)
        param_names = {p.name for p in original_sig.parameters.values()}

        # Import type annotations (resolve lazy strings)
        from openhcs.core.config import LazyDtypeConfig
        type_map = {'LazyDtypeConfig': LazyDtypeConfig}

        # Build injectable params list from declarative constants
        # Resolve lambda defaults by calling them
        injectable_params = []
        for name, default, annotation in self.INJECTABLE_PARAMS:
            resolved_default = default() if callable(default) else default
            resolved_annotation = type_map.get(annotation, annotation) if isinstance(annotation, str) else annotation
            injectable_params.append((name, resolved_default, resolved_annotation))

        if contract == ProcessingContract.FLEXIBLE:
            for name, default, annotation in self.FLEXIBLE_ONLY_PARAMS:
                resolved_default = default() if callable(default) else default
                resolved_annotation = type_map.get(annotation, annotation) if isinstance(annotation, str) else annotation
                injectable_params.append((name, resolved_default, resolved_annotation))

        # Filter out already-existing parameters
        params_to_add = [(name, default, annotation) for name, default, annotation in injectable_params if name not in param_names]

        # If nothing to inject, return original function
        if not params_to_add:
            # Still brand the callable as Enableable metadata.
            from python_introspect import mark_enableable
            mark_enableable(func, enabled_default=True)
            return func

        # Build new parameter list (insert before **kwargs)
        new_params = list(original_sig.parameters.values())
        insert_index = next((i for i, p in enumerate(new_params) if p.kind == inspect.Parameter.VAR_KEYWORD), len(new_params))

        for param_name, default_value, annotation in params_to_add:
            new_params.insert(insert_index, inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, default=default_value, annotation=annotation))
            insert_index += 1

        # Create wrapper
        @wraps(func)
        def wrapper(image, *args, **kwargs):
            # Extract injectable params and set them as attributes on func
            for param_name, _, _ in injectable_params:
                if param_name in kwargs:
                    setattr(func, param_name, kwargs[param_name])

            # Populate missing injectable params with their defaults from the signature
            # This is critical for internal calls between OpenHCS functions where
            # injectable params may not be explicitly passed (e.g., create_projection calling max_projection)
            from python_introspect import SignatureAnalyzer
            sig_params = SignatureAnalyzer.analyze(wrapper)
            for param_name, _, _ in injectable_params:
                if param_name not in kwargs and param_name in sig_params:
                    default_value = sig_params[param_name].default_value
                    if default_value is not inspect.Parameter.empty:
                        kwargs[param_name] = default_value

            # Filter injectable params from kwargs, EXCEPT dtype_config which needs to
            # flow through to ArrayBridge's dtype_wrapper for conversion logic
            params_to_filter = {name for name, _, _ in injectable_params if name != 'dtype_config'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in params_to_filter}

            return contract.execute(self, func, image, *args, **filtered_kwargs)

        # Set defaults and signature
        for param_name, default_value, _ in injectable_params:
            setattr(wrapper, param_name, default_value)

        wrapper.__signature__ = original_sig.replace(parameters=new_params)
        wrapper.__annotations__ = getattr(func, '__annotations__', {}).copy()
        for param_name, _, annotation in injectable_params:
            wrapper.__annotations__[param_name] = annotation

        # Explicitly copy __processing_contract__ if it exists
        if hasattr(func, '__processing_contract__'):
            wrapper.__processing_contract__ = func.__processing_contract__
        from openhcs.core.callable_contract import attach_callable_contract_metadata
        attach_callable_contract_metadata(wrapper, raw_processing_function=func)

        # Nominal enable semantics: decorated callables are Enableable.
        # (Enableable is metadata only; enabled remains a kw-only param for call sites.)
        from python_introspect import mark_enableable
        mark_enableable(wrapper, enabled_default=True)

        return wrapper

    def _inject_optional_dataclass_params(self, func: Callable) -> Callable:
        """Inject optional lazy dataclass parameters into function signature.

        Can be disabled by setting ENABLE_CONFIG_INJECTION = False.
        """
        # Configuration flag to enable/disable config injection
        ENABLE_CONFIG_INJECTION = False  # Set to True to re-enable config injection

        if not ENABLE_CONFIG_INJECTION:
            return func  # Return function unchanged when disabled

        # Original injection logic (commented out for now but preserved)
        import inspect
        from functools import wraps
        from typing import Optional

        # Get original signature
        original_sig = inspect.signature(func)
        original_params = list(original_sig.parameters.values())

        # Import existing lazy config types
        from openhcs.core.config import LazyNapariStreamingConfig, LazyFijiStreamingConfig, LazyStepMaterializationConfig

        # Define common lazy dataclass parameters to inject
        dataclass_params = [
            ('napari_streaming_config', 'Optional[LazyNapariStreamingConfig]', LazyNapariStreamingConfig),
            ('fiji_streaming_config', 'Optional[LazyFijiStreamingConfig]', LazyFijiStreamingConfig),
            ('step_materialization_config', 'Optional[LazyStepMaterializationConfig]', LazyStepMaterializationConfig),
        ]

        # Check if any parameters need to be added
        existing_param_names = {p.name for p in original_params}
        params_to_add = [(name, type_hint, lazy_class) for name, type_hint, lazy_class in dataclass_params
                        if name not in existing_param_names]

        if not params_to_add:
            return func  # No parameters to add

        # Create new parameters
        new_params = original_params.copy()

        # Find insertion point (before **kwargs if it exists)
        insert_index = len(new_params)
        for i, param in enumerate(new_params):
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                insert_index = i
                break

        # Add dataclass parameters
        for param_name, type_hint, lazy_class in params_to_add:
            new_param = inspect.Parameter(
                param_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[lazy_class]  # Use actual type object, not string
            )
            new_params.insert(insert_index, new_param)
            insert_index += 1

        # Create enhanced wrapper function
        @wraps(func)
        def enhanced_wrapper(*args, **kwargs):
            # Extract dataclass parameters from kwargs (they're just ignored for now)
            regular_kwargs = {k: v for k, v in kwargs.items()
                            if k not in [name for name, _, _ in dataclass_params]}

            # Call original function with regular parameters only
            return func(*args, **regular_kwargs)

        # Apply the modified signature
        new_sig = original_sig.replace(parameters=new_params)
        enhanced_wrapper.__signature__ = new_sig

        # Enhance annotations
        if hasattr(func, '__annotations__'):
            enhanced_wrapper.__annotations__ = func.__annotations__.copy()
        else:
            enhanced_wrapper.__annotations__ = {}

        # Add type annotations for injected parameters
        from typing import Optional
        for param_name, type_hint, lazy_class in params_to_add:
            enhanced_wrapper.__annotations__[param_name] = Optional[lazy_class]

        return enhanced_wrapper

    def _get_function_by_name(self, module_path: str, func_name: str):
        """Get function object by module path and name."""
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    def create_library_adapter(
        self,
        original_func: Callable,
        contract: ProcessingContract,
    ) -> Callable:
        """Return the callable shape used before contract wrapping."""
        return original_func

    # ===== PROCESSING CONTRACT EXECUTION METHODS =====
    def _execute_slice_by_slice(self, func, image, *args, **kwargs):
        """Shared slice-by-slice execution logic."""
        if image.ndim == 3:
            from openhcs.core.memory import unstack_slices, stack_slices
            from openhcs.core.memory import detect_memory_type
            mem = detect_memory_type(image)
            slices = unstack_slices(image, mem, 0)
            results = [func(sl, *args, **kwargs) for sl in slices]
            return stack_slices(results, mem, 0)
        return func(image, *args, **kwargs)

    def _execute_pure_3d(self, func, image, *args, **kwargs):
        """Execute 3D→3D function directly (no change)."""
        return func(image, *args, **kwargs)

    def _execute_pure_2d(self, func, image, *args, **kwargs):
        """Execute 2D→2D function with unstack/restack wrapper."""
        # Get memory type from the decorated function
        memory_type = func.output_memory_type
        slices = unstack_slices(image, memory_type, 0)
        slice_results = [
            _rewrite_slice_index(func(slice_2d, *args, **kwargs), slice_index)
            for slice_index, slice_2d in enumerate(slices)
        ]
        main_outputs, auxiliary_groups = _pure_2d_slice_results(slice_results)
        stacked_main_output = stack_slices(main_outputs, memory_type, 0)
        if not auxiliary_groups:
            return stacked_main_output
        aggregated_auxiliary_outputs = tuple(
            _aggregate_pure_2d_auxiliary_output(values, memory_type)
            for values in auxiliary_groups
        )
        return (stacked_main_output, *aggregated_auxiliary_outputs)

    def _execute_flexible(self, func, image, *args, **kwargs):
        """Execute function that handles both 3D→3D and 2D→2D with toggle."""
        # Check if slice_by_slice attribute is set on the function
        slice_by_slice = getattr(func, 'slice_by_slice', False)
        if slice_by_slice:
            # Reuse the 2D-only execution logic (unstack -> process -> restack)
            return self._execute_pure_2d(func, image, *args, **kwargs)
        else:
            # Use 3D-only execution logic (no modification)
            return self._execute_pure_3d(func, image, *args, **kwargs)

    def _execute_volumetric_to_slice(self, func, image, *args, **kwargs):
        """Execute 3D→2D function returning slice 3D array."""
        # Get memory type from the decorated function
        memory_type = func.output_memory_type
        result_2d = func(image, *args, **kwargs)
        return stack_slices([result_2d], memory_type, 0)

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
            logger.info(f"✅ Loaded {len(cached_functions)} {self.library_name} functions from cache")
            self._function_metadata_cache = cached_functions
            self._function_metadata_cache_modules = module_signature
            return cached_functions

        logger.info(f"🔍 Cache miss for {self.library_name} - performing full discovery")
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
            logger.debug(f"📂 LOAD FROM CACHE: No cache file exists at {self._cache_path}")
            return None

        try:
            with open(self._cache_path, 'r') as f:
                cache_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Corrupt cache file {self._cache_path}, rebuilding")
            self._cache_path.unlink(missing_ok=True)
            return None

        if 'functions' not in cache_data:
            return None

        cached_version = cache_data.get('library_version', 'unknown')
        current_version = self.get_library_version()
        if cached_version != current_version:
            logger.info(f"{self.library_name} version changed ({cached_version} → {current_version}) - cache invalid")
            return None

        cached_signature = cache_data.get('discovery_signature')
        current_signature = self.get_discovery_signature()
        if cached_signature != current_signature:
            logger.info(f"{self.library_name} discovery inputs changed - cache invalid")
            return None

        cache_timestamp = cache_data.get('timestamp', 0)
        cache_age_days = (time.time() - cache_timestamp) / (24 * 3600)
        if cache_age_days > 7:
            logger.debug(f"Cache is {cache_age_days:.1f} days old - rebuilding")
            return None

        logger.debug(f"📂 LOAD FROM CACHE: Loading {len(cache_data['functions'])} functions for {self.library_name}")

        functions = {}
        for func_name, cached_data in cache_data['functions'].items():
            original_name = cached_data.get('original_name', func_name)
            try:
                func = self._get_function_by_name(
                    cached_data['module'],
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
                self._cache_path.unlink(missing_ok=True)
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
                self._cache_path.unlink(missing_ok=True)
                return None
            contract = ProcessingContract[cached_data['contract']]

            adapted_func = self.create_library_adapter(func, contract)
            contract_wrapped_func = self.apply_contract_wrapper(adapted_func, contract)
            final_func = self._inject_optional_dataclass_params(contract_wrapped_func)

            metadata = FunctionMetadata(
                name=func_name,
                func=final_func,
                contract=contract,
                registry=self,
                module=cached_data.get('module', ''),
                doc=cached_data.get('doc', ''),
                tags=cached_data.get('tags', []),
                original_name=cached_data.get('original_name', func_name)
            )
            functions[func_name] = metadata

        return functions

    def get_discovery_signature(self) -> str:
        """Return the existing JSON cache's discovery-input signature."""
        signature = {
            'registry_class': f"{type(self).__module__}.{type(self).__qualname__}",
            'modules_to_scan': list(self.MODULES_TO_SCAN),
            'source_mtimes': self.cache_source_mtimes(),
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
            'cache_version': '1.0',
            'library_version': self.get_library_version(),
            'discovery_signature': self.get_discovery_signature(),
            'timestamp': time.time(),
            'functions': {
                func_name: {
                    'name': metadata.name,
                    'original_name': metadata.original_name,
                    'module': metadata.module,
                    'contract': metadata.contract.name,
                    'doc': metadata.doc,
                    'tags': metadata.tags
                }
                for func_name, metadata in functions.items()
            }
        }

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"💾 Saved {len(functions)} {self.library_name} functions to cache")

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
                module = getattr(library, module_name)
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
    def classify_function_behavior(self, func: Callable, declared_contract: Optional[ProcessingContract] = None) -> Tuple[ProcessingContract, bool]:
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
            except:
                return False, None

        works_3d, result_3d = test_function(test_3d)
        works_2d, _ = test_function(test_2d)

        # Classification lookup table
        classification_map = {
            (True, True): self._classify_dual_support(result_3d),
            (True, False): ProcessingContract.PURE_3D,
            (False, True): ProcessingContract.PURE_2D,
            (False, False): None  # Invalid function
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
                if hasattr(first_result, 'ndim') and first_result.ndim == 2:
                    return ProcessingContract.VOLUMETRIC_TO_SLICE
            # Handle single array results
            elif hasattr(result_3d, 'ndim') and result_3d.ndim == 2:
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

    def create_library_adapter(self, original_func: Callable, contract: ProcessingContract) -> Callable:
        """Create adapter with library-specific processing only."""
        import inspect
        func_name = getattr(original_func, '__name__', 'unknown')

        logger.debug(f"🔧 CREATE LIBRARY ADAPTER: {func_name} from {getattr(original_func, '__module__', 'unknown')}")

        # Get original signature to preserve it
        original_sig = inspect.signature(original_func)

        # Wrap external library functions with ArrayBridge decorator for dtype handling
        arraybridge_wrapped_func = original_func
        if self.MEMORY_TYPE is not None:
            from arraybridge.decorators import _create_dtype_wrapper
            from arraybridge.types import MemoryType as ABMemoryType
            # Map memory type string to ArrayBridge MemoryType enum
            mem_type = ABMemoryType(self.MEMORY_TYPE)
            arraybridge_wrapped_func = _create_dtype_wrapper(original_func, mem_type, func_name)

        def adapter(image, *args, **kwargs):
            processed_image = self._preprocess_input(image, func_name)
            result = arraybridge_wrapped_func(processed_image, *args, **kwargs)
            return self._postprocess_output(result, image, func_name)

        # Apply wraps and preserve signature
        wrapped_adapter = wraps(original_func)(adapter)
        wrapped_adapter.__signature__ = original_sig

        # Preserve and enhance annotations
        if hasattr(original_func, '__annotations__'):
            wrapped_adapter.__annotations__ = original_func.__annotations__.copy()
        else:
            wrapped_adapter.__annotations__ = {}

        # Extract type hints from docstring if annotations are missing
        self._enhance_annotations_from_docstring(wrapped_adapter, original_func)

        # Set memory type attributes for contract execution compatibility
        # Only set if registry has a specific memory type (external libraries)
        if self.MEMORY_TYPE is not None:
            wrapped_adapter.input_memory_type = self.MEMORY_TYPE
            wrapped_adapter.output_memory_type = self.MEMORY_TYPE

        return wrapped_adapter

    def _enhance_annotations_from_docstring(self, wrapped_func: Callable, original_func: Callable):
        """Extract type hints from docstring using mathematical simplification approach."""
        try:
            # Import from shared UI utilities (no circular dependency)
            from openhcs.introspection import SignatureAnalyzer
            import numpy as np

            logger.debug(f"🔍 ENHANCE ANNOTATIONS: {original_func.__name__} from {original_func.__module__}")

            # Unified type extraction with compatibility validation (mathematical simplification)
            TYPE_PATTERNS = {'ndarray': np.ndarray, 'array': np.ndarray, 'array_like': np.ndarray,
                           'int': int, 'integer': int, 'float': float, 'scalar': float,
                           'bool': bool, 'boolean': bool, 'str': str, 'string': str,
                           'tuple': tuple, 'list': list, 'dict': dict, 'sequence': list}

            COMPATIBLE_DEFAULTS = {float: (int, float, range), int: (int, float),
                                 list: (list, tuple, range), tuple: (list, tuple, range)}

            param_info = SignatureAnalyzer.analyze(original_func, skip_first_param=False)

            # Inline type extraction and validation (single-use function inlining rule)
            enhanced_count = 0
            for param_name, info in param_info.items():
                if param_name not in wrapped_func.__annotations__ and info.description:
                    # Extract first line of description (NumPy/SciPy convention: type is always on first line)
                    # This avoids false matches from type keywords appearing later in the description
                    first_line = info.description.split('\n')[0].strip().lower()
                    # Remove optional markers and split on 'or' for union types
                    first_line = first_line.replace(', optional', '').replace(' optional', '').split(' or ')[0].strip()

                    # Type extraction with priority patterns
                    python_type = (str if first_line.startswith('{') and '}' in first_line
                                 else list if any(p in first_line for p in ['sequence', 'iterable', 'array of', 'list of'])
                                 else next((t for pattern, t in TYPE_PATTERNS.items() if pattern in first_line), None))

                    # Inline compatibility check (single-use function inlining rule)
                    if python_type and (info.default_value is None or
                                      type(info.default_value) in COMPATIBLE_DEFAULTS.get(python_type, (python_type,))):
                        logger.debug(f"  ✓ Enhanced {param_name}: {python_type} (from first_line='{first_line[:50]}')")
                        wrapped_func.__annotations__[param_name] = python_type
                        enhanced_count += 1
                    elif info.description:
                        logger.debug(f"  ✗ Could not enhance {param_name}: first_line='{first_line[:50]}', extracted_type={python_type}")

            if enhanced_count > 0:
                logger.debug(f"  📝 Enhanced {enhanced_count} annotations for {original_func.__name__}")
                logger.debug(f"  Final annotations: {wrapped_func.__annotations__}")
        except Exception as e:
            logger.error(f"  ❌ Error enhancing annotations for {original_func.__name__}: {e}", exc_info=True)

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
        if func_name.startswith('_'):
            return False

        # Skip exclusions (check both common and library-specific)
        exclusions = getattr(self.__class__, 'EXCLUSIONS', self.COMMON_EXCLUSIONS)
        if func_name.lower() in exclusions:
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
            logger.warning(f"Skipping function '{func_name}' due to unresolvable type hints: {e}")
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
        logger.info(f"📦 Scanning {len(modules)} modules: {[name for name, _ in modules]}")

        total_tested = 0
        total_accepted = 0

        for module_name, module in modules:
            logger.info(f"  📦 Analyzing {module_name} ({module})...")
            module_tested = 0
            module_accepted = 0

            for name in dir(module):
                if name.startswith("_"):
                    continue

                func = getattr(module, name)
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
                logger.debug(f"       Classification: {contract.name if contract else contract}")

                if not is_valid:
                    logger.debug("       ❌ Rejected: Invalid classification")
                    continue

                doc_lines = (func.__doc__ or "").splitlines()
                first_line_doc = doc_lines[0] if doc_lines else ""
                func_name = self._generate_function_name(name, module_name)

                # Apply library adapter (preprocessing/postprocessing)
                adapted_func = self.create_library_adapter(func, contract)

                # Apply contract wrapper (slice_by_slice for FLEXIBLE)
                contract_wrapped_func = self.apply_contract_wrapper(adapted_func, contract)

                # Inject optional dataclass parameters
                final_func = self._inject_optional_dataclass_params(contract_wrapped_func)

                metadata = FunctionMetadata(
                    name=func_name,
                    func=final_func,
                    contract=contract,
                    registry=self,
                    module=func.__module__ or "",
                    doc=first_line_doc,
                    tags=self._generate_tags(name),
                    original_name=name
                )

                functions[func_name] = metadata
                module_accepted += 1
                total_accepted += 1
                logger.debug(f"       ✅ Accepted as '{func_name}'")

            logger.debug(f"  📊 Module {module_name}: {module_accepted}/{module_tested} functions accepted")

        logger.info(f"✅ Discovery complete: {total_accepted}/{total_tested} functions accepted")
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
        if func_name.startswith('_'):
            return "private"

        exclusions = getattr(self.__class__, 'EXCLUSIONS', self.COMMON_EXCLUSIONS)
        if func_name.lower() in exclusions:
            return "blacklisted"

        if inspect.isclass(func) or isinstance(func, type):
            return "is class/type"

        if not callable(func):
            return "not callable"

        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if not params:
                return "no parameters (not pure function)"
        except (ValueError, TypeError):
            return "invalid signature"

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
