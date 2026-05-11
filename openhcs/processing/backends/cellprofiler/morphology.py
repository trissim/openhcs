"""Morphology backend strategies for CellProfiler-compatible processing.

This module is the OpenHCS processing-backend seam for CellProfiler-compatible
semantics.  The default implementation is independent NumPy/SciPy/skimage code;
the optional Centrosome provider is allowed for matching legacy morphology
behavior when explicitly requested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import logging
import os
import time
from typing import Any, ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit, prange

from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy as numpy_decorator
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.image_shapes import (
    trailing_spatial_factors,
    trailing_spatial_target_shape,
)
from openhcs.core.runtime_semantics import (
    ExplicitObjectLabelDomainDeclaration,
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ParentChildRelationshipPayload,
    aligned_dense_object_label_mask_stack_alignment,
    aligned_dense_object_labels_and_mask,
    dense_object_label_plane_id_domains,
    dense_object_label_max_present_id,
    object_label_lineage_payload,
    project_dense_object_label_stack,
    relabel_dense_object_labels_consecutive,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ObjectLabelPayload,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelSet,
    image_payload_data,
    image_payload_metadata,
    object_label_dense_array,
    object_label_payload_with_dense_labels,
    with_image_payload_data,
)
from openhcs.interop.cellprofiler.expand_or_shrink_settings import (
    CellProfilerExpandShrinkOperation,
    ExpandShrinkMode,
)
from openhcs.interop.cellprofiler.image_module_settings import CombineObjectsMethod
from openhcs.interop.cellprofiler.mask_objects_settings import (
    MaskObjectsNumberingChoice,
    MaskObjectsOverlapHandling,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
    apply_structuring_element,
    build_structuring_element,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois

HolePredicate = Callable[[int, bool], bool]
ConnectivityStructureBuilder = Callable[[int], np.ndarray]
LabelBoundingBox = tuple[int, tuple[slice, ...]]
LabelBoundingBoxes = list[LabelBoundingBox]
SCIPY_CONSTANT_BOUNDARY_MODE = "constant"
MORPH_CONVOLUTION_MODE = "constant"
EIGHT_NEIGHBOR_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    dtype=np.uint8,
)
FOUR_CONNECTED_KERNEL = np.array(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    dtype=np.uint8,
)
PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


class MorphOperation(Enum):
    """CellProfiler Morph operation names."""

    BRANCHPOINTS = "branchpoints"
    BRIDGE = "bridge"
    CLEAN = "clean"
    CONVEX_HULL = "convex_hull"
    DIAG = "diag"
    DISTANCE = "distance"
    ENDPOINTS = "endpoints"
    FILL = "fill"
    HBREAK = "hbreak"
    MAJORITY = "majority"
    OPENLINES = "openlines"
    REMOVE = "remove"
    SHRINK = "shrink"
    SKELPE = "skelpe"
    SPUR = "spur"
    THICKEN = "thicken"
    THIN = "thin"
    VBREAK = "vbreak"


class RepeatMode(Enum):
    """CellProfiler Morph repeat policies."""

    ONCE = "once"
    FOREVER = "forever"
    CUSTOM = "custom"


class ResizeObjectsMethod(Enum):
    """CellProfiler ResizeObjects size policy."""

    DIMENSIONS = ("dimensions", "to_size", "manual")
    FACTOR = ("factor", "by_factor")

    def __new__(cls, value: str, *cellprofiler_literals: str):
        member = object.__new__(cls)
        member._value_ = value
        member.cellprofiler_literals = cellprofiler_literals
        return member


class FillMode(Enum):
    HOLES = "holes"
    CONVEX_HULL = "convex_hull"


class MaskChoice(Enum):
    """MaskObjects mask source kind."""

    OBJECTS = "objects"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class ResizeObjectsStats:
    slice_index: int
    original_height: int
    original_width: int
    new_height: int
    new_width: int
    object_count: int


@dataclass(frozen=True, slots=True)
class ErosionStats:
    slice_index: int
    input_object_count: int
    output_object_count: int
    objects_removed: int


@dataclass(frozen=True, slots=True)
class DilationStats:
    slice_index: int
    object_count: int
    mean_area_before: float
    mean_area_after: float


@dataclass(frozen=True, slots=True)
class DilationStats3D:
    object_count: int
    mean_volume_before: float
    mean_volume_after: float


@dataclass(frozen=True, slots=True)
class CentroidStats:
    slice_index: int
    object_count: int


@dataclass(frozen=True)
class MorphOperationRequest:
    """Execution context shared by registered Morph operation strategies."""

    image: np.ndarray
    iterations: int
    rescale_values: bool
    line_length: int
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
    memory_type: MemoryType = MemoryType.NUMPY


MorphOperationImplementation = Callable[[MorphOperationRequest], np.ndarray]
RepeatCountResolver = Callable[[int], int]
NeighborConvolutionTransition = Callable[[np.ndarray, np.ndarray], np.ndarray]
OpenLineStructureBuilder = Callable[[int], np.ndarray]


class RegisteredCallableStrategy(ABC, metaclass=AutoRegisterMeta):
    """Shared callback substrate for registered CellProfiler morphology families."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None
    callback: ClassVar[Callable[..., Any] | None] = None

    @classmethod
    def registered_type_for(cls, key: object, label: str) -> type:
        strategy_type = cls.__registry__.get(key)
        if strategy_type is None:
            raise ValueError(f"Unknown {label}: {key}")
        return strategy_type

    def invoke(self, *args: object) -> Any:
        callback = type(self).callback
        if callback is None:
            raise TypeError(f"{type(self).__name__} cannot invoke callback")
        return callback(*args)


class MorphOperationStrategy(RegisteredCallableStrategy, metaclass=AutoRegisterMeta):
    """Registered implementation authority for CellProfiler Morph operations."""

    __registry_key__ = "operation"
    __skip_if_no_key__ = True
    operation: ClassVar[MorphOperation | None] = None
    callback: ClassVar[MorphOperationImplementation | None] = None

    @classmethod
    def for_operation(cls, operation: MorphOperation) -> "MorphOperationStrategy":
        return cls.registered_type_for(operation, "Morph operation")()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if (
            cls.operation is not None
            and cls.callback is None
            and cls.apply is MorphOperationStrategy.apply
        ):
            raise TypeError(
                f"{cls.__name__} must declare implementation or apply() for Morph"
            )

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return self.invoke(request)


class RepeatModeStrategy(RegisteredCallableStrategy, metaclass=AutoRegisterMeta):
    """Registered iteration-count policy for CellProfiler Morph repeat modes."""

    __registry_key__ = "repeat_mode"
    __skip_if_no_key__ = True
    repeat_mode: ClassVar[RepeatMode | None] = None
    callback: ClassVar[RepeatCountResolver | None] = None

    @classmethod
    def for_repeat_mode(cls, repeat_mode: RepeatMode) -> "RepeatModeStrategy":
        return cls.registered_type_for(repeat_mode, "Morph repeat mode")()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.repeat_mode is not None and cls.callback is None:
            raise TypeError(f"{cls.__name__} must declare callback for RepeatMode")

    def repeat_count(self, custom_repeats: int) -> int:
        return self.invoke(custom_repeats)


class OnceRepeatModeStrategy(RepeatModeStrategy):
    """Run a Morph operation once."""

    repeat_mode = RepeatMode.ONCE
    callback = staticmethod(lambda custom_repeats: 1)


class ForeverRepeatModeStrategy(RepeatModeStrategy):
    """CellProfiler's bounded approximation of FOREVER repeat mode."""

    repeat_mode = RepeatMode.FOREVER
    callback = staticmethod(lambda custom_repeats: 10000)


class CustomRepeatModeStrategy(RepeatModeStrategy):
    """Run a Morph operation a declared number of times."""

    repeat_mode = RepeatMode.CUSTOM
    callback = staticmethod(lambda custom_repeats: custom_repeats)


def _ensure_binary(image: np.ndarray) -> np.ndarray:
    if image.dtype != bool:
        return image != 0
    return image


class IterativeConvolutionMorphOperationStrategy(MorphOperationStrategy):
    """Template strategy for Morph operations driven by neighbor convolution."""

    kernel: ClassVar[np.ndarray | None] = None
    transition: ClassVar[NeighborConvolutionTransition | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.operation is not None and (cls.kernel is None or cls.transition is None):
            raise TypeError(
                f"{cls.__name__} must declare kernel and transition for Morph"
            )

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        from scipy.ndimage import convolve

        kernel = type(self).kernel
        transition = type(self).transition
        if kernel is None or transition is None:
            raise TypeError(f"{type(self).__name__} cannot run convolutional Morph")

        result = _ensure_binary(request.image).astype(np.float32)
        for _ in range(request.iterations):
            neighbor_count = convolve(
                result.astype(np.uint8),
                kernel,
                mode=MORPH_CONVOLUTION_MODE,
                cval=0,
            )
            result = transition(result, neighbor_count)
        return result


def _branchpoints(image: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve

    binary = _ensure_binary(image)
    neighbor_count = convolve(
        binary.astype(np.uint8),
        EIGHT_NEIGHBOR_KERNEL,
        mode=MORPH_CONVOLUTION_MODE,
        cval=0,
    )
    return (binary & (neighbor_count > 2)).astype(np.float32)


def _bridge(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import convolve

    result = _ensure_binary(image).astype(np.float32)
    patterns = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),
    ]

    for _ in range(iterations):
        for pattern in patterns:
            match = convolve(result, pattern, mode=MORPH_CONVOLUTION_MODE, cval=0)
            result = np.where(match == 2, 1.0, result)
    return result


def _convex_hull(image: np.ndarray, morphology: "MorphologyBackendStrategy") -> np.ndarray:
    binary = _ensure_binary(image)
    if not np.any(binary):
        return np.zeros_like(image, dtype=np.float32)
    return morphology.convex_hull_image(binary).astype(np.float32)


def _diag(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    result = _ensure_binary(image).astype(np.float32)
    struct = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    for _ in range(iterations):
        dilated = binary_dilation(result > 0, structure=struct)
        result = np.maximum(result, dilated.astype(np.float32))
    return result


def _distance(image: np.ndarray, rescale: bool = True) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    binary = _ensure_binary(image)
    dist = distance_transform_edt(binary)
    if rescale and dist.max() > 0:
        dist = dist / dist.max()
    return dist.astype(np.float32)


def _endpoints(image: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve

    binary = _ensure_binary(image)
    neighbor_count = convolve(
        binary.astype(np.uint8),
        EIGHT_NEIGHBOR_KERNEL,
        mode=MORPH_CONVOLUTION_MODE,
        cval=0,
    )
    return (binary & (neighbor_count == 1)).astype(np.float32)


def _hbreak(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import convolve

    result = _ensure_binary(image).astype(np.float32)
    pattern = np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    for _ in range(iterations):
        match = convolve(result, pattern, mode=MORPH_CONVOLUTION_MODE, cval=0)
        result = np.where((match >= 6) & (result > 0), 0.0, result)
    return result


def _majority(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import convolve

    result = _ensure_binary(image).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    for _ in range(iterations):
        neighbor_sum = convolve(result, kernel, mode=MORPH_CONVOLUTION_MODE, cval=0)
        result = (neighbor_sum >= 5).astype(np.float32)
    return result


class OpenLineStructuringElement(RegisteredCallableStrategy, metaclass=AutoRegisterMeta):
    """Registered structuring-element authority for Morph OPENLINES angles."""

    __registry_key__ = "angle"
    __skip_if_no_key__ = True
    angle: ClassVar[int | None] = None
    callback: ClassVar[OpenLineStructureBuilder | None] = None

    @classmethod
    def registered_elements(cls) -> tuple["OpenLineStructuringElement", ...]:
        return tuple(strategy_type() for strategy_type in cls.__registry__.values())

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.angle is not None and cls.callback is None:
            raise TypeError(
                f"{cls.__name__} must declare callback for Morph OPENLINES"
            )

    def structure(self, line_length: int) -> np.ndarray:
        return self.invoke(line_length)


class HorizontalOpenLineStructuringElement(OpenLineStructuringElement):
    """Horizontal OPENLINES structuring element."""

    angle = 0
    callback = staticmethod(lambda line_length: np.ones((1, line_length), dtype=bool))


class RisingDiagonalOpenLineStructuringElement(OpenLineStructuringElement):
    """Rising diagonal OPENLINES structuring element."""

    angle = 45
    callback = staticmethod(lambda line_length: np.eye(line_length, dtype=bool))


class VerticalOpenLineStructuringElement(OpenLineStructuringElement):
    """Vertical OPENLINES structuring element."""

    angle = 90
    callback = staticmethod(lambda line_length: np.ones((line_length, 1), dtype=bool))


class FallingDiagonalOpenLineStructuringElement(OpenLineStructuringElement):
    """Falling diagonal OPENLINES structuring element."""

    angle = 135
    callback = staticmethod(lambda line_length: np.fliplr(np.eye(line_length, dtype=bool)))


def _openlines(image: np.ndarray, line_length: int = 3) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion

    binary = _ensure_binary(image)
    result = np.zeros_like(binary)
    for element in OpenLineStructuringElement.registered_elements():
        struct = element.structure(line_length)
        eroded = binary_erosion(binary, structure=struct)
        dilated = binary_dilation(eroded, structure=struct)
        result = result | dilated
    return result.astype(np.float32)


def _shrink(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from skimage.morphology import thin

    binary = _ensure_binary(image)
    return thin(binary, max_num_iter=iterations).astype(np.float32)


def _skelpe(image: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    binary = _ensure_binary(image)
    return skeletonize(binary).astype(np.float32)


def _thicken(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    result = _ensure_binary(image)
    for _ in range(iterations):
        result = binary_dilation(result)
    return result.astype(np.float32)


def _thin(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from skimage.morphology import thin

    binary = _ensure_binary(image)
    return thin(binary, max_num_iter=iterations).astype(np.float32)


def _vbreak(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import convolve

    result = _ensure_binary(image).astype(np.float32)
    pattern = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=np.float32)
    for _ in range(iterations):
        match = convolve(result, pattern, mode=MORPH_CONVOLUTION_MODE, cval=0)
        result = np.where((match >= 6) & (result > 0), 0.0, result)
    return result


def convex_hull_morph_operation(request: MorphOperationRequest) -> np.ndarray:
    """Run Morph CONVEX_HULL through the configured morphology backend."""
    morphology = MorphologyBackendStrategy.for_memory_type(
        request.memory_type,
        backend_provider=request.backend_provider,
    )
    return _convex_hull(request.image, morphology)


class BranchpointsMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.BRANCHPOINTS
    callback = staticmethod(lambda request: _branchpoints(request.image))


class BridgeMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.BRIDGE
    callback = staticmethod(lambda request: _bridge(request.image, request.iterations))


class CleanMorphOperationStrategy(IterativeConvolutionMorphOperationStrategy):
    operation = MorphOperation.CLEAN
    kernel = EIGHT_NEIGHBOR_KERNEL
    transition = staticmethod(
        lambda result, neighbor_count: np.where(neighbor_count == 0, 0.0, result)
    )


class ConvexHullMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.CONVEX_HULL
    callback = staticmethod(convex_hull_morph_operation)


class DiagMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.DIAG
    callback = staticmethod(lambda request: _diag(request.image, request.iterations))


class DistanceMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.DISTANCE
    callback = staticmethod(
        lambda request: _distance(request.image, request.rescale_values)
    )


class EndpointsMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.ENDPOINTS
    callback = staticmethod(lambda request: _endpoints(request.image))


class FillMorphOperationStrategy(IterativeConvolutionMorphOperationStrategy):
    operation = MorphOperation.FILL
    kernel = EIGHT_NEIGHBOR_KERNEL
    transition = staticmethod(
        lambda result, neighbor_count: np.where(neighbor_count == 8, 1.0, result)
    )


class HBreakMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.HBREAK
    callback = staticmethod(lambda request: _hbreak(request.image, request.iterations))


class MajorityMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.MAJORITY
    callback = staticmethod(lambda request: _majority(request.image, request.iterations))


class OpenLinesMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.OPENLINES
    callback = staticmethod(lambda request: _openlines(request.image, request.line_length))


class RemoveMorphOperationStrategy(IterativeConvolutionMorphOperationStrategy):
    operation = MorphOperation.REMOVE
    kernel = FOUR_CONNECTED_KERNEL
    transition = staticmethod(
        lambda result, neighbor_count: np.where(neighbor_count == 4, 0.0, result)
    )


class ShrinkMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.SHRINK
    callback = staticmethod(lambda request: _shrink(request.image, request.iterations))


class SkelpeMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.SKELPE
    callback = staticmethod(lambda request: _skelpe(request.image))


class SpurMorphOperationStrategy(IterativeConvolutionMorphOperationStrategy):
    operation = MorphOperation.SPUR
    kernel = EIGHT_NEIGHBOR_KERNEL
    transition = staticmethod(
        lambda result, neighbor_count: np.where(
            (neighbor_count == 1) & (result > 0),
            0.0,
            result,
        )
    )


class ThickenMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.THICKEN
    callback = staticmethod(lambda request: _thicken(request.image, request.iterations))


class ThinMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.THIN
    callback = staticmethod(lambda request: _thin(request.image, request.iterations))


class VBreakMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.VBREAK
    callback = staticmethod(lambda request: _vbreak(request.image, request.iterations))


def apply_morph_operation(
    image: np.ndarray,
    operation: MorphOperation = MorphOperation.THIN,
    repeat_mode: RepeatMode = RepeatMode.ONCE,
    custom_repeats: int = 2,
    rescale_values: bool = True,
    line_length: int = 3,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    memory_type: MemoryType = MemoryType.NUMPY,
) -> np.ndarray:
    """Apply one CellProfiler Morph operation through registered backend policies."""
    iterations = RepeatModeStrategy.for_repeat_mode(repeat_mode).repeat_count(
        custom_repeats
    )
    return MorphOperationStrategy.for_operation(operation).apply(
        MorphOperationRequest(
            image=image,
            iterations=iterations,
            rescale_values=rescale_values,
            line_length=line_length,
            backend_provider=morphology_backend_provider,
            memory_type=memory_type,
        )
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def morph(
    image: np.ndarray,
    operation: MorphOperation = MorphOperation.THIN,
    repeat_mode: RepeatMode = RepeatMode.ONCE,
    custom_repeats: int = 2,
    rescale_values: bool = True,
    line_length: int = 3,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Decorated CellProfiler Morph entrypoint backed by registered strategies."""
    return apply_morph_operation(
        image=image,
        operation=operation,
        repeat_mode=repeat_mode,
        custom_repeats=custom_repeats,
        rescale_values=rescale_values,
        line_length=line_length,
        morphology_backend_provider=morphology_backend_provider,
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def closing(
    image: np.ndarray,
    structuring_element: StructuringElement = StructuringElement.DISK,
    size: int = 3,
    morphology_backend_provider: CellProfilerBackendProvider | None = (
        CellProfilerBackendProvider.OPENCV
    ),
) -> np.ndarray:
    """Apply CellProfiler-compatible grayscale closing to an image plane."""
    pixel_data = image_payload_data(image)
    morphology = MorphologyBackendStrategy.for_callable(
        closing,
        backend_provider=morphology_backend_provider,
    )
    result = apply_structuring_element(
        pixel_data,
        build_structuring_element(structuring_element, size),
        morphology.grayscale_closing,
    )
    return with_image_payload_data(
        image,
        result.astype(pixel_data.dtype, copy=False),
        metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def opening(
    image: np.ndarray,
    structuring_element: StructuringElement = StructuringElement.DISK,
    size: int = 3,
    morphology_backend_provider: CellProfilerBackendProvider | None = (
        CellProfilerBackendProvider.OPENCV
    ),
) -> np.ndarray:
    """Apply CellProfiler-compatible grayscale opening to an image plane."""
    pixel_data = image_payload_data(image)
    morphology = MorphologyBackendStrategy.for_callable(
        opening,
        backend_provider=morphology_backend_provider,
    )
    result = apply_structuring_element(
        pixel_data,
        build_structuring_element(structuring_element, size),
        morphology.grayscale_opening,
    )
    return with_image_payload_data(
        image,
        result.astype(pixel_data.dtype, copy=False),
        metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def dilate_image(
    image: np.ndarray,
    structuring_element: StructuringElement = StructuringElement.DISK,
    size: int = 3,
) -> np.ndarray:
    """Apply grayscale dilation to an image plane."""
    from skimage.morphology import dilation

    dilated = apply_structuring_element(
        image,
        build_structuring_element(structuring_element, size),
        lambda spatial_image, footprint: dilation(spatial_image, footprint),
    )
    return dilated.astype(image.dtype)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def erode_image(
    image: np.ndarray,
    structuring_element: StructuringElement | str = StructuringElement.DISK,
    size: int = 3,
) -> np.ndarray:
    """Apply grayscale erosion to an image plane."""
    from skimage.morphology import erosion

    eroded = apply_structuring_element(
        image,
        build_structuring_element(structuring_element, size),
        lambda spatial_image, footprint: erosion(spatial_image, footprint),
    )
    return eroded.astype(image.dtype)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def remove_holes(
    image: np.ndarray,
    diameter: float = 1.0,
) -> np.ndarray:
    """Fill binary holes smaller than the CellProfiler diameter threshold."""
    return HoleRemovalDiameterPolicy(diameter=diameter, volumetric=False).apply(image)


@numpy_decorator(contract=ProcessingContract.PURE_3D)
def remove_holes_3d(
    image: np.ndarray,
    diameter: float = 1.0,
) -> np.ndarray:
    """Fill volumetric holes smaller than the CellProfiler diameter threshold."""
    return HoleRemovalDiameterPolicy(diameter=diameter, volumetric=True).apply(image)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def morphological_skeleton_2d(image: np.ndarray) -> np.ndarray:
    """Compute the 2-D morphological skeleton of a binary image."""
    from skimage.morphology import skeletonize

    return skeletonize(image > 0).astype(np.float32)


@numpy_decorator(contract=ProcessingContract.PURE_3D)
def morphological_skeleton_3d(image: np.ndarray) -> np.ndarray:
    """Compute the 3-D morphological skeleton of a binary volume."""
    from skimage.morphology import skeletonize_3d

    return skeletonize_3d(image > 0).astype(np.float32)


@numpy_decorator
def morphologicalskeleton(
    image: np.ndarray,
    volumetric: bool = False,
) -> np.ndarray:
    """Compute CellProfiler MorphologicalSkeleton on a stack or volume."""
    from skimage.morphology import skeletonize

    if volumetric:
        return morphological_skeleton_3d(image)
    binary = image > 0
    result = np.zeros_like(image, dtype=np.float32)
    for slice_index in range(image.shape[0]):
        result[slice_index] = skeletonize(binary[slice_index]).astype(np.float32)
    return result


@dataclass(frozen=True, slots=True)
class HoleRemovalDiameterPolicy:
    """CellProfiler RemoveHoles diameter threshold semantics."""

    diameter: float
    volumetric: bool = False

    @property
    def threshold(self) -> int:
        radius = self.diameter / 2.0
        if self.volumetric:
            threshold = (4.0 / 3.0) * np.pi * (radius**3)
        else:
            threshold = np.pi * (radius**2)
        return max(1, int(threshold))

    def binary_image(self, image: np.ndarray) -> np.ndarray:
        from skimage import img_as_bool

        if image.dtype.kind == "f":
            return img_as_bool(image)
        if image.dtype.kind in ("u", "i"):
            return image > 0
        return image.astype(bool)

    def apply(self, image: np.ndarray) -> np.ndarray:
        import skimage.morphology

        result = skimage.morphology.remove_small_holes(
            self.binary_image(image),
            area_threshold=self.threshold,
        )
        return result.astype(np.float32)


def face_connected_component_structure(ndim: int) -> np.ndarray:
    """Return the SciPy face-connectivity structure for an nd label image."""
    from scipy import ndimage as ndi

    return ndi.generate_binary_structure(ndim, 1)


def full_connected_component_structure(ndim: int) -> np.ndarray:
    """Return the full 3-wide neighborhood structure for an nd label image."""
    return np.ones((3,) * ndim, dtype=bool)


class ConnectedComponentConnectivity(ABC, metaclass=AutoRegisterMeta):
    """Registered structuring-element policy for connected components."""

    __registry_key__ = "connectivity"
    __skip_if_no_key__ = True
    connectivity: ClassVar[int | None] = None
    structure_builder: ClassVar[ConnectivityStructureBuilder | None] = None

    @classmethod
    def for_connectivity(cls, connectivity: int) -> "ConnectedComponentConnectivity":
        strategy_type = cls.__registry__.get(connectivity)
        if strategy_type is None:
            raise ValueError(
                f"Unsupported connected-component connectivity: {connectivity}"
            )
        return strategy_type()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.connectivity is not None and cls.structure_builder is None:
            raise TypeError(
                f"{cls.__name__} must declare structure_builder for connectivity"
            )

    def structure(self, ndim: int) -> np.ndarray:
        structure_builder = type(self).structure_builder
        if structure_builder is None:
            raise TypeError(f"{type(self).__name__} cannot build connectivity")
        return structure_builder(ndim)


class FaceConnectedComponents(ConnectedComponentConnectivity):
    """Face-connected components."""

    connectivity = 1
    structure_builder = staticmethod(face_connected_component_structure)


class FullConnectedComponents(ConnectedComponentConnectivity):
    """Fully connected components over a 3-wide neighborhood."""

    connectivity = 2
    structure_builder = staticmethod(full_connected_component_structure)


class CellProfilerDeclumpMethod(Enum):
    """Typed declumping modes that affect morphology backend geometry."""

    INTENSITY = "intensity"
    SHAPE = "shape"


class FillHolesOption(Enum):
    """CellProfiler IdentifyPrimaryObjects hole-fill phase policy."""

    NEVER = ("never", False, False)
    AFTER_BOTH = ("after_both", True, True)
    AFTER_DECLUMP = ("after_declump", False, True)

    def __new__(
        cls,
        value: str,
        fill_before_declump: bool,
        fill_after_declump: bool,
    ):
        option = object.__new__(cls)
        option._value_ = value
        option.fill_before_declump = fill_before_declump
        option.fill_after_declump = fill_after_declump
        return option

    def before_declump_requested(self, *, use_advanced_settings: bool) -> bool:
        """Return whether CP fills binary foreground holes before declumping."""
        return (not use_advanced_settings) or self.fill_before_declump

    def after_declump_requested(self, *, use_advanced_settings: bool) -> bool:
        """Return whether CP fills labeled-object holes after declumping/filtering."""
        return (not use_advanced_settings) or self.fill_after_declump


CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE = 7.0


@dataclass(frozen=True, slots=True)
class DeclumpingMaximaGeometry:
    """CellProfiler declumping maxima resize and suppression geometry."""

    image_resize_factor: float
    suppress_size: float

    @classmethod
    def from_cellprofiler_settings(
        cls,
        *,
        min_diameter: int,
        low_res_maxima: bool,
        automatic_suppression: bool,
        maxima_suppression_size: float,
    ) -> "DeclumpingMaximaGeometry":
        if min_diameter > 10 and low_res_maxima:
            image_resize_factor = 10.0 / float(min_diameter)
            if automatic_suppression:
                return cls(
                    image_resize_factor,
                    CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE,
                )
            return cls(
                image_resize_factor,
                manual_declumping_size(maxima_suppression_size)
                * image_resize_factor
                + 0.5,
            )

        if automatic_suppression:
            return cls(1.0, float(min_diameter) / 1.5)
        return cls(1.0, manual_declumping_size(maxima_suppression_size))


def manual_declumping_size(size: float) -> float:
    """Return the configured manual CP declumping size."""
    size = float(size)
    if size <= 0:
        return 0.0
    return size


class MorphologyBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal morphology operations keyed by OpenHCS memory type."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @classmethod
    def for_memory_type(
        cls,
        memory_type: MemoryType | str = MemoryType.NUMPY,
        *,
        backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
        prefer_centrosome: bool = False,
    ) -> "MorphologyBackendStrategy":
        if prefer_centrosome:
            if backend_provider not in (
                DEFAULT_CELLPROFILER_BACKEND_SELECTION,
                CellProfilerBackendProvider.CENTROSOME,
            ):
                raise ValueError(
                    "prefer_centrosome=True conflicts with explicit "
                    f"backend_provider={backend_provider!r}"
                )
            backend_provider = CellProfilerBackendProvider.CENTROSOME
        return super().for_memory_type(
            memory_type,
            backend_provider=backend_provider,
        )

    @classmethod
    def for_callable(
        cls,
        func: object,
        *,
        backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
        prefer_centrosome: bool = False,
    ) -> "MorphologyBackendStrategy":
        if prefer_centrosome:
            if backend_provider not in (
                DEFAULT_CELLPROFILER_BACKEND_SELECTION,
                CellProfilerBackendProvider.CENTROSOME,
            ):
                raise ValueError(
                    "prefer_centrosome=True conflicts with explicit "
                    f"backend_provider={backend_provider!r}"
                )
            backend_provider = CellProfilerBackendProvider.CENTROSOME
        return super().for_callable(
            func,
            backend_provider=backend_provider,
        )

    @abstractmethod
    def connected_components(
        self,
        mask: np.ndarray,
        *,
        connectivity: int = 2,
    ) -> tuple[np.ndarray, int]:
        """Label foreground components in a binary 2-D mask."""

    @abstractmethod
    def disk_footprint(self, radius: float) -> np.ndarray:
        """Return a 2-D disk footprint."""

    @abstractmethod
    def declumping_suppression_footprint(
        self,
        suppress_size: float,
        *,
        min_diameter: float,
        declump_method: CellProfilerDeclumpMethod,
    ) -> np.ndarray:
        """Return the local-maxima suppression footprint for declumping."""

    @abstractmethod
    def grayscale_opening(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        """Return grayscale morphological opening for a 2-D image."""

    @abstractmethod
    def grayscale_closing(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        """Return grayscale morphological closing for a 2-D image."""

    @abstractmethod
    def erode_labeled_objects(
        self,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        """Erode labeled objects while preserving label identities."""

    @abstractmethod
    def block_labels(
        self,
        image_shape: tuple[int, int],
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partition a 2-D plane into square block labels."""

    @abstractmethod
    def blockwise_minimum(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        block_size: int,
    ) -> np.ndarray:
        """Broadcast the masked minimum of each CellProfiler block to its pixels."""

    @abstractmethod
    def fix_labeled_result(self, values: np.ndarray) -> np.ndarray:
        """Normalize scipy.ndimage labeled reductions to an ndarray."""

    @abstractmethod
    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        """Fill enclosed background components."""

    @abstractmethod
    def fill_labeled_holes_below_size(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        """Fill enclosed background components smaller than a size limit."""

    @abstractmethod
    def restore_removed_declump_basins(
        self,
        pre_declump_labels: np.ndarray,
        labels_before_size_filter: np.ndarray,
        labels_after_size_filter: np.ndarray,
    ) -> np.ndarray:
        """Restore removed watershed basins with one surviving original identity."""

    @abstractmethod
    def local_maxima_by_label(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        """Find local maxima independently within each positive label."""

    @abstractmethod
    def smooth_image_for_declumping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        """Smooth an image using CP's mask-corrected declumping convention."""

    @abstractmethod
    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        """Return the binary convex hull of a 2-D mask."""

    @abstractmethod
    def declumping_seed_points(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        """Find CellProfiler-compatible declumping seed points."""

    @abstractmethod
    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        """Represent each connected component by one seed point."""

    @abstractmethod
    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        """Compact positive labels to 1..N."""


class NumpyMorphologyBackendStrategy(MorphologyBackendStrategy):
    """Independent NumPy/SciPy/skimage morphology backend."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False

    def connected_components(
        self,
        mask: np.ndarray,
        *,
        connectivity: int = 2,
    ) -> tuple[np.ndarray, int]:
        return _scipy_connected_components(mask, connectivity=connectivity)

    def disk_footprint(self, radius: float) -> np.ndarray:
        return _scipy_disk_footprint(radius)

    def declumping_suppression_footprint(
        self,
        suppress_size: float,
        *,
        min_diameter: float,
        declump_method: CellProfilerDeclumpMethod,
    ) -> np.ndarray:
        radius = _declumping_suppression_radius(
            suppress_size,
            min_diameter=min_diameter,
            declump_method=declump_method,
        )
        return _scipy_disk_footprint(radius)

    def grayscale_closing(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _skimage_grayscale_closing(image, footprint)

    def grayscale_opening(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _skimage_grayscale_opening(image, footprint)

    def erode_labeled_objects(
        self,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _scipy_erode_labeled_objects(labels, footprint)

    def block_labels(
        self,
        image_shape: tuple[int, int],
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _scipy_block_labels(image_shape, block_size)

    def blockwise_minimum(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        block_size: int,
    ) -> np.ndarray:
        return _scipy_blockwise_minimum(
            image,
            mask,
            block_size,
            morphology=self,
        )

    def fix_labeled_result(self, values: np.ndarray) -> np.ndarray:
        return _scipy_fix_labeled_result(values)

    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        return _scipy_fill_labeled_holes(labels, size_predicate=size_predicate)

    def fill_labeled_holes_below_size(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        return _scipy_fill_labeled_holes(
            labels,
            size_predicate=lambda size, _is_foreground: size < maximum_hole_size,
        )

    def restore_removed_declump_basins(
        self,
        pre_declump_labels: np.ndarray,
        labels_before_size_filter: np.ndarray,
        labels_after_size_filter: np.ndarray,
    ) -> np.ndarray:
        return _restore_removed_declump_basins_numba(
            np.ascontiguousarray(pre_declump_labels, dtype=np.int64),
            np.ascontiguousarray(labels_before_size_filter, dtype=np.int64),
            np.ascontiguousarray(labels_after_size_filter, dtype=np.int64),
        ).astype(np.asarray(labels_after_size_filter).dtype, copy=False)

    def local_maxima_by_label(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _scipy_local_maxima_by_label(image, labels, footprint)

    def smooth_image_for_declumping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        return _scipy_smooth_image_for_declumping(
            image,
            mask,
            filter_size,
            declump_method=declump_method,
            suppress_size=suppress_size,
            min_diameter=min_diameter,
        )

    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        return _skimage_convex_hull_image(mask)

    def declumping_seed_points(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        return _scipy_declumping_seed_points(
            image,
            labels,
            footprint,
            image_resize_factor,
            morphology=self,
        )

    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        return _scipy_shrink_components_to_seed_points(mask)

    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        return _scipy_relabel_sequential(labels)


class CentrosomeNumpyMorphologyBackendStrategy(NumpyMorphologyBackendStrategy):
    """Optional centrosome provider for NumPy-memory morphology."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def disk_footprint(self, radius: float) -> np.ndarray:
        from centrosome.cpmorphology import strel_disk

        return strel_disk(radius)

    def block_labels(
        self,
        image_shape: tuple[int, int],
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from centrosome.cpmorphology import block

        block_size = max(1, int(block_size))
        return block(image_shape, (block_size, block_size))

    def fix_labeled_result(self, values: np.ndarray) -> np.ndarray:
        from centrosome.cpmorphology import fixup_scipy_ndimage_result

        return fixup_scipy_ndimage_result(values)

    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        from centrosome.cpmorphology import fill_labeled_holes

        if size_predicate is None:
            return fill_labeled_holes(labels)
        return fill_labeled_holes(labels, size_fn=size_predicate)

    def fill_labeled_holes_below_size(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        return self.fill_labeled_holes(
            labels,
            size_predicate=lambda size, _is_foreground: size < maximum_hole_size,
        )

    def local_maxima_by_label(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        from centrosome.cpmorphology import is_local_maximum

        return np.asarray(is_local_maximum(image, labels, footprint), dtype=bool)

    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        from centrosome.cpmorphology import convex_hull_image

        return np.asarray(convex_hull_image(mask), dtype=bool)

    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        from centrosome.cpmorphology import binary_shrink

        return np.asarray(binary_shrink(mask), dtype=bool)

    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        from centrosome.cpmorphology import relabel

        relabeled, count = relabel(labels)
        return relabeled, int(count)


class NumbaNumpyMorphologyBackendStrategy(NumpyMorphologyBackendStrategy):
    """Numba-accelerated NumPy morphology backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def connected_components(
        self,
        mask: np.ndarray,
        *,
        connectivity: int = 2,
    ) -> tuple[np.ndarray, int]:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            return self._connected_components_planewise(
                mask_array,
                connectivity=connectivity,
            )
        if connectivity != 2:
            return super().connected_components(mask_array, connectivity=connectivity)
        return _foreground_components_2d_numba(np.ascontiguousarray(mask_array))

    def _connected_components_planewise(
        self,
        mask: np.ndarray,
        *,
        connectivity: int,
    ) -> tuple[np.ndarray, int]:
        if mask.ndim < 2:
            raise ValueError("Connected components requires at least two dimensions.")
        labels = np.zeros(mask.shape, dtype=np.int32)
        plane_count = int(np.prod(mask.shape[:-2], dtype=np.int64))
        source_planes = mask.reshape((plane_count, *mask.shape[-2:]))
        target_planes = labels.reshape((plane_count, *mask.shape[-2:]))
        label_offset = 0
        for plane_index in range(plane_count):
            plane_labels, plane_count_labels = self.connected_components(
                source_planes[plane_index],
                connectivity=connectivity,
            )
            if plane_count_labels:
                target_planes[plane_index] = np.where(
                    plane_labels > 0,
                    plane_labels + label_offset,
                    0,
                )
                label_offset += plane_count_labels
        return labels, label_offset

    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D convex hulls."
            )
        return _convex_hull_image_numba(np.ascontiguousarray(mask_array))

    def grayscale_closing(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        footprint_array = np.asarray(footprint, dtype=bool)
        if image_array.ndim != 2 or footprint_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D grayscale closing."
            )
        footprint_offsets = _footprint_offsets(footprint_array)
        return _grayscale_morphology_2d_numba(
            np.ascontiguousarray(image_array),
            footprint_offsets[:, 0],
            footprint_offsets[:, 1],
            True,
        )

    def grayscale_opening(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        footprint_array = np.asarray(footprint, dtype=bool)
        if image_array.ndim != 2 or footprint_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D grayscale opening."
            )
        footprint_offsets = _footprint_offsets(footprint_array)
        return _grayscale_morphology_2d_numba(
            np.ascontiguousarray(image_array),
            footprint_offsets[:, 0],
            footprint_offsets[:, 1],
            False,
        )

    def block_labels(
        self,
        image_shape: tuple[int, int],
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(image_shape) != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D block labels."
            )
        height, width = image_shape
        return _block_labels_2d_numba(
            int(height),
            int(width),
            max(1, int(block_size)),
        )

    def blockwise_minimum(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        block_size: int,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        if image_array.ndim not in (2, 3):
            return super().blockwise_minimum(image_array, mask, block_size)
        mask_array = (
            np.empty((0, 0), dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        if mask is not None and mask_array.shape != image_array.shape[:2]:
            raise ValueError(
                "Blockwise minimum mask must match image spatial shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        return _blockwise_minimum_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(mask_array),
            mask is not None,
            max(1, int(block_size)),
        )

    def erode_labeled_objects(
        self,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        footprint_array = np.asarray(footprint, dtype=bool)
        if labels_array.ndim not in (2, 3) or footprint_array.ndim != labels_array.ndim:
            return super().erode_labeled_objects(labels_array, footprint_array)
        offsets = _footprint_offsets_nd(footprint_array)
        return _erode_labeled_objects_numba(
            np.ascontiguousarray(labels_array),
            offsets,
        ).astype(labels_array.dtype, copy=False)

    def local_maxima_by_label(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        image_array = np.ascontiguousarray(image, dtype=np.float64)
        labels_array = np.ascontiguousarray(labels, dtype=np.int64)
        footprint_offsets = _footprint_offsets(footprint)
        return _local_maxima_by_label_numba(
            image_array,
            labels_array,
            footprint_offsets[:, 0],
            footprint_offsets[:, 1],
        )

    def smooth_image_for_declumping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        mask_array = np.asarray(mask, dtype=bool)
        if image_array.ndim != 2 or mask_array.ndim != 2:
            return self._smooth_image_for_declumping_planewise(
                image_array,
                mask_array,
                filter_size,
                declump_method=declump_method,
                suppress_size=suppress_size,
                min_diameter=min_diameter,
            )
        if image_array.shape != mask_array.shape:
            raise ValueError(
                "Declumping smoothing mask must match the image shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        kernel = _declumping_smoothing_kernel(
            filter_size,
            declump_method=declump_method,
            suppress_size=suppress_size,
            min_diameter=min_diameter,
        )
        if kernel.size == 0:
            return image_array
        return _smooth_image_for_declumping_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(mask_array),
            kernel,
        )

    def _smooth_image_for_declumping_planewise(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod,
        suppress_size: float | None,
        min_diameter: float | None,
    ) -> np.ndarray:
        if image.ndim < 2 or mask.ndim < 2:
            raise ValueError("Declumping smoothing requires at least two dimensions.")
        if image.shape != mask.shape:
            raise ValueError(
                "Declumping smoothing mask must match the image shape; got "
                f"mask {mask.shape!r} for image {image.shape!r}."
            )
        smoothed = np.empty_like(image)
        plane_count = int(np.prod(image.shape[:-2], dtype=np.int64))
        image_planes = image.reshape((plane_count, *image.shape[-2:]))
        mask_planes = mask.reshape((plane_count, *mask.shape[-2:]))
        target_planes = smoothed.reshape((plane_count, *image.shape[-2:]))
        for plane_index in range(plane_count):
            target_planes[plane_index] = self.smooth_image_for_declumping(
                image_planes[plane_index],
                mask_planes[plane_index],
                filter_size,
                declump_method=declump_method,
                suppress_size=suppress_size,
                min_diameter=min_diameter,
            )
        return smoothed

    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 2:
            return self._fill_labeled_holes_planewise(
                labels_array,
                size_predicate=size_predicate,
            )
        return self._fill_labeled_holes_2d(
            labels_array,
            size_predicate=size_predicate,
        )

    def _fill_labeled_holes_planewise(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        if labels.ndim < 2:
            raise ValueError("Hole filling requires at least two dimensions.")
        filled = np.empty_like(labels)
        plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
        source_planes = labels.reshape((plane_count, *labels.shape[-2:]))
        target_planes = filled.reshape((plane_count, *labels.shape[-2:]))
        for plane_index in range(plane_count):
            target_planes[plane_index] = self._fill_labeled_holes_2d(
                source_planes[plane_index],
                size_predicate=size_predicate,
            )
        return filled

    def _fill_labeled_holes_2d(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        components, sizes, touches_border, component_count = (
            _background_components_2d_numba(np.ascontiguousarray(labels))
        )
        fill_flags = np.zeros(component_count + 1, dtype=np.bool_)
        for component_id in range(1, component_count + 1):
            if touches_border[component_id]:
                continue
            if size_predicate is None or size_predicate(
                int(sizes[component_id]),
                False,
            ):
                fill_flags[component_id] = True
        if not np.any(fill_flags):
            return labels
        if labels.dtype == np.bool_:
            return _fill_binary_holes_from_components_numba(
                np.ascontiguousarray(labels),
                components,
                fill_flags,
            )
        return _fill_labeled_holes_single_label_components_numba(
            np.ascontiguousarray(labels),
            components,
            fill_flags,
        )

    def restore_removed_declump_basins(
        self,
        pre_declump_labels: np.ndarray,
        labels_before_size_filter: np.ndarray,
        labels_after_size_filter: np.ndarray,
    ) -> np.ndarray:
        pre_declump = np.asarray(pre_declump_labels)
        before = np.asarray(labels_before_size_filter)
        after = np.asarray(labels_after_size_filter)
        if pre_declump.ndim != 2 or before.ndim != 2 or after.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D declump basin restoration."
            )
        if pre_declump.shape != before.shape or before.shape != after.shape:
            raise ValueError(
                "Declump basin restoration inputs must have identical shapes; got "
                f"{pre_declump.shape!r}, {before.shape!r}, and {after.shape!r}."
            )
        return _restore_removed_declump_basins_numba(
            np.ascontiguousarray(pre_declump, dtype=np.int64),
            np.ascontiguousarray(before, dtype=np.int64),
            np.ascontiguousarray(after, dtype=np.int64),
        ).astype(after.dtype, copy=False)

    def fill_labeled_holes_below_size(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 2:
            return self._fill_labeled_holes_below_size_planewise(
                labels_array,
                maximum_hole_size,
            )
        return self._fill_labeled_holes_below_size_2d(
            labels_array,
            maximum_hole_size,
        )

    def _fill_labeled_holes_below_size_planewise(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        if labels.ndim < 2:
            raise ValueError("Hole filling requires at least two dimensions.")
        filled = np.empty_like(labels)
        plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
        source_planes = labels.reshape((plane_count, *labels.shape[-2:]))
        target_planes = filled.reshape((plane_count, *labels.shape[-2:]))
        for plane_index in range(plane_count):
            target_planes[plane_index] = self._fill_labeled_holes_below_size_2d(
                source_planes[plane_index],
                maximum_hole_size,
            )
        return filled

    def _fill_labeled_holes_below_size_2d(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        components, sizes, touches_border, component_count = (
            _background_components_2d_numba(np.ascontiguousarray(labels))
        )
        fill_flags = _hole_fill_flags_below_size_numba(
            sizes,
            touches_border,
            component_count,
            int(maximum_hole_size),
        )
        if labels.dtype == np.bool_:
            return _fill_binary_holes_from_components_numba(
                np.ascontiguousarray(labels),
                components,
                fill_flags,
            )
        return _fill_labeled_holes_single_label_components_numba(
            np.ascontiguousarray(labels),
            components,
            fill_flags,
        )

    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D seed shrinking."
            )
        return _binary_shrink_2d_numba(
            np.ascontiguousarray(mask_array),
            _binary_shrink_table_stack(),
        )

    def declumping_seed_points(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        labels_array = np.asarray(labels)
        if image_array.ndim != 2 or labels_array.ndim != 2:
            return self._declumping_seed_points_planewise(
                image_array,
                labels_array,
                footprint,
                image_resize_factor,
            )
        if image_array.shape != labels_array.shape:
            raise ValueError(
                "image and labels must have identical shapes for declumping seed extraction"
            )
        if float(image_resize_factor) != 1.0:
            return super().declumping_seed_points(
                image_array,
                labels_array,
                footprint,
                image_resize_factor,
            )

        maxima = self.local_maxima_by_label(
            image_array,
            labels_array,
            footprint,
        )
        maxima[np.asarray(image_array) <= 0] = 0
        return self.shrink_components_to_seed_points(maxima)

    def _declumping_seed_points_planewise(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        if image.ndim < 2 or labels.ndim < 2:
            raise ValueError("Declumping seed extraction requires at least two dimensions.")
        if image.shape != labels.shape:
            raise ValueError(
                "image and labels must have identical shapes for declumping seed extraction"
            )
        seeds = np.empty(labels.shape, dtype=bool)
        plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
        image_planes = image.reshape((plane_count, *image.shape[-2:]))
        label_planes = labels.reshape((plane_count, *labels.shape[-2:]))
        seed_planes = seeds.reshape((plane_count, *labels.shape[-2:]))
        for plane_index in range(plane_count):
            seed_planes[plane_index] = self.declumping_seed_points(
                image_planes[plane_index],
                label_planes[plane_index],
                footprint,
                image_resize_factor,
            )
        return seeds

    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        labels_array = np.asarray(labels)
        if labels_array.ndim > 2:
            relabeled_planes, count = _relabel_sequential_3d_numba(
                np.ascontiguousarray(
                    labels_array.reshape((-1, *labels_array.shape[-2:])),
                    dtype=np.int64,
                ),
            )
            return relabeled_planes.reshape(labels_array.shape), int(count)
        if labels_array.ndim != 2:
            raise ValueError("Relabeling requires at least two dimensions.")
        return _relabel_sequential_numba(
            np.ascontiguousarray(labels_array, dtype=np.int64),
        )


class OpenCVNumpyMorphologyBackendStrategy(NumbaNumpyMorphologyBackendStrategy):
    """OpenCV-accelerated NumPy morphology backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.OPENCV,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.OPENCV
    is_default_backend = False

    def grayscale_closing(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _opencv_morphology(image, footprint, operation="closing")

    def grayscale_opening(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _opencv_morphology(image, footprint, operation="opening")


def _scipy_disk_footprint(radius: float) -> np.ndarray:
    radius = max(0.0, float(radius))
    extent = int(radius)
    y, x = np.ogrid[-extent : extent + 1, -extent : extent + 1]
    return (x * x + y * y) <= radius * radius


def _declumping_suppression_radius(
    suppress_size: float,
    *,
    min_diameter: float,
    declump_method: CellProfilerDeclumpMethod,
) -> float:
    size = max(1.0, float(suppress_size))
    return max(1.0, size - 0.5)


def _scipy_block_labels(
    image_shape: tuple[int, int],
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    block_size = max(1, int(block_size))
    row_blocks = max(1, int(np.floor(float(height) / float(block_size))))
    column_blocks = max(1, int(np.floor(float(width) / float(block_size))))
    labels = np.empty((height, width), dtype=np.int32)
    indexes: list[int] = []
    for row in range(row_blocks):
        y_start = int(np.ceil(float(row * height) / float(row_blocks)))
        y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
        for column in range(column_blocks):
            x_start = int(np.ceil(float(column * width) / float(column_blocks)))
            x_stop = int(
                np.ceil(float((column + 1) * width) / float(column_blocks))
            )
            label = row * column_blocks + column
            labels[y_start:y_stop, x_start:x_stop] = label
            indexes.append(label)
    return labels, np.asarray(indexes, dtype=np.int32)


def _scipy_blockwise_minimum(
    image: np.ndarray,
    mask: np.ndarray | None,
    block_size: int,
    *,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    from scipy.ndimage import minimum

    image_array = np.asarray(image)
    labels, indexes = morphology.block_labels(image_array.shape[:2], block_size)
    labels = labels.copy()
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != image_array.shape[:2]:
            raise ValueError(
                "Blockwise minimum mask must match image spatial shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        labels[~mask_array] = -1

    valid = labels != -1
    result = np.zeros(image_array.shape, dtype=image_array.dtype)
    if not np.any(valid):
        return result

    if image_array.ndim == 2:
        minima = morphology.fix_labeled_result(minimum(image_array, labels, indexes))
        result[valid] = minima[labels[valid]]
        return result

    if image_array.ndim != 3:
        raise NotImplementedError(
            "Blockwise minimum currently supports 2-D images or 3-D color images."
        )
    for channel in range(image_array.shape[2]):
        minima = morphology.fix_labeled_result(
            minimum(image_array[:, :, channel], labels, indexes)
        )
        result[valid, channel] = minima[labels[valid]]
    return result


def _scipy_erode_labeled_objects(
    labels: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    import scipy.ndimage

    labels_array = np.asarray(labels)
    contours = scipy.ndimage.morphological_gradient(
        labels_array,
        footprint=np.asarray(footprint, dtype=bool),
    )
    return labels_array * (contours == 0)


@njit(cache=True)
def _block_labels_2d_numba(
    height: int,
    width: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_blocks = max(1, int(np.floor(float(height) / float(block_size))))
    column_blocks = max(1, int(np.floor(float(width) / float(block_size))))
    labels = np.empty((height, width), dtype=np.int32)
    indexes = np.empty(row_blocks * column_blocks, dtype=np.int32)
    for row in range(row_blocks):
        y_start = int(np.ceil(float(row * height) / float(row_blocks)))
        y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
        for column in range(column_blocks):
            x_start = int(np.ceil(float(column * width) / float(column_blocks)))
            x_stop = int(
                np.ceil(float((column + 1) * width) / float(column_blocks))
            )
            label = row * column_blocks + column
            indexes[label] = label
            for y in range(y_start, y_stop):
                for x in range(x_start, x_stop):
                    labels[y, x] = label
    return labels, indexes


@njit(cache=True)
def _blockwise_minimum_numba(
    image: np.ndarray,
    mask: np.ndarray,
    has_mask: bool,
    block_size: int,
) -> np.ndarray:
    height = image.shape[0]
    width = image.shape[1]
    row_blocks = max(1, int(np.floor(float(height) / float(block_size))))
    column_blocks = max(1, int(np.floor(float(width) / float(block_size))))
    label_count = row_blocks * column_blocks
    output = np.zeros(image.shape, dtype=image.dtype)

    if image.ndim == 2:
        minima = np.empty(label_count, dtype=image.dtype)
        has_value = np.zeros(label_count, dtype=np.bool_)
        for row in range(row_blocks):
            y_start = int(np.ceil(float(row * height) / float(row_blocks)))
            y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
            for column in range(column_blocks):
                x_start = int(np.ceil(float(column * width) / float(column_blocks)))
                x_stop = int(
                    np.ceil(float((column + 1) * width) / float(column_blocks))
                )
                label = row * column_blocks + column
                for y in range(y_start, y_stop):
                    for x in range(x_start, x_stop):
                        if has_mask and not mask[y, x]:
                            continue
                        value = image[y, x]
                        if not has_value[label] or value < minima[label]:
                            minima[label] = value
                            has_value[label] = True
                if has_value[label]:
                    value = minima[label]
                    for y in range(y_start, y_stop):
                        for x in range(x_start, x_stop):
                            if not has_mask or mask[y, x]:
                                output[y, x] = value
        return output

    channel_count = image.shape[2]
    minima = np.empty((label_count, channel_count), dtype=image.dtype)
    has_value = np.zeros(label_count, dtype=np.bool_)
    for row in range(row_blocks):
        y_start = int(np.ceil(float(row * height) / float(row_blocks)))
        y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
        for column in range(column_blocks):
            x_start = int(np.ceil(float(column * width) / float(column_blocks)))
            x_stop = int(np.ceil(float((column + 1) * width) / float(column_blocks)))
            label = row * column_blocks + column
            for y in range(y_start, y_stop):
                for x in range(x_start, x_stop):
                    if has_mask and not mask[y, x]:
                        continue
                    if not has_value[label]:
                        for channel in range(channel_count):
                            minima[label, channel] = image[y, x, channel]
                        has_value[label] = True
                    else:
                        for channel in range(channel_count):
                            value = image[y, x, channel]
                            if value < minima[label, channel]:
                                minima[label, channel] = value
            if has_value[label]:
                for y in range(y_start, y_stop):
                    for x in range(x_start, x_stop):
                        if not has_mask or mask[y, x]:
                            for channel in range(channel_count):
                                output[y, x, channel] = minima[label, channel]
    return output


@njit(cache=True)
def _erode_labeled_objects_numba(
    labels: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=labels.dtype)
    if labels.ndim == 2:
        height, width = labels.shape
        for y in range(height):
            for x in range(width):
                label = labels[y, x]
                if label == 0:
                    continue
                keep = True
                for offset_index in range(offsets.shape[0]):
                    yy = y + offsets[offset_index, 0]
                    xx = x + offsets[offset_index, 1]
                    if yy < 0 or xx < 0 or yy >= height or xx >= width:
                        continue
                    if labels[yy, xx] != label:
                        keep = False
                        break
                if keep:
                    output[y, x] = label
        return output

    z_size, y_size, x_size = labels.shape
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                label = labels[z, y, x]
                if label == 0:
                    continue
                keep = True
                for offset_index in range(offsets.shape[0]):
                    zz = z + offsets[offset_index, 0]
                    yy = y + offsets[offset_index, 1]
                    xx = x + offsets[offset_index, 2]
                    if (
                        zz < 0
                        or yy < 0
                        or xx < 0
                        or zz >= z_size
                        or yy >= y_size
                        or xx >= x_size
                    ):
                        continue
                    if labels[zz, yy, xx] != label:
                        keep = False
                        break
                if keep:
                    output[z, y, x] = label
    return output


def _scipy_connected_components(
    mask: np.ndarray,
    *,
    connectivity: int = 2,
) -> tuple[np.ndarray, int]:
    from scipy import ndimage as ndi

    mask_array = np.asarray(mask, dtype=bool)
    structure = ConnectedComponentConnectivity.for_connectivity(
        connectivity
    ).structure(mask_array.ndim)
    labels, count = ndi.label(mask_array, structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def _scipy_fix_labeled_result(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        return values.reshape(1)
    return values


def _scipy_fill_labeled_holes(
    labels: np.ndarray,
    *,
    size_predicate: HolePredicate | None = None,
) -> np.ndarray:
    from scipy import ndimage as ndi

    array = np.asarray(labels)
    foreground = array != 0
    background = ~foreground
    if not background.any():
        return array.copy()

    structure = ndi.generate_binary_structure(array.ndim, 1)
    background_labels, component_count = ndi.label(background, structure=structure)
    if component_count == 0:
        return array.copy()

    border_ids = _border_component_ids(background_labels)
    candidate_ids = set(range(1, component_count + 1)) - border_ids
    if size_predicate is not None:
        sizes = np.bincount(background_labels.ravel(), minlength=component_count + 1)
        candidate_ids = {
            component_id
            for component_id in candidate_ids
            if size_predicate(int(sizes[component_id]), False)
        }
    if not candidate_ids:
        return array.copy()

    fill_mask = np.isin(background_labels, tuple(sorted(candidate_ids)))
    if array.dtype == bool or np.array_equal(np.unique(array), np.array([False, True])):
        output = foreground.copy()
        output[fill_mask] = True
        return output.astype(array.dtype, copy=False)

    _, nearest_indices = ndi.distance_transform_edt(
        background,
        return_distances=True,
        return_indices=True,
    )
    output = array.copy()
    output[fill_mask] = array[
        tuple(axis_indices[fill_mask] for axis_indices in nearest_indices)
    ]
    return output


def _scipy_local_maxima_by_label(
    image: np.ndarray,
    labels: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    from scipy import ndimage as ndi

    image_array = np.asarray(image)
    labels_array = np.asarray(labels)
    maxima = np.zeros(labels_array.shape, dtype=bool)
    if image_array.shape != labels_array.shape:
        raise ValueError(
            "image and labels must have identical shapes for labeled local maxima"
        )

    for label_id, bounds in _positive_label_bounding_boxes(labels_array):
        label_crop = labels_array[bounds] == label_id
        image_crop = image_array[bounds]
        masked_image = np.where(label_crop, image_crop, -np.inf)
        local_max = ndi.maximum_filter(
            masked_image,
            footprint=footprint,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
            cval=-np.inf,
        )
        maxima[bounds] |= label_crop & (image_crop == local_max)
    return maxima


def _scipy_smooth_image_for_declumping(
    image: np.ndarray,
    mask: np.ndarray,
    filter_size: float,
    *,
    declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
    suppress_size: float | None = None,
    min_diameter: float | None = None,
) -> np.ndarray:
    import scipy.ndimage

    if filter_size == 0:
        return image
    kernel = _declumping_smoothing_kernel(
        filter_size,
        declump_method=declump_method,
        suppress_size=suppress_size,
        min_diameter=min_diameter,
    )

    def convolve(array: np.ndarray) -> np.ndarray:
        output = scipy.ndimage.convolve1d(
            array,
            kernel,
            axis=0,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
        )
        return scipy.ndimage.convolve1d(
            output,
            kernel,
            axis=1,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
        )

    mask_array = np.asarray(mask, dtype=bool)
    edge_array = convolve(mask_array.astype(float))
    masked_image = np.asarray(image).copy()
    masked_image[~mask_array] = 0
    smoothed_image = convolve(masked_image)
    valid = mask_array & (edge_array != 0)
    masked_image[valid] = smoothed_image[valid] / edge_array[valid]
    return masked_image


def _declumping_smoothing_kernel(
    filter_size: float,
    *,
    declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
    suppress_size: float | None = None,
    min_diameter: float | None = None,
) -> np.ndarray:
    if filter_size == 0:
        return np.empty((0,), dtype=np.float64)
    sigma_divisor = _declumping_smoothing_sigma_divisor(
        declump_method=declump_method,
        suppress_size=suppress_size,
        min_diameter=min_diameter,
    )
    sigma = float(filter_size) / sigma_divisor
    half_width = max(int(float(filter_size) / 2.0), 1)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = (
        1.0
        / np.sqrt(2.0 * np.pi)
        / sigma
        * np.exp(-0.5 * offsets**2 / sigma**2)
    )
    return np.ascontiguousarray(kernel, dtype=np.float64)


def _declumping_smoothing_sigma_divisor(
    *,
    declump_method: CellProfilerDeclumpMethod,
    suppress_size: float | None,
    min_diameter: float | None,
) -> float:
    return 2.35


def _skimage_convex_hull_image(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import convex_hull_image

    return np.asarray(convex_hull_image(np.asarray(mask, dtype=bool)), dtype=bool)


def _skimage_grayscale_closing(
    image: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    from skimage.morphology import closing as skimage_closing

    image_array = np.asarray(image)
    return np.asarray(
        skimage_closing(image_array, np.asarray(footprint, dtype=bool)),
        dtype=image_array.dtype,
    )


def _skimage_grayscale_opening(
    image: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    from skimage.morphology import opening as skimage_opening

    image_array = np.asarray(image)
    return np.asarray(
        skimage_opening(image_array, np.asarray(footprint, dtype=bool)),
        dtype=image_array.dtype,
    )


def _opencv_morphology(
    image: np.ndarray,
    footprint: np.ndarray,
    *,
    operation: str,
) -> np.ndarray:
    import cv2

    image_array = np.asarray(image)
    footprint_array = np.asarray(footprint, dtype=np.uint8)
    op = cv2.MORPH_OPEN if operation == "opening" else cv2.MORPH_CLOSE
    result = cv2.morphologyEx(
        np.ascontiguousarray(image_array),
        op,
        footprint_array,
        borderType=cv2.BORDER_REFLECT,
    )
    return np.asarray(result, dtype=image_array.dtype)


def _scipy_declumping_seed_points(
    image: np.ndarray,
    labels: np.ndarray,
    footprint: np.ndarray,
    image_resize_factor: float,
    *,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    from scipy import ndimage as ndi

    image_array = np.asarray(image)
    labels_array = np.asarray(labels)
    if image_array.shape != labels_array.shape:
        raise ValueError(
            "image and labels must have identical shapes for declumping seed extraction"
        )

    if image_resize_factor < 1.0:
        shape = np.maximum(
            1,
            np.ceil(np.asarray(image_array.shape) * float(image_resize_factor)),
        ).astype(int)
        coordinates = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float) / float(
            image_resize_factor
        )
        resized_image = ndi.map_coordinates(image_array, coordinates)
        resized_labels = ndi.map_coordinates(
            labels_array,
            coordinates,
            order=0,
        ).astype(labels_array.dtype, copy=False)
    else:
        resized_image = image_array
        resized_labels = labels_array

    maxima = morphology.local_maxima_by_label(
        resized_image,
        resized_labels,
        footprint,
    )
    maxima[resized_image <= 0] = 0

    if image_resize_factor < 1.0:
        inverse_resize_factor = float(image_array.shape[0]) / float(maxima.shape[0])
        coordinates = (
            np.mgrid[0 : image_array.shape[0], 0 : image_array.shape[1]].astype(float)
            / inverse_resize_factor
        )
        maxima = ndi.map_coordinates(maxima.astype(float), coordinates) > 0.5

    return morphology.shrink_components_to_seed_points(maxima)


def _positive_label_bounding_boxes(
    labels: np.ndarray,
) -> LabelBoundingBoxes:
    positive_coords = np.nonzero(labels > 0)
    if not positive_coords[0].size:
        return []

    label_values = labels[positive_coords]
    order = np.argsort(label_values, kind="stable")
    sorted_labels = label_values[order]
    sorted_coords = tuple(axis_coords[order] for axis_coords in positive_coords)
    change_offsets = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    group_starts = np.concatenate(([0], change_offsets))
    group_ends = np.concatenate((change_offsets, [sorted_labels.size]))

    boxes: LabelBoundingBoxes = []
    for start, end in zip(group_starts, group_ends):
        bounds = tuple(
            slice(int(axis_coords[start:end].min()), int(axis_coords[start:end].max()) + 1)
            for axis_coords in sorted_coords
        )
        boxes.append((int(sorted_labels[start]), bounds))
    return boxes


def _scipy_shrink_components_to_seed_points(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    mask_array = np.asarray(mask, dtype=bool)
    components, component_count = ndi.label(
        mask_array,
        structure=np.ones((3,) * mask_array.ndim, dtype=bool),
    )
    seeds = np.zeros(mask_array.shape, dtype=bool)
    for component_id, component_slice in enumerate(
        ndi.find_objects(components, max_label=component_count),
        start=1,
    ):
        if component_slice is None:
            continue
        component_crop = components[component_slice] == component_id
        coords = np.argwhere(component_crop)
        if coords.size == 0:
            continue
        centroid = coords.mean(axis=0)
        nearest = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
        seed_coord = tuple(
            int(axis_slice.start or 0) + int(coord)
            for axis_slice, coord in zip(component_slice, coords[nearest], strict=True)
        )
        seeds[seed_coord] = True
    return seeds


@lru_cache(maxsize=1)
def _binary_shrink_table_stack() -> np.ndarray:
    erode_table = np.array(
        [
            _binary_shrink_pattern_center(index)
            and _binary_shrink_component_count(index & ~16) != 1
            for index in range(512)
        ],
        dtype=np.bool_,
    )
    erode_table[_binary_shrink_index_of(np.ones((3, 3), dtype=bool))] = True

    tables = (
        erode_table
        | (
            _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
                np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
            )
            & _binary_shrink_make_table(
                False,
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
                np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
            )
        ),
        erode_table
        | (
            _binary_shrink_make_table(
                False,
                np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
                np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
            )
            & _binary_shrink_make_table(
                False,
                np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
                np.array([[0, 0, 1], [1, 1, 0], [1, 1, 0]], dtype=bool),
            )
        ),
        erode_table
        | (
            _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
                np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
            )
            & _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
                np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
            )
        ),
        erode_table
        | (
            _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
                np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
            )
            & _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=bool),
                np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]], dtype=bool),
            )
        ),
    )
    return np.ascontiguousarray(np.stack(tables), dtype=np.bool_)


def _binary_shrink_pattern_center(index: int) -> bool:
    return bool(index & 16)


def _binary_shrink_component_count(index: int) -> int:
    pattern = _binary_shrink_pattern_of(index)
    visited = np.zeros((3, 3), dtype=bool)
    components = 0
    for row in range(3):
        for col in range(3):
            if not pattern[row, col] or visited[row, col]:
                continue
            components += 1
            stack: list[tuple[int, int]] = [(row, col)]
            visited[row, col] = True
            while stack:
                current_row, current_col = stack.pop()
                for delta_row, delta_col in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if (
                        next_row < 0
                        or next_row >= 3
                        or next_col < 0
                        or next_col >= 3
                        or visited[next_row, next_col]
                        or not pattern[next_row, next_col]
                    ):
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
    return components


def _binary_shrink_pattern_of(index: int) -> np.ndarray:
    pattern = np.zeros((3, 3), dtype=bool)
    bit = 1
    for row in range(3):
        for col in range(3):
            pattern[row, col] = bool(index & bit)
            bit <<= 1
    return pattern


def _binary_shrink_index_of(pattern: np.ndarray) -> int:
    index = 0
    bit = 1
    for row in range(3):
        for col in range(3):
            if pattern[row, col]:
                index += bit
            bit <<= 1
    return index


def _binary_shrink_make_table(
    value: bool,
    pattern: np.ndarray,
    care: np.ndarray,
) -> np.ndarray:
    table = np.empty(512, dtype=np.bool_)
    for index in range(512):
        matches = True
        bit = 1
        for row in range(3):
            for col in range(3):
                if care[row, col] and bool(index & bit) != bool(pattern[row, col]):
                    matches = False
                    break
                bit <<= 1
            if not matches:
                break
        table[index] = value if matches else not value
    return table


def _scipy_relabel_sequential(labels: np.ndarray) -> tuple[np.ndarray, int]:
    labels_array = np.asarray(labels)
    positive = np.unique(labels_array[labels_array > 0])
    output = np.zeros(labels_array.shape, dtype=np.int32)
    for new_label, old_label in enumerate(positive, start=1):
        output[labels_array == old_label] = new_label
    return output, int(positive.size)


def _footprint_offsets(footprint: np.ndarray) -> np.ndarray:
    footprint_array = np.asarray(footprint, dtype=bool)
    if footprint_array.ndim != 2:
        raise NotImplementedError(
            "CellProfiler-compatible morphology currently supports 2-D footprints."
        )
    center_y = footprint_array.shape[0] // 2
    center_x = footprint_array.shape[1] // 2
    coords = np.argwhere(footprint_array)
    return np.ascontiguousarray(
        np.column_stack((coords[:, 0] - center_y, coords[:, 1] - center_x)),
        dtype=np.int64,
    )


def _footprint_offsets_nd(footprint: np.ndarray) -> np.ndarray:
    footprint_array = np.asarray(footprint, dtype=bool)
    if footprint_array.ndim not in (2, 3):
        raise NotImplementedError(
            "CellProfiler-compatible morphology currently supports 2-D and 3-D footprints."
        )
    center = np.asarray(footprint_array.shape, dtype=np.int64) // 2
    coords = np.argwhere(footprint_array).astype(np.int64)
    return np.ascontiguousarray(coords - center)


def _border_component_ids(component_labels: np.ndarray) -> set[int]:
    border_values: list[np.ndarray] = []
    for axis in range(component_labels.ndim):
        border_values.append(np.take(component_labels, 0, axis=axis).ravel())
        border_values.append(np.take(component_labels, -1, axis=axis).ravel())
    return {
        int(component_id)
        for component_id in np.concatenate(border_values)
        if component_id != 0
    }


@njit(cache=True, parallel=True)
def _grayscale_morphology_2d_numba(
    image: np.ndarray,
    offset_rows: np.ndarray,
    offset_cols: np.ndarray,
    first_pass_is_dilation: bool,
) -> np.ndarray:
    height, width = image.shape
    intermediate = np.empty_like(image)
    output = np.empty_like(image)
    footprint_size = offset_rows.size
    for row in prange(height):
        for col in range(width):
            best = image[
                _reflect_index_1d(row + int(offset_rows[0]), height),
                _reflect_index_1d(col + int(offset_cols[0]), width),
            ]
            for offset_index in range(1, footprint_size):
                value = image[
                    _reflect_index_1d(row + int(offset_rows[offset_index]), height),
                    _reflect_index_1d(col + int(offset_cols[offset_index]), width),
                ]
                if (first_pass_is_dilation and value > best) or (
                    not first_pass_is_dilation and value < best
                ):
                    best = value
            intermediate[row, col] = best

    for row in prange(height):
        for col in range(width):
            best = intermediate[
                _reflect_index_1d(row + int(offset_rows[0]), height),
                _reflect_index_1d(col + int(offset_cols[0]), width),
            ]
            for offset_index in range(1, footprint_size):
                value = intermediate[
                    _reflect_index_1d(row + int(offset_rows[offset_index]), height),
                    _reflect_index_1d(col + int(offset_cols[offset_index]), width),
                ]
                if (first_pass_is_dilation and value < best) or (
                    not first_pass_is_dilation and value > best
                ):
                    best = value
            output[row, col] = best
    return output


@njit(cache=True)
def _reflect_index_1d(index: int, size: int) -> int:
    if size <= 1:
        return 0
    reflected = index
    while reflected < 0 or reflected >= size:
        if reflected < 0:
            reflected = -reflected - 1
        else:
            reflected = 2 * size - reflected - 1
    return reflected


@njit(cache=True)
def _convex_hull_image_numba(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    output = np.zeros((height, width), dtype=np.bool_)
    point_count = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                point_count += 1

    if point_count == 0:
        return output

    row_count2 = height * 2 + 1
    min_col_by_row = np.empty(row_count2, dtype=np.int64)
    max_col_by_row = np.empty(row_count2, dtype=np.int64)
    point_capacity = max(2, row_count2 * 2)
    point_y = np.empty(point_capacity, dtype=np.int64)
    point_x = np.empty(point_capacity, dtype=np.int64)
    hull_y = np.empty(point_capacity * 2, dtype=np.int64)
    hull_x = np.empty(point_capacity * 2, dtype=np.int64)

    point_count = _collect_convex_hull_diamond_extreme_points_numba(
        mask,
        min_col_by_row,
        max_col_by_row,
        point_y,
        point_x,
    )
    if point_count == 0:
        return output

    hull_count = _monotone_chain_hull_numba(
        point_y,
        point_x,
        point_count,
        hull_y,
        hull_x,
    )
    _paint_convex_hull_mask_numba(output, hull_y, hull_x, hull_count)
    return output


@njit(cache=True)
def _collect_convex_hull_diamond_extreme_points_numba(
    mask: np.ndarray,
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    point_y: np.ndarray,
    point_x: np.ndarray,
) -> int:
    height, width = mask.shape
    row_count2 = height * 2 + 1
    for row_index in range(row_count2):
        min_col_by_row[row_index] = 9223372036854775807
        max_col_by_row[row_index] = -9223372036854775807

    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row,
                    max_col_by_row,
                    2 * y - 1,
                    2 * x,
                )
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row,
                    max_col_by_row,
                    2 * y + 1,
                    2 * x,
                )
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row,
                    max_col_by_row,
                    2 * y,
                    2 * x - 1,
                )
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row,
                    max_col_by_row,
                    2 * y,
                    2 * x + 1,
                )

    point_count = 0
    for row_index in range(row_count2):
        max_col = max_col_by_row[row_index]
        if max_col < -9223372036854775800:
            continue
        row2 = row_index - 1
        min_col = min_col_by_row[row_index]
        point_y[point_count] = row2
        point_x[point_count] = min_col
        point_count += 1
        if max_col != min_col:
            point_y[point_count] = row2
            point_x[point_count] = max_col
            point_count += 1
    return point_count


@njit(cache=True)
def _add_convex_hull_diamond_vertex_numba(
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    row2: int,
    col2: int,
) -> None:
    row_index = row2 + 1
    if col2 < min_col_by_row[row_index]:
        min_col_by_row[row_index] = col2
    if col2 > max_col_by_row[row_index]:
        max_col_by_row[row_index] = col2


@njit(cache=True)
def _cross_convex_hull_points_numba(
    ay: int,
    ax: int,
    by: int,
    bx: int,
    cy: int,
    cx: int,
) -> int:
    return (by - ay) * (cx - ax) - (bx - ax) * (cy - ay)


@njit(cache=True)
def _monotone_chain_hull_numba(
    point_y: np.ndarray,
    point_x: np.ndarray,
    point_count: int,
    hull_y: np.ndarray,
    hull_x: np.ndarray,
) -> int:
    if point_count <= 1:
        if point_count == 1:
            hull_y[0] = point_y[0]
            hull_x[0] = point_x[0]
        return point_count

    hull_count = 0
    for index in range(point_count):
        py = point_y[index]
        px = point_x[index]
        while hull_count >= 2 and _cross_convex_hull_points_numba(
            hull_y[hull_count - 2],
            hull_x[hull_count - 2],
            hull_y[hull_count - 1],
            hull_x[hull_count - 1],
            py,
            px,
        ) <= 0:
            hull_count -= 1
        hull_y[hull_count] = py
        hull_x[hull_count] = px
        hull_count += 1

    lower_count = hull_count
    for index in range(point_count - 2, -1, -1):
        py = point_y[index]
        px = point_x[index]
        while hull_count > lower_count and _cross_convex_hull_points_numba(
            hull_y[hull_count - 2],
            hull_x[hull_count - 2],
            hull_y[hull_count - 1],
            hull_x[hull_count - 1],
            py,
            px,
        ) <= 0:
            hull_count -= 1
        hull_y[hull_count] = py
        hull_x[hull_count] = px
        hull_count += 1

    if hull_count > 1:
        hull_count -= 1
    return hull_count


@njit(cache=True)
def _paint_convex_hull_mask_numba(
    output: np.ndarray,
    hull_y: np.ndarray,
    hull_x: np.ndarray,
    hull_count: int,
) -> None:
    if hull_count <= 0:
        return
    if hull_count == 1:
        if hull_y[0] % 2 != 0 or hull_x[0] % 2 != 0:
            return
        y = hull_y[0] // 2
        x = hull_x[0] // 2
        if y >= 0 and y < output.shape[0] and x >= 0 and x < output.shape[1]:
            output[y, x] = True
        return

    min_row2 = hull_y[0]
    max_row2 = hull_y[0]
    min_col2 = hull_x[0]
    max_col2 = hull_x[0]
    for index in range(1, hull_count):
        row2 = hull_y[index]
        col2 = hull_x[index]
        if row2 < min_row2:
            min_row2 = row2
        if row2 > max_row2:
            max_row2 = row2
        if col2 < min_col2:
            min_col2 = col2
        if col2 > max_col2:
            max_col2 = col2

    if hull_count == 2:
        _paint_convex_hull_line_mask_numba(
            output,
            hull_y[0],
            hull_x[0],
            hull_y[1],
            hull_x[1],
            min_row2,
            max_row2,
            min_col2,
            max_col2,
        )
        return

    area2 = 0
    for index in range(hull_count):
        next_index = 0 if index == hull_count - 1 else index + 1
        area2 += hull_y[index] * hull_x[next_index]
        area2 -= hull_y[next_index] * hull_x[index]
    positive_orientation = area2 >= 0

    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2_numba(min_row2))
    max_y = min(image_height - 1, _floor_div2_numba(max_row2))
    min_x = max(0, _ceil_div2_numba(min_col2))
    max_x = min(image_width - 1, _floor_div2_numba(max_col2))

    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            query_col2 = x * 2
            inside = True
            for index in range(hull_count):
                next_index = 0 if index == hull_count - 1 else index + 1
                cross = _cross_convex_hull_points_numba(
                    hull_y[index],
                    hull_x[index],
                    hull_y[next_index],
                    hull_x[next_index],
                    query_row2,
                    query_col2,
                )
                if positive_orientation:
                    if cross < 0:
                        inside = False
                        break
                elif cross > 0:
                    inside = False
                    break
            if inside:
                output[y, x] = True


@njit(cache=True)
def _ceil_div2_numba(value: int) -> int:
    if value >= 0:
        return (value + 1) // 2
    return value // 2


@njit(cache=True)
def _floor_div2_numba(value: int) -> int:
    if value >= 0:
        return value // 2
    return -((-value + 1) // 2)


@njit(cache=True)
def _paint_convex_hull_line_mask_numba(
    output: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    min_row2: int,
    max_row2: int,
    min_col2: int,
    max_col2: int,
) -> None:
    dy = y1 - y0
    dx = x1 - x0
    length2 = dy * dy + dx * dx
    if length2 == 0:
        if y0 % 2 == 0 and x0 % 2 == 0:
            y = y0 // 2
            x = x0 // 2
            if y >= 0 and y < output.shape[0] and x >= 0 and x < output.shape[1]:
                output[y, x] = True
        return

    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2_numba(min_row2))
    max_y = min(image_height - 1, _floor_div2_numba(max_row2))
    min_x = max(0, _ceil_div2_numba(min_col2))
    max_x = min(image_width - 1, _floor_div2_numba(max_col2))
    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            query_col2 = x * 2
            dot = (query_row2 - y0) * dy + (query_col2 - x0) * dx
            if dot < 0 or dot > length2:
                continue
            cross = dy * (query_col2 - x0) - dx * (query_row2 - y0)
            if cross == 0:
                output[y, x] = True


@njit(cache=True, parallel=True)
def _local_maxima_by_label_numba(
    image: np.ndarray,
    labels: np.ndarray,
    offset_y: np.ndarray,
    offset_x: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    maxima = np.zeros((height, width), dtype=np.bool_)
    for y in prange(height):
        for x in range(width):
            label = labels[y, x]
            if label <= 0:
                continue
            current = image[y, x]
            max_value = -np.inf
            for offset_index in range(offset_y.size):
                neighbor_y = y + offset_y[offset_index]
                neighbor_x = x + offset_x[offset_index]
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                ):
                    continue
                if labels[neighbor_y, neighbor_x] != label:
                    continue
                value = image[neighbor_y, neighbor_x]
                if value > max_value:
                    max_value = value
            maxima[y, x] = current == max_value
    return maxima


@njit(cache=True, parallel=True)
def _smooth_image_for_declumping_numba(
    image: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    radius = kernel.size // 2
    edge_vertical = np.empty((height, width), dtype=np.float64)
    image_vertical = np.empty((height, width), dtype=np.float64)
    edge_array = np.empty((height, width), dtype=np.float64)
    smoothed_image = np.empty((height, width), dtype=np.float64)

    for y in prange(height):
        for x in range(width):
            edge_sum = 0.0
            image_sum = 0.0
            for kernel_index in range(kernel.size):
                iy = y + kernel_index - radius
                if iy < 0 or iy >= height:
                    continue
                kernel_value = kernel[kernel_index]
                if mask[iy, x]:
                    edge_sum += kernel_value
                    image_sum += float(image[iy, x]) * kernel_value
            edge_vertical[y, x] = edge_sum
            image_vertical[y, x] = image_sum

    for y in prange(height):
        for x in range(width):
            edge_sum = 0.0
            image_sum = 0.0
            for kernel_index in range(kernel.size):
                ix = x + kernel_index - radius
                if ix < 0 or ix >= width:
                    continue
                kernel_value = kernel[kernel_index]
                edge_sum += edge_vertical[y, ix] * kernel_value
                image_sum += image_vertical[y, ix] * kernel_value
            edge_array[y, x] = edge_sum
            smoothed_image[y, x] = image_sum

    output = np.empty_like(image)
    for y in prange(height):
        for x in range(width):
            if mask[y, x]:
                edge_value = edge_array[y, x]
                if edge_value != 0.0:
                    output[y, x] = smoothed_image[y, x] / edge_value
                else:
                    output[y, x] = image[y, x]
            else:
                output[y, x] = 0
    return output


@njit(cache=True)
def _foreground_components_2d_numba(
    mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    capacity = height * width
    labels = np.zeros((height, width), dtype=np.int32)
    queue_y = np.empty(capacity, dtype=np.int64)
    queue_x = np.empty(capacity, dtype=np.int64)
    component_count = 0

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or labels[start_y, start_x] != 0:
                continue
            component_count += 1
            head = 0
            tail = 1
            queue_y[0] = start_y
            queue_x[0] = start_x
            labels[start_y, start_x] = component_count

            while head < tail:
                y = queue_y[head]
                x = queue_x[head]
                head += 1
                for dy in range(-1, 2):
                    ny = y + dy
                    if ny < 0 or ny >= height:
                        continue
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        nx = x + dx
                        if nx < 0 or nx >= width:
                            continue
                        if not mask[ny, nx] or labels[ny, nx] != 0:
                            continue
                        labels[ny, nx] = component_count
                        queue_y[tail] = ny
                        queue_x[tail] = nx
                        tail += 1
    return labels, component_count


@njit(cache=True)
def _background_components_2d_numba(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    height, width = labels.shape
    capacity = height * width
    components = np.zeros((height, width), dtype=np.int32)
    sizes = np.zeros(capacity + 1, dtype=np.int64)
    touches_border = np.zeros(capacity + 1, dtype=np.bool_)
    queue_y = np.empty(capacity, dtype=np.int64)
    queue_x = np.empty(capacity, dtype=np.int64)
    component_count = 0

    for start_y in range(height):
        for start_x in range(width):
            if labels[start_y, start_x] != 0 or components[start_y, start_x] != 0:
                continue
            component_count += 1
            head = 0
            tail = 1
            queue_y[0] = start_y
            queue_x[0] = start_x
            components[start_y, start_x] = component_count

            while head < tail:
                y = queue_y[head]
                x = queue_x[head]
                head += 1
                sizes[component_count] += 1
                if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                    touches_border[component_count] = True

                if y > 0 and labels[y - 1, x] == 0 and components[y - 1, x] == 0:
                    components[y - 1, x] = component_count
                    queue_y[tail] = y - 1
                    queue_x[tail] = x
                    tail += 1
                if (
                    y + 1 < height
                    and labels[y + 1, x] == 0
                    and components[y + 1, x] == 0
                ):
                    components[y + 1, x] = component_count
                    queue_y[tail] = y + 1
                    queue_x[tail] = x
                    tail += 1
                if x > 0 and labels[y, x - 1] == 0 and components[y, x - 1] == 0:
                    components[y, x - 1] = component_count
                    queue_y[tail] = y
                    queue_x[tail] = x - 1
                    tail += 1
                if (
                    x + 1 < width
                    and labels[y, x + 1] == 0
                    and components[y, x + 1] == 0
                ):
                    components[y, x + 1] = component_count
                    queue_y[tail] = y
                    queue_x[tail] = x + 1
                    tail += 1

    return components, sizes, touches_border, component_count


@njit(cache=True)
def _hole_fill_flags_below_size_numba(
    sizes: np.ndarray,
    touches_border: np.ndarray,
    component_count: int,
    maximum_hole_size: int,
) -> np.ndarray:
    fill_flags = np.zeros(component_count + 1, dtype=np.bool_)
    for component_id in range(1, component_count + 1):
        fill_flags[component_id] = (
            not touches_border[component_id]
            and sizes[component_id] < maximum_hole_size
        )
    return fill_flags


@njit(cache=True, parallel=True)
def _fill_binary_holes_from_components_numba(
    labels: np.ndarray,
    components: np.ndarray,
    fill_flags: np.ndarray,
) -> np.ndarray:
    height, width = labels.shape
    has_fillable_component = False
    for component in range(fill_flags.size):
        if fill_flags[component]:
            has_fillable_component = True
            break
    if not has_fillable_component:
        return labels

    output = labels.copy()
    for y in prange(height):
        for x in range(width):
            component = components[y, x]
            if component > 0 and fill_flags[component]:
                output[y, x] = True
    return output


@njit(cache=True)
def _fill_labeled_holes_single_label_components_numba(
    labels: np.ndarray,
    components: np.ndarray,
    fill_flags: np.ndarray,
) -> np.ndarray:
    height, width = labels.shape
    component_labels = np.zeros(fill_flags.size, dtype=np.int64)

    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0 or not fill_flags[component]:
                continue
            if y > 0:
                _record_component_boundary_label_numba(
                    component_labels,
                    component,
                    int(labels[y - 1, x]),
                )
            if y + 1 < height:
                _record_component_boundary_label_numba(
                    component_labels,
                    component,
                    int(labels[y + 1, x]),
                )
            if x > 0:
                _record_component_boundary_label_numba(
                    component_labels,
                    component,
                    int(labels[y, x - 1]),
                )
            if x + 1 < width:
                _record_component_boundary_label_numba(
                    component_labels,
                    component,
                    int(labels[y, x + 1]),
                )

    has_fillable_component = False
    for component in range(component_labels.size):
        if component_labels[component] > 0:
            has_fillable_component = True
            break
    if not has_fillable_component:
        return labels

    output = labels.copy()
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component > 0 and fill_flags[component]:
                label = component_labels[component]
                if label > 0:
                    output[y, x] = label
    return output


@njit(cache=True)
def _restore_removed_declump_basins_numba(
    pre_declump_labels: np.ndarray,
    labels_before_size_filter: np.ndarray,
    labels_after_size_filter: np.ndarray,
) -> np.ndarray:
    height, width = labels_before_size_filter.shape
    output = labels_after_size_filter.copy()
    max_pre_declump_label = 0
    for y in range(height):
        for x in range(width):
            pre_label = int(pre_declump_labels[y, x])
            if pre_label > max_pre_declump_label:
                max_pre_declump_label = pre_label

    component_surviving_label = np.zeros(max_pre_declump_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            pre_label = int(pre_declump_labels[y, x])
            after_label = int(labels_after_size_filter[y, x])
            if pre_label <= 0 or after_label <= 0:
                continue
            current_label = component_surviving_label[pre_label]
            if current_label == 0:
                component_surviving_label[pre_label] = after_label
            elif current_label != after_label:
                component_surviving_label[pre_label] = -1

    visited = np.zeros((height, width), dtype=np.bool_)
    stack_y = np.empty(height * width, dtype=np.int64)
    stack_x = np.empty(height * width, dtype=np.int64)
    component_y = np.empty(height * width, dtype=np.int64)
    component_x = np.empty(height * width, dtype=np.int64)

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x]:
                continue
            if (
                labels_before_size_filter[start_y, start_x] <= 0
                or labels_after_size_filter[start_y, start_x] > 0
            ):
                visited[start_y, start_x] = True
                continue

            stack_size = 1
            stack_y[0] = start_y
            stack_x[0] = start_x
            visited[start_y, start_x] = True
            component_size = 0
            boundary_label = 0
            has_multiple_boundary_labels = False
            pre_declump_label = 0
            has_multiple_pre_declump_labels = False

            while stack_size:
                stack_size -= 1
                y = stack_y[stack_size]
                x = stack_x[stack_size]
                component_y[component_size] = y
                component_x[component_size] = x
                component_size += 1
                current_pre_declump_label = int(pre_declump_labels[y, x])
                if current_pre_declump_label <= 0:
                    has_multiple_pre_declump_labels = True
                elif pre_declump_label == 0:
                    pre_declump_label = current_pre_declump_label
                elif pre_declump_label != current_pre_declump_label:
                    has_multiple_pre_declump_labels = True

                for dy in range(-1, 2):
                    yy = y + dy
                    if yy < 0 or yy >= height:
                        continue
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        xx = x + dx
                        if xx < 0 or xx >= width:
                            continue

                        neighbor_after = labels_after_size_filter[yy, xx]
                        if neighbor_after > 0:
                            if boundary_label == 0:
                                boundary_label = neighbor_after
                            elif boundary_label != neighbor_after:
                                has_multiple_boundary_labels = True
                            continue

                        if visited[yy, xx]:
                            continue
                        if (
                            labels_before_size_filter[yy, xx] > 0
                            and labels_after_size_filter[yy, xx] == 0
                        ):
                            visited[yy, xx] = True
                            stack_y[stack_size] = yy
                            stack_x[stack_size] = xx
                            stack_size += 1

            if (
                boundary_label <= 0
                or has_multiple_boundary_labels
                or pre_declump_label <= 0
                or has_multiple_pre_declump_labels
            ):
                continue
            surviving_label = component_surviving_label[pre_declump_label]
            if surviving_label <= 0 or surviving_label != boundary_label:
                continue
            for index in range(component_size):
                output[component_y[index], component_x[index]] = boundary_label
    return output


@njit(cache=True)
def _record_component_boundary_label_numba(
    component_labels: np.ndarray,
    component: int,
    label: int,
) -> None:
    if label <= 0:
        return
    current = component_labels[component]
    if current == 0:
        component_labels[component] = label
    elif current != label:
        component_labels[component] = -1


@njit(cache=True)
def _fill_labeled_holes_from_components_numba(
    labels: np.ndarray,
    components: np.ndarray,
    fill_flags: np.ndarray,
) -> np.ndarray:
    height, width = labels.shape
    capacity = height * width
    output = labels.copy()
    queue_y = np.empty(capacity, dtype=np.int64)
    queue_x = np.empty(capacity, dtype=np.int64)
    head = 0
    tail = 0

    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0 or not fill_flags[component]:
                continue
            label = _first_adjacent_foreground_label_numba(labels, y, x)
            if label == 0:
                continue
            output[y, x] = label
            queue_y[tail] = y
            queue_x[tail] = x
            tail += 1

    while head < tail:
        y = queue_y[head]
        x = queue_x[head]
        head += 1
        component = components[y, x]
        label = output[y, x]

        if (
            y > 0
            and components[y - 1, x] == component
            and fill_flags[component]
            and output[y - 1, x] == 0
        ):
            output[y - 1, x] = label
            queue_y[tail] = y - 1
            queue_x[tail] = x
            tail += 1
        if (
            y + 1 < height
            and components[y + 1, x] == component
            and fill_flags[component]
            and output[y + 1, x] == 0
        ):
            output[y + 1, x] = label
            queue_y[tail] = y + 1
            queue_x[tail] = x
            tail += 1
        if (
            x > 0
            and components[y, x - 1] == component
            and fill_flags[component]
            and output[y, x - 1] == 0
        ):
            output[y, x - 1] = label
            queue_y[tail] = y
            queue_x[tail] = x - 1
            tail += 1
        if (
            x + 1 < width
            and components[y, x + 1] == component
            and fill_flags[component]
            and output[y, x + 1] == 0
        ):
            output[y, x + 1] = label
            queue_y[tail] = y
            queue_x[tail] = x + 1
            tail += 1

    return output


@njit(cache=True)
def _first_adjacent_foreground_label_numba(
    labels: np.ndarray,
    y: int,
    x: int,
):
    height, width = labels.shape
    for dy in range(-1, 2):
        neighbor_y = y + dy
        if neighbor_y < 0 or neighbor_y >= height:
            continue
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            neighbor_x = x + dx
            if neighbor_x < 0 or neighbor_x >= width:
                continue
            label = labels[neighbor_y, neighbor_x]
            if label != 0:
                return label
    return labels[y, x]


@njit(cache=True)
def _binary_shrink_2d_numba(mask: np.ndarray, tables: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    current = np.zeros((height + 2, width + 2), dtype=np.bool_)
    capacity = height * width
    coords_y = np.empty(capacity, dtype=np.int64)
    coords_x = np.empty(capacity, dtype=np.int64)
    removed_y = np.empty(capacity, dtype=np.int64)
    removed_x = np.empty(capacity, dtype=np.int64)
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            padded_y = y + 1
            padded_x = x + 1
            current[padded_y, padded_x] = True
            coords_y[count] = padded_y
            coords_x[count] = padded_x
            count += 1

    iterations = count
    for _iteration in range(iterations):
        pixel_count = count
        for table_index in range(4):
            table = tables[table_index]
            new_count = 0
            removed_count = 0
            for coord_index in range(count):
                y = coords_y[coord_index]
                x = coords_x[coord_index]
                if not current[y, x]:
                    continue

                pattern_index = 0
                bit = 1
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if current[y + dy, x + dx]:
                            pattern_index += bit
                        bit <<= 1

                if table[pattern_index]:
                    coords_y[new_count] = y
                    coords_x[new_count] = x
                    new_count += 1
                else:
                    removed_y[removed_count] = y
                    removed_x[removed_count] = x
                    removed_count += 1
            for removed_index in range(removed_count):
                current[removed_y[removed_index], removed_x[removed_index]] = False
            count = new_count

        if count == pixel_count:
            break

    output = np.zeros((height, width), dtype=np.bool_)
    for coord_index in range(count):
        output[coords_y[coord_index] - 1, coords_x[coord_index] - 1] = True
    return output


@njit(cache=True)
def _seed_points_from_components_numba(
    components: np.ndarray,
    component_count: int,
) -> np.ndarray:
    height, width = components.shape
    counts = np.zeros(component_count + 1, dtype=np.int64)
    sum_y = np.zeros(component_count + 1, dtype=np.float64)
    sum_x = np.zeros(component_count + 1, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0:
                continue
            counts[component] += 1
            sum_y[component] += y
            sum_x[component] += x

    best_distance = np.empty(component_count + 1, dtype=np.float64)
    best_y = np.full(component_count + 1, -1, dtype=np.int64)
    best_x = np.full(component_count + 1, -1, dtype=np.int64)
    for component in range(component_count + 1):
        best_distance[component] = np.inf

    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0:
                continue
            centroid_y = sum_y[component] / counts[component]
            centroid_x = sum_x[component] / counts[component]
            dy = y - centroid_y
            dx = x - centroid_x
            distance = dy * dy + dx * dx
            if distance < best_distance[component]:
                best_distance[component] = distance
                best_y[component] = y
                best_x[component] = x

    seeds = np.zeros((height, width), dtype=np.bool_)
    for component in range(1, component_count + 1):
        y = best_y[component]
        x = best_x[component]
        if y >= 0 and x >= 0:
            seeds[y, x] = True
    return seeds


@njit(cache=True)
def _relabel_sequential_numba(labels: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label > max_label:
                max_label = label

    if max_label <= 0:
        return np.zeros((height, width), dtype=np.int32), 0

    present = np.zeros(max_label + 1, dtype=np.bool_)
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label > 0:
                present[label] = True

    mapping = np.zeros(max_label + 1, dtype=np.int32)
    count = 0
    for label in range(1, max_label + 1):
        if present[label]:
            count += 1
            mapping[label] = count

    output = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label > 0:
                output[y, x] = mapping[label]
    return output, count


@njit(cache=True)
def _relabel_sequential_3d_numba(labels: np.ndarray) -> tuple[np.ndarray, int]:
    plane_count, height, width = labels.shape
    max_label = 0
    for plane_index in range(plane_count):
        for y in range(height):
            for x in range(width):
                label = labels[plane_index, y, x]
                if label > max_label:
                    max_label = label

    if max_label <= 0:
        return np.zeros((plane_count, height, width), dtype=np.int32), 0

    present = np.zeros(max_label + 1, dtype=np.bool_)
    for plane_index in range(plane_count):
        for y in range(height):
            for x in range(width):
                label = labels[plane_index, y, x]
                if label > 0:
                    present[label] = True

    mapping = np.zeros(max_label + 1, dtype=np.int32)
    count = 0
    for label in range(1, max_label + 1):
        if present[label]:
            count += 1
            mapping[label] = count

    output = np.zeros((plane_count, height, width), dtype=np.int32)
    for plane_index in range(plane_count):
        for y in range(height):
            for x in range(width):
                label = labels[plane_index, y, x]
                if label > 0:
                    output[plane_index, y, x] = mapping[label]
    return output, count


class ExpandShrinkOperationStrategy(
    EnumKeyedStrategyMixin[ExpandShrinkMode],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal CellProfiler ExpandOrShrinkObjects operation strategy."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "mode"
    mode: ClassVar[ExpandShrinkMode | None] = None
    strategy_label: ClassVar[str | None] = None
    cellprofiler_operations: ClassVar[tuple[CellProfilerExpandShrinkOperation, ...]] = ()

    @classmethod
    def for_mode(
        cls,
        mode: ExpandShrinkMode | str,
    ) -> "ExpandShrinkOperationStrategy":
        resolved = coerce_cellprofiler_enum(ExpandShrinkMode, mode)
        return cls.for_enum_member(resolved)

    @abstractmethod
    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        """Return transformed labels for this operation mode."""

    def output_domain(self, labels: np.ndarray) -> ObjectLabelDomain:
        """Return CP's semantic object domain for transformed labels."""
        return ObjectLabelDomain(
            declared_object_count=dense_object_label_max_present_id(labels),
            scope=ObjectLabelDomainScope.PLANE,
        )

    @staticmethod
    def apply_label_planes(
        labels: np.ndarray,
        operation: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        output = np.empty_like(labels, dtype=np.int32)
        label_planes = labels.reshape((-1, *labels.shape[-2:]))
        output_planes = output.reshape((-1, *output.shape[-2:]))
        for plane_index in range(label_planes.shape[0]):
            output_planes[plane_index] = operation(label_planes[plane_index])
        return output


class ExpandDefinedPixelsStrategy(ExpandShrinkOperationStrategy):
    """Expand labeled objects by a fixed pixel radius."""

    mode = ExpandShrinkMode.EXPAND_DEFINED_PIXELS
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.EXPAND_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.EXPAND_BY_MEASUREMENT,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return self.expand_defined_pixels(labels, iterations)

    def expand_defined_pixels(
        self,
        labels: np.ndarray,
        iterations: int,
    ) -> np.ndarray:
        """Expand labeled objects by a defined number of pixels."""
        from scipy.ndimage import distance_transform_edt

        if iterations <= 0:
            return labels.copy()
        labels_int = labels.astype(np.int32, copy=False)
        if labels_int.ndim > 2:
            return self.apply_label_planes(
                labels_int,
                lambda plane: self.expand_defined_pixels(plane, iterations),
            )
        if _labels_are_points_numba(np.ascontiguousarray(labels_int)):
            return _expand_point_labels_defined_pixels_numba(
                np.ascontiguousarray(labels_int),
                int(iterations),
            )

        result = labels_int.copy()
        background = labels_int == 0
        distances, indices = distance_transform_edt(background, return_indices=True)
        expand_mask = background & (distances <= iterations)
        result[expand_mask] = labels_int[indices[0][expand_mask], indices[1][expand_mask]]
        return result


class ExpandInfiniteStrategy(ExpandShrinkOperationStrategy):
    """Expand labeled objects until all background is assigned."""

    mode = ExpandShrinkMode.EXPAND_INFINITE
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.EXPAND_UNTIL_TOUCHING,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _expand_until_touching(labels)


class ShrinkDefinedPixelsStrategy(ExpandShrinkOperationStrategy):
    """Shrink labeled objects by a fixed pixel radius."""

    mode = ExpandShrinkMode.SHRINK_DEFINED_PIXELS
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.SHRINK_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.SHRINK_BY_MEASUREMENT,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _shrink_defined_pixels(labels, iterations, fill_holes)


class ShrinkToPointStrategy(ExpandShrinkOperationStrategy):
    """Shrink each object to its center point."""

    mode = ExpandShrinkMode.SHRINK_TO_POINT
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.SHRINK_TO_POINT,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return self.shrink_to_point(labels, fill_holes)

    def shrink_to_point(
        self,
        labels: np.ndarray,
        fill: bool,
    ) -> np.ndarray:
        """Shrink each labeled object to a single point at its centroid."""
        labels_int = labels.astype(np.int32, copy=False)
        if labels_int.ndim > 2:
            return self.apply_label_planes(
                labels_int,
                lambda plane: self.shrink_to_point(plane, fill),
            )
        if labels_int.size == 0 or int(labels_int.max()) <= 0:
            return np.zeros_like(labels_int)
        return _shrink_to_point_numba(np.ascontiguousarray(labels_int))


class AddDividingLinesStrategy(ExpandShrinkOperationStrategy):
    """Remove touching object boundary pixels."""

    mode = ExpandShrinkMode.ADD_DIVIDING_LINES
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.ADD_DIVIDING_LINES,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _add_dividing_lines(labels)


class DespurStrategy(ExpandShrinkOperationStrategy):
    """Remove object spurs by repeated opening."""

    mode = ExpandShrinkMode.DESPUR
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.DESPUR,)

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _despur(labels, iterations)


class SkeletonizeStrategy(ExpandShrinkOperationStrategy):
    """Reduce each object to a skeleton."""

    mode = ExpandShrinkMode.SKELETONIZE
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.SKELETONIZE,)

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _skeletonize_labels(labels)


@njit(cache=True)
def _labels_are_points_numba(labels: np.ndarray) -> bool:
    max_label = 0
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label
    if max_label <= 0:
        return True

    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1
            if counts[label] > 1:
                return False
    return True


@njit(cache=True)
def _expand_point_labels_defined_pixels_numba(
    labels: np.ndarray,
    radius: int,
) -> np.ndarray:
    height, width = labels.shape
    output = labels.copy()
    radius_squared = radius * radius
    initial_distance = radius_squared + 1
    best_distance = np.full(labels.shape, initial_distance, dtype=np.int32)
    best_y = np.full(labels.shape, 2147483647, dtype=np.int32)
    best_x = np.full(labels.shape, 2147483647, dtype=np.int32)

    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            for dy in range(-radius, radius + 1):
                yy = y + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-radius, radius + 1):
                    xx = x + dx
                    if xx < 0 or xx >= width:
                        continue
                    distance = dy * dy + dx * dx
                    if distance > radius_squared:
                        continue
                    if (
                        distance < best_distance[yy, xx]
                        or (
                            distance == best_distance[yy, xx]
                            and (
                                x < best_x[yy, xx]
                                or (x == best_x[yy, xx] and y < best_y[yy, xx])
                            )
                        )
                    ):
                        best_distance[yy, xx] = distance
                        best_y[yy, xx] = y
                        best_x[yy, xx] = x
                        output[yy, xx] = label
    return output


def _expand_until_touching(labels: np.ndarray) -> np.ndarray:
    """Expand labeled objects until they touch."""
    from scipy.ndimage import distance_transform_edt

    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels,
            _expand_until_touching,
        )
    if labels.max() == 0:
        return labels.copy()
    mask = labels > 0
    _distances, indices = distance_transform_edt(~mask, return_indices=True)
    return labels[indices[0], indices[1]]


def _shrink_defined_pixels(labels: np.ndarray, iterations: int, fill: bool) -> np.ndarray:
    """Shrink labeled objects by a defined number of pixels."""
    if iterations <= 0:
        return labels.copy()

    original = labels.astype(np.int32, copy=False)
    if original.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            original,
            lambda plane: _shrink_defined_pixels(plane, iterations, fill),
        )
    result = original.copy()
    for _ in range(iterations):
        same_neighbors = np.zeros(result.shape, dtype=bool)
        center = result[1:-1, 1:-1]
        same_neighbors[1:-1, 1:-1] = (
            (center > 0)
            & (center == result[:-2, 1:-1])
            & (center == result[2:, 1:-1])
            & (center == result[1:-1, :-2])
            & (center == result[1:-1, 2:])
        )
        result = np.where(same_neighbors, result, 0).astype(np.int32, copy=False)

    if fill:
        _restore_eroded_objects_to_centroids(original, result)

    return result


def _restore_eroded_objects_to_centroids(
    original: np.ndarray,
    eroded: np.ndarray,
) -> None:
    """Preserve one centroid pixel for labels fully removed by shrinking."""
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        original.astype(np.int32, copy=False)
    )
    if region_props.label.size == 0:
        return
    remaining_ids = set(int(label_id) for label_id in np.unique(eroded) if label_id > 0)
    for index, label_id in enumerate(region_props.label):
        label_int = int(label_id)
        if label_int in remaining_ids:
            continue
        cy = int(region_props.centroid_y[index])
        cx = int(region_props.centroid_x[index])
        eroded[cy, cx] = label_int


@njit(cache=True)
def _shrink_to_point_numba(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label

    y_sums = np.zeros(max_label + 1, dtype=np.float64)
    x_sums = np.zeros(max_label + 1, dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            y_sums[label] += float(y)
            x_sums[label] += float(x)
            counts[label] += 1

    result = np.zeros(labels.shape, dtype=np.int32)
    for label in range(1, max_label + 1):
        count = counts[label]
        if count <= 0:
            continue
        cy = int(y_sums[label] / float(count))
        cx = int(x_sums[label] / float(count))
        if cy < 0:
            cy = 0
        elif cy >= height:
            cy = height - 1
        if cx < 0:
            cx = 0
        elif cx >= width:
            cx = width - 1
        result[cy, cx] = label
    return result


def _add_dividing_lines(labels: np.ndarray) -> np.ndarray:
    """Add 1-pixel dividing lines between touching objects."""
    from scipy.ndimage import maximum_filter, minimum_filter

    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels,
            _add_dividing_lines,
        )
    if labels.max() == 0:
        return labels.copy()
    result = labels.copy()
    max_filt = maximum_filter(labels, size=3)
    min_filt = minimum_filter(labels, size=3)
    boundary = (max_filt != min_filt) & (min_filt > 0)
    result[boundary] = 0
    return result


def _despur(labels: np.ndarray, iterations: int) -> np.ndarray:
    """Remove spurs from labeled objects."""
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

    if iterations <= 0:
        return labels.copy()
    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels,
            lambda plane: _despur(plane, iterations),
        )
    result = np.zeros_like(labels)
    struct = generate_binary_structure(2, 1)
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        opened = binary_erosion(obj_mask, structure=struct, iterations=iterations)
        opened = binary_dilation(opened, structure=struct, iterations=iterations)
        result[opened] = label_id
    return result


def _skeletonize_labels(labels: np.ndarray) -> np.ndarray:
    """Reduce labeled objects to their skeletons."""
    from skimage.morphology import skeletonize

    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels,
            _skeletonize_labels,
        )
    result = np.zeros_like(labels)
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        skeleton = skeletonize(obj_mask)
        result[skeleton] = label_id
    return result


def prepare_expand_or_shrink_objects() -> None:
    """Compile kernels used by common object expansion/shrink modes."""
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:5, 3:7] = 1
    labels[8:12, 9:14] = 2
    points = ShrinkToPointStrategy().shrink_to_point(labels, False)
    ExpandDefinedPixelsStrategy().expand_defined_pixels(points, 2)


@dataclass(frozen=True, slots=True)
class MaskObjectsStats:
    """MaskObjects count summary for one runtime plane."""

    slice_index: int
    original_object_count: int
    remaining_object_count: int
    objects_removed: int


@dataclass(frozen=True, slots=True)
class MaskObjectsPlaneResult:
    """MaskObjects result for one runtime plane."""

    labels: np.ndarray
    stats: MaskObjectsStats
    relationships: ParentChildRelationshipPayload


@dataclass(frozen=True, slots=True)
class MaskObjectsPlaneOperation:
    """CellProfiler MaskObjects semantics for one aligned object-label plane."""

    overlap_handling: MaskObjectsOverlapHandling
    overlap_fraction: float
    numbering: MaskObjectsNumberingChoice
    invert_mask: bool
    relationship_backend: ObjectRelationshipBackendStrategy

    def apply(
        self,
        label_image: np.ndarray,
        mask: np.ndarray,
        *,
        slice_index: int = 0,
    ) -> MaskObjectsPlaneResult:
        import scipy.ndimage as ndi

        label_image = np.asarray(label_image, dtype=np.int32)
        _aligned_labels, mask = aligned_dense_object_labels_and_mask(label_image, mask)
        label_image = _aligned_labels.astype(np.int32, copy=False)

        binary_mask = mask > 0 if mask.max() > 1 else mask.astype(bool)
        if self.invert_mask:
            binary_mask = ~binary_mask

        masked_labels = label_image.copy()
        nobjects = int(np.max(label_image))
        if nobjects == 0:
            return MaskObjectsPlaneResult(
                labels=masked_labels,
                stats=MaskObjectsStats(
                    slice_index=slice_index,
                    original_object_count=0,
                    remaining_object_count=0,
                    objects_removed=0,
                ),
                relationships=ParentChildRelationshipPayload(
                    parent_ids=(),
                    child_ids=(),
                ),
            )

        binary_mask = _size_binary_mask_like_labels(label_image, binary_mask)
        if self.overlap_handling == MaskObjectsOverlapHandling.MASK:
            masked_labels = masked_labels * binary_mask.astype(masked_labels.dtype)
        else:
            object_indices = np.arange(1, nobjects + 1, dtype=np.int32)
            pixel_counts = np.atleast_1d(
                ndi.sum(binary_mask.astype(np.float64), label_image, object_indices)
            )

            if self.overlap_handling == MaskObjectsOverlapHandling.KEEP:
                keep = pixel_counts > 0
            else:
                total_pixels = np.atleast_1d(
                    ndi.sum(
                        np.ones(label_image.shape, dtype=np.float64),
                        label_image,
                        object_indices,
                    )
                )

                if self.overlap_handling == MaskObjectsOverlapHandling.REMOVE:
                    keep = pixel_counts == total_pixels
                elif (
                    self.overlap_handling
                    == MaskObjectsOverlapHandling.REMOVE_PERCENTAGE
                ):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fractions = np.where(
                            total_pixels > 0,
                            pixel_counts / total_pixels,
                            0,
                        )
                    keep = fractions >= self.overlap_fraction
                else:
                    raise ValueError(
                        "Unsupported MaskObjects overlap handling: "
                        f"{self.overlap_handling!r}"
                    )

            keep_lookup = np.concatenate([[False], keep])
            masked_labels[~keep_lookup[label_image]] = 0

        if self.numbering == MaskObjectsNumberingChoice.RENUMBER:
            unique_labels = np.unique(masked_labels[masked_labels != 0])
            if len(unique_labels) > 0:
                indexer = np.zeros(nobjects + 1, dtype=np.int32)
                indexer[unique_labels] = np.arange(
                    1,
                    len(unique_labels) + 1,
                    dtype=np.int32,
                )
                masked_labels = indexer[masked_labels]
                remaining_count = len(unique_labels)
            else:
                remaining_count = 0
        elif self.numbering == MaskObjectsNumberingChoice.RETAIN:
            remaining_count = len(np.unique(masked_labels[masked_labels != 0]))
        else:
            raise ValueError(f"Unsupported MaskObjects numbering: {self.numbering!r}")

        return MaskObjectsPlaneResult(
            labels=masked_labels,
            stats=MaskObjectsStats(
                slice_index=slice_index,
                original_object_count=nobjects,
                remaining_object_count=remaining_count,
                objects_removed=nobjects - remaining_count,
            ),
            relationships=self.relationship_backend.parent_child_payload_from_labels(
                label_image,
                masked_labels,
            ),
        )


@dataclass(frozen=True, slots=True)
class MaskObjectsOutputLabels:
    """Typed MaskObjects label output preserving input object-label semantics."""

    source: object
    labels: np.ndarray

    def value(self) -> object:
        if not isinstance(self.source, (ObjectLabelPayload, ObjectLabelSet)):
            return self.labels
        plane_domains = dense_object_label_plane_id_domains(
            self.labels,
            domain_scope=ObjectLabelDomainScope.PLANE,
        )
        return object_label_payload_with_dense_labels(
            self.source,
            self.labels,
            domain_declaration=ExplicitObjectLabelDomainDeclaration(
                ObjectLabelDomain(
                    declared_object_id_domains=plane_domains,
                    scope=ObjectLabelDomainScope.PLANE,
                )
            ),
        )


@numpy_decorator
@special_inputs("labels", "mask")
@special_outputs(
    (
        "mask_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "original_object_count",
                "remaining_object_count",
                "objects_removed",
            ],
            analysis_type="mask_objects",
        ),
    ),
    "object_relationships",
    ("masked_labels", segmentation_mask_rois()),
)
def mask_objects(
    image: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    overlap_handling: MaskObjectsOverlapHandling = MaskObjectsOverlapHandling.MASK,
    overlap_fraction: float = 0.5,
    numbering: MaskObjectsNumberingChoice = MaskObjectsNumberingChoice.RENUMBER,
    invert_mask: bool = False,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[
    np.ndarray,
    MaskObjectsStats | list[MaskObjectsStats],
    ParentChildRelationshipPayload,
    object,
]:
    """Mask object labels while preserving OpenHCS object-label domain semantics."""

    overlap_handling = coerce_cellprofiler_enum(
        MaskObjectsOverlapHandling,
        overlap_handling,
    )
    numbering = coerce_cellprofiler_enum(MaskObjectsNumberingChoice, numbering)
    label_array = object_label_dense_array(labels, dtype=np.int32)
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=relationship_backend_provider,
    )
    operation = MaskObjectsPlaneOperation(
        overlap_handling=overlap_handling,
        overlap_fraction=overlap_fraction,
        numbering=numbering,
        invert_mask=invert_mask,
        relationship_backend=relationship_backend,
    )

    stack_slice_count = ObjectLabelRuntimeSliceStackContract.runtime_slice_count(labels)
    if stack_slice_count is None and label_array.ndim == 3:
        stack_slice_count = int(label_array.shape[0])
    if stack_slice_count is not None and stack_slice_count > 1:
        stack_alignment = aligned_dense_object_label_mask_stack_alignment(
            label_array,
            mask,
            slice_count=stack_slice_count,
        )
        if stack_alignment is not None:
            plane_results = tuple(
                operation.apply(
                    stack_alignment.label_stack[slice_index],
                    stack_alignment.mask_stack[slice_index],
                    slice_index=slice_index,
                )
                for slice_index in range(stack_slice_count)
            )
            masked_stack = stack_alignment.restore_label_stack(
                np.stack([result.labels for result in plane_results], axis=0)
            )
            plane_domains = dense_object_label_plane_id_domains(
                masked_stack,
                domain_scope=ObjectLabelDomainScope.PLANE,
            )
            masked_payload = object_label_payload_with_dense_labels(
                labels,
                masked_stack,
                domain_declaration=ExplicitObjectLabelDomainDeclaration(
                    ObjectLabelDomain(
                        declared_object_id_domains=plane_domains,
                        scope=ObjectLabelDomainScope.PLANE,
                    )
                ),
            )
            relationships = ParentChildRelationshipPayload(
                parent_ids=tuple(
                    parent_id
                    for result in plane_results
                    for parent_id in result.relationships.parent_ids
                ),
                child_ids=tuple(
                    child_id
                    for result in plane_results
                    for child_id in result.relationships.child_ids
                ),
                slice_indices=tuple(
                    slice_index
                    for slice_index, result in enumerate(plane_results)
                    for _child_id in result.relationships.child_ids
                ),
                slice_count=stack_slice_count,
            )
            return (
                image,
                [result.stats for result in plane_results],
                relationships,
                masked_payload,
            )

    try:
        label_image = project_dense_object_label_stack(label_array).astype(
            np.int32,
            copy=False,
        )
    except ValueError as exc:
        raise ValueError(
            "MaskObjects could not project object labels; "
            f"labels shape={label_array.shape!r}, "
            f"mask shape={mask.shape!r}."
        ) from exc
    result = operation.apply(label_image, mask)
    masked_labels = MaskObjectsOutputLabels(labels, result.labels).value()
    return image, result.stats, result.relationships, masked_labels


def _size_binary_mask_like_labels(
    labels: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Return a binary mask sized like CP size_similarly(labels, mask)."""
    if binary_mask.shape == labels.shape:
        return binary_mask
    result = np.zeros(labels.shape, dtype=bool)
    common_slices = tuple(
        slice(0, min(label_extent, mask_extent))
        for label_extent, mask_extent in zip(labels.shape, binary_mask.shape, strict=False)
    )
    if not common_slices:
        return result
    result[common_slices] = binary_mask[common_slices]
    return result


@dataclass(frozen=True, slots=True)
class CombineObjectsStats:
    """CellProfiler CombineObjects summary row."""

    slice_index: int
    method: str
    input_objects_x: int
    input_objects_y: int
    output_objects: int


class CombineObjectsStrategy(
    EnumKeyedStrategyMixin[CombineObjectsMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal object-label combination strategy."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    method_label: ClassVar[str | None] = None
    method: ClassVar[CombineObjectsMethod | None] = None

    @classmethod
    def for_method(
        cls,
        method: CombineObjectsMethod | str,
    ) -> "CombineObjectsStrategy":
        return cls.for_enum_member(
            coerce_cellprofiler_enum(CombineObjectsMethod, method)
        )

    @abstractmethod
    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        """Return combined labels for this policy."""

    def result(
        self,
        labels_x: np.ndarray,
        labels_y: np.ndarray,
    ) -> tuple[CombineObjectsStats, np.ndarray]:
        combined_labels = self.combine(labels_x, labels_y)
        method = type(self).method
        if method is None:
            raise TypeError(f"{type(self).__name__} must declare method.")
        return (
            CombineObjectsStats(
                slice_index=0,
                method=method.value,
                input_objects_x=positive_dense_label_count(labels_x),
                input_objects_y=positive_dense_label_count(labels_y),
                output_objects=positive_dense_label_count(combined_labels),
            ),
            combined_labels,
        )


class MergeCombineObjectsStrategy(CombineObjectsStrategy):
    """Merge overlapping objects from two label images into single objects."""

    method = CombineObjectsMethod.MERGE

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label as scipy_label

        combined_binary = ((labels_x > 0) | (labels_y > 0)).astype(np.uint8)
        merged_labels, _ = scipy_label(combined_binary)
        return merged_labels.astype(np.int32)


class PreserveCombineObjectsStrategy(CombineObjectsStrategy):
    """Preserve labels_x and add non-overlapping objects from labels_y."""

    method = CombineObjectsMethod.PRESERVE

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        result = labels_x.copy().astype(np.int32)
        max_label = labels_x.max()
        non_overlapping_mask = (labels_y > 0) & (labels_x == 0)
        if non_overlapping_mask.any():
            y_labels_in_mask = np.unique(labels_y[non_overlapping_mask])
            y_labels_in_mask = y_labels_in_mask[y_labels_in_mask > 0]
            for index, y_label in enumerate(y_labels_in_mask):
                y_object_mask = (labels_y == y_label) & non_overlapping_mask
                result[y_object_mask] = max_label + index + 1
        return result


class DiscardCombineObjectsStrategy(CombineObjectsStrategy):
    """Discard objects from labels_x that overlap labels_y."""

    method = CombineObjectsMethod.DISCARD

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label as scipy_label

        overlap_mask = (labels_x > 0) & (labels_y > 0)
        overlapping_labels = np.unique(labels_x[overlap_mask])
        result = labels_x.copy().astype(np.int32)
        for label_id in overlapping_labels:
            if label_id > 0:
                result[labels_x == label_id] = 0
        if result.max() > 0:
            result, _ = scipy_label(result > 0)
        return result.astype(np.int32)


class SegmentCombineObjectsStrategy(CombineObjectsStrategy):
    """Segment labels_x using labels_y as watershed markers."""

    method = CombineObjectsMethod.SEGMENT

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        from skimage.segmentation import watershed

        binary_x = labels_x > 0
        if not binary_x.any():
            return np.zeros_like(labels_x, dtype=np.int32)
        distance = distance_transform_edt(binary_x)
        markers = labels_y.copy()
        markers[~binary_x] = 0
        if markers.max() == 0:
            return labels_x.astype(np.int32)
        return watershed(-distance, markers, mask=binary_x).astype(np.int32)


class SplitOrMergeOperation(Enum):
    """CellProfiler SplitOrMergeObjects top-level operation."""

    MERGE = "merge"
    SPLIT = "split"


class SplitOrMergeMergeMethod(Enum):
    """CellProfiler SplitOrMergeObjects merge selection."""

    DISTANCE = "distance"
    PER_PARENT = "per_parent"


class SplitOrMergeOutputObjectType(Enum):
    """CellProfiler SplitOrMergeObjects per-parent output mode."""

    DISCONNECTED = "disconnected"
    CONVEX_HULL = "convex_hull"


class SplitOrMergeIntensityMethod(Enum):
    """CellProfiler SplitOrMergeObjects guide-image criterion."""

    CENTROIDS = "centroids"
    CLOSEST_POINT = "closest_point"


@dataclass
class SplitOrMergeStats:
    """CellProfiler SplitOrMergeObjects summary row."""

    slice_index: int
    input_object_count: int
    output_object_count: int
    operation: str


@dataclass(frozen=True, slots=True)
class SplitOrMergeRequest:
    """Complete semantic request for SplitOrMergeObjects."""

    image: np.ndarray
    labels: np.ndarray
    operation: SplitOrMergeOperation
    merge_method: SplitOrMergeMergeMethod
    output_object_type: SplitOrMergeOutputObjectType
    distance_threshold: int
    use_guide_image: bool
    minimum_intensity_fraction: float
    intensity_method: SplitOrMergeIntensityMethod
    parent_labels: np.ndarray | None
    morphology_backend_provider: BackendProviderInput

    @property
    def input_object_count(self) -> int:
        return positive_dense_label_count(self.labels)


class SplitOrMergeOperationStrategy(
    EnumKeyedStrategyMixin[SplitOrMergeOperation],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal implementation for one SplitOrMergeObjects operation."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "operation"

    operation: ClassVar[SplitOrMergeOperation]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_operation(
        cls,
        operation: SplitOrMergeOperation | str,
    ) -> "SplitOrMergeOperationStrategy":
        return cls.for_enum_member(
            coerce_cellprofiler_enum(SplitOrMergeOperation, operation)
        )

    @abstractmethod
    def execute(self, request: SplitOrMergeRequest) -> np.ndarray:
        """Return output labels for the operation."""


class SplitObjectsStrategy(SplitOrMergeOperationStrategy):
    operation = SplitOrMergeOperation.SPLIT

    def execute(self, request: SplitOrMergeRequest) -> np.ndarray:
        from scipy.ndimage import label as scipy_label

        output_labels, _ = scipy_label(
            request.labels > 0,
            structure=np.ones((3, 3), bool),
        )
        return output_labels


class MergeObjectsStrategy(SplitOrMergeOperationStrategy):
    operation = SplitOrMergeOperation.MERGE

    def execute(self, request: SplitOrMergeRequest) -> np.ndarray:
        return SplitOrMergeMergeMethodStrategy.for_method(
            request.merge_method,
        ).merge(request)


class SplitOrMergeMergeMethodStrategy(
    EnumKeyedStrategyMixin[SplitOrMergeMergeMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal implementation for one SplitOrMergeObjects merge method."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"

    method: ClassVar[SplitOrMergeMergeMethod]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_method(
        cls,
        method: SplitOrMergeMergeMethod | str,
    ) -> "SplitOrMergeMergeMethodStrategy":
        return cls.for_enum_member(
            coerce_cellprofiler_enum(SplitOrMergeMergeMethod, method)
        )

    @abstractmethod
    def merge(self, request: SplitOrMergeRequest) -> np.ndarray:
        """Return output labels for the merge method."""


class DistanceSplitOrMergeMergeMethodStrategy(SplitOrMergeMergeMethodStrategy):
    method = SplitOrMergeMergeMethod.DISTANCE

    def merge(self, request: SplitOrMergeRequest) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt, label as scipy_label

        mask = request.labels > 0
        if request.distance_threshold > 0:
            distance = distance_transform_edt(~mask)
            mask = distance < (request.distance_threshold / 2.0 + 1)

        output_labels, _ = scipy_label(mask, structure=np.ones((3, 3), bool))
        output_labels[request.labels == 0] = 0

        if request.use_guide_image:
            output_labels = SplitOrMergeGuideImageFilter().filter(
                request.labels,
                output_labels,
                request.image,
                request.minimum_intensity_fraction,
                request.intensity_method,
            )

        return relabel_dense_object_labels_consecutive(output_labels)


class ParentSplitOrMergeMergeMethodStrategy(SplitOrMergeMergeMethodStrategy):
    method = SplitOrMergeMergeMethod.PER_PARENT

    def merge(self, request: SplitOrMergeRequest) -> np.ndarray:
        if request.parent_labels is None:
            raise ValueError("parent_labels are required when merge_method is PER_PARENT")

        from skimage.measure import regionprops

        output_labels = np.zeros_like(request.labels)
        for prop in regionprops(request.labels):
            child_mask = request.labels == prop.label
            parent_values = request.parent_labels[child_mask]
            parent_values = parent_values[parent_values > 0]
            if len(parent_values) > 0:
                output_labels[child_mask] = np.bincount(parent_values).argmax()
            else:
                output_labels[child_mask] = prop.label

        if request.output_object_type == SplitOrMergeOutputObjectType.CONVEX_HULL:
            output_labels = SplitOrMergeConvexHull().labels(
                output_labels,
                MorphologyBackendStrategy.for_callable(
                    split_or_merge_objects,
                    backend_provider=request.morphology_backend_provider,
                ),
            )

        return relabel_dense_object_labels_consecutive(output_labels)


class SplitOrMergeGuideImageFilter:
    """Guide-image filtering policy for distance-based object merging."""

    def filter(
        self,
        original_labels: np.ndarray,
        merged_labels: np.ndarray,
        image: np.ndarray,
        minimum_intensity_fraction: float,
        intensity_method: SplitOrMergeIntensityMethod,
    ) -> np.ndarray:
        if intensity_method is not SplitOrMergeIntensityMethod.CLOSEST_POINT:
            return merged_labels.copy()

        from scipy.ndimage import distance_transform_edt, label as scipy_label

        _, indices = distance_transform_edt(
            original_labels == 0,
            return_indices=True,
        )
        closest_i, closest_j = indices
        object_intensity = image[closest_i, closest_j] * minimum_intensity_fraction
        valid_mask = (original_labels > 0) | (image >= object_intensity)
        output_labels, _ = scipy_label(
            valid_mask & (merged_labels > 0),
            structure=np.ones((3, 3), bool),
        )
        output_labels[original_labels == 0] = 0
        return output_labels


class SplitOrMergeConvexHull:
    """Convex-hull fill policy for per-parent merged labels."""

    def labels(
        self,
        labels: np.ndarray,
        morphology: MorphologyBackendStrategy,
    ) -> np.ndarray:
        output = np.zeros_like(labels)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        for label_id in unique_labels:
            mask = labels == label_id
            coords = np.argwhere(mask)
            if len(coords) < 3:
                output[mask] = label_id
                continue

            min_row = int(coords[:, 0].min())
            max_row = int(coords[:, 0].max()) + 1
            min_col = int(coords[:, 1].min())
            max_col = int(coords[:, 1].max()) + 1
            hull = morphology.convex_hull_image(mask[min_row:max_row, min_col:max_col])
            output[min_row:max_row, min_col:max_col][hull] = label_id

        return output


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "split_merge_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "input_object_count",
                "output_object_count",
                "operation",
            ],
            analysis_type="split_or_merge",
        ),
    ),
    ("output_labels", segmentation_mask_rois()),
)
def split_or_merge_objects(
    image: np.ndarray,
    labels: np.ndarray,
    operation: SplitOrMergeOperation = SplitOrMergeOperation.MERGE,
    merge_method: SplitOrMergeMergeMethod = SplitOrMergeMergeMethod.DISTANCE,
    output_object_type: SplitOrMergeOutputObjectType = (
        SplitOrMergeOutputObjectType.DISCONNECTED
    ),
    distance_threshold: int = 0,
    use_guide_image: bool = False,
    minimum_intensity_fraction: float = 0.9,
    intensity_method: SplitOrMergeIntensityMethod = SplitOrMergeIntensityMethod.CENTROIDS,
    parent_labels: np.ndarray | None = None,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, SplitOrMergeStats, np.ndarray]:
    """Split or merge dense object labels."""
    labels_array = object_label_dense_array(labels, dtype=np.int32)
    parent_array = (
        None
        if parent_labels is None
        else object_label_dense_array(parent_labels, dtype=np.int32)
    )
    request = SplitOrMergeRequest(
        image=image,
        labels=labels_array,
        operation=coerce_cellprofiler_enum(SplitOrMergeOperation, operation),
        merge_method=coerce_cellprofiler_enum(SplitOrMergeMergeMethod, merge_method),
        output_object_type=coerce_cellprofiler_enum(
            SplitOrMergeOutputObjectType,
            output_object_type,
        ),
        distance_threshold=distance_threshold,
        use_guide_image=use_guide_image,
        minimum_intensity_fraction=minimum_intensity_fraction,
        intensity_method=coerce_cellprofiler_enum(
            SplitOrMergeIntensityMethod,
            intensity_method,
        ),
        parent_labels=parent_array,
        morphology_backend_provider=morphology_backend_provider,
    )
    output_labels = SplitOrMergeOperationStrategy.for_operation(
        request.operation,
    ).execute(request)
    stats = SplitOrMergeStats(
        slice_index=0,
        input_object_count=int(request.input_object_count),
        output_object_count=int(positive_dense_label_count(output_labels)),
        operation=request.operation.value,
    )
    return image, stats, output_labels.astype(np.int32)


def positive_dense_label_count(labels: np.ndarray) -> int:
    """Return the count of positive labels present in a dense label image."""
    return int(len(np.unique(labels)) - (1 if 0 in labels else 0))


def dense_label_area_statistics(labels: np.ndarray) -> tuple[float, float, float]:
    """Return mean, median, and total positive-label area."""
    areas = np.bincount(np.asarray(labels).ravel())[1:]
    positive_areas = areas[areas > 0]
    if positive_areas.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(positive_areas)),
        float(np.median(positive_areas)),
        float(np.sum(positive_areas)),
    )


def filter_labels_below_minimum_diameter(
    labels: np.ndarray,
    min_diameter: float,
) -> np.ndarray:
    min_area = np.pi * (float(min_diameter) ** 2) / 4.0
    labels_array = np.ascontiguousarray(labels)
    areas = np.bincount(np.asarray(labels_array).ravel())
    return filter_labels_by_area_numba(
        labels_array,
        np.ascontiguousarray(areas),
        float(min_area),
        np.inf,
    )


def filter_labels_above_maximum_diameter(
    labels: np.ndarray,
    max_diameter: float,
) -> np.ndarray:
    max_area = np.pi * (float(max_diameter) ** 2) / 4.0
    labels_array = np.ascontiguousarray(labels)
    areas = np.bincount(np.asarray(labels_array).ravel())
    return filter_labels_by_area_numba(
        labels_array,
        np.ascontiguousarray(areas),
        0.0,
        float(max_area),
    )


def filter_labels_by_diameter_range(
    labels: np.ndarray,
    min_diameter: float,
    max_diameter: float,
) -> tuple[np.ndarray, np.ndarray]:
    min_area = np.pi * (float(min_diameter) ** 2) / 4.0
    max_area = np.pi * (float(max_diameter) ** 2) / 4.0
    labels_array = np.ascontiguousarray(labels)
    areas = np.ascontiguousarray(np.bincount(np.asarray(labels_array).ravel()))
    return filter_labels_by_diameter_range_numba(
        labels_array,
        areas,
        float(min_area),
        float(max_area),
    )


def filter_labels_by_area_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> np.ndarray:
    if labels.ndim == 2:
        return _filter_labels_by_area_2d_numba(
            labels,
            areas,
            min_area,
            max_area,
        )
    if labels.ndim == 3:
        return _filter_labels_by_area_3d_numba(
            labels,
            areas,
            min_area,
            max_area,
        )
    raise ValueError(
        "IdentifyPrimaryObjects area filtering expects 2-D planes or stacked "
        f"planes, got shape {labels.shape!r}."
    )


@njit(cache=True, parallel=True)
def _filter_labels_by_area_2d_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> np.ndarray:
    output = labels.copy()
    height, width = labels.shape
    for row in prange(height):
        for col in range(width):
            label = int(labels[row, col])
            if label <= 0:
                continue
            area = float(areas[label])
            if area < min_area or area > max_area:
                output[row, col] = 0
    return output


@njit(cache=True, parallel=True)
def _filter_labels_by_area_3d_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> np.ndarray:
    output = labels.copy()
    plane_count, height, width = labels.shape
    for plane_index in prange(plane_count):
        for row in range(height):
            for col in range(width):
                label = int(labels[plane_index, row, col])
                if label <= 0:
                    continue
                area = float(areas[label])
                if area < min_area or area > max_area:
                    output[plane_index, row, col] = 0
    return output


def filter_labels_by_diameter_range_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim == 2:
        return _filter_labels_by_diameter_range_2d_numba(
            labels,
            areas,
            min_area,
            max_area,
        )
    if labels.ndim == 3:
        return _filter_labels_by_diameter_range_3d_numba(
            labels,
            areas,
            min_area,
            max_area,
        )
    raise ValueError(
        "IdentifyPrimaryObjects size filtering expects 2-D planes or stacked "
        f"planes, got shape {labels.shape!r}."
    )


@njit(cache=True, parallel=True)
def _filter_labels_by_diameter_range_2d_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    small_removed = labels.copy()
    final = labels.copy()
    height, width = labels.shape
    for row in prange(height):
        for col in range(width):
            label = int(labels[row, col])
            if label <= 0:
                continue
            area = float(areas[label])
            if area < min_area:
                small_removed[row, col] = 0
                final[row, col] = 0
            elif area > max_area:
                final[row, col] = 0
    return small_removed, final


@njit(cache=True, parallel=True)
def _filter_labels_by_diameter_range_3d_numba(
    labels: np.ndarray,
    areas: np.ndarray,
    min_area: float,
    max_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    small_removed = labels.copy()
    final = labels.copy()
    plane_count, height, width = labels.shape
    for plane_index in prange(plane_count):
        for row in range(height):
            for col in range(width):
                label = int(labels[plane_index, row, col])
                if label <= 0:
                    continue
                area = float(areas[label])
                if area < min_area:
                    small_removed[plane_index, row, col] = 0
                    final[plane_index, row, col] = 0
                elif area > max_area:
                    final[plane_index, row, col] = 0
    return small_removed, final


def filter_border_objects(
    labeled_image: np.ndarray,
    *,
    image_mask: np.ndarray | None,
    image_metadata: ImagePayloadMetadata = ImagePayloadMetadata(),
) -> np.ndarray:
    """Remove labels touching the physical border or masked image border."""
    labeled_array = np.asarray(labeled_image)
    if labeled_array.ndim > 2:
        return filter_border_objects_planewise(
            labeled_array,
            image_mask=image_mask,
            image_metadata=image_metadata,
        )

    height, width = labeled_array.shape[:2]
    physical_edges = image_metadata.physical_border_edges_for_shape((height, width))
    output, removed_physical = filter_physical_border_objects_numba(
        np.ascontiguousarray(labeled_array),
        bool(physical_edges[0]),
        bool(physical_edges[1]),
        bool(physical_edges[2]),
        bool(physical_edges[3]),
    )
    if removed_physical:
        return output

    if image_mask is None or image_metadata.mask_defines_border is False:
        return output

    from scipy import ndimage as ndi

    max_label = int(output.max())
    if max_label <= 0:
        return output
    mask = np.asarray(image_mask, dtype=bool)
    mask_border = np.logical_not(ndi.binary_erosion(mask, border_value=1)) & mask
    masked_border_labels = output[mask_border].astype(np.int64, copy=False)
    masked_border_histogram = np.bincount(
        masked_border_labels,
        minlength=max_label + 1,
    )
    labels_to_remove = np.flatnonzero(masked_border_histogram[1:] > 0) + 1
    if labels_to_remove.size:
        output[np.isin(output, labels_to_remove)] = 0
    return output


def filter_border_objects_planewise(
    labeled_image: np.ndarray,
    *,
    image_mask: np.ndarray | None,
    image_metadata: ImagePayloadMetadata,
) -> np.ndarray:
    output = np.empty_like(labeled_image)
    label_planes = labeled_image.reshape((-1, *labeled_image.shape[-2:]))
    output_planes = output.reshape((-1, *output.shape[-2:]))
    mask_planes = mask_planes_for_labels(image_mask, label_planes.shape[0])
    for plane_index in range(label_planes.shape[0]):
        output_planes[plane_index] = filter_border_objects(
            label_planes[plane_index],
            image_mask=None if mask_planes is None else mask_planes[plane_index],
            image_metadata=image_metadata.for_channel(plane_index),
        )
    return output


def mask_planes_for_labels(
    image_mask: np.ndarray | None,
    plane_count: int,
) -> np.ndarray | None:
    if image_mask is None:
        return None
    mask = np.asarray(image_mask, dtype=bool)
    if mask.ndim == 2:
        return np.broadcast_to(mask, (plane_count, *mask.shape))
    mask_planes = mask.reshape((-1, *mask.shape[-2:]))
    if mask_planes.shape[0] == plane_count:
        return mask_planes
    if mask_planes.shape[0] == 1:
        return np.broadcast_to(mask_planes[0], (plane_count, *mask_planes.shape[-2:]))
    raise ValueError(
        "IdentifyPrimaryObjects mask stack must align with label stack; got "
        f"{mask.shape!r} for {plane_count} label planes."
    )


@njit(cache=True)
def filter_physical_border_objects_numba(
    labels: np.ndarray,
    top: bool,
    bottom: bool,
    left: bool,
    right: bool,
) -> tuple[np.ndarray, bool]:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label
    if max_label <= 0:
        return labels, False

    remove = np.zeros(max_label + 1, dtype=np.bool_)
    if top and height > 0:
        for x in range(width):
            label = int(labels[0, x])
            if label > 0:
                remove[label] = True
    if bottom and height > 0:
        for x in range(width):
            label = int(labels[height - 1, x])
            if label > 0:
                remove[label] = True
    if left and width > 0:
        for y in range(height):
            label = int(labels[y, 0])
            if label > 0:
                remove[label] = True
    if right and width > 0:
        for y in range(height):
            label = int(labels[y, width - 1])
            if label > 0:
                remove[label] = True

    any_removed = False
    for label in range(1, max_label + 1):
        if remove[label]:
            any_removed = True
            break
    if not any_removed:
        return labels, False

    output = labels.copy()
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > 0 and remove[label]:
                output[y, x] = 0
    return output, True


def profile_function_runtime_enabled() -> bool:
    return os.environ.get(PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def log_function_runtime_profile(label: str, seconds: float, **fields: object) -> None:
    if not profile_function_runtime_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "erosion_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "input_object_count",
                "output_object_count",
                "objects_removed",
            ],
            analysis_type="erosion",
        ),
    ),
    "parent_child_relationship",
    ("eroded_labels", segmentation_mask_rois()),
)
def erode_objects(
    image: np.ndarray,
    labels: np.ndarray,
    structuring_element: Any = "disk",
    size: int = 1,
    preserve_midpoints: bool = True,
    relabel_objects: bool = False,
) -> tuple[np.ndarray, ErosionStats, ParentChildRelationshipPayload, np.ndarray]:
    """Erode CellProfiler object labels while preserving optional midpoints."""
    from skimage.measure import label as relabel
    from openhcs.processing.backends.cellprofiler.structuring_elements import (
        adapt_structuring_element_rank,
    )

    total_started_at = time.perf_counter()
    labels = object_label_dense_array(labels, dtype=np.int32)
    footprint = adapt_structuring_element_rank(
        build_structuring_element(structuring_element, size),
        labels.ndim,
    )

    phase_started_at = time.perf_counter()
    input_labels = np.unique(labels)
    input_labels = input_labels[input_labels != 0]
    input_count = len(input_labels)
    log_function_runtime_profile(
        "erode_objects_input_labels",
        time.perf_counter() - phase_started_at,
    )

    phase_started_at = time.perf_counter()
    eroded = MorphologyBackendStrategy.for_memory_type().erode_labeled_objects(
        labels,
        footprint,
    )
    log_function_runtime_profile(
        "erode_objects_backend",
        time.perf_counter() - phase_started_at,
    )

    if preserve_midpoints:
        phase_started_at = time.perf_counter()
        missing_labels = np.setxor1d(labels, eroded)
        preservation = MidpointPreservationPolicy.for_footprint(footprint)
        eroded = preservation.preserve_missing_labels(
            labels,
            eroded,
            missing_labels,
        )
        log_function_runtime_profile(
            "erode_objects_preserve_midpoints",
            time.perf_counter() - phase_started_at,
            missing=len(missing_labels),
            policy=type(preservation).__name__,
        )

    if relabel_objects:
        phase_started_at = time.perf_counter()
        eroded = relabel(eroded > 0).astype(labels.dtype)
        log_function_runtime_profile(
            "erode_objects_relabel",
            time.perf_counter() - phase_started_at,
        )

    phase_started_at = time.perf_counter()
    output_labels = np.unique(eroded)
    output_labels = output_labels[output_labels != 0]
    output_count = len(output_labels)
    log_function_runtime_profile(
        "erode_objects_output_labels",
        time.perf_counter() - phase_started_at,
    )

    stats = ErosionStats(
        slice_index=0,
        input_object_count=input_count,
        output_object_count=output_count,
        objects_removed=input_count - output_count,
    )

    phase_started_at = time.perf_counter()
    relationship = object_label_lineage_payload(labels, eroded)
    log_function_runtime_profile(
        "erode_objects_lineage",
        time.perf_counter() - phase_started_at,
    )
    log_function_runtime_profile(
        "erode_objects_total",
        time.perf_counter() - total_started_at,
    )
    return image, stats, relationship, eroded


class MidpointPreservationPolicy:
    """CellProfiler midpoint preservation for labels lost during erosion."""

    def preserve_missing_labels(
        self,
        labels: np.ndarray,
        eroded: np.ndarray,
        missing_labels: np.ndarray,
    ) -> np.ndarray:
        for label_id in missing_labels:
            label_positions = np.argwhere(labels == label_id)
            if label_positions.size == 0:
                continue
            lower = label_positions.min(axis=0)
            upper = label_positions.max(axis=0) + 1
            expanded_lower = np.maximum(lower - 1, 0)
            expanded_upper = np.minimum(upper + 1, labels.shape)
            expanded_slices = tuple(
                slice(int(start), int(stop))
                for start, stop in zip(expanded_lower, expanded_upper, strict=True)
            )
            inner_slices = tuple(
                slice(int(start - expanded_start), int(stop - expanded_start))
                for start, stop, expanded_start in zip(
                    lower,
                    upper,
                    expanded_lower,
                    strict=True,
                )
            )
            output_slices = tuple(
                slice(int(start), int(stop))
                for start, stop in zip(lower, upper, strict=True)
            )
            binary = labels[expanded_slices] == label_id
            midpoint = self.midpoint_distance(binary)[inner_slices]
            eroded_region = eroded[output_slices]
            eroded_region[midpoint == np.max(midpoint)] = label_id
        return eroded

    def midpoint_distance(self, binary: np.ndarray) -> np.ndarray:
        import scipy.ndimage

        return scipy.ndimage.distance_transform_edt(binary)

    @classmethod
    def for_footprint(cls, footprint: np.ndarray) -> "MidpointPreservationPolicy":
        if SimpleDiskMidpointPreservationPolicy.matches(footprint):
            return SimpleDiskMidpointPreservationPolicy()
        return cls()


class SimpleDiskMidpointPreservationPolicy(MidpointPreservationPolicy):
    """CellProfiler's optimized disk-1 behavior restores entire missing labels."""

    @classmethod
    def matches(cls, footprint: np.ndarray) -> bool:
        import skimage.morphology

        return (
            footprint.ndim == 2
            and footprint.shape == (3, 3)
            and np.array_equal(footprint, skimage.morphology.disk(1))
        )

    def preserve_missing_labels(
        self,
        labels: np.ndarray,
        eroded: np.ndarray,
        missing_labels: np.ndarray,
    ) -> np.ndarray:
        return eroded + labels * np.isin(labels, missing_labels)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "dilation_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "object_count",
                "mean_area_before",
                "mean_area_after",
            ],
            analysis_type="dilation",
        ),
    ),
    ("dilated_labels", segmentation_mask_rois()),
)
def dilate_objects(
    image: np.ndarray,
    labels: np.ndarray,
    structuring_element_shape: StructuringElement | str = StructuringElement.DISK,
    structuring_element_size: int = 1,
) -> tuple[np.ndarray, DilationStats, np.ndarray]:
    """Dilate labels with CellProfiler's higher-label-overwrites policy."""
    from scipy.ndimage import grey_dilation

    label_array = object_label_dense_array(labels, dtype=np.int32)
    props_before = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        label_array
    )
    mean_area_before = (
        float(np.mean(props_before.area)) if props_before.label.size else 0.0
    )
    footprint = build_structuring_element(
        structuring_element_shape,
        structuring_element_size,
    )
    dilated_labels = grey_dilation(label_array, footprint=footprint)
    props_after = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        dilated_labels.astype(np.int32, copy=False)
    )
    mean_area_after = (
        float(np.mean(props_after.area)) if props_after.label.size else 0.0
    )
    stats = DilationStats(
        slice_index=0,
        object_count=int(props_after.label.size),
        mean_area_before=mean_area_before,
        mean_area_after=mean_area_after,
    )
    return image, stats, dilated_labels.astype(np.float32)


@numpy_decorator(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(
    (
        "dilation_stats_3d",
        csv_materializer(
            fields=["object_count", "mean_volume_before", "mean_volume_after"],
            analysis_type="dilation_3d",
        ),
    ),
    ("dilated_labels", segmentation_mask_rois()),
)
def dilate_objects_3d(
    image: np.ndarray,
    labels: np.ndarray,
    structuring_element_shape: StructuringElement | str = StructuringElement.BALL,
    structuring_element_size: int = 1,
) -> tuple[np.ndarray, DilationStats3D, np.ndarray]:
    """Dilate 3D labels with CellProfiler's higher-label-overwrites policy."""
    from scipy.ndimage import grey_dilation
    from skimage.measure import regionprops

    label_array = object_label_dense_array(labels, dtype=np.int32)
    props_before = regionprops(label_array)
    volumes_before = [prop.area for prop in props_before]
    mean_volume_before = float(np.mean(volumes_before)) if volumes_before else 0.0
    footprint = build_structuring_element(
        structuring_element_shape,
        structuring_element_size,
    )
    dilated_labels = grey_dilation(label_array, footprint=footprint)
    props_after = regionprops(dilated_labels)
    volumes_after = [prop.area for prop in props_after]
    mean_volume_after = float(np.mean(volumes_after)) if volumes_after else 0.0
    stats = DilationStats3D(
        object_count=len(props_after),
        mean_volume_before=mean_volume_before,
        mean_volume_after=mean_volume_after,
    )
    return image, stats, dilated_labels.astype(np.float32)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("labels", segmentation_mask_rois()))
def fill_objects(
    image: np.ndarray,
    labels: np.ndarray,
    mode: FillMode = FillMode.HOLES,
    diameter: float = 64.0,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill object holes or replace objects with convex hull labels."""
    from skimage.morphology import remove_small_holes

    label_array = object_label_dense_array(labels, dtype=np.int32)
    if label_array.max() == 0:
        return image, label_array.copy()

    mode = coerce_cellprofiler_enum(FillMode, mode)
    filled_labels = np.zeros_like(label_array)
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        label_array
    )

    if mode == FillMode.HOLES:
        max_hole_area = np.pi * (diameter / 2.0) ** 2
        for label_id in region_props.label:
            label_int = int(label_id)
            obj_mask = label_array == label_int
            filled_mask = remove_small_holes(
                obj_mask,
                area_threshold=int(max_hole_area),
                connectivity=1,
            )
            filled_labels[filled_mask] = label_int
    elif mode == FillMode.CONVEX_HULL:
        morphology = MorphologyBackendStrategy.for_callable(
            fill_objects,
            backend_provider=morphology_backend_provider,
        )
        for index, label_id in enumerate(region_props.label):
            label_int = int(label_id)
            obj_mask = label_array == label_int
            minr = int(region_props.bbox_min_y[index])
            minc = int(region_props.bbox_min_x[index])
            maxr = int(region_props.bbox_max_y[index])
            maxc = int(region_props.bbox_max_x[index])
            obj_crop = obj_mask[minr:maxr, minc:maxc]
            if obj_crop.sum() > 2:
                hull = morphology.convex_hull_image(obj_crop)
                filled_labels[minr:maxr, minc:maxc][hull] = label_int
            else:
                filled_labels[obj_mask] = label_int
    else:
        raise ValueError(
            f"Mode '{mode}' is not supported. "
            f"Available modes are: 'holes' and 'convex_hull'."
        )

    return image, filled_labels.astype(label_array.dtype)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "centroid_stats",
        csv_materializer(
            fields=["slice_index", "object_count"],
            analysis_type="centroid",
        ),
    ),
    ("centroid_labels", segmentation_mask_rois()),
)
def shrink_to_object_centers(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, CentroidStats, np.ndarray]:
    """Transform labeled objects into single-pixel centroid labels."""
    label_array = object_label_dense_array(labels, dtype=np.int32)
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        label_array
    )
    output_labels = np.zeros_like(label_array, dtype=np.int32)

    for index, label_id in enumerate(region_props.label):
        centroid_int = (
            int(round(float(region_props.centroid_y[index]))),
            int(round(float(region_props.centroid_x[index]))),
        )
        if all(
            0 <= centroid_int[axis] < label_array.shape[axis]
            for axis in range(len(centroid_int))
        ):
            output_labels[centroid_int] = int(label_id)

    return (
        image,
        CentroidStats(slice_index=0, object_count=int(region_props.label.size)),
        output_labels,
    )


@numpy_decorator(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(
    (
        "centroid_stats",
        csv_materializer(
            fields=["slice_index", "object_count"],
            analysis_type="centroid",
        ),
    ),
    ("centroid_labels", segmentation_mask_rois()),
)
def shrink_to_object_centers_3d(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, CentroidStats, np.ndarray]:
    """Transform 3D labeled objects into single-voxel centroid labels."""
    from skimage.measure import regionprops

    label_array = object_label_dense_array(labels, dtype=np.int32)
    props = regionprops(label_array)
    output_labels = np.zeros_like(label_array, dtype=np.int32)

    for region in props:
        centroid_int = tuple(int(round(coordinate)) for coordinate in region.centroid)
        if all(
            0 <= centroid_int[axis] < label_array.shape[axis]
            for axis in range(len(centroid_int))
        ):
            output_labels[centroid_int] = region.label

    return (
        image,
        CentroidStats(slice_index=0, object_count=len(props)),
        output_labels,
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "resize_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "original_height",
                "original_width",
                "new_height",
                "new_width",
                "object_count",
            ],
            analysis_type="resize_objects",
        ),
    ),
    "parent_child_relationship",
    ("resized_labels", segmentation_mask_rois()),
)
def resize_objects(
    image: np.ndarray,
    labels: np.ndarray,
    method: ResizeObjectsMethod = ResizeObjectsMethod.FACTOR,
    factor_x: float = 0.25,
    factor_y: float = 0.25,
    factor_z: float = 1.0,
    width: int = 100,
    height: int = 100,
    planes: int = 10,
) -> tuple[np.ndarray, ResizeObjectsStats, ParentChildRelationshipPayload, np.ndarray]:
    """Resize object labels by CellProfiler nearest-neighbor label semantics."""
    from scipy.ndimage import zoom

    labels = object_label_dense_array(labels, dtype=np.int32)
    original_shape = labels.shape
    method = coerce_cellprofiler_enum(ResizeObjectsMethod, method)

    if method == ResizeObjectsMethod.DIMENSIONS:
        target_size = resize_objects_target_shape(
            labels.shape,
            planes=planes,
            height=height,
            width=width,
        )
        zoom_factors = np.divide(np.multiply(1.0, target_size), labels.shape)
    else:
        zoom_factors = resize_objects_zoom_factors(
            labels.ndim,
            factor_z=factor_z,
            factor_y=factor_y,
            factor_x=factor_x,
        )
    resized_labels = zoom(labels, zoom_factors, order=0, mode="nearest").astype(
        np.int32
    )
    unique_labels = np.unique(resized_labels)
    object_count = len(unique_labels[unique_labels > 0])

    stats = ResizeObjectsStats(
        slice_index=0,
        original_height=original_shape[-2],
        original_width=original_shape[-1],
        new_height=resized_labels.shape[-2],
        new_width=resized_labels.shape[-1],
        object_count=object_count,
    )
    relationship = object_label_lineage_payload(labels, resized_labels)
    return image, stats, relationship, resized_labels


def resize_objects_target_shape(
    shape: tuple[int, ...],
    *,
    planes: int,
    height: int,
    width: int,
) -> tuple[int, ...]:
    spatial_shape = (planes, height, width) if len(shape) >= 3 else (height, width)
    return trailing_spatial_target_shape(shape, spatial_shape)


def resize_objects_zoom_factors(
    ndim: int,
    *,
    factor_z: float,
    factor_y: float,
    factor_x: float,
) -> tuple[float, ...]:
    spatial_factors = (
        (factor_z, factor_y, factor_x)
        if ndim >= 3
        else (factor_y, factor_x)
    )
    return trailing_spatial_factors(ndim, spatial_factors)


@numpy_decorator(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(
    (
        "resize_stats_3d",
        csv_materializer(
            fields=[
                "original_depth",
                "original_height",
                "original_width",
                "new_depth",
                "new_height",
                "new_width",
                "object_count",
            ],
            analysis_type="resize_objects_3d",
        ),
    ),
    "parent_child_relationship",
    ("resized_labels", segmentation_mask_rois()),
)
def resize_objects_3d(
    image: np.ndarray,
    labels: np.ndarray,
    method: ResizeObjectsMethod = ResizeObjectsMethod.FACTOR,
    factor_x: float = 0.25,
    factor_y: float = 0.25,
    factor_z: float = 0.25,
    width: int = 100,
    height: int = 100,
    planes: int = 10,
) -> tuple[np.ndarray, dict, ParentChildRelationshipPayload, np.ndarray]:
    """Resize 3D object labels by CellProfiler nearest-neighbor semantics."""
    from scipy.ndimage import zoom

    labels = object_label_dense_array(labels, dtype=np.int32)
    original_shape = labels.shape
    method = coerce_cellprofiler_enum(ResizeObjectsMethod, method)

    if method == ResizeObjectsMethod.DIMENSIONS:
        target_size = (planes, height, width)
        zoom_factors = np.divide(np.multiply(1.0, target_size), labels.shape)
    else:
        zoom_factors = (factor_z, factor_y, factor_x)
    resized_labels = zoom(labels, zoom_factors, order=0, mode="nearest").astype(
        np.int32
    )
    unique_labels = np.unique(resized_labels)
    object_count = len(unique_labels[unique_labels > 0])

    stats = {
        "original_depth": original_shape[0],
        "original_height": original_shape[1],
        "original_width": original_shape[2],
        "new_depth": resized_labels.shape[0],
        "new_height": resized_labels.shape[1],
        "new_width": resized_labels.shape[2],
        "object_count": object_count,
    }
    relationship = object_label_lineage_payload(labels, resized_labels)
    return image, stats, relationship, resized_labels


__all__ = [
    "CentrosomeNumpyMorphologyBackendStrategy",
    "CellProfilerDeclumpMethod",
    "CentroidStats",
    "CombineObjectsStats",
    "CombineObjectsStrategy",
    "DeclumpingMaximaGeometry",
    "ExpandDefinedPixelsStrategy",
    "ExpandInfiniteStrategy",
    "ExpandShrinkOperationStrategy",
    "FillHolesOption",
    "FillMode",
    "HolePredicate",
    "HoleRemovalDiameterPolicy",
    "MaskChoice",
    "MaskObjectsOutputLabels",
    "MaskObjectsPlaneOperation",
    "MaskObjectsPlaneResult",
    "MaskObjectsStats",
    "MorphOperation",
    "MorphOperationRequest",
    "MorphOperationStrategy",
    "MorphologyBackendStrategy",
    "NumbaNumpyMorphologyBackendStrategy",
    "NumpyMorphologyBackendStrategy",
    "RepeatMode",
    "RepeatModeStrategy",
    "ResizeObjectsMethod",
    "ResizeObjectsStats",
    "DilationStats",
    "DilationStats3D",
    "DistanceSplitOrMergeMergeMethodStrategy",
    "ErosionStats",
    "MergeObjectsStrategy",
    "MidpointPreservationPolicy",
    "ParentSplitOrMergeMergeMethodStrategy",
    "SplitObjectsStrategy",
    "SplitOrMergeConvexHull",
    "SplitOrMergeGuideImageFilter",
    "SplitOrMergeIntensityMethod",
    "SplitOrMergeMergeMethod",
    "SplitOrMergeMergeMethodStrategy",
    "SplitOrMergeOperation",
    "SplitOrMergeOperationStrategy",
    "SplitOrMergeOutputObjectType",
    "SplitOrMergeRequest",
    "SplitOrMergeStats",
    "apply_morph_operation",
    "closing",
    "dense_label_area_statistics",
    "dilate_image",
    "dilate_objects",
    "dilate_objects_3d",
    "erode_image",
    "erode_objects",
    "fill_objects",
    "filter_border_objects",
    "filter_border_objects_planewise",
    "filter_labels_above_maximum_diameter",
    "filter_labels_below_minimum_diameter",
    "filter_labels_by_area_numba",
    "filter_labels_by_diameter_range",
    "filter_labels_by_diameter_range_numba",
    "filter_physical_border_objects_numba",
    "manual_declumping_size",
    "mask_planes_for_labels",
    "mask_objects",
    "morph",
    "morphological_skeleton_2d",
    "morphological_skeleton_3d",
    "morphologicalskeleton",
    "opening",
    "positive_dense_label_count",
    "prepare_expand_or_shrink_objects",
    "resize_objects",
    "resize_objects_3d",
    "resize_objects_target_shape",
    "resize_objects_zoom_factors",
    "remove_holes",
    "remove_holes_3d",
    "SimpleDiskMidpointPreservationPolicy",
    "shrink_to_object_centers",
    "shrink_to_object_centers_3d",
    "split_or_merge_objects",
]
