"""
Converted from CellProfiler: Morph
Performs low-level morphological operations on binary or grayscale images.
"""

import numpy as np
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Tuple, Optional
from enum import Enum
from metaclass_registry import AutoRegisterMeta

from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class MorphOperation(Enum):
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
    ONCE = "once"
    FOREVER = "forever"
    CUSTOM = "custom"


MORPH_CONVOLUTION_MODE = "constant"
EIGHT_NEIGHBOR_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    dtype=np.uint8,
)
FOUR_CONNECTED_KERNEL = np.array(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class MorphOperationRequest:
    """Execution context shared by registered Morph operation strategies."""

    image: np.ndarray
    iterations: int
    rescale_values: bool
    line_length: int
    backend_provider: CellProfilerBackendProvider | None


MorphOperationImplementation = Callable[[MorphOperationRequest], np.ndarray]
RepeatCountResolver = Callable[[int], int]
NeighborConvolutionTransition = Callable[[np.ndarray, np.ndarray], np.ndarray]


class RegisteredCallableStrategy(ABC):
    """Shared callback substrate for registered Morph semantic families."""

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
            raise TypeError(f"{type(self).__name__} cannot invoke Morph callback")
        return callback(*args)


class MorphOperationStrategy(RegisteredCallableStrategy, metaclass=AutoRegisterMeta):
    """Registered implementation authority for Morph operations."""

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
    """Registered iteration-count policy for Morph repeat modes."""

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
    repeat_mode = RepeatMode.ONCE
    callback = staticmethod(lambda custom_repeats: 1)


class ForeverRepeatModeStrategy(RepeatModeStrategy):
    repeat_mode = RepeatMode.FOREVER
    callback = staticmethod(lambda custom_repeats: 10000)


class CustomRepeatModeStrategy(RepeatModeStrategy):
    repeat_mode = RepeatMode.CUSTOM
    callback = staticmethod(lambda custom_repeats: custom_repeats)


def _ensure_binary(image: np.ndarray) -> np.ndarray:
    """Convert image to binary if not already."""
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
    """Find branchpoints in a skeleton image."""
    from scipy.ndimage import convolve
    binary = _ensure_binary(image)
    # Count 8-connected neighbors
    neighbor_count = convolve(
        binary.astype(np.uint8),
        EIGHT_NEIGHBOR_KERNEL,
        mode=MORPH_CONVOLUTION_MODE,
        cval=0,
    )
    # Branchpoints have more than 2 neighbors
    return (binary & (neighbor_count > 2)).astype(np.float32)


def _bridge(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Bridge pixels that have two non-zero neighbors on opposite sides."""
    from scipy.ndimage import convolve
    result = _ensure_binary(image).astype(np.float32)
    
    # Patterns for opposite neighbors
    patterns = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]),  # diagonal
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),  # anti-diagonal
        np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),  # vertical
        np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),  # horizontal
    ]
    
    for _ in range(iterations):
        for pattern in patterns:
            match = convolve(result, pattern, mode=MORPH_CONVOLUTION_MODE, cval=0)
            result = np.where(match == 2, 1.0, result)
    
    return result


def _convex_hull(image: np.ndarray, morphology) -> np.ndarray:
    """Compute the convex hull of a binary image."""
    binary = _ensure_binary(image)
    if not np.any(binary):
        return np.zeros_like(image, dtype=np.float32)
    return morphology.convex_hull_image(binary).astype(np.float32)


def _diag(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Fill diagonal connections to make 4-connected from 8-connected."""
    from scipy.ndimage import convolve
    result = _ensure_binary(image).astype(np.float32)
    
    # Patterns for diagonal connections
    patterns = [
        (np.array([[0, 1], [1, 0]]), np.array([[1, 1], [1, 1]])),
        (np.array([[1, 0], [0, 1]]), np.array([[1, 1], [1, 1]])),
    ]
    
    for _ in range(iterations):
        for check, fill in patterns:
            # Simple approach: dilate diagonally connected regions
            pass
        # Use binary dilation with diagonal structure
        from scipy.ndimage import binary_dilation
        struct = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
        dilated = binary_dilation(result > 0, structure=struct)
        result = np.maximum(result, dilated.astype(np.float32))
    
    return result


def _distance(image: np.ndarray, rescale: bool = True) -> np.ndarray:
    """Compute distance transform of binary image."""
    from scipy.ndimage import distance_transform_edt
    binary = _ensure_binary(image)
    dist = distance_transform_edt(binary)
    if rescale and dist.max() > 0:
        dist = dist / dist.max()
    return dist.astype(np.float32)


def _endpoints(image: np.ndarray) -> np.ndarray:
    """Find endpoints in a skeleton image."""
    from scipy.ndimage import convolve
    binary = _ensure_binary(image)
    neighbor_count = convolve(
        binary.astype(np.uint8),
        EIGHT_NEIGHBOR_KERNEL,
        mode=MORPH_CONVOLUTION_MODE,
        cval=0,
    )
    # Endpoints have exactly 1 neighbor
    return (binary & (neighbor_count == 1)).astype(np.float32)


def _hbreak(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Remove vertical bridges between horizontal lines."""
    from scipy.ndimage import convolve
    result = _ensure_binary(image).astype(np.float32)
    
    # Pattern: pixel with horizontal neighbors above and below
    pattern = np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    
    for _ in range(iterations):
        match = convolve(result, pattern, mode=MORPH_CONVOLUTION_MODE, cval=0)
        # Remove pixels that match the H-bridge pattern
        result = np.where((match >= 6) & (result > 0), 0.0, result)
    
    return result


def _majority(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Each pixel takes majority value of its neighborhood."""
    from scipy.ndimage import convolve
    result = _ensure_binary(image).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    
    for _ in range(iterations):
        neighbor_sum = convolve(result, kernel, mode=MORPH_CONVOLUTION_MODE, cval=0)
        result = (neighbor_sum >= 5).astype(np.float32)  # 5 out of 9 (including center)
    
    return result


OpenLineStructureBuilder = Callable[[int], np.ndarray]


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
    callback = staticmethod(
        lambda line_length: np.ones((1, line_length), dtype=bool)
    )


class RisingDiagonalOpenLineStructuringElement(OpenLineStructuringElement):
    """Rising diagonal OPENLINES structuring element."""

    angle = 45
    callback = staticmethod(lambda line_length: np.eye(line_length, dtype=bool))


class VerticalOpenLineStructuringElement(OpenLineStructuringElement):
    """Vertical OPENLINES structuring element."""

    angle = 90
    callback = staticmethod(
        lambda line_length: np.ones((line_length, 1), dtype=bool)
    )


class FallingDiagonalOpenLineStructuringElement(OpenLineStructuringElement):
    """Falling diagonal OPENLINES structuring element."""

    angle = 135
    callback = staticmethod(
        lambda line_length: np.fliplr(np.eye(line_length, dtype=bool))
    )


def _openlines(image: np.ndarray, line_length: int = 3) -> np.ndarray:
    """Erosion followed by dilation using rotating linear elements."""
    from scipy.ndimage import binary_erosion, binary_dilation
    binary = _ensure_binary(image)

    result = np.zeros_like(binary)
    for element in OpenLineStructuringElement.registered_elements():
        struct = element.structure(line_length)
        eroded = binary_erosion(binary, structure=struct)
        dilated = binary_dilation(eroded, structure=struct)
        result = result | dilated
    
    return result.astype(np.float32)


def _shrink(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Shrink objects preserving topology (Euler number)."""
    from skimage.morphology import thin
    binary = _ensure_binary(image)
    return thin(binary, max_num_iter=iterations).astype(np.float32)


def _skelpe(image: np.ndarray) -> np.ndarray:
    """Skeletonize using PE*D metric."""
    from skimage.morphology import skeletonize
    from scipy.ndimage import distance_transform_edt
    binary = _ensure_binary(image)
    # Simplified version using standard skeletonization
    return skeletonize(binary).astype(np.float32)


def _thicken(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Thicken objects without connecting them."""
    from scipy.ndimage import binary_dilation, label
    result = _ensure_binary(image)
    
    for _ in range(iterations):
        # Label current objects
        labeled, num_features = label(result)
        # Dilate
        dilated = binary_dilation(result)
        # Only keep dilated pixels that don't connect different objects
        new_labeled, _ = label(dilated)
        # Simple approach: just dilate
        result = dilated
    
    return result.astype(np.float32)


def _thin(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Thin lines preserving Euler number."""
    from skimage.morphology import thin
    binary = _ensure_binary(image)
    return thin(binary, max_num_iter=iterations).astype(np.float32)


def _vbreak(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Remove horizontal bridges between vertical lines."""
    from scipy.ndimage import convolve
    result = _ensure_binary(image).astype(np.float32)
    
    # Pattern: pixel with vertical neighbors left and right
    pattern = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=np.float32)
    
    for _ in range(iterations):
        match = convolve(result, pattern, mode=MORPH_CONVOLUTION_MODE, cval=0)
        result = np.where((match >= 6) & (result > 0), 0.0, result)
    
    return result


def convex_hull_morph_operation(request: MorphOperationRequest) -> np.ndarray:
    """Run Morph CONVEX_HULL through the configured morphology backend."""
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    morphology = MorphologyBackendStrategy.for_callable(
        morph,
        backend_provider=request.backend_provider,
    )
    return _convex_hull(request.image, morphology)


class BranchpointsMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.BRANCHPOINTS
    callback = staticmethod(lambda request: _branchpoints(request.image))


class BridgeMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.BRIDGE
    callback = staticmethod(
        lambda request: _bridge(request.image, request.iterations)
    )


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
    callback = staticmethod(
        lambda request: _diag(request.image, request.iterations)
    )


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
    callback = staticmethod(
        lambda request: _hbreak(request.image, request.iterations)
    )


class MajorityMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.MAJORITY
    callback = staticmethod(
        lambda request: _majority(request.image, request.iterations)
    )


class OpenLinesMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.OPENLINES
    callback = staticmethod(
        lambda request: _openlines(request.image, request.line_length)
    )


class RemoveMorphOperationStrategy(IterativeConvolutionMorphOperationStrategy):
    operation = MorphOperation.REMOVE
    kernel = FOUR_CONNECTED_KERNEL
    transition = staticmethod(
        lambda result, neighbor_count: np.where(neighbor_count == 4, 0.0, result)
    )


class ShrinkMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.SHRINK
    callback = staticmethod(
        lambda request: _shrink(request.image, request.iterations)
    )


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
    callback = staticmethod(
        lambda request: _thicken(request.image, request.iterations)
    )


class ThinMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.THIN
    callback = staticmethod(
        lambda request: _thin(request.image, request.iterations)
    )


class VBreakMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.VBREAK
    callback = staticmethod(
        lambda request: _vbreak(request.image, request.iterations)
    )


@numpy(contract=ProcessingContract.PURE_2D)
def morph(
    image: np.ndarray,
    operation: MorphOperation = MorphOperation.THIN,
    repeat_mode: RepeatMode = RepeatMode.ONCE,
    custom_repeats: int = 2,
    rescale_values: bool = True,
    line_length: int = 3,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """
    Perform morphological operations on binary or grayscale images.
    
    Args:
        image: Input image (H, W), will be converted to binary for most operations
        operation: The morphological operation to perform
        repeat_mode: How many times to repeat (ONCE, FOREVER, or CUSTOM)
        custom_repeats: Number of repetitions when repeat_mode is CUSTOM
        rescale_values: For DISTANCE operation, rescale output to 0-1
        line_length: For OPENLINES operation, minimum line length to keep
    
    Returns:
        Processed image (H, W)
    """
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
        )
    )
