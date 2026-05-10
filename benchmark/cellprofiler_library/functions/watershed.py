"""
Converted from CellProfiler: Watershed
Original: watershed
"""

import logging
import numpy as np
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal, Tuple

from metaclass_registry import AutoRegisterMeta
from numba import njit

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
    adapt_structuring_element_rank,
    build_structuring_element,
)
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_semantics import relabel_dense_object_labels_consecutive
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois

PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


def watershed_profile_enabled() -> bool:
    return os.environ.get(PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def log_watershed_profile(label: str, seconds: float, **fields: object) -> None:
    if not watershed_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@dataclass
class WatershedStats:
    slice_index: int
    object_count: int
    mean_area: float


class WatershedMethod(str, Enum):
    """CellProfiler watershed surface source."""

    DISTANCE = "distance"
    INTENSITY = "intensity"
    MARKERS = "markers"


class WatershedDeclumpMethod(str, Enum):
    """CellProfiler watershed declumping priority family."""

    SHAPE = "shape"
    INTENSITY = "intensity"
    NONE = "none"


class WatershedSeedMethod(str, Enum):
    """CellProfiler watershed seed detector family."""

    LOCAL = "local"
    REGIONAL = "regional"
    CONNECTED_COMPONENTS = "connected_components"


class WatershedRuntimeFamily(str, Enum):
    """CellProfiler Watershed implementation family selected by module revision."""

    CELLPROFILER4 = "cellprofiler4"
    LIBRARY = "library"


class WatershedInputKeyword(str, Enum):
    """Special-input keyword names owned by the Watershed callable contract."""

    MARKERS = "markers"
    MASK = "mask"


@dataclass(frozen=True, slots=True)
class WatershedBasicDefaults:
    """CellProfiler Watershed defaults used when advanced settings are disabled."""

    seed_method: WatershedSeedMethod = WatershedSeedMethod.LOCAL
    max_seeds: int = -1
    min_distance: int = 1
    min_intensity: float = 0.0
    connectivity: int = 1
    compactness: float = 0.0
    watershed_line: bool = False
    gaussian_sigma: float = 0.0


CELLPROFILER_WATERSHED_BASIC_DEFAULTS = WatershedBasicDefaults()


def coerce_watershed_method(value: WatershedMethod | str | None) -> WatershedMethod:
    if value is None:
        return WatershedMethod.DISTANCE
    return _coerce_function_enum(WatershedMethod, value)


def coerce_watershed_declump_method(
    value: WatershedDeclumpMethod | str,
) -> WatershedDeclumpMethod:
    return _coerce_function_enum(WatershedDeclumpMethod, value)


def coerce_watershed_seed_method(value: WatershedSeedMethod | str) -> WatershedSeedMethod:
    return _coerce_function_enum(WatershedSeedMethod, value)


def coerce_watershed_runtime_family(
    value: WatershedRuntimeFamily | str,
) -> WatershedRuntimeFamily:
    return _coerce_function_enum(WatershedRuntimeFamily, value)


@dataclass(frozen=True, slots=True)
class WatershedInputs:
    image: np.ndarray
    binary: np.ndarray
    mask: np.ndarray
    markers: np.ndarray | None


@dataclass(frozen=True, slots=True)
class WatershedSegmentationSurface:
    watershed_input_image: np.ndarray
    seed_image: np.ndarray | None
    distance_image: np.ndarray | None
    markers: np.ndarray


@dataclass(frozen=True, slots=True)
class WatershedComputationImages:
    input_image: np.ndarray
    mask: np.ndarray | None
    markers: np.ndarray | None
    distance: np.ndarray | None


@dataclass(frozen=True, slots=True)
class WatershedParameters:
    method: WatershedMethod
    declump_method: WatershedDeclumpMethod
    seed_method: WatershedSeedMethod
    use_advanced_settings: bool
    max_seeds: int
    downsample: int
    min_distance: int
    min_intensity: float
    footprint: int
    connectivity: int
    compactness: float
    exclude_border: bool
    watershed_line: bool
    gaussian_sigma: float
    structuring_element: np.ndarray


class WatershedSeedStrategy(
    EnumKeyedStrategyMixin[WatershedSeedMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Build seed labels for non-marker Watershed modes."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedSeedMethod | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def markers(
        self,
        seed_image: np.ndarray,
        inputs: WatershedInputs,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        """Return marker labels for the supplied seed image."""


class LocalWatershedSeedStrategy(WatershedSeedStrategy):
    strategy_key = WatershedSeedMethod.LOCAL

    def markers(
        self,
        seed_image: np.ndarray,
        inputs: WatershedInputs,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        from skimage.feature import peak_local_max
        from scipy.ndimage import label as ndi_label
        from skimage.morphology import binary_dilation

        coords = peak_local_max(
            seed_image,
            min_distance=parameters.min_distance,
            footprint=np.ones((parameters.footprint,) * inputs.image.ndim),
            threshold_rel=parameters.min_intensity,
            num_peaks=(
                parameters.max_seeds
                if parameters.max_seeds != -1
                else np.inf
            ),
            exclude_border=False,
        )
        seeds = np.zeros(seed_image.shape, dtype=bool)
        seeds[tuple(coords.T)] = True
        seeds = binary_dilation(seeds, parameters.structuring_element)
        markers, _count = ndi_label(seeds)
        return markers.astype(np.int32, copy=False)


class RegionalWatershedSeedStrategy(WatershedSeedStrategy):
    strategy_key = WatershedSeedMethod.REGIONAL

    def markers(
        self,
        seed_image: np.ndarray,
        inputs: WatershedInputs,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        del inputs
        import mahotas
        from scipy.ndimage import label as ndi_label
        from skimage.morphology import binary_dilation

        maxima_footprint = np.ones((parameters.footprint,) * seed_image.ndim)
        seeds = mahotas.regmax(seed_image, maxima_footprint)
        seeds = binary_dilation(seeds, parameters.structuring_element)
        markers, _count = ndi_label(seeds)
        return markers.astype(np.int32, copy=False)


class ConnectedComponentsWatershedSeedStrategy(WatershedSeedStrategy):
    strategy_key = WatershedSeedMethod.CONNECTED_COMPONENTS

    def markers(
        self,
        seed_image: np.ndarray,
        inputs: WatershedInputs,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        del seed_image, parameters
        from scipy.ndimage import label as ndi_label

        markers, _count = ndi_label(inputs.mask)
        return markers.astype(np.int32, copy=False)


class WatershedDeclumpStrategy(
    EnumKeyedStrategyMixin[WatershedDeclumpMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Build the watershed priority surface for one declumping family."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedDeclumpMethod | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def priority_image(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
    ) -> np.ndarray:
        """Return the skimage watershed input image."""


class ShapeWatershedDeclumpStrategy(WatershedDeclumpStrategy):
    strategy_key = WatershedDeclumpMethod.SHAPE

    def priority_image(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
    ) -> np.ndarray:
        del inputs
        if computation.distance is None:
            raise ValueError("Shape declumping requires a distance image.")
        watershed_input = -computation.distance
        return watershed_input - watershed_input.min()


class IntensityWatershedDeclumpStrategy(WatershedDeclumpStrategy):
    strategy_key = WatershedDeclumpMethod.INTENSITY

    def priority_image(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
    ) -> np.ndarray:
        del computation
        return 1.0 - inputs.image


class NoneWatershedDeclumpStrategy(WatershedDeclumpStrategy):
    strategy_key = WatershedDeclumpMethod.NONE

    def priority_image(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
    ) -> np.ndarray:
        del inputs, computation
        raise ValueError("No-declump watershed should label the binary input directly.")


class WatershedMethodStrategy(
    EnumKeyedStrategyMixin[WatershedMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Build seed labels for one CellProfiler watershed seed source."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedMethod | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def seed_image(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
    ) -> np.ndarray | None:
        """Return the image used to derive watershed seeds."""

    def markers(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
        seed_strategy: WatershedSeedStrategy,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        seed_image = self.seed_image(inputs, computation)
        if seed_image is None:
            raise ValueError(f"{type(self).__name__} did not provide a seed image.")
        return seed_strategy.markers(seed_image, inputs, parameters)


class DistanceWatershedMethodStrategy(WatershedMethodStrategy):
    strategy_key = WatershedMethod.DISTANCE

    def seed_image(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
    ) -> np.ndarray:
        del inputs
        if computation.distance is None:
            raise ValueError("Distance watershed seed method requires a distance image.")
        return computation.distance


class IntensityWatershedMethodStrategy(WatershedMethodStrategy):
    strategy_key = WatershedMethod.INTENSITY

    def seed_image(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
    ) -> np.ndarray:
        del computation
        return inputs.image


class MarkerWatershedMethodStrategy(WatershedMethodStrategy):
    strategy_key = WatershedMethod.MARKERS

    def seed_image(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
    ) -> np.ndarray | None:
        del inputs, computation
        return None

    def markers(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
        seed_strategy: WatershedSeedStrategy,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        del computation, seed_strategy
        if inputs.markers is None:
            raise ValueError("Watershed marker mode requires marker labels.")
        markers = object_label_dense_array(inputs.markers, dtype=np.int32)
        if markers.shape != inputs.image.shape:
            raise ValueError(
                "Watershed marker shape must match the input image shape; "
                f"got markers={markers.shape!r}, image={inputs.image.shape!r}."
            )
        from skimage.morphology import dilation

        return dilation(markers, footprint=parameters.structuring_element).astype(
            np.int32,
            copy=False,
        )


class WatershedRuntimeStrategy(
    EnumKeyedStrategyMixin[WatershedRuntimeFamily],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Execute one nominal CellProfiler Watershed implementation family."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedRuntimeFamily | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def labels(
        self,
        image: np.ndarray,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        """Return object labels using one CellProfiler runtime family."""

    def segmentation_surface(
        self,
        inputs: WatershedInputs,
        parameters: WatershedParameters,
    ) -> WatershedSegmentationSurface:
        """Build the CellProfiler watershed priority image and marker labels."""
        from scipy.ndimage import distance_transform_edt
        from skimage.filters import gaussian

        needs_distance = (
            parameters.declump_method is WatershedDeclumpMethod.SHAPE
            or parameters.method is WatershedMethod.DISTANCE
        )
        distance = (
            distance_transform_edt(
                gaussian(inputs.image, sigma=parameters.gaussian_sigma)
            )
            if needs_distance
            else None
        )
        computation = WatershedComputationImages(
            input_image=inputs.image,
            mask=inputs.mask,
            markers=inputs.markers,
            distance=distance,
        )
        method_strategy = WatershedMethodStrategy.for_enum_member(parameters.method)
        seed_image = method_strategy.seed_image(inputs, computation)
        markers = method_strategy.markers(
            inputs,
            computation,
            WatershedSeedStrategy.for_enum_member(parameters.seed_method),
            parameters,
        )
        return WatershedSegmentationSurface(
            watershed_input_image=WatershedDeclumpStrategy.for_enum_member(
                parameters.declump_method
            ).priority_image(inputs, computation),
            seed_image=seed_image,
            distance_image=distance,
            markers=markers,
        )


class CellProfiler4InitialWatershedStrategy(
    EnumKeyedStrategyMixin[WatershedMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Build the CP4 module's initial watershed labels before advanced refinement."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedMethod | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def labels(
        self,
        image: np.ndarray,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the initial labels and the image-domain mask source."""


class CellProfiler4DistanceMarkerBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Build CellProfiler 4 distance-watershed markers through a typed backend."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def distance_markers(
        self,
        distance: np.ndarray,
        peak_footprint: np.ndarray,
        seed_connectivity: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Return seed markers, marker count, and the regional-maxima mask."""


class MahotasCellProfiler4DistanceMarkerBackendStrategy(
    CellProfiler4DistanceMarkerBackendStrategy
):
    """Reference CellProfiler 4 marker backend."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE)
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def distance_markers(
        self,
        distance: np.ndarray,
        peak_footprint: np.ndarray,
        seed_connectivity: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        import mahotas

        peaks = mahotas.regmax(distance, peak_footprint)
        seed_markers, marker_count = mahotas.label(peaks, seed_connectivity)
        return seed_markers, int(marker_count), peaks


class NumbaCellProfiler4DistanceMarkerBackendStrategy(
    CellProfiler4DistanceMarkerBackendStrategy
):
    """Exact numba backend for CellProfiler 4 regional-maxima markers."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA)
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def distance_markers(
        self,
        distance: np.ndarray,
        peak_footprint: np.ndarray,
        seed_connectivity: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        import mahotas
        import scipy.ndimage

        distance_array = np.ascontiguousarray(distance)
        peak_footprint_array = np.asarray(peak_footprint, dtype=bool)
        if distance_array.ndim != 3 or peak_footprint_array.ndim != 3:
            return MahotasCellProfiler4DistanceMarkerBackendStrategy().distance_markers(
                distance_array,
                peak_footprint_array,
                seed_connectivity,
            )
        local_maxima = (
            distance_array
            == scipy.ndimage.maximum_filter(
                distance_array,
                footprint=peak_footprint_array,
                mode="constant",
                cval=0,
            )
        ) & (distance_array > 0)
        peaks = _cellprofiler4_regional_maxima_from_candidates_3d_numba(
            distance_array,
            np.ascontiguousarray(local_maxima),
            _footprint_offsets_3d(peak_footprint_array),
        )
        seed_markers, marker_count = mahotas.label(peaks, seed_connectivity)
        return seed_markers, int(marker_count), peaks

    def prepare_backend(self) -> None:
        distance = np.zeros((8, 16, 16), dtype=np.uint8)
        distance[2:6, 4:12, 4:12] = 1
        distance[3:5, 6:10, 6:10] = 2
        footprint = np.ones((4, 4, 4), dtype=bool)
        connectivity = np.ones((4, 4, 4), dtype=bool)
        self.distance_markers(distance, footprint, connectivity)


def _footprint_offsets_3d(footprint: np.ndarray) -> np.ndarray:
    center = np.asarray(footprint.shape, dtype=np.int64) // 2
    offsets = np.argwhere(footprint).astype(np.int64) - center
    return np.ascontiguousarray(offsets[np.any(offsets != 0, axis=1)])


@njit(cache=True)
def _cellprofiler4_regional_maxima_from_candidates_3d_numba(
    image: np.ndarray,
    candidates: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    z_size, y_size, x_size = image.shape
    voxel_count = image.size
    visited = np.zeros(voxel_count, np.uint8)
    output = np.zeros(image.shape, np.bool_)
    stack = np.empty(voxel_count, np.int64)
    component = np.empty(voxel_count, np.int64)
    plane_size = y_size * x_size

    for start_index in range(voxel_count):
        z_start = start_index // plane_size
        start_remainder = start_index - z_start * plane_size
        y_start = start_remainder // x_size
        x_start = start_remainder - y_start * x_size
        if not candidates[z_start, y_start, x_start] or visited[start_index]:
            continue

        value = image[z_start, y_start, x_start]
        visited[start_index] = 1
        stack_size = 1
        stack[0] = start_index
        component_size = 0
        has_higher_neighbor = False

        while stack_size > 0:
            stack_size -= 1
            index = stack[stack_size]
            component[component_size] = index
            component_size += 1
            z_index = index // plane_size
            remainder = index - z_index * plane_size
            y_index = remainder // x_size
            x_index = remainder - y_index * x_size

            for offset_index in range(offsets.shape[0]):
                z_neighbor = z_index + offsets[offset_index, 0]
                y_neighbor = y_index + offsets[offset_index, 1]
                x_neighbor = x_index + offsets[offset_index, 2]
                if (
                    z_neighbor < 0
                    or y_neighbor < 0
                    or x_neighbor < 0
                    or z_neighbor >= z_size
                    or y_neighbor >= y_size
                    or x_neighbor >= x_size
                ):
                    continue

                neighbor_value = image[z_neighbor, y_neighbor, x_neighbor]
                if neighbor_value > value:
                    has_higher_neighbor = True
                elif neighbor_value == value:
                    neighbor_index = (
                        z_neighbor * plane_size
                        + y_neighbor * x_size
                        + x_neighbor
                    )
                    if not visited[neighbor_index]:
                        visited[neighbor_index] = 1
                        stack[stack_size] = neighbor_index
                        stack_size += 1

        if not has_higher_neighbor:
            for component_index in range(component_size):
                index = component[component_index]
                z_index = index // plane_size
                remainder = index - z_index * plane_size
                y_index = remainder // x_size
                x_index = remainder - y_index * x_size
                output[z_index, y_index, x_index] = True

    return output


class CellProfiler4DistanceInitialWatershedStrategy(
    CellProfiler4InitialWatershedStrategy
):
    strategy_key = WatershedMethod.DISTANCE

    def labels(
        self,
        image: np.ndarray,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        del markers, mask
        import mahotas
        import scipy.ndimage
        import skimage.filters
        import skimage.transform

        total_started_at = time.perf_counter()
        input_shape = image.shape
        factor = parameters.downsample
        x_data = image
        if factor > 1:
            phase_started_at = time.perf_counter()
            factors = (1, factor, factor) if image.ndim > 2 else (factor, factor)
            x_data = skimage.transform.downscale_local_mean(x_data, factors)
            log_watershed_profile(
                "watershed_cp4_distance_downsample",
                time.perf_counter() - phase_started_at,
                factor=factor,
                ndim=image.ndim,
            )

        phase_started_at = time.perf_counter()
        threshold = skimage.filters.threshold_otsu(x_data)
        x_data = x_data > threshold
        log_watershed_profile(
            "watershed_cp4_distance_threshold",
            time.perf_counter() - phase_started_at,
            factor=factor,
        )
        phase_started_at = time.perf_counter()
        distance = scipy.ndimage.distance_transform_edt(x_data)
        log_watershed_profile(
            "watershed_cp4_distance_edt",
            time.perf_counter() - phase_started_at,
            factor=factor,
        )
        phase_started_at = time.perf_counter()
        distance = mahotas.stretch(distance)
        surface = distance.max() - distance
        log_watershed_profile(
            "watershed_cp4_distance_surface",
            time.perf_counter() - phase_started_at,
            factor=factor,
        )
        phase_started_at = time.perf_counter()
        peak_footprint = np.ones((parameters.footprint,) * image.ndim)
        seed_connectivity = np.ones((16,) * image.ndim)
        seed_markers, marker_count, _peaks = (
            CellProfiler4DistanceMarkerBackendStrategy.for_memory_type(
                MemoryType.NUMPY
            ).distance_markers(
                distance,
                peak_footprint,
                seed_connectivity,
            )
        )
        log_watershed_profile(
            "watershed_cp4_distance_markers",
            time.perf_counter() - phase_started_at,
            factor=factor,
            seeds=marker_count,
        )
        phase_started_at = time.perf_counter()
        y_data = mahotas.cwatershed(surface, seed_markers) * x_data
        log_watershed_profile(
            "watershed_cp4_distance_cwatershed",
            time.perf_counter() - phase_started_at,
            factor=factor,
        )

        if factor > 1:
            phase_started_at = time.perf_counter()
            y_data = skimage.transform.resize(
                y_data,
                input_shape,
                mode="edge",
                order=0,
                preserve_range=True,
            )
            y_data = np.rint(y_data).astype(np.uint16)
            x_data = image > threshold
            log_watershed_profile(
                "watershed_cp4_distance_upsample",
                time.perf_counter() - phase_started_at,
                factor=factor,
            )
        log_watershed_profile(
            "watershed_cp4_distance_initial_total",
            time.perf_counter() - total_started_at,
            factor=factor,
        )
        return y_data, x_data


class CellProfiler4MarkerInitialWatershedStrategy(CellProfiler4InitialWatershedStrategy):
    strategy_key = WatershedMethod.MARKERS

    def labels(
        self,
        image: np.ndarray,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        if markers is None:
            raise ValueError("CellProfiler 4 marker watershed requires markers.")
        if parameters.compactness != 0.0:
            raise NotImplementedError(
                "CellProfiler 4 marker watershed compactness requires legacy "
                "compact watershed semantics."
            )
        if parameters.watershed_line:
            raise NotImplementedError(
                "CellProfiler 4 marker watershed lines require legacy watershed-line "
                "semantics."
            )

        image_array = np.asarray(image)
        mask_array = np.asarray(mask, dtype=bool) if mask is not None else image_array != 0
        y_data = cellprofiler_legacy_watershed(
            image=image,
            markers=object_label_dense_array(markers, dtype=np.int32),
            mask=mask_array,
            connectivity=parameters.connectivity,
        )
        return y_data, image


class CellProfiler4WatershedRuntimeStrategy(WatershedRuntimeStrategy):
    """CellProfiler 4.2 module-level Watershed semantics."""

    strategy_key = WatershedRuntimeFamily.CELLPROFILER4

    def labels(
        self,
        image: np.ndarray,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        import mahotas
        import scipy.ndimage
        import skimage.feature
        import skimage.filters
        import skimage.measure
        import skimage.morphology
        import skimage.segmentation

        phase_started_at = time.perf_counter()
        y_data, x_data = CellProfiler4InitialWatershedStrategy.for_enum_member(
            parameters.method
        ).labels(
            image,
            markers,
            mask,
            parameters,
        )
        log_watershed_profile(
            "watershed_cp4_initial",
            time.perf_counter() - phase_started_at,
            method=parameters.method.value,
        )

        if parameters.use_advanced_settings:
            if parameters.structuring_element.ndim != image.ndim:
                raise ValueError(
                    "Watershed structuring element dimensionality must match the image; "
                    f"got structuring element ndim={parameters.structuring_element.ndim} "
                    f"for image ndim={image.ndim}."
                )

            phase_started_at = time.perf_counter()
            peak_image = scipy.ndimage.distance_transform_edt(y_data > 0)
            log_watershed_profile(
                "watershed_cp4_peak_distance",
                time.perf_counter() - phase_started_at,
                method=parameters.method.value,
            )
            if parameters.declump_method is WatershedDeclumpMethod.SHAPE:
                watershed_image = -peak_image
                watershed_image -= watershed_image.min()
            else:
                watershed_image = 1.0 - image.astype(float, copy=False)

            phase_started_at = time.perf_counter()
            watershed_image = skimage.filters.gaussian(
                watershed_image,
                sigma=parameters.gaussian_sigma,
            )
            log_watershed_profile(
                "watershed_cp4_gaussian",
                time.perf_counter() - phase_started_at,
                method=parameters.method.value,
            )
            phase_started_at = time.perf_counter()
            seed_coords = skimage.feature.peak_local_max(
                peak_image,
                min_distance=parameters.min_distance,
                threshold_rel=parameters.min_intensity,
                exclude_border=parameters.exclude_border,
                num_peaks=(
                    parameters.max_seeds
                    if parameters.max_seeds != -1
                    else np.inf
                ),
            )
            log_watershed_profile(
                "watershed_cp4_peak_local_max",
                time.perf_counter() - phase_started_at,
                method=parameters.method.value,
                seeds=len(seed_coords),
            )
            phase_started_at = time.perf_counter()
            seeds = np.zeros_like(peak_image, dtype=bool)
            seeds[tuple(seed_coords.T)] = True
            seeds = skimage.morphology.binary_dilation(
                seeds,
                parameters.structuring_element,
            )
            number_objects = skimage.measure.label(y_data, return_num=True)[1]
            seeds_dtype = (
                np.uint16
                if number_objects < np.iinfo(np.uint16).max
                else np.uint32
            )
            seeds = scipy.ndimage.label(seeds)[0]
            advanced_markers = np.zeros_like(seeds, dtype=seeds_dtype)
            advanced_markers[seeds > 0] = -seeds[seeds > 0]
            log_watershed_profile(
                "watershed_cp4_markers",
                time.perf_counter() - phase_started_at,
                method=parameters.method.value,
                objects=number_objects,
            )
            phase_started_at = time.perf_counter()
            watershed_boundaries = skimage.segmentation.watershed(
                image=watershed_image,
                markers=advanced_markers,
                mask=x_data != 0,
                connectivity=parameters.connectivity,
            )
            log_watershed_profile(
                "watershed_cp4_segmentation",
                time.perf_counter() - phase_started_at,
                method=parameters.method.value,
            )
            phase_started_at = time.perf_counter()
            y_data = watershed_boundaries.copy()
            zeros = np.where(y_data == 0)
            y_data += np.abs(np.min(y_data)) + 1
            y_data[zeros] = 0
            log_watershed_profile(
                "watershed_cp4_relabel_prepare",
                time.perf_counter() - phase_started_at,
                method=parameters.method.value,
            )

        phase_started_at = time.perf_counter()
        labels = skimage.measure.label(y_data).astype(np.int32, copy=False)
        log_watershed_profile(
            "watershed_cp4_final_label",
            time.perf_counter() - phase_started_at,
            method=parameters.method.value,
        )
        return labels


class LibraryWatershedRuntimeStrategy(WatershedRuntimeStrategy):
    """CellProfiler library-style Watershed semantics."""

    strategy_key = WatershedRuntimeFamily.LIBRARY

    def labels(
        self,
        image: np.ndarray,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        from skimage.segmentation import watershed as skimage_watershed
        from skimage.segmentation import clear_border
        from skimage.filters import gaussian
        from skimage.transform import downscale_local_mean, resize
        from scipy.ndimage import distance_transform_edt, label as ndi_label

        binary = image.astype(bool, copy=False)
        mask_array = binary.astype(bool) if mask is None else np.asarray(mask) > 0
        if mask_array.shape != image.shape:
            raise ValueError(
                "Watershed mask shape must match the input image shape; "
                f"got mask={mask_array.shape!r}, image={image.shape!r}."
            )

        input_shape = binary.shape
        working_image = binary
        working_mask: np.ndarray | None = mask_array
        working_markers = (
            None
            if markers is None
            else object_label_dense_array(markers, dtype=np.int32)
        )
        if parameters.downsample > 1:
            factors = (
                (1, parameters.downsample, parameters.downsample)
                if binary.ndim > 2
                else (parameters.downsample, parameters.downsample)
            )
            working_image = downscale_local_mean(binary.astype(np.float32), factors)
            if working_mask is not None:
                working_mask = (
                    downscale_local_mean(working_mask.astype(np.float32), factors) > 0
                )
            if working_markers is not None:
                working_markers = downscale_local_mean(
                    working_markers.astype(np.float32),
                    factors,
                )

        working_mask_array = working_image != 0 if working_mask is None else working_mask
        working_inputs = WatershedInputs(
            working_image,
            working_image,
            working_mask_array,
            working_markers,
        )
        if parameters.declump_method is WatershedDeclumpMethod.NONE:
            labels, _count = ndi_label(working_image)
            if working_mask is not None:
                labels = np.where(working_mask_array, labels, 0)
        else:
            needs_distance = (
                parameters.declump_method is WatershedDeclumpMethod.SHAPE
                or parameters.method is WatershedMethod.DISTANCE
            )
            distance = (
                distance_transform_edt(
                    gaussian(working_image, sigma=parameters.gaussian_sigma)
                )
                if needs_distance
                else None
            )
            computation = WatershedComputationImages(
                input_image=working_image,
                mask=working_mask_array,
                markers=working_markers,
                distance=distance,
            )
            watershed_input_image = WatershedDeclumpStrategy.for_enum_member(
                parameters.declump_method
            ).priority_image(working_inputs, computation)
            markers_array = WatershedMethodStrategy.for_enum_member(
                parameters.method
            ).markers(
                working_inputs,
                computation,
                WatershedSeedStrategy.for_enum_member(parameters.seed_method),
                parameters,
            )
            labels = skimage_watershed(
                watershed_input_image,
                markers=markers_array,
                mask=working_mask_array,
                connectivity=parameters.connectivity,
                compactness=parameters.compactness,
                watershed_line=parameters.watershed_line,
            )

        if parameters.downsample > 1:
            labels = resize(labels, input_shape, mode="edge", order=0, preserve_range=True)
            labels = np.rint(labels).astype(np.uint16)

        if parameters.exclude_border:
            labels = clear_border(labels)

        return relabel_dense_object_labels_consecutive(labels, dtype=np.int32)


def cellprofiler_legacy_watershed(
    image: np.ndarray,
    *,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int | np.ndarray = 1,
    backend_provider: CellProfilerBackendProvider | None = None,
) -> np.ndarray:
    """Run CellProfiler 4.2/skimage 0.18 watershed semantics.

    Newer scikit-image raises a queued neighbor's priority to at least the
    source pixel priority. CellProfiler 4.2 did not, and that changes basin
    ownership for tied/near-tied object boundaries by a few pixels.
    """
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed as _cellprofiler_legacy_watershed,
    )

    return _cellprofiler_legacy_watershed(
        image,
        markers=markers,
        mask=mask,
        connectivity=connectivity,
        backend_provider=backend_provider,
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("markers", "mask")
@special_outputs(
    ("watershed_stats", csv_materializer(fields=["slice_index", "object_count", "mean_area"])),
    ("labels", segmentation_mask_rois())
)
def watershed(
    image: np.ndarray,
    markers: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    watershed_method: Literal["distance", "intensity", "markers"] = "distance",
    declump_method: Literal["shape", "intensity"] = "shape",
    seed_method: Literal["local", "regional"] = "local",
    use_advanced_settings: bool = True,
    max_seeds: int = -1,
    downsample: int = 1,
    min_distance: int = 1,
    min_intensity: float = 0.0,
    footprint: int = 8,
    connectivity: int = 1,
    compactness: float = 0.0,
    exclude_border: bool = False,
    watershed_line: bool = False,
    gaussian_sigma: float = 0.0,
    structuring_element: Literal[
        "ball", "cube", "diamond", "disk", "octahedron", "square", "star"
    ] = "disk",
    structuring_element_size: int = 1,
    runtime_family: WatershedRuntimeFamily | str = WatershedRuntimeFamily.LIBRARY,
) -> Tuple[np.ndarray, WatershedStats, np.ndarray]:
    """
    Apply watershed segmentation to separate touching objects.
    
    Args:
        image: Input binary or grayscale image (H, W)
        watershed_method: Method for watershed - 'distance' uses distance transform,
                         'intensity' uses intensity image, 'markers' uses marker image
        declump_method: Method for declumping - 'shape' or 'intensity'
        seed_method: Seed detection method - 'local' for local maxima, 'regional' for regional
        max_seeds: Maximum number of seeds (-1 for unlimited)
        downsample: Downsampling factor for speed
        min_distance: Minimum distance between seeds
        min_intensity: Minimum intensity for seeds
        footprint: Footprint size for local maxima detection
        connectivity: Connectivity for watershed (1 or 2)
        compactness: Compactness parameter for watershed
        exclude_border: Whether to exclude objects touching border
        watershed_line: Whether to draw watershed lines between objects
        gaussian_sigma: Sigma for Gaussian smoothing (0 for no smoothing)
        structuring_element: Shape of structuring element for morphological operations
        structuring_element_size: Size of structuring element
    
    Returns:
        Tuple of (original image, watershed statistics, labeled image)
    """
    from skimage.measure import regionprops
    method = coerce_watershed_method(watershed_method)
    resolved_declump_method = coerce_watershed_declump_method(declump_method)
    if not use_advanced_settings:
        defaults = CELLPROFILER_WATERSHED_BASIC_DEFAULTS
        seed_method = defaults.seed_method
        max_seeds = defaults.max_seeds
        min_distance = defaults.min_distance
        min_intensity = defaults.min_intensity
        connectivity = defaults.connectivity
        compactness = defaults.compactness
        watershed_line = defaults.watershed_line
        gaussian_sigma = defaults.gaussian_sigma
    
    if not np.array_equal(image, image.astype(bool)):
        raise ValueError("Watershed expects a thresholded image as input.")
    structuring_element_array = adapt_structuring_element_rank(
        build_structuring_element(
            _coerce_function_enum(StructuringElement, structuring_element),
            structuring_element_size,
        ),
        image.ndim,
    )
    if structuring_element_array.ndim != image.ndim:
        raise ValueError(
            "Watershed structuring element dimensionality must match the image; "
            f"got structuring element ndim={structuring_element_array.ndim} for "
            f"image ndim={image.ndim}."
        )

    labels = WatershedRuntimeStrategy.for_enum_member(
        coerce_watershed_runtime_family(runtime_family)
    ).labels(
        image,
        markers,
        mask,
        WatershedParameters(
            method=method,
            declump_method=resolved_declump_method,
            seed_method=coerce_watershed_seed_method(seed_method),
            use_advanced_settings=use_advanced_settings,
            max_seeds=max_seeds,
            downsample=downsample,
            min_distance=min_distance,
            min_intensity=min_intensity,
            footprint=footprint,
            connectivity=connectivity,
            compactness=compactness,
            exclude_border=exclude_border,
            watershed_line=watershed_line,
            gaussian_sigma=gaussian_sigma,
            structuring_element=structuring_element_array,
        ),
    )

    props = regionprops(labels)
    object_count = len(props)
    mean_area = np.mean([p.area for p in props]) if props else 0.0

    stats = WatershedStats(
        slice_index=0,
        object_count=object_count,
        mean_area=float(mean_area)
    )

    return image, stats, labels.astype(np.int32)
