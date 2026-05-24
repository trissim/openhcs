"""Watershed backend strategies for CellProfiler-compatible processing."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.image_shapes import trailing_spatial_factors
from openhcs.core.runtime_semantics import DenseObjectLabelConsecutiveRelabelingStrategy
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
    adapt_structuring_element_rank,
    build_structuring_element,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois

PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
NDIMAGE_CONSTANT_MODE = "constant"
WATERSHED_STRATEGY_REGISTRY_KEY = "strategy_label"
logger = logging.getLogger(__name__)


def watershed_xy_downsample_factors(ndim: int, factor: int) -> tuple[int, ...]:
    """Return rank-matched factors that downsample only the XY image domain."""
    return tuple(
        int(value)
        for value in trailing_spatial_factors(ndim, (factor, factor))
    )


def watershed_connected_components(labels_like: np.ndarray) -> np.ndarray:
    """Label connected components over skimage-supported trailing spatial axes."""
    import skimage.measure

    labels_array = np.asarray(labels_like)
    spatial_rank = min(labels_array.ndim, 3)
    if labels_array.ndim == spatial_rank:
        return skimage.measure.label(labels_array).astype(np.int32, copy=False)
    output = np.zeros(labels_array.shape, dtype=np.int32)
    leading_shape = labels_array.shape[: labels_array.ndim - spatial_rank]
    for leading_index in np.ndindex(leading_shape):
        output[leading_index] = skimage.measure.label(labels_array[leading_index])
    return output


def watershed_regionprops_stats(labels: np.ndarray) -> tuple[int, float]:
    """Return object count and mean area over skimage-supported spatial labels."""
    from skimage.measure import regionprops

    labels_array = np.asarray(labels)
    spatial_rank = min(labels_array.ndim, 3)
    if labels_array.ndim == spatial_rank:
        props = regionprops(labels_array)
    else:
        props = []
        leading_shape = labels_array.shape[: labels_array.ndim - spatial_rank]
        for leading_index in np.ndindex(leading_shape):
            props.extend(regionprops(labels_array[leading_index]))
    return (
        len(props),
        float(np.mean([prop.area for prop in props]) if props else 0.0),
    )


def watershed_profile_enabled() -> bool:
    return os.environ.get(PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def log_watershed_profile(label: str, seconds: float, **fields: object) -> None:
    if not watershed_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@dataclass(frozen=True)
class WatershedProfiler:
    """Bound profiler for CellProfiler watershed execution phases."""

    def record(self, label: str, started_at: float, **fields: object) -> None:
        log_watershed_profile(
            label,
            time.perf_counter() - started_at,
            **fields,
        )

    def record_factor(self, label: str, started_at: float, factor: int, **fields: object) -> None:
        self.record(label, started_at, factor=factor, **fields)

    def record_method(
        self,
        label: str,
        started_at: float,
        method: WatershedMethod,
        **fields: object,
    ) -> None:
        self.record(label, started_at, method=method.value, **fields)


@dataclass(frozen=True, slots=True)
class WatershedFactorProfiler:
    """Profiler projection for a watershed phase family sharing one factor."""

    profiler: WatershedProfiler
    factor: int
    ndim: int

    def record(
        self,
        label: str,
        started_at: float,
        **fields: object,
    ) -> None:
        self.profiler.record(label, started_at, factor=self.factor, **fields)

    def record_downsample(self, label: str, started_at: float) -> None:
        self.record(label, started_at, ndim=self.ndim)


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


@dataclass
class WatershedStats:
    """Watershed object-count measurement row."""

    slice_index: int
    object_count: int
    mean_area: float


def coerce_watershed_method(value: WatershedMethod | str | None) -> WatershedMethod:
    if value is None:
        return WatershedMethod.DISTANCE
    return coerce_cellprofiler_enum(WatershedMethod, value)


def coerce_watershed_declump_method(
    value: WatershedDeclumpMethod | str,
) -> WatershedDeclumpMethod:
    return coerce_cellprofiler_enum(WatershedDeclumpMethod, value)


def coerce_watershed_seed_method(value: WatershedSeedMethod | str) -> WatershedSeedMethod:
    return coerce_cellprofiler_enum(WatershedSeedMethod, value)


def coerce_watershed_runtime_family(
    value: WatershedRuntimeFamily | str,
) -> WatershedRuntimeFamily:
    return coerce_cellprofiler_enum(WatershedRuntimeFamily, value)


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

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
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

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
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

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
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

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
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

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
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

        profiler = WatershedProfiler()
        factor_profiler = WatershedFactorProfiler(
            profiler=profiler,
            factor=parameters.downsample,
            ndim=image.ndim,
        )
        total_started_at = time.perf_counter()
        input_shape = image.shape
        factor = parameters.downsample
        x_data = image
        if factor > 1:
            phase_started_at = time.perf_counter()
            factors = watershed_xy_downsample_factors(image.ndim, factor)
            x_data = skimage.transform.downscale_local_mean(x_data, factors)
            factor_profiler.record_downsample(
                "watershed_cp4_distance_downsample",
                phase_started_at,
            )

        phase_started_at = time.perf_counter()
        threshold = skimage.filters.threshold_otsu(x_data)
        x_data = x_data > threshold
        factor_profiler.record(
            "watershed_cp4_distance_threshold",
            phase_started_at,
        )
        phase_started_at = time.perf_counter()
        distance = scipy.ndimage.distance_transform_edt(x_data)
        factor_profiler.record(
            "watershed_cp4_distance_edt",
            phase_started_at,
        )
        phase_started_at = time.perf_counter()
        distance = mahotas.stretch(distance)
        surface = distance.max() - distance
        factor_profiler.record(
            "watershed_cp4_distance_surface",
            phase_started_at,
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
        factor_profiler.record(
            "watershed_cp4_distance_markers",
            phase_started_at,
            seeds=marker_count,
        )
        phase_started_at = time.perf_counter()
        y_data = mahotas.cwatershed(surface, seed_markers) * x_data
        factor_profiler.record(
            "watershed_cp4_distance_cwatershed",
            phase_started_at,
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
            factor_profiler.record(
                "watershed_cp4_distance_upsample",
                phase_started_at,
            )
        factor_profiler.record(
            "watershed_cp4_distance_initial_total",
            total_started_at,
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

        import skimage.segmentation

        image_array = np.asarray(image)
        markers_array = np.asarray(markers)
        mask_array = None if mask is None else np.asarray(mask, dtype=bool)
        y_data = skimage.segmentation.watershed(
            image=image_array,
            markers=markers_array,
            mask=mask_array,
            connectivity=parameters.connectivity,
            compactness=parameters.compactness,
            watershed_line=parameters.watershed_line,
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
        import scipy.ndimage
        import skimage.feature
        import skimage.filters
        import skimage.measure
        import skimage.morphology
        import skimage.segmentation

        profiler = WatershedProfiler()
        phase_started_at = time.perf_counter()
        y_data, x_data = CellProfiler4InitialWatershedStrategy.for_enum_member(
            parameters.method
        ).labels(
            image,
            markers,
            mask,
            parameters,
        )
        profiler.record_method(
            "watershed_cp4_initial",
            phase_started_at,
            parameters.method,
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
            profiler.record_method(
                "watershed_cp4_peak_distance",
                phase_started_at,
                parameters.method,
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
            profiler.record_method(
                "watershed_cp4_gaussian",
                phase_started_at,
                parameters.method,
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
            profiler.record_method(
                "watershed_cp4_peak_local_max",
                phase_started_at,
                parameters.method,
                seeds=len(seed_coords),
            )
            phase_started_at = time.perf_counter()
            seeds = np.zeros_like(peak_image, dtype=bool)
            seeds[tuple(seed_coords.T)] = True
            seeds = skimage.morphology.binary_dilation(
                seeds,
                parameters.structuring_element,
            )
            number_objects = int(np.max(watershed_connected_components(y_data)))
            seeds_dtype = (
                np.uint16
                if number_objects < np.iinfo(np.uint16).max
                else np.uint32
            )
            seeds = scipy.ndimage.label(seeds)[0]
            advanced_markers = np.zeros_like(seeds, dtype=seeds_dtype)
            advanced_markers[seeds > 0] = -seeds[seeds > 0]
            profiler.record_method(
                "watershed_cp4_markers",
                phase_started_at,
                parameters.method,
                objects=number_objects,
            )
            phase_started_at = time.perf_counter()
            watershed_boundaries = skimage.segmentation.watershed(
                image=watershed_image,
                markers=advanced_markers,
                mask=x_data != 0,
                connectivity=parameters.connectivity,
            )
            profiler.record_method(
                "watershed_cp4_segmentation",
                phase_started_at,
                parameters.method,
            )
            phase_started_at = time.perf_counter()
            y_data = watershed_boundaries.copy()
            zeros = np.where(y_data == 0)
            y_data += np.abs(np.min(y_data)) + 1
            y_data[zeros] = 0
            profiler.record_method(
                "watershed_cp4_relabel_prepare",
                phase_started_at,
                parameters.method,
            )

        phase_started_at = time.perf_counter()
        labels = watershed_connected_components(y_data)
        profiler.record_method(
            "watershed_cp4_final_label",
            phase_started_at,
            parameters.method,
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
            factors = watershed_xy_downsample_factors(binary.ndim, parameters.downsample)
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

        return DenseObjectLabelConsecutiveRelabelingStrategy.for_labels(labels).relabel(
            labels,
            dtype=np.int32,
        )


class LegacyWatershedBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Legacy watershed operations keyed by OpenHCS memory type."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True
    prefer_fast: ClassVar[bool]

    def validated_request(
        self,
        image: np.ndarray,
        *,
        markers: np.ndarray,
        mask: np.ndarray,
        connectivity: int | np.ndarray = 1,
    ) -> "LegacyWatershedRequest":
        """Build a validated legacy watershed request for this backend family."""
        return LegacyWatershedRequest.from_inputs(
            image,
            markers=markers,
            mask=mask,
            connectivity=connectivity,
            prefer_fast=self.prefer_fast,
        )


class NumpyLegacyWatershedBackendStrategy(LegacyWatershedBackendStrategy):
    """NumPy-memory reference legacy watershed backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False
    prefer_fast = False


class NumbaNumpyLegacyWatershedBackendStrategy(LegacyWatershedBackendStrategy):
    """NumPy-memory legacy watershed backend with required Numba acceleration."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True
    prefer_fast = True

    def prepare_backend(self) -> None:
        image = np.arange(9, dtype=np.float64).reshape((3, 3))
        markers = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]], dtype=np.int32)
        mask = np.ones(markers.shape, dtype=np.bool_)
        self.validated_request(
            image,
            markers=markers,
            mask=mask,
            connectivity=1,
        ).execute()

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

    backend_key = CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE)
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

    backend_key = CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA)
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
                mode=NDIMAGE_CONSTANT_MODE,
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


def cellprofiler_legacy_watershed(
    image: np.ndarray,
    *,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int | np.ndarray = 1,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Run CellProfiler 4.2/skimage 0.18 watershed semantics."""
    return LegacyWatershedBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    ).validated_request(
        image,
        markers=markers,
        mask=mask,
        connectivity=connectivity,
    ).execute()


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("markers", "mask")
@special_outputs(
    (
        "watershed_stats",
        csv_materializer(
            fields=["slice_index", "object_count", "mean_area"],
            analysis_type="watershed",
        ),
    ),
    ("labels", segmentation_mask_rois()),
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
) -> tuple[np.ndarray, WatershedStats, np.ndarray]:
    """Apply watershed segmentation using the registered CP runtime family."""
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
            coerce_cellprofiler_enum(StructuringElement, structuring_element),
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

    object_count, mean_area = watershed_regionprops_stats(labels)
    stats = WatershedStats(
        slice_index=0,
        object_count=object_count,
        mean_area=mean_area,
    )

    return image, stats, labels.astype(np.int32)


@dataclass(frozen=True, slots=True)
class LegacyWatershedRequest:
    """Validated legacy watershed inputs shared across whole-volume and plane paths."""

    image: np.ndarray
    markers: np.ndarray
    mask: np.ndarray
    connectivity: int | np.ndarray
    prefer_fast: bool

    @classmethod
    def from_inputs(
        cls,
        image: np.ndarray,
        *,
        markers: np.ndarray,
        mask: np.ndarray,
        connectivity: int | np.ndarray,
        prefer_fast: bool,
    ) -> "LegacyWatershedRequest":
        image_array = np.asarray(image, dtype=np.float64)
        mask_array = np.asarray(mask, dtype=bool)
        marker_array = np.asarray(markers) * mask_array
        if marker_array.shape != image_array.shape:
            raise ValueError("markers must have the same shape as image")
        if mask_array.shape != image_array.shape:
            raise ValueError("mask must have the same shape as image")
        return cls(
            image=image_array,
            markers=marker_array,
            mask=mask_array,
            connectivity=connectivity,
            prefer_fast=prefer_fast,
        )

    def plane(self, plane_index: int) -> "LegacyWatershedRequest":
        image_planes = self.image.reshape((-1, *self.image.shape[-2:]))
        marker_planes = self.markers.reshape((-1, *self.markers.shape[-2:]))
        mask_planes = self.mask.reshape((-1, *self.mask.shape[-2:]))
        return type(self)(
            image=image_planes[plane_index],
            markers=marker_planes[plane_index],
            mask=mask_planes[plane_index],
            connectivity=self.connectivity,
            prefer_fast=self.prefer_fast,
        )

    def execute(self) -> np.ndarray:
        """Execute the validated legacy watershed request."""
        from skimage.morphology._util import (
            _offsets_to_raveled_neighbors,
            _validate_connectivity,
        )
        from skimage.util import crop

        if self.is_planewise:
            return self.execute_planewise()

        connectivity_array, offset = _validate_connectivity(
            self.image.ndim,
            self.connectivity,
            None,
        )
        pad_width = [(int(width), int(width)) for width in offset]
        padded_image = np.pad(self.image, pad_width, mode=NDIMAGE_CONSTANT_MODE)
        padded_mask = np.pad(
            self.mask.astype(np.bool_, copy=False),
            pad_width,
            mode=NDIMAGE_CONSTANT_MODE,
        ).ravel()
        output = np.pad(
            self.markers.astype(np.int32, copy=False),
            pad_width,
            mode=NDIMAGE_CONSTANT_MODE,
        )
        state = LegacyWatershedRaveledState(
            image_flat=padded_image.ravel(),
            mask_flat=padded_mask,
            output_flat=output.ravel(),
            neighbor_offsets=_offsets_to_raveled_neighbors(
                padded_image.shape,
                connectivity_array,
                center=offset,
            ).astype(np.int64, copy=False),
            marker_locations=np.flatnonzero(output).astype(np.int64, copy=False),
        )
        if self.prefer_fast:
            state.execute_numba()
        else:
            state.execute_python()
        return crop(output, pad_width, copy=True)

    @property
    def is_planewise(self) -> bool:
        if self.image.ndim <= 2:
            return False
        if np.isscalar(self.connectivity):
            return False
        return np.asarray(self.connectivity).ndim == 2

    def execute_planewise(self) -> np.ndarray:
        output = np.empty(self.markers.shape, dtype=np.int32)
        output_planes = output.reshape((-1, *output.shape[-2:]))
        for plane_index in range(output_planes.shape[0]):
            output_planes[plane_index] = self.plane(plane_index).execute()
        return output


@dataclass(frozen=True, slots=True)
class LegacyWatershedRaveledState:
    """Raveled legacy watershed buffers and neighborhood provenance."""

    image_flat: np.ndarray
    mask_flat: np.ndarray
    output_flat: np.ndarray
    neighbor_offsets: np.ndarray
    marker_locations: np.ndarray

    def execute_python(self) -> None:
        heap = LegacyWatershedPythonHeap()
        for marker_location in self.marker_locations:
            location = int(marker_location)
            heap.push(float(self.image_flat[location]), 0, location, location)

        age = 1
        while heap:
            _value, _entry_age, index, source = heap.pop()
            label = int(self.output_flat[index])
            if label == 0:
                label = int(self.output_flat[source])
            for offset_value in self.neighbor_offsets:
                neighbor_index = int(index + offset_value)
                if (
                    not self.mask_flat[neighbor_index]
                    or self.output_flat[neighbor_index] != 0
                ):
                    continue
                self.output_flat[neighbor_index] = label
                age += 1
                heap.push(
                    float(self.image_flat[neighbor_index]),
                    age,
                    neighbor_index,
                    source,
                )

    def execute_numba(self) -> None:
        _legacy_watershed_raveled_numba(
            self.image_flat,
            self.mask_flat,
            self.output_flat,
            self.neighbor_offsets,
            self.marker_locations,
        )


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


@dataclass(slots=True)
class LegacyWatershedPythonHeap:
    """Priority heap for the Python legacy watershed reference path."""

    values: list[float]
    ages: list[int]
    indexes: list[int]
    sources: list[int]

    def __init__(self) -> None:
        self.values = []
        self.ages = []
        self.indexes = []
        self.sources = []

    def __bool__(self) -> bool:
        return bool(self.values)

    @staticmethod
    def item_less(
        left_value: float,
        left_age: int,
        right_value: float,
        right_age: int,
    ) -> bool:
        if left_value != right_value:
            return left_value < right_value
        return left_age < right_age

    def swap(self, left: int, right: int) -> None:
        self.values[left], self.values[right] = self.values[right], self.values[left]
        self.ages[left], self.ages[right] = self.ages[right], self.ages[left]
        self.indexes[left], self.indexes[right] = (
            self.indexes[right],
            self.indexes[left],
        )
        self.sources[left], self.sources[right] = (
            self.sources[right],
            self.sources[left],
        )

    def push(self, value: float, age: int, index: int, source: int) -> None:
        self.values.append(value)
        self.ages.append(age)
        self.indexes.append(index)
        self.sources.append(source)
        position = len(self.values) - 1
        while position > 0:
            parent = (position - 1) // 2
            if not self.item_less(
                self.values[position],
                self.ages[position],
                self.values[parent],
                self.ages[parent],
            ):
                break
            self.swap(position, parent)
            position = parent

    def pop(self) -> tuple[float, int, int, int]:
        value = self.values[0]
        age = self.ages[0]
        index = self.indexes[0]
        source = self.sources[0]
        last = len(self.values) - 1
        if last == 0:
            self.values.pop()
            self.ages.pop()
            self.indexes.pop()
            self.sources.pop()
            return value, age, index, source

        self.values[0] = self.values.pop()
        self.ages[0] = self.ages.pop()
        self.indexes[0] = self.indexes.pop()
        self.sources[0] = self.sources.pop()
        size = len(self.values)
        position = 0
        while True:
            left = position * 2 + 1
            right = left + 1
            if left >= size:
                break
            smallest = left
            if right < size and self.item_less(
                self.values[right],
                self.ages[right],
                self.values[left],
                self.ages[left],
            ):
                smallest = right
            if not self.item_less(
                self.values[smallest],
                self.ages[smallest],
                self.values[position],
                self.ages[position],
            ):
                break
            self.swap(position, smallest)
            position = smallest
        return value, age, index, source


@njit(cache=True)
def _heap_item_less(
    left_value: float,
    left_age: int,
    left_index: int,
    left_source: int,
    right_value: float,
    right_age: int,
    right_index: int,
    right_source: int,
) -> bool:
    if left_value != right_value:
        return left_value < right_value
    return left_age < right_age


@njit(cache=True)
def _heap_swap(
    heap_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    left: int,
    right: int,
) -> None:
    values, ages, indexes, sources = heap_arrays
    value = values[left]
    age = ages[left]
    index = indexes[left]
    source = sources[left]
    values[left] = values[right]
    ages[left] = ages[right]
    indexes[left] = indexes[right]
    sources[left] = sources[right]
    values[right] = value
    ages[right] = age
    indexes[right] = index
    sources[right] = source


@njit(cache=True)
def _heap_push(
    heap_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    size: int,
    value: float,
    age: int,
    index: int,
    source: int,
) -> int:
    values, ages, indexes, sources = heap_arrays
    values[size] = value
    ages[size] = age
    indexes[size] = index
    sources[size] = source
    size += 1
    position = size - 1
    while position > 0:
        parent = (position - 1) // 2
        if not _heap_item_less(
            values[position],
            ages[position],
            indexes[position],
            sources[position],
            values[parent],
            ages[parent],
            indexes[parent],
            sources[parent],
        ):
            break
        _heap_swap(heap_arrays, position, parent)
        position = parent
    return size


@njit(cache=True)
def _heap_pop(
    heap_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    size: int,
) -> tuple[int, float, int, int, int]:
    values, ages, indexes, sources = heap_arrays
    value = values[0]
    age = ages[0]
    index = indexes[0]
    source = sources[0]
    size -= 1
    if size > 0:
        values[0] = values[size]
        ages[0] = ages[size]
        indexes[0] = indexes[size]
        sources[0] = sources[size]
        position = 0
        while True:
            left = position * 2 + 1
            right = left + 1
            if left >= size:
                break
            smallest = left
            if right < size and _heap_item_less(
                values[right],
                ages[right],
                indexes[right],
                sources[right],
                values[left],
                ages[left],
                indexes[left],
                sources[left],
            ):
                smallest = right
            if not _heap_item_less(
                values[smallest],
                ages[smallest],
                indexes[smallest],
                sources[smallest],
                values[position],
                ages[position],
                indexes[position],
                sources[position],
            ):
                break
            _heap_swap(heap_arrays, position, smallest)
            position = smallest
    return size, value, age, index, source


@njit(cache=True)
def _legacy_watershed_raveled_numba(
    image_flat: np.ndarray,
    mask_flat: np.ndarray,
    output_flat: np.ndarray,
    neighbor_offsets: np.ndarray,
    marker_locations: np.ndarray,
) -> None:
    capacity = output_flat.size
    heap_values = np.empty(capacity, dtype=np.float64)
    heap_ages = np.empty(capacity, dtype=np.int64)
    heap_indexes = np.empty(capacity, dtype=np.int64)
    heap_sources = np.empty(capacity, dtype=np.int64)
    heap_arrays = (heap_values, heap_ages, heap_indexes, heap_sources)
    heap_size = 0

    for marker_location in marker_locations:
        location = int(marker_location)
        heap_size = _heap_push(
            heap_arrays,
            heap_size,
            float(image_flat[location]),
            0,
            location,
            location,
        )

    age = 1
    while heap_size > 0:
        heap_size, _value, _entry_age, index, source = _heap_pop(
            heap_arrays,
            heap_size,
        )
        label = int(output_flat[index])
        if label == 0:
            label = int(output_flat[source])
        for offset_value in neighbor_offsets:
            neighbor_index = int(index + offset_value)
            if (not mask_flat[neighbor_index]) or output_flat[neighbor_index] != 0:
                continue
            output_flat[neighbor_index] = label
            age += 1
            heap_size = _heap_push(
                heap_arrays,
                heap_size,
                float(image_flat[neighbor_index]),
                age,
                neighbor_index,
                source,
            )


__all__ = public_names_from_objects(
    CellProfiler4DistanceMarkerBackendStrategy,
    LegacyWatershedBackendStrategy,
    MahotasCellProfiler4DistanceMarkerBackendStrategy,
    NumbaCellProfiler4DistanceMarkerBackendStrategy,
    NumbaNumpyLegacyWatershedBackendStrategy,
    NumpyLegacyWatershedBackendStrategy,
    WatershedStats,
    cellprofiler_legacy_watershed,
    watershed,
)
