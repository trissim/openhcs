"""Intensity-distribution backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import logging
import os
import time

import numpy as np
import scipy.sparse
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.cellprofiler.shape import shape_measurement_backend
from openhcs.processing.backends.cellprofiler.secondary import (
    SecondaryPropagationBackendStrategy,
    secondary_propagation_backend,
)


_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)
_RADIAL_LABEL_GEOMETRY_CACHE_LIMIT = 16
_RADIAL_LABEL_GEOMETRY_CACHE: OrderedDict[
    "RadialLabelGeometryCacheKey",
    "RadialLabelGeometry",
] = OrderedDict()


def _runtime_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_profile(label: str, seconds: float, **fields: object) -> None:
    if not _runtime_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@dataclass(frozen=True, slots=True)
class RadialDistributionArrays:
    """Dense per-object radial intensity-distribution arrays."""

    fraction_at_distance: np.ndarray
    mean_pixel_fraction: np.ndarray
    radial_cv_by_bin: np.ndarray
    object_has_pixels: np.ndarray
    n_bins: int


@dataclass(frozen=True, slots=True)
class RadialCenterDistanceFields:
    """CellProfiler center-propagation fields for radial measurements."""

    d_from_center: np.ndarray
    center_labels: np.ndarray
    centers_i: np.ndarray
    centers_j: np.ndarray


@dataclass(frozen=True, slots=True)
class RadialCenterPropagationRequest:
    """Nearest-center propagation for radial intensity-distribution geometry."""

    center_labels: np.ndarray
    colors: np.ndarray
    propagation_backend: SecondaryPropagationBackendStrategy

    def fields(self) -> tuple[np.ndarray, np.ndarray]:
        """Return center distances and propagated center labels by color mask."""
        d_from_center = np.zeros(self.center_labels.shape, dtype=float)
        propagated_center_labels = np.zeros(self.center_labels.shape, dtype=int)
        max_color = int(np.max(self.colors)) if self.colors.size else 0
        seed_labels = np.asarray(self.center_labels, dtype=np.int32)
        for color in range(1, max_color + 1):
            mask = self.colors == color
            seed_mask = mask & (seed_labels > 0)
            if not np.any(seed_mask):
                continue
            propagation = self.propagation_backend.propagate_result(
                np.zeros(seed_labels.shape, dtype=float),
                seed_labels,
                mask,
                1,
            )
            propagated_labels = propagation.labels
            distances = propagation.distances
            d_from_center[mask] = distances[mask]
            propagated_center_labels[mask] = propagated_labels[mask]
        return d_from_center, propagated_center_labels


@dataclass(frozen=True, slots=True)
class RadialLabelGeometryCacheKey:
    """Content identity for radial geometry derived only from object labels."""

    dtype: str
    shape: tuple[int, ...]
    digest: bytes

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> "RadialLabelGeometryCacheKey":
        label_array = np.ascontiguousarray(labels, dtype=np.int32)
        digest = hashlib.blake2b(label_array.view(np.uint8), digest_size=16).digest()
        return cls(
            dtype=str(label_array.dtype),
            shape=tuple(int(value) for value in label_array.shape),
            digest=digest,
        )


@dataclass(frozen=True, slots=True)
class RadialLabelGeometry:
    """Label-only radial geometry shared by all intensity images for a label plane."""

    d_to_edge: np.ndarray
    center_fields: RadialCenterDistanceFields


@dataclass(frozen=True, slots=True)
class IntensityDistributionPlaneInputs:
    """2D image/label plane consumed by intensity-distribution measurement."""

    image: np.ndarray
    labels: object

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        label_array = object_label_dense_array(self.labels, dtype=np.int32)
        image_array = np.asarray(self.image)
        if image_array.ndim == 3:
            image_2d = image_array[0]
            labels_2d = label_array[0] if label_array.ndim == 3 else label_array
            return image_2d, labels_2d
        return image_array, label_array


class RadialDistributionBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Radial-distribution operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True
    center_propagation_backend_provider = CellProfilerBackendProvider.NUMBA

    def center_propagation_backend(self) -> SecondaryPropagationBackendStrategy:
        """Return the propagation backend used for center-distance geometry."""
        return secondary_propagation_backend(
            backend_provider=self.center_propagation_backend_provider,
        )

    @abstractmethod
    def measure(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        d_to_edge: np.ndarray,
        d_from_center: np.ndarray,
        center_labels: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays for one image plane."""

    @abstractmethod
    def measure_from_centers(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        d_to_edge: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays while computing center distances."""

    @abstractmethod
    def measure_self_centered(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays using each object's own center."""

    def center_distance_fields(
        self,
        labels: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
    ) -> RadialCenterDistanceFields:
        """Return CP-compatible propagated center labels and distances."""
        labels_array = np.asarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        object_labels = np.arange(1, object_count + 1, dtype=np.int32)
        center_labels = np.zeros(labels_array.shape, dtype=int)
        centers_i_int = np.asarray(centers_i, dtype=int)
        centers_j_int = np.asarray(centers_j, dtype=int)
        valid_center_bounds = (
            (centers_i_int >= 0)
            & (centers_i_int < labels_array.shape[0])
            & (centers_j_int >= 0)
            & (centers_j_int < labels_array.shape[1])
        )
        sampled_center_labels = np.zeros(object_count, dtype=np.int32)
        sampled_center_labels[valid_center_bounds] = labels_array[
            centers_i_int[valid_center_bounds],
            centers_j_int[valid_center_bounds],
        ]
        valid_centers = valid_center_bounds & (sampled_center_labels == object_labels)
        if np.any(valid_centers):
            center_labels[
                centers_i_int[valid_centers],
                centers_j_int[valid_centers],
            ] = labels_array[
                centers_i_int[valid_centers],
                centers_j_int[valid_centers],
            ]
        shape_backend = shape_measurement_backend(
            backend_provider=CellProfilerBackendProvider.CENTROSOME,
        )
        phase_started_at = time.perf_counter()
        colors = shape_backend.color_labels(labels_array)
        _log_profile(
            "idist_center_color_labels",
            time.perf_counter() - phase_started_at,
            objects=object_count,
            colors=int(np.max(colors)) if colors.size else 0,
        )
        phase_started_at = time.perf_counter()
        d_from_center, propagated_center_labels = RadialCenterPropagationRequest(
            center_labels=center_labels,
            colors=colors,
            propagation_backend=self.center_propagation_backend(),
        ).fields()
        _log_profile(
            "idist_center_propagate",
            time.perf_counter() - phase_started_at,
            objects=object_count,
            colors=int(np.max(colors)) if colors.size else 0,
        )

        return RadialCenterDistanceFields(
            d_from_center=d_from_center,
            center_labels=propagated_center_labels,
            centers_i=np.asarray(centers_i, dtype=np.float64),
            centers_j=np.asarray(centers_j, dtype=np.float64),
        )

    def label_geometry(self, labels: np.ndarray) -> RadialLabelGeometry:
        """Return CP-compatible radial geometry derived only from object labels."""
        labels_array = np.ascontiguousarray(labels, dtype=np.int32)
        cache_key = RadialLabelGeometryCacheKey.from_labels(labels_array)
        cached = _RADIAL_LABEL_GEOMETRY_CACHE.get(cache_key)
        if cached is not None:
            _RADIAL_LABEL_GEOMETRY_CACHE.move_to_end(cache_key)
            _log_profile(
                "idist_label_geometry_cache_hit",
                0.0,
                objects=int(labels_array.max()) if labels_array.size else 0,
            )
            return cached

        object_count = int(labels_array.max()) if labels_array.size else 0
        shape_backend = shape_measurement_backend(
            backend_provider=CellProfilerBackendProvider.CENTROSOME,
        )
        phase_started_at = time.perf_counter()
        d_to_edge = shape_backend.distance_to_edge(labels_array)
        _log_profile(
            "idist_distance_to_edge",
            time.perf_counter() - phase_started_at,
            objects=object_count,
        )
        phase_started_at = time.perf_counter()
        centers_i, centers_j = shape_backend.maximum_position_of_labels(
            d_to_edge,
            labels_array,
            np.arange(1, object_count + 1, dtype=np.int32),
        )
        _log_profile(
            "idist_maximum_position",
            time.perf_counter() - phase_started_at,
            objects=object_count,
        )
        geometry = RadialLabelGeometry(
            d_to_edge=d_to_edge,
            center_fields=self.center_distance_fields(
                labels_array,
                centers_i,
                centers_j,
            ),
        )
        _RADIAL_LABEL_GEOMETRY_CACHE[cache_key] = geometry
        _RADIAL_LABEL_GEOMETRY_CACHE.move_to_end(cache_key)
        while len(_RADIAL_LABEL_GEOMETRY_CACHE) > _RADIAL_LABEL_GEOMETRY_CACHE_LIMIT:
            _RADIAL_LABEL_GEOMETRY_CACHE.popitem(last=False)
        return geometry


class NativeNumpyRadialDistributionBackendStrategy(
    RadialDistributionBackendStrategy
):
    """CellProfiler-native NumPy radial-distribution backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def measure(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        d_to_edge: np.ndarray,
        d_from_center: np.ndarray,
        center_labels: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        image_array = np.asarray(image, dtype=np.float64)
        labels_array = np.asarray(labels, dtype=np.int32)
        d_to_edge_array = np.asarray(d_to_edge, dtype=np.float64)
        d_from_center_array = np.asarray(d_from_center, dtype=np.float64)
        center_labels_array = np.asarray(center_labels, dtype=np.int32)
        centers_i_array = np.asarray(centers_i, dtype=np.float64)
        centers_j_array = np.asarray(centers_j, dtype=np.float64)

        if image_array.ndim != 2 or labels_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler radial intensity distribution currently supports "
                f"2-D NumPy planes, got image {image_array.shape!r} and labels "
                f"{labels_array.shape!r}."
            )
        if labels_array.shape != image_array.shape:
            raise ValueError(
                "Radial distribution labels must match the image shape; got "
                f"labels {labels_array.shape!r} for image {image_array.shape!r}."
            )
        if bin_count <= 0:
            raise ValueError(f"bin_count must be positive, got {bin_count!r}.")

        object_count = int(labels_array.max()) if labels_array.size else 0
        n_bins = int(bin_count) if wants_scaled else int(bin_count) + 1
        if object_count <= 0:
            return RadialDistributionArrays(
                fraction_at_distance=np.zeros((0, int(bin_count) + 1), dtype=float),
                mean_pixel_fraction=np.zeros((0, int(bin_count) + 1), dtype=float),
                radial_cv_by_bin=np.zeros((n_bins, 0), dtype=float),
                object_has_pixels=np.zeros(0, dtype=bool),
                n_bins=n_bins,
            )

        good_mask = center_labels_array > 0
        normalized_distance = np.zeros(labels_array.shape, dtype=float)
        if wants_scaled:
            total_distance = d_from_center_array + d_to_edge_array
            normalized_distance[good_mask] = d_from_center_array[good_mask] / (
                total_distance[good_mask] + 0.001
            )
        else:
            normalized_distance[good_mask] = (
                d_from_center_array[good_mask] / maximum_radius
            )

        good_labels = labels_array[good_mask]
        bin_indexes = (normalized_distance * int(bin_count)).astype(int)
        bin_indexes[bin_indexes > int(bin_count)] = int(bin_count)
        labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

        histogram = scipy.sparse.coo_matrix(
            (image_array[good_mask], labels_and_bins),
            (object_count, int(bin_count) + 1),
        ).toarray()
        sum_by_object = np.sum(histogram, 1)
        fraction_at_distance = histogram / np.dstack(
            [sum_by_object] * (int(bin_count) + 1)
        )[0]

        ngood_pixels = int(np.sum(good_mask))
        number_at_distance = scipy.sparse.coo_matrix(
            (np.ones(ngood_pixels), labels_and_bins),
            (object_count, int(bin_count) + 1),
        ).toarray()
        sum_by_object = np.sum(number_at_distance, 1)
        fraction_at_bin = number_at_distance / np.dstack(
            [sum_by_object] * (int(bin_count) + 1)
        )[0]
        mean_pixel_fraction = fraction_at_distance / (
            fraction_at_bin + np.finfo(float).eps
        )

        object_has_pixels = sum_by_object > 0
        radial_cv_by_bin = np.zeros((n_bins, object_count), dtype=float)
        row_index, column_index = np.mgrid[0 : labels_array.shape[0], 0 : labels_array.shape[1]]
        i_center = np.zeros(labels_array.shape, dtype=float)
        j_center = np.zeros(labels_array.shape, dtype=float)
        i_center[good_mask] = centers_i_array[center_labels_array[good_mask] - 1]
        j_center[good_mask] = centers_j_array[center_labels_array[good_mask] - 1]
        imask = row_index[good_mask] > i_center[good_mask]
        jmask = column_index[good_mask] > j_center[good_mask]
        absmask = np.abs(row_index[good_mask] - i_center[good_mask]) > np.abs(
            column_index[good_mask] - j_center[good_mask]
        )
        radial_index = (
            imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4
        )

        for bin_index in range(n_bins):
            bin_mask = good_mask & (bin_indexes == bin_index)
            bin_pixels = int(np.sum(bin_mask))
            bin_labels = labels_array[bin_mask]
            bin_radial_index = radial_index[bin_indexes[good_mask] == bin_index]
            labels_and_radii = (bin_labels - 1, bin_radial_index)
            radial_values = scipy.sparse.coo_matrix(
                (image_array[bin_mask], labels_and_radii),
                (object_count, 8),
            ).toarray()
            pixel_count = scipy.sparse.coo_matrix(
                (np.ones(bin_pixels), labels_and_radii),
                (object_count, 8),
            ).toarray()
            mask = pixel_count == 0
            radial_means = np.ma.masked_array(radial_values / pixel_count, mask)
            radial_cv = np.std(radial_means, 1) / np.mean(radial_means, 1)
            radial_cv[np.sum(~mask, 1) == 0] = 0
            radial_cv_by_bin[bin_index] = np.asarray(radial_cv.filled(0), dtype=float)

        return RadialDistributionArrays(
            fraction_at_distance=fraction_at_distance,
            mean_pixel_fraction=mean_pixel_fraction,
            radial_cv_by_bin=radial_cv_by_bin,
            object_has_pixels=object_has_pixels,
            n_bins=n_bins,
        )

    def measure_from_centers(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        d_to_edge: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        labels_array = np.asarray(labels, dtype=np.int32)
        d_to_edge_array = np.asarray(d_to_edge, dtype=np.float64)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            n_bins = int(bin_count) if wants_scaled else int(bin_count) + 1
            return RadialDistributionArrays(
                fraction_at_distance=np.zeros((0, int(bin_count) + 1), dtype=float),
                mean_pixel_fraction=np.zeros((0, int(bin_count) + 1), dtype=float),
                radial_cv_by_bin=np.zeros((n_bins, 0), dtype=float),
                object_has_pixels=np.zeros(0, dtype=bool),
                n_bins=n_bins,
            )

        center_fields = self.center_distance_fields(
            labels_array,
            centers_i,
            centers_j,
        )

        return self.measure(
            image,
            labels_array,
            d_to_edge_array,
            center_fields.d_from_center,
            center_fields.center_labels,
            center_fields.centers_i,
            center_fields.centers_j,
            bin_count=bin_count,
            wants_scaled=wants_scaled,
            maximum_radius=maximum_radius,
        )

    def measure_self_centered(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        labels_array = np.asarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            n_bins = int(bin_count) if wants_scaled else int(bin_count) + 1
            return RadialDistributionArrays(
                fraction_at_distance=np.zeros((0, int(bin_count) + 1), dtype=float),
                mean_pixel_fraction=np.zeros((0, int(bin_count) + 1), dtype=float),
                radial_cv_by_bin=np.zeros((n_bins, 0), dtype=float),
                object_has_pixels=np.zeros(0, dtype=bool),
                n_bins=n_bins,
            )
        geometry = self.label_geometry(labels_array)
        return self.measure(
            image,
            labels_array,
            geometry.d_to_edge,
            geometry.center_fields.d_from_center,
            geometry.center_fields.center_labels,
            geometry.center_fields.centers_i,
            geometry.center_fields.centers_j,
            bin_count=bin_count,
            wants_scaled=wants_scaled,
            maximum_radius=maximum_radius,
        )


class NumbaNumpyRadialDistributionBackendStrategy(
    RadialDistributionBackendStrategy
):
    """Numba-accelerated NumPy radial-distribution backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[2:6, 2:6] = 1
        image = np.zeros((8, 8), dtype=np.float32)
        d_to_edge = np.ones((8, 8), dtype=np.float64)
        centers_i = np.array([3.5], dtype=np.float64)
        centers_j = np.array([3.5], dtype=np.float64)
        self.measure_from_centers(
            image,
            labels,
            d_to_edge,
            centers_i,
            centers_j,
            bin_count=4,
            wants_scaled=True,
            maximum_radius=100,
        )

    def measure(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        d_to_edge: np.ndarray,
        d_from_center: np.ndarray,
        center_labels: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        image_array = np.ascontiguousarray(image, dtype=np.float64)
        labels_array = np.ascontiguousarray(labels, dtype=np.int32)
        d_to_edge_array = np.ascontiguousarray(d_to_edge, dtype=np.float64)
        d_from_center_array = np.ascontiguousarray(d_from_center, dtype=np.float64)
        center_labels_array = np.ascontiguousarray(center_labels, dtype=np.int32)
        centers_i_array = np.ascontiguousarray(centers_i, dtype=np.float64)
        centers_j_array = np.ascontiguousarray(centers_j, dtype=np.float64)

        if image_array.ndim != 2 or labels_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler radial intensity distribution currently supports "
                f"2-D NumPy planes, got image {image_array.shape!r} and labels "
                f"{labels_array.shape!r}."
            )
        if labels_array.shape != image_array.shape:
            raise ValueError(
                "Radial distribution labels must match the image shape; got "
                f"labels {labels_array.shape!r} for image {image_array.shape!r}."
            )
        if bin_count <= 0:
            raise ValueError(f"bin_count must be positive, got {bin_count!r}.")

        object_count = int(labels_array.max()) if labels_array.size else 0
        n_bins = int(bin_count) if wants_scaled else int(bin_count) + 1
        if object_count <= 0:
            return RadialDistributionArrays(
                fraction_at_distance=np.zeros((0, int(bin_count) + 1), dtype=float),
                mean_pixel_fraction=np.zeros((0, int(bin_count) + 1), dtype=float),
                radial_cv_by_bin=np.zeros((n_bins, 0), dtype=float),
                object_has_pixels=np.zeros(0, dtype=bool),
                n_bins=n_bins,
            )

        (
            fraction_at_distance,
            mean_pixel_fraction,
            radial_cv_by_bin,
            object_has_pixels,
        ) = _measure_radial_distribution_numba(
            image_array,
            labels_array,
            d_to_edge_array,
            d_from_center_array,
            center_labels_array,
            centers_i_array,
            centers_j_array,
            int(bin_count),
            bool(wants_scaled),
            int(maximum_radius),
            object_count,
        )
        return RadialDistributionArrays(
            fraction_at_distance=fraction_at_distance,
            mean_pixel_fraction=mean_pixel_fraction,
            radial_cv_by_bin=radial_cv_by_bin,
            object_has_pixels=object_has_pixels,
            n_bins=n_bins,
        )

    def measure_from_centers(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        d_to_edge: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        image_array = np.ascontiguousarray(image, dtype=np.float64)
        labels_array = np.ascontiguousarray(labels, dtype=np.int32)
        d_to_edge_array = np.ascontiguousarray(d_to_edge, dtype=np.float64)
        centers_i_array = np.ascontiguousarray(centers_i, dtype=np.float64)
        centers_j_array = np.ascontiguousarray(centers_j, dtype=np.float64)

        if image_array.ndim != 2 or labels_array.ndim != 2:
            raise NotImplementedError(
                "CellProfiler radial intensity distribution currently supports "
                f"2-D NumPy planes, got image {image_array.shape!r} and labels "
                f"{labels_array.shape!r}."
            )
        if labels_array.shape != image_array.shape:
            raise ValueError(
                "Radial distribution labels must match the image shape; got "
                f"labels {labels_array.shape!r} for image {image_array.shape!r}."
            )
        if bin_count <= 0:
            raise ValueError(f"bin_count must be positive, got {bin_count!r}.")

        object_count = int(labels_array.max()) if labels_array.size else 0
        n_bins = int(bin_count) if wants_scaled else int(bin_count) + 1
        if object_count <= 0:
            return RadialDistributionArrays(
                fraction_at_distance=np.zeros((0, int(bin_count) + 1), dtype=float),
                mean_pixel_fraction=np.zeros((0, int(bin_count) + 1), dtype=float),
                radial_cv_by_bin=np.zeros((n_bins, 0), dtype=float),
                object_has_pixels=np.zeros(0, dtype=bool),
                n_bins=n_bins,
            )

        center_fields = self.center_distance_fields(
            labels_array,
            centers_i_array,
            centers_j_array,
        )
        (
            fraction_at_distance,
            mean_pixel_fraction,
            radial_cv_by_bin,
            object_has_pixels,
        ) = _measure_radial_distribution_numba(
            image_array,
            labels_array,
            d_to_edge_array,
            np.ascontiguousarray(center_fields.d_from_center, dtype=np.float64),
            np.ascontiguousarray(center_fields.center_labels, dtype=np.int32),
            np.ascontiguousarray(center_fields.centers_i, dtype=np.float64),
            np.ascontiguousarray(center_fields.centers_j, dtype=np.float64),
            int(bin_count),
            bool(wants_scaled),
            int(maximum_radius),
            object_count,
        )
        return RadialDistributionArrays(
            fraction_at_distance=fraction_at_distance,
            mean_pixel_fraction=mean_pixel_fraction,
            radial_cv_by_bin=radial_cv_by_bin,
            object_has_pixels=object_has_pixels,
            n_bins=n_bins,
        )

    def measure_self_centered(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        labels_array = np.asarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            n_bins = int(bin_count) if wants_scaled else int(bin_count) + 1
            return RadialDistributionArrays(
                fraction_at_distance=np.zeros((0, int(bin_count) + 1), dtype=float),
                mean_pixel_fraction=np.zeros((0, int(bin_count) + 1), dtype=float),
                radial_cv_by_bin=np.zeros((n_bins, 0), dtype=float),
                object_has_pixels=np.zeros(0, dtype=bool),
                n_bins=n_bins,
            )
        geometry = self.label_geometry(labels_array)
        return self.measure(
            image,
            labels_array,
            geometry.d_to_edge,
            geometry.center_fields.d_from_center,
            geometry.center_fields.center_labels,
            geometry.center_fields.centers_i,
            geometry.center_fields.centers_j,
            bin_count=bin_count,
            wants_scaled=wants_scaled,
            maximum_radius=maximum_radius,
        )


def radial_distribution_backend(
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> RadialDistributionBackendStrategy:
    """Return the selected radial-distribution backend."""
    return RadialDistributionBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


__all__ = [
    "NativeNumpyRadialDistributionBackendStrategy",
    "NumbaNumpyRadialDistributionBackendStrategy",
    "RadialDistributionArrays",
    "RadialDistributionBackendStrategy",
    "RadialLabelGeometry",
    "RadialLabelGeometryCacheKey",
    "radial_distribution_backend",
]


@njit(cache=True)
def _measure_radial_distribution_numba(
    image: np.ndarray,
    labels: np.ndarray,
    d_to_edge: np.ndarray,
    d_from_center: np.ndarray,
    center_labels: np.ndarray,
    centers_i: np.ndarray,
    centers_j: np.ndarray,
    bin_count: int,
    wants_scaled: bool,
    maximum_radius: int,
    object_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = labels.shape
    histogram = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    number_at_distance = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    n_bins = bin_count if wants_scaled else bin_count + 1
    radial_values = np.zeros((n_bins, object_count, 8), dtype=np.float64)
    radial_counts = np.zeros((n_bins, object_count, 8), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            label_id = labels[y, x]
            if label_id <= 0 or label_id > object_count or center_labels[y, x] <= 0:
                continue
            object_index = label_id - 1
            if wants_scaled:
                denominator = d_from_center[y, x] + d_to_edge[y, x] + 0.001
                normalized_distance = d_from_center[y, x] / denominator
            else:
                normalized_distance = d_from_center[y, x] / maximum_radius

            bin_index = int(normalized_distance * bin_count)
            if bin_index > bin_count:
                bin_index = bin_count
            if bin_index < 0:
                bin_index = 0

            pixel_value = image[y, x]
            histogram[object_index, bin_index] += pixel_value
            number_at_distance[object_index, bin_index] += 1.0

            if bin_index < n_bins:
                center_index = center_labels[y, x] - 1
                center_i = centers_i[center_index]
                center_j = centers_j[center_index]
                imask = 1 if y > center_i else 0
                jmask = 1 if x > center_j else 0
                absmask = 1 if abs(y - center_i) > abs(x - center_j) else 0
                radial_index = imask + jmask * 2 + absmask * 4
                radial_values[bin_index, object_index, radial_index] += pixel_value
                radial_counts[bin_index, object_index, radial_index] += 1.0

    fraction_at_distance = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    fraction_at_bin = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    object_has_pixels = np.zeros(object_count, dtype=np.bool_)
    eps = np.finfo(np.float64).eps

    for object_index in range(object_count):
        intensity_sum = 0.0
        pixel_count = 0.0
        for bin_index in range(bin_count + 1):
            intensity_sum += histogram[object_index, bin_index]
            pixel_count += number_at_distance[object_index, bin_index]
        if intensity_sum == 0.0:
            intensity_sum = 1.0
        if pixel_count > 0.0:
            object_has_pixels[object_index] = True
        else:
            pixel_count = 1.0
        for bin_index in range(bin_count + 1):
            fraction_at_distance[object_index, bin_index] = (
                histogram[object_index, bin_index] / intensity_sum
            )
            fraction_at_bin[object_index, bin_index] = (
                number_at_distance[object_index, bin_index] / pixel_count
            )

    mean_pixel_fraction = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    for object_index in range(object_count):
        for bin_index in range(bin_count + 1):
            mean_pixel_fraction[object_index, bin_index] = (
                fraction_at_distance[object_index, bin_index]
                / (fraction_at_bin[object_index, bin_index] + eps)
            )

    radial_cv_by_bin = np.zeros((n_bins, object_count), dtype=np.float64)
    for bin_index in range(n_bins):
        for object_index in range(object_count):
            populated_wedges = 0
            wedge_sum = 0.0
            wedge_sum_sq = 0.0
            for radial_index in range(8):
                count = radial_counts[bin_index, object_index, radial_index]
                if count <= 0.0:
                    continue
                radial_mean = (
                    radial_values[bin_index, object_index, radial_index] / count
                )
                populated_wedges += 1
                wedge_sum += radial_mean
                wedge_sum_sq += radial_mean * radial_mean
            if populated_wedges == 0:
                continue
            mean = wedge_sum / populated_wedges
            variance = wedge_sum_sq / populated_wedges - mean * mean
            if variance < 0.0:
                variance = 0.0
            radial_cv_by_bin[bin_index, object_index] = np.sqrt(variance) / (
                mean + eps
            )

    return (
        fraction_at_distance,
        mean_pixel_fraction,
        radial_cv_by_bin,
        object_has_pixels,
    )
