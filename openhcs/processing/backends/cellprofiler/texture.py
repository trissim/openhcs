"""Texture-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)


class ObjectTextureCropBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Extract masked object intensity crops for texture measurement."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @classmethod
    def for_callable(
        cls,
        func: object,
        *,
        backend_provider: BackendProviderInput | None = None,
    ) -> "ObjectTextureCropBackendStrategy":
        return super().for_callable(func, backend_provider=backend_provider)

    @abstractmethod
    def object_intensity_crops(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Return positive object labels and CP-style masked intensity crops."""


class HaralickTextureBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Compute CP-compatible 2-D Haralick feature matrices."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @classmethod
    def for_callable(
        cls,
        func: object,
        *,
        backend_provider: BackendProviderInput | None = None,
    ) -> "HaralickTextureBackendStrategy":
        return super().for_callable(func, backend_provider=backend_provider)

    @abstractmethod
    def haralick_features(
        self,
        pixel_data: np.ndarray,
        *,
        scale: int,
        ignore_zeros: bool,
    ) -> np.ndarray:
        """Return one Haralick feature row per 2-D direction."""


class NumbaNumpyObjectTextureCropBackendStrategy(ObjectTextureCropBackendStrategy):
    """Numba-accelerated NumPy backend for object texture crop extraction."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def object_intensity_crops(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        image_array = np.asarray(image)
        labels_array = np.asarray(labels)
        if image_array.ndim != 2 or labels_array.ndim != 2:
            raise NotImplementedError(
                "Numba texture crop backend currently supports 2-D NumPy planes."
            )
        if image_array.shape != labels_array.shape:
            raise ValueError(
                "Texture image and labels must have identical shapes; got "
                f"{image_array.shape!r} and {labels_array.shape!r}."
            )
        object_labels, boxes = _object_bounding_boxes_numba(
            np.ascontiguousarray(labels_array, dtype=np.int64)
        )
        crops: list[np.ndarray] = []
        for index, object_label in enumerate(object_labels):
            y0, y1, x0, x1 = boxes[index]
            label_crop = labels_array[y0:y1, x0:x1]
            intensity_crop = np.asarray(image_array[y0:y1, x0:x1]).copy()
            intensity_crop[label_crop != object_label] = 0
            crops.append(intensity_crop)
        return object_labels.astype(np.int64, copy=False), tuple(crops)


class NumbaNumpyHaralickTextureBackendStrategy(HaralickTextureBackendStrategy):
    """Numba implementation of mahotas' default 2-D Haralick semantics."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def haralick_features(
        self,
        pixel_data: np.ndarray,
        *,
        scale: int,
        ignore_zeros: bool,
    ) -> np.ndarray:
        pixel_array = np.ascontiguousarray(pixel_data)
        if pixel_array.ndim != 2:
            raise ValueError("Haralick texture backend expects a 2-D image plane.")
        if scale < 1:
            raise ValueError(f"Haralick texture scale must be positive, got {scale}.")
        if pixel_array.shape[0] <= scale or pixel_array.shape[1] <= scale:
            return np.zeros((4, 13), dtype=np.float64)
        return _haralick_2d_features_numba(
            pixel_array.astype(np.int64, copy=False),
            int(scale),
            bool(ignore_zeros),
        )


class NativeNumpyHaralickTextureBackendStrategy(HaralickTextureBackendStrategy):
    """Explicit mahotas backend used as the native reference implementation."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE

    def haralick_features(
        self,
        pixel_data: np.ndarray,
        *,
        scale: int,
        ignore_zeros: bool,
    ) -> np.ndarray:
        import mahotas.features as mahotas_features

        return np.asarray(
            mahotas_features.haralick(
                np.asarray(pixel_data),
                distance=scale,
                ignore_zeros=ignore_zeros,
            ),
            dtype=np.float64,
        )


@njit(cache=True)
def _object_bounding_boxes_numba(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label > max_label:
                max_label = label

    min_y = np.full(max_label + 1, height, dtype=np.int64)
    min_x = np.full(max_label + 1, width, dtype=np.int64)
    max_y = np.full(max_label + 1, -1, dtype=np.int64)
    max_x = np.full(max_label + 1, -1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label <= 0:
                continue
            if y < min_y[label]:
                min_y[label] = y
            if x < min_x[label]:
                min_x[label] = x
            if y > max_y[label]:
                max_y[label] = y
            if x > max_x[label]:
                max_x[label] = x

    object_count = 0
    for label in range(1, max_label + 1):
        if max_y[label] >= 0:
            object_count += 1

    object_labels = np.empty(object_count, dtype=np.int64)
    boxes = np.empty((object_count, 4), dtype=np.int64)
    index = 0
    for label in range(1, max_label + 1):
        if max_y[label] < 0:
            continue
        object_labels[index] = label
        boxes[index, 0] = min_y[label]
        boxes[index, 1] = max_y[label] + 1
        boxes[index, 2] = min_x[label]
        boxes[index, 3] = max_x[label] + 1
        index += 1
    return object_labels, boxes


@njit(cache=True)
def _haralick_2d_features_numba(
    image: np.ndarray,
    distance: int,
    ignore_zeros: bool,
) -> np.ndarray:
    max_value = 0
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value > max_value:
                max_value = value

    gray_count = max_value + 1
    features = np.zeros((4, 13), dtype=np.float64)
    deltas_y = np.array((0, 1, 1, 1), dtype=np.int64)
    deltas_x = np.array((1, 1, 0, -1), dtype=np.int64)

    for direction in range(4):
        cmat = np.zeros((gray_count, gray_count), dtype=np.float64)
        dy = deltas_y[direction] * distance
        dx = deltas_x[direction] * distance
        for y in range(height):
            yy = y + dy
            if yy < 0 or yy >= height:
                continue
            for x in range(width):
                xx = x + dx
                if xx < 0 or xx >= width:
                    continue
                a = image[y, x]
                b = image[yy, xx]
                if ignore_zeros and (a == 0 or b == 0):
                    continue
                cmat[a, b] += 1.0
                cmat[b, a] += 1.0

        total = cmat.sum()
        if total == 0.0:
            continue
        features[direction, :] = _haralick_features_from_cmat_numba(cmat, total)
    return features


@njit(cache=True)
def _haralick_features_from_cmat_numba(
    cmat: np.ndarray,
    total: float,
) -> np.ndarray:
    gray_count = cmat.shape[0]
    feats = np.zeros(13, dtype=np.float64)
    px = np.zeros(gray_count, dtype=np.float64)
    py = np.zeros(gray_count, dtype=np.float64)
    px_plus_y = np.zeros(gray_count * 2, dtype=np.float64)
    px_minus_y = np.zeros(gray_count, dtype=np.float64)

    for i in range(gray_count):
        for j in range(gray_count):
            p = cmat[i, j] / total
            px[j] += p
            py[i] += p
            px_plus_y[i + j] += p
            diff = i - j
            if diff < 0:
                diff = -diff
            px_minus_y[diff] += p
            feats[0] += p * p
            feats[1] += diff * diff * p
            feats[4] += p / (1.0 + diff * diff)

    ux = 0.0
    uy = 0.0
    for k in range(gray_count):
        ux += px[k] * k
        uy += py[k] * k

    vx = 0.0
    vy = 0.0
    for k in range(gray_count):
        vx += px[k] * k * k
        vy += py[k] * k * k
    vx -= ux * ux
    vy -= uy * uy

    sx = np.sqrt(vx)
    sy = np.sqrt(vy)
    if sx == 0.0 or sy == 0.0:
        feats[2] = 1.0
    else:
        ijp = 0.0
        for i in range(gray_count):
            for j in range(gray_count):
                ijp += i * j * (cmat[i, j] / total)
        feats[2] = (ijp - ux * uy) / (sx * sy)

    feats[3] = vx
    sum_average = 0.0
    sum_second = 0.0
    for k in range(gray_count * 2):
        sum_average += k * px_plus_y[k]
        sum_second += k * k * px_plus_y[k]
    feats[5] = sum_average
    feats[7] = _entropy_numba(px_plus_y)
    feats[6] = sum_second - sum_average * sum_average
    feats[8] = _entropy_matrix_numba(cmat, total)

    mean_minus = 0.0
    for k in range(gray_count):
        mean_minus += px_minus_y[k]
    mean_minus /= gray_count
    variance_minus = 0.0
    for k in range(gray_count):
        delta = px_minus_y[k] - mean_minus
        variance_minus += delta * delta
    feats[9] = variance_minus / gray_count
    feats[10] = _entropy_numba(px_minus_y)

    hx = _entropy_numba(px)
    hy = _entropy_numba(py)
    hxy1 = 0.0
    hxy2 = 0.0
    for i in range(gray_count):
        for j in range(gray_count):
            p = cmat[i, j] / total
            cross = py[i] * px[j]
            if cross > 0.0 and p > 0.0:
                hxy1 -= p * np.log2(cross)
            if cross > 0.0:
                hxy2 -= cross * np.log2(cross)

    if hx >= hy:
        max_h = hx
    else:
        max_h = hy
    if max_h == 0.0:
        feats[11] = feats[8] - hxy1
    else:
        feats[11] = (feats[8] - hxy1) / max_h
    info2 = 1.0 - np.exp(-2.0 * (hxy2 - feats[8]))
    if info2 < 0.0:
        info2 = 0.0
    feats[12] = np.sqrt(info2)
    return feats


@njit(cache=True)
def _entropy_numba(values: np.ndarray) -> float:
    result = 0.0
    for value in values:
        if value > 0.0:
            result -= value * np.log2(value)
    return result


@njit(cache=True)
def _entropy_matrix_numba(cmat: np.ndarray, total: float) -> float:
    result = 0.0
    height, width = cmat.shape
    for y in range(height):
        for x in range(width):
            p = cmat[y, x] / total
            if p > 0.0:
                result -= p * np.log2(p)
    return result


__all__ = [
    "HaralickTextureBackendStrategy",
    "NativeNumpyHaralickTextureBackendStrategy",
    "NumbaNumpyHaralickTextureBackendStrategy",
    "NumbaNumpyObjectTextureCropBackendStrategy",
    "ObjectTextureCropBackendStrategy",
]
