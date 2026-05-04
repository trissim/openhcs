"""Image-quality backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)


class ImageQualityBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Image-quality primitives keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        """Return CP-style Haralick H3 correlation for one image plane."""

    @abstractmethod
    def radial_power_spectrum(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return CP-style radial Fourier spectrum bins."""


class NumpyImageQualityBackendStrategy(ImageQualityBackendStrategy):
    """Independent NumPy implementation of image-quality primitives."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 2:
            raise NotImplementedError(
                "Image-quality Haralick correlation currently supports 2-D "
                f"NumPy planes, got shape {image_array.shape!r}."
            )
        return _haralick_h3_numpy(image_array, int(scale))

    def radial_power_spectrum(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_array = np.asarray(image, dtype=np.float64)
        if image_array.ndim != 2:
            raise NotImplementedError(
                "Image-quality radial power spectrum currently supports 2-D "
                f"NumPy planes, got shape {image_array.shape!r}."
            )
        return _radial_power_spectrum_numpy(image_array)


class NumbaNumpyImageQualityBackendStrategy(NumpyImageQualityBackendStrategy):
    """Numba-accelerated NumPy image-quality backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 2:
            raise NotImplementedError(
                "Numba image-quality Haralick correlation currently supports "
                f"2-D NumPy planes, got shape {image_array.shape!r}."
            )
        return float(
            _haralick_h3_numba(
                np.ascontiguousarray(image_array, dtype=np.float32),
                int(scale),
            )
        )


class CentrosomeNumpyImageQualityBackendStrategy(ImageQualityBackendStrategy):
    """Explicit centrosome provider for image-quality primitives."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        import centrosome.haralick

        image_array = np.asarray(image, dtype=np.float32)
        value = centrosome.haralick.Haralick(
            image_array,
            np.ones(image_array.shape, dtype=int),
            0,
            int(scale),
        ).H3()
        return _finite_scalar(value)

    def radial_power_spectrum(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import centrosome.radial_power_spectrum

        radii, magnitude, power = centrosome.radial_power_spectrum.rps(
            np.asarray(image)
        )
        return (
            np.asarray(radii),
            np.asarray(magnitude),
            np.asarray(power),
        )


def image_quality_backend(
    *,
    backend_provider: BackendProviderInput | None = None,
) -> ImageQualityBackendStrategy:
    """Return the selected image-quality backend."""
    return ImageQualityBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=backend_provider,
    )


def _haralick_h3_numpy(image: np.ndarray, scale: int) -> float:
    if image.size == 0 or scale < 1 or image.shape[1] <= scale:
        return 0.0

    minimum = float(np.min(image))
    maximum = float(np.max(image))
    divisor = maximum - minimum if maximum > minimum else 1.0
    quantized = np.floor(((image - minimum) / divisor) * 8.0).astype(np.int16)
    quantized = np.clip(quantized, 0, 7)
    level_count = int(np.max(quantized)) + 1
    if level_count <= 0:
        return 0.0

    left = quantized[:, :-scale].ravel()
    right = quantized[:, scale:].ravel()
    pair_count = left.size
    if pair_count == 0:
        return 0.0
    flat_indexes = level_count * left + right
    matrix = np.bincount(
        flat_indexes,
        minlength=level_count * level_count,
    ).reshape(level_count, level_count).astype(float)
    return _haralick_h3_from_matrix(matrix / float(pair_count))


def _haralick_h3_from_matrix(matrix: np.ndarray) -> float:
    total = float(np.sum(matrix))
    if total <= 0.0:
        return 0.0
    matrix = matrix / total
    px = matrix.sum(axis=1)
    py = matrix.sum(axis=0)
    px_total = float(np.sum(px))
    py_total = float(np.sum(py))
    if px_total <= 0.0 or py_total <= 0.0:
        return 0.0
    px = px / px_total
    py = py / py_total
    levels = np.arange(matrix.shape[0], dtype=float) + 1.0
    mux = float(np.sum(levels * px))
    muy = float(np.sum(levels * py))
    sigmax = float(np.sqrt(np.sum(((levels - mux) ** 2) * px)))
    sigmay = float(np.sqrt(np.sum(((levels - muy) ** 2) * py)))
    if sigmax <= 0.0 or sigmay <= 0.0:
        return 0.0
    summed = float(np.sum(np.outer(levels, levels) * matrix))
    value = (summed - mux * muy) / (sigmax * sigmay)
    return value if np.isfinite(value) else 0.0


def _radial_power_spectrum_numpy(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.fftpack import fft2

    height, width = image.shape
    row2 = np.arange(height).reshape((height, 1)) ** 2
    col2 = np.arange(width) ** 2
    radii2 = row2 + col2
    radii2 = np.minimum(radii2, np.flipud(radii2))
    radii2 = np.minimum(radii2, np.fliplr(radii2))
    max_width = min(height, width) / 8.0

    working = image.astype(np.float64, copy=False)
    if np.ptp(working) > 0.0:
        mean_value = float(np.mean(working))
        mad_value = float(np.median(np.abs(working - mean_value)))
        with np.errstate(divide="ignore", invalid="ignore"):
            working = working / mad_value

    centered = working - np.mean(working)
    magnitude = np.abs(fft2(centered))
    power = magnitude**2
    radii = np.floor(np.sqrt(radii2)).astype(int) + 1
    labels = np.arange(2, int(np.floor(max_width)), dtype=int)
    if labels.size == 0:
        return (
            np.array([2], dtype=int),
            np.array([0], dtype=int),
            np.array([0], dtype=int),
        )
    return (
        labels,
        np.asarray(scipy.ndimage.sum(magnitude, radii, labels)),
        np.asarray(scipy.ndimage.sum(power, radii, labels)),
    )


def _finite_scalar(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        return 0.0
    scalar = float(array.ravel()[0])
    return scalar if np.isfinite(scalar) else 0.0


@njit(cache=True)
def _haralick_h3_numba(image: np.ndarray, scale: int) -> float:
    height, width = image.shape
    if height == 0 or width == 0 or scale < 1 or width <= scale:
        return 0.0

    minimum = image[0, 0]
    maximum = image[0, 0]
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
    divisor = maximum - minimum
    if divisor <= 0.0:
        divisor = 1.0

    level_count = 1
    for y in range(height):
        for x in range(width):
            level = int(((image[y, x] - minimum) / divisor) * 8.0)
            if level < 0:
                level = 0
            elif level > 7:
                level = 7
            if level + 1 > level_count:
                level_count = level + 1

    matrix = np.zeros((level_count, level_count), dtype=np.float64)
    pair_count = 0
    for y in range(height):
        for x in range(width - scale):
            left = int(((image[y, x] - minimum) / divisor) * 8.0)
            if left < 0:
                left = 0
            elif left > 7:
                left = 7
            right = int(((image[y, x + scale] - minimum) / divisor) * 8.0)
            if right < 0:
                right = 0
            elif right > 7:
                right = 7
            matrix[left, right] += 1.0
            pair_count += 1

    if pair_count == 0:
        return 0.0
    for y in range(level_count):
        for x in range(level_count):
            matrix[y, x] /= pair_count

    px = np.zeros(level_count, dtype=np.float64)
    py = np.zeros(level_count, dtype=np.float64)
    for y in range(level_count):
        for x in range(level_count):
            px[y] += matrix[y, x]
            py[x] += matrix[y, x]

    px_total = 0.0
    py_total = 0.0
    for index in range(level_count):
        px_total += px[index]
        py_total += py[index]
    if px_total <= 0.0 or py_total <= 0.0:
        return 0.0
    for index in range(level_count):
        px[index] /= px_total
        py[index] /= py_total

    mux = 0.0
    muy = 0.0
    for index in range(level_count):
        level_value = index + 1.0
        mux += level_value * px[index]
        muy += level_value * py[index]

    sigmax2 = 0.0
    sigmay2 = 0.0
    for index in range(level_count):
        level_value = index + 1.0
        dx = level_value - mux
        dy = level_value - muy
        sigmax2 += dx * dx * px[index]
        sigmay2 += dy * dy * py[index]
    if sigmax2 <= 0.0 or sigmay2 <= 0.0:
        return 0.0
    sigmax = np.sqrt(sigmax2)
    sigmay = np.sqrt(sigmay2)

    summed = 0.0
    for y in range(level_count):
        for x in range(level_count):
            summed += (y + 1.0) * (x + 1.0) * matrix[y, x]
    value = (summed - mux * muy) / (sigmax * sigmay)
    if np.isfinite(value):
        return value
    return 0.0


__all__ = [
    "CentrosomeNumpyImageQualityBackendStrategy",
    "ImageQualityBackendStrategy",
    "NumbaNumpyImageQualityBackendStrategy",
    "NumpyImageQualityBackendStrategy",
    "image_quality_backend",
]
