"""Shared CellProfiler mask-normalized smoothing primitives."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GaussianKernel1D:
    """One-dimensional constant-boundary Gaussian kernel authority."""

    sigma: float

    @property
    def array(self) -> np.ndarray:
        sigma = float(self.sigma)
        if sigma <= 0:
            return np.ones((1,), dtype=np.float64)
        radius = max(1, int(round(4.0 * sigma)))
        coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
        kernel /= np.sum(kernel)
        return kernel.astype(np.float64, copy=False)


@dataclass(frozen=True, slots=True)
class MaskedFilterRequest(ABC):
    """Shared pixel/mask state for mask-normalized filtering."""

    pixels: np.ndarray
    mask: np.ndarray | None

    @property
    def resolved_mask(self) -> np.ndarray:
        if self.mask is None:
            return np.ones(self.pixels.shape, dtype=bool)
        return np.asarray(self.mask, dtype=bool)


@dataclass(frozen=True, slots=True)
class MaskedLinearFilterRequest(MaskedFilterRequest):
    """Mask-normalized linear filtering request."""

    operation: Callable[[np.ndarray], np.ndarray]

    def apply(self) -> np.ndarray:
        mask = self.resolved_mask
        masked_image = np.zeros(self.pixels.shape, dtype=self.pixels.dtype)
        masked_image[mask] = self.pixels[mask]
        weights = self.operation(mask.astype(float))
        filtered = self.operation(masked_image)
        return filtered / (weights + np.finfo(float).eps)


@dataclass(frozen=True, slots=True)
class OpenCVMaskedGaussianFilterRequest(MaskedFilterRequest):
    """OpenCV implementation of mask-normalized Gaussian smoothing."""

    sigma: float

    @property
    def image_array(self) -> np.ndarray:
        return np.ascontiguousarray(self.pixels, dtype=np.float32)

    @property
    def mask_array(self) -> np.ndarray:
        image_array = self.image_array
        if self.mask is None:
            return np.ones(image_array.shape, dtype=np.float32)
        mask_bool = np.asarray(self.mask, dtype=bool)
        if mask_bool.shape != image_array.shape:
            raise ValueError(
                "Smoothing mask must match image shape; got "
                f"{mask_bool.shape!r} for image {image_array.shape!r}."
            )
        return np.ascontiguousarray(mask_bool.astype(np.float32))

    def apply(self) -> np.ndarray:
        import cv2

        image_array = self.image_array
        mask_array = self.mask_array
        kernel = GaussianKernel1D(self.sigma).array.astype(np.float32, copy=False)
        masked_image = np.zeros(image_array.shape, dtype=np.float32)
        np.copyto(masked_image, image_array, where=mask_array.astype(bool, copy=False))
        filtered = cv2.sepFilter2D(
            masked_image,
            cv2.CV_32F,
            kernel,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        weights = cv2.sepFilter2D(
            mask_array,
            cv2.CV_32F,
            kernel,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        return filtered / (weights + np.finfo(np.float32).eps)

    @classmethod
    def apply_stack(
        cls,
        pixel_stack: np.ndarray,
        mask_stack: np.ndarray | None,
        sigma: float,
    ) -> np.ndarray:
        return np.stack(
            [
                cls(
                    pixel_stack[index],
                    None if mask_stack is None else mask_stack[index],
                    sigma,
                ).apply()
                for index in range(pixel_stack.shape[0])
            ],
            axis=0,
        )
