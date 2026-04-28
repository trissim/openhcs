"""Compatibility implementation for legacy CellProfiler Align."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class AlignCropMode(str, Enum):
    """Closed crop modes from legacy CellProfiler Align."""

    KEEP_SIZE = "Keep size"
    CROP_TO_ALIGNED_REGION = "Crop to aligned region"
    PAD_IMAGES = "Pad images"

    @classmethod
    def from_literal(cls, value: "AlignCropMode | str") -> "AlignCropMode":
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        for mode in cls:
            if normalized == mode.value.lower():
                return mode
        raise ValueError(f"Unsupported Align crop mode {value!r}.")


@dataclass(frozen=True, slots=True)
class AlignCropRequest:
    """Inputs shared by Align crop-mode strategies."""

    first_image: np.ndarray
    second_image: np.ndarray
    shift: tuple[float, float]


@numpy(contract=ProcessingContract.FLEXIBLE)
def align(
    image: np.ndarray,
    *,
    method: str = "Mutual Information",
    crop_mode: AlignCropMode | str = AlignCropMode.KEEP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Align the second image to the first image and return both output images."""
    del method
    first_image, second_image = _two_image_payload(image)
    shift = _translation_shift(first_image, second_image)
    aligned_second = _shift_image(second_image, shift)
    return _crop_mode_outputs(
        first_image,
        aligned_second,
        shift=shift,
        crop_mode=AlignCropMode.from_literal(crop_mode),
    )


def _two_image_payload(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(image, "ndim") or image.ndim != 3 or image.shape[0] != 2:
        raise ValueError("Align requires exactly two stacked image inputs.")
    return image[0], image[1]


def _translation_shift(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
) -> tuple[float, float]:
    from skimage.registration import phase_cross_correlation

    shift, _error, _phase = phase_cross_correlation(
        reference_image,
        moving_image,
        upsample_factor=10,
    )
    return float(shift[0]), float(shift[1])


def _shift_image(image: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    from scipy import ndimage

    shifted = ndimage.shift(
        np.asarray(image),
        shift=shift,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return shifted.astype(getattr(image, "dtype", shifted.dtype), copy=False)


def _crop_mode_outputs(
    first_image: np.ndarray,
    second_image: np.ndarray,
    *,
    shift: tuple[float, float],
    crop_mode: AlignCropMode,
) -> tuple[np.ndarray, np.ndarray]:
    return AlignCropModeStrategy.for_crop_mode(crop_mode).apply(
        AlignCropRequest(
            first_image=first_image,
            second_image=second_image,
            shift=shift,
        )
    )


class AlignCropModeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy family for legacy Align crop modes."""

    __registry_key__ = "crop_mode"
    __skip_if_no_key__ = True
    crop_mode: ClassVar[AlignCropMode | None] = None

    @classmethod
    def for_crop_mode(cls, crop_mode: AlignCropMode) -> "AlignCropModeStrategy":
        return cls.__registry__[crop_mode]()

    @abstractmethod
    def apply(self, request: AlignCropRequest) -> tuple[np.ndarray, np.ndarray]:
        """Return first/second image outputs for one crop mode."""


class KeepSizeAlignCropModeStrategy(AlignCropModeStrategy):
    """Keep aligned images in their original shape."""

    crop_mode = AlignCropMode.KEEP_SIZE

    def apply(self, request: AlignCropRequest) -> tuple[np.ndarray, np.ndarray]:
        return request.first_image, request.second_image


class PadImagesAlignCropModeStrategy(AlignCropModeStrategy):
    """Pad both images to preserve all shifted content."""

    crop_mode = AlignCropMode.PAD_IMAGES

    def apply(self, request: AlignCropRequest) -> tuple[np.ndarray, np.ndarray]:
        top, bottom, left, right = _integer_padding(request.shift)
        return (
            np.pad(request.first_image, ((top, bottom), (left, right))),
            np.pad(request.second_image, ((top, bottom), (left, right))),
        )


class CropToOverlapAlignCropModeStrategy(AlignCropModeStrategy):
    """Crop both images to the overlapping aligned region."""

    crop_mode = AlignCropMode.CROP_TO_ALIGNED_REGION

    def apply(self, request: AlignCropRequest) -> tuple[np.ndarray, np.ndarray]:
        row_shift, column_shift = (int(round(value)) for value in request.shift)
        row_start = max(0, row_shift)
        row_stop = min(
            request.first_image.shape[0],
            request.first_image.shape[0] + row_shift,
        )
        column_start = max(0, column_shift)
        column_stop = min(
            request.first_image.shape[1],
            request.first_image.shape[1] + column_shift,
        )
        slices = (slice(row_start, row_stop), slice(column_start, column_stop))
        return request.first_image[slices], request.second_image[slices]


def _integer_padding(shift: tuple[float, float]) -> tuple[int, int, int, int]:
    row_shift, column_shift = (int(round(value)) for value in shift)
    return (
        max(0, row_shift),
        max(0, -row_shift),
        max(0, column_shift),
        max(0, -column_shift),
    )
