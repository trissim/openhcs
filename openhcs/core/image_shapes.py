"""Nominal image payload shape roles for OpenHCS runtime paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar
from typing import TypeVar

import numpy as np
from metaclass_registry import AutoRegisterMeta


COLOR_CHANNEL_COUNTS = frozenset((3, 4))
CHANNEL_LAST_IMAGE_CHANNEL_COUNTS = frozenset((2, 3, 4))
ArrayT = TypeVar("ArrayT", bound=np.ndarray)


@dataclass(frozen=True, slots=True)
class ArrayShape:
    """Nominal shape capability for array-like image payloads."""

    ndim: int
    shape: tuple[int, ...]

    @classmethod
    def from_value(cls, value: Any) -> "ArrayShape | None":
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if array.ndim == 0:
            return None
        return cls(ndim=int(array.ndim), shape=tuple(int(axis) for axis in array.shape))

    def has_rank(self, ndim: int) -> bool:
        return self.ndim == ndim

    def has_channel_last(self) -> bool:
        return self.shape[-1] in COLOR_CHANNEL_COUNTS

    def has_channel_last_image_count(self) -> bool:
        return self.shape[-1] in CHANNEL_LAST_IMAGE_CHANNEL_COUNTS


class ImageShapeRole(ABC, metaclass=AutoRegisterMeta):
    """Nominal owner for one OpenHCS image slice/stack shape family."""

    __registry_key__ = "role_key"
    __skip_if_no_key__ = True
    role_key: ClassVar[str | None] = None

    @classmethod
    def matches_slice(cls, value: Any) -> bool:
        array_shape = ArrayShape.from_value(value)
        return array_shape is not None and cls.matches_slice_shape(array_shape)

    @classmethod
    def matches_stack(cls, value: Any) -> bool:
        array_shape = ArrayShape.from_value(value)
        return array_shape is not None and cls.matches_stack_shape(array_shape)

    @classmethod
    @abstractmethod
    def matches_slice_shape(cls, array_shape: ArrayShape) -> bool:
        """Return True when the shape is one file-level image payload."""

    @classmethod
    @abstractmethod
    def matches_stack_shape(cls, array_shape: ArrayShape) -> bool:
        """Return True when the shape is an OpenHCS main-flow image stack."""


class ColorImageShapeRole(ImageShapeRole):
    """Channel-last RGB/RGBA image planes and stacks."""

    role_key = "color"

    @classmethod
    def matches_slice_shape(cls, array_shape: ArrayShape) -> bool:
        return array_shape.has_rank(3) and array_shape.has_channel_last()

    @classmethod
    def matches_stack_shape(cls, array_shape: ArrayShape) -> bool:
        return array_shape.has_rank(4) and array_shape.has_channel_last()


class GrayscaleImageShapeRole(ImageShapeRole):
    """2D grayscale image planes and (N, H, W) stacks."""

    role_key = "grayscale"

    @classmethod
    def matches_slice_shape(cls, array_shape: ArrayShape) -> bool:
        return array_shape.has_rank(2)

    @classmethod
    def matches_stack_shape(cls, array_shape: ArrayShape) -> bool:
        return (
            array_shape.has_rank(3)
            and not ColorImageShapeRole.matches_slice_shape(array_shape)
        )


class GrayscaleVolumeShapeRole(ImageShapeRole):
    """ZYX grayscale volumes and (N, Z, H, W) volume stacks."""

    role_key = "grayscale_volume"

    @classmethod
    def matches_slice_shape(cls, array_shape: ArrayShape) -> bool:
        return (
            array_shape.has_rank(3)
            and not ColorImageShapeRole.matches_slice_shape(array_shape)
        )

    @classmethod
    def matches_stack_shape(cls, array_shape: ArrayShape) -> bool:
        return (
            array_shape.has_rank(4)
            and not ColorImageShapeRole.matches_stack_shape(array_shape)
        )


class ColorVolumeShapeRole(ImageShapeRole):
    """Channel-last RGB/RGBA volumes and volume stacks."""

    role_key = "color_volume"

    @classmethod
    def matches_slice_shape(cls, array_shape: ArrayShape) -> bool:
        return array_shape.has_rank(4) and array_shape.has_channel_last()

    @classmethod
    def matches_stack_shape(cls, array_shape: ArrayShape) -> bool:
        return array_shape.has_rank(5) and array_shape.has_channel_last()


class ChannelFirstVolumeShapeRole(ImageShapeRole):
    """CZYX channel-first volumes and (N, C, Z, H, W) stacks."""

    role_key = "channel_first_volume"

    @classmethod
    def matches_slice_shape(cls, array_shape: ArrayShape) -> bool:
        return array_shape.has_rank(4) and array_shape.shape[0] > 1

    @classmethod
    def matches_stack_shape(cls, array_shape: ArrayShape) -> bool:
        return array_shape.has_rank(5) and array_shape.shape[1] > 1


def is_grayscale_image_slice(value: Any) -> bool:
    """Return True for one 2D grayscale image plane."""
    return GrayscaleImageShapeRole.matches_slice(value)


def is_color_image_slice(value: Any) -> bool:
    """Return True for one HWC RGB/RGBA image plane."""
    return ColorImageShapeRole.matches_slice(value)


def is_grayscale_image_stack(value: Any) -> bool:
    """Return True for an OpenHCS grayscale stack shaped (N, H, W)."""
    return GrayscaleImageShapeRole.matches_stack(value)


def is_color_image_stack(value: Any) -> bool:
    """Return True for an OpenHCS color stack shaped (N, H, W, C)."""
    return ColorImageShapeRole.matches_stack(value)


def is_channel_last_image_slice(value: Any) -> bool:
    """Return True for one declared channel-last image plane."""
    array_shape = ArrayShape.from_value(value)
    return (
        array_shape is not None
        and array_shape.has_rank(3)
        and array_shape.has_channel_last_image_count()
    )


def is_channel_last_image_stack(value: Any) -> bool:
    """Return True for an OpenHCS stack of declared channel-last image planes."""
    array_shape = ArrayShape.from_value(value)
    return (
        array_shape is not None
        and array_shape.has_rank(4)
        and array_shape.has_channel_last_image_count()
    )


def is_grayscale_volume_slice(value: Any) -> bool:
    """Return True for one grayscale volume shaped (Z, H, W)."""
    return GrayscaleVolumeShapeRole.matches_slice(value)


def is_grayscale_volume_stack(value: Any) -> bool:
    """Return True for an OpenHCS grayscale volume stack shaped (N, Z, H, W)."""
    return GrayscaleVolumeShapeRole.matches_stack(value)


def is_color_volume_slice(value: Any) -> bool:
    """Return True for one channel-last RGB/RGBA volume shaped (Z, H, W, C)."""
    return ColorVolumeShapeRole.matches_slice(value)


def is_color_volume_stack(value: Any) -> bool:
    """Return True for OpenHCS color volume stacks shaped (N, Z, H, W, C)."""
    return ColorVolumeShapeRole.matches_stack(value)


def is_channel_first_volume_slice(value: Any) -> bool:
    """Return True for one channel-first volume shaped (C, Z, H, W)."""
    return ChannelFirstVolumeShapeRole.matches_slice(value)


def is_channel_first_volume_stack(value: Any) -> bool:
    """Return True for OpenHCS channel-first volume stacks shaped (N, C, Z, H, W)."""
    return ChannelFirstVolumeShapeRole.matches_stack(value)


def is_image_stack(value: Any) -> bool:
    """Return True for OpenHCS main-flow image stacks."""
    return any(
        shape_role.matches_stack(value)
        for shape_role in ImageShapeRole.__registry__.values()
    )


def image_spatial_shape_yx(value: Any) -> tuple[int, int] | None:
    """Return the XY frame for an image payload shape, if it is image-like."""
    array_shape = ArrayShape.from_value(value)
    if array_shape is None or array_shape.ndim < 2:
        return None
    spatial_axes = image_spatial_axis_indices(value)
    if spatial_axes is None:
        return None
    return tuple(int(array_shape.shape[axis]) for axis in spatial_axes)


def image_spatial_axis_indices(value: np.ndarray) -> tuple[int, int] | None:
    """Return the array axes that carry image Y/X coordinates."""
    array_shape = ArrayShape.from_value(value)
    if array_shape is None or array_shape.ndim < 2:
        return None
    if ColorImageShapeRole.matches_slice_shape(array_shape):
        return 0, 1
    if ColorImageShapeRole.matches_stack_shape(array_shape):
        return 1, 2
    if ColorVolumeShapeRole.matches_slice_shape(array_shape):
        return array_shape.ndim - 3, array_shape.ndim - 2
    if ColorVolumeShapeRole.matches_stack_shape(array_shape):
        return array_shape.ndim - 3, array_shape.ndim - 2
    return array_shape.ndim - 2, array_shape.ndim - 1


def trailing_spatial_target_shape(
    shape: tuple[int, ...],
    spatial_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Return a full shape that preserves leading axes before spatial axes."""
    spatial_rank = len(spatial_shape)
    if spatial_rank <= 0:
        raise ValueError("spatial_shape must contain at least one axis.")
    if len(shape) < spatial_rank:
        raise ValueError(
            "Cannot apply spatial shape with rank greater than input rank: "
            f"{spatial_shape!r} for {shape!r}."
        )
    return (*shape[: len(shape) - spatial_rank], *spatial_shape)


def trailing_spatial_factors(
    ndim: int,
    spatial_factors: tuple[float, ...],
) -> tuple[float, ...]:
    """Return full-rank factors that preserve leading axes before spatial axes."""
    spatial_rank = len(spatial_factors)
    if spatial_rank <= 0:
        raise ValueError("spatial_factors must contain at least one axis.")
    if ndim < spatial_rank:
        raise ValueError(
            "Cannot apply spatial factors with rank greater than input rank: "
            f"{spatial_factors!r} for ndim={ndim}."
        )
    return (*((1.0,) * (ndim - spatial_rank)), *spatial_factors)


def apply_over_trailing_spatial_axes(
    array: ArrayT,
    spatial_rank: int,
    operation: Callable[[ArrayT], ArrayT],
    *,
    fill_value: object = 0,
) -> ArrayT:
    """Apply an operation to trailing spatial axes while preserving leading axes."""
    if spatial_rank <= 0:
        raise ValueError("spatial_rank must be positive.")
    if array.ndim < spatial_rank:
        raise ValueError(
            f"Cannot apply spatial_rank={spatial_rank} to shape {array.shape!r}."
        )
    if array.ndim == spatial_rank:
        return operation(array)
    output = np.full_like(array, fill_value)
    leading_shape = array.shape[: array.ndim - spatial_rank]
    for leading_index in np.ndindex(leading_shape):
        output[leading_index] = operation(array[leading_index])
    return output
