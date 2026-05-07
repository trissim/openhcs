"""Nominal image-stack layouts for OpenHCS main-flow runtime data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Sequence

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_grayscale_image_slice,
    is_grayscale_image_stack,
    is_grayscale_volume_slice,
    is_grayscale_volume_stack,
)
from openhcs.core.memory import (
    MEMORY_TYPE_NUMPY,
    convert_memory,
    detect_memory_type,
    stack_slices,
    unstack_slices,
)


class ImageStackLayout(ABC, metaclass=AutoRegisterMeta):
    """Nominal family owning stack and unstack behavior for one image layout."""

    __registry_key__ = "layout_key"
    __skip_if_no_key__ = True
    layout_key: ClassVar[str | None] = None
    slice_predicate: ClassVar[Callable[[Any], bool]]
    stack_predicate: ClassVar[Callable[[Any], bool]]

    @classmethod
    def for_slices(cls, slices: Sequence[Any]) -> "ImageStackLayout":
        return cls._matching_layout(
            matches=lambda layout_type: all(
                layout_type.slice_predicate(slice_data)
                for slice_data in slices
            ),
            failure_message=(
                "OpenHCS image stacks require all loaded slices to be either 2D "
                "grayscale images, ZYX grayscale volumes, or HWC color images; "
                "got shapes "
                f"{[getattr(slice_data, 'shape', None) for slice_data in slices]!r}."
            ),
        )

    @classmethod
    def for_stack(cls, array: Any) -> "ImageStackLayout":
        return cls._matching_layout(
            matches=lambda layout_type: layout_type.stack_predicate(array),
            failure_message=(
                "OpenHCS image stack must be shaped (N, H, W), (N, Z, H, W), "
                "or (N, H, W, C), "
                f"got {getattr(array, 'shape', 'unknown')}."
            ),
        )

    @classmethod
    def stack_slices_or_single_stack(
        cls,
        slices: Sequence[Any],
        *,
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        """Stack slices, or pass through one payload already shaped as a stack."""
        if len(slices) == 1:
            candidate = slices[0]
            if cls._is_unambiguous_single_stack(candidate):
                source_type = detect_memory_type(candidate)
                if source_type == memory_type:
                    return candidate
                return _convert_memory(candidate, source_type, memory_type, gpu_id)
        return cls.for_slices(slices).stack(
            slices=slices,
            memory_type=memory_type,
            gpu_id=gpu_id,
        )

    @classmethod
    def _is_unambiguous_single_stack(cls, candidate: Any) -> bool:
        """Return True when one candidate is a stack and not also a valid slice."""
        return any(
            layout_type.stack_predicate(candidate)
            for layout_type in cls.__registry__.values()
        ) and not any(
            layout_type.slice_predicate(candidate)
            for layout_type in cls.__registry__.values()
        )

    @classmethod
    def _matching_layout(
        cls,
        *,
        matches: Callable[[type["ImageStackLayout"]], bool],
        failure_message: str,
    ) -> "ImageStackLayout":
        for layout_type in cls.__registry__.values():
            if matches(layout_type):
                return layout_type()
        raise ValueError(failure_message)

    @abstractmethod
    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        """Stack per-file image slices into an OpenHCS main-flow payload."""

    @abstractmethod
    def unstack(
        self,
        *,
        array: Any,
        memory_type: str,
        gpu_id: int,
    ) -> list[Any]:
        """Split an OpenHCS main-flow payload into per-file image slices."""


class GrayscaleImageStackLayout(ImageStackLayout):
    """OpenHCS grayscale stacks shaped (N, H, W)."""

    layout_key = "grayscale"
    slice_predicate = staticmethod(is_grayscale_image_slice)
    stack_predicate = staticmethod(is_grayscale_image_stack)

    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        return stack_slices(
            slices=list(slices),
            memory_type=memory_type,
            gpu_id=gpu_id,
        )

    def unstack(
        self,
        *,
        array: Any,
        memory_type: str,
        gpu_id: int,
    ) -> list[Any]:
        return unstack_slices(
            array=array,
            memory_type=memory_type,
            gpu_id=gpu_id,
            validate_slices=True,
        )


class ColorImageStackLayout(ImageStackLayout):
    """OpenHCS color stacks shaped (N, H, W, C)."""

    layout_key = "color"
    slice_predicate = staticmethod(is_color_image_slice)
    stack_predicate = staticmethod(is_color_image_stack)

    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        numpy_slices = [
            _as_numpy_slice(slice_data, gpu_id)
            for slice_data in slices
        ]
        channel_counts = {int(slice_data.shape[-1]) for slice_data in numpy_slices}
        if len(channel_counts) != 1:
            raise ValueError(
                "OpenHCS color image stacks require a stable channel count; "
                f"got {sorted(channel_counts)!r}."
            )
        stacked = np.stack(numpy_slices)
        if memory_type == MEMORY_TYPE_NUMPY:
            return stacked
        return _convert_memory(stacked, MEMORY_TYPE_NUMPY, memory_type, gpu_id)

    def unstack(
        self,
        *,
        array: Any,
        memory_type: str,
        gpu_id: int,
    ) -> list[Any]:
        source_type = detect_memory_type(array)
        if source_type != memory_type:
            array = _convert_memory(array, source_type, memory_type, gpu_id)
        return [array[index] for index in range(array.shape[0])]


class GrayscaleVolumeStackLayout(ImageStackLayout):
    """OpenHCS grayscale volume stacks shaped (N, Z, H, W)."""

    layout_key = "grayscale_volume"
    slice_predicate = staticmethod(is_grayscale_volume_slice)
    stack_predicate = staticmethod(is_grayscale_volume_stack)

    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        numpy_slices = [
            _as_numpy_slice(slice_data, gpu_id)
            for slice_data in slices
        ]
        volume_shapes = {tuple(slice_data.shape) for slice_data in numpy_slices}
        if len(volume_shapes) != 1:
            raise ValueError(
                "OpenHCS grayscale volume stacks require stable ZYX shape; "
                f"got {[slice_data.shape for slice_data in numpy_slices]!r}."
            )
        stacked = np.stack(numpy_slices)
        if memory_type == MEMORY_TYPE_NUMPY:
            return stacked
        return _convert_memory(stacked, MEMORY_TYPE_NUMPY, memory_type, gpu_id)

    def unstack(
        self,
        *,
        array: Any,
        memory_type: str,
        gpu_id: int,
    ) -> list[Any]:
        source_type = detect_memory_type(array)
        if source_type != memory_type:
            array = _convert_memory(array, source_type, memory_type, gpu_id)
        return [array[index] for index in range(array.shape[0])]


def _as_numpy_slice(slice_data: Any, gpu_id: int) -> np.ndarray:
    source_type = detect_memory_type(slice_data)
    if source_type == MEMORY_TYPE_NUMPY:
        return slice_data
    return _convert_memory(slice_data, source_type, MEMORY_TYPE_NUMPY, gpu_id)


def _convert_memory(
    data: Any,
    source_type: str,
    target_type: str,
    gpu_id: int,
) -> Any:
    return convert_memory(
        data=data,
        source_type=source_type,
        target_type=target_type,
        gpu_id=gpu_id,
    )
