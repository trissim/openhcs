"""Nominal image-stack layouts for OpenHCS main-flow runtime data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Sequence

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.image_shapes import (
    ChannelFirstVolumeShapeRole,
    ColorImageShapeRole,
    ColorVolumeShapeRole,
    GrayscaleImageShapeRole,
    GrayscaleVolumeShapeRole,
    ImageShapeRole,
)
from openhcs.core.memory import (
    MEMORY_TYPE_NUMPY,
    convert_memory,
    detect_memory_type,
    stack_slices,
    unstack_slices,
)

ImageStackData = np.ndarray
ImageStackSliceData = np.ndarray


@dataclass(frozen=True, slots=True)
class ImageStackLayoutSelection:
    """Typed request for selecting an image-stack layout from registered classes."""

    matches: Callable[[type["ImageStackLayout"]], bool]
    failure_message: str

    def select(self) -> "ImageStackLayout":
        for layout_type in ImageStackLayout.__registry__.values():
            if self.matches(layout_type):
                return layout_type()
        raise ValueError(self.failure_message)


@dataclass(frozen=True, slots=True)
class MemoryConversion:
    """Typed conversion request for moving payloads between memory domains."""

    data: ImageStackData
    source_type: str
    target_type: str
    gpu_id: int

    def materialize(self) -> ImageStackData:
        if self.source_type == self.target_type:
            return self.data
        return convert_memory(
            data=self.data,
            source_type=self.source_type,
            target_type=self.target_type,
            gpu_id=self.gpu_id,
        )


@dataclass(frozen=True, slots=True)
class ImageStackUnstackRequest:
    """Shared request fields and layout dispatch for image-stack unstacking."""

    array: ImageStackData
    memory_type: str
    gpu_id: int

    def layout_slices(
        self,
        layout_source: ImageStackData | None = None,
    ) -> tuple[ImageStackSliceData, ...]:
        layout_value = self.array if layout_source is None else layout_source
        return ImageStackLayout.for_stack(layout_value).unstack(
            array=self.array,
            memory_type=self.memory_type,
            gpu_id=self.gpu_id,
        )


@dataclass(frozen=True, slots=True)
class SourceSliceUnstackRequest(ImageStackUnstackRequest):
    """Nominal request for unstacking output against source-slice shape domains."""

    source_slice_shapes: Sequence[tuple[int, ...]]

    def slices(self) -> list[ImageStackSliceData]:
        if self.output_is_source_slice():
            return [
                MemoryConversionSource.DETECTED.conversion(
                    self.array,
                    target_type=self.memory_type,
                    gpu_id=self.gpu_id,
                ).materialize()
            ]
        return list(self.layout_slices())

    def output_is_source_slice(self) -> bool:
        from openhcs.core.runtime_values import runtime_array_operand

        output_shape = tuple(np.shape(runtime_array_operand(self.array)))
        if not output_shape:
            return False
        if output_shape in set(self.source_slice_shapes):
            return True
        if self.output_is_singleton_stack_for_only_source_slice(output_shape):
            return False
        return (
            len(self.source_slice_shapes) == 1
            and ImageStackLayout.is_unambiguous_slice(self.array)
        )

    def output_is_singleton_stack_for_only_source_slice(
        self,
        output_shape: tuple[int, ...],
    ) -> bool:
        if len(self.source_slice_shapes) != 1 or output_shape[:1] != (1,):
            return False
        if output_shape[1:] != tuple(self.source_slice_shapes[0]):
            return False
        try:
            ImageStackLayout.for_stack(self.array)
        except ValueError:
            return False
        return True


@dataclass(frozen=True, slots=True)
class ImageStackLayoutUnstackRequest(ImageStackUnstackRequest):
    """Nominal request for unstacking with an optional layout source."""

    layout_source: ImageStackData | None = None

    def slices(self) -> tuple[ImageStackSliceData, ...]:
        return self.layout_slices(self.layout_source)


class MemoryConversionSource(Enum):
    """Nominal source-domain variants for memory conversion requests."""

    DETECTED = "detected"
    NUMPY = "numpy"

    def conversion(
        self,
        data: ImageStackData,
        *,
        target_type: str,
        gpu_id: int,
    ) -> MemoryConversion:
        source_type = (
            detect_memory_type(data)
            if self is MemoryConversionSource.DETECTED
            else MEMORY_TYPE_NUMPY
        )
        return MemoryConversion(
            data=data,
            source_type=source_type,
            target_type=target_type,
            gpu_id=gpu_id,
        )


@dataclass(frozen=True, slots=True)
class NumpySliceConversion:
    """Nominal authority for converting one image-like slice to numpy."""

    slice_data: ImageStackSliceData
    gpu_id: int

    def array(self) -> np.ndarray:
        from openhcs.core.runtime_values import runtime_array_operand

        return MemoryConversionSource.DETECTED.conversion(
            runtime_array_operand(self.slice_data),
            target_type=MEMORY_TYPE_NUMPY,
            gpu_id=self.gpu_id,
        ).materialize()


class ImageStackLayout(ABC, metaclass=AutoRegisterMeta):
    """Nominal family owning stack and unstack behavior for one image layout."""

    __registry_key__ = "layout_key"
    __skip_if_no_key__ = True
    layout_key: ClassVar[str | None] = None
    shape_role: ClassVar[type[ImageShapeRole]]
    stable_slice_shape_error: ClassVar[str | None] = None
    disambiguate_single_candidate_stack_from_slices: ClassVar[bool] = True

    @classmethod
    def for_slices(cls, slices: Sequence[ImageStackSliceData]) -> "ImageStackLayout":
        return ImageStackLayoutSelection(
            matches=lambda layout_type: all(
                layout_type.shape_role.matches_slice(slice_data)
                for slice_data in slices
            ),
            failure_message=(
                "OpenHCS image stacks require all loaded slices to be either 2D "
                "grayscale images, ZYX grayscale volumes, HWC color images, "
                "ZYXC color volumes, or CZYX channel-first volumes; "
                "got shapes "
                f"{[tuple(np.shape(slice_data)) for slice_data in slices]!r}."
            ),
        ).select()

    @classmethod
    def for_stack(cls, array: ImageStackData) -> "ImageStackLayout":
        array_shape = tuple(np.shape(array))
        return ImageStackLayoutSelection(
            matches=lambda layout_type: layout_type.shape_role.matches_stack(array),
            failure_message=(
                "OpenHCS image stack must be shaped (N, H, W), (N, Z, H, W), "
                "(N, H, W, C), (N, Z, H, W, C), or (N, C, Z, H, W), "
                f"got {array_shape}."
            ),
        ).select()

    @classmethod
    def stack_slices_or_single_stack(
        cls,
        slices: Sequence[ImageStackSliceData],
        *,
        memory_type: str,
        gpu_id: int,
    ) -> ImageStackData:
        """Stack slices, or pass through one payload already shaped as a stack."""
        if len(slices) == 1:
            candidate = slices[0]
            if cls.is_unambiguous_stack(candidate):
                return MemoryConversionSource.DETECTED.conversion(
                    candidate,
                    target_type=memory_type,
                    gpu_id=gpu_id,
                ).materialize()
        return cls.for_slices(slices).stack(
            slices=slices,
            memory_type=memory_type,
            gpu_id=gpu_id,
        )

    @classmethod
    def unstack_with_layout_source(
        cls,
        array: ImageStackData,
        *,
        memory_type: str,
        gpu_id: int,
        layout_source: ImageStackData | None = None,
    ) -> tuple[ImageStackSliceData, ...]:
        """Unstack an array using an optional separate value for layout selection."""
        layout_value = array if layout_source is None else layout_source
        return cls.for_stack(layout_value).unstack(
            array=array,
            memory_type=memory_type,
            gpu_id=gpu_id,
        )

    @classmethod
    def stack_function_result_for_input_stack(
        cls,
        result: ImageStackData,
        *,
        input_stack: ImageStackData,
        memory_type: str,
        gpu_id: int,
    ) -> ImageStackData:
        """Return a main-flow stack for one function result in an input-stack domain."""
        result_shape = tuple(np.shape(result))
        input_shape = tuple(np.shape(input_stack))
        if (
            result_shape
            and input_shape
            and (
                result_shape[0] == input_shape[0]
                or result_shape[0] == 1
                or cls.is_unambiguous_stack(result)
            )
            and any(
                layout_type.shape_role.matches_stack(result)
                for layout_type in cls.__registry__.values()
            )
        ):
            return MemoryConversionSource.DETECTED.conversion(
                result,
                target_type=memory_type,
                gpu_id=gpu_id,
            ).materialize()
        return cls.for_slices((result,)).stack(
            slices=(result,),
            memory_type=memory_type,
            gpu_id=gpu_id,
        )

    @classmethod
    def is_slice(cls, candidate: ImageStackData) -> bool:
        """Return whether a value is one file-level image payload."""
        return any(
            layout_type.shape_role.matches_slice(candidate)
            for layout_type in cls.__registry__.values()
        )

    @classmethod
    def is_unambiguous_slice(cls, candidate: ImageStackData) -> bool:
        """Return whether a value is a slice without an explicit stack axis."""
        return cls.is_slice(candidate) and not cls.is_unambiguous_stack(candidate)

    @classmethod
    def is_unambiguous_stack(cls, candidate: ImageStackData) -> bool:
        """Return whether a value carries an explicit OpenHCS outer stack axis."""
        return any(
            layout_type.accepts_single_candidate_stack(candidate)
            for layout_type in cls.__registry__.values()
        )

    @classmethod
    def accepts_single_candidate_stack(cls, candidate: ImageStackData) -> bool:
        """Return True when this layout can own a lone candidate as a stack."""
        if not cls.shape_role.matches_stack(candidate):
            return False
        if not cls.disambiguate_single_candidate_stack_from_slices:
            return True
        return not ImageStackLayout.is_slice(candidate)

    def stack(
        self,
        *,
        slices: Sequence[ImageStackSliceData],
        memory_type: str,
        gpu_id: int,
    ) -> ImageStackData:
        """Stack per-file image slices into an OpenHCS main-flow payload."""
        if self.stable_slice_shape_error is None:
            return stack_slices(
                slices=list(slices),
                memory_type=memory_type,
                gpu_id=gpu_id,
            )
        return self.stack_stable_numpy_slices(
            slices=slices,
            memory_type=memory_type,
            gpu_id=gpu_id,
        )

    def stack_stable_numpy_slices(
        self,
        *,
        slices: Sequence[ImageStackSliceData],
        memory_type: str,
        gpu_id: int,
    ) -> ImageStackData:
        """Stack slices that must all share the same native numpy shape."""
        numpy_slices = [
            NumpySliceConversion(slice_data, gpu_id).array()
            for slice_data in slices
        ]
        slice_shapes = {tuple(slice_data.shape) for slice_data in numpy_slices}
        if len(slice_shapes) != 1:
            if self.stable_slice_shape_error is None:
                raise ValueError(
                    "OpenHCS image stacks require stable slice shapes; "
                    f"got {[slice_data.shape for slice_data in numpy_slices]!r}."
                )
            raise ValueError(
                f"{self.stable_slice_shape_error}; "
                f"got {[slice_data.shape for slice_data in numpy_slices]!r}."
            )
        stacked = np.stack(numpy_slices)
        return MemoryConversionSource.NUMPY.conversion(
            stacked,
            target_type=memory_type,
            gpu_id=gpu_id,
        ).materialize()

    def unstack(
        self,
        *,
        array: ImageStackData,
        memory_type: str,
        gpu_id: int,
    ) -> list[ImageStackSliceData]:
        """Split an OpenHCS main-flow payload into per-file image slices."""
        from openhcs.core.runtime_values import (
            RuntimeArrayPayload,
            image_payload_slice_context,
            runtime_array_operand,
        )

        array_data = MemoryConversionSource.DETECTED.conversion(
            runtime_array_operand(array),
            target_type=memory_type,
            gpu_id=gpu_id,
        ).materialize()
        if isinstance(array, RuntimeArrayPayload):
            return [
                image_payload_slice_context(array, array_data[index], index)
                for index in range(array_data.shape[0])
            ]
        return [array_data[index] for index in range(array_data.shape[0])]


class GrayscaleImageStackLayout(ImageStackLayout):
    """OpenHCS grayscale stacks shaped (N, H, W)."""

    layout_key = "grayscale"
    shape_role = GrayscaleImageShapeRole

    def unstack(
        self,
        *,
        array: ImageStackData,
        memory_type: str,
        gpu_id: int,
    ) -> list[ImageStackSliceData]:
        return unstack_slices(
            array=array,
            memory_type=memory_type,
            gpu_id=gpu_id,
            validate_slices=True,
        )


class ColorImageStackLayout(ImageStackLayout):
    """OpenHCS color stacks shaped (N, H, W, C)."""

    layout_key = "color"
    shape_role = ColorImageShapeRole
    stable_slice_shape_error = "OpenHCS color image stacks require stable HWC shape"


class GrayscaleVolumeStackLayout(ImageStackLayout):
    """OpenHCS grayscale volume stacks shaped (N, Z, H, W)."""

    layout_key = "grayscale_volume"
    shape_role = GrayscaleVolumeShapeRole
    stable_slice_shape_error = "OpenHCS grayscale volume stacks require stable ZYX shape"
    disambiguate_single_candidate_stack_from_slices = False


class ColorVolumeStackLayout(ImageStackLayout):
    """OpenHCS color volume stacks shaped (N, Z, H, W, C)."""

    layout_key = "color_volume"
    shape_role = ColorVolumeShapeRole
    stable_slice_shape_error = "OpenHCS color volume stacks require stable ZYXC shape"


class ChannelFirstVolumeStackLayout(ImageStackLayout):
    """OpenHCS channel-first volume stacks shaped (N, C, Z, H, W)."""

    layout_key = "channel_first_volume"
    shape_role = ChannelFirstVolumeShapeRole
    stable_slice_shape_error = (
        "OpenHCS channel-first volume stacks require stable CZYX shape"
    )
