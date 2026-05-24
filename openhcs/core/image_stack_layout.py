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

    data: Any
    source_type: str
    target_type: str
    gpu_id: int

    def materialize(self) -> Any:
        if self.source_type == self.target_type:
            return self.data
        return convert_memory(
            data=self.data,
            source_type=self.source_type,
            target_type=self.target_type,
            gpu_id=self.gpu_id,
        )


class MemoryConversionSource(Enum):
    """Nominal source-domain variants for memory conversion requests."""

    DETECTED = "detected"
    NUMPY = "numpy"

    def conversion(
        self,
        data: Any,
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

    slice_data: Any
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

    @classmethod
    def slice_predicate(cls, value: Any) -> bool:
        return cls.shape_role.matches_slice(value)

    @classmethod
    def stack_predicate(cls, value: Any) -> bool:
        return cls.shape_role.matches_stack(value)

    @classmethod
    def for_slices(cls, slices: Sequence[Any]) -> "ImageStackLayout":
        return ImageStackLayoutSelection(
            matches=lambda layout_type: all(
                layout_type.slice_predicate(slice_data)
                for slice_data in slices
            ),
            failure_message=(
                "OpenHCS image stacks require all loaded slices to be either 2D "
                "grayscale images, ZYX grayscale volumes, HWC color images, "
                "ZYXC color volumes, or CZYX channel-first volumes; "
                "got shapes "
                f"{[getattr(slice_data, 'shape', None) for slice_data in slices]!r}."
            ),
        ).select()

    @classmethod
    def for_stack(cls, array: Any) -> "ImageStackLayout":
        return ImageStackLayoutSelection(
            matches=lambda layout_type: layout_type.stack_predicate(array),
            failure_message=(
                "OpenHCS image stack must be shaped (N, H, W), (N, Z, H, W), "
                "(N, H, W, C), (N, Z, H, W, C), or (N, C, Z, H, W), "
                f"got {getattr(array, 'shape', 'unknown')}."
            ),
        ).select()

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
    def unstack_result_for_source_slices(
        cls,
        array: Any,
        *,
        source_slice_shapes: Sequence[tuple[int, ...]],
        memory_type: str,
        gpu_id: int,
    ) -> list[Any]:
        """Unstack runtime output while preserving source-slice shape domains."""
        output_shape = tuple(getattr(array, "shape", ()))
        if output_shape and output_shape in set(source_slice_shapes):
            return [
                MemoryConversionSource.DETECTED.conversion(
                    array,
                    target_type=memory_type,
                    gpu_id=gpu_id,
                ).materialize()
            ]
        return cls.for_stack(array).unstack(
            array=array,
            memory_type=memory_type,
            gpu_id=gpu_id,
        )

    @classmethod
    def unstack_with_layout_source(
        cls,
        array: Any,
        *,
        memory_type: str,
        gpu_id: int,
        layout_source: Any | None = None,
    ) -> tuple[Any, ...]:
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
        result: Any,
        *,
        input_stack: Any,
        memory_type: str,
        gpu_id: int,
    ) -> Any:
        """Return a main-flow stack for one function result in an input-stack domain."""
        result_shape = tuple(getattr(result, "shape", ()))
        input_shape = tuple(getattr(input_stack, "shape", ()))
        if (
            result_shape
            and input_shape
            and result_shape[0] == input_shape[0]
            and any(
                layout_type.stack_predicate(result)
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
    def _is_unambiguous_single_stack(cls, candidate: Any) -> bool:
        """Return True when one candidate is a stack and not also a valid slice."""
        return any(
            layout_type.accepts_single_candidate_stack(candidate)
            for layout_type in cls.__registry__.values()
        )

    @classmethod
    def accepts_single_candidate_stack(cls, candidate: Any) -> bool:
        """Return True when this layout can own a lone candidate as a stack."""
        return cls.stack_predicate(candidate) and not any(
            layout_type.slice_predicate(candidate)
            for layout_type in ImageStackLayout.__registry__.values()
        )

    @classmethod
    def is_unambiguous_stack(cls, candidate: Any) -> bool:
        """Return whether a value carries an explicit OpenHCS outer stack axis."""
        return any(
            layout_type.accepts_single_candidate_stack(candidate)
            for layout_type in ImageStackLayout.__registry__.values()
        )

    def stack(
        self,
        *,
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
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
        slices: Sequence[Any],
        memory_type: str,
        gpu_id: int,
    ) -> Any:
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
        array: Any,
        memory_type: str,
        gpu_id: int,
    ) -> list[Any]:
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
    shape_role = ColorImageShapeRole
    stable_slice_shape_error = "OpenHCS color image stacks require stable HWC shape"


class GrayscaleVolumeStackLayout(ImageStackLayout):
    """OpenHCS grayscale volume stacks shaped (N, Z, H, W)."""

    layout_key = "grayscale_volume"
    shape_role = GrayscaleVolumeShapeRole
    stable_slice_shape_error = "OpenHCS grayscale volume stacks require stable ZYX shape"

    @classmethod
    def accepts_single_candidate_stack(cls, candidate: Any) -> bool:
        return cls.stack_predicate(candidate)


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
