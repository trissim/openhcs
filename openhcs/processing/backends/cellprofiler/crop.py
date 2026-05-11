"""Crop geometry semantics for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.memory.decorators import numpy as numpy_decorator
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.runtime_semantics import coerce_enum
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
    image_payload_with_context,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.crop_settings import (
    CropShape,
    CroppingMethod,
    RemovalMethod,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer

CropBoundaryPair = tuple[int | None, int | None] | None


@dataclass(frozen=True, slots=True)
class CropMeasurement:
    """Measurements from one Crop invocation."""

    slice_index: int
    original_area: int
    area_retained: int
    fraction_retained: float


@dataclass(frozen=True, slots=True)
class CropMaskRequest:
    """Nominal crop-mask construction request."""

    orig_image_pixels: np.ndarray
    mask_plane: np.ndarray | None
    crop_shape: CropShape
    cropping_method: CroppingMethod
    left_right_rectangle_positions: CropBoundaryPair
    top_bottom_rectangle_positions: CropBoundaryPair
    ellipse_center: tuple[float, float] | None
    ellipse_x_radius: float | None
    ellipse_y_radius: float | None
    cropping_labels: Any | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "crop_shape",
            coerce_enum(CropShape, self.crop_shape, "Crop.crop_shape"),
        )
        object.__setattr__(
            self,
            "cropping_method",
            coerce_enum(
                CroppingMethod,
                self.cropping_method,
                "Crop.cropping_method",
            ),
        )


@dataclass(frozen=True, slots=True)
class CropSpatialBounds:
    """Spatial context for a crop output in the parent image domain."""

    offset_yx: tuple[int, int]
    output_shape_yx: tuple[int, int]
    first_last_yx: tuple[int, int, int, int] | None


class CropShapeMaskStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy family for CellProfiler Crop shape modes."""

    __registry_key__ = "crop_shape_label"
    __skip_if_no_key__ = True
    crop_shape_label: ClassVar[str | None] = None
    crop_shape: ClassVar[CropShape | None] = None

    @classmethod
    def for_shape(cls, crop_shape: CropShape) -> "CropShapeMaskStrategy":
        strategy_type = cls.__registry__.get(crop_shape.value)
        if strategy_type is None:
            raise NotImplementedError(
                f"Unsupported CellProfiler Crop shape {crop_shape.value!r}."
            )
        return strategy_type()

    @abstractmethod
    def mask(self, request: CropMaskRequest) -> np.ndarray:
        """Return a boolean crop mask for one shape mode."""

    def validate_crop_mask(
        self,
        mask: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        """Validate and normalize a shape-mode crop mask against input image XY."""
        crop_mask = np.asarray(mask).astype(bool)
        if crop_mask.shape != image.shape[:2]:
            raise ValueError(
                "Crop mask shape must match input image XY shape; "
                f"got mask {crop_mask.shape!r} for image {image.shape[:2]!r}."
            )
        return crop_mask


class PreviousCroppingMaskStrategy(CropShapeMaskStrategy):
    """Use the prior Crop sidecar mask."""

    crop_shape = CropShape.CROPPING
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        if request.mask_plane is None:
            raise ValueError("Crop Previous cropping mode requires a crop-mask plane.")
        return self.validate_crop_mask(request.mask_plane, request.orig_image_pixels)


class ImageMaskCropMaskStrategy(CropShapeMaskStrategy):
    """Use a supplied image mask."""

    crop_shape = CropShape.IMAGE
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        if request.mask_plane is None:
            raise ValueError("Crop image-mask mode requires a mask-image plane.")
        return self.validate_crop_mask(request.mask_plane > 0, request.orig_image_pixels)


class ObjectMaskCropMaskStrategy(CropShapeMaskStrategy):
    """Use supplied object labels as the crop mask."""

    crop_shape = CropShape.OBJECTS
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        if request.cropping_labels is None:
            raise ValueError("Crop object-mask mode requires cropping_labels.")
        return self.validate_crop_mask(
            object_label_crop_mask(request.cropping_labels, request.orig_image_pixels),
            request.orig_image_pixels,
        )


class RectangleCropMaskStrategy(CropShapeMaskStrategy):
    """Build a rectangular coordinate crop mask."""

    crop_shape = CropShape.RECTANGLE
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        require_coordinate_cropping(request)
        left, right = rectangle_pair(
            request.left_right_rectangle_positions,
            "left_right_rectangle_positions",
        )
        top, bottom = rectangle_pair(
            request.top_bottom_rectangle_positions,
            "top_bottom_rectangle_positions",
        )
        return rectangle_cropping(
            request.orig_image_pixels,
            (left, right, top, bottom),
        )


class EllipseCropMaskStrategy(CropShapeMaskStrategy):
    """Build an elliptical coordinate crop mask."""

    crop_shape = CropShape.ELLIPSE
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        require_coordinate_cropping(request)
        if (
            request.ellipse_center is None
            or request.ellipse_x_radius is None
            or request.ellipse_y_radius is None
        ):
            raise ValueError("Crop ellipse mode requires center and X/Y radii.")
        return ellipse_cropping(
            request.orig_image_pixels,
            float_pair(request.ellipse_center, "ellipse_center"),
            (float(request.ellipse_x_radius), float(request.ellipse_y_radius)),
        )


@dataclass(frozen=True, slots=True)
class CropImageRequest:
    """Nominal request for CellProfiler Crop row/column projection."""

    image: np.ndarray
    crop_mask: np.ndarray
    crop_internal: bool = False

    def cropped_pixels(self) -> np.ndarray:
        i_histogram = self.crop_mask.sum(axis=1)
        i_cumsum = np.cumsum(i_histogram != 0)
        j_histogram = self.crop_mask.sum(axis=0)
        j_cumsum = np.cumsum(j_histogram != 0)
        if i_cumsum[-1] == 0:
            return np.zeros((0, 0), dtype=self.image.dtype)
        if self.crop_internal:
            i_keep = np.argwhere(i_histogram > 0).flatten()
            j_keep = np.argwhere(j_histogram > 0).flatten()
            return self.image[i_keep, :][:, j_keep].copy()

        i_first = int(np.argwhere(i_cumsum == 1)[0][0])
        i_last = int(np.argwhere(i_cumsum == i_cumsum.max())[0][0])
        j_first = int(np.argwhere(j_cumsum == 1)[0][0])
        j_last = int(np.argwhere(j_cumsum == j_cumsum.max())[0][0])
        return self.image[i_first : i_last + 1, j_first : j_last + 1].copy()


@dataclass(frozen=True, slots=True)
class CropRequest:
    """Executable CellProfiler Crop request."""

    image: np.ndarray
    mask_plane: np.ndarray | None = None
    crop_shape: CropShape | str = CropShape.RECTANGLE
    cropping_method: CroppingMethod | str = CroppingMethod.COORDINATES
    removal_method: RemovalMethod | str = RemovalMethod.NO
    left_right_rectangle_positions: CropBoundaryPair = None
    top_bottom_rectangle_positions: CropBoundaryPair = None
    ellipse_center: tuple[float, float] | None = None
    ellipse_x_radius: float | None = None
    ellipse_y_radius: float | None = None
    cropping_labels: Any | None = None

    def execute(self) -> tuple[np.ndarray, np.ndarray, CropMeasurement]:
        input_pixels = image_payload_data(self.image)
        input_metadata = image_payload_metadata(self.image)
        orig_image_pixels, input_mask_plane = split_crop_input(input_pixels)
        if self.mask_plane is not None:
            input_mask_plane = image_payload_data(self.mask_plane)
        request = CropMaskRequest(
            orig_image_pixels=orig_image_pixels,
            mask_plane=input_mask_plane,
            crop_shape=self.crop_shape,
            cropping_method=self.cropping_method,
            left_right_rectangle_positions=self.left_right_rectangle_positions,
            top_bottom_rectangle_positions=self.top_bottom_rectangle_positions,
            ellipse_center=self.ellipse_center,
            ellipse_x_radius=self.ellipse_x_radius,
            ellipse_y_radius=self.ellipse_y_radius,
            cropping_labels=self.cropping_labels,
        )
        removal_method = coerce_enum(
            RemovalMethod,
            self.removal_method,
            "Crop.removal_method",
        )
        cropping = CropShapeMaskStrategy.for_shape(request.crop_shape).mask(request)
        cropped_mask = cropped_mask_for(cropping, None, removal_method)
        cropped_pixel_data = cropped_image_pixels(
            orig_image_pixels,
            cropping,
            cropped_mask,
            removal_method,
        )
        input_shape_yx = tuple(int(value) for value in orig_image_pixels.shape[:2])
        crop_bounds = crop_spatial_bounds(cropping, removal_method)
        output_metadata = crop_output_metadata(
            input_metadata,
            input_shape_yx=input_shape_yx,
            bounds=crop_bounds,
        )

        original_area = int(np.prod(orig_image_pixels.shape[:2]))
        area_retained = int(np.sum(cropping))
        measurements = CropMeasurement(
            slice_index=0,
            original_area=original_area,
            area_retained=area_retained,
            fraction_retained=area_retained / original_area if original_area else 0.0,
        )
        if not removal_method.removes_empty_rows_or_columns:
            output_metadata = replace(output_metadata, mask_defines_border=False)
        return (
            image_payload_with_context(
                cropped_pixel_data,
                mask=cropped_mask,
                metadata=output_metadata,
            ),
            cropping,
            measurements,
        )


def require_coordinate_cropping(request: CropMaskRequest) -> None:
    """Validate that a shape strategy is configured for coordinate cropping."""
    if request.cropping_method.is_coordinate_based:
        return
    raise NotImplementedError(
        f"Headless OpenHCS execution supports coordinate Crop, not "
        f"{request.cropping_method.value!r}."
    )


def ellipse_cropping(
    orig_image_pixels: np.ndarray,
    ellipse_center: tuple[float, float],
    ellipse_radius: tuple[float, float],
) -> np.ndarray:
    """Return a CP-compatible elliptical crop mask."""
    x_center, y_center = ellipse_center
    x_radius, y_radius = ellipse_radius
    x_max = orig_image_pixels.shape[1]
    y_max = orig_image_pixels.shape[0]
    if x_radius > y_radius:
        dist_x = np.sqrt(x_radius**2 - y_radius**2)
        dist_y = 0
        major_radius = x_radius
    else:
        dist_x = 0
        dist_y = np.sqrt(y_radius**2 - x_radius**2)
        major_radius = y_radius

    focus_1_x, focus_1_y = (x_center - dist_x, y_center - dist_y)
    focus_2_x, focus_2_y = (x_center + dist_x, y_center + dist_y)
    y, x = np.mgrid[0:y_max, 0:x_max]
    d1 = np.sqrt((x - focus_1_x) ** 2 + (y - focus_1_y) ** 2)
    d2 = np.sqrt((x - focus_2_x) ** 2 + (y - focus_2_y) ** 2)
    return d1 + d2 <= major_radius * 2


def rectangle_cropping(
    orig_image_pixels: np.ndarray,
    bounding_box: tuple[int | None, int | None, int | None, int | None],
) -> np.ndarray:
    """Return a CP-compatible rectangular crop mask."""
    cropping = np.ones(orig_image_pixels.shape[:2], bool)
    left, right, top, bottom = bounding_box
    if left and left > 0:
        cropping[:, :left] = False
    if right and right < cropping.shape[1]:
        cropping[:, right:] = False
    if top and top > 0:
        cropping[:top, :] = False
    if bottom and bottom < cropping.shape[0]:
        cropping[bottom:, :] = False
    return cropping


def cropped_mask_for(
    cropping: np.ndarray,
    mask: np.ndarray | None,
    removal_method: RemovalMethod,
) -> np.ndarray:
    """Return the output mask for a crop result."""
    if not removal_method.removes_empty_rows_or_columns:
        return cropping if mask is None else mask
    if mask is not None:
        return mask
    return CropImageRequest(
        image=cropping,
        crop_mask=cropping,
        crop_internal=removal_method.removes_internal_empty_rows_or_columns,
    ).cropped_pixels()


def cropped_image_pixels(
    orig_image_pixels: np.ndarray,
    cropping: np.ndarray,
    mask: np.ndarray | None,
    removal_method: RemovalMethod,
) -> np.ndarray:
    """Return cropped image pixels using CP removal semantics."""
    if not removal_method.removes_empty_rows_or_columns:
        cropped_pixel_data = orig_image_pixels.copy()
        cropped_pixel_data[~cropping] = 0
        return cropped_pixel_data
    cropped_pixel_data = CropImageRequest(
        image=orig_image_pixels,
        crop_mask=cropping,
        crop_internal=removal_method.removes_internal_empty_rows_or_columns,
    ).cropped_pixels()
    if mask is not None:
        cropped_pixel_data[~mask.astype(bool)] = 0
    return cropped_pixel_data


def crop_spatial_bounds(
    cropping: np.ndarray,
    removal_method: RemovalMethod,
) -> CropSpatialBounds:
    """Return crop bounds in parent-image coordinates."""
    input_shape = tuple(int(value) for value in cropping.shape[:2])
    if not removal_method.removes_empty_rows_or_columns:
        return CropSpatialBounds(
            offset_yx=(0, 0),
            output_shape_yx=input_shape,
            first_last_yx=(0, input_shape[0] - 1, 0, input_shape[1] - 1),
        )

    row_indexes = np.flatnonzero(np.sum(cropping, axis=1) > 0)
    column_indexes = np.flatnonzero(np.sum(cropping, axis=0) > 0)
    if row_indexes.size == 0 or column_indexes.size == 0:
        return CropSpatialBounds(
            offset_yx=(0, 0),
            output_shape_yx=(0, 0),
            first_last_yx=None,
        )

    first_row = int(row_indexes[0])
    last_row = int(row_indexes[-1])
    first_column = int(column_indexes[0])
    last_column = int(column_indexes[-1])
    if removal_method.removes_internal_empty_rows_or_columns:
        output_shape = (int(row_indexes.size), int(column_indexes.size))
    else:
        output_shape = (
            last_row - first_row + 1,
            last_column - first_column + 1,
        )
    return CropSpatialBounds(
        offset_yx=(first_row, first_column),
        output_shape_yx=output_shape,
        first_last_yx=(first_row, last_row, first_column, last_column),
    )


def crop_output_metadata(
    metadata: ImagePayloadMetadata,
    *,
    input_shape_yx: tuple[int, int],
    bounds: CropSpatialBounds,
) -> ImagePayloadMetadata:
    """Return image metadata with exact crop-local physical edge context."""
    if bounds.first_last_yx is None:
        physical_edges = (False, False, False, False)
    else:
        first_row, last_row, first_column, last_column = bounds.first_last_yx
        top_edge, bottom_edge, left_edge, right_edge = (
            metadata.physical_border_edges_for_shape(input_shape_yx)
        )
        physical_edges = (
            top_edge and first_row == 0,
            bottom_edge and last_row == input_shape_yx[0] - 1,
            left_edge and first_column == 0,
            right_edge and last_column == input_shape_yx[1] - 1,
        )
    return metadata.with_spatial_crop(
        input_shape_yx=input_shape_yx,
        output_shape_yx=bounds.output_shape_yx,
        offset_yx=bounds.offset_yx,
        physical_border_edges_yx=physical_edges,
    )


def split_crop_input(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Split an image/mask stack accepted by the CP Crop wrapper."""
    if image.ndim == 2:
        return image, None
    if is_color_image_slice(image):
        return image, None
    if image.ndim == 3 and image.shape[0] >= 1:
        mask_plane = image[1].astype(bool) if image.shape[0] >= 2 else None
        return image[0], mask_plane
    raise ValueError(
        "Crop expects a 2D image or a stacked image/mask payload; "
        f"got shape {getattr(image, 'shape', None)!r}."
    )


@numpy_decorator
@special_inputs("mask_plane")
@special_outputs(
    (
        "crop_measurements",
        csv_materializer(
            fields=[
                "slice_index",
                "original_area",
                "area_retained",
                "fraction_retained",
            ],
            analysis_type="crop",
        ),
    )
)
def crop(
    image: np.ndarray,
    mask_plane: np.ndarray | None = None,
    crop_shape: CropShape | str = CropShape.RECTANGLE,
    cropping_method: CroppingMethod | str = CroppingMethod.COORDINATES,
    removal_method: RemovalMethod | str = RemovalMethod.NO,
    left_right_rectangle_positions: CropBoundaryPair = None,
    top_bottom_rectangle_positions: CropBoundaryPair = None,
    ellipse_center: tuple[float, float] | None = None,
    ellipse_x_radius: float | None = None,
    ellipse_y_radius: float | None = None,
    cropping_labels: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, CropMeasurement]:
    """Crop an image and return its CellProfiler crop-mask sidecar."""
    return CropRequest(
        image=image,
        mask_plane=mask_plane,
        crop_shape=crop_shape,
        cropping_method=cropping_method,
        removal_method=removal_method,
        left_right_rectangle_positions=left_right_rectangle_positions,
        top_bottom_rectangle_positions=top_bottom_rectangle_positions,
        ellipse_center=ellipse_center,
        ellipse_x_radius=ellipse_x_radius,
        ellipse_y_radius=ellipse_y_radius,
        cropping_labels=cropping_labels,
    ).execute()


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def crop_simple(
    image: np.ndarray,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
) -> np.ndarray:
    """Simple rectangular crop by pixel counts removed from each edge."""
    height, width = image.shape
    y_start = max(0, min(crop_top, height - 1))
    y_end = height - crop_bottom if crop_bottom > 0 else height
    y_end = max(y_start + 1, min(y_end, height))
    x_start = max(0, min(crop_left, width - 1))
    x_end = width - crop_right if crop_right > 0 else width
    x_end = max(x_start + 1, min(x_end, width))
    return image[y_start:y_end, x_start:x_end].copy()


def object_label_crop_mask(labels: Any, image: np.ndarray) -> np.ndarray:
    """Return a 2-D foreground mask from dense object-label planes."""
    label_array = object_label_dense_array(labels)
    image_shape = tuple(np.asarray(image).shape[:2])
    if label_array.shape == image_shape:
        return label_array > 0
    if label_array.ndim > 2 and tuple(label_array.shape[-2:]) == image_shape:
        return np.any(label_array > 0, axis=tuple(range(label_array.ndim - 2)))
    return label_array > 0


def rectangle_pair(
    value: CropBoundaryPair,
    name: str,
) -> tuple[int | None, int | None]:
    """Validate and normalize a rectangle boundary pair."""
    if value is None:
        return None, None
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}.")
    return value[0], value[1]


def float_pair(
    value: tuple[float, float],
    name: str,
) -> tuple[float, float]:
    """Validate and normalize a coordinate pair."""
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}.")
    return float(value[0]), float(value[1])


__all__ = [
    "CropBoundaryPair",
    "CropImageRequest",
    "CropMaskRequest",
    "CropMeasurement",
    "CropRequest",
    "CropShapeMaskStrategy",
    "CropSpatialBounds",
    "crop",
    "crop_simple",
    "cropped_image_pixels",
    "cropped_mask_for",
    "crop_output_metadata",
    "crop_spatial_bounds",
    "ellipse_cropping",
    "float_pair",
    "object_label_crop_mask",
    "rectangle_cropping",
    "rectangle_pair",
    "require_coordinate_cropping",
    "split_crop_input",
]
