"""
Converted from CellProfiler: Crop
Original: crop, measure_area_retained_after_cropping, measure_original_image_area, get_measurements
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from benchmark.cellprofiler_semantics.crop import (
    CropShape,
    CroppingMethod,
    RemovalMethod,
)
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.runtime_semantics import coerce_enum
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


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
    left_right_rectangle_positions: tuple[int | None, int | None] | None
    top_bottom_rectangle_positions: tuple[int | None, int | None] | None
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


class CropShapeMaskStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy family for CellProfiler Crop shape modes."""

    __registry_key__ = "crop_shape"
    __skip_if_no_key__ = True
    crop_shape: ClassVar[CropShape | None] = None

    @classmethod
    def for_shape(cls, crop_shape: CropShape) -> "CropShapeMaskStrategy":
        strategy_type = cls.__registry__.get(crop_shape)
        if strategy_type is None:
            raise NotImplementedError(
                f"Unsupported CellProfiler Crop shape {crop_shape.value!r}."
            )
        return strategy_type()

    @abstractmethod
    def mask(self, request: CropMaskRequest) -> np.ndarray:
        """Return a boolean crop mask for one shape mode."""


class PreviousCroppingMaskStrategy(CropShapeMaskStrategy):
    """Use the prior Crop sidecar mask."""

    crop_shape = CropShape.CROPPING

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        if request.mask_plane is None:
            raise ValueError("Crop Previous cropping mode requires a crop-mask plane.")
        return _validate_crop_mask(request.mask_plane, request.orig_image_pixels)


class ImageMaskCropMaskStrategy(CropShapeMaskStrategy):
    """Use a supplied image mask."""

    crop_shape = CropShape.IMAGE

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        if request.mask_plane is None:
            raise ValueError("Crop image-mask mode requires a mask-image plane.")
        return _validate_crop_mask(request.mask_plane > 0, request.orig_image_pixels)


class ObjectMaskCropMaskStrategy(CropShapeMaskStrategy):
    """Use supplied object labels as the crop mask."""

    crop_shape = CropShape.OBJECTS

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        if request.cropping_labels is None:
            raise ValueError("Crop object-mask mode requires cropping_labels.")
        return _validate_crop_mask(
            np.asarray(request.cropping_labels) > 0,
            request.orig_image_pixels,
        )


class RectangleCropMaskStrategy(CropShapeMaskStrategy):
    """Build a rectangular coordinate crop mask."""

    crop_shape = CropShape.RECTANGLE

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        _require_coordinate_cropping(request)
        left, right = _rectangle_pair(
            request.left_right_rectangle_positions,
            "left_right_rectangle_positions",
        )
        top, bottom = _rectangle_pair(
            request.top_bottom_rectangle_positions,
            "top_bottom_rectangle_positions",
        )
        return _get_rectangle_cropping(
            request.orig_image_pixels,
            (left, right, top, bottom),
        )


class EllipseCropMaskStrategy(CropShapeMaskStrategy):
    """Build an elliptical coordinate crop mask."""

    crop_shape = CropShape.ELLIPSE

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        _require_coordinate_cropping(request)
        if (
            request.ellipse_center is None
            or request.ellipse_x_radius is None
            or request.ellipse_y_radius is None
        ):
            raise ValueError("Crop ellipse mode requires center and X/Y radii.")
        return _get_ellipse_cropping(
            request.orig_image_pixels,
            _float_pair(request.ellipse_center, "ellipse_center"),
            (float(request.ellipse_x_radius), float(request.ellipse_y_radius)),
        )


def _require_coordinate_cropping(request: CropMaskRequest) -> None:
    if request.cropping_method.is_coordinate_based:
        return
    raise NotImplementedError(
        f"Headless OpenHCS execution supports coordinate Crop, not "
        f"{request.cropping_method.value!r}."
    )


def _get_ellipse_cropping(
    orig_image_pixels: np.ndarray,
    ellipse_center: tuple[float, float],
    ellipse_radius: tuple[float, float],
) -> np.ndarray:
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


def _get_rectangle_cropping(
    orig_image_pixels: np.ndarray,
    bounding_box: tuple[int | None, int | None, int | None, int | None],
) -> np.ndarray:
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


def _crop_image(
    image: np.ndarray,
    crop_mask: np.ndarray,
    *,
    crop_internal: bool = False,
) -> np.ndarray:
    i_histogram = crop_mask.sum(axis=1)
    i_cumsum = np.cumsum(i_histogram != 0)
    j_histogram = crop_mask.sum(axis=0)
    j_cumsum = np.cumsum(j_histogram != 0)
    if i_cumsum[-1] == 0:
        return np.zeros((0, 0), dtype=image.dtype)
    if crop_internal:
        i_keep = np.argwhere(i_histogram > 0).flatten()
        j_keep = np.argwhere(j_histogram > 0).flatten()
        return image[i_keep, :][:, j_keep].copy()

    i_first = int(np.argwhere(i_cumsum == 1)[0][0])
    i_last = int(np.argwhere(i_cumsum == i_cumsum.max())[0][0])
    j_first = int(np.argwhere(j_cumsum == 1)[0][0])
    j_last = int(np.argwhere(j_cumsum == j_cumsum.max())[0][0])
    return image[i_first : i_last + 1, j_first : j_last + 1].copy()


def _get_cropped_mask(
    cropping: np.ndarray,
    mask: np.ndarray | None,
    removal_method: RemovalMethod,
) -> np.ndarray:
    if not removal_method.removes_empty_rows_or_columns:
        return cropping if mask is None else mask
    if mask is not None:
        return mask
    return _crop_image(
        cropping,
        cropping,
        crop_internal=removal_method.removes_internal_empty_rows_or_columns,
    )


def _get_cropped_image_pixels(
    orig_image_pixels: np.ndarray,
    cropping: np.ndarray,
    mask: np.ndarray | None,
    removal_method: RemovalMethod,
) -> np.ndarray:
    if not removal_method.removes_empty_rows_or_columns:
        cropped_pixel_data = orig_image_pixels.copy()
        cropped_pixel_data[~cropping] = 0
        return cropped_pixel_data
    cropped_pixel_data = _crop_image(
        orig_image_pixels,
        cropping,
        crop_internal=removal_method.removes_internal_empty_rows_or_columns,
    )
    if mask is not None:
        cropped_pixel_data[~mask.astype(bool)] = 0
    return cropped_pixel_data


@numpy
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
    crop_shape: CropShape | str = CropShape.RECTANGLE,
    cropping_method: CroppingMethod | str = CroppingMethod.COORDINATES,
    removal_method: RemovalMethod | str = RemovalMethod.NO,
    left_right_rectangle_positions: tuple[int | None, int | None] | None = None,
    top_bottom_rectangle_positions: tuple[int | None, int | None] | None = None,
    ellipse_center: tuple[float, float] | None = None,
    ellipse_x_radius: float | None = None,
    ellipse_y_radius: float | None = None,
    cropping_labels: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, CropMeasurement]:
    """Crop an image and return its CellProfiler crop_mask sidecar."""
    orig_image_pixels, mask_plane = _split_crop_input(image)
    request = CropMaskRequest(
        orig_image_pixels=orig_image_pixels,
        mask_plane=mask_plane,
        crop_shape=crop_shape,
        cropping_method=cropping_method,
        left_right_rectangle_positions=left_right_rectangle_positions,
        top_bottom_rectangle_positions=top_bottom_rectangle_positions,
        ellipse_center=ellipse_center,
        ellipse_x_radius=ellipse_x_radius,
        ellipse_y_radius=ellipse_y_radius,
        cropping_labels=cropping_labels,
    )
    removal_method = coerce_enum(
        RemovalMethod,
        removal_method,
        "Crop.removal_method",
    )
    cropping = CropShapeMaskStrategy.for_shape(request.crop_shape).mask(request)
    cropped_mask = _get_cropped_mask(cropping, None, removal_method)
    cropped_pixel_data = _get_cropped_image_pixels(
        orig_image_pixels,
        cropping,
        cropped_mask,
        removal_method,
    )

    original_area = int(np.prod(orig_image_pixels.shape[:2]))
    area_retained = int(np.sum(cropping))
    measurements = CropMeasurement(
        slice_index=0,
        original_area=original_area,
        area_retained=area_retained,
        fraction_retained=area_retained / original_area if original_area else 0.0,
    )
    return cropped_pixel_data, cropping, measurements


def _split_crop_input(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
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


def _validate_crop_mask(
    mask: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    crop_mask = np.asarray(mask).astype(bool)
    if crop_mask.shape != image.shape[:2]:
        raise ValueError(
            "Crop mask shape must match input image XY shape; "
            f"got mask {crop_mask.shape!r} for image {image.shape[:2]!r}."
        )
    return crop_mask


def _rectangle_pair(
    value: tuple[int | None, int | None] | None,
    name: str,
) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}.")
    return value[0], value[1]


def _float_pair(
    value: tuple[float, float],
    name: str,
) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}.")
    return float(value[0]), float(value[1])


@numpy(contract=ProcessingContract.PURE_2D)
def crop_simple(
    image: np.ndarray,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
) -> np.ndarray:
    """
    Simple rectangular crop by specifying pixel amounts to remove from each edge.

    Args:
        image: Input image (H, W)
        crop_top: Pixels to remove from top
        crop_bottom: Pixels to remove from bottom
        crop_left: Pixels to remove from left
        crop_right: Pixels to remove from right

    Returns:
        Cropped image
    """
    h, w = image.shape

    y_start = crop_top
    y_end = h - crop_bottom if crop_bottom > 0 else h
    x_start = crop_left
    x_end = w - crop_right if crop_right > 0 else w

    y_start = max(0, min(y_start, h - 1))
    y_end = max(y_start + 1, min(y_end, h))
    x_start = max(0, min(x_start, w - 1))
    x_end = max(x_start + 1, min(x_end, w))

    return image[y_start:y_end, x_start:x_end].copy()
