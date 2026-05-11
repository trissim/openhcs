"""Shared CellProfiler image-plane geometry semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Tuple

import numpy as np

from openhcs.core.aligned_image_payload import (
    aligned_payload_slice,
    payload_slices_for_alignment,
)
from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_grayscale_image_stack,
    is_grayscale_volume_slice,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import image_payload_data
from openhcs.core.runtime_values import image_payload_mask
from openhcs.core.runtime_values import image_payload_metadata
from openhcs.core.runtime_values import image_payload_with_context
from openhcs.core.measurement_image_alignment import ReplicatedChannelMonochromeProjection
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


class TileMethod(Enum):
    WITHIN_CYCLES = "within_cycles"
    ACROSS_CYCLES = "across_cycles"


class PlaceFirst(Enum):
    TOP_LEFT = "top_left"
    BOTTOM_LEFT = "bottom_left"
    TOP_RIGHT = "top_right"
    BOTTOM_RIGHT = "bottom_right"

    @property
    def row_from_bottom(self) -> bool:
        return self.value.startswith("bottom_")

    @property
    def column_from_right(self) -> bool:
        return self.value.endswith("_right")


class TileStyle(Enum):
    ROW = "row"
    COLUMN = "column"


class ResizeMethod(Enum):
    BY_FACTOR = "by_factor"
    TO_SIZE = "to_size"


class InterpolationMethod(Enum):
    NEAREST_NEIGHBOR = "nearest_neighbor"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class FlipMethod(Enum):
    NONE = "none"
    LEFT_TO_RIGHT = "left_to_right"
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTH = "both"


class RotateMethod(Enum):
    NONE = "none"
    ANGLE = "angle"
    COORDINATES = "coordinates"


class AlignmentDirection(Enum):
    HORIZONTALLY = "horizontally"
    VERTICALLY = "vertically"


@dataclass(frozen=True, slots=True)
class TileSettings:
    rows: int
    columns: int
    place_first: PlaceFirst
    tile_style: TileStyle
    meander: bool
    auto_rows: bool
    auto_columns: bool

    def geometry(self, image_count: int) -> "TileGeometry":
        grid_rows, grid_columns = tile_grid_dimensions(
            image_count,
            self.rows,
            self.columns,
            self.auto_rows,
            self.auto_columns,
        )
        return TileGeometry(
            rows=grid_rows,
            columns=grid_columns,
            tile_style=self.tile_style,
            place_first=self.place_first,
            meander=self.meander,
        )


@dataclass(frozen=True, slots=True)
class TileGeometry:
    rows: int
    columns: int
    tile_style: TileStyle
    place_first: PlaceFirst
    meander: bool

    @property
    def tile_count(self) -> int:
        return self.rows * self.columns

    def coordinates(self, image_index: int) -> tuple[int, int]:
        """Return row/column coordinates for one tile index."""
        if self.tile_style == TileStyle.ROW:
            tile_i = int(image_index / self.columns)
            tile_j = image_index % self.columns
            if self.meander and tile_i % 2 == 1:
                tile_j = self.columns - tile_j - 1
        else:
            tile_i = image_index % self.rows
            tile_j = int(image_index / self.rows)
            if self.meander and tile_j % 2 == 1:
                tile_i = self.rows - tile_i - 1

        if self.place_first.row_from_bottom:
            tile_i = self.rows - tile_i - 1
        if self.place_first.column_from_right:
            tile_j = self.columns - tile_j - 1

        return tile_i, tile_j


@dataclass(frozen=True, slots=True)
class ResizeGeometry:
    """CellProfiler resize geometry for pixels and per-pixel validity masks."""

    output_shape: tuple[int, ...]
    interpolation_order: int

    @classmethod
    def from_parameters(
        cls,
        input_shape: tuple[int, ...],
        *,
        resize_method: ResizeMethod,
        resizing_factors: tuple[float, ...],
        specific_shape: tuple[int, ...],
        interpolation: InterpolationMethod,
    ) -> "ResizeGeometry":
        if resize_method is ResizeMethod.BY_FACTOR:
            output_shape = tuple(
                int(np.round(axis_size * factor))
                for axis_size, factor in zip(input_shape, resizing_factors, strict=True)
            )
        else:
            output_shape = specific_shape
        return cls(
            output_shape=output_shape,
            interpolation_order=cls.resolve_interpolation_order(interpolation),
        )

    @staticmethod
    def resolve_interpolation_order(interpolation: InterpolationMethod) -> int:
        if interpolation is InterpolationMethod.NEAREST_NEIGHBOR:
            return 0
        if interpolation is InterpolationMethod.BILINEAR:
            return 1
        if interpolation is InterpolationMethod.BICUBIC:
            return 3
        raise TypeError(f"Unsupported Resize interpolation {interpolation!r}.")

    def resize_pixels(self, pixels: Any) -> np.ndarray:
        import skimage.transform

        return skimage.transform.resize(
            pixels,
            self.output_shape,
            order=self.interpolation_order,
            mode="symmetric",
            preserve_range=True,
        ).astype(np.asarray(pixels).dtype, copy=False)

    def resize_mask(self, mask: Any | None) -> np.ndarray | None:
        import scipy.ndimage as ndi

        if mask is None:
            return None
        mask_array = np.asarray(mask, dtype=bool)
        zoom = tuple(
            output_size / input_size
            for output_size, input_size in zip(
                self.output_shape,
                mask_array.shape,
                strict=True,
            )
        )
        return ndi.zoom(
            mask_array.astype(np.float32),
            zoom,
            order=0,
            mode="constant",
            grid_mode=True,
        ).astype(bool, copy=False)

    def resize_payload(self, image: Any) -> Any:
        pixels = image_payload_data(image)
        output_pixels = self.resize_pixels(pixels)
        return image_payload_with_context(
            output_pixels,
            mask=self.resize_mask(image_payload_mask(image)),
            metadata=image_payload_metadata(image).without_spatial_domain(),
        )


@dataclass(frozen=True, slots=True)
class RotationResult:
    slice_index: int
    rotation_angle: float


@dataclass(frozen=True, slots=True)
class CellProfilerPlaneGeometry:
    """One CellProfiler XY plane coordinate system."""

    shape: tuple[int, int]
    spatial_rank: int = 2

    @classmethod
    def from_image_plane(cls, image: np.ndarray) -> "CellProfilerPlaneGeometry":
        image_array = collapse_singleton_plane_stack(np.asarray(image_payload_data(image)))
        if not hasattr(image_array, "ndim") or image_array.ndim not in {2, 3}:
            raise ValueError(
                "CellProfiler image planes must be 2D grayscale, ZYX grayscale, "
                "or HWC color; "
                f"got shape {getattr(image_array, 'shape', None)!r}."
            )
        if is_grayscale_volume_slice(image_array):
            return cls(
                tuple(int(axis) for axis in image_array.shape[-2:]),
                spatial_rank=3,
            )
        if image_array.ndim == 3 and not is_color_image_slice(image_array):
            raise ValueError(
                "CellProfiler 3D image planes must be HWC color; got shape "
                f"{getattr(image_array, 'shape', None)!r}."
            )
        return cls(tuple(int(axis) for axis in image_array.shape[:2]))

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        if self.spatial_rank == 2:
            return self.shape
        if self.spatial_rank == 3:
            return self.shape
        raise ValueError(f"Unsupported CellProfiler spatial rank {self.spatial_rank}.")

    def binary_mask(
        self,
        mask: np.ndarray,
        *,
        threshold: float = 0.5,
        labels: bool = False,
    ) -> np.ndarray:
        mask_array = binary_mask_plane(mask, threshold=threshold, labels=labels)
        if self.spatial_rank == 3 and mask_array.ndim == 3:
            return align_volume_mask_to_shape(mask_array, self.shape)
        if self.spatial_rank == 3 and mask_array.ndim == 2:
            return align_binary_mask_to_shape(mask_array, self.shape)
        return align_binary_mask_to_shape(mask_array, self.shape)

    def label_plane(self, labels: np.ndarray) -> np.ndarray:
        return align_label_plane_to_shape(labels.astype(np.int32), self.shape)


@dataclass(frozen=True, slots=True)
class CellProfilerImageMaskPlane:
    """One image plane paired with a binary mask in the same XY geometry."""

    image: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        geometry = CellProfilerPlaneGeometry.from_image_plane(self.image)
        if tuple(self.mask.shape[-2:]) != geometry.shape:
            raise ValueError(
                "CellProfilerImageMaskPlane mask shape must match image spatial "
                f"shape; got mask {self.mask.shape!r} for image {geometry.shape!r}."
            )
        if geometry.spatial_rank == 2 and self.mask.ndim != 2:
            raise ValueError(
                "CellProfilerImageMaskPlane 2D images require 2D masks; "
                f"got mask {self.mask.shape!r}."
            )


def aligned_image_mask_planes(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    labels: bool = False,
) -> tuple[CellProfilerImageMaskPlane, ...]:
    """Align a mask payload to each image plane using CellProfiler slice rules."""
    image_planes = payload_slices_for_alignment(image)
    mask_planes = payload_slices_for_alignment(mask)
    if len(image_planes) == 1 and len(mask_planes) > 1:
        image_plane = image_planes[0]
        geometry = CellProfilerPlaneGeometry.from_image_plane(image_plane)
        if geometry.spatial_rank == 3:
            projected_volume_mask = np.stack(
                tuple(
                    geometry.binary_mask(
                        mask_plane,
                        threshold=threshold,
                        labels=labels,
                    )
                    for mask_plane in mask_planes
                ),
                axis=0,
            )
            return (
                CellProfilerImageMaskPlane(
                    image=image_plane,
                    mask=projected_volume_mask,
                ),
            )
        projected_mask = np.any(
            np.stack(
                tuple(
                    geometry.binary_mask(
                        mask_plane,
                        threshold=threshold,
                        labels=labels,
                    )
                    for mask_plane in mask_planes
                )
            ),
            axis=0,
        )
        return (
            CellProfilerImageMaskPlane(
                image=image_plane,
                mask=projected_mask,
            ),
        )
    if len(mask_planes) not in {1, len(image_planes)}:
        projected_mask_planes = _project_volume_mask_planes(
            image_planes,
            mask_planes,
            threshold=threshold,
            labels=labels,
        )
        if projected_mask_planes is not None:
            return projected_mask_planes
        projected_mask_planes = _project_flat_mask_plane_groups(
            image_planes,
            mask_planes,
            threshold=threshold,
            labels=labels,
        )
        if projected_mask_planes is not None:
            return projected_mask_planes
        raise ValueError(
            "CellProfiler mask payload must have one plane or match image plane "
            f"count; got image count {len(image_planes)} and mask count "
            f"{len(mask_planes)}."
        )
    return tuple(
        CellProfilerImageMaskPlane(
            image=image_plane,
            mask=CellProfilerPlaneGeometry.from_image_plane(
                image_plane
            ).binary_mask(
                aligned_payload_slice(mask_planes, plane_index),
                threshold=threshold,
                labels=labels,
            ),
        )
        for plane_index, image_plane in enumerate(image_planes)
    )


def _project_volume_mask_planes(
    image_planes: tuple[Any, ...],
    mask_planes: tuple[Any, ...],
    *,
    threshold: float,
    labels: bool,
) -> tuple[CellProfilerImageMaskPlane, ...] | None:
    """Project stacked volume masks onto each image plane when ranks permit."""
    image_count = len(image_planes)
    if image_count <= 1:
        return None
    mask_arrays = tuple(np.asarray(image_payload_data(mask)) for mask in mask_planes)
    if not mask_arrays or any(mask.ndim < 3 for mask in mask_arrays):
        return None
    if any(mask.shape[0] != image_count for mask in mask_arrays):
        return _project_all_volume_masks_to_image_planes(
            image_planes,
            mask_arrays,
            threshold=threshold,
            labels=labels,
        )
    return tuple(
        CellProfilerImageMaskPlane(
            image=image_plane,
            mask=np.any(
                np.stack(
                    tuple(
                        CellProfilerPlaneGeometry.from_image_plane(
                            image_plane
                        ).binary_mask(
                            mask_array[plane_index],
                            threshold=threshold,
                            labels=labels,
                        )
                        for mask_array in mask_arrays
                    )
                ),
                axis=0,
            ),
        )
        for plane_index, image_plane in enumerate(image_planes)
    )


def _project_flat_mask_plane_groups(
    image_planes: tuple[Any, ...],
    mask_planes: tuple[Any, ...],
    *,
    threshold: float,
    labels: bool,
) -> tuple[CellProfilerImageMaskPlane, ...] | None:
    """Project flattened grouped mask stacks onto matching image-plane indices."""
    image_count = len(image_planes)
    mask_count = len(mask_planes)
    if image_count <= 1 or mask_count <= image_count:
        return None
    if mask_count % image_count != 0:
        return None

    group_count = mask_count // image_count
    return tuple(
        CellProfilerImageMaskPlane(
            image=image_plane,
            mask=np.any(
                np.stack(
                    tuple(
                        CellProfilerPlaneGeometry.from_image_plane(
                            image_plane
                        ).binary_mask(
                            mask_planes[group_index * image_count + plane_index],
                            threshold=threshold,
                            labels=labels,
                        )
                        for group_index in range(group_count)
                    )
                ),
                axis=0,
            ),
        )
        for plane_index, image_plane in enumerate(image_planes)
    )


def _project_all_volume_masks_to_image_planes(
    image_planes: tuple[Any, ...],
    mask_arrays: tuple[np.ndarray, ...],
    *,
    threshold: float,
    labels: bool,
) -> tuple[CellProfilerImageMaskPlane, ...]:
    """Collapse all mask leading axes, then broadcast the XY mask to each image."""
    return tuple(
        CellProfilerImageMaskPlane(
            image=image_plane,
            mask=np.any(
                np.stack(
                    tuple(
                        _project_mask_array_to_geometry(
                            mask_array,
                            CellProfilerPlaneGeometry.from_image_plane(image_plane),
                            threshold=threshold,
                            labels=labels,
                        )
                        for mask_array in mask_arrays
                    )
                ),
                axis=0,
            ),
        )
        for image_plane in image_planes
    )


def _project_mask_array_to_geometry(
    mask_array: np.ndarray,
    geometry: CellProfilerPlaneGeometry,
    *,
    threshold: float,
    labels: bool,
) -> np.ndarray:
    """Project every leading-axis mask plane into one XY geometry."""
    mask_planes = mask_array.reshape((-1, *mask_array.shape[-2:]))
    return np.any(
        np.stack(
            tuple(
                geometry.binary_mask(
                    mask_plane,
                    threshold=threshold,
                    labels=labels,
                )
                for mask_plane in mask_planes
            )
        ),
        axis=0,
    )


def restore_image_mask_planes(
    original_image: np.ndarray,
    masked_planes: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Restore masked image planes to the original image payload rank."""
    if not masked_planes:
        raise ValueError("Cannot restore an empty CellProfiler image plane set.")
    if not _is_stack_payload(original_image) and len(masked_planes) == 1:
        return masked_planes[0]
    return np.stack(masked_planes).astype(masked_planes[0].dtype, copy=False)


def binary_mask_plane(
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    labels: bool = False,
) -> np.ndarray:
    """Convert one CellProfiler mask/label plane to a 2D boolean mask."""
    mask = collapse_singleton_plane_stack(np.asarray(mask))
    if labels:
        return mask > 0
    if is_color_image_slice(mask):
        return np.any(mask > threshold, axis=-1)
    unique_values = np.unique(mask)
    if len(unique_values) <= 2 and set(unique_values).issubset(
        {0, 1, False, True}
    ):
        return mask > 0
    return mask > threshold


def align_binary_mask_to_shape(
    mask: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbor align a boolean mask to an XY shape."""
    if mask.shape == shape:
        return mask.astype(bool, copy=False)
    return resize_nearest(mask.astype(np.uint8), shape).astype(bool)


def align_volume_mask_to_shape(
    mask: np.ndarray,
    shape_yx: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbor align every Z plane of a ZYX boolean mask."""
    if mask.shape[-2:] == shape_yx:
        return mask.astype(bool, copy=False)
    return np.stack(
        tuple(align_binary_mask_to_shape(plane, shape_yx) for plane in mask),
        axis=0,
    )


def align_label_plane_to_shape(
    labels: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbor align a dense label plane to an XY shape."""
    labels = collapse_singleton_plane_stack(np.asarray(labels))
    if labels.shape == shape:
        return labels.astype(np.int32, copy=False)
    return resize_nearest(labels, shape).astype(np.int32)


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a discrete 2D payload without interpolation artifacts."""
    from skimage.transform import resize

    return resize(
        image,
        shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )


def collapse_singleton_plane_stack(payload: Any) -> Any:
    """Collapse one-plane label/mask stacks to CellProfiler's 2D plane form."""
    if hasattr(payload, "ndim") and payload.ndim == 3 and payload.shape[0] == 1:
        return payload[0]
    return payload


def cellprofiler_grayscale_plane(payload: Any, name: str) -> np.ndarray:
    """Return CP's must_be_grayscale pixel plane for one image payload.

    CellProfiler does not split arbitrary RGB images for grayscale modules. It
    accepts a multichannel image only when the first three channels are identical
    and then exposes channel 0 through GrayscaleImage.pixel_data.
    """
    return ReplicatedChannelMonochromeProjection().plane(payload, name=name)


def tile_grid_dimensions(
    image_count: int,
    rows: int,
    columns: int,
    auto_rows: bool,
    auto_columns: bool,
) -> tuple[int, int]:
    """Calculate CellProfiler Tile grid dimensions from auto/manual settings."""
    if auto_rows:
        if auto_columns:
            row_count = int(np.sqrt(image_count))
            column_count = int((image_count + row_count - 1) / row_count)
            return row_count, column_count
        column_count = columns
        row_count = int((image_count + column_count - 1) / column_count)
        return row_count, column_count
    if auto_columns:
        row_count = rows
        column_count = int((image_count + row_count - 1) / row_count)
        return row_count, column_count
    return rows, columns


def put_tile(
    pixels: np.ndarray,
    output_pixels: np.ndarray,
    image_index: int,
    geometry: TileGeometry,
) -> None:
    """Place one image plane into a CellProfiler Tile output montage."""
    tile_height = int(output_pixels.shape[0] / geometry.rows)
    tile_width = int(output_pixels.shape[1] / geometry.columns)

    tile_i, tile_j = geometry.coordinates(image_index)
    tile_i *= tile_height
    tile_j *= tile_width

    img_height = min(tile_height, pixels.shape[0])
    img_width = min(tile_width, pixels.shape[1])

    output_pixels[
        tile_i:(tile_i + img_height),
        tile_j:(tile_j + img_width),
    ] = pixels[:img_height, :img_width]


def tile_output_shape(
    image: np.ndarray,
    output_height: int,
    output_width: int,
) -> tuple[int, ...]:
    """Return CellProfiler Tile output shape for grayscale or color stacks."""
    if image.ndim == 4:
        return (output_height, output_width, image.shape[3])
    return (output_height, output_width)


@numpy
def tile(
    image: np.ndarray,
    rows: int = 8,
    columns: int = 12,
    place_first: PlaceFirst = PlaceFirst.TOP_LEFT,
    tile_style: TileStyle = TileStyle.ROW,
    meander: bool = False,
    auto_rows: bool = False,
    auto_columns: bool = False,
) -> np.ndarray:
    """Tile multiple images together to form a CellProfiler montage image."""
    if image.ndim not in {3, 4}:
        raise ValueError(
            "Tile expects an image stack shaped (N, H, W) or (N, H, W, C), "
            f"got {image.shape!r}."
        )

    num_images = image.shape[0]
    if num_images == 0:
        raise ValueError("No images provided for tiling")

    geometry = TileSettings(
        rows=rows,
        columns=columns,
        place_first=place_first,
        tile_style=tile_style,
        meander=meander,
        auto_rows=auto_rows,
        auto_columns=auto_columns,
    ).geometry(num_images)

    if geometry.tile_count < num_images:
        raise ValueError(
            f"Grid size ({geometry.rows}x{geometry.columns}={geometry.tile_count}) "
            f"is too small for {num_images} images"
        )

    tile_height = image.shape[1]
    tile_width = image.shape[2]
    output_height = tile_height * geometry.rows
    output_width = tile_width * geometry.columns
    output_pixels = np.zeros(
        tile_output_shape(image, output_height, output_width),
        dtype=image.dtype,
    )

    for image_index in range(num_images):
        put_tile(image[image_index], output_pixels, image_index, geometry)

    return output_pixels[np.newaxis, ...]


@numpy(contract=ProcessingContract.PURE_2D)
def resize(
    image: np.ndarray,
    resize_method: ResizeMethod = ResizeMethod.BY_FACTOR,
    resizing_factor_x: float = 0.25,
    resizing_factor_y: float = 0.25,
    specific_width: int = 100,
    specific_height: int = 100,
    interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR,
) -> np.ndarray:
    """Resize a CellProfiler image plane by factor or explicit dimensions."""
    resize_method = coerce_cellprofiler_enum(ResizeMethod, resize_method)
    interpolation = coerce_cellprofiler_enum(InterpolationMethod, interpolation)

    pixels = image_payload_data(image)
    geometry = ResizeGeometry.from_parameters(
        tuple(np.asarray(pixels).shape[:2]),
        resize_method=resize_method,
        resizing_factors=(resizing_factor_y, resizing_factor_x),
        specific_shape=(specific_height, specific_width),
        interpolation=interpolation,
    )
    return geometry.resize_payload(image)


@numpy(contract=ProcessingContract.PURE_3D)
def resize_volumetric(
    image: np.ndarray,
    resize_method: ResizeMethod = ResizeMethod.BY_FACTOR,
    resizing_factor_x: float = 0.25,
    resizing_factor_y: float = 0.25,
    resizing_factor_z: float = 0.25,
    specific_width: int = 100,
    specific_height: int = 100,
    specific_planes: int = 10,
    interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR,
) -> np.ndarray:
    """Resize a CellProfiler ZYX image volume by factor or explicit dimensions."""
    resize_method = coerce_cellprofiler_enum(ResizeMethod, resize_method)
    interpolation = coerce_cellprofiler_enum(InterpolationMethod, interpolation)

    pixels = image_payload_data(image)
    geometry = ResizeGeometry.from_parameters(
        tuple(np.asarray(pixels).shape[:3]),
        resize_method=resize_method,
        resizing_factors=(resizing_factor_z, resizing_factor_y, resizing_factor_x),
        specific_shape=(specific_planes, specific_height, specific_width),
        interpolation=interpolation,
    )
    return geometry.resize_payload(image)


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "rotation_results",
        csv_materializer(
            fields=["slice_index", "rotation_angle"],
            analysis_type="rotation",
        ),
    )
)
def flip_and_rotate(
    image: np.ndarray,
    flip_method: FlipMethod = FlipMethod.NONE,
    rotate_method: RotateMethod = RotateMethod.NONE,
    rotation_angle: float = 0.0,
    first_pixel_x: int = 0,
    first_pixel_y: int = 0,
    second_pixel_x: int = 0,
    second_pixel_y: int = 100,
    alignment_direction: AlignmentDirection = AlignmentDirection.HORIZONTALLY,
    crop_rotated_edges: bool = True,
) -> Tuple[np.ndarray, RotationResult]:
    """Flip and/or rotate a CellProfiler image plane."""
    from scipy.ndimage import rotate as scipy_rotate

    pixel_data = image.copy()

    if flip_method != FlipMethod.NONE:
        if flip_method == FlipMethod.LEFT_TO_RIGHT:
            pixel_data = np.flip(pixel_data, axis=1)
        elif flip_method == FlipMethod.TOP_TO_BOTTOM:
            pixel_data = np.flip(pixel_data, axis=0)
        elif flip_method == FlipMethod.BOTH:
            pixel_data = np.flip(np.flip(pixel_data, axis=1), axis=0)

    angle = 0.0
    if rotate_method != RotateMethod.NONE:
        if rotate_method == RotateMethod.ANGLE:
            angle = rotation_angle
        elif rotate_method == RotateMethod.COORDINATES:
            xdiff = second_pixel_x - first_pixel_x
            ydiff = second_pixel_y - first_pixel_y
            if alignment_direction == AlignmentDirection.VERTICALLY:
                angle = -np.arctan2(ydiff, xdiff) * 180.0 / np.pi
            else:
                angle = np.arctan2(xdiff, ydiff) * 180.0 / np.pi

        if angle != 0.0:
            pixel_data = scipy_rotate(pixel_data, angle, reshape=True, order=1)

            if crop_rotated_edges:
                crop_mask = scipy_rotate(
                    np.ones(image.shape[:2]),
                    angle,
                    reshape=True,
                ) > 0.50

                half = (np.array(crop_mask.shape) // 2).astype(int)

                quartercrop = crop_mask[half[0]:, half[1]:]
                ci = np.cumsum(quartercrop, 0)
                cj = np.cumsum(quartercrop, 1)
                carea_d = ci * cj
                carea_d[quartercrop == 0] = 0

                quartercrop_u = crop_mask[
                    crop_mask.shape[0] - half[0] - 1::-1,
                    half[1]:,
                ]
                ci = np.cumsum(quartercrop_u, 0)
                cj = np.cumsum(quartercrop_u, 1)
                carea_u = ci * cj
                carea_u[quartercrop_u == 0] = 0

                min_shape = min(carea_d.shape[0], carea_u.shape[0])
                carea = carea_d[:min_shape] + carea_u[:min_shape]

                if carea.size > 0:
                    max_carea = np.max(carea)
                    if max_carea > 0:
                        max_area_idx = np.argwhere(carea == max_carea)[0] + half
                        min_i = max(crop_mask.shape[0] - max_area_idx[0] - 1, 0)
                        max_i = max_area_idx[0] + 1
                        min_j = max(crop_mask.shape[1] - max_area_idx[1] - 1, 0)
                        max_j = max_area_idx[1] + 1
                        pixel_data = pixel_data[min_i:max_i, min_j:max_j]

    return (
        pixel_data.astype(np.float32),
        RotationResult(slice_index=0, rotation_angle=angle),
    )


def _is_stack_payload(payload: Any) -> bool:
    return is_grayscale_image_stack(payload) or is_color_image_stack(payload)
