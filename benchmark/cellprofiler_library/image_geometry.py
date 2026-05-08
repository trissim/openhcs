"""Shared CellProfiler image-plane geometry semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from openhcs.core.runtime_values import image_payload_data


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
    array = collapse_singleton_plane_stack(np.asarray(payload))
    if array.ndim == 2:
        return array
    if is_color_image_slice(array) and array.shape[-1] >= 3:
        color = array[..., :3]
        if np.all(color == color[..., :1]):
            return array[..., 0]
    raise ValueError(
        f"CellProfiler requires a 2-D grayscale {name} plane or replicated "
        f"RGB/RGBA grayscale plane, got shape {array.shape!r}."
    )


def _is_stack_payload(payload: Any) -> bool:
    return is_grayscale_image_stack(payload) or is_color_image_stack(payload)
