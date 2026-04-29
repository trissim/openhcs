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
)


@dataclass(frozen=True, slots=True)
class CellProfilerPlaneGeometry:
    """One CellProfiler XY plane coordinate system."""

    shape: tuple[int, int]

    @classmethod
    def from_image_plane(cls, image: np.ndarray) -> "CellProfilerPlaneGeometry":
        if not hasattr(image, "ndim") or image.ndim not in {2, 3}:
            raise ValueError(
                "CellProfiler image planes must be 2D grayscale or HWC color; "
                f"got shape {getattr(image, 'shape', None)!r}."
            )
        if image.ndim == 3 and not is_color_image_slice(image):
            raise ValueError(
                "CellProfiler 3D image planes must be HWC color; got shape "
                f"{getattr(image, 'shape', None)!r}."
            )
        return cls(tuple(int(axis) for axis in image.shape[:2]))

    def binary_mask(
        self,
        mask: np.ndarray,
        *,
        threshold: float = 0.5,
        labels: bool = False,
    ) -> np.ndarray:
        return align_binary_mask_to_shape(
            binary_mask_plane(mask, threshold=threshold, labels=labels),
            self.shape,
        )

    def label_plane(self, labels: np.ndarray) -> np.ndarray:
        return align_label_plane_to_shape(labels.astype(np.int32), self.shape)


@dataclass(frozen=True, slots=True)
class CellProfilerImageMaskPlane:
    """One image plane paired with a binary mask in the same XY geometry."""

    image: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        image_shape = CellProfilerPlaneGeometry.from_image_plane(self.image).shape
        if self.mask.shape != image_shape:
            raise ValueError(
                "CellProfilerImageMaskPlane mask shape must match image XY shape; "
                f"got mask {self.mask.shape!r} for image {image_shape!r}."
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
    if len(mask_planes) not in {1, len(image_planes)}:
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


def _is_stack_payload(payload: Any) -> bool:
    return is_grayscale_image_stack(payload) or is_color_image_stack(payload)
