"""Object-label image rendering for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta


class ImageMode(Enum):
    BINARY = "binary"
    GRAYSCALE = "grayscale"
    COLOR = "color"
    UINT16 = "uint16"


class ImageModeRenderer(ABC, metaclass=AutoRegisterMeta):
    """Render object labels for one closed ImageMode case."""

    __registry_key__ = "image_mode_label"
    __skip_if_no_key__ = True
    image_mode_label: ClassVar[str | None] = None
    image_mode: ClassVar[ImageMode | None] = None

    @classmethod
    def for_image_mode(cls, image_mode: ImageMode) -> "ImageModeRenderer":
        return cls.__registry__[image_mode.value]()

    @abstractmethod
    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        """Return one rendered image payload for the requested ImageMode."""


class BinaryImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.BINARY
    image_mode_label = image_mode.value

    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        del colormap_value
        return (labels > 0).astype(np.float32)


class GrayscaleImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.GRAYSCALE
    image_mode_label = image_mode.value

    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        del colormap_value
        max_label = labels.max()
        if max_label > 0:
            return labels.astype(np.float32) / max_label
        return np.zeros(labels.shape, dtype=np.float32)


class ColorImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.COLOR
    image_mode_label = image_mode.value

    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        max_label = labels.max()
        colors = object_label_colormap(colormap_value, max_label)
        pixel_data = colors[labels]
        return (
            np.float32(0.299) * pixel_data[..., 0]
            + np.float32(0.587) * pixel_data[..., 1]
            + np.float32(0.114) * pixel_data[..., 2]
        ).astype(np.float32, copy=False)


class Uint16ImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.UINT16
    image_mode_label = image_mode.value

    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        del colormap_value
        return labels.astype(np.int32, copy=False)


def object_label_colormap(colormap_name: str, num_labels: int) -> np.ndarray:
    """Generate colors for object labels using a matplotlib colormap."""
    from matplotlib import colormaps

    cmap = colormaps.get_cmap(colormap_name)
    colors = np.zeros((num_labels + 1, 3), dtype=np.float32)
    for index in range(1, num_labels + 1):
        colors[index] = cmap(index / max(num_labels, 1))[:3]
    return colors
