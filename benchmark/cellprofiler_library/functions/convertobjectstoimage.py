"""
Converted from CellProfiler: ConvertObjectsToImage
Original: convert_objects_to_image

Converts object labels to various image representations (binary, grayscale, color, uint16).
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


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
        colors = _get_colormap(colormap_value, max_label)
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


def _get_colormap(colormap_name: str, num_labels: int) -> np.ndarray:
    """Generate colors for labels using matplotlib colormap."""
    from matplotlib import colormaps

    cmap = colormaps.get_cmap(colormap_name)
    
    colors = np.zeros((num_labels + 1, 3), dtype=np.float32)
    for i in range(1, num_labels + 1):
        colors[i] = cmap(i / max(num_labels, 1))[:3]
    return colors


def _coerce_image_mode(image_mode: ImageMode | str) -> ImageMode:
    return coerce_cellprofiler_enum(ImageMode, image_mode)


@numpy
@special_inputs("labels")
def convert_objects_to_image(
    image: np.ndarray,
    labels: np.ndarray,
    image_mode: ImageMode = ImageMode.COLOR,
    colormap_value: str = "jet",
) -> np.ndarray:
    """
    Convert object labels to an image representation.
    
    Args:
        image: Input image (H, W) - used for shape reference
        labels: Object labels (H, W) - integer labels where 0 is background
        image_mode: Output image format (BINARY, GRAYSCALE, COLOR, UINT16)
        colormap_value: Matplotlib colormap name for COLOR mode
    
    Returns:
        Converted image:
        - BINARY: (H, W) boolean mask where objects are True
        - GRAYSCALE: (H, W) float with normalized label values
        - COLOR: (H, W, 3) RGB image with colored objects
        - UINT16: (H, W) integer labels
    """
    del image
    labels = object_label_dense_array(labels, dtype=np.int32)
    resolved_image_mode = _coerce_image_mode(image_mode)
    return ImageModeRenderer.for_image_mode(resolved_image_mode).render(
        labels,
        colormap_value=colormap_value,
    )
