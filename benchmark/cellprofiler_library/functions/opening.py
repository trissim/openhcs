"""
Converted from CellProfiler: Opening
Morphological opening operation (erosion followed by dilation)
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class StructuringElement(str, Enum):
    DISK = "disk"
    SQUARE = "square"
    DIAMOND = "diamond"
    OCTAGON = "octagon"
    STAR = "star"


class StructuringElementFactory(ABC, metaclass=AutoRegisterMeta):
    """Create one skimage structuring element for a closed enum case."""

    __registry_key__ = "structuring_element"
    __skip_if_no_key__ = True
    structuring_element: ClassVar[StructuringElement | None] = None

    @classmethod
    def for_structuring_element(
        cls,
        structuring_element: StructuringElement,
    ) -> "StructuringElementFactory":
        return cls.__registry__[structuring_element]()

    @abstractmethod
    def build(self, size: int) -> np.ndarray:
        """Return the skimage structuring element for one closed case."""


class DiskStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.DISK

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import disk

        return disk(size)


class SquareStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.SQUARE

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import square

        return square(size)


class DiamondStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.DIAMOND

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import diamond

        return diamond(size)


class OctagonStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.OCTAGON

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import octagon

        return octagon(size, size)


class StarStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.STAR

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import star

        return star(size)


def _coerce_structuring_element(
    structuring_element: StructuringElement | str,
) -> StructuringElement:
    return (
        structuring_element
        if isinstance(structuring_element, StructuringElement)
        else StructuringElement(structuring_element)
    )


@numpy(contract=ProcessingContract.PURE_2D)
def opening(
    image: np.ndarray,
    structuring_element: StructuringElement = StructuringElement.DISK,
    size: int = 3,
) -> np.ndarray:
    """
    Apply morphological opening to an image.
    
    Opening is erosion followed by dilation. It removes small bright spots
    (noise) and smooths object boundaries while preserving object size.
    
    Args:
        image: Input image with shape (H, W)
        structuring_element: Shape of the structuring element.
            Options: "disk", "square", "diamond", "octagon", "star"
        size: Size of the structuring element (radius for disk, side length for square, etc.)
    
    Returns:
        Opened image with shape (H, W)
    """
    from skimage.morphology import opening as skimage_opening

    resolved_structuring_element = _coerce_structuring_element(structuring_element)
    selem = StructuringElementFactory.for_structuring_element(
        resolved_structuring_element
    ).build(size)
    result = skimage_opening(image, selem)
    return result.astype(image.dtype)
