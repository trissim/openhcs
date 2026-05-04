"""Shared CellProfiler morphology structuring-element semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta


class StructuringElement(str, Enum):
    """CellProfiler 2D structuring-element shapes."""

    DISK = "disk"
    SQUARE = "square"
    DIAMOND = "diamond"
    OCTAGON = "octagon"
    STAR = "star"


class StructuringElementFactory(ABC, metaclass=AutoRegisterMeta):
    """Create one skimage structuring element for a closed enum case."""

    __registry_key__ = "structuring_element_label"
    __skip_if_no_key__ = True
    structuring_element_label: ClassVar[str | None] = None
    structuring_element: ClassVar[StructuringElement | None] = None

    @classmethod
    def for_structuring_element(
        cls,
        structuring_element: StructuringElement,
    ) -> "StructuringElementFactory":
        return cls.__registry__[structuring_element.value]()

    @abstractmethod
    def build(self, size: int) -> np.ndarray:
        """Return the skimage structuring element for one closed case."""


class DiskStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.DISK
    structuring_element_label = structuring_element.value

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import disk

        return disk(size)


class SquareStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.SQUARE
    structuring_element_label = structuring_element.value

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import square

        return square(size)


class DiamondStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.DIAMOND
    structuring_element_label = structuring_element.value

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import diamond

        return diamond(size)


class OctagonStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.OCTAGON
    structuring_element_label = structuring_element.value

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import octagon

        return octagon(size, size)


class StarStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.STAR
    structuring_element_label = structuring_element.value

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import star

        return star(size)


def coerce_structuring_element(
    structuring_element: StructuringElement | str,
) -> StructuringElement:
    """Coerce CellProfiler setting text into the closed shape enum."""
    return (
        structuring_element
        if isinstance(structuring_element, StructuringElement)
        else StructuringElement(structuring_element.casefold())
    )


def build_structuring_element(
    structuring_element: StructuringElement | str,
    size: int,
) -> np.ndarray:
    """Build the requested skimage structuring element."""
    if size <= 0:
        raise ValueError(f"Structuring element size must be positive: {size!r}")
    resolved_structuring_element = coerce_structuring_element(structuring_element)
    return StructuringElementFactory.for_structuring_element(
        resolved_structuring_element
    ).build(size)
