"""Shared CellProfiler morphology structuring-element semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin


class StructuringElement(str, Enum):
    """CellProfiler morphology structuring-element shapes."""

    DISK = "disk"
    SQUARE = "square"
    DIAMOND = "diamond"
    OCTAGON = "octagon"
    STAR = "star"
    BALL = "ball"


class StructuringElementFactory(
    EnumKeyedStrategyMixin[StructuringElement],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Create one skimage structuring element for a closed enum case."""

    __registry_key__ = "structuring_element_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "structuring_element"
    __enum_label_attr__ = "structuring_element_label"
    structuring_element_label: ClassVar[str | None] = None
    structuring_element: ClassVar[StructuringElement | None] = None

    @classmethod
    def for_structuring_element(
        cls,
        structuring_element: StructuringElement,
    ) -> "StructuringElementFactory":
        return cls.for_enum_member(structuring_element)

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


class BallStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.BALL

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import ball

        return ball(size)


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


def adapt_structuring_element_rank(
    footprint: np.ndarray,
    spatial_rank: int,
) -> np.ndarray:
    """Return a footprint compatible with the target spatial rank.

    CellProfiler image modules can execute plane-wise even when a pipeline
    setting names a volumetric structuring element. A centered section preserves
    the requested morphology for lower-rank planes without inventing per-module
    shape conventions. Lower-rank footprints expand with singleton leading axes
    so they do not erode across spatial axes the setting did not define.
    """
    if spatial_rank <= 0:
        raise ValueError("spatial_rank must be positive.")
    if footprint.ndim == spatial_rank:
        return footprint
    if footprint.ndim > spatial_rank:
        reduced = footprint
        while reduced.ndim > spatial_rank:
            reduced = reduced[reduced.shape[0] // 2]
        return np.asarray(reduced)
    leading_axes = (1,) * (spatial_rank - footprint.ndim)
    return footprint.reshape((*leading_axes, *footprint.shape))
