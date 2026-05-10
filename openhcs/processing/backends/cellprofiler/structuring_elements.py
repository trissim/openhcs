"""Shared CellProfiler morphology structuring-element semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
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
    CUBE = "cube"
    OCTAHEDRON = "octahedron"


class StructuringElementDomain(str, Enum):
    """How a structuring element's rank relates to the image domain."""

    MATCHING_RANK = "matching_rank"
    STACKWISE = "stackwise"
    UNSUPPORTED_VOLUMETRIC_TO_PLANE = "unsupported_volumetric_to_plane"


StructuringElementOperation = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class StructuringElementApplication:
    """Typed request for applying one structuring element to an image."""

    image: np.ndarray
    footprint: np.ndarray
    operation: StructuringElementOperation

    @property
    def domain(self) -> StructuringElementDomain:
        if self.footprint.ndim == self.image.ndim:
            return StructuringElementDomain.MATCHING_RANK
        if 1 < self.footprint.ndim < self.image.ndim:
            return StructuringElementDomain.STACKWISE
        return StructuringElementDomain.UNSUPPORTED_VOLUMETRIC_TO_PLANE


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


class CubeStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.CUBE

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import cube

        return cube(size)


class OctahedronStructuringElementFactory(StructuringElementFactory):
    structuring_element = StructuringElement.OCTAHEDRON

    def build(self, size: int) -> np.ndarray:
        from skimage.morphology import octahedron

        return octahedron(size)


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


class StructuringElementApplicationStrategy(
    EnumKeyedStrategyMixin[StructuringElementDomain],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Apply CellProfiler structuring-element rank semantics."""

    __registry_key__ = "domain_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "domain_key"
    __enum_label_attr__ = "domain_label"
    domain_key: ClassVar[StructuringElementDomain | None] = None
    domain_label: ClassVar[str | None] = None

    @classmethod
    def apply(
        cls,
        image: np.ndarray,
        footprint: np.ndarray,
        operation: StructuringElementOperation,
    ) -> np.ndarray:
        request = StructuringElementApplication(
            image=np.asarray(image),
            footprint=np.asarray(footprint, dtype=bool),
            operation=operation,
        )
        return cls.for_enum_member(request.domain).apply_request(request)

    @abstractmethod
    def apply_request(self, request: StructuringElementApplication) -> np.ndarray:
        """Apply the requested operation under this domain relationship."""


class MatchingRankStructuringElementApplicationStrategy(
    StructuringElementApplicationStrategy
):
    domain_key = StructuringElementDomain.MATCHING_RANK

    def apply_request(self, request: StructuringElementApplication) -> np.ndarray:
        return request.operation(request.image, request.footprint)


class StackwiseStructuringElementApplicationStrategy(
    StructuringElementApplicationStrategy
):
    domain_key = StructuringElementDomain.STACKWISE

    def apply_request(self, request: StructuringElementApplication) -> np.ndarray:
        output = np.empty_like(request.image)
        spatial_shape = request.image.shape[-request.footprint.ndim :]
        plane_count = int(
            np.prod(request.image.shape[: -request.footprint.ndim], dtype=np.int64)
        )
        image_planes = request.image.reshape((plane_count, *spatial_shape))
        output_planes = output.reshape((plane_count, *spatial_shape))
        for plane_index in range(plane_count):
            output_planes[plane_index] = request.operation(
                image_planes[plane_index],
                request.footprint,
            )
        return output


class UnsupportedVolumetricToPlaneStructuringElementApplicationStrategy(
    StructuringElementApplicationStrategy
):
    domain_key = StructuringElementDomain.UNSUPPORTED_VOLUMETRIC_TO_PLANE

    def apply_request(self, request: StructuringElementApplication) -> np.ndarray:
        raise NotImplementedError(
            "A volumetric structuring element cannot be applied to a lower-rank "
            "CellProfiler image domain; got "
            f"footprint ndim={request.footprint.ndim}, image ndim={request.image.ndim}."
        )


def apply_structuring_element(
    image: np.ndarray,
    footprint: np.ndarray,
    operation: StructuringElementOperation,
) -> np.ndarray:
    """Apply a morphology operation using CellProfiler rank semantics."""
    return StructuringElementApplicationStrategy.apply(image, footprint, operation)


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
