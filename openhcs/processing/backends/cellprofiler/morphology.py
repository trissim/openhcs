"""Morphology backend strategies for CellProfiler-compatible processing.

This module is the OpenHCS processing-backend seam for CellProfiler-compatible
semantics.  The default implementation is independent NumPy/SciPy/skimage code;
the optional Centrosome provider is allowed for matching legacy morphology
behavior when explicitly requested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit, prange

from openhcs.constants.constants import MemoryType
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_semantics import (
    ExplicitObjectLabelDomainDeclaration,
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ParentChildRelationshipPayload,
    aligned_dense_object_labels_and_mask,
    dense_object_label_plane_id_domains,
    dense_object_label_max_present_id,
)
from openhcs.core.runtime_values import (
    ObjectLabelPayload,
    ObjectLabelSet,
    object_label_payload_with_dense_labels,
)
from openhcs.interop.cellprofiler.expand_or_shrink_settings import (
    CellProfilerExpandShrinkOperation,
    ExpandShrinkMode,
)
from openhcs.interop.cellprofiler.image_module_settings import CombineObjectsMethod
from openhcs.interop.cellprofiler.mask_objects_settings import (
    MaskObjectsNumberingChoice,
    MaskObjectsOverlapHandling,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    cellprofiler_backend_key,
)
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)

HolePredicate = Callable[[int, bool], bool]
ConnectivityStructureBuilder = Callable[[int], np.ndarray]
LabelBoundingBox = tuple[int, tuple[slice, ...]]
LabelBoundingBoxes = list[LabelBoundingBox]
SCIPY_CONSTANT_BOUNDARY_MODE = "constant"


def face_connected_component_structure(ndim: int) -> np.ndarray:
    """Return the SciPy face-connectivity structure for an nd label image."""
    from scipy import ndimage as ndi

    return ndi.generate_binary_structure(ndim, 1)


def full_connected_component_structure(ndim: int) -> np.ndarray:
    """Return the full 3-wide neighborhood structure for an nd label image."""
    return np.ones((3,) * ndim, dtype=bool)


class ConnectedComponentConnectivity(ABC, metaclass=AutoRegisterMeta):
    """Registered structuring-element policy for connected components."""

    __registry_key__ = "connectivity"
    __skip_if_no_key__ = True
    connectivity: ClassVar[int | None] = None
    structure_builder: ClassVar[ConnectivityStructureBuilder | None] = None

    @classmethod
    def for_connectivity(cls, connectivity: int) -> "ConnectedComponentConnectivity":
        strategy_type = cls.__registry__.get(connectivity)
        if strategy_type is None:
            raise ValueError(
                f"Unsupported connected-component connectivity: {connectivity}"
            )
        return strategy_type()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.connectivity is not None and cls.structure_builder is None:
            raise TypeError(
                f"{cls.__name__} must declare structure_builder for connectivity"
            )

    def structure(self, ndim: int) -> np.ndarray:
        structure_builder = type(self).structure_builder
        if structure_builder is None:
            raise TypeError(f"{type(self).__name__} cannot build connectivity")
        return structure_builder(ndim)


class FaceConnectedComponents(ConnectedComponentConnectivity):
    """Face-connected components."""

    connectivity = 1
    structure_builder = staticmethod(face_connected_component_structure)


class FullConnectedComponents(ConnectedComponentConnectivity):
    """Fully connected components over a 3-wide neighborhood."""

    connectivity = 2
    structure_builder = staticmethod(full_connected_component_structure)


class CellProfilerDeclumpMethod(Enum):
    """Typed declumping modes that affect morphology backend geometry."""

    INTENSITY = "intensity"
    SHAPE = "shape"


class MorphologyBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal morphology operations keyed by OpenHCS memory type."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @classmethod
    def for_memory_type(
        cls,
        memory_type: MemoryType | str = MemoryType.NUMPY,
        *,
        backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
        prefer_centrosome: bool = False,
    ) -> "MorphologyBackendStrategy":
        if prefer_centrosome:
            if backend_provider not in (
                DEFAULT_CELLPROFILER_BACKEND_SELECTION,
                CellProfilerBackendProvider.CENTROSOME,
            ):
                raise ValueError(
                    "prefer_centrosome=True conflicts with explicit "
                    f"backend_provider={backend_provider!r}"
                )
            backend_provider = CellProfilerBackendProvider.CENTROSOME
        return super().for_memory_type(
            memory_type,
            backend_provider=backend_provider,
        )

    @classmethod
    def for_callable(
        cls,
        func: object,
        *,
        backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
        prefer_centrosome: bool = False,
    ) -> "MorphologyBackendStrategy":
        if prefer_centrosome:
            if backend_provider not in (
                DEFAULT_CELLPROFILER_BACKEND_SELECTION,
                CellProfilerBackendProvider.CENTROSOME,
            ):
                raise ValueError(
                    "prefer_centrosome=True conflicts with explicit "
                    f"backend_provider={backend_provider!r}"
                )
            backend_provider = CellProfilerBackendProvider.CENTROSOME
        return super().for_callable(
            func,
            backend_provider=backend_provider,
        )

    @abstractmethod
    def connected_components(
        self,
        mask: np.ndarray,
        *,
        connectivity: int = 2,
    ) -> tuple[np.ndarray, int]:
        """Label foreground components in a binary 2-D mask."""

    @abstractmethod
    def disk_footprint(self, radius: float) -> np.ndarray:
        """Return a 2-D disk footprint."""

    @abstractmethod
    def declumping_suppression_footprint(
        self,
        suppress_size: float,
        *,
        min_diameter: float,
        declump_method: CellProfilerDeclumpMethod,
    ) -> np.ndarray:
        """Return the local-maxima suppression footprint for declumping."""

    @abstractmethod
    def grayscale_opening(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        """Return grayscale morphological opening for a 2-D image."""

    @abstractmethod
    def grayscale_closing(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        """Return grayscale morphological closing for a 2-D image."""

    @abstractmethod
    def erode_labeled_objects(
        self,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        """Erode labeled objects while preserving label identities."""

    @abstractmethod
    def block_labels(
        self,
        image_shape: tuple[int, int],
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partition a 2-D plane into square block labels."""

    @abstractmethod
    def blockwise_minimum(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        block_size: int,
    ) -> np.ndarray:
        """Broadcast the masked minimum of each CellProfiler block to its pixels."""

    @abstractmethod
    def fix_labeled_result(self, values: np.ndarray) -> np.ndarray:
        """Normalize scipy.ndimage labeled reductions to an ndarray."""

    @abstractmethod
    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        """Fill enclosed background components."""

    @abstractmethod
    def fill_labeled_holes_below_size(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        """Fill enclosed background components smaller than a size limit."""

    @abstractmethod
    def restore_removed_declump_basins(
        self,
        pre_declump_labels: np.ndarray,
        labels_before_size_filter: np.ndarray,
        labels_after_size_filter: np.ndarray,
    ) -> np.ndarray:
        """Restore removed watershed basins with one surviving original identity."""

    @abstractmethod
    def local_maxima_by_label(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        """Find local maxima independently within each positive label."""

    @abstractmethod
    def smooth_image_for_declumping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        """Smooth an image using CP's mask-corrected declumping convention."""

    @abstractmethod
    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        """Return the binary convex hull of a 2-D mask."""

    @abstractmethod
    def declumping_seed_points(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        """Find CellProfiler-compatible declumping seed points."""

    @abstractmethod
    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        """Represent each connected component by one seed point."""

    @abstractmethod
    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        """Compact positive labels to 1..N."""


class NumpyMorphologyBackendStrategy(MorphologyBackendStrategy):
    """Independent NumPy/SciPy/skimage morphology backend."""

    backend_key = cellprofiler_backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False

    def connected_components(
        self,
        mask: np.ndarray,
        *,
        connectivity: int = 2,
    ) -> tuple[np.ndarray, int]:
        return _scipy_connected_components(mask, connectivity=connectivity)

    def disk_footprint(self, radius: float) -> np.ndarray:
        return _scipy_disk_footprint(radius)

    def declumping_suppression_footprint(
        self,
        suppress_size: float,
        *,
        min_diameter: float,
        declump_method: CellProfilerDeclumpMethod,
    ) -> np.ndarray:
        radius = _declumping_suppression_radius(
            suppress_size,
            min_diameter=min_diameter,
            declump_method=declump_method,
        )
        return _scipy_disk_footprint(radius)

    def grayscale_closing(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _skimage_grayscale_closing(image, footprint)

    def grayscale_opening(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _skimage_grayscale_opening(image, footprint)

    def erode_labeled_objects(
        self,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _scipy_erode_labeled_objects(labels, footprint)

    def block_labels(
        self,
        image_shape: tuple[int, int],
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _scipy_block_labels(image_shape, block_size)

    def blockwise_minimum(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        block_size: int,
    ) -> np.ndarray:
        return _scipy_blockwise_minimum(
            image,
            mask,
            block_size,
            morphology=self,
        )

    def fix_labeled_result(self, values: np.ndarray) -> np.ndarray:
        return _scipy_fix_labeled_result(values)

    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        return _scipy_fill_labeled_holes(labels, size_predicate=size_predicate)

    def fill_labeled_holes_below_size(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        return _scipy_fill_labeled_holes(
            labels,
            size_predicate=lambda size, _is_foreground: size < maximum_hole_size,
        )

    def restore_removed_declump_basins(
        self,
        pre_declump_labels: np.ndarray,
        labels_before_size_filter: np.ndarray,
        labels_after_size_filter: np.ndarray,
    ) -> np.ndarray:
        return _restore_removed_declump_basins_numba(
            np.ascontiguousarray(pre_declump_labels, dtype=np.int64),
            np.ascontiguousarray(labels_before_size_filter, dtype=np.int64),
            np.ascontiguousarray(labels_after_size_filter, dtype=np.int64),
        ).astype(np.asarray(labels_after_size_filter).dtype, copy=False)

    def local_maxima_by_label(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _scipy_local_maxima_by_label(image, labels, footprint)

    def smooth_image_for_declumping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        return _scipy_smooth_image_for_declumping(
            image,
            mask,
            filter_size,
            declump_method=declump_method,
            suppress_size=suppress_size,
            min_diameter=min_diameter,
        )

    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        return _skimage_convex_hull_image(mask)

    def declumping_seed_points(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        return _scipy_declumping_seed_points(
            image,
            labels,
            footprint,
            image_resize_factor,
            morphology=self,
        )

    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        return _scipy_shrink_components_to_seed_points(mask)

    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        return _scipy_relabel_sequential(labels)


class CentrosomeNumpyMorphologyBackendStrategy(NumpyMorphologyBackendStrategy):
    """Optional centrosome provider for NumPy-memory morphology."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.CENTROSOME,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def disk_footprint(self, radius: float) -> np.ndarray:
        from centrosome.cpmorphology import strel_disk

        return strel_disk(radius)

    def block_labels(
        self,
        image_shape: tuple[int, int],
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from centrosome.cpmorphology import block

        block_size = max(1, int(block_size))
        return block(image_shape, (block_size, block_size))

    def fix_labeled_result(self, values: np.ndarray) -> np.ndarray:
        from centrosome.cpmorphology import fixup_scipy_ndimage_result

        return fixup_scipy_ndimage_result(values)

    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        from centrosome.cpmorphology import fill_labeled_holes

        if size_predicate is None:
            return fill_labeled_holes(labels)
        return fill_labeled_holes(labels, size_fn=size_predicate)

    def fill_labeled_holes_below_size(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        return self.fill_labeled_holes(
            labels,
            size_predicate=lambda size, _is_foreground: size < maximum_hole_size,
        )

    def local_maxima_by_label(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        from centrosome.cpmorphology import is_local_maximum

        return np.asarray(is_local_maximum(image, labels, footprint), dtype=bool)

    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        from centrosome.cpmorphology import convex_hull_image

        return np.asarray(convex_hull_image(mask), dtype=bool)

    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        from centrosome.cpmorphology import binary_shrink

        return np.asarray(binary_shrink(mask), dtype=bool)

    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        from centrosome.cpmorphology import relabel

        relabeled, count = relabel(labels)
        return relabeled, int(count)


class NumbaNumpyMorphologyBackendStrategy(NumpyMorphologyBackendStrategy):
    """Numba-accelerated NumPy morphology backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def connected_components(
        self,
        mask: np.ndarray,
        *,
        connectivity: int = 2,
    ) -> tuple[np.ndarray, int]:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            return self._connected_components_planewise(
                mask_array,
                connectivity=connectivity,
            )
        if connectivity != 2:
            return super().connected_components(mask_array, connectivity=connectivity)
        return _foreground_components_2d_numba(np.ascontiguousarray(mask_array))

    def _connected_components_planewise(
        self,
        mask: np.ndarray,
        *,
        connectivity: int,
    ) -> tuple[np.ndarray, int]:
        if mask.ndim < 2:
            raise ValueError("Connected components requires at least two dimensions.")
        labels = np.zeros(mask.shape, dtype=np.int32)
        plane_count = int(np.prod(mask.shape[:-2], dtype=np.int64))
        source_planes = mask.reshape((plane_count, *mask.shape[-2:]))
        target_planes = labels.reshape((plane_count, *mask.shape[-2:]))
        label_offset = 0
        for plane_index in range(plane_count):
            plane_labels, plane_count_labels = self.connected_components(
                source_planes[plane_index],
                connectivity=connectivity,
            )
            if plane_count_labels:
                target_planes[plane_index] = np.where(
                    plane_labels > 0,
                    plane_labels + label_offset,
                    0,
                )
                label_offset += plane_count_labels
        return labels, label_offset

    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D convex hulls."
            )
        return _convex_hull_image_numba(np.ascontiguousarray(mask_array))

    def grayscale_closing(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        footprint_array = np.asarray(footprint, dtype=bool)
        if image_array.ndim != 2 or footprint_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D grayscale closing."
            )
        footprint_offsets = _footprint_offsets(footprint_array)
        return _grayscale_morphology_2d_numba(
            np.ascontiguousarray(image_array),
            footprint_offsets[:, 0],
            footprint_offsets[:, 1],
            True,
        )

    def grayscale_opening(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        footprint_array = np.asarray(footprint, dtype=bool)
        if image_array.ndim != 2 or footprint_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D grayscale opening."
            )
        footprint_offsets = _footprint_offsets(footprint_array)
        return _grayscale_morphology_2d_numba(
            np.ascontiguousarray(image_array),
            footprint_offsets[:, 0],
            footprint_offsets[:, 1],
            False,
        )

    def block_labels(
        self,
        image_shape: tuple[int, int],
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(image_shape) != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D block labels."
            )
        height, width = image_shape
        return _block_labels_2d_numba(
            int(height),
            int(width),
            max(1, int(block_size)),
        )

    def blockwise_minimum(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        block_size: int,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        if image_array.ndim not in (2, 3):
            return super().blockwise_minimum(image_array, mask, block_size)
        mask_array = (
            np.empty((0, 0), dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        if mask is not None and mask_array.shape != image_array.shape[:2]:
            raise ValueError(
                "Blockwise minimum mask must match image spatial shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        return _blockwise_minimum_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(mask_array),
            mask is not None,
            max(1, int(block_size)),
        )

    def erode_labeled_objects(
        self,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        footprint_array = np.asarray(footprint, dtype=bool)
        if labels_array.ndim not in (2, 3) or footprint_array.ndim != labels_array.ndim:
            return super().erode_labeled_objects(labels_array, footprint_array)
        offsets = _footprint_offsets_nd(footprint_array)
        return _erode_labeled_objects_numba(
            np.ascontiguousarray(labels_array),
            offsets,
        ).astype(labels_array.dtype, copy=False)

    def local_maxima_by_label(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        image_array = np.ascontiguousarray(image, dtype=np.float64)
        labels_array = np.ascontiguousarray(labels, dtype=np.int64)
        footprint_offsets = _footprint_offsets(footprint)
        return _local_maxima_by_label_numba(
            image_array,
            labels_array,
            footprint_offsets[:, 0],
            footprint_offsets[:, 1],
        )

    def smooth_image_for_declumping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        mask_array = np.asarray(mask, dtype=bool)
        if image_array.ndim != 2 or mask_array.ndim != 2:
            return self._smooth_image_for_declumping_planewise(
                image_array,
                mask_array,
                filter_size,
                declump_method=declump_method,
                suppress_size=suppress_size,
                min_diameter=min_diameter,
            )
        if image_array.shape != mask_array.shape:
            raise ValueError(
                "Declumping smoothing mask must match the image shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        kernel = _declumping_smoothing_kernel(
            filter_size,
            declump_method=declump_method,
            suppress_size=suppress_size,
            min_diameter=min_diameter,
        )
        if kernel.size == 0:
            return image_array
        return _smooth_image_for_declumping_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(mask_array),
            kernel,
        )

    def _smooth_image_for_declumping_planewise(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod,
        suppress_size: float | None,
        min_diameter: float | None,
    ) -> np.ndarray:
        if image.ndim < 2 or mask.ndim < 2:
            raise ValueError("Declumping smoothing requires at least two dimensions.")
        if image.shape != mask.shape:
            raise ValueError(
                "Declumping smoothing mask must match the image shape; got "
                f"mask {mask.shape!r} for image {image.shape!r}."
            )
        smoothed = np.empty_like(image)
        plane_count = int(np.prod(image.shape[:-2], dtype=np.int64))
        image_planes = image.reshape((plane_count, *image.shape[-2:]))
        mask_planes = mask.reshape((plane_count, *mask.shape[-2:]))
        target_planes = smoothed.reshape((plane_count, *image.shape[-2:]))
        for plane_index in range(plane_count):
            target_planes[plane_index] = self.smooth_image_for_declumping(
                image_planes[plane_index],
                mask_planes[plane_index],
                filter_size,
                declump_method=declump_method,
                suppress_size=suppress_size,
                min_diameter=min_diameter,
            )
        return smoothed

    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 2:
            return self._fill_labeled_holes_planewise(
                labels_array,
                size_predicate=size_predicate,
            )
        return self._fill_labeled_holes_2d(
            labels_array,
            size_predicate=size_predicate,
        )

    def _fill_labeled_holes_planewise(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        if labels.ndim < 2:
            raise ValueError("Hole filling requires at least two dimensions.")
        filled = np.empty_like(labels)
        plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
        source_planes = labels.reshape((plane_count, *labels.shape[-2:]))
        target_planes = filled.reshape((plane_count, *labels.shape[-2:]))
        for plane_index in range(plane_count):
            target_planes[plane_index] = self._fill_labeled_holes_2d(
                source_planes[plane_index],
                size_predicate=size_predicate,
            )
        return filled

    def _fill_labeled_holes_2d(
        self,
        labels: np.ndarray,
        *,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        components, sizes, touches_border, component_count = (
            _background_components_2d_numba(np.ascontiguousarray(labels))
        )
        fill_flags = np.zeros(component_count + 1, dtype=np.bool_)
        for component_id in range(1, component_count + 1):
            if touches_border[component_id]:
                continue
            if size_predicate is None or size_predicate(
                int(sizes[component_id]),
                False,
            ):
                fill_flags[component_id] = True
        if not np.any(fill_flags):
            return labels
        if labels.dtype == np.bool_:
            return _fill_binary_holes_from_components_numba(
                np.ascontiguousarray(labels),
                components,
                fill_flags,
            )
        return _fill_labeled_holes_single_label_components_numba(
            np.ascontiguousarray(labels),
            components,
            fill_flags,
        )

    def restore_removed_declump_basins(
        self,
        pre_declump_labels: np.ndarray,
        labels_before_size_filter: np.ndarray,
        labels_after_size_filter: np.ndarray,
    ) -> np.ndarray:
        pre_declump = np.asarray(pre_declump_labels)
        before = np.asarray(labels_before_size_filter)
        after = np.asarray(labels_after_size_filter)
        if pre_declump.ndim != 2 or before.ndim != 2 or after.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D declump basin restoration."
            )
        if pre_declump.shape != before.shape or before.shape != after.shape:
            raise ValueError(
                "Declump basin restoration inputs must have identical shapes; got "
                f"{pre_declump.shape!r}, {before.shape!r}, and {after.shape!r}."
            )
        return _restore_removed_declump_basins_numba(
            np.ascontiguousarray(pre_declump, dtype=np.int64),
            np.ascontiguousarray(before, dtype=np.int64),
            np.ascontiguousarray(after, dtype=np.int64),
        ).astype(after.dtype, copy=False)

    def fill_labeled_holes_below_size(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 2:
            return self._fill_labeled_holes_below_size_planewise(
                labels_array,
                maximum_hole_size,
            )
        return self._fill_labeled_holes_below_size_2d(
            labels_array,
            maximum_hole_size,
        )

    def _fill_labeled_holes_below_size_planewise(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        if labels.ndim < 2:
            raise ValueError("Hole filling requires at least two dimensions.")
        filled = np.empty_like(labels)
        plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
        source_planes = labels.reshape((plane_count, *labels.shape[-2:]))
        target_planes = filled.reshape((plane_count, *labels.shape[-2:]))
        for plane_index in range(plane_count):
            target_planes[plane_index] = self._fill_labeled_holes_below_size_2d(
                source_planes[plane_index],
                maximum_hole_size,
            )
        return filled

    def _fill_labeled_holes_below_size_2d(
        self,
        labels: np.ndarray,
        maximum_hole_size: int,
    ) -> np.ndarray:
        components, sizes, touches_border, component_count = (
            _background_components_2d_numba(np.ascontiguousarray(labels))
        )
        fill_flags = _hole_fill_flags_below_size_numba(
            sizes,
            touches_border,
            component_count,
            int(maximum_hole_size),
        )
        if labels.dtype == np.bool_:
            return _fill_binary_holes_from_components_numba(
                np.ascontiguousarray(labels),
                components,
                fill_flags,
            )
        return _fill_labeled_holes_single_label_components_numba(
            np.ascontiguousarray(labels),
            components,
            fill_flags,
        )

    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D seed shrinking."
            )
        return _binary_shrink_2d_numba(
            np.ascontiguousarray(mask_array),
            _binary_shrink_table_stack(),
        )

    def declumping_seed_points(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        labels_array = np.asarray(labels)
        if image_array.ndim != 2 or labels_array.ndim != 2:
            return self._declumping_seed_points_planewise(
                image_array,
                labels_array,
                footprint,
                image_resize_factor,
            )
        if image_array.shape != labels_array.shape:
            raise ValueError(
                "image and labels must have identical shapes for declumping seed extraction"
            )
        if float(image_resize_factor) != 1.0:
            return super().declumping_seed_points(
                image_array,
                labels_array,
                footprint,
                image_resize_factor,
            )

        maxima = self.local_maxima_by_label(
            image_array,
            labels_array,
            footprint,
        )
        maxima[np.asarray(image_array) <= 0] = 0
        return self.shrink_components_to_seed_points(maxima)

    def _declumping_seed_points_planewise(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        if image.ndim < 2 or labels.ndim < 2:
            raise ValueError("Declumping seed extraction requires at least two dimensions.")
        if image.shape != labels.shape:
            raise ValueError(
                "image and labels must have identical shapes for declumping seed extraction"
            )
        seeds = np.empty(labels.shape, dtype=bool)
        plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
        image_planes = image.reshape((plane_count, *image.shape[-2:]))
        label_planes = labels.reshape((plane_count, *labels.shape[-2:]))
        seed_planes = seeds.reshape((plane_count, *labels.shape[-2:]))
        for plane_index in range(plane_count):
            seed_planes[plane_index] = self.declumping_seed_points(
                image_planes[plane_index],
                label_planes[plane_index],
                footprint,
                image_resize_factor,
            )
        return seeds

    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        labels_array = np.asarray(labels)
        if labels_array.ndim > 2:
            relabeled_planes, count = _relabel_sequential_3d_numba(
                np.ascontiguousarray(
                    labels_array.reshape((-1, *labels_array.shape[-2:])),
                    dtype=np.int64,
                ),
            )
            return relabeled_planes.reshape(labels_array.shape), int(count)
        if labels_array.ndim != 2:
            raise ValueError("Relabeling requires at least two dimensions.")
        return _relabel_sequential_numba(
            np.ascontiguousarray(labels_array, dtype=np.int64),
        )


class OpenCVNumpyMorphologyBackendStrategy(NumbaNumpyMorphologyBackendStrategy):
    """OpenCV-accelerated NumPy morphology backend."""

    backend_key = cellprofiler_backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.OPENCV,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.OPENCV
    is_default_backend = False

    def grayscale_closing(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _opencv_morphology(image, footprint, operation="closing")

    def grayscale_opening(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
    ) -> np.ndarray:
        return _opencv_morphology(image, footprint, operation="opening")


def _scipy_disk_footprint(radius: float) -> np.ndarray:
    radius = max(0.0, float(radius))
    extent = int(radius)
    y, x = np.ogrid[-extent : extent + 1, -extent : extent + 1]
    return (x * x + y * y) <= radius * radius


def _declumping_suppression_radius(
    suppress_size: float,
    *,
    min_diameter: float,
    declump_method: CellProfilerDeclumpMethod,
) -> float:
    size = max(1.0, float(suppress_size))
    return max(1.0, size - 0.5)


def _scipy_block_labels(
    image_shape: tuple[int, int],
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    block_size = max(1, int(block_size))
    row_blocks = max(1, int(np.floor(float(height) / float(block_size))))
    column_blocks = max(1, int(np.floor(float(width) / float(block_size))))
    labels = np.empty((height, width), dtype=np.int32)
    indexes: list[int] = []
    for row in range(row_blocks):
        y_start = int(np.ceil(float(row * height) / float(row_blocks)))
        y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
        for column in range(column_blocks):
            x_start = int(np.ceil(float(column * width) / float(column_blocks)))
            x_stop = int(
                np.ceil(float((column + 1) * width) / float(column_blocks))
            )
            label = row * column_blocks + column
            labels[y_start:y_stop, x_start:x_stop] = label
            indexes.append(label)
    return labels, np.asarray(indexes, dtype=np.int32)


def _scipy_blockwise_minimum(
    image: np.ndarray,
    mask: np.ndarray | None,
    block_size: int,
    *,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    from scipy.ndimage import minimum

    image_array = np.asarray(image)
    labels, indexes = morphology.block_labels(image_array.shape[:2], block_size)
    labels = labels.copy()
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != image_array.shape[:2]:
            raise ValueError(
                "Blockwise minimum mask must match image spatial shape; got "
                f"mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        labels[~mask_array] = -1

    valid = labels != -1
    result = np.zeros(image_array.shape, dtype=image_array.dtype)
    if not np.any(valid):
        return result

    if image_array.ndim == 2:
        minima = morphology.fix_labeled_result(minimum(image_array, labels, indexes))
        result[valid] = minima[labels[valid]]
        return result

    if image_array.ndim != 3:
        raise NotImplementedError(
            "Blockwise minimum currently supports 2-D images or 3-D color images."
        )
    for channel in range(image_array.shape[2]):
        minima = morphology.fix_labeled_result(
            minimum(image_array[:, :, channel], labels, indexes)
        )
        result[valid, channel] = minima[labels[valid]]
    return result


def _scipy_erode_labeled_objects(
    labels: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    import scipy.ndimage

    labels_array = np.asarray(labels)
    contours = scipy.ndimage.morphological_gradient(
        labels_array,
        footprint=np.asarray(footprint, dtype=bool),
    )
    return labels_array * (contours == 0)


@njit(cache=True)
def _block_labels_2d_numba(
    height: int,
    width: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_blocks = max(1, int(np.floor(float(height) / float(block_size))))
    column_blocks = max(1, int(np.floor(float(width) / float(block_size))))
    labels = np.empty((height, width), dtype=np.int32)
    indexes = np.empty(row_blocks * column_blocks, dtype=np.int32)
    for row in range(row_blocks):
        y_start = int(np.ceil(float(row * height) / float(row_blocks)))
        y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
        for column in range(column_blocks):
            x_start = int(np.ceil(float(column * width) / float(column_blocks)))
            x_stop = int(
                np.ceil(float((column + 1) * width) / float(column_blocks))
            )
            label = row * column_blocks + column
            indexes[label] = label
            for y in range(y_start, y_stop):
                for x in range(x_start, x_stop):
                    labels[y, x] = label
    return labels, indexes


@njit(cache=True)
def _blockwise_minimum_numba(
    image: np.ndarray,
    mask: np.ndarray,
    has_mask: bool,
    block_size: int,
) -> np.ndarray:
    height = image.shape[0]
    width = image.shape[1]
    row_blocks = max(1, int(np.floor(float(height) / float(block_size))))
    column_blocks = max(1, int(np.floor(float(width) / float(block_size))))
    label_count = row_blocks * column_blocks
    output = np.zeros(image.shape, dtype=image.dtype)

    if image.ndim == 2:
        minima = np.empty(label_count, dtype=image.dtype)
        has_value = np.zeros(label_count, dtype=np.bool_)
        for row in range(row_blocks):
            y_start = int(np.ceil(float(row * height) / float(row_blocks)))
            y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
            for column in range(column_blocks):
                x_start = int(np.ceil(float(column * width) / float(column_blocks)))
                x_stop = int(
                    np.ceil(float((column + 1) * width) / float(column_blocks))
                )
                label = row * column_blocks + column
                for y in range(y_start, y_stop):
                    for x in range(x_start, x_stop):
                        if has_mask and not mask[y, x]:
                            continue
                        value = image[y, x]
                        if not has_value[label] or value < minima[label]:
                            minima[label] = value
                            has_value[label] = True
                if has_value[label]:
                    value = minima[label]
                    for y in range(y_start, y_stop):
                        for x in range(x_start, x_stop):
                            if not has_mask or mask[y, x]:
                                output[y, x] = value
        return output

    channel_count = image.shape[2]
    minima = np.empty((label_count, channel_count), dtype=image.dtype)
    has_value = np.zeros(label_count, dtype=np.bool_)
    for row in range(row_blocks):
        y_start = int(np.ceil(float(row * height) / float(row_blocks)))
        y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
        for column in range(column_blocks):
            x_start = int(np.ceil(float(column * width) / float(column_blocks)))
            x_stop = int(np.ceil(float((column + 1) * width) / float(column_blocks)))
            label = row * column_blocks + column
            for y in range(y_start, y_stop):
                for x in range(x_start, x_stop):
                    if has_mask and not mask[y, x]:
                        continue
                    if not has_value[label]:
                        for channel in range(channel_count):
                            minima[label, channel] = image[y, x, channel]
                        has_value[label] = True
                    else:
                        for channel in range(channel_count):
                            value = image[y, x, channel]
                            if value < minima[label, channel]:
                                minima[label, channel] = value
            if has_value[label]:
                for y in range(y_start, y_stop):
                    for x in range(x_start, x_stop):
                        if not has_mask or mask[y, x]:
                            for channel in range(channel_count):
                                output[y, x, channel] = minima[label, channel]
    return output


@njit(cache=True)
def _erode_labeled_objects_numba(
    labels: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=labels.dtype)
    if labels.ndim == 2:
        height, width = labels.shape
        for y in range(height):
            for x in range(width):
                label = labels[y, x]
                if label == 0:
                    continue
                keep = True
                for offset_index in range(offsets.shape[0]):
                    yy = y + offsets[offset_index, 0]
                    xx = x + offsets[offset_index, 1]
                    if yy < 0 or xx < 0 or yy >= height or xx >= width:
                        continue
                    if labels[yy, xx] != label:
                        keep = False
                        break
                if keep:
                    output[y, x] = label
        return output

    z_size, y_size, x_size = labels.shape
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                label = labels[z, y, x]
                if label == 0:
                    continue
                keep = True
                for offset_index in range(offsets.shape[0]):
                    zz = z + offsets[offset_index, 0]
                    yy = y + offsets[offset_index, 1]
                    xx = x + offsets[offset_index, 2]
                    if (
                        zz < 0
                        or yy < 0
                        or xx < 0
                        or zz >= z_size
                        or yy >= y_size
                        or xx >= x_size
                    ):
                        continue
                    if labels[zz, yy, xx] != label:
                        keep = False
                        break
                if keep:
                    output[z, y, x] = label
    return output


def _scipy_connected_components(
    mask: np.ndarray,
    *,
    connectivity: int = 2,
) -> tuple[np.ndarray, int]:
    from scipy import ndimage as ndi

    mask_array = np.asarray(mask, dtype=bool)
    structure = ConnectedComponentConnectivity.for_connectivity(
        connectivity
    ).structure(mask_array.ndim)
    labels, count = ndi.label(mask_array, structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def _scipy_fix_labeled_result(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        return values.reshape(1)
    return values


def _scipy_fill_labeled_holes(
    labels: np.ndarray,
    *,
    size_predicate: HolePredicate | None = None,
) -> np.ndarray:
    from scipy import ndimage as ndi

    array = np.asarray(labels)
    foreground = array != 0
    background = ~foreground
    if not background.any():
        return array.copy()

    structure = ndi.generate_binary_structure(array.ndim, 1)
    background_labels, component_count = ndi.label(background, structure=structure)
    if component_count == 0:
        return array.copy()

    border_ids = _border_component_ids(background_labels)
    candidate_ids = set(range(1, component_count + 1)) - border_ids
    if size_predicate is not None:
        sizes = np.bincount(background_labels.ravel(), minlength=component_count + 1)
        candidate_ids = {
            component_id
            for component_id in candidate_ids
            if size_predicate(int(sizes[component_id]), False)
        }
    if not candidate_ids:
        return array.copy()

    fill_mask = np.isin(background_labels, tuple(sorted(candidate_ids)))
    if array.dtype == bool or np.array_equal(np.unique(array), np.array([False, True])):
        output = foreground.copy()
        output[fill_mask] = True
        return output.astype(array.dtype, copy=False)

    _, nearest_indices = ndi.distance_transform_edt(
        background,
        return_distances=True,
        return_indices=True,
    )
    output = array.copy()
    output[fill_mask] = array[
        tuple(axis_indices[fill_mask] for axis_indices in nearest_indices)
    ]
    return output


def _scipy_local_maxima_by_label(
    image: np.ndarray,
    labels: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    from scipy import ndimage as ndi

    image_array = np.asarray(image)
    labels_array = np.asarray(labels)
    maxima = np.zeros(labels_array.shape, dtype=bool)
    if image_array.shape != labels_array.shape:
        raise ValueError(
            "image and labels must have identical shapes for labeled local maxima"
        )

    for label_id, bounds in _positive_label_bounding_boxes(labels_array):
        label_crop = labels_array[bounds] == label_id
        image_crop = image_array[bounds]
        masked_image = np.where(label_crop, image_crop, -np.inf)
        local_max = ndi.maximum_filter(
            masked_image,
            footprint=footprint,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
            cval=-np.inf,
        )
        maxima[bounds] |= label_crop & (image_crop == local_max)
    return maxima


def _scipy_smooth_image_for_declumping(
    image: np.ndarray,
    mask: np.ndarray,
    filter_size: float,
    *,
    declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
    suppress_size: float | None = None,
    min_diameter: float | None = None,
) -> np.ndarray:
    import scipy.ndimage

    if filter_size == 0:
        return image
    kernel = _declumping_smoothing_kernel(
        filter_size,
        declump_method=declump_method,
        suppress_size=suppress_size,
        min_diameter=min_diameter,
    )

    def convolve(array: np.ndarray) -> np.ndarray:
        output = scipy.ndimage.convolve1d(
            array,
            kernel,
            axis=0,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
        )
        return scipy.ndimage.convolve1d(
            output,
            kernel,
            axis=1,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
        )

    mask_array = np.asarray(mask, dtype=bool)
    edge_array = convolve(mask_array.astype(float))
    masked_image = np.asarray(image).copy()
    masked_image[~mask_array] = 0
    smoothed_image = convolve(masked_image)
    valid = mask_array & (edge_array != 0)
    masked_image[valid] = smoothed_image[valid] / edge_array[valid]
    return masked_image


def _declumping_smoothing_kernel(
    filter_size: float,
    *,
    declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
    suppress_size: float | None = None,
    min_diameter: float | None = None,
) -> np.ndarray:
    if filter_size == 0:
        return np.empty((0,), dtype=np.float64)
    sigma_divisor = _declumping_smoothing_sigma_divisor(
        declump_method=declump_method,
        suppress_size=suppress_size,
        min_diameter=min_diameter,
    )
    sigma = float(filter_size) / sigma_divisor
    half_width = max(int(float(filter_size) / 2.0), 1)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = (
        1.0
        / np.sqrt(2.0 * np.pi)
        / sigma
        * np.exp(-0.5 * offsets**2 / sigma**2)
    )
    return np.ascontiguousarray(kernel, dtype=np.float64)


def _declumping_smoothing_sigma_divisor(
    *,
    declump_method: CellProfilerDeclumpMethod,
    suppress_size: float | None,
    min_diameter: float | None,
) -> float:
    return 2.35


def _skimage_convex_hull_image(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import convex_hull_image

    return np.asarray(convex_hull_image(np.asarray(mask, dtype=bool)), dtype=bool)


def _skimage_grayscale_closing(
    image: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    from skimage.morphology import closing as skimage_closing

    image_array = np.asarray(image)
    return np.asarray(
        skimage_closing(image_array, np.asarray(footprint, dtype=bool)),
        dtype=image_array.dtype,
    )


def _skimage_grayscale_opening(
    image: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    from skimage.morphology import opening as skimage_opening

    image_array = np.asarray(image)
    return np.asarray(
        skimage_opening(image_array, np.asarray(footprint, dtype=bool)),
        dtype=image_array.dtype,
    )


def _opencv_morphology(
    image: np.ndarray,
    footprint: np.ndarray,
    *,
    operation: str,
) -> np.ndarray:
    import cv2

    image_array = np.asarray(image)
    footprint_array = np.asarray(footprint, dtype=np.uint8)
    op = cv2.MORPH_OPEN if operation == "opening" else cv2.MORPH_CLOSE
    result = cv2.morphologyEx(
        np.ascontiguousarray(image_array),
        op,
        footprint_array,
        borderType=cv2.BORDER_REFLECT,
    )
    return np.asarray(result, dtype=image_array.dtype)


def _scipy_declumping_seed_points(
    image: np.ndarray,
    labels: np.ndarray,
    footprint: np.ndarray,
    image_resize_factor: float,
    *,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    from scipy import ndimage as ndi

    image_array = np.asarray(image)
    labels_array = np.asarray(labels)
    if image_array.shape != labels_array.shape:
        raise ValueError(
            "image and labels must have identical shapes for declumping seed extraction"
        )

    if image_resize_factor < 1.0:
        shape = np.maximum(
            1,
            np.ceil(np.asarray(image_array.shape) * float(image_resize_factor)),
        ).astype(int)
        coordinates = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float) / float(
            image_resize_factor
        )
        resized_image = ndi.map_coordinates(image_array, coordinates)
        resized_labels = ndi.map_coordinates(
            labels_array,
            coordinates,
            order=0,
        ).astype(labels_array.dtype, copy=False)
    else:
        resized_image = image_array
        resized_labels = labels_array

    maxima = morphology.local_maxima_by_label(
        resized_image,
        resized_labels,
        footprint,
    )
    maxima[resized_image <= 0] = 0

    if image_resize_factor < 1.0:
        inverse_resize_factor = float(image_array.shape[0]) / float(maxima.shape[0])
        coordinates = (
            np.mgrid[0 : image_array.shape[0], 0 : image_array.shape[1]].astype(float)
            / inverse_resize_factor
        )
        maxima = ndi.map_coordinates(maxima.astype(float), coordinates) > 0.5

    return morphology.shrink_components_to_seed_points(maxima)


def _positive_label_bounding_boxes(
    labels: np.ndarray,
) -> LabelBoundingBoxes:
    positive_coords = np.nonzero(labels > 0)
    if not positive_coords[0].size:
        return []

    label_values = labels[positive_coords]
    order = np.argsort(label_values, kind="stable")
    sorted_labels = label_values[order]
    sorted_coords = tuple(axis_coords[order] for axis_coords in positive_coords)
    change_offsets = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    group_starts = np.concatenate(([0], change_offsets))
    group_ends = np.concatenate((change_offsets, [sorted_labels.size]))

    boxes: LabelBoundingBoxes = []
    for start, end in zip(group_starts, group_ends):
        bounds = tuple(
            slice(int(axis_coords[start:end].min()), int(axis_coords[start:end].max()) + 1)
            for axis_coords in sorted_coords
        )
        boxes.append((int(sorted_labels[start]), bounds))
    return boxes


def _scipy_shrink_components_to_seed_points(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    mask_array = np.asarray(mask, dtype=bool)
    components, component_count = ndi.label(
        mask_array,
        structure=np.ones((3,) * mask_array.ndim, dtype=bool),
    )
    seeds = np.zeros(mask_array.shape, dtype=bool)
    for component_id, component_slice in enumerate(
        ndi.find_objects(components, max_label=component_count),
        start=1,
    ):
        if component_slice is None:
            continue
        component_crop = components[component_slice] == component_id
        coords = np.argwhere(component_crop)
        if coords.size == 0:
            continue
        centroid = coords.mean(axis=0)
        nearest = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
        seed_coord = tuple(
            int(axis_slice.start or 0) + int(coord)
            for axis_slice, coord in zip(component_slice, coords[nearest], strict=True)
        )
        seeds[seed_coord] = True
    return seeds


@lru_cache(maxsize=1)
def _binary_shrink_table_stack() -> np.ndarray:
    erode_table = np.array(
        [
            _binary_shrink_pattern_center(index)
            and _binary_shrink_component_count(index & ~16) != 1
            for index in range(512)
        ],
        dtype=np.bool_,
    )
    erode_table[_binary_shrink_index_of(np.ones((3, 3), dtype=bool))] = True

    tables = (
        erode_table
        | (
            _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
                np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
            )
            & _binary_shrink_make_table(
                False,
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
                np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
            )
        ),
        erode_table
        | (
            _binary_shrink_make_table(
                False,
                np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
                np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
            )
            & _binary_shrink_make_table(
                False,
                np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
                np.array([[0, 0, 1], [1, 1, 0], [1, 1, 0]], dtype=bool),
            )
        ),
        erode_table
        | (
            _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
                np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
            )
            & _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
                np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
            )
        ),
        erode_table
        | (
            _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
                np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
            )
            & _binary_shrink_make_table(
                False,
                np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=bool),
                np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]], dtype=bool),
            )
        ),
    )
    return np.ascontiguousarray(np.stack(tables), dtype=np.bool_)


def _binary_shrink_pattern_center(index: int) -> bool:
    return bool(index & 16)


def _binary_shrink_component_count(index: int) -> int:
    pattern = _binary_shrink_pattern_of(index)
    visited = np.zeros((3, 3), dtype=bool)
    components = 0
    for row in range(3):
        for col in range(3):
            if not pattern[row, col] or visited[row, col]:
                continue
            components += 1
            stack: list[tuple[int, int]] = [(row, col)]
            visited[row, col] = True
            while stack:
                current_row, current_col = stack.pop()
                for delta_row, delta_col in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if (
                        next_row < 0
                        or next_row >= 3
                        or next_col < 0
                        or next_col >= 3
                        or visited[next_row, next_col]
                        or not pattern[next_row, next_col]
                    ):
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
    return components


def _binary_shrink_pattern_of(index: int) -> np.ndarray:
    pattern = np.zeros((3, 3), dtype=bool)
    bit = 1
    for row in range(3):
        for col in range(3):
            pattern[row, col] = bool(index & bit)
            bit <<= 1
    return pattern


def _binary_shrink_index_of(pattern: np.ndarray) -> int:
    index = 0
    bit = 1
    for row in range(3):
        for col in range(3):
            if pattern[row, col]:
                index += bit
            bit <<= 1
    return index


def _binary_shrink_make_table(
    value: bool,
    pattern: np.ndarray,
    care: np.ndarray,
) -> np.ndarray:
    table = np.empty(512, dtype=np.bool_)
    for index in range(512):
        matches = True
        bit = 1
        for row in range(3):
            for col in range(3):
                if care[row, col] and bool(index & bit) != bool(pattern[row, col]):
                    matches = False
                    break
                bit <<= 1
            if not matches:
                break
        table[index] = value if matches else not value
    return table


def _scipy_relabel_sequential(labels: np.ndarray) -> tuple[np.ndarray, int]:
    labels_array = np.asarray(labels)
    positive = np.unique(labels_array[labels_array > 0])
    output = np.zeros(labels_array.shape, dtype=np.int32)
    for new_label, old_label in enumerate(positive, start=1):
        output[labels_array == old_label] = new_label
    return output, int(positive.size)


def _footprint_offsets(footprint: np.ndarray) -> np.ndarray:
    footprint_array = np.asarray(footprint, dtype=bool)
    if footprint_array.ndim != 2:
        raise NotImplementedError(
            "CellProfiler-compatible morphology currently supports 2-D footprints."
        )
    center_y = footprint_array.shape[0] // 2
    center_x = footprint_array.shape[1] // 2
    coords = np.argwhere(footprint_array)
    return np.ascontiguousarray(
        np.column_stack((coords[:, 0] - center_y, coords[:, 1] - center_x)),
        dtype=np.int64,
    )


def _footprint_offsets_nd(footprint: np.ndarray) -> np.ndarray:
    footprint_array = np.asarray(footprint, dtype=bool)
    if footprint_array.ndim not in (2, 3):
        raise NotImplementedError(
            "CellProfiler-compatible morphology currently supports 2-D and 3-D footprints."
        )
    center = np.asarray(footprint_array.shape, dtype=np.int64) // 2
    coords = np.argwhere(footprint_array).astype(np.int64)
    return np.ascontiguousarray(coords - center)


def _border_component_ids(component_labels: np.ndarray) -> set[int]:
    border_values: list[np.ndarray] = []
    for axis in range(component_labels.ndim):
        border_values.append(np.take(component_labels, 0, axis=axis).ravel())
        border_values.append(np.take(component_labels, -1, axis=axis).ravel())
    return {
        int(component_id)
        for component_id in np.concatenate(border_values)
        if component_id != 0
    }


@njit(cache=True, parallel=True)
def _grayscale_morphology_2d_numba(
    image: np.ndarray,
    offset_rows: np.ndarray,
    offset_cols: np.ndarray,
    first_pass_is_dilation: bool,
) -> np.ndarray:
    height, width = image.shape
    intermediate = np.empty_like(image)
    output = np.empty_like(image)
    footprint_size = offset_rows.size
    for row in prange(height):
        for col in range(width):
            best = image[
                _reflect_index_1d(row + int(offset_rows[0]), height),
                _reflect_index_1d(col + int(offset_cols[0]), width),
            ]
            for offset_index in range(1, footprint_size):
                value = image[
                    _reflect_index_1d(row + int(offset_rows[offset_index]), height),
                    _reflect_index_1d(col + int(offset_cols[offset_index]), width),
                ]
                if (first_pass_is_dilation and value > best) or (
                    not first_pass_is_dilation and value < best
                ):
                    best = value
            intermediate[row, col] = best

    for row in prange(height):
        for col in range(width):
            best = intermediate[
                _reflect_index_1d(row + int(offset_rows[0]), height),
                _reflect_index_1d(col + int(offset_cols[0]), width),
            ]
            for offset_index in range(1, footprint_size):
                value = intermediate[
                    _reflect_index_1d(row + int(offset_rows[offset_index]), height),
                    _reflect_index_1d(col + int(offset_cols[offset_index]), width),
                ]
                if (first_pass_is_dilation and value < best) or (
                    not first_pass_is_dilation and value > best
                ):
                    best = value
            output[row, col] = best
    return output


@njit(cache=True)
def _reflect_index_1d(index: int, size: int) -> int:
    if size <= 1:
        return 0
    reflected = index
    while reflected < 0 or reflected >= size:
        if reflected < 0:
            reflected = -reflected - 1
        else:
            reflected = 2 * size - reflected - 1
    return reflected


@njit(cache=True)
def _convex_hull_image_numba(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    output = np.zeros((height, width), dtype=np.bool_)
    point_count = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                point_count += 1

    if point_count == 0:
        return output

    row_count2 = height * 2 + 1
    min_col_by_row = np.empty(row_count2, dtype=np.int64)
    max_col_by_row = np.empty(row_count2, dtype=np.int64)
    point_capacity = max(2, row_count2 * 2)
    point_y = np.empty(point_capacity, dtype=np.int64)
    point_x = np.empty(point_capacity, dtype=np.int64)
    hull_y = np.empty(point_capacity * 2, dtype=np.int64)
    hull_x = np.empty(point_capacity * 2, dtype=np.int64)

    point_count = _collect_convex_hull_diamond_extreme_points_numba(
        mask,
        min_col_by_row,
        max_col_by_row,
        point_y,
        point_x,
    )
    if point_count == 0:
        return output

    hull_count = _monotone_chain_hull_numba(
        point_y,
        point_x,
        point_count,
        hull_y,
        hull_x,
    )
    _paint_convex_hull_mask_numba(output, hull_y, hull_x, hull_count)
    return output


@njit(cache=True)
def _collect_convex_hull_diamond_extreme_points_numba(
    mask: np.ndarray,
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    point_y: np.ndarray,
    point_x: np.ndarray,
) -> int:
    height, width = mask.shape
    row_count2 = height * 2 + 1
    for row_index in range(row_count2):
        min_col_by_row[row_index] = 9223372036854775807
        max_col_by_row[row_index] = -9223372036854775807

    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row,
                    max_col_by_row,
                    2 * y - 1,
                    2 * x,
                )
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row,
                    max_col_by_row,
                    2 * y + 1,
                    2 * x,
                )
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row,
                    max_col_by_row,
                    2 * y,
                    2 * x - 1,
                )
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row,
                    max_col_by_row,
                    2 * y,
                    2 * x + 1,
                )

    point_count = 0
    for row_index in range(row_count2):
        max_col = max_col_by_row[row_index]
        if max_col < -9223372036854775800:
            continue
        row2 = row_index - 1
        min_col = min_col_by_row[row_index]
        point_y[point_count] = row2
        point_x[point_count] = min_col
        point_count += 1
        if max_col != min_col:
            point_y[point_count] = row2
            point_x[point_count] = max_col
            point_count += 1
    return point_count


@njit(cache=True)
def _add_convex_hull_diamond_vertex_numba(
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    row2: int,
    col2: int,
) -> None:
    row_index = row2 + 1
    if col2 < min_col_by_row[row_index]:
        min_col_by_row[row_index] = col2
    if col2 > max_col_by_row[row_index]:
        max_col_by_row[row_index] = col2


@njit(cache=True)
def _cross_convex_hull_points_numba(
    ay: int,
    ax: int,
    by: int,
    bx: int,
    cy: int,
    cx: int,
) -> int:
    return (by - ay) * (cx - ax) - (bx - ax) * (cy - ay)


@njit(cache=True)
def _monotone_chain_hull_numba(
    point_y: np.ndarray,
    point_x: np.ndarray,
    point_count: int,
    hull_y: np.ndarray,
    hull_x: np.ndarray,
) -> int:
    if point_count <= 1:
        if point_count == 1:
            hull_y[0] = point_y[0]
            hull_x[0] = point_x[0]
        return point_count

    hull_count = 0
    for index in range(point_count):
        py = point_y[index]
        px = point_x[index]
        while hull_count >= 2 and _cross_convex_hull_points_numba(
            hull_y[hull_count - 2],
            hull_x[hull_count - 2],
            hull_y[hull_count - 1],
            hull_x[hull_count - 1],
            py,
            px,
        ) <= 0:
            hull_count -= 1
        hull_y[hull_count] = py
        hull_x[hull_count] = px
        hull_count += 1

    lower_count = hull_count
    for index in range(point_count - 2, -1, -1):
        py = point_y[index]
        px = point_x[index]
        while hull_count > lower_count and _cross_convex_hull_points_numba(
            hull_y[hull_count - 2],
            hull_x[hull_count - 2],
            hull_y[hull_count - 1],
            hull_x[hull_count - 1],
            py,
            px,
        ) <= 0:
            hull_count -= 1
        hull_y[hull_count] = py
        hull_x[hull_count] = px
        hull_count += 1

    if hull_count > 1:
        hull_count -= 1
    return hull_count


@njit(cache=True)
def _paint_convex_hull_mask_numba(
    output: np.ndarray,
    hull_y: np.ndarray,
    hull_x: np.ndarray,
    hull_count: int,
) -> None:
    if hull_count <= 0:
        return
    if hull_count == 1:
        if hull_y[0] % 2 != 0 or hull_x[0] % 2 != 0:
            return
        y = hull_y[0] // 2
        x = hull_x[0] // 2
        if y >= 0 and y < output.shape[0] and x >= 0 and x < output.shape[1]:
            output[y, x] = True
        return

    min_row2 = hull_y[0]
    max_row2 = hull_y[0]
    min_col2 = hull_x[0]
    max_col2 = hull_x[0]
    for index in range(1, hull_count):
        row2 = hull_y[index]
        col2 = hull_x[index]
        if row2 < min_row2:
            min_row2 = row2
        if row2 > max_row2:
            max_row2 = row2
        if col2 < min_col2:
            min_col2 = col2
        if col2 > max_col2:
            max_col2 = col2

    if hull_count == 2:
        _paint_convex_hull_line_mask_numba(
            output,
            hull_y[0],
            hull_x[0],
            hull_y[1],
            hull_x[1],
            min_row2,
            max_row2,
            min_col2,
            max_col2,
        )
        return

    area2 = 0
    for index in range(hull_count):
        next_index = 0 if index == hull_count - 1 else index + 1
        area2 += hull_y[index] * hull_x[next_index]
        area2 -= hull_y[next_index] * hull_x[index]
    positive_orientation = area2 >= 0

    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2_numba(min_row2))
    max_y = min(image_height - 1, _floor_div2_numba(max_row2))
    min_x = max(0, _ceil_div2_numba(min_col2))
    max_x = min(image_width - 1, _floor_div2_numba(max_col2))

    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            query_col2 = x * 2
            inside = True
            for index in range(hull_count):
                next_index = 0 if index == hull_count - 1 else index + 1
                cross = _cross_convex_hull_points_numba(
                    hull_y[index],
                    hull_x[index],
                    hull_y[next_index],
                    hull_x[next_index],
                    query_row2,
                    query_col2,
                )
                if positive_orientation:
                    if cross < 0:
                        inside = False
                        break
                elif cross > 0:
                    inside = False
                    break
            if inside:
                output[y, x] = True


@njit(cache=True)
def _ceil_div2_numba(value: int) -> int:
    if value >= 0:
        return (value + 1) // 2
    return value // 2


@njit(cache=True)
def _floor_div2_numba(value: int) -> int:
    if value >= 0:
        return value // 2
    return -((-value + 1) // 2)


@njit(cache=True)
def _paint_convex_hull_line_mask_numba(
    output: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    min_row2: int,
    max_row2: int,
    min_col2: int,
    max_col2: int,
) -> None:
    dy = y1 - y0
    dx = x1 - x0
    length2 = dy * dy + dx * dx
    if length2 == 0:
        if y0 % 2 == 0 and x0 % 2 == 0:
            y = y0 // 2
            x = x0 // 2
            if y >= 0 and y < output.shape[0] and x >= 0 and x < output.shape[1]:
                output[y, x] = True
        return

    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2_numba(min_row2))
    max_y = min(image_height - 1, _floor_div2_numba(max_row2))
    min_x = max(0, _ceil_div2_numba(min_col2))
    max_x = min(image_width - 1, _floor_div2_numba(max_col2))
    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            query_col2 = x * 2
            dot = (query_row2 - y0) * dy + (query_col2 - x0) * dx
            if dot < 0 or dot > length2:
                continue
            cross = dy * (query_col2 - x0) - dx * (query_row2 - y0)
            if cross == 0:
                output[y, x] = True


@njit(cache=True, parallel=True)
def _local_maxima_by_label_numba(
    image: np.ndarray,
    labels: np.ndarray,
    offset_y: np.ndarray,
    offset_x: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    maxima = np.zeros((height, width), dtype=np.bool_)
    for y in prange(height):
        for x in range(width):
            label = labels[y, x]
            if label <= 0:
                continue
            current = image[y, x]
            max_value = -np.inf
            for offset_index in range(offset_y.size):
                neighbor_y = y + offset_y[offset_index]
                neighbor_x = x + offset_x[offset_index]
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                ):
                    continue
                if labels[neighbor_y, neighbor_x] != label:
                    continue
                value = image[neighbor_y, neighbor_x]
                if value > max_value:
                    max_value = value
            maxima[y, x] = current == max_value
    return maxima


@njit(cache=True, parallel=True)
def _smooth_image_for_declumping_numba(
    image: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    radius = kernel.size // 2
    edge_vertical = np.empty((height, width), dtype=np.float64)
    image_vertical = np.empty((height, width), dtype=np.float64)
    edge_array = np.empty((height, width), dtype=np.float64)
    smoothed_image = np.empty((height, width), dtype=np.float64)

    for y in prange(height):
        for x in range(width):
            edge_sum = 0.0
            image_sum = 0.0
            for kernel_index in range(kernel.size):
                iy = y + kernel_index - radius
                if iy < 0 or iy >= height:
                    continue
                kernel_value = kernel[kernel_index]
                if mask[iy, x]:
                    edge_sum += kernel_value
                    image_sum += float(image[iy, x]) * kernel_value
            edge_vertical[y, x] = edge_sum
            image_vertical[y, x] = image_sum

    for y in prange(height):
        for x in range(width):
            edge_sum = 0.0
            image_sum = 0.0
            for kernel_index in range(kernel.size):
                ix = x + kernel_index - radius
                if ix < 0 or ix >= width:
                    continue
                kernel_value = kernel[kernel_index]
                edge_sum += edge_vertical[y, ix] * kernel_value
                image_sum += image_vertical[y, ix] * kernel_value
            edge_array[y, x] = edge_sum
            smoothed_image[y, x] = image_sum

    output = np.empty_like(image)
    for y in prange(height):
        for x in range(width):
            if mask[y, x]:
                edge_value = edge_array[y, x]
                if edge_value != 0.0:
                    output[y, x] = smoothed_image[y, x] / edge_value
                else:
                    output[y, x] = image[y, x]
            else:
                output[y, x] = 0
    return output


@njit(cache=True)
def _foreground_components_2d_numba(
    mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    capacity = height * width
    labels = np.zeros((height, width), dtype=np.int32)
    queue_y = np.empty(capacity, dtype=np.int64)
    queue_x = np.empty(capacity, dtype=np.int64)
    component_count = 0

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or labels[start_y, start_x] != 0:
                continue
            component_count += 1
            head = 0
            tail = 1
            queue_y[0] = start_y
            queue_x[0] = start_x
            labels[start_y, start_x] = component_count

            while head < tail:
                y = queue_y[head]
                x = queue_x[head]
                head += 1
                for dy in range(-1, 2):
                    ny = y + dy
                    if ny < 0 or ny >= height:
                        continue
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        nx = x + dx
                        if nx < 0 or nx >= width:
                            continue
                        if not mask[ny, nx] or labels[ny, nx] != 0:
                            continue
                        labels[ny, nx] = component_count
                        queue_y[tail] = ny
                        queue_x[tail] = nx
                        tail += 1
    return labels, component_count


@njit(cache=True)
def _background_components_2d_numba(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    height, width = labels.shape
    capacity = height * width
    components = np.zeros((height, width), dtype=np.int32)
    sizes = np.zeros(capacity + 1, dtype=np.int64)
    touches_border = np.zeros(capacity + 1, dtype=np.bool_)
    queue_y = np.empty(capacity, dtype=np.int64)
    queue_x = np.empty(capacity, dtype=np.int64)
    component_count = 0

    for start_y in range(height):
        for start_x in range(width):
            if labels[start_y, start_x] != 0 or components[start_y, start_x] != 0:
                continue
            component_count += 1
            head = 0
            tail = 1
            queue_y[0] = start_y
            queue_x[0] = start_x
            components[start_y, start_x] = component_count

            while head < tail:
                y = queue_y[head]
                x = queue_x[head]
                head += 1
                sizes[component_count] += 1
                if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                    touches_border[component_count] = True

                if y > 0 and labels[y - 1, x] == 0 and components[y - 1, x] == 0:
                    components[y - 1, x] = component_count
                    queue_y[tail] = y - 1
                    queue_x[tail] = x
                    tail += 1
                if (
                    y + 1 < height
                    and labels[y + 1, x] == 0
                    and components[y + 1, x] == 0
                ):
                    components[y + 1, x] = component_count
                    queue_y[tail] = y + 1
                    queue_x[tail] = x
                    tail += 1
                if x > 0 and labels[y, x - 1] == 0 and components[y, x - 1] == 0:
                    components[y, x - 1] = component_count
                    queue_y[tail] = y
                    queue_x[tail] = x - 1
                    tail += 1
                if (
                    x + 1 < width
                    and labels[y, x + 1] == 0
                    and components[y, x + 1] == 0
                ):
                    components[y, x + 1] = component_count
                    queue_y[tail] = y
                    queue_x[tail] = x + 1
                    tail += 1

    return components, sizes, touches_border, component_count


@njit(cache=True)
def _hole_fill_flags_below_size_numba(
    sizes: np.ndarray,
    touches_border: np.ndarray,
    component_count: int,
    maximum_hole_size: int,
) -> np.ndarray:
    fill_flags = np.zeros(component_count + 1, dtype=np.bool_)
    for component_id in range(1, component_count + 1):
        fill_flags[component_id] = (
            not touches_border[component_id]
            and sizes[component_id] < maximum_hole_size
        )
    return fill_flags


@njit(cache=True, parallel=True)
def _fill_binary_holes_from_components_numba(
    labels: np.ndarray,
    components: np.ndarray,
    fill_flags: np.ndarray,
) -> np.ndarray:
    height, width = labels.shape
    has_fillable_component = False
    for component in range(fill_flags.size):
        if fill_flags[component]:
            has_fillable_component = True
            break
    if not has_fillable_component:
        return labels

    output = labels.copy()
    for y in prange(height):
        for x in range(width):
            component = components[y, x]
            if component > 0 and fill_flags[component]:
                output[y, x] = True
    return output


@njit(cache=True)
def _fill_labeled_holes_single_label_components_numba(
    labels: np.ndarray,
    components: np.ndarray,
    fill_flags: np.ndarray,
) -> np.ndarray:
    height, width = labels.shape
    component_labels = np.zeros(fill_flags.size, dtype=np.int64)

    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0 or not fill_flags[component]:
                continue
            if y > 0:
                _record_component_boundary_label_numba(
                    component_labels,
                    component,
                    int(labels[y - 1, x]),
                )
            if y + 1 < height:
                _record_component_boundary_label_numba(
                    component_labels,
                    component,
                    int(labels[y + 1, x]),
                )
            if x > 0:
                _record_component_boundary_label_numba(
                    component_labels,
                    component,
                    int(labels[y, x - 1]),
                )
            if x + 1 < width:
                _record_component_boundary_label_numba(
                    component_labels,
                    component,
                    int(labels[y, x + 1]),
                )

    has_fillable_component = False
    for component in range(component_labels.size):
        if component_labels[component] > 0:
            has_fillable_component = True
            break
    if not has_fillable_component:
        return labels

    output = labels.copy()
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component > 0 and fill_flags[component]:
                label = component_labels[component]
                if label > 0:
                    output[y, x] = label
    return output


@njit(cache=True)
def _restore_removed_declump_basins_numba(
    pre_declump_labels: np.ndarray,
    labels_before_size_filter: np.ndarray,
    labels_after_size_filter: np.ndarray,
) -> np.ndarray:
    height, width = labels_before_size_filter.shape
    output = labels_after_size_filter.copy()
    max_pre_declump_label = 0
    for y in range(height):
        for x in range(width):
            pre_label = int(pre_declump_labels[y, x])
            if pre_label > max_pre_declump_label:
                max_pre_declump_label = pre_label

    component_surviving_label = np.zeros(max_pre_declump_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            pre_label = int(pre_declump_labels[y, x])
            after_label = int(labels_after_size_filter[y, x])
            if pre_label <= 0 or after_label <= 0:
                continue
            current_label = component_surviving_label[pre_label]
            if current_label == 0:
                component_surviving_label[pre_label] = after_label
            elif current_label != after_label:
                component_surviving_label[pre_label] = -1

    visited = np.zeros((height, width), dtype=np.bool_)
    stack_y = np.empty(height * width, dtype=np.int64)
    stack_x = np.empty(height * width, dtype=np.int64)
    component_y = np.empty(height * width, dtype=np.int64)
    component_x = np.empty(height * width, dtype=np.int64)

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x]:
                continue
            if (
                labels_before_size_filter[start_y, start_x] <= 0
                or labels_after_size_filter[start_y, start_x] > 0
            ):
                visited[start_y, start_x] = True
                continue

            stack_size = 1
            stack_y[0] = start_y
            stack_x[0] = start_x
            visited[start_y, start_x] = True
            component_size = 0
            boundary_label = 0
            has_multiple_boundary_labels = False
            pre_declump_label = 0
            has_multiple_pre_declump_labels = False

            while stack_size:
                stack_size -= 1
                y = stack_y[stack_size]
                x = stack_x[stack_size]
                component_y[component_size] = y
                component_x[component_size] = x
                component_size += 1
                current_pre_declump_label = int(pre_declump_labels[y, x])
                if current_pre_declump_label <= 0:
                    has_multiple_pre_declump_labels = True
                elif pre_declump_label == 0:
                    pre_declump_label = current_pre_declump_label
                elif pre_declump_label != current_pre_declump_label:
                    has_multiple_pre_declump_labels = True

                for dy in range(-1, 2):
                    yy = y + dy
                    if yy < 0 or yy >= height:
                        continue
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        xx = x + dx
                        if xx < 0 or xx >= width:
                            continue

                        neighbor_after = labels_after_size_filter[yy, xx]
                        if neighbor_after > 0:
                            if boundary_label == 0:
                                boundary_label = neighbor_after
                            elif boundary_label != neighbor_after:
                                has_multiple_boundary_labels = True
                            continue

                        if visited[yy, xx]:
                            continue
                        if (
                            labels_before_size_filter[yy, xx] > 0
                            and labels_after_size_filter[yy, xx] == 0
                        ):
                            visited[yy, xx] = True
                            stack_y[stack_size] = yy
                            stack_x[stack_size] = xx
                            stack_size += 1

            if (
                boundary_label <= 0
                or has_multiple_boundary_labels
                or pre_declump_label <= 0
                or has_multiple_pre_declump_labels
            ):
                continue
            surviving_label = component_surviving_label[pre_declump_label]
            if surviving_label <= 0 or surviving_label != boundary_label:
                continue
            for index in range(component_size):
                output[component_y[index], component_x[index]] = boundary_label
    return output


@njit(cache=True)
def _record_component_boundary_label_numba(
    component_labels: np.ndarray,
    component: int,
    label: int,
) -> None:
    if label <= 0:
        return
    current = component_labels[component]
    if current == 0:
        component_labels[component] = label
    elif current != label:
        component_labels[component] = -1


@njit(cache=True)
def _fill_labeled_holes_from_components_numba(
    labels: np.ndarray,
    components: np.ndarray,
    fill_flags: np.ndarray,
) -> np.ndarray:
    height, width = labels.shape
    capacity = height * width
    output = labels.copy()
    queue_y = np.empty(capacity, dtype=np.int64)
    queue_x = np.empty(capacity, dtype=np.int64)
    head = 0
    tail = 0

    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0 or not fill_flags[component]:
                continue
            label = _first_adjacent_foreground_label_numba(labels, y, x)
            if label == 0:
                continue
            output[y, x] = label
            queue_y[tail] = y
            queue_x[tail] = x
            tail += 1

    while head < tail:
        y = queue_y[head]
        x = queue_x[head]
        head += 1
        component = components[y, x]
        label = output[y, x]

        if (
            y > 0
            and components[y - 1, x] == component
            and fill_flags[component]
            and output[y - 1, x] == 0
        ):
            output[y - 1, x] = label
            queue_y[tail] = y - 1
            queue_x[tail] = x
            tail += 1
        if (
            y + 1 < height
            and components[y + 1, x] == component
            and fill_flags[component]
            and output[y + 1, x] == 0
        ):
            output[y + 1, x] = label
            queue_y[tail] = y + 1
            queue_x[tail] = x
            tail += 1
        if (
            x > 0
            and components[y, x - 1] == component
            and fill_flags[component]
            and output[y, x - 1] == 0
        ):
            output[y, x - 1] = label
            queue_y[tail] = y
            queue_x[tail] = x - 1
            tail += 1
        if (
            x + 1 < width
            and components[y, x + 1] == component
            and fill_flags[component]
            and output[y, x + 1] == 0
        ):
            output[y, x + 1] = label
            queue_y[tail] = y
            queue_x[tail] = x + 1
            tail += 1

    return output


@njit(cache=True)
def _first_adjacent_foreground_label_numba(
    labels: np.ndarray,
    y: int,
    x: int,
):
    height, width = labels.shape
    for dy in range(-1, 2):
        neighbor_y = y + dy
        if neighbor_y < 0 or neighbor_y >= height:
            continue
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            neighbor_x = x + dx
            if neighbor_x < 0 or neighbor_x >= width:
                continue
            label = labels[neighbor_y, neighbor_x]
            if label != 0:
                return label
    return labels[y, x]


@njit(cache=True)
def _binary_shrink_2d_numba(mask: np.ndarray, tables: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    current = np.zeros((height + 2, width + 2), dtype=np.bool_)
    capacity = height * width
    coords_y = np.empty(capacity, dtype=np.int64)
    coords_x = np.empty(capacity, dtype=np.int64)
    removed_y = np.empty(capacity, dtype=np.int64)
    removed_x = np.empty(capacity, dtype=np.int64)
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            padded_y = y + 1
            padded_x = x + 1
            current[padded_y, padded_x] = True
            coords_y[count] = padded_y
            coords_x[count] = padded_x
            count += 1

    iterations = count
    for _iteration in range(iterations):
        pixel_count = count
        for table_index in range(4):
            table = tables[table_index]
            new_count = 0
            removed_count = 0
            for coord_index in range(count):
                y = coords_y[coord_index]
                x = coords_x[coord_index]
                if not current[y, x]:
                    continue

                pattern_index = 0
                bit = 1
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if current[y + dy, x + dx]:
                            pattern_index += bit
                        bit <<= 1

                if table[pattern_index]:
                    coords_y[new_count] = y
                    coords_x[new_count] = x
                    new_count += 1
                else:
                    removed_y[removed_count] = y
                    removed_x[removed_count] = x
                    removed_count += 1
            for removed_index in range(removed_count):
                current[removed_y[removed_index], removed_x[removed_index]] = False
            count = new_count

        if count == pixel_count:
            break

    output = np.zeros((height, width), dtype=np.bool_)
    for coord_index in range(count):
        output[coords_y[coord_index] - 1, coords_x[coord_index] - 1] = True
    return output


@njit(cache=True)
def _seed_points_from_components_numba(
    components: np.ndarray,
    component_count: int,
) -> np.ndarray:
    height, width = components.shape
    counts = np.zeros(component_count + 1, dtype=np.int64)
    sum_y = np.zeros(component_count + 1, dtype=np.float64)
    sum_x = np.zeros(component_count + 1, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0:
                continue
            counts[component] += 1
            sum_y[component] += y
            sum_x[component] += x

    best_distance = np.empty(component_count + 1, dtype=np.float64)
    best_y = np.full(component_count + 1, -1, dtype=np.int64)
    best_x = np.full(component_count + 1, -1, dtype=np.int64)
    for component in range(component_count + 1):
        best_distance[component] = np.inf

    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0:
                continue
            centroid_y = sum_y[component] / counts[component]
            centroid_x = sum_x[component] / counts[component]
            dy = y - centroid_y
            dx = x - centroid_x
            distance = dy * dy + dx * dx
            if distance < best_distance[component]:
                best_distance[component] = distance
                best_y[component] = y
                best_x[component] = x

    seeds = np.zeros((height, width), dtype=np.bool_)
    for component in range(1, component_count + 1):
        y = best_y[component]
        x = best_x[component]
        if y >= 0 and x >= 0:
            seeds[y, x] = True
    return seeds


@njit(cache=True)
def _relabel_sequential_numba(labels: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label > max_label:
                max_label = label

    if max_label <= 0:
        return np.zeros((height, width), dtype=np.int32), 0

    present = np.zeros(max_label + 1, dtype=np.bool_)
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label > 0:
                present[label] = True

    mapping = np.zeros(max_label + 1, dtype=np.int32)
    count = 0
    for label in range(1, max_label + 1):
        if present[label]:
            count += 1
            mapping[label] = count

    output = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label > 0:
                output[y, x] = mapping[label]
    return output, count


@njit(cache=True)
def _relabel_sequential_3d_numba(labels: np.ndarray) -> tuple[np.ndarray, int]:
    plane_count, height, width = labels.shape
    max_label = 0
    for plane_index in range(plane_count):
        for y in range(height):
            for x in range(width):
                label = labels[plane_index, y, x]
                if label > max_label:
                    max_label = label

    if max_label <= 0:
        return np.zeros((plane_count, height, width), dtype=np.int32), 0

    present = np.zeros(max_label + 1, dtype=np.bool_)
    for plane_index in range(plane_count):
        for y in range(height):
            for x in range(width):
                label = labels[plane_index, y, x]
                if label > 0:
                    present[label] = True

    mapping = np.zeros(max_label + 1, dtype=np.int32)
    count = 0
    for label in range(1, max_label + 1):
        if present[label]:
            count += 1
            mapping[label] = count

    output = np.zeros((plane_count, height, width), dtype=np.int32)
    for plane_index in range(plane_count):
        for y in range(height):
            for x in range(width):
                label = labels[plane_index, y, x]
                if label > 0:
                    output[plane_index, y, x] = mapping[label]
    return output, count


class ExpandShrinkOperationStrategy(
    EnumKeyedStrategyMixin[ExpandShrinkMode],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal CellProfiler ExpandOrShrinkObjects operation strategy."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "mode"
    mode: ClassVar[ExpandShrinkMode | None] = None
    strategy_label: ClassVar[str | None] = None
    cellprofiler_operations: ClassVar[tuple[CellProfilerExpandShrinkOperation, ...]] = ()

    @classmethod
    def for_mode(
        cls,
        mode: ExpandShrinkMode | str,
    ) -> "ExpandShrinkOperationStrategy":
        resolved = coerce_cellprofiler_enum(ExpandShrinkMode, mode)
        return cls.for_enum_member(resolved)

    @abstractmethod
    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        """Return transformed labels for this operation mode."""

    def output_domain(self, labels: np.ndarray) -> ObjectLabelDomain:
        """Return CP's semantic object domain for transformed labels."""
        return ObjectLabelDomain(
            declared_object_count=dense_object_label_max_present_id(labels),
            scope=ObjectLabelDomainScope.PLANE,
        )

    @staticmethod
    def apply_label_planes(
        labels: np.ndarray,
        operation: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        output = np.empty_like(labels, dtype=np.int32)
        label_planes = labels.reshape((-1, *labels.shape[-2:]))
        output_planes = output.reshape((-1, *output.shape[-2:]))
        for plane_index in range(label_planes.shape[0]):
            output_planes[plane_index] = operation(label_planes[plane_index])
        return output


class ExpandDefinedPixelsStrategy(ExpandShrinkOperationStrategy):
    """Expand labeled objects by a fixed pixel radius."""

    mode = ExpandShrinkMode.EXPAND_DEFINED_PIXELS
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.EXPAND_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.EXPAND_BY_MEASUREMENT,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return self.expand_defined_pixels(labels, iterations)

    def expand_defined_pixels(
        self,
        labels: np.ndarray,
        iterations: int,
    ) -> np.ndarray:
        """Expand labeled objects by a defined number of pixels."""
        from scipy.ndimage import distance_transform_edt

        if iterations <= 0:
            return labels.copy()
        labels_int = labels.astype(np.int32, copy=False)
        if labels_int.ndim > 2:
            return self.apply_label_planes(
                labels_int,
                lambda plane: self.expand_defined_pixels(plane, iterations),
            )
        if _labels_are_points_numba(np.ascontiguousarray(labels_int)):
            return _expand_point_labels_defined_pixels_numba(
                np.ascontiguousarray(labels_int),
                int(iterations),
            )

        result = labels_int.copy()
        background = labels_int == 0
        distances, indices = distance_transform_edt(background, return_indices=True)
        expand_mask = background & (distances <= iterations)
        result[expand_mask] = labels_int[indices[0][expand_mask], indices[1][expand_mask]]
        return result


class ExpandInfiniteStrategy(ExpandShrinkOperationStrategy):
    """Expand labeled objects until all background is assigned."""

    mode = ExpandShrinkMode.EXPAND_INFINITE
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.EXPAND_UNTIL_TOUCHING,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _expand_until_touching(labels)


class ShrinkDefinedPixelsStrategy(ExpandShrinkOperationStrategy):
    """Shrink labeled objects by a fixed pixel radius."""

    mode = ExpandShrinkMode.SHRINK_DEFINED_PIXELS
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.SHRINK_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.SHRINK_BY_MEASUREMENT,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _shrink_defined_pixels(labels, iterations, fill_holes)


class ShrinkToPointStrategy(ExpandShrinkOperationStrategy):
    """Shrink each object to its center point."""

    mode = ExpandShrinkMode.SHRINK_TO_POINT
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.SHRINK_TO_POINT,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return self.shrink_to_point(labels, fill_holes)

    def shrink_to_point(
        self,
        labels: np.ndarray,
        fill: bool,
    ) -> np.ndarray:
        """Shrink each labeled object to a single point at its centroid."""
        labels_int = labels.astype(np.int32, copy=False)
        if labels_int.ndim > 2:
            return self.apply_label_planes(
                labels_int,
                lambda plane: self.shrink_to_point(plane, fill),
            )
        if labels_int.size == 0 or int(labels_int.max()) <= 0:
            return np.zeros_like(labels_int)
        return _shrink_to_point_numba(np.ascontiguousarray(labels_int))


class AddDividingLinesStrategy(ExpandShrinkOperationStrategy):
    """Remove touching object boundary pixels."""

    mode = ExpandShrinkMode.ADD_DIVIDING_LINES
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.ADD_DIVIDING_LINES,
    )

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _add_dividing_lines(labels)


class DespurStrategy(ExpandShrinkOperationStrategy):
    """Remove object spurs by repeated opening."""

    mode = ExpandShrinkMode.DESPUR
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.DESPUR,)

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _despur(labels, iterations)


class SkeletonizeStrategy(ExpandShrinkOperationStrategy):
    """Reduce each object to a skeleton."""

    mode = ExpandShrinkMode.SKELETONIZE
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.SKELETONIZE,)

    def apply(
        self,
        labels: np.ndarray,
        *,
        iterations: int,
        fill_holes: bool,
    ) -> np.ndarray:
        return _skeletonize_labels(labels)


@njit(cache=True)
def _labels_are_points_numba(labels: np.ndarray) -> bool:
    max_label = 0
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label
    if max_label <= 0:
        return True

    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1
            if counts[label] > 1:
                return False
    return True


@njit(cache=True)
def _expand_point_labels_defined_pixels_numba(
    labels: np.ndarray,
    radius: int,
) -> np.ndarray:
    height, width = labels.shape
    output = labels.copy()
    radius_squared = radius * radius
    initial_distance = radius_squared + 1
    best_distance = np.full(labels.shape, initial_distance, dtype=np.int32)
    best_y = np.full(labels.shape, 2147483647, dtype=np.int32)
    best_x = np.full(labels.shape, 2147483647, dtype=np.int32)

    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            for dy in range(-radius, radius + 1):
                yy = y + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-radius, radius + 1):
                    xx = x + dx
                    if xx < 0 or xx >= width:
                        continue
                    distance = dy * dy + dx * dx
                    if distance > radius_squared:
                        continue
                    if (
                        distance < best_distance[yy, xx]
                        or (
                            distance == best_distance[yy, xx]
                            and (
                                x < best_x[yy, xx]
                                or (x == best_x[yy, xx] and y < best_y[yy, xx])
                            )
                        )
                    ):
                        best_distance[yy, xx] = distance
                        best_y[yy, xx] = y
                        best_x[yy, xx] = x
                        output[yy, xx] = label
    return output


def _expand_until_touching(labels: np.ndarray) -> np.ndarray:
    """Expand labeled objects until they touch."""
    from scipy.ndimage import distance_transform_edt

    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels,
            _expand_until_touching,
        )
    if labels.max() == 0:
        return labels.copy()
    mask = labels > 0
    _distances, indices = distance_transform_edt(~mask, return_indices=True)
    return labels[indices[0], indices[1]]


def _shrink_defined_pixels(labels: np.ndarray, iterations: int, fill: bool) -> np.ndarray:
    """Shrink labeled objects by a defined number of pixels."""
    if iterations <= 0:
        return labels.copy()

    original = labels.astype(np.int32, copy=False)
    if original.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            original,
            lambda plane: _shrink_defined_pixels(plane, iterations, fill),
        )
    result = original.copy()
    for _ in range(iterations):
        same_neighbors = np.zeros(result.shape, dtype=bool)
        center = result[1:-1, 1:-1]
        same_neighbors[1:-1, 1:-1] = (
            (center > 0)
            & (center == result[:-2, 1:-1])
            & (center == result[2:, 1:-1])
            & (center == result[1:-1, :-2])
            & (center == result[1:-1, 2:])
        )
        result = np.where(same_neighbors, result, 0).astype(np.int32, copy=False)

    if fill:
        _restore_eroded_objects_to_centroids(original, result)

    return result


def _restore_eroded_objects_to_centroids(
    original: np.ndarray,
    eroded: np.ndarray,
) -> None:
    """Preserve one centroid pixel for labels fully removed by shrinking."""
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        original.astype(np.int32, copy=False)
    )
    if region_props.label.size == 0:
        return
    remaining_ids = set(int(label_id) for label_id in np.unique(eroded) if label_id > 0)
    for index, label_id in enumerate(region_props.label):
        label_int = int(label_id)
        if label_int in remaining_ids:
            continue
        cy = int(region_props.centroid_y[index])
        cx = int(region_props.centroid_x[index])
        eroded[cy, cx] = label_int


@njit(cache=True)
def _shrink_to_point_numba(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label

    y_sums = np.zeros(max_label + 1, dtype=np.float64)
    x_sums = np.zeros(max_label + 1, dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            y_sums[label] += float(y)
            x_sums[label] += float(x)
            counts[label] += 1

    result = np.zeros(labels.shape, dtype=np.int32)
    for label in range(1, max_label + 1):
        count = counts[label]
        if count <= 0:
            continue
        cy = int(y_sums[label] / float(count))
        cx = int(x_sums[label] / float(count))
        if cy < 0:
            cy = 0
        elif cy >= height:
            cy = height - 1
        if cx < 0:
            cx = 0
        elif cx >= width:
            cx = width - 1
        result[cy, cx] = label
    return result


def _add_dividing_lines(labels: np.ndarray) -> np.ndarray:
    """Add 1-pixel dividing lines between touching objects."""
    from scipy.ndimage import maximum_filter, minimum_filter

    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels,
            _add_dividing_lines,
        )
    if labels.max() == 0:
        return labels.copy()
    result = labels.copy()
    max_filt = maximum_filter(labels, size=3)
    min_filt = minimum_filter(labels, size=3)
    boundary = (max_filt != min_filt) & (min_filt > 0)
    result[boundary] = 0
    return result


def _despur(labels: np.ndarray, iterations: int) -> np.ndarray:
    """Remove spurs from labeled objects."""
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

    if iterations <= 0:
        return labels.copy()
    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels,
            lambda plane: _despur(plane, iterations),
        )
    result = np.zeros_like(labels)
    struct = generate_binary_structure(2, 1)
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        opened = binary_erosion(obj_mask, structure=struct, iterations=iterations)
        opened = binary_dilation(opened, structure=struct, iterations=iterations)
        result[opened] = label_id
    return result


def _skeletonize_labels(labels: np.ndarray) -> np.ndarray:
    """Reduce labeled objects to their skeletons."""
    from skimage.morphology import skeletonize

    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels,
            _skeletonize_labels,
        )
    result = np.zeros_like(labels)
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        skeleton = skeletonize(obj_mask)
        result[skeleton] = label_id
    return result


def prepare_expand_or_shrink_objects() -> None:
    """Compile kernels used by common object expansion/shrink modes."""
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:5, 3:7] = 1
    labels[8:12, 9:14] = 2
    points = ShrinkToPointStrategy().shrink_to_point(labels, False)
    ExpandDefinedPixelsStrategy().expand_defined_pixels(points, 2)


@dataclass(frozen=True, slots=True)
class MaskObjectsStats:
    """MaskObjects count summary for one runtime plane."""

    slice_index: int
    original_object_count: int
    remaining_object_count: int
    objects_removed: int


@dataclass(frozen=True, slots=True)
class MaskObjectsPlaneResult:
    """MaskObjects result for one runtime plane."""

    labels: np.ndarray
    stats: MaskObjectsStats
    relationships: ParentChildRelationshipPayload


@dataclass(frozen=True, slots=True)
class MaskObjectsPlaneOperation:
    """CellProfiler MaskObjects semantics for one aligned object-label plane."""

    overlap_handling: MaskObjectsOverlapHandling
    overlap_fraction: float
    numbering: MaskObjectsNumberingChoice
    invert_mask: bool
    relationship_backend: ObjectRelationshipBackendStrategy

    def apply(
        self,
        label_image: np.ndarray,
        mask: np.ndarray,
        *,
        slice_index: int = 0,
    ) -> MaskObjectsPlaneResult:
        import scipy.ndimage as ndi

        label_image = np.asarray(label_image, dtype=np.int32)
        _aligned_labels, mask = aligned_dense_object_labels_and_mask(label_image, mask)
        label_image = _aligned_labels.astype(np.int32, copy=False)

        binary_mask = mask > 0 if mask.max() > 1 else mask.astype(bool)
        if self.invert_mask:
            binary_mask = ~binary_mask

        masked_labels = label_image.copy()
        nobjects = int(np.max(label_image))
        if nobjects == 0:
            return MaskObjectsPlaneResult(
                labels=masked_labels,
                stats=MaskObjectsStats(
                    slice_index=slice_index,
                    original_object_count=0,
                    remaining_object_count=0,
                    objects_removed=0,
                ),
                relationships=ParentChildRelationshipPayload(
                    parent_ids=(),
                    child_ids=(),
                ),
            )

        binary_mask = _size_binary_mask_like_labels(label_image, binary_mask)
        if self.overlap_handling == MaskObjectsOverlapHandling.MASK:
            masked_labels = masked_labels * binary_mask.astype(masked_labels.dtype)
        else:
            object_indices = np.arange(1, nobjects + 1, dtype=np.int32)
            pixel_counts = np.atleast_1d(
                ndi.sum(binary_mask.astype(np.float64), label_image, object_indices)
            )

            if self.overlap_handling == MaskObjectsOverlapHandling.KEEP:
                keep = pixel_counts > 0
            else:
                total_pixels = np.atleast_1d(
                    ndi.sum(
                        np.ones(label_image.shape, dtype=np.float64),
                        label_image,
                        object_indices,
                    )
                )

                if self.overlap_handling == MaskObjectsOverlapHandling.REMOVE:
                    keep = pixel_counts == total_pixels
                elif (
                    self.overlap_handling
                    == MaskObjectsOverlapHandling.REMOVE_PERCENTAGE
                ):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fractions = np.where(
                            total_pixels > 0,
                            pixel_counts / total_pixels,
                            0,
                        )
                    keep = fractions >= self.overlap_fraction
                else:
                    raise ValueError(
                        "Unsupported MaskObjects overlap handling: "
                        f"{self.overlap_handling!r}"
                    )

            keep_lookup = np.concatenate([[False], keep])
            masked_labels[~keep_lookup[label_image]] = 0

        if self.numbering == MaskObjectsNumberingChoice.RENUMBER:
            unique_labels = np.unique(masked_labels[masked_labels != 0])
            if len(unique_labels) > 0:
                indexer = np.zeros(nobjects + 1, dtype=np.int32)
                indexer[unique_labels] = np.arange(
                    1,
                    len(unique_labels) + 1,
                    dtype=np.int32,
                )
                masked_labels = indexer[masked_labels]
                remaining_count = len(unique_labels)
            else:
                remaining_count = 0
        elif self.numbering == MaskObjectsNumberingChoice.RETAIN:
            remaining_count = len(np.unique(masked_labels[masked_labels != 0]))
        else:
            raise ValueError(f"Unsupported MaskObjects numbering: {self.numbering!r}")

        return MaskObjectsPlaneResult(
            labels=masked_labels,
            stats=MaskObjectsStats(
                slice_index=slice_index,
                original_object_count=nobjects,
                remaining_object_count=remaining_count,
                objects_removed=nobjects - remaining_count,
            ),
            relationships=self.relationship_backend.parent_child_payload_from_labels(
                label_image,
                masked_labels,
            ),
        )


@dataclass(frozen=True, slots=True)
class MaskObjectsOutputLabels:
    """Typed MaskObjects label output preserving input object-label semantics."""

    source: object
    labels: np.ndarray

    def value(self) -> object:
        if not isinstance(self.source, (ObjectLabelPayload, ObjectLabelSet)):
            return self.labels
        plane_domains = dense_object_label_plane_id_domains(
            self.labels,
            domain_scope=ObjectLabelDomainScope.PLANE,
        )
        return object_label_payload_with_dense_labels(
            self.source,
            self.labels,
            domain_declaration=ExplicitObjectLabelDomainDeclaration(
                ObjectLabelDomain(
                    declared_object_id_domains=plane_domains,
                    scope=ObjectLabelDomainScope.PLANE,
                )
            ),
        )


def _size_binary_mask_like_labels(
    labels: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Return a binary mask sized like CP size_similarly(labels, mask)."""
    if binary_mask.shape == labels.shape:
        return binary_mask
    result = np.zeros(labels.shape, dtype=bool)
    common_slices = tuple(
        slice(0, min(label_extent, mask_extent))
        for label_extent, mask_extent in zip(labels.shape, binary_mask.shape, strict=False)
    )
    if not common_slices:
        return result
    result[common_slices] = binary_mask[common_slices]
    return result


@dataclass(frozen=True, slots=True)
class CombineObjectsStats:
    """CellProfiler CombineObjects summary row."""

    slice_index: int
    method: str
    input_objects_x: int
    input_objects_y: int
    output_objects: int


class CombineObjectsStrategy(
    EnumKeyedStrategyMixin[CombineObjectsMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal object-label combination strategy."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    method_label: ClassVar[str | None] = None
    method: ClassVar[CombineObjectsMethod | None] = None

    @classmethod
    def for_method(
        cls,
        method: CombineObjectsMethod | str,
    ) -> "CombineObjectsStrategy":
        return cls.for_enum_member(
            coerce_cellprofiler_enum(CombineObjectsMethod, method)
        )

    @abstractmethod
    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        """Return combined labels for this policy."""

    def result(
        self,
        labels_x: np.ndarray,
        labels_y: np.ndarray,
    ) -> tuple[CombineObjectsStats, np.ndarray]:
        combined_labels = self.combine(labels_x, labels_y)
        method = type(self).method
        if method is None:
            raise TypeError(f"{type(self).__name__} must declare method.")
        return (
            CombineObjectsStats(
                slice_index=0,
                method=method.value,
                input_objects_x=positive_dense_label_count(labels_x),
                input_objects_y=positive_dense_label_count(labels_y),
                output_objects=positive_dense_label_count(combined_labels),
            ),
            combined_labels,
        )


class MergeCombineObjectsStrategy(CombineObjectsStrategy):
    """Merge overlapping objects from two label images into single objects."""

    method = CombineObjectsMethod.MERGE

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label as scipy_label

        combined_binary = ((labels_x > 0) | (labels_y > 0)).astype(np.uint8)
        merged_labels, _ = scipy_label(combined_binary)
        return merged_labels.astype(np.int32)


class PreserveCombineObjectsStrategy(CombineObjectsStrategy):
    """Preserve labels_x and add non-overlapping objects from labels_y."""

    method = CombineObjectsMethod.PRESERVE

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        result = labels_x.copy().astype(np.int32)
        max_label = labels_x.max()
        non_overlapping_mask = (labels_y > 0) & (labels_x == 0)
        if non_overlapping_mask.any():
            y_labels_in_mask = np.unique(labels_y[non_overlapping_mask])
            y_labels_in_mask = y_labels_in_mask[y_labels_in_mask > 0]
            for index, y_label in enumerate(y_labels_in_mask):
                y_object_mask = (labels_y == y_label) & non_overlapping_mask
                result[y_object_mask] = max_label + index + 1
        return result


class DiscardCombineObjectsStrategy(CombineObjectsStrategy):
    """Discard objects from labels_x that overlap labels_y."""

    method = CombineObjectsMethod.DISCARD

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label as scipy_label

        overlap_mask = (labels_x > 0) & (labels_y > 0)
        overlapping_labels = np.unique(labels_x[overlap_mask])
        result = labels_x.copy().astype(np.int32)
        for label_id in overlapping_labels:
            if label_id > 0:
                result[labels_x == label_id] = 0
        if result.max() > 0:
            result, _ = scipy_label(result > 0)
        return result.astype(np.int32)


class SegmentCombineObjectsStrategy(CombineObjectsStrategy):
    """Segment labels_x using labels_y as watershed markers."""

    method = CombineObjectsMethod.SEGMENT

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        from skimage.segmentation import watershed

        binary_x = labels_x > 0
        if not binary_x.any():
            return np.zeros_like(labels_x, dtype=np.int32)
        distance = distance_transform_edt(binary_x)
        markers = labels_y.copy()
        markers[~binary_x] = 0
        if markers.max() == 0:
            return labels_x.astype(np.int32)
        return watershed(-distance, markers, mask=binary_x).astype(np.int32)


def positive_dense_label_count(labels: np.ndarray) -> int:
    """Return the count of positive labels present in a dense label image."""
    return int(len(np.unique(labels)) - (1 if 0 in labels else 0))


__all__ = [
    "CentrosomeNumpyMorphologyBackendStrategy",
    "CellProfilerDeclumpMethod",
    "CombineObjectsStats",
    "CombineObjectsStrategy",
    "ExpandDefinedPixelsStrategy",
    "ExpandInfiniteStrategy",
    "ExpandShrinkOperationStrategy",
    "HolePredicate",
    "MaskObjectsOutputLabels",
    "MaskObjectsPlaneOperation",
    "MaskObjectsPlaneResult",
    "MaskObjectsStats",
    "MorphologyBackendStrategy",
    "NumbaNumpyMorphologyBackendStrategy",
    "NumpyMorphologyBackendStrategy",
    "positive_dense_label_count",
    "prepare_expand_or_shrink_objects",
]
