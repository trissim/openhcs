"""
Converted from CellProfiler: MeasureImageAreaOccupied
Measures the total area in an image that is occupied by objects or foreground.
"""

import numpy as np
from typing import Optional, Sequence, Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.analysis.region_properties import (
    binary_area_and_perimeter_2d,
    label_area_and_rounded_perimeter_2d,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer


class OperandChoice(Enum):
    BINARY_IMAGE = "binary_image"
    OBJECTS = "objects"

    @classmethod
    def from_literal(cls, value: "OperandChoice | str") -> "OperandChoice":
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        if "binary" in normalized:
            return cls.BINARY_IMAGE
        if "object" in normalized:
            return cls.OBJECTS
        return cls(normalized)


@dataclass(frozen=True, slots=True)
class AreaOccupiedRuntimeRow:
    """One typed runtime row for the generic area-occupied runner."""

    operand: OperandChoice
    input_name: str
    retained_image_name: str | None

    @classmethod
    def from_literals(
        cls,
        *,
        operand: OperandChoice | str,
        input_name: str,
        retained_image_name: str | None,
    ) -> "AreaOccupiedRuntimeRow":
        normalized_input_name = input_name.strip()
        if not normalized_input_name:
            raise ValueError("AreaOccupiedRuntimeRow.input_name cannot be empty.")
        return cls(
            operand=OperandChoice.from_literal(operand),
            input_name=normalized_input_name,
            retained_image_name=retained_image_name,
        )


@dataclass
class AreaOccupiedMeasurement:
    """Measurements for area occupied analysis."""
    slice_index: int
    area_occupied: float
    perimeter: float
    total_area: float
    source_image_name: str | None = None

    @classmethod
    def from_area(
        cls,
        *,
        area_occupied: float,
        perimeter: float,
        total_area: float,
        slice_index: int = 0,
        source_image_name: str | None = None,
    ) -> "AreaOccupiedMeasurement":
        return cls(
            slice_index=slice_index,
            area_occupied=area_occupied,
            perimeter=perimeter,
            total_area=total_area,
            source_image_name=source_image_name,
        )


@dataclass(frozen=True, slots=True)
class BinaryAreaOccupiedRequest:
    """Measure occupied area for one binary image plane."""

    image: np.ndarray
    slice_index: int = 0
    source_image_name: str | None = None

    def measure(self) -> tuple[np.ndarray, AreaOccupiedMeasurement]:
        binary_mask = self.image > 0
        area_occupied, perimeter_value = binary_area_and_perimeter_2d(binary_mask)
        measurement = AreaOccupiedMeasurement.from_area(
            area_occupied=area_occupied,
            perimeter=perimeter_value,
            total_area=float(np.prod(self.image.shape)),
            slice_index=self.slice_index,
            source_image_name=self.source_image_name,
        )
        return self.image, measurement


@dataclass(frozen=True, slots=True)
class ObjectLabelsAreaOccupiedRequest:
    """Measure occupied area for one object-label plane."""

    image: np.ndarray
    labels: np.ndarray
    slice_index: int = 0
    source_image_name: str | None = None

    def measure(self) -> tuple[np.ndarray, AreaOccupiedMeasurement]:
        label_array = object_label_dense_array(self.labels)
        area_occupied, perimeter_value = label_area_and_perimeter(label_array)
        measurement = AreaOccupiedMeasurement.from_area(
            area_occupied=area_occupied,
            perimeter=perimeter_value,
            total_area=float(np.prod(label_array.shape)),
            slice_index=self.slice_index,
            source_image_name=self.source_image_name,
        )
        object_region_mask = (label_array > 0).astype(np.asarray(self.image).dtype)
        return object_region_mask, measurement


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(("area_measurements", csv_materializer(
    fields=["slice_index", "area_occupied", "perimeter", "total_area"],
    analysis_type="area_occupied"
)))
def measure_image_area_occupied(
    image: np.ndarray,
    *,
    operand_choices: Sequence[OperandChoice | str] = (OperandChoice.BINARY_IMAGE,),
    input_names: Sequence[str] = ("image",),
    retained_image_names: Sequence[str | None] = (None,),
    object_labels: Sequence[np.ndarray] = (),
    slice_by_slice: bool = True,
) -> tuple:
    """Measure area occupied for ordered binary-image and object rows."""
    rows = _area_occupied_runtime_rows(
        operand_choices,
        input_names,
        retained_image_names,
    )
    binary_images = _binary_images_from_payload(
        image,
        sum(row.operand is OperandChoice.BINARY_IMAGE for row in rows),
    )
    if len(object_labels) != sum(row.operand is OperandChoice.OBJECTS for row in rows):
        raise ValueError(
            "MeasureImageAreaOccupied object_labels count must match object rows."
        )

    retained_outputs = []
    measurements = []
    binary_index = 0
    object_index = 0
    for row_index, row in enumerate(rows):
        if row.operand is OperandChoice.BINARY_IMAGE:
            output_image, measurement = BinaryAreaOccupiedRequest(
                image=binary_images[binary_index],
                slice_index=row_index,
                source_image_name=row.input_name,
            ).measure()
            binary_index += 1
        else:
            labels = object_labels[object_index]
            output_image, measurement = ObjectLabelsAreaOccupiedRequest(
                image=_reference_image_for_labels(image, labels),
                labels=labels,
                slice_index=row_index,
                source_image_name=row.input_name,
            ).measure()
            object_index += 1
        measurements.append(measurement)
        if row.retained_image_name is not None:
            retained_outputs.append(output_image)

    if retained_outputs:
        return (*retained_outputs, measurements)
    return image, measurements


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("area_measurements", csv_materializer(
    fields=["slice_index", "area_occupied", "perimeter", "total_area"],
    analysis_type="area_occupied"
)))
def measure_image_area_occupied_binary(
    image: np.ndarray,
    source_image_name: str | None = None,
) -> Tuple[np.ndarray, AreaOccupiedMeasurement]:
    """
    Measure area occupied by foreground in a binary image.
    
    Args:
        image: Binary image (H, W) where foreground > 0
        
    Returns:
        Tuple of (original image, AreaOccupiedMeasurement)
    """
    return BinaryAreaOccupiedRequest(
        image=image,
        source_image_name=source_image_name,
    ).measure()


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(("area_measurements", csv_materializer(
    fields=["slice_index", "area_occupied", "perimeter", "total_area"],
    analysis_type="area_occupied"
)))
def measure_image_area_occupied_objects(
    image: np.ndarray,
    labels: np.ndarray,
    source_image_name: str | None = None,
) -> Tuple[np.ndarray, AreaOccupiedMeasurement]:
    """
    Measure area occupied by labeled objects.
    
    Args:
        image: Intensity image (H, W)
        labels: Label image from segmentation (H, W)
        
    Returns:
        Tuple of (original image, AreaOccupiedMeasurement)
    """
    return ObjectLabelsAreaOccupiedRequest(
        image=image,
        labels=labels,
        source_image_name=source_image_name,
    ).measure()


def _area_occupied_runtime_rows(
    operand_choices: Sequence[OperandChoice | str],
    input_names: Sequence[str],
    retained_image_names: Sequence[str | None],
) -> tuple[AreaOccupiedRuntimeRow, ...]:
    if len(operand_choices) != len(input_names) or len(input_names) != len(
        retained_image_names
    ):
        raise ValueError(
            "MeasureImageAreaOccupied row kwargs must have matching lengths."
        )
    return tuple(
        AreaOccupiedRuntimeRow.from_literals(
            operand=operand,
            input_name=input_name,
            retained_image_name=retained_image_name,
        )
        for operand, input_name, retained_image_name in zip(
            operand_choices,
            input_names,
            retained_image_names,
            strict=True,
        )
    )


def _binary_images_from_payload(
    image: np.ndarray,
    binary_image_count: int,
) -> tuple[np.ndarray, ...]:
    if binary_image_count == 0:
        return ()
    if binary_image_count == 1:
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] == 1:
            return (image[0],)
        return (image,)
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError(
            "MeasureImageAreaOccupied requires a stacked image payload for "
            "multiple binary-image rows."
        )
    if image.shape[0] != binary_image_count:
        raise ValueError(
            "MeasureImageAreaOccupied binary image stack length must match "
            "binary-image row count."
        )
    return tuple(image[index] for index in range(binary_image_count))


def label_area_and_perimeter(labels: np.ndarray) -> tuple[float, float]:
    label_array = object_label_dense_array(labels)
    if label_array.ndim <= 2:
        return _label_plane_area_and_perimeter(label_array)
    plane_measurements = tuple(
        _label_plane_area_and_perimeter(label_array[index])
        for index in range(label_array.shape[0])
    )
    return (
        float(sum(area for area, _perimeter in plane_measurements)),
        float(sum(perimeter for _area, perimeter in plane_measurements)),
    )


def _label_plane_area_and_perimeter(labels: np.ndarray) -> tuple[float, float]:
    labels_array = object_label_dense_array(labels, dtype=np.int32)
    return label_area_and_rounded_perimeter_2d(labels_array)


def _reference_image_for_labels(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_array = object_label_dense_array(labels)
    if image.ndim == label_array.ndim:
        return image
    if image.ndim == label_array.ndim + 1 and image.shape[0] >= 1:
        return image[0]
    return image


@dataclass
class VolumeOccupiedMeasurement:
    """Measurements for volume occupied analysis (3D)."""
    volume_occupied: float
    surface_area: float
    total_volume: float

    @classmethod
    def from_volume(
        cls,
        *,
        volume_occupied: float,
        surface_area: float,
        total_volume: float,
    ) -> "VolumeOccupiedMeasurement":
        return cls(
            volume_occupied=volume_occupied,
            surface_area=surface_area,
            total_volume=total_volume,
        )


@dataclass(frozen=True, slots=True)
class SurfaceAreaRequest:
    """Compute rounded surface area for one 3D label image."""

    label_image: np.ndarray
    spacing: Optional[Tuple[float, ...]] = None

    def surface_area(self) -> float:
        from skimage.measure import marching_cubes, mesh_surface_area

        spacing = self.spacing
        label_image = np.asarray(self.label_image)
        if spacing is None:
            spacing = (1.0,) * label_image.ndim

        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) == 0:
            return 0.0

        total_surface = 0.0
        for label in unique_labels:
            binary_mask = (label_image == label).astype(np.float32)
            try:
                verts, faces, _, _ = marching_cubes(
                    binary_mask,
                    spacing=spacing,
                    level=0.5,
                    method="lorensen",
                )
                total_surface += mesh_surface_area(verts, faces)
            except (ValueError, RuntimeError):
                continue
        return float(np.round(total_surface))


@dataclass(frozen=True, slots=True)
class VolumeOccupiedRequest:
    """Materialize a volume-occupied measurement from voxel totals."""

    volume_occupied: float
    surface_area: float
    total_volume: float

    def measurement(self) -> VolumeOccupiedMeasurement:
        return VolumeOccupiedMeasurement.from_volume(
            volume_occupied=self.volume_occupied,
            surface_area=self.surface_area,
            total_volume=self.total_volume,
        )


@numpy(contract=ProcessingContract.PURE_3D)
@special_outputs(("volume_measurements", csv_materializer(
    fields=["volume_occupied", "surface_area", "total_volume"],
    analysis_type="volume_occupied"
)))
def measure_image_volume_occupied_binary(
    image: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, VolumeOccupiedMeasurement]:
    """
    Measure volume occupied by foreground in a 3D binary image.
    
    Args:
        image: 3D binary image (D, H, W) where foreground > 0
        spacing: Voxel spacing (z, y, x) for surface area calculation
        
    Returns:
        Tuple of (original image, VolumeOccupiedMeasurement)
    """
    # Calculate volume occupied (number of foreground voxels)
    binary_mask = image > 0
    volume_occupied = float(np.sum(binary_mask))
    
    # Calculate surface area
    if volume_occupied > 0:
        surface_area_value = SurfaceAreaRequest(
            binary_mask.astype(np.int32),
            spacing=spacing,
        ).surface_area()
    else:
        surface_area_value = 0.0
    
    # Total volume is the total number of voxels
    total_volume = float(np.prod(image.shape))
    
    measurement = VolumeOccupiedRequest(
        volume_occupied=volume_occupied,
        surface_area=surface_area_value,
        total_volume=total_volume,
    ).measurement()
    
    return image, measurement


@numpy(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(("volume_measurements", csv_materializer(
    fields=["volume_occupied", "surface_area", "total_volume"],
    analysis_type="volume_occupied"
)))
def measure_image_volume_occupied_objects(
    image: np.ndarray,
    labels: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, VolumeOccupiedMeasurement]:
    """
    Measure volume occupied by labeled objects in 3D.
    
    Args:
        image: 3D intensity image (D, H, W)
        labels: 3D label image from segmentation (D, H, W)
        spacing: Voxel spacing (z, y, x) for surface area calculation
        
    Returns:
        Tuple of (original image, VolumeOccupiedMeasurement)
    """
    labels_array = object_label_dense_array(labels, dtype=np.int32)
    volume_occupied = float(np.count_nonzero(labels_array))
    
    # Calculate surface area
    if volume_occupied > 0:
        surface_area_value = SurfaceAreaRequest(
            labels_array,
            spacing=spacing,
        ).surface_area()
    else:
        surface_area_value = 0.0
    
    # Total volume is the total number of voxels
    total_volume = float(np.prod(labels_array.shape))
    
    measurement = VolumeOccupiedRequest(
        volume_occupied=volume_occupied,
        surface_area=surface_area_value,
        total_volume=total_volume,
    ).measurement()
    
    return image, measurement


@processing_prepare(measure_image_area_occupied)
def _prepare_measure_image_area_occupied() -> None:
    """Compile reusable area/perimeter kernels before timed execution."""
    binary = np.zeros((64, 64), dtype=np.float32)
    binary[8:40, 12:48] = 1.0
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    BinaryAreaOccupiedRequest(binary).measure()
    ObjectLabelsAreaOccupiedRequest(binary, labels).measure()
