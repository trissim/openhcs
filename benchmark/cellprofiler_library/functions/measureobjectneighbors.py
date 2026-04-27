"""
Converted from CellProfiler: MeasureObjectNeighbors
Original: MeasureObjectNeighbors.run

Measures neighbor relationships between objects including:
- Number of neighbors
- Percent of boundary touching neighbors
- First and second closest object distances
- Angle between neighbors
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple
from dataclasses import dataclass
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory import numpy


class DistanceMethod(Enum):
    ADJACENT = "adjacent"
    EXPAND = "expand"
    WITHIN = "within"


@dataclass
class NeighborMeasurements:
    """Per-object neighbor measurements."""
    slice_index: int
    object_id: int
    number_of_neighbors: int
    percent_touching: float
    first_closest_object_number: int
    first_closest_distance: float
    second_closest_object_number: int
    second_closest_distance: float
    angle_between_neighbors: float


@dataclass(frozen=True)
class NeighborDistancePlan:
    working_labels: np.ndarray
    distance: int


class NeighborDistancePlanner(ABC, metaclass=AutoRegisterMeta):
    """Prepare neighbor-distance state for one closed distance method."""

    __registry_key__ = "method"
    method: ClassVar[DistanceMethod | None] = None

    @classmethod
    def for_method(cls, method: DistanceMethod) -> "NeighborDistancePlanner":
        return cls.__registry__[method]()

    @abstractmethod
    def plan(
        self,
        labels: np.ndarray,
        neighbor_distance: int,
    ) -> NeighborDistancePlan:
        """Return working labels and neighborhood distance."""


class AdjacentNeighborDistancePlanner(NeighborDistancePlanner):
    method = DistanceMethod.ADJACENT

    def plan(
        self,
        labels: np.ndarray,
        neighbor_distance: int,
    ) -> NeighborDistancePlan:
        return NeighborDistancePlan(labels.copy(), 1)


class ExpandedNeighborDistancePlanner(NeighborDistancePlanner):
    method = DistanceMethod.EXPAND

    def plan(
        self,
        labels: np.ndarray,
        neighbor_distance: int,
    ) -> NeighborDistancePlan:
        from scipy.ndimage import distance_transform_edt

        i, j = distance_transform_edt(
            labels == 0,
            return_distances=False,
            return_indices=True,
        )
        return NeighborDistancePlan(labels[i, j], 1)


class WithinNeighborDistancePlanner(NeighborDistancePlanner):
    method = DistanceMethod.WITHIN

    def plan(
        self,
        labels: np.ndarray,
        neighbor_distance: int,
    ) -> NeighborDistancePlan:
        return NeighborDistancePlan(labels.copy(), neighbor_distance)


def _strel_disk(radius: int) -> np.ndarray:
    """Create a disk-shaped structuring element."""
    from skimage.morphology import disk
    return disk(radius)


def _centers_of_labels(labels: np.ndarray) -> np.ndarray:
    """Calculate centers of mass for each labeled object."""
    from scipy.ndimage import center_of_mass
    num_labels = labels.max()
    if num_labels == 0:
        return np.zeros((0, 2))
    centers = center_of_mass(np.ones_like(labels), labels, range(1, num_labels + 1))
    return np.array(centers)


def _outline(labels: np.ndarray) -> np.ndarray:
    """Create outline of labeled objects."""
    from scipy.ndimage import binary_erosion
    outline = np.zeros_like(labels)
    for i in range(1, labels.max() + 1):
        mask = labels == i
        eroded = binary_erosion(mask)
        outline[mask & ~eroded] = i
    return outline


@numpy
def measure_object_neighbors(
    image: np.ndarray,
    labels: np.ndarray,
    distance_method: DistanceMethod = DistanceMethod.EXPAND,
    neighbor_distance: int = 5,
    neighbors_are_same_objects: bool = True,
) -> Tuple[np.ndarray, list]:
    """
    Measure neighbor relationships between objects.
    
    Args:
        image: Input image (H, W)
        labels: Label image with segmented objects (H, W)
        distance_method: Method to determine neighbors:
            - ADJACENT: Objects must have adjacent boundary pixels
            - EXPAND: Expand objects until all boundaries touch
            - WITHIN: Expand by specified distance
        neighbor_distance: Distance for WITHIN method
        neighbors_are_same_objects: If True, measure neighbors within same object set
    
    Returns:
        Tuple of (image, list of NeighborMeasurements)
    """
    from scipy.ndimage import binary_dilation
    from scipy.signal import fftconvolve
    
    labels = labels.astype(np.int32)
    nobjects = labels.max()
    
    if nobjects == 0:
        return image, []
    
    # Initialize measurement arrays
    neighbor_count = np.zeros(nobjects)
    pixel_count = np.zeros(nobjects)
    first_object_number = np.zeros(nobjects, dtype=int)
    second_object_number = np.zeros(nobjects, dtype=int)
    first_x_vector = np.zeros(nobjects)
    second_x_vector = np.zeros(nobjects)
    first_y_vector = np.zeros(nobjects)
    second_y_vector = np.zeros(nobjects)
    angle = np.zeros(nobjects)
    percent_touching = np.zeros(nobjects)
    
    distance_plan = NeighborDistancePlanner.for_method(distance_method).plan(
        labels,
        neighbor_distance,
    )
    working_labels = distance_plan.working_labels
    distance = distance_plan.distance
    
    neighbor_labels = working_labels.copy()
    
    if nobjects > (1 if neighbors_are_same_objects else 0):
        # Calculate object centers
        ocenters = _centers_of_labels(labels)
        ncenters = ocenters.copy()
        
        # Calculate perimeters
        object_indexes = np.arange(nobjects) + 1
        perimeter_outlines = _outline(labels)
        perimeters = np.array([np.sum(perimeter_outlines == i) for i in object_indexes])
        perimeters = np.maximum(perimeters, 1)  # Avoid division by zero
        
        # Find nearest neighbors using center distances
        if nobjects >= 2:
            for i in range(nobjects):
                distances = np.sqrt(
                    (ocenters[i, 0] - ncenters[:, 0])**2 + 
                    (ocenters[i, 1] - ncenters[:, 1])**2
                )
                if neighbors_are_same_objects:
                    distances[i] = np.inf  # Exclude self
                
                sorted_idx = np.argsort(distances)
                first_neighbor_idx = 0 if not neighbors_are_same_objects else 0
                
                if len(sorted_idx) > first_neighbor_idx:
                    first_idx = sorted_idx[first_neighbor_idx]
                    first_object_number[i] = first_idx + 1
                    first_x_vector[i] = ncenters[first_idx, 1] - ocenters[i, 1]
                    first_y_vector[i] = ncenters[first_idx, 0] - ocenters[i, 0]
                
                if len(sorted_idx) > first_neighbor_idx + 1:
                    second_idx = sorted_idx[first_neighbor_idx + 1]
                    second_object_number[i] = second_idx + 1
                    second_x_vector[i] = ncenters[second_idx, 1] - ocenters[i, 1]
                    second_y_vector[i] = ncenters[second_idx, 0] - ocenters[i, 0]
        
        # Calculate angles between neighbors
        for i in range(nobjects):
            v1 = np.array([first_x_vector[i], first_y_vector[i]])
            v2 = np.array([second_x_vector[i], second_y_vector[i]])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                dot = np.dot(v1, v2) / (norm1 * norm2)
                dot = np.clip(dot, -1, 1)
                angle[i] = np.arccos(dot) * 180.0 / np.pi
        
        # Create structuring elements
        strel = _strel_disk(distance)
        strel_touching = _strel_disk(distance + 1)
        
        # Calculate neighbor counts and touching percentages
        for obj_idx in range(nobjects):
            obj_num = obj_idx + 1
            
            # Get bounding box with padding
            obj_mask = labels == obj_num
            if not np.any(obj_mask):
                continue
                
            rows, cols = np.where(obj_mask)
            min_i = max(0, rows.min() - distance)
            max_i = min(labels.shape[0], rows.max() + distance + 1)
            min_j = max(0, cols.min() - distance)
            max_j = min(labels.shape[1], cols.max() + distance + 1)
            
            patch = working_labels[min_i:max_i, min_j:max_j]
            npatch = neighbor_labels[min_i:max_i, min_j:max_j]
            
            # Find neighbors by dilation
            patch_mask = patch == obj_num
            if distance <= 5:
                extended = binary_dilation(patch_mask, strel)
            else:
                extended = fftconvolve(patch_mask.astype(float), strel.astype(float), mode='same') > 0.5
            
            neighbors = np.unique(npatch[extended])
            neighbors = neighbors[neighbors != 0]
            if neighbors_are_same_objects:
                neighbors = neighbors[neighbors != obj_num]
            
            neighbor_count[obj_idx] = len(neighbors)
            
            # Calculate percent touching
            outline_patch = perimeter_outlines[min_i:max_i, min_j:max_j] == obj_num
            
            if neighbors_are_same_objects:
                extendme = (patch != 0) & (patch != obj_num)
            else:
                extendme = npatch != 0
            
            if distance <= 5:
                extended_touch = binary_dilation(extendme, strel_touching)
            else:
                extended_touch = fftconvolve(extendme.astype(float), strel_touching.astype(float), mode='same') > 0.5
            
            overlap = np.sum(outline_patch & extended_touch)
            pixel_count[obj_idx] = overlap
        
        # Calculate percent touching
        percent_touching = pixel_count * 100 / perimeters
    
    # Build measurement results
    measurements = []
    for i in range(nobjects):
        first_dist = np.sqrt(first_x_vector[i]**2 + first_y_vector[i]**2)
        second_dist = np.sqrt(second_x_vector[i]**2 + second_y_vector[i]**2)
        
        measurements.append(NeighborMeasurements(
            slice_index=0,
            object_id=i + 1,
            number_of_neighbors=int(neighbor_count[i]),
            percent_touching=float(percent_touching[i]),
            first_closest_object_number=int(first_object_number[i]),
            first_closest_distance=float(first_dist),
            second_closest_object_number=int(second_object_number[i]),
            second_closest_distance=float(second_dist),
            angle_between_neighbors=float(angle[i])
        ))
    
    return image, measurements
