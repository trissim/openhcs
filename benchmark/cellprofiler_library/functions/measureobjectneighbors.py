"""
Converted from CellProfiler: MeasureObjectNeighbors.

Compatibility facade for the OpenHCS CellProfiler backend implementation.
"""

from openhcs.processing.backends.cellprofiler.neighbors import (
    AdjacentNeighborDistancePlanner,
    DistanceMethod,
    ExpandedNeighborDistancePlanner,
    NeighborClosestArrays,
    NeighborDistancePlan,
    NeighborDistancePlanner,
    NeighborMeasurements,
    NeighborRetainedImageRequest,
    NeighborTopologyArrays,
    NeighborTopologyBackendStrategy,
    NumbaNumpyNeighborTopologyBackendStrategy,
    WithinNeighborDistancePlanner,
    labels_or_default,
    measure_object_neighbors,
    neighbor_topology_backend,
    require_matching_shape,
    variant_numbers_for_final_labels,
)

__all__ = [
    "AdjacentNeighborDistancePlanner",
    "DistanceMethod",
    "ExpandedNeighborDistancePlanner",
    "NeighborClosestArrays",
    "NeighborDistancePlan",
    "NeighborDistancePlanner",
    "NeighborMeasurements",
    "NeighborRetainedImageRequest",
    "NeighborTopologyArrays",
    "NeighborTopologyBackendStrategy",
    "NumbaNumpyNeighborTopologyBackendStrategy",
    "WithinNeighborDistancePlanner",
    "labels_or_default",
    "measure_object_neighbors",
    "neighbor_topology_backend",
    "require_matching_shape",
    "variant_numbers_for_final_labels",
]
