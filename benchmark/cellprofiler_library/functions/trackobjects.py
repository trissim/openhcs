"""
Converted from CellProfiler: TrackObjects.

Compatibility facade for the OpenHCS CellProfiler backend implementation.
"""

from openhcs.processing.backends.cellprofiler.tracking import (
    DistanceTrackObjectsMethodStrategy,
    MovementModel,
    ObjectTrackingData,
    OverlapTrackObjectsMethodStrategy,
    TrackObjectsMethodStrategy,
    TrackingMethod,
    TrackingResult,
    track_objects,
)

__all__ = [
    "DistanceTrackObjectsMethodStrategy",
    "MovementModel",
    "ObjectTrackingData",
    "OverlapTrackObjectsMethodStrategy",
    "TrackObjectsMethodStrategy",
    "TrackingMethod",
    "TrackingResult",
    "track_objects",
]
