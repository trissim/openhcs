"""CellProfiler relationship measurement payload semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RelationshipMeasurements:
    """Measurements emitted by CellProfiler RelateObjects."""

    slice_index: int
    parent_object_count: int
    child_object_count: int
    children_with_parents_count: int
    mean_children_per_parent: float
    mean_centroid_distance: float
    mean_minimum_distance: float

    @property
    def declares_distance_measurements(self) -> bool:
        """Return whether this payload contains distance features."""
        return bool(
            np.isfinite(self.mean_centroid_distance)
            or np.isfinite(self.mean_minimum_distance)
        )
