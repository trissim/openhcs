"""Compatibility aliases for CellProfiler worm geometry backend semantics."""

from openhcs.processing.backends.cellprofiler.worm_geometry import (
    CellProfilerLookupPattern,
    CellProfilerLookupTableProjection,
    branchpoints,
    calculate_cumulative_lengths,
    control_points_for_label_image,
    eight_connectivity,
    endpoints,
    rebuild_worm_from_control_points_approx,
    sample_control_points,
    skeletonize_worm_mask,
    trace_skeleton_path,
)

__all__ = [
    "CellProfilerLookupPattern",
    "CellProfilerLookupTableProjection",
    "branchpoints",
    "calculate_cumulative_lengths",
    "control_points_for_label_image",
    "eight_connectivity",
    "endpoints",
    "rebuild_worm_from_control_points_approx",
    "sample_control_points",
    "skeletonize_worm_mask",
    "trace_skeleton_path",
]
