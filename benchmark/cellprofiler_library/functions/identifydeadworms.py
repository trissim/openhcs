"""Converted from CellProfiler: IdentifyDeadWorms

Identifies dead worms by their straight shape using diamond-shaped template
matching at multiple angles. Dead C. elegans worms typically have a straight
shape whereas live worms assume a sinusoidal shape.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.analysis.cell_counting_cpu import materialize_segmentation_masks


@dataclass
class DeadWormStats:
    slice_index: int
    object_count: int
    mean_center_x: float
    mean_center_y: float
    mean_angle: float


def _get_line_pts(y0, x0, y1, x1):
    """Get points along lines between start and end coordinates.
    
    Simple Bresenham-style line drawing for multiple line segments.
    """
    n_lines = len(y0)
    all_i = []
    all_j = []
    
    for idx in range(n_lines):
        # Bresenham's line algorithm
        dy = abs(y1[idx] - y0[idx])
        dx = abs(x1[idx] - x0[idx])
        sy = 1 if y0[idx] < y1[idx] else -1
        sx = 1 if x0[idx] < x1[idx] else -1
        err = dx - dy
        
        cy, cx = y0[idx], x0[idx]
        while True:
            all_i.append(cy)
            all_j.append(cx)
            if cy == y1[idx] and cx == x1[idx]:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy
    
    return np.array(all_i), np.array(all_j)


def _get_diamond(worm_width: int, worm_length: int, angle: float) -> np.ndarray:
    """Get a diamond-shaped structuring element at given angle.
    
    Args:
        worm_width: Width of the diamond (short axis)
        worm_length: Length of the diamond (long axis)
        angle: Rotation angle in radians
    
    Returns:
        Binary array for use as erosion footprint
    """
    from scipy.ndimage import binary_fill_holes
    
    # Diamond vertices
    x0 = int(np.sin(angle) * worm_length / 2)
    x1 = int(np.cos(angle) * worm_width / 2)
    x2 = -x0
    x3 = -x1
    y2 = int(np.cos(angle) * worm_length / 2)
    y1 = int(np.sin(angle) * worm_width / 2)
    y0 = -y2
    y3 = -y1
    
    xmax = np.max(np.abs([x0, x1, x2, x3]))
    ymax = np.max(np.abs([y0, y1, y2, y3]))
    
    strel = np.zeros((ymax * 2 + 1, xmax * 2 + 1), bool)
    
    # Draw diamond outline
    pts_y0 = np.array([y0, y1, y2, y3]) + ymax
    pts_x0 = np.array([x0, x1, x2, x3]) + xmax
    pts_y1 = np.array([y1, y2, y3, y0]) + ymax
    pts_x1 = np.array([x1, x2, x3, x0]) + xmax
    
    i_pts, j_pts = _get_line_pts(pts_y0, pts_x0, pts_y1, pts_x1)
    
    # Clip to valid indices
    valid = (i_pts >= 0) & (i_pts < strel.shape[0]) & (j_pts >= 0) & (j_pts < strel.shape[1])
    strel[i_pts[valid], j_pts[valid]] = True
    strel = binary_fill_holes(strel)
    
    return strel


def _all_connected_components(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Find connected components from edge list.
    
    Args:
        first: First vertex of each edge
        second: Second vertex of each edge
    
    Returns:
        Label array where each unique value represents a connected component
    """
    if len(first) == 0:
        return np.zeros(0, dtype=int)
    
    n_vertices = max(np.max(first), np.max(second)) + 1
    labels = np.arange(n_vertices)
    
    # Union-find with path compression
    def find(x):
        root = x
        while labels[root] != root:
            root = labels[root]
        # Path compression
        while labels[x] != root:
            next_x = labels[x]
            labels[x] = root
            x = next_x
        return root
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            labels[rx] = ry
    
    for f, s in zip(first, second):
        union(f, s)
    
    # Compress labels
    for i in range(n_vertices):
        labels[i] = find(i)
    
    # Renumber to consecutive integers
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[l] for l in labels])


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    ("dead_worm_stats", csv_materializer(
        fields=["slice_index", "object_count", "mean_center_x", "mean_center_y", "mean_angle"],
        analysis_type="dead_worm_identification"
    )),
    ("labels", materialize_segmentation_masks)
)
def identify_dead_worms(
    image: np.ndarray,
    worm_width: int = 10,
    worm_length: int = 100,
    angle_count: int = 32,
    auto_distance: bool = True,
    space_distance: float = 5.0,
    angular_distance: float = 30.0,
) -> Tuple[np.ndarray, DeadWormStats, np.ndarray]:
    """Identify dead worms by fitting straight diamond shapes at multiple angles.
    
    Dead C. elegans worms typically have a straight shape whereas live worms
    assume a sinusoidal shape. This function identifies dead worms by fitting
    a diamond-shaped template at many angles.
    
    Args:
        image: Binary input image (H, W) with worms as foreground
        worm_width: Width of diamond template in pixels (short axis)
        worm_length: Length of diamond template in pixels (long axis)
        angle_count: Number of angles to test (0 to 180 degrees)
        auto_distance: Whether to auto-calculate distance parameters
        space_distance: Spatial distance threshold for grouping centers
        angular_distance: Angular distance threshold in degrees
    
    Returns:
        Tuple of (original image, statistics, label image)
    """
    from scipy.ndimage import binary_erosion, mean as mean_of_labels
    
    # Ensure binary
    mask = image > 0
    
    # Collect erosion points at all angles
    i_coords = []
    j_coords = []
    a_coords = []
    
    ig, jg = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    
    for angle_idx in range(angle_count):
        angle = float(angle_idx) * np.pi / float(angle_count)
        strel = _get_diamond(worm_width, worm_length, angle)
        erosion = binary_erosion(mask, strel)
        
        this_count = np.sum(erosion)
        if this_count > 0:
            i_coords.append(ig[erosion])
            j_coords.append(jg[erosion])
            a_coords.append(np.ones(this_count) * angle)
    
    if len(i_coords) == 0:
        # No worms found
        labels = np.zeros(mask.shape, dtype=np.int32)
        stats = DeadWormStats(
            slice_index=0,
            object_count=0,
            mean_center_x=0.0,
            mean_center_y=0.0,
            mean_angle=0.0
        )
        return image, stats, labels
    
    i = np.concatenate(i_coords)
    j = np.concatenate(j_coords)
    a = np.concatenate(a_coords)
    
    # Calculate distance parameters
    if auto_distance:
        space_dist = float(worm_width)
        angle_dist = np.arctan2(worm_width, worm_length) + np.pi / angle_count
    else:
        space_dist = space_distance
        angle_dist = angular_distance * np.pi / 180.0
    
    # Find adjacent points by distance
    if len(i) < 2:
        first = np.zeros(0, dtype=int)
        second = np.zeros(0, dtype=int)
    else:
        # Sort by i coordinate
        order = np.lexsort((a, j, i))
        i_sorted = i[order]
        j_sorted = j[order]
        a_sorted = a[order]
        
        # Find pairs within distance threshold
        first_list = []
        second_list = []
        
        # Simple O(n^2) approach for correctness - can be optimized
        for idx1 in range(len(i)):
            for idx2 in range(idx1 + 1, len(i)):
                spatial_dist_sq = (i_sorted[idx1] - i_sorted[idx2])**2 + (j_sorted[idx1] - j_sorted[idx2])**2
                if spatial_dist_sq <= space_dist**2:
                    angle_diff = abs(a_sorted[idx1] - a_sorted[idx2])
                    # Handle wrap-around
                    if angle_diff <= angle_dist or (np.pi - angle_diff) <= angle_dist:
                        first_list.append(order[idx1])
                        second_list.append(order[idx2])
        
        first = np.array(first_list, dtype=int)
        second = np.array(second_list, dtype=int)
    
    # Connected components
    if len(first) > 0:
        ij_labels = _all_connected_components(first, second) + 1
        nlabels = np.max(ij_labels)
        label_indexes = np.arange(1, nlabels + 1)
        
        # Compute measurements
        center_x = np.array([np.mean(j[ij_labels == lbl]) for lbl in label_indexes])
        center_y = np.array([np.mean(i[ij_labels == lbl]) for lbl in label_indexes])
        angles = np.array([np.mean(a[ij_labels == lbl]) for lbl in label_indexes])
        
        # Create 2D label image
        labels = np.zeros(mask.shape, dtype=np.int32)
        labels[i, j] = ij_labels
    else:
        # Each point is its own object
        nlabels = len(i)
        labels = np.zeros(mask.shape, dtype=np.int32)
        if nlabels > 0:
            labels[i, j] = np.arange(1, nlabels + 1)
            center_x = j.astype(float)
            center_y = i.astype(float)
            angles = a
        else:
            center_x = np.array([])
            center_y = np.array([])
            angles = np.array([])
    
    # Create statistics
    stats = DeadWormStats(
        slice_index=0,
        object_count=int(nlabels),
        mean_center_x=float(np.mean(center_x)) if len(center_x) > 0 else 0.0,
        mean_center_y=float(np.mean(center_y)) if len(center_y) > 0 else 0.0,
        mean_angle=float(np.mean(angles) * 180 / np.pi) if len(angles) > 0 else 0.0
    )
    
    return image, stats, labels