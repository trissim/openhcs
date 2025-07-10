"""
CPU implementation of Random-Reaction-Seed (RRS) neurite tracing algorithm.

This module provides a CPU-based implementation that maintains exact algorithmic
fidelity to the original paper while being compatible with OpenHCS framework.

Based on: "Random-Reaction-Seed Method for Automated Identification of Neurite
Elongation and Branching" by Alvason L., Lawrance C., and Jia Z. (2019)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from skimage.feature import canny, blob_dog
from skimage.filters import threshold_li, median
from skimage.morphology import remove_small_objects, skeletonize, disk, dilation
from skimage import exposure
import queue

# Import OpenHCS decorators and utilities
from openhcs.core.memory.decorators import numpy as numpy_func
from openhcs.core.pipeline.function_contracts import special_outputs
import json
import pandas as pd

@numpy_func
@special_outputs("trace_results")
def trace_neurites_rrs_exact_author_implementation_cpu(
    image: np.ndarray,  # First argument must be 3D array
    # Core algorithm parameters (from author's implementation)
    total_node: int = 16,  # HMM chain length
    total_path: int = 8,   # candidate directions per node
    node_r: int = 5,       # radial distance between nodes (pixels)
    node_angle_max: float = 90.0,  # max search angle (degrees)
    # Validation parameters
    chain_level: float = 1.05,  # validation threshold multiplier
    prob_multiplier: float = 255.0,  # for log calculation scaling
    min_high_nodes: int = 3,  # minimum valid chain length
    # Boundary and filtering parameters
    boundary: int = 4,  # edge avoidance pixels
    line_length_min: int = 16,  # minimum branch length
    free_zone_from_y0: int = 4,  # root detection zone
    # Seed parameters
    seed_angle_max: float = 360.0,  # seed angular search range (degrees)
    seed_density: float = 0.01,  # density of initial seed points
    # Processing parameters
    enable_preprocessing: bool = True,  # enable author's preprocessing pipeline
    preprocessing_method: str = "canny",  # "canny", "threshold", or "blob"
    # Output parameters
    overlay_traces_on_image: bool = True,  # overlay binary trace mask on original image
    # Blob detection parameters (for preprocessing_method="blob")
    min_sigma: float = 1.0,
    max_sigma: float = 2.0,
    blob_threshold: float = 0.02
) -> Tuple[np.ndarray, Dict[str, List[Tuple[float, ...]]]]:
    """
    CPU implementation of exact author's Random-Reaction-Seed (RRS) neurite tracing.

    Maintains mathematical fidelity to the original paper while providing CPU-based
    processing for systems without GPU acceleration.

    This implementation follows the exact algorithm from:
    "Random-Reaction-Seed Method for Automated Identification of Neurite Elongation and Branching"
    by Alvason L., Lawrance C., and Jia Z. (2019)

    Args:
        image: Input 3D image array (Z, Y, X) or (1, Y, X) for 2D images
        total_node: HMM chain length (author's default: 16)
        total_path: Candidate directions per node (author's default: 8)
        node_r: Radial distance between nodes in pixels (author's default: 5)
        node_angle_max: Maximum angular search range in degrees (author's default: 90)
        chain_level: Validation threshold multiplier (author's default: 1.05)
        prob_multiplier: Scaling factor for log calculations (author's default: 255)
        min_high_nodes: Minimum nodes for valid chain (author's default: 3)
        boundary: Edge avoidance pixels (author's default: 4)
        line_length_min: Minimum branch length (author's default: 16)
        free_zone_from_y0: Root detection zone (author's default: 4)
        seed_angle_max: Seed angular search range in degrees (author's default: 360)
        seed_density: Density of initial seed points (fraction of pixels)
        enable_preprocessing: Enable author's edge/skeleton/blob preprocessing
        preprocessing_method: Type of preprocessing ("canny", "threshold", "blob")
        overlay_traces_on_image: If True, overlay binary trace mask on original image; if False, return original image unchanged
        min_sigma: Blob detection minimum sigma
        max_sigma: Blob detection maximum sigma
        blob_threshold: Blob detection threshold

    Returns:
        output_stack : np.ndarray
            3D array - if overlay_traces_on_image=True: original image with traced neurites overlaid,
            if overlay_traces_on_image=False: original image unchanged
        trace_results : Dict[str, List[Tuple[float, ...]]]
            Dictionary of traces where keys are trace IDs and values are coordinate lists
    """
    # Validate input is 3D array
    if image.ndim != 3:
        raise ValueError(f"Input image must be 3D array (Z, Y, X), got {image.ndim}D")

    # Process each slice in the 3D stack
    output_stack = np.zeros_like(image, dtype=np.float64)
    all_traces = {}  # Collect all traces across slices

    for z_idx in range(image.shape[0]):
        slice_2d = image[z_idx].astype(np.float64)

        # Convert angle parameters from degrees to radians
        node_angle_max_rad = np.radians(node_angle_max)
        seed_angle_max_rad = np.radians(seed_angle_max)

        # Compute derived parameters following author's formulas
        total_path_seed = int(1 + 8 + 8 * np.floor(total_path / 8.0))

        # Validation cut levels (author's exact formula)
        cut_level_first_node = 4 * chain_level
        cut_level_other_nodes = chain_level

        # Apply preprocessing if enabled
        if enable_preprocessing:
            edge_map = _apply_preprocessing(slice_2d, preprocessing_method, boundary,
                                          min_sigma, max_sigma, blob_threshold)
            seed_xx, seed_yy = _generate_seeds_from_map(edge_map, seed_density)
        else:
            seed_xx, seed_yy = _generate_random_seeds(slice_2d, seed_density)

        if len(seed_xx) == 0:
            # No seeds found, keep original slice
            output_stack[z_idx] = slice_2d
            continue

        # Run RRS algorithm on this slice
        traces = _run_rrs_algorithm_cpu(
            slice_2d, seed_xx, seed_yy,
            total_node, total_path, total_path_seed, node_r,
            node_angle_max_rad, seed_angle_max_rad,
            chain_level, cut_level_first_node, cut_level_other_nodes,
            prob_multiplier, min_high_nodes, line_length_min, free_zone_from_y0
        )

        # Apply output transformation based on flag
        if overlay_traces_on_image:
            # Convert traces to binary mask and overlay on original slice
            trace_mask = _traces_to_binary_mask(traces, slice_2d.shape)
            output_stack[z_idx] = slice_2d + trace_mask  # Overlay traces on original
        else:
            # Return original image unchanged
            output_stack[z_idx] = slice_2d

        # Collect traces with z-coordinate information
        for trace_id, coordinates in traces.items():
            # Add z-coordinate to each point and create unique trace ID
            z_trace_id = f"z{z_idx}_{trace_id}"
            z_coordinates = [(coord[0], coord[1], float(z_idx)) for coord in coordinates]
            all_traces[z_trace_id] = z_coordinates

    return output_stack, all_traces


def materialize_rrs_cpu_trace_results(data: Dict[str, List[Tuple[float, ...]]], path: str, filemanager) -> str:
    """Materialize RRS CPU trace results as JSON with analysis metadata and CSV coordinates."""
    # JSON for visualization tools
    json_path = path.replace('.pkl', '_rrs_traces.json')

    trace_summary = {
        "analysis_type": "rrs_neurite_tracing_cpu",
        "algorithm": "exact_author_implementation",
        "total_traces": len(data),
        "traces": data,
        "summary_statistics": {
            "total_points": sum(len(coords) for coords in data.values()),
            "avg_trace_length": sum(len(coords) for coords in data.values()) / len(data) if data else 0,
            "longest_trace": max(len(coords) for coords in data.values()) if data else 0,
            "shortest_trace": min(len(coords) for coords in data.values()) if data else 0
        }
    }

    json_content = json.dumps(trace_summary, indent=2, default=str)
    filemanager.save(json_content, json_path, "disk")

    # CSV for statistical analysis
    csv_path = path.replace('.pkl', '_rrs_coordinates.csv')

    rows = []
    for trace_id, coordinates in data.items():
        for i, coord in enumerate(coordinates):
            rows.append({
                'trace_id': trace_id,
                'point_index': i,
                'x_coordinate': coord[0],
                'y_coordinate': coord[1],
                'z_coordinate': coord[2] if len(coord) > 2 else 0,
                'trace_length': len(coordinates)
            })

    if rows:
        df = pd.DataFrame(rows)
        csv_content = df.to_csv(index=False)
        filemanager.save(csv_content, csv_path, "disk")

    return json_path


def _traces_to_binary_mask(traces: Dict[str, List[Tuple[float, ...]]], shape: Tuple[int, int]) -> np.ndarray:
    """Convert trace coordinates to binary mask."""
    mask = np.zeros(shape, dtype=np.float64)

    for trace_id, coordinates in traces.items():
        for coord in coordinates:
            x, y = int(coord[0]), int(coord[1])
            # Ensure coordinates are within bounds
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                mask[y, x] = 1.0

    return mask


def _apply_preprocessing(
    image: np.ndarray,
    method: str,
    boundary: int,
    min_sigma: float = 1.0,
    max_sigma: float = 2.0,
    blob_threshold: float = 0.02
) -> np.ndarray:
    """Apply author's preprocessing methods."""

    if method == "canny":
        # Author's Canny edge detection
        edge_map = canny(image)
        edge_map = edge_map.astype(np.int64)

    elif method == "threshold":
        # Author's threshold + skeletonization
        thresh_val = threshold_li(image)
        bool_image = image > thresh_val
        bool_image = remove_small_objects(bool_image, min_size=2)
        edge_map = skeletonize(bool_image).astype(np.int64)

    elif method == "blob":
        # Author's blob detection
        image_median = median(image)
        blobs = blob_dog(image_median, min_sigma=min_sigma,
                        max_sigma=max_sigma, threshold=blob_threshold)
        edge_map = np.zeros_like(image, dtype=np.int64)
        if len(blobs) > 0:
            yy = np.int64(blobs[:, 0])
            xx = np.int64(blobs[:, 1])
            # Ensure coordinates are within bounds
            valid_mask = (yy >= 0) & (yy < image.shape[0]) & (xx >= 0) & (xx < image.shape[1])
            edge_map[yy[valid_mask], xx[valid_mask]] = 1
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    # Apply boundary masking (author's method)
    edge_map[:boundary, :] = 0
    edge_map[-boundary:, :] = 0
    edge_map[:, :boundary] = 0
    edge_map[:, -boundary:] = 0

    return edge_map


def _generate_seeds_from_map(edge_map: np.ndarray, seed_density: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate seeds from edge map following author's method."""
    yy, xx = edge_map.nonzero()

    if len(xx) == 0:
        return np.array([]), np.array([])

    # Author's random sampling from edge pixels
    num_seeds = max(1, int(len(xx) * seed_density))
    seed_indices = np.random.choice(len(xx), size=min(num_seeds, len(xx)), replace=False)

    seed_xx = xx[seed_indices]
    seed_yy = yy[seed_indices]

    return seed_xx, seed_yy


def _generate_random_seeds(image: np.ndarray, seed_density: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random seeds across the image."""
    height, width = image.shape
    num_seeds = int(seed_density * height * width)

    if num_seeds == 0:
        return np.array([]), np.array([])

    seed_xx = np.random.randint(0, width, size=num_seeds)
    seed_yy = np.random.randint(0, height, size=num_seeds)

    return seed_xx, seed_yy


def _run_rrs_algorithm_cpu(
    image: np.ndarray,
    seed_xx: np.ndarray,
    seed_yy: np.ndarray,
    total_node: int,
    total_path: int,
    total_path_seed: int,
    node_r: int,
    node_angle_max_rad: float,
    seed_angle_max_rad: float,
    chain_level: float,
    cut_level_first_node: float,
    cut_level_other_nodes: float,
    prob_multiplier: float,
    min_high_nodes: int,
    line_length_min: int,
    free_zone_from_y0: int
) -> Dict[str, List[Tuple[float, ...]]]:
    """
    Run the complete RRS algorithm following author's exact implementation.

    This is the main algorithm that processes all seeds and returns traces.
    """
    if len(seed_xx) == 0:
        return {}

    # Create AlvaHMM instance (inlined)
    alva_hmm = _AlvaHMMCPU(
        image, total_node, total_path, node_r, node_angle_max_rad, total_path_seed
    )

    # Run pair HMM chain algorithm
    chain_hmm_1st, pair_chain_hmm, pair_seed_xx, pair_seed_yy = alva_hmm.pair_HMM_chain(
        seed_xx, seed_yy,
        seed_aa=None,  # Will be set to zeros
        seed_angle_max=seed_angle_max_rad,
        chain_level=chain_level,
        cut_level_first_node=cut_level_first_node,
        cut_level_other_nodes=cut_level_other_nodes,
        prob_multiplier=prob_multiplier,
        min_high_nodes=min_high_nodes
    )

    # Convert results to OpenHCS format
    traces = _convert_chains_to_traces(chain_hmm_1st, pair_chain_hmm)

    return traces


def _convert_chains_to_traces(chain_hmm_1st, pair_chain_hmm) -> Dict[str, List[Tuple[float, ...]]]:
    """Convert author's chain format to OpenHCS trace format."""
    traces = {}
    trace_id = 0

    # Process first chains
    if chain_hmm_1st is not None:
        real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_hmm_1st[0:4]

        for i in range(len(real_chain_ii)):
            if len(real_chain_ii[i]) >= 3:  # Minimum chain length
                chain_xx = real_chain_xx[i][real_chain_ii[i]]
                chain_yy = real_chain_yy[i][real_chain_ii[i]]

                # Convert to coordinate tuples
                coordinates = [(float(x), float(y)) for x, y in zip(chain_xx, chain_yy)]
                traces[f"trace_{trace_id}"] = coordinates
                trace_id += 1

    # Process pair chains (secondary/reaction chains)
    if pair_chain_hmm is not None:
        real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = pair_chain_hmm[0:4]

        for i in range(len(real_chain_ii)):
            if len(real_chain_ii[i]) >= 3:  # Minimum chain length
                chain_xx = real_chain_xx[i][real_chain_ii[i]]
                chain_yy = real_chain_yy[i][real_chain_ii[i]]

                # Convert to coordinate tuples
                coordinates = [(float(x), float(y)) for x, y in zip(chain_xx, chain_yy)]
                traces[f"trace_{trace_id}"] = coordinates
                trace_id += 1

    return traces


class _AlvaHMMCPU:
    """
    CPU implementation of author's AlvaHMM class.

    This class contains the exact algorithm from the original paper,
    inlined to remove external dependencies while maintaining algorithmic fidelity.
    """

    def __init__(self, likelihood_mmm, total_node=None, total_path=None,
                 node_r=None, node_angle_max=None, total_path_seed=None):
        """Initialize AlvaHMM with author's exact parameters."""

        # Normalize image following author's method
        likelihood_mmm = likelihood_mmm - likelihood_mmm.min()
        likelihood_mmm = likelihood_mmm / likelihood_mmm.max()

        # Set default parameters (author's values)
        if total_node is None:
            total_node = 16
        if total_path is None:
            total_path = 8
        if node_r is None:
            node_r = 5
        if node_angle_max is None:
            node_angle_max = 90 * (np.pi / 180)
        if total_path_seed is None:
            total_path_seed = int(1 + 8 + 8 * np.floor(total_path / 8))

        self.mmm = likelihood_mmm
        self.total_node = int(total_node)
        self.total_path = int(total_path)
        self.node_r = int(node_r)
        self.node_angle_max = node_angle_max
        self.total_path_seed = int(total_path_seed)

    def _prob_sum_state(self, x0, y0, node_angle, prob_multiplier=255.0):
        """
        Author's exact probability calculation method.

        Formula: prob += np.log(mmm[ry, rx] * 255)
        """
        total_pixel_y, total_pixel_x = self.mmm.shape

        x1 = int(self.node_r * np.cos(node_angle) + x0)
        y1 = int(self.node_r * np.sin(node_angle) + y0)

        # Boundary check
        if (y1 < 0 or y1 >= total_pixel_y - 1 or
            x1 < 0 or x1 >= total_pixel_x - 1 or
            y0 < 0 or y0 >= total_pixel_y - 1 or
            x0 < 0 or x0 >= total_pixel_x - 1):
            prob = -np.inf
        else:
            prob = 0
            # Author's exact log-probability accumulation
            for rn in range(1, self.node_r + 1):
                rx = int(rn * np.cos(node_angle) + x0)
                ry = int(rn * np.sin(node_angle) + y0)

                # Author's zero handling
                if self.mmm[ry, rx] == 0:
                    prob = prob + 0
                else:
                    # Author's exact formula: prob += np.log(mmm[ry, rx] * 255)
                    prob = prob + np.log(self.mmm[ry, rx] * prob_multiplier)

        return (prob, x1, y1)

    def _node_link_intensity(self, node_A, node_B):
        """
        Author's exact neurite object validation method.

        Computes link_mean and zone_median for validation.
        """
        total_pixel_y, total_pixel_x = self.mmm.shape

        node_A_x, node_A_y = np.int64(node_A)
        node_B_x, node_B_y = np.int64(node_B)

        ox = (node_B_x - node_A_x)
        oy = (node_B_y - node_A_y)
        link_r = int((ox * ox + oy * oy)**0.5)

        link_zone = []
        link_path = []

        for zn in np.append(np.arange(-link_r, link_r, 1), np.int64([0])):
            try:
                zn_xn = int(-oy * zn / link_r)
            except:
                zn_xn = 0
            try:
                zn_yn = int(ox * zn / link_r)
            except:
                zn_yn = 0

            for rn in range(link_r):
                link_xn = int(node_A_x + ox * (rn / link_r)) + zn_xn
                link_yn = int(node_A_y + oy * (rn / link_r)) + zn_yn

                # Boundary handling
                if link_xn < 0:
                    link_xn = 0
                if link_xn >= total_pixel_x:
                    link_xn = total_pixel_x - 1
                if link_yn < 0:
                    link_yn = 0
                if link_yn >= total_pixel_y:
                    link_yn = total_pixel_y - 1

                link_zone.append(self.mmm[link_yn, link_xn])

                # Three adjacent lines for better evaluation
                if zn in [0]:
                    link_path.append(self.mmm[link_yn, link_xn])

        zone_median = np.median(link_zone)
        link_mean = np.mean(link_path)

        return (link_mean, zone_median)

    def node_HMM_path(self, seed_x, seed_y, seed_angle=None, seed_angle_max=None, prob_multiplier=255.0):
        """
        Author's exact HMM path finding algorithm with dynamic programming.

        This implements the 3-step process:
        1. Initialize with seed candidates
        2. Forward pass with state transitions
        3. Backward pass to find optimal path
        """
        if seed_angle is None:
            seed_angle = 0
        if seed_angle_max is None:
            seed_angle_max = 360 * (np.pi / 180)

        total_pixel_y, total_pixel_x = self.mmm.shape

        # Initialize arrays following author's structure
        node_aa = np.zeros([self.total_node], dtype=np.float64)
        node_xx = np.zeros([self.total_node], dtype=np.int64)
        node_yy = np.zeros([self.total_node], dtype=np.int64)

        # 3D probability tensor - author's exact structure
        node_path_path0_aa = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.float64)
        node_path_path0_xx = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.int64)
        node_path_path0_yy = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.int64)
        node_path_path0_pp = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.float64)

        # Track optimal previous states
        node_path_path0max = np.zeros([self.total_node, self.total_path], dtype=np.int64)

        # Angular spacing
        dAngle = self.node_angle_max / (self.total_path - 1)

        # STEP 1: Initialize first node with seed candidates
        Nn = 0

        # Generate seed path candidates
        node_path_path0_aa_seed = np.zeros([self.total_path_seed], dtype=np.float64)
        node_path_path0_xx_seed = np.zeros([self.total_path_seed], dtype=np.int64)
        node_path_path0_yy_seed = np.zeros([self.total_path_seed], dtype=np.int64)
        node_path_path0_pp_seed = np.zeros([self.total_path_seed], dtype=np.float64)

        # Author's seed path generation
        dAngle_seed = seed_angle_max / (self.total_path_seed - 1)

        for Pn in range(self.total_path_seed):
            node_angle = (Pn * dAngle_seed) + (seed_angle - seed_angle_max / 2)
            prob, x1, y1 = self._prob_sum_state(seed_x, seed_y, node_angle, prob_multiplier)

            node_path_path0_aa_seed[Pn] = node_angle
            node_path_path0_xx_seed[Pn] = x1
            node_path_path0_yy_seed[Pn] = y1
            node_path_path0_pp_seed[Pn] = prob

        # Select top paths from seed candidates
        top_path_from_seed = np.argsort(node_path_path0_pp_seed)[-self.total_path:]

        # Initialize first node states
        for Pn in range(self.total_path):
            Pn_seed = top_path_from_seed[Pn]
            Pn_now = 0

            node_path_path0_aa[Nn, Pn, Pn_now] = node_path_path0_aa_seed[Pn_seed]
            node_path_path0_xx[Nn, Pn, Pn_now] = node_path_path0_xx_seed[Pn_seed]
            node_path_path0_yy[Nn, Pn, Pn_now] = node_path_path0_yy_seed[Pn_seed]
            node_path_path0_pp[Nn, Pn, Pn_now] = node_path_path0_pp_seed[Pn_seed]

            # Track optimal state
            Pn_now_max = np.argmax(node_path_path0_pp[Nn, Pn, :])
            node_path_path0max[Nn, Pn] = Pn_now_max

        # Set seed position
        node_aa[0] = seed_angle
        node_xx[0] = seed_x
        node_yy[0] = seed_y

        # STEP 2: Forward pass - compute transitions for remaining nodes
        for Nn in range(1, self.total_node):
            Nn_now = Nn - 1

            for Pn in range(self.total_path):
                for Pn_now in range(self.total_path):
                    Pn_now_max = node_path_path0max[Nn_now, Pn_now]

                    # Calculate new angle based on previous state
                    prev_angle = node_path_path0_aa[Nn_now, Pn_now, Pn_now_max]
                    node_angle = (prev_angle - self.node_angle_max / 2) + (Pn * dAngle)

                    # Get previous position
                    prev_x = node_path_path0_xx[Nn_now, Pn_now, Pn_now_max]
                    prev_y = node_path_path0_yy[Nn_now, Pn_now, Pn_now_max]

                    # Compute probability for this transition
                    prob, x1, y1 = self._prob_sum_state(prev_x, prev_y, node_angle, prob_multiplier)

                    # Store state information
                    node_path_path0_aa[Nn, Pn, Pn_now] = node_angle
                    node_path_path0_xx[Nn, Pn, Pn_now] = x1
                    node_path_path0_yy[Nn, Pn, Pn_now] = y1
                    node_path_path0_pp[Nn, Pn, Pn_now] = node_path_path0_pp[Nn_now, Pn_now, Pn_now_max] + prob

                # Find optimal previous state for this path
                Pn_now_max = np.argmax(node_path_path0_pp[Nn, Pn, :])
                node_path_path0max[Nn, Pn] = Pn_now_max

        # STEP 3: Backward pass - find optimal path
        for Nn in np.arange(self.total_node - 1, 0, -1):
            Nn_now = Nn - 1

            if Nn == self.total_node - 1:
                # Find best final state
                final_scores = np.zeros(self.total_path)
                for Pn in range(self.total_path):
                    Pn_now_max = node_path_path0max[Nn, Pn]
                    final_scores[Pn] = node_path_path0_pp[Nn, Pn, Pn_now_max]
                Pn_max = np.argmax(final_scores)
            else:
                Pn_max = Pn_max_Pn_now_max

            Pn_max_Pn_now_max = node_path_path0max[Nn, Pn_max]

            # Store optimal path values
            node_aa[Nn] = node_path_path0_aa[Nn_now, Pn_max_Pn_now_max, 0] if Nn_now >= 0 else seed_angle
            node_xx[Nn] = node_path_path0_xx[Nn_now, Pn_max_Pn_now_max, 0] if Nn_now >= 0 else seed_x
            node_yy[Nn] = node_path_path0_yy[Nn_now, Pn_max_Pn_now_max, 0] if Nn_now >= 0 else seed_y

        return node_aa, node_xx, node_yy

    def chain_HMM_node(self, seed_xx, seed_yy, seed_aa=None, seed_angle_max=None,
                       chain_level=None, cut_level_first_node=None, cut_level_other_nodes=None,
                       prob_multiplier=255.0, min_high_nodes=3):
        """
        Author's exact chain HMM processing with validation.

        Processes multiple seeds and applies neurite object validation.
        """
        total_seed = len(seed_xx)

        if chain_level is None:
            chain_level = 1
        if cut_level_first_node is None:
            cut_level_first_node = 4 * chain_level
        if cut_level_other_nodes is None:
            cut_level_other_nodes = chain_level
        if seed_aa is None:
            seed_aa = np.zeros(total_seed)
        if seed_angle_max is None:
            seed_angle_max = 360 * (np.pi / 180)

        # Initialize storage arrays
        seed_node_aa = np.zeros([total_seed, self.total_node], dtype=np.float64)
        seed_node_xx = np.zeros([total_seed, self.total_node], dtype=np.int64)
        seed_node_yy = np.zeros([total_seed, self.total_node], dtype=np.int64)

        real_chain_ii_list = []
        real_chain_aa_list = []
        real_chain_xx_list = []
        real_chain_yy_list = []

        # Process each seed
        for i in range(total_seed):
            seed_a = seed_aa[i]
            seed_x = seed_xx[i]
            seed_y = seed_yy[i]

            # Run HMM path finding for this seed
            node_HMM = self.node_HMM_path(
                seed_x, seed_y,
                seed_angle=seed_a,
                seed_angle_max=seed_angle_max,
                prob_multiplier=prob_multiplier
            )
            seed_node_aa[i], seed_node_xx[i], seed_node_yy[i] = node_HMM

            # Apply author's exact neurite object validation
            high_node = []
            for Nn in range(self.total_node):
                if Nn == 0:
                    Nn_A = Nn
                    Nn_B = Nn + 1
                    cut_level = cut_level_first_node  # 4 * chain_level for first node
                else:
                    Nn_A = Nn - 1
                    Nn_B = Nn
                    cut_level = cut_level_other_nodes  # chain_level for other nodes

                node_A = np.array([seed_node_xx[i, Nn_A], seed_node_yy[i, Nn_A]])
                node_B = np.array([seed_node_xx[i, Nn_B], seed_node_yy[i, Nn_B]])

                link_mean, zone_median = self._node_link_intensity(node_A, node_B)

                # Author's exact validation formula
                if link_mean > cut_level * zone_median:
                    high_node.append(Nn)

            # Author's continuous chain validation
            if len(high_node) >= min_high_nodes:
                real_chain = []
                j = high_node[0]
                real_chain.append(high_node[0])

                for k in high_node[1:]:
                    if k == j + 1:
                        real_chain.append(k)
                        j = j + 1

                # Store valid chains
                real_chain_ii_list.append(real_chain)
                real_chain_aa_list.append(seed_node_aa[i])
                real_chain_xx_list.append(seed_node_xx[i])
                real_chain_yy_list.append(seed_node_yy[i])

        # Convert to numpy arrays (author's format)
        real_chain_ii = np.array(real_chain_ii_list, dtype=object)
        real_chain_aa = np.array(real_chain_aa_list)
        real_chain_xx = np.array(real_chain_xx_list)
        real_chain_yy = np.array(real_chain_yy_list)

        return (real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy, seed_node_xx, seed_node_yy)

    def pair_HMM_chain(self, seed_xx, seed_yy, seed_aa=None, seed_angle_max=None,
                       chain_level=None, cut_level_first_node=None, cut_level_other_nodes=None,
                       prob_multiplier=255.0, min_high_nodes=3):
        """
        Author's exact pair HMM chain algorithm with reaction mechanism.

        This implements the complete RRS algorithm:
        1. First chain generation
        2. Reaction seed generation (180° opposite direction)
        3. Secondary chain generation
        """
        # STEP 1: Generate first chains
        chain_HMM = self.chain_HMM_node(
            seed_xx, seed_yy,
            seed_aa=seed_aa,
            seed_angle_max=seed_angle_max,
            chain_level=chain_level,
            cut_level_first_node=cut_level_first_node,
            cut_level_other_nodes=cut_level_other_nodes,
            prob_multiplier=prob_multiplier,
            min_high_nodes=min_high_nodes
        )

        real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_HMM[0:4]

        # STEP 2: Generate reaction seeds (author's exact method)
        pair_seed_aa = []
        pair_seed_xx = []
        pair_seed_yy = []

        for ri in range(real_chain_ii.shape[0]):
            chain_aa = real_chain_aa[ri][real_chain_ii[ri]]

            if len(chain_aa) >= 2:
                # Author's exact reaction mechanism:
                # - Birth from first active node (chain_xx[0], chain_yy[0])
                # - 180° opposite direction (chain_aa[1] + 180°)
                pair_seed_aa.append(chain_aa[1] + 180 * (np.pi / 180))

                chain_xx = real_chain_xx[ri][real_chain_ii[ri]]
                pair_seed_xx.append(chain_xx[0])  # First active node

                chain_yy = real_chain_yy[ri][real_chain_ii[ri]]
                pair_seed_yy.append(chain_yy[0])  # First active node

        # STEP 3: Generate secondary chains if we have reaction seeds
        if len(pair_seed_xx) > 0:
            # Author's method: limit secondary search to 180° (half circle)
            secondary_seed_angle_max = 180 * (np.pi / 180)

            pair_chain_HMM = self.chain_HMM_node(
                np.array(pair_seed_xx),
                np.array(pair_seed_yy),
                seed_aa=np.array(pair_seed_aa),
                seed_angle_max=secondary_seed_angle_max,
                chain_level=chain_level,
                cut_level_first_node=cut_level_first_node,
                cut_level_other_nodes=cut_level_other_nodes,
                prob_multiplier=prob_multiplier,
                min_high_nodes=min_high_nodes
            )
        else:
            pair_chain_HMM = None

        return (chain_HMM, pair_chain_HMM, pair_seed_xx, pair_seed_yy)


# All old functions have been replaced by the modern implementation above.
# The CPU version now provides exact algorithmic fidelity to the original paper
# while being fully integrated with the OpenHCS framework.

# Legacy code removed - all functionality now available through:
# trace_neurites_rrs_exact_author_implementation_cpu()

