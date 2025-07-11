# BACKUP: hmm_axon_backup.py created before OpenHCS conversion
"""
OpenHCS-compatible neurite tracing using alvahmm RRS algorithm.

Converted from file-based processing to pure array-in/array-out functions
following OpenHCS patterns.
"""

import numpy as np
import networkx as nx
import skimage
import math
from enum import Enum
from typing import Tuple, Dict, List, Optional
from skimage.feature import canny, blob_dog as local_max
from skimage.filters import median, threshold_li
from skimage.morphology import remove_small_objects, skeletonize
from openhcs.core.memory.decorators import numpy

# Import alvahmm from local copy
from .alvahmm.alva_machinery.markov import aChain as alva_MCMC
from .alvahmm.alva_machinery.branching import aWay as alva_branch


class SeedingMethod(Enum):
    """Seeding methods for neurite tracing."""
    RANDOM = "random"              # Paper's original method - random seeds across entire image
    BLOB_DETECTION = "blob"        # Enhanced method - seeds on detected blob structures
    CANNY_EDGES = "canny"          # Alternative - seeds on Canny edge detection
    GROWTH_CONES = "growth_cones"  # Alternative - seeds on detected growth cones


class VisualizationMode(Enum):
    """Visualization modes for trace output."""
    NONE = "none"           # Return zeros array (no visualization)
    TRACE_ONLY = "trace"    # Show only traced neurites (binary mask)
    OVERLAY = "overlay"     # Show original image with traced neurites overlaid


class OutputMode(Enum):
    """Output visualization modes."""
    TRACE_ONLY = "trace_only"      # Binary mask of traced neurites only
    OVERLAY = "overlay"            # Original image with traces overlaid
    NONE = "none"                  # Return original image unchanged

def normalize(img,percentile=99.9):
    percentile_value = np.percentile(img, percentile)
    img = img / percentile_value  # Scale the image to the nth percentile value
    img = np.clip(img, 0, 100)  # You can change 1 to 100 if you want percentages
    #img = img - img.min()
    #img = img / img.max()
    return img

def boundary_masking_canny(image):
    bool_im_axon_edit = canny(image)
    bool_im_axon_edit[:,:2] = False
    bool_im_axon_edit[:,-2:] = False
    bool_im_axon_edit[:2,:] = False
    bool_im_axon_edit[-2:,:] = False
    return np.array(bool_im_axon_edit,dtype=np.int64)

def boundary_masking_threshold(image,threshold=threshold_li,min_size=2):
    threshed=threshold(image)
    bool_image = image > threshed
    bool_image[:,:2] = False
    bool_image[:,-2:] = False
    bool_image[:2,:] = False
    bool_image[-2:,:] = False
    cleaned_bool_im_axon_edit = skeletonize(bool_image)
    return np.array(bool_image,dtype=np.int64)

def boundary_masking_blob(image,min_sigma = 1, max_sigma = 2, threshold = 0.02):
    if min_sigma is None:
        min_sigma = 1
    if max_sigma is None:
        max_sigma = 2
    if threshold is None:
        threshold = 0.02

    image_median = median(image)
    galaxy = local_max(image_median, min_sigma = min_sigma, max_sigma = max_sigma, threshold = threshold)
    yy = np.int64(galaxy[:, 0])
    xx = np.int64(galaxy[:, 1])
    boundary_mask = np.copy(image) * 0
    boundary_mask[yy, xx] = 1
    return boundary_mask

def random_seed_by_edge_map(edge_map):
    """Generate random seeds from detected edge/blob locations."""
    yy, xx = edge_map.nonzero()
    if len(xx) == 0:
        # No edges detected, fall back to random seeding
        return generate_random_seeds(edge_map.shape, num_seeds=100)
    seed_index = np.random.choice(len(xx), len(xx))
    seed_xx = xx[seed_index]
    seed_yy = yy[seed_index]
    return seed_xx, seed_yy


def generate_random_seeds(image_shape: Tuple[int, int], num_seeds: int = 100):
    """Generate completely random seeds across the entire image (paper's original method)."""
    height, width = image_shape
    seed_xx = np.random.randint(0, width, num_seeds)
    seed_yy = np.random.randint(0, height, num_seeds)
    return seed_xx, seed_yy


def generate_seeds_by_method(
    image: np.ndarray,
    method: SeedingMethod = SeedingMethod.BLOB_DETECTION,
    num_seeds: int = 100,
    min_sigma: float = 1.0,
    max_sigma: float = 2.0,
    threshold: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate seeds using the specified method.

    Args:
        image: Input image for seed generation
        method: Seeding method to use
        num_seeds: Number of seeds for random method
        min_sigma: Min sigma for blob detection
        max_sigma: Max sigma for blob detection
        threshold: Threshold for blob detection

    Returns:
        seed_xx, seed_yy: Arrays of seed coordinates
    """
    if method == SeedingMethod.RANDOM:
        # Paper's original method - pure random seeding
        return generate_random_seeds(image.shape, num_seeds)

    elif method == SeedingMethod.BLOB_DETECTION:
        # Enhanced method - seeds on detected blobs
        edge_map = boundary_masking_blob(image, min_sigma, max_sigma, threshold)
        return random_seed_by_edge_map(edge_map)

    elif method == SeedingMethod.CANNY_EDGES:
        # Alternative - seeds on Canny edges
        edge_map = boundary_masking_canny(image)
        return random_seed_by_edge_map(edge_map)

    elif method == SeedingMethod.GROWTH_CONES:
        # Alternative - seeds on growth cones
        return get_growth_cone_positions(image)

    else:
        raise ValueError(f"Unknown seeding method: {method}")

def get_growth_cone_positions(image):
    """
    Detect growth cone positions using morphological operations (OpenHCS-compatible).

    Args:
        image: Input image for growth cone detection

    Returns:
        seed_xx, seed_yy: Arrays of growth cone center coordinates
    """
    # Threshold the image to create a binary mask
    mask = image > skimage.filters.threshold_otsu(image)

    # Use morphological closing to fill in small gaps in the mask
    mask = skimage.morphology.closing(mask, skimage.morphology.disk(3))
    labeled = skimage.measure.label(mask)
    props = skimage.measure.regionprops(labeled)

    seed_xx = []
    seed_yy = []
    for prop in props:
        seed_xx.append(prop.centroid[1])  # x coordinate
        seed_yy.append(prop.centroid[0])  # y coordinate

    return np.array(seed_xx), np.array(seed_yy)

def selected_seeding(image,seed_xx,seed_yy,chain_level=1.05,total_node=8,node_r=None,line_length_min=32):
    im_copy=np.copy(image)
    alva_HMM = alva_MCMC.AlvaHmm(im_copy,
                                total_node = total_node,
                                total_path = None,
                                node_r = node_r,
                                node_angle_max = None,)
    chain_HMM_1st, pair_chain_HMM, pair_seed_xx, pair_seed_yy = alva_HMM.pair_HMM_chain(seed_xx = seed_xx,
                                                                                        seed_yy = seed_yy,
                                                                                        chain_level = chain_level,)
    for chain_i in [0, 1]:
                chain_HMM = [chain_HMM_1st, pair_chain_HMM][chain_i]
                real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_HMM[0:4]
                seed_node_xx, seed_node_yy = chain_HMM[4:6]

    chain_im_fine = alva_HMM.chain_image(chain_HMM_1st, pair_chain_HMM,)
    return alva_branch.connect_way(chain_im_fine,
                                    line_length_min = line_length_min,
                                    free_zone_from_y0 = None,)

def euclidian_distance(x1, y1, x2, y2):
  distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
  return distance

def extract_graph(root_tree_xx,root_tree_yy):
    graph = nx.Graph()
    for path_x,path_y in zip(root_tree_xx,root_tree_yy):
        for x,y in zip(path_x,path_y):
            graph.add_node((x,y))
        for i in range(len(path_x)-1):
            distance=euclidian_distance(path_x[i], path_y[i], path_x[i + 1], path_y[i + 1])
            graph.add_edge((path_x[i], path_y[i]), (path_x[i + 1], path_y[i + 1]),weight=distance)
    return graph

def graph_to_length(graph):
    total_distance = 0
    for u, v, data in graph.edges(data=True):
        total_distance += data['weight']
    return total_distance

def create_visualization_array(
    original_image: np.ndarray,
    graph: nx.Graph,
    mode: VisualizationMode
) -> np.ndarray:
    """
    Create visualization array based on the specified mode.

    Args:
        original_image: Original 2D image
        graph: NetworkX graph with traced neurites
        mode: Visualization mode

    Returns:
        2D array for visualization
    """
    if mode == VisualizationMode.NONE:
        # Return zeros array
        return np.zeros_like(original_image, dtype=original_image.dtype)

    elif mode == VisualizationMode.TRACE_ONLY:
        # Create binary mask with traced neurites
        trace_mask = np.zeros_like(original_image, dtype=np.uint8)
        for u, v in graph.edges:
            y1, x1 = u
            y2, x2 = v
            # Bounds checking
            if (0 <= y1 < original_image.shape[0] and 0 <= x1 < original_image.shape[1]):
                trace_mask[y1, x1] = 1
            if (0 <= y2 < original_image.shape[0] and 0 <= x2 < original_image.shape[1]):
                trace_mask[y2, x2] = 1
        return trace_mask

    elif mode == VisualizationMode.OVERLAY:
        # Create overlay of original image with traces
        overlay = original_image.copy()
        # Set trace pixels to maximum intensity for visibility
        max_val = np.max(original_image) if original_image.size > 0 else 1
        for u, v in graph.edges:
            y1, x1 = u
            y2, x2 = v
            # Bounds checking
            if (0 <= y1 < original_image.shape[0] and 0 <= x1 < original_image.shape[1]):
                overlay[y1, x1] = max_val
            if (0 <= y2 < original_image.shape[0] and 0 <= x2 < original_image.shape[1]):
                overlay[y2, x2] = max_val
        return overlay

    else:
        raise ValueError(f"Unknown visualization mode: {mode}")

@numpy
def trace_neurites_rrs_alva(
    image_stack: np.ndarray,
    seeding_method: SeedingMethod = SeedingMethod.BLOB_DETECTION,
    visualization_mode: VisualizationMode = VisualizationMode.TRACE_ONLY,
    chain_level: float = 1.05,
    node_r: Optional[int] = None,
    total_node: Optional[int] = None,
    line_length_min: int = 32,
    num_seeds: int = 100,
    min_sigma: float = 1.0,
    max_sigma: float = 2.0,
    threshold: float = 0.02,
    normalize_image: bool = False,
    percentile: float = 99.9
) -> Tuple[np.ndarray, nx.Graph]:
    """
    Trace neurites using the alvahmm RRS (Random-Reaction-Seed) algorithm.

    This is the OpenHCS-compatible version of the original alvahmm implementation.
    Performs bidirectional HMM tracing with branching analysis to reconstruct
    complete neurite morphology.

    Args:
        image_stack: 3D array of shape (Z, Y, X) - input image stack
        seeding_method: Method for seed generation (RANDOM=paper default, BLOB_DETECTION=enhanced)
        visualization_mode: How to visualize results (NONE, TRACE_ONLY, OVERLAY)
        chain_level: Validation threshold for HMM chains (default: 1.05)
        node_r: Path length between adjacent nodes (default: None, uses alvahmm default)
        total_node: Number of HMM nodes in chain (default: None, uses alvahmm default)
        line_length_min: Minimum line length for connection (default: 32)
        num_seeds: Number of seeds for random seeding method (default: 100)
        min_sigma: Minimum sigma for blob detection (default: 1.0)
        max_sigma: Maximum sigma for blob detection (default: 2.0)
        threshold: Threshold for blob detection (default: 0.02)
        normalize_image: Whether to apply percentile normalization (default: False, paper doesn't use)
        percentile: Percentile for normalization if enabled (default: 99.9)

    Returns:
        result_image: 3D array (same dimensions as input) with visualization based on mode
        graph: NetworkX graph object with traced neurites and edge weights
    """
    # Validate input is 3D
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D array, got {image_stack.ndim}D")

    # For now, process the first slice (Z=0) - can be extended for 3D later
    # This follows the original implementation which was 2D
    im_axon = image_stack[0].astype(np.float64)

    # Optional normalization (removed from default, paper doesn't use)
    if normalize_image:
        im_axon = normalize(im_axon, percentile=percentile)

    # Generate seeds using selected method
    seed_xx, seed_yy = generate_seeds_by_method(
        im_axon,
        method=seeding_method,
        num_seeds=num_seeds,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold
    )

    # Perform RRS tracing with bidirectional HMM chains
    root_tree_yy, root_tree_xx, root_tip_yy, root_tip_xx = selected_seeding(
        im_axon,
        seed_xx,
        seed_yy,
        chain_level=chain_level,
        node_r=node_r,
        total_node=total_node,
        line_length_min=line_length_min
    )

    # Extract graph representation
    graph = extract_graph(root_tree_xx, root_tree_yy)

    # Create visualization based on selected mode
    result_2d = create_visualization_array(im_axon, graph, visualization_mode)

    # Convert back to 3D format (same dimensions as input)
    Z, Y, X = image_stack.shape
    result_image = np.zeros((Z, Y, X), dtype=result_2d.dtype)
    result_image[0] = result_2d

    return result_image, graph


# Legacy file-based processing function (kept for reference)
def process_file_legacy(filename, input_folder, output_folder, **kwargs):
    """
    Legacy file-based processing function.

    This is kept for reference but should not be used in OpenHCS.
    Use trace_neurites_rrs_alva() instead for array-in/array-out processing.
    """
    raise NotImplementedError(
        "Legacy file-based processing not supported in OpenHCS. "
        "Use trace_neurites_rrs_alva() for array-in/array-out processing."
    )

