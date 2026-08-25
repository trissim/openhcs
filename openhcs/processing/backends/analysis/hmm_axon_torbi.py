# Torbi-accelerated version of hmm_axon.py
"""
OpenHCS-compatible neurite tracing using alvahmm RRS algorithm with torbi GPU acceleration.

This version uses torbi for GPU-accelerated Viterbi decoding while maintaining
the same API as the original CPU version.
"""

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import skimage
from skimage.feature import blob_dog as local_max
from skimage.feature import canny
from skimage.filters import median, threshold_li
from skimage.morphology import skeletonize

from openhcs.constants.constants import Backend
from openhcs.core.lazy_gpu_imports import torch
from openhcs.core.memory import torch as torch_func
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import (
    CsvOptions,
    JsonOptions,
    MaterializationSpec,
    TextOptions,
    TiffStackOptions,
)

# Import torch using the established optional import pattern
from openhcs.utils.import_utils import optional_import_placeholder

# Import torbi for GPU-accelerated Viterbi decoding
torbi = optional_import_placeholder("torbi")


## Greenfield: materialization is writer-driven (no custom materializers).


def materialize_trace_visualizations(
    data: List[np.ndarray], path: str, filemanager
) -> str:
    """Materialize trace visualizations as individual TIFF files."""

    if not data:
        # Create empty summary file to indicate no visualizations were generated
        summary_path = path.replace(".pkl", "_trace_summary.txt")
        summary_content = (
            "No trace visualizations generated (return_trace_visualizations=False)\n"
        )
        from openhcs.constants.constants import Backend

        filemanager.save(summary_content, summary_path, Backend.DISK.value)
        return summary_path

    # Generate output file paths based on the input path
    base_path = path.replace(".pkl", "")

    # Save each visualization as a separate TIFF file
    for i, visualization in enumerate(data):
        viz_filename = f"{base_path}_slice_{i:03d}.tif"

        # Convert visualization to uint8 for tracing visualizations (standard format)
        if visualization.dtype != np.uint8:
            # Normalize to uint8 range if needed
            if visualization.max() <= 1.0:
                viz_uint8 = (visualization * 255).astype(np.uint8)
            else:
                viz_uint8 = visualization.astype(np.uint8)
        else:
            viz_uint8 = visualization

        # Save using filemanager
        from openhcs.constants.constants import Backend

        filemanager.save(viz_uint8, viz_filename, Backend.DISK.value)

    # Return summary path
    summary_path = f"{base_path}_trace_summary.txt"
    summary_content = f"Trace visualizations saved: {len(data)} files\n"
    summary_content += f"Base filename pattern: {base_path}_slice_XXX.tif\n"
    summary_content += f"Visualization dtype: {data[0].dtype}\n"
    summary_content += f"Visualization shape: {data[0].shape}\n"

    filemanager.save(summary_content, summary_path, Backend.DISK.value)

    return summary_path


# Import alvahmm submodules through the optional import boundary so a partial
# installation cannot break module discovery or seize root logging.
alva_MCMC_torbi = optional_import_placeholder("alva_machinery.markov.aChain_torbi")
alva_MCMC = optional_import_placeholder("alva_machinery.markov.aChain")
alva_branch = optional_import_placeholder("alva_machinery.branching.aWay")
ALVA_AVAILABLE = bool(alva_MCMC_torbi and alva_MCMC and alva_branch)


class SeedingMethod(Enum):
    """Seeding methods for neurite tracing."""

    RANDOM = "random"  # Paper's original method - random seeds across entire image
    BLOB_DETECTION = "blob"  # Enhanced method - seeds on detected blob structures
    CANNY_EDGES = "canny"  # Alternative - seeds on Canny edge detection
    GROWTH_CONES = "growth_cones"  # Alternative - seeds on detected growth cones


class VisualizationMode(Enum):
    """Visualization modes for trace output."""

    NONE = "none"  # Return zeros array (no visualization)
    TRACE_ONLY = "trace"  # Show only traced neurites (binary mask)
    OVERLAY = "overlay"  # Show original image with traced neurites overlaid


class OutputMode(Enum):
    """Output visualization modes."""

    TRACE_ONLY = "trace_only"  # Binary mask of traced neurites only
    OVERLAY = "overlay"  # Original image with traces overlaid
    NONE = "none"  # Return original image unchanged


def normalize(img, percentile=99.9):
    percentile_value = np.percentile(img, percentile)
    img = img / percentile_value  # Scale the image to the nth percentile value
    img = np.clip(img, 0, 100)  # You can change 1 to 100 if you want percentages
    # img = img - img.min()
    # img = img / img.max()
    return img


def boundary_masking_canny(image):
    bool_im_axon_edit = canny(image)
    bool_im_axon_edit[:, :2] = False
    bool_im_axon_edit[:, -2:] = False
    bool_im_axon_edit[:2, :] = False
    bool_im_axon_edit[-2:, :] = False
    return np.array(bool_im_axon_edit, dtype=np.int64)


def boundary_masking_threshold(image, threshold=threshold_li, min_size=2):
    threshed = threshold(image)
    bool_image = image > threshed
    bool_image[:, :2] = False
    bool_image[:, -2:] = False
    bool_image[:2, :] = False
    bool_image[-2:, :] = False
    cleaned_bool_im_axon_edit = skeletonize(bool_image)
    return np.array(bool_image, dtype=np.int64)


def boundary_masking_blob(image, min_sigma=1, max_sigma=2, threshold=0.02):
    if min_sigma is None:
        min_sigma = 1
    if max_sigma is None:
        max_sigma = 2
    if threshold is None:
        threshold = 0.02

    image_median = median(image)
    galaxy = local_max(
        image_median, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold
    )
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
    threshold: float = 0.02,
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


def selected_seeding(
    image,
    seed_xx,
    seed_yy,
    chain_level=1.05,
    total_node=8,
    node_r=None,
    line_length_min=32,
):
    """Original CPU version for reference."""
    im_copy = np.copy(image)
    alva_HMM = alva_MCMC.AlvaHmm(
        im_copy,
        total_node=total_node,
        total_path=None,
        node_r=node_r,
        node_angle_max=None,
    )
    chain_HMM_1st, pair_chain_HMM, pair_seed_xx, pair_seed_yy = alva_HMM.pair_HMM_chain(
        seed_xx=seed_xx,
        seed_yy=seed_yy,
        chain_level=chain_level,
    )
    for chain_i in [0, 1]:
        chain_HMM = [chain_HMM_1st, pair_chain_HMM][chain_i]
        real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_HMM[0:4]
        seed_node_xx, seed_node_yy = chain_HMM[4:6]

    chain_im_fine = alva_HMM.chain_image(
        chain_HMM_1st,
        pair_chain_HMM,
    )
    return alva_branch.connect_way(
        chain_im_fine,
        line_length_min=line_length_min,
        free_zone_from_y0=None,
    )


def selected_seeding_torbi(
    image,
    seed_xx,
    seed_yy,
    chain_level=1.05,
    total_node=8,
    node_r=None,
    line_length_min=32,
    device=None,
):
    """
    Torbi-accelerated version of selected_seeding with batched processing.

    Processes all seeds in parallel using torbi's batch capabilities for maximum GPU utilization.
    """
    print(f"🚀 Processing {len(seed_xx)} seeds in parallel with torbi GPU acceleration")

    im_copy = np.copy(image)

    # Use torbi-accelerated HMM class with batched processing
    alva_HMM = alva_MCMC_torbi.AlvaHmmTorbi(
        im_copy,
        total_node=total_node,
        total_path=None,
        node_r=node_r,
        node_angle_max=None,
        device=device,
    )

    # Perform batched bidirectional HMM tracing with torbi acceleration
    chain_HMM_1st, pair_chain_HMM, pair_seed_xx, pair_seed_yy = (
        alva_HMM.pair_HMM_chain_batched(
            seed_xx=seed_xx, seed_yy=seed_yy, chain_level=chain_level
        )
    )

    for chain_i in [0, 1]:
        chain_HMM = [chain_HMM_1st, pair_chain_HMM][chain_i]
        real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_HMM[0:4]
        seed_node_xx, seed_node_yy = chain_HMM[4:6]

    chain_im_fine = alva_HMM.chain_image(chain_HMM_1st, pair_chain_HMM)
    return alva_branch.connect_way(
        chain_im_fine, line_length_min=line_length_min, free_zone_from_y0=None
    )


def euclidian_distance(x1, y1, x2, y2):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def extract_graph(root_tree_xx, root_tree_yy):
    graph = nx.Graph()
    for path_x, path_y in zip(root_tree_xx, root_tree_yy):
        for x, y in zip(path_x, path_y):
            graph.add_node((x, y))
        for i in range(len(path_x) - 1):
            distance = euclidian_distance(
                path_x[i], path_y[i], path_x[i + 1], path_y[i + 1]
            )
            graph.add_edge(
                (path_x[i], path_y[i]), (path_x[i + 1], path_y[i + 1]), weight=distance
            )
    return graph


def graph_to_length(graph):
    total_distance = 0
    for u, v, data in graph.edges(data=True):
        total_distance += data["weight"]
    return total_distance


def create_overlay_from_graph(
    original_image: np.ndarray, graph: nx.Graph
) -> np.ndarray:
    """
    Create overlay visualization with traces on original image.

    Args:
        original_image: Original 2D input image (numpy array)
        graph: NetworkX graph containing trace coordinates as (x, y) tuples

    Returns:
        2D overlay array with traces highlighted on original image
    """
    overlay = original_image.copy()
    # Set trace pixels to maximum intensity for visibility
    max_val = np.max(original_image) if original_image.size > 0 else 1

    for u, v in graph.edges:
        x1, y1 = u  # Nodes are stored as (x, y) tuples
        x2, y2 = v  # Nodes are stored as (x, y) tuples
        # Bounds checking
        if 0 <= y1 < original_image.shape[0] and 0 <= x1 < original_image.shape[1]:
            overlay[y1, x1] = max_val
        if 0 <= y2 < original_image.shape[0] and 0 <= x2 < original_image.shape[1]:
            overlay[y2, x2] = max_val

    return overlay


def create_visualization_array(
    original_image: np.ndarray, graph: nx.Graph, mode: VisualizationMode
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
            x1, y1 = u  # Nodes are stored as (x, y) tuples
            x2, y2 = v  # Nodes are stored as (x, y) tuples
            # Bounds checking
            if 0 <= y1 < original_image.shape[0] and 0 <= x1 < original_image.shape[1]:
                trace_mask[y1, x1] = 1
            if 0 <= y2 < original_image.shape[0] and 0 <= x2 < original_image.shape[1]:
                trace_mask[y2, x2] = 1
        return trace_mask

    elif mode == VisualizationMode.OVERLAY:
        # Use shared overlay function
        return create_overlay_from_graph(original_image, graph)

    else:
        raise ValueError(f"Unknown visualization mode: {mode}")


@artifact_outputs(
    (
        "hmm_analysis",
        MaterializationSpec(
            JsonOptions(source="summary", filename_suffix=".json"),
            TextOptions(source="graphml", filename_suffix="_graph.graphml"),
            CsvOptions(source="edges", filename_suffix="_edges.csv"),
            primary=0,
            allowed_backends=[Backend.DISK.value],
        ),
    ),
    (
        "trace_visualizations",
        MaterializationSpec(
            TiffStackOptions(normalize_uint8=True, summary_suffix="_trace_summary.txt")
        ),
    ),
)
@torch_func
def trace_neurites_rrs_alva_torbi(
    image_stack: "torch.Tensor",
    seeding_method: SeedingMethod = SeedingMethod.BLOB_DETECTION,
    return_trace_visualizations: bool = False,
    trace_visualization_mode: VisualizationMode = VisualizationMode.TRACE_ONLY,
    chain_level: float = 1.05,
    node_r: Optional[int] = None,
    total_node: Optional[int] = None,
    line_length_min: int = 32,
    num_seeds: int = 100,
    min_sigma: float = 1.0,
    max_sigma: float = 2.0,
    threshold: float = 0.02,
    normalize_image: bool = False,
    percentile: float = 99.9,
) -> "Tuple[torch.Tensor, Dict[str, Any], List[np.ndarray]]":
    """
    Trace neurites using the alvahmm RRS algorithm with torbi GPU acceleration.

    This is the GPU-accelerated version using torbi for Viterbi decoding while
    maintaining the same API as the CPU version. Falls back to CPU if torbi
    is not available.

    Args:
        image_stack: 3D torch tensor of shape (Z, Y, X) - input image stack
        seeding_method: Method for seed generation (RANDOM=paper default, BLOB_DETECTION=enhanced)
        return_trace_visualizations: Whether to generate trace visualizations as special output
        trace_visualization_mode: How to visualize results (NONE, TRACE_ONLY, OVERLAY)
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
        result_image: Original image stack unchanged (Z, Y, X)
        analysis_results: HMM analysis data structure with graph and metrics
        trace_visualizations: (Special output) List of visualization arrays if return_trace_visualizations=True
    """
    # Validate input is 3D
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {image_stack.ndim}D")

    # Get device from input tensor
    device = image_stack.device

    # Process each slice individually
    Z, Y, X = image_stack.shape
    all_graphs = []
    trace_visualizations = []

    for z in range(Z):
        im_axon = image_stack[z].cpu().numpy().astype(np.float64)

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
            threshold=threshold,
        )

        # Perform RRS tracing with bidirectional HMM chains (torbi-accelerated)
        root_tree_yy, root_tree_xx, root_tip_yy, root_tip_xx = selected_seeding_torbi(
            im_axon,
            seed_xx,
            seed_yy,
            chain_level=chain_level,
            node_r=node_r,
            total_node=total_node,
            line_length_min=line_length_min,
            device=device,
        )

        # Extract graph representation for this slice
        graph = extract_graph(root_tree_xx, root_tree_yy)
        all_graphs.append(graph)

        # Create visualization for this slice if requested
        if return_trace_visualizations:
            visualization = create_visualization_array(
                im_axon, graph, trace_visualization_mode
            )
            trace_visualizations.append(visualization)

    # Combine all graphs (for compatibility, return the first one)
    combined_graph = all_graphs[0] if all_graphs else nx.Graph()

    # Compile analysis results
    analysis_results = _compile_hmm_analysis_results(
        combined_graph,
        all_graphs,
        image_stack.shape,
        seeding_method,
        trace_visualization_mode,
        chain_level,
        node_r,
        total_node,
        line_length_min,
    )

    # Always return original image, analysis results, and trace visualizations
    return image_stack, analysis_results, trace_visualizations


def _compile_hmm_analysis_results(
    combined_graph: nx.Graph,
    all_graphs: List[nx.Graph],
    image_shape: Tuple[int, int, int],
    seeding_method,  # SeedingMethod enum
    visualization_mode,  # VisualizationMode enum
    chain_level: float,
    node_r: Optional[int],
    total_node: Optional[int],
    line_length_min: int,
) -> Dict[str, Any]:
    """Compile comprehensive HMM analysis results for torbi version."""
    import io
    from datetime import datetime

    # Compute summary metrics from the graph
    num_nodes = combined_graph.number_of_nodes()
    num_edges = combined_graph.number_of_edges()

    # Calculate total trace length
    total_length = 0.0
    edge_lengths = []
    for u, v, data in combined_graph.edges(data=True):
        # Calculate Euclidean distance between nodes
        x1, y1 = u
        x2, y2 = v
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        edge_lengths.append(length)
        total_length += length

    # Summary metrics
    summary = {
        "total_trace_length": float(total_length),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "num_slices_processed": len(all_graphs),
        "mean_edge_length": (
            float(sum(edge_lengths) / len(edge_lengths)) if edge_lengths else 0.0
        ),
        "max_edge_length": float(max(edge_lengths)) if edge_lengths else 0.0,
        "graph_density": float(nx.density(combined_graph)) if num_nodes > 1 else 0.0,
        "num_connected_components": int(nx.number_connected_components(combined_graph)),
    }

    # Metadata
    metadata = {
        "algorithm": "alvahmm_rrs_torbi",
        "seeding_method": seeding_method.value,
        "visualization_mode": visualization_mode.value,
        "chain_level": chain_level,
        "node_r": node_r,
        "total_node": total_node,
        "line_length_min": line_length_min,
        "image_shape": image_shape,
        "processing_timestamp": datetime.now().isoformat(),
        "gpu_accelerated": True,
    }

    graphml: str = ""
    edges: List[Dict[str, Any]] = []
    if combined_graph and combined_graph.number_of_nodes() > 0:
        try:
            buf = io.StringIO()
            nx.write_graphml(combined_graph, buf)
            graphml = buf.getvalue()
        except Exception:
            graphml = ""

        for u, v, edge_attrs in combined_graph.edges(data=True):
            edges.append(
                {
                    "source_x": u[0],
                    "source_y": u[1],
                    "target_x": v[0],
                    "target_y": v[1],
                    **(edge_attrs or {}),
                }
            )

    return {
        "summary": {**summary, **metadata},
        "graphml": graphml,
        "edges": edges,
    }


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
