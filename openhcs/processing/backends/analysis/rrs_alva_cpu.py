#!/usr/bin/env python3
"""
OpenHCS-compatible version of the original alvahmm RRS algorithm.

Based on the CPU implementation from:
https://github.com/trissim/alvahmm

Original paper:
"Random-Reaction-Seed Method for Automated Identification of Neurite Elongation and Branching"
https://doi.org/10.1038/s41598-019-39962-0

This version preserves the exact algorithm from the paper while making it compatible
with OpenHCS conventions and preparing for GPU acceleration with torbi.
"""

import numpy as np
import torch
import logging
from typing import Tuple, Dict, List, Optional
from skimage.morphology import dilation, disk, skeletonize

logger = logging.getLogger(__name__)


class AlvaHmmCPU:
    """
    CPU implementation of the RRS (Random-Reaction-Seed) algorithm.
    
    This preserves the exact algorithm from the original paper while preparing
    for incremental conversion to GPU acceleration with torbi.
    """
    
    def __init__(self,
                 likelihood_image: np.ndarray,
                 total_node: int = 16,
                 total_path: int = 8,
                 node_r: int = 5,
                 node_angle_max: float = np.pi / 2):
        """
        Initialize the AlvaHmm processor.
        
        Args:
            likelihood_image: Input image normalized to [0,1]
            total_node: N = number of HMM nodes in chain (paper default: 16)
            total_path: Number of possible path directions (paper default: 8)
            node_r: r = path length between adjacent nodes (paper default: 5)
            node_angle_max: Maximum angle range for path search (paper default: π/2)
        """
        # Normalize image if needed
        if likelihood_image.min() < 0 or likelihood_image.max() > 1:
            likelihood_image = likelihood_image - likelihood_image.min()
            likelihood_image = likelihood_image / likelihood_image.max()
            logger.info(f'Normalized image range: [{likelihood_image.min():.4f}, {likelihood_image.max():.4f}]')
        
        self.mmm = likelihood_image
        self.total_node = int(total_node)
        self.total_path = int(total_path)
        self.node_r = int(node_r)
        self.node_angle_max = node_angle_max
        
        # Possible paths starting from seed (paper's symmetric computation)
        self.total_path_seed = int(1 + 8 + 8 * np.floor(total_path / 8))
        
        logger.info(f"Initialized AlvaHmm: {total_node} nodes, {total_path} paths, r={node_r}")
    
    def _prob_sum_state(self, x0: int, y0: int, node_angle: float) -> Tuple[float, int, int]:
        """
        Calculate probability sum along a path from (x0,y0) in direction node_angle.
        
        This is the core observation model from the paper.
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
            # Sum log probabilities along the path
            for rn in range(1, self.node_r + 1):
                rx = int(rn * np.cos(node_angle) + x0)
                ry = int(rn * np.sin(node_angle) + y0)
                # Avoid log(0) problem
                if self.mmm[ry, rx] == 0:
                    prob = prob + 0
                else:
                    # 255 factor to avoid negative log values (paper's approach)
                    prob = prob + np.log(self.mmm[ry, rx] * 255)
        
        return (prob, x1, y1)
    
    def _node_link_intensity(self, node_A: np.ndarray, node_B: np.ndarray) -> Tuple[float, float]:
        """
        Calculate link intensity between two nodes for validation.
        
        This implements the paper's validation criterion:
        - link_mean: mean intensity along the direct path
        - zone_median: median intensity in the surrounding zone
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
                
                # Boundary clipping
                link_xn = max(0, min(link_xn, total_pixel_x - 1))
                link_yn = max(0, min(link_yn, total_pixel_y - 1))
                
                link_zone.append(self.mmm[link_yn, link_xn])
                # Central line for path evaluation
                if zn in [0]:
                    link_path.append(self.mmm[link_yn, link_xn])
        
        zone_median = np.median(link_zone)
        link_mean = np.mean(link_path)
        return (link_mean, zone_median)

    def node_HMM_path(self,
                      seed_x: int,
                      seed_y: int,
                      seed_angle: float = 0,
                      seed_angle_max: float = 2 * np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Core HMM path finding algorithm from the paper.

        This is the method that will be replaced with torbi acceleration.
        It implements the forward-backward algorithm for finding the optimal
        path through the HMM state space.

        Args:
            seed_x, seed_y: Starting position
            seed_angle: Initial direction
            seed_angle_max: Maximum angle range for search

        Returns:
            node_aa: Angles for each node in optimal path
            node_xx: X coordinates for each node in optimal path
            node_yy: Y coordinates for each node in optimal path
        """
        # Initialize arrays for nodes and paths
        node_aa = np.zeros([self.total_node], dtype=np.float64)
        node_xx = np.zeros([self.total_node], dtype=np.int64)
        node_yy = np.zeros([self.total_node], dtype=np.int64)

        # Arrays for path exploration
        node_path_aa = np.zeros([self.total_node, self.total_path], dtype=np.float64)
        node_path_xx = np.zeros([self.total_node, self.total_path], dtype=np.int64)
        node_path_yy = np.zeros([self.total_node, self.total_path], dtype=np.int64)
        node_path_pp = np.zeros([self.total_node, self.total_path], dtype=np.float64)
        node_path_path0max = np.zeros([self.total_node, self.total_path], dtype=np.int64)

        # 3D array for full state space
        node_path_path0_aa = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.float64)
        node_path_path0_xx = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.int64)
        node_path_path0_yy = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.int64)
        node_path_path0_pp = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.float64)

        dAngle = self.node_angle_max / (self.total_path - 1)

        # Initialize first node (seed)
        Nn = 0

        # Generate seed paths in all directions
        node_path_path0_aa_seed = np.zeros([self.total_path_seed], dtype=np.float64)
        node_path_path0_xx_seed = np.zeros([self.total_path_seed], dtype=np.int64)
        node_path_path0_yy_seed = np.zeros([self.total_path_seed], dtype=np.int64)
        node_path_path0_pp_seed = np.zeros([self.total_path_seed], dtype=np.float64)

        dAngle_seed = seed_angle_max / (self.total_path_seed - 1)

        # Evaluate all possible directions from seed
        for Pn in range(self.total_path_seed):
            node_angle = (Pn * dAngle_seed) + (seed_angle - seed_angle_max / 2)
            prob, x1, y1 = self._prob_sum_state(seed_x, seed_y, node_angle)

            node_path_path0_aa_seed[Pn] = node_angle
            node_path_path0_xx_seed[Pn] = x1
            node_path_path0_yy_seed[Pn] = y1
            node_path_path0_pp_seed[Pn] = prob

        # Select top paths from seed
        top_path_from_seed = np.argsort(node_path_path0_pp_seed)[-self.total_path:]

        for Pn in range(self.total_path):
            Pn_seed = top_path_from_seed[Pn]
            Pn_now = 0

            node_path_path0_aa[Nn, Pn, Pn_now] = node_path_path0_aa_seed[Pn_seed]
            node_path_path0_xx[Nn, Pn, Pn_now] = node_path_path0_xx_seed[Pn_seed]
            node_path_path0_yy[Nn, Pn, Pn_now] = node_path_path0_yy_seed[Pn_seed]
            node_path_path0_pp[Nn, Pn, Pn_now] = node_path_path0_pp_seed[Pn_seed]

            Pn_now_max = np.argmax(node_path_path0_pp[Nn, Pn, :])
            node_path_path0max[Nn, Pn] = Pn_now_max

            node_path_aa[Nn, Pn] = node_path_path0_aa[Nn, Pn, Pn_now_max]
            node_path_xx[Nn, Pn] = node_path_path0_xx[Nn, Pn, Pn_now_max]
            node_path_yy[Nn, Pn] = node_path_path0_yy[Nn, Pn, Pn_now_max]
            node_path_pp[Nn, Pn] = node_path_path0_pp[Nn, Pn, Pn_now_max]

        # Set initial node
        node_aa[0] = seed_angle
        node_xx[0] = seed_x
        node_yy[0] = seed_y

        # Forward pass: compute future nodes
        for Nn in range(1, self.total_node):
            Nn_now = Nn - 1

            for Pn in range(self.total_path):
                for Pn_now in range(self.total_path):
                    Pn_now_max = node_path_path0max[Nn_now, Pn_now]
                    node_angle = (node_path_path0_aa[Nn_now, Pn_now, Pn_now_max]
                                  - self.node_angle_max / 2) + (Pn * dAngle)

                    prob, x1, y1 = self._prob_sum_state(
                        node_path_path0_xx[Nn_now, Pn_now, Pn_now_max],
                        node_path_path0_yy[Nn_now, Pn_now, Pn_now_max],
                        node_angle
                    )

                    node_path_path0_aa[Nn, Pn, Pn_now] = node_angle
                    node_path_path0_xx[Nn, Pn, Pn_now] = x1
                    node_path_path0_yy[Nn, Pn, Pn_now] = y1
                    node_path_path0_pp[Nn, Pn, Pn_now] = node_path_path0_pp[Nn_now, Pn_now, Pn_now_max] + prob

                Pn_now_max = np.argmax(node_path_path0_pp[Nn, Pn, :])
                node_path_path0max[Nn, Pn] = Pn_now_max

                node_path_aa[Nn, Pn] = node_path_path0_aa[Nn, Pn, Pn_now_max]
                node_path_xx[Nn, Pn] = node_path_path0_xx[Nn, Pn, Pn_now_max]
                node_path_yy[Nn, Pn] = node_path_path0_yy[Nn, Pn, Pn_now_max]
                node_path_pp[Nn, Pn] = node_path_path0_pp[Nn, Pn, Pn_now_max]

        # Backward pass: trace optimal path
        for Nn in np.arange(self.total_node - 1, 0, -1):
            Nn_now = Nn - 1
            if Nn == self.total_node - 1:
                Pn_max = np.argmax(node_path_pp[Nn, :])
            else:
                Pn_max = Pn_max_Pn_now_max

            Pn_max_Pn_now_max = node_path_path0max[Nn, Pn_max]

            node_aa[Nn] = node_path_aa[Nn_now, Pn_max_Pn_now_max]
            node_xx[Nn] = node_path_xx[Nn_now, Pn_max_Pn_now_max]
            node_yy[Nn] = node_path_yy[Nn_now, Pn_max_Pn_now_max]

        return (node_aa, node_xx, node_yy)

    def chain_HMM_node(self,
                       node_aa: np.ndarray,
                       node_xx: np.ndarray,
                       node_yy: np.ndarray,
                       chain_level: float = 1.0) -> List[int]:
        """
        Validate HMM chain using the paper's criterion.

        This implements the exact validation logic from the original paper:
        - First node: link_mean > 4 * chain_level * zone_median
        - Other nodes: link_mean > chain_level * zone_median
        - Requires at least 3 valid nodes for a successful chain

        Args:
            node_aa: Node angles
            node_xx: Node x coordinates
            node_yy: Node y coordinates
            chain_level: Validation threshold

        Returns:
            high_node: List of indices of valid nodes
        """
        high_node = []

        for Nn in range(self.total_node - 1):
            node_A = np.array([node_xx[Nn], node_yy[Nn]])
            node_B = np.array([node_xx[Nn + 1], node_yy[Nn + 1]])

            link_mean, zone_median = self._node_link_intensity(node_A, node_B)

            # Paper's validation criterion with different thresholds
            if Nn == 0:
                cut_level = 4 * chain_level  # First node: stricter threshold
            else:
                cut_level = chain_level      # Other nodes: normal threshold

            if link_mean > cut_level * zone_median:
                high_node.append(Nn)

        # Paper requires at least 3 valid nodes
        if len(high_node) >= 3:
            return high_node
        else:
            return []

    def node_HMM_path(self,
                      seed_x: int,
                      seed_y: int,
                      seed_angle: float = 0,
                      seed_angle_max: float = 2 * np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Core HMM path finding algorithm from the paper.

        This is the method that will be replaced with torbi acceleration.
        It implements the forward-backward algorithm for finding the optimal
        path through the HMM state space.

        Args:
            seed_x, seed_y: Starting position
            seed_angle: Initial direction
            seed_angle_max: Maximum angle range for search

        Returns:
            node_aa: Angles for each node in optimal path
            node_xx: X coordinates for each node in optimal path
            node_yy: Y coordinates for each node in optimal path
        """
        # Initialize arrays for nodes and paths
        node_aa = np.zeros([self.total_node], dtype=np.float64)
        node_xx = np.zeros([self.total_node], dtype=np.int64)
        node_yy = np.zeros([self.total_node], dtype=np.int64)

        # Arrays for path exploration
        node_path_aa = np.zeros([self.total_node, self.total_path], dtype=np.float64)
        node_path_xx = np.zeros([self.total_node, self.total_path], dtype=np.int64)
        node_path_yy = np.zeros([self.total_node, self.total_path], dtype=np.int64)
        node_path_pp = np.zeros([self.total_node, self.total_path], dtype=np.float64)
        node_path_path0max = np.zeros([self.total_node, self.total_path], dtype=np.int64)

        # 3D array for full state space
        node_path_path0_aa = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.float64)
        node_path_path0_xx = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.int64)
        node_path_path0_yy = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.int64)
        node_path_path0_pp = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.float64)

        dAngle = self.node_angle_max / (self.total_path - 1)

        # Initialize first node (seed)
        Nn = 0

        # Generate seed paths in all directions
        node_path_path0_aa_seed = np.zeros([self.total_path_seed], dtype=np.float64)
        node_path_path0_xx_seed = np.zeros([self.total_path_seed], dtype=np.int64)
        node_path_path0_yy_seed = np.zeros([self.total_path_seed], dtype=np.int64)
        node_path_path0_pp_seed = np.zeros([self.total_path_seed], dtype=np.float64)

        dAngle_seed = seed_angle_max / (self.total_path_seed - 1)

        # Evaluate all possible directions from seed
        for Pn in range(self.total_path_seed):
            node_angle = (Pn * dAngle_seed) + (seed_angle - seed_angle_max / 2)
            prob, x1, y1 = self._prob_sum_state(seed_x, seed_y, node_angle)

            node_path_path0_aa_seed[Pn] = node_angle
            node_path_path0_xx_seed[Pn] = x1
            node_path_path0_yy_seed[Pn] = y1
            node_path_path0_pp_seed[Pn] = prob

        # Select top paths from seed
        top_path_from_seed = np.argsort(node_path_path0_pp_seed)[-self.total_path:]

        for Pn in range(self.total_path):
            Pn_seed = top_path_from_seed[Pn]
            Pn_now = 0

            node_path_path0_aa[Nn, Pn, Pn_now] = node_path_path0_aa_seed[Pn_seed]
            node_path_path0_xx[Nn, Pn, Pn_now] = node_path_path0_xx_seed[Pn_seed]
            node_path_path0_yy[Nn, Pn, Pn_now] = node_path_path0_yy_seed[Pn_seed]
            node_path_path0_pp[Nn, Pn, Pn_now] = node_path_path0_pp_seed[Pn_seed]

            Pn_now_max = np.argmax(node_path_path0_pp[Nn, Pn, :])
            node_path_path0max[Nn, Pn] = Pn_now_max

            node_path_aa[Nn, Pn] = node_path_path0_aa[Nn, Pn, Pn_now_max]
            node_path_xx[Nn, Pn] = node_path_path0_xx[Nn, Pn, Pn_now_max]
            node_path_yy[Nn, Pn] = node_path_path0_yy[Nn, Pn, Pn_now_max]
            node_path_pp[Nn, Pn] = node_path_path0_pp[Nn, Pn, Pn_now_max]

        # Set initial node
        node_aa[0] = seed_angle
        node_xx[0] = seed_x
        node_yy[0] = seed_y

        # Forward pass: compute future nodes
        for Nn in range(1, self.total_node):
            Nn_now = Nn - 1

            for Pn in range(self.total_path):
                for Pn_now in range(self.total_path):
                    Pn_now_max = node_path_path0max[Nn_now, Pn_now]
                    node_angle = (node_path_path0_aa[Nn_now, Pn_now, Pn_now_max]
                                  - self.node_angle_max / 2) + (Pn * dAngle)

                    prob, x1, y1 = self._prob_sum_state(
                        node_path_path0_xx[Nn_now, Pn_now, Pn_now_max],
                        node_path_path0_yy[Nn_now, Pn_now, Pn_now_max],
                        node_angle
                    )

                    node_path_path0_aa[Nn, Pn, Pn_now] = node_angle
                    node_path_path0_xx[Nn, Pn, Pn_now] = x1
                    node_path_path0_yy[Nn, Pn, Pn_now] = y1
                    node_path_path0_pp[Nn, Pn, Pn_now] = node_path_path0_pp[Nn_now, Pn_now, Pn_now_max] + prob

                Pn_now_max = np.argmax(node_path_path0_pp[Nn, Pn, :])
                node_path_path0max[Nn, Pn] = Pn_now_max

                node_path_aa[Nn, Pn] = node_path_path0_aa[Nn, Pn, Pn_now_max]
                node_path_xx[Nn, Pn] = node_path_path0_xx[Nn, Pn, Pn_now_max]
                node_path_yy[Nn, Pn] = node_path_path0_yy[Nn, Pn, Pn_now_max]
                node_path_pp[Nn, Pn] = node_path_path0_pp[Nn, Pn, Pn_now_max]

        # Backward pass: trace optimal path
        for Nn in np.arange(self.total_node - 1, 0, -1):
            Nn_now = Nn - 1
            if Nn == self.total_node - 1:
                Pn_max = np.argmax(node_path_pp[Nn, :])
            else:
                Pn_max = Pn_max_Pn_now_max

            Pn_max_Pn_now_max = node_path_path0max[Nn, Pn_max]

            node_aa[Nn] = node_path_aa[Nn_now, Pn_max_Pn_now_max]
            node_xx[Nn] = node_path_xx[Nn_now, Pn_max_Pn_now_max]
            node_yy[Nn] = node_path_yy[Nn_now, Pn_max_Pn_now_max]

        return (node_aa, node_xx, node_yy)


def trace_neurites_rrs_cpu(
    image: torch.Tensor,
    seed_density: float = 0.001,
    total_node: int = 16,
    total_path: int = 8,
    node_r: int = 5,
    node_angle_max: float = np.pi / 2,
    chain_level: float = 1.0
) -> Tuple[torch.Tensor, Dict]:
    """
    CPU implementation of the RRS algorithm for neurite tracing.
    
    This function provides the exact algorithm from the paper as a baseline
    for comparison with the GPU-accelerated version.
    
    Args:
        image: Input image tensor [H, W] (normalized to [0,1])
        seed_density: Density of random seeds
        total_node: N = number of HMM nodes in chain
        total_path: Number of possible path directions
        node_r: r = path length between adjacent nodes
        node_angle_max: Maximum angle range for path search
        chain_level: Validation threshold
    
    Returns:
        result_image: Binary image with traced neurites
        trace_dict: Dictionary with trace coordinates
    """
    device = image.device
    H, W = image.shape
    
    # Convert to numpy for CPU processing
    image_np = image.cpu().numpy()
    
    logger.info(f"Starting CPU RRS algorithm: {total_node} nodes, {total_path} paths, r={node_r}")
    
    # Initialize AlvaHmm processor
    alva_hmm = AlvaHmmCPU(
        likelihood_image=image_np,
        total_node=total_node,
        total_path=total_path,
        node_r=node_r,
        node_angle_max=node_angle_max
    )
    
    # Generate random seeds
    num_seeds = int(seed_density * H * W)
    seed_xx = np.random.randint(0, W, num_seeds)
    seed_yy = np.random.randint(0, H, num_seeds)
    seed_aa = np.zeros(num_seeds)  # Start with 0 angle
    
    logger.info(f"Generated {num_seeds} random seeds")
    
    # Process random seeds
    valid_traces = 0
    trace_dict = {}
    result_image_np = np.zeros_like(image_np)

    logger.info(f"Processing {num_seeds} random seeds...")

    for i in range(num_seeds):
        # Run HMM path finding
        node_aa, node_xx, node_yy = alva_hmm.node_HMM_path(
            seed_x=seed_xx[i],
            seed_y=seed_yy[i],
            seed_angle=seed_aa[i],
            seed_angle_max=node_angle_max
        )

        # Validate the chain
        valid_nodes = alva_hmm.chain_HMM_node(
            node_aa=node_aa,
            node_xx=node_xx,
            node_yy=node_yy,
            chain_level=chain_level
        )

        # If chain is valid, add to results
        if len(valid_nodes) > 0:
            trace_coords = []

            # Draw the trace and collect coordinates
            for j in range(len(node_xx) - 1):
                x0, y0 = int(node_xx[j]), int(node_yy[j])
                x1, y1 = int(node_xx[j+1]), int(node_yy[j+1])

                # Simple line drawing
                if (0 <= x0 < W and 0 <= y0 < H and 0 <= x1 < W and 0 <= y1 < H):
                    steps = max(abs(x1 - x0), abs(y1 - y0))
                    if steps > 0:
                        for step in range(steps + 1):
                            t = step / steps
                            x_interp = int(x0 + t * (x1 - x0))
                            y_interp = int(y0 + t * (y1 - y0))
                            if 0 <= x_interp < W and 0 <= y_interp < H:
                                result_image_np[y_interp, x_interp] = 1.0
                                trace_coords.append((0.0, float(y_interp), float(x_interp)))

            # Store trace if it has coordinates
            if trace_coords:
                trace_dict[f"slice_000_trace_{valid_traces+1:05d}"] = trace_coords
                valid_traces += 1

    logger.info(f"CPU RRS algorithm complete: {valid_traces} valid traces from {num_seeds} seeds")

    # Convert back to torch tensor
    result_image = torch.from_numpy(result_image_np).to(device)

    return result_image, trace_dict
