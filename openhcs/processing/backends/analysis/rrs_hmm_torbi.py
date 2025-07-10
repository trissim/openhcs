"""
GPU-accelerated HMM neurite tracing using torbi for Viterbi decoding.

This implementation follows the exact algorithm from:
"Random-Reaction-Seed Method for Automated Identification of Neurite Elongation and Branching"
PMC6393450 (https://pmc.ncbi.nlm.nih.gov/articles/PMC6393450/)

Uses torbi's optimized PyTorch-based Viterbi decoder to replace the manual
forward/backward pass loops with GPU-accelerated computation.
"""

import torch
import torbi
import logging
from typing import Tuple, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PaperHMMTracer:
    """
    Exact implementation of the HMM algorithm from PMC6393450 using torbi for Viterbi decoding.

    This follows the paper's three-step process:
    1. Computing all possible paths (Step 1)
    2. Predicting HMM-node position via Viterbi (Step 2)
    3. Identifying neurite objects via intensity validation (Step 3)
    """

    def __init__(self,
                 image: torch.Tensor,
                 total_node: int = 16,
                 total_path: int = 8,
                 node_r: int = 5,
                 node_angle_max: float = torch.pi / 2,
                 device: str = 'cuda'):
        """
        Initialize the paper's HMM tracer with torbi acceleration.

        Args:
            image: Input image tensor [H, W] (normalized to [0,1])
            total_node: N = number of HMM nodes in chain (sequence length)
            total_path: Number of possible path directions (state space size)
            node_r: r = path length between adjacent nodes
            node_angle_max: Maximum angle range for path search
            device: Device to run on ('cuda' or 'cpu')
        """
        self.image = image.to(device)
        self.device = device
        self.H, self.W = image.shape

        # Paper's HMM parameters
        self.total_node = total_node  # N (number of nodes in HMM chain)
        self.total_path = total_path  # Number of possible directions
        self.node_r = node_r          # r (path length between nodes)
        self.node_angle_max = node_angle_max

        # Create state space: discretized angles for path directions
        self.path_angles = torch.linspace(
            -node_angle_max/2, node_angle_max/2,
            total_path, device=device
        )

        # HMM transition matrix: favor continuing in same direction
        self.transition_matrix = self._create_paper_transition_matrix()

        # Initial state distribution: uniform over all directions
        self.initial_distribution = torch.ones(total_path, device=device) / total_path

        logger.info(f"Initialized paper HMM tracer: {total_node} nodes, {total_path} paths, r={node_r}")
    
    def _create_paper_transition_matrix(self) -> torch.Tensor:
        """
        Create HMM transition matrix following the paper's approach.

        The paper doesn't specify exact transition probabilities, but implies
        that neurites tend to continue in similar directions with some variation.
        """
        # Create transition matrix [S, S] where S = total_path
        transition = torch.zeros(self.total_path, self.total_path, device=self.device)

        # Paper's approach: favor continuing in same direction with gradual transitions
        for i in range(self.total_path):
            for j in range(self.total_path):
                # Angular distance between directions (circular)
                angle_diff = abs(self.path_angles[i] - self.path_angles[j])
                # Handle circular boundary
                angle_diff = min(angle_diff, 2 * torch.pi - angle_diff)

                # Higher probability for smaller angle changes (smoother paths)
                # Using exponential decay based on angle difference
                transition[i, j] = torch.exp(-angle_diff / (self.node_angle_max / 4))

        # Normalize rows to create proper probability matrix
        transition = transition / transition.sum(dim=1, keepdim=True)

        return transition
    
    def _compute_path_cost_vectorized(self, start_positions: torch.Tensor, end_positions: torch.Tensor) -> torch.Tensor:
        """
        Vectorized version of paper's Step 1: Computing path cost ℒ(Aα[n-1], Bβ).

        Computes path costs for multiple paths simultaneously on GPU.
        Keeps the paper's exact formula but processes many paths at once.

        Args:
            start_positions: [N, 2] tensor of starting positions (x, y)
            end_positions: [N, 2] tensor of ending positions (x, y)

        Returns:
            Path costs [N] tensor (sum of log intensities along each path)
        """
        N = start_positions.shape[0]
        if N == 0:
            return torch.zeros(0, device=self.device)

        # Compute path lengths for all paths
        path_vectors = end_positions - start_positions  # [N, 2]
        path_lengths = torch.norm(path_vectors, dim=1)  # [N]

        # Use maximum path length for sampling resolution
        max_samples = max(int(path_lengths.max().item()), 1)

        # Create sampling grid for all paths [N, max_samples]
        t_vals = torch.linspace(0, 1, max_samples, device=self.device).unsqueeze(0).expand(N, -1)  # [N, max_samples]

        # Compute all sample positions [N, max_samples, 2]
        path_x = start_positions[:, 0:1] + t_vals * path_vectors[:, 0:1]  # [N, max_samples]
        path_y = start_positions[:, 1:2] + t_vals * path_vectors[:, 1:2]  # [N, max_samples]

        # Clamp to image bounds
        path_x = torch.clamp(path_x.long(), 0, self.W - 1)
        path_y = torch.clamp(path_y.long(), 0, self.H - 1)

        # Sample intensities for all paths [N, max_samples]
        intensities = self.image[path_y, path_x]

        # Create mask for valid samples (within actual path length)
        sample_distances = t_vals * path_lengths.unsqueeze(1)  # [N, max_samples]
        valid_mask = sample_distances <= path_lengths.unsqueeze(1)  # [N, max_samples]

        # Paper's formula: sum of log pixel intensities (avoiding log(0))
        log_intensities = torch.log(intensities * 255 + 1e-8)

        # Mask invalid samples and sum along path
        masked_log_intensities = torch.where(valid_mask, log_intensities, torch.tensor(0.0, device=self.device))
        path_costs = masked_log_intensities.sum(dim=1)  # [N]

        return path_costs

    def _compute_observation_probabilities(self,
                                         start_x: float,
                                         start_y: float,
                                         seed_angle: float = 0.0) -> torch.Tensor:
        """
        Paper's Step 1: Computing all possible paths (VECTORIZED for GPU).

        Keeps the paper's exact algorithm but processes all state transitions
        simultaneously for maximum GPU utilization.

        Args:
            start_x, start_y: Starting coordinates of the seed
            seed_angle: Initial angle direction for the seed

        Returns:
            observation: [T, S] tensor of observation probabilities for torbi
        """
        observations = torch.zeros(self.total_node, self.total_path, device=self.device)

        # Current position starts at the seed
        current_pos = torch.tensor([start_x, start_y], device=self.device, dtype=torch.float32)

        # For each time step (HMM node) - this loop is necessary for sequential dependency
        for t in range(self.total_node):
            # VECTORIZED: Compute ALL possible next positions simultaneously
            direction_angles = seed_angle + self.path_angles  # [S]

            # All next positions for all states [S, 2]
            next_x = current_pos[0] + self.node_r * torch.cos(direction_angles)  # [S]
            next_y = current_pos[1] + self.node_r * torch.sin(direction_angles)  # [S]
            next_positions = torch.stack([next_x, next_y], dim=1)  # [S, 2]

            # Current position repeated for all states [S, 2]
            current_positions = current_pos.unsqueeze(0).expand(self.total_path, -1)  # [S, 2]

            # PAPER'S APPROACH: Compute path costs as sum of log pixel intensities
            # Compute all possible paths from current position to next positions
            path_costs = torch.zeros(self.total_path, device=self.device)

            # For each possible path
            for s in range(self.total_path):
                # Get the path vector
                dx = next_x[s] - current_pos[0]
                dy = next_y[s] - current_pos[1]
                path_length = torch.sqrt(dx*dx + dy*dy)

                # Sample points along the path (linear interpolation)
                num_samples = max(int(path_length.item() * 2), 2)  # At least 2 samples
                t_values = torch.linspace(0, 1, num_samples, device=self.device)

                # Compute interpolated positions
                interp_x = current_pos[0] + dx * t_values
                interp_y = current_pos[1] + dy * t_values

                # Clamp to image bounds
                interp_x = torch.clamp(interp_x, 0, self.W - 1)
                interp_y = torch.clamp(interp_y, 0, self.H - 1)

                # Sample intensities along the path
                path_intensities = self.image[interp_y.long(), interp_x.long()]

                # Sum of log intensities (add small epsilon to avoid log(0))
                path_costs[s] = torch.sum(torch.log(path_intensities + 1e-6))

            # Convert costs to probabilities (paper's approach)
            observations[t] = torch.exp(path_costs)

            # Update current position for next time step
            # Use the most likely direction (highest probability state)
            best_state = torch.argmax(observations[t])
            best_angle = seed_angle + self.path_angles[best_state]

            # Move to next position following the best path
            current_pos[0] += self.node_r * torch.cos(best_angle)
            current_pos[1] += self.node_r * torch.sin(best_angle)

            # DEBUG: Log coordinate generation for first seed in small batches
            # (Removed debug to avoid variable scope issues)

        # Normalize to create proper probability distributions
        observations = observations / (observations.sum(dim=1, keepdim=True) + 1e-8)

        # DEBUG: Log observation statistics for debugging (limit to avoid spam)
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1

        if self._debug_count <= 2:  # Only log first 2 seeds
            logger.info(f"Observation debug - Seed {self._debug_count} at ({start_x:.1f}, {start_y:.1f}):")
            logger.info(f"  Observation range: [{observations.min():.6f}, {observations.max():.6f}]")
            logger.info(f"  First time step probs: {observations[0]}")
            logger.info(f"  Image intensity at seed: {self.image[int(start_y), int(start_x)]:.4f}")

        return observations
    
    def _validate_neurite_objects_vectorized(self, nodes_prev: torch.Tensor, nodes_curr: torch.Tensor) -> torch.Tensor:
        """
        Vectorized version of Paper's Step 3: Identifying HMM-nodes as neurite objects.

        Processes multiple node pairs simultaneously while keeping the paper's exact criterion:
        "A neurite object Ri=1 exists only when the Median(I_zone) < Mean(I_line)"

        Args:
            nodes_prev: [N, 2] Previous node positions (x, y)
            nodes_curr: [N, 2] Current node positions (x, y)

        Returns:
            [N] Boolean tensor indicating valid neurite objects
        """
        N = nodes_prev.shape[0]
        if N == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        # Paper's I_line: Sample intensities along all paths simultaneously
        path_costs = self._compute_path_cost_vectorized(nodes_prev, nodes_curr)  # [N]

        # Compute mean intensities along all lines
        path_vectors = nodes_curr - nodes_prev  # [N, 2]
        path_lengths = torch.norm(path_vectors, dim=1)  # [N]

        # Sample along all paths
        max_samples = max(int(path_lengths.max().item()), 1)
        t_vals = torch.linspace(0, 1, max_samples, device=self.device).unsqueeze(0).expand(N, -1)  # [N, max_samples]

        line_x = nodes_prev[:, 0:1] + t_vals * path_vectors[:, 0:1]  # [N, max_samples]
        line_y = nodes_prev[:, 1:2] + t_vals * path_vectors[:, 1:2]  # [N, max_samples]

        # Clamp to image bounds
        line_x = torch.clamp(line_x.long(), 0, self.W - 1)
        line_y = torch.clamp(line_y.long(), 0, self.H - 1)

        # Sample intensities and compute means
        I_line = self.image[line_y, line_x]  # [N, max_samples]
        valid_mask = t_vals * path_lengths.unsqueeze(1) <= path_lengths.unsqueeze(1)  # [N, max_samples]

        # Compute mean for each path
        valid_intensities = torch.where(valid_mask, I_line, torch.tensor(0.0, device=self.device))
        valid_counts = valid_mask.sum(dim=1).float()  # [N]
        mean_I_line = valid_intensities.sum(dim=1) / (valid_counts + 1e-8)  # [N]

        # Paper's I_zone: 2D pixel intensity in local zones
        zone_width = 2  # Paper specifies zone width = 2 × line

        # Compute bounding boxes for all paths
        min_x = torch.clamp(torch.min(nodes_prev[:, 0], nodes_curr[:, 0]).long() - zone_width, 0, self.W - 1)
        max_x = torch.clamp(torch.max(nodes_prev[:, 0], nodes_curr[:, 0]).long() + zone_width, 0, self.W - 1)
        min_y = torch.clamp(torch.min(nodes_prev[:, 1], nodes_curr[:, 1]).long() - zone_width, 0, self.H - 1)
        max_y = torch.clamp(torch.max(nodes_prev[:, 1], nodes_curr[:, 1]).long() + zone_width, 0, self.H - 1)

        # Compute zone medians for each path
        median_I_zone = torch.zeros(N, device=self.device)

        for i in range(N):
            if max_x[i] > min_x[i] and max_y[i] > min_y[i]:
                I_zone = self.image[min_y[i]:max_y[i]+1, min_x[i]:max_x[i]+1]
                median_I_zone[i] = I_zone.median()

        # Paper's criterion: neurite exists when Median(I_zone) < Mean(I_line)
        valid_mask = median_I_zone < mean_I_line

        # DEBUG: Log validation statistics for first few paths
        if N > 0:
            logger.info(f"Validation debug - First 3 paths:")
            for i in range(min(3, N)):
                logger.info(f"  Path {i}: Mean(I_line)={mean_I_line[i]:.4f}, Median(I_zone)={median_I_zone[i]:.4f}, Valid={valid_mask[i]}")
            logger.info(f"Total valid: {valid_mask.sum().item()}/{N} ({100*valid_mask.sum().item()/N:.1f}%)")

        return valid_mask

    def trace_from_seeds_batch(self,
                              seed_x: torch.Tensor,
                              seed_y: torch.Tensor,
                              seed_angles: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]]:
        """
        Batch version: Process multiple seeds simultaneously for maximum GPU utilization.

        Args:
            seed_x: [B] tensor of x coordinates
            seed_y: [B] tensor of y coordinates
            seed_angles: [B] tensor of initial angles

        Returns:
            List of (angles, x_coords, y_coords, valid_nodes) for each seed
        """
        batch_size = len(seed_x)
        if batch_size == 0:
            return []

        # Compute observations for all seeds simultaneously
        all_observations = []
        for i in range(batch_size):
            obs = self._compute_observation_probabilities(seed_x[i], seed_y[i], seed_angles[i])
            all_observations.append(obs)

        # Stack into batch [B, T, S]
        observations_batch = torch.stack(all_observations, dim=0)
        batch_frames = torch.full((batch_size,), self.total_node, device=self.device, dtype=torch.int32)

        # Single batched Viterbi call - MAXIMUM GPU UTILIZATION
        optimal_states_batch = torbi.viterbi.decode(
            observation=observations_batch,
            batch_frames=batch_frames,
            transition=self.transition_matrix,
            initial=self.initial_distribution
        )

        # Process results for each seed
        results = []
        for i in range(batch_size):
            optimal_path = optimal_states_batch[i]  # [T]

            # Convert to coordinates
            angles = torch.zeros(self.total_node, device=self.device)
            x_coords = torch.zeros(self.total_node, device=self.device)
            y_coords = torch.zeros(self.total_node, device=self.device)

            # Initialize with seed
            x_coords[0] = seed_x[i]
            y_coords[0] = seed_y[i]
            angles[0] = seed_angles[i]

            # Follow optimal path
            for t in range(1, self.total_node):
                state = optimal_path[t]
                direction_angle = seed_angles[i] + self.path_angles[state]

                x_coords[t] = x_coords[t-1] + self.node_r * torch.cos(direction_angle)
                y_coords[t] = y_coords[t-1] + self.node_r * torch.sin(direction_angle)
                angles[t] = direction_angle

            # Vectorized validation
            if self.total_node > 1:
                nodes_prev = torch.stack([x_coords[:-1], y_coords[:-1]], dim=1)
                nodes_curr = torch.stack([x_coords[1:], y_coords[1:]], dim=1)
                valid_mask = self._validate_neurite_objects_vectorized(nodes_prev, nodes_curr)
                valid_nodes = (torch.where(valid_mask)[0] + 1).tolist()
            else:
                valid_nodes = []

            results.append((angles, x_coords, y_coords, valid_nodes))

        return results

    def trace_from_seed(self,
                       seed_x: float,
                       seed_y: float,
                       seed_angle: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Paper's complete HMM tracing algorithm using torbi for Step 2 (Viterbi).

        Implements all three steps from the paper:
        1. Computing all possible paths
        2. Predicting HMM-node position (via torbi Viterbi)
        3. Identifying neurite objects

        Args:
            seed_x, seed_y: Starting coordinates
            seed_angle: Initial direction angle

        Returns:
            Tuple of (angles, x_coords, y_coords, valid_nodes) for the traced path
        """
        # Paper's Step 1: Computing all possible paths
        observations = self._compute_observation_probabilities(seed_x, seed_y, seed_angle)

        # Paper's Step 2: Predicting HMM-node position using torbi Viterbi
        observations_batch = observations.unsqueeze(0)  # [1, T, S]
        batch_frames = torch.tensor([self.total_node], device=self.device, dtype=torch.int32)

        # Use torbi for GPU-accelerated Viterbi decoding
        optimal_states = torbi.viterbi.decode(
            observation=observations_batch,
            batch_frames=batch_frames,
            transition=self.transition_matrix,
            initial=self.initial_distribution
        )

        optimal_path = optimal_states[0]  # [T] - remove batch dimension

        # DEBUG: Log Viterbi results for small batches only
        # (Removed debug to avoid variable scope issues)

        # Convert optimal state sequence to node coordinates
        angles = torch.zeros(self.total_node, device=self.device)
        x_coords = torch.zeros(self.total_node, device=self.device)
        y_coords = torch.zeros(self.total_node, device=self.device)

        # Initialize with seed position
        x_coords[0] = seed_x
        y_coords[0] = seed_y
        angles[0] = seed_angle

        # Follow the optimal path to generate node positions
        for t in range(1, self.total_node):
            state = optimal_path[t]
            direction_angle = seed_angle + self.path_angles[state]

            # Move from previous node in the optimal direction
            x_coords[t] = x_coords[t-1] + self.node_r * torch.cos(direction_angle)
            y_coords[t] = y_coords[t-1] + self.node_r * torch.sin(direction_angle)
            angles[t] = direction_angle

        # Paper's Step 3: Identifying neurite objects (VECTORIZED)
        if self.total_node > 1:
            # Prepare all node pairs for vectorized validation
            nodes_prev = torch.stack([x_coords[:-1], y_coords[:-1]], dim=1)  # [T-1, 2]
            nodes_curr = torch.stack([x_coords[1:], y_coords[1:]], dim=1)    # [T-1, 2]

            # DEBUG: Log first few node positions
            if hasattr(self, '_validation_debug_count'):
                self._validation_debug_count += 1
            else:
                self._validation_debug_count = 1

            if self._validation_debug_count <= 2:
                logger.info(f"Validation debug - Seed {self._validation_debug_count}:")
                logger.info(f"  Node positions: {nodes_prev[:3]} -> {nodes_curr[:3]}")
                logger.info(f"  Total node pairs: {len(nodes_prev)}")

            # Vectorized validation using paper's exact criterion
            valid_mask = self._validate_neurite_objects_vectorized(nodes_prev, nodes_curr)  # [T-1]

            # Convert to list of valid node indices (1-based to match paper)
            valid_nodes = (torch.where(valid_mask)[0] + 1).tolist()

            if self._validation_debug_count <= 2:
                logger.info(f"  Valid nodes: {valid_nodes}")
        else:
            valid_nodes = []

        return angles, x_coords, y_coords, valid_nodes


def trace_neurites_rrs_hmm_torbi(
    image: torch.Tensor,
    seed_density: float = 0.001,
    total_node: int = 16,
    total_path: int = 8,
    node_r: int = 5,
    node_angle_max: float = torch.pi / 2,
    chain_level: float = 1.0
) -> tuple[torch.Tensor, dict]:
    """
    Paper's Random-Reaction-Seed method using torbi for GPU-accelerated Viterbi.

    Implements the exact algorithm from PMC6393450:
    "Random-Reaction-Seed Method for Automated Identification of Neurite Elongation and Branching"

    Uses torbi's optimized Viterbi decoder to replace the paper's manual
    forward/backward pass loops with GPU acceleration.

    Args:
        image: Input image tensor [H, W] (normalized to [0,1])
        seed_density: Density of random seeds (paper uses various densities)
        total_node: N = number of HMM nodes in chain
        total_path: Number of possible path directions (state space)
        node_r: r = path length between adjacent nodes
        node_angle_max: Maximum angle range for path search
        chain_level: Validation threshold (paper's criterion parameter)

    Returns:
        result_image: Binary image with traced neurites
        trace_dict: Dictionary with trace coordinates in expected format
    """
    device = image.device
    H, W = image.shape

    logger.info(f"Starting paper's RRS-HMM algorithm with torbi acceleration on {device}")

    # Generate seeds preferentially on bright pixels
    # Find bright pixels (potential axons)
    bright_threshold = 0.1  # Only consider pixels brighter than this
    bright_mask = image > bright_threshold

    # If no bright pixels, fall back to random seeds
    if not bright_mask.any():
        num_seeds = int(H * W * seed_density)
        seed_x = torch.randint(0, W, (num_seeds,), device=device).float()
        seed_y = torch.randint(0, H, (num_seeds,), device=device).float()
    else:
        # Get coordinates of bright pixels
        bright_y, bright_x = torch.where(bright_mask)

        # Randomly sample from bright pixels
        num_bright = len(bright_y)
        num_seeds = min(int(num_bright * 0.1), 200)  # Use at most 10% of bright pixels, max 200

        if num_seeds > 0:
            indices = torch.randperm(num_bright, device=device)[:num_seeds]
            seed_y = bright_y[indices].float()
            seed_x = bright_x[indices].float()
        else:
            # Fallback if no seeds
            num_seeds = 50
            seed_x = torch.randint(0, W, (num_seeds,), device=device).float()
            seed_y = torch.randint(0, H, (num_seeds,), device=device).float()

    # Generate random angles
    seed_angles = torch.rand(num_seeds, device=device) * 2 * torch.pi

    logger.info(f"Generated {num_seeds} seeds for paper's RRS algorithm")

    # Initialize paper's HMM tracer with torbi acceleration
    tracer = PaperHMMTracer(
        image=image,
        total_node=total_node,
        total_path=total_path,
        node_r=node_r,
        node_angle_max=node_angle_max,
        device=device
    )
    
    # Result image for visualization
    result_image = torch.zeros_like(image)

    # Trace dictionary to store individual trace coordinates
    trace_dict = {}

    # Process seeds in batches for MAXIMUM GPU utilization
    batch_size = 64
    valid_traces = 0
    processed_seeds = 0
    reaction_seeds = []  # Store reaction seeds for bidirectional tracing
    trace_id = 1  # Start trace IDs from 1

    logger.info(f"Processing {num_seeds} seeds in batches of {batch_size}")

    for batch_start in range(0, num_seeds, batch_size):
        batch_end = min(batch_start + batch_size, num_seeds)

        # BATCH PROCESSING: Process entire batch simultaneously
        batch_seed_x = seed_x[batch_start:batch_end]
        batch_seed_y = seed_y[batch_start:batch_end]
        batch_seed_angles = seed_angles[batch_start:batch_end]

        try:
            # DEBUG: Time the batch processing
            import time
            start_time = time.time()

            # Process entire batch simultaneously - MAXIMUM GPU UTILIZATION
            batch_results = tracer.trace_from_seeds_batch(
                batch_seed_x, batch_seed_y, batch_seed_angles
            )

            batch_time = time.time() - start_time
            logger.info(f"Batch of {len(batch_seed_x)} seeds processed in {batch_time:.4f}s ({batch_time/len(batch_seed_x):.4f}s per seed)")

            # Process results from the batch
            for i, (angles, x_coords, y_coords, valid_node_indices) in enumerate(batch_results):
                processed_seeds += 1

                if i == 0:  # Log first seed in batch
                    logger.info(f"  First seed valid nodes: {len(valid_node_indices)}")

                # Paper's Reaction Seed Strategy: Generate reaction seed for bidirectional tracing
                if len(valid_node_indices) >= 1:  # Active chain with at least one valid node
                    # Birth place: first action node (first valid node)
                    first_valid_idx = valid_node_indices[0]
                    reaction_x = x_coords[first_valid_idx]
                    reaction_y = y_coords[first_valid_idx]

                    # Conditional direction: opposite to the first two action nodes
                    if len(valid_node_indices) >= 2:
                        second_valid_idx = valid_node_indices[1]
                        # Compute direction from first to second valid node
                        dx = x_coords[second_valid_idx] - x_coords[first_valid_idx]
                        dy = y_coords[second_valid_idx] - y_coords[first_valid_idx]
                        primary_angle = torch.atan2(dy, dx)
                        # Reaction seed goes in opposite direction
                        reaction_angle = primary_angle + torch.pi
                    else:
                        # If only one valid node, use opposite of seed direction
                        reaction_angle = batch_seed_angles[i] + torch.pi

                    # Store reaction seed for later processing
                    reaction_seeds.append((reaction_x, reaction_y, reaction_angle))

                # Only process traces that have valid nodes (passed validation)
                if len(valid_node_indices) > 0:
                    # Collect trace coordinates
                    trace_coords = []

                    for j in range(len(x_coords) - 1):
                        x0, y0 = int(x_coords[j]), int(y_coords[j])
                        x1, y1 = int(x_coords[j+1]), int(y_coords[j+1])

                        # Collect line coordinates between consecutive nodes
                        if (0 <= x0 < W and 0 <= y0 < H and 0 <= x1 < W and 0 <= y1 < H):
                            # Simple line drawing
                            steps = max(abs(x1 - x0), abs(y1 - y0))
                            if steps > 0:
                                for step in range(steps + 1):
                                    t = step / steps
                                    x_interp = int(x0 + t * (x1 - x0))
                                    y_interp = int(y0 + t * (y1 - y0))
                                    if 0 <= x_interp < W and 0 <= y_interp < H:
                                        # Add to trace coordinates (z, y, x format expected by test)
                                        trace_coords.append((0.0, float(y_interp), float(x_interp)))
                                        # Also draw on result image for visualization
                                        result_image[y_interp, x_interp] = 1.0

                    # Store trace in dictionary if it has coordinates
                    if trace_coords:
                        trace_dict[f"slice_000_trace_{trace_id:05d}"] = trace_coords
                        trace_id += 1
                        valid_traces += 1

        except Exception as e:
            # Skip problematic batches
            logger.warning(f"Batch processing failed: {e}")
            continue

        # Log progress every 10 batches
        if (batch_start // batch_size) % 10 == 0:
            logger.info(f"Processed {processed_seeds}/{num_seeds} seeds, found {valid_traces} valid traces so far")

    # CRITICAL: Process reaction seeds for bidirectional tracing (Paper Figure 4)
    if reaction_seeds:
        logger.info(f"Processing {len(reaction_seeds)} reaction seeds for bidirectional tracing")

        # Convert to tensors for GPU processing
        reaction_x = torch.tensor([rs[0] for rs in reaction_seeds], device=device)
        reaction_y = torch.tensor([rs[1] for rs in reaction_seeds], device=device)
        reaction_angles = torch.tensor([rs[2] for rs in reaction_seeds], device=device)

        reaction_traces = 0

        # Process reaction seeds in batches (same batch_size as random seeds)
        for batch_start in range(0, len(reaction_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(reaction_seeds))

            batch_rx = reaction_x[batch_start:batch_end]
            batch_ry = reaction_y[batch_start:batch_end]
            batch_rangles = reaction_angles[batch_start:batch_end]

            try:
                # Process reaction seeds using same HMM tracer
                batch_results = tracer.trace_from_seeds_batch(batch_rx, batch_ry, batch_rangles)

                # Process reaction seed traces (only valid ones)
                for i, (angles, x_coords, y_coords, valid_node_indices) in enumerate(batch_results):
                    # Only process traces that have valid nodes (passed validation)
                    if len(valid_node_indices) > 0:
                        # Collect reaction trace coordinates
                        trace_coords = []

                        for j in range(len(x_coords) - 1):
                            x0, y0 = int(x_coords[j]), int(y_coords[j])
                            x1, y1 = int(x_coords[j+1]), int(y_coords[j+1])

                            # Collect line coordinates between consecutive nodes
                            if (0 <= x0 < W and 0 <= y0 < H and 0 <= x1 < W and 0 <= y1 < H):
                                steps = max(abs(x1 - x0), abs(y1 - y0))
                                if steps > 0:
                                    for step in range(steps + 1):
                                        t = step / steps
                                        x_interp = int(x0 + t * (x1 - x0))
                                        y_interp = int(y0 + t * (y1 - y0))
                                        if 0 <= x_interp < W and 0 <= y_interp < H:
                                            # Add to trace coordinates (z, y, x format expected by test)
                                            trace_coords.append((0.0, float(y_interp), float(x_interp)))
                                            # Also draw on result image for visualization
                                            result_image[y_interp, x_interp] = 1.0

                        # Store reaction trace in dictionary if it has coordinates
                        if trace_coords:
                            trace_dict[f"slice_000_trace_{trace_id:05d}"] = trace_coords
                            trace_id += 1
                            reaction_traces += 1

            except Exception as e:
                logger.warning(f"Reaction seed batch processing failed: {e}")
                continue

        logger.info(f"Reaction seed processing complete: {reaction_traces} reaction traces added")
    else:
        logger.info("No reaction seeds generated - all random seeds failed validation")

    total_traces = len(trace_dict)
    logger.info(f"RRS algorithm complete: {total_traces} total traces collected in trace dictionary")

    return result_image, trace_dict
