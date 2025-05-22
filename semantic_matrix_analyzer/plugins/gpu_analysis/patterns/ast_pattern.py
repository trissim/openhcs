"""
GPU-Accelerated AST Pattern Matcher

This module provides a GPU-accelerated implementation of AST pattern matching
for code analysis. It uses PyTorch to accelerate the matching process and can
run on both CPU and GPU.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from pattern_base to avoid circular imports
from .pattern_base import Pattern, PatternMatch, PatternType

# Set up logging
logger = logging.getLogger(__name__)


class ASTPattern:
    """
    GPU-accelerated AST pattern matcher.

    This class provides methods for matching AST patterns against code using GPU acceleration.
    It uses a neural network to match patterns in the AST tensor representation.

    Attributes:
        device: Device to use for matching ("cuda" or "cpu")
        model: Neural network model for matching
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the AST pattern matcher.

        Args:
            device: Device to use for matching ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = ASTPatternModel().to(self.device)

    def match(
        self,
        pattern: Pattern,
        file_path: Path,
        file_content: str,
        ast_tensors: Dict[str, torch.Tensor]
    ) -> List[PatternMatch]:
        """
        Match an AST pattern against a file.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            file_content: Content of the file
            ast_tensors: AST tensors

        Returns:
            List of pattern matches
        """
        if pattern.pattern_type != PatternType.AST:
            logger.warning(f"Expected AST pattern, got {pattern.pattern_type}")
            return []

        # Check if tensors are empty (parsing error)
        if ast_tensors["nodes"].numel() == 0:
            return []

        # Extract pattern details
        node_type, condition = pattern.pattern

        # Find matching nodes
        matches = []

        # Use the model to find matching nodes
        match_scores = self.model(ast_tensors, node_type, condition)

        # Convert match scores to matches
        for i, score in enumerate(match_scores):
            if score > 0.5:  # Confidence threshold
                # Get node information
                node_id = i
                node_type_id = ast_tensors["nodes"][i].item()

                # Find source range
                source_range = self._get_source_range(file_content, node_id, ast_tensors)

                # Extract source code
                source_code = self._get_source_code(file_content, source_range)

                # Create match
                matches.append(PatternMatch(
                    pattern=pattern,
                    file_path=file_path,
                    source_range=source_range,
                    source_code=source_code,
                    confidence=score.item()
                ))

        return matches

    def match_batch(
        self,
        patterns: List[Pattern],
        file_path: Path,
        file_content: str,
        ast_tensors: Dict[str, torch.Tensor]
    ) -> List[PatternMatch]:
        """
        Match multiple AST patterns against a file.

        Args:
            patterns: Patterns to match
            file_path: Path to the file
            file_content: Content of the file
            ast_tensors: AST tensors

        Returns:
            List of pattern matches
        """
        matches = []

        # Check if tensors are empty (parsing error)
        if ast_tensors["nodes"].numel() == 0:
            return []

        # Match each pattern
        for pattern in patterns:
            pattern_matches = self.match(pattern, file_path, file_content, ast_tensors)
            matches.extend(pattern_matches)

        return matches

    def _get_source_range(
        self,
        file_content: str,
        node_id: int,
        ast_tensors: Dict[str, torch.Tensor]
    ) -> Tuple[int, int]:
        """
        Get the source range for a node.

        Args:
            file_content: Content of the file
            node_id: ID of the node
            ast_tensors: AST tensors

        Returns:
            Tuple of (start_line, end_line)
        """
        # This is an approximation since we don't have line numbers in the tensor representation
        # In a real implementation, we would need to store line numbers in the tensor

        # Find all edges where this node is the parent
        edges = ast_tensors["edges"]
        child_edges = edges[edges[:, 0] == node_id]

        # If no children, assume it's a single line
        if len(child_edges) == 0:
            return (1, 1)

        # Find the min and max line numbers of children
        min_line = 1
        max_line = 1

        # In a real implementation, we would use the actual line numbers
        # For now, we'll just return a range based on the node ID
        return (node_id % 10 + 1, node_id % 10 + 3)

    def _get_source_code(self, file_content: str, source_range: Tuple[int, int]) -> str:
        """
        Get the source code for a range.

        Args:
            file_content: Content of the file
            source_range: Range of lines

        Returns:
            Source code
        """
        start_line, end_line = source_range
        lines = file_content.splitlines()

        # Adjust for 0-based indexing
        start_idx = start_line - 1
        end_idx = end_line

        if start_idx < 0:
            start_idx = 0
        if end_idx > len(lines):
            end_idx = len(lines)

        return "\n".join(lines[start_idx:end_idx])


class ASTPatternModel(nn.Module):
    """
    Neural network model for matching AST patterns.

    This model takes AST tensors and a pattern specification and returns
    a tensor of match scores for each node in the AST.
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__()

        # Node embedding
        self.node_embedding = nn.Embedding(100, 64)  # 100 node types, 64-dim embedding

        # Feature embedding
        self.feature_embedding = nn.Linear(10, 64)

        # Graph convolution layers
        self.conv1 = nn.Linear(64, 64)
        self.conv2 = nn.Linear(64, 64)

        # Pattern matching layers
        self.pattern_embedding = nn.Embedding(100, 64)  # 100 node types, 64-dim embedding
        self.match_layer = nn.Linear(64, 1)

    def forward(
        self,
        ast_tensors: Dict[str, torch.Tensor],
        node_type: str,
        condition: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Match a pattern against AST tensors.

        Args:
            ast_tensors: AST tensors
            node_type: Type of node to match
            condition: Optional condition for the node

        Returns:
            Tensor of match scores for each node
        """
        # Extract tensors
        nodes = ast_tensors["nodes"]
        edges = ast_tensors["edges"]
        features = ast_tensors["features"]

        # Embed nodes
        node_embeds = self.node_embedding(nodes)

        # Embed features
        feature_embeds = self.feature_embedding(features)

        # Combine embeddings
        embeds = node_embeds + feature_embeds

        # Apply graph convolution
        for _ in range(2):
            # Aggregate messages from neighbors
            new_embeds = torch.zeros_like(embeds)
            for edge in edges:
                src, dst = edge
                new_embeds[dst] += embeds[src]

            # Update embeddings
            embeds = F.relu(self.conv1(embeds + new_embeds))

        # Convert node type to ID
        # In a real implementation, we would use a mapping from string to ID
        # For now, we'll just use a simple hash
        node_type_id = hash(node_type) % 100

        # Embed pattern
        pattern_embed = self.pattern_embedding(torch.tensor([node_type_id], device=embeds.device))

        # Compute match scores
        match_scores = torch.zeros(len(nodes), device=embeds.device)
        for i, embed in enumerate(embeds):
            # Compute similarity
            similarity = F.cosine_similarity(embed.unsqueeze(0), pattern_embed, dim=1)
            match_scores[i] = similarity

        # Apply condition
        if condition:
            # In a real implementation, we would apply the condition
            # For now, we'll just return the match scores
            pass

        return match_scores
