"""
GPU-Accelerated Pattern Matching Module

This module provides GPU-accelerated pattern matching functionality for the
Semantic Matrix Analyzer. It implements the same interface as SMA's pattern
matching system, but uses GPU acceleration for improved performance.
"""

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

import torch
import numpy as np

from gpu_analysis.ast_tensor import GPUASTTensorizer

# Set up logging
logger = logging.getLogger(__name__)

# Define pattern types (should match SMA's pattern types)
class PatternType(Enum):
    STRING = "string"
    REGEX = "regex"
    AST = "ast"
    SEMANTIC = "semantic"

class PatternMatch:
    """
    Represents a pattern match.

    This class represents a match of a pattern in a file. It contains
    information about the pattern, the file, and the location of the match.
    """

    def __init__(self, pattern, file_path, start_pos, end_pos, match_text=None):
        """
        Initialize a pattern match.

        Args:
            pattern: Pattern that was matched
            file_path: Path to the file where the match was found
            start_pos: Start position of the match
            end_pos: End position of the match
            match_text: Text of the match (optional)
        """
        self.pattern = pattern
        self.file_path = file_path
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.match_text = match_text

    def __repr__(self):
        return f"PatternMatch(pattern={self.pattern}, file_path={self.file_path}, start_pos={self.start_pos}, end_pos={self.end_pos})"

class GPUPatternMatcher:
    """
    Base class for GPU-accelerated pattern matchers.

    This class provides the base functionality for GPU-accelerated pattern
    matchers. It implements the same interface as SMA's PatternMatcher class.

    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the GPU pattern matcher.

        Args:
            device: Device to place tensors on ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    def match_pattern(self, pattern, file_path, file_content, ast_node):
        """
        Match a pattern against a file using GPU acceleration.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            file_content: Content of the file
            ast_node: AST of the file

        Returns:
            List of PatternMatch objects
        """
        # Convert inputs to GPU tensors
        gpu_inputs = self._prepare_inputs(pattern, file_content, ast_node)

        # Dispatch to appropriate matcher based on pattern type
        if pattern.pattern_type == PatternType.STRING:
            matches = self._match_string_pattern(pattern, file_path, gpu_inputs)
        elif pattern.pattern_type == PatternType.REGEX:
            matches = self._match_regex_pattern(pattern, file_path, gpu_inputs)
        elif pattern.pattern_type == PatternType.AST:
            matches = self._match_ast_pattern(pattern, file_path, gpu_inputs)
        elif pattern.pattern_type == PatternType.SEMANTIC:
            matches = self._match_semantic_pattern(pattern, file_path, gpu_inputs)
        else:
            matches = []

        return matches

    def _prepare_inputs(self, pattern, file_content, ast_node):
        """
        Prepare inputs for GPU processing.

        Args:
            pattern: Pattern to match
            file_content: Content of the file
            ast_node: AST of the file

        Returns:
            Dictionary of GPU tensors
        """
        # Convert file content to tensor
        if isinstance(file_content, str):
            content_tensor = torch.tensor([ord(c) for c in file_content],
                                         dtype=torch.int32,
                                         device=self.device)
        else:
            content_tensor = file_content

        # Tensorize AST if not already tensorized
        if ast_node is not None and not isinstance(ast_node, dict):
            tensorizer = GPUASTTensorizer(device=self.device)
            ast_tensors = tensorizer.tensorize(ast_node)
        else:
            ast_tensors = ast_node

        return {
            "content_tensor": content_tensor,
            "ast_tensors": ast_tensors
        }

    def _match_string_pattern(self, pattern, file_path, gpu_inputs):
        """
        Match a string pattern using GPU acceleration.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            gpu_inputs: Dictionary of GPU tensors

        Returns:
            List of PatternMatch objects
        """
        # This method should be implemented by subclasses
        raise NotImplementedError

    def _match_regex_pattern(self, pattern, file_path, gpu_inputs):
        """
        Match a regex pattern using GPU acceleration.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            gpu_inputs: Dictionary of GPU tensors

        Returns:
            List of PatternMatch objects
        """
        # This method should be implemented by subclasses
        raise NotImplementedError

    def _match_ast_pattern(self, pattern, file_path, gpu_inputs):
        """
        Match an AST pattern using GPU acceleration.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            gpu_inputs: Dictionary of GPU tensors

        Returns:
            List of PatternMatch objects
        """
        # This method should be implemented by subclasses
        raise NotImplementedError

    def _match_semantic_pattern(self, pattern, file_path, gpu_inputs):
        """
        Match a semantic pattern using GPU acceleration.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            gpu_inputs: Dictionary of GPU tensors

        Returns:
            List of PatternMatch objects
        """
        # This method should be implemented by subclasses
        raise NotImplementedError

class GPUStringPatternMatcher(GPUPatternMatcher):
    """
    GPU-accelerated matcher for string patterns.

    This class provides GPU-accelerated matching for string patterns.
    It uses a parallel algorithm to find all occurrences of a string pattern
    in a file.
    """

    def _match_string_pattern(self, pattern, file_path, gpu_inputs):
        """
        Match a string pattern using GPU acceleration.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            gpu_inputs: Dictionary of GPU tensors

        Returns:
            List of PatternMatch objects
        """
        content_tensor = gpu_inputs["content_tensor"]
        string_pattern = pattern.pattern

        # Convert pattern to tensor
        pattern_tensor = torch.tensor([ord(c) for c in string_pattern],
                                     dtype=torch.int32,
                                     device=self.device)

        # Find all occurrences using parallel algorithm
        matches = self._find_string_matches(content_tensor, pattern_tensor)

        # Convert matches to PatternMatch objects
        return self._convert_to_pattern_matches(pattern, file_path, content_tensor, matches, string_pattern)

    def _find_string_matches(self, content_tensor, pattern_tensor):
        """
        Find all occurrences of pattern in content using parallel algorithm.

        Args:
            content_tensor: Tensor of content characters
            pattern_tensor: Tensor of pattern characters

        Returns:
            Tensor of match positions
        """
        return parallel_string_match(content_tensor, pattern_tensor)

    def _convert_to_pattern_matches(self, pattern, file_path, content_tensor, matches, pattern_str):
        """
        Convert tensor matches to PatternMatch objects.

        Args:
            pattern: Pattern that was matched
            file_path: Path to the file
            content_tensor: Tensor of content characters
            matches: Tensor of match positions
            pattern_str: String representation of the pattern

        Returns:
            List of PatternMatch objects
        """
        result = []

        # Convert tensor to CPU for processing
        matches_cpu = matches.cpu().numpy()

        for pos in matches_cpu:
            start_pos = int(pos)
            end_pos = start_pos + len(pattern_str)

            # Create PatternMatch object
            match = PatternMatch(
                pattern=pattern,
                file_path=file_path,
                start_pos=start_pos,
                end_pos=end_pos,
                match_text=pattern_str
            )

            result.append(match)

        return result

class GPURegexPatternMatcher(GPUPatternMatcher):
    """
    GPU-accelerated matcher for regex patterns.

    This class provides GPU-accelerated matching for regex patterns.
    It uses a parallel algorithm to find all occurrences of a regex pattern
    in a file.
    """

    def _match_regex_pattern(self, pattern, file_path, gpu_inputs):
        """
        Match a regex pattern using GPU acceleration.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            gpu_inputs: Dictionary of GPU tensors

        Returns:
            List of PatternMatch objects
        """
        content_tensor = gpu_inputs["content_tensor"]
        regex_pattern = pattern.pattern

        # For now, we'll use Python's re module for regex matching
        # In the future, we could implement a GPU-accelerated regex matcher

        # Convert content tensor to string
        content = "".join(chr(c) for c in content_tensor.cpu().numpy())

        # Find all matches
        matches = []
        for match in re.finditer(regex_pattern, content):
            start_pos = match.start()
            end_pos = match.end()
            match_text = match.group()

            # Create PatternMatch object
            pattern_match = PatternMatch(
                pattern=pattern,
                file_path=file_path,
                start_pos=start_pos,
                end_pos=end_pos,
                match_text=match_text
            )

            matches.append(pattern_match)

        return matches

class GPUASTPatternMatcher(GPUPatternMatcher):
    """
    GPU-accelerated matcher for AST patterns.

    This class provides GPU-accelerated matching for AST patterns.
    It uses a parallel algorithm to find all occurrences of an AST pattern
    in a file.
    """

    def _match_ast_pattern(self, pattern, file_path, gpu_inputs):
        """
        Match an AST pattern using GPU acceleration.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            gpu_inputs: Dictionary of GPU tensors

        Returns:
            List of PatternMatch objects
        """
        ast_tensors = gpu_inputs["ast_tensors"]
        if ast_tensors is None:
            return []

        node_type, condition = pattern.pattern

        # Find nodes of the specified type
        matching_nodes = self._find_nodes_by_type(ast_tensors, node_type)

        # Apply condition to matching nodes
        if condition:
            matching_nodes = self._apply_condition(ast_tensors, matching_nodes, condition)

        # Convert matches to PatternMatch objects
        return self._convert_to_pattern_matches(pattern, file_path, ast_tensors, matching_nodes)

    def _find_nodes_by_type(self, ast_tensors, node_type):
        """
        Find nodes of the specified type in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors
            node_type: Type of node to find

        Returns:
            Tensor of node indices
        """
        # Get node types tensor
        node_types = ast_tensors["node_types"]

        # Find nodes with matching type
        from gpu_analysis.ast_tensor import AST_NODE_TYPES
        type_id = AST_NODE_TYPES.get(node_type, 0)

        return torch.nonzero(node_types == type_id).squeeze(-1)

    def _apply_condition(self, ast_tensors, nodes, condition):
        """
        Apply condition to nodes in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors
            nodes: Tensor of node indices
            condition: Condition to apply

        Returns:
            Tensor of node indices that satisfy the condition
        """
        # For now, we'll implement a simplified version
        # In the future, we could implement a more sophisticated condition evaluator

        # Convert nodes to CPU for processing
        nodes_cpu = nodes.cpu().numpy()

        # Filter nodes based on condition
        result = []
        for node_idx in nodes_cpu:
            # Check if node satisfies condition
            if self._check_condition(ast_tensors, node_idx, condition):
                result.append(node_idx)

        # Convert back to tensor
        return torch.tensor(result, device=self.device)

    def _check_condition(self, ast_tensors, node_idx, condition):
        """
        Check if a node satisfies a condition.

        Args:
            ast_tensors: Dictionary of AST tensors
            node_idx: Index of the node
            condition: Condition to check

        Returns:
            True if the node satisfies the condition, False otherwise
        """
        # For now, we'll implement a simplified version
        # In the future, we could implement a more sophisticated condition evaluator

        # Always return True for now
        return True

    def _convert_to_pattern_matches(self, pattern, file_path, ast_tensors, matching_nodes):
        """
        Convert matching nodes to PatternMatch objects.

        Args:
            pattern: Pattern that was matched
            file_path: Path to the file
            ast_tensors: Dictionary of AST tensors
            matching_nodes: Tensor of matching node indices

        Returns:
            List of PatternMatch objects
        """
        result = []

        # Convert matching_nodes to CPU for processing
        matching_nodes_cpu = matching_nodes.cpu().numpy()

        for node_idx in matching_nodes_cpu:
            # Create PatternMatch object
            # For now, we'll use dummy start and end positions
            match = PatternMatch(
                pattern=pattern,
                file_path=file_path,
                start_pos=node_idx,
                end_pos=node_idx,
                match_text=f"AST node {node_idx}"
            )

            result.append(match)

        return result

def parallel_string_match(content_tensor, pattern_tensor):
    """
    Find all occurrences of pattern in content using a parallel algorithm.

    This implementation is based on the parallel string matching algorithm
    described in the Voetter paper.

    Args:
        content_tensor: Tensor of content characters
        pattern_tensor: Tensor of pattern characters

    Returns:
        Tensor of match positions
    """
    content_len = content_tensor.size(0)
    pattern_len = pattern_tensor.size(0)

    if pattern_len > content_len:
        return torch.tensor([], dtype=torch.int64, device=content_tensor.device)

    # Step 1: Compute a boolean mask for each character in the pattern
    masks = []
    for i in range(pattern_len):
        mask = (content_tensor == pattern_tensor[i])
        masks.append(mask)

    # Step 2: Compute a sliding window product of the masks
    result = torch.ones(content_len - pattern_len + 1, dtype=torch.bool,
                       device=content_tensor.device)

    for i in range(pattern_len):
        result = result & masks[i][i:i+content_len-pattern_len+1]

    # Step 3: Find the indices where the result is True
    matches = torch.nonzero(result).squeeze(-1)

    return matches

class GPUPatternMatcherRegistry:
    """
    Registry for GPU-accelerated pattern matchers.

    This class provides a registry for GPU-accelerated pattern matchers.
    It allows looking up the appropriate matcher for a given pattern type.

    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the GPU pattern matcher registry.

        Args:
            device: Device to place tensors on ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._matchers = {}

        # Register default matchers
        self.register_matcher(PatternType.STRING, GPUStringPatternMatcher)
        self.register_matcher(PatternType.REGEX, GPURegexPatternMatcher)
        self.register_matcher(PatternType.AST, GPUASTPatternMatcher)

    def register_matcher(self, pattern_type, matcher_class):
        """
        Register a matcher for a pattern type.

        Args:
            pattern_type: Type of pattern
            matcher_class: Class of matcher to register
        """
        self._matchers[pattern_type] = matcher_class

    def get_matcher(self, pattern_type):
        """
        Get the matcher for a pattern type.

        Args:
            pattern_type: Type of pattern

        Returns:
            Matcher for the pattern type, or None if not found
        """
        matcher_class = self._matchers.get(pattern_type)
        if matcher_class is None:
            return None

        # Initialize matcher
        return matcher_class(device=self.device)

    def match_patterns(self, patterns, file_path, file_content, ast_node):
        """
        Match multiple patterns against a file.

        Args:
            patterns: List of patterns to match
            file_path: Path to the file
            file_content: Content of the file
            ast_node: AST of the file

        Returns:
            List of PatternMatch objects
        """
        result = []

        # Group patterns by type
        pattern_groups = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(pattern)

        # Match each group of patterns
        for pattern_type, patterns in pattern_groups.items():
            matcher = self.get_matcher(pattern_type)
            if matcher is None:
                continue

            for pattern in patterns:
                matches = matcher.match_pattern(pattern, file_path, file_content, ast_node)
                result.extend(matches)

        return result
