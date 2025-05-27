"""
AST Tensorizer Module

This module provides functionality to convert Python ASTs to tensor representations
that can be efficiently processed on GPU.

The ASTTensorizer class converts Python's Abstract Syntax Tree (AST) to a tensor
representation that can be processed by GPU-accelerated neural networks. This enables
faster analysis of code structure and semantics.

The GPUASTTensorizer class extends this functionality by using a parent pointer
representation instead of child pointers, which is more efficient for GPU processing
as described in the paper "Parallel Lexing, Parsing and Semantic Analysis on the GPU"
by R. F. Voetter.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

import torch
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Define AST node type mapping
# Each AST node type gets a unique integer ID for tensor representation
AST_NODE_TYPES = {
    ast.Module: 1,
    ast.FunctionDef: 2,
    ast.AsyncFunctionDef: 3,
    ast.ClassDef: 4,
    ast.Return: 5,
    ast.Delete: 6,
    ast.Assign: 7,
    ast.AugAssign: 8,
    ast.AnnAssign: 9,
    ast.For: 10,
    ast.AsyncFor: 11,
    ast.While: 12,
    ast.If: 13,
    ast.With: 14,
    ast.AsyncWith: 15,
    ast.Raise: 16,
    ast.Try: 17,
    ast.Assert: 18,
    ast.Import: 19,
    ast.ImportFrom: 20,
    ast.Global: 21,
    ast.Nonlocal: 22,
    ast.Expr: 23,
    ast.Pass: 24,
    ast.Break: 25,
    ast.Continue: 26,
    # Expressions
    ast.BoolOp: 27,
    ast.BinOp: 28,
    ast.UnaryOp: 29,
    ast.Lambda: 30,
    ast.IfExp: 31,
    ast.Dict: 32,
    ast.Set: 33,
    ast.ListComp: 34,
    ast.SetComp: 35,
    ast.DictComp: 36,
    ast.GeneratorExp: 37,
    ast.Await: 38,
    ast.Yield: 39,
    ast.YieldFrom: 40,
    ast.Compare: 41,
    ast.Call: 42,
    ast.FormattedValue: 43,
    ast.JoinedStr: 44,
    ast.Constant: 45,
    ast.Attribute: 46,
    ast.Subscript: 47,
    ast.Starred: 48,
    ast.Name: 49,
    ast.List: 50,
    ast.Tuple: 51,
    ast.Slice: 52,
}

# Reverse mapping for debugging and visualization
AST_NODE_TYPES_REV = {v: k.__name__ for k, v in AST_NODE_TYPES.items()}

class ASTTensorizer:
    """
    Converts Python AST to tensor representation for GPU processing.

    This class provides methods to convert Python's Abstract Syntax Tree (AST)
    to a tensor representation that can be efficiently processed on GPU.

    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
        config: Optional configuration for the tensorizer
    """

    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AST tensorizer.

        Args:
            device: Device to place tensors on ("cuda" or "cpu")
            config: Optional configuration dictionary
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}

        # Extract configuration values
        self.feature_dim = self.config.get("feature_dim", 10)
        self.max_nodes = self.config.get("max_nodes", 10000)

    def tensorize(self, code: str) -> Dict[str, torch.Tensor]:
        """
        Convert Python code to tensor representation.

        Args:
            code: Python code to convert

        Returns:
            Dictionary of tensors representing the AST:
            - nodes: Tensor of node type IDs
            - edges: Tensor of edge indices (parent, child)
            - features: Tensor of node features
        """
        try:
            # Parse code to AST
            tree = ast.parse(code)

            # Extract node features
            nodes, edges, node_features = self._extract_graph(tree)

            # Convert to tensors
            node_tensor = torch.tensor(nodes, dtype=torch.int32, device=self.device)
            edge_tensor = torch.tensor(edges, dtype=torch.int32, device=self.device)
            feature_tensor = torch.tensor(node_features, dtype=torch.float32, device=self.device)

            return {
                "nodes": node_tensor,
                "edges": edge_tensor,
                "features": feature_tensor
            }
        except Exception as e:
            logger.error(f"Error tensorizing AST: {e}")
            # Return empty tensors on error
            return {
                "nodes": torch.tensor([], dtype=torch.int32, device=self.device),
                "edges": torch.tensor([], dtype=torch.int32, device=self.device),
                "features": torch.tensor([], dtype=torch.float32, device=self.device)
            }

    def tensorize_file(self, file_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Convert Python file to tensor representation.

        Args:
            file_path: Path to Python file

        Returns:
            Dictionary of tensors representing the AST
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        return self.tensorize(code)

    def _extract_graph(self, tree: ast.AST) -> Tuple[List[int], List[Tuple[int, int]], List[List[float]]]:
        """
        Extract graph representation from AST.

        Args:
            tree: AST to extract graph from

        Returns:
            Tuple of (nodes, edges, node_features):
            - nodes: List of node type IDs
            - edges: List of (parent_id, child_id) pairs
            - node_features: List of feature vectors for each node
        """
        nodes = []  # Node IDs
        edges = []  # (parent_id, child_id) pairs
        node_features = []  # Feature vectors for each node

        # Assign unique IDs to nodes
        node_ids = {}
        next_id = 0

        # First pass: assign IDs to all nodes
        for node in ast.walk(tree):
            node_ids[node] = next_id
            next_id += 1

            # Get node type ID
            node_type = type(node)
            type_id = AST_NODE_TYPES.get(node_type, 0)
            nodes.append(type_id)

            # Extract features
            features = self._extract_node_features(node)
            node_features.append(features)

        # Second pass: create edges
        for node in ast.walk(tree):
            parent_id = node_ids[node]

            for child_name, child in ast.iter_fields(node):
                if isinstance(child, ast.AST):
                    child_id = node_ids[child]
                    edges.append((parent_id, child_id))
                elif isinstance(child, list):
                    for grandchild in child:
                        if isinstance(grandchild, ast.AST):
                            child_id = node_ids[grandchild]
                            edges.append((parent_id, child_id))

        return nodes, edges, node_features

    def _extract_node_features(self, node: ast.AST) -> List[float]:
        """
        Extract feature vector for an AST node.

        Args:
            node: AST node to extract features from

        Returns:
            Feature vector (list of floats)
        """
        # Initialize with zeros
        features = [0.0] * self.feature_dim

        # Feature 1: Complexity (number of children)
        child_count = sum(1 for _ in ast.iter_child_nodes(node))
        features[0] = float(child_count)

        # Feature 2: Depth in the tree (approximated by node ID)
        # This will be normalized later
        features[1] = 1.0

        # Feature 3: Line number (normalized later)
        if hasattr(node, 'lineno'):
            features[2] = float(node.lineno)

        # Feature 4: Is it a control flow node?
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
            features[3] = 1.0

        # Feature 5: Is it a function/class definition?
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            features[4] = 1.0

        # Feature 6: Is it an assignment?
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            features[5] = 1.0

        # Feature 7: Is it a call?
        if isinstance(node, ast.Call):
            features[6] = 1.0

        # Feature 8: Is it an import?
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            features[7] = 1.0

        # Feature 9: Is it a name/attribute?
        if isinstance(node, (ast.Name, ast.Attribute)):
            features[8] = 1.0

        # Feature 10: Is it a constant?
        if isinstance(node, ast.Constant):
            features[9] = 1.0

        return features


class GPUASTTensorizer:
    """
    GPU-optimized AST tensorizer using parent pointers.

    This class provides methods to convert Python's Abstract Syntax Tree (AST)
    to a GPU-friendly tensor representation using parent pointers instead of
    child pointers, as described in the paper "Parallel Lexing, Parsing and
    Semantic Analysis on the GPU" by R. F. Voetter.

    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
        config: Optional configuration for the tensorizer
    """

    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GPU AST tensorizer.

        Args:
            device: Device to place tensors on ("cuda" or "cpu")
            config: Optional configuration dictionary
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}

        # Extract configuration values
        self.feature_dim = self.config.get("feature_dim", 10)
        self.max_nodes = self.config.get("max_nodes", 10000)

    def tensorize(self, code_or_ast: Union[str, ast.AST]) -> Dict[str, torch.Tensor]:
        """
        Convert Python code or AST to GPU-friendly tensor representation.

        Args:
            code_or_ast: Python code string or AST to convert

        Returns:
            Dictionary of tensors representing the AST:
            - nodes: Tensor of node type IDs
            - parents: Tensor of parent indices (-1 for root)
            - features: Tensor of node features
            - node_types: Tensor of node type IDs as strings (for debugging)
            - depths: Tensor of node depths in the tree
            - siblings: Tensor of sibling indices
        """
        try:
            # Parse code to AST if needed
            tree = code_or_ast if isinstance(code_or_ast, ast.AST) else ast.parse(code_or_ast)

            # Extract node features with parent pointers
            nodes, parents, node_features, node_types, field_names = self._extract_graph_with_parents(tree)

            # Convert to tensors
            node_tensor = torch.tensor(nodes, dtype=torch.int32, device=self.device)
            parent_tensor = torch.tensor(parents, dtype=torch.int32, device=self.device)
            feature_tensor = torch.tensor(node_features, dtype=torch.float32, device=self.device)
            node_type_tensor = torch.tensor(node_types, dtype=torch.int32, device=self.device)

            # Compute additional tensors
            depths = self._compute_depths(parent_tensor)
            siblings = self._compute_siblings(parent_tensor)

            return {
                "nodes": node_tensor,
                "parents": parent_tensor,
                "features": feature_tensor,
                "node_types": node_type_tensor,
                "depths": depths,
                "siblings": siblings,
                "field_names": field_names  # This is a list, not a tensor
            }
        except Exception as e:
            logger.error(f"Error tensorizing AST with GPU format: {e}")
            # Return empty tensors on error
            return {
                "nodes": torch.tensor([], dtype=torch.int32, device=self.device),
                "parents": torch.tensor([], dtype=torch.int32, device=self.device),
                "features": torch.tensor([], dtype=torch.float32, device=self.device),
                "node_types": torch.tensor([], dtype=torch.int32, device=self.device),
                "depths": torch.tensor([], dtype=torch.int32, device=self.device),
                "siblings": torch.tensor([], dtype=torch.int32, device=self.device),
                "field_names": []
            }

    def tensorize_file(self, file_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Convert Python file to GPU-friendly tensor representation.

        Args:
            file_path: Path to Python file

        Returns:
            Dictionary of tensors representing the AST
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        return self.tensorize(code)

    def _extract_graph_with_parents(self, tree: ast.AST) -> Tuple[List[int], List[int], List[List[float]], List[int], List[str]]:
        """
        Extract graph representation from AST using parent pointers.

        Args:
            tree: AST to extract graph from

        Returns:
            Tuple of (nodes, parents, node_features, node_types, field_names):
            - nodes: List of node IDs
            - parents: List of parent indices (-1 for root)
            - node_features: List of feature vectors for each node
            - node_types: List of node type IDs
            - field_names: List of field names for each node
        """
        nodes = []  # Node IDs
        parents = []  # Parent indices
        node_features = []  # Feature vectors for each node
        node_types = []  # Node type IDs
        field_names = []  # Field names

        # Assign unique IDs to nodes
        node_ids = {}
        next_id = 0

        # First pass: assign IDs to all nodes
        for node in ast.walk(tree):
            node_ids[node] = next_id
            next_id += 1

            # Get node type ID
            node_type = type(node)
            type_id = AST_NODE_TYPES.get(node_type, 0)
            nodes.append(next_id - 1)  # Node ID is its index
            node_types.append(type_id)

            # Extract features
            features = self._extract_node_features(node)
            node_features.append(features)

            # Initialize parent to -1 (will be set in second pass)
            parents.append(-1)
            field_names.append("")

        # Second pass: set parent indices
        for node in ast.walk(tree):
            node_id = node_ids[node]

            for child_name, child in ast.iter_fields(node):
                if isinstance(child, ast.AST):
                    child_id = node_ids[child]
                    parents[child_id] = node_id
                    field_names[child_id] = child_name
                elif isinstance(child, list):
                    for i, grandchild in enumerate(child):
                        if isinstance(grandchild, ast.AST):
                            child_id = node_ids[grandchild]
                            parents[child_id] = node_id
                            field_names[child_id] = f"{child_name}[{i}]"

        return nodes, parents, node_features, node_types, field_names

    def _extract_node_features(self, node: ast.AST) -> List[float]:
        """
        Extract feature vector for an AST node.

        Args:
            node: AST node to extract features from

        Returns:
            Feature vector (list of floats)
        """
        # Initialize with zeros
        features = [0.0] * self.feature_dim

        # Feature 1: Complexity (number of children)
        child_count = sum(1 for _ in ast.iter_child_nodes(node))
        features[0] = float(child_count)

        # Feature 2: Depth in the tree (approximated by node ID)
        # This will be normalized later
        features[1] = 1.0

        # Feature 3: Line number (normalized later)
        if hasattr(node, 'lineno'):
            features[2] = float(node.lineno)

        # Feature 4: Is it a control flow node?
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
            features[3] = 1.0

        # Feature 5: Is it a function/class definition?
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            features[4] = 1.0

        # Feature 6: Is it an assignment?
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            features[5] = 1.0

        # Feature 7: Is it a call?
        if isinstance(node, ast.Call):
            features[6] = 1.0

        # Feature 8: Is it an import?
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            features[7] = 1.0

        # Feature 9: Is it a name/attribute?
        if isinstance(node, (ast.Name, ast.Attribute)):
            features[8] = 1.0

        # Feature 10: Is it a constant?
        if isinstance(node, ast.Constant):
            features[9] = 1.0

        return features

    def _compute_depths(self, parents_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute node depths in the tree using parent pointers.

        This uses a parallel algorithm based on pointer jumping to compute
        depths in logarithmic time.

        Args:
            parents_tensor: Tensor of parent indices (-1 for root)

        Returns:
            Tensor of node depths
        """
        # Initialize depths to 0
        n = parents_tensor.size(0)
        depths = torch.zeros(n, dtype=torch.int32, device=self.device)

        # Create a mask for non-root nodes
        non_root_mask = (parents_tensor != -1)

        # Compute depths using pointer jumping
        # This is O(log n) instead of O(n) for a recursive approach
        changed = True
        while changed:
            # Get the depths of parents
            parent_depths = torch.zeros_like(depths)
            parent_depths[non_root_mask] = depths[parents_tensor[non_root_mask]]

            # Add 1 to get the depths of children
            new_depths = parent_depths + 1
            new_depths[~non_root_mask] = 0  # Root nodes have depth 0

            # Check if depths have changed
            changed = not torch.all(depths == new_depths)
            depths = new_depths

        return depths

    def _compute_siblings(self, parents_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute sibling indices for each node.

        Args:
            parents_tensor: Tensor of parent indices (-1 for root)

        Returns:
            Tensor of sibling indices (-1 for no siblings)
        """
        n = parents_tensor.size(0)
        siblings = torch.full((n,), -1, dtype=torch.int32, device=self.device)

        # Group nodes by parent
        for i in range(n):
            parent = parents_tensor[i].item()
            if parent == -1:
                continue

            # Find other nodes with the same parent
            same_parent_mask = (parents_tensor == parent)
            same_parent_indices = torch.nonzero(same_parent_mask).squeeze(-1)

            # Exclude self
            same_parent_indices = same_parent_indices[same_parent_indices != i]

            # If there are siblings, set the first one
            if same_parent_indices.size(0) > 0:
                siblings[i] = same_parent_indices[0]

        return siblings

    def find_root_nodes(self, parents_tensor: torch.Tensor) -> torch.Tensor:
        """
        Find root nodes in the tree (nodes with parent = -1).

        Args:
            parents_tensor: Tensor of parent indices (-1 for root)

        Returns:
            Tensor of root node indices
        """
        return torch.nonzero(parents_tensor == -1).squeeze(-1)

    def find_leaf_nodes(self, ast_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Find leaf nodes in the tree (nodes with no children).

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Tensor of leaf node indices
        """
        parents = ast_tensors["parents"]
        n = parents.size(0)

        # A node is a leaf if it's not a parent of any other node
        is_parent = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Mark all parent indices
        valid_parents = parents[parents != -1]
        is_parent[valid_parents] = True

        # Nodes that are not parents are leaves
        return torch.nonzero(~is_parent).squeeze(-1)

    def find_leftmost_descendants(self, parents_tensor: torch.Tensor) -> torch.Tensor:
        """
        Find leftmost descendants for each node.

        Args:
            parents_tensor: Tensor of parent indices (-1 for root)

        Returns:
            Tensor mapping each node to its leftmost descendant
        """
        n = parents_tensor.size(0)
        leftmost = torch.arange(n, dtype=torch.int32, device=self.device)

        # Initialize with self (leaf nodes are their own leftmost descendants)
        leaf_nodes = self.find_leaf_nodes({"parents": parents_tensor})

        # Propagate leftmost descendants up the tree
        # This is done by iteratively updating the leftmost descendant of each node
        # based on the leftmost descendants of its children
        changed = True
        while changed:
            new_leftmost = leftmost.clone()

            # For each node, check if any of its children have a smaller leftmost descendant
            for i in range(n):
                # Find children of this node
                children = torch.nonzero(parents_tensor == i).squeeze(-1)

                if children.size(0) > 0:
                    # Get the leftmost descendants of all children
                    child_leftmost = leftmost[children]

                    # Update if any child has a smaller leftmost descendant
                    new_leftmost[i] = torch.min(child_leftmost)

            # Check if leftmost descendants have changed
            changed = not torch.all(leftmost == new_leftmost)
            leftmost = new_leftmost

        return leftmost

    def find_rightmost_descendants(self, parents_tensor: torch.Tensor) -> torch.Tensor:
        """
        Find rightmost descendants for each node.

        Args:
            parents_tensor: Tensor of parent indices (-1 for root)

        Returns:
            Tensor mapping each node to its rightmost descendant
        """
        n = parents_tensor.size(0)
        rightmost = torch.arange(n, dtype=torch.int32, device=self.device)

        # Initialize with self (leaf nodes are their own rightmost descendants)
        leaf_nodes = self.find_leaf_nodes({"parents": parents_tensor})

        # Propagate rightmost descendants up the tree
        changed = True
        while changed:
            new_rightmost = rightmost.clone()

            # For each node, check if any of its children have a larger rightmost descendant
            for i in range(n):
                # Find children of this node
                children = torch.nonzero(parents_tensor == i).squeeze(-1)

                if children.size(0) > 0:
                    # Get the rightmost descendants of all children
                    child_rightmost = rightmost[children]

                    # Update if any child has a larger rightmost descendant
                    new_rightmost[i] = torch.max(child_rightmost)

            # Check if rightmost descendants have changed
            changed = not torch.all(rightmost == new_rightmost)
            rightmost = new_rightmost

        return rightmost

    def tree_compactification(self, ast_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compact the tree representation to reduce memory usage.

        This removes unnecessary nodes and reindexes the remaining nodes to
        ensure contiguous indices.

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Dictionary of compacted AST tensors
        """
        nodes = ast_tensors["nodes"]
        parents = ast_tensors["parents"]
        features = ast_tensors["features"]
        node_types = ast_tensors["node_types"]

        n = nodes.size(0)

        # Find nodes to keep (non-trivial nodes)
        # A node is trivial if it has exactly one child and no other properties
        to_keep = torch.ones(n, dtype=torch.bool, device=self.device)

        for i in range(n):
            # Find children of this node
            children = torch.nonzero(parents == i).squeeze(-1)

            # If node has exactly one child and is not a root
            if children.size(0) == 1 and parents[i] != -1:
                # Check if node has no special properties
                # For simplicity, we just check if all features are zero
                if torch.all(features[i] == 0):
                    to_keep[i] = False

        # Create a mapping from old indices to new indices
        new_indices = torch.cumsum(to_keep.int(), dim=0) - 1

        # Filter tensors to keep only non-trivial nodes
        new_nodes = nodes[to_keep]
        new_features = features[to_keep]
        new_node_types = node_types[to_keep]

        # Update parent indices
        new_parents = torch.full_like(parents[to_keep], -1)
        for i in range(n):
            if to_keep[i]:
                parent = parents[i].item()
                if parent != -1 and to_keep[parent]:
                    new_parents[new_indices[i]] = new_indices[parent]
                elif parent != -1:
                    # If parent was removed, find the nearest non-trivial ancestor
                    while parent != -1 and not to_keep[parent]:
                        parent = parents[parent].item()
                    if parent != -1:
                        new_parents[new_indices[i]] = new_indices[parent]

        # Create result dictionary with only the necessary tensors
        result = {
            "nodes": new_nodes,
            "parents": new_parents,
            "features": new_features,
            "node_types": new_node_types
        }

        # Add other tensors from the original dictionary if they exist
        for key, value in ast_tensors.items():
            if key not in result and key != "field_names":
                # Skip field_names as it's a list, not a tensor
                if isinstance(value, torch.Tensor):
                    if value.size(0) == n:
                        # If tensor has the same size as nodes, filter it
                        result[key] = value[to_keep]
                    else:
                        # Otherwise, keep it as is
                        result[key] = value

        # Handle field_names separately
        if "field_names" in ast_tensors:
            field_names = ast_tensors["field_names"]
            new_field_names = [field_names[i] for i in range(n) if to_keep[i]]
            result["field_names"] = new_field_names

        return result

    def to_dot(self, ast_tensors: Dict[str, torch.Tensor], max_nodes: int = 100) -> str:
        """
        Convert AST tensors to DOT format for visualization.

        Args:
            ast_tensors: Dictionary of AST tensors
            max_nodes: Maximum number of nodes to include in the visualization

        Returns:
            DOT representation of the AST
        """
        nodes = ast_tensors["nodes"]
        parents = ast_tensors["parents"]
        node_types = ast_tensors["node_types"]
        field_names = ast_tensors.get("field_names", [""] * nodes.size(0))

        n = min(nodes.size(0), max_nodes)

        # Create DOT representation
        dot = ["digraph AST {"]
        dot.append("  node [shape=box];")

        # Add nodes
        for i in range(n):
            node_type = node_types[i].item()
            node_type_name = AST_NODE_TYPES_REV.get(node_type, f"Unknown_{node_type}")
            dot.append(f'  {i} [label="{i}: {node_type_name}"];')

        # Add edges
        for i in range(n):
            parent = parents[i].item()
            if parent != -1 and parent < n:
                field_name = field_names[i] if i < len(field_names) else ""
                label = f' [label="{field_name}"]' if field_name else ""
                dot.append(f"  {parent} -> {i}{label};")

        dot.append("}")
        return "\n".join(dot)

    def get_node_by_type(self, ast_tensors: Dict[str, torch.Tensor], node_type: type) -> torch.Tensor:
        """
        Find nodes of a specific type in the AST.

        Args:
            ast_tensors: Dictionary of AST tensors
            node_type: AST node type to find

        Returns:
            Tensor of indices for nodes of the specified type
        """
        node_types = ast_tensors["node_types"]
        type_id = AST_NODE_TYPES.get(node_type, 0)

        return torch.nonzero(node_types == type_id).squeeze(-1)

    def get_type_name(self, type_id: int) -> str:
        """
        Get the name of a node type from its ID.

        Args:
            type_id: Node type ID

        Returns:
            Name of the node type
        """
        return AST_NODE_TYPES_REV.get(type_id, f"Unknown_{type_id}")

    def get_name(self, name_idx: int, ast_tensors: Optional[Dict[str, torch.Tensor]] = None) -> Optional[str]:
        """
        Get the name of a node from its name index.

        Args:
            name_idx: Name index
            ast_tensors: Optional dictionary of AST tensors containing string table

        Returns:
            Name of the node, or None if not available
        """
        if name_idx == -1:
            return None

        if ast_tensors is not None and "string_table" in ast_tensors:
            string_table = ast_tensors["string_table"]
            if name_idx < len(string_table):
                return string_table[name_idx]

        return None

    def get_children_indices(self, ast_tensors: Dict[str, torch.Tensor], node_idx: int) -> List[int]:
        """
        Get the indices of a node's children.

        Args:
            ast_tensors: Dictionary of AST tensors
            node_idx: Index of the node

        Returns:
            List of child indices
        """
        # Check if node_children exists
        if "node_children" in ast_tensors:
            node_children = ast_tensors["node_children"]

            # If node_children is a list of lists
            if isinstance(node_children, list) and node_idx < len(node_children):
                return node_children[node_idx]

        # Find children based on parent-child relationship
        if "parents" in ast_tensors:
            parents = ast_tensors["parents"]
            # Find indices where parent is node_idx
            children = torch.nonzero(parents == node_idx).squeeze(-1)
            return children.tolist() if children.numel() > 0 else []

        return []

    def extract_subtree(self, ast_tensors: Dict[str, torch.Tensor], node_idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract a subtree from the AST.

        Args:
            ast_tensors: Dictionary of AST tensors
            node_idx: Index of the root node of the subtree

        Returns:
            Dictionary of tensors representing the subtree
        """
        # Create a new dictionary for the subtree
        subtree = {}

        # Copy tensors that don't depend on node indices
        for key, value in ast_tensors.items():
            if key not in ["nodes", "parents", "node_types", "node_names", "node_children"]:
                subtree[key] = value

        # Extract the subtree nodes
        if "nodes" in ast_tensors and isinstance(ast_tensors["nodes"], torch.Tensor):
            subtree["nodes"] = ast_tensors["nodes"][node_idx:node_idx+1]

        # Extract the subtree node types
        if "node_types" in ast_tensors and isinstance(ast_tensors["node_types"], torch.Tensor):
            subtree["node_types"] = ast_tensors["node_types"][node_idx:node_idx+1]

        # Extract the subtree node names
        if "node_names" in ast_tensors and isinstance(ast_tensors["node_names"], torch.Tensor):
            subtree["node_names"] = ast_tensors["node_names"][node_idx:node_idx+1]

        # Set the parent to -1 (root)
        if "parents" in ast_tensors:
            subtree["parents"] = torch.tensor([-1], device=ast_tensors["parents"].device)

        # Extract children recursively
        children_indices = self.get_children_indices(ast_tensors, node_idx)
        if children_indices:
            subtree["node_children"] = [list(range(1, len(children_indices) + 1))]

            # Add children to the subtree
            for i, child_idx in enumerate(children_indices):
                child_subtree = self.extract_subtree(ast_tensors, child_idx)

                # Append child tensors
                for key, value in child_subtree.items():
                    if key in ["nodes", "node_types", "node_names"] and key in subtree:
                        subtree[key] = torch.cat([subtree[key], value])

                # Update parents
                if "parents" in child_subtree and "parents" in subtree:
                    # Adjust parent indices
                    child_parents = child_subtree["parents"] + len(subtree["parents"])
                    # Set the parent of the root of the child subtree to node_idx
                    child_parents[0] = 0
                    subtree["parents"] = torch.cat([subtree["parents"], child_parents[1:]])

        return subtree

    def get_node_attributes(self, ast_tensors: Dict[str, torch.Tensor], node_idx: int) -> Dict[str, Any]:
        """
        Get the attributes of a node.

        Args:
            ast_tensors: Dictionary of AST tensors
            node_idx: Index of the node

        Returns:
            Dictionary of node attributes
        """
        attributes = {}

        # Extract type
        if "node_types" in ast_tensors and node_idx < len(ast_tensors["node_types"]):
            type_idx = ast_tensors["node_types"][node_idx].item() if isinstance(ast_tensors["node_types"], torch.Tensor) else ast_tensors["node_types"][node_idx]
            attributes["type"] = self.get_type_name(type_idx)

        # Extract name
        if "node_names" in ast_tensors and node_idx < len(ast_tensors["node_names"]):
            name_idx = ast_tensors["node_names"][node_idx].item() if isinstance(ast_tensors["node_names"], torch.Tensor) else ast_tensors["node_names"][node_idx]
            name = self.get_name(name_idx, ast_tensors)
            if name is not None:
                attributes["name"] = name

        # Extract source range
        if "node_line_ranges" in ast_tensors:
            line_ranges = ast_tensors["node_line_ranges"]
            if isinstance(line_ranges, torch.Tensor) and line_ranges.shape[0] > node_idx:
                start_line = line_ranges[node_idx, 0].item()
                end_line = line_ranges[node_idx, 1].item()
                attributes["lineno"] = start_line
                attributes["end_lineno"] = end_line
            elif isinstance(line_ranges, list) and node_idx < len(line_ranges):
                start_line, end_line = line_ranges[node_idx]
                attributes["lineno"] = start_line
                attributes["end_lineno"] = end_line

        # Extract children
        children_indices = self.get_children_indices(ast_tensors, node_idx)
        if children_indices:
            attributes["children"] = children_indices

        return attributes

    def create_empty_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Create empty tensors for error cases.

        Returns:
            Dictionary of empty tensors
        """
        return {
            "nodes": torch.tensor([], dtype=torch.int32, device=self.device),
            "parents": torch.tensor([], dtype=torch.int32, device=self.device),
            "features": torch.tensor([], dtype=torch.float32, device=self.device),
            "node_types": torch.tensor([], dtype=torch.int32, device=self.device),
            "depths": torch.tensor([], dtype=torch.int32, device=self.device),
            "siblings": torch.tensor([], dtype=torch.int32, device=self.device),
            "field_names": []
        }

    def batch_tensorize(self, ast_nodes: List[ast.AST]) -> List[Dict[str, torch.Tensor]]:
        """
        Convert multiple ASTs to GPU-friendly tensor format in batch.

        Args:
            ast_nodes: List of AST nodes to convert

        Returns:
            List of dictionaries of tensors representing the ASTs
        """
        try:
            # Process each AST individually
            # In the future, this could be optimized to process them in a single batch
            return [self.tensorize(node) for node in ast_nodes]
        except Exception as e:
            logger.error(f"Error batch tensorizing ASTs: {e}")
            return [self.create_empty_tensors() for _ in ast_nodes]

    def batch_detensorize(self, gpu_asts: List[Dict[str, torch.Tensor]]) -> List[ast.AST]:
        """
        Convert multiple GPU-friendly tensor formats back to ASTs in batch.

        Args:
            gpu_asts: List of dictionaries of tensors representing the ASTs

        Returns:
            List of AST nodes
        """
        try:
            # Process each GPU AST individually
            # In the future, this could be optimized to process them in a single batch
            return [self.detensorize(gpu_ast) for gpu_ast in gpu_asts]
        except Exception as e:
            logger.error(f"Error batch detensorizing ASTs: {e}")
            return [ast.Module(body=[], type_ignores=[]) for _ in gpu_asts]

    def detensorize(self, gpu_ast: Dict[str, torch.Tensor]) -> ast.AST:
        """
        Convert GPU-friendly tensor format back to AST.

        Args:
            gpu_ast: Dictionary of tensors representing the AST

        Returns:
            AST node
        """
        try:
            # Extract tensors
            nodes = gpu_ast["nodes"]
            parents = gpu_ast["parents"]
            node_types = gpu_ast["node_types"]
            field_names = gpu_ast.get("field_names", [])

            # Move tensors to CPU for processing
            nodes_cpu = nodes.cpu().numpy()
            parents_cpu = parents.cpu().numpy()
            node_types_cpu = node_types.cpu().numpy()

            # Create a mapping from node type IDs to AST node classes
            node_type_map = {v: k for k, v in AST_NODE_TYPES.items()}

            # Create empty AST nodes
            ast_nodes = {}
            for i in range(len(nodes_cpu)):
                node_type_id = node_types_cpu[i]
                node_class = node_type_map.get(node_type_id)

                if node_class is None:
                    logger.warning(f"Unknown node type ID: {node_type_id}")
                    continue

                # Create an empty instance of the node class
                if node_class == ast.Module:
                    ast_nodes[i] = ast.Module(body=[], type_ignores=[])
                elif node_class == ast.FunctionDef:
                    ast_nodes[i] = ast.FunctionDef(name="", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[])
                elif node_class == ast.ClassDef:
                    ast_nodes[i] = ast.ClassDef(name="", bases=[], keywords=[], body=[], decorator_list=[])
                elif node_class == ast.Assign:
                    ast_nodes[i] = ast.Assign(targets=[], value=None)
                elif node_class == ast.Name:
                    ast_nodes[i] = ast.Name(id="", ctx=ast.Load())
                elif node_class == ast.Constant:
                    ast_nodes[i] = ast.Constant(value=None)
                else:
                    # For other node types, create a generic node
                    try:
                        ast_nodes[i] = node_class()
                    except:
                        logger.warning(f"Failed to create node of type {node_class}")
                        continue

            # Build the tree structure
            for i in range(len(nodes_cpu)):
                parent_idx = parents_cpu[i]
                if parent_idx == -1:
                    # Root node
                    root = ast_nodes.get(i)
                    continue

                # Get parent and child nodes
                parent = ast_nodes.get(parent_idx)
                child = ast_nodes.get(i)

                if parent is None or child is None:
                    continue

                # Get field name
                field_name = field_names[i] if i < len(field_names) else ""

                # Handle list fields (e.g., body, targets)
                if "[" in field_name:
                    base_field, idx = field_name.split("[")
                    idx = int(idx.rstrip("]"))

                    # Ensure the field exists and is a list
                    if not hasattr(parent, base_field):
                        setattr(parent, base_field, [])

                    field = getattr(parent, base_field)
                    if not isinstance(field, list):
                        setattr(parent, base_field, [])
                        field = getattr(parent, base_field)

                    # Extend the list if needed
                    while len(field) <= idx:
                        field.append(None)

                    # Set the child
                    field[idx] = child
                else:
                    # Regular field
                    if hasattr(parent, field_name):
                        setattr(parent, field_name, child)

            # Return the root node
            return root if 'root' in locals() else None
        except Exception as e:
            logger.error(f"Error detensorizing AST: {e}")
            # Return an empty module as fallback
            return ast.Module(body=[], type_ignores=[])