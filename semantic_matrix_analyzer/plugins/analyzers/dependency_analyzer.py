"""
GPU-Accelerated Dependency Analyzer

This module provides a GPU-accelerated implementation of code dependency analysis.
It uses PyTorch to accelerate the analysis process and can run on both CPU and GPU.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


class DependencyAnalyzer(nn.Module):
    """
    Neural network for analyzing code dependencies.
    
    This module analyzes AST tensors to compute dependency matrices.
    
    Attributes:
        device: Device to use for analysis ("cuda" or "cpu")
        config: Configuration for the analyzer
    """
    
    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dependency analyzer.
        
        Args:
            device: Device to use for analysis ("cuda" or "cpu")
            config: Configuration for the analyzer
        """
        super().__init__()
        
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}
        
        # Extract configuration values
        self.feature_dim = self.config.get("feature_dim", 10)
        self.hidden_dim = self.config.get("hidden_dim", 64)
        
        # Graph neural network layers
        self.node_embedding = nn.Linear(self.feature_dim, self.hidden_dim)
        self.edge_embedding = nn.Linear(2, self.hidden_dim)
        self.graph_conv1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.graph_conv2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, ast_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute dependency matrices from AST tensors.
        
        Args:
            ast_tensors: Dictionary of AST tensors
            
        Returns:
            Dictionary of dependency matrices:
            - function_dependencies: Function dependency matrix
            - node_similarity: Node similarity matrix
        """
        # Check if tensors are empty (parsing error)
        if ast_tensors["nodes"].numel() == 0:
            return {
                "function_dependencies": torch.tensor([], device=self.device),
                "node_similarity": torch.tensor([], device=self.device)
            }
        
        # Move tensors to device
        nodes = ast_tensors["nodes"].to(self.device)
        edges = ast_tensors["edges"].to(self.device)
        features = ast_tensors["features"].to(self.device)
        
        # Embed nodes
        node_embeds = self.node_embedding(features)
        node_embeds = F.relu(node_embeds)
        
        # Simple graph convolution
        for _ in range(2):
            # Aggregate messages from neighbors
            new_embeds = torch.zeros_like(node_embeds)
            for edge in edges:
                src, dst = edge
                new_embeds[dst] += node_embeds[src]
            
            # Update node embeddings
            node_embeds = F.relu(self.graph_conv1(node_embeds + new_embeds))
        
        # Compute dependency matrices
        # For simplicity, we'll just compute a similarity matrix between nodes
        similarity = torch.matmul(node_embeds, node_embeds.t())
        
        # Extract function nodes
        function_mask = (nodes == 2)  # FunctionDef nodes
        function_embeds = node_embeds[function_mask]
        
        # Compute function dependency matrix
        if function_embeds.size(0) > 0:
            function_deps = torch.matmul(function_embeds, function_embeds.t())
        else:
            function_deps = torch.tensor([], device=self.device)
        
        return {
            "function_dependencies": function_deps,
            "node_similarity": similarity
        }
    
    def analyze(self, code: str) -> Dict[str, np.ndarray]:
        """
        Analyze code dependencies.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary of dependency matrices
        """
        # Tensorize AST
        from ..ast_tensor import ASTTensorizer
        tensorizer = ASTTensorizer(device=self.device)
        ast_tensors = tensorizer.tensorize(code)
        
        # Compute dependency matrices
        matrices = self(ast_tensors)
        
        # Convert to NumPy arrays
        result = {}
        for key, tensor in matrices.items():
            if tensor.numel() > 0:
                result[key] = tensor.cpu().numpy()
            else:
                result[key] = np.array([])
        
        return result
    
    def analyze_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Analyze file dependencies.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dictionary of dependency matrices
        """
        # Read file
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Analyze code
        return self.analyze(code)
    
    def batch_analyze(self, codes: List[str]) -> List[Dict[str, np.ndarray]]:
        """
        Analyze multiple code snippets in batch.
        
        Args:
            codes: List of Python code snippets
            
        Returns:
            List of dictionaries of dependency matrices
        """
        # Tensorize ASTs
        from ..ast_tensor import ASTTensorizer
        tensorizer = ASTTensorizer(device=self.device)
        
        results = []
        for code in codes:
            # Tensorize AST
            ast_tensors = tensorizer.tensorize(code)
            
            # Compute dependency matrices
            matrices = self(ast_tensors)
            
            # Convert to NumPy arrays
            result = {}
            for key, tensor in matrices.items():
                if tensor.numel() > 0:
                    result[key] = tensor.cpu().numpy()
                else:
                    result[key] = np.array([])
            
            results.append(result)
        
        return results
    
    def extract_function_dependencies(self, code: str) -> Dict[str, List[str]]:
        """
        Extract function dependencies from code.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary of function name to list of dependent function names
        """
        import ast
        
        # Parse code
        try:
            tree = ast.parse(code)
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
            return {}
        
        # Find all functions
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
        
        # Find function calls
        dependencies = {func_name: [] for func_name in functions}
        
        class FunctionCallVisitor(ast.NodeVisitor):
            def __init__(self, current_function):
                self.current_function = current_function
                self.calls = []
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in functions:
                    self.calls.append(node.func.id)
                self.generic_visit(node)
        
        for func_name, func_node in functions.items():
            visitor = FunctionCallVisitor(func_name)
            visitor.visit(func_node)
            dependencies[func_name] = visitor.calls
        
        return dependencies
