"""
GPU-Accelerated Complexity Analyzer

This module provides a GPU-accelerated implementation of code complexity analysis.
It uses PyTorch to accelerate the analysis process and can run on both CPU and GPU.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up logging
logger = logging.getLogger(__name__)


class ComplexityAnalyzer(nn.Module):
    """
    Neural network for analyzing code complexity.
    
    This module analyzes AST tensors to compute complexity metrics.
    
    Attributes:
        device: Device to use for analysis ("cuda" or "cpu")
        config: Configuration for the analyzer
    """
    
    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the complexity analyzer.
        
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
        self.output_layer = nn.Linear(self.hidden_dim, 5)  # 5 complexity metrics
        
        # Move to device
        self.to(self.device)
    
    def forward(self, ast_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute complexity metrics from AST tensors.
        
        Args:
            ast_tensors: Dictionary of AST tensors
            
        Returns:
            Dictionary of complexity metrics:
            - cyclomatic_complexity: Cyclomatic complexity
            - cognitive_complexity: Cognitive complexity
            - nesting_depth: Maximum nesting depth
            - num_statements: Number of statements
            - num_functions: Number of functions
        """
        # Check if tensors are empty (parsing error)
        if ast_tensors["nodes"].numel() == 0:
            return {
                "cyclomatic_complexity": torch.tensor(0.0, device=self.device),
                "cognitive_complexity": torch.tensor(0.0, device=self.device),
                "nesting_depth": torch.tensor(0.0, device=self.device),
                "num_statements": torch.tensor(0.0, device=self.device),
                "num_functions": torch.tensor(0.0, device=self.device)
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
        
        # Global pooling
        graph_embedding = torch.mean(node_embeds, dim=0)
        
        # Compute complexity metrics
        metrics = self.output_layer(graph_embedding)
        
        return {
            "cyclomatic_complexity": metrics[0],
            "cognitive_complexity": metrics[1],
            "nesting_depth": metrics[2],
            "num_statements": metrics[3],
            "num_functions": metrics[4]
        }
    
    def analyze(self, code: str) -> Dict[str, float]:
        """
        Analyze code complexity.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary of complexity metrics
        """
        # Tensorize AST
        from ..ast_tensor import ASTTensorizer
        tensorizer = ASTTensorizer(device=self.device)
        ast_tensors = tensorizer.tensorize(code)
        
        # Compute complexity metrics
        metrics = self(ast_tensors)
        
        # Convert to Python floats
        return {
            "cyclomatic_complexity": metrics["cyclomatic_complexity"].item(),
            "cognitive_complexity": metrics["cognitive_complexity"].item(),
            "nesting_depth": metrics["nesting_depth"].item(),
            "num_statements": metrics["num_statements"].item(),
            "num_functions": metrics["num_functions"].item()
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, float]:
        """
        Analyze file complexity.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dictionary of complexity metrics
        """
        # Read file
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Analyze code
        return self.analyze(code)
    
    def batch_analyze(self, codes: List[str]) -> List[Dict[str, float]]:
        """
        Analyze multiple code snippets in batch.
        
        Args:
            codes: List of Python code snippets
            
        Returns:
            List of dictionaries of complexity metrics
        """
        # Tensorize ASTs
        from ..ast_tensor import ASTTensorizer
        tensorizer = ASTTensorizer(device=self.device)
        
        results = []
        for code in codes:
            # Tensorize AST
            ast_tensors = tensorizer.tensorize(code)
            
            # Compute complexity metrics
            metrics = self(ast_tensors)
            
            # Convert to Python floats
            result = {
                "cyclomatic_complexity": metrics["cyclomatic_complexity"].item(),
                "cognitive_complexity": metrics["cognitive_complexity"].item(),
                "nesting_depth": metrics["nesting_depth"].item(),
                "num_statements": metrics["num_statements"].item(),
                "num_functions": metrics["num_functions"].item()
            }
            
            results.append(result)
        
        return results
