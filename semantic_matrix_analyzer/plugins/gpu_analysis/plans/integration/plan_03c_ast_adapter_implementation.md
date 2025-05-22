# Plan 03c: AST Adapter Implementation

## Objective

Implement the AST Adapter class to convert between SMA's AST representation and GPU-friendly tensor format, enabling GPU-accelerated analysis of unimplemented SMA methods.

## Background

The AST Adapter is a critical component for integrating GPU acceleration with SMA's language parsing system. It converts between SMA's AST representation and a GPU-friendly tensor format that can be processed efficiently on GPUs. This conversion is essential for enabling GPU-accelerated analysis of unimplemented SMA methods.

## Current State

The current AST Adapter in `brain/gpu_analysis/ast_adapter.py` is implemented as:

```python
class ASTAdapter:
    """Adapter for converting between AST representations."""

    def __init__(self, base_parser):
        self.base_parser = base_parser

    def convert_to_gpu_format(self, ast_node):
        """Convert AST to GPU-friendly format."""
        # Implementation details omitted
        pass
```

This implementation is incomplete and doesn't fully support SMA's AST representation. It needs to be refactored to work with SMA's language parsing system and to support the GPU-accelerated implementation of unimplemented SMA methods.

## Implementation Plan

### 1. Update AST Adapter Class Definition

Refactor the `ASTAdapter` class to work with SMA's AST representation:

```python
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import logging
import ast
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ASTAdapter:
    """
    Adapter between SMA's AST representation and GPU-friendly format.

    This class provides methods to convert between SMA's AST representation
    and the GPU-friendly tensor format, enabling GPU-accelerated analysis
    of unimplemented SMA methods.
    """

    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AST adapter.

        Args:
            device: Device to use for GPU acceleration ("cuda" or "cpu")
            config: Configuration dictionary
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}

        # Initialize tensorizer
        from gpu_analysis.ast_tensor import GPUASTTensorizer
        self.tensorizer = GPUASTTensorizer(device=self.device)

        logger.info(f"AST Adapter initialized with device: {self.device}")

    def convert_to_gpu_format(self, ast_node: Any) -> Dict[str, torch.Tensor]:
        """
        Convert SMA's AST representation to GPU-friendly format.

        This method converts an AST node from SMA's representation to a
        GPU-friendly tensor format that can be processed efficiently on GPUs.

        Args:
            ast_node: AST node from SMA's parser

        Returns:
            Dictionary of tensors representing the AST
        """
        try:
            # If ast_node is already in GPU format, return it
            if isinstance(ast_node, dict) and "gpu_ast" in ast_node:
                return ast_node["gpu_ast"]

            # If ast_node is a dict with ast, use the ast
            if isinstance(ast_node, dict) and "ast" in ast_node:
                ast_node = ast_node["ast"]

            # Convert to GPU format using tensorizer
            gpu_ast = self.tensorizer.tensorize(ast_node)

            return gpu_ast
        except Exception as e:
            logger.error(f"Error converting AST to GPU format: {e}")
            # Return empty tensors as fallback
            return self.tensorizer.create_empty_tensors()
```

### 2. Implement Conversion from GPU Format

Add a method to convert from GPU-friendly format back to SMA's AST representation:

```python
def convert_from_gpu_format(self, gpu_ast: Dict[str, torch.Tensor]) -> Any:
    """
    Convert GPU-friendly format back to SMA's AST representation.

    This method converts a GPU-friendly tensor format back to an AST node
    compatible with SMA's parser. This is useful for integrating GPU-accelerated
    analysis results back into SMA's workflow.

    Args:
        gpu_ast: Dictionary of tensors representing the AST

    Returns:
        AST node compatible with SMA's parser
    """
    try:
        # Check if gpu_ast is valid
        if not isinstance(gpu_ast, dict) or not all(k in gpu_ast for k in ["node_types", "node_indices"]):
            raise ValueError("Invalid GPU AST format")

        # Convert from GPU format using tensorizer
        if hasattr(self.tensorizer, 'detensorize'):
            return self.tensorizer.detensorize(gpu_ast)
        else:
            # If detensorize is not implemented, raise an error
            raise NotImplementedError("Converting from GPU format to SMA format is not implemented")
    except Exception as e:
        logger.error(f"Error converting from GPU format: {e}")
        raise
```

### 3. Implement AST Preprocessing

Add methods to preprocess ASTs before conversion to GPU format:

```python
def preprocess_ast(self, ast_node: Any) -> Any:
    """
    Preprocess an AST node before conversion to GPU format.

    This method preprocesses an AST node to ensure it can be converted
    to GPU format correctly. It handles special cases and normalizes
    the AST structure.

    Args:
        ast_node: AST node from SMA's parser

    Returns:
        Preprocessed AST node
    """
    try:
        # If ast_node is a dict with ast, use the ast
        if isinstance(ast_node, dict) and "ast" in ast_node:
            ast_node = ast_node["ast"]

        # Handle different AST node types
        if isinstance(ast_node, ast.AST):
            # Add parent pointers to AST nodes
            return self.add_parent_pointers(ast_node)
        else:
            # Return as is for other types
            return ast_node
    except Exception as e:
        logger.error(f"Error preprocessing AST: {e}")
        return ast_node

def add_parent_pointers(self, ast_node: ast.AST) -> ast.AST:
    """
    Add parent pointers to AST nodes.

    This method adds parent pointers to AST nodes, which are useful for
    traversing the AST in both directions. This is important for certain
    types of analysis.

    Args:
        ast_node: AST node from SMA's parser

    Returns:
        AST node with parent pointers
    """
    # Create a copy of the AST to avoid modifying the original
    import copy
    ast_copy = copy.deepcopy(ast_node)

    # Add parent pointers
    for node in ast.walk(ast_copy):
        for child_name, child in ast.iter_fields(node):
            if isinstance(child, ast.AST):
                # Add parent pointer to child
                if not hasattr(child, 'parent'):
                    child.parent = node
            elif isinstance(child, list):
                for grandchild in child:
                    if isinstance(grandchild, ast.AST):
                        # Add parent pointer to grandchild
                        if not hasattr(grandchild, 'parent'):
                            grandchild.parent = node

    return ast_copy
```

### 4. Implement AST Postprocessing

Add methods to postprocess ASTs after conversion from GPU format:

```python
def postprocess_ast(self, ast_node: Any) -> Any:
    """
    Postprocess an AST node after conversion from GPU format.

    This method postprocesses an AST node to ensure it can be used
    with SMA's parser correctly. It handles special cases and normalizes
    the AST structure.

    Args:
        ast_node: AST node from GPU format conversion

    Returns:
        Postprocessed AST node
    """
    try:
        # Handle different AST node types
        if isinstance(ast_node, ast.AST):
            # Remove parent pointers from AST nodes
            return self.remove_parent_pointers(ast_node)
        else:
            # Return as is for other types
            return ast_node
    except Exception as e:
        logger.error(f"Error postprocessing AST: {e}")
        return ast_node

def remove_parent_pointers(self, ast_node: ast.AST) -> ast.AST:
    """
    Remove parent pointers from AST nodes.

    This method removes parent pointers from AST nodes, which were added
    during preprocessing. This is important for ensuring the AST is compatible
    with SMA's parser.

    Args:
        ast_node: AST node with parent pointers

    Returns:
        AST node without parent pointers
    """
    # Create a copy of the AST to avoid modifying the original
    import copy
    ast_copy = copy.deepcopy(ast_node)

    # Remove parent pointers
    for node in ast.walk(ast_copy):
        if hasattr(node, 'parent'):
            delattr(node, 'parent')

    return ast_copy
```

### 5. Implement Batch Processing

Add methods for batch processing of ASTs:

```python
def batch_convert_to_gpu_format(self, ast_nodes: List[Any]) -> List[Dict[str, torch.Tensor]]:
    """
    Convert multiple AST nodes to GPU format in batch.

    This method converts multiple AST nodes to GPU format in a single batch,
    which is more efficient than converting them one by one.

    Args:
        ast_nodes: List of AST nodes from SMA's parser

    Returns:
        List of dictionaries of tensors representing the ASTs
    """
    try:
        # Preprocess AST nodes
        preprocessed_nodes = [self.preprocess_ast(node) for node in ast_nodes]

        # Convert to GPU format in batch
        if hasattr(self.tensorizer, 'batch_tensorize'):
            return self.tensorizer.batch_tensorize(preprocessed_nodes)
        else:
            # Fall back to individual conversion
            return [self.tensorizer.tensorize(node) for node in preprocessed_nodes]
    except Exception as e:
        logger.error(f"Error batch converting ASTs to GPU format: {e}")
        # Return empty tensors as fallback
        return [self.tensorizer.create_empty_tensors() for _ in ast_nodes]

def batch_convert_from_gpu_format(self, gpu_asts: List[Dict[str, torch.Tensor]]) -> List[Any]:
    """
    Convert multiple GPU-format ASTs back to SMA's AST representation in batch.

    This method converts multiple GPU-format ASTs back to SMA's AST representation
    in a single batch, which is more efficient than converting them one by one.

    Args:
        gpu_asts: List of dictionaries of tensors representing the ASTs

    Returns:
        List of AST nodes compatible with SMA's parser
    """
    try:
        # Check if detensorize is implemented
        if hasattr(self.tensorizer, 'batch_detensorize'):
            # Convert from GPU format in batch
            ast_nodes = self.tensorizer.batch_detensorize(gpu_asts)

            # Postprocess AST nodes
            return [self.postprocess_ast(node) for node in ast_nodes]
        elif hasattr(self.tensorizer, 'detensorize'):
            # Fall back to individual conversion
            ast_nodes = [self.tensorizer.detensorize(gpu_ast) for gpu_ast in gpu_asts]

            # Postprocess AST nodes
            return [self.postprocess_ast(node) for node in ast_nodes]
        else:
            # If detensorize is not implemented, raise an error
            raise NotImplementedError("Converting from GPU format to SMA format is not implemented")
    except Exception as e:
        logger.error(f"Error batch converting from GPU format: {e}")
        raise
```

## Implementation Focus

The implementation should focus on:

1. **Conversion Functionality**: Implementing conversion between SMA's AST representation and GPU-friendly tensor format.

2. **Information Preservation**: Ensuring the conversion preserves all necessary information for analysis.

3. **Basic Error Handling**: Implementing essential error handling for architectural correctness.

4. **Batch Processing**: Implementing batch processing for efficiency.

5. **GPU Acceleration**: Enabling GPU-accelerated analysis of unimplemented SMA methods.

## Success Criteria

1. The AST Adapter correctly converts between SMA's AST representation and GPU-friendly tensor format.

2. The conversion preserves all necessary information for analysis.

3. The AST Adapter supports batch processing for efficiency.

4. The AST Adapter enables GPU-accelerated analysis of unimplemented SMA methods.

Note: Testing will be deferred until the complete architecture is implemented and stable. The focus is on velocity and architectural completion rather than incremental validation.

## References

1. SMA Language Parser Interface: `semantic_matrix_analyzer/semantic_matrix_analyzer/language/__init__.py`

2. SMA Python Parser: `semantic_matrix_analyzer/semantic_matrix_analyzer/language/python_parser.py`

3. GPU AST Adapter: `brain/gpu_analysis/ast_adapter.py`

4. GPU AST Tensorizer: `brain/gpu_analysis/ast_tensor.py`
