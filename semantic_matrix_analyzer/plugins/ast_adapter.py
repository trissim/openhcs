"""
AST Adapter Module

This module provides adapters between SMA's AST representation and our GPU-friendly format.
It allows seamless conversion between the two representations, enabling integration with
SMA's existing codebase while leveraging the performance benefits of GPU acceleration.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from abc import ABC, abstractmethod

import torch
import numpy as np

from gpu_analysis.ast_tensor import GPUASTTensorizer
from gpu_analysis.logging_integration import get_logger
from gpu_analysis.error_handling import (
    GPUAnalysisError, GPUAnalysisParsingError, GPUAnalysisComponentError,
    handle_error, check_gpu_available, with_error_handling
)

# Import SMA's LanguageParser interface
try:
    from semantic_matrix_analyzer.semantic_matrix_analyzer.language import LanguageParser
except ImportError:
    # Define a fallback LanguageParser for development without SMA
    class LanguageParser(ABC):
        """Abstract base class for language-specific parsers."""

        @classmethod
        @abstractmethod
        def get_supported_extensions(cls) -> Set[str]:
            """Return file extensions supported by this parser."""
            pass

        @abstractmethod
        def parse_file(self, file_path: Path) -> Any:
            """Parse a file and return its AST."""
            pass

        @abstractmethod
        def get_node_type(self, node: Any) -> str:
            """Get the type of an AST node."""
            pass

        @abstractmethod
        def get_node_name(self, node: Any) -> Optional[str]:
            """Get the name of an AST node, if applicable."""
            pass

        @abstractmethod
        def get_node_children(self, node: Any) -> List[Any]:
            """Get the children of an AST node."""
            pass

        @abstractmethod
        def get_node_source_range(self, node: Any) -> Optional[Tuple[int, int]]:
            """Get the source range of an AST node."""
            pass

        @abstractmethod
        def get_node_source(self, node: Any, file_content: str) -> Optional[str]:
            """Get the source code for an AST node."""
            pass

# Set up logging
logger = get_logger(__name__)

class ASTAdapter:
    """
    Adapter between SMA's AST representation and GPU-friendly format.

    This class provides methods to convert between SMA's AST representation
    and our GPU-friendly format with parent pointers.

    Attributes:
        language_parser: SMA language parser to use for AST generation
        device: Device to place tensors on ("cuda" or "cpu")
        config: Optional configuration for the adapter
    """

    def __init__(self, language_parser=None, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AST adapter.

        Args:
            language_parser: SMA language parser to use for AST generation
            device: Device to place tensors on ("cuda" or "cpu")
            config: Optional configuration dictionary
        """
        self.language_parser = language_parser
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}

        # Initialize tensorizer
        self.tensorizer = GPUASTTensorizer(device=self.device, config=self.config)

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

            # Preprocess the AST node
            preprocessed_node = self.preprocess_ast(ast_node)

            # Convert to GPU format using tensorizer
            gpu_ast = self.tensorizer.tensorize(preprocessed_node)

            return gpu_ast
        except Exception as e:
            logger.error(f"Error converting AST to GPU format: {e}")
            # Return empty tensors as fallback
            return self.tensorizer.create_empty_tensors()

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
            if not isinstance(gpu_ast, dict) or not all(k in gpu_ast for k in ["node_types", "parents"]):
                raise ValueError("Invalid GPU AST format")

            # Convert from GPU format using tensorizer
            if hasattr(self.tensorizer, 'detensorize'):
                ast_node = self.tensorizer.detensorize(gpu_ast)

                # Postprocess the AST node
                return self.postprocess_ast(ast_node)
            else:
                # If detensorize is not implemented, use the fallback method
                ast_node = self._reconstruct_ast(gpu_ast)

                # Postprocess the AST node
                return self.postprocess_ast(ast_node)
        except Exception as e:
            logger.error(f"Error converting from GPU format: {e}")
            # Return an empty module as fallback
            return ast.Module(body=[], type_ignores=[])

    def _reconstruct_ast(self, gpu_ast: Dict[str, torch.Tensor]) -> ast.AST:
        """
        Reconstruct AST from GPU-friendly format.

        Args:
            gpu_ast: Dictionary of tensors representing the AST in GPU-friendly format

        Returns:
            AST node compatible with SMA's parser
        """
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
        from gpu_analysis.ast_tensor import AST_NODE_TYPES
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

    def cleanup(self) -> None:
        """
        Clean up resources used by the AST adapter.

        This method should be called when the adapter is no longer needed
        to free up GPU memory and other resources.
        """
        # Clean up GPU memory
        if self.device == "cuda":
            try:
                # Clean up any cached tensors
                torch.cuda.empty_cache()

                # Clean up tensorizer resources
                if hasattr(self.tensorizer, 'cleanup'):
                    self.tensorizer.cleanup()

                logger.info("AST adapter resources cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up AST adapter resources: {e}")

    def clear_cache(self) -> None:
        """
        Clear any cached data.
        """
        # Clear tensorizer cache
        if hasattr(self.tensorizer, 'clear_cache'):
            self.tensorizer.clear_cache()

        # Clear GPU memory cache
        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("AST adapter cache cleared")

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

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.

        Returns:
            Dictionary of cache information
        """
        info = {
            "size": 0,
            "hits": 0,
            "misses": 0
        }

        # Get tensorizer cache info
        if hasattr(self.tensorizer, 'get_cache_info'):
            tensorizer_cache_info = self.tensorizer.get_cache_info()
            info["tensorizer"] = tensorizer_cache_info
            info["size"] += tensorizer_cache_info.get("size", 0)
            info["hits"] += tensorizer_cache_info.get("hits", 0)
            info["misses"] += tensorizer_cache_info.get("misses", 0)

        return info

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
            # Return empty modules as fallback
            return [ast.Module(body=[], type_ignores=[]) for _ in gpu_asts]


class GPULanguageParser(LanguageParser):
    """
    GPU-accelerated language parser.

    This class implements SMA's LanguageParser interface with GPU acceleration.
    It uses the ASTAdapter to convert between SMA's AST representation and our
    GPU-friendly format.

    Attributes:
        base_parser: SMA language parser to use as a base
        device: Device to place tensors on ("cuda" or "cpu")
        config: Optional configuration for the parser
    """

    def __init__(self, base_parser=None, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GPU language parser.

        Args:
            base_parser: SMA language parser to use as a base (optional)
            device: Device to place tensors on ("cuda" or "cpu")
            config: Optional configuration dictionary
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}

        # Create a base parser for Python if none is provided
        if base_parser is None:
            try:
                from semantic_matrix_analyzer.semantic_matrix_analyzer.language.python_parser import PythonParser
                self.base_parser = PythonParser()
            except ImportError:
                # Create a simple Python parser if SMA is not available
                self.base_parser = self._create_fallback_python_parser()
        else:
            self.base_parser = base_parser

        # Initialize adapter
        self.adapter = ASTAdapter(self.base_parser, device=self.device, config=self.config)

        # Initialize tensorizer
        self.tensorizer = GPUASTTensorizer(device=self.device)

        logger.info(f"GPU Language Parser initialized with device: {self.device}")

    def _create_fallback_python_parser(self):
        """Create a fallback Python parser when SMA is not available."""
        class FallbackPythonParser:
            def parse_file(self, file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return ast.parse(f.read(), filename=str(file_path))

            def parse_code(self, code):
                return ast.parse(code)

        return FallbackPythonParser()

    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        """
        Return file extensions supported by this parser.

        Returns:
            A set of file extensions (including the dot) that this parser supports.
        """
        return {".py", ".pyi"}

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a file and return its AST.

        Args:
            file_path: Path to the file to parse

        Returns:
            Dictionary containing both the original AST and the GPU-friendly format

        Raises:
            FileNotFoundError: If the file does not exist.
            SyntaxError: If the file contains syntax errors.
            ValueError: If the file is not supported by this parser.
        """
        try:
            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check if file extension is supported
            if file_path.suffix.lower() not in self.get_supported_extensions():
                raise ValueError(f"Unsupported file extension: {file_path.suffix}")

            # Use base parser to parse the file
            ast_node = self.base_parser.parse_file(file_path)

            # Convert to GPU-friendly format
            gpu_ast = self.adapter.convert_to_gpu_format(ast_node)

            # Return both representations for flexibility
            return {
                "ast": ast_node,
                "gpu_ast": gpu_ast,
                "file_path": str(file_path)
            }
        except SyntaxError as e:
            logger.error(f"Syntax error in file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise

    def parse_string(self, code: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Parse a string of code and return its AST.

        This method parses a string of code and returns both the standard AST
        representation and the GPU-friendly tensor representation, enabling
        GPU-accelerated analysis.

        Args:
            code: String of code to parse.
            file_path: Optional path to associate with the code.

        Returns:
            The AST representation of the code, with both standard and GPU-friendly formats.

        Raises:
            SyntaxError: If the code contains syntax errors.
        """
        try:
            # Use base parser to parse the string
            if hasattr(self.base_parser, 'parse_code'):
                ast_node = self.base_parser.parse_code(code)
            else:
                # Fall back to ast module if base parser doesn't support string parsing
                ast_node = ast.parse(code)

            # Convert to GPU-friendly format
            gpu_ast = self.adapter.convert_to_gpu_format(ast_node)

            # Return both representations for flexibility
            return {
                "ast": ast_node,
                "gpu_ast": gpu_ast,
                "file_path": str(file_path) if file_path else None
            }
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
            raise

    # Alias for compatibility with existing code
    def parse_code(self, code: str) -> Dict[str, Any]:
        """
        Parse code string and return its AST.

        Args:
            code: Code string to parse

        Returns:
            Dictionary containing both the original AST and the GPU-friendly format
        """
        return self.parse_string(code)

    @with_error_handling
    def get_node_type(self, node: Any) -> str:
        """
        Get the type of an AST node.

        This method handles both standard AST nodes and GPU-accelerated AST nodes,
        delegating to the base parser for standard nodes and extracting type information
        from GPU-accelerated nodes.

        Args:
            node: An AST node returned by parse_file.

        Returns:
            A string representing the type of the node.
        """
        logger.debug("Getting node type")

        try:
            # If node is a dict with both ast and gpu_ast, use the ast
            if isinstance(node, dict) and "ast" in node:
                logger.debug("Using standard AST for node type")
                return self.base_parser.get_node_type(node["ast"])

            # If node is a dict with gpu_ast only, extract type from tensor
            elif isinstance(node, dict) and "gpu_ast" in node:
                logger.debug("Extracting type from GPU AST")
                # Extract type from GPU tensor
                gpu_ast = node["gpu_ast"]
                if "node_types" in gpu_ast and "node_indices" in gpu_ast:
                    # Get the type index for the root node (index 0)
                    type_idx = gpu_ast["node_types"][0].item()
                    # Map type index to type name using tensorizer's type mapping
                    from gpu_analysis.ast_tensor import AST_NODE_TYPES_REV
                    return AST_NODE_TYPES_REV.get(type_idx, f"Unknown_{type_idx}")
                else:
                    logger.warning("Invalid GPU AST format: missing node_types or node_indices")
                    raise ValueError("Invalid GPU AST format: missing node_types or node_indices")

            # Otherwise, delegate to base parser
            elif hasattr(self.base_parser, 'get_node_type'):
                logger.debug("Delegating to base parser for node type")
                return self.base_parser.get_node_type(node)

            # Fallback implementation for ast nodes
            elif isinstance(node, ast.AST):
                logger.debug("Using fallback implementation for AST node type")
                return node.__class__.__name__

            # Unknown node type
            else:
                logger.warning(f"Unknown node type: {type(node)}")
                return "Unknown"

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error getting node type: {error}")
            # Fall back to a generic type
            return "Unknown"

    @with_error_handling
    def get_node_name(self, node: Any) -> Optional[str]:
        """
        Get the name of an AST node, if applicable.

        This method handles both standard AST nodes and GPU-accelerated AST nodes,
        delegating to the base parser for standard nodes and extracting name information
        from GPU-accelerated nodes.

        Args:
            node: An AST node returned by parse_file.

        Returns:
            The name of the node, or None if the node does not have a name.
        """
        logger.debug("Getting node name")

        try:
            # If node is a dict with both ast and gpu_ast, use the ast
            if isinstance(node, dict) and "ast" in node:
                logger.debug("Using standard AST for node name")
                return self.base_parser.get_node_name(node["ast"])

            # If node is a dict with gpu_ast only, extract name from tensor
            elif isinstance(node, dict) and "gpu_ast" in node:
                logger.debug("Extracting name from GPU AST")
                # Extract name from GPU tensor
                gpu_ast = node["gpu_ast"]

                # Check if node has a name attribute
                if "node_names" in gpu_ast and "node_indices" in gpu_ast:
                    # Get the name index for the root node (index 0)
                    if "node_names" in gpu_ast and len(gpu_ast["node_names"]) > 0:
                        name_idx = gpu_ast["node_names"][0].item() if isinstance(gpu_ast["node_names"], torch.Tensor) else gpu_ast["node_names"][0]

                        # If name index is -1, node has no name
                        if name_idx == -1:
                            return None

                        # Get name from string table
                        if "string_table" in gpu_ast and name_idx < len(gpu_ast["string_table"]):
                            return gpu_ast["string_table"][name_idx]

                # Node might not have a name
                return None

            # Otherwise, delegate to base parser
            elif hasattr(self.base_parser, 'get_node_name'):
                logger.debug("Delegating to base parser for node name")
                return self.base_parser.get_node_name(node)

            # Fallback implementation for ast nodes
            elif isinstance(node, ast.AST):
                logger.debug("Using fallback implementation for AST node name")
                if hasattr(node, 'name'):
                    return node.name
                elif hasattr(node, 'id'):
                    return node.id

            # Unknown node type
            return None

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error getting node name: {error}")
            return None

    @with_error_handling
    def get_node_children(self, node: Any) -> List[Any]:
        """
        Get the children of an AST node.

        This method handles both standard AST nodes and GPU-accelerated AST nodes,
        delegating to the base parser for standard nodes and extracting children
        from GPU-accelerated nodes.

        Args:
            node: An AST node returned by parse_file.

        Returns:
            A list of child nodes.
        """
        logger.debug("Getting node children")

        try:
            # If node is a dict with both ast and gpu_ast, use the ast
            if isinstance(node, dict) and "ast" in node:
                logger.debug("Using standard AST for node children")
                # Get children from standard AST
                if hasattr(self.base_parser, 'get_node_children'):
                    ast_children = self.base_parser.get_node_children(node["ast"])
                else:
                    # Fallback implementation for ast nodes
                    ast_children = []
                    if isinstance(node["ast"], ast.AST):
                        for field, value in ast.iter_fields(node["ast"]):
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, ast.AST):
                                        ast_children.append(item)
                            elif isinstance(value, ast.AST):
                                ast_children.append(value)

                # If gpu_ast is available, create combined nodes for children
                if "gpu_ast" in node:
                    logger.debug("Creating combined nodes for children")
                    gpu_ast = node["gpu_ast"]
                    if "node_children" in gpu_ast and "node_indices" in gpu_ast:
                        # Get children indices for the root node (index 0)
                        children_indices = self._get_children_indices(gpu_ast, 0)

                        # Create combined nodes for each child
                        combined_children = []
                        for i, ast_child in enumerate(ast_children):
                            if i < len(children_indices):
                                child_idx = children_indices[i]
                                gpu_child = self._extract_subtree(gpu_ast, child_idx)
                                combined_children.append({
                                    "ast": ast_child,
                                    "gpu_ast": gpu_child,
                                    "file_path": node.get("file_path")
                                })
                            else:
                                # Fall back to standard AST if indices don't match
                                combined_children.append({
                                    "ast": ast_child,
                                    "file_path": node.get("file_path")
                                })

                        return combined_children

                # If no gpu_ast or error, return standard AST children
                return ast_children

            # If node is a dict with gpu_ast only, extract children from tensor
            elif isinstance(node, dict) and "gpu_ast" in node:
                logger.debug("Extracting children from GPU AST")
                gpu_ast = node["gpu_ast"]
                if "node_children" in gpu_ast and "node_indices" in gpu_ast:
                    # Get children indices for the root node (index 0)
                    children_indices = self._get_children_indices(gpu_ast, 0)

                    # Create GPU-only nodes for each child
                    gpu_children = []
                    for child_idx in children_indices:
                        gpu_child = self._extract_subtree(gpu_ast, child_idx)
                        gpu_children.append({
                            "gpu_ast": gpu_child,
                            "file_path": node.get("file_path")
                        })

                    return gpu_children
                else:
                    # No children information available
                    logger.warning("Invalid GPU AST format: missing node_children or node_indices")
                    return []

            # Otherwise, delegate to base parser
            elif hasattr(self.base_parser, 'get_node_children'):
                logger.debug("Delegating to base parser for node children")
                return self.base_parser.get_node_children(node)

            # Fallback implementation for ast nodes
            elif isinstance(node, ast.AST):
                logger.debug("Using fallback implementation for AST node children")
                children = []
                for field, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, ast.AST):
                                children.append(item)
                    elif isinstance(value, ast.AST):
                        children.append(value)
                return children

            # Unknown node type
            else:
                logger.warning(f"Unknown node type for children: {type(node)}")
                return []

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error getting node children: {error}")
            return []

    def _get_children_indices(self, gpu_ast: Dict[str, torch.Tensor], node_idx: int) -> List[int]:
        """
        Get the indices of a node's children in the GPU AST.

        Args:
            gpu_ast: GPU AST tensor dictionary
            node_idx: Index of the node

        Returns:
            List of child indices
        """
        try:
            # Check if node_children exists
            if "node_children" not in gpu_ast:
                return []

            node_children = gpu_ast["node_children"]

            # If node_children is a list of lists
            if isinstance(node_children, list) and node_idx < len(node_children):
                return node_children[node_idx]

            # If node_children is a tensor
            elif isinstance(node_children, torch.Tensor):
                # Find children based on parent-child relationship
                if "parents" in gpu_ast:
                    parents = gpu_ast["parents"]
                    # Find indices where parent is node_idx
                    children = torch.nonzero(parents == node_idx).squeeze(-1)
                    return children.tolist() if children.numel() > 0 else []

            return []
        except Exception as e:
            logger.error(f"Error getting children indices: {e}")
            return []

    def _extract_subtree(self, gpu_ast: Dict[str, torch.Tensor], node_idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract a subtree from the GPU AST.

        Args:
            gpu_ast: GPU AST tensor dictionary
            node_idx: Index of the root node of the subtree

        Returns:
            Dictionary of tensors representing the subtree
        """
        try:
            # Create a new dictionary for the subtree
            subtree = {}

            # Copy tensors that don't depend on node indices
            for key, value in gpu_ast.items():
                if key not in ["nodes", "parents", "node_types", "node_names", "node_children"]:
                    subtree[key] = value

            # Extract the subtree nodes
            if "nodes" in gpu_ast and isinstance(gpu_ast["nodes"], torch.Tensor):
                subtree["nodes"] = gpu_ast["nodes"][node_idx:node_idx+1]

            # Extract the subtree node types
            if "node_types" in gpu_ast and isinstance(gpu_ast["node_types"], torch.Tensor):
                subtree["node_types"] = gpu_ast["node_types"][node_idx:node_idx+1]

            # Extract the subtree node names
            if "node_names" in gpu_ast and isinstance(gpu_ast["node_names"], torch.Tensor):
                subtree["node_names"] = gpu_ast["node_names"][node_idx:node_idx+1]

            # Set the parent to -1 (root)
            if "parents" in gpu_ast:
                subtree["parents"] = torch.tensor([-1], device=gpu_ast["parents"].device)

            # Extract children recursively
            children_indices = self._get_children_indices(gpu_ast, node_idx)
            if children_indices:
                subtree["node_children"] = [list(range(1, len(children_indices) + 1))]

                # Add children to the subtree
                for i, child_idx in enumerate(children_indices):
                    child_subtree = self._extract_subtree(gpu_ast, child_idx)

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
        except Exception as e:
            logger.error(f"Error extracting subtree: {e}")
            # Return a minimal valid subtree
            return {
                "nodes": torch.tensor([0], device=gpu_ast.get("nodes", torch.tensor([0])).device),
                "parents": torch.tensor([-1], device=gpu_ast.get("parents", torch.tensor([-1])).device),
                "node_types": torch.tensor([0], device=gpu_ast.get("node_types", torch.tensor([0])).device),
                "node_names": torch.tensor([-1], device=gpu_ast.get("node_names", torch.tensor([-1])).device)
            }

    @with_error_handling
    def get_node_source_range(self, node: Any) -> Optional[Tuple[int, int]]:
        """
        Get the source range of an AST node.

        This method handles both standard AST nodes and GPU-accelerated AST nodes,
        delegating to the base parser for standard nodes and extracting source range
        from GPU-accelerated nodes.

        Args:
            node: An AST node returned by parse_file.

        Returns:
            A tuple of (start_line, end_line), or None if not available.
            Line numbers are 1-based.
        """
        logger.debug("Getting node source range")

        try:
            # If node is a dict with both ast and gpu_ast, use the ast
            if isinstance(node, dict) and "ast" in node:
                logger.debug("Using standard AST for node source range")
                if hasattr(self.base_parser, 'get_node_source_range'):
                    return self.base_parser.get_node_source_range(node["ast"])
                else:
                    # Fallback implementation for ast nodes
                    ast_node = node["ast"]
                    if isinstance(ast_node, ast.AST):
                        if hasattr(ast_node, 'lineno') and hasattr(ast_node, 'end_lineno'):
                            return (ast_node.lineno, ast_node.end_lineno)
                        elif hasattr(ast_node, 'lineno'):
                            return (ast_node.lineno, ast_node.lineno)

            # If node is a dict with gpu_ast only, extract source range from tensor
            elif isinstance(node, dict) and "gpu_ast" in node:
                logger.debug("Extracting source range from GPU AST")
                gpu_ast = node["gpu_ast"]
                if "node_line_ranges" in gpu_ast:
                    # Get the source range for the root node (index 0)
                    line_ranges = gpu_ast["node_line_ranges"]
                    if isinstance(line_ranges, torch.Tensor) and line_ranges.shape[0] > 0:
                        start_line = line_ranges[0, 0].item()
                        end_line = line_ranges[0, 1].item()
                        # Convert to 1-based line numbers if they're 0-based
                        if start_line == 0:
                            start_line = 1
                        return (start_line, end_line)
                    elif isinstance(line_ranges, list) and len(line_ranges) > 0:
                        start_line, end_line = line_ranges[0]
                        # Convert to 1-based line numbers if they're 0-based
                        if start_line == 0:
                            start_line = 1
                        return (start_line, end_line)

                # No source range information available
                logger.debug("No source range information available in GPU AST")
                return None

            # Otherwise, delegate to base parser
            elif hasattr(self.base_parser, 'get_node_source_range'):
                logger.debug("Delegating to base parser for node source range")
                return self.base_parser.get_node_source_range(node)

            # Fallback implementation for ast nodes
            elif isinstance(node, ast.AST):
                logger.debug("Using fallback implementation for AST node source range")
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    return (node.lineno, node.end_lineno)
                elif hasattr(node, 'lineno'):
                    return (node.lineno, node.lineno)

            # Unknown node type
            else:
                logger.warning(f"Unknown node type for source range: {type(node)}")
                return None

            return None

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error getting node source range: {error}")
            return None

    @with_error_handling
    def get_node_source(self, node: Any, file_content: str) -> Optional[str]:
        """
        Get the source code for an AST node.

        This method handles both standard AST nodes and GPU-accelerated AST nodes,
        delegating to the base parser for standard nodes and extracting source code
        from GPU-accelerated nodes using source ranges.

        Args:
            node: An AST node returned by parse_file.
            file_content: The content of the file.

        Returns:
            The source code for the node, or None if not available.
        """
        logger.debug("Getting node source")

        try:
            # If node is a dict with both ast and gpu_ast, use the ast
            if isinstance(node, dict) and "ast" in node:
                logger.debug("Using standard AST for node source")
                if hasattr(self.base_parser, 'get_node_source'):
                    return self.base_parser.get_node_source(node["ast"], file_content)
                else:
                    # Fallback implementation for ast nodes
                    source_range = self.get_node_source_range(node)
                    if source_range and file_content:
                        start_line, end_line = source_range
                        lines = file_content.splitlines()
                        if 1 <= start_line <= len(lines) and 1 <= end_line <= len(lines):
                            return '\n'.join(lines[start_line-1:end_line])

            # If node is a dict with gpu_ast only, extract source using source range
            elif isinstance(node, dict) and "gpu_ast" in node:
                logger.debug("Extracting source from GPU AST using source range")
                # Get source range
                source_range = self.get_node_source_range(node)
                if source_range is None:
                    logger.debug("No source range available for GPU AST node")
                    return None

                # Extract source code using source range
                start_line, end_line = source_range
                lines = file_content.splitlines()

                # Check if line numbers are valid
                if start_line < 1 or start_line > len(lines) or end_line < 1 or end_line > len(lines):
                    logger.warning(f"Invalid source range: {start_line}-{end_line} for file with {len(lines)} lines")
                    return None

                # Extract source code (convert to 0-based indices)
                source_lines = lines[start_line - 1:end_line]
                return "\n".join(source_lines)

            # Otherwise, delegate to base parser
            elif hasattr(self.base_parser, 'get_node_source'):
                logger.debug("Delegating to base parser for node source")
                return self.base_parser.get_node_source(node, file_content)

            # Fallback implementation for ast nodes
            elif isinstance(node, ast.AST):
                logger.debug("Using fallback implementation for AST node source")
                source_range = self.get_node_source_range(node)
                if source_range and file_content:
                    start_line, end_line = source_range
                    lines = file_content.splitlines()
                    if 1 <= start_line <= len(lines) and 1 <= end_line <= len(lines):
                        return '\n'.join(lines[start_line-1:end_line])

            # Unknown node type
            else:
                logger.warning(f"Unknown node type for source: {type(node)}")
                return None

            return None

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error getting node source: {error}")
            return None

    @with_error_handling
    def get_node_attributes(self, node: Any) -> Dict[str, Any]:
        """
        Get the attributes of an AST node.

        This method extracts attributes from both standard and GPU-accelerated AST nodes,
        providing a unified interface for accessing node attributes.

        Args:
            node: An AST node returned by parse_file.

        Returns:
            A dictionary of node attributes.
        """
        logger.debug("Getting node attributes")

        try:
            # If node is a dict with both ast and gpu_ast, use the ast
            if isinstance(node, dict) and "ast" in node:
                logger.debug("Using standard AST for node attributes")
                # Get attributes from standard AST
                if hasattr(self.base_parser, 'get_node_attributes'):
                    return self.base_parser.get_node_attributes(node["ast"])
                else:
                    # Fall back to ast module
                    if isinstance(node["ast"], ast.AST):
                        return {name: getattr(node["ast"], name) for name in node["ast"]._fields}

            # If node is a dict with gpu_ast only, extract attributes from tensor
            elif isinstance(node, dict) and "gpu_ast" in node:
                logger.debug("Extracting attributes from GPU AST")
                gpu_ast = node["gpu_ast"]
                if "node_attributes" in gpu_ast:
                    # Get attributes for the root node (index 0)
                    if isinstance(gpu_ast["node_attributes"], list) and len(gpu_ast["node_attributes"]) > 0:
                        return gpu_ast["node_attributes"][0]
                    elif isinstance(gpu_ast["node_attributes"], dict):
                        return gpu_ast["node_attributes"]

                # Try to extract attributes from other tensors
                attributes = {}

                # Extract type
                if "node_types" in gpu_ast and len(gpu_ast["node_types"]) > 0:
                    type_idx = gpu_ast["node_types"][0].item() if isinstance(gpu_ast["node_types"], torch.Tensor) else gpu_ast["node_types"][0]
                    from gpu_analysis.ast_tensor import AST_NODE_TYPES_REV
                    attributes["type"] = AST_NODE_TYPES_REV.get(type_idx, f"Unknown_{type_idx}")

                # Extract name
                if "node_names" in gpu_ast and len(gpu_ast["node_names"]) > 0:
                    name_idx = gpu_ast["node_names"][0].item() if isinstance(gpu_ast["node_names"], torch.Tensor) else gpu_ast["node_names"][0]
                    if name_idx != -1 and "string_table" in gpu_ast and name_idx < len(gpu_ast["string_table"]):
                        attributes["name"] = gpu_ast["string_table"][name_idx]

                # Extract source range
                if "node_line_ranges" in gpu_ast:
                    line_ranges = gpu_ast["node_line_ranges"]
                    if isinstance(line_ranges, torch.Tensor) and line_ranges.shape[0] > 0:
                        start_line = line_ranges[0, 0].item()
                        end_line = line_ranges[0, 1].item()
                        attributes["lineno"] = start_line
                        attributes["end_lineno"] = end_line
                    elif isinstance(line_ranges, list) and len(line_ranges) > 0:
                        start_line, end_line = line_ranges[0]
                        attributes["lineno"] = start_line
                        attributes["end_lineno"] = end_line

                return attributes

            # Otherwise, delegate to base parser
            elif hasattr(self.base_parser, 'get_node_attributes'):
                logger.debug("Delegating to base parser for node attributes")
                return self.base_parser.get_node_attributes(node)

            # Fallback implementation for ast nodes
            elif isinstance(node, ast.AST):
                logger.debug("Using fallback implementation for AST node attributes")
                return {name: getattr(node, name) for name in node._fields}

            # Unknown node type
            else:
                logger.warning(f"Unknown node type for attributes: {type(node)}")
                return {}

            # No attributes available
            return {}

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error getting node attributes: {error}")
            return {}

    @with_error_handling
    def is_node_of_type(self, node: Any, node_type: str) -> bool:
        """
        Check if a node is of a specific type.

        This method checks if a node is of a specific type, handling both
        standard and GPU-accelerated AST nodes.

        Args:
            node: An AST node returned by parse_file.
            node_type: The type to check for.

        Returns:
            True if the node is of the specified type, False otherwise.
        """
        logger.debug(f"Checking if node is of type: {node_type}")

        try:
            # Get node type
            actual_type = self.get_node_type(node)

            # Check if types match
            return actual_type == node_type

        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error checking node type: {error}")
            return False

    @classmethod
    def is_supported_file(cls, file_path: Path) -> bool:
        """
        Check if a file is supported by this parser.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the file is supported, False otherwise.
        """
        return file_path.suffix.lower() in cls.get_supported_extensions()

    @classmethod
    def get_language_name(cls) -> str:
        """
        Get the name of the language supported by this parser.

        Returns:
            The name of the language.
        """
        return "Python-GPU"

    def get_gpu_ast(self, node: Any) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get the GPU-friendly AST representation of a node.

        This method extracts the GPU-friendly AST representation from a node,
        converting it if necessary.

        Args:
            node: An AST node returned by parse_file.

        Returns:
            The GPU-friendly AST representation, or None if not available.
        """
        try:
            # If node is a dict with gpu_ast, return it
            if isinstance(node, dict) and "gpu_ast" in node:
                return node["gpu_ast"]

            # If node is a dict with ast, convert it
            if isinstance(node, dict) and "ast" in node:
                return self.adapter.convert_to_gpu_format(node["ast"])

            # If node is a standard AST node, convert it
            return self.adapter.convert_to_gpu_format(node)
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error getting GPU AST: {error}")
            return None

    @with_error_handling
    def analyze_complexity(self, node: Any) -> Dict[str, Any]:
        """
        Analyze the complexity of an AST node using GPU acceleration.

        This method analyzes the complexity of an AST node using GPU acceleration,
        providing metrics such as cyclomatic complexity, cognitive complexity,
        and maintainability index.

        Args:
            node: An AST node returned by parse_file.

        Returns:
            Dictionary of complexity metrics.
        """
        try:
            # Convert to GPU format if needed
            gpu_ast = self.get_gpu_ast(node)

            if gpu_ast is None:
                raise ValueError("Cannot analyze node: GPU AST not available")

            # Perform complexity analysis
            from gpu_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer
            analyzer = ComplexityAnalyzer(device=self.device)

            # Analyze complexity
            complexity_results = analyzer.analyze(gpu_ast)

            # Format results
            return {
                "cyclomatic_complexity": complexity_results.get("cyclomatic_complexity", 0),
                "cognitive_complexity": complexity_results.get("cognitive_complexity", 0),
                "maintainability_index": complexity_results.get("maintainability_index", 0),
                "halstead_metrics": complexity_results.get("halstead_metrics", {}),
                "loc_metrics": complexity_results.get("loc_metrics", {})
            }
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error analyzing complexity: {error}")
            # Return default values as fallback
            return {
                "cyclomatic_complexity": 0,
                "cognitive_complexity": 0,
                "maintainability_index": 0,
                "halstead_metrics": {},
                "loc_metrics": {}
            }

    @with_error_handling
    def match_patterns(self, node: Any, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match patterns in an AST node using GPU acceleration.

        This method matches patterns in an AST node using GPU acceleration,
        providing a list of matches with their locations and confidence scores.

        Args:
            node: An AST node returned by parse_file.
            patterns: List of pattern definitions to match.

        Returns:
            List of pattern matches.
        """
        try:
            # Convert to GPU format if needed
            gpu_ast = self.get_gpu_ast(node)

            if gpu_ast is None:
                raise ValueError("Cannot match patterns: GPU AST not available")

            # Perform pattern matching
            from gpu_analysis.pattern_matcher import PatternMatcher
            matcher = PatternMatcher(device=self.device)

            # Match patterns
            matches = matcher.match_patterns(gpu_ast, patterns)

            # Format results
            formatted_matches = []
            for match in matches:
                formatted_match = {
                    "pattern_id": match.get("pattern_id", "unknown"),
                    "pattern_name": match.get("pattern_name", "Unknown Pattern"),
                    "confidence": match.get("confidence", 0.0),
                    "node_type": match.get("node_type", "Unknown"),
                    "location": match.get("location", {}),
                    "description": match.get("description", "")
                }
                formatted_matches.append(formatted_match)

            return formatted_matches
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error matching patterns: {error}")
            # Return empty list as fallback
            return []

    @with_error_handling
    def analyze_dependencies(self, node: Any) -> Dict[str, Any]:
        """
        Analyze the dependencies of an AST node using GPU acceleration.

        This method analyzes the dependencies of an AST node using GPU acceleration,
        providing information about imports, function calls, and variable references.

        Args:
            node: An AST node returned by parse_file.

        Returns:
            Dictionary of dependency information.
        """
        try:
            # Convert to GPU format if needed
            gpu_ast = self.get_gpu_ast(node)

            if gpu_ast is None:
                raise ValueError("Cannot analyze dependencies: GPU AST not available")

            # Perform dependency analysis
            from gpu_analysis.analyzers.dependency_analyzer import DependencyAnalyzer
            analyzer = DependencyAnalyzer(device=self.device)

            # Analyze dependencies
            dependency_results = analyzer.analyze(gpu_ast)

            # Format results
            return {
                "imports": dependency_results.get("imports", []),
                "function_calls": dependency_results.get("function_calls", []),
                "variable_references": dependency_results.get("variable_references", []),
                "class_references": dependency_results.get("class_references", []),
                "external_dependencies": dependency_results.get("external_dependencies", {})
            }
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error analyzing dependencies: {error}")
            # Return default values as fallback
            return {
                "imports": [],
                "function_calls": [],
                "variable_references": [],
                "class_references": [],
                "external_dependencies": {}
            }

    @with_error_handling
    def analyze_semantics(self, node: Any) -> Dict[str, Any]:
        """
        Perform semantic analysis on an AST node using GPU acceleration.

        This method performs semantic analysis on an AST node using GPU acceleration,
        providing information about the semantic structure and meaning of the code.

        Args:
            node: An AST node returned by parse_file.

        Returns:
            Dictionary of semantic analysis results.
        """
        try:
            # Convert to GPU format if needed
            gpu_ast = self.get_gpu_ast(node)

            if gpu_ast is None:
                raise ValueError("Cannot analyze semantics: GPU AST not available")

            # Perform semantic analysis
            from gpu_analysis.analyzers.semantic_analyzer import SemanticAnalyzer
            analyzer = SemanticAnalyzer(device=self.device)

            # Analyze semantics
            semantic_results = analyzer.analyze(gpu_ast)

            # Return results directly
            return semantic_results
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error analyzing semantics: {error}")
            # Return default values as fallback
            return {
                "complexity": {},
                "dependencies": {},
                "pattern_matches": [],
                "metrics": {},
                "semantic_features": {}
            }

    def get_file_type(self, file_path: Path) -> str:
        """
        Get the type of a file.

        Args:
            file_path: Path to the file to check.

        Returns:
            The type of the file.
        """
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")

        # For Python files, we can distinguish between different types
        if file_path.suffix.lower() == ".py":
            return "python"
        elif file_path.suffix.lower() == ".pyi":
            return "python-interface"
        else:
            return "unknown"

    def cleanup(self) -> None:
        """
        Clean up resources used by the GPU language parser.

        This method should be called when the parser is no longer needed
        to free up GPU memory and other resources.
        """
        # Clean up adapter resources
        if hasattr(self.adapter, 'cleanup'):
            self.adapter.cleanup()

        # Clean up base parser resources if it has a cleanup method
        if hasattr(self.base_parser, 'cleanup'):
            self.base_parser.cleanup()

        logger.info("GPU language parser resources cleaned up")

    def clear_cache(self) -> None:
        """
        Clear any cached data.
        """
        # Clear adapter cache
        if hasattr(self.adapter, 'clear_cache'):
            self.adapter.clear_cache()

        # Clear base parser cache if it has a clear_cache method
        if hasattr(self.base_parser, 'clear_cache'):
            self.base_parser.clear_cache()

        logger.info("GPU language parser cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.

        Returns:
            Dictionary of cache information
        """
        info = {
            "size": 0,
            "hits": 0,
            "misses": 0
        }

        # Get adapter cache info
        if hasattr(self.adapter, 'get_cache_info'):
            adapter_cache_info = self.adapter.get_cache_info()
            info["adapter"] = adapter_cache_info
            info["size"] += adapter_cache_info.get("size", 0)
            info["hits"] += adapter_cache_info.get("hits", 0)
            info["misses"] += adapter_cache_info.get("misses", 0)

        # Get base parser cache info
        if hasattr(self.base_parser, 'get_cache_info'):
            base_parser_cache_info = self.base_parser.get_cache_info()
            info["base_parser"] = base_parser_cache_info
            info["size"] += base_parser_cache_info.get("size", 0)
            info["hits"] += base_parser_cache_info.get("hits", 0)
            info["misses"] += base_parser_cache_info.get("misses", 0)

        return info

    @with_error_handling
    def analyze_component_code(self, component_name: str, code: str) -> Dict[str, Any]:
        """
        Analyze component code using GPU acceleration.

        This method is designed to be used by SMA's `analyze_component` method
        to provide GPU-accelerated analysis of component code. It performs a
        comprehensive analysis using all available GPU-accelerated analyzers.

        Args:
            component_name: Name of the component
            code: Code to analyze

        Returns:
            Analysis results
        """
        logger.info(f"Analyzing component code: {component_name}")

        try:
            # Parse the code
            logger.debug("Parsing component code")
            parsed = self.parse_string(code)

            # Get the GPU AST
            gpu_ast = parsed.get("gpu_ast")
            if gpu_ast is None:
                raise ValueError("GPU AST not available for component code")

            # Perform comprehensive analysis
            logger.debug("Performing comprehensive analysis of component code")

            # Analyze complexity
            logger.debug("Analyzing complexity")
            complexity_results = self.analyze_complexity(parsed)

            # Analyze dependencies
            logger.debug("Analyzing dependencies")
            dependency_results = self.analyze_dependencies(parsed)

            # Match patterns
            logger.debug("Matching patterns")
            # Get default patterns from pattern matcher
            from gpu_analysis.pattern_matcher import PatternMatcher
            matcher = PatternMatcher(device=self.device)
            default_patterns = matcher.get_default_patterns()
            pattern_matches = self.match_patterns(parsed, default_patterns)

            # Perform semantic analysis
            logger.debug("Performing semantic analysis")
            semantic_results = self.analyze_semantics(parsed)

            # Combine results
            results = {
                "component_name": component_name,
                "complexity": complexity_results,
                "dependencies": dependency_results,
                "pattern_matches": pattern_matches,
                "semantic_features": semantic_results.get("semantic_features", {}),
                "metrics": semantic_results.get("metrics", {})
            }

            logger.info(f"Component code analysis complete for {component_name}")
            return results
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error analyzing component {component_name}: {error}")
            raise GPUAnalysisComponentError(f"Error analyzing component {component_name}", e)

    @with_error_handling
    def analyze_file_for_cli(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a file for CLI output using GPU acceleration.

        This method is designed to be used by SMA's CLI commands to provide
        GPU-accelerated analysis of files. It performs a comprehensive analysis
        using all available GPU-accelerated analyzers and formats the results
        for CLI output.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Analysis results formatted for CLI output
        """
        logger.info(f"Analyzing file for CLI: {file_path}")

        try:
            # Parse the file
            logger.debug(f"Parsing file: {file_path}")
            parsed = self.parse_file(file_path)

            # Get the GPU AST
            gpu_ast = parsed.get("gpu_ast")
            if gpu_ast is None:
                raise ValueError("GPU AST not available for file")

            # Perform comprehensive analysis
            logger.debug("Performing comprehensive analysis of file")

            # Analyze complexity
            logger.debug("Analyzing complexity")
            complexity_results = self.analyze_complexity(parsed)

            # Analyze dependencies
            logger.debug("Analyzing dependencies")
            dependency_results = self.analyze_dependencies(parsed)

            # Match patterns
            logger.debug("Matching patterns")
            # Get default patterns from pattern matcher
            from gpu_analysis.pattern_matcher import PatternMatcher
            matcher = PatternMatcher(device=self.device)
            default_patterns = matcher.get_default_patterns()
            pattern_matches = self.match_patterns(parsed, default_patterns)

            # Perform semantic analysis
            logger.debug("Performing semantic analysis")
            semantic_results = self.analyze_semantics(parsed)

            # Format results for CLI output
            cli_results = {
                "file_path": str(file_path),
                "complexity": complexity_results,
                "dependencies": dependency_results,
                "patterns": pattern_matches,
                "metrics": semantic_results.get("metrics", {}),
                "semantic_features": semantic_results.get("semantic_features", {})
            }

            # Add summary for CLI output
            cli_results["summary"] = {
                "cyclomatic_complexity": complexity_results.get("cyclomatic_complexity", 0),
                "cognitive_complexity": complexity_results.get("cognitive_complexity", 0),
                "maintainability_index": complexity_results.get("maintainability_index", 0),
                "num_dependencies": len(dependency_results.get("imports", [])),
                "num_patterns": len(pattern_matches),
                "loc": complexity_results.get("loc_metrics", {}).get("loc", 0)
            }

            logger.info(f"File analysis complete for CLI: {file_path}")
            return cli_results
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error analyzing file {file_path} for CLI: {error}")
            raise GPUAnalysisParsingError(f"Error analyzing file {file_path} for CLI", e)

    @with_error_handling
    def extract_component(self, file_path: Path, component_name: str) -> Dict[str, Any]:
        """
        Extract a component from a file.

        This method extracts a component (class, function, etc.) from a file
        and returns its AST representation.

        Args:
            file_path: Path to the file containing the component
            component_name: Name of the component to extract

        Returns:
            Component AST representation
        """
        logger.info(f"Extracting component {component_name} from {file_path}")

        try:
            # Parse the file
            logger.debug(f"Parsing file: {file_path}")
            parsed = self.parse_file(file_path)

            # Extract the component
            logger.debug(f"Extracting component: {component_name}")
            ast_node = parsed["ast"]

            # Find the component in the AST
            component_node = None
            for node in ast.walk(ast_node):
                if (isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and
                    hasattr(node, 'name') and node.name == component_name):
                    component_node = node
                    break

            if component_node is None:
                logger.warning(f"Component {component_name} not found in {file_path}")
                return {
                    "error": f"Component {component_name} not found in {file_path}",
                    "component_name": component_name,
                    "file_path": str(file_path)
                }

            # Convert component to GPU-friendly format
            logger.debug(f"Converting component to GPU-friendly format")
            gpu_ast = self.adapter.convert_to_gpu_format(component_node)

            # Return component representation
            return {
                "ast": component_node,
                "gpu_ast": gpu_ast,
                "component_name": component_name,
                "file_path": str(file_path)
            }
        except Exception as e:
            error = handle_error(e)
            logger.error(f"Error extracting component {component_name} from {file_path}: {error}")
            raise GPUAnalysisComponentError(f"Error extracting component {component_name}", e)
        if hasattr(self.base_parser, 'get_cache_info'):
            base_parser_cache_info = self.base_parser.get_cache_info()
            info["base_parser"] = base_parser_cache_info
            info["size"] += base_parser_cache_info.get("size", 0)
            info["hits"] += base_parser_cache_info.get("hits", 0)
            info["misses"] += base_parser_cache_info.get("misses", 0)

        return info

@with_error_handling
def register_gpu_parser(language_registry):
    """
    Register GPU language parsers with SMA's language registry.

    This function registers the GPU language parser with SMA's language registry,
    enabling it to be used for parsing and analyzing code in SMA.

    Args:
        language_registry: SMA's language registry
    """
    logger.info("Registering GPU language parser with SMA")

    try:
        # Register the GPU language parser class directly
        language_registry.register_parser(GPULanguageParser)
        logger.info("GPU Language Parser registered with SMA")

        # Check if registration was successful
        registered_parsers = language_registry.get_registered_parsers()
        if GPULanguageParser in registered_parsers:
            logger.info("GPU Language Parser successfully registered")
        else:
            logger.warning("GPU Language Parser registration check failed")

        # Return the registered parser class for reference
        return GPULanguageParser
    except Exception as e:
        error = handle_error(e)
        logger.error(f"Error registering GPU language parser with SMA: {error}")

        # Fallback: try to register for each existing parser
        try:
            logger.info("Attempting fallback registration")

            # Get all parsers from the registry
            parsers = language_registry.get_all_parsers()
            logger.debug(f"Found {len(parsers)} existing parsers")

            if not parsers:
                # If no parsers are registered, register a standalone GPU parser
                logger.info("No existing parsers found, registering standalone GPU parser")
                gpu_parser = GPULanguageParser()
                language_registry.register_parser("PythonGPUParser", gpu_parser)
                logger.info("Registered standalone GPU language parser with SMA")
                return GPULanguageParser

            # Create GPU parser for each existing parser
            registered_count = 0
            for parser in parsers:
                try:
                    parser_name = parser.__class__.__name__
                    logger.debug(f"Creating GPU wrapper for {parser_name}")
                    gpu_parser = GPULanguageParser(parser)
                    wrapper_name = f"{parser_name}GPU"
                    language_registry.register_parser(wrapper_name, gpu_parser)
                    logger.info(f"Registered GPU wrapper for {parser_name}")
                    registered_count += 1
                except Exception as parser_error:
                    error = handle_error(parser_error)
                    logger.warning(f"Error registering GPU wrapper for {parser_name}: {error}")

            logger.info(f"Registered {registered_count} GPU wrappers")
            return GPULanguageParser
        except Exception as fallback_error:
            error = handle_error(fallback_error)
            logger.error(f"Error in fallback registration: {error}")
            return None
