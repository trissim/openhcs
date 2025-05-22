"""
GPU-Accelerated Semantic Analyzer

This module provides a GPU-accelerated implementation of semantic code analysis.
It uses PyTorch to accelerate the analysis process and can run on both CPU and GPU.

This implementation uses the two-phase approach described in the paper
"Parallel Lexing, Parsing and Semantic Analysis on the GPU" by R. F. Voetter
for efficient semantic analysis on GPU.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import ast
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .complexity_analyzer import ComplexityAnalyzer
from .dependency_analyzer import DependencyAnalyzer

# Set up logging
logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """
    GPU-accelerated semantic analyzer.

    This class provides methods for semantic analysis of code using GPU acceleration,
    including AST traversal, token scoring, and pattern weight computation.

    This implementation uses the two-phase approach described in the paper
    "Parallel Lexing, Parsing and Semantic Analysis on the GPU" by R. F. Voetter
    for efficient semantic analysis on GPU.

    Attributes:
        device: Device to use for analysis ("cuda" or "cpu")
        config: Configuration for the analyzer
        memory_manager: GPU memory manager to keep data in GPU memory
        complexity_analyzer: Complexity analyzer
        dependency_analyzer: Dependency analyzer
        type_checker: Type checker using two-phase approach
        variable_resolver: Variable reference resolver
        function_resolver: Function call resolver
    """

    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the semantic analyzer.

        Args:
            device: Device to use for analysis ("cuda" or "cpu")
            config: Configuration for the analyzer
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}

        # Initialize analyzers
        self.complexity_analyzer = ComplexityAnalyzer(device=self.device)
        self.dependency_analyzer = DependencyAnalyzer(device=self.device)

        # Initialize pattern matcher
        try:
            from ..pattern_matcher import GPUPatternMatcherRegistry
            self.pattern_matcher = GPUPatternMatcherRegistry(device=self.device)
        except ImportError:
            from ..patterns import GPUPatternMatcher
            self.pattern_matcher = GPUPatternMatcher(device=self.device)

        # Initialize type checker, variable resolver, and function resolver
        # Use stub implementations for now
        self.type_checker = None
        self.variable_resolver = None
        self.function_resolver = None

        # Initialize patterns
        self.patterns = []

    def add_pattern(self, pattern):
        """
        Add a pattern for matching.

        Args:
            pattern: Pattern to add
        """
        self.patterns.append(pattern)

    def clear_patterns(self):
        """Clear all patterns."""
        self.patterns = []

    def analyze(
        self,
        code: str,
        file_path: Optional[Union[str, Path]] = None,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze code semantically using GPU acceleration.

        Args:
            code: Python code to analyze
            file_path: Optional path to the file
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary of analysis results
        """
        start_time = time.time()

        # Convert file path to Path
        if file_path is not None:
            file_path = Path(file_path)
        else:
            file_path = Path("input.py")

        # Tensorize the AST
        ast_tensors = self._tensorize_ast(code)

        # Determine which analyses to run
        if analysis_types is None:
            analysis_types = ["complexity", "dependency", "pattern", "type", "variable", "function"]

        # Run analyses
        results = {}

        # Complexity analysis
        if "complexity" in analysis_types:
            complexity_metrics = self.complexity_analyzer(ast_tensors)
            results["complexity"] = {
                key: value.detach().item() for key, value in complexity_metrics.items()
            }

        # Dependency analysis
        if "dependency" in analysis_types:
            dependency_matrices = self.dependency_analyzer(ast_tensors)
            results["dependency"] = {
                key: value.detach().cpu().numpy() if value.numel() > 0 else np.array([])
                for key, value in dependency_matrices.items()
            }

        # Type checking (two-phase approach)
        if "type" in analysis_types and self.type_checker is not None:
            type_results = self.type_checker.check_types(ast_tensors)
            results["types"] = {
                "types": type_results["types"].detach().cpu().numpy(),
                "errors": type_results["errors"].detach().cpu().numpy()
            }
        elif "type" in analysis_types:
            # Stub implementation
            results["types"] = {
                "types": np.array([]),
                "errors": np.array([])
            }

        # Variable resolution
        if "variable" in analysis_types and self.variable_resolver is not None:
            variable_results = self.variable_resolver.resolve_variables(ast_tensors)
            results["variables"] = {
                "declarations": variable_results["declarations"].detach().cpu().numpy(),
                "references": variable_results["references"].detach().cpu().numpy(),
                "resolution": variable_results["resolution"].detach().cpu().numpy()
            }
        elif "variable" in analysis_types:
            # Stub implementation
            results["variables"] = {
                "declarations": np.array([]),
                "references": np.array([]),
                "resolution": np.array([])
            }

        # Function resolution
        if "function" in analysis_types and self.function_resolver is not None:
            function_results = self.function_resolver.resolve_functions(ast_tensors)
            results["functions"] = {
                "declarations": function_results["declarations"].detach().cpu().numpy(),
                "calls": function_results["calls"].detach().cpu().numpy(),
                "resolution": function_results["resolution"].detach().cpu().numpy()
            }
        elif "function" in analysis_types:
            # Stub implementation
            results["functions"] = {
                "declarations": np.array([]),
                "calls": np.array([]),
                "resolution": np.array([])
            }

        # Pattern matching
        if "pattern" in analysis_types and self.patterns:
            try:
                pattern_matches = self.pattern_matcher.match_patterns(
                    self.patterns, file_path, code, ast_tensors
                )
                results["pattern_matches"] = pattern_matches
            except AttributeError:
                # Fall back to old pattern matcher interface
                pattern_matches = []
                for pattern in self.patterns:
                    matches = self.pattern_matcher.match_pattern(
                        pattern, file_path, code, ast_tensors
                    )
                    pattern_matches.extend(matches)
                results["pattern_matches"] = pattern_matches

        # Add metadata
        results["metadata"] = {
            "analysis_time": time.time() - start_time,
            "code_length": len(code),
            "device": self.device
        }

        return results

    def _tensorize_ast(self, code: str) -> Dict[str, torch.Tensor]:
        """
        Tensorize AST.

        Args:
            code: Python code to tensorize

        Returns:
            Dictionary of AST tensors
        """
        # Tensorize AST
        try:
            # Try to use GPU-optimized tensorizer first
            from ..ast_tensor import GPUASTTensorizer
            tensorizer = GPUASTTensorizer(device=self.device)
            ast_tensors = tensorizer.tensorize(code)

            # Convert parent pointers to edges for compatibility with complexity analyzer
            if "parents" in ast_tensors and "edges" not in ast_tensors:
                parents = ast_tensors["parents"]
                # Create edges from parent pointers
                edges = []
                for i, parent in enumerate(parents):
                    if parent >= 0:  # Skip root node
                        edges.append([parent.item(), i])

                if edges:
                    ast_tensors["edges"] = torch.tensor(edges, dtype=torch.int32, device=self.device)
                else:
                    # Create empty edges tensor with correct shape
                    ast_tensors["edges"] = torch.zeros((0, 2), dtype=torch.int32, device=self.device)

        except (ImportError, Exception) as e:
            # Fall back to regular tensorizer
            logger.warning(f"Falling back to regular tensorizer: {e}")
            from ..ast_tensor import ASTTensorizer
            tensorizer = ASTTensorizer(device=self.device)
            ast_tensors = tensorizer.tensorize(code)

        # Ensure all required keys are present
        required_keys = ["nodes", "edges", "features"]
        for key in required_keys:
            if key not in ast_tensors:
                if key == "edges":
                    # Create empty edges tensor with correct shape
                    ast_tensors[key] = torch.zeros((0, 2), dtype=torch.int32, device=self.device)
                elif key == "nodes":
                    # Create empty nodes tensor
                    ast_tensors[key] = torch.tensor([], dtype=torch.int32, device=self.device)
                elif key == "features":
                    # Create empty features tensor with correct feature dimension
                    feature_dim = self.config.get("feature_dim", 10)
                    ast_tensors[key] = torch.zeros((0, feature_dim), dtype=torch.float32, device=self.device)

        return ast_tensors

    def analyze_file(
        self,
        file_path: Union[str, Path],
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a file semantically.

        Args:
            file_path: Path to the file
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary of analysis results
        """
        # Convert file path to Path
        file_path = Path(file_path)

        # Read file
        with open(file_path, 'r') as f:
            code = f.read()

        # Analyze code
        return self.analyze(code, file_path, analysis_types)

    def batch_analyze(
        self,
        codes: List[str],
        file_paths: Optional[List[Union[str, Path]]] = None,
        analysis_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple code snippets in batch.

        Args:
            codes: List of Python code snippets
            file_paths: Optional list of file paths
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            List of analysis results dictionaries
        """
        # Convert file paths to Paths
        if file_paths is not None:
            file_paths = [Path(path) for path in file_paths]
        else:
            file_paths = [Path(f"input_{i}.py") for i in range(len(codes))]

        results = []

        # Get batch size from config or use default
        batch_size = self.config.get("batch_size", 16)

        # Process in batches for better GPU utilization
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]
            batch_paths = file_paths[i:i+batch_size]

            # Tensorize all ASTs in the batch
            batch_ast_tensors = []
            for code in batch_codes:
                ast_tensors = self._tensorize_ast(code)
                batch_ast_tensors.append(ast_tensors)

            # Analyze each code snippet
            batch_results = []
            for code, path, ast_tensors in zip(batch_codes, batch_paths, batch_ast_tensors):
                result = self.analyze(code, path, analysis_types)
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def extract_semantic_features(self, code: str) -> Dict[str, Any]:
        """
        Extract semantic features from code.

        Args:
            code: Python code to analyze

        Returns:
            Dictionary of semantic features
        """
        # Analyze code
        analysis = self.analyze(code)

        # Extract features
        features = {}

        # Complexity features
        if "complexity" in analysis:
            features.update(analysis["complexity"])

        # Dependency features
        if "dependency" in analysis:
            # Extract number of dependencies
            if "function_dependencies" in analysis["dependency"]:
                deps = analysis["dependency"]["function_dependencies"]
                if deps.size > 0:
                    # Make sure we're working with a numpy array, not a tensor
                    if isinstance(deps, torch.Tensor):
                        deps = deps.detach().cpu().numpy()
                    features["num_function_dependencies"] = np.sum(deps > 0.5)
                else:
                    features["num_function_dependencies"] = 0

        # Type features
        if "types" in analysis:
            # Extract number of type errors
            if "errors" in analysis["types"]:
                errors = analysis["types"]["errors"]
                # Make sure we're working with a numpy array, not a tensor
                if isinstance(errors, torch.Tensor):
                    errors = errors.detach().cpu().numpy()
                features["num_type_errors"] = np.sum(errors)

        # Variable features
        if "variables" in analysis:
            # Extract number of variable declarations and references
            if "declarations" in analysis["variables"]:
                features["num_variable_declarations"] = len(analysis["variables"]["declarations"])
            if "references" in analysis["variables"]:
                features["num_variable_references"] = len(analysis["variables"]["references"])

        # Function features
        if "functions" in analysis:
            # Extract number of function declarations and calls
            if "declarations" in analysis["functions"]:
                features["num_function_declarations"] = len(analysis["functions"]["declarations"])
            if "calls" in analysis["functions"]:
                features["num_function_calls"] = len(analysis["functions"]["calls"])

        # Pattern features
        if "pattern_matches" in analysis:
            features["num_pattern_matches"] = len(analysis["pattern_matches"])

        return features

    def cleanup(self) -> None:
        """
        Clean up resources used by the semantic analyzer.

        This method should be called when the analyzer is no longer needed
        to free up GPU memory and other resources.
        """
        # Clean up GPU memory
        if self.device == "cuda":
            try:
                # Clean up any cached tensors
                torch.cuda.empty_cache()

                # Clean up component resources
                if hasattr(self.complexity_analyzer, 'cleanup'):
                    self.complexity_analyzer.cleanup()

                if hasattr(self.dependency_analyzer, 'cleanup'):
                    self.dependency_analyzer.cleanup()

                if hasattr(self.pattern_matcher, 'cleanup'):
                    self.pattern_matcher.cleanup()

                if self.type_checker is not None and hasattr(self.type_checker, 'cleanup'):
                    self.type_checker.cleanup()

                if self.variable_resolver is not None and hasattr(self.variable_resolver, 'cleanup'):
                    self.variable_resolver.cleanup()

                if self.function_resolver is not None and hasattr(self.function_resolver, 'cleanup'):
                    self.function_resolver.cleanup()

                logger.info("Semantic analyzer resources cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up semantic analyzer resources: {e}")

    def clear_cache(self) -> None:
        """
        Clear any cached data.
        """
        # Clear pattern matcher cache
        if hasattr(self.pattern_matcher, 'clear_cache'):
            self.pattern_matcher.clear_cache()

        # Clear other component caches
        if hasattr(self.complexity_analyzer, 'clear_cache'):
            self.complexity_analyzer.clear_cache()

        if hasattr(self.dependency_analyzer, 'clear_cache'):
            self.dependency_analyzer.clear_cache()

        # Clear GPU memory cache
        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Semantic analyzer cache cleared")

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

        # Get pattern matcher cache info
        if hasattr(self.pattern_matcher, 'get_cache_info'):
            pattern_cache_info = self.pattern_matcher.get_cache_info()
            info["pattern_matcher"] = pattern_cache_info
            info["size"] += pattern_cache_info.get("size", 0)
            info["hits"] += pattern_cache_info.get("hits", 0)
            info["misses"] += pattern_cache_info.get("misses", 0)

        # Get other component cache info
        if hasattr(self.complexity_analyzer, 'get_cache_info'):
            complexity_cache_info = self.complexity_analyzer.get_cache_info()
            info["complexity_analyzer"] = complexity_cache_info
            info["size"] += complexity_cache_info.get("size", 0)
            info["hits"] += complexity_cache_info.get("hits", 0)
            info["misses"] += complexity_cache_info.get("misses", 0)

        if hasattr(self.dependency_analyzer, 'get_cache_info'):
            dependency_cache_info = self.dependency_analyzer.get_cache_info()
            info["dependency_analyzer"] = dependency_cache_info
            info["size"] += dependency_cache_info.get("size", 0)
            info["hits"] += dependency_cache_info.get("hits", 0)
            info["misses"] += dependency_cache_info.get("misses", 0)

        return info


class TypeChecker:
    """
    GPU-accelerated type checker.

    This class implements the two-phase type checking approach described in the Voetter paper.

    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the type checker.

        Args:
            device: Device to place tensors on ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    def check_types(self, ast_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Check types in an AST using the two-phase approach.

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Dictionary of type information
        """
        # Phase 1: Assign initial types to all nodes
        initial_types = self._assign_initial_types(ast_tensors)

        # Phase 2: Verify type compatibility
        type_errors = self._verify_type_compatibility(ast_tensors, initial_types)

        result = {
            "types": initial_types,
            "errors": type_errors
        }

        return result

    def _assign_initial_types(self, ast_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Assign initial types to all nodes in parallel.

        This is the first phase of the two-phase type checking approach.

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Tensor of initial types
        """
        nodes = ast_tensors["nodes"]
        node_types = ast_tensors["node_types"]

        # Create a tensor to store the type of each node
        initial_types = torch.zeros(nodes.size(0), dtype=torch.int64, device=self.device)

        # Assign types based on node type
        # This can be done in parallel for all nodes

        # Import AST node types
        from ..ast_tensor import AST_NODE_TYPES

        # Example: Assign type 1 to all literal nodes
        literal_mask = (node_types == AST_NODE_TYPES.get(ast.Constant, 0))
        initial_types[literal_mask] = 1

        # Example: Assign type 2 to all variable nodes
        variable_mask = (node_types == AST_NODE_TYPES.get(ast.Name, 0))
        initial_types[variable_mask] = 2

        # Example: Assign type 3 to all function nodes
        function_mask = (node_types == AST_NODE_TYPES.get(ast.FunctionDef, 0))
        initial_types[function_mask] = 3

        # And so on for other node types...

        return initial_types

    def _verify_type_compatibility(self, ast_tensors: Dict[str, torch.Tensor],
                                  initial_types: torch.Tensor) -> torch.Tensor:
        """
        Verify type compatibility in parallel.

        This is the second phase of the two-phase type checking approach.

        Args:
            ast_tensors: Dictionary of AST tensors
            initial_types: Tensor of initial types

        Returns:
            Tensor of type errors
        """
        nodes = ast_tensors["nodes"]
        parents = ast_tensors["parents"]
        node_types = ast_tensors["node_types"]

        # Create a tensor to store type errors
        type_errors = torch.zeros(nodes.size(0), dtype=torch.bool, device=self.device)

        # Import AST node types
        from ..ast_tensor import AST_NODE_TYPES

        # Verify type compatibility for each node
        # This can be done in parallel for all nodes

        # Example: Verify that binary operations have compatible operands
        binary_op_mask = (node_types == AST_NODE_TYPES.get(ast.BinOp, 0))
        binary_op_indices = torch.nonzero(binary_op_mask).squeeze(-1)

        for node_idx in binary_op_indices:
            # Find children of this node
            children_mask = (parents == node_idx)
            children = torch.nonzero(children_mask).squeeze(-1)

            if children.size(0) >= 2:
                # Get the types of the operands
                left_type = initial_types[children[0]]
                right_type = initial_types[children[1]]

                # Check if types are compatible
                if not self._are_types_compatible(left_type, right_type):
                    type_errors[node_idx] = True

        # And so on for other type compatibility checks...

        return type_errors

    def _are_types_compatible(self, type1: int, type2: int) -> bool:
        """
        Check if two types are compatible.

        Args:
            type1: First type
            type2: Second type

        Returns:
            True if the types are compatible, False otherwise
        """
        # For now, we'll implement a simplified version
        # In the future, we could implement a more sophisticated type compatibility check

        # Example: Types are compatible if they are the same
        return type1 == type2


class VariableResolver:
    """
    GPU-accelerated variable resolver.

    This class resolves variable references to their declarations in parallel.

    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the variable resolver.

        Args:
            device: Device to place tensors on ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    def resolve_variables(self, ast_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Resolve variable references to their declarations in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Dictionary mapping variable references to declarations
        """
        # Find all variable declarations
        declarations = self._find_variable_declarations(ast_tensors)

        # Find all variable references
        references = self._find_variable_references(ast_tensors)

        # Resolve references to declarations
        resolution = self._resolve_references(ast_tensors, declarations, references)

        result = {
            "declarations": declarations,
            "references": references,
            "resolution": resolution
        }

        return result

    def _find_variable_declarations(self, ast_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Find all variable declarations in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Tensor of variable declaration indices
        """
        # Import AST node types
        from ..ast_tensor import AST_NODE_TYPES

        node_types = ast_tensors["node_types"]
        parents = ast_tensors["parents"]

        # Find all assignment nodes
        assign_mask = (node_types == AST_NODE_TYPES.get(ast.Assign, 0))
        assign_indices = torch.nonzero(assign_mask).squeeze(-1)

        # Find all variable declarations (names on the left side of assignments)
        declarations = []

        for assign_idx in assign_indices:
            # Find children of this assignment
            children_mask = (parents == assign_idx)
            children = torch.nonzero(children_mask).squeeze(-1)

            # The first child is the target (left side)
            if children.size(0) > 0:
                target_idx = children[0]

                # If the target is a name, it's a variable declaration
                if node_types[target_idx] == AST_NODE_TYPES.get(ast.Name, 0):
                    declarations.append(target_idx)

        return torch.tensor(declarations, device=self.device)

    def _find_variable_references(self, ast_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Find all variable references in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Tensor of variable reference indices
        """
        # Import AST node types
        from ..ast_tensor import AST_NODE_TYPES

        node_types = ast_tensors["node_types"]

        # Find all name nodes
        name_mask = (node_types == AST_NODE_TYPES.get(ast.Name, 0))
        name_indices = torch.nonzero(name_mask).squeeze(-1)

        return name_indices

    def _resolve_references(self, ast_tensors: Dict[str, torch.Tensor],
                           declarations: torch.Tensor,
                           references: torch.Tensor) -> torch.Tensor:
        """
        Resolve references to declarations in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors
            declarations: Tensor of variable declaration indices
            references: Tensor of variable reference indices

        Returns:
            Tensor mapping each reference to its declaration
        """
        # For now, we'll implement a simplified version
        # In the future, we could implement a more sophisticated resolution algorithm

        # Create a tensor to store the resolution
        resolution = torch.full((references.size(0),), -1, dtype=torch.int64, device=self.device)

        # For each reference, find the corresponding declaration
        for i, ref_idx in enumerate(references):
            # For now, just assign the first declaration
            if declarations.size(0) > 0:
                resolution[i] = declarations[0]

        return resolution


class FunctionResolver:
    """
    GPU-accelerated function resolver.

    This class resolves function calls to their declarations in parallel.

    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the function resolver.

        Args:
            device: Device to place tensors on ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    def resolve_functions(self, ast_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Resolve function calls to their declarations in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Dictionary mapping function calls to declarations
        """
        # Find all function declarations
        declarations = self._find_function_declarations(ast_tensors)

        # Find all function calls
        calls = self._find_function_calls(ast_tensors)

        # Resolve calls to declarations
        resolution = self._resolve_calls(ast_tensors, declarations, calls)

        result = {
            "declarations": declarations,
            "calls": calls,
            "resolution": resolution
        }

        return result

    def _find_function_declarations(self, ast_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Find all function declarations in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Tensor of function declaration indices
        """
        # Import AST node types
        from ..ast_tensor import AST_NODE_TYPES

        node_types = ast_tensors["node_types"]

        # Find all function definition nodes
        func_def_mask = (node_types == AST_NODE_TYPES.get(ast.FunctionDef, 0))
        func_def_indices = torch.nonzero(func_def_mask).squeeze(-1)

        return func_def_indices

    def _find_function_calls(self, ast_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Find all function calls in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors

        Returns:
            Tensor of function call indices
        """
        # Import AST node types
        from ..ast_tensor import AST_NODE_TYPES

        node_types = ast_tensors["node_types"]

        # Find all call nodes
        call_mask = (node_types == AST_NODE_TYPES.get(ast.Call, 0))
        call_indices = torch.nonzero(call_mask).squeeze(-1)

        return call_indices

    def _resolve_calls(self, ast_tensors: Dict[str, torch.Tensor],
                      declarations: torch.Tensor,
                      calls: torch.Tensor) -> torch.Tensor:
        """
        Resolve calls to declarations in parallel.

        Args:
            ast_tensors: Dictionary of AST tensors
            declarations: Tensor of function declaration indices
            calls: Tensor of function call indices

        Returns:
            Tensor mapping each call to its declaration
        """
        # For now, we'll implement a simplified version
        # In the future, we could implement a more sophisticated resolution algorithm

        # Create a tensor to store the resolution
        resolution = torch.full((calls.size(0),), -1, dtype=torch.int64, device=self.device)

        # For each call, find the corresponding declaration
        for i, call_idx in enumerate(calls):
            # For now, just assign the first declaration
            if declarations.size(0) > 0:
                resolution[i] = declarations[0]

        return resolution
