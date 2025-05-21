"""
Inconsistency detection module for cross-file analysis.

This module provides functionality for detecting inconsistencies in the codebase.
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.cross_file.dependency_graph import DependencyGraph, Node, Edge

logger = logging.getLogger(__name__)


@dataclass
class Inconsistency:
    """An inconsistency detected in the codebase."""
    
    type: str  # "naming", "interface", "implementation", etc.
    description: str
    nodes: List[Node]
    severity: str  # "info", "warning", "error"
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "description": self.description,
            "nodes": [node.to_dict() for node in self.nodes],
            "severity": self.severity,
            "suggestion": self.suggestion
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], dependency_graph: DependencyGraph) -> 'Inconsistency':
        """Create from dictionary after deserialization.
        
        Args:
            data: The dictionary to create from.
            dependency_graph: The dependency graph to get nodes from.
            
        Returns:
            The created inconsistency.
        """
        nodes = []
        for node_data in data.get("nodes", []):
            node_id = node_data.get("id")
            if node_id and node_id in dependency_graph.nodes:
                nodes.append(dependency_graph.nodes[node_id])
        
        return cls(
            type=data["type"],
            description=data["description"],
            nodes=nodes,
            severity=data["severity"],
            suggestion=data.get("suggestion")
        )


class InconsistencyDetector:
    """Detects inconsistencies in the codebase."""
    
    def __init__(self, dependency_graph: DependencyGraph):
        """Initialize the inconsistency detector.
        
        Args:
            dependency_graph: The dependency graph to analyze.
        """
        self.dependency_graph = dependency_graph
    
    def detect_inconsistencies(self) -> List[Inconsistency]:
        """Detect inconsistencies in the codebase.
        
        Returns:
            A list of detected inconsistencies.
        """
        inconsistencies = []
        
        # Detect naming inconsistencies
        naming_inconsistencies = self._detect_naming_inconsistencies()
        inconsistencies.extend(naming_inconsistencies)
        
        # Detect interface inconsistencies
        interface_inconsistencies = self._detect_interface_inconsistencies()
        inconsistencies.extend(interface_inconsistencies)
        
        # Detect implementation inconsistencies
        implementation_inconsistencies = self._detect_implementation_inconsistencies()
        inconsistencies.extend(implementation_inconsistencies)
        
        # Detect documentation inconsistencies
        documentation_inconsistencies = self._detect_documentation_inconsistencies()
        inconsistencies.extend(documentation_inconsistencies)
        
        return inconsistencies
    
    def _detect_naming_inconsistencies(self) -> List[Inconsistency]:
        """Detect naming inconsistencies.
        
        Returns:
            A list of detected inconsistencies.
        """
        inconsistencies = []
        
        # Check for inconsistent naming conventions
        
        # 1. Check class naming conventions
        class_nodes = self.dependency_graph.get_nodes_by_type("class")
        
        # Group classes by naming convention
        pascal_case_classes = []
        snake_case_classes = []
        camel_case_classes = []
        other_case_classes = []
        
        for node in class_nodes:
            if self._is_pascal_case(node.name):
                pascal_case_classes.append(node)
            elif self._is_snake_case(node.name):
                snake_case_classes.append(node)
            elif self._is_camel_case(node.name):
                camel_case_classes.append(node)
            else:
                other_case_classes.append(node)
        
        # Check if there are multiple naming conventions
        class_conventions = []
        if pascal_case_classes:
            class_conventions.append(("PascalCase", pascal_case_classes))
        if snake_case_classes:
            class_conventions.append(("snake_case", snake_case_classes))
        if camel_case_classes:
            class_conventions.append(("camelCase", camel_case_classes))
        if other_case_classes:
            class_conventions.append(("other", other_case_classes))
        
        if len(class_conventions) > 1:
            # Find the dominant convention
            dominant_convention = max(class_conventions, key=lambda x: len(x[1]))
            
            # Create inconsistencies for non-dominant conventions
            for convention_name, convention_classes in class_conventions:
                if convention_name != dominant_convention[0]:
                    inconsistency = Inconsistency(
                        type="naming",
                        description=f"Inconsistent class naming convention: {len(convention_classes)} classes use {convention_name} instead of {dominant_convention[0]}",
                        nodes=convention_classes,
                        severity="warning",
                        suggestion=f"Rename classes to use {dominant_convention[0]} convention"
                    )
                    inconsistencies.append(inconsistency)
        
        # 2. Check function naming conventions
        function_nodes = self.dependency_graph.get_nodes_by_type("function")
        method_nodes = self.dependency_graph.get_nodes_by_type("method")
        
        # Group functions by naming convention
        pascal_case_functions = []
        snake_case_functions = []
        camel_case_functions = []
        other_case_functions = []
        
        for node in function_nodes + method_nodes:
            if self._is_pascal_case(node.name):
                pascal_case_functions.append(node)
            elif self._is_snake_case(node.name):
                snake_case_functions.append(node)
            elif self._is_camel_case(node.name):
                camel_case_functions.append(node)
            else:
                other_case_functions.append(node)
        
        # Check if there are multiple naming conventions
        function_conventions = []
        if pascal_case_functions:
            function_conventions.append(("PascalCase", pascal_case_functions))
        if snake_case_functions:
            function_conventions.append(("snake_case", snake_case_functions))
        if camel_case_functions:
            function_conventions.append(("camelCase", camel_case_functions))
        if other_case_functions:
            function_conventions.append(("other", other_case_functions))
        
        if len(function_conventions) > 1:
            # Find the dominant convention
            dominant_convention = max(function_conventions, key=lambda x: len(x[1]))
            
            # Create inconsistencies for non-dominant conventions
            for convention_name, convention_functions in function_conventions:
                if convention_name != dominant_convention[0]:
                    inconsistency = Inconsistency(
                        type="naming",
                        description=f"Inconsistent function naming convention: {len(convention_functions)} functions use {convention_name} instead of {dominant_convention[0]}",
                        nodes=convention_functions,
                        severity="warning",
                        suggestion=f"Rename functions to use {dominant_convention[0]} convention"
                    )
                    inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _is_pascal_case(self, name: str) -> bool:
        """Check if a name is in PascalCase.
        
        Args:
            name: The name to check.
            
        Returns:
            True if the name is in PascalCase, False otherwise.
        """
        return name and name[0].isupper() and "_" not in name
    
    def _is_snake_case(self, name: str) -> bool:
        """Check if a name is in snake_case.
        
        Args:
            name: The name to check.
            
        Returns:
            True if the name is in snake_case, False otherwise.
        """
        return name and name.islower() and "_" in name
    
    def _is_camel_case(self, name: str) -> bool:
        """Check if a name is in camelCase.
        
        Args:
            name: The name to check.
            
        Returns:
            True if the name is in camelCase, False otherwise.
        """
        return name and name[0].islower() and not name.islower() and "_" not in name
    
    def _detect_interface_inconsistencies(self) -> List[Inconsistency]:
        """Detect interface inconsistencies.
        
        Returns:
            A list of detected inconsistencies.
        """
        inconsistencies = []
        
        # Check for inconsistent interfaces
        
        # 1. Check for similar classes with different methods
        class_nodes = self.dependency_graph.get_nodes_by_type("class")
        
        # Group classes by name similarity
        similar_classes: Dict[str, List[Node]] = {}
        for node in class_nodes:
            # Extract the base name (without suffixes like "Impl", "Interface", etc.)
            base_name = self._extract_base_name(node.name)
            if base_name:
                if base_name not in similar_classes:
                    similar_classes[base_name] = []
                similar_classes[base_name].append(node)
        
        # Check for inconsistencies in similar classes
        for base_name, classes in similar_classes.items():
            if len(classes) > 1:
                # Get methods for each class
                class_methods: Dict[str, Set[str]] = {}
                for class_node in classes:
                    methods = set()
                    for edge in self.dependency_graph.get_outgoing_edges(class_node.id):
                        if edge.type == "contains":
                            target_node = self.dependency_graph.get_node(edge.target_id)
                            if target_node and target_node.type == "method":
                                methods.add(target_node.name)
                    class_methods[class_node.id] = methods
                
                # Check for inconsistencies
                if len(class_methods) > 1:
                    # Find methods that are in some classes but not others
                    all_methods = set()
                    for methods in class_methods.values():
                        all_methods.update(methods)
                    
                    inconsistent_classes = []
                    for class_node in classes:
                        missing_methods = all_methods - class_methods[class_node.id]
                        if missing_methods:
                            inconsistent_classes.append(class_node)
                    
                    if inconsistent_classes:
                        inconsistency = Inconsistency(
                            type="interface",
                            description=f"Inconsistent interfaces for similar classes: {', '.join(node.name for node in inconsistent_classes)}",
                            nodes=inconsistent_classes,
                            severity="warning",
                            suggestion="Ensure all similar classes have consistent interfaces"
                        )
                        inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _extract_base_name(self, name: str) -> Optional[str]:
        """Extract the base name from a class name.
        
        Args:
            name: The class name.
            
        Returns:
            The base name, or None if not found.
        """
        # Remove common suffixes
        suffixes = ["Impl", "Implementation", "Interface", "Abstract", "Base", "Default"]
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        
        return name
    
    def _detect_implementation_inconsistencies(self) -> List[Inconsistency]:
        """Detect implementation inconsistencies.
        
        Returns:
            A list of detected inconsistencies.
        """
        inconsistencies = []
        
        # Check for inconsistent implementations
        
        # 1. Check for similar methods with different implementations
        method_nodes = self.dependency_graph.get_nodes_by_type("method")
        
        # Group methods by name
        methods_by_name: Dict[str, List[Node]] = {}
        for node in method_nodes:
            if node.name not in methods_by_name:
                methods_by_name[node.name] = []
            methods_by_name[node.name].append(node)
        
        # Check for methods with the same name but in different classes
        for method_name, methods in methods_by_name.items():
            if len(methods) > 1:
                # Group methods by class
                methods_by_class: Dict[str, List[Node]] = {}
                for method_node in methods:
                    class_name = method_node.attributes.get("class")
                    if class_name:
                        if class_name not in methods_by_class:
                            methods_by_class[class_name] = []
                        methods_by_class[class_name].append(method_node)
                
                # Check for methods in different classes
                if len(methods_by_class) > 1:
                    # This is a potential inconsistency, but we need more information
                    # to determine if it's actually inconsistent
                    # For now, we'll just flag it as an info
                    inconsistency = Inconsistency(
                        type="implementation",
                        description=f"Method '{method_name}' is implemented in multiple classes: {', '.join(methods_by_class.keys())}",
                        nodes=methods,
                        severity="info",
                        suggestion="Verify that the implementations are consistent"
                    )
                    inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _detect_documentation_inconsistencies(self) -> List[Inconsistency]:
        """Detect documentation inconsistencies.
        
        Returns:
            A list of detected inconsistencies.
        """
        inconsistencies = []
        
        # Check for inconsistent documentation
        
        # 1. Check for classes without documentation
        class_nodes = self.dependency_graph.get_nodes_by_type("class")
        
        # This is a placeholder implementation
        # In a real implementation, we would need to analyze the actual code
        # to determine if a class has documentation
        
        return inconsistencies
