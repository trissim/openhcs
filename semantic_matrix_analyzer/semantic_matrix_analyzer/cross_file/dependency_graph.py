"""
Dependency graph module for cross-file analysis.

This module provides functionality for constructing and analyzing a dependency graph
of the codebase, representing relationships between files, classes, functions, etc.
"""

import ast
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """A node in the dependency graph."""
    
    id: str
    type: str  # "file", "class", "function", "variable", etc.
    name: str
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "attributes": self.attributes
        }
        
        if self.file_path:
            result["file_path"] = str(self.file_path)
        
        if self.line_number:
            result["line_number"] = self.line_number
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create from dictionary after deserialization."""
        file_path = data.get("file_path")
        if file_path:
            file_path = Path(file_path)
        
        return cls(
            id=data["id"],
            type=data["type"],
            name=data["name"],
            file_path=file_path,
            line_number=data.get("line_number"),
            attributes=data.get("attributes", {})
        )


@dataclass
class Edge:
    """An edge in the dependency graph."""
    
    source_id: str
    target_id: str
    type: str  # "imports", "calls", "inherits", "references", etc.
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        """Create from dictionary after deserialization."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            attributes=data.get("attributes", {})
        )


class DependencyGraph:
    """A graph of dependencies between files, classes, functions, etc."""
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize the dependency graph.
        
        Args:
            storage_path: Path to store the dependency graph.
        """
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.node_index: Dict[str, List[str]] = {}  # type -> [node_id]
        self.edge_index: Dict[str, List[Edge]] = {}  # type -> [edge]
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Load the dependency graph if a storage path is provided
        if self.storage_path and self.storage_path.exists():
            self.load()
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph.
        
        Args:
            node: The node to add.
        """
        self.nodes[node.id] = node
        
        # Update the node index
        if node.type not in self.node_index:
            self.node_index[node.type] = []
        self.node_index[node.type].append(node.id)
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph.
        
        Args:
            edge: The edge to add.
        """
        self.edges.append(edge)
        
        # Update the edge index
        if edge.type not in self.edge_index:
            self.edge_index[edge.type] = []
        self.edge_index[edge.type].append(edge)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID.
        
        Args:
            node_id: The ID of the node.
            
        Returns:
            The node, or None if not found.
        """
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type.
        
        Args:
            node_type: The type of nodes to get.
            
        Returns:
            A list of nodes of the specified type.
        """
        node_ids = self.node_index.get(node_type, [])
        return [self.nodes[node_id] for node_id in node_ids]
    
    def get_edges_by_type(self, edge_type: str) -> List[Edge]:
        """Get all edges of a specific type.
        
        Args:
            edge_type: The type of edges to get.
            
        Returns:
            A list of edges of the specified type.
        """
        return self.edge_index.get(edge_type, [])
    
    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get all edges originating from a node.
        
        Args:
            node_id: The ID of the node.
            
        Returns:
            A list of edges originating from the node.
        """
        return [edge for edge in self.edges if edge.source_id == node_id]
    
    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Get all edges targeting a node.
        
        Args:
            node_id: The ID of the node.
            
        Returns:
            A list of edges targeting the node.
        """
        return [edge for edge in self.edges if edge.target_id == node_id]
    
    def get_neighbors(self, node_id: str, direction: str = "outgoing") -> List[Node]:
        """Get all neighbors of a node.
        
        Args:
            node_id: The ID of the node.
            direction: The direction of the edges ("outgoing" or "incoming").
            
        Returns:
            A list of neighboring nodes.
        """
        if direction == "outgoing":
            edges = self.get_outgoing_edges(node_id)
            return [self.nodes[edge.target_id] for edge in edges if edge.target_id in self.nodes]
        elif direction == "incoming":
            edges = self.get_incoming_edges(node_id)
            return [self.nodes[edge.source_id] for edge in edges if edge.source_id in self.nodes]
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def save(self) -> None:
        """Save the dependency graph to storage."""
        if not self.storage_path:
            logger.warning("No storage path provided, dependency graph not saved.")
            return
        
        try:
            # Create the parent directory if it doesn't exist
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the dependency graph to a file
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
                    "edges": [edge.to_dict() for edge in self.edges]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving dependency graph: {e}")
    
    def load(self) -> None:
        """Load the dependency graph from storage."""
        if not self.storage_path or not self.storage_path.exists():
            logger.warning("No storage path provided or file does not exist, dependency graph not loaded.")
            return
        
        try:
            # Load the dependency graph from a file
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Clear existing data
                self.nodes = {}
                self.edges = []
                self.node_index = {}
                self.edge_index = {}
                
                # Load nodes
                for node_id, node_data in data.get("nodes", {}).items():
                    node = Node.from_dict(node_data)
                    self.add_node(node)
                
                # Load edges
                for edge_data in data.get("edges", []):
                    edge = Edge.from_dict(edge_data)
                    self.add_edge(edge)
        except Exception as e:
            logger.error(f"Error loading dependency graph: {e}")


class DependencyExtractor:
    """Extracts dependencies from code."""
    
    def __init__(self):
        """Initialize the dependency extractor."""
        pass
    
    def extract_dependencies(self, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract dependencies from a file.
        
        Args:
            file_path: The path to the file.
            
        Returns:
            A tuple of (nodes, edges).
        """
        # Check if the file exists
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return [], []
        
        # Check if the file is a Python file
        if file_path.suffix != ".py":
            logger.warning(f"Not a Python file: {file_path}")
            return [], []
        
        try:
            # Read the file
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            
            # Parse the file
            tree = ast.parse(code, filename=str(file_path))
            
            # Extract nodes and edges
            nodes = []
            edges = []
            
            # Add a node for the file
            file_node = Node(
                id=f"file:{file_path}",
                type="file",
                name=file_path.name,
                file_path=file_path
            )
            nodes.append(file_node)
            
            # Extract imports
            import_nodes, import_edges = self._extract_imports(tree, file_node.id, file_path)
            nodes.extend(import_nodes)
            edges.extend(import_edges)
            
            # Extract classes
            class_nodes, class_edges = self._extract_classes(tree, file_node.id, file_path)
            nodes.extend(class_nodes)
            edges.extend(class_edges)
            
            # Extract functions
            function_nodes, function_edges = self._extract_functions(tree, file_node.id, file_path)
            nodes.extend(function_nodes)
            edges.extend(function_edges)
            
            # Extract variables
            variable_nodes, variable_edges = self._extract_variables(tree, file_node.id, file_path)
            nodes.extend(variable_nodes)
            edges.extend(variable_edges)
            
            return nodes, edges
        except Exception as e:
            logger.error(f"Error extracting dependencies from {file_path}: {e}")
            return [], []
    
    def _extract_imports(self, tree: ast.AST, file_id: str, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract import dependencies from an AST node.
        
        Args:
            tree: The AST node.
            file_id: The ID of the file node.
            file_path: The path to the file.
            
        Returns:
            A tuple of (nodes, edges).
        """
        nodes = []
        edges = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Create a node for the imported module
                    module_id = f"module:{name.name}"
                    module_node = Node(
                        id=module_id,
                        type="module",
                        name=name.name
                    )
                    nodes.append(module_node)
                    
                    # Create an edge from the file to the module
                    edge = Edge(
                        source_id=file_id,
                        target_id=module_id,
                        type="imports"
                    )
                    edges.append(edge)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Create a node for the imported module
                    module_id = f"module:{node.module}"
                    module_node = Node(
                        id=module_id,
                        type="module",
                        name=node.module
                    )
                    nodes.append(module_node)
                    
                    # Create an edge from the file to the module
                    edge = Edge(
                        source_id=file_id,
                        target_id=module_id,
                        type="imports"
                    )
                    edges.append(edge)
                    
                    # Create nodes and edges for imported names
                    for name in node.names:
                        # Create a node for the imported name
                        name_id = f"name:{node.module}.{name.name}"
                        name_node = Node(
                            id=name_id,
                            type="name",
                            name=name.name,
                            attributes={"module": node.module}
                        )
                        nodes.append(name_node)
                        
                        # Create an edge from the file to the name
                        edge = Edge(
                            source_id=file_id,
                            target_id=name_id,
                            type="imports"
                        )
                        edges.append(edge)
                        
                        # Create an edge from the module to the name
                        edge = Edge(
                            source_id=module_id,
                            target_id=name_id,
                            type="contains"
                        )
                        edges.append(edge)
        
        return nodes, edges
    
    def _extract_classes(self, tree: ast.AST, file_id: str, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract class dependencies from an AST node.
        
        Args:
            tree: The AST node.
            file_id: The ID of the file node.
            file_path: The path to the file.
            
        Returns:
            A tuple of (nodes, edges).
        """
        nodes = []
        edges = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Create a node for the class
                class_id = f"class:{file_path.stem}.{node.name}"
                class_node = Node(
                    id=class_id,
                    type="class",
                    name=node.name,
                    file_path=file_path,
                    line_number=node.lineno
                )
                nodes.append(class_node)
                
                # Create an edge from the file to the class
                edge = Edge(
                    source_id=file_id,
                    target_id=class_id,
                    type="contains"
                )
                edges.append(edge)
                
                # Extract base classes
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        # Create a node for the base class
                        base_id = f"class:{base.id}"
                        base_node = Node(
                            id=base_id,
                            type="class",
                            name=base.id
                        )
                        nodes.append(base_node)
                        
                        # Create an edge from the class to the base class
                        edge = Edge(
                            source_id=class_id,
                            target_id=base_id,
                            type="inherits"
                        )
                        edges.append(edge)
        
        return nodes, edges
    
    def _extract_functions(self, tree: ast.AST, file_id: str, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract function dependencies from an AST node.
        
        Args:
            tree: The AST node.
            file_id: The ID of the file node.
            file_path: The path to the file.
            
        Returns:
            A tuple of (nodes, edges).
        """
        nodes = []
        edges = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if this is a method
                is_method = False
                parent_class = None
                
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef) and node in parent.body:
                        is_method = True
                        parent_class = parent
                        break
                
                if is_method:
                    # Create a node for the method
                    method_id = f"method:{file_path.stem}.{parent_class.name}.{node.name}"
                    method_node = Node(
                        id=method_id,
                        type="method",
                        name=node.name,
                        file_path=file_path,
                        line_number=node.lineno,
                        attributes={"class": parent_class.name}
                    )
                    nodes.append(method_node)
                    
                    # Create an edge from the class to the method
                    class_id = f"class:{file_path.stem}.{parent_class.name}"
                    edge = Edge(
                        source_id=class_id,
                        target_id=method_id,
                        type="contains"
                    )
                    edges.append(edge)
                else:
                    # Create a node for the function
                    function_id = f"function:{file_path.stem}.{node.name}"
                    function_node = Node(
                        id=function_id,
                        type="function",
                        name=node.name,
                        file_path=file_path,
                        line_number=node.lineno
                    )
                    nodes.append(function_node)
                    
                    # Create an edge from the file to the function
                    edge = Edge(
                        source_id=file_id,
                        target_id=function_id,
                        type="contains"
                    )
                    edges.append(edge)
        
        return nodes, edges
    
    def _extract_variables(self, tree: ast.AST, file_id: str, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract variable dependencies from an AST node.
        
        Args:
            tree: The AST node.
            file_id: The ID of the file node.
            file_path: The path to the file.
            
        Returns:
            A tuple of (nodes, edges).
        """
        nodes = []
        edges = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if this is a class variable
                        is_class_var = False
                        parent_class = None
                        
                        for parent in ast.walk(tree):
                            if isinstance(parent, ast.ClassDef) and node in parent.body:
                                is_class_var = True
                                parent_class = parent
                                break
                        
                        if is_class_var:
                            # Create a node for the class variable
                            var_id = f"variable:{file_path.stem}.{parent_class.name}.{target.id}"
                            var_node = Node(
                                id=var_id,
                                type="class_variable",
                                name=target.id,
                                file_path=file_path,
                                line_number=node.lineno,
                                attributes={"class": parent_class.name}
                            )
                            nodes.append(var_node)
                            
                            # Create an edge from the class to the variable
                            class_id = f"class:{file_path.stem}.{parent_class.name}"
                            edge = Edge(
                                source_id=class_id,
                                target_id=var_id,
                                type="contains"
                            )
                            edges.append(edge)
                        else:
                            # Create a node for the global variable
                            var_id = f"variable:{file_path.stem}.{target.id}"
                            var_node = Node(
                                id=var_id,
                                type="global_variable",
                                name=target.id,
                                file_path=file_path,
                                line_number=node.lineno
                            )
                            nodes.append(var_node)
                            
                            # Create an edge from the file to the variable
                            edge = Edge(
                                source_id=file_id,
                                target_id=var_id,
                                type="contains"
                            )
                            edges.append(edge)
        
        return nodes, edges
