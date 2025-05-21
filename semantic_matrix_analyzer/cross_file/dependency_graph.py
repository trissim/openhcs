"""
Dependency graph for representing dependencies between code elements.
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """A node in the dependency graph."""
    
    id: str
    name: str
    type: str  # "file", "class", "method", "variable", etc.
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """An edge in the dependency graph."""
    
    source_id: str
    target_id: str
    type: str  # "imports", "calls", "contains", "inherits", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class DependencyGraph:
    """A graph of dependencies between code elements."""
    
    def __init__(self):
        """Initialize the dependency graph."""
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.outgoing_edges: Dict[str, List[Edge]] = {}
        self.incoming_edges: Dict[str, List[Edge]] = {}
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph.
        
        Args:
            node: The node to add.
        """
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph.
        
        Args:
            edge: The edge to add.
        """
        self.edges.append(edge)
        
        if edge.source_id not in self.outgoing_edges:
            self.outgoing_edges[edge.source_id] = []
        self.outgoing_edges[edge.source_id].append(edge)
        
        if edge.target_id not in self.incoming_edges:
            self.incoming_edges[edge.target_id] = []
        self.incoming_edges[edge.target_id].append(edge)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID.
        
        Args:
            node_id: The node ID.
            
        Returns:
            The node, or None if not found.
        """
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type.
        
        Args:
            node_type: The node type.
            
        Returns:
            A list of nodes.
        """
        return [node for node in self.nodes.values() if node.type == node_type]
    
    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get all outgoing edges from a node.
        
        Args:
            node_id: The node ID.
            
        Returns:
            A list of edges.
        """
        return self.outgoing_edges.get(node_id, [])
    
    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Get all incoming edges to a node.
        
        Args:
            node_id: The node ID.
            
        Returns:
            A list of edges.
        """
        return self.incoming_edges.get(node_id, [])


class DependencyExtractor:
    """Extracts dependencies from Python code."""
    
    def __init__(self):
        """Initialize the dependency extractor."""
        pass
    
    def extract_dependencies(self, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract dependencies from a Python file.
        
        Args:
            file_path: The path to the Python file.
            
        Returns:
            A tuple of (nodes, edges).
        """
        nodes = []
        edges = []
        
        try:
            # Parse the file
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            
            tree = ast.parse(code, filename=str(file_path))
            
            # Create a file node
            file_id = f"file:{file_path}"
            file_node = Node(
                id=file_id,
                name=file_path.name,
                type="file",
                file_path=file_path
            )
            nodes.append(file_node)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        import_id = f"module:{name.name}"
                        import_node = Node(
                            id=import_id,
                            name=name.name,
                            type="module"
                        )
                        nodes.append(import_node)
                        
                        edge = Edge(
                            source_id=file_id,
                            target_id=import_id,
                            type="imports"
                        )
                        edges.append(edge)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_id = f"module:{node.module}"
                        module_node = Node(
                            id=module_id,
                            name=node.module,
                            type="module"
                        )
                        nodes.append(module_node)
                        
                        for name in node.names:
                            import_id = f"module:{node.module}.{name.name}"
                            import_node = Node(
                                id=import_id,
                                name=f"{node.module}.{name.name}",
                                type="module"
                            )
                            nodes.append(import_node)
                            
                            edge1 = Edge(
                                source_id=file_id,
                                target_id=import_id,
                                type="imports"
                            )
                            edges.append(edge1)
                            
                            edge2 = Edge(
                                source_id=module_id,
                                target_id=import_id,
                                type="contains"
                            )
                            edges.append(edge2)
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_id = f"class:{file_path}:{node.name}"
                    class_node = Node(
                        id=class_id,
                        name=node.name,
                        type="class",
                        file_path=file_path,
                        line_number=node.lineno
                    )
                    nodes.append(class_node)
                    
                    edge = Edge(
                        source_id=file_id,
                        target_id=class_id,
                        type="contains"
                    )
                    edges.append(edge)
                    
                    # Extract base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_id = f"class:{base.id}"
                            base_node = Node(
                                id=base_id,
                                name=base.id,
                                type="class"
                            )
                            nodes.append(base_node)
                            
                            edge = Edge(
                                source_id=class_id,
                                target_id=base_id,
                                type="inherits"
                            )
                            edges.append(edge)
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_id = f"method:{file_path}:{node.name}.{item.name}"
                            method_node = Node(
                                id=method_id,
                                name=item.name,
                                type="method",
                                file_path=file_path,
                                line_number=item.lineno
                            )
                            nodes.append(method_node)
                            
                            edge = Edge(
                                source_id=class_id,
                                target_id=method_id,
                                type="contains"
                            )
                            edges.append(edge)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not any(
                    isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                    if node in parent.body
                ):
                    function_id = f"function:{file_path}:{node.name}"
                    function_node = Node(
                        id=function_id,
                        name=node.name,
                        type="function",
                        file_path=file_path,
                        line_number=node.lineno
                    )
                    nodes.append(function_node)
                    
                    edge = Edge(
                        source_id=file_id,
                        target_id=function_id,
                        type="contains"
                    )
                    edges.append(edge)
        
        except Exception as e:
            logger.error(f"Error extracting dependencies from {file_path}: {e}")
        
        return nodes, edges
