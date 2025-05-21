"""
Architectural pattern detection module for cross-file analysis.

This module provides functionality for detecting architectural patterns in the dependency graph.
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.cross_file.dependency_graph import DependencyGraph, Node, Edge

logger = logging.getLogger(__name__)


@dataclass
class ArchitecturalPattern:
    """An architectural pattern detected in the codebase."""

    name: str
    description: str
    nodes: List[Node]
    edges: List[Edge]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], dependency_graph: DependencyGraph) -> 'ArchitecturalPattern':
        """Create from dictionary after deserialization.

        Args:
            data: The dictionary to create from.
            dependency_graph: The dependency graph to get nodes and edges from.

        Returns:
            The created architectural pattern.
        """
        nodes = []
        for node_data in data.get("nodes", []):
            node_id = node_data.get("id")
            if node_id and node_id in dependency_graph.nodes:
                nodes.append(dependency_graph.nodes[node_id])

        edges = []
        for edge_data in data.get("edges", []):
            source_id = edge_data.get("source_id")
            target_id = edge_data.get("target_id")
            edge_type = edge_data.get("type")

            for edge in dependency_graph.edges:
                if edge.source_id == source_id and edge.target_id == target_id and edge.type == edge_type:
                    edges.append(edge)
                    break

        return cls(
            name=data["name"],
            description=data["description"],
            nodes=nodes,
            edges=edges,
            confidence=data["confidence"]
        )


class ArchitecturalPatternDetector:
    """Detects architectural patterns in the dependency graph."""

    def __init__(self, dependency_graph: DependencyGraph):
        """Initialize the architectural pattern detector.

        Args:
            dependency_graph: The dependency graph to analyze.
        """
        self.dependency_graph = dependency_graph
        self.patterns: List[ArchitecturalPattern] = []

    def detect_patterns(self) -> List[ArchitecturalPattern]:
        """Detect architectural patterns in the dependency graph.

        Returns:
            A list of detected architectural patterns.
        """
        self.patterns = []

        # Detect layered architecture
        self._detect_layered_architecture()

        # Detect microservices architecture
        self._detect_microservices_architecture()

        # Detect event-driven architecture
        self._detect_event_driven_architecture()

        # Detect model-view-controller architecture
        self._detect_mvc_architecture()

        # Detect repository pattern
        self._detect_repository_pattern()

        # Detect factory pattern
        self._detect_factory_pattern()

        # Detect singleton pattern
        self._detect_singleton_pattern()

        return self.patterns

    def _detect_layered_architecture(self) -> None:
        """Detect layered architecture pattern."""
        # Look for common layer names
        layer_names = ["presentation", "ui", "application", "service", "domain", "model", "data", "persistence", "infrastructure"]

        # Get all file nodes
        file_nodes = self.dependency_graph.get_nodes_by_type("file")

        # Group files by layer
        layers: Dict[str, List[Node]] = {}
        for node in file_nodes:
            for layer_name in layer_names:
                if layer_name in str(node.file_path).lower():
                    if layer_name not in layers:
                        layers[layer_name] = []
                    layers[layer_name].append(node)
                    break

        # Check if we have at least 2 layers
        if len(layers) >= 2:
            # Check for dependencies between layers
            layer_dependencies: Dict[str, Set[str]] = {}
            for layer_name, layer_nodes in layers.items():
                layer_dependencies[layer_name] = set()
                for node in layer_nodes:
                    for edge in self.dependency_graph.get_outgoing_edges(node.id):
                        target_node = self.dependency_graph.get_node(edge.target_id)
                        if target_node and target_node.type == "file":
                            for other_layer_name, other_layer_nodes in layers.items():
                                if other_layer_name != layer_name and target_node in other_layer_nodes:
                                    layer_dependencies[layer_name].add(other_layer_name)

            # Check if we have dependencies between layers
            if any(deps for deps in layer_dependencies.values()):
                # Create a pattern
                nodes = []
                for layer_nodes in layers.values():
                    nodes.extend(layer_nodes)

                edges = []
                for node in nodes:
                    for edge in self.dependency_graph.get_outgoing_edges(node.id):
                        target_node = self.dependency_graph.get_node(edge.target_id)
                        if target_node and target_node in nodes:
                            edges.append(edge)

                pattern = ArchitecturalPattern(
                    name="Layered Architecture",
                    description="The codebase is organized in layers, with higher layers depending on lower layers.",
                    nodes=nodes,
                    edges=edges,
                    confidence=0.7
                )

                self.patterns.append(pattern)

    def _detect_microservices_architecture(self) -> None:
        """Detect microservices architecture pattern."""
        # Look for common microservice indicators
        service_indicators = ["service", "api", "client", "server"]

        # Get all file nodes
        file_nodes = self.dependency_graph.get_nodes_by_type("file")

        # Group files by service
        services: Dict[str, List[Node]] = {}
        for node in file_nodes:
            for indicator in service_indicators:
                if indicator in str(node.file_path).lower():
                    service_name = self._extract_service_name(str(node.file_path), indicator)
                    if service_name:
                        if service_name not in services:
                            services[service_name] = []
                        services[service_name].append(node)
                        break

        # Check if we have at least 2 services
        if len(services) >= 2:
            # Check for dependencies between services
            service_dependencies: Dict[str, Set[str]] = {}
            for service_name, service_nodes in services.items():
                service_dependencies[service_name] = set()
                for node in service_nodes:
                    for edge in self.dependency_graph.get_outgoing_edges(node.id):
                        target_node = self.dependency_graph.get_node(edge.target_id)
                        if target_node and target_node.type == "file":
                            for other_service_name, other_service_nodes in services.items():
                                if other_service_name != service_name and target_node in other_service_nodes:
                                    service_dependencies[service_name].add(other_service_name)

            # Check if we have dependencies between services
            if any(deps for deps in service_dependencies.values()):
                # Create a pattern
                nodes = []
                for service_nodes in services.values():
                    nodes.extend(service_nodes)

                edges = []
                for node in nodes:
                    for edge in self.dependency_graph.get_outgoing_edges(node.id):
                        target_node = self.dependency_graph.get_node(edge.target_id)
                        if target_node and target_node in nodes:
                            edges.append(edge)

                pattern = ArchitecturalPattern(
                    name="Microservices Architecture",
                    description="The codebase is organized as a set of loosely coupled services.",
                    nodes=nodes,
                    edges=edges,
                    confidence=0.6
                )

                self.patterns.append(pattern)

    def _extract_service_name(self, file_path: str, indicator: str) -> Optional[str]:
        """Extract a service name from a file path.

        Args:
            file_path: The file path.
            indicator: The service indicator.

        Returns:
            The service name, or None if not found.
        """
        # This is a simple implementation that looks for patterns like "user_service" or "user-service"
        pattern = rf"([a-zA-Z0-9_-]+)_{indicator}|([a-zA-Z0-9_-]+)-{indicator}"
        match = re.search(pattern, file_path)
        if match:
            return match.group(1) or match.group(2)

        # Also look for patterns like "service_user" or "service-user"
        pattern = rf"{indicator}_([a-zA-Z0-9_-]+)|{indicator}-([a-zA-Z0-9_-]+)"
        match = re.search(pattern, file_path)
        if match:
            return match.group(1) or match.group(2)

        return None

    def _detect_event_driven_architecture(self) -> None:
        """Detect event-driven architecture pattern."""
        # Look for common event-driven indicators
        event_indicators = ["event", "listener", "handler", "subscriber", "publisher", "emitter"]

        # Get all class and function nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")
        function_nodes = self.dependency_graph.get_nodes_by_type("function")

        # Find event-related nodes
        event_nodes = []
        for node in class_nodes + function_nodes:
            for indicator in event_indicators:
                if indicator in node.name.lower():
                    event_nodes.append(node)
                    break

        # Check if we have enough event-related nodes
        if len(event_nodes) >= 3:
            # Get all edges between event-related nodes
            edges = []
            for node in event_nodes:
                for edge in self.dependency_graph.get_outgoing_edges(node.id):
                    target_node = self.dependency_graph.get_node(edge.target_id)
                    if target_node and target_node in event_nodes:
                        edges.append(edge)

            # Check if we have enough edges
            if len(edges) >= 2:
                pattern = ArchitecturalPattern(
                    name="Event-Driven Architecture",
                    description="The codebase uses events to communicate between components.",
                    nodes=event_nodes,
                    edges=edges,
                    confidence=0.5
                )

                self.patterns.append(pattern)

    def _detect_mvc_architecture(self) -> None:
        """Detect model-view-controller architecture pattern."""
        # Look for common MVC indicators
        model_indicators = ["model", "entity", "domain"]
        view_indicators = ["view", "template", "page", "ui"]
        controller_indicators = ["controller", "handler", "resource"]

        # Get all class nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")

        # Group classes by MVC component
        models = []
        views = []
        controllers = []

        for node in class_nodes:
            for indicator in model_indicators:
                if indicator in node.name.lower():
                    models.append(node)
                    break

            for indicator in view_indicators:
                if indicator in node.name.lower():
                    views.append(node)
                    break

            for indicator in controller_indicators:
                if indicator in node.name.lower():
                    controllers.append(node)
                    break

        # Check if we have all three components
        if models and views and controllers:
            # Get all edges between MVC components
            edges = []
            mvc_nodes = models + views + controllers

            for node in mvc_nodes:
                for edge in self.dependency_graph.get_outgoing_edges(node.id):
                    target_node = self.dependency_graph.get_node(edge.target_id)
                    if target_node and target_node in mvc_nodes:
                        edges.append(edge)

            # Check if we have enough edges
            if len(edges) >= 2:
                pattern = ArchitecturalPattern(
                    name="Model-View-Controller Architecture",
                    description="The codebase is organized according to the MVC pattern.",
                    nodes=mvc_nodes,
                    edges=edges,
                    confidence=0.7
                )

                self.patterns.append(pattern)

    def _detect_repository_pattern(self) -> None:
        """Detect repository pattern."""
        # Look for repository classes
        repository_indicators = ["repository", "repo", "dao", "data_access"]

        # Get all class nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")

        # Find repository classes
        repository_nodes = []
        for node in class_nodes:
            for indicator in repository_indicators:
                if indicator in node.name.lower():
                    repository_nodes.append(node)
                    break

        # Check if we have repository classes
        if repository_nodes:
            # Get all edges to and from repository classes
            edges = []
            for node in repository_nodes:
                for edge in self.dependency_graph.get_outgoing_edges(node.id):
                    edges.append(edge)
                for edge in self.dependency_graph.get_incoming_edges(node.id):
                    edges.append(edge)

            # Get all nodes connected to repository classes
            connected_nodes = set()
            for edge in edges:
                source_node = self.dependency_graph.get_node(edge.source_id)
                target_node = self.dependency_graph.get_node(edge.target_id)
                if source_node:
                    connected_nodes.add(source_node.id)
                if target_node:
                    connected_nodes.add(target_node.id)

            # Convert node IDs to actual nodes
            connected_node_objects = [self.dependency_graph.get_node(node_id) for node_id in connected_nodes]
            connected_node_objects = [node for node in connected_node_objects if node]

            # Create a pattern
            pattern = ArchitecturalPattern(
                name="Repository Pattern",
                description="The codebase uses the repository pattern to abstract data access.",
                nodes=connected_node_objects,
                edges=edges,
                confidence=0.6
            )

            self.patterns.append(pattern)

    def _detect_factory_pattern(self) -> None:
        """Detect factory pattern."""
        # Look for factory classes
        factory_indicators = ["factory", "creator", "builder"]

        # Get all class nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")

        # Find factory classes
        factory_nodes = []
        for node in class_nodes:
            for indicator in factory_indicators:
                if indicator in node.name.lower():
                    factory_nodes.append(node)
                    break

        # Check if we have factory classes
        if factory_nodes:
            # Get all edges from factory classes
            edges = []
            for node in factory_nodes:
                for edge in self.dependency_graph.get_outgoing_edges(node.id):
                    edges.append(edge)

            # Get all nodes created by factory classes
            created_node_ids = set()
            for edge in edges:
                target_node = self.dependency_graph.get_node(edge.target_id)
                if target_node:
                    created_node_ids.add(target_node.id)

            # Convert node IDs to actual nodes
            created_nodes = [self.dependency_graph.get_node(node_id) for node_id in created_node_ids]
            created_nodes = [node for node in created_nodes if node]

            # Create a pattern
            pattern = ArchitecturalPattern(
                name="Factory Pattern",
                description="The codebase uses the factory pattern to create objects.",
                nodes=factory_nodes + list(created_nodes),
                edges=edges,
                confidence=0.6
            )

            self.patterns.append(pattern)

    def _detect_singleton_pattern(self) -> None:
        """Detect singleton pattern."""
        # Get all class nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")

        # Find singleton classes
        singleton_nodes = []
        for node in class_nodes:
            # Check if the class has a getInstance or instance method/property
            is_singleton = False

            # Get all methods of the class
            class_id = node.id
            for edge in self.dependency_graph.get_outgoing_edges(class_id):
                if edge.type == "contains":
                    target_node = self.dependency_graph.get_node(edge.target_id)
                    if target_node and target_node.type == "method":
                        method_name = target_node.name.lower()
                        if method_name in ["getinstance", "get_instance", "instance"]:
                            is_singleton = True
                            break

            if is_singleton or "singleton" in node.name.lower():
                singleton_nodes.append(node)

        # Check if we have singleton classes
        if singleton_nodes:
            # Get all edges to singleton classes
            edges = []
            for node in singleton_nodes:
                for edge in self.dependency_graph.get_incoming_edges(node.id):
                    edges.append(edge)

            # Create a pattern
            pattern = ArchitecturalPattern(
                name="Singleton Pattern",
                description="The codebase uses the singleton pattern to ensure a class has only one instance.",
                nodes=singleton_nodes,
                edges=edges,
                confidence=0.7
            )

            self.patterns.append(pattern)
