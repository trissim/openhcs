"""
Structural analysis components for extracting intent from code structure.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, StructuralIntent, IntentType, CodeLocation
)
from semantic_matrix_analyzer.intent.config.configuration import (
    Configuration, ConfigurableAnalyzer
)

logger = logging.getLogger(__name__)


class ConfigurableStructuralAnalyzer(ConfigurableAnalyzer):
    """Configurable analyzer for structural patterns."""

    def __init__(self, dependency_graph, config: Optional[Configuration] = None):
        """Initialize the configurable structural analyzer.

        Args:
            dependency_graph: The dependency graph to analyze.
            config: The configuration to use (optional).
        """
        super().__init__(config)
        self.dependency_graph = dependency_graph

    def _get_config_section(self) -> str:
        """Get the configuration section for this analyzer.

        Returns:
            The configuration section name.
        """
        return "structural_analysis"

    def is_enabled(self) -> bool:
        """Check if the analyzer is enabled.

        Returns:
            True if the analyzer is enabled, False otherwise.
        """
        return self.get_config_value("enabled", True)

    def analyze_dependency_graph(self) -> List[StructuralIntent]:
        """Analyze the dependency graph to extract intent.

        Returns:
            A list of StructuralIntent objects.
        """
        # Check if the analyzer is enabled
        if not self.is_enabled():
            return []

        intents = []

        # Detect layered architecture
        if self.get_config_value("patterns.layered_architecture.enabled", True):
            layered_intents = self._detect_layered_architecture()
            intents.extend(layered_intents)

        # Detect microservices architecture
        if self.get_config_value("patterns.microservices_architecture.enabled", True):
            microservices_intents = self._detect_microservices_architecture()
            intents.extend(microservices_intents)

        # Detect event-driven architecture
        if self.get_config_value("patterns.event_driven_architecture.enabled", True):
            event_driven_intents = self._detect_event_driven_architecture()
            intents.extend(event_driven_intents)

        # Detect MVC architecture
        if self.get_config_value("patterns.mvc_architecture.enabled", True):
            mvc_intents = self._detect_mvc_architecture()
            intents.extend(mvc_intents)

        # Detect repository pattern
        if self.get_config_value("patterns.repository_pattern.enabled", True):
            repository_intents = self._detect_repository_pattern()
            intents.extend(repository_intents)

        # Detect factory pattern
        if self.get_config_value("patterns.factory_pattern.enabled", True):
            factory_intents = self._detect_factory_pattern()
            intents.extend(factory_intents)

        # Detect singleton pattern
        if self.get_config_value("patterns.singleton_pattern.enabled", True):
            singleton_intents = self._detect_singleton_pattern()
            intents.extend(singleton_intents)

        return intents

    def _detect_layered_architecture(self) -> List[StructuralIntent]:
        """Detect layered architecture pattern.

        Returns:
            A list of StructuralIntent objects.
        """
        intents = []

        # Look for common layer names
        layer_names = self.get_config_value("patterns.layered_architecture.layer_names", [
            "presentation", "ui", "application", "service", "domain", "model", "data", "persistence", "infrastructure"
        ])

        # Get all file nodes
        file_nodes = self.dependency_graph.get_nodes_by_type("file")

        # Group files by layer
        layers: Dict[str, List[Any]] = {}
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
                # Create a structural intent
                components = []
                for layer_name, layer_nodes in layers.items():
                    components.extend([node.id for node in layer_nodes])

                relationships = []
                for layer_name, deps in layer_dependencies.items():
                    for dep in deps:
                        relationships.append({
                            "source": layer_name,
                            "target": dep,
                            "type": "depends_on"
                        })

                confidence = self.get_config_value("patterns.layered_architecture.confidence", 0.7)

                intent = StructuralIntent(
                    name="Layered Architecture",
                    description="The codebase is organized in layers, with higher layers depending on lower layers.",
                    type=IntentType.PATTERN,
                    confidence=confidence,
                    pattern_name="Layered Architecture",
                    components=components,
                    relationships=relationships
                )

                intents.append(intent)

        return intents

    def _detect_microservices_architecture(self) -> List[StructuralIntent]:
        """Detect microservices architecture pattern.

        Returns:
            A list of StructuralIntent objects.
        """
        intents = []

        # Look for service indicators
        service_indicators = self.get_config_value("patterns.microservices_architecture.service_indicators", [
            "service", "api", "client", "server"
        ])

        # Get all file nodes
        file_nodes = self.dependency_graph.get_nodes_by_type("file")

        # Group files by service
        services: Dict[str, List[Any]] = {}
        for node in file_nodes:
            for service_indicator in service_indicators:
                if service_indicator in str(node.file_path).lower():
                    service_name = self._extract_service_name(str(node.file_path), service_indicator)
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

            # Create a structural intent
            components = []
            for service_name, service_nodes in services.items():
                components.extend([node.id for node in service_nodes])

            relationships = []
            for service_name, deps in service_dependencies.items():
                for dep in deps:
                    relationships.append({
                        "source": service_name,
                        "target": dep,
                        "type": "depends_on"
                    })

            confidence = self.get_config_value("patterns.microservices_architecture.confidence", 0.6)

            intent = StructuralIntent(
                name="Microservices Architecture",
                description="The codebase is organized as a collection of loosely coupled services.",
                type=IntentType.PATTERN,
                confidence=confidence,
                pattern_name="Microservices Architecture",
                components=components,
                relationships=relationships
            )

            intents.append(intent)

        return intents

    def _extract_service_name(self, file_path: str, service_indicator: str) -> Optional[str]:
        """Extract the service name from a file path.

        Args:
            file_path: The file path.
            service_indicator: The service indicator.

        Returns:
            The service name, or None if not found.
        """
        # Extract the directory containing the service indicator
        match = re.search(f"(\\w+{service_indicator}\\w*|\\w*{service_indicator}\\w+)", file_path, re.IGNORECASE)
        if match:
            return match.group(1)

        # Extract the directory name before or after the service indicator
        parts = file_path.split("/")
        for i, part in enumerate(parts):
            if service_indicator in part.lower():
                if i > 0:
                    return parts[i-1]
                elif i < len(parts) - 1:
                    return parts[i+1]

        return None

    def _detect_event_driven_architecture(self) -> List[StructuralIntent]:
        """Detect event-driven architecture pattern.

        Returns:
            A list of StructuralIntent objects.
        """
        intents = []

        # Look for event indicators
        event_indicators = self.get_config_value("patterns.event_driven_architecture.event_indicators", [
            "event", "listener", "handler", "subscriber", "publisher"
        ])

        # Get all class nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")

        # Find event-related classes
        event_classes = []
        for node in class_nodes:
            for indicator in event_indicators:
                if indicator in node.name.lower():
                    event_classes.append(node)
                    break

        # Check if we have event-related classes
        if event_classes:
            # Get all edges to and from event classes
            edges = []
            for node in event_classes:
                for edge in self.dependency_graph.get_outgoing_edges(node.id):
                    edges.append(edge)
                for edge in self.dependency_graph.get_incoming_edges(node.id):
                    edges.append(edge)

            # Get all nodes connected to event classes
            connected_node_ids = set()
            for edge in edges:
                connected_node_ids.add(edge.source_id)
                connected_node_ids.add(edge.target_id)

            # Create a structural intent
            components = list(connected_node_ids)

            relationships = []
            for edge in edges:
                relationships.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.type
                })

            confidence = self.get_config_value("patterns.event_driven_architecture.confidence", 0.6)

            intent = StructuralIntent(
                name="Event-Driven Architecture",
                description="The codebase uses events to communicate between components.",
                type=IntentType.PATTERN,
                confidence=confidence,
                pattern_name="Event-Driven Architecture",
                components=components,
                relationships=relationships
            )

            intents.append(intent)

        return intents

    def _detect_mvc_architecture(self) -> List[StructuralIntent]:
        """Detect Model-View-Controller architecture pattern.

        Returns:
            A list of StructuralIntent objects.
        """
        intents = []

        # Look for MVC indicators
        model_indicators = self.get_config_value("patterns.mvc_architecture.model_indicators", [
            "model", "entity", "domain"
        ])
        view_indicators = self.get_config_value("patterns.mvc_architecture.view_indicators", [
            "view", "template", "page", "screen"
        ])
        controller_indicators = self.get_config_value("patterns.mvc_architecture.controller_indicators", [
            "controller", "handler"
        ])

        # Get all class nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")

        # Find MVC classes
        model_classes = []
        view_classes = []
        controller_classes = []

        for node in class_nodes:
            for indicator in model_indicators:
                if indicator in node.name.lower():
                    model_classes.append(node)
                    break

            for indicator in view_indicators:
                if indicator in node.name.lower():
                    view_classes.append(node)
                    break

            for indicator in controller_indicators:
                if indicator in node.name.lower():
                    controller_classes.append(node)
                    break

        # Check if we have at least one class from each MVC component
        if model_classes and view_classes and controller_classes:
            # Get all edges between MVC classes
            edges = []
            mvc_classes = model_classes + view_classes + controller_classes

            for node in mvc_classes:
                for edge in self.dependency_graph.get_outgoing_edges(node.id):
                    target_node = self.dependency_graph.get_node(edge.target_id)
                    if target_node in mvc_classes:
                        edges.append(edge)

            # Create a structural intent
            components = [node.id for node in mvc_classes]

            relationships = []
            for edge in edges:
                relationships.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.type
                })

            confidence = self.get_config_value("patterns.mvc_architecture.confidence", 0.7)

            intent = StructuralIntent(
                name="MVC Architecture",
                description="The codebase follows the Model-View-Controller pattern.",
                type=IntentType.PATTERN,
                confidence=confidence,
                pattern_name="MVC Architecture",
                components=components,
                relationships=relationships
            )

            intents.append(intent)

        return intents

    def _detect_repository_pattern(self) -> List[StructuralIntent]:
        """Detect repository pattern.

        Returns:
            A list of StructuralIntent objects.
        """
        intents = []

        # Look for repository classes
        repository_indicators = self.get_config_value("patterns.repository_pattern.repository_indicators", [
            "repository", "repo", "dao", "data_access"
        ])

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
            connected_node_ids = set()
            for edge in edges:
                connected_node_ids.add(edge.source_id)
                connected_node_ids.add(edge.target_id)

            # Create a structural intent
            components = list(connected_node_ids)

            relationships = []
            for edge in edges:
                relationships.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.type
                })

            confidence = self.get_config_value("patterns.repository_pattern.confidence", 0.6)

            intent = StructuralIntent(
                name="Repository Pattern",
                description="The codebase uses the repository pattern to abstract data access.",
                type=IntentType.PATTERN,
                confidence=confidence,
                pattern_name="Repository Pattern",
                components=components,
                relationships=relationships
            )

            intents.append(intent)

        return intents

    def _detect_factory_pattern(self) -> List[StructuralIntent]:
        """Detect factory pattern.

        Returns:
            A list of StructuralIntent objects.
        """
        intents = []

        # Look for factory classes
        factory_indicators = self.get_config_value("patterns.factory_pattern.factory_indicators", [
            "factory", "creator", "builder"
        ])

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
            # Get all edges to and from factory classes
            edges = []
            for node in factory_nodes:
                for edge in self.dependency_graph.get_outgoing_edges(node.id):
                    edges.append(edge)
                for edge in self.dependency_graph.get_incoming_edges(node.id):
                    edges.append(edge)

            # Get all nodes connected to factory classes
            connected_node_ids = set()
            for edge in edges:
                connected_node_ids.add(edge.source_id)
                connected_node_ids.add(edge.target_id)

            # Create a structural intent
            components = list(connected_node_ids)

            relationships = []
            for edge in edges:
                relationships.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.type
                })

            confidence = self.get_config_value("patterns.factory_pattern.confidence", 0.6)

            intent = StructuralIntent(
                name="Factory Pattern",
                description="The codebase uses the factory pattern to create objects.",
                type=IntentType.PATTERN,
                confidence=confidence,
                pattern_name="Factory Pattern",
                components=components,
                relationships=relationships
            )

            intents.append(intent)

        return intents

    def _detect_singleton_pattern(self) -> List[StructuralIntent]:
        """Detect singleton pattern.

        Returns:
            A list of StructuralIntent objects.
        """
        intents = []

        # Look for singleton classes
        singleton_indicators = self.get_config_value("patterns.singleton_pattern.singleton_indicators", [
            "singleton", "instance"
        ])

        # Get all class nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")

        # Find singleton classes
        singleton_nodes = []
        for node in class_nodes:
            # Check class name
            for indicator in singleton_indicators:
                if indicator in node.name.lower():
                    singleton_nodes.append(node)
                    break

            # Check for instance method or variable
            if node not in singleton_nodes:
                for edge in self.dependency_graph.get_outgoing_edges(node.id):
                    target_node = self.dependency_graph.get_node(edge.target_id)
                    if target_node and (target_node.type == "method" or target_node.type == "variable"):
                        for indicator in singleton_indicators:
                            if indicator in target_node.name.lower():
                                singleton_nodes.append(node)
                                break
                        if node in singleton_nodes:
                            break

        # Check if we have singleton classes
        if singleton_nodes:
            # Get all edges to and from singleton classes
            edges = []
            for node in singleton_nodes:
                for edge in self.dependency_graph.get_outgoing_edges(node.id):
                    edges.append(edge)
                for edge in self.dependency_graph.get_incoming_edges(node.id):
                    edges.append(edge)

            # Get all nodes connected to singleton classes
            connected_node_ids = set()
            for edge in edges:
                connected_node_ids.add(edge.source_id)
                connected_node_ids.add(edge.target_id)

            # Create a structural intent
            components = list(connected_node_ids)

            relationships = []
            for edge in edges:
                relationships.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.type
                })

            confidence = self.get_config_value("patterns.singleton_pattern.confidence", 0.6)

            intent = StructuralIntent(
                name="Singleton Pattern",
                description="The codebase uses the singleton pattern to ensure a single instance of a class.",
                type=IntentType.PATTERN,
                confidence=confidence,
                pattern_name="Singleton Pattern",
                components=components,
                relationships=relationships
            )

            intents.append(intent)

        return intents
