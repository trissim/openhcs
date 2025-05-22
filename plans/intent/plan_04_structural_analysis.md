# plan_04_structural_analysis.md
## Component: Structural Analysis

### Objective
Create components for analyzing structural patterns in the codebase to extract intent. These components will analyze the dependency graph to identify architectural patterns, relationships between components, and responsibilities.

### Plan
1. Enhance the `DependencyGraph` to support intent extraction
2. Create a `StructuralAnalyzer` class for analyzing structural patterns
3. Implement detection of common architectural patterns
4. Create an `ArchitecturalIntentExtractor` for extracting intent from patterns
5. Implement analysis of component relationships and responsibilities

### Findings
Structural analysis provides insights into the architecture and design of the codebase. By analyzing the dependency graph, we can identify patterns like layered architecture, microservices, event-driven architecture, and more.

Key patterns to recognize:
- Layered architecture (presentation → service → data)
- Microservices architecture (loosely coupled services)
- Event-driven architecture (publishers and subscribers)
- Model-View-Controller (MVC) pattern
- Repository pattern (data access abstraction)
- Factory pattern (object creation)
- Singleton pattern (single instance)

### Implementation Draft

```python
"""
Structural analysis components for extracting intent from code structure.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.cross_file.dependency_graph import (
    DependencyGraph, Node, Edge
)
from semantic_matrix_analyzer.intent.models.intent import (
    Intent, StructuralIntent, IntentType, CodeLocation
)

logger = logging.getLogger(__name__)


class StructuralAnalyzer:
    """Analyzes structural patterns in the codebase to extract intent."""
    
    def __init__(self, dependency_graph: DependencyGraph):
        """Initialize the structural analyzer.
        
        Args:
            dependency_graph: The dependency graph to analyze.
        """
        self.dependency_graph = dependency_graph
    
    def analyze_dependency_graph(self) -> List[StructuralIntent]:
        """Analyze the dependency graph to extract intent.
        
        Returns:
            A list of StructuralIntent objects.
        """
        intents = []
        
        # Detect layered architecture
        layered_intents = self._detect_layered_architecture()
        intents.extend(layered_intents)
        
        # Detect microservices architecture
        microservices_intents = self._detect_microservices_architecture()
        intents.extend(microservices_intents)
        
        # Detect event-driven architecture
        event_driven_intents = self._detect_event_driven_architecture()
        intents.extend(event_driven_intents)
        
        # Detect MVC architecture
        mvc_intents = self._detect_mvc_architecture()
        intents.extend(mvc_intents)
        
        # Detect repository pattern
        repository_intents = self._detect_repository_pattern()
        intents.extend(repository_intents)
        
        # Detect factory pattern
        factory_intents = self._detect_factory_pattern()
        intents.extend(factory_intents)
        
        # Detect singleton pattern
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
                
                intent = StructuralIntent(
                    name="Layered Architecture",
                    description="The codebase is organized in layers, with higher layers depending on lower layers.",
                    type=IntentType.PATTERN,
                    confidence=0.7,
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
        # Similar implementation to _detect_layered_architecture, but for microservices
        return []
    
    def _detect_event_driven_architecture(self) -> List[StructuralIntent]:
        """Detect event-driven architecture pattern.
        
        Returns:
            A list of StructuralIntent objects.
        """
        # Similar implementation to _detect_layered_architecture, but for event-driven architecture
        return []
    
    def _detect_mvc_architecture(self) -> List[StructuralIntent]:
        """Detect Model-View-Controller architecture pattern.
        
        Returns:
            A list of StructuralIntent objects.
        """
        # Similar implementation to _detect_layered_architecture, but for MVC
        return []
    
    def _detect_repository_pattern(self) -> List[StructuralIntent]:
        """Detect repository pattern.
        
        Returns:
            A list of StructuralIntent objects.
        """
        intents = []
        
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
            
            intent = StructuralIntent(
                name="Repository Pattern",
                description="The codebase uses the repository pattern to abstract data access.",
                type=IntentType.PATTERN,
                confidence=0.6,
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
        # Similar implementation to _detect_repository_pattern, but for factory pattern
        return []
    
    def _detect_singleton_pattern(self) -> List[StructuralIntent]:
        """Detect singleton pattern.
        
        Returns:
            A list of StructuralIntent objects.
        """
        # Similar implementation to _detect_repository_pattern, but for singleton pattern
        return []


class ArchitecturalIntentExtractor:
    """Extracts intent from architectural patterns."""
    
    def __init__(self, dependency_graph: DependencyGraph):
        """Initialize the architectural intent extractor.
        
        Args:
            dependency_graph: The dependency graph to analyze.
        """
        self.dependency_graph = dependency_graph
        self.structural_analyzer = StructuralAnalyzer(dependency_graph)
    
    def extract_architectural_intents(self) -> List[StructuralIntent]:
        """Extract architectural intents from the dependency graph.
        
        Returns:
            A list of StructuralIntent objects.
        """
        return self.structural_analyzer.analyze_dependency_graph()
    
    def extract_component_responsibilities(self) -> Dict[str, List[str]]:
        """Extract component responsibilities from the dependency graph.
        
        Returns:
            A dictionary mapping component IDs to lists of responsibilities.
        """
        responsibilities = {}
        
        # Get all class nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")
        
        for node in class_nodes:
            # Get all methods of the class
            methods = []
            for edge in self.dependency_graph.get_outgoing_edges(node.id):
                if edge.type == "contains":
                    target_node = self.dependency_graph.get_node(edge.target_id)
                    if target_node and target_node.type == "method":
                        methods.append(target_node)
            
            # Extract responsibilities from method names
            class_responsibilities = []
            for method in methods:
                if method.name.startswith("get") or method.name.startswith("set"):
                    continue  # Skip getters and setters
                
                # Extract responsibility from method name
                responsibility = method.name.replace("_", " ")
                class_responsibilities.append(responsibility)
            
            if class_responsibilities:
                responsibilities[node.id] = class_responsibilities
        
        return responsibilities
    
    def extract_component_relationships(self) -> List[Dict[str, Any]]:
        """Extract component relationships from the dependency graph.
        
        Returns:
            A list of relationship dictionaries.
        """
        relationships = []
        
        # Get all class nodes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")
        
        for node in class_nodes:
            # Get all outgoing edges
            for edge in self.dependency_graph.get_outgoing_edges(node.id):
                target_node = self.dependency_graph.get_node(edge.target_id)
                if target_node and target_node.type == "class":
                    relationship = {
                        "source": node.id,
                        "source_name": node.name,
                        "target": target_node.id,
                        "target_name": target_node.name,
                        "type": edge.type
                    }
                    relationships.append(relationship)
        
        return relationships
```
