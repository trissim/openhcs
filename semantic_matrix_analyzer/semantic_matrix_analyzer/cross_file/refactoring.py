"""
Refactoring opportunity detection module for cross-file analysis.

This module provides functionality for detecting refactoring opportunities in the codebase.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.cross_file.dependency_graph import DependencyGraph, Node, Edge

logger = logging.getLogger(__name__)


@dataclass
class RefactoringOpportunity:
    """A refactoring opportunity detected in the codebase."""
    
    type: str  # "extract_class", "move_method", "rename", etc.
    description: str
    nodes: List[Node]
    edges: List[Edge]
    benefit: str
    effort: str  # "low", "medium", "high"
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "description": self.description,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "benefit": self.benefit,
            "effort": self.effort,
            "suggestion": self.suggestion
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], dependency_graph: DependencyGraph) -> 'RefactoringOpportunity':
        """Create from dictionary after deserialization.
        
        Args:
            data: The dictionary to create from.
            dependency_graph: The dependency graph to get nodes and edges from.
            
        Returns:
            The created refactoring opportunity.
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
            type=data["type"],
            description=data["description"],
            nodes=nodes,
            edges=edges,
            benefit=data["benefit"],
            effort=data["effort"],
            suggestion=data.get("suggestion")
        )


class RefactoringOpportunityDetector:
    """Detects refactoring opportunities in the codebase."""
    
    def __init__(self, dependency_graph: DependencyGraph):
        """Initialize the refactoring opportunity detector.
        
        Args:
            dependency_graph: The dependency graph to analyze.
        """
        self.dependency_graph = dependency_graph
    
    def detect_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect refactoring opportunities in the codebase.
        
        Returns:
            A list of detected refactoring opportunities.
        """
        opportunities = []
        
        # Detect extract class opportunities
        extract_class_opportunities = self._detect_extract_class_opportunities()
        opportunities.extend(extract_class_opportunities)
        
        # Detect move method opportunities
        move_method_opportunities = self._detect_move_method_opportunities()
        opportunities.extend(move_method_opportunities)
        
        # Detect rename opportunities
        rename_opportunities = self._detect_rename_opportunities()
        opportunities.extend(rename_opportunities)
        
        # Detect extract interface opportunities
        extract_interface_opportunities = self._detect_extract_interface_opportunities()
        opportunities.extend(extract_interface_opportunities)
        
        return opportunities
    
    def _detect_extract_class_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect extract class opportunities.
        
        Returns:
            A list of detected refactoring opportunities.
        """
        opportunities = []
        
        # Look for large classes
        class_nodes = self.dependency_graph.get_nodes_by_type("class")
        
        for class_node in class_nodes:
            # Get all methods and variables of the class
            class_members = []
            for edge in self.dependency_graph.get_outgoing_edges(class_node.id):
                if edge.type == "contains":
                    target_node = self.dependency_graph.get_node(edge.target_id)
                    if target_node and target_node.type in ["method", "class_variable"]:
                        class_members.append(target_node)
            
            # Check if the class has many members
            if len(class_members) >= 10:
                # Look for cohesive groups of members
                groups = self._find_cohesive_groups(class_node, class_members)
                
                # Create opportunities for each group
                for group_name, group_members in groups.items():
                    if len(group_members) >= 3:
                        # Get edges between group members
                        edges = []
                        for member in group_members:
                            for edge in self.dependency_graph.get_outgoing_edges(member.id):
                                target_node = self.dependency_graph.get_node(edge.target_id)
                                if target_node and target_node in group_members:
                                    edges.append(edge)
                        
                        opportunity = RefactoringOpportunity(
                            type="extract_class",
                            description=f"Extract class from {class_node.name} for {group_name} functionality",
                            nodes=[class_node] + group_members,
                            edges=edges,
                            benefit="Improved cohesion and reduced class size",
                            effort="medium",
                            suggestion=f"Create a new class for {group_name} functionality and move the related methods and variables"
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _find_cohesive_groups(self, class_node: Node, class_members: List[Node]) -> Dict[str, List[Node]]:
        """Find cohesive groups of class members.
        
        Args:
            class_node: The class node.
            class_members: The class members.
            
        Returns:
            A dictionary mapping group names to lists of members.
        """
        # This is a simple implementation that groups members by name prefix
        groups: Dict[str, List[Node]] = {}
        
        for member in class_members:
            # Look for common prefixes
            prefixes = ["get", "set", "is", "has", "create", "build", "parse", "format", "validate", "compute", "calculate"]
            
            for prefix in prefixes:
                if member.name.lower().startswith(prefix):
                    # Extract the entity name (e.g., "User" from "getUser")
                    entity_name = member.name[len(prefix):]
                    if entity_name:
                        if entity_name not in groups:
                            groups[entity_name] = []
                        groups[entity_name].append(member)
                        break
        
        return groups
    
    def _detect_move_method_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect move method opportunities.
        
        Returns:
            A list of detected refactoring opportunities.
        """
        opportunities = []
        
        # Look for methods that use more members from another class than from their own class
        method_nodes = self.dependency_graph.get_nodes_by_type("method")
        
        for method_node in method_nodes:
            # Get the class of the method
            class_name = method_node.attributes.get("class")
            if not class_name:
                continue
            
            # Find the class node
            class_node = None
            for node in self.dependency_graph.get_nodes_by_type("class"):
                if node.name == class_name:
                    class_node = node
                    break
            
            if not class_node:
                continue
            
            # Get all members of the class
            class_members = []
            for edge in self.dependency_graph.get_outgoing_edges(class_node.id):
                if edge.type == "contains":
                    target_node = self.dependency_graph.get_node(edge.target_id)
                    if target_node and target_node.type in ["method", "class_variable"]:
                        class_members.append(target_node)
            
            # Get all members used by the method
            used_members: Dict[str, List[Node]] = {}
            for edge in self.dependency_graph.get_outgoing_edges(method_node.id):
                target_node = self.dependency_graph.get_node(edge.target_id)
                if target_node and target_node.type in ["method", "class_variable"]:
                    # Get the class of the target node
                    target_class_name = target_node.attributes.get("class")
                    if target_class_name:
                        if target_class_name not in used_members:
                            used_members[target_class_name] = []
                        used_members[target_class_name].append(target_node)
            
            # Check if the method uses more members from another class
            for other_class_name, other_members in used_members.items():
                if other_class_name != class_name and len(other_members) > len(used_members.get(class_name, [])):
                    # Find the other class node
                    other_class_node = None
                    for node in self.dependency_graph.get_nodes_by_type("class"):
                        if node.name == other_class_name:
                            other_class_node = node
                            break
                    
                    if other_class_node:
                        # Create an opportunity
                        opportunity = RefactoringOpportunity(
                            type="move_method",
                            description=f"Move method {method_node.name} from {class_name} to {other_class_name}",
                            nodes=[method_node, class_node, other_class_node] + other_members,
                            edges=[],
                            benefit="Improved cohesion and reduced coupling",
                            effort="low",
                            suggestion=f"Move the method {method_node.name} to class {other_class_name}"
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_rename_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect rename opportunities.
        
        Returns:
            A list of detected refactoring opportunities.
        """
        opportunities = []
        
        # Look for poorly named entities
        
        # 1. Look for single-letter variable names
        variable_nodes = self.dependency_graph.get_nodes_by_type("global_variable")
        variable_nodes.extend(self.dependency_graph.get_nodes_by_type("class_variable"))
        
        for node in variable_nodes:
            if len(node.name) == 1 and node.name not in ["i", "j", "k", "x", "y", "z"]:
                # Create an opportunity
                opportunity = RefactoringOpportunity(
                    type="rename",
                    description=f"Rename variable {node.name} to a more descriptive name",
                    nodes=[node],
                    edges=[],
                    benefit="Improved code readability",
                    effort="low",
                    suggestion=f"Choose a more descriptive name for the variable {node.name}"
                )
                opportunities.append(opportunity)
        
        # 2. Look for methods with unclear names
        method_nodes = self.dependency_graph.get_nodes_by_type("method")
        
        for node in method_nodes:
            if len(node.name) <= 2 or node.name in ["process", "handle", "do", "run", "execute"]:
                # Create an opportunity
                opportunity = RefactoringOpportunity(
                    type="rename",
                    description=f"Rename method {node.name} to a more descriptive name",
                    nodes=[node],
                    edges=[],
                    benefit="Improved code readability",
                    effort="medium",
                    suggestion=f"Choose a more descriptive name for the method {node.name}"
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_extract_interface_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect extract interface opportunities.
        
        Returns:
            A list of detected refactoring opportunities.
        """
        opportunities = []
        
        # Look for classes with similar interfaces
        class_nodes = self.dependency_graph.get_nodes_by_type("class")
        
        # Group classes by their methods
        class_methods: Dict[str, Set[str]] = {}
        for class_node in class_nodes:
            methods = set()
            for edge in self.dependency_graph.get_outgoing_edges(class_node.id):
                if edge.type == "contains":
                    target_node = self.dependency_graph.get_node(edge.target_id)
                    if target_node and target_node.type == "method":
                        methods.add(target_node.name)
            class_methods[class_node.id] = methods
        
        # Find classes with similar methods
        for class1_id, methods1 in class_methods.items():
            for class2_id, methods2 in class_methods.items():
                if class1_id != class2_id:
                    # Calculate the intersection of methods
                    common_methods = methods1.intersection(methods2)
                    
                    # Check if there are enough common methods
                    if len(common_methods) >= 3:
                        class1_node = self.dependency_graph.get_node(class1_id)
                        class2_node = self.dependency_graph.get_node(class2_id)
                        
                        if class1_node and class2_node:
                            # Get the method nodes
                            method_nodes = []
                            for edge in self.dependency_graph.get_outgoing_edges(class1_id):
                                if edge.type == "contains":
                                    target_node = self.dependency_graph.get_node(edge.target_id)
                                    if target_node and target_node.type == "method" and target_node.name in common_methods:
                                        method_nodes.append(target_node)
                            
                            # Create an opportunity
                            opportunity = RefactoringOpportunity(
                                type="extract_interface",
                                description=f"Extract interface from {class1_node.name} and {class2_node.name}",
                                nodes=[class1_node, class2_node] + method_nodes,
                                edges=[],
                                benefit="Improved abstraction and code reuse",
                                effort="medium",
                                suggestion=f"Create an interface with the common methods: {', '.join(common_methods)}"
                            )
                            opportunities.append(opportunity)
        
        return opportunities
