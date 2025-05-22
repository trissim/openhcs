# plan_05_integration.md
## Component: Intent Integration

### Objective
Create components for integrating the various intent extraction methods, combining intents from different sources, building a hierarchical intent model, and generating reports. These components will provide a unified interface for extracting and analyzing intent from the codebase.

### Plan
1. Create an `IntentCombiner` class for combining intents from different sources
2. Create a `HierarchyBuilder` class for building a hierarchical intent model
3. Create an `IntentRelationshipAnalyzer` for analyzing relationships between intents
4. Create an `IntentReporter` class for generating reports
5. Create a `GapAnalyzer` class for analyzing gaps between intent and implementation
6. Integrate with existing evidence collection and verification systems

### Findings
Integration of the various intent extraction methods is critical for providing a comprehensive understanding of the codebase. By combining intents from names, types, and structure, we can build a hierarchical model that represents the intended behavior of the code.

Key integration points:
- Combining intents from different sources
- Building a hierarchical model of intents
- Analyzing relationships between intents
- Generating reports of extracted intents
- Analyzing gaps between intent and implementation

### Implementation Draft

```python
"""
Integration components for the intent extraction system.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, NameIntent, TypeIntent, StructuralIntent, IntentType,
    IntentHierarchy, CodeLocation
)
from semantic_matrix_analyzer.intent.analyzers.name_analyzer import NameAnalyzer
from semantic_matrix_analyzer.intent.analyzers.type_analyzer import TypeHintAnalyzer
from semantic_matrix_analyzer.intent.analyzers.structural_analyzer import (
    StructuralAnalyzer, ArchitecturalIntentExtractor
)

logger = logging.getLogger(__name__)


class IntentCombiner:
    """Combines intents from different sources."""
    
    def __init__(self):
        """Initialize the intent combiner."""
        pass
    
    def combine_intents(self, intent_lists: List[List[Intent]]) -> List[Intent]:
        """Combine intents from different sources.
        
        Args:
            intent_lists: A list of intent lists from different sources.
            
        Returns:
            A combined list of intents.
        """
        # Flatten the list of lists
        all_intents = [intent for intent_list in intent_lists for intent in intent_list]
        
        # Group intents by location
        intents_by_location = {}
        for intent in all_intents:
            if intent.location:
                location_key = f"{intent.location.file_path}:{intent.location.start_line}-{intent.location.end_line}"
                if location_key not in intents_by_location:
                    intents_by_location[location_key] = []
                intents_by_location[location_key].append(intent)
        
        # Combine intents at the same location
        combined_intents = []
        for location_key, location_intents in intents_by_location.items():
            if len(location_intents) == 1:
                combined_intents.append(location_intents[0])
            else:
                combined_intent = self._combine_intents_at_location(location_intents)
                combined_intents.append(combined_intent)
        
        # Add intents without a location
        for intent in all_intents:
            if not intent.location:
                combined_intents.append(intent)
        
        return combined_intents
    
    def _combine_intents_at_location(self, intents: List[Intent]) -> Intent:
        """Combine intents at the same location.
        
        Args:
            intents: A list of intents at the same location.
            
        Returns:
            A combined intent.
        """
        # Use the intent with the highest confidence as the base
        base_intent = max(intents, key=lambda i: i.confidence)
        
        # Collect related intents
        related_intents = set(base_intent.related_intents)
        for intent in intents:
            if intent.id != base_intent.id:
                related_intents.add(intent.id)
                related_intents.update(intent.related_intents)
        
        # Create a new intent with combined information
        combined_intent = Intent(
            id=base_intent.id,
            name=base_intent.name,
            description=base_intent.description,
            source=base_intent.source,
            type=base_intent.type,
            confidence=base_intent.confidence,
            location=base_intent.location,
            related_intents=list(related_intents),
            metadata=base_intent.metadata.copy()
        )
        
        # Add metadata from other intents
        for intent in intents:
            if intent.id != base_intent.id:
                for key, value in intent.metadata.items():
                    if key not in combined_intent.metadata:
                        combined_intent.metadata[key] = value
        
        return combined_intent


class HierarchyBuilder:
    """Builds a hierarchical intent model."""
    
    def __init__(self):
        """Initialize the hierarchy builder."""
        pass
    
    def build_hierarchy(self, intents: List[Intent]) -> IntentHierarchy:
        """Build a hierarchical intent model.
        
        Args:
            intents: A list of intents.
            
        Returns:
            An IntentHierarchy object.
        """
        hierarchy = IntentHierarchy()
        
        # First, add all intents to the hierarchy
        for intent in intents:
            hierarchy.add_intent(intent)
        
        # Then, establish parent-child relationships
        self._establish_relationships(hierarchy, intents)
        
        return hierarchy
    
    def _establish_relationships(self, hierarchy: IntentHierarchy, intents: List[Intent]) -> None:
        """Establish parent-child relationships between intents.
        
        Args:
            hierarchy: The intent hierarchy.
            intents: A list of intents.
        """
        # Group intents by file
        intents_by_file = {}
        for intent in intents:
            if intent.location:
                file_path = str(intent.location.file_path)
                if file_path not in intents_by_file:
                    intents_by_file[file_path] = []
                intents_by_file[file_path].append(intent)
        
        # For each file, establish relationships based on location
        for file_path, file_intents in intents_by_file.items():
            # Sort intents by location
            file_intents.sort(key=lambda i: (i.location.start_line, i.location.end_line))
            
            # Establish relationships based on containment
            for i, intent1 in enumerate(file_intents):
                for intent2 in file_intents[i+1:]:
                    if (intent1.location.start_line <= intent2.location.start_line and
                        intent1.location.end_line >= intent2.location.end_line):
                        # intent1 contains intent2
                        hierarchy.add_intent(intent2, intent1.id)
        
        # Establish relationships based on related_intents
        for intent in intents:
            for related_id in intent.related_intents:
                related_intent = hierarchy.get_intent(related_id)
                if related_intent:
                    # Check if there's already a parent-child relationship
                    if hierarchy.get_parent(related_id) is None:
                        hierarchy.add_intent(related_intent, intent.id)


class IntentRelationshipAnalyzer:
    """Analyzes relationships between intents."""
    
    def __init__(self):
        """Initialize the intent relationship analyzer."""
        pass
    
    def analyze_relationships(self, hierarchy: IntentHierarchy) -> Dict[str, Any]:
        """Analyze relationships between intents.
        
        Args:
            hierarchy: The intent hierarchy.
            
        Returns:
            A dictionary of relationship analysis results.
        """
        results = {
            "root_intents": len(hierarchy.get_root_intents()),
            "total_intents": len(hierarchy.intents),
            "max_depth": self._calculate_max_depth(hierarchy),
            "intent_types": self._count_intent_types(hierarchy),
            "intent_sources": self._count_intent_sources(hierarchy),
            "orphaned_intents": self._find_orphaned_intents(hierarchy)
        }
        
        return results
    
    def _calculate_max_depth(self, hierarchy: IntentHierarchy) -> int:
        """Calculate the maximum depth of the hierarchy.
        
        Args:
            hierarchy: The intent hierarchy.
            
        Returns:
            The maximum depth.
        """
        max_depth = 0
        
        def calculate_depth(intent_id: str, current_depth: int) -> None:
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            for child in hierarchy.get_children(intent_id):
                calculate_depth(child.id, current_depth + 1)
        
        for root_intent in hierarchy.get_root_intents():
            calculate_depth(root_intent.id, 1)
        
        return max_depth
    
    def _count_intent_types(self, hierarchy: IntentHierarchy) -> Dict[str, int]:
        """Count the number of intents of each type.
        
        Args:
            hierarchy: The intent hierarchy.
            
        Returns:
            A dictionary mapping intent types to counts.
        """
        counts = {}
        
        for intent in hierarchy.intents.values():
            intent_type = intent.type.value
            counts[intent_type] = counts.get(intent_type, 0) + 1
        
        return counts
    
    def _count_intent_sources(self, hierarchy: IntentHierarchy) -> Dict[str, int]:
        """Count the number of intents from each source.
        
        Args:
            hierarchy: The intent hierarchy.
            
        Returns:
            A dictionary mapping intent sources to counts.
        """
        counts = {}
        
        for intent in hierarchy.intents.values():
            source = intent.source.value
            counts[source] = counts.get(source, 0) + 1
        
        return counts
    
    def _find_orphaned_intents(self, hierarchy: IntentHierarchy) -> List[str]:
        """Find intents that are not related to any other intent.
        
        Args:
            hierarchy: The intent hierarchy.
            
        Returns:
            A list of orphaned intent IDs.
        """
        orphaned = []
        
        for intent_id, intent in hierarchy.intents.items():
            if (not intent.related_intents and
                not hierarchy.get_parent(intent_id) and
                not hierarchy.get_children(intent_id)):
                orphaned.append(intent_id)
        
        return orphaned


class IntentReporter:
    """Generates reports of extracted intents."""
    
    def __init__(self):
        """Initialize the intent reporter."""
        pass
    
    def generate_report(self, hierarchy: IntentHierarchy) -> Dict[str, Any]:
        """Generate a report of extracted intents.
        
        Args:
            hierarchy: The intent hierarchy.
            
        Returns:
            A report dictionary.
        """
        # Analyze relationships
        relationship_analyzer = IntentRelationshipAnalyzer()
        relationship_analysis = relationship_analyzer.analyze_relationships(hierarchy)
        
        # Generate the report
        report = {
            "summary": {
                "total_intents": len(hierarchy.intents),
                "root_intents": len(hierarchy.get_root_intents()),
                "max_depth": relationship_analysis["max_depth"],
                "intent_types": relationship_analysis["intent_types"],
                "intent_sources": relationship_analysis["intent_sources"]
            },
            "hierarchy": self._generate_hierarchy_report(hierarchy),
            "orphaned_intents": [
                hierarchy.intents[intent_id].to_dict()
                for intent_id in relationship_analysis["orphaned_intents"]
            ]
        }
        
        return report
    
    def _generate_hierarchy_report(self, hierarchy: IntentHierarchy) -> List[Dict[str, Any]]:
        """Generate a report of the intent hierarchy.
        
        Args:
            hierarchy: The intent hierarchy.
            
        Returns:
            A list of hierarchy node dictionaries.
        """
        report = []
        
        def add_node(intent_id: str, depth: int) -> Dict[str, Any]:
            intent = hierarchy.get_intent(intent_id)
            if not intent:
                return None
            
            node = {
                "id": intent.id,
                "name": intent.name,
                "description": intent.description,
                "type": intent.type.value,
                "source": intent.source.value,
                "confidence": intent.confidence,
                "depth": depth,
                "children": []
            }
            
            for child in hierarchy.get_children(intent_id):
                child_node = add_node(child.id, depth + 1)
                if child_node:
                    node["children"].append(child_node)
            
            return node
        
        for root_intent in hierarchy.get_root_intents():
            root_node = add_node(root_intent.id, 0)
            if root_node:
                report.append(root_node)
        
        return report
    
    def format_report(self, report: Dict[str, Any], format: str = "text") -> str:
        """Format a report in the specified format.
        
        Args:
            report: The report to format.
            format: The format to use ("text", "markdown", "json").
            
        Returns:
            The formatted report.
        """
        if format == "json":
            import json
            return json.dumps(report, indent=2)
        elif format == "markdown":
            return self._format_markdown_report(report)
        else:
            return self._format_text_report(report)
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format a report as plain text.
        
        Args:
            report: The report to format.
            
        Returns:
            The formatted report.
        """
        lines = []
        
        # Add summary
        lines.append("# Intent Analysis Report")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"Total intents: {report['summary']['total_intents']}")
        lines.append(f"Root intents: {report['summary']['root_intents']}")
        lines.append(f"Maximum depth: {report['summary']['max_depth']}")
        lines.append("")
        
        # Add intent types
        lines.append("### Intent Types")
        for intent_type, count in report['summary']['intent_types'].items():
            lines.append(f"{intent_type}: {count}")
        lines.append("")
        
        # Add intent sources
        lines.append("### Intent Sources")
        for source, count in report['summary']['intent_sources'].items():
            lines.append(f"{source}: {count}")
        lines.append("")
        
        # Add hierarchy
        lines.append("## Intent Hierarchy")
        
        def add_hierarchy_node(node: Dict[str, Any]) -> None:
            indent = "  " * node["depth"]
            lines.append(f"{indent}- {node['name']}: {node['description']} ({node['type']}, {node['confidence']:.2f})")
            
            for child in node["children"]:
                add_hierarchy_node(child)
        
        for root_node in report["hierarchy"]:
            add_hierarchy_node(root_node)
        
        # Add orphaned intents
        if report["orphaned_intents"]:
            lines.append("")
            lines.append("## Orphaned Intents")
            
            for intent in report["orphaned_intents"]:
                lines.append(f"- {intent['name']}: {intent['description']} ({intent['type']}, {intent['confidence']:.2f})")
        
        return "\n".join(lines)
    
    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format a report as Markdown.
        
        Args:
            report: The report to format.
            
        Returns:
            The formatted report.
        """
        # Similar to _format_text_report, but with Markdown formatting
        return self._format_text_report(report)


class GapAnalyzer:
    """Analyzes gaps between intent and implementation."""
    
    def __init__(self):
        """Initialize the gap analyzer."""
        pass
    
    def analyze_gaps(self, intent_hierarchy: IntentHierarchy, actual_implementation: Any) -> List[Dict[str, Any]]:
        """Analyze gaps between intent and implementation.
        
        Args:
            intent_hierarchy: The intent hierarchy.
            actual_implementation: The actual implementation.
            
        Returns:
            A list of gap dictionaries.
        """
        # This is a placeholder implementation
        # In a real implementation, we would compare the intent hierarchy with the actual implementation
        
        return []
```
