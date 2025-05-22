# plan_09_configurable_integration.md
## Component: Configurable Integration

### Objective
Make the integration components of the Structural Intent Analysis system configurable, allowing users to customize how intents are combined, how the hierarchy is built, and how reports are generated.

### Plan
1. Update the `IntentCombiner` to use configuration for combining intents
2. Update the `HierarchyBuilder` to use configuration for building the hierarchy
3. Update the `IntentReporter` to use configuration for generating reports
4. Create a `ConfigurableIntentAnalyzer` class that orchestrates the entire analysis process
5. Implement validation for integration-specific configuration

### Findings
Configurable integration components allow users to customize how the different analyzers work together and how the results are presented. This is important for adapting the system to different use cases and preferences.

### Implementation Draft

```python
"""
Configurable integration components for the Structural Intent Analysis system.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, NameIntent, TypeIntent, StructuralIntent, IntentType,
    IntentHierarchy, CodeLocation
)
from semantic_matrix_analyzer.intent.config.configuration import (
    Configuration, ConfigurableAnalyzer
)
from semantic_matrix_analyzer.intent.analyzers.configurable_name_analyzer import ConfigurableNameAnalyzer
from semantic_matrix_analyzer.intent.analyzers.configurable_type_analyzer import ConfigurableTypeHintAnalyzer
from semantic_matrix_analyzer.intent.analyzers.configurable_structural_analyzer import ConfigurableStructuralAnalyzer

logger = logging.getLogger(__name__)


class ConfigurableIntentCombiner(ConfigurableAnalyzer):
    """Configurable combiner for intents from different sources."""
    
    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the configurable intent combiner.
        
        Args:
            config: The configuration to use (optional).
        """
        super().__init__(config)
    
    def _get_config_section(self) -> str:
        """Get the configuration section for this analyzer.
        
        Returns:
            The configuration section name.
        """
        return "integration"
    
    def is_enabled(self) -> bool:
        """Check if the combiner is enabled.
        
        Returns:
            True if the combiner is enabled, False otherwise.
        """
        return self.get_config_value("combine_intents", True)
    
    def combine_intents(self, intent_lists: List[List[Intent]]) -> List[Intent]:
        """Combine intents from different sources.
        
        Args:
            intent_lists: A list of intent lists from different sources.
            
        Returns:
            A combined list of intents.
        """
        # Check if the combiner is enabled
        if not self.is_enabled():
            # If disabled, just flatten the list of lists
            return [intent for intent_list in intent_lists for intent in intent_list]
        
        # Flatten the list of lists
        all_intents = [intent for intent_list in intent_lists for intent in intent_list]
        
        # Filter by minimum confidence
        min_confidence = self.get_config_value("min_confidence", 0.3)
        all_intents = [intent for intent in all_intents if intent.confidence >= min_confidence]
        
        # Limit the number of results
        max_results = self.get_config_value("max_results", 100)
        if len(all_intents) > max_results:
            # Sort by confidence and take the top max_results
            all_intents.sort(key=lambda i: i.confidence, reverse=True)
            all_intents = all_intents[:max_results]
        
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


class ConfigurableHierarchyBuilder(ConfigurableAnalyzer):
    """Configurable builder for intent hierarchies."""
    
    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the configurable hierarchy builder.
        
        Args:
            config: The configuration to use (optional).
        """
        super().__init__(config)
    
    def _get_config_section(self) -> str:
        """Get the configuration section for this analyzer.
        
        Returns:
            The configuration section name.
        """
        return "integration"
    
    def is_enabled(self) -> bool:
        """Check if the builder is enabled.
        
        Returns:
            True if the builder is enabled, False otherwise.
        """
        return self.get_config_value("build_hierarchy", True)
    
    def build_hierarchy(self, intents: List[Intent]) -> IntentHierarchy:
        """Build a hierarchical intent model.
        
        Args:
            intents: A list of intents.
            
        Returns:
            An IntentHierarchy object.
        """
        hierarchy = IntentHierarchy()
        
        # Check if the builder is enabled
        if not self.is_enabled():
            # If disabled, just add all intents to the hierarchy without relationships
            for intent in intents:
                hierarchy.add_intent(intent)
            return hierarchy
        
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


class ConfigurableIntentReporter(ConfigurableAnalyzer):
    """Configurable reporter for intent analysis."""
    
    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the configurable intent reporter.
        
        Args:
            config: The configuration to use (optional).
        """
        super().__init__(config)
    
    def _get_config_section(self) -> str:
        """Get the configuration section for this analyzer.
        
        Returns:
            The configuration section name.
        """
        return "integration"
    
    def generate_report(self, hierarchy: IntentHierarchy) -> Dict[str, Any]:
        """Generate a report of extracted intents.
        
        Args:
            hierarchy: The intent hierarchy.
            
        Returns:
            A report dictionary.
        """
        # Create a relationship analyzer
        relationship_analyzer = IntentRelationshipAnalyzer()
        
        # Analyze relationships
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
    
    def format_report(self, report: Dict[str, Any], format: Optional[str] = None) -> str:
        """Format a report in the specified format.
        
        Args:
            report: The report to format.
            format: The format to use ("text", "markdown", "json").
            
        Returns:
            The formatted report.
        """
        # Use the configured format if none is specified
        if format is None:
            format = self.get_config_value("report_format", "text")
        
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


class ConfigurableIntentAnalyzer:
    """Configurable analyzer for extracting intent from code structure."""
    
    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the configurable intent analyzer.
        
        Args:
            config: The configuration to use (optional).
        """
        self.config = config or Configuration()
        self.name_analyzer = ConfigurableNameAnalyzer(self.config)
        self.type_analyzer = ConfigurableTypeHintAnalyzer(self.config)
        self.intent_combiner = ConfigurableIntentCombiner(self.config)
        self.hierarchy_builder = ConfigurableHierarchyBuilder(self.config)
        self.intent_reporter = ConfigurableIntentReporter(self.config)
    
    def analyze_codebase(self, file_paths: List[Path], dependency_graph=None) -> Dict[str, Any]:
        """Analyze a codebase to extract intent.
        
        Args:
            file_paths: A list of file paths to analyze.
            dependency_graph: A dependency graph (optional).
            
        Returns:
            A report of the analysis.
        """
        # Analyze names
        name_intents = self.analyze_names(file_paths)
        
        # Analyze type hints
        type_intents = self.analyze_type_hints(file_paths)
        
        # Analyze structure
        structural_intents = []
        if dependency_graph:
            structural_analyzer = ConfigurableStructuralAnalyzer(dependency_graph, self.config)
            structural_intents = structural_analyzer.analyze_dependency_graph()
        
        # Combine intents
        combined_intents = self.intent_combiner.combine_intents([name_intents, type_intents, structural_intents])
        
        # Build hierarchy
        intent_hierarchy = self.hierarchy_builder.build_hierarchy(combined_intents)
        
        # Generate report
        report = self.intent_reporter.generate_report(intent_hierarchy)
        
        return report
    
    def analyze_names(self, file_paths: List[Path]) -> List[NameIntent]:
        """Analyze names in the codebase.
        
        Args:
            file_paths: A list of file paths.
            
        Returns:
            A list of NameIntent objects.
        """
        name_intents = []
        
        for file_path in file_paths:
            # Parse the file
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            
            tree = ast.parse(code, filename=str(file_path))
            
            # Analyze class names
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    intent = self.name_analyzer.analyze_class_name(
                        node.name,
                        file_path,
                        node.lineno
                    )
                    name_intents.append(intent)
                
                elif isinstance(node, ast.FunctionDef):
                    # Check if this is a method
                    is_method = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef) and node in parent.body:
                            is_method = True
                            break
                    
                    if is_method:
                        intent = self.name_analyzer.analyze_method_name(
                            node.name,
                            file_path,
                            node.lineno
                        )
                    else:
                        intent = self.name_analyzer.analyze_name(
                            node.name,
                            "function",
                            file_path,
                            node.lineno
                        )
                    name_intents.append(intent)
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            intent = self.name_analyzer.analyze_variable_name(
                                target.id,
                                file_path,
                                node.lineno
                            )
                            name_intents.append(intent)
        
        return name_intents
    
    def analyze_type_hints(self, file_paths: List[Path]) -> List[TypeIntent]:
        """Analyze type hints in the codebase.
        
        Args:
            file_paths: A list of file paths.
            
        Returns:
            A list of TypeIntent objects.
        """
        type_intents = []
        
        # Create a type hint extractor
        type_hint_extractor = TypeHintExtractor()
        
        for file_path in file_paths:
            # Parse the file
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            
            tree = ast.parse(code, filename=str(file_path))
            
            # Extract type hints from functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    type_hints = type_hint_extractor.extract_type_hints_from_ast(node)
                    
                    # Analyze parameter type hints
                    for param_name, type_hint in type_hints.items():
                        if param_name != "return":
                            intent = self.type_analyzer.analyze_parameter_type(
                                param_name,
                                type_hint,
                                file_path,
                                node.lineno
                            )
                            type_intents.append(intent)
                    
                    # Analyze return type hint
                    if "return" in type_hints:
                        intent = self.type_analyzer.analyze_return_type(
                            type_hints["return"],
                            file_path,
                            node.lineno
                        )
                        type_intents.append(intent)
        
        return type_intents
    
    def format_report(self, report: Dict[str, Any], format: Optional[str] = None) -> str:
        """Format a report in the specified format.
        
        Args:
            report: The report to format.
            format: The format to use ("text", "markdown", "json").
            
        Returns:
            The formatted report.
        """
        return self.intent_reporter.format_report(report, format)
```
