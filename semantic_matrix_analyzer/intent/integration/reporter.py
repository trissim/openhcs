"""
Reporter for generating reports of extracted intents.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.models.intent import Intent, IntentHierarchy
from semantic_matrix_analyzer.intent.config.configuration import Configuration, ConfigurableAnalyzer

logger = logging.getLogger(__name__)


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
        lines = []

        # Add summary
        lines.append("# Intent Analysis Report")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- **Total intents**: {report['summary']['total_intents']}")
        lines.append(f"- **Root intents**: {report['summary']['root_intents']}")
        lines.append(f"- **Maximum depth**: {report['summary']['max_depth']}")
        lines.append("")

        # Add intent types
        lines.append("### Intent Types")
        for intent_type, count in report['summary']['intent_types'].items():
            lines.append(f"- **{intent_type}**: {count}")
        lines.append("")

        # Add intent sources
        lines.append("### Intent Sources")
        for source, count in report['summary']['intent_sources'].items():
            lines.append(f"- **{source}**: {count}")
        lines.append("")

        # Add hierarchy
        lines.append("## Intent Hierarchy")
        lines.append("")

        def add_hierarchy_node(node: Dict[str, Any]) -> None:
            indent = "  " * node["depth"]
            lines.append(f"{indent}- **{node['name']}**: {node['description']} (*{node['type']}*, {node['confidence']:.2f})")

            for child in node["children"]:
                add_hierarchy_node(child)

        for root_node in report["hierarchy"]:
            add_hierarchy_node(root_node)

        # Add orphaned intents
        if report["orphaned_intents"]:
            lines.append("")
            lines.append("## Orphaned Intents")
            lines.append("")

            for intent in report["orphaned_intents"]:
                lines.append(f"- **{intent['name']}**: {intent['description']} (*{intent['type']}*, {intent['confidence']:.2f})")

        return "\n".join(lines)
