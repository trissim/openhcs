"""
Hierarchy builder for building intent hierarchies.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.models.intent import Intent, IntentHierarchy
from semantic_matrix_analyzer.intent.config.configuration import Configuration, ConfigurableAnalyzer

logger = logging.getLogger(__name__)


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
