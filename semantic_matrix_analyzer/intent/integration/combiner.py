"""
Intent combiner for combining intents from different sources.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.models.intent import Intent
from semantic_matrix_analyzer.intent.config.configuration import Configuration, ConfigurableAnalyzer

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
