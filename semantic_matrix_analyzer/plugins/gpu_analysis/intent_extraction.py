"""
Intent Extraction System for GPU Analysis Plugin.

This module provides functionality for determining user intentions
from code, text, and feedback using pattern matching and context.
"""

from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass

# Use relative imports to avoid circular imports
from .dynamic_config import ConfigObserver
from .pattern_extraction import Pattern, PatternMatch, PatternExtractor


@dataclass
class Intent:
    """Represents a detected intent."""

    id: str  # Unique identifier
    name: str  # Human-readable name
    confidence: float  # Confidence in the intent (0.0 to 1.0)
    patterns: List[PatternMatch]  # Patterns that contributed to this intent
    context: Dict[str, Any]  # Additional context
    metadata: Dict[str, Any]  # Additional metadata


class IntentExtractor(ConfigObserver):
    """
    Extracts user intents from content using pattern matching.

    This class provides:
    1. Intent extraction from various sources
    2. Intent confidence scoring
    3. Intent-to-pattern mapping
    """

    def __init__(self, config_manager, pattern_extractor: PatternExtractor):
        """
        Initialize the intent extractor.

        Args:
            config_manager: Dynamic configuration manager
            pattern_extractor: Pattern extractor
        """
        self.config_manager = config_manager
        self.pattern_extractor = pattern_extractor

        # Register as observer for configuration changes
        self.config_manager.register_observer(self)

        # Initialize default intent configuration if not present
        config = config_manager.get_config()
        if "intents" not in config:
            intent_config = {
                "intents": {
                    "min_confidence_threshold": 0.6,
                    "max_intents_per_content": 5,
                    "intent_types": {
                        "code_improvement": {
                            "enabled": True,
                            "weight": 1.0,
                            "patterns": ["function", "class", "optimize"]
                        },
                        "bug_fix": {
                            "enabled": True,
                            "weight": 1.0,
                            "patterns": ["fix", "bug", "issue", "error"]
                        },
                        "feature_request": {
                            "enabled": True,
                            "weight": 0.9,
                            "patterns": ["add", "feature", "implement", "new"]
                        }
                    }
                }
            }
            config_manager.update_config(intent_config, "system")

    def on_config_changed(self, config: Dict[str, Any], changed_keys: Set[str], source: str) -> None:
        """
        Called when configuration changes.

        Args:
            config: The new configuration
            changed_keys: Set of keys that were changed
            source: Source of the change (e.g., "user", "system", "feedback")
        """
        # No specific action needed for now
        pass

    def extract_intent(self, content: str, content_type: str,
                      context: Optional[Dict[str, Any]] = None) -> List[Intent]:
        """
        Extract intents from content.

        Args:
            content: Content to extract intents from
            content_type: Type of content ("code", "text", "behavior")
            context: Optional context information

        Returns:
            List of extracted intents
        """
        config = self.config_manager.get_config()
        intents = []

        # Extract patterns from content
        patterns = self.pattern_extractor.extract_patterns(content, content_type, context)

        # Match against known patterns
        matches = self.pattern_extractor.match_patterns(content, content_type)

        # Combine all pattern matches
        all_matches = matches + [
            PatternMatch(
                pattern=p,
                content=content,
                score=1.0,  # New patterns are exact matches
                locations=[],  # No specific locations for new patterns
                context=context or {}
            ) for p in patterns
        ]

        # Map patterns to intents
        intent_matches: Dict[str, List[PatternMatch]] = {}

        for match in all_matches:
            pattern_content = match.pattern.content

            # Check which intents this pattern contributes to
            for intent_type, intent_config in config["intents"]["intent_types"].items():
                if not intent_config.get("enabled", True):
                    continue

                # Check if pattern matches any of the intent's patterns
                for intent_pattern in intent_config.get("patterns", []):
                    if intent_pattern.lower() in pattern_content.lower():
                        if intent_type not in intent_matches:
                            intent_matches[intent_type] = []
                        intent_matches[intent_type].append(match)
                        break

        # Create intent objects
        for intent_type, matches in intent_matches.items():
            # Calculate confidence based on pattern matches
            total_score = sum(match.score * match.pattern.weight for match in matches)
            avg_score = total_score / len(matches) if matches else 0

            # Apply intent type weight
            intent_weight = config["intents"]["intent_types"][intent_type].get("weight", 1.0)
            confidence = avg_score * intent_weight

            # Only include intents above threshold
            if confidence >= config["intents"]["min_confidence_threshold"]:
                intents.append(Intent(
                    id=f"{intent_type}_{hash(content) % 10000}",
                    name=intent_type.replace("_", " ").title(),
                    confidence=confidence,
                    patterns=matches,
                    context=context or {},
                    metadata={
                        "intent_type": intent_type,
                        "pattern_count": len(matches)
                    }
                ))

        # Sort by confidence (highest first)
        intents.sort(key=lambda i: i.confidence, reverse=True)

        # Limit number of intents
        max_intents = config["intents"]["max_intents_per_content"]
        return intents[:max_intents]
