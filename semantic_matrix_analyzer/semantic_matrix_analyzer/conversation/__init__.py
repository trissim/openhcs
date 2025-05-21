"""
Conversation module for Semantic Matrix Analyzer.

This module provides functionality for extracting intents from conversations
and analyzing code based on those intents.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.patterns import Intent

logger = logging.getLogger(__name__)


class ConversationIntentExtractor:
    """Extracts intents from conversations."""

    @staticmethod
    def extract_intents_from_text(conversation_text: str) -> List[Intent]:
        """Extract intents from conversation text.

        Args:
            conversation_text: The conversation text.

        Returns:
            A list of extracted intents.
        """
        intents = []

        # Split the conversation into sections by intent
        intent_sections = re.split(r"(?=Intent:\s*[^\n]+)", conversation_text)

        for section in intent_sections:
            # Skip empty sections
            if not section.strip():
                continue

            # Extract intent name
            intent_match = re.search(r"Intent:\s*([^\n]+)", section)
            if not intent_match:
                continue

            intent_name = intent_match.group(1).strip()
            intent_description = f"Intent extracted from conversation: {intent_name}"

            # Create intent
            intent = Intent(name=intent_name, description=intent_description)

            # Look for patterns associated with this intent
            pattern_matches = re.finditer(
                r"Pattern:\s*([^\n]+)\s*Type:\s*([^\n]+)\s*Description:\s*([^\n]+)(?:\s*Is_negative:\s*(true|false))?",
                section
            )

            for pattern_match in pattern_matches:
                pattern_str = pattern_match.group(1).strip()
                pattern_type = pattern_match.group(2).strip().lower()
                description = pattern_match.group(3).strip()
                is_negative = pattern_match.group(4) == "true" if pattern_match.group(4) else False

                pattern_name = f"{intent_name.lower().replace(' ', '_')}_pattern_{len(intent.patterns) + 1}"

                if pattern_type == "string":
                    intent.add_string_pattern(
                        name=pattern_name,
                        description=description,
                        pattern=pattern_str,
                        weight=1.0,
                        is_negative=is_negative
                    )
                elif pattern_type == "regex":
                    intent.add_regex_pattern(
                        name=pattern_name,
                        description=description,
                        pattern=pattern_str,
                        weight=1.0,
                        is_negative=is_negative
                    )
                elif pattern_type == "ast":
                    # Parse node_type and condition from pattern_str
                    parts = pattern_str.split(",", 1)
                    node_type = parts[0].strip()
                    condition = json.loads(parts[1]) if len(parts) > 1 else None

                    intent.add_ast_pattern(
                        name=pattern_name,
                        description=description,
                        node_type=node_type,
                        condition=condition,
                        weight=1.0,
                        is_negative=is_negative
                    )

            if intent.patterns:
                intents.append(intent)

        return intents

    @staticmethod
    def extract_intents_from_file(file_path: Path) -> List[Intent]:
        """Extract intents from a conversation file.

        Args:
            file_path: Path to the conversation file.

        Returns:
            A list of extracted intents.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            conversation_text = f.read()

        return ConversationIntentExtractor.extract_intents_from_text(conversation_text)

    @staticmethod
    def save_intents_to_file(intents: List[Intent], file_path: Path, append: bool = False) -> None:
        """Save intents to a file.

        Args:
            intents: The intents to save.
            file_path: Path to save the intents to.
            append: Whether to append to an existing file.

        Raises:
            IOError: If the file cannot be written.
        """
        # Convert intents to a serializable format
        intent_data = []
        for intent in intents:
            patterns = []
            for pattern in intent.patterns:
                pattern_data = {
                    "name": pattern.name,
                    "description": pattern.description,
                    "pattern_type": pattern.pattern_type.name.lower(),
                    "pattern": pattern.pattern if isinstance(pattern.pattern, str) else str(pattern.pattern),
                    "weight": pattern.weight,
                    "is_negative": pattern.is_negative
                }
                patterns.append(pattern_data)

            intent_data.append({
                "name": intent.name,
                "description": intent.description,
                "patterns": patterns
            })

        # Create the output directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        if append and file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

                existing_intents = existing_data.get("intents", [])

                # Merge intents
                merged_intents = existing_intents
                for intent in intent_data:
                    # Check if intent already exists
                    existing_intent = next((i for i in merged_intents if i["name"] == intent["name"]), None)
                    if existing_intent:
                        # Merge patterns
                        existing_patterns = existing_intent.get("patterns", [])
                        new_patterns = intent.get("patterns", [])

                        # Add new patterns that don't already exist
                        for pattern in new_patterns:
                            if not any(p["pattern"] == pattern["pattern"] and p["pattern_type"] == pattern["pattern_type"] for p in existing_patterns):
                                existing_patterns.append(pattern)

                        existing_intent["patterns"] = existing_patterns
                    else:
                        merged_intents.append(intent)

                # Save merged intents
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump({"intents": merged_intents}, f, indent=2)
            except Exception as e:
                import logging
                logging.error(f"Error appending intents to file {file_path}: {e}")
                raise
        else:
            # Save new intents
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({"intents": intent_data}, f, indent=2)


# Import memory-related classes for convenience
try:
    from semantic_matrix_analyzer.conversation.memory import (
        Conversation, ConversationEntry, ConversationStore
    )
    from semantic_matrix_analyzer.conversation.memory.agent import (
        AgentFactory, MemoryAugmentedAgent
    )
    from semantic_matrix_analyzer.conversation.memory.context import (
        ContextManager, ConversationContext
    )
    from semantic_matrix_analyzer.conversation.memory.intent_extraction import (
        ConversationMemoryIntentExtractor, IntentMatcher
    )
    from semantic_matrix_analyzer.conversation.memory.knowledge_graph import (
        KnowledgeGraph, KnowledgeGraphBuilder, KnowledgeGraphEntity, KnowledgeGraphRelationship
    )
except ImportError:
    logger.debug("Conversation memory modules not available.")
