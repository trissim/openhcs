#!/usr/bin/env python3
"""
Conversation Intent Plugin

This plugin extracts intents from conversations and converts them to AST patterns
that can be used to check for correctness in code.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_matrix_analyzer import IntentPlugin, Intent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("conversation_plugin")


class ConversationIntentPlugin(IntentPlugin):
    """Plugin for extracting intents from conversations."""

    @staticmethod
    def get_intents() -> List[Intent]:
        """Get intents from conversation files."""
        intents = []

        # Look for conversation files in the conversations directory
        conversations_dir = Path("conversations")
        if not conversations_dir.exists():
            # Create the directory if it doesn't exist
            conversations_dir.mkdir(exist_ok=True)
            logger.info(f"Created conversations directory: {conversations_dir}")

            # Create a sample conversation file
            sample_file = conversations_dir / "sample_conversation.json"
            sample_conversation = {
                "intents": [
                    {
                        "name": "Clean Code",
                        "description": "Writing clean, maintainable code",
                        "patterns": [
                            {
                                "name": "meaningful_names",
                                "description": "Using meaningful variable and function names",
                                "pattern_type": "string",
                                "pattern": "meaningful name",
                                "weight": 1.0
                            },
                            {
                                "name": "small_functions",
                                "description": "Writing small, focused functions",
                                "pattern_type": "regex",
                                "pattern": r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:\s*(?:\s*\"\"\"[^\"]*\"\"\"\s*)?(?:[^\n]*\n){1,10}\s*return",
                                "weight": 0.8
                            },
                            {
                                "name": "comments",
                                "description": "Using comments to explain complex code",
                                "pattern_type": "string",
                                "pattern": "# ",
                                "weight": 0.5
                            }
                        ]
                    },
                    {
                        "name": "SOLID Principles",
                        "description": "Following SOLID design principles",
                        "patterns": [
                            {
                                "name": "single_responsibility",
                                "description": "Single Responsibility Principle",
                                "pattern_type": "string",
                                "pattern": "single responsibility",
                                "weight": 1.0
                            },
                            {
                                "name": "open_closed",
                                "description": "Open/Closed Principle",
                                "pattern_type": "string",
                                "pattern": "open for extension",
                                "weight": 1.0
                            },
                            {
                                "name": "liskov_substitution",
                                "description": "Liskov Substitution Principle",
                                "pattern_type": "string",
                                "pattern": "liskov substitution",
                                "weight": 1.0
                            },
                            {
                                "name": "interface_segregation",
                                "description": "Interface Segregation Principle",
                                "pattern_type": "string",
                                "pattern": "interface segregation",
                                "weight": 1.0
                            },
                            {
                                "name": "dependency_inversion",
                                "description": "Dependency Inversion Principle",
                                "pattern_type": "string",
                                "pattern": "dependency inversion",
                                "weight": 1.0
                            }
                        ]
                    }
                ]
            }

            with open(sample_file, "w", encoding="utf-8") as f:
                json.dump(sample_conversation, f, indent=2)

            logger.info(f"Created sample conversation file: {sample_file}")

        # Load intents from conversation files
        for conversation_file in conversations_dir.glob("*.json"):
            try:
                with open(conversation_file, "r", encoding="utf-8") as f:
                    conversation_data = json.load(f)

                if "intents" in conversation_data:
                    for intent_data in conversation_data["intents"]:
                        intent = ConversationIntentPlugin._create_intent_from_data(intent_data)
                        if intent:
                            intents.append(intent)
            except Exception as e:
                logger.error(f"Error loading conversation file {conversation_file}: {e}")

        return intents

    @staticmethod
    def _create_intent_from_data(intent_data: Dict[str, Any]) -> Optional[Intent]:
        """Create an Intent object from intent data."""
        try:
            name = intent_data.get("name")
            description = intent_data.get("description", "")

            if not name:
                logger.warning("Intent data missing name, skipping")
                return None

            intent = Intent(name=name, description=description)

            patterns = intent_data.get("patterns", [])
            for pattern_data in patterns:
                pattern_name = pattern_data.get("name")
                pattern_description = pattern_data.get("description", "")
                pattern_type = pattern_data.get("pattern_type")
                pattern = pattern_data.get("pattern")
                weight = pattern_data.get("weight", 1.0)
                is_negative = pattern_data.get("is_negative", False)

                if not pattern_name or not pattern_type or not pattern:
                    logger.warning(f"Pattern data for intent '{name}' missing required fields, skipping")
                    continue

                if pattern_type == "string":
                    intent.add_string_pattern(
                        name=pattern_name,
                        description=pattern_description,
                        pattern=pattern,
                        weight=weight,
                        is_negative=is_negative
                    )
                elif pattern_type == "regex":
                    intent.add_regex_pattern(
                        name=pattern_name,
                        description=pattern_description,
                        pattern=pattern,
                        weight=weight,
                        is_negative=is_negative
                    )
                else:
                    logger.warning(f"Unknown pattern type '{pattern_type}' for intent '{name}', skipping")

            return intent

        except Exception as e:
            logger.error(f"Error creating intent from data: {e}")
            return None

    @staticmethod
    def extract_intents_from_conversation(conversation_text: str) -> List[Dict[str, Any]]:
        """Extract intents from a conversation text."""
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

            # Look for patterns associated with this intent
            patterns = []
            pattern_matches = re.finditer(
                r"Pattern:\s*([^\n]+)\s*Type:\s*([^\n]+)\s*Description:\s*([^\n]+)(?:\s*Is_negative:\s*(true|false))?",
                section
            )

            for pattern_match in pattern_matches:
                pattern = pattern_match.group(1).strip()
                pattern_type = pattern_match.group(2).strip().lower()
                description = pattern_match.group(3).strip()
                is_negative = pattern_match.group(4) == "true" if pattern_match.group(4) else False

                if pattern_type in ["string", "regex"]:
                    patterns.append({
                        "name": f"{intent_name.lower().replace(' ', '_')}_pattern_{len(patterns) + 1}",
                        "description": description,
                        "pattern_type": pattern_type,
                        "pattern": pattern,
                        "weight": 1.0,
                        "is_negative": is_negative
                    })

            if patterns:
                intents.append({
                    "name": intent_name,
                    "description": f"Intent extracted from conversation: {intent_name}",
                    "patterns": patterns
                })

        return intents

    @staticmethod
    def save_intents_to_file(intents: List[Dict[str, Any]], file_path: str) -> None:
        """Save intents to a file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({"intents": intents}, f, indent=2)

            logger.info(f"Saved intents to file: {file_path}")
        except Exception as e:
            logger.error(f"Error saving intents to file {file_path}: {e}")


# For testing
if __name__ == "__main__":
    plugin = ConversationIntentPlugin()
    intents = plugin.get_intents()

    print(f"Loaded {len(intents)} intents from conversations:")
    for intent in intents:
        print(f"- {intent.name}: {len(intent.patterns)} patterns")
