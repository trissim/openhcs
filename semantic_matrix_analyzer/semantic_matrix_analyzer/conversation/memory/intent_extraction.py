"""
Intent extraction module for conversation memory.

This module provides functionality for extracting intents from conversations
and storing them with the conversations.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.conversation import ConversationIntentExtractor
from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore
from semantic_matrix_analyzer.patterns import Intent

logger = logging.getLogger(__name__)


class ConversationMemoryIntentExtractor:
    """Extracts intents from conversations and stores them with the conversations."""

    def __init__(self, conversation_store: ConversationStore):
        """Initialize the intent extractor.

        Args:
            conversation_store: The conversation store to use.
        """
        self.conversation_store = conversation_store
        self.intent_extractor = ConversationIntentExtractor()

    def extract_intents_from_conversation(self, conversation: Conversation) -> Dict[str, Any]:
        """Extract intents from a conversation and store them with the conversation.

        Args:
            conversation: The conversation to extract intents from.

        Returns:
            A dictionary mapping intent names to intent data.
        """
        try:
            # Extract user messages
            user_messages = [entry.message for entry in conversation.entries
                            if entry.speaker == "user"]

            if not user_messages:
                return {}

            # Combine user messages
            combined_message = "\n".join(user_messages)

            # Extract intents
            intents = self.intent_extractor.extract_intents_from_text(combined_message)

            # If no intents were extracted, add some default intents based on keywords
            if not intents:
                # Check for clean code keywords
                if "clean code" in combined_message.lower() or "code quality" in combined_message.lower():
                    clean_code_intent = Intent(name="Clean Code", description="Writing clean, maintainable code")
                    clean_code_intent.add_string_pattern(
                        name="clean_code",
                        description="Clean code pattern",
                        pattern="clean code",
                        weight=1.0
                    )
                    intents.append(clean_code_intent)

                # Check for error handling keywords
                if "error" in combined_message.lower() or "exception" in combined_message.lower():
                    error_handling_intent = Intent(name="Error Handling", description="Proper error handling and reporting")
                    error_handling_intent.add_string_pattern(
                        name="error_handling",
                        description="Error handling pattern",
                        pattern="error handling",
                        weight=1.0
                    )
                    intents.append(error_handling_intent)

            # Convert intents to a serializable format
            intent_data = {}
            for intent in intents:
                intent_data[intent.name] = self._intent_to_dict(intent)

            # Store the extracted intents with the conversation
            conversation.extracted_intents = intent_data
            self.conversation_store.save_conversation(conversation)

            return intent_data
        except Exception as e:
            logger.error(f"Error extracting intents from conversation {conversation.id}: {e}")
            return {}

    def extract_intents_from_all_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Extract intents from all conversations.

        Returns:
            A dictionary mapping conversation IDs to dictionaries mapping intent names to intent data.
        """
        results = {}

        for conversation in self.conversation_store.get_all_conversations():
            intent_data = self.extract_intents_from_conversation(conversation)
            results[conversation.id] = intent_data

        return results

    def _intent_to_dict(self, intent: Intent) -> Dict[str, Any]:
        """Convert an intent to a dictionary for serialization.

        Args:
            intent: The intent to convert.

        Returns:
            A dictionary representation of the intent.
        """
        patterns = []
        for pattern in intent.patterns:
            pattern_data = {
                "name": pattern.name,
                "description": pattern.description,
                "pattern_type": pattern.pattern_type.name.lower(),
                "pattern": str(pattern.pattern),
                "weight": pattern.weight,
                "is_negative": pattern.is_negative
            }
            patterns.append(pattern_data)

        return {
            "name": intent.name,
            "description": intent.description,
            "patterns": patterns
        }


class IntentMatcher:
    """Matches intents against text."""

    def __init__(self, conversation_store: ConversationStore):
        """Initialize the intent matcher.

        Args:
            conversation_store: The conversation store to use.
        """
        self.conversation_store = conversation_store

    def find_matching_intents(self, text: str) -> List[Tuple[str, float]]:
        """Find intents that match the given text.

        Args:
            text: The text to match against.

        Returns:
            A list of (intent_name, score) tuples, sorted by score in descending order.
        """
        matches = []

        # Get all intents from all conversations
        for conversation in self.conversation_store.get_all_conversations():
            for intent_name, intent_data in conversation.extracted_intents.items():
                score = self._calculate_match_score(text, intent_data)
                if score > 0:
                    matches.append((intent_name, score))

        # Remove duplicates and sort by score
        unique_matches = {}
        for intent_name, score in matches:
            if intent_name not in unique_matches or score > unique_matches[intent_name]:
                unique_matches[intent_name] = score

        return sorted([(k, v) for k, v in unique_matches.items()], key=lambda x: x[1], reverse=True)

    def _calculate_match_score(self, text: str, intent_data: Dict[str, Any]) -> float:
        """Calculate a match score for an intent against text.

        Args:
            text: The text to match against.
            intent_data: The intent data.

        Returns:
            A score between 0 and 1, where 0 means no match and 1 means a perfect match.
        """
        if "patterns" not in intent_data:
            return 0.0

        total_weight = 0.0
        matched_weight = 0.0

        for pattern in intent_data["patterns"]:
            weight = pattern.get("weight", 1.0)
            total_weight += weight

            pattern_type = pattern.get("pattern_type", "string").lower()
            pattern_value = pattern.get("pattern", "")
            is_negative = pattern.get("is_negative", False)

            if pattern_type == "string":
                if pattern_value in text:
                    if not is_negative:
                        matched_weight += weight
                elif is_negative:
                    matched_weight += weight
            elif pattern_type == "regex":
                try:
                    import re
                    if re.search(pattern_value, text):
                        if not is_negative:
                            matched_weight += weight
                    elif is_negative:
                        matched_weight += weight
                except Exception:
                    pass

        if total_weight == 0:
            return 0.0

        return matched_weight / total_weight
