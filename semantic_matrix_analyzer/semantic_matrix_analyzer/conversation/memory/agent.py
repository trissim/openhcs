"""
AI agent integration module for conversation memory.

This module provides functionality for integrating AI agents with the conversation memory system.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.conversation.memory import ConversationStore
from semantic_matrix_analyzer.conversation.memory.context import ContextManager
from semantic_matrix_analyzer.conversation.memory.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class MemoryAugmentedAgent:
    """An AI agent that uses conversation memory."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        knowledge_graph: KnowledgeGraph
    ):
        """Initialize the memory-augmented agent.

        Args:
            conversation_store: The conversation store to use.
            knowledge_graph: The knowledge graph to use.
        """
        self.conversation_store = conversation_store
        self.knowledge_graph = knowledge_graph
        self.context_manager = ContextManager(conversation_store, knowledge_graph)
        self.current_context = None

    def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Start a new conversation or continue an existing one.

        Args:
            conversation_id: The ID of the conversation, or None to create a new conversation.

        Returns:
            The ID of the conversation.
        """
        self.current_context = self.context_manager.get_context(conversation_id)
        return self.current_context.conversation.id

    def process_message(self, message: str) -> str:
        """Process a message from the user.

        Args:
            message: The message from the user.

        Returns:
            The response from the agent.
        """
        if not self.current_context:
            self.start_conversation()

        # Add the user's message to the conversation
        self.current_context.add_entry("user", message)

        # Generate a response based on the conversation context
        response = self._generate_response(message)

        # Add the AI's response to the conversation
        self.current_context.add_entry("ai", response)

        return response

    def _generate_response(self, message: str) -> str:
        """Generate a response based on the conversation context.

        Args:
            message: The message from the user.

        Returns:
            The response from the agent.
        """
        # Get active intents and preferences
        active_intents = self.current_context.get_active_intents()
        active_preferences = self.current_context.get_active_preferences()

        # Find related intents
        related_intents = self.current_context.get_related_intents(message)

        # This is a placeholder for response generation
        # In a real implementation, this would use the active intents, preferences,
        # and related intents to generate a response

        # For now, we'll just return a simple response
        if active_intents:
            intent_list = ", ".join(active_intents)
            return f"I understand you're interested in: {intent_list}. How can I help with that?"
        elif related_intents:
            intent_name, score = related_intents[0]
            return f"I see you might be interested in {intent_name}. Is that correct?"
        else:
            return "How can I help you today?"

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history.

        Returns:
            A list of conversation entries as dictionaries.
        """
        if not self.current_context:
            return []

        return self.current_context.get_conversation_history()

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation.

        Returns:
            A dictionary with conversation summary information.
        """
        if not self.current_context:
            return {}

        return self.current_context.get_conversation_summary()

    def get_active_intents(self) -> Set[str]:
        """Get the active intents for the current conversation.

        Returns:
            A set of active intent names.
        """
        if not self.current_context:
            return set()

        return self.current_context.get_active_intents()

    def get_active_preferences(self) -> Dict[str, Any]:
        """Get the active preferences for the current conversation.

        Returns:
            A dictionary of active preferences.
        """
        if not self.current_context:
            return {}

        return self.current_context.get_active_preferences()

    def close_conversation(self) -> None:
        """Close the current conversation."""
        if self.current_context:
            self.context_manager.close_context(self.current_context.conversation.id)
            self.current_context = None


class AgentFactory:
    """Factory for creating memory-augmented agents."""

    @staticmethod
    def create_agent(
        storage_dir: Union[str, Path],
        knowledge_graph_path: Optional[Union[str, Path]] = None
    ) -> MemoryAugmentedAgent:
        """Create a memory-augmented agent.

        Args:
            storage_dir: Directory to store conversations.
            knowledge_graph_path: Path to store the knowledge graph (optional).

        Returns:
            A memory-augmented agent.
        """
        # Create the conversation store
        conversation_store = ConversationStore(storage_dir)

        # Create the knowledge graph
        knowledge_graph = KnowledgeGraph(knowledge_graph_path)

        # Create the agent
        return MemoryAugmentedAgent(conversation_store, knowledge_graph)
