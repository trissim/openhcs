"""
Conversation context module for conversation memory.

This module provides functionality for managing conversation context across sessions.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore
from semantic_matrix_analyzer.conversation.memory.intent_extraction import ConversationMemoryIntentExtractor, IntentMatcher
from semantic_matrix_analyzer.conversation.memory.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


class ConversationContext:
    """Manages context for a conversation."""
    
    def __init__(
        self,
        conversation_id: Optional[str],
        conversation_store: ConversationStore,
        knowledge_graph: KnowledgeGraph
    ):
        """Initialize the conversation context.
        
        Args:
            conversation_id: The ID of the conversation, or None to create a new conversation.
            conversation_store: The conversation store to use.
            knowledge_graph: The knowledge graph to use.
        """
        self.conversation_store = conversation_store
        self.knowledge_graph = knowledge_graph
        self.intent_extractor = ConversationMemoryIntentExtractor(conversation_store)
        self.intent_matcher = IntentMatcher(conversation_store)
        self.graph_builder = KnowledgeGraphBuilder(conversation_store, knowledge_graph)
        
        # Get or create the conversation
        if conversation_id:
            self.conversation = conversation_store.get_conversation(conversation_id)
            if not self.conversation:
                logger.warning(f"Conversation {conversation_id} not found, creating a new conversation.")
                self.conversation = conversation_store.create_conversation("New Conversation")
        else:
            self.conversation = conversation_store.create_conversation("New Conversation")
        
        # Initialize context from the conversation
        self.active_intents: Set[str] = set()
        self.active_preferences: Dict[str, Any] = {}
        self._initialize_from_conversation()
    
    def _initialize_from_conversation(self) -> None:
        """Initialize context from the conversation."""
        # Extract intents from the conversation
        self.intent_extractor.extract_intents_from_conversation(self.conversation)
        
        # Add the conversation to the knowledge graph
        self.graph_builder.add_conversation_to_graph(self.conversation)
        
        # Set active intents
        self.active_intents = set(self.conversation.extracted_intents.keys())
        
        # Extract preferences from the conversation
        self._extract_preferences()
    
    def _extract_preferences(self) -> None:
        """Extract preferences from the conversation."""
        # This is a placeholder for preference extraction
        # In a real implementation, this would analyze the conversation to extract user preferences
        pass
    
    def add_entry(self, speaker: str, message: str) -> None:
        """Add an entry to the conversation.
        
        Args:
            speaker: The speaker ("user" or "ai").
            message: The message.
        """
        # Add the entry to the conversation
        self.conversation.add_entry(speaker, message)
        self.conversation_store.save_conversation(self.conversation)
        
        # If the speaker is the user, update the context
        if speaker == "user":
            # Extract intents
            self.intent_extractor.extract_intents_from_conversation(self.conversation)
            
            # Update the knowledge graph
            self.graph_builder.add_conversation_to_graph(self.conversation)
            
            # Update active intents
            self.active_intents = set(self.conversation.extracted_intents.keys())
            
            # Extract preferences
            self._extract_preferences()
    
    def get_active_intents(self) -> Set[str]:
        """Get the active intents for this conversation.
        
        Returns:
            A set of active intent names.
        """
        return self.active_intents
    
    def get_active_preferences(self) -> Dict[str, Any]:
        """Get the active preferences for this conversation.
        
        Returns:
            A dictionary of active preferences.
        """
        return self.active_preferences
    
    def get_related_intents(self, text: str) -> List[Tuple[str, float]]:
        """Get intents related to the given text.
        
        Args:
            text: The text to match against.
            
        Returns:
            A list of (intent_name, score) tuples, sorted by score in descending order.
        """
        return self.intent_matcher.find_matching_intents(text)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history.
        
        Returns:
            A list of conversation entries as dictionaries.
        """
        return [entry.to_dict() for entry in self.conversation.entries]
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation.
        
        Returns:
            A dictionary with conversation summary information.
        """
        return {
            "id": self.conversation.id,
            "title": self.conversation.title,
            "created_at": self.conversation.created_at.isoformat(),
            "updated_at": self.conversation.updated_at.isoformat(),
            "entry_count": len(self.conversation.entries),
            "active_intents": list(self.active_intents),
            "active_preferences": self.active_preferences
        }


class ContextManager:
    """Manages conversation contexts."""
    
    def __init__(
        self,
        conversation_store: ConversationStore,
        knowledge_graph: KnowledgeGraph
    ):
        """Initialize the context manager.
        
        Args:
            conversation_store: The conversation store to use.
            knowledge_graph: The knowledge graph to use.
        """
        self.conversation_store = conversation_store
        self.knowledge_graph = knowledge_graph
        self.active_contexts: Dict[str, ConversationContext] = {}
    
    def get_context(self, conversation_id: Optional[str] = None) -> ConversationContext:
        """Get a conversation context.
        
        Args:
            conversation_id: The ID of the conversation, or None to create a new conversation.
            
        Returns:
            The conversation context.
        """
        if conversation_id and conversation_id in self.active_contexts:
            return self.active_contexts[conversation_id]
        
        # Create a new context
        context = ConversationContext(
            conversation_id=conversation_id,
            conversation_store=self.conversation_store,
            knowledge_graph=self.knowledge_graph
        )
        
        # Store the context
        self.active_contexts[context.conversation.id] = context
        
        return context
    
    def close_context(self, conversation_id: str) -> None:
        """Close a conversation context.
        
        Args:
            conversation_id: The ID of the conversation.
        """
        if conversation_id in self.active_contexts:
            del self.active_contexts[conversation_id]
