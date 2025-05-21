"""
Conversation memory module for Semantic Matrix Analyzer.

This module provides functionality for storing, retrieving, and utilizing conversation
history to build a persistent understanding of user concerns, preferences, and priorities.
"""

from datetime import datetime
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ConversationEntry:
    """A single entry in a conversation."""
    
    timestamp: datetime
    speaker: str  # "user" or "ai"
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "speaker": self.speaker,
            "message": self.message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Create from dictionary after deserialization."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            speaker=data["speaker"],
            message=data["message"]
        )


@dataclass
class Conversation:
    """A conversation between a user and an AI agent."""
    
    id: str
    title: str
    entries: List[ConversationEntry] = field(default_factory=list)
    extracted_intents: Dict[str, Any] = field(default_factory=dict)  # Intent name -> Intent data
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_entry(self, speaker: str, message: str) -> None:
        """Add an entry to the conversation."""
        self.entries.append(ConversationEntry(
            timestamp=datetime.now(),
            speaker=speaker,
            message=message
        ))
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "entries": [entry.to_dict() for entry in self.entries],
            "extracted_intents": self.extracted_intents,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create from dictionary after deserialization."""
        conversation = cls(
            id=data["id"],
            title=data["title"],
            entries=[ConversationEntry.from_dict(entry) for entry in data["entries"]],
            extracted_intents=data.get("extracted_intents", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        return conversation


class ConversationStore:
    """Stores conversations and provides access to them."""
    
    def __init__(self, storage_dir: Union[str, Path]):
        """Initialize the conversation store.
        
        Args:
            storage_dir: Directory to store conversations.
        """
        self.storage_dir = Path(storage_dir)
        self.conversations: Dict[str, Conversation] = {}
        
        # Create the storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing conversations
        self._load_conversations()
    
    def _load_conversations(self) -> None:
        """Load conversations from storage."""
        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        conversation = Conversation.from_dict(data)
                        self.conversations[conversation.id] = conversation
                except Exception as e:
                    logger.error(f"Error loading conversation from {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
    
    def save_conversation(self, conversation: Conversation) -> None:
        """Save a conversation to storage.
        
        Args:
            conversation: The conversation to save.
        """
        try:
            # Update the conversation in memory
            self.conversations[conversation.id] = conversation
            
            # Save the conversation to a file
            file_path = self.storage_dir / f"{conversation.id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(conversation.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.id}: {e}")
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            The conversation, or None if not found.
        """
        return self.conversations.get(conversation_id)
    
    def get_all_conversations(self) -> List[Conversation]:
        """Get all conversations.
        
        Returns:
            A list of all conversations.
        """
        return list(self.conversations.values())
    
    def get_conversations_by_intent(self, intent_name: str) -> List[Conversation]:
        """Get conversations that mention a specific intent.
        
        Args:
            intent_name: The name of the intent.
            
        Returns:
            A list of conversations that mention the intent.
        """
        return [c for c in self.conversations.values() 
                if intent_name in c.extracted_intents]
    
    def create_conversation(self, title: str) -> Conversation:
        """Create a new conversation.
        
        Args:
            title: The title of the conversation.
            
        Returns:
            The created conversation.
        """
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            title=title
        )
        self.save_conversation(conversation)
        return conversation
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            True if the conversation was deleted, False otherwise.
        """
        if conversation_id not in self.conversations:
            return False
        
        try:
            # Remove the conversation from memory
            del self.conversations[conversation_id]
            
            # Remove the conversation file
            file_path = self.storage_dir / f"{conversation_id}.json"
            if file_path.exists():
                file_path.unlink()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False
