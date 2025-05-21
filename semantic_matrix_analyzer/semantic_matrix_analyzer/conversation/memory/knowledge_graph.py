"""
Knowledge graph module for conversation memory.

This module provides functionality for building and querying a knowledge graph
of intents, patterns, and user preferences extracted from conversations.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraphEntity:
    """An entity in the knowledge graph."""
    
    id: str
    type: str  # "intent", "pattern", "preference", etc.
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeGraphEntity':
        """Create from dictionary after deserialization."""
        return cls(
            id=data["id"],
            type=data["type"],
            name=data["name"],
            attributes=data.get("attributes", {})
        )


@dataclass
class KnowledgeGraphRelationship:
    """A relationship between entities in the knowledge graph."""
    
    source_id: str
    target_id: str
    type: str  # "has_pattern", "mentions", "prefers", etc.
    strength: float = 1.0  # 0.0 to 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "strength": self.strength,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeGraphRelationship':
        """Create from dictionary after deserialization."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            strength=data["strength"],
            attributes=data.get("attributes", {})
        )


class KnowledgeGraph:
    """A graph of knowledge extracted from conversations."""
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize the knowledge graph.
        
        Args:
            storage_path: Path to store the knowledge graph.
        """
        self.entities: Dict[str, KnowledgeGraphEntity] = {}
        self.relationships: List[KnowledgeGraphRelationship] = []
        self.entity_index: Dict[str, List[str]] = {}  # type -> [entity_id]
        self.relationship_index: Dict[str, List[KnowledgeGraphRelationship]] = {}  # type -> [relationship]
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Load the knowledge graph if a storage path is provided
        if self.storage_path and self.storage_path.exists():
            self.load()
    
    def add_entity(self, entity: KnowledgeGraphEntity) -> None:
        """Add an entity to the graph.
        
        Args:
            entity: The entity to add.
        """
        self.entities[entity.id] = entity
        
        # Update the entity index
        if entity.type not in self.entity_index:
            self.entity_index[entity.type] = []
        self.entity_index[entity.type].append(entity.id)
    
    def add_relationship(self, relationship: KnowledgeGraphRelationship) -> None:
        """Add a relationship to the graph.
        
        Args:
            relationship: The relationship to add.
        """
        self.relationships.append(relationship)
        
        # Update the relationship index
        if relationship.type not in self.relationship_index:
            self.relationship_index[relationship.type] = []
        self.relationship_index[relationship.type].append(relationship)
    
    def get_entity(self, entity_id: str) -> Optional[KnowledgeGraphEntity]:
        """Get an entity by ID.
        
        Args:
            entity_id: The ID of the entity.
            
        Returns:
            The entity, or None if not found.
        """
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[KnowledgeGraphEntity]:
        """Get all entities of a specific type.
        
        Args:
            entity_type: The type of entities to get.
            
        Returns:
            A list of entities of the specified type.
        """
        entity_ids = self.entity_index.get(entity_type, [])
        return [self.entities[entity_id] for entity_id in entity_ids]
    
    def get_relationships_by_type(self, relationship_type: str) -> List[KnowledgeGraphRelationship]:
        """Get all relationships of a specific type.
        
        Args:
            relationship_type: The type of relationships to get.
            
        Returns:
            A list of relationships of the specified type.
        """
        return self.relationship_index.get(relationship_type, [])
    
    def get_relationships_for_entity(self, entity_id: str, direction: str = "outgoing") -> List[KnowledgeGraphRelationship]:
        """Get all relationships for an entity.
        
        Args:
            entity_id: The ID of the entity.
            direction: The direction of the relationships ("outgoing" or "incoming").
            
        Returns:
            A list of relationships for the entity.
        """
        if direction == "outgoing":
            return [r for r in self.relationships if r.source_id == entity_id]
        elif direction == "incoming":
            return [r for r in self.relationships if r.target_id == entity_id]
        else:
            return [r for r in self.relationships if r.source_id == entity_id or r.target_id == entity_id]
    
    def get_related_entities(self, entity_id: str, relationship_type: Optional[str] = None, direction: str = "outgoing") -> List[Tuple[KnowledgeGraphEntity, float]]:
        """Get entities related to an entity.
        
        Args:
            entity_id: The ID of the entity.
            relationship_type: The type of relationships to consider (optional).
            direction: The direction of the relationships ("outgoing" or "incoming").
            
        Returns:
            A list of (entity, strength) tuples.
        """
        relationships = self.get_relationships_for_entity(entity_id, direction)
        
        if relationship_type:
            relationships = [r for r in relationships if r.type == relationship_type]
        
        related_entities = []
        for relationship in relationships:
            related_id = relationship.target_id if direction == "outgoing" else relationship.source_id
            related_entity = self.get_entity(related_id)
            if related_entity:
                related_entities.append((related_entity, relationship.strength))
        
        return related_entities
    
    def save(self) -> None:
        """Save the knowledge graph to storage."""
        if not self.storage_path:
            logger.warning("No storage path provided, knowledge graph not saved.")
            return
        
        try:
            # Create the parent directory if it doesn't exist
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the knowledge graph to a file
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "entities": {entity_id: entity.to_dict() for entity_id, entity in self.entities.items()},
                    "relationships": [relationship.to_dict() for relationship in self.relationships]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
    
    def load(self) -> None:
        """Load the knowledge graph from storage."""
        if not self.storage_path or not self.storage_path.exists():
            logger.warning("No storage path provided or file does not exist, knowledge graph not loaded.")
            return
        
        try:
            # Load the knowledge graph from a file
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Clear existing data
                self.entities = {}
                self.relationships = []
                self.entity_index = {}
                self.relationship_index = {}
                
                # Load entities
                for entity_id, entity_data in data.get("entities", {}).items():
                    entity = KnowledgeGraphEntity.from_dict(entity_data)
                    self.add_entity(entity)
                
                # Load relationships
                for relationship_data in data.get("relationships", []):
                    relationship = KnowledgeGraphRelationship.from_dict(relationship_data)
                    self.add_relationship(relationship)
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")


class KnowledgeGraphBuilder:
    """Builds a knowledge graph from conversations."""
    
    def __init__(self, conversation_store: ConversationStore, knowledge_graph: KnowledgeGraph):
        """Initialize the knowledge graph builder.
        
        Args:
            conversation_store: The conversation store to use.
            knowledge_graph: The knowledge graph to build.
        """
        self.conversation_store = conversation_store
        self.knowledge_graph = knowledge_graph
    
    def build_from_conversations(self) -> None:
        """Build the knowledge graph from all conversations."""
        for conversation in self.conversation_store.get_all_conversations():
            self.add_conversation_to_graph(conversation)
    
    def add_conversation_to_graph(self, conversation: Conversation) -> None:
        """Add a conversation to the knowledge graph.
        
        Args:
            conversation: The conversation to add.
        """
        # Add a conversation entity
        conversation_entity = KnowledgeGraphEntity(
            id=f"conversation:{conversation.id}",
            type="conversation",
            name=conversation.title,
            attributes={
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat()
            }
        )
        self.knowledge_graph.add_entity(conversation_entity)
        
        # Add intent entities and relationships
        for intent_name, intent_data in conversation.extracted_intents.items():
            intent_id = f"intent:{intent_name}"
            
            # Add the intent entity if it doesn't exist
            if not self.knowledge_graph.get_entity(intent_id):
                intent_entity = KnowledgeGraphEntity(
                    id=intent_id,
                    type="intent",
                    name=intent_name,
                    attributes={
                        "description": intent_data.get("description", "")
                    }
                )
                self.knowledge_graph.add_entity(intent_entity)
            
            # Add a relationship between the conversation and the intent
            self.knowledge_graph.add_relationship(KnowledgeGraphRelationship(
                source_id=conversation_entity.id,
                target_id=intent_id,
                type="mentions",
                strength=1.0
            ))
            
            # Add pattern entities and relationships
            for pattern in intent_data.get("patterns", []):
                pattern_name = pattern.get("name", "")
                pattern_id = f"pattern:{pattern_name}"
                
                # Add the pattern entity if it doesn't exist
                if not self.knowledge_graph.get_entity(pattern_id):
                    pattern_entity = KnowledgeGraphEntity(
                        id=pattern_id,
                        type="pattern",
                        name=pattern_name,
                        attributes={
                            "description": pattern.get("description", ""),
                            "pattern_type": pattern.get("pattern_type", ""),
                            "pattern": pattern.get("pattern", ""),
                            "is_negative": pattern.get("is_negative", False)
                        }
                    )
                    self.knowledge_graph.add_entity(pattern_entity)
                
                # Add a relationship between the intent and the pattern
                self.knowledge_graph.add_relationship(KnowledgeGraphRelationship(
                    source_id=intent_id,
                    target_id=pattern_id,
                    type="has_pattern",
                    strength=pattern.get("weight", 1.0)
                ))
        
        # Save the knowledge graph
        self.knowledge_graph.save()
