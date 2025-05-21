"""
Core models for representing and organizing intents extracted from code structure.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class IntentSource(Enum):
    """Source of an extracted intent."""
    
    NAME = "name"  # From a name (class, method, variable)
    TYPE = "type"  # From a type hint
    STRUCTURE = "structure"  # From structural analysis
    DOCUMENTATION = "documentation"  # From documentation
    COMMENT = "comment"  # From a comment
    CONVERSATION = "conversation"  # From a conversation


class IntentType(Enum):
    """Type of an intent."""
    
    ACTION = "action"  # An action or operation
    ENTITY = "entity"  # A business entity or domain object
    STATE = "state"  # A state or condition
    PATTERN = "pattern"  # A design or architectural pattern
    CONSTRAINT = "constraint"  # A constraint or limitation
    RESPONSIBILITY = "responsibility"  # A responsibility or role
    OTHER = "other"  # Other type of intent


@dataclass
class CodeLocation:
    """Location of code in a file."""
    
    file_path: Path
    start_line: int
    end_line: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": str(self.file_path),
            "start_line": self.start_line,
            "end_line": self.end_line
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeLocation':
        """Create from dictionary after deserialization."""
        return cls(
            file_path=Path(data["file_path"]),
            start_line=data["start_line"],
            end_line=data["end_line"]
        )


@dataclass
class Intent:
    """Base class for representing extracted intent."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    source: IntentSource = IntentSource.NAME
    type: IntentType = IntentType.OTHER
    confidence: float = 0.0
    location: Optional[CodeLocation] = None
    related_intents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source": self.source.value,
            "type": self.type.value,
            "confidence": self.confidence,
            "related_intents": self.related_intents,
            "metadata": self.metadata
        }
        
        if self.location:
            result["location"] = self.location.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intent':
        """Create from dictionary after deserialization."""
        intent = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            source=IntentSource(data["source"]),
            type=IntentType(data["type"]),
            confidence=data["confidence"],
            related_intents=data.get("related_intents", []),
            metadata=data.get("metadata", {})
        )
        
        if "location" in data:
            intent.location = CodeLocation.from_dict(data["location"])
        
        return intent


@dataclass
class NameIntent(Intent):
    """Intent extracted from a name (class, method, variable)."""
    
    original_name: str = ""
    tokens: List[str] = field(default_factory=list)
    name_type: str = ""  # "class", "method", "variable", etc.
    
    def __post_init__(self):
        """Initialize after creation."""
        self.source = IntentSource.NAME
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            "original_name": self.original_name,
            "tokens": self.tokens,
            "name_type": self.name_type
        })
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NameIntent':
        """Create from dictionary after deserialization."""
        intent = super().from_dict(data)
        name_intent = cls(
            id=intent.id,
            name=intent.name,
            description=intent.description,
            type=intent.type,
            confidence=intent.confidence,
            location=intent.location,
            related_intents=intent.related_intents,
            metadata=intent.metadata,
            original_name=data["original_name"],
            tokens=data["tokens"],
            name_type=data["name_type"]
        )
        return name_intent


@dataclass
class TypeIntent(Intent):
    """Intent extracted from a type hint."""
    
    type_string: str = ""
    is_optional: bool = False
    is_collection: bool = False
    is_custom_type: bool = False
    
    def __post_init__(self):
        """Initialize after creation."""
        self.source = IntentSource.TYPE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            "type_string": self.type_string,
            "is_optional": self.is_optional,
            "is_collection": self.is_collection,
            "is_custom_type": self.is_custom_type
        })
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TypeIntent':
        """Create from dictionary after deserialization."""
        intent = super().from_dict(data)
        type_intent = cls(
            id=intent.id,
            name=intent.name,
            description=intent.description,
            type=intent.type,
            confidence=intent.confidence,
            location=intent.location,
            related_intents=intent.related_intents,
            metadata=intent.metadata,
            type_string=data["type_string"],
            is_optional=data["is_optional"],
            is_collection=data["is_collection"],
            is_custom_type=data["is_custom_type"]
        )
        return type_intent


@dataclass
class StructuralIntent(Intent):
    """Intent extracted from structural analysis."""
    
    pattern_name: str = ""
    components: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize after creation."""
        self.source = IntentSource.STRUCTURE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = super().to_dict()
        result.update({
            "pattern_name": self.pattern_name,
            "components": self.components,
            "relationships": self.relationships
        })
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuralIntent':
        """Create from dictionary after deserialization."""
        intent = super().from_dict(data)
        structural_intent = cls(
            id=intent.id,
            name=intent.name,
            description=intent.description,
            type=intent.type,
            confidence=intent.confidence,
            location=intent.location,
            related_intents=intent.related_intents,
            metadata=intent.metadata,
            pattern_name=data["pattern_name"],
            components=data["components"],
            relationships=data["relationships"]
        )
        return structural_intent


class IntentHierarchy:
    """Hierarchical organization of intents."""
    
    def __init__(self):
        """Initialize the intent hierarchy."""
        self.intents: Dict[str, Intent] = {}
        self.children: Dict[str, List[str]] = {}
        self.parents: Dict[str, str] = {}
    
    def add_intent(self, intent: Intent, parent_id: Optional[str] = None) -> None:
        """Add an intent to the hierarchy.
        
        Args:
            intent: The intent to add.
            parent_id: The ID of the parent intent (optional).
        """
        self.intents[intent.id] = intent
        
        if parent_id:
            if parent_id not in self.children:
                self.children[parent_id] = []
            self.children[parent_id].append(intent.id)
            self.parents[intent.id] = parent_id
    
    def get_intent(self, intent_id: str) -> Optional[Intent]:
        """Get an intent by ID.
        
        Args:
            intent_id: The intent ID.
            
        Returns:
            The intent, or None if not found.
        """
        return self.intents.get(intent_id)
    
    def get_children(self, intent_id: str) -> List[Intent]:
        """Get the children of an intent.
        
        Args:
            intent_id: The intent ID.
            
        Returns:
            A list of child intents.
        """
        child_ids = self.children.get(intent_id, [])
        return [self.intents[child_id] for child_id in child_ids if child_id in self.intents]
    
    def get_parent(self, intent_id: str) -> Optional[Intent]:
        """Get the parent of an intent.
        
        Args:
            intent_id: The intent ID.
            
        Returns:
            The parent intent, or None if not found.
        """
        parent_id = self.parents.get(intent_id)
        if parent_id:
            return self.intents.get(parent_id)
        return None
    
    def get_root_intents(self) -> List[Intent]:
        """Get all root intents (intents with no parent).
        
        Returns:
            A list of root intents.
        """
        root_ids = [intent_id for intent_id in self.intents if intent_id not in self.parents]
        return [self.intents[intent_id] for intent_id in root_ids]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intents": {intent_id: intent.to_dict() for intent_id, intent in self.intents.items()},
            "children": self.children,
            "parents": self.parents
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentHierarchy':
        """Create from dictionary after deserialization."""
        hierarchy = cls()
        
        # Load intents
        for intent_id, intent_data in data.get("intents", {}).items():
            if "source" in intent_data:
                source = IntentSource(intent_data["source"])
                
                if source == IntentSource.NAME:
                    intent = NameIntent.from_dict(intent_data)
                elif source == IntentSource.TYPE:
                    intent = TypeIntent.from_dict(intent_data)
                elif source == IntentSource.STRUCTURE:
                    intent = StructuralIntent.from_dict(intent_data)
                else:
                    intent = Intent.from_dict(intent_data)
                
                hierarchy.intents[intent_id] = intent
        
        # Load relationships
        hierarchy.children = data.get("children", {})
        hierarchy.parents = data.get("parents", {})
        
        return hierarchy
