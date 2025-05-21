# Plan 01: Conversation Memory System

## Objective

Develop a robust system for storing, retrieving, and utilizing conversation history to build a persistent understanding of user concerns, preferences, and priorities, reducing the cognitive load on users by eliminating the need to repeat information.

## Rationale

Currently, each conversation with an AI agent about code quality is isolated, requiring users to repeat their concerns and preferences. This increases cognitive load and reduces efficiency. By implementing conversation memory:

1. **Reduced Repetition**: Users don't need to restate their concerns and preferences
2. **Contextual Understanding**: AI agents can understand references to previous conversations
3. **Progressive Refinement**: Analysis can improve over time based on accumulated knowledge
4. **Personalization**: Recommendations can be tailored to user preferences
5. **Organizational Knowledge**: Teams can build a shared understanding of code quality standards

## Implementation Details

### 1. Conversation Storage

Create a system for storing conversations and extracted intents:

```python
@dataclass
class ConversationEntry:
    """A single entry in a conversation."""
    timestamp: datetime
    speaker: str  # "user" or "ai"
    message: str
    
@dataclass
class Conversation:
    """A conversation between a user and an AI agent."""
    id: str
    title: str
    entries: List[ConversationEntry]
    extracted_intents: Dict[str, Any]  # Intent name -> Intent data
    created_at: datetime
    updated_at: datetime
    
class ConversationStore:
    """Stores conversations and provides access to them."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.conversations = {}
        self._load_conversations()
    
    def _load_conversations(self) -> None:
        """Load conversations from storage."""
        # Load conversations from JSON files in the storage directory
        pass
    
    def save_conversation(self, conversation: Conversation) -> None:
        """Save a conversation to storage."""
        # Save the conversation to a JSON file
        pass
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def get_all_conversations(self) -> List[Conversation]:
        """Get all conversations."""
        return list(self.conversations.values())
    
    def get_conversations_by_intent(self, intent_name: str) -> List[Conversation]:
        """Get conversations that mention a specific intent."""
        return [c for c in self.conversations.values() 
                if intent_name in c.extracted_intents]
```

### 2. Intent Extraction and Persistence

Enhance the intent extraction system to store extracted intents with conversations:

```python
class ConversationIntentExtractor:
    """Extracts intents from conversations and stores them."""
    
    def __init__(self, conversation_store: ConversationStore):
        self.conversation_store = conversation_store
    
    def extract_intents_from_conversation(self, conversation: Conversation) -> Dict[str, Any]:
        """Extract intents from a conversation and store them."""
        # Use the existing intent extraction logic
        intents = extract_intents_from_text("\n".join(
            entry.message for entry in conversation.entries if entry.speaker == "user"
        ))
        
        # Store the extracted intents with the conversation
        conversation.extracted_intents = intents
        self.conversation_store.save_conversation(conversation)
        
        return intents
```

### 3. Knowledge Graph Construction

Build a knowledge graph of intents, patterns, and user preferences:

```python
class KnowledgeGraph:
    """A graph of knowledge extracted from conversations."""
    
    def __init__(self):
        self.intents = {}  # Intent name -> Intent data
        self.patterns = {}  # Pattern name -> Pattern data
        self.preferences = {}  # Preference name -> Preference data
        self.relationships = {}  # (entity1, relationship, entity2) -> strength
    
    def add_intent(self, intent_name: str, intent_data: Dict[str, Any]) -> None:
        """Add an intent to the knowledge graph."""
        self.intents[intent_name] = intent_data
        
        # Add relationships between the intent and its patterns
        for pattern in intent_data.get("patterns", []):
            pattern_name = pattern.get("name")
            if pattern_name:
                self.patterns[pattern_name] = pattern
                self.add_relationship(intent_name, "has_pattern", pattern_name, 1.0)
    
    def add_preference(self, preference_name: str, preference_data: Dict[str, Any]) -> None:
        """Add a user preference to the knowledge graph."""
        self.preferences[preference_name] = preference_data
    
    def add_relationship(self, entity1: str, relationship: str, entity2: str, strength: float) -> None:
        """Add a relationship between two entities."""
        self.relationships[(entity1, relationship, entity2)] = strength
    
    def get_related_entities(self, entity: str, relationship: str) -> List[Tuple[str, float]]:
        """Get entities related to the given entity by the given relationship."""
        return [(e2, s) for (e1, r, e2), s in self.relationships.items() 
                if e1 == entity and r == relationship]
```

### 4. Conversation Context Manager

Create a system for managing conversation context across sessions:

```python
class ConversationContext:
    """Manages context for a conversation."""
    
    def __init__(self, conversation_id: str, conversation_store: ConversationStore):
        self.conversation_id = conversation_id
        self.conversation_store = conversation_store
        self.conversation = conversation_store.get_conversation(conversation_id)
        self.active_intents = set()
        self.active_preferences = {}
        self.knowledge_graph = KnowledgeGraph()
        
        if self.conversation:
            self._initialize_from_conversation()
    
    def _initialize_from_conversation(self) -> None:
        """Initialize context from the conversation."""
        # Extract intents and preferences from the conversation
        for intent_name, intent_data in self.conversation.extracted_intents.items():
            self.active_intents.add(intent_name)
            self.knowledge_graph.add_intent(intent_name, intent_data)
    
    def add_entry(self, speaker: str, message: str) -> None:
        """Add an entry to the conversation."""
        if not self.conversation:
            # Create a new conversation
            self.conversation = Conversation(
                id=self.conversation_id,
                title=message[:50],
                entries=[],
                extracted_intents={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        # Add the entry
        self.conversation.entries.append(ConversationEntry(
            timestamp=datetime.now(),
            speaker=speaker,
            message=message
        ))
        
        # Update the conversation
        self.conversation.updated_at = datetime.now()
        self.conversation_store.save_conversation(self.conversation)
        
        # If the speaker is the user, extract intents
        if speaker == "user":
            extractor = ConversationIntentExtractor(self.conversation_store)
            extractor.extract_intents_from_conversation(self.conversation)
            self._initialize_from_conversation()
    
    def get_active_intents(self) -> Set[str]:
        """Get the active intents for this conversation."""
        return self.active_intents
    
    def get_active_preferences(self) -> Dict[str, Any]:
        """Get the active preferences for this conversation."""
        return self.active_preferences
```

### 5. Integration with AI Agents

Integrate the conversation memory system with AI agents:

```python
class AIAgent:
    """An AI agent that uses conversation memory."""
    
    def __init__(self, conversation_store: ConversationStore):
        self.conversation_store = conversation_store
        self.current_context = None
    
    def start_conversation(self, conversation_id: str = None) -> str:
        """Start a new conversation or continue an existing one."""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        self.current_context = ConversationContext(
            conversation_id=conversation_id,
            conversation_store=self.conversation_store
        )
        
        return conversation_id
    
    def process_message(self, message: str) -> str:
        """Process a message from the user."""
        if not self.current_context:
            self.start_conversation()
        
        # Add the user's message to the conversation
        self.current_context.add_entry("user", message)
        
        # Generate a response based on the conversation context
        response = self._generate_response()
        
        # Add the AI's response to the conversation
        self.current_context.add_entry("ai", response)
        
        return response
    
    def _generate_response(self) -> str:
        """Generate a response based on the conversation context."""
        # Use the active intents and preferences to generate a response
        active_intents = self.current_context.get_active_intents()
        active_preferences = self.current_context.get_active_preferences()
        
        # This would be implemented by the specific AI agent
        return f"Response based on intents: {active_intents} and preferences: {active_preferences}"
```

## Success Criteria

1. Persistent storage of conversations and extracted intents
2. Knowledge graph of intents, patterns, and preferences
3. Context management across conversation sessions
4. Integration with AI agents for utilizing conversation memory
5. Demonstration of reduced repetition in follow-up conversations

## Dependencies

- Existing intent extraction system
- Existing pattern detection system

## Timeline

- Research and design: 1 week
- Conversation storage implementation: 1 week
- Intent extraction and persistence: 1 week
- Knowledge graph construction: 2 weeks
- Conversation context manager: 1 week
- AI agent integration: 1 week
- Testing and documentation: 1 week

Total: 8 weeks
