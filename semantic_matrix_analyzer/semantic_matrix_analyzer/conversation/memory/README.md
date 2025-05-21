# Conversation Memory System

The Conversation Memory System is a component of the Semantic Matrix Analyzer that provides functionality for storing, retrieving, and utilizing conversation history to build a persistent understanding of user concerns, preferences, and priorities.

## Overview

The Conversation Memory System consists of several key components:

1. **Conversation Storage**: Persistent storage of conversations and extracted intents
2. **Intent Extraction**: Enhanced system to extract and persist intents from conversations
3. **Knowledge Graph**: Construction of a knowledge graph linking intents, patterns, and preferences
4. **Context Management**: System for maintaining context across conversation sessions
5. **AI Integration**: Integration with AI agents to utilize conversation memory

## Usage

### Command-Line Interface

The Conversation Memory System provides a command-line interface for interacting with the system:

```bash
# Start a chat session
conversation-memory chat

# List all conversations
conversation-memory list

# Show a specific conversation
conversation-memory show --conversation-id <id>

# Extract intents from conversations
conversation-memory extract

# Build the knowledge graph
conversation-memory build

# Query the knowledge graph
conversation-memory query
```

### Programmatic Usage

You can also use the Conversation Memory System programmatically:

```python
from semantic_matrix_analyzer.conversation.memory import ConversationStore
from semantic_matrix_analyzer.conversation.memory.agent import AgentFactory
from semantic_matrix_analyzer.conversation.memory.intent_extraction import ConversationMemoryIntentExtractor
from semantic_matrix_analyzer.conversation.memory.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder

# Create a conversation store
conversation_store = ConversationStore("conversation_memory")

# Create a new conversation
conversation = conversation_store.create_conversation("Example Conversation")

# Add entries to the conversation
conversation.add_entry("user", "I want to improve the clean code in my project.")
conversation.add_entry("ai", "I can help you with that.")

# Save the conversation
conversation_store.save_conversation(conversation)

# Extract intents
intent_extractor = ConversationMemoryIntentExtractor(conversation_store)
intents = intent_extractor.extract_intents_from_conversation(conversation)

# Build a knowledge graph
knowledge_graph = KnowledgeGraph("knowledge_graph.json")
graph_builder = KnowledgeGraphBuilder(conversation_store, knowledge_graph)
graph_builder.build_from_conversations()

# Create a memory-augmented agent
agent = AgentFactory.create_agent("conversation_memory", "knowledge_graph.json")

# Start a conversation
conversation_id = agent.start_conversation()

# Process a message
response = agent.process_message("I want to improve error handling.")

# Get active intents
active_intents = agent.get_active_intents()
```

## Components

### Conversation Storage

The conversation storage system provides functionality for storing and retrieving conversations:

- `ConversationEntry`: A single entry in a conversation
- `Conversation`: A conversation between a user and an AI agent
- `ConversationStore`: Stores conversations and provides access to them

### Intent Extraction

The intent extraction system extracts intents from conversations and stores them with the conversations:

- `ConversationMemoryIntentExtractor`: Extracts intents from conversations
- `IntentMatcher`: Matches intents against text

### Knowledge Graph

The knowledge graph system builds a graph of knowledge extracted from conversations:

- `KnowledgeGraphEntity`: An entity in the knowledge graph
- `KnowledgeGraphRelationship`: A relationship between entities in the knowledge graph
- `KnowledgeGraph`: A graph of knowledge extracted from conversations
- `KnowledgeGraphBuilder`: Builds a knowledge graph from conversations

### Context Management

The context management system manages conversation context across sessions:

- `ConversationContext`: Manages context for a conversation
- `ContextManager`: Manages conversation contexts

### AI Integration

The AI integration system integrates AI agents with the conversation memory system:

- `MemoryAugmentedAgent`: An AI agent that uses conversation memory
- `AgentFactory`: Factory for creating memory-augmented agents

## Benefits

The Conversation Memory System provides several benefits:

1. **Reduced Repetition**: Users don't need to repeat their concerns and preferences
2. **Contextual Understanding**: AI agents can understand references to previous conversations
3. **Progressive Refinement**: Analysis can improve over time based on accumulated knowledge
4. **Personalization**: Recommendations can be tailored to user preferences
5. **Organizational Knowledge**: Teams can build a shared understanding of code quality standards
