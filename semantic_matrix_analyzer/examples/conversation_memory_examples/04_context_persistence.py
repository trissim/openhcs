#!/usr/bin/env python3
"""
Example 4: Context Persistence

This example demonstrates how conversation context persists across multiple sessions,
allowing the system to remember user preferences and concerns.

Key concepts demonstrated:
1. Context persistence across sessions
2. Memory-augmented agents
3. Automatic recall of user preferences

Cognitive load reduction:
- Users don't need to repeat their preferences in each session
- The system remembers previous conversations and their context
- The system can recall relevant information from past sessions
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore
from semantic_matrix_analyzer.conversation.memory.agent import AgentFactory, MemoryAugmentedAgent
from semantic_matrix_analyzer.conversation.memory.context import ContextManager, ConversationContext
from semantic_matrix_analyzer.conversation.memory.knowledge_graph import KnowledgeGraph


def simulate_session(agent, session_name, messages):
    """Simulate a conversation session.
    
    Args:
        agent: The memory-augmented agent.
        session_name: The name of the session.
        messages: A list of user messages.
        
    Returns:
        The conversation ID.
    """
    print(f"\n=== {session_name} ===")
    
    # Start a conversation
    conversation_id = agent.start_conversation()
    print(f"Started conversation: {conversation_id}")
    
    # Process messages
    for message in messages:
        print(f"\nUser: {message}")
        response = agent.process_message(message)
        print(f"AI: {response}")
        
        # Print active intents
        active_intents = agent.get_active_intents()
        if active_intents:
            print(f"Active intents: {', '.join(active_intents)}")
    
    # Close the conversation
    agent.close_conversation()
    print(f"Ended conversation: {conversation_id}")
    
    return conversation_id


def main():
    """Main entry point for the example."""
    # Create a directory for storing conversations
    storage_dir = Path("conversation_memory_examples/context_persistence")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a knowledge graph path
    knowledge_graph_path = storage_dir / "knowledge_graph.json"
    
    # Create a memory-augmented agent
    agent = AgentFactory.create_agent(storage_dir, knowledge_graph_path)
    
    # Simulate the first session
    print("\nIn this first session, the user expresses interest in clean code and error handling.")
    session_1_messages = [
        "I want to improve the clean code in my project.",
        "I'm particularly interested in function length and variable naming.",
        "I also want to make sure my error handling is robust."
    ]
    session_1_id = simulate_session(agent, "Session 1", session_1_messages)
    
    # Simulate a delay between sessions
    print("\nSimulating a delay between sessions (3 seconds)...")
    time.sleep(3)
    
    # Simulate the second session
    print("\nIn this second session, the user doesn't explicitly mention clean code or error handling,")
    print("but the system remembers these concerns from the previous session.")
    session_2_messages = [
        "I'm working on a new module for my project.",
        "Can you help me analyze it?"
    ]
    session_2_id = simulate_session(agent, "Session 2", session_2_messages)
    
    # Simulate a delay between sessions
    print("\nSimulating a delay between sessions (3 seconds)...")
    time.sleep(3)
    
    # Simulate the third session
    print("\nIn this third session, the user introduces a new concern (performance),")
    print("which is added to the existing concerns.")
    session_3_messages = [
        "I'm concerned about the performance of my application.",
        "Can you help me optimize it?"
    ]
    session_3_id = simulate_session(agent, "Session 3", session_3_messages)
    
    # Print a summary of the conversations
    print("\n=== Summary ===")
    conversation_store = ConversationStore(storage_dir)
    
    session_1 = conversation_store.get_conversation(session_1_id)
    session_2 = conversation_store.get_conversation(session_2_id)
    session_3 = conversation_store.get_conversation(session_3_id)
    
    print(f"\nSession 1 intents: {', '.join(session_1.extracted_intents.keys())}")
    print(f"Session 2 intents: {', '.join(session_2.extracted_intents.keys())}")
    print(f"Session 3 intents: {', '.join(session_3.extracted_intents.keys())}")
    
    # Load the knowledge graph
    knowledge_graph = KnowledgeGraph(knowledge_graph_path)
    
    # Print statistics
    entity_count = len(knowledge_graph.entities)
    relationship_count = len(knowledge_graph.relationships)
    print(f"\nKnowledge graph has {entity_count} entities and {relationship_count} relationships.")
    
    # Print intent entities
    intent_entities = knowledge_graph.get_entities_by_type("intent")
    print(f"\nFound {len(intent_entities)} intents in the knowledge graph:")
    for entity in intent_entities:
        print(f"- {entity.name}")
        
        # Get conversations that mention this intent
        related_entities = knowledge_graph.get_related_entities(entity.id, "mentions", direction="incoming")
        if related_entities:
            print(f"  Mentioned in conversations:")
            for related_entity, strength in related_entities:
                print(f"  - {related_entity.name} (strength: {strength})")
    
    print("\nThis example demonstrates how the system maintains context across multiple sessions.")
    print("The user's concerns (clean code, error handling, performance) are remembered,")
    print("even when they are not explicitly mentioned in each session.")


if __name__ == "__main__":
    main()
