#!/usr/bin/env python3
"""
Example 3: Knowledge Graph

This example demonstrates how a knowledge graph is built from conversations and how it can be
used to understand relationships between intents, patterns, and conversations.

Key concepts demonstrated:
1. Building a knowledge graph from conversations
2. Querying the knowledge graph
3. Understanding relationships between entities

Cognitive load reduction:
- The system builds a structured representation of knowledge
- Relationships between concepts are automatically captured
- The system can answer questions about related concepts
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore
from semantic_matrix_analyzer.conversation.memory.intent_extraction import ConversationMemoryIntentExtractor
from semantic_matrix_analyzer.conversation.memory.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder


def create_sample_conversations(conversation_store):
    """Create sample conversations for the knowledge graph.
    
    Args:
        conversation_store: The conversation store to use.
        
    Returns:
        A list of conversation IDs.
    """
    # Create conversations about different topics
    conversation_ids = []
    
    # Conversation about clean code
    clean_code_conversation = conversation_store.create_conversation("Clean Code Discussion")
    clean_code_conversation.add_entry("user", "I want to improve the clean code in my project.")
    clean_code_conversation.add_entry("ai", "I can help with that. What specific aspects are you interested in?")
    clean_code_conversation.add_entry("user", "I'm particularly interested in function length and variable naming.")
    clean_code_conversation.add_entry("ai", "Those are important aspects of clean code. Let's start by analyzing your functions.")
    conversation_store.save_conversation(clean_code_conversation)
    conversation_ids.append(clean_code_conversation.id)
    
    # Conversation about error handling
    error_handling_conversation = conversation_store.create_conversation("Error Handling Discussion")
    error_handling_conversation.add_entry("user", "I want to make sure my error handling is robust.")
    error_handling_conversation.add_entry("ai", "That's a good focus. What kind of errors are you concerned about?")
    error_handling_conversation.add_entry("user", "I'm worried about network errors and invalid user input.")
    error_handling_conversation.add_entry("ai", "Those are common sources of errors. Let's look at how to handle them properly.")
    conversation_store.save_conversation(error_handling_conversation)
    conversation_ids.append(error_handling_conversation.id)
    
    # Conversation about performance
    performance_conversation = conversation_store.create_conversation("Performance Discussion")
    performance_conversation.add_entry("user", "My application is running slowly. I need to improve performance.")
    performance_conversation.add_entry("ai", "I can help you identify performance bottlenecks. What part of the application is slow?")
    performance_conversation.add_entry("user", "The data processing functions take too long to execute.")
    performance_conversation.add_entry("ai", "Let's analyze those functions to find optimization opportunities.")
    conversation_store.save_conversation(performance_conversation)
    conversation_ids.append(performance_conversation.id)
    
    return conversation_ids


def extract_intents(conversation_store):
    """Extract intents from all conversations.
    
    Args:
        conversation_store: The conversation store to use.
    """
    # Create an intent extractor
    intent_extractor = ConversationMemoryIntentExtractor(conversation_store)
    
    # Extract intents from all conversations
    results = intent_extractor.extract_intents_from_all_conversations()
    
    print(f"Extracted intents from {len(results)} conversations:")
    for conversation_id, intents in results.items():
        conversation = conversation_store.get_conversation(conversation_id)
        print(f"- {conversation.title}: {', '.join(intents.keys())}")


def build_knowledge_graph(conversation_store, knowledge_graph_path):
    """Build a knowledge graph from conversations.
    
    Args:
        conversation_store: The conversation store to use.
        knowledge_graph_path: Path to store the knowledge graph.
        
    Returns:
        The knowledge graph.
    """
    # Create a knowledge graph
    knowledge_graph = KnowledgeGraph(knowledge_graph_path)
    
    # Create a knowledge graph builder
    graph_builder = KnowledgeGraphBuilder(conversation_store, knowledge_graph)
    
    # Build the knowledge graph
    graph_builder.build_from_conversations()
    
    return knowledge_graph


def explore_knowledge_graph(knowledge_graph):
    """Explore the knowledge graph.
    
    Args:
        knowledge_graph: The knowledge graph to explore.
    """
    # Print statistics
    entity_count = len(knowledge_graph.entities)
    relationship_count = len(knowledge_graph.relationships)
    print(f"\nKnowledge graph has {entity_count} entities and {relationship_count} relationships.")
    
    # Print entity types
    entity_types = knowledge_graph.entity_index.keys()
    print(f"Entity types: {', '.join(entity_types)}")
    
    # Print relationship types
    relationship_types = knowledge_graph.relationship_index.keys()
    print(f"Relationship types: {', '.join(relationship_types)}")
    
    # Explore conversations
    conversation_entities = knowledge_graph.get_entities_by_type("conversation")
    print(f"\nFound {len(conversation_entities)} conversations:")
    for entity in conversation_entities:
        print(f"- {entity.name} ({entity.id})")
        
        # Get intents mentioned in this conversation
        related_entities = knowledge_graph.get_related_entities(entity.id, "mentions")
        if related_entities:
            print(f"  Mentions intents:")
            for related_entity, strength in related_entities:
                print(f"  - {related_entity.name} (strength: {strength})")
    
    # Explore intents
    intent_entities = knowledge_graph.get_entities_by_type("intent")
    print(f"\nFound {len(intent_entities)} intents:")
    for entity in intent_entities:
        print(f"- {entity.name} ({entity.id})")
        
        # Get patterns associated with this intent
        related_entities = knowledge_graph.get_related_entities(entity.id, "has_pattern")
        if related_entities:
            print(f"  Has patterns:")
            for related_entity, strength in related_entities:
                print(f"  - {related_entity.name} (strength: {strength})")
        
        # Get conversations that mention this intent
        related_entities = knowledge_graph.get_related_entities(entity.id, "mentions", direction="incoming")
        if related_entities:
            print(f"  Mentioned in conversations:")
            for related_entity, strength in related_entities:
                print(f"  - {related_entity.name} (strength: {strength})")


def main():
    """Main entry point for the example."""
    # Create a directory for storing conversations
    storage_dir = Path("conversation_memory_examples/knowledge_graph")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a conversation store
    conversation_store = ConversationStore(storage_dir)
    
    # Create sample conversations
    print("Creating sample conversations...")
    conversation_ids = create_sample_conversations(conversation_store)
    
    # Extract intents
    print("\nExtracting intents from conversations...")
    extract_intents(conversation_store)
    
    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    knowledge_graph_path = storage_dir / "knowledge_graph.json"
    knowledge_graph = build_knowledge_graph(conversation_store, knowledge_graph_path)
    
    # Explore knowledge graph
    print("\nExploring knowledge graph...")
    explore_knowledge_graph(knowledge_graph)
    
    print("\nKnowledge graph saved to:", knowledge_graph_path)


if __name__ == "__main__":
    main()
