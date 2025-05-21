#!/usr/bin/env python3
"""
Example of using the conversation memory system.

This script demonstrates how to use the conversation memory system to store and retrieve
conversations, extract intents, and build a knowledge graph.
"""

import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore
from semantic_matrix_analyzer.conversation.memory.agent import AgentFactory, MemoryAugmentedAgent
from semantic_matrix_analyzer.conversation.memory.intent_extraction import ConversationMemoryIntentExtractor
from semantic_matrix_analyzer.conversation.memory.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_example_conversation(storage_dir):
    """Create an example conversation.
    
    Args:
        storage_dir: Directory to store conversations.
        
    Returns:
        The created conversation.
    """
    # Create the conversation store
    conversation_store = ConversationStore(storage_dir)
    
    # Create a new conversation
    conversation = conversation_store.create_conversation("Example Conversation")
    
    # Add entries to the conversation
    conversation.add_entry("user", "I want to improve the clean code in my project.")
    conversation.add_entry("ai", "I can help you with that. What specific aspects of clean code are you interested in?")
    conversation.add_entry("user", "I'm particularly interested in improving function length and variable naming.")
    conversation.add_entry("ai", "Those are important aspects of clean code. Let's start by analyzing your functions.")
    conversation.add_entry("user", "I also want to make sure my error handling is robust.")
    conversation.add_entry("ai", "Error handling is another important aspect. We can look at that as well.")
    
    # Save the conversation
    conversation_store.save_conversation(conversation)
    
    return conversation


def extract_intents(conversation, conversation_store):
    """Extract intents from a conversation.
    
    Args:
        conversation: The conversation to extract intents from.
        conversation_store: The conversation store.
        
    Returns:
        The extracted intents.
    """
    # Create the intent extractor
    intent_extractor = ConversationMemoryIntentExtractor(conversation_store)
    
    # Extract intents
    intents = intent_extractor.extract_intents_from_conversation(conversation)
    
    return intents


def build_knowledge_graph(conversation_store, knowledge_graph_path):
    """Build a knowledge graph from conversations.
    
    Args:
        conversation_store: The conversation store.
        knowledge_graph_path: Path to store the knowledge graph.
        
    Returns:
        The knowledge graph.
    """
    # Create the knowledge graph
    knowledge_graph = KnowledgeGraph(knowledge_graph_path)
    
    # Create the knowledge graph builder
    graph_builder = KnowledgeGraphBuilder(conversation_store, knowledge_graph)
    
    # Build the knowledge graph
    graph_builder.build_from_conversations()
    
    return knowledge_graph


def chat_with_agent(storage_dir, knowledge_graph_path):
    """Chat with a memory-augmented agent.
    
    Args:
        storage_dir: Directory to store conversations.
        knowledge_graph_path: Path to store the knowledge graph.
    """
    # Create the agent
    agent = AgentFactory.create_agent(storage_dir, knowledge_graph_path)
    
    # Start a new conversation
    conversation_id = agent.start_conversation()
    
    print(f"Conversation ID: {conversation_id}")
    print("Type 'exit' to end the conversation.")
    print()
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                break
            
            # Process the message
            response = agent.process_message(user_input)
            
            # Print the response
            print(f"AI: {response}")
            print()
            
            # Print active intents
            active_intents = agent.get_active_intents()
            if active_intents:
                print(f"Active intents: {', '.join(active_intents)}")
                print()
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Close the conversation
    agent.close_conversation()
    print("Conversation ended.")


def main():
    """Main entry point for the script."""
    setup_logging()
    
    # Create directories
    storage_dir = Path("conversation_memory")
    storage_dir.mkdir(exist_ok=True)
    
    knowledge_graph_path = storage_dir / "knowledge_graph.json"
    
    # Create an example conversation
    conversation = create_example_conversation(storage_dir)
    print(f"Created conversation: {conversation.id}")
    
    # Extract intents
    intents = extract_intents(conversation, ConversationStore(storage_dir))
    print(f"Extracted intents: {', '.join(intents.keys())}")
    
    # Build knowledge graph
    knowledge_graph = build_knowledge_graph(ConversationStore(storage_dir), knowledge_graph_path)
    print(f"Built knowledge graph with {len(knowledge_graph.entities)} entities and {len(knowledge_graph.relationships)} relationships.")
    
    # Chat with agent
    print("\nStarting chat with memory-augmented agent...")
    chat_with_agent(storage_dir, knowledge_graph_path)


if __name__ == "__main__":
    main()
