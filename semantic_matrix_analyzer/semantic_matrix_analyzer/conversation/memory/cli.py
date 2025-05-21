"""
Command-line interface for conversation memory.

This module provides a command-line interface for interacting with the conversation memory system.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.conversation.memory import ConversationStore
from semantic_matrix_analyzer.conversation.memory.agent import AgentFactory, MemoryAugmentedAgent
from semantic_matrix_analyzer.conversation.memory.context import ContextManager
from semantic_matrix_analyzer.conversation.memory.intent_extraction import ConversationMemoryIntentExtractor
from semantic_matrix_analyzer.conversation.memory.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder


def setup_logging(verbose: bool = False) -> None:
    """Set up logging.
    
    Args:
        verbose: Whether to enable verbose logging.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Conversation Memory CLI")
    
    # Common arguments
    parser.add_argument("--storage-dir", type=str, default="conversation_memory",
                        help="Directory to store conversations")
    parser.add_argument("--knowledge-graph", type=str, default="conversation_memory/knowledge_graph.json",
                        help="Path to store the knowledge graph")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session")
    chat_parser.add_argument("--conversation-id", type=str, help="ID of an existing conversation")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List conversations")
    
    # show command
    show_parser = subparsers.add_parser("show", help="Show a conversation")
    show_parser.add_argument("--conversation-id", type=str, required=True, help="ID of the conversation")
    
    # extract command
    extract_parser = subparsers.add_parser("extract", help="Extract intents from conversations")
    extract_parser.add_argument("--conversation-id", type=str, help="ID of the conversation (optional)")
    
    # build command
    build_parser = subparsers.add_parser("build", help="Build the knowledge graph")
    
    # query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("--entity-type", type=str, help="Type of entities to query")
    query_parser.add_argument("--entity-id", type=str, help="ID of the entity to query")
    query_parser.add_argument("--relationship-type", type=str, help="Type of relationships to query")
    
    return parser.parse_args()


def chat_command(args: argparse.Namespace) -> None:
    """Run the chat command.
    
    Args:
        args: The parsed arguments.
    """
    # Create the agent
    agent = AgentFactory.create_agent(args.storage_dir, args.knowledge_graph)
    
    # Start the conversation
    conversation_id = agent.start_conversation(args.conversation_id)
    
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


def list_command(args: argparse.Namespace) -> None:
    """Run the list command.
    
    Args:
        args: The parsed arguments.
    """
    # Create the conversation store
    conversation_store = ConversationStore(args.storage_dir)
    
    # Get all conversations
    conversations = conversation_store.get_all_conversations()
    
    if not conversations:
        print("No conversations found.")
        return
    
    # Print conversations
    print(f"Found {len(conversations)} conversations:")
    for conversation in conversations:
        print(f"- {conversation.id}: {conversation.title} ({len(conversation.entries)} entries, {len(conversation.extracted_intents)} intents)")


def show_command(args: argparse.Namespace) -> None:
    """Run the show command.
    
    Args:
        args: The parsed arguments.
    """
    # Create the conversation store
    conversation_store = ConversationStore(args.storage_dir)
    
    # Get the conversation
    conversation = conversation_store.get_conversation(args.conversation_id)
    
    if not conversation:
        print(f"Conversation {args.conversation_id} not found.")
        return
    
    # Print conversation details
    print(f"Conversation: {conversation.title}")
    print(f"ID: {conversation.id}")
    print(f"Created: {conversation.created_at}")
    print(f"Updated: {conversation.updated_at}")
    print(f"Entries: {len(conversation.entries)}")
    print(f"Intents: {', '.join(conversation.extracted_intents.keys())}")
    print()
    
    # Print conversation entries
    print("Conversation history:")
    for entry in conversation.entries:
        print(f"{entry.timestamp} - {entry.speaker}: {entry.message}")


def extract_command(args: argparse.Namespace) -> None:
    """Run the extract command.
    
    Args:
        args: The parsed arguments.
    """
    # Create the conversation store
    conversation_store = ConversationStore(args.storage_dir)
    
    # Create the intent extractor
    intent_extractor = ConversationMemoryIntentExtractor(conversation_store)
    
    if args.conversation_id:
        # Get the conversation
        conversation = conversation_store.get_conversation(args.conversation_id)
        
        if not conversation:
            print(f"Conversation {args.conversation_id} not found.")
            return
        
        # Extract intents
        intents = intent_extractor.extract_intents_from_conversation(conversation)
        
        # Print intents
        print(f"Extracted {len(intents)} intents from conversation {conversation.id}:")
        for intent_name, intent_data in intents.items():
            print(f"- {intent_name}: {intent_data.get('description', '')}")
            print(f"  Patterns: {len(intent_data.get('patterns', []))}")
    else:
        # Extract intents from all conversations
        results = intent_extractor.extract_intents_from_all_conversations()
        
        # Print results
        print(f"Extracted intents from {len(results)} conversations:")
        for conversation_id, intents in results.items():
            print(f"- Conversation {conversation_id}: {len(intents)} intents")


def build_command(args: argparse.Namespace) -> None:
    """Run the build command.
    
    Args:
        args: The parsed arguments.
    """
    # Create the conversation store
    conversation_store = ConversationStore(args.storage_dir)
    
    # Create the knowledge graph
    knowledge_graph = KnowledgeGraph(args.knowledge_graph)
    
    # Create the knowledge graph builder
    graph_builder = KnowledgeGraphBuilder(conversation_store, knowledge_graph)
    
    # Build the knowledge graph
    graph_builder.build_from_conversations()
    
    # Print statistics
    entity_count = len(knowledge_graph.entities)
    relationship_count = len(knowledge_graph.relationships)
    print(f"Built knowledge graph with {entity_count} entities and {relationship_count} relationships.")
    
    # Print entity types
    entity_types = knowledge_graph.entity_index.keys()
    print(f"Entity types: {', '.join(entity_types)}")
    
    # Print relationship types
    relationship_types = knowledge_graph.relationship_index.keys()
    print(f"Relationship types: {', '.join(relationship_types)}")


def query_command(args: argparse.Namespace) -> None:
    """Run the query command.
    
    Args:
        args: The parsed arguments.
    """
    # Create the knowledge graph
    knowledge_graph = KnowledgeGraph(args.knowledge_graph)
    
    if args.entity_type:
        # Query entities by type
        entities = knowledge_graph.get_entities_by_type(args.entity_type)
        
        print(f"Found {len(entities)} entities of type {args.entity_type}:")
        for entity in entities:
            print(f"- {entity.id}: {entity.name}")
    
    elif args.entity_id:
        # Query entity by ID
        entity = knowledge_graph.get_entity(args.entity_id)
        
        if not entity:
            print(f"Entity {args.entity_id} not found.")
            return
        
        print(f"Entity: {entity.name}")
        print(f"ID: {entity.id}")
        print(f"Type: {entity.type}")
        print(f"Attributes: {json.dumps(entity.attributes, indent=2)}")
        print()
        
        # Get relationships
        relationships = knowledge_graph.get_relationships_for_entity(entity.id)
        
        if args.relationship_type:
            relationships = [r for r in relationships if r.type == args.relationship_type]
        
        print(f"Found {len(relationships)} relationships:")
        for relationship in relationships:
            source = knowledge_graph.get_entity(relationship.source_id)
            target = knowledge_graph.get_entity(relationship.target_id)
            
            source_name = source.name if source else relationship.source_id
            target_name = target.name if target else relationship.target_id
            
            print(f"- {source_name} --[{relationship.type}]--> {target_name} (strength: {relationship.strength})")
    
    elif args.relationship_type:
        # Query relationships by type
        relationships = knowledge_graph.get_relationships_by_type(args.relationship_type)
        
        print(f"Found {len(relationships)} relationships of type {args.relationship_type}:")
        for relationship in relationships:
            source = knowledge_graph.get_entity(relationship.source_id)
            target = knowledge_graph.get_entity(relationship.target_id)
            
            source_name = source.name if source else relationship.source_id
            target_name = target.name if target else relationship.target_id
            
            print(f"- {source_name} --[{relationship.type}]--> {target_name} (strength: {relationship.strength})")
    
    else:
        # Print statistics
        entity_count = len(knowledge_graph.entities)
        relationship_count = len(knowledge_graph.relationships)
        print(f"Knowledge graph has {entity_count} entities and {relationship_count} relationships.")
        
        # Print entity types
        entity_types = knowledge_graph.entity_index.keys()
        print(f"Entity types: {', '.join(entity_types)}")
        
        # Print relationship types
        relationship_types = knowledge_graph.relationship_index.keys()
        print(f"Relationship types: {', '.join(relationship_types)}")


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Create the storage directory if it doesn't exist
    os.makedirs(args.storage_dir, exist_ok=True)
    
    # Run the appropriate command
    if args.command == "chat":
        chat_command(args)
    elif args.command == "list":
        list_command(args)
    elif args.command == "show":
        show_command(args)
    elif args.command == "extract":
        extract_command(args)
    elif args.command == "build":
        build_command(args)
    elif args.command == "query":
        query_command(args)
    else:
        print("No command specified. Use --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
