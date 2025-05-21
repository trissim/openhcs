#!/usr/bin/env python3
"""
Example 1: Basic Conversation Memory

This example demonstrates the most basic functionality of the Conversation Memory System:
storing and retrieving conversations.

Key concepts demonstrated:
1. Creating and storing conversations
2. Adding entries to conversations
3. Retrieving conversations
4. Persistence across program runs

Cognitive load reduction:
- Users don't need to remember or repeat information from previous conversations
- The system maintains a history that can be referenced later
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore


def main():
    """Main entry point for the example."""
    # Create a directory for storing conversations
    storage_dir = Path("conversation_memory_examples/basic_memory")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a conversation store
    conversation_store = ConversationStore(storage_dir)
    
    # Check if we have any existing conversations
    existing_conversations = conversation_store.get_all_conversations()
    
    if existing_conversations:
        print(f"Found {len(existing_conversations)} existing conversations:")
        for conversation in existing_conversations:
            print(f"- {conversation.id}: {conversation.title} ({len(conversation.entries)} entries)")
        
        # Ask the user if they want to continue an existing conversation
        print("\nDo you want to continue an existing conversation? (y/n)")
        choice = input("> ")
        
        if choice.lower() == "y":
            # Let the user choose a conversation
            print("\nEnter the ID of the conversation you want to continue:")
            for conversation in existing_conversations:
                print(f"- {conversation.id}: {conversation.title}")
            
            conversation_id = input("> ")
            conversation = conversation_store.get_conversation(conversation_id)
            
            if conversation:
                print(f"\nContinuing conversation: {conversation.title}")
                
                # Show the conversation history
                print("\nConversation history:")
                for entry in conversation.entries:
                    print(f"{entry.timestamp} - {entry.speaker}: {entry.message}")
            else:
                print(f"\nConversation {conversation_id} not found. Creating a new conversation.")
                conversation = conversation_store.create_conversation("New Conversation")
        else:
            # Create a new conversation
            print("\nCreating a new conversation.")
            conversation = conversation_store.create_conversation("New Conversation")
    else:
        # No existing conversations, create a new one
        print("No existing conversations found. Creating a new conversation.")
        conversation = conversation_store.create_conversation("New Conversation")
    
    # Chat loop
    print("\nType 'exit' to end the conversation.")
    print()
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        # Add the user's message to the conversation
        conversation.add_entry("user", user_input)
        
        # Generate a simple response
        response = f"I received your message: {user_input}"
        
        # Add the AI's response to the conversation
        conversation.add_entry("ai", response)
        
        # Print the response
        print(f"AI: {response}")
        print()
    
    # Save the conversation
    conversation_store.save_conversation(conversation)
    
    print("\nConversation saved. You can continue it later by running this example again.")
    
    # Show the conversation ID for reference
    print(f"Conversation ID: {conversation.id}")


if __name__ == "__main__":
    main()
