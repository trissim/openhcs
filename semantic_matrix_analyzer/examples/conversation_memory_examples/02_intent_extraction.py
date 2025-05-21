#!/usr/bin/env python3
"""
Example 2: Intent Extraction

This example demonstrates how intents are extracted from conversations.

Key concepts demonstrated:
1. Creating and storing conversations
2. Extracting intents from conversations
3. Using extracted intents to guide responses

Cognitive load reduction:
- The system automatically identifies user concerns without explicit labeling
- Users can express concerns in natural language
- The system remembers these concerns across the conversation
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore
from semantic_matrix_analyzer.conversation.memory.intent_extraction import ConversationMemoryIntentExtractor


def main():
    """Main entry point for the example."""
    # Create a directory for storing conversations
    storage_dir = Path("conversation_memory_examples/intent_extraction")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a conversation store
    conversation_store = ConversationStore(storage_dir)
    
    # Create a new conversation
    conversation = conversation_store.create_conversation("Intent Extraction Example")
    
    # Create an intent extractor
    intent_extractor = ConversationMemoryIntentExtractor(conversation_store)
    
    # Chat loop
    print("This example demonstrates intent extraction from conversations.")
    print("Try mentioning topics like 'clean code', 'error handling', or 'performance'.")
    print("The system will automatically identify these as intents.")
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
        
        # Extract intents from the conversation
        intents = intent_extractor.extract_intents_from_conversation(conversation)
        
        # Generate a response based on the extracted intents
        if intents:
            intent_names = list(intents.keys())
            response = f"I understand you're interested in: {', '.join(intent_names)}. How can I help with that?"
        else:
            response = "I'm not sure what you're interested in. Could you tell me more about your concerns?"
        
        # Add the AI's response to the conversation
        conversation.add_entry("ai", response)
        
        # Print the response
        print(f"AI: {response}")
        
        # Print the extracted intents
        if intents:
            print("\nExtracted intents:")
            for intent_name, intent_data in intents.items():
                print(f"- {intent_name}: {intent_data.get('description', '')}")
                
                # Print patterns
                patterns = intent_data.get("patterns", [])
                if patterns:
                    print(f"  Patterns:")
                    for pattern in patterns:
                        pattern_type = pattern.get("pattern_type", "")
                        pattern_value = pattern.get("pattern", "")
                        print(f"  - {pattern_type}: {pattern_value}")
        
        print()
    
    # Save the conversation
    conversation_store.save_conversation(conversation)
    
    print("\nConversation saved with extracted intents.")
    
    # Show a summary of the extracted intents
    intents = conversation.extracted_intents
    if intents:
        print("\nSummary of extracted intents:")
        for intent_name, intent_data in intents.items():
            print(f"- {intent_name}: {intent_data.get('description', '')}")
    else:
        print("\nNo intents were extracted from the conversation.")


if __name__ == "__main__":
    main()
