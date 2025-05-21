#!/usr/bin/env python3
"""
Example 6: Refactoring Assistant

This example demonstrates a refactoring assistant that remembers user preferences
and applies them consistently across multiple refactoring sessions.

Key concepts demonstrated:
1. Preference-driven refactoring
2. Consistent application of style preferences
3. Memory-augmented refactoring suggestions

Cognitive load reduction:
- Users don't need to repeat style preferences for each refactoring
- The system applies consistent style across multiple files
- The system remembers previous refactoring decisions
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore
from semantic_matrix_analyzer.conversation.memory.agent import AgentFactory, MemoryAugmentedAgent


class RefactoringAssistant:
    """A refactoring assistant that remembers user preferences."""
    
    def __init__(self, storage_dir: str):
        """Initialize the refactoring assistant.
        
        Args:
            storage_dir: Directory to store conversations and knowledge graph.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.knowledge_graph_path = self.storage_dir / "knowledge_graph.json"
        self.agent = AgentFactory.create_agent(self.storage_dir, self.knowledge_graph_path)
        self.conversation_id = None
        
        # Default style preferences
        self.style_preferences = {
            "max_line_length": 80,
            "indentation": 4,
            "function_style": "snake_case",
            "class_style": "PascalCase",
            "docstring_style": "google"
        }
    
    def start_conversation(self, title: str = "Refactoring") -> str:
        """Start a new conversation.
        
        Args:
            title: The title of the conversation.
            
        Returns:
            The ID of the conversation.
        """
        self.conversation_id = self.agent.start_conversation()
        return self.conversation_id
    
    def process_message(self, message: str) -> str:
        """Process a message from the user.
        
        Args:
            message: The message from the user.
            
        Returns:
            The response from the agent.
        """
        # Update style preferences based on the message
        self._update_style_preferences(message)
        
        return self.agent.process_message(message)
    
    def _update_style_preferences(self, message: str) -> None:
        """Update style preferences based on the message.
        
        Args:
            message: The message from the user.
        """
        # This is a simple implementation that looks for specific keywords
        # In a real implementation, this would use more sophisticated NLP
        
        # Check for line length preferences
        if "line length" in message.lower():
            if "120" in message:
                self.style_preferences["max_line_length"] = 120
            elif "100" in message:
                self.style_preferences["max_line_length"] = 100
            elif "80" in message:
                self.style_preferences["max_line_length"] = 80
        
        # Check for indentation preferences
        if "indentation" in message.lower() or "spaces" in message.lower():
            if "2 spaces" in message.lower() or "2-space" in message.lower():
                self.style_preferences["indentation"] = 2
            elif "4 spaces" in message.lower() or "4-space" in message.lower():
                self.style_preferences["indentation"] = 4
            elif "tabs" in message.lower():
                self.style_preferences["indentation"] = "tabs"
        
        # Check for function naming style
        if "function" in message.lower() and "style" in message.lower():
            if "camel" in message.lower():
                self.style_preferences["function_style"] = "camelCase"
            elif "snake" in message.lower():
                self.style_preferences["function_style"] = "snake_case"
        
        # Check for docstring style
        if "docstring" in message.lower() and "style" in message.lower():
            if "google" in message.lower():
                self.style_preferences["docstring_style"] = "google"
            elif "numpy" in message.lower():
                self.style_preferences["docstring_style"] = "numpy"
            elif "sphinx" in message.lower():
                self.style_preferences["docstring_style"] = "sphinx"
    
    def get_active_intents(self) -> set:
        """Get the active intents for the current conversation.
        
        Returns:
            A set of active intent names.
        """
        return self.agent.get_active_intents()
    
    def refactor_code(self, code: str, file_path: str) -> str:
        """Refactor code based on style preferences and active intents.
        
        Args:
            code: The code to refactor.
            file_path: The path to the file containing the code.
            
        Returns:
            The refactored code.
        """
        # Get active intents
        active_intents = self.get_active_intents()
        
        # Apply refactorings based on style preferences and active intents
        refactored_code = code
        
        # Apply line length refactoring
        refactored_code = self._refactor_line_length(refactored_code)
        
        # Apply indentation refactoring
        refactored_code = self._refactor_indentation(refactored_code)
        
        # Apply function style refactoring
        refactored_code = self._refactor_function_style(refactored_code)
        
        # Apply docstring style refactoring
        refactored_code = self._refactor_docstring_style(refactored_code)
        
        # Apply intent-specific refactorings
        if "Clean Code" in active_intents:
            refactored_code = self._refactor_for_clean_code(refactored_code)
        
        if "Error Handling" in active_intents:
            refactored_code = self._refactor_for_error_handling(refactored_code)
        
        return refactored_code
    
    def _refactor_line_length(self, code: str) -> str:
        """Refactor code to match line length preference.
        
        Args:
            code: The code to refactor.
            
        Returns:
            The refactored code.
        """
        # This is a placeholder implementation
        # In a real implementation, this would use a proper code formatter
        
        max_length = self.style_preferences["max_line_length"]
        lines = code.splitlines()
        refactored_lines = []
        
        for line in lines:
            if len(line) > max_length:
                # Very simple line wrapping for demonstration
                # In a real implementation, this would be much more sophisticated
                if "=" in line:
                    parts = line.split("=", 1)
                    refactored_lines.append(parts[0].rstrip() + "=")
                    refactored_lines.append("    " + parts[1].lstrip())
                else:
                    refactored_lines.append(line)
            else:
                refactored_lines.append(line)
        
        return "\n".join(refactored_lines)
    
    def _refactor_indentation(self, code: str) -> str:
        """Refactor code to match indentation preference.
        
        Args:
            code: The code to refactor.
            
        Returns:
            The refactored code.
        """
        # This is a placeholder implementation
        # In a real implementation, this would use a proper code formatter
        
        # For demonstration purposes, we'll just return the original code
        return code
    
    def _refactor_function_style(self, code: str) -> str:
        """Refactor code to match function style preference.
        
        Args:
            code: The code to refactor.
            
        Returns:
            The refactored code.
        """
        # This is a placeholder implementation
        # In a real implementation, this would use a proper code formatter
        
        # For demonstration purposes, we'll just return the original code
        return code
    
    def _refactor_docstring_style(self, code: str) -> str:
        """Refactor code to match docstring style preference.
        
        Args:
            code: The code to refactor.
            
        Returns:
            The refactored code.
        """
        # This is a placeholder implementation
        # In a real implementation, this would use a proper code formatter
        
        # For demonstration purposes, we'll just return the original code
        return code
    
    def _refactor_for_clean_code(self, code: str) -> str:
        """Refactor code for clean code principles.
        
        Args:
            code: The code to refactor.
            
        Returns:
            The refactored code.
        """
        # This is a placeholder implementation
        # In a real implementation, this would apply clean code refactorings
        
        # For demonstration purposes, we'll just return the original code
        return code
    
    def _refactor_for_error_handling(self, code: str) -> str:
        """Refactor code for better error handling.
        
        Args:
            code: The code to refactor.
            
        Returns:
            The refactored code.
        """
        # This is a placeholder implementation
        # In a real implementation, this would apply error handling refactorings
        
        # For demonstration purposes, we'll make a simple change
        if "except:" in code and "except Exception:" not in code:
            code = code.replace("except:", "except Exception:")
        
        return code
    
    def close_conversation(self) -> None:
        """Close the current conversation."""
        self.agent.close_conversation()
        self.conversation_id = None


def main():
    """Main entry point for the example."""
    # Create a directory for storing conversations
    storage_dir = Path("conversation_memory_examples/refactoring_assistant")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the refactoring assistant
    assistant = RefactoringAssistant(storage_dir)
    
    # Sample code to refactor
    code = """
def process_data(data):
    # This is a function that processes data
    result = []
    for item in data:
        if item is None:
            continue
        try:
            processed = item.process()
            result.append(processed)
        except:
            print("Error processing item")
    return result
"""
    
    # Start a conversation
    conversation_id = assistant.start_conversation("Refactoring")
    print(f"Started conversation: {conversation_id}")
    
    # First session: Set style preferences
    print("\n=== Session 1: Setting Style Preferences ===")
    response = assistant.process_message("I prefer 100 character line length and 4 spaces for indentation.")
    print(f"User: I prefer 100 character line length and 4 spaces for indentation.")
    print(f"AI: {response}")
    
    # Print style preferences
    print("\nStyle preferences:")
    for key, value in assistant.style_preferences.items():
        print(f"  {key}: {value}")
    
    # Close the conversation
    assistant.close_conversation()
    
    # Start a new conversation
    conversation_id = assistant.start_conversation("Refactoring 2")
    print(f"\n=== Session 2: Refactoring for Error Handling ===")
    
    # Second session: Focus on error handling
    response = assistant.process_message("I want to improve error handling in my code.")
    print(f"User: I want to improve error handling in my code.")
    print(f"AI: {response}")
    
    # Get active intents
    active_intents = assistant.get_active_intents()
    print(f"Active intents: {', '.join(active_intents)}")
    
    # Print style preferences (should remember from previous session)
    print("\nStyle preferences (remembered from previous session):")
    for key, value in assistant.style_preferences.items():
        print(f"  {key}: {value}")
    
    # Refactor the code
    print("\nOriginal code:")
    print(code)
    
    refactored_code = assistant.refactor_code(code, "example.py")
    
    print("\nRefactored code:")
    print(refactored_code)
    
    # Close the conversation
    assistant.close_conversation()
    
    print("\nThis example demonstrates how the system remembers user preferences across sessions.")
    print("In the first session, the user set style preferences.")
    print("In the second session, the system remembered these preferences and applied them,")
    print("along with error handling improvements based on the user's new request.")


if __name__ == "__main__":
    main()
