#!/usr/bin/env python3
"""
Example of using the conversation memory system for code analysis.

This script demonstrates how to use the conversation memory system to improve code analysis
by remembering user preferences and concerns across multiple conversations.
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


class MemoryAugmentedAnalyzer:
    """A code analyzer that uses conversation memory to improve analysis."""
    
    def __init__(self, storage_dir: str):
        """Initialize the analyzer.
        
        Args:
            storage_dir: Directory to store conversations and knowledge graph.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.knowledge_graph_path = self.storage_dir / "knowledge_graph.json"
        self.agent = AgentFactory.create_agent(self.storage_dir, self.knowledge_graph_path)
        self.conversation_id = None
    
    def start_conversation(self, title: str = "Code Analysis") -> str:
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
        return self.agent.process_message(message)
    
    def get_active_intents(self) -> set:
        """Get the active intents for the current conversation.
        
        Returns:
            A set of active intent names.
        """
        return self.agent.get_active_intents()
    
    def analyze_code(self, code: str, file_path: str) -> dict:
        """Analyze code based on active intents.
        
        Args:
            code: The code to analyze.
            file_path: The path to the file containing the code.
            
        Returns:
            A dictionary with analysis results.
        """
        # Get active intents
        active_intents = self.get_active_intents()
        
        # Analyze code based on active intents
        results = {}
        
        if "Clean Code" in active_intents:
            results["clean_code"] = self._analyze_clean_code(code)
        
        if "Error Handling" in active_intents:
            results["error_handling"] = self._analyze_error_handling(code)
        
        if "Performance" in active_intents:
            results["performance"] = self._analyze_performance(code)
        
        return results
    
    def _analyze_clean_code(self, code: str) -> dict:
        """Analyze code for clean code principles.
        
        Args:
            code: The code to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        # This is a placeholder for clean code analysis
        # In a real implementation, this would use more sophisticated techniques
        
        lines = code.splitlines()
        line_count = len(lines)
        
        # Check for long lines
        long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 80]
        
        # Check for long functions
        # This is a very simplistic approach
        function_lines = {}
        current_function = None
        function_start = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                if current_function:
                    function_lines[current_function] = i - function_start
                
                current_function = line.strip().split("def ")[1].split("(")[0].strip()
                function_start = i
            
            if line.strip().startswith("class "):
                current_function = None
        
        if current_function:
            function_lines[current_function] = len(lines) - function_start
        
        long_functions = {func: lines for func, lines in function_lines.items() if lines > 20}
        
        return {
            "line_count": line_count,
            "long_lines": long_lines,
            "long_functions": long_functions
        }
    
    def _analyze_error_handling(self, code: str) -> dict:
        """Analyze code for error handling.
        
        Args:
            code: The code to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        # This is a placeholder for error handling analysis
        # In a real implementation, this would use more sophisticated techniques
        
        # Check for try-except blocks
        try_except_count = code.count("try:") if "try:" in code else 0
        
        # Check for bare excepts
        bare_except_count = code.count("except:") if "except:" in code else 0
        
        # Check for specific exception types
        specific_except_count = code.count("except ") if "except " in code else 0
        
        return {
            "try_except_count": try_except_count,
            "bare_except_count": bare_except_count,
            "specific_except_count": specific_except_count
        }
    
    def _analyze_performance(self, code: str) -> dict:
        """Analyze code for performance issues.
        
        Args:
            code: The code to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        # This is a placeholder for performance analysis
        # In a real implementation, this would use more sophisticated techniques
        
        # Check for list comprehensions vs. loops
        list_comprehension_count = code.count("[") and code.count("for") and code.count("]")
        
        # Check for generator expressions
        generator_expression_count = code.count("(") and code.count("for") and code.count(")")
        
        return {
            "list_comprehension_count": list_comprehension_count,
            "generator_expression_count": generator_expression_count
        }
    
    def close_conversation(self) -> None:
        """Close the current conversation."""
        self.agent.close_conversation()
        self.conversation_id = None


def main():
    """Main entry point for the script."""
    setup_logging()
    
    # Create the analyzer
    analyzer = MemoryAugmentedAnalyzer("conversation_memory")
    
    # Start a conversation
    conversation_id = analyzer.start_conversation("Code Analysis")
    print(f"Started conversation: {conversation_id}")
    
    # Process some messages
    print("\nConversation 1:")
    response = analyzer.process_message("I want to improve the clean code in my project.")
    print(f"User: I want to improve the clean code in my project.")
    print(f"AI: {response}")
    
    response = analyzer.process_message("I'm particularly interested in function length and variable naming.")
    print(f"User: I'm particularly interested in function length and variable naming.")
    print(f"AI: {response}")
    
    # Get active intents
    active_intents = analyzer.get_active_intents()
    print(f"Active intents: {', '.join(active_intents)}")
    
    # Analyze some code
    code = """
def process_data(data):
    # This is a long function that does many things
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
    
    results = analyzer.analyze_code(code, "example.py")
    print("\nAnalysis results:")
    for category, category_results in results.items():
        print(f"  {category}:")
        for key, value in category_results.items():
            print(f"    {key}: {value}")
    
    # Close the conversation
    analyzer.close_conversation()
    
    # Start a new conversation
    conversation_id = analyzer.start_conversation("Code Analysis 2")
    print(f"\nStarted conversation: {conversation_id}")
    
    # Process some messages
    print("\nConversation 2:")
    response = analyzer.process_message("I want to make sure my error handling is robust.")
    print(f"User: I want to make sure my error handling is robust.")
    print(f"AI: {response}")
    
    # Get active intents
    active_intents = analyzer.get_active_intents()
    print(f"Active intents: {', '.join(active_intents)}")
    
    # Analyze the same code
    results = analyzer.analyze_code(code, "example.py")
    print("\nAnalysis results:")
    for category, category_results in results.items():
        print(f"  {category}:")
        for key, value in category_results.items():
            print(f"    {key}: {value}")
    
    # Close the conversation
    analyzer.close_conversation()


if __name__ == "__main__":
    main()
